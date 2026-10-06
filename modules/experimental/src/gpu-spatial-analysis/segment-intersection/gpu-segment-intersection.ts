// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUScan,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import {validateGraphViewsBelongToGraph} from '../../utils/gpu-contributor-utils';
import type {
  GPUSegmentGeometry,
  GPUSegmentIntersectionColumns,
  GPUSegmentIntersectionPairs
} from './segment-intersection-types';
import {
  getSegmentTableAccessorsWGSL,
  SEGMENT_PREDICATES_WGSL,
  SEGMENT_TABLE_STRIDE
} from './segment-intersection-wgsl';
import {getSegmentBVHNodes} from './segment-bvh';
import {createSegmentTableNode, getSegmentSlotCount} from './segment-table';

const OPERATION = 'GPUSegmentIntersection';

/** Vertex count from which `spatialSort` defaults to on. */
const SPATIAL_SORT_MINIMUM_SLOTS = 1024;

/**
 * Properties for {@link GPUSegmentIntersection}.
 *
 * Per-frame: the contents of every input buffer. Topology: view lengths, `sameFeatureOnly`,
 * `leafCapacity`, whether `right` is given, and which optional views exist.
 */
export type GPUSegmentIntersectionProps = GPUSegmentIntersectionColumns & {
  /** Prefix for generated node and transient IDs. Defaults to `'segment-intersection'`. */
  id?: string;
  /** Left segments. Output rows are left segment IDs (start-vertex indices of `left`). */
  left: GPUSegmentGeometry;
  /**
   * Right segments, indexed by a BVH. Omit for self mode, where `left` is intersected with itself.
   * The two sides may use different geometry kinds.
   */
  right?: GPUSegmentGeometry;
  /**
   * Self mode only: keep only pairs whose segments belong to the same feature. This keeps the
   * output small when the question is per-feature validity rather than overlap between features.
   * Defaults to `false`.
   */
  sameFeatureOnly?: boolean;
  /**
   * Compile-time. Reorders the right segments along a Morton (Z-order) curve before the BVH build
   * and sorts each left segment's hits afterwards. The output is identical either way; only cost
   * changes. Without it, a source order that jumps around the plane makes upper BVH nodes cover
   * everything and every probe visits most of the tree. Defaults to `true` from 1024 vertices on,
   * and `false` below, where the extra passes cost more than they save. On overflow the last
   * partly written left segment may hold an arbitrary subset of its hits when this is enabled.
   */
  spatialSort?: boolean;
  /** Power-of-two BVH leaf slots over the right segments. Defaults to the next power of two of its vertex count. */
  leafCapacity?: number;
  /**
   * Caller-owned `(left, right)` segment pairs, sorted by left then right segment ID, with the
   * capacity `pairs.leftIds.length`. `count` is `min(total, capacity)`; `overflow` is 1 when the
   * total exceeded the capacity, in which case the output holds the sorted prefix of the pairs.
   */
  pairs: GPUSegmentIntersectionPairs;
  /**
   * Optional one-row count of pairs within the capacity that were classified `uncertain`. Those
   * pairs are listed in the output with the `uncertain` kind; none are silently dropped.
   */
  uncertainCount?: GraphDataView<'uint32'>;
};

/** Returns the smallest power of two that is at least `value`. */
function getNextPowerOfTwo(value: number): number {
  let result = 1;
  while (result < value) {
    result *= 2;
  }
  return result;
}

/**
 * Finds every intersecting pair of line segments between two geometries, or within one.
 *
 * Pipeline: a segment table per side (one row per vertex, so segment IDs are vertex indices),
 * a `GPUBVH` over the right segment bounds, a counted and scanned probe from every left segment,
 * and an exact classification of every candidate pair. Candidates come out ordered by left
 * segment and then right segment (BVH leaves keep source order), so the output is sorted and
 * deterministic with no sort pass, and a capacity overflow keeps a consistent sorted prefix.
 * Nothing is read back.
 *
 * Classification uses exact orientation signs (an f32 filter with an exact integer fallback), so
 * a pair is `proper`, `touch`, `collinearTouch` or `overlap` exactly, for finite coordinates whose
 * products stay within 200 binary orders of magnitude of each other. A pair the predicates cannot certify is listed as `uncertain` and counted
 * in `uncertainCount`; it is never dropped. The reported intersection point of a `proper`
 * crossing is rounded to f32; touch and overlap points are input vertices, reported exactly.
 *
 * **Two-sided mode** (`right` given) lists every `(left, right)` segment pair: turf `lineIntersect`,
 * and the segment-level step of line/polygon overlay and noding.
 *
 * **Self mode** (`right` omitted) lists each unordered pair once with `left < right`, within one
 * geometry (turf `kinks`, self-intersection tests, `gpu-network` noding). Segments that follow
 * each other in the same ring or linestring are adjacent and are skipped, because meeting at the
 * shared vertex is not an intersection; an adjacent pair is still listed when it `overlap`s (a
 * spike that doubles back). The successor of a segment is the next non-degenerate segment, so a
 * repeated vertex does not make its neighbors count as non-adjacent. A closed linestring or a
 * polygon ring is cyclic. A segment that touches a non-adjacent segment of its own ring is listed.
 *
 * Skipped segments never intersect anything: zero-length segments, segments with non-finite
 * coordinates, polygon rings with fewer than three vertices and linestrings with one vertex.
 * Segment slots are vertices, so the cost scales with the vertex count. The self-mode successor
 * search looks at most 1024 vertices ahead.
 *
 * Output beyond `pairs.count` is unspecified.
 */
export class GPUSegmentIntersection implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUSegmentIntersectionProps;
  /** Whether the contributor intersects `left` with itself. */
  readonly isSelfMode: boolean;
  /** Number of left segment slots (vertices). */
  readonly leftSlotCount: number;
  /** Number of right segment slots (vertices). */
  readonly rightSlotCount: number;
  /** Pair capacity. */
  readonly pairCapacity: number;
  /** Resolved BVH leaf capacity. */
  readonly leafCapacity: number;
  /** Whether segments are Morton sorted before the BVH build. */
  readonly spatialSort: boolean;

  constructor(props: GPUSegmentIntersectionProps) {
    this.id = props.id ?? 'segment-intersection';
    this.props = props;
    const {id} = this;
    this.isSelfMode = !props.right;
    for (const [name, geometry] of [
      ['left', props.left],
      ['right', props.right]
    ] as const) {
      if (!geometry) {
        continue;
      }
      validatePackedView(geometry.positions, ['float32x2'], `${id} ${name}.positions`);
      const offsetViews =
        geometry.kind === 'lines'
          ? [['lineOffsets', geometry.lineOffsets] as const]
          : [
              ['featureOffsets', geometry.featureOffsets] as const,
              ['polygonOffsets', geometry.polygonOffsets] as const,
              ['ringOffsets', geometry.ringOffsets] as const
            ];
      for (const [offsetName, view] of offsetViews) {
        validatePackedUint32View(view, `${id} ${name}.${offsetName}`);
        if (view.length < 1) {
          throw new Error(`${id} ${name}.${offsetName} requires a terminal entry`);
        }
      }
      if (geometry.positions.length < 1) {
        throw new Error(`${id} ${name} must contain at least one vertex`);
      }
      if (geometry.positions.length >= 0x80000000) {
        throw new Error(`${id} ${name} has too many vertices`);
      }
    }
    if (props.sameFeatureOnly && !this.isSelfMode) {
      throw new Error(`${id} sameFeatureOnly requires self mode (omit right)`);
    }
    this.leftSlotCount = getSegmentSlotCount(props.left);
    this.rightSlotCount = getSegmentSlotCount(props.right ?? props.left);
    this.spatialSort = props.spatialSort ?? this.rightSlotCount >= SPATIAL_SORT_MINIMUM_SLOTS;
    this.leafCapacity = props.leafCapacity ?? getNextPowerOfTwo(this.rightSlotCount);
    if (
      !Number.isSafeInteger(this.leafCapacity) ||
      this.leafCapacity < 1 ||
      (this.leafCapacity & (this.leafCapacity - 1)) !== 0
    ) {
      throw new Error(`${id} leafCapacity must be a positive power of two`);
    }
    const {pairs} = props;
    validatePackedUint32View(pairs.leftIds, `${id} pairs.leftIds`);
    validatePackedUint32View(pairs.rightIds, `${id} pairs.rightIds`);
    this.pairCapacity = pairs.leftIds.length;
    if (this.pairCapacity < 1 || pairs.rightIds.length !== this.pairCapacity) {
      throw new Error(`${id} pairs.leftIds and pairs.rightIds must have equal nonzero length`);
    }
    for (const [name, view] of [
      ['count', pairs.count],
      ['overflow', pairs.overflow],
      ['totalCount', pairs.totalCount],
      ['uncertainCount', props.uncertainCount]
    ] as const) {
      if (view) {
        validatePackedUint32View(view, `${id} ${name}`);
        if (view.length < 1) {
          throw new Error(`${id} ${name} must contain one uint32 row`);
        }
      }
    }
    for (const [name, view] of [
      ['kinds', props.kinds],
      ['leftFeatures', props.leftFeatures],
      ['rightFeatures', props.rightFeatures],
      ['leftRings', props.leftRings],
      ['rightRings', props.rightRings]
    ] as const) {
      if (view) {
        validatePackedUint32View(view, `${id} ${name}`);
        if (view.length !== this.pairCapacity) {
          throw new Error(`${id} ${name} length must equal the pair capacity`);
        }
      }
    }
    for (const [name, view] of [
      ['points', props.points],
      ['endPoints', props.endPoints]
    ] as const) {
      if (view) {
        validatePackedView(view, ['float32x2'], `${id} ${name}`);
        if (view.length !== this.pairCapacity) {
          throw new Error(`${id} ${name} length must equal the pair capacity`);
        }
      }
    }
  }

  private getInputViews(): GraphDataView[] {
    const views: GraphDataView[] = [];
    for (const geometry of [this.props.left, this.props.right]) {
      if (!geometry) {
        continue;
      }
      views.push(geometry.positions);
      if (geometry.kind === 'lines') {
        views.push(geometry.lineOffsets);
      } else {
        views.push(geometry.featureOffsets, geometry.polygonOffsets, geometry.ringOffsets);
      }
    }
    return views;
  }

  private getOutputViews(): (GraphDataView | undefined)[] {
    const {pairs, uncertainCount} = this.props;
    return [
      pairs.leftIds,
      pairs.rightIds,
      pairs.count,
      pairs.overflow,
      pairs.totalCount,
      uncertainCount,
      this.props.kinds,
      this.props.points,
      this.props.endPoints,
      this.props.leftFeatures,
      this.props.rightFeatures,
      this.props.leftRings,
      this.props.rightRings
    ];
  }

  /** Returns table, BVH, probe, scan, classification, column and scalar nodes in order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, isSelfMode, leftSlotCount, rightSlotCount, pairCapacity} = this;
    const {pairs} = props;
    validateGraphViewsBelongToGraph(id, graph, [...this.getInputViews(), ...this.getOutputViews()]);
    const nodes: GPUCommandNode<Parameters>[] = [];

    // Segment tables, with bounds for the indexed (right) side.
    const rightGeometry = props.right ?? props.left;
    const rightTable = createTransientView(
      graph,
      `${id}-right-table`,
      'uint32',
      rightSlotCount * SEGMENT_TABLE_STRIDE
    );
    const minima = createTransientView(graph, `${id}-bounds-minima`, 'float32x2', rightSlotCount);
    const maxima = createTransientView(graph, `${id}-bounds-maxima`, 'float32x2', rightSlotCount);
    nodes.push(
      createSegmentTableNode<Parameters>(graph, {
        id: `${id}-right-table`,
        operation: OPERATION,
        geometry: rightGeometry,
        table: rightTable,
        minima,
        maxima
      })
    );
    let leftTable = rightTable;
    if (!isSelfMode) {
      leftTable = createTransientView(
        graph,
        `${id}-left-table`,
        'uint32',
        leftSlotCount * SEGMENT_TABLE_STRIDE
      );
      nodes.push(
        createSegmentTableNode<Parameters>(graph, {
          id: `${id}-left-table`,
          operation: OPERATION,
          geometry: props.left,
          table: leftTable
        })
      );
    }

    // BVH over right segment bounds, Morton sorted for spatial coherence when requested.
    const {bvh, nodes: bvhNodes} = getSegmentBVHNodes(graph, {
      id,
      operation: OPERATION,
      minima,
      maxima,
      leafCapacity: this.leafCapacity,
      spatialSort: this.spatialSort
    });
    nodes.push(...bvhNodes);

    const state = createTransientView(graph, `${id}-state`, 'uint32', 4);
    const counts = createTransientView(graph, `${id}-counts`, 'uint32', leftSlotCount);
    const offsets = createTransientView(graph, `${id}-offsets`, 'uint32', leftSlotCount);

    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-clear`,
        operation: OPERATION,
        variant: 'clear',
        bindings: [{name: 'state', view: state, type: 'u32', access: 'read_write'}],
        invocationCount: 4,
        body: 'state[stateOffset + index] = 0u;'
      })
    );

    const pairDeclarations = `${SEGMENT_PREDICATES_WGSL}
${getSegmentTableAccessorsWGSL('left')}
${getSegmentTableAccessorsWGSL('right')}
const SELF: bool = ${isSelfMode};
const SAME_FEATURE: bool = ${Boolean(props.sameFeatureOnly)};
const RIGHT_COUNT: u32 = ${rightSlotCount}u;
const INTERNAL_NODE_COUNT: u32 = ${bvh.internalNodeCount}u;
const PAIR_CAPACITY: u32 = ${pairCapacity}u;
fn pairHit(left: u32, right: u32, withGeometry: bool) -> SegmentHit {
  var none = SegmentHit(KIND_NONE, vec2f(0.0), vec2f(0.0));
  if (!leftValid(left) || !rightValid(right)) { return none; }
  if (SELF && right <= left) { return none; }
  if (SAME_FEATURE && leftFeature(left) != rightFeature(right)) { return none; }
  var hit = classifySegments(leftStart(left), leftEnd(left), rightStart(right), rightEnd(right), withGeometry);
  if (SELF && (hit.kind == KIND_TOUCH || hit.kind == KIND_COLLINEAR_TOUCH) &&
      (leftSuccessor(left) == right || rightSuccessor(right) == left)) {
    hit.kind = KIND_NONE;
  }
  return hit;
}
fn nodeOverlaps(node: u32, queryMinimum: vec2f, queryMaximum: vec2f) -> bool {
  let component = node * 2u;
  let minimum = vec2f(nodeMinima[nodeMinimaOffset + component], nodeMinima[nodeMinimaOffset + component + 1u]);
  let maximum = vec2f(nodeMaxima[nodeMaximaOffset + component], nodeMaxima[nodeMaximaOffset + component + 1u]);
  return all(minimum <= queryMaximum) && all(queryMinimum <= maximum);
}`;
    const probeBindings: WGSLKernelBinding[] = [
      {name: 'leftTable', view: leftTable, type: 'u32', access: 'read'},
      {name: 'rightTable', view: rightTable, type: 'u32', access: 'read'},
      {name: 'nodeMinima', view: bvh.nodeMinima, type: 'f32', access: 'read'},
      {name: 'nodeMaxima', view: bvh.nodeMaxima, type: 'f32', access: 'read'},
      {name: 'leafIds', view: bvh.leafIds, type: 'u32', access: 'read'}
    ];
    const probeBody = (leafBody: string, prologue: string, epilogue: string) => `${prologue}
  if (leftValid(index)) {
    let start = leftStart(index);
    let end = leftEnd(index);
    let queryMinimum = min(start, end);
    let queryMaximum = max(start, end);
    var node = 0u;
    loop {
      if (nodeOverlaps(node, queryMinimum, queryMaximum)) {
        if (node < INTERNAL_NODE_COUNT) {
          node = node * 2u + 1u;
          continue;
        }
        let rightRow = leafIds[leafIdsOffset + node - INTERNAL_NODE_COUNT];
        if (rightRow < RIGHT_COUNT && pairHit(index, rightRow, false).kind != KIND_NONE) {
          ${leafBody}
        }
      }
      // Climb while the node is a right child, then step to the right sibling.
      loop {
        if (node == 0u || (node & 1u) == 1u) { break; }
        node = (node - 1u) / 2u;
      }
      if (node == 0u) { break; }
      node = node + 1u;
    }
  }
  ${epilogue}`;
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-count`,
        operation: OPERATION,
        variant: 'count',
        bindings: [
          ...probeBindings,
          {name: 'counts', view: counts, type: 'u32', access: 'read_write'}
        ],
        invocationCount: leftSlotCount,
        declarations: pairDeclarations,
        body: probeBody(
          'found = found + 1u;',
          'var found = 0u;',
          'counts[countsOffset + index] = found;'
        )
      })
    );
    nodes.push(
      ...new GPUScan({
        id: `${id}-scan`,
        input: counts,
        output: offsets,
        mode: 'exclusive'
      }).getCommandNodes(graph)
    );
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-total`,
        operation: OPERATION,
        variant: 'total',
        bindings: [
          {name: 'counts', view: counts, type: 'u32', access: 'read'},
          {name: 'offsets', view: offsets, type: 'u32', access: 'read'},
          {name: 'state', view: state, type: 'u32', access: 'read_write'}
        ],
        invocationCount: 1,
        body: `state[stateOffset] = offsets[offsetsOffset + ${leftSlotCount - 1}u] + counts[countsOffset + ${leftSlotCount - 1}u];`
      })
    );
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-write`,
        operation: OPERATION,
        variant: 'write',
        bindings: [
          ...probeBindings,
          {name: 'offsets', view: offsets, type: 'u32', access: 'read'},
          {name: 'pairLeft', view: pairs.leftIds, type: 'u32', access: 'read_write'},
          {name: 'pairRight', view: pairs.rightIds, type: 'u32', access: 'read_write'}
        ],
        invocationCount: leftSlotCount,
        declarations: pairDeclarations,
        body: probeBody(
          `let slot = offsets[offsetsOffset + index] + found;
          if (slot < PAIR_CAPACITY) {
            pairLeft[pairLeftOffset + slot] = index;
            pairRight[pairRightOffset + slot] = rightRow;
          }
          found = found + 1u;`,
          'var found = 0u;',
          ''
        )
      })
    );

    if (this.spatialSort) {
      // Morton-ordered leaves emit each left segment's hits out of order: sort every range.
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-sort-hits`,
          operation: OPERATION,
          variant: 'sort-hits',
          bindings: [
            {name: 'counts', view: counts, type: 'u32', access: 'read'},
            {name: 'offsets', view: offsets, type: 'u32', access: 'read'},
            {name: 'pairRight', view: pairs.rightIds, type: 'u32', access: 'read_write'}
          ],
          invocationCount: leftSlotCount,
          declarations: `const PAIR_CAPACITY: u32 = ${pairCapacity}u;
const GAPS = array<u32, 8>(701u, 301u, 132u, 57u, 23u, 10u, 4u, 1u);`,
          body: `let first = offsets[offsetsOffset + index];
  let last = min(first + counts[countsOffset + index], PAIR_CAPACITY);
  if (last <= first + 1u) { return; }
  // Shell sort with Ciura gaps; ranges are short unless one segment crosses very many others.
  for (var gapIndex = 0u; gapIndex < 8u; gapIndex++) {
    let gap = GAPS[gapIndex];
    for (var slot = first + gap; slot < last; slot++) {
      let value = pairRight[pairRightOffset + slot];
      var hole = slot;
      loop {
        if (hole < first + gap) { break; }
        let earlier = pairRight[pairRightOffset + hole - gap];
        if (earlier <= value) { break; }
        pairRight[pairRightOffset + hole] = earlier;
        hole = hole - gap;
      }
      pairRight[pairRightOffset + hole] = value;
    }
  }`
        })
      );
    }

    // Per-pair classification columns.
    const {kinds, points, endPoints, uncertainCount} = props;
    if (kinds || points || endPoints || uncertainCount) {
      const bindings: WGSLKernelBinding[] = [
        {name: 'pairLeft', view: pairs.leftIds, type: 'u32', access: 'read'},
        {name: 'pairRight', view: pairs.rightIds, type: 'u32', access: 'read'},
        {name: 'leftTable', view: leftTable, type: 'u32', access: 'read'},
        {name: 'rightTable', view: rightTable, type: 'u32', access: 'read'},
        {name: 'state', view: state, type: 'atomic<u32>', access: 'read_write'}
      ];
      if (kinds) {
        bindings.push({name: 'kinds', view: kinds, type: 'u32', access: 'read_write'});
      }
      if (points) {
        bindings.push({name: 'points', view: points, type: 'f32', access: 'read_write'});
      }
      if (endPoints) {
        bindings.push({name: 'endPoints', view: endPoints, type: 'f32', access: 'read_write'});
      }
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-classify`,
          operation: OPERATION,
          variant: 'classify',
          bindings,
          invocationCount: pairCapacity,
          declarations: pairDeclarations.replace(/fn nodeOverlaps[\s\S]*$/, ''),
          body: `if (index >= min(atomicLoad(&state[stateOffset]), PAIR_CAPACITY)) { return; }
  let hit = pairHit(pairLeft[pairLeftOffset + index], pairRight[pairRightOffset + index], true);
  if (hit.kind == KIND_UNCERTAIN) { atomicAdd(&state[stateOffset + 1u], 1u); }
  ${kinds ? 'kinds[kindsOffset + index] = hit.kind;' : ''}
  ${points ? 'points[pointsOffset + index * 2u] = hit.point.x; points[pointsOffset + index * 2u + 1u] = hit.point.y;' : ''}
  ${endPoints ? 'endPoints[endPointsOffset + index * 2u] = hit.endPoint.x; endPoints[endPointsOffset + index * 2u + 1u] = hit.endPoint.y;' : ''}`
        })
      );
    }
    for (const [side, ids, table, featureColumn, ringColumn] of [
      ['left', pairs.leftIds, leftTable, props.leftFeatures, props.leftRings],
      ['right', pairs.rightIds, rightTable, props.rightFeatures, props.rightRings]
    ] as const) {
      if (!featureColumn && !ringColumn) {
        continue;
      }
      const bindings: WGSLKernelBinding[] = [
        {name: 'ids', view: ids, type: 'u32', access: 'read'},
        {name: `${side}Table`, view: table, type: 'u32', access: 'read'},
        {name: 'state', view: state, type: 'u32', access: 'read'}
      ];
      if (featureColumn) {
        bindings.push({
          name: 'featureColumn',
          view: featureColumn,
          type: 'u32',
          access: 'read_write'
        });
      }
      if (ringColumn) {
        bindings.push({name: 'ringColumn', view: ringColumn, type: 'u32', access: 'read_write'});
      }
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-${side}-columns`,
          operation: OPERATION,
          variant: `${side}-columns`,
          bindings,
          invocationCount: pairCapacity,
          declarations: `const SEGMENT_NONE: u32 = 0xffffffffu;
const PAIR_CAPACITY: u32 = ${pairCapacity}u;
${getSegmentTableAccessorsWGSL(side)}`,
          body: `if (index >= min(state[stateOffset], PAIR_CAPACITY)) { return; }
  let row = ids[idsOffset + index];
  ${featureColumn ? `featureColumn[featureColumnOffset + index] = ${side}Feature(row);` : ''}
  ${ringColumn ? `ringColumn[ringColumnOffset + index] = ${side}Ring(row);` : ''}`
        })
      );
    }

    // Scalars.
    const scalarBindings: WGSLKernelBinding[] = [
      {name: 'state', view: state, type: 'u32', access: 'read'},
      {name: 'bvhOverflow', view: bvh.overflow, type: 'u32', access: 'read'}
    ];
    const scalars: [string, GraphDataView<'uint32'> | undefined][] = [
      ['count', pairs.count],
      ['overflow', pairs.overflow],
      ['totalCount', pairs.totalCount],
      ['uncertainCount', uncertainCount]
    ];
    for (const [name, view] of scalars) {
      if (view) {
        scalarBindings.push({name, view, type: 'u32', access: 'read_write'});
      }
    }
    const has = (name: string) => scalars.some(([key, view]) => key === name && view);
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-finalize`,
        operation: OPERATION,
        variant: 'finalize',
        bindings: scalarBindings,
        invocationCount: 1,
        declarations: `const PAIR_CAPACITY: u32 = ${pairCapacity}u;`,
        body: `let total = state[stateOffset];
  let overflowed = bvhOverflow[bvhOverflowOffset] != 0u || total > PAIR_CAPACITY;
  ${has('count') ? 'count[countOffset] = min(total, PAIR_CAPACITY);' : ''}
  ${has('overflow') ? 'overflow[overflowOffset] = select(0u, 1u, overflowed);' : ''}
  ${has('totalCount') ? 'totalCount[totalCountOffset] = total;' : ''}
  ${has('uncertainCount') ? 'uncertainCount[uncertainCountOffset] = state[stateOffset + 1u];' : ''}`
      })
    );
    return nodes;
  }
}
