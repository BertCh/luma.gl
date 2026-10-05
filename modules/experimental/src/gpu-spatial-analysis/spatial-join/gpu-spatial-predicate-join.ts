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
import {
  GPU_POINT_IN_POLYGON_CLASSIFICATION,
  GPUPairwisePointInPolygon
} from '../../geospatial/gpu-pairwise-point-in-polygon';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import {
  captureGraphCommandNodes,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {validateGPUSpatialWeights, type GPUSpatialWeights} from '../spatial-weights/index';
import {
  createSpatialJoinBoundsNode,
  getFeatureBVHNodes,
  getNextPowerOfTwo,
  isPowerOfTwo,
  SPATIAL_JOIN_WGSL_HELPERS,
  validateDisjointOutputs
} from './spatial-join-passes';
import {
  getSpatialPredicateWGSL,
  type SpatialPredicateName,
  type SpatialPredicateSide
} from './spatial-predicate-wgsl';
import type {GPUSpatialJoinGeometry, GPUSpatialJoinPairs} from './spatial-join-types';

const OPERATION = 'GPUSpatialPredicateJoin';

/** Predicates supported by {@link GPUSpatialPredicateJoin}. */
export type GPUSpatialPredicate = SpatialPredicateName;

/**
 * Properties for {@link GPUSpatialPredicateJoin}.
 *
 * Per-frame: the contents of every input buffer. Topology: view lengths, `predicate`, `distance`,
 * `candidateCapacity`, `leafCapacity`, `excludeSameRow`, and which optional views exist.
 */
export type GPUSpatialPredicateJoinProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'spatial-predicate-join'`. */
  id?: string;
  /** Left features. Output rows (and CSR rows of `weights`) are left features. */
  left: GPUSpatialJoinGeometry;
  /** Right features, indexed by a BVH. Prefer the side with more features or larger extent. */
  right: GPUSpatialJoinGeometry;
  /**
   * Predicate evaluated as `predicate(left, right)`, following OGC Simple Features and the
   * DE-9IM matrix. `within(a, b)` is exactly `contains(b, a)`. See {@link GPUSpatialPredicateJoin}
   * for the boundary rules.
   */
  predicate: GPUSpatialPredicate;
  /** Planar distance in coordinate units for `'dwithin'` (inclusive). Required for it, ignored otherwise. */
  distance?: number;
  /**
   * Skip pairs with equal left and right rows. Set it for a self-join (`left` and `right` are the
   * same features) so each feature is not matched with itself, as `GPUSpatialWeights` requires.
   * Defaults to `false`.
   */
  excludeSameRow?: boolean;
  /** Maximum `(left, right)` bounding-box candidates evaluated per encoding. */
  candidateCapacity: number;
  /** Power-of-two BVH leaf slots over `right`. Defaults to the next power of two of its feature count. */
  leafCapacity?: number;
  /** Matched pairs sorted by `(left, right)`. At least one of `pairs` and `weights` is required. */
  pairs?: GPUSpatialJoinPairs;
  /**
   * Matches as cross spatial weights: one CSR row per left feature, neighbors are right rows
   * ascending, every weight is `1`. `weights.neighbors.length` is the pair capacity and must equal
   * `pairs.leftIds.length` when both are given. `weights.distances` is not supported.
   */
  weights?: GPUSpatialWeights;
  /** One-row flag: 1 when the BVH, candidate, or pair capacity overflowed. Required without `pairs`. */
  overflow?: GraphDataView<'uint32'>;
  /** Optional one-row unclamped candidate count, for sizing `candidateCapacity`. */
  candidateCount?: GraphDataView<'uint32'>;
  /**
   * Optional one-row count of point/polygon candidates the robust classifier could not certify.
   * Those pairs are decided by the f32 test instead (exact for small-integer or dyadic input,
   * so a point exactly on a diagonal edge is still found), and may be wrong only for razor-thin
   * near-degenerate input. Always 0 for other geometry combinations.
   */
  uncertainCount?: GraphDataView<'uint32'>;
};

function getFeatureCount(geometry: GPUSpatialJoinGeometry): number {
  switch (geometry.kind) {
    case 'points':
      return geometry.positions.length;
    case 'lines':
      return geometry.lineOffsets.length - 1;
    case 'polygons':
      return geometry.featureOffsets.length - 1;
  }
}

/**
 * Joins two sets of planar features by a spatial predicate: the GPU equivalent of a GeoPandas
 * `sjoin` or a PostGIS `JOIN ... ON ST_Intersects(l, r)`.
 *
 * Pipeline: per-feature bounds, a `GPUBVH` over the right bounds, a counted and scanned bounding
 * box probe from every left feature (candidates come out ordered by left row and then right row),
 * an exact predicate test per candidate, and a scan-based stable compaction into the caller's
 * capacity-bounded output. Nothing is read back; `overflow` reports when a capacity was exceeded,
 * in which case the output holds a prefix-consistent subset (candidates past `candidateCapacity`
 * and matches past the pair capacity are dropped).
 *
 * Geometry kinds on either side: points, linestrings and polygons or multipolygons with holes (see
 * {@link GPUSpatialJoinGeometry}); all nine combinations are supported.
 *
 * **Semantics (OGC / DE-9IM).** A point's interior is the point and its boundary is empty. A
 * linestring's boundary is its two endpoints, unless it is closed, and the rest is interior. A
 * polygon's boundary is its rings and its interior is the area between them (holes are exterior).
 * - `intersects(a, b)`: the closed geometries share at least one point. A point on a polygon
 *   boundary intersects the polygon; polygons that only touch along an edge or at a corner intersect.
 * - `contains(a, b)`: every point of `b` lies in the closure of `a` and the interiors intersect.
 *   A point on the boundary of a polygon is not contained by it; a linestring lying entirely on a
 *   polygon boundary is not contained; a polygon contains itself; a linestring does not contain its
 *   own endpoints (unless closed); a polygon contains a polygon that touches its boundary from
 *   inside; a point contains only an equal point; a lower-dimension geometry never contains a
 *   higher-dimension one.
 * - `within(a, b)`: `contains(b, a)`.
 * - `dwithin(a, b)`: the minimum planar distance between the closed geometries is at most
 *   `distance`. Zero distance counts, so it includes everything that intersects.
 *
 * Point/polygon pairs for `intersects`, `contains` and `within` use the robust double-single
 * classifier `GPUPairwisePointInPolygon`. Every other combination uses f32 orientation tests that
 * are exact for coordinates whose products are exactly representable (small integers, dyadic
 * rationals) and may misclassify razor-thin near-degenerate configurations otherwise. Coordinates
 * must be finite; empty or invalid features never match. Polygon-in-polygon containment assumes
 * valid polygons; a polygon that coincides with a hole is handled by a short probe along shared edges.
 *
 * Weights output: a join is a cross spatial-weights matrix (rows are left features, neighbors are
 * right rows, ascending), so `weights` writes the CSR directly from the sorted pairs. For a
 * self-join pass `excludeSameRow` to satisfy the no-self-neighbor invariant.
 */
export class GPUSpatialPredicateJoin implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUSpatialPredicateJoinProps;
  /** Number of left features. */
  readonly leftCount: number;
  /** Number of right features. */
  readonly rightCount: number;
  /** Candidate pair capacity. */
  readonly candidateCapacity: number;
  /** Matched pair capacity. */
  readonly pairCapacity: number;
  /** Resolved BVH leaf capacity. */
  readonly leafCapacity: number;

  constructor(props: GPUSpatialPredicateJoinProps) {
    this.id = props.id ?? 'spatial-predicate-join';
    this.props = props;
    const {id} = this;
    for (const [name, geometry] of [
      ['left', props.left],
      ['right', props.right]
    ] as const) {
      validatePackedView(geometry.positions, ['float32x2'], `${id} ${name}.positions`);
      const offsetViews =
        geometry.kind === 'lines'
          ? [['lineOffsets', geometry.lineOffsets] as const]
          : geometry.kind === 'polygons'
            ? [
                ['featureOffsets', geometry.featureOffsets] as const,
                ['polygonOffsets', geometry.polygonOffsets] as const,
                ['ringOffsets', geometry.ringOffsets] as const
              ]
            : [];
      for (const [offsetName, view] of offsetViews) {
        validatePackedUint32View(view, `${id} ${name}.${offsetName}`);
        if (view.length < 1) {
          throw new Error(`${id} ${name}.${offsetName} requires a terminal entry`);
        }
      }
      if (getFeatureCount(geometry) < 1) {
        throw new Error(`${id} ${name} must contain at least one feature`);
      }
    }
    this.leftCount = getFeatureCount(props.left);
    this.rightCount = getFeatureCount(props.right);
    if (props.predicate === 'dwithin') {
      if (props.distance === undefined || !Number.isFinite(props.distance) || props.distance < 0) {
        throw new Error(`${id} dwithin requires a finite, non-negative distance`);
      }
    } else if (!['intersects', 'contains', 'within'].includes(props.predicate)) {
      throw new Error(`${id} unknown predicate ${String(props.predicate)}`);
    }
    this.candidateCapacity = props.candidateCapacity;
    if (
      !Number.isSafeInteger(this.candidateCapacity) ||
      this.candidateCapacity < 1 ||
      2 * this.candidateCapacity + 2 > 0xffffffff
    ) {
      throw new Error(`${id} candidateCapacity must be a positive integer`);
    }
    this.leafCapacity = props.leafCapacity ?? getNextPowerOfTwo(Math.max(this.rightCount, 1));
    if (!isPowerOfTwo(this.leafCapacity)) {
      throw new Error(`${id} leafCapacity must be a positive power of two`);
    }
    if (!props.pairs && !props.weights) {
      throw new Error(`${id} requires pairs, weights, or both`);
    }
    if (!props.pairs && !props.overflow) {
      throw new Error(`${id} requires overflow when pairs is not given`);
    }
    if (props.pairs) {
      const {pairs} = props;
      validatePackedUint32View(pairs.leftIds, `${id} pairs.leftIds`);
      validatePackedUint32View(pairs.rightIds, `${id} pairs.rightIds`);
      if (pairs.leftIds.length < 1 || pairs.rightIds.length !== pairs.leftIds.length) {
        throw new Error(`${id} pairs.leftIds and pairs.rightIds must have equal nonzero length`);
      }
      for (const [name, view] of [
        ['count', pairs.count],
        ['overflow', pairs.overflow],
        ['totalCount', pairs.totalCount]
      ] as const) {
        if (view) {
          validatePackedUint32View(view, `${id} pairs.${name}`);
          if (view.length < 1) {
            throw new Error(`${id} pairs.${name} must contain one uint32 row`);
          }
        }
      }
    }
    if (props.weights) {
      const rows = validateGPUSpatialWeights(id, props.weights);
      if (rows !== this.leftCount) {
        throw new Error(`${id} weights must have one row per left feature`);
      }
      if (props.weights.distances) {
        throw new Error(`${id} weights.distances is not supported`);
      }
      if (props.pairs && props.pairs.leftIds.length !== props.weights.neighbors.length) {
        throw new Error(`${id} weights.neighbors length must equal the pair capacity`);
      }
    }
    this.pairCapacity = props.pairs
      ? props.pairs.leftIds.length
      : (props.weights as GPUSpatialWeights).neighbors.length;
    for (const [name, view] of [
      ['overflow', props.overflow],
      ['candidateCount', props.candidateCount],
      ['uncertainCount', props.uncertainCount]
    ] as const) {
      if (view) {
        validatePackedUint32View(view, `${id} ${name}`);
        if (view.length < 1) {
          throw new Error(`${id} ${name} must contain one uint32 row`);
        }
      }
    }
    validateDisjointOutputs(id, this.getInputViews(), this.getOutputViews());
  }

  private getInputViews(): GraphDataView[] {
    const views: GraphDataView[] = [];
    for (const geometry of [this.props.left, this.props.right]) {
      views.push(geometry.positions);
      if (geometry.kind === 'lines') {
        views.push(geometry.lineOffsets);
      } else if (geometry.kind === 'polygons') {
        views.push(geometry.featureOffsets, geometry.polygonOffsets, geometry.ringOffsets);
      }
    }
    return views;
  }

  private getOutputViews(): (GraphDataView | undefined)[] {
    const {pairs, weights, overflow, candidateCount, uncertainCount} = this.props;
    return [
      pairs?.leftIds,
      pairs?.rightIds,
      pairs?.count,
      pairs?.overflow,
      pairs?.totalCount,
      weights?.offsets,
      weights?.neighbors,
      weights?.weights,
      overflow,
      candidateCount,
      uncertainCount
    ];
  }

  /** Returns bounds, BVH, candidate, exact-test, compaction, and optional CSR nodes in order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, leftCount, rightCount, candidateCapacity, pairCapacity} = this;
    const {left, right, predicate, pairs, weights} = props;
    validateGraphViewsBelongToGraph(id, graph, [...this.getInputViews(), ...this.getOutputViews()]);
    const nodes: GPUCommandNode<Parameters>[] = [];

    // Per-side feature-to-ring tables (polygons) and bounds.
    const sides = [
      {name: 'left' as const, geometry: left, count: leftCount},
      {name: 'right' as const, geometry: right, count: rightCount}
    ].map(side => {
      const minima = createTransientView(
        graph,
        `${id}-${side.name}-minima`,
        'float32x2',
        side.count
      );
      const maxima = createTransientView(
        graph,
        `${id}-${side.name}-maxima`,
        'float32x2',
        side.count
      );
      let featureRings: GraphDataView<'uint32x2'> | undefined;
      if (side.geometry.kind === 'polygons') {
        featureRings = createTransientView(
          graph,
          `${id}-${side.name}-feature-rings`,
          'uint32x2',
          side.count
        );
        nodes.push(
          createFeatureRingsNode<Parameters>(graph, `${id}-${side.name}-rings`, {
            featureCount: side.count,
            geometry: side.geometry,
            featureRings
          })
        );
      }
      nodes.push(
        createBoundsNode<Parameters>(graph, `${id}-${side.name}-bounds`, {
          featureCount: side.count,
          geometry: side.geometry,
          featureRings,
          minima,
          maxima
        })
      );
      return {...side, minima, maxima, featureRings};
    });
    const [leftSide, rightSide] = sides;

    const {bvh, nodes: bvhNodes} = getFeatureBVHNodes(
      graph,
      id,
      rightSide.minima,
      rightSide.maxima,
      this.leafCapacity
    );
    nodes.push(...bvhNodes);

    const state = createTransientView(graph, `${id}-state`, 'uint32', 4);
    const leftCounts = createTransientView(graph, `${id}-left-counts`, 'uint32', leftCount);
    const leftOffsets = createTransientView(graph, `${id}-left-offsets`, 'uint32', leftCount);
    const candidatePairs = createTransientView(
      graph,
      `${id}-candidate-pairs`,
      'uint32x2',
      candidateCapacity
    );
    const flags = createTransientView(graph, `${id}-flags`, 'uint32', candidateCapacity);
    const flagOffsets = createTransientView(
      graph,
      `${id}-flag-offsets`,
      'uint32',
      candidateCapacity
    );
    const leftIds =
      pairs?.leftIds ?? createTransientView(graph, `${id}-left-ids`, 'uint32', pairCapacity);
    const rightIds =
      pairs?.rightIds ?? createTransientView(graph, `${id}-right-ids`, 'uint32', pairCapacity);
    const leftMatchCounts = weights
      ? createTransientView(graph, `${id}-left-match-counts`, 'uint32', leftCount + 1)
      : undefined;

    const usesRobustPointPolygon =
      predicate !== 'dwithin' &&
      ((left.kind === 'points' && right.kind === 'polygons') ||
        (left.kind === 'polygons' && right.kind === 'points'));
    const pairRowCount = 2 * candidateCapacity + 1;
    const polygonSide = left.kind === 'polygons' ? left : right.kind === 'polygons' ? right : null;
    const robust = usesRobustPointPolygon
      ? {
          pairPoints: createTransientView(graph, `${id}-pair-points`, 'float32x2', pairRowCount),
          pairGeometryOffsets: createTransientView(
            graph,
            `${id}-pair-geometry-offsets`,
            'uint32',
            pairRowCount + 1
          ),
          pairClassifications: createTransientView(
            graph,
            `${id}-pair-classifications`,
            'uint32',
            pairRowCount
          )
        }
      : undefined;

    // Clear state, candidate sentinels, match counts, and the robust pair layout.
    {
      const bindings: WGSLKernelBinding[] = [
        {name: 'state', view: state, type: 'u32', access: 'read_write'},
        {name: 'candidatePairs', view: candidatePairs, type: 'u32', access: 'read_write'}
      ];
      if (leftMatchCounts) {
        bindings.push({
          name: 'leftMatchCounts',
          view: leftMatchCounts,
          type: 'u32',
          access: 'read_write'
        });
      }
      if (robust) {
        bindings.push(
          {name: 'pairPoints', view: robust.pairPoints, type: 'f32', access: 'read_write'},
          {
            name: 'pairGeometryOffsets',
            view: robust.pairGeometryOffsets,
            type: 'u32',
            access: 'read_write'
          }
        );
      }
      const lengths = [
        4,
        candidateCapacity,
        leftMatchCounts?.length ?? 0,
        robust?.pairGeometryOffsets.length ?? 0
      ];
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-clear`,
          operation: OPERATION,
          variant: 'clear',
          bindings,
          invocationCount: Math.max(...lengths),
          declarations: SPATIAL_JOIN_WGSL_HELPERS,
          body: `if (index < 4u) { state[stateOffset + index] = 0u; }
  if (index < ${candidateCapacity}u) {
    candidatePairs[candidatePairsOffset + index * 2u] = NO_FEATURE;
    candidatePairs[candidatePairsOffset + index * 2u + 1u] = NO_FEATURE;
  }
  ${leftMatchCounts ? `if (index <= ${leftCount}u) { leftMatchCounts[leftMatchCountsOffset + index] = 0u; }` : ''}
  ${
    robust
      ? `if (index < ${robust.pairPoints.length}u) {
    // A runtime expression keeps the quiet-NaN bit pattern out of const evaluation.
    let quietNaN = bitcast<f32>(0x7fc00000u | (index & 0u));
    pairPoints[pairPointsOffset + index * 2u] = quietNaN;
    pairPoints[pairPointsOffset + index * 2u + 1u] = quietNaN;
  }
  if (index < ${robust.pairGeometryOffsets.length}u) {
    pairGeometryOffsets[pairGeometryOffsetsOffset + index] =
      select(0u, ${(polygonSide as {polygonOffsets: GraphDataView}).polygonOffsets.length - 1}u, index == ${robust.pairGeometryOffsets.length - 1}u);
  }`
      : ''
  }`
        })
      );
    }

    // Candidate generation: count per left feature, scan, then write in (left, right) order.
    const probeBindings = (extra: WGSLKernelBinding[]): WGSLKernelBinding[] => [
      {name: 'leftMinima', view: leftSide.minima, type: 'f32', access: 'read'},
      {name: 'leftMaxima', view: leftSide.maxima, type: 'f32', access: 'read'},
      {name: 'nodeMinima', view: bvh.nodeMinima, type: 'f32', access: 'read'},
      {name: 'nodeMaxima', view: bvh.nodeMaxima, type: 'f32', access: 'read'},
      {name: 'leafIds', view: bvh.leafIds, type: 'u32', access: 'read'},
      ...extra
    ];
    const margin = predicate === 'dwithin' ? Math.fround(props.distance as number) : 0;
    const probeDeclarations = `${SPATIAL_JOIN_WGSL_HELPERS}
const INTERNAL_NODE_COUNT: u32 = ${bvh.internalNodeCount}u;
const RIGHT_COUNT: u32 = ${rightCount}u;
const CANDIDATE_CAPACITY: u32 = ${candidateCapacity}u;
const MARGIN: f32 = ${formatFloat(margin)};
fn nodeOverlaps(node: u32, queryMinimum: vec2f, queryMaximum: vec2f) -> bool {
  let component = node * 2u;
  let minimum = vec2f(nodeMinima[nodeMinimaOffset + component], nodeMinima[nodeMinimaOffset + component + 1u]);
  let maximum = vec2f(nodeMaxima[nodeMaximaOffset + component], nodeMaxima[nodeMaximaOffset + component + 1u]);
  return all(minimum <= queryMaximum) && all(queryMinimum <= maximum);
}`;
    const probeBody = (
      leaf: string,
      prologue: string,
      epilogue: string
    ) => `let boxMinimum = vec2f(leftMinima[leftMinimaOffset + index * 2u], leftMinima[leftMinimaOffset + index * 2u + 1u]);
  let boxMaximum = vec2f(leftMaxima[leftMaximaOffset + index * 2u], leftMaxima[leftMaximaOffset + index * 2u + 1u]);
  ${prologue}
  if (boxMinimum.x <= boxMaximum.x && boxMinimum.y <= boxMaximum.y) {
    let queryMinimum = boxMinimum - vec2f(MARGIN);
    let queryMaximum = boxMaximum + vec2f(MARGIN);
    var node = 0u;
    loop {
      if (nodeOverlaps(node, queryMinimum, queryMaximum)) {
        if (node < INTERNAL_NODE_COUNT) {
          node = node * 2u + 1u;
          continue;
        }
        let rightRow = leafIds[leafIdsOffset + node - INTERNAL_NODE_COUNT];
        if (rightRow < RIGHT_COUNT) {
          ${leaf}
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
        id: `${id}-count-candidates`,
        operation: OPERATION,
        variant: 'count-candidates',
        bindings: probeBindings([
          {name: 'leftCounts', view: leftCounts, type: 'u32', access: 'read_write'}
        ]),
        invocationCount: leftCount,
        declarations: probeDeclarations,
        body: probeBody(
          'found = found + 1u;',
          'var found = 0u;',
          'leftCounts[leftCountsOffset + index] = found;'
        )
      })
    );
    nodes.push(
      ...new GPUScan({
        id: `${id}-scan-candidates`,
        input: leftCounts,
        output: leftOffsets,
        mode: 'exclusive'
      }).getCommandNodes(graph)
    );
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-total-candidates`,
        operation: OPERATION,
        variant: 'total-candidates',
        bindings: [
          {name: 'leftCounts', view: leftCounts, type: 'u32', access: 'read'},
          {name: 'leftOffsets', view: leftOffsets, type: 'u32', access: 'read'},
          {name: 'state', view: state, type: 'u32', access: 'read_write'}
        ],
        invocationCount: 1,
        body: `state[stateOffset] = leftOffsets[leftOffsetsOffset + ${leftCount - 1}u] + leftCounts[leftCountsOffset + ${leftCount - 1}u];`
      })
    );
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-write-candidates`,
        operation: OPERATION,
        variant: 'write-candidates',
        bindings: probeBindings([
          {name: 'leftOffsets', view: leftOffsets, type: 'u32', access: 'read'},
          {name: 'candidatePairs', view: candidatePairs, type: 'u32', access: 'read_write'}
        ]),
        invocationCount: leftCount,
        declarations: probeDeclarations,
        body: probeBody(
          `let slot = leftOffsets[leftOffsetsOffset + index] + found;
          if (slot < CANDIDATE_CAPACITY) {
            candidatePairs[candidatePairsOffset + slot * 2u] = index;
            candidatePairs[candidatePairsOffset + slot * 2u + 1u] = rightRow;
          }
          found = found + 1u;`,
          'var found = 0u;',
          ''
        )
      })
    );

    // Exact predicate per candidate slot.
    const sameRowTest = props.excludeSameRow ? ' && left != right' : '';
    const sideSpecs: SpatialPredicateSide[] = sides.map(side => ({
      prefix: side.name,
      kind: side.geometry.kind,
      vertexCount: side.geometry.positions.length
    }));
    const sideBindings: WGSLKernelBinding[] = [];
    for (const side of sides) {
      const prefix = side.name;
      sideBindings.push({
        name: `${prefix}Positions`,
        view: side.geometry.positions,
        type: 'f32',
        access: 'read'
      });
      if (side.geometry.kind === 'lines') {
        sideBindings.push({
          name: `${prefix}RingOffsets`,
          view: side.geometry.lineOffsets,
          type: 'u32',
          access: 'read'
        });
      } else if (side.geometry.kind === 'polygons') {
        sideBindings.push(
          {
            name: `${prefix}FeatureRings`,
            view: side.featureRings as GraphDataView,
            type: 'u32',
            access: 'read'
          },
          {
            name: `${prefix}RingOffsets`,
            view: side.geometry.ringOffsets,
            type: 'u32',
            access: 'read'
          }
        );
      }
    }
    if (robust) {
      const pointsOnLeft = left.kind === 'points';
      const pointGeometry = pointsOnLeft ? left : right;
      const polygonGeometry = (pointsOnLeft ? right : left) as Extract<
        GPUSpatialJoinGeometry,
        {kind: 'polygons'}
      >;
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-expand`,
          operation: OPERATION,
          variant: 'expand',
          bindings: [
            {name: 'candidatePairs', view: candidatePairs, type: 'u32', access: 'read'},
            {name: 'pointPositions', view: pointGeometry.positions, type: 'f32', access: 'read'},
            {
              name: 'featureOffsets',
              view: polygonGeometry.featureOffsets,
              type: 'u32',
              access: 'read'
            },
            {name: 'pairPoints', view: robust.pairPoints, type: 'f32', access: 'read_write'},
            {
              name: 'pairGeometryOffsets',
              view: robust.pairGeometryOffsets,
              type: 'u32',
              access: 'read_write'
            }
          ],
          invocationCount: candidateCapacity,
          declarations: SPATIAL_JOIN_WGSL_HELPERS,
          body: `let left = candidatePairs[candidatePairsOffset + index * 2u];
  if (left == NO_FEATURE) { return; }
  let right = candidatePairs[candidatePairsOffset + index * 2u + 1u];
  let pointRow = ${pointsOnLeft ? 'left' : 'right'};
  let polygonRow = ${pointsOnLeft ? 'right' : 'left'};
  let row = index * 2u + 1u;
  pairPoints[pairPointsOffset + row * 2u] = pointPositions[pointPositionsOffset + pointRow * 2u];
  pairPoints[pairPointsOffset + row * 2u + 1u] = pointPositions[pointPositionsOffset + pointRow * 2u + 1u];
  pairGeometryOffsets[pairGeometryOffsetsOffset + row] = featureOffsets[featureOffsetsOffset + polygonRow];
  pairGeometryOffsets[pairGeometryOffsetsOffset + row + 1u] = featureOffsets[featureOffsetsOffset + polygonRow + 1u];`
        })
      );
      nodes.push(
        ...captureGraphCommandNodes(graph, () =>
          new GPUPairwisePointInPolygon({
            id: `${id}-classify`,
            points: robust.pairPoints,
            polygonPositions: polygonGeometry.positions,
            geometryOffsets: robust.pairGeometryOffsets,
            polygonOffsets: polygonGeometry.polygonOffsets,
            ringOffsets: polygonGeometry.ringOffsets,
            output: robust.pairClassifications
          }).addToGraph(graph)
        )
      );
      // A point never contains a polygon and a polygon is never within a point; otherwise
      // `contains`/`within` need the point strictly inside and `intersects` accepts the boundary.
      const impossible =
        (predicate === 'contains' && pointsOnLeft) || (predicate === 'within' && !pointsOnLeft);
      const accepted =
        predicate === 'intersects'
          ? 'classification == INSIDE || classification == BOUNDARY'
          : 'classification == INSIDE';
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-resolve`,
          operation: OPERATION,
          variant: 'resolve',
          bindings: [
            {name: 'candidatePairs', view: candidatePairs, type: 'u32', access: 'read'},
            {
              name: 'pairClassifications',
              view: robust.pairClassifications,
              type: 'u32',
              access: 'read'
            },
            {name: 'flags', view: flags, type: 'u32', access: 'read_write'},
            {name: 'state', view: state, type: 'atomic<u32>', access: 'read_write'}
          ],
          invocationCount: candidateCapacity,
          declarations: `${SPATIAL_JOIN_WGSL_HELPERS}
const INSIDE: u32 = ${GPU_POINT_IN_POLYGON_CLASSIFICATION.inside}u;
const BOUNDARY: u32 = ${GPU_POINT_IN_POLYGON_CLASSIFICATION.boundary}u;
const UNCERTAIN: u32 = ${GPU_POINT_IN_POLYGON_CLASSIFICATION.uncertain}u;`,
          body: `let left = candidatePairs[candidatePairsOffset + index * 2u];
  var matched = false;
  if (left != NO_FEATURE) {
    let right = candidatePairs[candidatePairsOffset + index * 2u + 1u];
    let classification = pairClassifications[pairClassificationsOffset + index * 2u + 1u];
    if (classification == UNCERTAIN) {
      atomicAdd(&state[stateOffset + 1u], 1u);
    } else {
      matched = ${impossible ? 'false' : `(${accepted})`}${sameRowTest};
    }
  }
  flags[flagsOffset + index] = select(0u, 1u, matched);`
        })
      );
      // Candidates the robust classifier could not certify are decided by the f32 test.
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-fallback`,
          operation: OPERATION,
          variant: `fallback-${left.kind}-${right.kind}-${predicate}`,
          bindings: [
            ...sideBindings,
            {name: 'candidatePairs', view: candidatePairs, type: 'u32', access: 'read'},
            {
              name: 'pairClassifications',
              view: robust.pairClassifications,
              type: 'u32',
              access: 'read'
            },
            {name: 'flags', view: flags, type: 'u32', access: 'read_write'}
          ],
          invocationCount: candidateCapacity,
          declarations: `${getSpatialPredicateWGSL(sideSpecs[0], sideSpecs[1], predicate, 0)}
const UNCERTAIN: u32 = ${GPU_POINT_IN_POLYGON_CLASSIFICATION.uncertain}u;`,
          body: `let left = candidatePairs[candidatePairsOffset + index * 2u];
  if (left == NO_FEATURE) { return; }
  if (pairClassifications[pairClassificationsOffset + index * 2u + 1u] != UNCERTAIN) { return; }
  let right = candidatePairs[candidatePairsOffset + index * 2u + 1u];
  flags[flagsOffset + index] = select(0u, 1u, pairMatches(left, right)${sameRowTest});`
        })
      );
    } else {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-exact`,
          operation: OPERATION,
          variant: `exact-${left.kind}-${right.kind}-${predicate}`,
          bindings: [
            ...sideBindings,
            {name: 'candidatePairs', view: candidatePairs, type: 'u32', access: 'read'},
            {name: 'flags', view: flags, type: 'u32', access: 'read_write'}
          ],
          invocationCount: candidateCapacity,
          declarations: getSpatialPredicateWGSL(
            sideSpecs[0],
            sideSpecs[1],
            predicate,
            props.distance ?? 0
          ),
          body: `let left = candidatePairs[candidatePairsOffset + index * 2u];
  var matched = false;
  if (left != NO_FEATURE) {
    let right = candidatePairs[candidatePairsOffset + index * 2u + 1u];
    matched = pairMatches(left, right)${sameRowTest};
  }
  flags[flagsOffset + index] = select(0u, 1u, matched);`
        })
      );
    }

    // Stable compaction of flagged candidates into the (left, right) output.
    nodes.push(
      ...new GPUScan({
        id: `${id}-scan-flags`,
        input: flags,
        output: flagOffsets,
        mode: 'exclusive'
      }).getCommandNodes(graph)
    );
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-scatter`,
        operation: OPERATION,
        variant: 'scatter',
        bindings: [
          {name: 'candidatePairs', view: candidatePairs, type: 'u32', access: 'read'},
          {name: 'flags', view: flags, type: 'u32', access: 'read'},
          {name: 'flagOffsets', view: flagOffsets, type: 'u32', access: 'read'},
          {name: 'leftIds', view: leftIds, type: 'u32', access: 'read_write'},
          {name: 'rightIds', view: rightIds, type: 'u32', access: 'read_write'},
          {name: 'state', view: state, type: 'u32', access: 'read_write'}
        ],
        invocationCount: candidateCapacity,
        declarations: `const PAIR_CAPACITY: u32 = ${pairCapacity}u;`,
        body: `if (flags[flagsOffset + index] != 0u) {
    let slot = flagOffsets[flagOffsetsOffset + index];
    if (slot < PAIR_CAPACITY) {
      leftIds[leftIdsOffset + slot] = candidatePairs[candidatePairsOffset + index * 2u];
      rightIds[rightIdsOffset + slot] = candidatePairs[candidatePairsOffset + index * 2u + 1u];
    }
  }
  if (index == ${candidateCapacity - 1}u) {
    state[stateOffset + 2u] = flagOffsets[flagOffsetsOffset + index] + flags[flagsOffset + index];
  }`
      })
    );

    // Scalars.
    {
      const bindings: WGSLKernelBinding[] = [
        {name: 'state', view: state, type: 'u32', access: 'read'},
        {name: 'bvhOverflow', view: bvh.overflow, type: 'u32', access: 'read'}
      ];
      const scalars: [string, GraphDataView<'uint32'> | undefined][] = [
        ['overflow', props.overflow],
        ['pairsOverflow', pairs?.overflow],
        ['pairsCount', pairs?.count],
        ['pairsTotal', pairs?.totalCount],
        ['candidateCount', props.candidateCount],
        ['uncertainCount', props.uncertainCount]
      ];
      for (const [name, view] of scalars) {
        if (view) {
          bindings.push({name, view, type: 'u32', access: 'read_write'});
        }
      }
      const has = (name: string) => scalars.some(([key, view]) => key === name && view);
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-finalize`,
          operation: OPERATION,
          variant: 'finalize',
          bindings,
          invocationCount: 1,
          declarations: `const CANDIDATE_CAPACITY: u32 = ${candidateCapacity}u;
const PAIR_CAPACITY: u32 = ${pairCapacity}u;`,
          body: `let candidateTotal = state[stateOffset];
  let matchTotal = state[stateOffset + 2u];
  let overflowed = bvhOverflow[bvhOverflowOffset] != 0u || candidateTotal > CANDIDATE_CAPACITY || matchTotal > PAIR_CAPACITY;
  let overflowValue = select(0u, 1u, overflowed);
  ${has('overflow') ? 'overflow[overflowOffset] = overflowValue;' : ''}
  ${has('pairsOverflow') ? 'pairsOverflow[pairsOverflowOffset] = overflowValue;' : ''}
  ${has('pairsCount') ? 'pairsCount[pairsCountOffset] = min(matchTotal, PAIR_CAPACITY);' : ''}
  ${has('pairsTotal') ? 'pairsTotal[pairsTotalOffset] = matchTotal;' : ''}
  ${has('candidateCount') ? 'candidateCount[candidateCountOffset] = candidateTotal;' : ''}
  ${has('uncertainCount') ? 'uncertainCount[uncertainCountOffset] = state[stateOffset + 1u];' : ''}`
        })
      );
    }

    // Optional CSR: rows are left features and pairs are already sorted by left row.
    if (weights && leftMatchCounts) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-count-matches`,
          operation: OPERATION,
          variant: 'count-matches',
          bindings: [
            {name: 'leftIds', view: leftIds, type: 'u32', access: 'read'},
            {name: 'state', view: state, type: 'u32', access: 'read'},
            {
              name: 'leftMatchCounts',
              view: leftMatchCounts,
              type: 'atomic<u32>',
              access: 'read_write'
            }
          ],
          invocationCount: pairCapacity,
          declarations: `const PAIR_CAPACITY: u32 = ${pairCapacity}u;`,
          body: `if (index >= min(state[stateOffset + 2u], PAIR_CAPACITY)) { return; }
  atomicAdd(&leftMatchCounts[leftMatchCountsOffset + leftIds[leftIdsOffset + index]], 1u);`
        })
      );
      nodes.push(
        ...new GPUScan({
          id: `${id}-scan-matches`,
          input: leftMatchCounts,
          output: weights.offsets,
          mode: 'exclusive'
        }).getCommandNodes(graph)
      );
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-fill-weights`,
          operation: OPERATION,
          variant: 'fill-weights',
          bindings: [
            {name: 'rightIds', view: rightIds, type: 'u32', access: 'read'},
            {name: 'state', view: state, type: 'u32', access: 'read'},
            {name: 'neighbors', view: weights.neighbors, type: 'u32', access: 'read_write'},
            {name: 'weightValues', view: weights.weights, type: 'f32', access: 'read_write'}
          ],
          invocationCount: pairCapacity,
          declarations: `const PAIR_CAPACITY: u32 = ${pairCapacity}u;`,
          body: `if (index >= min(state[stateOffset + 2u], PAIR_CAPACITY)) { return; }
  neighbors[neighborsOffset + index] = rightIds[rightIdsOffset + index];
  weightValues[weightValuesOffset + index] = 1.0;`
        })
      );
    }
    return nodes;
  }
}

function formatFloat(value: number): string {
  const text = String(Math.fround(value));
  return /[.eE]/.test(text) ? text : `${text}.0`;
}

/** Writes `(ringStart, ringEnd)` per polygon feature; malformed offsets give an empty range. */
function createFeatureRingsNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  props: {
    featureCount: number;
    geometry: Extract<GPUSpatialJoinGeometry, {kind: 'polygons'}>;
    featureRings: GraphDataView<'uint32x2'>;
  }
): GPUCommandNode<Parameters> {
  const {geometry} = props;
  return createWGSLKernelNode<Parameters>(graph, {
    id,
    operation: OPERATION,
    variant: 'feature-rings',
    bindings: [
      {name: 'featureOffsets', view: geometry.featureOffsets, type: 'u32', access: 'read'},
      {name: 'polygonOffsets', view: geometry.polygonOffsets, type: 'u32', access: 'read'},
      {name: 'featureRings', view: props.featureRings, type: 'u32', access: 'read_write'}
    ],
    invocationCount: props.featureCount,
    declarations: `const POLYGON_COUNT: u32 = ${geometry.polygonOffsets.length - 1}u;
const RING_COUNT: u32 = ${geometry.ringOffsets.length - 1}u;`,
    body: `let polygonStart = featureOffsets[featureOffsetsOffset + index];
  let polygonEnd = featureOffsets[featureOffsetsOffset + index + 1u];
  var ringStart = 0u;
  var ringEnd = 0u;
  if (polygonStart <= polygonEnd && polygonEnd <= POLYGON_COUNT) {
    let first = polygonOffsets[polygonOffsetsOffset + polygonStart];
    let last = polygonOffsets[polygonOffsetsOffset + polygonEnd];
    if (first <= last && last <= RING_COUNT) { ringStart = first; ringEnd = last; }
  }
  featureRings[featureRingsOffset + index * 2u] = ringStart;
  featureRings[featureRingsOffset + index * 2u + 1u] = ringEnd;`
  });
}

/** Writes per-feature bounds for any geometry kind; empty features get inverted bounds. */
function createBoundsNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  props: {
    featureCount: number;
    geometry: GPUSpatialJoinGeometry;
    featureRings?: GraphDataView<'uint32x2'>;
    minima: GraphDataView<'float32x2'>;
    maxima: GraphDataView<'float32x2'>;
  }
): GPUCommandNode<Parameters> {
  const {geometry} = props;
  if (geometry.kind === 'points') {
    return createSpatialJoinBoundsNode<Parameters>(graph, {
      id,
      operation: OPERATION,
      featureCount: props.featureCount,
      source: {kind: 'points', positions: geometry.positions},
      minima: props.minima,
      maxima: props.maxima
    });
  }
  const bindings: WGSLKernelBinding[] = [
    {name: 'positions', view: geometry.positions, type: 'f32', access: 'read'}
  ];
  let range: string;
  if (geometry.kind === 'lines') {
    bindings.push({name: 'lineOffsets', view: geometry.lineOffsets, type: 'u32', access: 'read'});
    range = `let vertexStart = lineOffsets[lineOffsetsOffset + index];
  let vertexEnd = min(lineOffsets[lineOffsetsOffset + index + 1u], VERTEX_COUNT);`;
  } else {
    bindings.push(
      {
        name: 'featureRings',
        view: props.featureRings as GraphDataView,
        type: 'u32',
        access: 'read'
      },
      {name: 'ringOffsets', view: geometry.ringOffsets, type: 'u32', access: 'read'}
    );
    range = `let ringStart = featureRings[featureRingsOffset + index * 2u];
  let ringEnd = featureRings[featureRingsOffset + index * 2u + 1u];
  var vertexStart = 0u;
  var vertexEnd = 0u;
  if (ringStart < ringEnd) {
    vertexStart = ringOffsets[ringOffsetsOffset + ringStart];
    vertexEnd = min(ringOffsets[ringOffsetsOffset + ringEnd], VERTEX_COUNT);
  }`;
  }
  bindings.push(
    {name: 'featureMinima', view: props.minima, type: 'f32', access: 'read_write'},
    {name: 'featureMaxima', view: props.maxima, type: 'f32', access: 'read_write'}
  );
  return createWGSLKernelNode<Parameters>(graph, {
    id,
    operation: OPERATION,
    variant: `bounds-${geometry.kind}`,
    bindings,
    invocationCount: props.featureCount,
    declarations: `${SPATIAL_JOIN_WGSL_HELPERS}
const VERTEX_COUNT: u32 = ${geometry.positions.length}u;`,
    body: `${range}
  var minimum = vec2f(FLOAT32_MAXIMUM);
  var maximum = vec2f(-FLOAT32_MAXIMUM);
  for (var vertex = vertexStart; vertex < vertexEnd; vertex++) {
    let position = vec2f(positions[positionsOffset + vertex * 2u], positions[positionsOffset + vertex * 2u + 1u]);
    if (isFiniteValue(position.x) && isFiniteValue(position.y)) {
      minimum = min(minimum, position);
      maximum = max(maximum, position);
    }
  }
  featureMinima[featureMinimaOffset + index * 2u] = minimum.x;
  featureMinima[featureMinimaOffset + index * 2u + 1u] = minimum.y;
  featureMaxima[featureMaximaOffset + index * 2u] = maximum.x;
  featureMaxima[featureMaximaOffset + index * 2u + 1u] = maximum.y;`
  });
}
