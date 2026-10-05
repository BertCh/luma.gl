// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUScan,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {
  createFillNode,
  createWGSLKernelNode,
  createPublishNode,
  type WGSLKernelBinding
} from '../../utils/wgsl-kernel-nodes';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {validateGraphViewsBelongToGraph} from '../../utils/gpu-contributor-utils';
import {GPU_ISOLINES_PARAMETER_LENGTH} from './isolines-parameters';
import {
  getIsolinesCornersWGSL,
  validateIsolinesAliasing,
  validateIsolinesView
} from './isolines-utils';

const OPERATION = 'GPUIsolines';

/** Caller-owned, capacity-bounded segment outputs of {@link GPUIsolines}. */
export type GPUIsolinesOutput = {
  /**
   * `[x0, y0, x1, y1]` world coordinates per segment. The segment capacity is `segments.length`.
   * The high side (`value >= level`) is on the left when walking from p0 to p1 (x right, y up).
   */
  segments: GraphDataView<'float32x4'>;
  /** Level index (row of the `levels` view) per segment. At least `segments.length` rows. */
  segmentLevels: GraphDataView<'uint32'>;
  /**
   * Optional `[startEdge, endEdge]` per segment: the global ids of the raster edges that hold p0
   * and p1. Horizontal edge `(c, r) -> (c + 1, r)` is `r * (width - 1) + c`; vertical edge
   * `(c, r) -> (c, r + 1)` is `height * (width - 1) + r * width + c`.
   */
  segmentEdges?: GraphDataView<'uint32x2'>;
  /** One row receiving `min(totalCount, segments.length)`. */
  count: GraphDataView<'uint32'>;
  /** One row receiving 1 when segments were dropped for lack of capacity, otherwise 0. */
  overflow: GraphDataView<'uint32'>;
  /** Optional one row receiving the unclamped segment count. */
  totalCount?: GraphDataView<'uint32'>;
};

/**
 * Optional stitched polyline outputs of {@link GPUIsolines}. Presence of this object schedules the
 * stitching pass.
 *
 * Chains are formed per level. Open chains start at the segment with no predecessor. Closed rings
 * start at their smallest segment index and repeat the first vertex at the end. Polylines are
 * ordered by head segment index. Vertices are copies of segment endpoints. Polyline `p` spans
 * vertex rows `[polylineOffsets[p], polylineOffsets[p + 1])` and is complete: a polyline that does
 * not fit into `vertices` is dropped, with all later ones, and `overflow` is set. Worst case needs
 * `2 * segmentCapacity` vertices (every chain one segment long).
 *
 * If the segment output overflowed, no polyline is produced and `overflow` is set.
 */
export type GPUIsolinesPolylineOutput = {
  /** World positions; the vertex capacity is `vertices.length`. */
  vertices: GraphDataView<'float32x2'>;
  /** `segments.length + 1` rows. Rows `0..polylineCount` are the vertex offsets of each polyline. */
  polylineOffsets: GraphDataView<'uint32'>;
  /** Level index per polyline, at least `segments.length` rows. */
  polylineLevels: GraphDataView<'uint32'>;
  /** 1 for a closed ring, 0 for an open chain, at least `segments.length` rows. */
  polylineClosed: GraphDataView<'uint32'>;
  /** One row receiving the number of complete polylines written. */
  polylineCount: GraphDataView<'uint32'>;
  /** One row receiving the number of vertices written (`polylineOffsets[polylineCount]`). */
  vertexCount: GraphDataView<'uint32'>;
  /** One row receiving 1 when segments or polylines were dropped, otherwise 0. */
  overflow: GraphDataView<'uint32'>;
};

/**
 * Properties for {@link GPUIsolines}.
 *
 * Topology: `width`, `height`, `levels.length` (the maximum level count), the segment and vertex
 * capacities, `noDataValue`, and which optional views exist. Per-frame (no recompile): raster and
 * `levels` contents, and `parameters` (`levelCount`, extent).
 */
export type GPUIsolinesProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'isolines'`. */
  id?: string;
  /** Raster width in samples, at least 2. */
  width: number;
  /** Raster height in samples, at least 2. */
  height: number;
  /** Packed row-major float32 raster of at least `width * height` rows. NaN is nodata. */
  values: GraphDataView<'float32'>;
  /** Optional nonzero-is-valid mask over the raster. */
  validity?: GraphDataView<'uint32'>;
  /** Optional finite sentinel treated as nodata. */
  noDataValue?: number;
  /** Level values; rows `[0, levelCount)` are active. Contents may change per frame. */
  levels: GraphDataView<'float32'>;
  /** Per-frame float32 view written with `getGPUIsolinesParameterValues`. */
  parameters: GraphDataView<'float32'>;
  /** Caller-owned segment outputs. */
  output: GPUIsolinesOutput;
  /** Optional polyline outputs; schedules stitching. */
  polylines?: GPUIsolinesPolylineOutput;
};

/**
 * Marching-squares isolines of a raster at several levels, as ordered segments and optionally
 * stitched polylines.
 *
 * A cell with a nodata corner emits nothing. Corners with `value >= level` are high; saddles are
 * resolved by the cell centre average. Segments are ordered by (cell index, level index, slot), so
 * output is deterministic. The pipeline is count, `GPUScan`, scatter, publish, and then for
 * polylines a link pass, a fixed number of pointer-jumping rounds, scans and a gather. Changing
 * levels, `levelCount`, or the extent never recompiles.
 */
export class GPUIsolines implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUIsolinesProps;
  /** Cells read one sample right and above. */
  readonly requiredHalo = 1;
  /** Segment capacity (`output.segments.length`). */
  readonly segmentCapacity: number;
  /** Maximum number of levels (`levels.length`). */
  readonly maximumLevelCount: number;

  constructor(props: GPUIsolinesProps) {
    this.id = props.id ?? 'isolines';
    this.props = props;
    const {id} = this;
    const {width, height, output, polylines} = props;
    for (const [name, value] of [
      ['width', width],
      ['height', height]
    ] as const) {
      if (!Number.isSafeInteger(value) || value < 2) {
        throw new Error(`${id} ${name} must be an integer of at least 2`);
      }
    }
    if (width * height > 0xffffffff / 2) {
      throw new Error(`${id} raster is too large for 32-bit edge identifiers`);
    }
    if (props.noDataValue !== undefined && !Number.isFinite(props.noDataValue)) {
      throw new Error(`${id} noDataValue must be finite (NaN cells are always nodata)`);
    }
    validateIsolinesView(id, 'values', props.values, 'float32', width * height);
    validateIsolinesView(id, 'validity', props.validity, 'uint32', width * height);
    validateIsolinesView(id, 'levels', props.levels, 'float32', 1);
    validateIsolinesView(
      id,
      'parameters',
      props.parameters,
      'float32',
      GPU_ISOLINES_PARAMETER_LENGTH
    );
    if (!output?.segments) {
      throw new Error(`${id} needs output.segments`);
    }
    this.maximumLevelCount = props.levels.length;
    this.segmentCapacity = output.segments.length;
    const capacity = this.segmentCapacity;
    if (capacity < 1) {
      throw new Error(`${id} output.segments must hold at least one row`);
    }
    validateIsolinesView(id, 'output.segments', output.segments, 'float32x4', 1);
    validateIsolinesView(id, 'output.segmentLevels', output.segmentLevels, 'uint32', capacity);
    validateIsolinesView(id, 'output.segmentEdges', output.segmentEdges, 'uint32x2', capacity);
    for (const [name, scalar] of [
      ['count', output.count],
      ['overflow', output.overflow],
      ['totalCount', output.totalCount]
    ] as const) {
      if (name !== 'totalCount' && !scalar) {
        throw new Error(`${id} needs output.${name}`);
      }
      validateIsolinesView(id, `output.${name}`, scalar, 'uint32', 1);
    }
    if (polylines) {
      validateIsolinesView(id, 'polylines.vertices', polylines.vertices, 'float32x2', 1);
      validateIsolinesView(
        id,
        'polylines.polylineOffsets',
        polylines.polylineOffsets,
        'uint32',
        capacity + 1
      );
      validateIsolinesView(
        id,
        'polylines.polylineLevels',
        polylines.polylineLevels,
        'uint32',
        capacity
      );
      validateIsolinesView(
        id,
        'polylines.polylineClosed',
        polylines.polylineClosed,
        'uint32',
        capacity
      );
      for (const name of ['polylineCount', 'vertexCount', 'overflow'] as const) {
        if (!polylines[name]) {
          throw new Error(`${id} needs polylines.${name}`);
        }
        validateIsolinesView(id, `polylines.${name}`, polylines[name], 'uint32', 1);
      }
    }
    validateIsolinesAliasing(
      id,
      [...this.getOutputViews()],
      [props.values, props.validity, props.levels, props.parameters]
    );
  }

  private getOutputViews(): GraphDataView[] {
    const {output, polylines} = this.props;
    return [
      output.segments,
      output.segmentLevels,
      output.segmentEdges,
      output.count,
      output.overflow,
      output.totalCount,
      polylines?.vertices,
      polylines?.polylineOffsets,
      polylines?.polylineLevels,
      polylines?.polylineClosed,
      polylines?.polylineCount,
      polylines?.vertexCount,
      polylines?.overflow
    ].filter(view => view !== undefined);
  }

  /** Returns count, scan, scatter and publish nodes, plus stitching nodes with `polylines`. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, segmentCapacity, maximumLevelCount} = this;
    const {width, height, output, polylines} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      props.values,
      props.validity,
      props.levels,
      props.parameters,
      ...this.getOutputViews()
    ]);
    const nodes: GPUCommandNode<Parameters>[] = [];
    const cellCount = (width - 1) * (height - 1);
    const hasValidity = Boolean(props.validity);
    const common = `const WIDTH: u32 = ${width}u;
const HEIGHT: u32 = ${height}u;
const MAXIMUM_LEVELS: u32 = ${maximumLevelCount}u;
const CAPACITY: u32 = ${segmentCapacity}u;
${getIsolinesCornersWGSL(hasValidity, props.noDataValue)}`;
    const rasterBindings: WGSLKernelBinding[] = [
      {name: 'params', view: props.parameters, type: 'f32', access: 'read'},
      {name: 'levels', view: props.levels, type: 'f32', access: 'read'},
      {name: 'raster', view: props.values, type: 'f32', access: 'read'},
      ...(props.validity
        ? [{name: 'validity', view: props.validity, type: 'u32', access: 'read'} as const]
        : [])
    ];

    const counts = createTransientView(graph, `${id}-cell-counts`, 'uint32', cellCount);
    const offsets = createTransientView(graph, `${id}-cell-offsets`, 'uint32', cellCount);
    const total = createTransientView(graph, `${id}-total`, 'uint32', 1);
    const needsEdges = Boolean(polylines);
    const edges =
      output.segmentEdges ??
      (needsEdges
        ? createTransientView(graph, `${id}-edges`, 'uint32x2', segmentCapacity)
        : undefined);

    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-count`,
        operation: OPERATION,
        variant: 'count',
        bindings: [
          ...rasterBindings,
          {name: 'counts', view: counts, type: 'u32', access: 'read_write'}
        ],
        invocationCount: cellCount,
        declarations: common,
        body: `let cx = index % (WIDTH - 1u);
  let cy = index / (WIDTH - 1u);
  let corners = loadCorners(cx, cy);
  var cellTotal = 0u;
  if (corners.valid) {
    let levelCount = min(u32(max(params[paramsOffset], 0.0)), MAXIMUM_LEVELS);
    for (var levelIndex = 0u; levelIndex < levelCount; levelIndex++) {
      let mask = getHighMask(corners.v, levels[levelsOffset + levelIndex]);
      cellTotal += countOneBits(getExitMask(mask));
    }
  }
  counts[countsOffset + index] = cellTotal;`
      }),
      ...new GPUScan({id: `${id}-cell-scan`, input: counts, output: offsets}).getCommandNodes(
        graph
      ),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-total`,
        operation: OPERATION,
        variant: 'total',
        bindings: [
          {name: 'counts', view: counts, type: 'u32', access: 'read'},
          {name: 'offsets', view: offsets, type: 'u32', access: 'read'},
          {name: 'totalOut', view: total, type: 'u32', access: 'read_write'}
        ],
        invocationCount: 1,
        body: `totalOut[totalOutOffset] =
    offsets[offsetsOffset + ${cellCount - 1}u] + counts[countsOffset + ${cellCount - 1}u];`
      })
    );

    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-scatter`,
        operation: OPERATION,
        variant: 'scatter',
        bindings: [
          ...rasterBindings,
          {name: 'offsets', view: offsets, type: 'u32', access: 'read'},
          {name: 'segments', view: output.segments, type: 'f32', access: 'read_write'},
          {name: 'segmentLevels', view: output.segmentLevels, type: 'u32', access: 'read_write'},
          ...(edges
            ? [{name: 'edges', view: edges, type: 'u32', access: 'read_write'} as const]
            : [])
        ],
        invocationCount: cellCount,
        declarations: `${common}
${getCrossingWGSL()}`,
        body: `let cx = index % (WIDTH - 1u);
  let cy = index / (WIDTH - 1u);
  let corners = loadCorners(cx, cy);
  if (!corners.valid) {
    return;
  }
  let v = corners.v;
  var position = offsets[offsetsOffset + index];
  let levelCount = min(u32(max(params[paramsOffset], 0.0)), MAXIMUM_LEVELS);
  for (var levelIndex = 0u; levelIndex < levelCount; levelIndex++) {
    let level = levels[levelsOffset + levelIndex];
    let mask = getHighMask(v, level);
    let exits = getExitMask(mask);
    if (exits == 0u) {
      continue;
    }
    let entries = getEntryMask(mask);
    // Saddles have two exits. High corners are joined when the centre is high.
    let centre = ((v.x + v.y) + (v.z + v.w)) * 0.25;
    let isJoined = countOneBits(exits) < 2u || centre >= level;
    for (var exitEdge = 0u; exitEdge < 4u; exitEdge++) {
      if (((exits >> exitEdge) & 1u) == 0u) {
        continue;
      }
      var entryEdge = exitEdge;
      for (var step = 1u; step < 4u; step++) {
        let candidate = select((exitEdge + 4u - step) % 4u, (exitEdge + step) % 4u, isJoined);
        if (((entries >> candidate) & 1u) != 0u) {
          entryEdge = candidate;
          break;
        }
      }
      if (position < CAPACITY) {
        let start = getEdgeCrossing(exitEdge, level, v, cx, cy);
        let end = getEdgeCrossing(entryEdge, level, v, cx, cy);
        let base = segmentsOffset + 4u * position;
        segments[base] = start.x;
        segments[base + 1u] = start.y;
        segments[base + 2u] = end.x;
        segments[base + 3u] = end.y;
        segmentLevels[segmentLevelsOffset + position] = levelIndex;
        ${
          edges
            ? `edges[edgesOffset + 2u * position] = getEdgeId(exitEdge, cx, cy);
        edges[edgesOffset + 2u * position + 1u] = getEdgeId(entryEdge, cx, cy);`
            : ''
        }
      }
      position++;
    }
  }`
      }),
      // Segment levels double as the compact ids: segments are already in place.
      createPublishNode<Parameters>(graph, {
        id: `${id}-publish`,
        operation: OPERATION,
        totalCount: total,
        output: {
          ids: output.segmentLevels,
          count: output.count,
          overflow: output.overflow,
          totalCount: output.totalCount
        }
      })
    );

    if (polylines && edges) {
      nodes.push(
        ...this.getStitchNodes(graph, {
          counts,
          offsets,
          edges,
          polylines,
          common: getStitchWGSL(width, height)
        })
      );
    }
    return nodes;
  }

  private getStitchNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    context: {
      counts: GraphDataView<'uint32'>;
      offsets: GraphDataView<'uint32'>;
      edges: GraphDataView<'uint32x2'>;
      polylines: GPUIsolinesPolylineOutput;
      common: string;
    }
  ): GPUCommandNode<Parameters>[] {
    const {id, props, segmentCapacity: capacity} = this;
    const {output} = props;
    const {counts, offsets, edges, polylines} = context;
    const nodes: GPUCommandNode<Parameters>[] = [];
    const transient = (name: string, length: number = capacity) =>
      createTransientView(graph, `${id}-${name}`, 'uint32', length);
    const kernel = (
      step: string,
      bindings: WGSLKernelBinding[],
      body: string,
      declarations: string = '',
      invocationCount: number = capacity
    ) =>
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-${step}`,
          operation: OPERATION,
          variant: step,
          bindings,
          invocationCount,
          declarations: `const CAPACITY: u32 = ${capacity}u;
${context.common}
${declarations}`,
          body
        })
      );
    const read = (name: string, view: GraphDataView): WGSLKernelBinding => ({
      name,
      view,
      type: 'u32',
      access: 'read'
    });
    const write = (name: string, view: GraphDataView): WGSLKernelBinding => ({
      name,
      view,
      type: 'u32',
      access: 'read_write'
    });

    // Active segment count: zero when the segment output overflowed (documented behaviour).
    const active = transient('active', 1);
    kernel(
      'stitch-active',
      [
        read('countIn', output.count),
        read('overflowIn', output.overflow),
        write('activeOut', active)
      ],
      `activeOut[activeOutOffset] =
    select(min(countIn[countInOffset], CAPACITY), 0u, overflowIn[overflowInOffset] != 0u);`,
      '',
      1
    );

    // Predecessor link: the same-level segment in the neighbour cell across the start edge whose
    // end edge is that edge. Successors follow from ranks, so no successor array is needed.
    const previous = transient('previous');
    kernel(
      'links',
      [
        read('edges', edges),
        read('levels', output.segmentLevels),
        read('offsets', offsets),
        read('counts', counts),
        read('activeCount', active),
        write('previousOut', previous)
      ],
      `var previousSegment = NONE;
  if (index < activeCount[activeCountOffset]) {
    let startEdge = edges[edgesOffset + 2u * index];
    let endEdge = edges[edgesOffset + 2u * index + 1u];
    let level = levels[levelsOffset + index];
    previousSegment = findLinkedSegment(getNeighbourCell(endEdge, startEdge), level, 1u, startEdge);
  }
  previousOut[previousOutOffset + index] = previousSegment;`,
      getLinkWGSL()
    );

    // Pointer jumping on the predecessor links with a fixed round count. Each node tracks its
    // jump pointer, the smallest key seen (heads of open chains have key 0, every other node
    // index + 1, so a ring converges on its smallest index), the distance to that node, and the
    // number of steps its pointer spans.
    const roundCount = Math.ceil(Math.log2(capacity)) + 1;
    const stateNames = ['pointer', 'key', 'distance', 'span'] as const;
    const states = [0, 1].map(
      copy =>
        Object.fromEntries(
          stateNames.map(name => [name, transient(`jump-${copy}-${name}`)])
        ) as Record<(typeof stateNames)[number], GraphDataView<'uint32'>>
    );
    kernel(
      'jump-init',
      [read('previous', previous), ...stateNames.map(name => write(`${name}Out`, states[0][name]))],
      `let hasPrevious = previous[previousOffset + index] != NONE;
  pointerOut[pointerOutOffset + index] =
    select(index, previous[previousOffset + index], hasPrevious);
  keyOut[keyOutOffset + index] = select(0u, index + 1u, hasPrevious);
  distanceOut[distanceOutOffset + index] = 0u;
  spanOut[spanOutOffset + index] = select(0u, 1u, hasPrevious);`
    );
    for (let round = 0; round < roundCount; round++) {
      const from = states[round % 2];
      const to = states[(round + 1) % 2];
      kernel(
        `jump-${round}`,
        [
          ...stateNames.map(name => read(`${name}In`, from[name])),
          ...stateNames.map(name => write(`${name}Out`, to[name]))
        ],
        `let jump = pointerIn[pointerInOffset + index];
  let ownKey = keyIn[keyInOffset + index];
  let jumpKey = keyIn[keyInOffset + jump];
  let ownWins = ownKey <= jumpKey;
  keyOut[keyOutOffset + index] = select(jumpKey, ownKey, ownWins);
  distanceOut[distanceOutOffset + index] = select(
    distanceIn[distanceInOffset + jump] + spanIn[spanInOffset + index],
    distanceIn[distanceInOffset + index],
    ownWins
  );
  spanOut[spanOutOffset + index] =
    spanIn[spanInOffset + index] + spanIn[spanInOffset + jump];
  pointerOut[pointerOutOffset + index] = pointerIn[pointerInOffset + jump];`
      );
    }
    const final = states[roundCount % 2];

    // Head and rank per segment; bit 31 of the rank marks a closed ring.
    const headOf = transient('head-of');
    const rankOf = transient('rank-of');
    kernel(
      'resolve',
      [
        read('pointerIn', final.pointer),
        read('keyIn', final.key),
        read('distanceIn', final.distance),
        write('headOut', headOf),
        write('rankOut', rankOf)
      ],
      `let key = keyIn[keyInOffset + index];
  let isRing = key != 0u;
  headOut[headOutOffset + index] = select(pointerIn[pointerInOffset + index], key - 1u, isRing);
  rankOut[rankOutOffset + index] =
    distanceIn[distanceInOffset + index] | select(0u, 0x80000000u, isRing);`
    );

    // Chain lengths via integer atomicMax, then head flags (1 open, 2 closed).
    const chainLength = transient('chain-length');
    const headKind = transient('head-kind');
    nodes.push(
      createFillNode<Parameters>(graph, {
        id: `${id}-chain-length-clear`,
        operation: OPERATION,
        view: chainLength,
        type: 'u32',
        value: '0u'
      })
    );
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-chain-length`,
        operation: OPERATION,
        variant: 'chain-length',
        bindings: [
          read('headOf', headOf),
          read('rankOf', rankOf),
          read('activeCount', active),
          {name: 'chainLength', view: chainLength, type: 'atomic<u32>', access: 'read_write'},
          write('headKindOut', headKind)
        ],
        invocationCount: capacity,
        body: `var kind = 0u;
  if (index < activeCount[activeCountOffset]) {
    let packed = rankOf[rankOfOffset + index];
    let rank = packed & 0x7fffffffu;
    atomicMax(&chainLength[chainLengthOffset + headOf[headOfOffset + index]], rank + 1u);
    kind = select(0u, select(1u, 2u, (packed >> 31u) != 0u), rank == 0u);
  }
  headKindOut[headKindOutOffset + index] = kind;`
      })
    );

    const headFlag = transient('head-flag');
    const vertexCountPerHead = transient('head-vertex-count');
    const headIndex = transient('head-index');
    const vertexOffset = transient('vertex-offset');
    kernel(
      'scan-input',
      [
        read('headKind', headKind),
        read('chainLength', chainLength),
        write('flagOut', headFlag),
        write('vertexCountOut', vertexCountPerHead)
      ],
      `let isHead = headKind[headKindOffset + index] != 0u;
  flagOut[flagOutOffset + index] = select(0u, 1u, isHead);
  vertexCountOut[vertexCountOutOffset + index] =
    select(0u, chainLength[chainLengthOffset + index] + 1u, isHead);`
    );
    nodes.push(
      ...new GPUScan({
        id: `${id}-head-scan`,
        input: headFlag,
        output: headIndex
      }).getCommandNodes(graph),
      ...new GPUScan({
        id: `${id}-vertex-scan`,
        input: vertexCountPerHead,
        output: vertexOffset
      }).getCommandNodes(graph)
    );

    kernel(
      'polyline-records',
      [
        read('headKind', headKind),
        read('headIndex', headIndex),
        read('vertexOffset', vertexOffset),
        read('segmentLevels', output.segmentLevels),
        write('polylineOffsets', polylines.polylineOffsets),
        write('polylineLevels', polylines.polylineLevels),
        write('polylineClosed', polylines.polylineClosed)
      ],
      `let kind = headKind[headKindOffset + index];
  if (kind != 0u) {
    let polyline = headIndex[headIndexOffset + index];
    polylineOffsets[polylineOffsetsOffset + polyline] = vertexOffset[vertexOffsetOffset + index];
    polylineLevels[polylineLevelsOffset + polyline] = segmentLevels[segmentLevelsOffset + index];
    polylineClosed[polylineClosedOffset + polyline] = select(0u, 1u, kind == 2u);
  }`
    );

    const vertexCapacity = polylines.vertices.length;
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-vertices`,
        operation: OPERATION,
        variant: 'vertices',
        bindings: [
          read('headOf', headOf),
          read('rankOf', rankOf),
          read('activeCount', active),
          read('chainLength', chainLength),
          read('vertexOffset', vertexOffset),
          {name: 'segments', view: output.segments, type: 'f32', access: 'read'},
          {name: 'vertices', view: polylines.vertices, type: 'f32', access: 'read_write'}
        ],
        invocationCount: capacity,
        declarations: `const VERTEX_CAPACITY: u32 = ${vertexCapacity}u;`,
        body: `if (index >= activeCount[activeCountOffset]) {
    return;
  }
  let head = headOf[headOfOffset + index];
  let packed = rankOf[rankOfOffset + index];
  let rank = packed & 0x7fffffffu;
  let length = chainLength[chainLengthOffset + head];
  let base = vertexOffset[vertexOffsetOffset + head];
  if (base + length + 1u > VERTEX_CAPACITY) {
    return;
  }
  let own = segmentsOffset + 4u * index;
  let slot = verticesOffset + 2u * (base + rank);
  vertices[slot] = segments[own];
  vertices[slot + 1u] = segments[own + 1u];
  if (rank + 1u == length) {
    // The last vertex of a ring repeats the first one; an open chain ends at its last p1.
    let closing = (packed >> 31u) != 0u;
    let source = select(own + 2u, segmentsOffset + 4u * head, closing);
    vertices[slot + 2u] = segments[source];
    vertices[slot + 3u] = segments[source + 1u];
  }`
      })
    );

    const totals = transient('stitch-totals', 2);
    kernel(
      'stitch-totals',
      [
        read('headIndex', headIndex),
        read('headFlag', headFlag),
        read('vertexOffset', vertexOffset),
        read('vertexCount', vertexCountPerHead),
        write('totalsOut', totals)
      ],
      `totalsOut[totalsOutOffset] =
    headIndex[headIndexOffset + CAPACITY - 1u] + headFlag[headFlagOffset + CAPACITY - 1u];
  totalsOut[totalsOutOffset + 1u] =
    vertexOffset[vertexOffsetOffset + CAPACITY - 1u] +
    vertexCount[vertexCountOffset + CAPACITY - 1u];`,
      '',
      1
    );
    kernel(
      'stitch-publish',
      [
        read('totals', totals),
        read('segmentOverflow', output.overflow),
        write('polylineOffsets', polylines.polylineOffsets),
        write('polylineCountOut', polylines.polylineCount),
        write('vertexCountOut', polylines.vertexCount),
        write('overflowOut', polylines.overflow)
      ],
      `let polylineTotal = totals[totalsOffset];
  let vertexTotal = totals[totalsOffset + 1u];
  polylineOffsets[polylineOffsetsOffset + polylineTotal] = vertexTotal;
  // Offsets are nondecreasing: keep the longest prefix of polylines that fits the vertex capacity.
  var low = 0u;
  var high = polylineTotal;
  loop {
    if (low >= high) {
      break;
    }
    let middle = (low + high + 1u) / 2u;
    if (polylineOffsets[polylineOffsetsOffset + middle] <= VERTEX_CAPACITY) {
      low = middle;
    } else {
      high = middle - 1u;
    }
  }
  polylineCountOut[polylineCountOutOffset] = low;
  vertexCountOut[vertexCountOutOffset] = polylineOffsets[polylineOffsetsOffset + low];
  overflowOut[overflowOutOffset] =
    select(0u, 1u, segmentOverflow[segmentOverflowOffset] != 0u || low < polylineTotal);`,
      `const VERTEX_CAPACITY: u32 = ${vertexCapacity}u;`,
      1
    );
    return nodes;
  }
}

/** WGSL `getEdgeCrossing(k, level, v, cx, cy)` in world coordinates, canonical edge direction. */
function getCrossingWGSL(): string {
  return /* wgsl */ `
fn getEdgeCrossing(k: u32, level: f32, v: vec4<f32>, cx: u32, cy: u32) -> vec2<f32> {
  let minX = params[paramsOffset + 1u];
  let minY = params[paramsOffset + 2u];
  let cellWidth = params[paramsOffset + 3u];
  let cellHeight = params[paramsOffset + 4u];
  // Canonical orientation: horizontal edges run +x, vertical edges run +y.
  var a = v.x;
  var b = v.y;
  if (k == 1u) {
    a = v.y;
    b = v.z;
  } else if (k == 2u) {
    a = v.w;
    b = v.z;
  } else if (k == 3u) {
    a = v.x;
    b = v.w;
  }
  let t = (level - a) / (b - a);
  if (k == 0u || k == 2u) {
    let gridX = f32(cx) + t;
    let gridY = f32(cy + select(0u, 1u, k == 2u));
    return vec2<f32>(minX + (gridX + 0.5) * cellWidth, minY + (gridY + 0.5) * cellHeight);
  }
  let gridX = f32(cx + select(0u, 1u, k == 1u));
  let gridY = f32(cy) + t;
  return vec2<f32>(minX + (gridX + 0.5) * cellWidth, minY + (gridY + 0.5) * cellHeight);
}`;
}

/** WGSL edge/cell topology helpers shared by the stitching kernels. */
function getStitchWGSL(width: number, height: number): string {
  return /* wgsl */ `
const WIDTH: u32 = ${width}u;
const HEIGHT: u32 = ${height}u;
const NONE: u32 = 0xffffffffu;

// The two cells that share edge e: x is below or left of it, y above or right of it.
fn getEdgeCells(e: u32) -> vec2<u32> {
  let cellColumns = WIDTH - 1u;
  let cellRows = HEIGHT - 1u;
  let horizontalCount = HEIGHT * cellColumns;
  var low = NONE;
  var high = NONE;
  if (e < horizontalCount) {
    let r = e / cellColumns;
    let c = e % cellColumns;
    if (r >= 1u) { low = (r - 1u) * cellColumns + c; }
    if (r < cellRows) { high = r * cellColumns + c; }
  } else {
    let i = e - horizontalCount;
    let r = i / WIDTH;
    let c = i % WIDTH;
    if (c >= 1u && r < cellRows) { low = r * cellColumns + c - 1u; }
    if (c < cellColumns && r < cellRows) { high = r * cellColumns + c; }
  }
  return vec2<u32>(low, high);
}

fn cellHasEdge(cell: u32, e: u32) -> bool {
  let cellColumns = WIDTH - 1u;
  let cx = cell % cellColumns;
  let cy = cell / cellColumns;
  let bottom = cy * cellColumns + cx;
  let left = HEIGHT * cellColumns + cy * WIDTH + cx;
  return e == bottom || e == bottom + cellColumns || e == left || e == left + 1u;
}

// Cell on the other side of edge \`across\`, given another edge of the segment's own cell.
fn getNeighbourCell(known: u32, across: u32) -> u32 {
  let cells = getEdgeCells(across);
  if (cells.x != NONE && cellHasEdge(cells.x, known)) {
    return cells.y;
  }
  return cells.x;
}`;
}

function getLinkWGSL(): string {
  return /* wgsl */ `
// Segment of \`level\` in \`cell\` whose start (field 0) or end (field 1) edge is \`edge\`.
fn findLinkedSegment(cell: u32, level: u32, field: u32, edge: u32) -> u32 {
  if (cell == NONE) {
    return NONE;
  }
  let first = offsets[offsetsOffset + cell];
  let segmentCount = counts[countsOffset + cell];
  for (var i = 0u; i < segmentCount; i++) {
    let position = first + i;
    if (position >= CAPACITY) {
      break;
    }
    if (levels[levelsOffset + position] == level &&
        edges[edgesOffset + 2u * position + field] == edge) {
      return position;
    }
  }
  return NONE;
}`;
}
