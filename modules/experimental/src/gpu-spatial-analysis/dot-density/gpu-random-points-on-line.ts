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
  createFillNode,
  createPublishNode,
  createWGSLKernelNode
} from '../../utils/wgsl-kernel-nodes';
import type {GPUCompactOutput} from '../../utils/gpu-contributor-types';
import {
  validateCompactOutput,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {getDotSlotCountLimit} from './dot-density-cpu';
import {GPU_DOT_DENSITY_PARAMETER_LENGTH} from './dot-density-parameters';
import {PHILOX_WGSL} from './dot-density-random';

const OPERATION = 'GPURandomPointsOnLine';
const NONE = '0xffffffffu';

/** Philox key word 1 of the line-position draw (1 and 2 are the dot density draws). */
export const RANDOM_POINTS_ON_LINE_PURPOSE = 3;

/** Caller-owned outputs of {@link GPURandomPointsOnLine}. */
export type GPURandomPointsOnLineOutput = {
  /** Point positions, capacity rows. Rows past `points.count` are NaN. */
  positions: GraphDataView<'float32x2'>;
  /**
   * Compact point list: `ids` holds the line row of each point (capacity rows, `0xffffffff` past
   * the count), `count = min(total, capacity)` is safe as an instance count, and `overflow` is 1
   * when the total exceeds the capacity or a per-line count was clamped.
   */
  points: GPUCompactOutput;
  /**
   * Optional position of each point along its line as a fraction `[0, 1)` of the line's length,
   * capacity rows. Rows past `points.count` are NaN.
   */
  fractions?: GraphDataView<'float32'>;
};

/** Properties for {@link GPURandomPointsOnLine}. */
export type GPURandomPointsOnLineProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'random-points-on-line'`. */
  id?: string;
  /** Vertices of every input linestring, concatenated. */
  positions: GraphDataView<'float32x2'>;
  /** Line-to-vertex offsets with `lineCount + 1` rows (GeoArrow layout). */
  lineOffsets: GraphDataView<'uint32'>;
  /** Number of points to place on each line, one row per line. */
  counts: GraphDataView<'uint32'>;
  /** Per-frame uint32 parameters written with `getGPUDotDensityParameterValues` (only `seed` is read). */
  parameters: GraphDataView<'uint32'>;
  /** Caller-owned outputs. The capacity is `output.positions.length`. */
  output: GPURandomPointsOnLineOutput;
};

/**
 * Places `counts[line]` uniformly random points along each linestring, like GeoPandas
 * `GeoSeries.sample_points(size)` on lines (uniform in arc length) and the line counterpart of
 * {@link GPURandomPointsInPolygon}.
 *
 * Point `j` of line `l` sits at arc length `u * length(l)` with
 * `u = philox((l, j, 0, 0), (seed, 3))`, so it depends on neither the other lines nor the count:
 * raising a count only appends points and existing points never move (a stable prefix). Lines
 * with fewer than two vertices, zero or non-finite length draw no points. Segment lengths are
 * summed in vertex order by one thread per line, so results are bitwise reproducible. Output order
 * is by line, then rank, with offsets from an exclusive prefix scan, and capacity, count and
 * overflow semantics are those of {@link GPURandomPointsInPolygon}.
 *
 * Precision: lengths and interpolation are f32 in the input coordinates (planar, not geodesic);
 * use extent-relative coordinates for long lines far from the origin.
 */
export class GPURandomPointsOnLine implements GPUCommandNodeProducer {
  /** Prefix for graph node and transient IDs. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPURandomPointsOnLineProps;
  /** Number of input lines. */
  readonly lineCount: number;

  constructor(props: GPURandomPointsOnLineProps) {
    this.id = props.id ?? 'random-points-on-line';
    this.props = props;
    const {id} = this;
    const {output} = props;
    validatePackedView(props.positions, ['float32x2'], `${id} positions`);
    validatePackedUint32View(props.lineOffsets, `${id} lineOffsets`);
    validatePackedUint32View(props.counts, `${id} counts`);
    validatePackedUint32View(props.parameters, `${id} parameters`);
    if (props.lineOffsets.length < 2) {
      throw new Error(`${id} lineOffsets needs at least one line`);
    }
    this.lineCount = props.lineOffsets.length - 1;
    if (props.counts.length !== this.lineCount) {
      throw new Error(`${id} counts must have one row per line`);
    }
    if (props.parameters.length < GPU_DOT_DENSITY_PARAMETER_LENGTH) {
      throw new Error(
        `${id} parameters must contain ${GPU_DOT_DENSITY_PARAMETER_LENGTH} uint32 rows`
      );
    }
    validatePackedView(output.positions, ['float32x2'], `${id} output.positions`);
    validateCompactOutput(id, output.points);
    const capacity = output.positions.length;
    if (capacity < 1 || output.points.ids.length !== capacity) {
      throw new Error(
        `${id} output.positions and output.points.ids must share a positive capacity`
      );
    }
    if (output.fractions) {
      validatePackedView(output.fractions, ['float32'], `${id} output.fractions`);
      if (output.fractions.length !== capacity) {
        throw new Error(`${id} output.fractions must have capacity rows`);
      }
    }
    validateGraphOutputsDisjointFromInputs(id, getOutputs(props), getInputs(props));
  }

  /** Returns the `lengths`, `counts`, scan, `total`, `locate`, `sample` and publish nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, lineCount} = this;
    const {output} = props;
    validateGraphViewsBelongToGraph(id, graph, [...getInputs(props), ...getOutputs(props)]);
    const capacity = output.positions.length;
    const limit = getDotSlotCountLimit(lineCount, capacity);
    const lengths = createTransientView(graph, `${id}-lengths`, 'float32', lineCount);
    // Arc length at every vertex (0 at each line's first vertex), summed in vertex order.
    const arcLengths = createTransientView(
      graph,
      `${id}-arc-lengths`,
      'float32',
      Math.max(props.positions.length, 1)
    );
    const lineCounts = createTransientView(graph, `${id}-line-counts`, 'uint32', lineCount);
    const lineOffsets = createTransientView(graph, `${id}-line-offsets`, 'uint32', lineCount);
    const clampFlag = createTransientView(graph, `${id}-clamped`, 'uint32', 1);
    const total = createTransientView(graph, `${id}-total`, 'uint32', 1);
    const pointSlots = createTransientView(graph, `${id}-point-slots`, 'uint32', capacity * 2);
    const segmentLength = /* wgsl */ `
fn readVertex(vertex: u32) -> vec2<f32> {
  return vec2<f32>(positions[positionsOffset + 2u * vertex], positions[positionsOffset + 2u * vertex + 1u]);
}
fn segmentLength(vertex: u32) -> f32 {
  let delta = readVertex(vertex + 1u) - readVertex(vertex);
  return sqrt(delta.x * delta.x + delta.y * delta.y);
}`;

    const nodes: GPUCommandNode<Parameters>[] = [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-lengths`,
        operation: OPERATION,
        variant: 'lengths',
        bindings: [
          {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
          {name: 'inputLineOffsets', view: props.lineOffsets, type: 'u32', access: 'read'},
          {name: 'lengths', view: lengths, type: 'f32', access: 'read_write'},
          {name: 'arcLengths', view: arcLengths, type: 'f32', access: 'read_write'}
        ],
        invocationCount: lineCount,
        declarations: segmentLength,
        // The running sum is kept per vertex so that sampling can binary-search the arc length
        // instead of re-walking the line for every point.
        body: `let first = inputLineOffsets[inputLineOffsetsOffset + index];
  let last = inputLineOffsets[inputLineOffsetsOffset + index + 1u];
  var total = 0.0;
  if (last > first) {
    arcLengths[arcLengthsOffset + first] = 0.0;
  }
  for (var vertex = first; vertex + 1u < last; vertex = vertex + 1u) {
    total = total + segmentLength(vertex);
    arcLengths[arcLengthsOffset + vertex + 1u] = total;
  }
  lengths[lengthsOffset + index] = total;`
      }),
      createFillNode<Parameters>(graph, {
        id: `${id}-clear-clamped`,
        operation: OPERATION,
        view: clampFlag,
        type: 'u32',
        value: '0u'
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-counts`,
        operation: OPERATION,
        variant: 'counts',
        bindings: [
          {name: 'sourceCounts', view: props.counts, type: 'u32', access: 'read'},
          {name: 'lengths', view: lengths, type: 'f32', access: 'read'},
          {name: 'lineCounts', view: lineCounts, type: 'u32', access: 'read_write'},
          {name: 'clampFlag', view: clampFlag, type: 'u32', access: 'read_write'}
        ],
        invocationCount: lineCount,
        declarations: `const LINE_LIMIT: u32 = ${limit}u;
fn isFiniteFloat(value: f32) -> bool { return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u; }`,
        body: `let lineLength = lengths[lengthsOffset + index];
  var count = select(0u, sourceCounts[sourceCountsOffset + index], isFiniteFloat(lineLength) && lineLength > 0.0);
  if (count > LINE_LIMIT) {
    count = LINE_LIMIT;
    clampFlag[clampFlagOffset] = 1u;
  }
  lineCounts[lineCountsOffset + index] = count;`
      }),
      ...new GPUScan({
        id: `${id}-line-scan`,
        input: lineCounts,
        output: lineOffsets,
        mode: 'exclusive'
      }).getCommandNodes(graph),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-total`,
        operation: OPERATION,
        variant: 'total',
        bindings: [
          {name: 'lineCounts', view: lineCounts, type: 'u32', access: 'read'},
          {name: 'lineOffsets', view: lineOffsets, type: 'u32', access: 'read'},
          {name: 'totalOut', view: total, type: 'u32', access: 'read_write'}
        ],
        invocationCount: 1,
        body: `let last = ${lineCount - 1}u;
  totalOut[totalOutOffset] = lineOffsets[lineOffsetsOffset + last] + lineCounts[lineCountsOffset + last];`
      }),
      // Line and rank of every published point (binary search on the inclusive prefix).
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-locate`,
        operation: OPERATION,
        variant: 'locate',
        bindings: [
          {name: 'lineCounts', view: lineCounts, type: 'u32', access: 'read'},
          {name: 'lineOffsets', view: lineOffsets, type: 'u32', access: 'read'},
          {name: 'total', view: total, type: 'u32', access: 'read'},
          {name: 'pointSlots', view: pointSlots, type: 'u32', access: 'read_write'},
          {name: 'lineIds', view: output.points.ids, type: 'u32', access: 'read_write'}
        ],
        invocationCount: capacity,
        declarations: `const LINE_COUNT: u32 = ${lineCount}u;`,
        body: `let count = min(total[totalOffset], ${capacity}u);
  if (index >= count) {
    pointSlots[pointSlotsOffset + 2u * index] = ${NONE};
    pointSlots[pointSlotsOffset + 2u * index + 1u] = 0u;
    lineIds[lineIdsOffset + index] = ${NONE};
    return;
  }
  var low = 0u;
  var high = LINE_COUNT - 1u;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (lineOffsets[lineOffsetsOffset + middle] + lineCounts[lineCountsOffset + middle] > index) {
      high = middle;
    } else {
      low = middle + 1u;
    }
  }
  pointSlots[pointSlotsOffset + 2u * index] = low;
  pointSlots[pointSlotsOffset + 2u * index + 1u] = index - lineOffsets[lineOffsetsOffset + low];
  lineIds[lineIdsOffset + index] = low;`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-sample`,
        operation: OPERATION,
        variant: output.fractions ? 'sample-fractions' : 'sample',
        bindings: [
          {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
          {name: 'inputLineOffsets', view: props.lineOffsets, type: 'u32', access: 'read'},
          {name: 'lengths', view: lengths, type: 'f32', access: 'read'},
          {name: 'arcLengths', view: arcLengths, type: 'f32', access: 'read'},
          {name: 'pointSlots', view: pointSlots, type: 'u32', access: 'read'},
          {name: 'parameters', view: props.parameters, type: 'u32', access: 'read'},
          {name: 'pointsOut', view: output.positions, type: 'f32', access: 'read_write'},
          ...(output.fractions
            ? [{name: 'fractionsOut', view: output.fractions, type: 'f32', access: 'read_write'}]
            : [])
        ] as never,
        invocationCount: capacity,
        declarations: `${PHILOX_WGSL}${segmentLength}`,
        body: `var nanBits = 0x7fc00000u;
  var point = vec2<f32>(bitcast<f32>(nanBits));
  var fraction = bitcast<f32>(nanBits);
  let line = pointSlots[pointSlotsOffset + 2u * index];
  if (line != ${NONE}) {
    let rank = pointSlots[pointSlotsOffset + 2u * index + 1u];
    let random = philox4x32(vec4<u32>(line, rank, 0u, 0u), vec2<u32>(parameters[parametersOffset], ${RANDOM_POINTS_ON_LINE_PURPOSE}u));
    fraction = philoxUnitFloat(random.x);
    let arcTarget = fraction * lengths[lengthsOffset + line];
    let first = inputLineOffsets[inputLineOffsetsOffset + line];
    let last = inputLineOffsets[inputLineOffsetsOffset + line + 1u];
    // First segment whose cumulative end exceeds the target (arc lengths are non-decreasing); zero
    // length segments never qualify. No such segment means rounding put the target at or past the
    // summed length, so the point clamps to the final vertex.
    var low = first;
    var high = last - 1u;
    while (low < high) {
      let middle = (low + high) / 2u;
      if (arcLengths[arcLengthsOffset + middle + 1u] > arcTarget) {
        high = middle;
      } else {
        low = middle + 1u;
      }
    }
    if (low + 1u < last && arcLengths[arcLengthsOffset + low + 1u] > arcTarget) {
      let travelled = arcLengths[arcLengthsOffset + low];
      point = mix(readVertex(low), readVertex(low + 1u), (arcTarget - travelled) / segmentLength(low));
    } else {
      point = readVertex(last - 1u);
    }
  }
  pointsOut[pointsOutOffset + 2u * index] = point.x;
  pointsOut[pointsOutOffset + 2u * index + 1u] = point.y;
  ${output.fractions ? 'fractionsOut[fractionsOutOffset + index] = fraction;' : ''}`
      }),
      createPublishNode<Parameters>(graph, {
        id: `${id}-publish`,
        operation: OPERATION,
        requiredCount: total,
        output: output.points,
        overflowSources: [clampFlag]
      })
    ];
    return nodes;
  }
}

/** Returns every read-only view of a random-points-on-line contributor. */
function getInputs(props: GPURandomPointsOnLineProps): GraphDataView[] {
  return [props.positions, props.lineOffsets, props.counts, props.parameters];
}

/** Returns every writable view of a random-points-on-line contributor. */
function getOutputs(props: GPURandomPointsOnLineProps): (GraphDataView | undefined)[] {
  const {output} = props;
  return [
    output.positions,
    output.points.ids,
    output.points.count,
    output.points.overflow,
    output.points.requiredCount,
    output.fractions
  ];
}
