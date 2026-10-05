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
  type WGSLKernelBinding
} from '../../utils/wgsl-kernel-nodes';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {validateGraphViewsBelongToGraph} from '../../utils/gpu-contributor-utils';
import {GPU_RASTER_PROFILE_PARAMETER_LENGTH} from './raster-sampling-parameters';
import {
  getRasterSamplingWGSL,
  validateRasterDescription,
  validateRasterSamplingView
} from './raster-sampling-utils';

const OPERATION = 'GPURasterProfile';
const MAXIMUM_SAMPLES_PER_PATH = 1 << 24;

/** Path ID written to profile sample rows at or beyond `count`. */
export const GPU_RASTER_PROFILE_NO_PATH_ID = 0xffffffff;

/** Caller-owned outputs of {@link GPURasterProfile}. Any subset of the optional views may be given. */
export type GPURasterProfileOutput = {
  /** One-row count of emitted sample rows, clamped to the sample capacity. */
  count: GraphDataView<'uint32'>;
  /** One-row flag: 1 when samples exceeded the capacity or a path hit the per-path limit. */
  overflow: GraphDataView<'uint32'>;
  /** Optional one-row unclamped sample count. */
  totalCount?: GraphDataView<'uint32'>;
  /** Sample positions; NaN at rows at or beyond `count`. */
  samplePositions?: GraphDataView<'float32x2'>;
  /** Planar distance along the path of each sample; NaN beyond `count`. */
  sampleDistances?: GraphDataView<'float32'>;
  /** Raster value at each sample; NaN outside the extent, at nodata, and beyond `count`. */
  sampleValues?: GraphDataView<'float32'>;
  /** Source path index of each sample; {@link GPU_RASTER_PROFILE_NO_PATH_ID} beyond `count`. */
  samplePathIds?: GraphDataView<'uint32'>;
  /** Running positive elevation change along the path up to and including each sample; rows at or beyond `count` are not written. */
  sampleCumulativeGain?: GraphDataView<'float32'>;
  /** Running descent (a positive magnitude) along the path up to and including each sample; rows at or beyond `count` are not written. */
  sampleCumulativeLoss?: GraphDataView<'float32'>;
  /** `pathCount + 1` first-sample rows per path, clamped to the sample capacity. */
  pathSampleOffsets?: GraphDataView<'uint32'>;
  /** Planar length of each path (0 for paths with fewer than two vertices). */
  pathLength?: GraphDataView<'float32'>;
  /** Total gain per path over its emitted samples. */
  pathGain?: GraphDataView<'float32'>;
  /** Total loss per path (a positive magnitude). */
  pathLoss?: GraphDataView<'float32'>;
  /** Minimum finite sample value per path; NaN when none is finite. */
  pathMinimum?: GraphDataView<'float32'>;
  /** Maximum finite sample value per path; NaN when none is finite. */
  pathMaximum?: GraphDataView<'float32'>;
};

/**
 * Properties for {@link GPURasterProfile}.
 *
 * Per-frame (no recompile): raster, vertex and offset contents and `parameters` (extent, method,
 * nodata policy, spacing). Topology: `width`, `height`, `noDataValue`, `pathOffsets.length`,
 * vertex and sample capacities, and which optional outputs exist.
 */
export type GPURasterProfileProps = {
  /** Prefix for generated node IDs. Defaults to `'raster-profile'`. */
  id?: string;
  /** Raster width in cells. */
  width: number;
  /** Raster height in cells. */
  height: number;
  /** Row-major raster of at least `width * height` rows; row 0 is at `minY`. NaN is nodata. */
  values: GraphDataView<'float32'>;
  /** Optional finite sentinel treated as nodata. */
  noDataValue?: number;
  /** Optional raster validity; zero marks a nodata cell. */
  validity?: GraphDataView<'uint32'>;
  /** Packed path vertices in extent units. */
  pathPositions: GraphDataView<'float32x2'>;
  /** CSR spans: path `p` owns vertices `[pathOffsets[p], pathOffsets[p + 1])`; `pathCount + 1` rows. */
  pathOffsets: GraphDataView<'uint32'>;
  /**
   * Sample row capacity. Defaults to the length of the first provided sample column; every
   * provided sample column must hold at least this many rows. Required when none is provided.
   */
  sampleCapacity?: number;
  /** Per-frame float32 view written with `getGPURasterProfileParameterValues`. */
  parameters: GraphDataView<'float32'>;
  /** Caller-owned outputs. */
  output: GPURasterProfileOutput;
};

/**
 * Elevation profiles: samples a raster along polylines at a per-frame spacing.
 *
 * Distances are planar in extent units. Each path is sampled at along-distances `0, s, 2s, ...`
 * while below its length `L`, plus the final vertex at exactly `L`; a zero-length path (one vertex,
 * or coincident vertices) gives one sample and a path without vertices gives none. Sample rows are
 * ordered by path then distance. A path may emit at most `min(2^24, floor((2^32 - 1) / (pathCount +
 * 1)))` samples: beyond that its samples are truncated (the last row still sits at the final
 * vertex), and `overflow` is set. Samples beyond `sampleCapacity` are dropped and set `overflow`.
 *
 * Pipeline: per-path serial walk (cumulative vertex distances and sample counts), `GPUScan` of the
 * counts, a per-sample kernel that binary searches its path and segment and interpolates the
 * position, the shared raster sampling kernel ({@link GPURasterSampling} semantics for `method` and
 * `noDataPolicy`), then a per-path serial pass for cumulative gain and loss and summaries. Gain and
 * loss sum positive and negative differences between consecutive finite samples (skipping
 * non-finite ones); minimum and maximum cover finite samples. The serial passes are fixed-order, so
 * results are deterministic, but a single very long path is processed by one invocation.
 *
 * Rows at or beyond `count` of the position, distance, value and path ID columns hold NaN (path IDs
 * {@link GPU_RASTER_PROFILE_NO_PATH_ID}); the cumulative columns are left unwritten there.
 */
export class GPURasterProfile implements GPUCommandNodeProducer {
  /** Prefix for every node ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPURasterProfileProps;
  /** Number of paths. */
  readonly pathCount: number;
  /** Sample row capacity. */
  readonly sampleCapacity: number;

  constructor(props: GPURasterProfileProps) {
    this.id = props.id ?? 'raster-profile';
    this.props = props;
    const {id} = this;
    const {width, height, output} = props;
    validateRasterDescription(id, width, height, props.noDataValue);
    validateRasterSamplingView(id, 'values', props.values, 'float32', width * height);
    if (props.validity) {
      validateRasterSamplingView(id, 'validity', props.validity, 'uint32', width * height);
    }
    validateRasterSamplingView(id, 'pathPositions', props.pathPositions, 'float32x2', 1);
    validateRasterSamplingView(id, 'pathOffsets', props.pathOffsets, 'uint32', 2);
    validateRasterSamplingView(
      id,
      'parameters',
      props.parameters,
      'float32',
      GPU_RASTER_PROFILE_PARAMETER_LENGTH
    );
    if (!output?.count || !output.overflow) {
      throw new Error(`${id} needs output.count and output.overflow`);
    }
    this.pathCount = props.pathOffsets.length - 1;
    const sampleColumns = [
      output.samplePositions,
      output.sampleDistances,
      output.sampleValues,
      output.samplePathIds,
      output.sampleCumulativeGain,
      output.sampleCumulativeLoss
    ].filter(view => view !== undefined);
    const capacity = props.sampleCapacity ?? sampleColumns[0]?.length;
    if (capacity === undefined) {
      throw new Error(`${id} needs sampleCapacity when no sample column is provided`);
    }
    if (!Number.isSafeInteger(capacity) || capacity < 1) {
      throw new Error(`${id} sampleCapacity must be a positive integer`);
    }
    this.sampleCapacity = capacity;
    const sampleFormats = [
      ['samplePositions', 'float32x2'],
      ['sampleDistances', 'float32'],
      ['sampleValues', 'float32'],
      ['samplePathIds', 'uint32'],
      ['sampleCumulativeGain', 'float32'],
      ['sampleCumulativeLoss', 'float32']
    ] as const;
    for (const [name, format] of sampleFormats) {
      if (output[name]) {
        validateRasterSamplingView(id, `output.${name}`, output[name], format, capacity);
      }
    }
    for (const name of ['count', 'overflow', 'totalCount'] as const) {
      if (output[name]) {
        validateRasterSamplingView(id, `output.${name}`, output[name], 'uint32', 1);
      }
    }
    if (output.pathSampleOffsets) {
      validateRasterSamplingView(
        id,
        'output.pathSampleOffsets',
        output.pathSampleOffsets,
        'uint32',
        this.pathCount + 1
      );
    }
    for (const name of [
      'pathLength',
      'pathGain',
      'pathLoss',
      'pathMinimum',
      'pathMaximum'
    ] as const) {
      if (output[name]) {
        validateRasterSamplingView(id, `output.${name}`, output[name], 'float32', this.pathCount);
      }
    }
    const outputs = Object.values(output).filter(view => view !== undefined);
    if (new Set(outputs.map(view => view.buffer)).size !== outputs.length) {
      throw new Error(`${id} outputs must not share buffers`);
    }
    const inputBuffers = new Set(
      [props.values, props.validity, props.pathPositions, props.pathOffsets, props.parameters]
        .filter(view => view !== undefined)
        .map(view => view.buffer)
    );
    if (outputs.some(view => inputBuffers.has(view.buffer))) {
      throw new Error(`${id} outputs must not share buffers with inputs`);
    }
  }

  /** Returns the measure, scan, locate, sample, cumulative and finish nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, pathCount, sampleCapacity} = this;
    const {output} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      props.values,
      props.validity,
      props.pathPositions,
      props.pathOffsets,
      props.parameters,
      ...Object.values(output)
    ]);
    const vertexCount = props.pathPositions.length;
    const maximumPerPath = Math.max(
      1,
      Math.min(MAXIMUM_SAMPLES_PER_PATH, Math.floor(0xffffffff / (pathCount + 1)))
    );
    const nodes: GPUCommandNode<Parameters>[] = [];
    const vertexDistances = createTransientView(
      graph,
      `${id}-vertex-distances`,
      'float32',
      vertexCount
    );
    const sampleCounts = createTransientView(graph, `${id}-sample-counts`, 'uint32', pathCount + 1);
    const sampleOffsets = createTransientView(
      graph,
      `${id}-sample-offsets`,
      'uint32',
      pathCount + 1
    );
    const truncated = createTransientView(graph, `${id}-truncated`, 'uint32', 1);
    const samplePositions =
      output.samplePositions ??
      createTransientView(graph, `${id}-sample-positions`, 'float32x2', sampleCapacity);
    const sampleValues =
      output.sampleValues ??
      createTransientView(graph, `${id}-sample-values`, 'float32', sampleCapacity);
    const pathConstants = `const PATH_COUNT: u32 = ${pathCount}u;
const VERTEX_COUNT: u32 = ${vertexCount}u;
const SAMPLE_CAPACITY: u32 = ${sampleCapacity}u;
const MAXIMUM_PER_PATH: u32 = ${maximumPerPath}u;
fn getVertexBegin(path: u32) -> u32 {
  return min(pathOffsets[pathOffsetsOffset + path], VERTEX_COUNT);
}
fn getVertexEnd(path: u32, vertexBegin: u32) -> u32 {
  return max(min(pathOffsets[pathOffsetsOffset + path + 1u], VERTEX_COUNT), vertexBegin);
}`;

    // 1. Per-path serial walk: cumulative vertex distances and sample counts.
    nodes.push(
      createFillNode<Parameters>(graph, {
        id: `${id}-truncated-fill`,
        operation: OPERATION,
        view: truncated,
        type: 'u32',
        value: '0u'
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-measure`,
        operation: OPERATION,
        variant: 'measure',
        bindings: [
          {name: 'pathPositions', view: props.pathPositions, type: 'f32', access: 'read'},
          {name: 'pathOffsets', view: props.pathOffsets, type: 'u32', access: 'read'},
          {name: 'params', view: props.parameters, type: 'f32', access: 'read'},
          {name: 'vertexDistances', view: vertexDistances, type: 'f32', access: 'read_write'},
          {name: 'sampleCounts', view: sampleCounts, type: 'u32', access: 'read_write'},
          {name: 'truncated', view: truncated, type: 'atomic<u32>', access: 'read_write'}
        ],
        invocationCount: pathCount + 1,
        declarations: pathConstants,
        body: `if (index >= PATH_COUNT) {
    sampleCounts[sampleCountsOffset + index] = 0u;
    return;
  }
  let vertexBegin = getVertexBegin(index);
  let vertexEnd = getVertexEnd(index, vertexBegin);
  var count = 0u;
  if (vertexEnd > vertexBegin) {
    var distance = 0.0;
    vertexDistances[vertexDistancesOffset + vertexBegin] = 0.0;
    var previousX = pathPositions[pathPositionsOffset + 2u * vertexBegin];
    var previousY = pathPositions[pathPositionsOffset + 2u * vertexBegin + 1u];
    for (var vertex = vertexBegin + 1u; vertex < vertexEnd; vertex++) {
      let currentX = pathPositions[pathPositionsOffset + 2u * vertex];
      let currentY = pathPositions[pathPositionsOffset + 2u * vertex + 1u];
      let deltaX = currentX - previousX;
      let deltaY = currentY - previousY;
      distance += sqrt(deltaX * deltaX + deltaY * deltaY);
      vertexDistances[vertexDistancesOffset + vertex] = distance;
      previousX = currentX;
      previousY = currentY;
    }
    count = 1u;
    if (distance > 0.0) {
      let spacing = params[paramsOffset + 10u];
      var segments = u32(min(ceil(distance / spacing), f32(MAXIMUM_PER_PATH)));
      if (segments > 0u && f32(segments - 1u) * spacing >= distance) {
        segments -= 1u;
      }
      if (f32(segments) * spacing < distance) {
        segments += 1u;
      }
      count = segments + 1u;
      if (count > MAXIMUM_PER_PATH) {
        count = MAXIMUM_PER_PATH;
        atomicMax(&truncated[truncatedOffset], 1u);
      }
    }
  }
  sampleCounts[sampleCountsOffset + index] = count;`
      })
    );

    // 2. Exclusive scan; the sentinel row at the end holds the unclamped total.
    nodes.push(
      ...new GPUScan({
        id: `${id}-scan`,
        input: sampleCounts,
        output: sampleOffsets
      }).getCommandNodes(graph)
    );

    // 3. Per-sample path, distance and position.
    const locateBindings: WGSLKernelBinding[] = [
      {name: 'pathPositions', view: props.pathPositions, type: 'f32', access: 'read'},
      {name: 'pathOffsets', view: props.pathOffsets, type: 'u32', access: 'read'},
      {name: 'vertexDistances', view: vertexDistances, type: 'f32', access: 'read'},
      {name: 'sampleOffsets', view: sampleOffsets, type: 'u32', access: 'read'},
      {name: 'params', view: props.parameters, type: 'f32', access: 'read'},
      {name: 'positionsOut', view: samplePositions, type: 'f32', access: 'read_write'}
    ];
    if (output.sampleDistances) {
      locateBindings.push({
        name: 'distancesOut',
        view: output.sampleDistances,
        type: 'f32',
        access: 'read_write'
      });
    }
    if (output.samplePathIds) {
      locateBindings.push({
        name: 'pathIdsOut',
        view: output.samplePathIds,
        type: 'u32',
        access: 'read_write'
      });
    }
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-locate`,
        operation: OPERATION,
        variant: 'locate',
        bindings: locateBindings,
        invocationCount: sampleCapacity,
        declarations: `${pathConstants}
fn getNaN() -> f32 {
  var bits = 0x7fc00000u;
  return bitcast<f32>(bits);
}`,
        body: `let total = sampleOffsets[sampleOffsetsOffset + PATH_COUNT];
  if (index >= total) {
    positionsOut[positionsOutOffset + 2u * index] = getNaN();
    positionsOut[positionsOutOffset + 2u * index + 1u] = getNaN();
    ${output.sampleDistances ? 'distancesOut[distancesOutOffset + index] = getNaN();' : ''}
    ${output.samplePathIds ? 'pathIdsOut[pathIdsOutOffset + index] = 0xffffffffu;' : ''}
    return;
  }
  var low = 0u;
  var high = PATH_COUNT;
  loop {
    if (low >= high) {
      break;
    }
    let middle = (low + high) / 2u;
    if (sampleOffsets[sampleOffsetsOffset + middle] <= index) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  let path = low - 1u;
  let vertexBegin = getVertexBegin(path);
  let vertexEnd = getVertexEnd(path, vertexBegin);
  let firstSample = sampleOffsets[sampleOffsetsOffset + path];
  let sampleCount = sampleOffsets[sampleOffsetsOffset + path + 1u] - firstSample;
  let localSample = index - firstSample;
  let lastVertex = vertexEnd - 1u;
  let length = vertexDistances[vertexDistancesOffset + lastVertex];
  var distance = length;
  var positionX = pathPositions[pathPositionsOffset + 2u * lastVertex];
  var positionY = pathPositions[pathPositionsOffset + 2u * lastVertex + 1u];
  if (localSample + 1u < sampleCount) {
    distance = min(f32(localSample) * params[paramsOffset + 10u], length);
    var searchLow = vertexBegin + 1u;
    var searchHigh = vertexEnd;
    loop {
      if (searchLow >= searchHigh) {
        break;
      }
      let middle = (searchLow + searchHigh) / 2u;
      if (vertexDistances[vertexDistancesOffset + middle] <= distance) {
        searchLow = middle + 1u;
      } else {
        searchHigh = middle;
      }
    }
    let segmentStart = searchLow - 1u;
    if (segmentStart < lastVertex) {
      let startDistance = vertexDistances[vertexDistancesOffset + segmentStart];
      let endDistance = vertexDistances[vertexDistancesOffset + segmentStart + 1u];
      let t = (distance - startDistance) / (endDistance - startDistance);
      let startX = pathPositions[pathPositionsOffset + 2u * segmentStart];
      let startY = pathPositions[pathPositionsOffset + 2u * segmentStart + 1u];
      let endX = pathPositions[pathPositionsOffset + 2u * segmentStart + 2u];
      let endY = pathPositions[pathPositionsOffset + 2u * segmentStart + 3u];
      positionX = startX + (endX - startX) * t;
      positionY = startY + (endY - startY) * t;
    }
  }
  positionsOut[positionsOutOffset + 2u * index] = positionX;
  positionsOut[positionsOutOffset + 2u * index + 1u] = positionY;
  ${output.sampleDistances ? 'distancesOut[distancesOutOffset + index] = distance;' : ''}
  ${output.samplePathIds ? 'pathIdsOut[pathIdsOutOffset + index] = path;' : ''}`
      })
    );

    // 4. Raster value per sample (NaN positions beyond the total give NaN).
    const valueBindings: WGSLKernelBinding[] = [
      {name: 'raster', view: props.values, type: 'f32', access: 'read'}
    ];
    if (props.validity) {
      valueBindings.push({name: 'validity', view: props.validity, type: 'u32', access: 'read'});
    }
    valueBindings.push(
      {name: 'samplePositions', view: samplePositions, type: 'f32', access: 'read'},
      {name: 'params', view: props.parameters, type: 'f32', access: 'read'},
      {name: 'valuesOut', view: sampleValues, type: 'f32', access: 'read_write'}
    );
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-sample`,
        operation: OPERATION,
        variant: 'sample',
        bindings: valueBindings,
        invocationCount: sampleCapacity,
        declarations: getRasterSamplingWGSL({
          width: props.width,
          height: props.height,
          noDataValue: props.noDataValue,
          hasValidity: Boolean(props.validity)
        }),
        body: `valuesOut[valuesOutOffset + index] = sampleRaster(
    samplePositions[samplePositionsOffset + 2u * index],
    samplePositions[samplePositionsOffset + 2u * index + 1u]
  );`
      })
    );

    // 5. Per-path serial pass: cumulative gain and loss, summaries.
    const cumulativeBindings: WGSLKernelBinding[] = [
      {name: 'sampleOffsets', view: sampleOffsets, type: 'u32', access: 'read'},
      {name: 'sampleValues', view: sampleValues, type: 'f32', access: 'read'}
    ];
    const cumulativeOutputs = [
      ['cumulativeGainOut', output.sampleCumulativeGain],
      ['cumulativeLossOut', output.sampleCumulativeLoss],
      ['pathGainOut', output.pathGain],
      ['pathLossOut', output.pathLoss],
      ['pathMinimumOut', output.pathMinimum],
      ['pathMaximumOut', output.pathMaximum]
    ] as const;
    for (const [name, view] of cumulativeOutputs) {
      if (view) {
        cumulativeBindings.push({name, view, type: 'f32', access: 'read_write'});
      }
    }
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-cumulative`,
        operation: OPERATION,
        variant: 'cumulative',
        bindings: cumulativeBindings,
        invocationCount: pathCount,
        declarations: `const PATH_COUNT: u32 = ${pathCount}u;
const SAMPLE_CAPACITY: u32 = ${sampleCapacity}u;
fn getNaN() -> f32 {
  var bits = 0x7fc00000u;
  return bitcast<f32>(bits);
}
fn isFiniteValue(value: f32) -> bool { return (bitcast<u32>(value) & 0x7fffffffu) < 0x7f800000u; }`,
        body: `let firstSample = min(sampleOffsets[sampleOffsetsOffset + index], SAMPLE_CAPACITY);
  let endSample = min(sampleOffsets[sampleOffsetsOffset + index + 1u], SAMPLE_CAPACITY);
  var gain = 0.0;
  var loss = 0.0;
  var minimum = getNaN();
  var maximum = getNaN();
  var previous = 0.0;
  var hasPrevious = false;
  for (var sample = firstSample; sample < endSample; sample++) {
    let value = sampleValues[sampleValuesOffset + sample];
    if (isFiniteValue(value)) {
      if (hasPrevious) {
        let delta = value - previous;
        if (delta > 0.0) {
          gain += delta;
        } else if (delta < 0.0) {
          loss -= delta;
        }
        minimum = select(minimum, value, value < minimum);
        maximum = select(maximum, value, value > maximum);
      } else {
        minimum = value;
        maximum = value;
      }
      previous = value;
      hasPrevious = true;
    }
    ${output.sampleCumulativeGain ? 'cumulativeGainOut[cumulativeGainOutOffset + sample] = gain;' : ''}
    ${output.sampleCumulativeLoss ? 'cumulativeLossOut[cumulativeLossOutOffset + sample] = loss;' : ''}
  }
  ${output.pathGain ? 'pathGainOut[pathGainOutOffset + index] = gain;' : ''}
  ${output.pathLoss ? 'pathLossOut[pathLossOutOffset + index] = loss;' : ''}
  ${output.pathMinimum ? 'pathMinimumOut[pathMinimumOutOffset + index] = minimum;' : ''}
  ${output.pathMaximum ? 'pathMaximumOut[pathMaximumOutOffset + index] = maximum;' : ''}`
      })
    );

    // 6. Per-path offsets and lengths, then the scalar count, overflow and total.
    if (output.pathSampleOffsets || output.pathLength) {
      const finishBindings: WGSLKernelBinding[] = [
        {name: 'sampleOffsets', view: sampleOffsets, type: 'u32', access: 'read'},
        {name: 'pathOffsets', view: props.pathOffsets, type: 'u32', access: 'read'},
        {name: 'vertexDistances', view: vertexDistances, type: 'f32', access: 'read'}
      ];
      if (output.pathSampleOffsets) {
        finishBindings.push({
          name: 'pathSampleOffsetsOut',
          view: output.pathSampleOffsets,
          type: 'u32',
          access: 'read_write'
        });
      }
      if (output.pathLength) {
        finishBindings.push({
          name: 'pathLengthOut',
          view: output.pathLength,
          type: 'f32',
          access: 'read_write'
        });
      }
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-finish-paths`,
          operation: OPERATION,
          variant: 'finish-paths',
          bindings: finishBindings,
          invocationCount: pathCount + 1,
          declarations: pathConstants,
          body: `${output.pathSampleOffsets ? 'pathSampleOffsetsOut[pathSampleOffsetsOutOffset + index] = min(sampleOffsets[sampleOffsetsOffset + index], SAMPLE_CAPACITY);' : ''}
  if (index < PATH_COUNT) {
    let vertexBegin = getVertexBegin(index);
    let vertexEnd = getVertexEnd(index, vertexBegin);
    var length = 0.0;
    if (vertexEnd > vertexBegin) {
      length = vertexDistances[vertexDistancesOffset + vertexEnd - 1u];
    }
    ${output.pathLength ? 'pathLengthOut[pathLengthOutOffset + index] = length;' : ''}
  }`
        })
      );
    }
    const countBindings: WGSLKernelBinding[] = [
      {name: 'sampleOffsets', view: sampleOffsets, type: 'u32', access: 'read'},
      {name: 'truncated', view: truncated, type: 'u32', access: 'read'},
      {name: 'countOut', view: output.count, type: 'u32', access: 'read_write'},
      {name: 'overflowOut', view: output.overflow, type: 'u32', access: 'read_write'}
    ];
    if (output.totalCount) {
      countBindings.push({
        name: 'totalOut',
        view: output.totalCount,
        type: 'u32',
        access: 'read_write'
      });
    }
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-finish-counts`,
        operation: OPERATION,
        variant: 'finish-counts',
        bindings: countBindings,
        invocationCount: 1,
        declarations: `const PATH_COUNT: u32 = ${pathCount}u;
const SAMPLE_CAPACITY: u32 = ${sampleCapacity}u;`,
        body: `let total = sampleOffsets[sampleOffsetsOffset + PATH_COUNT];
  countOut[countOutOffset] = min(total, SAMPLE_CAPACITY);
  overflowOut[overflowOutOffset] = select(0u, 1u, total > SAMPLE_CAPACITY || truncated[truncatedOffset] != 0u);
  ${output.totalCount ? 'totalOut[totalOutOffset] = total;' : ''}`
      })
    );
    return nodes;
  }
}
