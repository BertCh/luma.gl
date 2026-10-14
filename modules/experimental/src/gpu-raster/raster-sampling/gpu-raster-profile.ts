// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
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
import {
  getStorageBindingElementLimit,
  validateGraphViewsBelongToGraph,
  validateMaximumCount
} from '../../utils/gpu-contributor-utils';
import type {GPUCardinalityPort, GPUOverflowStatusPort} from '../../utils/gpu-contributor-types';
import type {GPURasterMetadata} from '../types';
import {GPU_RASTER_PROFILE_PARAMETER_LENGTH} from './raster-sampling-parameters';
import {
  getRasterSamplingWGSL,
  validateRasterDescription,
  validateRasterSamplingView
} from './raster-sampling-utils';
import {GET_NAN_WGSL, IS_FINITE_VALUE_WGSL} from '../../utils/wgsl-common';
import {
  DISTANCE_SCAN_STATE_WORDS,
  PROFILE_SCAN_STATE_WORDS,
  SEGMENTED_SCAN_WORKGROUP_SIZE,
  getDistanceScanDeclarations,
  getProfileScanDeclarations,
  getScanBlockBody,
  getScanCarryBody,
  getSegmentedScanBlockCount
} from './gpu-raster-profile-scan';

const OPERATION = 'GPURasterProfile';
const MAXIMUM_SAMPLES_PER_LINE = 1 << 24;

/** Line ID written to profile sample rows at or beyond `count`. */
export const GPU_RASTER_PROFILE_NO_LINE_ID = 0xffffffff;

/** Caller-owned outputs of {@link GPURasterProfile}. Any subset of the optional views may be given. */
export type GPURasterProfileOutput = {
  /**
   * One-row status scalars: `count` receives the emitted sample rows, clamped to the sample
   * capacity; `overflow` is 1 when samples exceeded the capacity or a line hit the per-line limit;
   * the optional `requiredCount` receives the unclamped sample count.
   */
  status: GPUCardinalityPort & GPUOverflowStatusPort;
  /** Sample positions relative to the parameters' `localOrigin`; NaN at rows at or beyond `count`. */
  samplePositions?: GraphDataView<'float32x2'>;
  /** Planar distance along the line of each sample; NaN beyond `count`. */
  sampleDistances?: GraphDataView<'float32'>;
  /** Raster value at each sample; NaN outside the extent, at nodata, and beyond `count`. */
  sampleValues?: GraphDataView<'float32'>;
  /** Source line index of each sample; {@link GPU_RASTER_PROFILE_NO_LINE_ID} beyond `count`. */
  sampleLineIds?: GraphDataView<'uint32'>;
  /** Running positive elevation change along the line up to and including each sample; rows at or beyond `count` are not written. */
  sampleCumulativeGain?: GraphDataView<'float32'>;
  /** Running descent (a positive magnitude) along the line up to and including each sample; rows at or beyond `count` are not written. */
  sampleCumulativeLoss?: GraphDataView<'float32'>;
  /** `lineCount + 1` first-sample rows per line, clamped to the sample capacity. */
  lineSampleOffsets?: GraphDataView<'uint32'>;
  /** Planar length of each line (0 for lines with fewer than two vertices). */
  lineLengths?: GraphDataView<'float32'>;
  /** Total gain per line over its emitted samples. */
  lineGains?: GraphDataView<'float32'>;
  /** Total loss per line (a positive magnitude). */
  lineLosses?: GraphDataView<'float32'>;
  /** Minimum finite sample value per line; NaN when none is finite. */
  lineMinimums?: GraphDataView<'float32'>;
  /** Maximum finite sample value per line; NaN when none is finite. */
  lineMaximums?: GraphDataView<'float32'>;
};

/**
 * Properties for {@link GPURasterProfile}.
 *
 * Per-frame (no recompile): raster, vertex and offset contents and `parameters` (extent, method,
 * nodata policy, spacing). Topology: `grid` size, `noDataValue`, `lineOffsets.length`,
 * vertex and sample capacities, and which optional outputs exist.
 */
export type GPURasterProfileProps = {
  /** Prefix for generated node IDs. Defaults to `'raster-profile'`. */
  id?: string;
  /** Raster grid: its width and height size the kernels; its extent places the samples. */
  grid: GPURasterMetadata;
  /** Row-major raster of at least `width * height` rows (row 0 is the grid's first row). NaN is nodata. */
  values: GraphDataView<'float32'>;
  /** Optional finite sentinel treated as nodata. */
  noDataValue?: number;
  /** Optional raster validity; zero marks a nodata cell. */
  validity?: GraphDataView<'uint32'>;
  /** Packed line vertices in grid world units, relative to the parameters' `localOrigin`. */
  positions: GraphDataView<'float32x2'>;
  /**
   * CSR spans: line `p` owns vertices `[lineOffsets[p], lineOffsets[p + 1])`; `lineCount + 1` rows,
   * with `lineCount` at most {@link getGPURasterProfileMaximumLineCount} (8388608 on a default device).
   */
  lineOffsets: GraphDataView<'uint32'>;
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
 * Largest line count (`lineOffsets.length - 1`) of {@link GPURasterProfile} on `device`: the line
 * summaries keep four `float32` (16 bytes) per line in one transient storage binding, so 8388608
 * on a default device (128 MiB bindings).
 */
export function getGPURasterProfileMaximumLineCount(device: Device): number {
  return getStorageBindingElementLimit(device, 16);
}

/**
 * Elevation profiles: samples a raster along polylines at a per-frame spacing.
 *
 * Distances are planar in extent units. Each line is sampled at along-distances `0, s, 2s, ...`
 * while below its length `L`, plus the final vertex at exactly `L`; a zero-length line (one vertex,
 * or coincident vertices) gives one sample and a line without vertices gives none. Sample rows are
 * ordered by line then distance. A line may emit at most `min(2^24, floor((2^32 - 1) / (lineCount +
 * 1)))` samples: beyond that its samples are truncated (the last row still sits at the final
 * vertex), and `overflow` is set. Samples beyond `sampleCapacity` are dropped and set `overflow`.
 *
 * Pipeline: a segmented scan of edge lengths over vertices (cumulative vertex distances; lines are
 * segments), per-line sample counts, `GPUScan` of the counts, a per-sample kernel that binary
 * searches its line and segment and interpolates the position, the shared raster sampling kernel
 * ({@link GPURasterSampling} semantics for `method` and `noDataPolicy`), then a segmented scan
 * over samples for cumulative gain and loss and the per-line summaries. Gain and loss sum positive
 * and negative differences between consecutive finite samples (skipping non-finite ones); minimum
 * and maximum cover finite samples. Both scans are reduce-then-scan over 256-item blocks, so the
 * work is parallel over vertices and samples and a single line of millions of samples is not a
 * single-invocation loop. The carry pass walks at most `ceil(blocks / 256)` blocks per lane.
 *
 * Summation order is a fixed tree (inside a block, then across blocks), so results are
 * deterministic, but a sum differs from a left-to-right serial sum in float32 rounding. Minimum,
 * maximum and which samples are skipped are unaffected; a sample count differs only where a line
 * length lies within that rounding of a multiple of the spacing. For `n` items the tree has height
 * at most `2 * ceil(ceil(n / 256) / 256) + 15` (23 for 2e5 samples; gain and loss add up to two
 * terms per combine, so twice that), so a distance or a total of non-negative terms is within about
 * that many float32 ulps relative to its exact value,
 * where a serial sum may drift by up to `n - 1`. Integer-valued or dyadic data below 2^24 units sum
 * exactly in any order. Line offsets must be non-decreasing (CSR), as the vertex scan finds a
 * vertex's line by binary search.
 *
 * Rows at or beyond `count` of the position, distance, value and line ID columns hold NaN (line IDs
 * {@link GPU_RASTER_PROFILE_NO_LINE_ID}); the cumulative columns are left unwritten there.
 *
 * Precision: line vertices, sample positions and the extent are float32 relative to the packer's
 * `localOrigin`. Absolute positions (the default origin `[0, 0]`) are quantized to the float32 ulp,
 * 0.5 m at a UTM northing of 4.6e6 m (2 m at 2e7 m in EPSG:3857). Set `localOrigin` to the raster
 * or tile corner, upload line vertices relative to it, and add it back to sample positions on the
 * CPU or as the layer's coordinate origin; positions then keep about `1.2e-7` of their distance
 * from it.
 */
export class GPURasterProfile implements GPUCommandNodeProducer {
  /** Prefix for every node ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPURasterProfileProps;
  /** Number of lines. */
  readonly lineCount: number;
  /** Sample row capacity. */
  readonly sampleCapacity: number;

  constructor(props: GPURasterProfileProps) {
    this.id = props.id ?? 'raster-profile';
    this.props = props;
    const {id} = this;
    const {output} = props;
    const pixelCount = validateRasterDescription(id, props.grid, props.noDataValue);
    validateRasterSamplingView(id, 'values', props.values, 'float32', pixelCount);
    if (props.validity) {
      validateRasterSamplingView(id, 'validity', props.validity, 'uint32', pixelCount);
    }
    validateRasterSamplingView(id, 'positions', props.positions, 'float32x2', 1);
    validateRasterSamplingView(id, 'lineOffsets', props.lineOffsets, 'uint32', 2);
    validateRasterSamplingView(
      id,
      'parameters',
      props.parameters,
      'float32',
      GPU_RASTER_PROFILE_PARAMETER_LENGTH
    );
    if (!output?.status?.count || !output.status.overflow) {
      throw new Error(`${id} needs output.status.count and output.status.overflow`);
    }
    this.lineCount = props.lineOffsets.length - 1;
    const sampleColumns = [
      output.samplePositions,
      output.sampleDistances,
      output.sampleValues,
      output.sampleLineIds,
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
      ['sampleLineIds', 'uint32'],
      ['sampleCumulativeGain', 'float32'],
      ['sampleCumulativeLoss', 'float32']
    ] as const;
    for (const [name, format] of sampleFormats) {
      if (output[name]) {
        validateRasterSamplingView(id, `output.${name}`, output[name], format, capacity);
      }
    }
    for (const name of ['count', 'overflow', 'requiredCount'] as const) {
      if (output.status[name]) {
        validateRasterSamplingView(id, `output.status.${name}`, output.status[name], 'uint32', 1);
      }
    }
    if (output.lineSampleOffsets) {
      validateRasterSamplingView(
        id,
        'output.lineSampleOffsets',
        output.lineSampleOffsets,
        'uint32',
        this.lineCount + 1
      );
    }
    for (const name of [
      'lineLengths',
      'lineGains',
      'lineLosses',
      'lineMinimums',
      'lineMaximums'
    ] as const) {
      if (output[name]) {
        validateRasterSamplingView(id, `output.${name}`, output[name], 'float32', this.lineCount);
      }
    }
    const {status, ...outputColumns} = output;
    const outputs = [...Object.values(outputColumns), ...Object.values(status)].filter(
      view => view !== undefined
    );
    if (new Set(outputs.map(view => view.buffer)).size !== outputs.length) {
      throw new Error(`${id} outputs must not share buffers`);
    }
    const inputBuffers = new Set(
      [props.values, props.validity, props.positions, props.lineOffsets, props.parameters]
        .filter(view => view !== undefined)
        .map(view => view.buffer)
    );
    if (outputs.some(view => inputBuffers.has(view.buffer))) {
      throw new Error(`${id} outputs must not share buffers with inputs`);
    }
  }

  /** Returns the vertex-distance scan, count, locate, sample, cumulative scan and finish nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, lineCount, sampleCapacity} = this;
    const {output} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      props.values,
      props.validity,
      props.positions,
      props.lineOffsets,
      props.parameters,
      ...Object.values({...output, status: undefined}),
      ...Object.values(output.status)
    ]);
    validateMaximumCount(
      id,
      'line count (lineOffsets.length - 1)',
      lineCount,
      getGPURasterProfileMaximumLineCount(graph.device)
    );
    const vertexCount = props.positions.length;
    const maximumPerLine = Math.max(
      1,
      Math.min(MAXIMUM_SAMPLES_PER_LINE, Math.floor(0xffffffff / (lineCount + 1)))
    );
    const nodes: GPUCommandNode<Parameters>[] = [];
    const vertexDistances = createTransientView(
      graph,
      `${id}-vertex-distances`,
      'float32',
      vertexCount
    );
    const sampleCounts = createTransientView(graph, `${id}-sample-counts`, 'uint32', lineCount + 1);
    const sampleOffsets = createTransientView(
      graph,
      `${id}-sample-offsets`,
      'uint32',
      lineCount + 1
    );
    const truncated = createTransientView(graph, `${id}-truncated`, 'uint32', 1);
    const samplePositions =
      output.samplePositions ??
      createTransientView(graph, `${id}-sample-positions`, 'float32x2', sampleCapacity);
    const sampleValues =
      output.sampleValues ??
      createTransientView(graph, `${id}-sample-values`, 'float32', sampleCapacity);
    const lineConstants = `const LINE_COUNT: u32 = ${lineCount}u;
const VERTEX_COUNT: u32 = ${vertexCount}u;
const SAMPLE_CAPACITY: u32 = ${sampleCapacity}u;
const MAXIMUM_PER_LINE: u32 = ${maximumPerLine}u;
fn getVertexBegin(line: u32) -> u32 {
  return min(lineOffsets[lineOffsetsOffset + line], VERTEX_COUNT);
}
fn getVertexEnd(line: u32, vertexBegin: u32) -> u32 {
  return max(min(lineOffsets[lineOffsetsOffset + line + 1u], VERTEX_COUNT), vertexBegin);
}`;

    // 1. Vertex distances: a segmented float sum over vertices (lines are segments), as a
    // block reduce, a one-workgroup carry scan of the block aggregates and a block down-sweep.
    const vertexBlockCount = getSegmentedScanBlockCount(vertexCount);
    const vertexAggregates = createTransientView(
      graph,
      `${id}-vertex-aggregates`,
      'uint32',
      vertexBlockCount * DISTANCE_SCAN_STATE_WORDS
    );
    const vertexScanBindings = (aggregatesAccess: 'read' | 'read_write'): WGSLKernelBinding[] => [
      {name: 'linePositions', view: props.positions, type: 'f32', access: 'read'},
      {name: 'lineOffsets', view: props.lineOffsets, type: 'u32', access: 'read'},
      {name: 'aggregates', view: vertexAggregates, type: 'u32', access: aggregatesAccess}
    ];
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
        bindings: vertexScanBindings('read_write'),
        invocationCount: vertexBlockCount * SEGMENTED_SCAN_WORKGROUP_SIZE,
        guardIndex: false,
        workgroupSize: SEGMENTED_SCAN_WORKGROUP_SIZE,
        declarations: `${lineConstants}
${getDistanceScanDeclarations({writesAggregates: true, loadsItems: true})}`,
        body: getScanBlockBody(`var itemState = getIdentityScanState();
  if (index < VERTEX_COUNT) {
    itemState = getVertexScanState(index);
  }`)
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-measure-carries`,
        operation: OPERATION,
        variant: 'measure-carries',
        bindings: [vertexScanBindings('read_write')[2]],
        invocationCount: SEGMENTED_SCAN_WORKGROUP_SIZE,
        guardIndex: false,
        workgroupSize: SEGMENTED_SCAN_WORKGROUP_SIZE,
        declarations: `const BLOCK_COUNT: u32 = ${vertexBlockCount}u;
${getDistanceScanDeclarations({writesAggregates: true, loadsItems: false})}`,
        body: getScanCarryBody()
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-measure-distances`,
        operation: OPERATION,
        variant: 'measure-distances',
        bindings: [
          ...vertexScanBindings('read'),
          {name: 'vertexDistances', view: vertexDistances, type: 'f32', access: 'read_write'}
        ],
        invocationCount: vertexBlockCount * SEGMENTED_SCAN_WORKGROUP_SIZE,
        guardIndex: false,
        workgroupSize: SEGMENTED_SCAN_WORKGROUP_SIZE,
        declarations: `${lineConstants}
${getDistanceScanDeclarations({writesAggregates: false, loadsItems: true})}`,
        body: `var itemState = getIdentityScanState();
  if (index < VERTEX_COUNT) {
    itemState = getVertexScanState(index);
  }
  let prefix = scanWorkgroup(localInvocationIndex, itemState);
  let result = combineScanStates(loadAggregate(workgroupIndex), prefix);
  if (index < VERTEX_COUNT) {
    vertexDistances[vertexDistancesOffset + index] = result.sum;
  }`
      }),
      // Per-line sample counts from each line's last cumulative vertex distance.
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-sample-counts`,
        operation: OPERATION,
        variant: 'sample-counts',
        bindings: [
          {name: 'lineOffsets', view: props.lineOffsets, type: 'u32', access: 'read'},
          {name: 'params', view: props.parameters, type: 'f32', access: 'read'},
          {name: 'vertexDistances', view: vertexDistances, type: 'f32', access: 'read'},
          {name: 'sampleCounts', view: sampleCounts, type: 'u32', access: 'read_write'},
          {name: 'truncated', view: truncated, type: 'atomic<u32>', access: 'read_write'}
        ],
        invocationCount: lineCount + 1,
        declarations: lineConstants,
        body: `if (index >= LINE_COUNT) {
    sampleCounts[sampleCountsOffset + index] = 0u;
    return;
  }
  let vertexBegin = getVertexBegin(index);
  let vertexEnd = getVertexEnd(index, vertexBegin);
  var count = 0u;
  if (vertexEnd > vertexBegin) {
    let distance = vertexDistances[vertexDistancesOffset + vertexEnd - 1u];
    count = 1u;
    if (distance > 0.0) {
      let spacing = params[paramsOffset + 10u];
      var segments = u32(min(ceil(distance / spacing), f32(MAXIMUM_PER_LINE)));
      if (segments > 0u && f32(segments - 1u) * spacing >= distance) {
        segments -= 1u;
      }
      if (f32(segments) * spacing < distance) {
        segments += 1u;
      }
      count = segments + 1u;
      if (count > MAXIMUM_PER_LINE) {
        count = MAXIMUM_PER_LINE;
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

    // 3. Per-sample line, distance and position.
    const locateBindings: WGSLKernelBinding[] = [
      {name: 'linePositions', view: props.positions, type: 'f32', access: 'read'},
      {name: 'lineOffsets', view: props.lineOffsets, type: 'u32', access: 'read'},
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
    if (output.sampleLineIds) {
      locateBindings.push({
        name: 'lineIdsOut',
        view: output.sampleLineIds,
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
        declarations: `${lineConstants}
${GET_NAN_WGSL}`,
        body: `let total = sampleOffsets[sampleOffsetsOffset + LINE_COUNT];
  if (index >= total) {
    positionsOut[positionsOutOffset + 2u * index] = getNaN();
    positionsOut[positionsOutOffset + 2u * index + 1u] = getNaN();
    ${output.sampleDistances ? 'distancesOut[distancesOutOffset + index] = getNaN();' : ''}
    ${output.sampleLineIds ? 'lineIdsOut[lineIdsOutOffset + index] = 0xffffffffu;' : ''}
    return;
  }
  var low = 0u;
  var high = LINE_COUNT;
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
  let line = low - 1u;
  let vertexBegin = getVertexBegin(line);
  let vertexEnd = getVertexEnd(line, vertexBegin);
  let firstSample = sampleOffsets[sampleOffsetsOffset + line];
  let sampleCount = sampleOffsets[sampleOffsetsOffset + line + 1u] - firstSample;
  let localSample = index - firstSample;
  let lastVertex = vertexEnd - 1u;
  let length = vertexDistances[vertexDistancesOffset + lastVertex];
  var distance = length;
  var positionX = linePositions[linePositionsOffset + 2u * lastVertex];
  var positionY = linePositions[linePositionsOffset + 2u * lastVertex + 1u];
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
      let startX = linePositions[linePositionsOffset + 2u * segmentStart];
      let startY = linePositions[linePositionsOffset + 2u * segmentStart + 1u];
      let endX = linePositions[linePositionsOffset + 2u * segmentStart + 2u];
      let endY = linePositions[linePositionsOffset + 2u * segmentStart + 3u];
      positionX = startX + (endX - startX) * t;
      positionY = startY + (endY - startY) * t;
    }
  }
  positionsOut[positionsOutOffset + 2u * index] = positionX;
  positionsOut[positionsOutOffset + 2u * index + 1u] = positionY;
  ${output.sampleDistances ? 'distancesOut[distancesOutOffset + index] = distance;' : ''}
  ${output.sampleLineIds ? 'lineIdsOut[lineIdsOutOffset + index] = line;' : ''}`
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
          width: props.grid.width,
          height: props.grid.height,
          noDataValue: props.noDataValue,
          hasValidity: Boolean(props.validity)
        }),
        body: `valuesOut[valuesOutOffset + index] = sampleRaster(
    samplePositions[samplePositionsOffset + 2u * index],
    samplePositions[samplePositionsOffset + 2u * index + 1u]
  );`
      })
    );

    // 5. Cumulative gain and loss and line summaries: a segmented scan over samples whose state
    // carries gain, loss, minimum and maximum (lines are segments), as block reduce, carry scan
    // and down-sweep. The down-sweep writes the per-sample columns and, at the last sample of each
    // line, its totals into a transient table that a per-line kernel copies to the outputs.
    const sampleBlockCount = getSegmentedScanBlockCount(sampleCapacity);
    const sampleAggregates = createTransientView(
      graph,
      `${id}-sample-aggregates`,
      'uint32',
      sampleBlockCount * PROFILE_SCAN_STATE_WORDS
    );
    const hasLineSummaries = Boolean(
      output.lineGains || output.lineLosses || output.lineMinimums || output.lineMaximums
    );
    const lineSummaries = hasLineSummaries
      ? createTransientView(graph, `${id}-line-summaries`, 'float32', 4 * lineCount)
      : undefined;
    const profileScanConstants = `const LINE_COUNT: u32 = ${lineCount}u;
const SAMPLE_CAPACITY: u32 = ${sampleCapacity}u;
${GET_NAN_WGSL}
${IS_FINITE_VALUE_WGSL}`;
    const profileScanInputs = (aggregatesAccess: 'read' | 'read_write'): WGSLKernelBinding[] => [
      {name: 'sampleOffsets', view: sampleOffsets, type: 'u32', access: 'read'},
      {name: 'sampleValues', view: sampleValues, type: 'f32', access: 'read'},
      {name: 'aggregates', view: sampleAggregates, type: 'u32', access: aggregatesAccess}
    ];
    const loadSampleState = `var itemState = getIdentityScanState();
  var line = 0u;
  let isActive = index < getEmittedSampleCount();
  if (isActive) {
    line = findSampleLine(index);
    itemState = getSampleScanState(index, line);
  }`;
    const cumulativeBindings = profileScanInputs('read');
    const cumulativeOutputs = [
      ['cumulativeGainOut', output.sampleCumulativeGain],
      ['cumulativeLossOut', output.sampleCumulativeLoss],
      ['lineSummaries', lineSummaries]
    ] as const;
    for (const [name, view] of cumulativeOutputs) {
      if (view) {
        cumulativeBindings.push({name, view, type: 'f32', access: 'read_write'});
      }
    }
    const hasCumulativeConsumer =
      hasLineSummaries || Boolean(output.sampleCumulativeGain || output.sampleCumulativeLoss);
    // Nothing reads the scan unless a cumulative column or a line summary is requested.
    if (hasCumulativeConsumer) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-cumulative-blocks`,
          operation: OPERATION,
          variant: 'cumulative-blocks',
          bindings: profileScanInputs('read_write'),
          invocationCount: sampleBlockCount * SEGMENTED_SCAN_WORKGROUP_SIZE,
          guardIndex: false,
          workgroupSize: SEGMENTED_SCAN_WORKGROUP_SIZE,
          declarations: `${profileScanConstants}
${getProfileScanDeclarations({writesAggregates: true, loadsItems: true})}`,
          body: getScanBlockBody(loadSampleState)
        }),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-cumulative-carries`,
          operation: OPERATION,
          variant: 'cumulative-carries',
          bindings: [profileScanInputs('read_write')[2]],
          invocationCount: SEGMENTED_SCAN_WORKGROUP_SIZE,
          guardIndex: false,
          workgroupSize: SEGMENTED_SCAN_WORKGROUP_SIZE,
          declarations: `const BLOCK_COUNT: u32 = ${sampleBlockCount}u;
${getProfileScanDeclarations({writesAggregates: true, loadsItems: false})}`,
          body: getScanCarryBody()
        }),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-cumulative`,
          operation: OPERATION,
          variant: 'cumulative',
          bindings: cumulativeBindings,
          invocationCount: sampleBlockCount * SEGMENTED_SCAN_WORKGROUP_SIZE,
          guardIndex: false,
          workgroupSize: SEGMENTED_SCAN_WORKGROUP_SIZE,
          declarations: `${profileScanConstants}
${getProfileScanDeclarations({writesAggregates: false, loadsItems: true})}`,
          body: `${loadSampleState}
  let prefix = scanWorkgroup(localInvocationIndex, itemState);
  let result = combineScanStates(loadAggregate(workgroupIndex), prefix);
  if (isActive) {
    ${output.sampleCumulativeGain ? 'cumulativeGainOut[cumulativeGainOutOffset + index] = result.gain;' : ''}
    ${output.sampleCumulativeLoss ? 'cumulativeLossOut[cumulativeLossOutOffset + index] = result.loss;' : ''}
    ${
      hasLineSummaries
        ? `// The last emitted sample of a line holds the line totals.
    if (min(sampleOffsets[sampleOffsetsOffset + line + 1u], SAMPLE_CAPACITY) == index + 1u) {
      let summaryBase = lineSummariesOffset + 4u * line;
      lineSummaries[summaryBase] = result.gain;
      lineSummaries[summaryBase + 1u] = result.loss;
      lineSummaries[summaryBase + 2u] = select(getNaN(), result.minimum, result.hasValue != 0u);
      lineSummaries[summaryBase + 3u] = select(getNaN(), result.maximum, result.hasValue != 0u);
    }`
        : ''
    }
  }`
        })
      );
    }
    if (lineSummaries) {
      const summaryBindings: WGSLKernelBinding[] = [
        {name: 'sampleOffsets', view: sampleOffsets, type: 'u32', access: 'read'},
        {name: 'lineSummaries', view: lineSummaries, type: 'f32', access: 'read'}
      ];
      for (const [name, view] of [
        ['lineGainsOut', output.lineGains],
        ['lineLossesOut', output.lineLosses],
        ['lineMinimumsOut', output.lineMinimums],
        ['lineMaximumsOut', output.lineMaximums]
      ] as const) {
        if (view) {
          summaryBindings.push({name, view, type: 'f32', access: 'read_write'});
        }
      }
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-summaries`,
          operation: OPERATION,
          variant: 'summaries',
          bindings: summaryBindings,
          invocationCount: lineCount,
          declarations: `const SAMPLE_CAPACITY: u32 = ${sampleCapacity}u;
${GET_NAN_WGSL}`,
          // Lines without emitted samples have no table row: gain and loss 0, extrema NaN.
          body: `let firstSample = min(sampleOffsets[sampleOffsetsOffset + index], SAMPLE_CAPACITY);
  let endSample = min(sampleOffsets[sampleOffsetsOffset + index + 1u], SAMPLE_CAPACITY);
  let hasSamples = endSample > firstSample;
  let summaryBase = lineSummariesOffset + 4u * index;
  ${output.lineGains ? 'lineGainsOut[lineGainsOutOffset + index] = select(0.0, lineSummaries[summaryBase], hasSamples);' : ''}
  ${output.lineLosses ? 'lineLossesOut[lineLossesOutOffset + index] = select(0.0, lineSummaries[summaryBase + 1u], hasSamples);' : ''}
  ${output.lineMinimums ? 'lineMinimumsOut[lineMinimumsOutOffset + index] = select(getNaN(), lineSummaries[summaryBase + 2u], hasSamples);' : ''}
  ${output.lineMaximums ? 'lineMaximumsOut[lineMaximumsOutOffset + index] = select(getNaN(), lineSummaries[summaryBase + 3u], hasSamples);' : ''}`
        })
      );
    }

    // 6. Per-line offsets and lengths, then the scalar count, overflow and total.
    if (output.lineSampleOffsets || output.lineLengths) {
      const finishBindings: WGSLKernelBinding[] = [
        {name: 'sampleOffsets', view: sampleOffsets, type: 'u32', access: 'read'},
        {name: 'lineOffsets', view: props.lineOffsets, type: 'u32', access: 'read'},
        {name: 'vertexDistances', view: vertexDistances, type: 'f32', access: 'read'}
      ];
      if (output.lineSampleOffsets) {
        finishBindings.push({
          name: 'lineSampleOffsetsOut',
          view: output.lineSampleOffsets,
          type: 'u32',
          access: 'read_write'
        });
      }
      if (output.lineLengths) {
        finishBindings.push({
          name: 'lineLengthsOut',
          view: output.lineLengths,
          type: 'f32',
          access: 'read_write'
        });
      }
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-finish-lines`,
          operation: OPERATION,
          variant: 'finish-lines',
          bindings: finishBindings,
          invocationCount: lineCount + 1,
          declarations: lineConstants,
          body: `${output.lineSampleOffsets ? 'lineSampleOffsetsOut[lineSampleOffsetsOutOffset + index] = min(sampleOffsets[sampleOffsetsOffset + index], SAMPLE_CAPACITY);' : ''}
  if (index < LINE_COUNT) {
    let vertexBegin = getVertexBegin(index);
    let vertexEnd = getVertexEnd(index, vertexBegin);
    var length = 0.0;
    if (vertexEnd > vertexBegin) {
      length = vertexDistances[vertexDistancesOffset + vertexEnd - 1u];
    }
    ${output.lineLengths ? 'lineLengthsOut[lineLengthsOutOffset + index] = length;' : ''}
  }`
        })
      );
    }
    const countBindings: WGSLKernelBinding[] = [
      {name: 'sampleOffsets', view: sampleOffsets, type: 'u32', access: 'read'},
      {name: 'truncated', view: truncated, type: 'u32', access: 'read'},
      {name: 'countOut', view: output.status.count, type: 'u32', access: 'read_write'},
      {name: 'overflowOut', view: output.status.overflow, type: 'u32', access: 'read_write'}
    ];
    if (output.status.requiredCount) {
      countBindings.push({
        name: 'totalOut',
        view: output.status.requiredCount,
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
        declarations: `const LINE_COUNT: u32 = ${lineCount}u;
const SAMPLE_CAPACITY: u32 = ${sampleCapacity}u;`,
        body: `let total = sampleOffsets[sampleOffsetsOffset + LINE_COUNT];
  countOut[countOutOffset] = min(total, SAMPLE_CAPACITY);
  overflowOut[overflowOutOffset] = select(0u, 1u, total > SAMPLE_CAPACITY || truncated[truncatedOffset] != 0u);
  ${output.status.requiredCount ? 'totalOut[totalOutOffset] = total;' : ''}`
      })
    );
    return nodes;
  }
}
