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
import {createFillNode, createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import type {WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {
  getRasterAlgebraValueWGSL,
  validateRasterAlgebraAliasing,
  validateRasterAlgebraCount,
  validateRasterAlgebraGraph,
  validateRasterAlgebraNoData,
  validateRasterAlgebraView
} from '../raster-algebra/raster-algebra-utils';
import {
  GPU_RASTER_STRETCH_MAXIMUM_EXTENT,
  GPU_RASTER_STRETCH_PARAMETER_LENGTH,
  GPU_RASTER_STRETCH_STATISTICS_LENGTH
} from './raster-stretch-parameters';

const OPERATION = 'GPURasterStretch';
const DEFAULT_BIN_COUNT = 1024;
const DEFAULT_LUT_SIZE = 256;
const MAXIMUM_PALETTE_SIZE = 65536;

/** Caller-owned outputs of {@link GPURasterStretch}. At least one is required. */
export type GPURasterStretchOutput = {
  /** Stretched value per cell in `[0, 1]`; NaN for nodata. At least `width * height` rows. */
  stretched?: GraphDataView<'float32'>;
  /** Packed rgba8 per cell (`r | g << 8 | b << 16 | a << 24`); 0 for nodata. Requires `palette`. */
  colors?: GraphDataView<'uint32'>;
  /** Stretch curve sampled at `lutSize` evenly spaced values across `[lo, hi]`. */
  lut?: GraphDataView<'float32'>;
  /** `lut` mapped through the palette, `lutSize` rows. Requires `palette`. */
  lutColors?: GraphDataView<'uint32'>;
  /** Histogram of the included valid cells inside the domain; exactly `binCount` rows. */
  histogram?: GraphDataView<'uint32'>;
  /** Eight rows `[domainMin, domainMax, lo, hi, validCount, binWidth, 0, 0]` (see `GPU_RASTER_STRETCH_STATISTICS_INDEX`). */
  statistics?: GraphDataView<'float32'>;
};

/**
 * Properties for {@link GPURasterStretch}.
 *
 * Per-frame (no recompile): `values` contents, `regionMask` contents, `palette` contents and
 * `parameters` written with `getGPURasterStretchParameterValues`. Topology: width, height,
 * `binCount`, `lutSize`, `noDataValue`, palette size, and which views exist.
 */
export type GPURasterStretchProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'raster-stretch'`. */
  id?: string;
  /** Packed row-major float32 raster; cell `(column, row)` is row `row * width + column`. */
  values: GraphDataView<'float32'>;
  /** Raster width in cells, `[1, 2^24]`. */
  width: number;
  /** Raster height in cells, `[1, 2^24]`. */
  height: number;
  /** Optional finite nodata sentinel compared exactly. NaN is always nodata. */
  noDataValue?: number;
  /** Optional validity per cell; zero marks nodata. */
  validity?: GraphDataView<'uint32'>;
  /** Optional per-frame region mask per cell; nonzero cells contribute to statistics. */
  regionMask?: GraphDataView<'uint32'>;
  /** Per-frame float32 view of {@link GPU_RASTER_STRETCH_PARAMETER_LENGTH} values. */
  parameters: GraphDataView<'float32'>;
  /** Optional packed rgba8 palette; its length is the palette size (topology), contents per frame. */
  palette?: GraphDataView<'uint32'>;
  /** Histogram bins, integer in `[16, 65536]`. Defaults to 1024. */
  binCount?: number;
  /** Lookup-table samples, integer in `[2, 65536]`. Defaults to 256. */
  lutSize?: number;
  /** Caller-owned outputs. */
  output: GPURasterStretchOutput;
};

/**
 * Raster contrast stretch and colormap helper: linear, percentile and histogram-equalization
 * stretches of a float32 raster, computed on the GPU with integer atomics only (deterministic).
 *
 * Pipeline: (1) exact min/max of the included valid finite cells through order-preserving u32
 * keys, (2) a `binCount` histogram over the domain, (3) an inclusive `GPUScan` CDF, (4) a finalize
 * kernel finding the stretch bounds `lo`/`hi` and the statistics row, (5) a lookup table and
 * (6) a per-cell apply step producing `stretched` and `colors`.
 *
 * Semantics:
 * - Statistics include cells that are inside the `window`, nonzero in `regionMask`, not nodata and
 *   finite (infinities are values for the apply step, where they clamp to 0 or 1, but are excluded
 *   from min/max and the histogram). The apply step always runs on every cell.
 * - Histogram bins are equal-width over the domain; the last bin includes the maximum. Cells outside
 *   an explicit domain are not counted.
 * - Percentile bounds find the first bin whose cumulative count reaches `p * total` and interpolate
 *   linearly inside it, so `lo`/`hi` are within one bin width (`binWidth`) of the exact percentile.
 * - After the mode, `t` goes through `t ** gamma` and then the sigmoidal contrast.
 * - A degenerate range (`lo >= hi`, e.g. a constant raster, or no included cell for percentile and
 *   equalize) maps values below `lo` to 0, above `hi` to 1 and equal values to 0.5, skipping gamma
 *   and sigmoid.
 * - With no included cell and an automatic domain, `statistics` holds NaN domain and bounds and
 *   `validCount` 0; valid cells then use the degenerate mapping against `lo = hi = 0`.
 */
export class GPURasterStretch implements GPUCommandNodeProducer {
  /** Prefix for every node ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPURasterStretchProps;
  /** Histogram bin count. */
  readonly binCount: number;
  /** Lookup table size. */
  readonly lutSize: number;

  constructor(props: GPURasterStretchProps) {
    this.id = props.id ?? 'raster-stretch';
    this.props = props;
    const {id} = this;
    const {output, width, height} = props;
    validateRasterAlgebraCount(id, 'width', width, 1, GPU_RASTER_STRETCH_MAXIMUM_EXTENT);
    validateRasterAlgebraCount(id, 'height', height, 1, GPU_RASTER_STRETCH_MAXIMUM_EXTENT);
    const cellCount = width * height;
    if (cellCount > 0xffffffff) {
      throw new Error(`${id} width * height must fit in uint32`);
    }
    this.binCount = props.binCount ?? DEFAULT_BIN_COUNT;
    this.lutSize = props.lutSize ?? DEFAULT_LUT_SIZE;
    validateRasterAlgebraCount(id, 'binCount', this.binCount, 16, 65536);
    validateRasterAlgebraCount(id, 'lutSize', this.lutSize, 2, 65536);
    validateRasterAlgebraNoData(id, props.noDataValue);
    validateRasterAlgebraView(id, 'values', props.values, 'float32', cellCount);
    if (!props.values) {
      throw new Error(`${id} needs values`);
    }
    validateRasterAlgebraView(id, 'validity', props.validity, 'uint32', cellCount);
    validateRasterAlgebraView(id, 'regionMask', props.regionMask, 'uint32', cellCount);
    validateRasterAlgebraView(
      id,
      'parameters',
      props.parameters,
      'float32',
      GPU_RASTER_STRETCH_PARAMETER_LENGTH
    );
    if (!props.parameters) {
      throw new Error(`${id} needs parameters`);
    }
    validateRasterAlgebraView(id, 'palette', props.palette, 'uint32', 1);
    if (props.palette && props.palette.length > MAXIMUM_PALETTE_SIZE) {
      throw new Error(`${id} palette must hold at most ${MAXIMUM_PALETTE_SIZE} colors`);
    }
    if (!output || Object.values(output).every(view => view === undefined)) {
      throw new Error(`${id} needs at least one output`);
    }
    if ((output.colors || output.lutColors) && !props.palette) {
      throw new Error(`${id} output.colors and output.lutColors require palette`);
    }
    validateRasterAlgebraView(id, 'output.stretched', output.stretched, 'float32', cellCount);
    validateRasterAlgebraView(id, 'output.colors', output.colors, 'uint32', cellCount);
    validateRasterAlgebraView(id, 'output.lut', output.lut, 'float32', this.lutSize);
    validateRasterAlgebraView(id, 'output.lutColors', output.lutColors, 'uint32', this.lutSize);
    validateRasterAlgebraView(id, 'output.histogram', output.histogram, 'uint32', this.binCount);
    if (output.histogram && output.histogram.length !== this.binCount) {
      throw new Error(`${id} output.histogram must hold exactly binCount (${this.binCount}) rows`);
    }
    validateRasterAlgebraView(
      id,
      'output.statistics',
      output.statistics,
      'float32',
      GPU_RASTER_STRETCH_STATISTICS_LENGTH
    );
    validateRasterAlgebraAliasing(
      id,
      [
        output.stretched,
        output.colors,
        output.lut,
        output.lutColors,
        output.histogram,
        output.statistics
      ],
      [props.values, props.validity, props.regionMask, props.parameters, props.palette]
    );
  }

  /** Returns the reduction, histogram, scan, finalize, lookup-table and apply nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, binCount, lutSize} = this;
    const {output, width, height} = props;
    const cellCount = width * height;
    validateRasterAlgebraGraph(id, graph, [
      props.values,
      props.validity,
      props.regionMask,
      props.parameters,
      props.palette,
      output.stretched,
      output.colors,
      output.lut,
      output.lutColors,
      output.histogram,
      output.statistics
    ]);
    const nodes: GPUCommandNode<Parameters>[] = [];
    const keys = createTransientView(graph, `${id}-keys`, 'uint32', 4);
    const state = createTransientView(graph, `${id}-state`, 'float32', 8);
    const histogram =
      output.histogram ?? createTransientView(graph, `${id}-histogram`, 'uint32', binCount);
    const cdf = createTransientView(graph, `${id}-cdf`, 'uint32', binCount);
    const needsApply = Boolean(output.stretched || output.colors);
    const needsLut = Boolean(output.lut || output.lutColors);
    const paletteSize = props.palette?.length ?? 0;

    const cellInputs: WGSLKernelBinding[] = [
      {name: 'values', view: props.values, type: 'f32', access: 'read'}
    ];
    if (props.validity) {
      cellInputs.push({name: 'validity', view: props.validity, type: 'u32', access: 'read'});
    }
    const statisticsInputs: WGSLKernelBinding[] = [...cellInputs];
    if (props.regionMask) {
      statisticsInputs.push({
        name: 'regionMask',
        view: props.regionMask,
        type: 'u32',
        access: 'read'
      });
    }
    const paramsBinding: WGSLKernelBinding = {
      name: 'params',
      view: props.parameters,
      type: 'f32',
      access: 'read'
    };
    const constants = `const WIDTH: u32 = ${width}u;
const HEIGHT: u32 = ${height}u;
const BIN_COUNT: u32 = ${binCount}u;
const LUT_SIZE: u32 = ${lutSize}u;
const PALETTE_SIZE: u32 = ${paletteSize}u;`;
    const valueWGSL = `${constants}
${getRasterAlgebraValueWGSL(props.noDataValue)}
fn readCellValue(index: u32, value: ptr<function, f32>) -> bool {
  let sample = values[valuesOffset + index];
  *value = sample;
  var isValid = !isNoDataValue(sample);
  ${props.validity ? 'isValid = isValid && validity[validityOffset + index] != 0u;' : ''}
  return isValid;
}`;
    const statisticsWGSL = `${valueWGSL}
// Statistics include cells inside the window, in the region mask, valid and finite.
fn readStatisticsValue(index: u32, value: ptr<function, f32>) -> bool {
  let column = index % WIDTH;
  let row = index / WIDTH;
  let column0 = min(u32(params[paramsOffset]), WIDTH);
  let row0 = min(u32(params[paramsOffset + 1u]), HEIGHT);
  let column1 = min(u32(params[paramsOffset + 2u]), WIDTH);
  let row1 = min(u32(params[paramsOffset + 3u]), HEIGHT);
  if (column < column0 || column >= column1 || row < row0 || row >= row1) {
    return false;
  }
  ${props.regionMask ? 'if (regionMask[regionMaskOffset + index] == 0u) { return false; }' : ''}
  var sample = 0.0;
  if (!readCellValue(index, &sample)) {
    return false;
  }
  *value = sample;
  return isFiniteValue(sample);
}`;

    // 1. Minimum and maximum keys plus the included-cell count.
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-clear-keys`,
        operation: OPERATION,
        variant: 'clear-keys',
        bindings: [{name: 'keys', view: keys, type: 'u32', access: 'read_write'}],
        invocationCount: 1,
        body: `keys[keysOffset] = 0xffffffffu;
  keys[keysOffset + 1u] = 0u;
  keys[keysOffset + 2u] = 0u;
  keys[keysOffset + 3u] = 0u;`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-reduce`,
        operation: OPERATION,
        variant: 'reduce',
        bindings: [
          ...statisticsInputs,
          paramsBinding,
          {name: 'keys', view: keys, type: 'atomic<u32>', access: 'read_write'}
        ],
        invocationCount: cellCount,
        declarations: `${statisticsWGSL}
fn getOrderedKey(value: f32) -> u32 {
  let bits = bitcast<u32>(value);
  return bits ^ select(0x80000000u, 0xffffffffu, (bits >> 31u) != 0u);
}`,
        body: `var value = 0.0;
  if (!readStatisticsValue(index, &value)) {
    return;
  }
  atomicAdd(&keys[keysOffset + 2u], 1u);
  if (params[paramsOffset + 4u] == 0.0) {
    let key = getOrderedKey(value);
    atomicMin(&keys[keysOffset], key);
    atomicMax(&keys[keysOffset + 1u], key);
  }`
      }),
      // 2. Domain, bin scale and bin width.
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-domain`,
        operation: OPERATION,
        variant: 'domain',
        bindings: [
          paramsBinding,
          {name: 'keys', view: keys, type: 'u32', access: 'read'},
          {name: 'state', view: state, type: 'f32', access: 'read_write'}
        ],
        invocationCount: 1,
        declarations: `const BIN_COUNT: u32 = ${binCount}u;
fn decodeOrderedKey(key: u32) -> f32 {
  return bitcast<f32>(key ^ select(0xffffffffu, 0x80000000u, (key >> 31u) != 0u));
}`,
        body: `let count = keys[keysOffset + 2u];
  var domainMin = 0.0;
  var domainMax = 0.0;
  if (params[paramsOffset + 4u] != 0.0) {
    domainMin = params[paramsOffset + 5u];
    domainMax = params[paramsOffset + 6u];
  } else if (count > 0u) {
    domainMin = decodeOrderedKey(keys[keysOffset]);
    domainMax = decodeOrderedKey(keys[keysOffset + 1u]);
  }
  let hasRange = domainMax > domainMin;
  let range = domainMax - domainMin;
  state[stateOffset] = domainMin;
  state[stateOffset + 1u] = domainMax;
  state[stateOffset + 2u] = select(0.0, f32(BIN_COUNT) / range, hasRange);
  state[stateOffset + 3u] = select(0.0, range / f32(BIN_COUNT), hasRange);
  state[stateOffset + 4u] = domainMin;
  state[stateOffset + 5u] = domainMax;
  state[stateOffset + 6u] = 0.0;
  state[stateOffset + 7u] = f32(count);`
      }),
      // 3. Histogram (cleared every encoding, transient buffers are not zero-initialised).
      createFillNode<Parameters>(graph, {
        id: `${id}-clear-histogram`,
        operation: OPERATION,
        view: histogram,
        type: 'u32',
        value: '0u'
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-histogram`,
        operation: OPERATION,
        variant: 'histogram',
        bindings: [
          ...statisticsInputs,
          paramsBinding,
          {name: 'state', view: state, type: 'f32', access: 'read'},
          {name: 'histogram', view: histogram, type: 'atomic<u32>', access: 'read_write'}
        ],
        invocationCount: cellCount,
        declarations: statisticsWGSL,
        body: `var value = 0.0;
  if (!readStatisticsValue(index, &value)) {
    return;
  }
  let domainMin = state[stateOffset];
  let domainMax = state[stateOffset + 1u];
  if (value < domainMin || value > domainMax) {
    return;
  }
  let position = (value - domainMin) * state[stateOffset + 2u];
  let bin = min(u32(position), BIN_COUNT - 1u);
  atomicAdd(&histogram[histogramOffset + bin], 1u);`
      }),
      // 4. Cumulative distribution.
      ...new GPUScan({
        id: `${id}-cdf-scan`,
        input: histogram,
        output: cdf,
        mode: 'inclusive'
      }).getCommandNodes(graph)
    );

    // 5. Bounds and statistics.
    const finalizeBindings: WGSLKernelBinding[] = [
      paramsBinding,
      {name: 'state', view: state, type: 'f32', access: 'read_write'},
      {name: 'cdf', view: cdf, type: 'u32', access: 'read'}
    ];
    if (output.statistics) {
      finalizeBindings.push({
        name: 'statisticsOut',
        view: output.statistics,
        type: 'f32',
        access: 'read_write'
      });
    }
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-finalize`,
        operation: OPERATION,
        variant: 'finalize',
        bindings: finalizeBindings,
        invocationCount: 1,
        declarations: `${constants}
${getRasterAlgebraValueWGSL(undefined)}
// Smallest bin whose cumulative count is positive and reaches target, then linear inside it.
fn getPercentileValue(rank: f32) -> f32 {
  var low = 0u;
  var high = BIN_COUNT - 1u;
  loop {
    if (low >= high) {
      break;
    }
    let middle = (low + high) / 2u;
    let cumulative = cdf[cdfOffset + middle];
    if (cumulative > 0u && f32(cumulative) >= rank) {
      high = middle;
    } else {
      low = middle + 1u;
    }
  }
  var before = 0u;
  if (low > 0u) {
    before = cdf[cdfOffset + low - 1u];
  }
  let count = cdf[cdfOffset + low] - before;
  var fraction = 0.0;
  if (count > 0u) {
    fraction = clamp((rank - f32(before)) / f32(count), 0.0, 1.0);
  }
  let domainMin = state[stateOffset];
  let domainMax = state[stateOffset + 1u];
  return clamp(domainMin + (f32(low) + fraction) * state[stateOffset + 3u], domainMin, domainMax);
}`,
        body: `let domainMin = state[stateOffset];
  let domainMax = state[stateOffset + 1u];
  let validCount = state[stateOffset + 7u];
  let total = cdf[cdfOffset + BIN_COUNT - 1u];
  let mode = u32(params[paramsOffset + 7u]);
  var lo = domainMin;
  var hi = domainMax;
  if (mode != 0u && total == 0u) {
    hi = lo;
  } else if (mode == 1u) {
    lo = getPercentileValue(params[paramsOffset + 8u] * f32(total));
    hi = getPercentileValue(params[paramsOffset + 9u] * f32(total));
  }
  state[stateOffset + 4u] = lo;
  state[stateOffset + 5u] = hi;
  state[stateOffset + 6u] = f32(total);
  ${
    output.statistics
      ? `let isEmpty = params[paramsOffset + 4u] == 0.0 && validCount == 0.0;
  let nan = getNaN();
  statisticsOut[statisticsOutOffset] = select(domainMin, nan, isEmpty);
  statisticsOut[statisticsOutOffset + 1u] = select(domainMax, nan, isEmpty);
  statisticsOut[statisticsOutOffset + 2u] = select(lo, nan, isEmpty);
  statisticsOut[statisticsOutOffset + 3u] = select(hi, nan, isEmpty);
  statisticsOut[statisticsOutOffset + 4u] = validCount;
  statisticsOut[statisticsOutOffset + 5u] = select(state[stateOffset + 3u], nan, isEmpty);
  statisticsOut[statisticsOutOffset + 6u] = 0.0;
  statisticsOut[statisticsOutOffset + 7u] = 0.0;`
      : ''
  }`
      })
    );

    if (!needsApply && !needsLut) {
      return nodes;
    }
    const stretchWGSL = `${getStretchWGSL()}
${props.palette ? getPaletteWGSL() : ''}`;
    const stretchInputs: WGSLKernelBinding[] = [
      paramsBinding,
      {name: 'state', view: state, type: 'f32', access: 'read'},
      {name: 'cdf', view: cdf, type: 'u32', access: 'read'}
    ];
    const paletteBinding: WGSLKernelBinding[] = props.palette
      ? [{name: 'palette', view: props.palette, type: 'u32', access: 'read'}]
      : [];

    // 6. Lookup table.
    if (needsLut) {
      const bindings = [...stretchInputs];
      if (output.lut) {
        bindings.push({name: 'lutOut', view: output.lut, type: 'f32', access: 'read_write'});
      }
      if (output.lutColors) {
        bindings.push({
          name: 'lutColorsOut',
          view: output.lutColors,
          type: 'u32',
          access: 'read_write'
        });
      }
      bindings.push(...paletteBinding);
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-lut`,
          operation: OPERATION,
          variant: 'lut',
          bindings,
          invocationCount: lutSize,
          declarations: `${constants}
${stretchWGSL}`,
          body: `let lo = state[stateOffset + 4u];
  let hi = state[stateOffset + 5u];
  let value = lo + (f32(index) + 0.5) / f32(LUT_SIZE) * (hi - lo);
  let t = getStretchedValue(value);
  ${output.lut ? 'lutOut[lutOutOffset + index] = t;' : ''}
  ${output.lutColors ? 'lutColorsOut[lutColorsOutOffset + index] = getPaletteColor(t);' : ''}`
        })
      );
    }

    // 7. Apply to every cell.
    if (needsApply) {
      const bindings = [...cellInputs, ...stretchInputs];
      if (output.stretched) {
        bindings.push({
          name: 'stretchedOut',
          view: output.stretched,
          type: 'f32',
          access: 'read_write'
        });
      }
      if (output.colors) {
        bindings.push({name: 'colorsOut', view: output.colors, type: 'u32', access: 'read_write'});
      }
      bindings.push(...paletteBinding);
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-apply`,
          operation: OPERATION,
          variant: 'apply',
          bindings,
          invocationCount: cellCount,
          declarations: `${valueWGSL}
${stretchWGSL}`,
          body: `var value = 0.0;
  if (!readCellValue(index, &value)) {
    ${output.stretched ? 'stretchedOut[stretchedOutOffset + index] = getNaN();' : ''}
    ${output.colors ? 'colorsOut[colorsOutOffset + index] = 0u;' : ''}
    return;
  }
  let t = getStretchedValue(value);
  ${output.stretched ? 'stretchedOut[stretchedOutOffset + index] = t;' : ''}
  ${output.colors ? 'colorsOut[colorsOutOffset + index] = getPaletteColor(t);' : ''}`
        })
      );
    }
    return nodes;
  }
}

/** WGSL `getStretchedValue(value)`: mode, gamma and sigmoid over `params`, `state` and `cdf`. */
function getStretchWGSL(): string {
  return /* wgsl */ `
fn applyCurve(unit: f32) -> f32 {
  var t = unit;
  let gamma = params[paramsOffset + 10u];
  if (gamma != 1.0 && t > 0.0 && t < 1.0) {
    t = pow(t, gamma);
  }
  let contrast = params[paramsOffset + 11u];
  if (contrast > 0.0) {
    let midpoint = params[paramsOffset + 12u];
    let lowEnd = 1.0 / (1.0 + exp(contrast * midpoint));
    let highEnd = 1.0 / (1.0 + exp(contrast * (midpoint - 1.0)));
    let denominator = highEnd - lowEnd;
    if (denominator > 0.0) {
      t = (1.0 / (1.0 + exp(contrast * (midpoint - t))) - lowEnd) / denominator;
    }
    t = clamp(t, 0.0, 1.0);
  }
  return t;
}

fn getStretchedValue(value: f32) -> f32 {
  let lo = state[stateOffset + 4u];
  let hi = state[stateOffset + 5u];
  if (hi <= lo) {
    return select(select(0.5, 1.0, value > hi), 0.0, value < lo);
  }
  var unit = 0.0;
  if (u32(params[paramsOffset + 7u]) == 2u) {
    let domainMin = state[stateOffset];
    let domainMax = state[stateOffset + 1u];
    if (value <= domainMin) {
      unit = 0.0;
    } else if (value >= domainMax) {
      unit = 1.0;
    } else {
      let position = (value - domainMin) * state[stateOffset + 2u];
      let bin = min(u32(position), BIN_COUNT - 1u);
      let fraction = clamp(position - f32(bin), 0.0, 1.0);
      var before = 0u;
      if (bin > 0u) {
        before = cdf[cdfOffset + bin - 1u];
      }
      let count = cdf[cdfOffset + bin] - before;
      unit = (f32(before) + fraction * f32(count)) / state[stateOffset + 6u];
    }
  } else {
    unit = clamp((value - lo) / (hi - lo), 0.0, 1.0);
  }
  return applyCurve(unit);
}`;
}

/** WGSL `getPaletteColor(t)`: nearest or linear lookup into the packed rgba8 `palette`. */
function getPaletteWGSL(): string {
  return /* wgsl */ `
fn unpackColor(color: u32) -> vec4<f32> {
  return vec4<f32>(
    f32(color & 255u),
    f32((color >> 8u) & 255u),
    f32((color >> 16u) & 255u),
    f32((color >> 24u) & 255u)
  );
}

fn getPaletteColor(t: f32) -> u32 {
  if (params[paramsOffset + 13u] == 0.0 || PALETTE_SIZE == 1u) {
    let index = min(u32(t * f32(PALETTE_SIZE)), PALETTE_SIZE - 1u);
    return palette[paletteOffset + index];
  }
  let position = t * f32(PALETTE_SIZE - 1u);
  let lower = min(u32(position), PALETTE_SIZE - 2u);
  let fraction = position - f32(lower);
  let first = unpackColor(palette[paletteOffset + lower]);
  let second = unpackColor(palette[paletteOffset + lower + 1u]);
  let channels = vec4<u32>(clamp(floor(mix(first, second, fraction) + 0.5), vec4<f32>(0.0), vec4<f32>(255.0)));
  return channels.x | (channels.y << 8u) | (channels.z << 16u) | (channels.w << 24u);
}`;
}
