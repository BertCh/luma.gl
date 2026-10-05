// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GraphVectorView,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {
  createMapGraphKernelNode,
  getWGSLFloatLiteral,
  type MapGraphKernelBinding
} from '../map-graph-kernels';
import type {GPUMapGraphRecipe} from '../map-graph-types';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../map-graph-utils';
import {GPU_FOCAL_STATISTICS_PARAMETER_LENGTH} from './spatial-interpolation-parameters';

const OPERATION = 'GPUFocalStatistics';
/** Largest compile-time window radius: a `(2 * 64 + 1)^2` window is 16,641 reads per cell. */
const MAXIMUM_FOCAL_RADIUS = 64;

/**
 * Caller-owned row-major outputs of {@link GPUFocalStatistics}, each at least `width * height`
 * rows. Provide any non-empty subset. Float outputs are NaN for nodata cells.
 */
export type GPUFocalStatisticsOutput = {
  /** Mean of valid window cells. */
  mean?: GraphDataView<'float32'>;
  /** Sum of valid window cells, accumulated in fixed row-major window order. */
  sum?: GraphDataView<'float32'>;
  /** Minimum of valid window cells. */
  min?: GraphDataView<'float32'>;
  /** Maximum of valid window cells. */
  max?: GraphDataView<'float32'>;
  /** `max - min` of valid window cells. */
  range?: GraphDataView<'float32'>;
  /** Population standard deviation, `sqrt(sum((x - mean)^2) / n)`, from a centered second pass. */
  standardDeviation?: GraphDataView<'float32'>;
  /** Number of valid window cells, written for every cell (also nodata ones). */
  count?: GraphDataView<'uint32'>;
};

/**
 * Properties for {@link GPUFocalStatistics}.
 *
 * Per-frame (no recompile): the contents of `parameters` (radius, shape, minimum count, center
 * nodata policy) and of the input raster. Topology: `width`, `height`, `maximumRadius`,
 * `noDataValue`, whether `validity` is present, and which outputs are present.
 */
export type GPUFocalStatisticsProps = {
  /** Prefix for generated node IDs. Defaults to `'focal-statistics'`. */
  id?: string;
  /** Packed row-major float32 raster, at least `width * height` rows. NaN cells are nodata. */
  values: GraphDataView<'float32'>;
  /** Optional row-major `uint32` validity; zero marks a nodata cell. */
  validity?: GraphDataView<'uint32'>;
  /** Optional finite nodata sentinel compared exactly against cell values. */
  noDataValue?: number;
  /** Raster width in cells. */
  width: number;
  /** Raster height in cells. */
  height: number;
  /**
   * Compile-time cap on the per-frame radius, an integer in `[0, 64]`. The per-frame radius is
   * clamped to it, which bounds the worst-case cost at `(2 * maximumRadius + 1)^2` reads per cell.
   */
  maximumRadius: number;
  /** Per-frame float32 view of at least 4 values written with `getGPUFocalStatisticsParameterValues`. */
  parameters: GraphDataView<'float32'>;
  /** Caller-owned outputs. */
  output: GPUFocalStatisticsOutput;
};

/**
 * Computes moving-window statistics over a float32 raster: mean, sum, min, max, range, population
 * standard deviation, and the count of valid cells.
 *
 * Direct gather: one invocation per output cell visits the window in fixed row-major order, so
 * every result is deterministic and no atomics are used. Window cells outside the raster and
 * nodata cells (NaN, the `noDataValue` sentinel, or zero `validity`) are skipped, so edge windows
 * are clipped like ArcGIS `FocalStatistics` and GRASS `r.neighbors`. A cell is nodata when fewer
 * than `max(minimumCount, 1)` window cells are valid, or when its own value is nodata and
 * `propagateCenterNoData` is set. Standard deviation subtracts the window mean in a second pass
 * over the window instead of using `E[x^2] - E[x]^2`.
 *
 * Moments (sum, mean, standard deviation, count) and extremes (min, max, range) are separate
 * kernels; each is scheduled only when one of its outputs is requested.
 */
export class GPUFocalStatistics implements GPUMapGraphRecipe {
  /** Prefix for every node ID. */
  readonly id: string;
  /** Stable recipe name. */
  readonly recipe = 'focal-statistics';
  /** Validated properties. */
  readonly props: GPUFocalStatisticsProps;
  /** `width * height`. */
  readonly cellCount: number;

  constructor(props: GPUFocalStatisticsProps) {
    this.id = props.id ?? this.recipe;
    this.props = props;
    const {id} = this;
    for (const [name, size] of [
      ['width', props.width],
      ['height', props.height]
    ] as const) {
      if (!Number.isSafeInteger(size) || size < 1) {
        throw new Error(`${id} ${name} must be a positive integer`);
      }
    }
    this.cellCount = props.width * props.height;
    if (this.cellCount > 0xffffffff) {
      throw new Error(`${id} cell count must fit in uint32`);
    }
    if (
      !Number.isSafeInteger(props.maximumRadius) ||
      props.maximumRadius < 0 ||
      props.maximumRadius > MAXIMUM_FOCAL_RADIUS
    ) {
      throw new Error(`${id} maximumRadius must be an integer in [0, ${MAXIMUM_FOCAL_RADIUS}]`);
    }
    if (props.noDataValue !== undefined && !Number.isFinite(props.noDataValue)) {
      throw new Error(`${id} noDataValue must be finite (NaN cells are always nodata)`);
    }
    const {output} = props;
    const outputs = getOutputEntries(output);
    if (outputs.length === 0) {
      throw new Error(`${id} needs at least one output`);
    }
    for (const [name, view] of [
      ['values', props.values],
      ['validity', props.validity],
      ['parameters', props.parameters],
      ...outputs
    ] as const) {
      if ((view as unknown) instanceof GraphVectorView) {
        throw new Error(`${id} ${name} must be a single packed view, not a chunked vector`);
      }
    }
    validatePackedView(props.values, ['float32'], `${id} values`);
    if (props.values.length < this.cellCount) {
      throw new Error(`${id} values must hold width * height rows`);
    }
    if (props.validity) {
      validatePackedUint32View(props.validity, `${id} validity`);
      if (props.validity.length < this.cellCount) {
        throw new Error(`${id} validity must hold width * height rows`);
      }
    }
    validatePackedView(props.parameters, ['float32'], `${id} parameters`);
    if (props.parameters.length < GPU_FOCAL_STATISTICS_PARAMETER_LENGTH) {
      throw new Error(
        `${id} parameters must hold ${GPU_FOCAL_STATISTICS_PARAMETER_LENGTH} float32 values`
      );
    }
    for (const [name, view] of outputs) {
      if (name === 'count') {
        validatePackedUint32View(view, `${id} output.count`);
      } else {
        validatePackedView(view, ['float32'], `${id} output.${name}`);
      }
      if (view.length < this.cellCount) {
        throw new Error(`${id} output.${name} must hold width * height rows`);
      }
    }
    const outputBuffers = outputs.map(([, view]) => view.buffer);
    if (new Set(outputBuffers).size !== outputBuffers.length) {
      throw new Error(`${id} outputs must not share buffers`);
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      outputs.map(([, view]) => view),
      [props.values, props.validity, props.parameters]
    );
  }

  /** Returns the moments kernel and the extremes kernel, each only when one of its outputs is set. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, cellCount} = this;
    const {output} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      props.values,
      props.validity,
      props.parameters,
      ...getOutputEntries(output).map(([, view]) => view)
    ]);
    const inputs: MapGraphKernelBinding[] = [
      {name: 'raster', view: props.values, type: 'f32', access: 'read'},
      ...(props.validity
        ? [
            {
              name: 'validity',
              view: props.validity,
              type: 'u32',
              access: 'read'
            } as const
          ]
        : []),
      {name: 'params', view: props.parameters, type: 'f32', access: 'read'}
    ];
    const declarations = `const WIDTH: u32 = ${props.width}u;
const HEIGHT: u32 = ${props.height}u;
const MAXIMUM_RADIUS: i32 = ${props.maximumRadius};
${getValidityWGSL(Boolean(props.validity), props.noDataValue)}`;
    const nodes: GPUCommandNode<Parameters>[] = [];
    const write = (
      name: string,
      view: GraphDataView | undefined,
      type: 'f32' | 'u32' = 'f32'
    ): MapGraphKernelBinding[] => (view ? [{name, view, type, access: 'read_write'}] : []);

    if (output.sum || output.mean || output.standardDeviation || output.count) {
      nodes.push(
        createMapGraphKernelNode<Parameters>(graph, {
          id: `${id}-moments`,
          operation: OPERATION,
          variant: 'moments',
          bindings: [
            ...inputs,
            ...write('sumOut', output.sum),
            ...write('meanOut', output.mean),
            ...write('deviationOut', output.standardDeviation),
            ...write('countOut', output.count, 'u32')
          ],
          invocationCount: cellCount,
          workgroupSize: 64,
          declarations,
          body: `${WINDOW_SETUP_WGSL}
  var count = 0u;
  var sum = 0.0;
  ${forEachWindowCell('count += 1u;\n      sum += value;')}
  let isDefined = count >= minimumCount && (!propagateCenterNoData || centerValid);
  let nan = getNaN();
  let mean = sum / f32(max(count, 1u));
  ${output.sum ? 'sumOut[sumOutOffset + index] = select(nan, sum, isDefined);' : ''}
  ${output.mean ? 'meanOut[meanOutOffset + index] = select(nan, mean, isDefined);' : ''}
  ${output.count ? 'countOut[countOutOffset + index] = count;' : ''}
  ${
    output.standardDeviation
      ? `var squaredDeviationSum = 0.0;
  if (isDefined) {
    ${forEachWindowCell('let deviation = value - mean;\n      squaredDeviationSum += deviation * deviation;')}
  }
  deviationOut[deviationOutOffset + index] = select(nan, sqrt(squaredDeviationSum / f32(max(count, 1u))), isDefined);`
      : ''
  }`
        })
      );
    }

    if (output.min || output.max || output.range) {
      nodes.push(
        createMapGraphKernelNode<Parameters>(graph, {
          id: `${id}-extremes`,
          operation: OPERATION,
          variant: 'extremes',
          bindings: [
            ...inputs,
            ...write('minOut', output.min),
            ...write('maxOut', output.max),
            ...write('rangeOut', output.range)
          ],
          invocationCount: cellCount,
          workgroupSize: 64,
          declarations,
          body: `${WINDOW_SETUP_WGSL}
  var count = 0u;
  var minimum = 0.0;
  var maximum = 0.0;
  ${forEachWindowCell(`if (count == 0u) {
        minimum = value;
        maximum = value;
      } else {
        minimum = min(minimum, value);
        maximum = max(maximum, value);
      }
      count += 1u;`)}
  let isDefined = count >= minimumCount && (!propagateCenterNoData || centerValid);
  let nan = getNaN();
  ${output.min ? 'minOut[minOutOffset + index] = select(nan, minimum, isDefined);' : ''}
  ${output.max ? 'maxOut[maxOutOffset + index] = select(nan, maximum, isDefined);' : ''}
  ${output.range ? 'rangeOut[rangeOutOffset + index] = select(nan, maximum - minimum, isDefined);' : ''}`
        })
      );
    }
    return nodes;
  }
}

function getOutputEntries(
  output: GPUFocalStatisticsOutput
): (readonly [keyof GPUFocalStatisticsOutput, GraphDataView])[] {
  return (['mean', 'sum', 'min', 'max', 'range', 'standardDeviation', 'count'] as const).flatMap(
    name => {
      const view = output[name];
      return view ? [[name, view as GraphDataView] as const] : [];
    }
  );
}

function getValidityWGSL(hasValidity: boolean, noDataValue: number | undefined): string {
  return /* wgsl */ `
// WGSL rejects NaN constants, so build one from a runtime bit pattern.
fn getNaN() -> f32 {
  var bits = 0x7fc00000u;
  return bitcast<f32>(bits);
}
fn isFiniteValue(value: f32) -> bool { return (bitcast<u32>(value) & 0x7fffffffu) < 0x7f800000u; }
fn readCell(cell: u32, value: ptr<function, f32>) -> bool {
  let sample = raster[rasterOffset + cell];
  *value = sample;
  // Compare bits: NaN is nodata, and infinities are kept as values.
  var isValid = (bitcast<u32>(sample) & 0x7fffffffu) <= 0x7f800000u;
  ${noDataValue !== undefined ? `isValid = isValid && sample != ${getWGSLFloatLiteral(noDataValue)};` : ''}
  ${hasValidity ? 'isValid = isValid && validity[validityOffset + cell] != 0u;' : ''}
  return isValid;
}`;
}

/** Reads per-frame parameters and the center cell; defines the clamped window radius. */
const WINDOW_SETUP_WGSL = /* wgsl */ `let column = i32(index % WIDTH);
  let row = i32(index / WIDTH);
  let radiusValue = params[paramsOffset];
  let isCircle = params[paramsOffset + 1u] == 1.0;
  let minimumCountValue = params[paramsOffset + 2u];
  let propagateCenterNoData = params[paramsOffset + 3u] != 0.0;
  let minimumCount = max(u32(clamp(select(1.0, minimumCountValue, isFiniteValue(minimumCountValue)), 0.0, 4294967040.0)), 1u);
  // NaN or negative radii give an empty window.
  let hasWindow = radiusValue >= 0.0;
  let windowRadius = select(-1, i32(min(floor(radiusValue), f32(MAXIMUM_RADIUS))), hasWindow);
  let radiusSquared = radiusValue * radiusValue;
  var centerValue = 0.0;
  let centerValid = readCell(index, &centerValue);`;

/** Visits valid window cells in row-major order, binding `value` for `statement`. */
function forEachWindowCell(statement: string): string {
  return `for (var dy = -windowRadius; dy <= windowRadius; dy++) {
    let sampleRow = row + dy;
    if (sampleRow < 0 || sampleRow >= i32(HEIGHT)) {
      continue;
    }
    for (var dx = -windowRadius; dx <= windowRadius; dx++) {
      let sampleColumn = column + dx;
      if (sampleColumn < 0 || sampleColumn >= i32(WIDTH)) {
        continue;
      }
      if (isCircle && f32(dx * dx + dy * dy) > radiusSquared) {
        continue;
      }
      var value = 0.0;
      if (!readCell(u32(sampleRow) * WIDTH + u32(sampleColumn), &value)) {
        continue;
      }
      ${statement}
    }
  }`;
}
