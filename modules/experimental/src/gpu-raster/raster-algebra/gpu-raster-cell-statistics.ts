// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {GPU_RASTER_CELL_STATISTICS_PARAMETER_LENGTH} from './local-operations-parameters';
import {
  getRasterAlgebraValueWGSL,
  validateRasterAlgebraAliasing,
  validateRasterAlgebraCount,
  validateRasterAlgebraGraph,
  validateRasterAlgebraNoData,
  validateRasterAlgebraView
} from './raster-algebra-utils';

const OPERATION = 'GPURasterCellStatistics';
/** Largest stack depth: the frequency kernel is `O(layerCount^2)` per cell. */
const MAXIMUM_LAYER_COUNT = 64;

/**
 * Caller-owned outputs of {@link GPURasterCellStatistics}, each at least `cellCount` rows. Provide
 * any non-empty subset. Float outputs are NaN where the cell is nodata.
 */
export type GPURasterCellStatisticsOutput = {
  /** Smallest valid layer value. */
  minimum?: GraphDataView<'float32'>;
  /** Largest valid layer value. */
  maximum?: GraphDataView<'float32'>;
  /** `maximum - minimum`. */
  range?: GraphDataView<'float32'>;
  /** Sum of valid layer values in layer order. */
  sum?: GraphDataView<'float32'>;
  /** `sum / count`. */
  mean?: GraphDataView<'float32'>;
  /** Population standard deviation from a centred second pass. */
  standardDeviation?: GraphDataView<'float32'>;
  /** Most frequent valid value (exact f32 equality); ties pick the smallest value. */
  majority?: GraphDataView<'float32'>;
  /** Least frequent valid value; ties pick the smallest value. */
  minority?: GraphDataView<'float32'>;
  /** Number of distinct valid values (0 where the cell is nodata). */
  variety?: GraphDataView<'uint32'>;
  /** Number of valid layers, written for every cell. */
  count?: GraphDataView<'uint32'>;
};

/**
 * Properties for {@link GPURasterCellStatistics}.
 *
 * Per-frame (no recompile): stack contents and `parameters` (nodata policy, minimum valid count).
 * Topology: `layerCount`, `cellCount`, `noDataValue`, and which outputs exist.
 */
export type GPURasterCellStatisticsProps = {
  /** Prefix for generated node IDs. Defaults to `'raster-cell-statistics'`. */
  id?: string;
  /** Band-sequential float32 stack: layer `i` occupies rows `[i * cellCount, (i + 1) * cellCount)`. */
  stack: GraphDataView<'float32'>;
  /** Layer count in `[1, 64]`. */
  layerCount: number;
  /** Cells per layer. */
  cellCount: number;
  /** Optional finite nodata sentinel compared exactly. NaN is always nodata. */
  noDataValue?: number;
  /** Per-frame float32 view written with `getGPURasterCellStatisticsParameterValues`. */
  parameters: GraphDataView<'float32'>;
  /** Caller-owned outputs. */
  output: GPURasterCellStatisticsOutput;
};

/**
 * Local (per-cell) statistics across a stack of rasters, like ArcGIS "Cell Statistics" or GDAL
 * `gdal_calc` reductions: minimum, maximum, range, sum, mean, standard deviation, majority,
 * minority, variety, and valid count.
 *
 * Each cell is one invocation that reads its layers in fixed order, so every output is
 * deterministic and no atomics are used. Moments, extremes, and frequencies are separate kernels,
 * each scheduled only when one of its outputs is requested. Frequencies compare every pair of
 * layers (`O(layerCount^2)` per cell, at most 64 layers) with exact f32 equality, so `-0` and `+0`
 * count as one value.
 */
export class GPURasterCellStatistics implements GPUCommandNodeProducer {
  /** Prefix for every node ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPURasterCellStatisticsProps;

  constructor(props: GPURasterCellStatisticsProps) {
    this.id = props.id ?? 'raster-cell-statistics';
    this.props = props;
    const {id} = this;
    const {layerCount, cellCount, output} = props;
    validateRasterAlgebraCount(id, 'layerCount', layerCount, 1, MAXIMUM_LAYER_COUNT);
    validateRasterAlgebraCount(id, 'cellCount', cellCount, 0, 0xffffffff);
    if (layerCount * cellCount > 0xffffffff) {
      throw new Error(`${id} layerCount * cellCount must fit in uint32`);
    }
    validateRasterAlgebraNoData(id, props.noDataValue);
    validateRasterAlgebraView(id, 'stack', props.stack, 'float32', layerCount * cellCount);
    validateRasterAlgebraView(
      id,
      'parameters',
      props.parameters,
      'float32',
      GPU_RASTER_CELL_STATISTICS_PARAMETER_LENGTH
    );
    const outputs = getOutputEntries(output);
    if (outputs.length === 0) {
      throw new Error(`${id} needs at least one output`);
    }
    for (const [name, view] of outputs) {
      validateRasterAlgebraView(
        id,
        `output.${name}`,
        view,
        name === 'count' || name === 'variety' ? 'uint32' : 'float32',
        cellCount
      );
    }
    validateRasterAlgebraAliasing(
      id,
      outputs.map(([, view]) => view),
      [props.stack, props.parameters]
    );
  }

  /** Returns up to three kernels: moments, extremes, and frequencies. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {output, layerCount, cellCount} = props;
    validateRasterAlgebraGraph(id, graph, [
      props.stack,
      props.parameters,
      ...getOutputEntries(output).map(([, view]) => view)
    ]);
    const inputs: WGSLKernelBinding[] = [
      {name: 'stack', view: props.stack, type: 'f32', access: 'read'},
      {name: 'params', view: props.parameters, type: 'f32', access: 'read'}
    ];
    const write = (
      name: string,
      view: GraphDataView | undefined,
      type: 'f32' | 'u32' = 'f32'
    ): WGSLKernelBinding[] => (view ? [{name, view, type, access: 'read_write'}] : []);
    const declarations = `const LAYER_COUNT: u32 = ${layerCount}u;
const CELL_COUNT: u32 = ${cellCount}u;
${getRasterAlgebraValueWGSL(props.noDataValue)}
fn readLayer(layer: u32, cell: u32, value: ptr<function, f32>) -> bool {
  let sample = stack[stackOffset + layer * CELL_COUNT + cell];
  *value = sample;
  return !isNoDataValue(sample);
}`;
    // Valid-layer counting rides along with each kernel's own layer sweep, so every kernel reads
    // the stack once (moments twice with a standard deviation) instead of one extra counting sweep.
    const setupPrefix = `let propagateNoData = params[paramsOffset] != 0.0;
  let minimumCountValue = params[paramsOffset + 1u];
  let minimumCount = max(u32(clamp(select(1.0, minimumCountValue, isFiniteValue(minimumCountValue)), 0.0, 4294967040.0)), 1u);
  var count = 0u;`;
    const setupSuffix = `let isDefined = count >= minimumCount && (!propagateNoData || count == LAYER_COUNT);
  let nan = getNaN();`;
    const forEachValid = (
      statement: string
    ) => `for (var layer = 0u; layer < LAYER_COUNT; layer++) {
    var value = 0.0;
    if (!readLayer(layer, index, &value)) {
      continue;
    }
    ${statement}
  }`;
    const nodes: GPUCommandNode<Parameters>[] = [];
    if (output.sum || output.mean || output.standardDeviation || output.count) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
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
          declarations,
          body: `${setupPrefix}
  var sum = 0.0;
  ${forEachValid('sum = sum + value;\n    count += 1u;')}
  ${setupSuffix}
  let mean = sum / f32(max(count, 1u));
  ${output.sum ? 'sumOut[sumOutOffset + index] = select(nan, sum, isDefined);' : ''}
  ${output.mean ? 'meanOut[meanOutOffset + index] = select(nan, mean, isDefined);' : ''}
  ${output.count ? 'countOut[countOutOffset + index] = count;' : ''}
  ${
    output.standardDeviation
      ? `var squaredDeviationSum = 0.0;
  ${forEachValid('let deviation = value - mean;\n    squaredDeviationSum = squaredDeviationSum + deviation * deviation;')}
  deviationOut[deviationOutOffset + index] = select(nan, sqrt(squaredDeviationSum / f32(max(count, 1u))), isDefined);`
      : ''
  }`
        })
      );
    }
    if (output.minimum || output.maximum || output.range) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-extremes`,
          operation: OPERATION,
          variant: 'extremes',
          bindings: [
            ...inputs,
            ...write('minimumOut', output.minimum),
            ...write('maximumOut', output.maximum),
            ...write('rangeOut', output.range)
          ],
          invocationCount: cellCount,
          declarations,
          body: `${setupPrefix}
  var minimum = 0.0;
  var maximum = 0.0;
  ${forEachValid(`if (count == 0u) {
      minimum = value;
      maximum = value;
    } else {
      minimum = min(minimum, value);
      maximum = max(maximum, value);
    }
    count += 1u;`)}
  ${setupSuffix}
  ${output.minimum ? 'minimumOut[minimumOutOffset + index] = select(nan, minimum, isDefined);' : ''}
  ${output.maximum ? 'maximumOut[maximumOutOffset + index] = select(nan, maximum, isDefined);' : ''}
  ${output.range ? 'rangeOut[rangeOutOffset + index] = select(nan, maximum - minimum, isDefined);' : ''}`
        })
      );
    }
    if (output.majority || output.minority || output.variety) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-frequencies`,
          operation: OPERATION,
          variant: 'frequencies',
          bindings: [
            ...inputs,
            ...write('majorityOut', output.majority),
            ...write('minorityOut', output.minority),
            ...write('varietyOut', output.variety, 'u32')
          ],
          invocationCount: cellCount,
          workgroupSize: 64,
          declarations,
          // Valid values are gathered once into a private array in layer order, so the pairwise
          // frequency comparison reads registers/private memory instead of re-reading the stack
          // LAYER_COUNT times per layer. Order is preserved, so first-occurrence rules are unchanged.
          body: `${setupPrefix}
  var gathered: array<f32, ${layerCount}>;
  ${forEachValid('gathered[count] = value;\n    count += 1u;')}
  ${setupSuffix}
  var majority = 0.0;
  var majorityCount = 0u;
  var minority = 0.0;
  var minorityCount = 0xffffffffu;
  var variety = 0u;
  for (var position = 0u; position < count; position++) {
    let value = gathered[position];
    var frequency = 0u;
    var isFirst = true;
    for (var other = 0u; other < count; other++) {
      if (gathered[other] == value) {
        frequency += 1u;
        if (other < position) {
          isFirst = false;
        }
      }
    }
    if (!isFirst) {
      continue;
    }
    variety += 1u;
    if (frequency > majorityCount || (frequency == majorityCount && value < majority)) {
      majority = value;
      majorityCount = frequency;
    }
    if (frequency < minorityCount || (frequency == minorityCount && value < minority)) {
      minority = value;
      minorityCount = frequency;
    }
  }
  ${output.majority ? 'majorityOut[majorityOutOffset + index] = select(nan, majority, isDefined);' : ''}
  ${output.minority ? 'minorityOut[minorityOutOffset + index] = select(nan, minority, isDefined);' : ''}
  ${output.variety ? 'varietyOut[varietyOutOffset + index] = select(0u, variety, isDefined);' : ''}`
        })
      );
    }
    return nodes;
  }
}

function getOutputEntries(
  output: GPURasterCellStatisticsOutput
): (readonly [keyof GPURasterCellStatisticsOutput, GraphDataView])[] {
  return (
    [
      'minimum',
      'maximum',
      'range',
      'sum',
      'mean',
      'standardDeviation',
      'majority',
      'minority',
      'variety',
      'count'
    ] as const
  ).flatMap(name => {
    const view = output[name];
    return view ? [[name, view as GraphDataView] as const] : [];
  });
}
