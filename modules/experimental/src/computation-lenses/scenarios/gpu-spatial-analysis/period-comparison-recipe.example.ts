// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {
  getGPUClassBreaksParameterLength,
  getGPUClassBreaksParameterValues,
  getGPUColorScaleParameterValues,
  GPU_COLOR_SCALE_PARAMETER_LENGTH,
  packGPUColor
} from '@luma.gl/experimental/gpu-dataframe';
import {
  addPeriodComparisonRecipe,
  GPUParameterBuffer
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Inputs the lens scenario uploads from `s2-chicago-north-nature.json`, split by day of year. */
export type PeriodComparisonRecipeExampleInput = {
  /** Spring observations as longitude/latitude degrees. */
  spring: GraphDataView<'float32x2'>;
  /** Autumn observations as longitude/latitude degrees. */
  autumn: GraphDataView<'float32x2'>;
};

/** Spring-to-autumn change in observation counts per Quadbin cell, in five diverging classes. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {spring, autumn}: PeriodComparisonRecipeExampleInput
) {
  const tableCapacity = 1024;
  const unionCapacity = 2048;
  const classEdges = [-1e6, -2.5, -0.5, 0.5, 2.5, 1e6];
  const palette = [
    packGPUColor(33, 102, 172),
    packGPUColor(146, 197, 222),
    packGPUColor(247, 247, 247),
    packGPUColor(244, 165, 130),
    packGPUColor(178, 24, 43)
  ];
  const parameters = [
    new GPUParameterBuffer(graph.device, {
      id: 'class-breaks-parameters',
      format: 'float32',
      length: getGPUClassBreaksParameterLength(5),
      values: getGPUClassBreaksParameterValues({method: 'custom', customEdges: classEdges}, 5)
    }),
    new GPUParameterBuffer(graph.device, {
      id: 'color-scale-parameters',
      format: 'float32',
      length: GPU_COLOR_SCALE_PARAMETER_LENGTH,
      values: getGPUColorScaleParameterValues({
        scale: 'threshold',
        domainCount: 6,
        paletteCount: 5,
        noDataColor: packGPUColor(0, 0, 0, 0)
      })
    }),
    new GPUParameterBuffer(graph.device, {
      id: 'palette',
      format: 'uint32',
      length: 5,
      values: Uint32Array.from(palette)
    })
  ] as const;
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const output = <Format extends 'float32' | 'uint32' | 'uint32x2'>(
    id: string,
    format: Format,
    length: number
  ) => createTransientView(graph, id, format, length, usage);
  const cells = output('cells', 'uint32x2', unionCapacity);
  const delta = output('delta', 'float32', unionCapacity);
  const zScore = output('zScore', 'float32', unionCapacity);
  const count = output('count', 'uint32', 1);
  const overflow = output('overflow', 'uint32', 1);
  const classIndices = output('classIndices', 'uint32', unionCapacity);
  const colors = output('colors', 'uint32', unionCapacity);
  // #region example
  const comparison = addPeriodComparisonRecipe(graph, {
    family: 'quadbin',
    resolution: 16,
    tableCapacity,
    unionCapacity,
    before: {positions: spring},
    after: {positions: autumn},
    classify: 'delta',
    classBreaksParameters: parameters[0].importToGraph(graph),
    colorScaleParameters: parameters[1].importToGraph(graph),
    palette: parameters[2].importToGraph(graph),
    methods: ['custom'],
    maximumClassCount: 5,
    maximumPaletteCount: 5,
    outputs: {comparison: {cells, delta, zScore, status: {count, overflow}}, classIndices, colors}
  });
  // delta is autumn minus spring per cell; classIndices holds its diverging class
  // #endregion example
  return {comparison, cells, delta, zScore, count, overflow, classIndices, colors, parameters};
}
