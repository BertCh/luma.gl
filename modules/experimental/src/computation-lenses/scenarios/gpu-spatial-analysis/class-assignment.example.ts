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
  GPUClassBreaks
} from '@luma.gl/experimental/gpu-dataframe';
import {GPUClassAssignment, GPUParameterBuffer} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Inputs the lens scenario uploads from `s2-midwest-counties.json`. */
export type ClassAssignmentExampleInput = {
  /** 2023 unemployment rate (%) of the 437 Midwest counties. */
  unemployment: GraphDataView<'float32'>;
};

/** Quintile classes of the 2023 county unemployment rate. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {unemployment}: ClassAssignmentExampleInput
) {
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const breaks = createTransientView(graph, 'breaks', 'float32', 6, usage);
  const classCount = createTransientView(graph, 'classCount', 'uint32', 1, usage);
  const classes = createTransientView(graph, 'classes', 'uint32', unemployment.length, usage);
  // #region example
  const breakParameters = new GPUParameterBuffer(graph.device, {
    id: 'break-parameters',
    format: 'float32',
    length: getGPUClassBreaksParameterLength(5),
    values: getGPUClassBreaksParameterValues({method: 'quantile', classCount: 5}, 5)
  });
  graph.add(
    new GPUClassBreaks({
      values: unemployment,
      parameters: breakParameters.importToGraph(graph),
      maximumClassCount: 5,
      methods: ['quantile'],
      output: {breaks, classCount}
    })
  );
  graph.add(new GPUClassAssignment({values: unemployment, breaks, classCount, output: classes}));
  // classes: 0 (lowest fifth of counties) to 4 (highest fifth), one per county
  // #endregion example
  return {breaks, classCount, classes, breakParameters};
}
