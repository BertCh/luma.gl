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
import {
  GPUClassAssignment,
  GPUClassificationFit,
  GPUParameterBuffer
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Inputs the lens scenario uploads from `s2-midwest-counties.json`. */
export type ClassificationFitExampleInput = {
  /** 2023 unemployment rate (%) of the 437 Midwest counties. */
  unemployment: GraphDataView<'float32'>;
};

/** Goodness of fit of a five-class quantile classification of county unemployment. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {unemployment}: ClassificationFitExampleInput
) {
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const breaks = createTransientView(graph, 'breaks', 'float32', 6, usage);
  const classCount = createTransientView(graph, 'classCount', 'uint32', 1, usage);
  const classes = createTransientView(graph, 'classes', 'uint32', unemployment.length, usage);
  const counts = createTransientView(graph, 'counts', 'uint32', 6, usage);
  const medians = createTransientView(graph, 'medians', 'float32', 6, usage);
  const absoluteDeviations = createTransientView(graph, 'absoluteDeviations', 'float32', 6, usage);
  const squaredDeviations = createTransientView(graph, 'squaredDeviations', 'float32', 6, usage);
  const summary = createTransientView(graph, 'summary', 'float32', 5, usage);
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
  // #region example
  graph.add(
    new GPUClassificationFit({
      values: unemployment,
      classes,
      classCount: 5,
      output: {counts, medians, absoluteDeviations, squaredDeviations, summary}
    })
  );
  // summary: [GADF, ADCM, ADAM, GVF, TSS]; counts and medians: one row per class, then all rows
  // #endregion example
  return {
    breaks,
    classes,
    counts,
    medians,
    absoluteDeviations,
    squaredDeviations,
    summary,
    breakParameters
  };
}
