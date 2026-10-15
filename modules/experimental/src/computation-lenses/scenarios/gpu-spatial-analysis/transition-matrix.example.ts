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
  GPUParameterBuffer,
  GPUTransitionMatrix
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Inputs the lens scenario uploads from `s2-midwest-counties.json`. */
export type TransitionMatrixExampleInput = {
  /** Unemployment rate (%) 2008-2023, period-major: county `i` of year `t` at `t * 437 + i`. */
  unemployment: GraphDataView<'float32'>;
  /** Number of yearly periods (16). */
  periodCount: number;
};

/** Class transitions of county unemployment between consecutive years, one legend for all years. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {unemployment, periodCount}: TransitionMatrixExampleInput
) {
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const breaks = createTransientView(graph, 'breaks', 'float32', 6, usage);
  const classCount = createTransientView(graph, 'classCount', 'uint32', 1, usage);
  const classes = createTransientView(graph, 'classes', 'uint32', unemployment.length, usage);
  const counts = createTransientView(graph, 'counts', 'uint32', 25, usage);
  const probabilities = createTransientView(graph, 'probabilities', 'float32', 25, usage);
  const rowTotals = createTransientView(graph, 'rowTotals', 'uint32', 5, usage);
  const ignored = createTransientView(graph, 'ignored', 'uint32', 1, usage);
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
  graph.add(
    new GPUTransitionMatrix({
      classes,
      periodCount,
      classCount: 5,
      output: {counts, probabilities, rowTotals, ignored}
    })
  );
  // counts and probabilities: row from * 5 + to; rowTotals: transitions out of each class
  // #endregion example
  return {breaks, classes, counts, probabilities, rowTotals, ignored, breakParameters};
}
