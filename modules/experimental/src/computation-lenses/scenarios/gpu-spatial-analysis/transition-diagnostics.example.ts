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
  GPUTransitionDiagnostics,
  GPUTransitionMatrix
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Inputs the lens scenario uploads from `s2-midwest-counties.json`. */
export type TransitionDiagnosticsExampleInput = {
  /** Unemployment rate (%) 2008-2023, period-major: county `i` of year `t` at `t * 437 + i`. */
  unemployment: GraphDataView<'float32'>;
  /** State of each county in the same layout (0 IL, 1 IN, 2 MI, 3 OH, 4 WI). */
  states: GraphDataView<'uint32'>;
  /** Number of yearly periods (16). */
  periodCount: number;
};

/** Do the five states share one transition matrix of county unemployment classes? */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {unemployment, states, periodCount}: TransitionDiagnosticsExampleInput
) {
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const breaks = createTransientView(graph, 'breaks', 'float32', 6, usage);
  const classCount = createTransientView(graph, 'classCount', 'uint32', 1, usage);
  const classes = createTransientView(graph, 'classes', 'uint32', unemployment.length, usage);
  const counts = createTransientView(graph, 'counts', 'uint32', 125, usage);
  const homogeneity = createTransientView(graph, 'homogeneity', 'float32', 5, usage);
  const steadyStates = createTransientView(graph, 'steadyStates', 'float32', 30, usage);
  const stateSparsity = createTransientView(graph, 'stateSparsity', 'uint32', 2, usage);
  const converged = createTransientView(graph, 'converged', 'uint32', 1, usage);
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
  // One transition count matrix per state: counts is 5 states x 5 classes x 5 classes.
  graph.add(
    new GPUTransitionMatrix({
      classes,
      periodCount,
      classCount: 5,
      conditionClasses: states,
      conditionCount: 5,
      output: {counts}
    })
  );
  // #region example
  graph.add(
    new GPUTransitionDiagnostics({
      counts,
      classCount: 5,
      conditionCount: 5,
      output: {homogeneity, steadyStates, stateSparsity, status: {converged}}
    })
  );
  // homogeneity: [Q, Q p-value, LR, LR p-value, degrees of freedom]; steadyStates: pooled chain, then one per state
  // #endregion example
  return {
    breaks,
    classes,
    counts,
    homogeneity,
    steadyStates,
    stateSparsity,
    converged,
    breakParameters
  };
}
