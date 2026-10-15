// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert, Buffer, type Device} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {
  GPUEmpiricalBayesRates,
  GPU_EMPIRICAL_BAYES_SUMMARY
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Event counts and populations per tract; the caller owns the recording graph. */
export type EmpiricalBayesRatesInput = {
  graph: GPUCommandGraph;
  events: GraphDataView<'float32'>;
  populations: GraphDataView<'float32'>;
};

/** Runs the exact public-API example shown below the empirical-Bayes result figure. */
export async function runExample(device: Device, input: EmpiricalBayesRatesInput) {
  const {graph, events, populations} = input;
  // Every view must belong to the supplied device's graph.
  assert(graph.device === device);
  const usage = Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST;
  const rawRates = createTransientView(graph, 'rawRates', 'float32', events.length, usage);
  const smoothedRates = createTransientView(
    graph,
    'smoothedRates',
    'float32',
    events.length,
    usage
  );
  const summary = createTransientView(
    graph,
    'summary',
    'float32',
    GPU_EMPIRICAL_BAYES_SUMMARY.length,
    usage
  );
  // #region example
  // Shrinks each crude rate toward the pooled rate; small populations move the most.
  graph.add(
    new GPUEmpiricalBayesRates({
      events,
      populations,
      clampPriorVariance: true,
      output: {rawRates, smoothedRates, summary}
    })
  );
  // #endregion example
  return {
    graph,
    encodeOptions: {parameters: undefined},
    views: {events, populations, rawRates, smoothedRates, summary}
  };
}
