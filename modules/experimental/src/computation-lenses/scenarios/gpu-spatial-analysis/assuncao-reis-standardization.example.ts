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
  GPUAssuncaoReisStandardization,
  GPU_RATE_STANDARDIZATION_STATISTICS_LENGTH
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Event counts and populations per tract; the caller owns the recording graph. */
export type AssuncaoReisStandardizationInput = {
  graph: GPUCommandGraph;
  events: GraphDataView<'float32'>;
  populations: GraphDataView<'float32'>;
};

/** Runs the exact public-API example shown below the Assuncao-Reis result figure. */
export async function runExample(device: Device, input: AssuncaoReisStandardizationInput) {
  const {graph, events, populations} = input;
  // Every view must belong to the supplied device's graph.
  assert(graph.device === device);
  const usage = Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST;
  const standardizedRates = createTransientView(
    graph,
    'standardizedRates',
    'float32',
    events.length,
    usage
  );
  const statistics = createTransientView(
    graph,
    'statistics',
    'float32',
    GPU_RATE_STANDARDIZATION_STATISTICS_LENGTH,
    usage
  );
  const invalidCount = createTransientView(graph, 'invalidCount', 'uint32', 1, usage);
  // #region example
  // z = (rate - b) / sqrt(v): the pooled rate b and the prior variance a come from the whole table.
  graph.add(
    new GPUAssuncaoReisStandardization({
      events,
      populations,
      output: {standardizedRates, statistics, status: {invalidCount}}
    })
  );
  // #endregion example
  return {
    graph,
    encodeOptions: {parameters: undefined},
    views: {events, populations, standardizedRates, statistics, invalidCount}
  };
}
