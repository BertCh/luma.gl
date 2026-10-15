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
  getGPUSpatialAutocorrelationParameterValues,
  GPULISAMarkov,
  GPUParameterBuffer,
  type GPUSpatialWeights
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Inputs the lens scenario uploads from `s2-midwest-counties.json`. */
export type LISAMarkovExampleInput = {
  /** Unemployment rate (%) 2008-2023, period-major: county `i` of year `t` at `t * 437 + i`. */
  unemployment: GraphDataView<'float32'>;
  /** Row-standardised queen contiguity of the 437 counties. */
  queen: GPUSpatialWeights;
  /** Number of yearly periods (16). */
  periodCount: number;
};

/** Moves between local Moran quadrants of county unemployment from year to year. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {unemployment, queen, periodCount}: LISAMarkovExampleInput
) {
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const counts = createTransientView(graph, 'counts', 'uint32', 25, usage);
  const probabilities = createTransientView(graph, 'probabilities', 'float32', 25, usage);
  const rowTotals = createTransientView(graph, 'rowTotals', 'uint32', 5, usage);
  const ignored = createTransientView(graph, 'ignored', 'uint32', 1, usage);
  const quadrants = createTransientView(graph, 'quadrants', 'uint32', unemployment.length, usage);
  // #region example
  const moranParameters = new GPUParameterBuffer(graph.device, {
    id: 'moran-parameters',
    format: 'float32',
    length: 8,
    values: getGPUSpatialAutocorrelationParameterValues({significanceLevel: 0.05})
  });
  graph.add(
    new GPULISAMarkov({
      values: unemployment,
      weights: queen,
      periodCount,
      parameters: moranParameters.importToGraph(graph),
      gating: 'none',
      output: {counts, probabilities, rowTotals, ignored, quadrants}
    })
  );
  // counts: from * 5 + to over states 0 (unused), 1 HH, 2 LH, 3 LL, 4 HL; quadrants: one state per county and year
  // #endregion example
  return {counts, probabilities, rowTotals, ignored, quadrants, moranParameters};
}
