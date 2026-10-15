// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
// biome-ignore-all format: the shown example is hand-formatted to fit the 25-line card limit

import {Buffer, type Device} from '@luma.gl/core';
import {createTransientView, type GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  getGPUInequalityParameterValues,
  GPUInequality,
  GPUParameterBuffer,
  GPU_INEQUALITY_GLOBAL_SUMMARY_LENGTH,
  GPU_INEQUALITY_PARAMETER_LENGTH
} from '@luma.gl/experimental/gpu-dataframe';

/** Tract columns the lens scenario reads from `f-chicago-tracts.json`; suppressed estimates are NaN. */
export type InequalityExampleInput = {
  graph: GPUCommandGraph<undefined>;
  tracts: {
    /** Per-capita income in dollars. */
    perCapitaIncome: Float32Array;
    /** Residents of each tract, the Lorenz weight. */
    population: Float32Array;
    /** Zero-based zone of each tract: its community area minus one. */
    zoneIds: Uint32Array;
  };
  /** Number of zones, 77 community areas. */
  zoneCount: number;
};

/** Gini coefficient and Lorenz curve of per-capita income within each community area. */
export async function runExample(
  device: Device,
  {graph, tracts, zoneCount}: InequalityExampleInput
) {
  const buffers: Buffer[] = [];
  const importColumn = <Format extends 'float32' | 'uint32'>(
    id: string,
    data: Float32Array | Uint32Array,
    format: Format
  ) => {
    const buffer = device.createBuffer({
      id,
      data,
      usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
    });
    buffers.push(buffer);
    const handle = graph.importBuffer(
      {id, byteLength: buffer.byteLength, usage: buffer.usage},
      buffer
    );
    return graph.createDataView(handle, {format, length: data.length});
  };
  const view = <Format extends 'float32' | 'uint32'>(id: string, format: Format, length: number) =>
    createTransientView(graph, id, format, length, Buffer.STORAGE | Buffer.COPY_SRC);
  // #region example
  const values = importColumn('values', tracts.perCapitaIncome, 'float32');
  const weights = importColumn('weights', tracts.population, 'float32');
  const zoneIds = importColumn('zoneIds', tracts.zoneIds, 'uint32');
  const parameters = new GPUParameterBuffer(device, {
    id: 'inequality-parameters',
    format: 'float32',
    length: GPU_INEQUALITY_PARAMETER_LENGTH,
    values: getGPUInequalityParameterValues() // Atkinson epsilon 1, Palma cuts 10% and 40%
  });
  const lorenzKnotCount = 11;
  const output = {
    gini: view('gini', 'float32', zoneCount),
    counts: view('counts', 'uint32', zoneCount), // included tracts per zone
    mean: view('mean', 'float32', zoneCount), // population-weighted mean income
    lorenzKnots: view('lorenzKnots', 'float32', zoneCount * lorenzKnotCount),
    globalSummary: view('globalSummary', 'float32', GPU_INEQUALITY_GLOBAL_SUMMARY_LENGTH)
  };
  const props = {values, weights, zoneIds, zoneCount, lorenzKnotCount, parameters: parameters.importToGraph(graph), output};
  graph.add(new GPUInequality(props));
  const {gini, lorenzKnots, globalSummary} = output; // lorenzKnots[zone * 11 + k] = L(k / 10)
  // #endregion example
  return {
    values,
    weights,
    zoneIds,
    gini,
    counts: output.counts,
    mean: output.mean,
    lorenzKnots,
    globalSummary,
    lorenzKnotCount,
    parameters,
    buffers
  };
}