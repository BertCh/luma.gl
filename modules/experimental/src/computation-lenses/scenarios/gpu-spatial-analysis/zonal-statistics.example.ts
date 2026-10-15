// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert, Buffer, type Device} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {GPUZonalStatistics} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Zone row and count of every tract, and the zone count; the caller owns the recording graph. */
export type ZonalStatisticsInput = {
  graph: GPUCommandGraph;
  /** Zone row of each tract, in `[0, zoneCount)`. */
  pointFeatureRows: GraphDataView<'uint32'>;
  values: GraphDataView<'float32'>;
  zoneCount: number;
};

/** Runs the exact public-API example shown below the zonal statistics result figure. */
export async function runExample(device: Device, input: ZonalStatisticsInput) {
  const {graph, pointFeatureRows, values, zoneCount} = input;
  // Every view must belong to the supplied device's graph.
  assert(graph.device === device);
  const usage = Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST;
  const createFloat = (name: string) =>
    createTransientView(graph, name, 'float32', zoneCount, usage);
  const counts = createTransientView(graph, 'counts', 'uint32', zoneCount, usage);
  const valueCounts = createTransientView(graph, 'valueCounts', 'uint32', zoneCount, usage);
  const sums = createFloat('sums');
  const means = createFloat('means');
  const standardDeviations = createFloat('standardDeviations');
  // #region example
  // Each tract is a row assigned to its community area; values are summed per zone.
  graph.add(
    new GPUZonalStatistics({
      features: {kind: 'feature-rows', pointFeatureRows, featureCount: zoneCount},
      values,
      output: {counts, valueCounts, sums, means, standardDeviations}
    })
  );
  // #endregion example
  return {
    graph,
    encodeOptions: {parameters: undefined},
    views: {pointFeatureRows, values, counts, valueCounts, sums, means, standardDeviations}
  };
}
