// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {GPUGeodesicDestination} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Uploaded origin, bearing and distance columns for GPUGeodesicDestination. */
export type GeodesicDestinationExampleInput = {
  origins: GraphDataView<'float32x2'>;
  bearings: GraphDataView<'float32'>;
  distances: GraphDataView<'float32'>;
};

/** Runs GPUGeodesicDestination and returns the views drawn or inspected by its card. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  input: GeodesicDestinationExampleInput
) {
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const {origins, bearings, distances} = input;
  const rowCount = origins.length;
  const output = {
    destinations: createTransientView(graph, 'destinations', 'float32x2', rowCount, usage),
    finalBearings: createTransientView(graph, 'finalBearings', 'float32', rowCount, usage),
    status: {converged: createTransientView(graph, 'converged', 'uint32', rowCount, usage)}
  };
  // #region example
  // Bearings in degrees clockwise from north, distances in WGS84 metres.
  graph.add(
    new GPUGeodesicDestination({
      origins,
      bearings,
      distances,
      spatialContext: {
        coordinateSpace: 'longitude-latitude',
        metric: 'ellipsoidal',
        units: 'meters'
      },
      maximumIterations: 16,
      output
    })
  );
  const {destinations} = output;
  // #endregion example
  return {...output, converged: output.status.converged, destinations};
}
