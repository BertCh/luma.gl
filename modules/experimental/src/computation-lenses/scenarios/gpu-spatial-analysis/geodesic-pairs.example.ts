// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {GPUGeodesicPairs} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Uploaded O'Hare routes for GPUGeodesicPairs. */
export type GeodesicPairsExampleInput = {
  origins: GraphDataView<'float32x2'>;
  targets: GraphDataView<'float32x2'>;
};

/** Runs GPUGeodesicPairs and returns the views drawn or inspected by its card. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  input: GeodesicPairsExampleInput
) {
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const {origins, targets} = input;
  const pairCount = origins.length;
  const output = {
    distances: createTransientView(graph, 'distances', 'float32', pairCount, usage),
    initialBearings: createTransientView(graph, 'initialBearings', 'float32', pairCount, usage),
    midpoints: createTransientView(graph, 'midpoints', 'float32x2', pairCount, usage),
    status: {converged: createTransientView(graph, 'converged', 'uint32', pairCount, usage)}
  };
  // #region example
  // Longitude/latitude degrees in, WGS84 metres and degrees clockwise from north out.
  graph.add(
    new GPUGeodesicPairs({
      origins,
      targets,
      spatialContext: {
        coordinateSpace: 'longitude-latitude',
        metric: 'ellipsoidal',
        units: 'meters'
      },
      maximumIterations: 16,
      output
    })
  );
  const {midpoints} = output;
  // #endregion example
  return {...output, converged: output.status.converged, midpoints};
}
