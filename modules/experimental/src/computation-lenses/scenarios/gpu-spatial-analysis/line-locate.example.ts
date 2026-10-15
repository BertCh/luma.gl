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
  GPULineLocate,
  type GPULineGeometryFields
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Uploaded Chicago geometry and parameters for GPULineLocate. */
export type LineLocateExampleInput = {
  roads: GPULineGeometryFields;
  eventLineIndices: GraphDataView<'uint32'>;
  eventMeasures: GraphDataView<'float32'>;
  parameters: GraphDataView<'float32'>;
};

/** Runs GPULineLocate and returns the views drawn or inspected by its card. */
export async function runExample(graph: GPUCommandGraph<undefined>, input: LineLocateExampleInput) {
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const {roads, eventLineIndices, eventMeasures, parameters} = input;
  const placeCount = eventMeasures.length;
  // #region example
  const output = {
    positions: createTransientView(graph, 'locatedPlaces', 'float32x2', placeCount, usage),
    statuses: createTransientView(graph, 'statuses', 'uint32', placeCount, usage)
  };
  // Nearest-road indices and metre measures come from GPULinearReferencing.
  graph.add(
    new GPULineLocate({
      ...roads,
      eventLineIndices,
      eventMeasures,
      parameters,
      measureMode: 'distance',
      output
    })
  );
  const {positions: locatedPlaces} = output;
  // #endregion example
  return {...output, locatedPlaces};
}
