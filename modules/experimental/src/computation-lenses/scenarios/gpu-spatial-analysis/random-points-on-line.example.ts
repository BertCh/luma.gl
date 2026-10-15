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
  GPURandomPointsOnLine,
  GPUParameterBuffer,
  getGPUDotDensityParameterValues,
  type GPULineGeometryFields,
  type GPUSpatialContext
} from '@luma.gl/experimental/gpu-spatial-analysis';

export type RandomPointsOnLineExampleInput = {
  geometry: GPULineGeometryFields;
  counts: GraphDataView<'uint32'>;
  spatialContext: GPUSpatialContext;
  capacity: number;
};

export async function runExample(
  graph: GPUCommandGraph<undefined>,
  input: RandomPointsOnLineExampleInput
) {
  const {geometry: roads, counts, spatialContext, capacity} = input;
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const points = createTransientView(graph, 'points', 'float32x2', capacity, usage);
  const sourceIds = createTransientView(graph, 'source-ids', 'uint32', capacity, usage);
  const count = createTransientView(graph, 'count', 'uint32', 1, usage);
  const overflow = createTransientView(graph, 'overflow', 'uint32', 1, usage);
  const requiredCount = createTransientView(graph, 'required-count', 'uint32', 1, usage);
  const fractions = createTransientView(graph, 'fractions', 'float32', capacity, usage);
  const parameters = new GPUParameterBuffer(graph.device, {
    id: 'random-points-on-line-parameters',
    format: 'uint32',
    length: 8,
    values: getGPUDotDensityParameterValues({seed: 2026})
  });
  // #region example
  graph.add(
    new GPURandomPointsOnLine({
      ...roads,
      spatialContext,
      counts,
      parameters: parameters.importToGraph(graph),
      output: {
        positions: points,
        points: {ids: sourceIds, status: {count, overflow, requiredCount}},
        fractions
      }
    })
  );
  // points and sourceIds: seeded samples on each input road edge
  // #endregion example
  return {points, sourceIds, count, overflow, requiredCount, fractions, parameters};
}
