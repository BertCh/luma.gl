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
  GPUDotDensity,
  GPUParameterBuffer,
  getGPUDotDensityParameterValues,
  type GPUPolygonGeometryFields,
  type GPUSpatialContext
} from '@luma.gl/experimental/gpu-spatial-analysis';

export type DotDensityExampleInput = {
  geometry: GPUPolygonGeometryFields;
  population: GraphDataView<'float32'>;
  spatialContext: GPUSpatialContext;
  capacity: number;
};

export async function runExample(graph: GPUCommandGraph<undefined>, input: DotDensityExampleInput) {
  const {geometry: tracts, population, spatialContext, capacity} = input;
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const points = createTransientView(graph, 'points', 'float32x2', capacity, usage);
  const sourceIds = createTransientView(graph, 'source-ids', 'uint32', capacity, usage);
  const count = createTransientView(graph, 'count', 'uint32', 1, usage);
  const overflow = createTransientView(graph, 'overflow', 'uint32', 1, usage);
  const requiredCount = createTransientView(graph, 'required-count', 'uint32', 1, usage);
  const slotCounts = createTransientView(
    graph,
    'slot-counts',
    'uint32',
    tracts.featureOffsets.length - 1,
    usage
  );
  const failedCount = createTransientView(graph, 'failed-count', 'uint32', 1, usage);
  const parameters = new GPUParameterBuffer(graph.device, {
    id: 'dot-density-parameters',
    format: 'uint32',
    length: 8,
    values: getGPUDotDensityParameterValues({seed: 2026, dotsPerUnit: 1 / 500})
  });
  // #region example
  graph.add(
    new GPUDotDensity({
      ...tracts,
      spatialContext,
      values: population,
      parameters: parameters.importToGraph(graph),
      output: {
        positions: points,
        dots: {ids: sourceIds, status: {count, overflow, requiredCount}},
        slotCounts,
        status: {failedCount}
      }
    })
  );
  // points and sourceIds: seeded population dots, one dot per 500 people in expectation
  // #endregion example
  return {points, sourceIds, count, overflow, requiredCount, slotCounts, failedCount, parameters};
}
