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
  getGPUPolygonRasterizationParameterValues,
  GPU_POLYGON_RASTERIZATION_PARAMETER_LENGTH,
  makeGPURasterMetadata
} from '@luma.gl/experimental/gpu-raster';
import {
  addChangeOfSupportRecipe,
  GPUParameterBuffer,
  type GPUChangeOfSupportZones,
  type GPUSpatialContext
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Zone systems the lens scenario uploads from the Chicago North Side tracts and areas. */
export type ChangeOfSupportRecipeExampleInput = {
  /** Census tracts as four-level polygon views in local-frame metres. */
  tracts: Omit<GPUChangeOfSupportZones, 'crossingCapacity'>;
  /** Community areas as four-level polygon views in the same frame. */
  areas: Omit<GPUChangeOfSupportZones, 'crossingCapacity'>;
  /** Observation count of each tract. */
  tractCounts: GraphDataView<'float32'>;
  /** Planar, equal-area context of the local frame. */
  spatialContext: GPUSpatialContext;
  /** Raster extent `[minX, minY, maxX, maxY]` in frame metres, a whole number of cells. */
  extent: readonly [number, number, number, number];
  /** Raster cell size in metres. */
  cellSize: number;
};

/** Observation counts moved from census tracts onto community areas by raster overlap area. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {tracts, areas, tractCounts, spatialContext, extent, cellSize}: ChangeOfSupportRecipeExampleInput
) {
  const [minX, minY, maxX, maxY] = extent;
  const width = Math.round((maxX - minX) / cellSize);
  const height = Math.round((maxY - minY) / cellSize);
  const grid = makeGPURasterMetadata({width, height, cellSize, origin: [minX, maxY]});
  const parameters = new GPUParameterBuffer(graph.device, {
    id: 'rasterization-parameters',
    format: 'float32',
    length: GPU_POLYGON_RASTERIZATION_PARAMETER_LENGTH,
    values: getGPUPolygonRasterizationParameterValues(grid)
  });
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const output = <Format extends 'float32' | 'uint32'>(
    id: string,
    format: Format,
    length: number
  ) => createTransientView(graph, id, format, length, usage);
  const sourceCount = tracts.featureOffsets.length - 1;
  const targetCount = areas.featureOffsets.length - 1;
  const areaCounts = output('areaCounts', 'float32', targetCount);
  const areaMeans = output('areaMeans', 'float32', targetCount);
  const sourceAreas = output('sourceAreas', 'float32', sourceCount);
  const sourceTransferredFractions = output('sourceTransferredFractions', 'float32', sourceCount);
  const overflow = output('overflow', 'uint32', 1);
  const requiredCount = output('requiredCount', 'uint32', 1);
  const invalidCount = output('invalidCount', 'uint32', 1);
  // #region example
  const transfer = addChangeOfSupportRecipe(graph, {
    spatialContext,
    grid,
    parameters: parameters.importToGraph(graph),
    source: {...tracts, crossingCapacity: 65536},
    target: {...areas, crossingCapacity: 65536},
    pairCapacity: 2600,
    sourceValues: tractCounts,
    outputs: {
      extensiveValues: areaCounts,
      intensiveValues: areaMeans,
      sourceAreas,
      sourceTransferredFractions,
      status: {overflow, requiredCount, invalidCount}
    }
  });
  // areaCounts moves tract counts onto areas; sourceTransferredFractions shows mass reaching a target
  // #endregion example
  return {
    transfer,
    areaCounts,
    areaMeans,
    sourceAreas,
    sourceTransferredFractions,
    overflow,
    requiredCount,
    invalidCount,
    parameters
  };
}
