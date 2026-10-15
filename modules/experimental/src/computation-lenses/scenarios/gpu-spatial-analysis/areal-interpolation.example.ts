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
  GPUPolygonRasterization,
  makeGPURasterMetadata
} from '@luma.gl/experimental/gpu-raster';
import {
  GPUArealInterpolation,
  GPUParameterBuffer,
  GPUSpatialLag,
  type GPUPolygonGeometryFields,
  type GPUSpatialContext
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Inputs the lens scenario uploads from the tract and community-area data files. */
export type ArealInterpolationExampleInput = {
  /** Source zones: census tracts in an equal-area frame. */
  tracts: GPUPolygonGeometryFields;
  /** Target zones: community areas in the same frame. */
  communityAreas: GPUPolygonGeometryFields;
  /** Residents per tract, an extensive count. */
  population: GraphDataView<'float32'>;
  /** Lambert equal-area context of the local frame. */
  spatialContext: GPUSpatialContext;
  /** Raster size, cell size and top-left corner `[minX, maxY]` in frame metres. */
  raster: {width: number; height: number; cellSize: number; origin: readonly [number, number]};
  /** Upper bound on edge-row crossings of the larger polygon set. */
  crossingCapacity: number;
};

/** Area-weighted transfer of tract population to community areas on a 100 m raster. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {
    tracts,
    communityAreas,
    population,
    spatialContext,
    raster,
    crossingCapacity
  }: ArealInterpolationExampleInput
) {
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const cellCount = raster.width * raster.height;
  const tractCount = tracts.featureOffsets.length - 1;
  const areaCount = communityAreas.featureOffsets.length - 1;
  const pairCapacity = 1024;
  const tractZones = createTransientView(graph, 'tractZones', 'uint32', cellCount, usage);
  const areaZones = createTransientView(graph, 'areaZones', 'uint32', cellCount, usage);
  const tractRasterOverflow = createTransientView(graph, 'tractRasterOverflow', 'uint32', 1, usage);
  const areaRasterOverflow = createTransientView(graph, 'areaRasterOverflow', 'uint32', 1, usage);
  const weights = {
    offsets: createTransientView(graph, 'weightOffsets', 'uint32', areaCount + 1, usage),
    neighbors: createTransientView(graph, 'weightNeighbors', 'uint32', pairCapacity, usage),
    weights: createTransientView(graph, 'weightValues', 'float32', pairCapacity, usage)
  };
  const overlapAreas = createTransientView(graph, 'overlapAreas', 'float32', pairCapacity, usage);
  const pairOverflow = createTransientView(graph, 'pairOverflow', 'uint32', 1, usage);
  const transferredFractions = createTransientView(
    graph,
    'transferredFractions',
    'float32',
    tractCount,
    usage
  );
  const areaPopulation = createTransientView(graph, 'areaPopulation', 'float32', areaCount, usage);
  const grid = makeGPURasterMetadata(raster);
  const placement = new GPUParameterBuffer(graph.device, {
    id: 'raster-placement',
    format: 'float32',
    length: 4,
    values: getGPUPolygonRasterizationParameterValues(grid)
  });
  const placementView = placement.importToGraph(graph);
  const rasterize = (
    id: string,
    polygons: GPUPolygonGeometryFields,
    zones: GraphDataView<'uint32'>,
    overflow: GraphDataView<'uint32'>
  ) =>
    graph.add(
      new GPUPolygonRasterization({
        id,
        grid,
        crossingCapacity,
        ...polygons,
        parameters: placementView,
        output: {zones, status: {overflow}}
      })
    );
  rasterize('tract-zones', tracts, tractZones, tractRasterOverflow);
  rasterize('area-zones', communityAreas, areaZones, areaRasterOverflow);
  // #region example
  // tractZones and areaZones: both systems rasterized to one grid by GPUPolygonRasterization
  graph.add(
    new GPUArealInterpolation({
      grid,
      spatialContext,
      mode: 'extensive',
      denominator: 'zone',
      sourceZones: tractZones,
      targetZones: areaZones,
      sourceZoneCount: tractCount,
      targetZoneCount: areaCount,
      output: {
        weights,
        areas: overlapAreas,
        sourceTransferredFractions: transferredFractions,
        status: {overflow: pairOverflow}
      }
    })
  );
  graph.add(new GPUSpatialLag({weights, values: population, output: areaPopulation}));
  // areaPopulation: tract residents shared out to the 13 community areas by overlap area
  // #endregion example
  return {
    areaPopulation,
    transferredFractions,
    weights,
    overlapAreas,
    pairOverflow,
    tractRasterOverflow,
    areaRasterOverflow,
    placement
  };
}
