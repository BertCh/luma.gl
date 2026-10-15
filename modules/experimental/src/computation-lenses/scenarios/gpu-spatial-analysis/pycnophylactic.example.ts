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
  GPUParameterBuffer,
  GPUPycnophylactic,
  type GPUPolygonGeometryFields,
  type GPUSpatialContext
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Inputs the lens scenario uploads from `s2-chicago-north-tracts.json`. */
export type PycnophylacticExampleInput = {
  /** Census tracts in an equal-area frame. */
  tracts: GPUPolygonGeometryFields;
  /** Residents per tract, the totals each zone must keep. */
  population: GraphDataView<'float32'>;
  /** Lambert equal-area context of the local frame. */
  spatialContext: GPUSpatialContext;
  /** Raster size, cell size and top-left corner `[minX, maxY]` in frame metres. */
  raster: {width: number; height: number; cellSize: number; origin: readonly [number, number]};
  /** Upper bound on edge-row crossings of the tract rings. */
  crossingCapacity: number;
};

/** Smooth tract population surface on a 100 m raster that keeps every tract total. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {tracts, population, spatialContext, raster, crossingCapacity}: PycnophylacticExampleInput
) {
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const cellCount = raster.width * raster.height;
  const zones = createTransientView(graph, 'zones', 'uint32', cellCount, usage);
  const rasterOverflow = createTransientView(graph, 'rasterOverflow', 'uint32', 1, usage);
  const mass = createTransientView(graph, 'mass', 'float32', cellCount, usage);
  const converged = createTransientView(graph, 'converged', 'uint32', 1, usage);
  const iterationCount = createTransientView(graph, 'iterationCount', 'uint32', 1, usage);
  const invalidCount = createTransientView(graph, 'invalidCount', 'uint32', 1, usage);
  const grid = makeGPURasterMetadata(raster);
  const placement = new GPUParameterBuffer(graph.device, {
    id: 'raster-placement',
    format: 'float32',
    length: 4,
    values: getGPUPolygonRasterizationParameterValues(grid)
  });
  const tolerance = new GPUParameterBuffer(graph.device, {
    id: 'tolerance',
    format: 'float32',
    length: 1,
    values: Float32Array.of(0.001)
  });
  graph.add(
    new GPUPolygonRasterization({
      id: 'tract-zones',
      grid,
      crossingCapacity,
      ...tracts,
      parameters: placement.importToGraph(graph),
      output: {zones, status: {overflow: rasterOverflow}}
    })
  );
  // #region example
  // zones: tracts rasterized to the grid by GPUPolygonRasterization; stops early at tolerance 0.001
  graph.add(
    new GPUPycnophylactic({
      grid,
      zones,
      zoneCount: tracts.featureOffsets.length - 1,
      totals: population,
      spatialContext,
      maximumIterations: 30,
      tolerance: tolerance.importToGraph(graph),
      output: {mass, status: {converged, iterationCount, invalidCount}}
    })
  );
  // mass: residents per cell; the cells of each tract sum back to its population
  // #endregion example
  return {
    zones,
    rasterOverflow,
    mass,
    converged,
    iterationCount,
    invalidCount,
    placement,
    tolerance
  };
}
