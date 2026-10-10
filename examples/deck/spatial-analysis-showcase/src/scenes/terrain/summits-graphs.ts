// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * The compiled graphs of the summits story: the disc-and-ring summit kernel (one variant per
 * compile option), the peak snap, the critical-point classifier and the contour extractor. Each
 * builder imports the buffers the scene owns, so the scene decides when they are written and read.
 */

import type {Buffer} from '@luma.gl/core';
import {
  DrawCommandBuffer,
  GPUCommandGraph,
  type CompiledGPUCommandGraph
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUParameterBuffer} from '@luma.gl/experimental/gpu-spatial-analysis';
import {
  GPUTerrainContours,
  GPUTerrainCriticalPoints,
  GPUTerrainPeakSnap,
  GPUTerrainSummits,
  GPU_TERRAIN_CRITICAL_POINT_CLASS_COUNT
} from '@luma.gl/experimental/gpu-terrain';
import {importGraphBuffer} from '../../engine/graph-buffers';
import type {SpatialAnalysisResources} from '../../engine/resources';
import {createElevationBand} from './b14b-terrain';

/** Level slots of the contour extractor (the contributor's maximum is a compile-time choice). */
export const CONTOUR_LEVEL_COUNT = 40;
/** Segments each contour level can hold before it overflows. */
export const CONTOUR_SEGMENT_CAPACITY = 30000;
/** Catalogue points the snap graph holds (the tile has about forty named peaks). */
export const CANDIDATE_CAPACITY = 64;

/** The output buffers of one summit run. */
export type SummitBuffers = {
  ids: Buffer;
  drops: Buffer;
  count: Buffer;
  total: Buffer;
  overflow: Buffer;
  clamped: Buffer;
};

/** Creates the buffers a summit run writes, with room for `capacity` listed candidates. */
export function createSummitBuffers(
  resources: SpatialAnalysisResources,
  name: string,
  capacity: number
): SummitBuffers {
  return {
    ids: resources.createBuffer(`${name}-ids`, capacity * 4),
    drops: resources.createBuffer(`${name}-drops`, capacity * 4),
    count: resources.createBuffer(`${name}-count`, 4),
    total: resources.createBuffer(`${name}-total`, 4),
    overflow: resources.createBuffer(`${name}-overflow`, 4),
    clamped: resources.createBuffer(`${name}-clamped`, 4)
  };
}

/** What a summit graph needs to be built. */
export type SummitGraphConfig = {
  name: string;
  width: number;
  height: number;
  pixelCount: number;
  elevation: Buffer;
  validity: Buffer;
  /** Compile-time loop bound, pixels per axis. */
  maximumRadiusPixels: number;
  /** Compile-time handling of discs that leave the grid. */
  incompleteNeighborhood: 'reject' | 'ignore';
  settings: GPUParameterBuffer<'float32'>;
  buffers: SummitBuffers;
  capacity: number;
};

/** Compiles `GPUTerrainSummits` with a compact list of ids and drops. */
export function compileSummitGraph(
  resources: SpatialAnalysisResources,
  config: SummitGraphConfig
): CompiledGPUCommandGraph<void> {
  const {buffers, capacity} = config;
  const graph = new GPUCommandGraph<void>(resources.device, {id: config.name});
  graph.add(
    new GPUTerrainSummits({
      id: 'summits',
      width: config.width,
      height: config.height,
      elevation: createElevationBand(
        graph,
        'summits',
        config.elevation,
        config.validity,
        config.pixelCount
      ),
      cellSizeMode: 'web-mercator',
      maximumRadiusPixels: config.maximumRadiusPixels,
      incompleteNeighborhood: config.incompleteNeighborhood,
      settings: config.settings.importToGraph(graph),
      output: {
        ids: importGraphBuffer(graph, 'ids', buffers.ids, 'uint32', capacity),
        count: importGraphBuffer(graph, 'count', buffers.count, 'uint32', 1),
        overflow: importGraphBuffer(graph, 'overflow', buffers.overflow, 'uint32', 1),
        requiredCount: importGraphBuffer(graph, 'total', buffers.total, 'uint32', 1)
      },
      outputDrop: importGraphBuffer(graph, 'drops', buffers.drops, 'float32', capacity),
      overflow: importGraphBuffer(graph, 'clamped', buffers.clamped, 'uint32', 1)
    })
  );
  return resources.track(graph.compile());
}

/** The buffers of the peak-snap graph. */
export type SnapBuffers = {
  candidates: Buffer;
  candidateHeights: Buffer;
  candidateRadii: Buffer;
  positions: Buffer;
  heights: Buffer;
  status: Buffer;
  distance: Buffer;
  overflow: Buffer;
};

/** Creates the snap buffers for {@link CANDIDATE_CAPACITY} candidates. */
export function createSnapBuffers(resources: SpatialAnalysisResources): SnapBuffers {
  return {
    candidates: resources.createBuffer('candidates', CANDIDATE_CAPACITY * 8),
    candidateHeights: resources.createBuffer('candidate-heights', CANDIDATE_CAPACITY * 4),
    candidateRadii: resources.createBuffer('candidate-radii', CANDIDATE_CAPACITY * 4),
    positions: resources.createBuffer('snap-positions', CANDIDATE_CAPACITY * 8),
    heights: resources.createBuffer('snap-heights', CANDIDATE_CAPACITY * 4),
    status: resources.createBuffer('snap-status', CANDIDATE_CAPACITY * 4),
    distance: resources.createBuffer('snap-distance', CANDIDATE_CAPACITY * 4),
    overflow: resources.createBuffer('snap-overflow', 4)
  };
}

/** Compiles `GPUTerrainPeakSnap` over the catalogue buffers. */
export function compileSnapGraph(
  resources: SpatialAnalysisResources,
  config: {
    width: number;
    height: number;
    pixelCount: number;
    elevation: Buffer;
    validity: Buffer;
    settings: GPUParameterBuffer<'float32'>;
    buffers: SnapBuffers;
  }
): CompiledGPUCommandGraph<void> {
  const {buffers} = config;
  const graph = new GPUCommandGraph<void>(resources.device, {id: 'summits-snap'});
  graph.add(
    new GPUTerrainPeakSnap({
      id: 'snap',
      width: config.width,
      height: config.height,
      elevation: createElevationBand(
        graph,
        'snap',
        config.elevation,
        config.validity,
        config.pixelCount
      ),
      cellSizeMode: 'web-mercator',
      maximumRadiusPixels: 16,
      candidates: importGraphBuffer(
        graph,
        'candidates',
        buffers.candidates,
        'float32x2',
        CANDIDATE_CAPACITY
      ),
      candidateHeights: importGraphBuffer(
        graph,
        'candidate-heights',
        buffers.candidateHeights,
        'float32',
        CANDIDATE_CAPACITY
      ),
      candidateRadii: importGraphBuffer(
        graph,
        'candidate-radii',
        buffers.candidateRadii,
        'float32',
        CANDIDATE_CAPACITY
      ),
      settings: config.settings.importToGraph(graph),
      positions: importGraphBuffer(
        graph,
        'positions',
        buffers.positions,
        'float32x2',
        CANDIDATE_CAPACITY
      ),
      heights: importGraphBuffer(graph, 'heights', buffers.heights, 'float32', CANDIDATE_CAPACITY),
      status: importGraphBuffer(graph, 'status', buffers.status, 'uint32', CANDIDATE_CAPACITY),
      snapDistance: importGraphBuffer(
        graph,
        'distance',
        buffers.distance,
        'float32',
        CANDIDATE_CAPACITY
      ),
      overflow: importGraphBuffer(graph, 'overflow', buffers.overflow, 'uint32', 1)
    })
  );
  return resources.track(graph.compile());
}

/** Compiles `GPUTerrainCriticalPoints` for a ring of 8 or 6 neighbours. */
export function compileCriticalGraph(
  resources: SpatialAnalysisResources,
  config: {
    width: number;
    height: number;
    pixelCount: number;
    elevation: Buffer;
    validity: Buffer;
    connectivity: 8 | 6;
    classes: Buffer;
    signChanges: Buffer;
    counts: Buffer;
  }
): CompiledGPUCommandGraph<void> {
  const graph = new GPUCommandGraph<void>(resources.device, {
    id: `summits-critical-${config.connectivity}`
  });
  graph.add(
    new GPUTerrainCriticalPoints({
      id: 'critical',
      width: config.width,
      height: config.height,
      elevation: createElevationBand(
        graph,
        'critical',
        config.elevation,
        config.validity,
        config.pixelCount
      ),
      connectivity: config.connectivity,
      classes: importGraphBuffer(graph, 'classes', config.classes, 'uint32', config.pixelCount),
      signChanges: importGraphBuffer(
        graph,
        'signs',
        config.signChanges,
        'uint32',
        config.pixelCount
      ),
      counts: importGraphBuffer(
        graph,
        'counts',
        config.counts,
        'uint32',
        GPU_TERRAIN_CRITICAL_POINT_CLASS_COUNT
      )
    })
  );
  return resources.track(graph.compile());
}

/** The buffers of the contour graph. */
export type ContourBuffers = {
  levels: GPUParameterBuffer<'float32'>;
  overflow: Buffer;
  vertices: Buffer[];
  counts: Buffer[];
  drawCommands: DrawCommandBuffer;
};

/** Creates the contour level, vertex and draw-record buffers. */
export function createContourBuffers(resources: SpatialAnalysisResources): ContourBuffers {
  const vertices: Buffer[] = [];
  const counts: Buffer[] = [];
  for (let level = 0; level < CONTOUR_LEVEL_COUNT; level++) {
    vertices.push(
      resources.createBuffer(`contour-vertices-${level}`, CONTOUR_SEGMENT_CAPACITY * 16)
    );
    counts.push(resources.createBuffer(`contour-count-${level}`, 4));
  }
  return {
    levels: resources.createParameterBuffer('levels', 'float32', CONTOUR_LEVEL_COUNT),
    overflow: resources.createBuffer('contour-overflow', 4),
    vertices,
    counts,
    drawCommands: resources.track(
      new DrawCommandBuffer(resources.device, {
        id: 'summits-contour-draw',
        type: 'draw',
        commands: Array.from({length: CONTOUR_LEVEL_COUNT}, () => ({
          vertexCount: 6,
          instanceCount: 0
        }))
      })
    )
  };
}

/** Compiles `GPUTerrainContours` over the full-resolution DEM, one slot per level. */
export function compileContourGraph(
  resources: SpatialAnalysisResources,
  config: {
    width: number;
    height: number;
    pixelCount: number;
    elevation: Buffer;
    validity: Buffer;
    buffers: ContourBuffers;
  }
): CompiledGPUCommandGraph<void> {
  const {buffers} = config;
  const graph = new GPUCommandGraph<void>(resources.device, {id: 'summits-contours'});
  const levelsView = buffers.levels.importToGraph(graph);
  const drawView = buffers.drawCommands.importToGraph(graph);
  graph.add(
    new GPUTerrainContours({
      id: 'contours',
      width: config.width,
      height: config.height,
      elevation: createElevationBand(
        graph,
        'contours',
        config.elevation,
        config.validity,
        config.pixelCount
      ),
      overflow: importGraphBuffer(graph, 'overflow', buffers.overflow, 'uint32', 1),
      levels: Array.from({length: CONTOUR_LEVEL_COUNT}, (_, level) => ({
        level: graph.createDataView(levelsView.buffer, {
          format: 'float32',
          length: 1,
          byteOffset: level * 4
        }),
        vertices: importGraphBuffer(
          graph,
          `vertices-${level}`,
          buffers.vertices[level],
          'float32x2',
          CONTOUR_SEGMENT_CAPACITY * 2
        ),
        segmentCount: importGraphBuffer(
          graph,
          `count-${level}`,
          buffers.counts[level],
          'uint32',
          1
        ),
        // The contributor rewrites [verticesPerInstance, segmentCount, 0, 0] on the GPU, so the
        // segment layer draws exactly the segments found, with no readback.
        draw: drawView,
        drawCommandIndex: level,
        drawLayout: 'instanced' as const,
        verticesPerInstance: 6,
        capacity: CONTOUR_SEGMENT_CAPACITY
      }))
    })
  );
  return resources.track(graph.compile());
}
