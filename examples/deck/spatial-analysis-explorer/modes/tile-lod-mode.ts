// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Tile LOD: GPU-driven level-of-detail selection over a procedural 80 km quadtree.
 *
 * One compiled `GPUTileLODSelection` graph picks the tiles to draw and the tiles to request from
 * the camera alone. The view (frustum planes, camera, error threshold, foveation), the budget, and
 * the residency flags are parameter buffers rewritten every frame. A small readback of the bounded
 * request list every few frames drives a simulated streaming loop: requested tiles become resident
 * after a deterministic latency, which writes only the changed residency words. Drawing binds the
 * contributor's compact drawn IDs and indirect draw record directly, so the selection never returns to
 * the CPU.
 */

import type {Viewport} from '@deck.gl/core';
import type {Layer} from '@deck.gl/core';
import type {CommandEncoder} from '@luma.gl/core';
import {
  DrawCommandBuffer,
  GPUCommandGraph,
  GPUReadbackRing,
  type CompiledGPUCommandGraph
} from '@luma.gl/gpgpu/gpu-core';
import {
  getGPUTileLODQuadtreeTile,
  getGPUTileLODViewParameterValues,
  GPU_TILE_LOD_STATISTICS_LENGTH,
  GPU_TILE_LOD_UNLIMITED,
  GPU_TILE_LOD_VIEW_LENGTH,
  GPUTileLODSelection,
  makeGPUTileLODQuadtree
} from '@luma.gl/experimental/gpu-tables';
import {importGraphBuffer} from '../graph-buffers';
import {NEW_YORK_ORIGIN} from '../spatial-analysis-data';
import type {
  SpatialAnalysisModeDefinition,
  SpatialAnalysisModeInstance
} from '../spatial-analysis-mode';
import {formatCount, SpatialAnalysisResources} from '../spatial-analysis-resources';
import {TILE_OUTLINE_VERTEX_COUNT, TileOutlineLayer} from './tile-lod-layers';

/** Half the side of the square hierarchy, in meters around the New York origin. */
const HALF_EXTENT_METERS = 40000;
const MAXIMUM_LEVEL = 7;
/** Tile resolution used for the geometric error, `tileWidth / TILE_SIZE_PIXELS` meters per pixel. */
const TILE_SIZE_PIXELS = 64;
const DRAWN_CAPACITY = 8192;
const REQUEST_CAPACITY = 64;
const READBACK_INTERVAL_FRAMES = 10;
const MINIMUM_LATENCY_SECONDS = 0.3;
const LATENCY_SPREAD_SECONDS = 0.5;
/** Budget slider values at or above this mean "no limit". */
const BUDGET_SLIDER_MAXIMUM = 1024;

// Summary readback layout in uint32 words.
const SUMMARY_COUNT_WORD = GPU_TILE_LOD_STATISTICS_LENGTH;
const SUMMARY_OVERFLOW_WORD = SUMMARY_COUNT_WORD + 1;
const SUMMARY_TOTAL_WORD = SUMMARY_COUNT_WORD + 2;
const SUMMARY_IDS_WORD = SUMMARY_COUNT_WORD + 3;
const SUMMARY_PRIORITIES_WORD = SUMMARY_IDS_WORD + REQUEST_CAPACITY;
const SUMMARY_WORD_COUNT = SUMMARY_PRIORITIES_WORD + REQUEST_CAPACITY;

export const tileLodMode: SpatialAnalysisModeDefinition = {
  id: 'tile-lod',
  title: 'Tile LOD',
  contributors: ['GPUTileLODSelection'],
  description:
    'GPU level-of-detail selection over an 80 km quadtree around New York. Tiles refine where the ' +
    'camera looks; missing tiles are requested, stream in after a simulated delay, and fill in. ' +
    'Pan and zoom: the view, budget, and residency are parameter buffers, never a recompile.',
  initialViewState: {
    longitude: NEW_YORK_ORIGIN[0],
    latitude: NEW_YORK_ORIGIN[1],
    // A pitched camera makes screen-space error vary with distance, so the LOD gradient is visible.
    zoom: 12,
    pitch: 60,
    bearing: 20
  },

  async create(context) {
    const {device} = context;
    const resources = new SpatialAnalysisResources(device, 'tile-lod');
    const quadtree = makeGPUTileLODQuadtree({
      bounds: [-HALF_EXTENT_METERS, -HALF_EXTENT_METERS, HALF_EXTENT_METERS, HALF_EXTENT_METERS],
      maximumLevel: MAXIMUM_LEVEL,
      tileSize: TILE_SIZE_PIXELS
    });
    const nodeCount = quadtree.nodeCount;

    // Static per-node tile rectangles and levels, for drawing only.
    const tileBounds = new Float32Array(nodeCount * 4);
    const tileLevels = new Uint32Array(nodeCount);
    for (let node = 0; node < nodeCount; node++) {
      const {level, x, y} = getGPUTileLODQuadtreeTile(node, MAXIMUM_LEVEL);
      const width = (2 * HALF_EXTENT_METERS) / 2 ** level;
      const minX = -HALF_EXTENT_METERS + x * width;
      const minY = -HALF_EXTENT_METERS + y * width;
      tileBounds.set([minX, minY, minX + width, minY + width], node * 4);
      tileLevels[node] = level;
    }
    const tileBoundsBuffer = resources.createBuffer('tile-bounds', tileBounds);
    const tileLevelsBuffer = resources.createBuffer('tile-levels', tileLevels);

    const sphereBoundsBuffer = resources.createBuffer('sphere-bounds', quadtree.sphereBounds);
    const geometricErrorsBuffer = resources.createBuffer(
      'geometric-errors',
      quadtree.geometricErrors
    );
    const childrenBuffer = resources.createBuffer('children', quadtree.children);

    // Per-frame parameters.
    const viewBuffer = resources.createParameterBuffer('view', 'float32', GPU_TILE_LOD_VIEW_LENGTH);
    const budgetBuffer = resources.createParameterBuffer(
      'budget',
      'uint32',
      2,
      Uint32Array.of(GPU_TILE_LOD_UNLIMITED, GPU_TILE_LOD_UNLIMITED)
    );
    const residency = new Uint32Array(nodeCount);
    residency[0] = 1;
    const residencyBuffer = resources.createParameterBuffer(
      'residency',
      'uint32',
      nodeCount,
      residency
    );

    // Contributor outputs.
    const drawnIds = resources.createBuffer('drawn-ids', DRAWN_CAPACITY * 4);
    const drawnCount = resources.createBuffer('drawn-count', 4);
    const drawnOverflow = resources.createBuffer('drawn-overflow', 4);
    const drawnTotal = resources.createBuffer('drawn-total', 4);
    const requestIds = resources.createBuffer('request-ids', REQUEST_CAPACITY * 4);
    const requestCount = resources.createBuffer('request-count', 4);
    const requestOverflow = resources.createBuffer('request-overflow', 4);
    const requestTotal = resources.createBuffer('request-total', 4);
    const requestPriorities = resources.createBuffer('request-priorities', REQUEST_CAPACITY * 4);
    const statistics = resources.createBuffer('statistics', GPU_TILE_LOD_STATISTICS_LENGTH * 4);
    const drawCommands = resources.track(
      new DrawCommandBuffer(device, {
        id: 'tile-lod-draw',
        type: 'draw',
        commands: [{vertexCount: TILE_OUTLINE_VERTEX_COUNT, instanceCount: 0}]
      })
    );

    const graph = new GPUCommandGraph<void>(device, {id: 'tile-lod'});
    graph.add(
      new GPUTileLODSelection({
        id: 'tile-lod',
        hierarchy: {
          sphereBounds: importGraphBuffer(
            graph,
            'sphere-bounds',
            sphereBoundsBuffer,
            'float32x4',
            nodeCount
          ),
          geometricErrors: importGraphBuffer(
            graph,
            'geometric-errors',
            geometricErrorsBuffer,
            'float32',
            nodeCount
          ),
          children: importGraphBuffer(graph, 'children', childrenBuffer, 'uint32x2', nodeCount),
          levelOffsets: quadtree.levelOffsets
        },
        view: viewBuffer.importToGraph(graph),
        residency: residencyBuffer.importToGraph(graph),
        budget: budgetBuffer.importToGraph(graph),
        output: {
          ids: importGraphBuffer(graph, 'drawn-ids', drawnIds, 'uint32', DRAWN_CAPACITY),
          count: importGraphBuffer(graph, 'drawn-count', drawnCount, 'uint32', 1),
          overflow: importGraphBuffer(graph, 'drawn-overflow', drawnOverflow, 'uint32', 1),
          totalCount: importGraphBuffer(graph, 'drawn-total', drawnTotal, 'uint32', 1)
        },
        requests: {
          ids: importGraphBuffer(graph, 'request-ids', requestIds, 'uint32', REQUEST_CAPACITY),
          count: importGraphBuffer(graph, 'request-count', requestCount, 'uint32', 1),
          overflow: importGraphBuffer(graph, 'request-overflow', requestOverflow, 'uint32', 1),
          totalCount: importGraphBuffer(graph, 'request-total', requestTotal, 'uint32', 1),
          priorities: importGraphBuffer(
            graph,
            'request-priorities',
            requestPriorities,
            'float32',
            REQUEST_CAPACITY
          )
        },
        statistics: importGraphBuffer(
          graph,
          'statistics',
          statistics,
          'uint32',
          GPU_TILE_LOD_STATISTICS_LENGTH
        ),
        indirectDraw: {commands: drawCommands.importToGraph(graph)}
      })
    );
    const compiled: CompiledGPUCommandGraph<void> = resources.track(graph.compile());
    const readbackRing = resources.track(
      new GPUReadbackRing(device, {id: 'tile-lod-summary', byteLength: SUMMARY_WORD_COUNT * 4})
    );

    // Controls.
    let maximumScreenSpaceError = 2;
    // Off by default: the pitched camera already varies screen-space error with distance.
    let foveationStrength = 0;
    let foveationRadius = 0.1;
    let tileBudget = GPU_TILE_LOD_UNLIMITED;
    let showRequests = true;
    context.controls.addSlider({
      label: 'Maximum screen-space error (per-frame)',
      min: 1,
      max: 64,
      step: 0.5,
      value: maximumScreenSpaceError,
      format: value => `${value} px`,
      onChange: value => {
        maximumScreenSpaceError = value;
      }
    });
    context.controls.addSlider({
      label: 'Foveation strength (per-frame, 0 = off)',
      min: 0,
      max: 12,
      step: 0.5,
      value: foveationStrength,
      format: value => (value === 0 ? 'off' : value.toFixed(1)),
      onChange: value => {
        foveationStrength = value;
      }
    });
    context.controls.addSlider({
      label: 'Foveation full-detail radius (per-frame)',
      min: 0.02,
      max: 0.6,
      step: 0.01,
      value: foveationRadius,
      format: value => value.toFixed(2),
      onChange: value => {
        foveationRadius = value;
      }
    });
    context.controls.addSlider({
      label: 'Tile budget (per-frame)',
      min: 8,
      max: BUDGET_SLIDER_MAXIMUM,
      step: 8,
      value: BUDGET_SLIDER_MAXIMUM,
      format: value => (value >= BUDGET_SLIDER_MAXIMUM ? 'unlimited' : `${value} tiles`),
      onChange: value => {
        tileBudget = value >= BUDGET_SLIDER_MAXIMUM ? GPU_TILE_LOD_UNLIMITED : value;
      }
    });
    context.controls.addToggle({
      label: 'Show requested tiles',
      value: showRequests,
      onChange: value => {
        showRequests = value;
        context.updateLayers();
      }
    });
    context.controls.addButton({label: 'Clear cache', onClick: () => clearCache()});
    context.controls.addLegend({
      title: 'Drawn tile level (outline and fill)',
      gradient: {
        colors: [
          [64, 140, 255],
          [89, 242, 166],
          [255, 230, 64]
        ],
        minimumLabel: 'level 0 (coarse)',
        maximumLabel: `level ${MAXIMUM_LEVEL}`
      }
    });
    context.controls.addLegend({
      title: 'Requested, not yet resident',
      entries: [{color: [255, 140, 40], label: 'Request (bounded to 64, coarse first)'}]
    });
    const totalReadout = context.controls.addReadout('Tiles in hierarchy', formatCount(nodeCount));
    const drawnReadout = context.controls.addReadout('Drawn tiles');
    const desiredReadout = context.controls.addReadout('Desired frontier');
    const requestReadout = context.controls.addReadout('Requested (this read)');
    const residentReadout = context.controls.addReadout('Resident');
    const inFlightReadout = context.controls.addReadout('In flight');
    const budgetReadout = context.controls.addReadout('Budget');
    const errorReadout = context.controls.addReadout('Max screen-space error');
    context.controls.addNote(
      'Hierarchy, view, residency, and budget are GPU buffers; only a 556-byte summary is read ' +
        'back every 10 frames to drive the simulated loader (300-800 ms latency, nothing evicted).'
    );
    totalReadout.setValue(`${formatCount(nodeCount)} (${MAXIMUM_LEVEL + 1} levels)`);

    // Simulated streaming loader.
    const pending = new Map<number, number>();
    let residentCount = 1;
    let currentTime = 0;
    let destroyed = false;
    let readbackPending = false;

    function clearCache(): void {
      residency.fill(0);
      residency[0] = 1;
      residentCount = 1;
      pending.clear();
      residencyBuffer.write(residency);
    }

    function getLatencySeconds(node: number): number {
      const unit = Math.imul(node + 1, 2654435761) >>> 0;
      return MINIMUM_LATENCY_SECONDS + (unit / 4294967296) * LATENCY_SPREAD_SECONDS;
    }

    function completeArrivedTiles(): void {
      for (const [node, readyTime] of pending) {
        if (readyTime <= currentTime) {
          pending.delete(node);
          if (residency[node] === 0) {
            residency[node] = 1;
            residentCount++;
            // Only the changed word is uploaded.
            residencyBuffer.write(Uint32Array.of(1), node);
          }
        }
      }
    }

    async function readSummary(commandEncoder: CommandEncoder): Promise<void> {
      const ticket = readbackRing.tryAcquire();
      if (!ticket) return;
      const copy = (sourceBuffer: typeof statistics, byteLength: number, destinationWord: number) =>
        commandEncoder.copyBufferToBuffer({
          sourceBuffer,
          sourceOffset: 0,
          destinationBuffer: ticket.buffer,
          destinationOffset: destinationWord * 4,
          size: byteLength
        });
      copy(statistics, GPU_TILE_LOD_STATISTICS_LENGTH * 4, 0);
      copy(requestCount, 4, SUMMARY_COUNT_WORD);
      copy(requestOverflow, 4, SUMMARY_OVERFLOW_WORD);
      copy(requestTotal, 4, SUMMARY_TOTAL_WORD);
      copy(requestIds, REQUEST_CAPACITY * 4, SUMMARY_IDS_WORD);
      copy(requestPriorities, REQUEST_CAPACITY * 4, SUMMARY_PRIORITIES_WORD);
      ticket.markEncoded({byteOffset: 0, byteLength: SUMMARY_WORD_COUNT * 4});
      readbackPending = true;
      try {
        const bytes = await ticket.read();
        if (destroyed) return;
        const words = new Uint32Array(
          bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength)
        );
        const priorities = new Float32Array(
          words.buffer,
          SUMMARY_PRIORITIES_WORD * 4,
          REQUEST_CAPACITY
        );
        const count = Math.min(words[SUMMARY_COUNT_WORD], REQUEST_CAPACITY);
        let maximumPriority = 0;
        for (let index = 0; index < count; index++) {
          const node = words[SUMMARY_IDS_WORD + index];
          if (node < nodeCount && residency[node] === 0 && !pending.has(node)) {
            pending.set(node, currentTime + getLatencySeconds(node));
          }
          maximumPriority = Math.max(maximumPriority, priorities[index]);
        }
        drawnReadout.setValue(formatCount(words[2]));
        desiredReadout.setValue(formatCount(words[0]));
        requestReadout.setValue(
          `${count} of ${formatCount(words[SUMMARY_TOTAL_WORD])}${words[SUMMARY_OVERFLOW_WORD] ? ' (list full)' : ''}`
        );
        budgetReadout.setValue(words[5] ? 'exhausted' : 'not reached');
        errorReadout.setValue(
          count > 0 ? `${maximumPriority.toFixed(1)} px (top request)` : 'none requested'
        );
      } catch {
        // The ring or device was destroyed while the read was in flight.
      } finally {
        readbackPending = false;
      }
    }

    const instance: SpatialAnalysisModeInstance = {
      getCompiledGraphs: () => [compiled],
      encode(commandEncoder, frame) {
        currentTime = frame.timeSeconds;
        completeArrivedTiles();
        viewBuffer.write(
          getGPUTileLODViewParameterValues(
            getHierarchyViewProps(frame.viewport, {
              maximumScreenSpaceError,
              foveationStrength,
              foveationRadius
            })
          )
        );
        budgetBuffer.write(Uint32Array.of(GPU_TILE_LOD_UNLIMITED, tileBudget));
        compiled.encode(commandEncoder, {parameters: undefined});
        residentReadoutUpdate();
        if (frame.frameIndex % READBACK_INTERVAL_FRAMES === 0 && !readbackPending) {
          void readSummary(commandEncoder);
        }
      },
      getLayers() {
        const coordinateOrigin: [number, number, number] = [
          NEW_YORK_ORIGIN[0],
          NEW_YORK_ORIGIN[1],
          0
        ];
        const layers: Layer[] = [
          new TileOutlineLayer({
            id: 'tile-lod-drawn',
            coordinateOrigin,
            bounds: tileBoundsBuffer,
            levels: tileLevelsBuffer,
            ids: drawnIds,
            drawCommands,
            levelColors: true,
            maximumLevel: MAXIMUM_LEVEL,
            color: [255, 255, 255, 235],
            widthPixels: 1.5,
            fillAlpha: 0.18
          })
        ];
        if (showRequests) {
          layers.push(
            new TileOutlineLayer({
              id: 'tile-lod-requests',
              coordinateOrigin,
              bounds: tileBoundsBuffer,
              levels: tileLevelsBuffer,
              ids: requestIds,
              countBuffer: requestCount,
              instanceCount: REQUEST_CAPACITY,
              color: [255, 140, 40, 255],
              widthPixels: 2.5,
              fillAlpha: 0.22
            })
          );
        }
        return layers;
      },
      destroy() {
        destroyed = true;
        resources.destroy();
      }
    };

    function residentReadoutUpdate(): void {
      residentReadout.setValue(`${formatCount(residentCount)} of ${formatCount(nodeCount)}`);
      inFlightReadout.setValue(formatCount(pending.size));
    }
    residentReadoutUpdate();
    return instance;
  }
};

/**
 * Converts a Deck viewport into the contributor's view description in hierarchy space (planar meters
 * around the New York origin). Deck's `viewProjectionMatrix` maps common space, so the hierarchy
 * matrix is `viewProjection * translate(commonOrigin) * scale(unitsPerMeter)`, the same chain the
 * layers apply for `METER_OFFSETS`.
 */
function getHierarchyViewProps(
  viewport: Viewport,
  options: {maximumScreenSpaceError: number; foveationStrength: number; foveationRadius: number}
): Parameters<typeof getGPUTileLODViewParameterValues>[0] {
  const commonOrigin = viewport.projectPosition([NEW_YORK_ORIGIN[0], NEW_YORK_ORIGIN[1], 0]);
  const unitsPerMeter = viewport.distanceScales.unitsPerMeter;
  const viewProjection = viewport.viewProjectionMatrix;
  const matrix = new Array<number>(16);
  for (let row = 0; row < 4; row++) {
    matrix[row] = viewProjection[row] * unitsPerMeter[0];
    matrix[4 + row] = viewProjection[4 + row] * unitsPerMeter[1];
    matrix[8 + row] = viewProjection[8 + row] * unitsPerMeter[2];
    matrix[12 + row] =
      viewProjection[row] * commonOrigin[0] +
      viewProjection[4 + row] * commonOrigin[1] +
      viewProjection[8 + row] * commonOrigin[2] +
      viewProjection[12 + row];
  }
  const camera = viewport.cameraPosition;
  // projectionMatrix[5] is 1 / tan(fovy / 2) for Deck's perspective projection.
  const fieldOfView = 2 * Math.atan(1 / viewport.projectionMatrix[5]);
  return {
    viewProjectionMatrix: matrix,
    cameraPosition: [
      (camera[0] - commonOrigin[0]) / unitsPerMeter[0],
      (camera[1] - commonOrigin[1]) / unitsPerMeter[1],
      (camera[2] - commonOrigin[2]) / unitsPerMeter[2]
    ],
    viewportSize: [viewport.width, viewport.height],
    maximumScreenSpaceError: options.maximumScreenSpaceError,
    verticalFieldOfView: fieldOfView,
    foveation: {
      center: [0.5, 0.5],
      radius: options.foveationRadius,
      strength: options.foveationStrength
    }
  };
}
