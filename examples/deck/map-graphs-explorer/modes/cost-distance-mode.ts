// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Least-cost travel over the San Francisco elevation raster, entirely on the GPU. Two compiled
 * graphs share caller-owned buffers. The cost graph computes slope (`GPUTerrainDerivatives`), a
 * friction surface from it (`1 + weight * (slope / 10 degrees)^2`, a small mode-local kernel
 * because the recipe's friction calibration is compile-time) and the accumulated cost, back-links,
 * isochrone bands and convergence flags (`GPUCostDistance`). The path graph follows the back-links
 * from the destination (`GPUCostDistancePath`) and converts the cell list to line segments that a
 * layer draws with an indirect instance count. Sea (elevation 0) is invalid and impassable.
 *
 * The source cell, destination cell, slope weight and cost limit are parameter buffer writes.
 * The cost graph is encoded only when the source, weight or limit changed (relaxation is the
 * expensive part); the path graph also re-encodes when only the destination moved.
 */

import type {Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {
  DrawCommandBuffer,
  GPUCommandGraph,
  GPUReadbackRing,
  type CompiledGPUCommandGraph
} from '@luma.gl/gpgpu/gpu-core';
import {getGPUCostDistanceParameterValues, GPU_COST_DISTANCE_NONE, GPU_COST_DISTANCE_PARAMETER_LENGTH, GPUCostDistance, GPUCostDistancePath} from '@luma.gl/experimental/gpu-raster';
import {getGPUTerrainDerivativesParameterValues, GPU_TERRAIN_DERIVATIVES_PARAMETER_LENGTH, GPUTerrainDerivatives} from '@luma.gl/experimental/gpu-terrain';
import {importGraphBuffer} from '@luma.gl/experimental/UNRESOLVED';
import {LocalMetricProjection} from '../map-graphs-data';
import {
  MapGraphsPointLayer,
  MapGraphsRasterLayer,
  MapGraphsSegmentLayer
} from '../map-graphs-layers';
import type {
  MapGraphsModeDefinition,
  MapGraphsModeInstance,
  MapGraphsPointerEvent
} from '../map-graphs-mode';
import {formatCount, MapGraphsResources} from '../map-graphs-resources';
import {addPathSegmentsPass, addSlopeFrictionPass} from './terrain-kernels';

/** Compile-time relaxation limit; the graph stops early on the GPU once costs converge. */
const MAXIMUM_ITERATIONS = 128;
/** Compile-time path capacity in cells. Longer paths report overflow. */
const PATH_CAPACITY = 2048;
const BAND_COUNT = 8;
const READBACK_INTERVAL_FRAMES = 15;
const SUN_AZIMUTH_DEGREES = 315;
const SUN_ALTITUDE_DEGREES = 40;
/** Cost limit slider maximum; this value means no limit. Units are kilometers of flat ground. */
const NO_LIMIT_KILOMETERS = 60;
/** Color range of the cost and band displays when there is no limit, in kilometers. */
const UNLIMITED_RANGE_KILOMETERS = 30;
const SOURCE_LONGITUDE_LATITUDE: readonly [number, number] = [-122.4194, 37.7599];
const DESTINATION_LONGITUDE_LATITUDE: readonly [number, number] = [-122.4725, 37.7536];
const BAND_PALETTE = [
  [253, 231, 37, 235],
  [160, 218, 57, 235],
  [74, 193, 109, 235],
  [31, 161, 135, 235],
  [39, 127, 142, 235],
  [54, 92, 141, 235],
  [70, 50, 127, 235],
  [68, 1, 84, 235]
] as const;

type Display = 'cost' | 'bands' | 'friction';
type Placement = 'source' | 'destination';

export const costDistanceMode: MapGraphsModeDefinition = {
  id: 'cost-distance',
  title: 'Cost distance',
  recipes: ['GPUCostDistance', 'GPUCostDistancePath', 'GPUTerrainDerivatives'],
  description:
    'Least-cost travel over San Francisco terrain, where steep slopes are expensive. Choose ' +
    'whether a click places the source or the destination; slope weight, cost limit, source ' +
    'and destination are buffer writes, never a recompile.',
  initialViewState: {longitude: -122.445, latitude: 37.758, zoom: 12.2},

  async create(context) {
    const terrain = await context.data.getSanFranciscoTerrain();
    context.signal.throwIfAborted();
    const {device} = context;
    const {width, height, bounds, cellSize} = terrain;
    const cellCount = width * height;
    const projection = new LocalMetricProjection(terrain.origin);
    const origin: [number, number, number] = [terrain.origin[0], terrain.origin[1], 0];
    const resources = new MapGraphsResources(device, 'cost-distance');

    // Sea (elevation 0) is invalid: it has no slope, no friction and is impassable.
    const validityValues = new Uint32Array(cellCount);
    for (let index = 0; index < cellCount; index++) {
      validityValues[index] = terrain.elevation[index] > 0.5 ? 1 : 0;
    }
    const elevationBuffer = resources.createBuffer('elevation', terrain.elevation);
    const validityBuffer = resources.createBuffer('validity', validityValues);
    const hillshadeBuffer = resources.createBuffer('hillshade', cellCount * 4);
    const slopeBuffer = resources.createBuffer('slope', cellCount * 4);
    const frictionBuffer = resources.createBuffer('friction', cellCount * 4);
    const costsBuffer = resources.createBuffer('costs', cellCount * 4);
    const backLinksBuffer = resources.createBuffer('back-links', cellCount * 4);
    const bandsBuffer = resources.createBuffer('bands', cellCount * 4);
    const bandCountsBuffer = resources.createBuffer('band-counts', BAND_COUNT * 4);
    const convergedBuffer = resources.createBuffer('converged', 4);
    const iterationCountBuffer = resources.createBuffer('iteration-count', 4);
    const pathIdsBuffer = resources.createBuffer('path-ids', PATH_CAPACITY * 4);
    const pathCountBuffer = resources.createBuffer('path-count', 4);
    const pathOverflowBuffer = resources.createBuffer('path-overflow', 4);
    const pathTotalBuffer = resources.createBuffer('path-total', 4);
    const pathSegmentsBuffer = resources.createBuffer('path-segments', PATH_CAPACITY * 16);
    const markerBuffer = resources.createBuffer('markers', 16);
    const destinationIdBuffer = resources.createBuffer('destination-id', Uint32Array.of(1));
    const derivativesSettings = resources.createParameterBuffer(
      'derivatives-settings',
      'float32',
      GPU_TERRAIN_DERIVATIVES_PARAMETER_LENGTH
    );
    const costSettings = resources.createParameterBuffer(
      'cost-settings',
      'float32',
      GPU_COST_DISTANCE_PARAMETER_LENGTH
    );
    const slopeWeight = resources.createParameterBuffer('slope-weight', 'float32', 1);
    const sources = resources.createParameterBuffer('sources', 'uint32', 1);
    const target = resources.createParameterBuffer('target', 'uint32', 1);
    const bandThresholds = resources.createParameterBuffer(
      'band-thresholds',
      'float32',
      BAND_COUNT
    );
    const drawCommands = resources.track(
      new DrawCommandBuffer(device, {
        id: 'cost-distance-path-draw',
        type: 'draw',
        commands: [{vertexCount: 6, instanceCount: 0}]
      })
    );
    // Cost at the destination, path count, overflow and total, converged, iterations, band counts.
    const summaryWordCount = 6 + BAND_COUNT;
    const readbackRing = resources.track(
      new GPUReadbackRing(device, {id: 'cost-distance-summary', byteLength: summaryWordCount * 4})
    );

    // --- Cost graph: slope, friction, accumulated cost, back-links, bands -----------------------
    const costGraph = new GPUCommandGraph<void>(device, {id: 'cost-distance'});
    const elevation = {
      id: 'elevation',
      format: 'float32' as const,
      storage: {
        kind: 'buffer' as const,
        values: importGraphBuffer(costGraph, 'elevation', elevationBuffer, 'float32', cellCount)
      },
      validity: importGraphBuffer(costGraph, 'validity', validityBuffer, 'uint32', cellCount)
    };
    const slope = importGraphBuffer(costGraph, 'slope', slopeBuffer, 'float32', cellCount);
    const friction = importGraphBuffer(costGraph, 'friction', frictionBuffer, 'float32', cellCount);
    costGraph.add(
      new GPUTerrainDerivatives({
        id: 'derivatives',
        width,
        height,
        elevation,
        settings: derivativesSettings.importToGraph(costGraph),
        slope,
        hillshade: importGraphBuffer(costGraph, 'hillshade', hillshadeBuffer, 'float32', cellCount),
        cellSizeMode: 'uniform',
        rowDirection: 'south'
      })
    );
    addSlopeFrictionPass(costGraph, {
      id: 'slope-friction',
      cellCount,
      slope,
      weight: slopeWeight.importToGraph(costGraph),
      output: friction
    });
    costGraph.add(
      new GPUCostDistance({
        id: 'cost',
        width,
        height,
        friction: {
          id: 'friction',
          format: 'float32',
          storage: {kind: 'buffer', values: friction},
          validity: importGraphBuffer(
            costGraph,
            'friction-validity',
            validityBuffer,
            'uint32',
            cellCount
          )
        },
        settings: costSettings.importToGraph(costGraph),
        cellSizeMode: 'uniform',
        sources: sources.importToGraph(costGraph),
        maxIterations: MAXIMUM_ITERATIONS,
        costs: importGraphBuffer(costGraph, 'costs', costsBuffer, 'float32', cellCount),
        backLinks: importGraphBuffer(costGraph, 'back-links', backLinksBuffer, 'uint32', cellCount),
        bandThresholds: bandThresholds.importToGraph(costGraph),
        bands: importGraphBuffer(costGraph, 'bands', bandsBuffer, 'uint32', cellCount),
        bandCounts: importGraphBuffer(
          costGraph,
          'band-counts',
          bandCountsBuffer,
          'uint32',
          BAND_COUNT
        ),
        converged: importGraphBuffer(costGraph, 'converged', convergedBuffer, 'uint32', 1),
        iterationCount: importGraphBuffer(
          costGraph,
          'iteration-count',
          iterationCountBuffer,
          'uint32',
          1
        )
      })
    );
    const compiledCost: CompiledGPUCommandGraph<void> = resources.track(costGraph.compile());

    // --- Path graph: back-links to a cell list to line segments ---------------------------------
    const pathGraph = new GPUCommandGraph<void>(device, {id: 'cost-distance-path'});
    const pathIds = importGraphBuffer(
      pathGraph,
      'path-ids',
      pathIdsBuffer,
      'uint32',
      PATH_CAPACITY
    );
    const pathCount = importGraphBuffer(pathGraph, 'path-count', pathCountBuffer, 'uint32', 1);
    pathGraph.add(
      new GPUCostDistancePath({
        id: 'path',
        width,
        height,
        backLinks: importGraphBuffer(pathGraph, 'back-links', backLinksBuffer, 'uint32', cellCount),
        target: target.importToGraph(pathGraph),
        output: {
          ids: pathIds,
          count: pathCount,
          overflow: importGraphBuffer(pathGraph, 'path-overflow', pathOverflowBuffer, 'uint32', 1),
          totalCount: importGraphBuffer(
            pathGraph,
            'path-total-output',
            pathTotalBuffer,
            'uint32',
            1
          )
        }
      })
    );
    addPathSegmentsPass(pathGraph, {
      id: 'path-segments',
      width,
      capacity: PATH_CAPACITY,
      bounds,
      cellSize,
      ids: pathIds,
      count: pathCount,
      segments: importGraphBuffer(
        pathGraph,
        'path-segments',
        pathSegmentsBuffer,
        'float32',
        PATH_CAPACITY * 4
      )
    });
    const compiledPath: CompiledGPUCommandGraph<void> = resources.track(pathGraph.compile());

    // --- State ---------------------------------------------------------------------------------
    let display: Display = 'cost';
    let placement: Placement = 'source';
    let slopeWeightValue = 2;
    let limitKilometers = 25;
    let sourceCell = getDefaultCell(SOURCE_LONGITUDE_LATITUDE, [0.35, 0.4]);
    let destinationCell = getDefaultCell(DESTINATION_LONGITUDE_LATITUDE, [0.8, 0.7]);
    let costDirty = true;
    let pathDirty = true;
    let destroyed = false;
    let readbackPending = false;

    function getCellCenter(cell: number): [number, number] {
      return [
        bounds[0] + ((cell % width) + 0.5) * cellSize[0],
        bounds[3] - (Math.floor(cell / width) + 0.5) * cellSize[1]
      ];
    }

    function getCellAt(longitude: number, latitude: number): number {
      const [x, y] = projection.project(longitude, latitude);
      const column = Math.min(width - 1, Math.max(0, Math.floor((x - bounds[0]) / cellSize[0])));
      const row = Math.min(height - 1, Math.max(0, Math.floor((bounds[3] - y) / cellSize[1])));
      return row * width + column;
    }

    /** Returns the land cell closest to `cell` in grid space. */
    function findNearestLandCell(cell: number): number {
      const column = cell % width;
      const row = Math.floor(cell / width);
      let best = cell;
      let bestDistance = Infinity;
      for (let candidate = 0; candidate < cellCount; candidate++) {
        if (validityValues[candidate] === 0) continue;
        const distance =
          ((candidate % width) - column) ** 2 + (Math.floor(candidate / width) - row) ** 2;
        if (distance < bestDistance) {
          bestDistance = distance;
          best = candidate;
        }
      }
      return best;
    }

    /** Default endpoint: a real place for San Francisco data, a fixed grid fraction otherwise. */
    function getDefaultCell(
      place: readonly [number, number],
      fraction: readonly [number, number]
    ): number {
      const cell =
        terrain.source === 'synthetic'
          ? Math.floor(height * fraction[1]) * width + Math.floor(width * fraction[0])
          : getCellAt(place[0], place[1]);
      return findNearestLandCell(cell);
    }

    const getLimit = () =>
      limitKilometers >= NO_LIMIT_KILOMETERS ? Infinity : limitKilometers * 1000;
    const getRange = () =>
      getLimit() === Infinity ? UNLIMITED_RANGE_KILOMETERS * 1000 : getLimit();

    const writeSettings = () => {
      costSettings.write(getGPUCostDistanceParameterValues({cellSize, costLimit: getLimit()}));
      bandThresholds.write(
        Float32Array.from({length: BAND_COUNT}, (_, band) => ((band + 1) * getRange()) / BAND_COUNT)
      );
      slopeWeight.write(Float32Array.of(slopeWeightValue));
      costDirty = true;
      pathDirty = true;
    };
    const writeEndpoints = () => {
      sources.write(Uint32Array.of(sourceCell));
      target.write(Uint32Array.of(destinationCell));
      markerBuffer.write(
        Float32Array.from([...getCellCenter(sourceCell), ...getCellCenter(destinationCell)])
      );
    };
    derivativesSettings.write(
      getGPUTerrainDerivativesParameterValues({
        cellSize,
        azimuthDegrees: SUN_AZIMUTH_DEGREES,
        altitudeDegrees: SUN_ALTITUDE_DEGREES
      })
    );
    writeSettings();
    writeEndpoints();

    // --- Controls ------------------------------------------------------------------------------
    context.controls.addSelect<Placement>({
      label: 'Click places',
      options: [
        {value: 'source', label: 'Source (white)'},
        {value: 'destination', label: 'Destination (red)'}
      ],
      value: placement,
      onChange: value => {
        placement = value;
      }
    });
    context.controls.addSelect<Display>({
      label: 'Raster display',
      options: [
        {value: 'cost', label: 'Accumulated cost'},
        {value: 'bands', label: 'Cost isochrone bands'},
        {value: 'friction', label: 'Friction surface'}
      ],
      value: display,
      onChange: value => {
        display = value;
        context.updateLayers();
      }
    });
    context.controls.addSlider({
      label: 'Slope weight (per-frame)',
      min: 0,
      max: 8,
      step: 0.25,
      value: slopeWeightValue,
      format: value => (value === 0 ? '0 (straight line)' : `${value}`),
      onChange: value => {
        slopeWeightValue = value;
        writeSettings();
      }
    });
    context.controls.addSlider({
      label: 'Cost limit (per-frame)',
      min: 1,
      max: NO_LIMIT_KILOMETERS,
      step: 1,
      value: limitKilometers,
      format: value => (value >= NO_LIMIT_KILOMETERS ? 'none' : `${value} km of flat ground`),
      onChange: value => {
        limitKilometers = value;
        writeSettings();
        context.updateLayers();
      }
    });
    context.controls.addLegend({
      title: 'Accumulated cost (0 to the limit)',
      gradient: {
        colors: [
          [68, 1, 84],
          [33, 145, 140],
          [253, 231, 37]
        ],
        minimumLabel: '0',
        maximumLabel: 'limit'
      }
    });
    context.controls.addLegend({
      title: 'Cost bands (share of the limit)',
      entries: BAND_PALETTE.map((color, band) => ({
        color,
        label: `${(band * 100) / BAND_COUNT}-${((band + 1) * 100) / BAND_COUNT}%`
      }))
    });
    context.controls.addNote(
      'Friction = 1 + weight × (slope / 10°)², cost per meter. Pick what a click places, then ' +
        'click the map. Sea is impassable.'
    );
    context.controls.addReadout('Raster', `${width} × ${height} cells`);
    const convergedReadout = context.controls.addReadout('Converged', 'pending');
    const pathReadout = context.controls.addReadout('Path', 'pending');
    const bandReadout = context.controls.addReadout('Cells per band', 'pending');
    context.controls.addReadout('Data', terrain.attribution);

    function placeEndpoint(event: MapGraphsPointerEvent): boolean {
      if (!event.coordinate) return false;
      const cell = getCellAt(event.coordinate[0], event.coordinate[1]);
      if (validityValues[cell] === 0) {
        context.setStatus('Sea is impassable: click on land.');
        return true;
      }
      context.setStatus('');
      if (placement === 'source') {
        sourceCell = cell;
        costDirty = true;
      } else {
        destinationCell = cell;
      }
      pathDirty = true;
      writeEndpoints();
      return true;
    }

    // --- Per-frame encode ----------------------------------------------------------------------
    const readSummary = async (commandEncoder: Parameters<MapGraphsModeInstance['encode']>[0]) => {
      const ticket = readbackRing.tryAcquire();
      if (!ticket) return;
      const copies: [Buffer, number, number, number][] = [
        [costsBuffer, destinationCell * 4, 0, 4],
        [pathCountBuffer, 0, 4, 4],
        [pathOverflowBuffer, 0, 8, 4],
        [pathTotalBuffer, 0, 12, 4],
        [convergedBuffer, 0, 16, 4],
        [iterationCountBuffer, 0, 20, 4],
        [bandCountsBuffer, 0, 24, BAND_COUNT * 4]
      ];
      for (const [sourceBuffer, sourceOffset, destinationOffset, size] of copies) {
        commandEncoder.copyBufferToBuffer({
          sourceBuffer,
          sourceOffset,
          destinationBuffer: ticket.buffer,
          destinationOffset,
          size
        });
      }
      ticket.markEncoded({byteOffset: 0, byteLength: summaryWordCount * 4});
      readbackPending = true;
      try {
        const bytes = await ticket.read();
        if (destroyed) return;
        const words = new Uint32Array(bytes.buffer, bytes.byteOffset, summaryWordCount);
        const cost = new Float32Array(words.buffer, words.byteOffset, 1)[0];
        convergedReadout.setValue(
          `${words[4] ? 'yes' : 'no (iteration limit)'} after ${words[5]} of ${MAXIMUM_ITERATIONS} iterations`
        );
        pathReadout.setValue(
          words[3] === 0
            ? 'destination unreached'
            : `${formatCount(words[3])} cells, cost ${(cost / 1000).toFixed(1)} km-equivalent${words[2] ? ', overflow' : ''}`
        );
        bandReadout.setValue(Array.from(words.subarray(6), formatCount).join(' / '));
      } catch {
        // The ring or device was destroyed while the read was in flight.
      } finally {
        readbackPending = false;
      }
    };

    const instance: MapGraphsModeInstance = {
      getCompiledGraphs: () => [compiledCost, compiledPath],
      encode(commandEncoder, frame) {
        // Outputs persist in their buffers: relax only when source, weight or limit changed.
        if (costDirty || frame.frameIndex < 2) {
          compiledCost.encode(commandEncoder, {parameters: undefined});
          costDirty = false;
          pathDirty = true;
        }
        if (pathDirty) {
          compiledPath.encode(commandEncoder, {parameters: undefined});
          // Instance-count word of the one 16-byte draw record: one segment per path cell.
          commandEncoder.copyBufferToBuffer({
            sourceBuffer: pathCountBuffer,
            sourceOffset: 0,
            destinationBuffer: drawCommands.buffer,
            destinationOffset: 4,
            size: 4
          });
          pathDirty = false;
        }
        if (frame.frameIndex % READBACK_INTERVAL_FRAMES === 1 && !readbackPending) {
          void readSummary(commandEncoder);
        }
      },
      getLayers() {
        const rasterProps = {
          coordinateOrigin: origin,
          gridSize: [width, height] as const,
          bounds,
          rowOrigin: 'north' as const
        };
        const range = getRange();
        const layers: Layer[] = [
          new MapGraphsRasterLayer({
            ...rasterProps,
            id: 'cost-distance-hillshade',
            values: hillshadeBuffer,
            valueFormat: 'float32',
            colormap: 'grayscale',
            valueRange: [0, 1],
            color: [255, 255, 255, 140]
          })
        ];
        if (display === 'cost') {
          layers.push(
            new MapGraphsRasterLayer({
              ...rasterProps,
              id: 'cost-distance-cost',
              values: costsBuffer,
              valueFormat: 'float32',
              colormap: 'viridis',
              valueRange: [0, range],
              color: [255, 255, 255, 170]
            })
          );
        } else if (display === 'bands') {
          layers.push(
            new MapGraphsRasterLayer({
              ...rasterProps,
              id: 'cost-distance-bands',
              values: bandsBuffer,
              valueFormat: 'uint32',
              colormap: 'category',
              palette: BAND_PALETTE,
              noDataValue: GPU_COST_DISTANCE_NONE
            })
          );
        } else {
          layers.push(
            new MapGraphsRasterLayer({
              ...rasterProps,
              id: 'cost-distance-friction',
              values: frictionBuffer,
              valueFormat: 'float32',
              colormap: 'inferno',
              valueRange: [1, 1 + slopeWeightValue * 9],
              color: [255, 255, 255, 200]
            })
          );
        }
        layers.push(
          new MapGraphsSegmentLayer({
            id: 'cost-distance-path-halo',
            coordinateOrigin: origin,
            segments: pathSegmentsBuffer,
            drawCommands,
            drawCommandIndex: 0,
            widthPixels: 6,
            color: [0, 0, 0, 200]
          }),
          new MapGraphsSegmentLayer({
            id: 'cost-distance-path',
            coordinateOrigin: origin,
            segments: pathSegmentsBuffer,
            drawCommands,
            drawCommandIndex: 0,
            widthPixels: 3.5,
            color: [255, 80, 70, 255]
          }),
          new MapGraphsPointLayer({
            id: 'cost-distance-source',
            coordinateOrigin: origin,
            positions: markerBuffer,
            instanceCount: 1,
            radiusPixels: 8,
            color: [255, 255, 255, 255]
          }),
          new MapGraphsPointLayer({
            id: 'cost-distance-destination',
            coordinateOrigin: origin,
            positions: markerBuffer,
            ids: destinationIdBuffer,
            instanceCount: 1,
            radiusPixels: 8,
            color: [235, 40, 60, 255]
          })
        );
        return layers;
      },
      onClick: placeEndpoint,
      destroy() {
        destroyed = true;
        resources.destroy();
      }
    };
    return instance;
  }
};
