// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Terrain hydrology of the San Francisco elevation raster, entirely on the GPU. One compiled graph
 * holds GPUTerrainDerivatives (a hillshade backdrop), GPUTerrainFlow (Planchon-Darboux depression
 * fill, D8 flow directions, flow accumulation and a stream mask) and three small mode-local
 * kernels that prepare display rasters (fill depth, log10 accumulation, direction index). The
 * stream threshold and fill epsilon are a per-frame parameter buffer. Sea (elevation 0) is
 * invalid, so the coast is where water leaves the grid.
 *
 * The graph is encoded only when an input changed: every output persists in its buffer, and the
 * fill plus accumulation rounds are the expensive part. Readbacks are one 16-byte summary
 * (convergence flags and the stream cell count from `GPUHistogram`) through `GPUReadbackRing`.
 */

import type {Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {
  GPUCommandGraph,
  GPUHistogram,
  GPUReadbackRing,
  type CompiledGPUCommandGraph
} from '@luma.gl/gpgpu/gpu-core';
import {getGPUTerrainDerivativesParameterValues, getGPUTerrainFlowParameterValues, GPU_TERRAIN_DERIVATIVES_PARAMETER_LENGTH, GPU_TERRAIN_FLOW_PARAMETER_LENGTH, GPUTerrainDerivatives, GPUTerrainFlow} from '@luma.gl/experimental/gpu-terrain';
import {importGraphBuffer} from '@luma.gl/experimental/UNRESOLVED';
import {MapGraphsRasterLayer} from '../map-graphs-layers';
import type {MapGraphsModeDefinition, MapGraphsModeInstance} from '../map-graphs-mode';
import {formatCount, MapGraphsResources} from '../map-graphs-resources';
import {addDirectionIndexPass, addFillDepthPass, addLogAccumulationPass} from './terrain-kernels';

/** Compile-time iteration limits of the fill and accumulation relaxations. */
const MAXIMUM_FILL_ITERATIONS = 256;
const MAXIMUM_ACCUMULATION_ITERATIONS = 128;
const SUN_AZIMUTH_DEGREES = 315;
const SUN_ALTITUDE_DEGREES = 40;
/** Frames between summary readbacks. */
const READBACK_INTERVAL_FRAMES = 15;
/** Fill epsilon choices in elevation meters; index 0 leaves filled depressions as exact flats. */
const FILL_EPSILONS = [0, 0.0001, 0.001, 0.01, 0.1] as const;
const DIRECTION_PALETTE = [
  [255, 90, 90, 230],
  [255, 160, 60, 230],
  [245, 225, 80, 230],
  [140, 220, 90, 230],
  [70, 210, 190, 230],
  [70, 150, 255, 230],
  [150, 110, 255, 230],
  [235, 100, 220, 230]
] as const;

type Display = 'streams' | 'accumulation' | 'fill' | 'directions';

export const hydrologyMode: MapGraphsModeDefinition = {
  id: 'hydrology',
  title: 'Hydrology',
  recipes: ['GPUTerrainFlow', 'GPUTerrainDerivatives', 'GPUHistogram'],
  description:
    'Depression fill, D8 flow directions, flow accumulation and streams over the San Francisco ' +
    'elevation raster. The stream threshold and fill epsilon are per-frame parameters; the ' +
    'graph is encoded only when one changes.',
  initialViewState: {longitude: -122.44, latitude: 37.735, zoom: 12},

  async create(context) {
    const terrain = await context.data.getSanFranciscoTerrain();
    context.signal.throwIfAborted();
    const {device} = context;
    const {width, height, bounds, cellSize} = terrain;
    const cellCount = width * height;
    const origin: [number, number, number] = [terrain.origin[0], terrain.origin[1], 0];
    const resources = new MapGraphsResources(device, 'hydrology');
    const cellAreaSquareMeters = cellSize[0] * cellSize[1];

    // Sea (elevation 0) is invalid: it has no elevation, so flow leaves the grid at the coast.
    const validityValues = new Uint32Array(cellCount);
    let landCellCount = 0;
    for (let index = 0; index < cellCount; index++) {
      validityValues[index] = terrain.elevation[index] > 0.5 ? 1 : 0;
      landCellCount += validityValues[index];
    }
    const elevationBuffer = resources.createBuffer('elevation', terrain.elevation);
    const validityBuffer = resources.createBuffer('validity', validityValues);
    const hillshadeBuffer = resources.createBuffer('hillshade', cellCount * 4);
    const filledBuffer = resources.createBuffer('filled', cellCount * 4);
    const directionsBuffer = resources.createBuffer('directions', cellCount * 4);
    const accumulationBuffer = resources.createBuffer('accumulation', cellCount * 4);
    const streamsBuffer = resources.createBuffer('streams', cellCount * 4);
    const fillDepthBuffer = resources.createBuffer('fill-depth', cellCount * 4);
    const logAccumulationBuffer = resources.createBuffer('log-accumulation', cellCount * 4);
    const directionIndexBuffer = resources.createBuffer('direction-index', cellCount * 4);
    // Summary outputs. Disjoint ranges of one buffer are rejected by the recipe's aliasing check,
    // so each output is its own buffer and the readback copies them side by side.
    const fillConvergedBuffer = resources.createBuffer('fill-converged', 4);
    const accumulationConvergedBuffer = resources.createBuffer('accumulation-converged', 4);
    const streamCountsBuffer = resources.createBuffer('stream-counts', 8);
    const flowSettings = resources.createParameterBuffer(
      'flow-settings',
      'float32',
      GPU_TERRAIN_FLOW_PARAMETER_LENGTH
    );
    const derivativesSettings = resources.createParameterBuffer(
      'derivatives-settings',
      'float32',
      GPU_TERRAIN_DERIVATIVES_PARAMETER_LENGTH
    );
    const readbackRing = resources.track(
      new GPUReadbackRing(device, {id: 'hydrology-summary', byteLength: 16})
    );

    const graph = new GPUCommandGraph<void>(device, {id: 'hydrology'});
    const elevationView = importGraphBuffer(
      graph,
      'elevation',
      elevationBuffer,
      'float32',
      cellCount
    );
    const elevation = {
      id: 'elevation',
      format: 'float32' as const,
      storage: {kind: 'buffer' as const, values: elevationView},
      validity: importGraphBuffer(graph, 'validity', validityBuffer, 'uint32', cellCount)
    };
    const filled = importGraphBuffer(graph, 'filled', filledBuffer, 'float32', cellCount);
    const directions = importGraphBuffer(
      graph,
      'directions',
      directionsBuffer,
      'uint32',
      cellCount
    );
    const accumulation = importGraphBuffer(
      graph,
      'accumulation',
      accumulationBuffer,
      'float32',
      cellCount
    );
    const streams = importGraphBuffer(graph, 'streams', streamsBuffer, 'uint32', cellCount);
    const fillConverged = importGraphBuffer(
      graph,
      'fill-converged',
      fillConvergedBuffer,
      'uint32',
      1
    );
    const accumulationConverged = importGraphBuffer(
      graph,
      'accumulation-converged',
      accumulationConvergedBuffer,
      'uint32',
      1
    );
    const streamCounts = importGraphBuffer(graph, 'stream-counts', streamCountsBuffer, 'uint32', 2);
    graph.add(
      new GPUTerrainDerivatives({
        id: 'derivatives',
        width,
        height,
        elevation,
        settings: derivativesSettings.importToGraph(graph),
        hillshade: importGraphBuffer(graph, 'hillshade', hillshadeBuffer, 'float32', cellCount),
        cellSizeMode: 'uniform',
        rowDirection: 'south'
      })
    );
    graph.add(
      new GPUTerrainFlow({
        id: 'flow',
        width,
        height,
        elevation,
        settings: flowSettings.importToGraph(graph),
        cellSizeMode: 'uniform',
        fillDepressions: true,
        maxFillIterations: MAXIMUM_FILL_ITERATIONS,
        maxAccumulationIterations: MAXIMUM_ACCUMULATION_ITERATIONS,
        accumulationUnits: 'cells',
        filledElevation: filled,
        flowDirections: directions,
        accumulation,
        streams,
        fillConverged,
        accumulationConverged
      })
    );
    graph.add(
      new GPUHistogram({
        id: 'stream-count',
        input: streams,
        output: streamCounts,
        domain: [0, 2]
      })
    );
    addFillDepthPass(graph, {
      id: 'fill-depth',
      cellCount,
      elevation: elevationView,
      filled,
      output: importGraphBuffer(graph, 'fill-depth', fillDepthBuffer, 'float32', cellCount)
    });
    addLogAccumulationPass(graph, {
      id: 'log-accumulation',
      cellCount,
      accumulation,
      output: importGraphBuffer(
        graph,
        'log-accumulation',
        logAccumulationBuffer,
        'float32',
        cellCount
      )
    });
    addDirectionIndexPass(graph, {
      id: 'direction-index',
      cellCount,
      directions,
      output: importGraphBuffer(graph, 'direction-index', directionIndexBuffer, 'uint32', cellCount)
    });
    const compiled: CompiledGPUCommandGraph<void> = resources.track(graph.compile());

    // --- State ---------------------------------------------------------------------------------
    let display: Display = 'streams';
    let showStreams = true;
    let thresholdExponent = 2.3;
    let epsilonIndex = 2;
    let dirty = true;
    let destroyed = false;
    let readbackPending = false;

    const getThresholdCells = () => Math.round(10 ** thresholdExponent);
    const writeFlowSettings = () => {
      flowSettings.write(
        getGPUTerrainFlowParameterValues({
          cellSize,
          fillEpsilon: FILL_EPSILONS[epsilonIndex],
          streamThreshold: getThresholdCells()
        })
      );
      dirty = true;
    };
    derivativesSettings.write(
      getGPUTerrainDerivativesParameterValues({
        cellSize,
        azimuthDegrees: SUN_AZIMUTH_DEGREES,
        altitudeDegrees: SUN_ALTITUDE_DEGREES
      })
    );
    writeFlowSettings();

    // --- Controls ------------------------------------------------------------------------------
    context.controls.addSelect<Display>({
      label: 'Raster display',
      options: [
        {value: 'streams', label: 'Hillshade only (streams overlay)'},
        {value: 'accumulation', label: 'Log flow accumulation'},
        {value: 'fill', label: 'Depression fill depth'},
        {value: 'directions', label: 'D8 flow direction'}
      ],
      value: display,
      onChange: value => {
        display = value;
        context.updateLayers();
      }
    });
    context.controls.addToggle({
      label: 'Stream overlay',
      value: showStreams,
      onChange: value => {
        showStreams = value;
        context.updateLayers();
      }
    });
    context.controls.addSlider({
      label: 'Stream threshold (per-frame)',
      min: 1,
      max: 4,
      step: 0.1,
      value: thresholdExponent,
      format: value => {
        const cells = Math.round(10 ** value);
        return `${formatCount(cells)} cells, ${(cells * cellAreaSquareMeters * 1e-6).toFixed(2)} km²`;
      },
      onChange: value => {
        thresholdExponent = value;
        writeFlowSettings();
      }
    });
    context.controls.addSlider({
      label: 'Fill epsilon (per-frame)',
      min: 0,
      max: FILL_EPSILONS.length - 1,
      step: 1,
      value: epsilonIndex,
      format: value => (value === 0 ? '0 (flats)' : `${FILL_EPSILONS[value]} m per step`),
      onChange: value => {
        epsilonIndex = value;
        writeFlowSettings();
      }
    });
    context.controls.addLegend({
      title: 'Log10 accumulation (cells upstream)',
      gradient: {
        colors: [
          [68, 1, 84],
          [33, 145, 140],
          [253, 231, 37]
        ],
        minimumLabel: '1',
        maximumLabel: '100,000'
      }
    });
    context.controls.addLegend({
      title: 'D8 direction (downslope)',
      entries: ['E', 'SE', 'S', 'SW', 'W', 'NW', 'N', 'NE'].map((label, index) => ({
        color: DIRECTION_PALETTE[index],
        label
      }))
    });
    context.controls.addNote(
      'Sea is no data, so water leaves the grid at the coast. Fill depth shows cells raised ' +
        'to remove sinks; the 8-bit DEM has 2 m steps, so plateaus fill as flats.'
    );
    context.controls.addReadout('Raster', `${width} × ${height} cells`);
    context.controls.addReadout(
      'Cell size',
      `${cellSize[0].toFixed(1)} × ${cellSize[1].toFixed(1)} m`
    );
    const convergedReadout = context.controls.addReadout('Converged', 'pending');
    const streamReadout = context.controls.addReadout('Stream cells', 'pending');
    context.controls.addReadout('Data', terrain.attribution);

    // --- Per-frame encode ----------------------------------------------------------------------
    const readSummary = async (commandEncoder: Parameters<MapGraphsModeInstance['encode']>[0]) => {
      const ticket = readbackRing.tryAcquire();
      if (!ticket) return;
      [
        [fillConvergedBuffer, 0, 4],
        [accumulationConvergedBuffer, 4, 4],
        [streamCountsBuffer, 8, 8]
      ].forEach(([sourceBuffer, destinationOffset, size]) => {
        commandEncoder.copyBufferToBuffer({
          sourceBuffer: sourceBuffer as Buffer,
          destinationBuffer: ticket.buffer,
          destinationOffset: destinationOffset as number,
          size: size as number
        });
      });
      ticket.markEncoded({byteOffset: 0, byteLength: 16});
      readbackPending = true;
      try {
        const bytes = await ticket.read();
        if (destroyed) return;
        const words = new Uint32Array(bytes.buffer, bytes.byteOffset, 4);
        const fillText = words[0] ? 'fill yes' : `fill no (${MAXIMUM_FILL_ITERATIONS} limit)`;
        const accumulationText = words[1]
          ? 'accumulation yes'
          : `accumulation no (${MAXIMUM_ACCUMULATION_ITERATIONS} limit)`;
        convergedReadout.setValue(`${fillText}, ${accumulationText}`);
        streamReadout.setValue(
          `${formatCount(words[3])} of ${formatCount(landCellCount)} land cells`
        );
      } catch {
        // The ring or device was destroyed while the read was in flight.
      } finally {
        readbackPending = false;
      }
    };

    const instance: MapGraphsModeInstance = {
      getCompiledGraphs: () => [compiled],
      encode(commandEncoder, frame) {
        // Every output persists in its buffer, so the graph only re-encodes when a setting changed.
        if (dirty || frame.frameIndex < 2) {
          compiled.encode(commandEncoder, {parameters: undefined});
          dirty = false;
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
        const layers: Layer[] = [
          new MapGraphsRasterLayer({
            ...rasterProps,
            id: 'hydrology-hillshade',
            values: hillshadeBuffer,
            valueFormat: 'float32',
            colormap: 'grayscale',
            valueRange: [0, 1],
            color: [255, 255, 255, 150]
          })
        ];
        if (display === 'accumulation') {
          layers.push(
            new MapGraphsRasterLayer({
              ...rasterProps,
              id: 'hydrology-accumulation',
              values: logAccumulationBuffer,
              valueFormat: 'float32',
              colormap: 'viridis',
              valueRange: [0, 5],
              discardAtOrBelow: -0.5,
              color: [255, 255, 255, 215]
            })
          );
        } else if (display === 'fill') {
          layers.push(
            new MapGraphsRasterLayer({
              ...rasterProps,
              id: 'hydrology-fill-depth',
              values: fillDepthBuffer,
              valueFormat: 'float32',
              colormap: 'inferno',
              valueRange: [0, 10],
              discardAtOrBelow: 0.5,
              color: [255, 255, 255, 230]
            })
          );
        } else if (display === 'directions') {
          layers.push(
            new MapGraphsRasterLayer({
              ...rasterProps,
              id: 'hydrology-directions',
              values: directionIndexBuffer,
              valueFormat: 'uint32',
              colormap: 'category',
              palette: DIRECTION_PALETTE
            })
          );
        }
        if (showStreams) {
          layers.push(
            new MapGraphsRasterLayer({
              ...rasterProps,
              id: 'hydrology-streams',
              values: streamsBuffer,
              valueFormat: 'uint32',
              colormap: 'mask',
              color: [40, 170, 255, 255],
              noDataColor: [0, 0, 0, 0]
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
    return instance;
  }
};
