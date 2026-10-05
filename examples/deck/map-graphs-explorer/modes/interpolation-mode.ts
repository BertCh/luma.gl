// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Spatial interpolation of scattered San Francisco elevation samples. Four thousand random land
 * points sample the terrain raster; `GPUInverseDistanceWeighting` rebuilds a grid index over them
 * and interpolates a 256 x 192 raster that follows the camera (the extent is a parameter buffer
 * rewritten from the viewport each frame). `GPUFocalStatistics` then runs over that raster with a
 * per-frame window radius and shape, and `GPUTerrainContours` extracts contours from the smoothed
 * mean, all in one compiled graph. Power, radius, nearest-k, minimum neighbors, focal radius and
 * shape, contour interval, and the resample button are buffer writes; the statistic select only
 * chooses which already-computed output buffer a layer draws.
 *
 * With focal radius 0 the mean output equals the IDW surface (every cell is its own window), so
 * the contours always have a valid source and no recompile is needed to switch smoothing on or off.
 *
 * The only readback is the IDW surface every 20 frames, to count nodata cells exactly.
 */

import type {Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {
  DrawCommandBuffer,
  GPUCommandGraph,
  GPUReadbackRing,
  type CompiledGPUCommandGraph
} from '@luma.gl/gpgpu/gpu-core';
import {
  getGPUFocalStatisticsParameterValues,
  getGPUInverseDistanceWeightingParameterValues,
  GPU_FOCAL_STATISTICS_PARAMETER_LENGTH,
  GPU_INVERSE_DISTANCE_WEIGHTING_PARAMETER_LENGTH,
  GPUFocalStatistics,
  GPUInverseDistanceWeighting,
  GPUTerrainContours,
  importGraphBuffer,
  type GPUFocalStatisticsShape
} from '@luma.gl/experimental/map-graphs';
import {createSeededRandom, LocalMetricProjection} from '../map-graphs-data';
import {MapGraphsPointLayer, MapGraphsRasterLayer} from '../map-graphs-layers';
import type {MapGraphsModeDefinition, MapGraphsModeInstance} from '../map-graphs-mode';
import {formatCount, getViewportMetricBounds, MapGraphsResources} from '../map-graphs-resources';
import {createExtentFollowingSegmentLayer} from './interpolation-layers';
import {formatCompiledGraphTiming, measureCompiledGraph} from './vector-timing';

const SAMPLE_COUNT = 4000;
const RASTER_WIDTH = 256;
const RASTER_HEIGHT = 192;
const CELL_COUNT = RASTER_WIDTH * RASTER_HEIGHT;
/** Compile-time capacity of the nearest-k limit. */
const MAXIMUM_NEIGHBOR_COUNT = 16;
/** Compile-time cap on the focal window radius, in raster cells. */
const MAXIMUM_FOCAL_RADIUS = 8;
/** Compile-time number of contour level slots; level `k` is `interval * (k + 1)`. */
const CONTOUR_LEVEL_COUNT = 12;
const CONTOUR_SEGMENT_CAPACITY = 12000;
const INDEX_GRID_SIZE: readonly [number, number] = [48, 48];
const READBACK_INTERVAL_FRAMES = 20;
const AUTO_MEASURE_FRAME = 40;
const SAMPLE_SEED = 7;
const ExtentFollowingSegmentLayer = createExtentFollowingSegmentLayer([
  RASTER_WIDTH,
  RASTER_HEIGHT
]);

type Statistic = 'mean' | 'min' | 'max' | 'standardDeviation';

export const interpolationMode: MapGraphsModeDefinition = {
  id: 'interpolation',
  title: 'Interpolate',
  recipes: ['GPUInverseDistanceWeighting', 'GPUFocalStatistics'],
  description:
    'Four thousand scattered elevation samples are interpolated on the GPU with inverse distance ' +
    'weighting onto a raster that follows the camera, smoothed with focal statistics and ' +
    'contoured. Power, radius, nearest-k and window size are per-frame parameters.',
  initialViewState: {longitude: -122.44, latitude: 37.755, zoom: 11.6},

  async create(context) {
    const terrain = await context.data.getSanFranciscoTerrain();
    context.signal.throwIfAborted();
    const {device} = context;
    const projection = new LocalMetricProjection(terrain.origin);
    const origin: [number, number, number] = [terrain.origin[0], terrain.origin[1], 0];
    const resources = new MapGraphsResources(device, 'interpolation');
    const {bounds: terrainBounds, cellSize, width: terrainWidth, height: terrainHeight} = terrain;
    const maximumElevation = Math.max(terrain.elevationRange[1], 1);

    // --- Samples: random land points that read the terrain raster -------------------------------
    const positionValues = new Float32Array(SAMPLE_COUNT * 2);
    const sampleValues = new Float32Array(SAMPLE_COUNT);
    let resampleCount = 0;
    function generateSamples(): void {
      const random = createSeededRandom(SAMPLE_SEED + resampleCount * 7919);
      for (let sample = 0; sample < SAMPLE_COUNT; ) {
        const x = terrainBounds[0] + random() * (terrainBounds[2] - terrainBounds[0]);
        const y = terrainBounds[1] + random() * (terrainBounds[3] - terrainBounds[1]);
        const column = Math.min(terrainWidth - 1, Math.floor((x - terrainBounds[0]) / cellSize[0]));
        const row = Math.min(terrainHeight - 1, Math.floor((terrainBounds[3] - y) / cellSize[1]));
        const elevation = terrain.elevation[row * terrainWidth + column];
        if (elevation <= 0.5) continue; // Skip the sea.
        positionValues[sample * 2] = x;
        positionValues[sample * 2 + 1] = y;
        sampleValues[sample] = elevation;
        sample++;
      }
    }
    generateSamples();

    const positionsBuffer = resources.createBuffer('positions', positionValues);
    const valuesBuffer = resources.createBuffer('values', sampleValues);
    const surfaceBuffer = resources.createBuffer('surface', CELL_COUNT * 4);
    const countsBuffer = resources.createBuffer('counts', CELL_COUNT * 4);
    const focalBuffers: Record<Statistic, Buffer> = {
      mean: resources.createBuffer('focal-mean', CELL_COUNT * 4),
      min: resources.createBuffer('focal-min', CELL_COUNT * 4),
      max: resources.createBuffer('focal-max', CELL_COUNT * 4),
      standardDeviation: resources.createBuffer('focal-standard-deviation', CELL_COUNT * 4)
    };
    const focalCountBuffer = resources.createBuffer('focal-count', CELL_COUNT * 4);
    const overflowBuffer = resources.createBuffer('overflow', 4);
    const idwParameters = resources.createParameterBuffer(
      'idw-parameters',
      'float32',
      GPU_INVERSE_DISTANCE_WEIGHTING_PARAMETER_LENGTH
    );
    const focalParameters = resources.createParameterBuffer(
      'focal-parameters',
      'float32',
      GPU_FOCAL_STATISTICS_PARAMETER_LENGTH
    );
    const levelValues = resources.createParameterBuffer('levels', 'float32', CONTOUR_LEVEL_COUNT);
    const contourVertices: Buffer[] = [];
    const contourCounts: Buffer[] = [];
    for (let level = 0; level < CONTOUR_LEVEL_COUNT; level++) {
      contourVertices.push(
        resources.createBuffer(`contour-vertices-${level}`, CONTOUR_SEGMENT_CAPACITY * 16)
      );
      contourCounts.push(resources.createBuffer(`contour-count-${level}`, 4));
    }
    const drawCommands = resources.track(
      new DrawCommandBuffer(device, {
        id: 'interpolation-contour-draw',
        type: 'draw',
        commands: Array.from({length: CONTOUR_LEVEL_COUNT}, () => ({
          vertexCount: 6,
          instanceCount: 0
        }))
      })
    );
    const readbackRing = resources.track(
      new GPUReadbackRing(device, {id: 'interpolation-surface', byteLength: CELL_COUNT * 4})
    );

    const graph = new GPUCommandGraph<void>(device, {id: 'interpolation'});
    const surfaceView = importGraphBuffer(graph, 'surface', surfaceBuffer, 'float32', CELL_COUNT);
    graph.add(
      new GPUInverseDistanceWeighting({
        id: 'idw',
        positions: importGraphBuffer(
          graph,
          'positions',
          positionsBuffer,
          'float32x2',
          SAMPLE_COUNT
        ),
        values: importGraphBuffer(graph, 'values', valuesBuffer, 'float32', SAMPLE_COUNT),
        parameters: idwParameters.importToGraph(graph),
        width: RASTER_WIDTH,
        height: RASTER_HEIGHT,
        indexGridSize: INDEX_GRID_SIZE,
        indexBounds: terrainBounds,
        maximumNeighborCount: MAXIMUM_NEIGHBOR_COUNT,
        output: {
          values: surfaceView,
          counts: importGraphBuffer(graph, 'counts', countsBuffer, 'uint32', CELL_COUNT)
        }
      })
    );
    const focalMean = importGraphBuffer(
      graph,
      'focal-mean',
      focalBuffers.mean,
      'float32',
      CELL_COUNT
    );
    graph.add(
      new GPUFocalStatistics({
        id: 'focal',
        values: surfaceView,
        width: RASTER_WIDTH,
        height: RASTER_HEIGHT,
        maximumRadius: MAXIMUM_FOCAL_RADIUS,
        parameters: focalParameters.importToGraph(graph),
        output: {
          mean: focalMean,
          min: importGraphBuffer(graph, 'focal-min', focalBuffers.min, 'float32', CELL_COUNT),
          max: importGraphBuffer(graph, 'focal-max', focalBuffers.max, 'float32', CELL_COUNT),
          standardDeviation: importGraphBuffer(
            graph,
            'focal-standard-deviation',
            focalBuffers.standardDeviation,
            'float32',
            CELL_COUNT
          ),
          count: importGraphBuffer(graph, 'focal-count', focalCountBuffer, 'uint32', CELL_COUNT)
        }
      })
    );
    const levelsView = levelValues.importToGraph(graph);
    graph.add(
      new GPUTerrainContours({
        id: 'contours',
        width: RASTER_WIDTH,
        height: RASTER_HEIGHT,
        elevation: {
          id: 'smoothed',
          format: 'float32',
          storage: {kind: 'buffer', values: focalMean}
        },
        overflow: importGraphBuffer(graph, 'contour-overflow', overflowBuffer, 'uint32', 1),
        levels: Array.from({length: CONTOUR_LEVEL_COUNT}, (_, level) => ({
          level: graph.createDataView(levelsView.buffer, {
            format: 'float32',
            length: 1,
            byteOffset: level * 4
          }),
          vertices: importGraphBuffer(
            graph,
            `vertices-${level}`,
            contourVertices[level],
            'float32x2',
            CONTOUR_SEGMENT_CAPACITY * 2
          ),
          segmentCount: importGraphBuffer(
            graph,
            `count-${level}`,
            contourCounts[level],
            'uint32',
            1
          )
        }))
      })
    );
    const compiled: CompiledGPUCommandGraph<void> = resources.track(graph.compile());

    // --- State ---------------------------------------------------------------------------------
    let power = 2;
    let searchRadius = 900;
    let neighborCount = 0;
    let minimumNeighborCount = 1;
    let focalRadius = 2;
    let focalShape: GPUFocalStatisticsShape = 'circle';
    let statistic: Statistic = 'mean';
    let contourInterval = 20;
    let showContours = true;
    let showSamples = true;
    let destroyed = false;
    let readbackPending = false;
    let measuring = false;
    let autoMeasured = false;
    let encodedFrames = 0;
    let viewBounds: [number, number, number, number] = [...terrainBounds];

    function writeIdwParameters(): void {
      idwParameters.write(
        getGPUInverseDistanceWeightingParameterValues({
          extent: viewBounds,
          searchRadius,
          power,
          neighborCount,
          minimumNeighborCount
        })
      );
    }
    function writeFocalParameters(): void {
      focalParameters.write(
        getGPUFocalStatisticsParameterValues({radius: focalRadius, shape: focalShape})
      );
    }
    function writeLevels(): void {
      levelValues.write(
        Float32Array.from(
          {length: CONTOUR_LEVEL_COUNT},
          (_, level) => contourInterval * (level + 1)
        )
      );
    }
    writeIdwParameters();
    writeFocalParameters();
    writeLevels();

    // --- Controls ------------------------------------------------------------------------------
    context.controls.addSelect<Statistic>({
      label: 'Focal statistic shown (output buffer swap)',
      options: [
        {value: 'mean', label: 'Mean (smoothed surface)'},
        {value: 'min', label: 'Minimum'},
        {value: 'max', label: 'Maximum'},
        {value: 'standardDeviation', label: 'Standard deviation (roughness)'}
      ],
      value: statistic,
      onChange: value => {
        statistic = value;
        context.updateLayers();
      }
    });
    context.controls.addSlider({
      label: 'IDW power (per-frame)',
      min: 0,
      max: 6,
      step: 0.5,
      value: power,
      format: value => value.toFixed(1),
      onChange: value => {
        power = value;
        writeIdwParameters();
      }
    });
    context.controls.addSlider({
      label: 'IDW search radius (per-frame)',
      min: 100,
      max: 3000,
      step: 50,
      value: searchRadius,
      format: value => `${value} m`,
      onChange: value => {
        searchRadius = value;
        writeIdwParameters();
      }
    });
    context.controls.addSlider({
      label: 'IDW nearest neighbors k (per-frame)',
      min: 0,
      max: MAXIMUM_NEIGHBOR_COUNT,
      step: 1,
      value: neighborCount,
      format: value => (value === 0 ? 'all in radius' : String(value)),
      onChange: value => {
        neighborCount = value;
        writeIdwParameters();
      }
    });
    context.controls.addSlider({
      label: 'IDW minimum neighbors (per-frame)',
      min: 1,
      max: 8,
      step: 1,
      value: minimumNeighborCount,
      onChange: value => {
        minimumNeighborCount = value;
        writeIdwParameters();
      }
    });
    context.controls.addSlider({
      label: 'Focal radius, cells (per-frame)',
      min: 0,
      max: MAXIMUM_FOCAL_RADIUS,
      step: 1,
      value: focalRadius,
      format: value => (value === 0 ? 'off' : `${value} cells`),
      onChange: value => {
        focalRadius = value;
        writeFocalParameters();
      }
    });
    context.controls.addSelect<GPUFocalStatisticsShape>({
      label: 'Focal window shape (per-frame)',
      options: [
        {value: 'circle', label: 'Circle'},
        {value: 'square', label: 'Square'}
      ],
      value: focalShape,
      onChange: value => {
        focalShape = value;
        writeFocalParameters();
      }
    });
    context.controls.addSlider({
      label: 'Contour interval (per-frame)',
      min: 10,
      max: 60,
      step: 5,
      value: contourInterval,
      format: value => `${value} m`,
      onChange: value => {
        contourInterval = value;
        writeLevels();
      }
    });
    context.controls.addToggle({
      label: 'Contours',
      value: showContours,
      onChange: value => {
        showContours = value;
        context.updateLayers();
      }
    });
    context.controls.addToggle({
      label: 'Sample points',
      value: showSamples,
      onChange: value => {
        showSamples = value;
        context.updateLayers();
      }
    });
    context.controls.addButton({
      label: 'Resample points (buffer write)',
      onClick: () => {
        resampleCount++;
        generateSamples();
        positionsBuffer.write(positionValues);
        valuesBuffer.write(sampleValues);
      }
    });
    context.controls.addLegend({
      title: 'Interpolated elevation',
      gradient: {
        colors: [
          [68, 1, 84],
          [33, 145, 140],
          [253, 231, 37]
        ],
        minimumLabel: '0 m',
        maximumLabel: `${maximumElevation.toFixed(0)} m`
      }
    });
    context.controls.addNote(
      'Cells with no sample inside the radius are nodata (transparent). Raise the radius or ' +
        'lower the minimum neighbors to fill them; the focal window shrinks near nodata.'
    );
    context.controls.addReadout('Samples', formatCount(SAMPLE_COUNT));
    context.controls.addReadout('Raster', `${RASTER_WIDTH} × ${RASTER_HEIGHT} cells`);
    const cellSizeReadout = context.controls.addReadout('Cell size');
    const nodataReadout = context.controls.addReadout('IDW nodata cells');
    const timingReadout = context.controls.addReadout('Graph time (all stages)');
    context.controls.addButton({label: 'Time the graph now', onClick: () => void measureGraph()});
    context.controls.addReadout('Data', terrain.attribution);

    // --- Timing and readback -------------------------------------------------------------------
    async function measureGraph(): Promise<void> {
      if (measuring || destroyed) return;
      measuring = true;
      timingReadout.setValue('measuring...');
      try {
        const timing = await measureCompiledGraph(device, compiled, {
          parameters: undefined,
          completionBuffer: overflowBuffer,
          signal: context.signal
        });
        if (!destroyed) timingReadout.setValue(formatCompiledGraphTiming(timing));
      } catch {
        if (!destroyed) timingReadout.setValue('interrupted');
      } finally {
        measuring = false;
      }
    }

    const readSummary = async (commandEncoder: Parameters<MapGraphsModeInstance['encode']>[0]) => {
      const ticket = readbackRing.tryAcquire();
      if (!ticket) return;
      commandEncoder.copyBufferToBuffer({
        sourceBuffer: surfaceBuffer,
        sourceOffset: 0,
        destinationBuffer: ticket.buffer,
        destinationOffset: 0,
        size: CELL_COUNT * 4
      });
      ticket.markEncoded({byteOffset: 0, byteLength: CELL_COUNT * 4});
      readbackPending = true;
      try {
        const bytes = await ticket.read();
        if (destroyed) return;
        const surface = new Uint32Array(bytes.buffer, bytes.byteOffset, CELL_COUNT);
        let nodata = 0;
        for (let index = 0; index < CELL_COUNT; index++) {
          if ((surface[index] & 0x7fffffff) >= 0x7f800000) nodata++;
        }
        nodataReadout.setValue(
          `${formatCount(nodata)} of ${formatCount(CELL_COUNT)} (${((100 * nodata) / CELL_COUNT).toFixed(1)}%)`
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
        encodedFrames++;
        viewBounds = getViewportMetricBounds(frame.viewport, projection);
        writeIdwParameters();
        cellSizeReadout.setValue(
          `${((viewBounds[2] - viewBounds[0]) / RASTER_WIDTH).toFixed(0)} m`
        );
        compiled.encode(commandEncoder, {parameters: undefined});
        for (let level = 0; level < CONTOUR_LEVEL_COUNT; level++) {
          // Instance-count word of record `level` (16-byte records of four uint32 words).
          commandEncoder.copyBufferToBuffer({
            sourceBuffer: contourCounts[level],
            sourceOffset: 0,
            destinationBuffer: drawCommands.buffer,
            destinationOffset: level * 16 + 4,
            size: 4
          });
        }
        if (!readbackPending && frame.frameIndex % READBACK_INTERVAL_FRAMES === 0) {
          void readSummary(commandEncoder);
        }
        if (!autoMeasured && encodedFrames >= AUTO_MEASURE_FRAME) {
          autoMeasured = true;
          void measureGraph();
        }
      },
      getLayers() {
        const layers: Layer[] = [];
        const isSpread = statistic === 'standardDeviation';
        layers.push(
          new MapGraphsRasterLayer({
            id: `interpolation-${statistic}`,
            coordinateOrigin: origin,
            gridSize: [RASTER_WIDTH, RASTER_HEIGHT],
            bounds: idwParameters.buffer,
            rowOrigin: 'south',
            values: focalBuffers[statistic],
            valueFormat: 'float32',
            colormap: isSpread ? 'inferno' : 'viridis',
            valueRange: isSpread ? [0, 30] : [0, maximumElevation],
            noDataColor: [0, 0, 0, 0],
            color: [255, 255, 255, 200]
          })
        );
        if (showContours) {
          for (let level = 0; level < CONTOUR_LEVEL_COUNT; level++) {
            const isIndexContour = (level + 1) % 5 === 0;
            layers.push(
              new ExtentFollowingSegmentLayer({
                id: `interpolation-contour-${level}`,
                coordinateOrigin: origin,
                segments: contourVertices[level],
                drawCommands,
                drawCommandIndex: level,
                extent: idwParameters.buffer,
                widthPixels: isIndexContour ? 1.8 : 1,
                color: isIndexContour ? [255, 255, 255, 255] : [255, 255, 255, 170]
              })
            );
          }
        }
        if (showSamples) {
          layers.push(
            new MapGraphsPointLayer({
              id: 'interpolation-samples-halo',
              coordinateOrigin: origin,
              positions: positionsBuffer,
              instanceCount: SAMPLE_COUNT,
              radiusPixels: 2.6,
              color: [0, 0, 0, 200]
            }),
            new MapGraphsPointLayer({
              id: 'interpolation-samples',
              coordinateOrigin: origin,
              positions: positionsBuffer,
              values: valuesBuffer,
              valueFormat: 'float32',
              colormap: 'viridis',
              valueRange: [0, maximumElevation],
              instanceCount: SAMPLE_COUNT,
              radiusPixels: 1.8,
              color: [255, 255, 255, 255]
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
