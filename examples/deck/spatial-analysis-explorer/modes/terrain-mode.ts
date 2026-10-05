// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Terrain analysis of a San Francisco elevation raster, entirely on the GPU. One compiled graph
 * holds GPUTerrainDerivatives (slope, aspect, hillshade), GPUTerrainContours (32 level slots whose
 * values are a per-frame parameter buffer) and GPUTerrainViewshed (observer, height and radius
 * are a per-frame parameter buffer). Deck layers read every output buffer directly; the only
 * readback is a one-word contour overflow flag.
 *
 * Contour draws: the contributor's own `draw` record uses two vertices per instance, while the shared
 * segment layer draws quads (six vertices), so each level copies its GPU-written segment count
 * into the instance-count word of a six-vertex `DrawCommandBuffer` record instead.
 */

import type {Layer} from '@deck.gl/core';
import {
  DrawCommandBuffer,
  GPUCommandGraph,
  GPUReadbackRing,
  type CompiledGPUCommandGraph
} from '@luma.gl/gpgpu/gpu-core';
import type {Buffer} from '@luma.gl/core';
import {
  getGPUTerrainDerivativesParameterValues,
  getGPUTerrainViewshedParameterValues,
  GPU_TERRAIN_DERIVATIVES_PARAMETER_LENGTH,
  GPU_TERRAIN_VIEWSHED_PARAMETER_LENGTH,
  GPUTerrainContours,
  GPUTerrainDerivatives,
  GPUTerrainViewshed
} from '@luma.gl/experimental/gpu-terrain';
import {importGraphBuffer} from '../graph-buffers';
import {LocalMetricProjection} from '../spatial-analysis-data';
import {
  SpatialAnalysisPointLayer,
  SpatialAnalysisRasterLayer,
  SpatialAnalysisSegmentLayer
} from '../spatial-analysis-layers';
import type {
  SpatialAnalysisModeDefinition,
  SpatialAnalysisModeInstance,
  SpatialAnalysisPointerEvent
} from '../spatial-analysis-mode';
import {formatCount, SpatialAnalysisResources} from '../spatial-analysis-resources';

/** Compile-time number of contour level slots; level `k` is `interval * (k + 1)`. */
const CONTOUR_LEVEL_COUNT = 32;
/** Compile-time segment capacity per contour level. */
const CONTOUR_SEGMENT_CAPACITY = 24000;
const SUN_ALTITUDE_DEGREES = 40;
const DRAG_RADIUS_PIXELS = 20;
/** Twin Peaks, the default observer when real data is loaded. */
const TWIN_PEAKS: readonly [number, number] = [-122.4475, 37.7544];
/** Frames between contour overflow readbacks. */
const READBACK_INTERVAL_FRAMES = 20;

type Display = 'hillshade' | 'slope' | 'aspect';

export const terrainMode: SpatialAnalysisModeDefinition = {
  id: 'terrain',
  title: 'Terrain',
  contributors: ['GPUTerrainDerivatives', 'GPUTerrainContours', 'GPUTerrainViewshed'],
  description:
    'Hillshade, slope, aspect, contours and a viewshed computed from one elevation raster. ' +
    'Drag the observer (or click the map) to move the viewshed; sun, contour interval and ' +
    'observer height are per-frame parameters, never a recompile.',
  initialViewState: {longitude: -122.44, latitude: 37.735, zoom: 12},

  async create(context) {
    const terrain = await context.data.getSanFranciscoTerrain();
    context.signal.throwIfAborted();
    const {device} = context;
    const {width, height, bounds, cellSize} = terrain;
    const pixelCount = width * height;
    const projection = new LocalMetricProjection(terrain.origin);
    const origin: [number, number, number] = [terrain.origin[0], terrain.origin[1], 0];
    const resources = new SpatialAnalysisResources(device, 'terrain');

    // Sea (elevation 0) is invalid so derivatives and the viewshed leave it transparent.
    const validityValues = new Uint32Array(pixelCount);
    for (let index = 0; index < pixelCount; index++) {
      validityValues[index] = terrain.elevation[index] > 0.5 ? 1 : 0;
    }
    const elevationBuffer = resources.createBuffer('elevation', terrain.elevation);
    const validityBuffer = resources.createBuffer('validity', validityValues);
    const slopeBuffer = resources.createBuffer('slope', pixelCount * 4);
    const aspectBuffer = resources.createBuffer('aspect', pixelCount * 4);
    const hillshadeBuffer = resources.createBuffer('hillshade', pixelCount * 4);
    const visibilityBuffer = resources.createBuffer('visibility', pixelCount * 4);
    const overflowBuffer = resources.createBuffer('overflow', 4);
    const derivativesSettings = resources.createParameterBuffer(
      'derivatives-settings',
      'float32',
      GPU_TERRAIN_DERIVATIVES_PARAMETER_LENGTH
    );
    const viewshedSettings = resources.createParameterBuffer(
      'viewshed-settings',
      'float32',
      GPU_TERRAIN_VIEWSHED_PARAMETER_LENGTH
    );
    const levelValues = resources.createParameterBuffer('levels', 'float32', CONTOUR_LEVEL_COUNT);
    const observerMarker = resources.createBuffer('observer', Float32Array.of(0, 0));
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
        id: 'terrain-contour-draw',
        type: 'draw',
        commands: Array.from({length: CONTOUR_LEVEL_COUNT}, () => ({
          vertexCount: 6,
          instanceCount: 0
        }))
      })
    );
    const readbackRing = resources.track(
      new GPUReadbackRing(device, {id: 'terrain-overflow', byteLength: 4})
    );

    const graph = new GPUCommandGraph<void>(device, {id: 'terrain'});
    const elevation = {
      id: 'elevation',
      format: 'float32' as const,
      storage: {
        kind: 'buffer' as const,
        values: importGraphBuffer(graph, 'elevation', elevationBuffer, 'float32', pixelCount)
      },
      validity: importGraphBuffer(graph, 'validity', validityBuffer, 'uint32', pixelCount)
    };
    const derivativesView = derivativesSettings.importToGraph(graph);
    const viewshedView = viewshedSettings.importToGraph(graph);
    const levelsView = levelValues.importToGraph(graph);
    graph.add(
      new GPUTerrainDerivatives({
        id: 'derivatives',
        width,
        height,
        elevation,
        settings: derivativesView,
        slope: importGraphBuffer(graph, 'slope', slopeBuffer, 'float32', pixelCount),
        aspect: importGraphBuffer(graph, 'aspect', aspectBuffer, 'float32', pixelCount),
        hillshade: importGraphBuffer(graph, 'hillshade', hillshadeBuffer, 'float32', pixelCount),
        cellSizeMode: 'uniform',
        rowDirection: 'south'
      })
    );
    graph.add(
      new GPUTerrainContours({
        id: 'contours',
        width,
        height,
        elevation,
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
    graph.add(
      new GPUTerrainViewshed({
        id: 'viewshed',
        width,
        height,
        elevation,
        settings: viewshedView,
        visibility: importGraphBuffer(graph, 'visibility', visibilityBuffer, 'uint32', pixelCount)
      })
    );
    const compiled: CompiledGPUCommandGraph<void> = resources.track(graph.compile());

    // --- State ---------------------------------------------------------------------------------
    let display: Display = 'hillshade';
    let showContours = true;
    let showViewshed = true;
    let sunAzimuth = 315;
    let contourInterval = 20;
    let observerHeight = 30;
    let maximumDistance = 6000;
    let observer = getDefaultObserver();
    let dragging = false;
    let destroyed = false;
    let readbackPending = false;
    let overflowFlag = false;

    function getDefaultObserver(): [number, number] {
      if (terrain.source === 'synthetic') {
        let best = 0;
        for (let index = 1; index < pixelCount; index++) {
          if (terrain.elevation[index] > terrain.elevation[best]) best = index;
        }
        const column = best % width;
        const row = Math.floor(best / width);
        return [bounds[0] + (column + 0.5) * cellSize[0], bounds[3] - (row + 0.5) * cellSize[1]];
      }
      return projection.project(TWIN_PEAKS[0], TWIN_PEAKS[1]);
    }

    function writeObserver(): void {
      observerMarker.write(Float32Array.of(observer[0], observer[1]));
      viewshedSettings.write(
        getGPUTerrainViewshedParameterValues({
          // Pixel-center index space; row 0 is the north edge.
          observer: [
            (observer[0] - bounds[0]) / cellSize[0] - 0.5,
            (bounds[3] - observer[1]) / cellSize[1] - 0.5
          ],
          observerHeight,
          maxDistance: maximumDistance,
          cellSize
        })
      );
    }

    function writeSun(): void {
      derivativesSettings.write(
        getGPUTerrainDerivativesParameterValues({
          cellSize,
          azimuthDegrees: sunAzimuth,
          altitudeDegrees: SUN_ALTITUDE_DEGREES
        })
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

    function moveObserver(event: SpatialAnalysisPointerEvent): void {
      if (!event.coordinate) return;
      const [x, y] = projection.project(event.coordinate[0], event.coordinate[1]);
      observer = [
        Math.min(Math.max(x, bounds[0]), bounds[2]),
        Math.min(Math.max(y, bounds[1]), bounds[3])
      ];
      writeObserver();
    }

    writeSun();
    writeLevels();
    writeObserver();

    // --- Controls ------------------------------------------------------------------------------
    context.controls.addSelect<Display>({
      label: 'Raster display',
      options: [
        {value: 'hillshade', label: 'Hillshade'},
        {value: 'slope', label: 'Slope'},
        {value: 'aspect', label: 'Aspect'}
      ],
      value: display,
      onChange: value => {
        display = value;
        context.updateLayers();
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
      label: 'Viewshed',
      value: showViewshed,
      onChange: value => {
        showViewshed = value;
        context.updateLayers();
      }
    });
    context.controls.addSlider({
      label: 'Sun azimuth (per-frame)',
      min: 0,
      max: 360,
      step: 5,
      value: sunAzimuth,
      format: value => `${value}°`,
      onChange: value => {
        sunAzimuth = value;
        writeSun();
      }
    });
    context.controls.addSlider({
      label: 'Contour interval (per-frame)',
      min: 10,
      max: 100,
      step: 10,
      value: contourInterval,
      format: value => `${value} m`,
      onChange: value => {
        contourInterval = value;
        writeLevels();
      }
    });
    context.controls.addSlider({
      label: 'Observer height (per-frame)',
      min: 2,
      max: 200,
      step: 2,
      value: observerHeight,
      format: value => `${value} m`,
      onChange: value => {
        observerHeight = value;
        writeObserver();
      }
    });
    context.controls.addSlider({
      label: 'Viewshed radius (per-frame)',
      min: 1000,
      max: 12000,
      step: 500,
      value: maximumDistance,
      format: value => `${(value / 1000).toFixed(1)} km`,
      onChange: value => {
        maximumDistance = value;
        writeObserver();
      }
    });
    context.controls.addLegend({
      title: 'Viewshed from the observer',
      entries: [
        {color: [60, 230, 110, 150], label: 'Visible'},
        {color: [0, 0, 0, 90], label: 'Hidden'}
      ]
    });
    context.controls.addLegend({
      title: 'Slope, degrees (0 to 45)',
      gradient: {
        colors: [
          [68, 1, 84],
          [33, 145, 140],
          [253, 231, 37]
        ],
        minimumLabel: '0',
        maximumLabel: '45'
      }
    });
    context.controls.addNote('Drag the red observer, or click the map, to move the viewshed.');
    context.controls.addReadout('Raster', `${width} × ${height} cells`);
    context.controls.addReadout(
      'Cell size',
      `${cellSize[0].toFixed(1)} × ${cellSize[1].toFixed(1)} m`
    );
    context.controls.addReadout(
      'Elevation',
      `${terrain.elevationRange[0].toFixed(0)} to ${terrain.elevationRange[1].toFixed(0)} m`
    );
    const contourReadout = context.controls.addReadout('Contour levels', 'ok');
    context.controls.addReadout('Data', terrain.attribution);

    // --- Per-frame encode ----------------------------------------------------------------------
    const readOverflow = async (
      commandEncoder: Parameters<SpatialAnalysisModeInstance['encode']>[0]
    ) => {
      const ticket = readbackRing.tryAcquire();
      if (!ticket) return;
      commandEncoder.copyBufferToBuffer({
        sourceBuffer: overflowBuffer,
        sourceOffset: 0,
        destinationBuffer: ticket.buffer,
        destinationOffset: 0,
        size: 4
      });
      ticket.markEncoded({byteOffset: 0, byteLength: 4});
      readbackPending = true;
      try {
        const bytes = await ticket.read();
        if (destroyed) return;
        overflowFlag = new Uint32Array(bytes.buffer, bytes.byteOffset, 1)[0] !== 0;
        contourReadout.setValue(
          overflowFlag
            ? `overflow (capacity ${formatCount(CONTOUR_SEGMENT_CAPACITY)} segments/level)`
            : `${CONTOUR_LEVEL_COUNT} slots, no overflow`
        );
      } catch {
        // The ring or device was destroyed while the read was in flight.
      } finally {
        readbackPending = false;
      }
    };

    const instance: SpatialAnalysisModeInstance = {
      getCompiledGraphs: () => [compiled],
      encode(commandEncoder, frame) {
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
          void readOverflow(commandEncoder);
        }
      },
      getLayers() {
        const layers: Layer[] = [];
        const rasterProps = {
          coordinateOrigin: origin,
          gridSize: [width, height] as const,
          bounds,
          rowOrigin: 'north' as const,
          valueFormat: 'float32' as const
        };
        if (display === 'hillshade') {
          layers.push(
            new SpatialAnalysisRasterLayer({
              ...rasterProps,
              id: 'terrain-hillshade',
              values: hillshadeBuffer,
              colormap: 'grayscale',
              valueRange: [0, 1],
              color: [255, 255, 255, 150]
            })
          );
        } else if (display === 'slope') {
          layers.push(
            new SpatialAnalysisRasterLayer({
              ...rasterProps,
              id: 'terrain-slope',
              values: slopeBuffer,
              colormap: 'viridis',
              valueRange: [0, 45],
              color: [255, 255, 255, 190]
            })
          );
        } else {
          layers.push(
            new SpatialAnalysisRasterLayer({
              ...rasterProps,
              id: 'terrain-aspect',
              values: aspectBuffer,
              colormap: 'inferno',
              valueRange: [0, 360],
              discardAtOrBelow: -0.5,
              color: [255, 255, 255, 190]
            })
          );
        }
        if (showViewshed) {
          layers.push(
            new SpatialAnalysisRasterLayer({
              ...rasterProps,
              id: 'terrain-viewshed',
              values: visibilityBuffer,
              valueFormat: 'uint32',
              colormap: 'category',
              // GPU_TERRAIN_VISIBILITY: hidden, visible, outOfRange, noData.
              palette: [
                [10, 10, 30, 55],
                [60, 230, 110, 95],
                [0, 0, 0, 0],
                [0, 0, 0, 0]
              ]
            })
          );
        }
        if (showContours) {
          for (let level = 0; level < CONTOUR_LEVEL_COUNT; level++) {
            const isIndexContour = (level + 1) % 5 === 0;
            layers.push(
              new SpatialAnalysisSegmentLayer({
                id: `terrain-contour-${level}`,
                coordinateOrigin: origin,
                segments: contourVertices[level],
                drawCommands,
                drawCommandIndex: level,
                // Vertices are in pixel-edge grid units with row 0 at the north edge.
                positionScale: [cellSize[0], -cellSize[1]],
                positionOffset: [bounds[0], bounds[3]],
                widthPixels: isIndexContour ? 1.6 : 0.9,
                color: isIndexContour ? [255, 214, 120, 255] : [255, 160, 60, 190]
              })
            );
          }
        }
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'terrain-observer-halo',
            coordinateOrigin: origin,
            positions: observerMarker,
            instanceCount: 1,
            radiusPixels: 11,
            color: [255, 255, 255, 230]
          }),
          new SpatialAnalysisPointLayer({
            id: 'terrain-observer',
            coordinateOrigin: origin,
            positions: observerMarker,
            instanceCount: 1,
            radiusPixels: 7,
            color: [230, 40, 60, 255]
          })
        );
        return layers;
      },
      onClick(event) {
        if (!event.coordinate) return false;
        moveObserver(event);
        return true;
      },
      onDragStart(event) {
        const viewport = context.getViewport();
        if (!viewport || !event.pixel) return false;
        const [longitude, latitude] = projection.unproject(observer[0], observer[1]);
        const [x, y] = viewport.project([longitude, latitude]);
        if (Math.hypot(x - event.pixel[0], y - event.pixel[1]) > DRAG_RADIUS_PIXELS) return false;
        dragging = true;
        context.setMapDragEnabled(false);
        return true;
      },
      onDrag(event) {
        if (dragging) moveObserver(event);
      },
      onDragEnd(event) {
        if (!dragging) return;
        moveObserver(event);
        dragging = false;
        context.setMapDragEnabled(true);
      },
      destroy() {
        destroyed = true;
        if (dragging) context.setMapDragEnabled(true);
        resources.destroy();
      }
    };
    return instance;
  }
};
