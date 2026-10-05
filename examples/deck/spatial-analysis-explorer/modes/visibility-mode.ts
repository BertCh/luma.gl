// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Visibility analysis of the San Francisco elevation raster, entirely on the GPU. Every graph reads
 * one elevation buffer:
 *
 * - `GPURasterExtremaPyramid` (setup graph, once): the one min-max pyramid every `'pyramid'`
 *   traversal reads through its `pyramid` prop (nothing builds a pyramid per frame), shown as a
 *   coarse maximum raster and as the DEM extrema of its root cell.
 * - `GPUTerrainViewshed`: visible, marginal (inside the tolerance band) and hidden cells from the
 *   observer, tolerance band and refraction are per-frame parameters.
 * - `GPUPointHorizonProfile`: the 360 degree skyline (elevation angle, distance and evaluated
 *   sample count per azimuth) from the same observer, drawn as a strip chart and as a ring on the map.
 * - `GPUTerrainLineOfSight`: one observer-target pair with a clearance in meters.
 * - `GPUTerrainCumulativeViewshed`: how many of six observers see each cell.
 *
 * Traversal (`'march'` or `'pyramid'`) is a compile-time choice of each contributor, so the mode
 * compiles both variants once into separate graphs with separate output buffers. The toggle only
 * chooses which one is encoded and drawn, and the "Pyramid = march" readout encodes both outside
 * the frame and compares every output bit for bit.
 */

import type {Layer} from '@deck.gl/core';
import {
  GPUCommandGraph,
  GPUReadbackRing,
  type CompiledGPUCommandGraph
} from '@luma.gl/gpgpu/gpu-core';
import type {Buffer, CommandEncoder} from '@luma.gl/core';
import {
  getGPURasterExtremaPyramidLayout,
  GPURasterExtremaPyramid
} from '@luma.gl/experimental/gpu-raster';
import {
  getGPUPointHorizonParameterValues,
  getGPUTerrainCurvatureCoefficient,
  getGPUTerrainDerivativesParameterValues,
  getGPUTerrainSightLineParameterValues,
  getGPUTerrainViewshedParameterValues,
  getGPUTerrainVisibilityToleranceParameterValues,
  GPU_POINT_HORIZON_PARAMETER_LENGTH,
  GPU_TERRAIN_DERIVATIVES_PARAMETER_LENGTH,
  GPU_TERRAIN_SIGHT_LINE_PARAMETER_LENGTH,
  GPU_TERRAIN_VIEWSHED_PARAMETER_LENGTH,
  GPU_TERRAIN_VISIBILITY,
  GPU_TERRAIN_VISIBILITY_TOLERANCE_PARAMETER_LENGTH,
  GPUPointHorizonProfile,
  GPUTerrainCumulativeViewshed,
  GPUTerrainDerivatives,
  GPUTerrainLineOfSight,
  GPUTerrainViewshed,
  type GPUTerrainSightLineTraversal,
  type GPUTerrainViewshedProps
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
import {MiniChart} from './point-pattern-chart';
import {
  formatCompiledGraphTiming,
  formatSpeedup,
  measureCompiledGraph,
  type CompiledGraphTiming
} from './vector-timing';

/** Compile-time number of skyline azimuth divisions (0.5 degrees). */
const AZIMUTH_COUNT = 720;
/** Compile-time ray length of the skyline profile, the upper bound of the radius slider. */
const HORIZON_MAXIMUM_DISTANCE = 12000;
/** Observers of the cumulative viewshed: the draggable one plus the highest separate summits. */
const CUMULATIVE_OBSERVER_COUNT = 6;
/** Minimum pixel distance between cumulative observers. */
const OBSERVER_SEPARATION_PIXELS = 45;
const DRAG_RADIUS_PIXELS = 20;
const READBACK_INTERVAL_FRAMES = 6;
/** Twin Peaks, the default observer when real data is loaded. */
const TWIN_PEAKS: readonly [number, number] = [-122.4475, 37.7544];
const SUN_AZIMUTH_DEGREES = 315;
const SUN_ALTITUDE_DEGREES = 40;
const VISIBILITY_NAMES: Record<number, string> = {
  [GPU_TERRAIN_VISIBILITY.hidden]: 'hidden',
  [GPU_TERRAIN_VISIBILITY.visible]: 'visible',
  [GPU_TERRAIN_VISIBILITY.outOfRange]: 'out of range',
  [GPU_TERRAIN_VISIBILITY.noData]: 'no data',
  [GPU_TERRAIN_VISIBILITY.marginal]: 'marginal'
};

type Traversal = GPUTerrainSightLineTraversal;
type Refraction = 'none' | 'mt-image' | 'gdal';
type PyramidDisplay = 'off' | `${number}`;

const REFRACTION_COEFFICIENTS: Record<Refraction, number | null> = {
  none: null,
  'mt-image': 0.13,
  gdal: 1 / 7
};

/** The prebuilt pyramid form every terrain contributor accepts as `pyramid`. */
type SharedPyramid = NonNullable<GPUTerrainViewshedProps['pyramid']>;

/** Output buffers and compiled graphs of one traversal. */
type TraversalGraphs = {
  traversal: Traversal;
  main: CompiledGPUCommandGraph<void>;
  cumulative: CompiledGPUCommandGraph<void>;
  visibility: Buffer;
  skylineAngle: Buffer;
  skylineDistance: Buffer;
  samples: Buffer;
  lineOfSightCode: Buffer;
  lineOfSightClearance: Buffer;
  cumulativeCount: Buffer;
};

export const visibilityMode: SpatialAnalysisModeDefinition = {
  id: 'visibility',
  title: 'Visibility',
  contributors: [
    'GPUTerrainViewshed',
    'GPUPointHorizonProfile',
    'GPUTerrainLineOfSight',
    'GPUTerrainCumulativeViewshed',
    'GPURasterExtremaPyramid'
  ],
  description:
    'Viewshed, 360° skyline, line of sight and a cumulative viewshed over one elevation raster. ' +
    'Drag the red observer or the cyan target (click the map to place the target). Tolerance, ' +
    'heights, radius and refraction are per-frame parameters; march vs pyramid traversal are two ' +
    'precompiled graphs proven identical on the GPU.',
  initialViewState: {longitude: -122.44, latitude: 37.755, zoom: 11.8},

  async create(context) {
    const terrain = await context.data.getSanFranciscoTerrain();
    context.signal.throwIfAborted();
    const {device} = context;
    const {width, height, bounds, cellSize} = terrain;
    const pixelCount = width * height;
    const projection = new LocalMetricProjection(terrain.origin);
    const origin: [number, number, number] = [terrain.origin[0], terrain.origin[1], 0];
    const resources = new SpatialAnalysisResources(device, 'visibility');
    const rayCount = AZIMUTH_COUNT;

    // --- Terrain buffers -----------------------------------------------------------------------
    // Sea (elevation 0) is invalid so every contributor leaves it out.
    const validityValues = new Uint32Array(pixelCount);
    for (let index = 0; index < pixelCount; index++) {
      validityValues[index] = terrain.elevation[index] > 0.5 ? 1 : 0;
    }
    const elevationBuffer = resources.createBuffer('elevation', terrain.elevation);
    const validityBuffer = resources.createBuffer('validity', validityValues);
    const hillshadeBuffer = resources.createBuffer('hillshade', pixelCount * 4);
    const derivativesSettings = resources.createParameterBuffer(
      'derivatives-settings',
      'float32',
      GPU_TERRAIN_DERIVATIVES_PARAMETER_LENGTH
    );

    // --- Setup graph: hillshade and the extrema pyramid, run once ------------------------------
    const setupGraph = new GPUCommandGraph<void>(device, {id: 'visibility-setup'});
    const setupElevation = {
      id: 'elevation',
      format: 'float32' as const,
      storage: {
        kind: 'buffer' as const,
        values: importGraphBuffer(setupGraph, 'elevation', elevationBuffer, 'float32', pixelCount)
      },
      validity: importGraphBuffer(setupGraph, 'validity', validityBuffer, 'uint32', pixelCount)
    };
    const pyramidLayout = getGPURasterExtremaPyramidLayout(width, height, {
      firstBlockSize: 4,
      footprint: 'bilinear'
    });
    // ONE combined buffer, maxima levels then minima levels: built once by the setup graph and
    // read by every pyramid-traversal contributor below through their `pyramid` prop.
    const pyramidCombined = resources.createBuffer('pyramid-combined', pyramidLayout.length * 8);
    setupGraph.add(
      new GPUTerrainDerivatives({
        id: 'derivatives',
        width,
        height,
        elevation: setupElevation,
        settings: derivativesSettings.importToGraph(setupGraph),
        hillshade: importGraphBuffer(
          setupGraph,
          'hillshade',
          hillshadeBuffer,
          'float32',
          pixelCount
        ),
        cellSizeMode: 'uniform',
        rowDirection: 'south'
      })
    );
    setupGraph.add(
      new GPURasterExtremaPyramid({
        id: 'extrema-pyramid',
        width,
        height,
        input: setupElevation,
        firstBlockSize: 4,
        footprint: 'bilinear',
        combined: importGraphBuffer(
          setupGraph,
          'pyramid-combined',
          pyramidCombined,
          'float32',
          2 * pyramidLayout.length
        )
      })
    );
    const compiledSetup = resources.track(setupGraph.compile());
    derivativesSettings.write(
      getGPUTerrainDerivativesParameterValues({
        cellSize,
        azimuthDegrees: SUN_AZIMUTH_DEGREES,
        altitudeDegrees: SUN_ALTITUDE_DEGREES
      })
    );
    // One coarse maximum raster per pyramid level, copied once after the setup graph runs.
    const levelBuffers = pyramidLayout.levels.map(level =>
      resources.createBuffer(`pyramid-level-${level.level}`, level.width * level.height * 4)
    );

    // --- Per-frame parameter buffers shared by both traversals ---------------------------------
    const viewshedSettings = resources.createParameterBuffer(
      'viewshed-settings',
      'float32',
      GPU_TERRAIN_VIEWSHED_PARAMETER_LENGTH
    );
    const toleranceSettings = resources.createParameterBuffer(
      'tolerance-settings',
      'float32',
      GPU_TERRAIN_VISIBILITY_TOLERANCE_PARAMETER_LENGTH
    );
    const horizonSettings = resources.createParameterBuffer(
      'horizon-settings',
      'float32',
      GPU_POINT_HORIZON_PARAMETER_LENGTH
    );
    const sightSettings = resources.createParameterBuffer(
      'sight-settings',
      'float32',
      GPU_TERRAIN_SIGHT_LINE_PARAMETER_LENGTH
    );
    const horizonObserver = resources.createBuffer('horizon-observer', new Float32Array(4));
    const sightPair = resources.createBuffer('sight-pair', new Float32Array(4));
    const cumulativeObservers = resources.createBuffer(
      'cumulative-observers',
      new Float32Array(CUMULATIVE_OBSERVER_COUNT * 2)
    );

    // --- One pair of graphs per traversal, with its own outputs --------------------------------
    function buildTraversalGraphs(traversal: Traversal): TraversalGraphs {
      // The pyramid traversal reads the setup graph's pyramid instead of building five of its own.
      const sharedPyramids = new Map<GPUCommandGraph<void>, SharedPyramid>();
      const getSharedPyramid = (graph: GPUCommandGraph<void>) => {
        if (traversal !== 'pyramid') return undefined;
        let shared = sharedPyramids.get(graph);
        if (!shared) {
          shared = {
            layout: pyramidLayout,
            combined: importGraphBuffer(
              graph,
              'pyramid-combined',
              pyramidCombined,
              'float32',
              2 * pyramidLayout.length
            )
          };
          sharedPyramids.set(graph, shared);
        }
        return shared;
      };
      const visibility = resources.createBuffer(`${traversal}-visibility`, pixelCount * 4);
      const skylineAngle = resources.createBuffer(`${traversal}-skyline-angle`, rayCount * 4);
      const skylineDistance = resources.createBuffer(`${traversal}-skyline-distance`, rayCount * 4);
      const samples = resources.createBuffer(`${traversal}-samples`, rayCount * 4);
      const lineOfSightCode = resources.createBuffer(`${traversal}-los-code`, 4);
      const lineOfSightClearance = resources.createBuffer(`${traversal}-los-clearance`, 4);
      const cumulativeCount = resources.createBuffer(`${traversal}-cumulative`, pixelCount * 4);

      const main = new GPUCommandGraph<void>(device, {id: `visibility-${traversal}`});
      const elevation = {
        id: 'elevation',
        format: 'float32' as const,
        storage: {
          kind: 'buffer' as const,
          values: importGraphBuffer(main, 'elevation', elevationBuffer, 'float32', pixelCount)
        },
        validity: importGraphBuffer(main, 'validity', validityBuffer, 'uint32', pixelCount)
      };
      main.add(
        new GPUTerrainViewshed({
          id: 'viewshed',
          width,
          height,
          elevation,
          traversal,
          pyramid: getSharedPyramid(main),
          settings: viewshedSettings.importToGraph(main),
          tolerance: toleranceSettings.importToGraph(main),
          visibility: importGraphBuffer(main, 'visibility', visibility, 'uint32', pixelCount)
        })
      );
      main.add(
        new GPUPointHorizonProfile({
          id: 'horizon',
          width,
          height,
          elevation,
          traversal,
          pyramid: getSharedPyramid(main),
          azimuthCount: AZIMUTH_COUNT,
          maximumDistance: HORIZON_MAXIMUM_DISTANCE,
          cellSize: cellSize[0],
          observers: importGraphBuffer(main, 'horizon-observer', horizonObserver, 'float32x4', 1),
          settings: horizonSettings.importToGraph(main),
          skylineAngle: importGraphBuffer(main, 'skyline-angle', skylineAngle, 'float32', rayCount),
          distance: importGraphBuffer(
            main,
            'skyline-distance',
            skylineDistance,
            'float32',
            rayCount
          ),
          samples: importGraphBuffer(main, 'skyline-samples', samples, 'uint32', rayCount)
        })
      );
      main.add(
        new GPUTerrainLineOfSight({
          id: 'line-of-sight',
          width,
          height,
          elevation,
          traversal,
          pyramid: getSharedPyramid(main),
          pairs: importGraphBuffer(main, 'sight-pair', sightPair, 'float32x4', 1),
          settings: sightSettings.importToGraph(main),
          visibility: importGraphBuffer(main, 'los-code', lineOfSightCode, 'uint32', 1),
          clearance: importGraphBuffer(main, 'los-clearance', lineOfSightClearance, 'float32', 1)
        })
      );

      const cumulative = new GPUCommandGraph<void>(device, {
        id: `visibility-cumulative-${traversal}`
      });
      const cumulativeElevation = {
        id: 'elevation',
        format: 'float32' as const,
        storage: {
          kind: 'buffer' as const,
          values: importGraphBuffer(cumulative, 'elevation', elevationBuffer, 'float32', pixelCount)
        },
        validity: importGraphBuffer(cumulative, 'validity', validityBuffer, 'uint32', pixelCount)
      };
      cumulative.add(
        new GPUTerrainCumulativeViewshed({
          id: 'cumulative',
          width,
          height,
          elevation: cumulativeElevation,
          traversal,
          pyramid: getSharedPyramid(cumulative),
          observers: importGraphBuffer(
            cumulative,
            'cumulative-observers',
            cumulativeObservers,
            'float32x2',
            CUMULATIVE_OBSERVER_COUNT
          ),
          settings: sightSettings.importToGraph(cumulative),
          visibleCount: importGraphBuffer(
            cumulative,
            'cumulative-count',
            cumulativeCount,
            'uint32',
            pixelCount
          )
        })
      );
      return {
        traversal,
        main: resources.track(main.compile()),
        cumulative: resources.track(cumulative.compile()),
        visibility,
        skylineAngle,
        skylineDistance,
        samples,
        lineOfSightCode,
        lineOfSightClearance,
        cumulativeCount
      };
    }
    const marchGraphs = buildTraversalGraphs('march');
    const pyramidGraphs = buildTraversalGraphs('pyramid');
    const graphsByTraversal: Record<Traversal, TraversalGraphs> = {
      march: marchGraphs,
      pyramid: pyramidGraphs
    };

    // --- Overlay buffers -----------------------------------------------------------------------
    const observerMarker = resources.createBuffer('observer-marker', new Float32Array(2));
    const targetMarker = resources.createBuffer('target-marker', new Float32Array(2));
    const extraObserverMarkers = resources.createBuffer(
      'extra-observers',
      new Float32Array((CUMULATIVE_OBSERVER_COUNT - 1) * 2)
    );
    const sightSegment = resources.createBuffer('sight-segment', new Float32Array(4));
    const skylinePoints = resources.createBuffer('skyline-points', rayCount * 8);
    // angles, distances, samples, then the sight-line code and clearance.
    const readbackWords = rayCount * 3 + 2;
    const readbackRing = resources.track(
      new GPUReadbackRing(device, {id: 'visibility-readback', byteLength: readbackWords * 4})
    );

    // --- State ---------------------------------------------------------------------------------
    let traversal: Traversal = 'pyramid';
    let showCumulative = false;
    let showSkyline = true;
    let pyramidDisplay: PyramidDisplay = 'off';
    let observerHeight = 10;
    let targetHeight = 2;
    let maximumDistance = 8000;
    let toleranceMeters = 1;
    let tolerancePerKilometer = 0.5;
    let refraction: Refraction = 'mt-image';
    let observer = getDefaultObserver();
    const extraObservers = findSummits(observer);
    let target = getDefaultTarget();
    let dragging: 'observer' | 'target' | null = null;
    let destroyed = false;
    let readbackPending = false;
    let cumulativeDirty = true;
    let lineOfSightCode = -1;
    let skylineReady = false;
    let verifyTimer: ReturnType<typeof setTimeout> | undefined;
    let verifying = false;
    let verifyAgain = false;

    function getCurvature(): number {
      const coefficient = REFRACTION_COEFFICIENTS[refraction];
      return coefficient === null ? 0 : getGPUTerrainCurvatureCoefficient(coefficient);
    }

    function pixelToMeters(point: readonly [number, number]): [number, number] {
      return [
        bounds[0] + (point[0] + 0.5) * cellSize[0],
        bounds[3] - (point[1] + 0.5) * cellSize[1]
      ];
    }

    function clampPixel(column: number, row: number): [number, number] {
      return [Math.min(Math.max(column, 0), width - 1), Math.min(Math.max(row, 0), height - 1)];
    }

    function metersToPixel(x: number, y: number): [number, number] {
      return clampPixel((x - bounds[0]) / cellSize[0] - 0.5, (bounds[3] - y) / cellSize[1] - 0.5);
    }

    function getDefaultObserver(): [number, number] {
      if (terrain.source === 'synthetic') {
        let best = 0;
        for (let index = 1; index < pixelCount; index++) {
          if (terrain.elevation[index] > terrain.elevation[best]) best = index;
        }
        return [best % width, Math.floor(best / width)];
      }
      const [x, y] = projection.project(TWIN_PEAKS[0], TWIN_PEAKS[1]);
      return metersToPixel(x, y);
    }

    function getDefaultTarget(): [number, number] {
      // A point a few kilometers away; a dry land cell if one lies nearby.
      const wanted = clampPixel(observer[0] + 70, observer[1] - 45);
      let best: [number, number] = wanted;
      let bestDistance = Infinity;
      for (let row = 0; row < height; row += 2) {
        for (let column = 0; column < width; column += 2) {
          if (validityValues[row * width + column] === 0) continue;
          const distance = Math.hypot(column - wanted[0], row - wanted[1]);
          if (distance < bestDistance) {
            bestDistance = distance;
            best = [column, row];
          }
        }
      }
      return best;
    }

    /** Highest valid cells at least {@link OBSERVER_SEPARATION_PIXELS} from each other and `main`. */
    function findSummits(main: readonly [number, number]): [number, number][] {
      const chosen: [number, number][] = [];
      const taken: [number, number][] = [[main[0], main[1]]];
      while (chosen.length < CUMULATIVE_OBSERVER_COUNT - 1) {
        let best = -1;
        for (let index = 0; index < pixelCount; index++) {
          if (validityValues[index] === 0) continue;
          if (best >= 0 && terrain.elevation[index] <= terrain.elevation[best]) continue;
          const column = index % width;
          const row = Math.floor(index / width);
          if (
            taken.some(
              point => Math.hypot(point[0] - column, point[1] - row) < OBSERVER_SEPARATION_PIXELS
            )
          ) {
            continue;
          }
          best = index;
        }
        if (best < 0) {
          // Fewer separated summits than requested: reuse the main observer position.
          chosen.push([main[0], main[1]]);
          continue;
        }
        const point: [number, number] = [best % width, Math.floor(best / width)];
        chosen.push(point);
        taken.push(point);
      }
      return chosen;
    }

    /** Bilinear ground elevation in meters at a pixel-center position, NaN near invalid cells. */
    function sampleElevation(column: number, row: number): number {
      const baseColumn = Math.min(Math.floor(column), width - 2);
      const baseRow = Math.min(Math.floor(row), height - 2);
      const fractionX = column - baseColumn;
      const fractionY = row - baseRow;
      let value = 0;
      for (let dy = 0; dy < 2; dy++) {
        for (let dx = 0; dx < 2; dx++) {
          const index = (baseRow + dy) * width + baseColumn + dx;
          if (validityValues[index] === 0) return NaN;
          value +=
            terrain.elevation[index] *
            (dx ? fractionX : 1 - fractionX) *
            (dy ? fractionY : 1 - fractionY);
        }
      }
      return value;
    }

    function writeParameters(): void {
      const curvatureCoefficient = getCurvature();
      viewshedSettings.write(
        getGPUTerrainViewshedParameterValues({
          observer,
          observerHeight,
          targetHeight,
          maxDistance: maximumDistance,
          cellSize,
          curvatureCoefficient
        })
      );
      toleranceSettings.write(
        getGPUTerrainVisibilityToleranceParameterValues({toleranceMeters, tolerancePerKilometer})
      );
      horizonSettings.write(
        getGPUPointHorizonParameterValues({
          curvatureCoefficient,
          maximumDistance,
          cellSize
        })
      );
      sightSettings.write(
        getGPUTerrainSightLineParameterValues({
          observerHeight,
          targetHeight,
          maxDistance: maximumDistance,
          cellSize,
          curvatureCoefficient,
          toleranceMeters,
          tolerancePerKilometer
        })
      );
      horizonObserver.write(Float32Array.of(observer[0], observer[1], observerHeight, 0));
      sightPair.write(Float32Array.of(observer[0], observer[1], target[0], target[1]));
      cumulativeObservers.write(
        Float32Array.of(observer[0], observer[1], ...extraObservers.flat())
      );
      const observerMeters = pixelToMeters(observer);
      const targetMeters = pixelToMeters(target);
      observerMarker.write(Float32Array.of(...observerMeters));
      targetMarker.write(Float32Array.of(...targetMeters));
      extraObserverMarkers.write(Float32Array.from(extraObservers.flatMap(pixelToMeters)));
      sightSegment.write(Float32Array.of(...observerMeters, ...targetMeters));
      cumulativeDirty = true;
      scheduleVerification();
    }

    writeParameters();

    // --- Controls ------------------------------------------------------------------------------
    context.controls.addSelect<Traversal>({
      label: 'Traversal (compile-time, both prebuilt)',
      options: [
        {value: 'pyramid', label: 'Pyramid (min-max skip)'},
        {value: 'march', label: 'March (every sample)'}
      ],
      value: traversal,
      onChange: value => {
        traversal = value;
        cumulativeDirty = true;
        context.updateLayers();
      }
    });
    context.controls.addToggle({
      label: 'Cumulative viewshed (6 observers)',
      value: showCumulative,
      onChange: value => {
        showCumulative = value;
        cumulativeDirty = true;
        context.updateLayers();
      }
    });
    context.controls.addToggle({
      label: 'Skyline points on the map',
      value: showSkyline,
      onChange: value => {
        showSkyline = value;
        context.updateLayers();
      }
    });
    context.controls.addSlider({
      label: 'Tolerance band (per-frame)',
      min: 0,
      max: 10,
      step: 0.5,
      value: toleranceMeters,
      format: value => `${value.toFixed(1)} m`,
      onChange: value => {
        toleranceMeters = value;
        writeParameters();
      }
    });
    context.controls.addSlider({
      label: 'Tolerance per km (per-frame)',
      min: 0,
      max: 5,
      step: 0.5,
      value: tolerancePerKilometer,
      format: value => `${value.toFixed(1)} m/km`,
      onChange: value => {
        tolerancePerKilometer = value;
        writeParameters();
      }
    });
    context.controls.addSlider({
      label: 'Observer height (per-frame)',
      min: 0,
      max: 200,
      step: 2,
      value: observerHeight,
      format: value => `${value} m`,
      onChange: value => {
        observerHeight = value;
        writeParameters();
      }
    });
    context.controls.addSlider({
      label: 'Target height (per-frame)',
      min: 0,
      max: 100,
      step: 1,
      value: targetHeight,
      format: value => `${value} m`,
      onChange: value => {
        targetHeight = value;
        writeParameters();
      }
    });
    context.controls.addSlider({
      label: 'Radius (per-frame)',
      min: 1000,
      max: HORIZON_MAXIMUM_DISTANCE,
      step: 500,
      value: maximumDistance,
      format: value => `${(value / 1000).toFixed(1)} km`,
      onChange: value => {
        maximumDistance = value;
        writeParameters();
      }
    });
    const dropReadout = context.controls.addReadout('Curvature drop at radius');
    function updateDropReadout(): void {
      const drop = getCurvature() * maximumDistance * maximumDistance;
      dropReadout.setValue(`${drop.toFixed(2)} m`);
    }
    context.controls.addSelect<Refraction>({
      label: 'Earth curvature and refraction k (per-frame)',
      options: [
        {value: 'mt-image', label: 'k = 0.13 (mt-image, geodetic)'},
        {value: 'gdal', label: 'k = 1/7 (GDAL -cc 0.85714)'},
        {value: 'none', label: 'Flat earth'}
      ],
      value: refraction,
      onChange: value => {
        refraction = value;
        updateDropReadout();
        writeParameters();
      }
    });
    updateDropReadout();
    context.controls.addSelect<PyramidDisplay>({
      label: 'Extrema pyramid, maximum per cell',
      options: [
        {value: 'off', label: 'Off'},
        ...pyramidLayout.levels.map(level => ({
          value: `${level.level}` as PyramidDisplay,
          label: `Level ${level.level}: ${level.blockSize} px blocks (${level.width} × ${level.height})`
        }))
      ],
      value: pyramidDisplay,
      onChange: value => {
        pyramidDisplay = value;
        context.updateLayers();
      }
    });
    context.controls.addLegend({
      title: 'Viewshed from the observer',
      entries: [
        {color: [60, 230, 110, 150], label: 'Visible'},
        {color: [255, 190, 60, 190], label: 'Marginal'},
        {color: [10, 10, 30, 110], label: 'Hidden'}
      ]
    });
    context.controls.addLegend({
      title: 'Cumulative count (1 to 6 observers) and skyline angle',
      gradient: {
        colors: [
          [68, 1, 84],
          [33, 145, 140],
          [253, 231, 37]
        ],
        minimumLabel: '1 observer',
        maximumLabel: '6 observers'
      }
    });
    context.controls.addNote(
      'Drag the red observer or the cyan target, or click the map to place the target. ' +
        'Marginal cells are inside the tolerance band of the sight line.'
    );

    const skylineChart = new MiniChart('Skyline angle', '° az');
    skylineChart.insertBefore(document.querySelector('[data-mode-readouts]'));

    const equalityReadout = context.controls.addReadout('Pyramid = march', 'checking...');
    const equalityCellsReadout = context.controls.addReadout('  cells differing');
    const equalitySkylineReadout = context.controls.addReadout('  skyline differing');
    const equalityLineOfSightReadout = context.controls.addReadout('  sight line');
    const sampleReadout = context.controls.addReadout('  samples march/pyr');
    const skylineReadout = context.controls.addReadout('Skyline highest');
    const lineOfSightReadout = context.controls.addReadout('Line of sight');
    const pyramidReadout = context.controls.addReadout('Extrema pyramid');
    const marchTimingReadout = context.controls.addReadout('Graph, march');
    const pyramidTimingReadout = context.controls.addReadout('Graph, pyramid');
    const cumulativeMarchReadout = context.controls.addReadout('Cumulative, march');
    const cumulativePyramidReadout = context.controls.addReadout('Cumulative, pyramid');
    context.controls.addReadout('Raster', `${width} × ${height} cells`);
    context.controls.addReadout(
      'Cell size',
      `${cellSize[0].toFixed(1)} × ${cellSize[1].toFixed(1)} m`
    );
    context.controls.addReadout('Data', terrain.attribution);

    // --- Setup run: hillshade, pyramid, level copies -------------------------------------------
    {
      const encoder = device.createCommandEncoder({id: 'visibility-setup-encoder'});
      compiledSetup.encode(encoder, {parameters: undefined});
      pyramidLayout.levels.forEach((level, index) => {
        encoder.copyBufferToBuffer({
          sourceBuffer: pyramidCombined,
          sourceOffset: level.offset * 4,
          destinationBuffer: levelBuffers[index],
          destinationOffset: 0,
          size: level.width * level.height * 4
        });
      });
      device.submit(encoder.finish());
    }
    void readPyramidRoot();

    async function readPyramidRoot(): Promise<void> {
      try {
        const root = pyramidLayout.levels[pyramidLayout.levels.length - 1];
        const [maximumBytes, minimumBytes] = await Promise.all([
          pyramidCombined.readAsync(root.offset * 4, 4),
          pyramidCombined.readAsync((pyramidLayout.length + root.offset) * 4, 4)
        ]);
        if (destroyed) return;
        const maximum = new Float32Array(maximumBytes.slice().buffer)[0];
        const minimum = new Float32Array(minimumBytes.slice().buffer)[0];
        pyramidReadout.setValue(
          `${pyramidLayout.levels.length} levels, ${formatCount(pyramidLayout.length)} cells · ` +
            `root ${minimum.toFixed(0)} to ${maximum.toFixed(0)} m`
        );
      } catch {
        // Destroyed while reading.
      }
    }

    // --- Verification: both traversals into their own buffers, compared bit for bit ------------
    function scheduleVerification(): void {
      if (destroyed) return;
      clearTimeout(verifyTimer);
      verifyTimer = setTimeout(() => void verifyTraversals(), 700);
    }

    async function readWords(buffer: Buffer, wordCount: number): Promise<Uint32Array> {
      const bytes = await buffer.readAsync(0, wordCount * 4);
      return new Uint32Array(bytes.slice().buffer);
    }

    function countDifferences(left: Uint32Array, right: Uint32Array): number {
      let count = 0;
      for (let index = 0; index < left.length; index++) {
        if (left[index] !== right[index]) count++;
      }
      return count;
    }

    async function verifyTraversals(): Promise<void> {
      if (destroyed) return;
      if (verifying) {
        verifyAgain = true;
        return;
      }
      verifying = true;
      try {
        const encoder = device.createCommandEncoder({id: 'visibility-verify-encoder'});
        marchGraphs.main.encode(encoder, {parameters: undefined});
        pyramidGraphs.main.encode(encoder, {parameters: undefined});
        device.submit(encoder.finish());
        const [
          marchCells,
          pyramidCells,
          marchSkyline,
          pyramidSkyline,
          marchDistance,
          pyramidDistance,
          marchSamples,
          pyramidSamples,
          marchSight,
          pyramidSight
        ] = await Promise.all([
          readWords(marchGraphs.visibility, pixelCount),
          readWords(pyramidGraphs.visibility, pixelCount),
          readWords(marchGraphs.skylineAngle, rayCount),
          readWords(pyramidGraphs.skylineAngle, rayCount),
          readWords(marchGraphs.skylineDistance, rayCount),
          readWords(pyramidGraphs.skylineDistance, rayCount),
          readWords(marchGraphs.samples, rayCount),
          readWords(pyramidGraphs.samples, rayCount),
          Promise.all([
            readWords(marchGraphs.lineOfSightCode, 1),
            readWords(marchGraphs.lineOfSightClearance, 1)
          ]),
          Promise.all([
            readWords(pyramidGraphs.lineOfSightCode, 1),
            readWords(pyramidGraphs.lineOfSightClearance, 1)
          ])
        ]);
        if (destroyed) return;
        const cellDifferences = countDifferences(marchCells, pyramidCells);
        const skylineDifferences =
          countDifferences(marchSkyline, pyramidSkyline) +
          countDifferences(marchDistance, pyramidDistance);
        const sightDifferences =
          countDifferences(marchSight[0], pyramidSight[0]) +
          countDifferences(marchSight[1], pyramidSight[1]);
        let marchTotal = 0;
        let pyramidTotal = 0;
        for (let index = 0; index < rayCount; index++) {
          marchTotal += marchSamples[index];
          pyramidTotal += pyramidSamples[index];
        }
        const identical = cellDifferences + skylineDifferences + sightDifferences === 0;
        equalityReadout.setValue(identical ? 'identical (bit for bit)' : 'MISMATCH');
        equalityCellsReadout.setValue(
          `${formatCount(cellDifferences)} of ${formatCount(pixelCount)}`
        );
        equalitySkylineReadout.setValue(
          `${formatCount(skylineDifferences)} of ${formatCount(rayCount * 2)} values`
        );
        equalityLineOfSightReadout.setValue(
          sightDifferences === 0 ? 'identical' : `${sightDifferences} words differ`
        );
        sampleReadout.setValue(
          marchTotal > 0
            ? `${formatCount(marchTotal)} / ${formatCount(pyramidTotal)} ` +
                `(${((1 - pyramidTotal / marchTotal) * 100).toFixed(0)}% skipped)`
            : 'no valid observer'
        );
      } catch (error) {
        if (!destroyed) equalityReadout.setValue(`failed: ${(error as Error).message}`);
      } finally {
        verifying = false;
        if (verifyAgain && !destroyed) {
          verifyAgain = false;
          scheduleVerification();
        }
      }
    }
    context.controls.addButton({
      label: 'Verify pyramid = march',
      onClick: () => void verifyTraversals()
    });

    // --- Measure -------------------------------------------------------------------------------
    async function measureAll(): Promise<void> {
      try {
        const options = {
          parameters: undefined,
          completionBuffer: marchGraphs.visibility,
          signal: context.signal
        };
        const marchTiming: CompiledGraphTiming = await measureCompiledGraph(
          device,
          marchGraphs.main,
          options
        );
        const pyramidTiming: CompiledGraphTiming = await measureCompiledGraph(
          device,
          pyramidGraphs.main,
          options
        );
        marchTimingReadout.setValue(
          `${marchGraphs.main.stats.nodeOrder.length} nodes · ${formatCompiledGraphTiming(marchTiming)}`
        );
        pyramidTimingReadout.setValue(
          `${pyramidGraphs.main.stats.nodeOrder.length} nodes · ${formatCompiledGraphTiming(pyramidTiming)} · ` +
            formatSpeedup(marchTiming.milliseconds, pyramidTiming.milliseconds)
        );
        const light = {...options, runs: 3, warmUpRuns: 1, repetitions: 2};
        const cumulativeMarchTiming: CompiledGraphTiming = await measureCompiledGraph(
          device,
          marchGraphs.cumulative,
          light
        );
        cumulativeMarchReadout.setValue(
          `${marchGraphs.cumulative.stats.nodeOrder.length} nodes · ${formatCompiledGraphTiming(cumulativeMarchTiming)}`
        );
        const cumulativePyramidTiming: CompiledGraphTiming = await measureCompiledGraph(
          device,
          pyramidGraphs.cumulative,
          light
        );
        cumulativePyramidReadout.setValue(
          `${pyramidGraphs.cumulative.stats.nodeOrder.length} nodes · ${formatCompiledGraphTiming(cumulativePyramidTiming)} · ` +
            formatSpeedup(cumulativeMarchTiming.milliseconds, cumulativePyramidTiming.milliseconds)
        );
      } catch (error) {
        if (!destroyed) marchTimingReadout.setValue(`failed: ${(error as Error).message}`);
      }
    }
    context.controls.addButton({label: 'Measure GPU cost', onClick: () => void measureAll()});
    marchTimingReadout.setValue('measuring...');
    // Measure once the first frames have run, outside Deck's frame encoder.
    const measureTimer = setTimeout(() => {
      if (!destroyed) void measureAll();
    }, 2500);

    // --- Readback of skyline and sight line ----------------------------------------------------
    async function readSkyline(commandEncoder: CommandEncoder): Promise<void> {
      const ticket = readbackRing.tryAcquire();
      if (!ticket) return;
      const active = graphsByTraversal[traversal];
      let offset = 0;
      for (const [buffer, byteLength] of [
        [active.skylineAngle, rayCount * 4],
        [active.skylineDistance, rayCount * 4],
        [active.samples, rayCount * 4],
        [active.lineOfSightCode, 4],
        [active.lineOfSightClearance, 4]
      ] as const) {
        commandEncoder.copyBufferToBuffer({
          sourceBuffer: buffer,
          sourceOffset: 0,
          destinationBuffer: ticket.buffer,
          destinationOffset: offset,
          size: byteLength
        });
        offset += byteLength;
      }
      ticket.markEncoded({byteOffset: 0, byteLength: offset});
      readbackPending = true;
      try {
        const bytes = await ticket.read();
        if (destroyed) return;
        const words = new Uint32Array(bytes.slice(0, offset).buffer);
        applySkyline(new Float32Array(words.buffer), words);
      } catch {
        // The ring or device was destroyed while the read was in flight.
      } finally {
        readbackPending = false;
      }
    }

    function applySkyline(floats: Float32Array, words: Uint32Array): void {
      const angles = floats.subarray(0, rayCount);
      const distances = floats.subarray(rayCount, rayCount * 2);
      const observerMeters = pixelToMeters(observer);
      const azimuths = new Float32Array(rayCount);
      const chartAngles = new Float32Array(rayCount);
      let peak = -Infinity;
      let peakAzimuth = 0;
      const points: [number, number][] = [];
      for (let ray = 0; ray < rayCount; ray++) {
        const azimuth = (ray * 360) / rayCount;
        const radians = (azimuth * Math.PI) / 180;
        azimuths[ray] = azimuth;
        const angle = angles[ray];
        const valid = Number.isFinite(angle) && angle > -89.5 && distances[ray] > 0;
        chartAngles[ray] = valid ? angle : NaN;
        points.push(
          valid
            ? [
                observerMeters[0] + distances[ray] * Math.sin(radians),
                observerMeters[1] + distances[ray] * Math.cos(radians)
              ]
            : [observerMeters[0], observerMeters[1]]
        );
        if (valid && angle > peak) {
          peak = angle;
          peakAzimuth = azimuth;
        }
      }
      skylinePoints.write(Float32Array.from(points.flat()));
      skylineReady = true;

      // The target as seen from the observer, on the same angular axis as the skyline.
      const eye = sampleElevation(observer[0], observer[1]) + observerHeight;
      const targetElevation = sampleElevation(target[0], target[1]) + targetHeight;
      const deltaX = (target[0] - observer[0]) * cellSize[0];
      const deltaY = (target[1] - observer[1]) * cellSize[1];
      const distance = Math.hypot(deltaX, deltaY);
      const targetAzimuth = ((Math.atan2(deltaX, -deltaY) * 180) / Math.PI + 360) % 360;
      const targetAngle =
        (Math.atan((targetElevation - eye) / distance - getCurvature() * distance) * 180) / Math.PI;
      skylineChart.update({
        series: [
          {kind: 'line', x: azimuths, y: chartAngles, color: '#f1c96b'},
          {
            kind: 'points',
            x: [targetAzimuth],
            y: [targetAngle],
            color: '#4ee0ff',
            radius: 4
          }
        ],
        referenceLines: [{y: 0, color: '#4d6aa8', dashed: true}],
        xRange: [0, 360],
        caption: Number.isFinite(peak)
          ? `peak ${peak.toFixed(1)}° at ${peakAzimuth.toFixed(0)}°`
          : ''
      });
      skylineReadout.setValue(
        Number.isFinite(peak)
          ? `${peak.toFixed(2)}° at azimuth ${peakAzimuth.toFixed(1)}°`
          : 'no data (observer on sea or outside)'
      );

      const code = words[rayCount * 3];
      const clearance = new Float32Array(words.buffer, (rayCount * 3 + 1) * 4, 1)[0];
      const name = VISIBILITY_NAMES[code] ?? `code ${code}`;
      const showClearance =
        code === GPU_TERRAIN_VISIBILITY.visible ||
        code === GPU_TERRAIN_VISIBILITY.hidden ||
        code === GPU_TERRAIN_VISIBILITY.marginal;
      const sightAbove = targetAngle >= (Number.isFinite(peak) ? peak : -90);
      lineOfSightReadout.setValue(
        `${name} · ${(distance / 1000).toFixed(2)} km` +
          (showClearance && Number.isFinite(clearance) && Math.abs(clearance) < 1e30
            ? ` · clearance ${clearance >= 0 ? '+' : ''}${clearance.toFixed(1)} m`
            : '') +
          (Number.isFinite(peak) ? (sightAbove ? ' · above skyline' : ' · below skyline') : '')
      );
      if (code !== lineOfSightCode) {
        lineOfSightCode = code;
        context.updateLayers();
      }
    }

    // --- Pointer interaction -------------------------------------------------------------------
    function movePoint(which: 'observer' | 'target', event: SpatialAnalysisPointerEvent): void {
      if (!event.coordinate) return;
      const [x, y] = projection.project(event.coordinate[0], event.coordinate[1]);
      const point = metersToPixel(x, y);
      if (which === 'observer') observer = point;
      else target = point;
      writeParameters();
    }

    function getPointerDistance(
      point: readonly [number, number],
      event: SpatialAnalysisPointerEvent
    ): number {
      const viewport = context.getViewport();
      if (!viewport) return Infinity;
      const [x, y] = pixelToMeters(point);
      const [longitude, latitude] = projection.unproject(x, y);
      const [screenX, screenY] = viewport.project([longitude, latitude]);
      return Math.hypot(screenX - event.pixel[0], screenY - event.pixel[1]);
    }

    // --- Instance ------------------------------------------------------------------------------
    let frameCount = 0;
    const instance: SpatialAnalysisModeInstance = {
      getCompiledGraphs: () => [
        compiledSetup,
        marchGraphs.main,
        marchGraphs.cumulative,
        pyramidGraphs.main,
        pyramidGraphs.cumulative
      ],
      encode(commandEncoder, frame) {
        frameCount = frame.frameIndex;
        const active = graphsByTraversal[traversal];
        active.main.encode(commandEncoder, {parameters: undefined});
        if (showCumulative && cumulativeDirty) {
          active.cumulative.encode(commandEncoder, {parameters: undefined});
          cumulativeDirty = false;
        }
        if (!readbackPending && frameCount % READBACK_INTERVAL_FRAMES === 0) {
          void readSkyline(commandEncoder);
        }
      },
      getLayers() {
        const active = graphsByTraversal[traversal];
        const layers: Layer[] = [];
        const rasterProps = {
          coordinateOrigin: origin,
          gridSize: [width, height] as const,
          bounds,
          rowOrigin: 'north' as const
        };
        layers.push(
          new SpatialAnalysisRasterLayer({
            ...rasterProps,
            id: 'visibility-hillshade',
            values: hillshadeBuffer,
            valueFormat: 'float32',
            colormap: 'grayscale',
            valueRange: [0, 1],
            color: [255, 255, 255, 150]
          })
        );
        if (pyramidDisplay !== 'off') {
          const level = pyramidLayout.levels[Number(pyramidDisplay)];
          const blockX = level.blockSize * cellSize[0];
          const blockY = level.blockSize * cellSize[1];
          layers.push(
            new SpatialAnalysisRasterLayer({
              coordinateOrigin: origin,
              id: `visibility-pyramid-${level.level}`,
              gridSize: [level.width, level.height],
              // Cells start at the raster's north-west corner and may overhang the south-east edge.
              bounds: [
                bounds[0],
                bounds[3] - level.height * blockY,
                bounds[0] + level.width * blockX,
                bounds[3]
              ],
              rowOrigin: 'north',
              values: levelBuffers[level.level],
              valueFormat: 'float32',
              colormap: 'viridis',
              valueRange: [terrain.elevationRange[0], terrain.elevationRange[1]],
              discardAtOrBelow: -1e30,
              color: [255, 255, 255, 200]
            })
          );
        } else if (showCumulative) {
          layers.push(
            new SpatialAnalysisRasterLayer({
              ...rasterProps,
              id: 'visibility-cumulative',
              values: active.cumulativeCount,
              valueFormat: 'uint32',
              colormap: 'viridis',
              valueRange: [1, CUMULATIVE_OBSERVER_COUNT],
              discardAtOrBelow: 0.5,
              color: [255, 255, 255, 170]
            })
          );
        } else {
          layers.push(
            new SpatialAnalysisRasterLayer({
              ...rasterProps,
              id: 'visibility-viewshed',
              values: active.visibility,
              valueFormat: 'uint32',
              colormap: 'category',
              // GPU_TERRAIN_VISIBILITY: hidden, visible, outOfRange, noData, marginal.
              palette: [
                [10, 10, 30, 70],
                [60, 230, 110, 100],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [255, 190, 60, 150]
              ]
            })
          );
        }
        if (showSkyline && skylineReady) {
          layers.push(
            new SpatialAnalysisPointLayer({
              id: 'visibility-skyline',
              coordinateOrigin: origin,
              positions: skylinePoints,
              instanceCount: rayCount,
              values: active.skylineAngle,
              valueFormat: 'float32',
              colormap: 'viridis',
              valueRange: [-3, 10],
              radiusPixels: 3
            })
          );
        }
        const sightColor: [number, number, number, number] =
          lineOfSightCode === GPU_TERRAIN_VISIBILITY.visible
            ? [60, 230, 110, 255]
            : lineOfSightCode === GPU_TERRAIN_VISIBILITY.marginal
              ? [255, 190, 60, 255]
              : [255, 80, 80, 255];
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'visibility-sight-line',
            coordinateOrigin: origin,
            segments: sightSegment,
            instanceCount: 1,
            widthPixels: 2.5,
            color: sightColor
          })
        );
        if (showCumulative) {
          layers.push(
            new SpatialAnalysisPointLayer({
              id: 'visibility-extra-observers',
              coordinateOrigin: origin,
              positions: extraObserverMarkers,
              instanceCount: CUMULATIVE_OBSERVER_COUNT - 1,
              radiusPixels: 5,
              color: [255, 160, 60, 255]
            })
          );
        }
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'visibility-target-halo',
            coordinateOrigin: origin,
            positions: targetMarker,
            instanceCount: 1,
            radiusPixels: 10,
            color: [255, 255, 255, 230]
          }),
          new SpatialAnalysisPointLayer({
            id: 'visibility-target',
            coordinateOrigin: origin,
            positions: targetMarker,
            instanceCount: 1,
            radiusPixels: 6,
            color: [40, 215, 255, 255]
          }),
          new SpatialAnalysisPointLayer({
            id: 'visibility-observer-halo',
            coordinateOrigin: origin,
            positions: observerMarker,
            instanceCount: 1,
            radiusPixels: 11,
            color: [255, 255, 255, 230]
          }),
          new SpatialAnalysisPointLayer({
            id: 'visibility-observer',
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
        movePoint('target', event);
        return true;
      },
      onDragStart(event) {
        if (!event.pixel) return false;
        const observerDistance = getPointerDistance(observer, event);
        const targetDistance = getPointerDistance(target, event);
        if (Math.min(observerDistance, targetDistance) > DRAG_RADIUS_PIXELS) return false;
        dragging = observerDistance <= targetDistance ? 'observer' : 'target';
        context.setMapDragEnabled(false);
        return true;
      },
      onDrag(event) {
        if (dragging) movePoint(dragging, event);
      },
      onDragEnd(event) {
        if (!dragging) return;
        movePoint(dragging, event);
        dragging = null;
        context.setMapDragEnabled(true);
      },
      destroy() {
        destroyed = true;
        clearTimeout(verifyTimer);
        clearTimeout(measureTimer);
        if (dragging) context.setMapDragEnabled(true);
        skylineChart.destroy();
        resources.destroy();
      }
    };
    return instance;
  }
};
