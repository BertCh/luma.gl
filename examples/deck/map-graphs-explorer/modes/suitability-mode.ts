// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Site suitability on the San Francisco terrain grid (512 x 512 cells, 256 x 256 synthetic), in
 * the style of kepler.gl or Studio map algebra, entirely on the GPU.
 *
 * Three criteria rasters (slope from `GPUTerrainDerivatives`, elevation, and distance to the
 * nearest bike parking from `GPUDistanceField`, computed once by a "prepare" graph) are each
 * reclassified into ratings 5 (best) to 1 by `GPURasterReclassify` with live break sliders; steep
 * slopes and far sites can be made restricted (NaN class value). The ratings are written into one
 * band-sequential stack that `GPUWeightedOverlay` combines with live weights; a second overlay
 * with equal weights feeds `GPURasterArithmetic` (weights effect), `GPURasterCellStatistics`
 * gives the limiting factor (minimum rating), `GPURasterConditional` masks cells below a live
 * score threshold, and a last `GPURasterReclassify` counts the suitable cells.
 *
 * A separate small graph runs on the pointer: `GPURasterProfile` samples elevation and the score
 * along a line drawn with two clicks (endpoints are draggable) and `GPURasterSampling` reports the
 * values at a probe point and drapes slope onto the bike-parking sites. Every slider, select and
 * toggle is a buffer write; the graphs are compiled once and encoded only when an input changed.
 */

import type {Layer} from '@deck.gl/core';
import type {Buffer, CommandEncoder} from '@luma.gl/core';
import {
  GPUCommandGraph,
  GPUReadbackRing,
  type CompiledGPUCommandGraph
} from '@luma.gl/gpgpu/gpu-core';
import {
  getGPUDistanceFieldParameterValues,
  getGPURasterArithmeticParameterValues,
  getGPURasterCellStatisticsParameterValues,
  getGPURasterConditionalParameterValues,
  getGPURasterProfileParameterValues,
  getGPURasterReclassifyParameterValues,
  getGPURasterSamplingParameterValues,
  getGPUTerrainDerivativesParameterValues,
  getGPUWeightedOverlayParameterLength,
  getGPUWeightedOverlayParameterValues,
  GPU_DISTANCE_FIELD_PARAMETER_LENGTH,
  GPU_RASTER_ARITHMETIC_PARAMETER_LENGTH,
  GPU_RASTER_CELL_STATISTICS_PARAMETER_LENGTH,
  GPU_RASTER_CONDITIONAL_PARAMETER_LENGTH,
  GPU_RASTER_PROFILE_PARAMETER_LENGTH,
  GPU_RASTER_RECLASSIFY_PARAMETER_LENGTH,
  GPU_RASTER_SAMPLING_PARAMETER_LENGTH,
  GPU_TERRAIN_DERIVATIVES_PARAMETER_LENGTH,
  GPUDistanceField,
  GPURasterArithmetic,
  GPURasterCellStatistics,
  GPURasterConditional,
  GPURasterProfile,
  GPURasterReclassify,
  GPURasterSampling,
  GPUTerrainDerivatives,
  GPUWeightedOverlay,
  importGraphBuffer,
  type GPURasterSamplingMethod
} from '@luma.gl/experimental/map-graphs';
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
import {createOffsetRasterLayerClass} from './suitability-layers';
import {
  formatCompiledGraphTiming,
  measureCompiledGraph,
  type CompiledGraphTiming
} from './vector-timing';

/** Criteria, in stack order. */
const CRITERIA = ['slope', 'elevation', 'distance'] as const;
type Criterion = (typeof CRITERIA)[number];
const CRITERION_COUNT = CRITERIA.length;
/** Break count per criterion; five classes with ratings 5 to 1. */
const BREAK_COUNT = 4;
const CLASS_COUNT = BREAK_COUNT + 1;
const RATINGS = [5, 4, 3, 2, 1];
/** Compile-time sample capacity of each profile. */
const PROFILE_CAPACITY = 2048;
const MINIMUM_SPACING = 15;
const GRAB_RADIUS_PIXELS = 22;

/** Slider ranges and defaults per criterion; breaks run linearly from `good` to `poor`. */
const CRITERION_INFO: Record<
  Criterion,
  {
    label: string;
    unit: string;
    goodRange: [number, number, number];
    poorRange: [number, number, number];
    step: number;
  }
> = {
  slope: {
    label: 'Slope',
    unit: '°',
    goodRange: [1, 15, 4],
    poorRange: [10, 45, 25],
    step: 1
  },
  elevation: {
    label: 'Elevation',
    unit: ' m',
    goodRange: [5, 150, 40],
    poorRange: [100, 450, 220],
    step: 5
  },
  distance: {
    label: 'Distance to bike parking',
    unit: ' m',
    goodRange: [50, 1000, 250],
    poorRange: [500, 5000, 2000],
    step: 50
  }
};

type Display =
  | 'suitable'
  | 'score'
  | 'rating-slope'
  | 'rating-elevation'
  | 'rating-distance'
  | 'limiting'
  | 'effect';
type Tool = 'profile' | 'probe';

/** Byte layout of the criteria summary readback. */
const SUMMARY_COUNTS_OFFSET = 0;
const SUMMARY_SCORE_RANGE_OFFSET = CRITERION_COUNT * CLASS_COUNT * 4;
const SUMMARY_SUITABLE_OFFSET = SUMMARY_SCORE_RANGE_OFFSET + 8;
const SUMMARY_BYTE_LENGTH = SUMMARY_SUITABLE_OFFSET + 8;
/** Byte layout of the path readback: scalars, then three sample columns. */
const PATH_SCALARS_OFFSET = 0;
const PATH_SCALAR_COUNT = 16;
const PATH_DISTANCES_OFFSET = PATH_SCALAR_COUNT * 4;
const PATH_ELEVATIONS_OFFSET = PATH_DISTANCES_OFFSET + PROFILE_CAPACITY * 4;
const PATH_SCORES_OFFSET = PATH_ELEVATIONS_OFFSET + PROFILE_CAPACITY * 4;
const PATH_BYTE_LENGTH = PATH_SCORES_OFFSET + PROFILE_CAPACITY * 4;

type PathReadback = {
  scalars: Float32Array;
  words: Uint32Array;
  distances: Float32Array;
  elevations: Float32Array;
  scores: Float32Array;
};

export const suitabilityMode: MapGraphsModeDefinition = {
  id: 'suitability',
  title: 'Suitability',
  recipes: [
    'GPURasterReclassify',
    'GPUWeightedOverlay',
    'GPURasterCellStatistics',
    'GPURasterConditional',
    'GPURasterArithmetic',
    'GPURasterSampling',
    'GPURasterProfile'
  ],
  description:
    'Weighted-overlay site suitability from slope, elevation and distance to bike parking: ' +
    'live class breaks and weights, a score threshold mask, then an elevation and score profile ' +
    'along a line you click (two clicks, endpoints draggable) and sampled values at a probe.',
  initialViewState: {longitude: -122.44, latitude: 37.735, zoom: 11.6},

  async create(context) {
    const [terrain, parking] = await Promise.all([
      context.data.getSanFranciscoTerrain(),
      context.data.getSanFranciscoBikeParking()
    ]);
    context.signal.throwIfAborted();
    const {device} = context;
    const {width, height, bounds, cellSize} = terrain;
    const cellCount = width * height;
    const projection = new LocalMetricProjection(terrain.origin);
    const origin: [number, number, number] = [terrain.origin[0], terrain.origin[1], 0];
    const resources = new MapGraphsResources(device, 'suitability');
    // The recipes put row 0 at the minimum y; the terrain has row 0 at the north edge. All
    // analysis geometry therefore lives in a frame with y negated, drawn with positionScale -1.
    const flippedExtent: [number, number, number, number] = [
      bounds[0],
      -bounds[3],
      bounds[2],
      -bounds[1]
    ];

    // --- Data buffers --------------------------------------------------------------------------
    const validityValues = new Uint32Array(cellCount);
    for (let index = 0; index < cellCount; index++) {
      validityValues[index] = terrain.elevation[index] > 0.5 ? 1 : 0;
    }
    const parkingPositions: number[] = [];
    for (let index = 0; index + 1 < parking.positions.length; index += 2) {
      const x = parking.positions[index];
      const y = parking.positions[index + 1];
      if (x >= bounds[0] && x <= bounds[2] && y >= bounds[1] && y <= bounds[3]) {
        parkingPositions.push(x, -y);
      }
    }
    if (parkingPositions.length === 0) {
      parkingPositions.push((bounds[0] + bounds[2]) / 2, -(bounds[1] + bounds[3]) / 2);
    }
    const parkingCount = parkingPositions.length / 2;

    const elevationBuffer = resources.createBuffer('elevation', terrain.elevation);
    const validityBuffer = resources.createBuffer('validity', validityValues);
    const slopeBuffer = resources.createBuffer('slope', cellCount * 4);
    const distanceBuffer = resources.createBuffer('distance', cellCount * 4);
    const parkingBuffer = resources.createBuffer('parking', Float32Array.from(parkingPositions));
    const parkingIdsBuffer = resources.createBuffer('parking-ids', new Uint32Array(parkingCount));
    const parkingCountBuffer = resources.createParameterBuffer(
      'parking-count',
      'uint32',
      1,
      Uint32Array.of(parkingCount)
    );
    const derivativesSettings = resources.createParameterBuffer(
      'derivatives-settings',
      'float32',
      GPU_TERRAIN_DERIVATIVES_PARAMETER_LENGTH
    );
    const distanceSettings = resources.createParameterBuffer(
      'distance-settings',
      'float32',
      GPU_DISTANCE_FIELD_PARAMETER_LENGTH
    );

    // Criteria graph buffers.
    const stackBuffer = resources.createBuffer('stack', cellCount * CRITERION_COUNT * 4);
    const classCountBuffers = CRITERIA.map(name =>
      resources.createBuffer(`class-counts-${name}`, CLASS_COUNT * 4)
    );
    const breakBuffers = CRITERIA.map(name =>
      resources.createBuffer(`breaks-${name}`, BREAK_COUNT * 4)
    );
    const classValueBuffers = CRITERIA.map(name =>
      resources.createBuffer(`class-values-${name}`, CLASS_COUNT * 4)
    );
    const reclassifyParameters = CRITERIA.map(name =>
      resources.createParameterBuffer(
        `reclassify-${name}`,
        'float32',
        GPU_RASTER_RECLASSIFY_PARAMETER_LENGTH
      )
    );
    const overlayParameters = resources.createParameterBuffer(
      'overlay',
      'float32',
      getGPUWeightedOverlayParameterLength(CRITERION_COUNT)
    );
    const equalOverlayParameters = resources.createParameterBuffer(
      'overlay-equal',
      'float32',
      getGPUWeightedOverlayParameterLength(CRITERION_COUNT)
    );
    const cellStatisticsParameters = resources.createParameterBuffer(
      'cell-statistics',
      'float32',
      GPU_RASTER_CELL_STATISTICS_PARAMETER_LENGTH
    );
    const conditionalParameters = resources.createParameterBuffer(
      'conditional',
      'float32',
      GPU_RASTER_CONDITIONAL_PARAMETER_LENGTH
    );
    const arithmeticParameters = resources.createParameterBuffer(
      'arithmetic',
      'float32',
      GPU_RASTER_ARITHMETIC_PARAMETER_LENGTH
    );
    const countParameters = resources.createParameterBuffer(
      'count-reclassify',
      'float32',
      GPU_RASTER_RECLASSIFY_PARAMETER_LENGTH
    );
    const countBreakBuffer = resources.createBuffer('count-break', 4);
    const suitableCountsBuffer = resources.createBuffer('suitable-counts', 8);
    const scoreBuffer = resources.createBuffer('score', cellCount * 4);
    const equalScoreBuffer = resources.createBuffer('score-equal', cellCount * 4);
    const scoreRangeBuffer = resources.createBuffer('score-range', 8);
    const equalScoreRangeBuffer = resources.createBuffer('score-range-equal', 8);
    const limitingBuffer = resources.createBuffer('limiting', cellCount * 4);
    const maskedBuffer = resources.createBuffer('masked', cellCount * 4);
    const effectBuffer = resources.createBuffer('effect', cellCount * 4);

    // Path graph buffers.
    const pathPositionsBuffer = resources.createBuffer('path-positions', 16);
    const pathOffsetsBuffer = resources.createBuffer('path-offsets', Uint32Array.of(0, 2));
    const pathSegmentBuffer = resources.createBuffer('path-segment', 16);
    const profileParameters = resources.createParameterBuffer(
      'profile',
      'float32',
      GPU_RASTER_PROFILE_PARAMETER_LENGTH
    );
    const samplingParameters = resources.createParameterBuffer(
      'sampling',
      'float32',
      GPU_RASTER_SAMPLING_PARAMETER_LENGTH
    );
    const elevationProfile = {
      count: resources.createBuffer('profile-count', 4),
      overflow: resources.createBuffer('profile-overflow', 4),
      distances: resources.createBuffer('profile-distances', PROFILE_CAPACITY * 4),
      values: resources.createBuffer('profile-elevations', PROFILE_CAPACITY * 4),
      length: resources.createBuffer('profile-length', 4),
      gain: resources.createBuffer('profile-gain', 4),
      loss: resources.createBuffer('profile-loss', 4),
      minimum: resources.createBuffer('profile-minimum', 4),
      maximum: resources.createBuffer('profile-maximum', 4)
    };
    const scoreProfile = {
      count: resources.createBuffer('score-profile-count', 4),
      overflow: resources.createBuffer('score-profile-overflow', 4),
      values: resources.createBuffer('profile-scores', PROFILE_CAPACITY * 4),
      minimum: resources.createBuffer('score-profile-minimum', 4),
      maximum: resources.createBuffer('score-profile-maximum', 4)
    };
    const probeBuffer = resources.createBuffer('probe', 8);
    const probeElevationBuffer = resources.createBuffer('probe-elevation', 4);
    const probeSlopeBuffer = resources.createBuffer('probe-slope', 4);
    const probeScoreBuffer = resources.createBuffer('probe-score', 4);
    const parkingSlopeBuffer = resources.createBuffer('parking-slope', parkingCount * 4);
    const endpointBuffer = resources.createBuffer('endpoints', 16);

    const summaryRing = resources.track(
      new GPUReadbackRing(device, {id: 'suitability-summary', byteLength: SUMMARY_BYTE_LENGTH})
    );
    const pathRing = resources.track(
      new GPUReadbackRing(device, {id: 'suitability-path', byteLength: PATH_BYTE_LENGTH})
    );

    // --- Graphs --------------------------------------------------------------------------------
    const prepareGraph = new GPUCommandGraph<void>(device, {id: 'suitability-prepare'});
    prepareGraph.add(
      new GPUTerrainDerivatives({
        id: 'slope',
        width,
        height,
        elevation: {
          id: 'elevation',
          format: 'float32',
          storage: {
            kind: 'buffer',
            values: importGraphBuffer(
              prepareGraph,
              'elevation',
              elevationBuffer,
              'float32',
              cellCount
            )
          },
          validity: importGraphBuffer(prepareGraph, 'validity', validityBuffer, 'uint32', cellCount)
        },
        settings: derivativesSettings.importToGraph(prepareGraph),
        slope: importGraphBuffer(prepareGraph, 'slope', slopeBuffer, 'float32', cellCount),
        cellSizeMode: 'uniform',
        rowDirection: 'south'
      })
    );
    prepareGraph.add(
      new GPUDistanceField({
        id: 'distance',
        width,
        height,
        mode: 'exact',
        settings: distanceSettings.importToGraph(prepareGraph),
        seedPositions: importGraphBuffer(
          prepareGraph,
          'parking',
          parkingBuffer,
          'float32x2',
          parkingCount
        ),
        seedIds: importGraphBuffer(
          prepareGraph,
          'parking-ids',
          parkingIdsBuffer,
          'uint32',
          parkingCount
        ),
        seedCount: parkingCountBuffer.importToGraph(prepareGraph),
        output: {
          distances: importGraphBuffer(
            prepareGraph,
            'distances',
            distanceBuffer,
            'float32',
            cellCount
          )
        }
      })
    );
    const prepared: CompiledGPUCommandGraph<void> = resources.track(prepareGraph.compile());

    const suitabilityGraph = new GPUCommandGraph<void>(device, {id: 'suitability'});
    {
      const graph = suitabilityGraph;
      const validity = importGraphBuffer(graph, 'validity', validityBuffer, 'uint32', cellCount);
      const sources = [slopeBuffer, elevationBuffer, distanceBuffer];
      const stack = importGraphBuffer(
        graph,
        'stack',
        stackBuffer,
        'float32',
        cellCount * CRITERION_COUNT
      );
      CRITERIA.forEach((name, index) => {
        graph.add(
          new GPURasterReclassify({
            id: `reclassify-${name}`,
            values: importGraphBuffer(
              graph,
              `source-${name}`,
              sources[index],
              'float32',
              cellCount
            ),
            validity,
            breaks: importGraphBuffer(
              graph,
              `breaks-${name}`,
              breakBuffers[index],
              'float32',
              BREAK_COUNT
            ),
            classValues: importGraphBuffer(
              graph,
              `class-values-${name}`,
              classValueBuffers[index],
              'float32',
              CLASS_COUNT
            ),
            parameters: reclassifyParameters[index].importToGraph(graph),
            output: {
              reclassified: graph.createDataView(stack.buffer, {
                format: 'float32',
                length: cellCount,
                byteOffset: index * cellCount * 4
              }),
              classCounts: importGraphBuffer(
                graph,
                `class-counts-${name}`,
                classCountBuffers[index],
                'uint32',
                CLASS_COUNT
              )
            }
          })
        );
      });
      const score = importGraphBuffer(graph, 'score', scoreBuffer, 'float32', cellCount);
      const equalScore = importGraphBuffer(
        graph,
        'score-equal',
        equalScoreBuffer,
        'float32',
        cellCount
      );
      graph.add(
        new GPUWeightedOverlay({
          id: 'overlay',
          stack,
          layerCount: CRITERION_COUNT,
          cellCount,
          parameters: overlayParameters.importToGraph(graph),
          output: {
            score,
            scoreRange: importGraphBuffer(graph, 'score-range', scoreRangeBuffer, 'float32', 2)
          }
        })
      );
      graph.add(
        new GPUWeightedOverlay({
          id: 'overlay-equal',
          stack,
          layerCount: CRITERION_COUNT,
          cellCount,
          parameters: equalOverlayParameters.importToGraph(graph),
          output: {
            score: equalScore,
            scoreRange: importGraphBuffer(
              graph,
              'score-range-equal',
              equalScoreRangeBuffer,
              'float32',
              2
            )
          }
        })
      );
      graph.add(
        new GPURasterCellStatistics({
          id: 'limiting-factor',
          stack,
          layerCount: CRITERION_COUNT,
          cellCount,
          parameters: cellStatisticsParameters.importToGraph(graph),
          output: {
            minimum: importGraphBuffer(graph, 'limiting', limitingBuffer, 'float32', cellCount)
          }
        })
      );
      graph.add(
        new GPURasterConditional({
          id: 'mask',
          cellCount,
          conditionValues: score,
          a: score,
          parameters: conditionalParameters.importToGraph(graph),
          output: {values: importGraphBuffer(graph, 'masked', maskedBuffer, 'float32', cellCount)}
        })
      );
      graph.add(
        new GPURasterArithmetic({
          id: 'weights-effect',
          cellCount,
          a: score,
          b: equalScore,
          parameters: arithmeticParameters.importToGraph(graph),
          output: {values: importGraphBuffer(graph, 'effect', effectBuffer, 'float32', cellCount)}
        })
      );
      graph.add(
        new GPURasterReclassify({
          id: 'suitable-count',
          values: score,
          breaks: importGraphBuffer(graph, 'count-break', countBreakBuffer, 'float32', 1),
          parameters: countParameters.importToGraph(graph),
          output: {
            classCounts: importGraphBuffer(
              graph,
              'suitable-counts',
              suitableCountsBuffer,
              'uint32',
              2
            )
          }
        })
      );
    }
    const suitability: CompiledGPUCommandGraph<void> = resources.track(suitabilityGraph.compile());

    const pathGraph = new GPUCommandGraph<void>(device, {id: 'suitability-path'});
    {
      const graph = pathGraph;
      const validity = importGraphBuffer(graph, 'validity', validityBuffer, 'uint32', cellCount);
      const elevation = importGraphBuffer(
        graph,
        'elevation',
        elevationBuffer,
        'float32',
        cellCount
      );
      const slope = importGraphBuffer(graph, 'slope', slopeBuffer, 'float32', cellCount);
      const score = importGraphBuffer(graph, 'score', scoreBuffer, 'float32', cellCount);
      const pathPositions = importGraphBuffer(
        graph,
        'path-positions',
        pathPositionsBuffer,
        'float32x2',
        2
      );
      const pathOffsets = importGraphBuffer(graph, 'path-offsets', pathOffsetsBuffer, 'uint32', 2);
      const profileView = profileParameters.importToGraph(graph);
      graph.add(
        new GPURasterProfile({
          id: 'elevation-profile',
          width,
          height,
          values: elevation,
          validity,
          pathPositions,
          pathOffsets,
          parameters: profileView,
          output: {
            count: importGraphBuffer(graph, 'profile-count', elevationProfile.count, 'uint32', 1),
            overflow: importGraphBuffer(
              graph,
              'profile-overflow',
              elevationProfile.overflow,
              'uint32',
              1
            ),
            sampleDistances: importGraphBuffer(
              graph,
              'profile-distances',
              elevationProfile.distances,
              'float32',
              PROFILE_CAPACITY
            ),
            sampleValues: importGraphBuffer(
              graph,
              'profile-elevations',
              elevationProfile.values,
              'float32',
              PROFILE_CAPACITY
            ),
            pathLength: importGraphBuffer(
              graph,
              'profile-length',
              elevationProfile.length,
              'float32',
              1
            ),
            pathGain: importGraphBuffer(graph, 'profile-gain', elevationProfile.gain, 'float32', 1),
            pathLoss: importGraphBuffer(graph, 'profile-loss', elevationProfile.loss, 'float32', 1),
            pathMinimum: importGraphBuffer(
              graph,
              'profile-minimum',
              elevationProfile.minimum,
              'float32',
              1
            ),
            pathMaximum: importGraphBuffer(
              graph,
              'profile-maximum',
              elevationProfile.maximum,
              'float32',
              1
            )
          }
        })
      );
      graph.add(
        new GPURasterProfile({
          id: 'score-profile',
          width,
          height,
          values: score,
          pathPositions,
          pathOffsets,
          parameters: profileView,
          sampleCapacity: PROFILE_CAPACITY,
          output: {
            count: importGraphBuffer(graph, 'score-profile-count', scoreProfile.count, 'uint32', 1),
            overflow: importGraphBuffer(
              graph,
              'score-profile-overflow',
              scoreProfile.overflow,
              'uint32',
              1
            ),
            sampleValues: importGraphBuffer(
              graph,
              'profile-scores',
              scoreProfile.values,
              'float32',
              PROFILE_CAPACITY
            ),
            pathMinimum: importGraphBuffer(
              graph,
              'score-profile-minimum',
              scoreProfile.minimum,
              'float32',
              1
            ),
            pathMaximum: importGraphBuffer(
              graph,
              'score-profile-maximum',
              scoreProfile.maximum,
              'float32',
              1
            )
          }
        })
      );
      const samplingView = samplingParameters.importToGraph(graph);
      const probePosition = importGraphBuffer(graph, 'probe', probeBuffer, 'float32x2', 1);
      for (const [name, values, output] of [
        ['elevation', elevation, probeElevationBuffer],
        ['slope', slope, probeSlopeBuffer],
        ['score', score, probeScoreBuffer]
      ] as const) {
        graph.add(
          new GPURasterSampling({
            id: `probe-${name}`,
            width,
            height,
            values,
            positions: probePosition,
            parameters: samplingView,
            output: {values: importGraphBuffer(graph, `probe-${name}`, output, 'float32', 1)}
          })
        );
      }
      graph.add(
        new GPURasterSampling({
          id: 'parking-slope',
          width,
          height,
          values: slope,
          positions: importGraphBuffer(graph, 'parking', parkingBuffer, 'float32x2', parkingCount),
          parameters: samplingView,
          output: {
            values: importGraphBuffer(
              graph,
              'parking-slope',
              parkingSlopeBuffer,
              'float32',
              parkingCount
            )
          }
        })
      );
    }
    const pathCompiled: CompiledGPUCommandGraph<void> = resources.track(pathGraph.compile());

    // --- State ---------------------------------------------------------------------------------
    const good: Record<Criterion, number> = {
      slope: CRITERION_INFO.slope.goodRange[2],
      elevation: CRITERION_INFO.elevation.goodRange[2],
      distance: CRITERION_INFO.distance.goodRange[2]
    };
    const poor: Record<Criterion, number> = {
      slope: CRITERION_INFO.slope.poorRange[2],
      elevation: CRITERION_INFO.elevation.poorRange[2],
      distance: CRITERION_INFO.distance.poorRange[2]
    };
    const weights: Record<Criterion, number> = {slope: 0.4, elevation: 0.2, distance: 0.4};
    let restrictSteep = true;
    let restrictFar = false;
    let scoreThreshold = 0.6;
    let display: Display = 'suitable';
    let samplingMethod: GPURasterSamplingMethod = 'bilinear';
    let spacing = 30;
    let tool: Tool = 'profile';
    let showParking = true;
    let showChart = true;
    let rasterOpacity = 0.75;
    let suitabilityDirty = true;
    let pathDirty = true;
    let wantSummary = true;
    let wantPath = true;
    let prepareEncoded = false;
    let destroyed = false;
    let measuring = false;
    let summaryPending = false;
    let pathPending = false;
    let clickStage = 0;
    let draggingHandle = -1;
    const timings: Record<'prepare' | 'suitability' | 'path', CompiledGraphTiming | null> = {
      prepare: null,
      suitability: null,
      path: null
    };

    const defaultMeters = (fractionX: number, fractionY: number): [number, number] => [
      bounds[0] + (bounds[2] - bounds[0]) * fractionX,
      bounds[1] + (bounds[3] - bounds[1]) * fractionY
    ];
    // West to east across the peninsula through the Twin Peaks latitude.
    let pathStart = defaultMeters(0.04, 0.62);
    let pathEnd = defaultMeters(0.88, 0.62);
    let probe = defaultMeters(0.5, 0.6);
    let pathData: PathReadback | null = null;

    // --- Chart ---------------------------------------------------------------------------------
    const chart = document.createElement('canvas');
    chart.style.cssText =
      'position:fixed;right:14px;bottom:14px;width:460px;height:200px;z-index:5;' +
      'border-radius:12px;background:rgba(11,18,38,.92);border:1px solid rgba(113,161,242,.3);' +
      'box-shadow:0 12px 40px rgba(0,0,0,.4);pointer-events:none';
    document.body.appendChild(chart);

    function drawChart(): void {
      const ratio = Math.min(window.devicePixelRatio || 1, 2);
      const cssWidth = 460;
      const cssHeight = 200;
      chart.width = cssWidth * ratio;
      chart.height = cssHeight * ratio;
      chart.style.display = showChart ? 'block' : 'none';
      const context2d = chart.getContext('2d');
      if (!context2d) return;
      context2d.scale(ratio, ratio);
      context2d.clearRect(0, 0, cssWidth, cssHeight);
      context2d.font = '11px system-ui, sans-serif';
      context2d.fillStyle = '#c8d6f0';
      const data = pathData;
      if (!data) {
        context2d.fillText('Profile: waiting for the GPU...', 14, 24);
        return;
      }
      const sampleCount = Math.min(data.words[0], PROFILE_CAPACITY);
      const length = data.scalars[2];
      const gain = data.scalars[3];
      const loss = data.scalars[4];
      context2d.fillText(
        `Profile ${(length / 1000).toFixed(2)} km   gain +${gain.toFixed(0)} m   loss -${loss.toFixed(0)} m` +
          `   ${formatCount(sampleCount)} samples`,
        14,
        20
      );
      if (sampleCount < 2 || !(length > 0)) {
        context2d.fillText('Click the map twice to draw a profile line.', 14, 44);
        return;
      }
      const left = 44;
      const right = cssWidth - 40;
      const top = 46;
      const bottom = cssHeight - 26;
      let minimum = Infinity;
      let maximum = -Infinity;
      for (let index = 0; index < sampleCount; index++) {
        const value = data.elevations[index];
        if (Number.isFinite(value)) {
          minimum = Math.min(minimum, value);
          maximum = Math.max(maximum, value);
        }
      }
      if (!(maximum > minimum)) {
        maximum = minimum + 1;
        if (!Number.isFinite(minimum)) {
          minimum = 0;
          maximum = 1;
        }
      }
      const x = (distance: number) => left + (distance / length) * (right - left);
      const yElevation = (value: number) =>
        bottom - ((value - minimum) / (maximum - minimum)) * (bottom - top);
      const yScore = (value: number) => bottom - Math.min(Math.max(value, 0), 1) * (bottom - top);
      context2d.strokeStyle = 'rgba(160,180,215,.25)';
      context2d.lineWidth = 1;
      for (let tick = 0; tick <= 4; tick++) {
        const y = top + ((bottom - top) * tick) / 4;
        context2d.beginPath();
        context2d.moveTo(left, y);
        context2d.lineTo(right, y);
        context2d.stroke();
      }
      context2d.fillStyle = '#8ea3c8';
      context2d.textAlign = 'right';
      context2d.fillText(`${maximum.toFixed(0)} m`, left - 4, top + 4);
      context2d.fillText(`${minimum.toFixed(0)} m`, left - 4, bottom);
      context2d.textAlign = 'left';
      context2d.fillText('1.0', right + 4, top + 4);
      context2d.fillText('0.0', right + 4, bottom);
      context2d.textAlign = 'center';
      context2d.fillText('0', left, cssHeight - 8);
      context2d.fillText(`${(length / 1000).toFixed(1)} km`, right, cssHeight - 8);
      context2d.fillText('distance along the line', (left + right) / 2, cssHeight - 8);
      // Elevation area.
      context2d.fillStyle = 'rgba(80,190,170,.35)';
      context2d.strokeStyle = 'rgb(110,225,200)';
      context2d.lineWidth = 1.5;
      let open = false;
      context2d.beginPath();
      for (let index = 0; index < sampleCount; index++) {
        const value = data.elevations[index];
        if (!Number.isFinite(value)) {
          open = false;
          continue;
        }
        const px = x(data.distances[index]);
        if (!open) {
          context2d.moveTo(px, yElevation(value));
          open = true;
        } else {
          context2d.lineTo(px, yElevation(value));
        }
      }
      context2d.stroke();
      // Score line.
      context2d.strokeStyle = 'rgb(255,170,80)';
      context2d.lineWidth = 1.5;
      open = false;
      context2d.beginPath();
      for (let index = 0; index < sampleCount; index++) {
        const value = data.scores[index];
        if (!Number.isFinite(value)) {
          open = false;
          continue;
        }
        const px = x(data.distances[index]);
        if (!open) {
          context2d.moveTo(px, yScore(value));
          open = true;
        } else {
          context2d.lineTo(px, yScore(value));
        }
      }
      context2d.stroke();
      context2d.textAlign = 'left';
      context2d.fillStyle = 'rgb(110,225,200)';
      context2d.fillText('elevation', left + 6, top - 6);
      context2d.fillStyle = 'rgb(255,170,80)';
      context2d.fillText('suitability score', left + 70, top - 6);
    }

    // --- Parameter writes ----------------------------------------------------------------------
    function writeCriterion(index: number): void {
      const name = CRITERIA[index];
      const goodValue = good[name];
      const poorValue = Math.max(poor[name], goodValue + 1);
      const breaks = Float32Array.from(
        {length: BREAK_COUNT},
        (_, step) => goodValue + ((poorValue - goodValue) * step) / (BREAK_COUNT - 1)
      );
      const restricted =
        (name === 'slope' && restrictSteep) || (name === 'distance' && restrictFar);
      const classValues = Float32Array.from(RATINGS);
      if (restricted) classValues[CLASS_COUNT - 1] = Number.NaN;
      breakBuffers[index].write(breaks);
      classValueBuffers[index].write(classValues);
      reclassifyParameters[index].write(
        getGPURasterReclassifyParameterValues({breakCount: BREAK_COUNT})
      );
      suitabilityDirty = true;
    }

    function writeWeights(): void {
      overlayParameters.write(
        getGPUWeightedOverlayParameterValues({
          layers: CRITERIA.map(name => ({
            weight: weights[name],
            inputMin: 1,
            inputMax: RATINGS[0]
          })),
          normalizeWeights: true
        })
      );
      suitabilityDirty = true;
    }

    function writeThreshold(): void {
      conditionalParameters.write(
        getGPURasterConditionalParameterValues({
          comparison: '>=',
          threshold: scoreThreshold,
          constantB: Number.NaN
        })
      );
      countBreakBuffer.write(Float32Array.of(scoreThreshold));
      countParameters.write(getGPURasterReclassifyParameterValues({breakCount: 1}));
      suitabilityDirty = true;
    }

    function writeStaticSettings(): void {
      derivativesSettings.write(getGPUTerrainDerivativesParameterValues({cellSize}));
      distanceSettings.write(
        getGPUDistanceFieldParameterValues({bounds: flippedExtent, gridSize: [width, height]})
      );
      equalOverlayParameters.write(
        getGPUWeightedOverlayParameterValues({
          layers: CRITERIA.map(() => ({weight: 1, inputMin: 1, inputMax: RATINGS[0]})),
          normalizeWeights: true
        })
      );
      cellStatisticsParameters.write(
        getGPURasterCellStatisticsParameterValues({noDataPolicy: 'propagate'})
      );
      arithmeticParameters.write(getGPURasterArithmeticParameterValues({operation: 'subtract'}));
    }

    function writeSampling(): void {
      const settings = {width, height, extent: flippedExtent, method: samplingMethod} as const;
      samplingParameters.write(getGPURasterSamplingParameterValues(settings));
      profileParameters.write(getGPURasterProfileParameterValues({...settings, spacing}));
      pathDirty = true;
    }

    function writePath(): void {
      pathPositionsBuffer.write(
        Float32Array.of(pathStart[0], -pathStart[1], pathEnd[0], -pathEnd[1])
      );
      pathSegmentBuffer.write(Float32Array.of(pathStart[0], pathStart[1], pathEnd[0], pathEnd[1]));
      endpointBuffer.write(Float32Array.of(pathStart[0], -pathStart[1], pathEnd[0], -pathEnd[1]));
      probeBuffer.write(Float32Array.of(probe[0], -probe[1]));
      pathDirty = true;
    }

    // --- Controls ------------------------------------------------------------------------------
    context.controls.addSelect<Display>({
      label: 'Show',
      options: [
        {value: 'suitable', label: 'Suitable sites (score >= threshold)'},
        {value: 'score', label: 'Weighted suitability score'},
        {value: 'rating-slope', label: 'Slope rating (5 best to 1)'},
        {value: 'rating-elevation', label: 'Elevation rating'},
        {value: 'rating-distance', label: 'Distance-to-parking rating'},
        {value: 'limiting', label: 'Limiting factor (lowest rating)'},
        {value: 'effect', label: 'Weights effect (score minus equal weights)'}
      ],
      value: display,
      onChange: value => {
        display = value;
        context.updateLayers();
      }
    });
    context.controls.addSlider({
      label: 'Score threshold (per-frame)',
      min: 0,
      max: 1,
      step: 0.02,
      value: scoreThreshold,
      format: value => value.toFixed(2),
      onChange: value => {
        scoreThreshold = value;
        writeThreshold();
      }
    });
    for (const name of CRITERIA) {
      const info = CRITERION_INFO[name];
      const index = CRITERIA.indexOf(name);
      context.controls.addSlider({
        label: `${info.label}: weight`,
        min: 0,
        max: 1,
        step: 0.05,
        value: weights[name],
        format: value => value.toFixed(2),
        onChange: value => {
          weights[name] = value;
          writeWeights();
        }
      });
      context.controls.addSlider({
        label: `${info.label}: rating 5 up to`,
        min: info.goodRange[0],
        max: info.goodRange[1],
        step: info.step,
        value: good[name],
        format: value => `${value}${info.unit}`,
        onChange: value => {
          good[name] = value;
          writeCriterion(index);
        }
      });
      context.controls.addSlider({
        label: `${info.label}: rating 1 from`,
        min: info.poorRange[0],
        max: info.poorRange[1],
        step: info.step,
        value: poor[name],
        format: value => `${value}${info.unit}`,
        onChange: value => {
          poor[name] = value;
          writeCriterion(index);
        }
      });
    }
    context.controls.addToggle({
      label: 'Exclude slopes in the lowest class (restricted)',
      value: restrictSteep,
      onChange: value => {
        restrictSteep = value;
        writeCriterion(0);
      }
    });
    context.controls.addToggle({
      label: 'Exclude sites in the farthest distance class',
      value: restrictFar,
      onChange: value => {
        restrictFar = value;
        writeCriterion(2);
      }
    });
    context.controls.addSlider({
      label: 'Raster opacity',
      min: 0,
      max: 1,
      step: 0.05,
      value: rasterOpacity,
      format: value => value.toFixed(2),
      onChange: value => {
        rasterOpacity = value;
        context.updateLayers();
      }
    });
    context.controls.addSelect<Tool>({
      label: 'Click action',
      options: [
        {value: 'profile', label: 'Draw profile (two clicks, drag handles)'},
        {value: 'probe', label: 'Move probe point'}
      ],
      value: tool,
      onChange: value => {
        tool = value;
        clickStage = 0;
      }
    });
    context.controls.addSlider({
      label: 'Profile spacing (per-frame)',
      min: MINIMUM_SPACING,
      max: 150,
      step: 5,
      value: spacing,
      format: value => `${value} m`,
      onChange: value => {
        spacing = value;
        writeSampling();
      }
    });
    context.controls.addSelect<GPURasterSamplingMethod>({
      label: 'Sampling method (per-frame)',
      options: [
        {value: 'nearest', label: 'Nearest'},
        {value: 'bilinear', label: 'Bilinear'},
        {value: 'bicubic', label: 'Bicubic (Catmull-Rom)'}
      ],
      value: samplingMethod,
      onChange: value => {
        samplingMethod = value;
        writeSampling();
      }
    });
    context.controls.addToggle({
      label: 'Bike parking colored by sampled slope (bright = flat)',
      value: showParking,
      onChange: value => {
        showParking = value;
        context.updateLayers();
      }
    });
    context.controls.addToggle({
      label: 'Profile chart',
      value: showChart,
      onChange: value => {
        showChart = value;
        drawChart();
      }
    });
    context.controls.addLegend({
      title: 'Suitability score (viridis), 0 = unsuitable, 1 = best; restricted cells are empty',
      gradient: {
        colors: [
          [68, 1, 84],
          [59, 82, 139],
          [33, 145, 140],
          [94, 201, 98],
          [253, 231, 37]
        ],
        minimumLabel: '0',
        maximumLabel: '1'
      }
    });
    context.controls.addNote(
      'Each criterion is reclassified into five classes (rating 5 to 1) between its two ' +
        'sliders, the three ratings are combined with the weights, and the threshold keeps the ' +
        'best cells. Click twice to draw a profile; drag the white handles to refine it.'
    );
    context.controls.addReadout(
      'Grid',
      `${width} × ${height} cells of ${cellSize[0].toFixed(1)} × ${cellSize[1].toFixed(1)} m`
    );
    const classReadouts = CRITERIA.map(name =>
      context.controls.addReadout(`${CRITERION_INFO[name].label} classes (5..1)`)
    );
    const scoreReadout = context.controls.addReadout('Score range');
    const suitableReadout = context.controls.addReadout('Suitable cells');
    const lengthReadout = context.controls.addReadout('Profile length');
    const gainReadout = context.controls.addReadout('Gain / loss');
    const extremeReadout = context.controls.addReadout('Elevation min / max');
    const scoreProfileReadout = context.controls.addReadout('Score min / max on line');
    const probeReadout = context.controls.addReadout('Probe (elev, slope, score)');
    const overflowReadout = context.controls.addReadout('Profile overflow');
    const encodeReadout = context.controls.addReadout('Last encode (CPU)');
    const prepareReadout = context.controls.addReadout('Prepare graph (once)');
    const suitabilityReadout = context.controls.addReadout('Suitability graph');
    const pathReadout = context.controls.addReadout('Path graph');
    context.controls.addButton({label: 'Measure all graphs', onClick: () => void measure()});
    context.controls.addReadout('Parking sites', formatCount(parkingCount));
    context.controls.addReadout('Data', `${terrain.attribution}; ${parking.attribution}`);

    // --- Readback ------------------------------------------------------------------------------
    async function readSummary(commandEncoder: CommandEncoder): Promise<void> {
      const ticket = summaryRing.tryAcquire();
      if (!ticket) return;
      wantSummary = false;
      const copy = (sourceBuffer: Buffer, destinationOffset: number, size: number) =>
        commandEncoder.copyBufferToBuffer({
          sourceBuffer,
          sourceOffset: 0,
          destinationBuffer: ticket.buffer,
          destinationOffset,
          size
        });
      classCountBuffers.forEach((buffer, index) =>
        copy(buffer, SUMMARY_COUNTS_OFFSET + index * CLASS_COUNT * 4, CLASS_COUNT * 4)
      );
      copy(scoreRangeBuffer, SUMMARY_SCORE_RANGE_OFFSET, 8);
      copy(suitableCountsBuffer, SUMMARY_SUITABLE_OFFSET, 8);
      ticket.markEncoded({byteOffset: 0, byteLength: SUMMARY_BYTE_LENGTH});
      summaryPending = true;
      try {
        const bytes = (await ticket.read()).slice(0, SUMMARY_BYTE_LENGTH);
        if (destroyed) return;
        const words = new Uint32Array(bytes.buffer, 0, SUMMARY_BYTE_LENGTH / 4);
        const floats = new Float32Array(bytes.buffer, 0, SUMMARY_BYTE_LENGTH / 4);
        CRITERIA.forEach((_, index) => {
          const counts = Array.from(words.subarray(index * CLASS_COUNT, (index + 1) * CLASS_COUNT));
          const total = counts.reduce((sum, value) => sum + value, 0) || 1;
          classReadouts[index].setValue(
            counts.map(value => `${((value / total) * 100).toFixed(0)}%`).join(' · ')
          );
        });
        const rangeIndex = SUMMARY_SCORE_RANGE_OFFSET / 4;
        scoreReadout.setValue(
          Number.isFinite(floats[rangeIndex])
            ? `${floats[rangeIndex].toFixed(2)} to ${floats[rangeIndex + 1].toFixed(2)}`
            : 'no valid cells'
        );
        const suitableIndex = SUMMARY_SUITABLE_OFFSET / 4;
        const below = words[suitableIndex];
        const above = words[suitableIndex + 1];
        const valid = below + above || 1;
        suitableReadout.setValue(
          `${formatCount(above)} of ${formatCount(below + above)} (${((above / valid) * 100).toFixed(1)}%)`
        );
      } catch {
        // The ring or device was destroyed while the read was in flight.
      } finally {
        summaryPending = false;
      }
    }

    async function readPath(commandEncoder: CommandEncoder): Promise<void> {
      const ticket = pathRing.tryAcquire();
      if (!ticket) return;
      wantPath = false;
      const copy = (sourceBuffer: Buffer, destinationOffset: number, size: number) =>
        commandEncoder.copyBufferToBuffer({
          sourceBuffer,
          sourceOffset: 0,
          destinationBuffer: ticket.buffer,
          destinationOffset,
          size
        });
      // Scalars: [count, overflow, length, gain, loss, minimum, maximum, scoreMinimum,
      // scoreMaximum, probeElevation, probeSlope, probeScore].
      const scalarBuffers = [
        elevationProfile.count,
        elevationProfile.overflow,
        elevationProfile.length,
        elevationProfile.gain,
        elevationProfile.loss,
        elevationProfile.minimum,
        elevationProfile.maximum,
        scoreProfile.minimum,
        scoreProfile.maximum,
        probeElevationBuffer,
        probeSlopeBuffer,
        probeScoreBuffer
      ];
      scalarBuffers.forEach((buffer, index) => copy(buffer, PATH_SCALARS_OFFSET + index * 4, 4));
      copy(elevationProfile.distances, PATH_DISTANCES_OFFSET, PROFILE_CAPACITY * 4);
      copy(elevationProfile.values, PATH_ELEVATIONS_OFFSET, PROFILE_CAPACITY * 4);
      copy(scoreProfile.values, PATH_SCORES_OFFSET, PROFILE_CAPACITY * 4);
      ticket.markEncoded({byteOffset: 0, byteLength: PATH_BYTE_LENGTH});
      pathPending = true;
      try {
        const bytes = (await ticket.read()).slice(0, PATH_BYTE_LENGTH);
        if (destroyed) return;
        pathData = {
          scalars: new Float32Array(bytes.buffer, PATH_SCALARS_OFFSET, PATH_SCALAR_COUNT),
          words: new Uint32Array(bytes.buffer, PATH_SCALARS_OFFSET, PATH_SCALAR_COUNT),
          distances: new Float32Array(bytes.buffer, PATH_DISTANCES_OFFSET, PROFILE_CAPACITY),
          elevations: new Float32Array(bytes.buffer, PATH_ELEVATIONS_OFFSET, PROFILE_CAPACITY),
          scores: new Float32Array(bytes.buffer, PATH_SCORES_OFFSET, PROFILE_CAPACITY)
        };
        const {scalars, words} = pathData;
        lengthReadout.setValue(
          `${(scalars[2] / 1000).toFixed(2)} km, ${formatCount(words[0])} samples`
        );
        gainReadout.setValue(`+${scalars[3].toFixed(0)} m / -${scalars[4].toFixed(0)} m`);
        extremeReadout.setValue(
          Number.isFinite(scalars[5])
            ? `${scalars[5].toFixed(0)} / ${scalars[6].toFixed(0)} m`
            : 'no land'
        );
        scoreProfileReadout.setValue(
          Number.isFinite(scalars[7])
            ? `${scalars[7].toFixed(2)} / ${scalars[8].toFixed(2)}`
            : 'none'
        );
        probeReadout.setValue(
          [scalars[9], scalars[10], scalars[11]]
            .map((value, index) =>
              Number.isFinite(value) ? value.toFixed(index === 2 ? 2 : 1) : 'n/a'
            )
            .join(', ')
        );
        overflowReadout.setValue(
          words[1] ? 'OVERFLOW: raise spacing' : `ok (capacity ${PROFILE_CAPACITY})`
        );
        drawChart();
      } catch {
        // The ring or device was destroyed while the read was in flight.
      } finally {
        pathPending = false;
      }
    }

    async function measure(): Promise<void> {
      if (measuring || destroyed) return;
      measuring = true;
      try {
        const entries = [
          ['prepare', prepared, prepareReadout],
          ['suitability', suitability, suitabilityReadout],
          ['path', pathCompiled, pathReadout]
        ] as const;
        for (const [key, graph, readout] of entries) {
          timings[key] = await measureCompiledGraph(device, graph, {
            parameters: undefined,
            completionBuffer: scoreRangeBuffer,
            signal: context.signal
          });
          if (destroyed) return;
          readout.setValue(formatCompiledGraphTiming(timings[key]));
        }
      } catch {
        // Aborted or destroyed while measuring.
      } finally {
        measuring = false;
        suitabilityDirty = true;
      }
    }

    writeStaticSettings();
    CRITERIA.forEach((_, index) => writeCriterion(index));
    writeWeights();
    writeThreshold();
    writeSampling();
    writePath();
    drawChart();
    void measure();

    // --- Pointer -------------------------------------------------------------------------------
    const clampToTerrain = (event: MapGraphsPointerEvent): [number, number] | null => {
      if (!event.coordinate) return null;
      const [x, y] = projection.project(event.coordinate[0], event.coordinate[1]);
      return [
        Math.min(Math.max(x, bounds[0] + 1), bounds[2] - 1),
        Math.min(Math.max(y, bounds[1] + 1), bounds[3] - 1)
      ];
    };
    const findHandle = (pixel: readonly [number, number]): number => {
      const viewport = context.getViewport();
      if (!viewport) return -1;
      let best = -1;
      let bestDistance = GRAB_RADIUS_PIXELS;
      [pathStart, pathEnd].forEach((point, index) => {
        const [longitude, latitude] = projection.unproject(point[0], point[1]);
        const [pixelX, pixelY] = viewport.project([longitude, latitude]);
        const distance = Math.hypot(pixelX - pixel[0], pixelY - pixel[1]);
        if (distance < bestDistance) {
          bestDistance = distance;
          best = index;
        }
      });
      return best;
    };

    // --- Frame ---------------------------------------------------------------------------------
    const ratingLayerClasses = CRITERIA.map((_, index) =>
      createOffsetRasterLayerClass(index * cellCount)
    );
    const instance: MapGraphsModeInstance = {
      getCompiledGraphs: () => [prepared, suitability, pathCompiled],
      encode(commandEncoder) {
        let cpuMilliseconds = 0;
        if (!prepareEncoded) {
          prepared.encode(commandEncoder, {parameters: undefined});
          prepareEncoded = true;
          suitabilityDirty = true;
        }
        if (suitabilityDirty) {
          cpuMilliseconds += suitability.encode(commandEncoder, {parameters: undefined}).stats
            .cpuEncodeTimeMilliseconds;
          suitabilityDirty = false;
          pathDirty = true;
          wantSummary = true;
        }
        if (pathDirty) {
          cpuMilliseconds += pathCompiled.encode(commandEncoder, {parameters: undefined}).stats
            .cpuEncodeTimeMilliseconds;
          pathDirty = false;
          wantPath = true;
        }
        if (cpuMilliseconds > 0) encodeReadout.setValue(`${cpuMilliseconds.toFixed(2)} ms`);
        if (wantSummary && !summaryPending) void readSummary(commandEncoder);
        if (wantPath && !pathPending) void readPath(commandEncoder);
      },
      getLayers() {
        const layers: Layer[] = [];
        const frameProps = {coordinateOrigin: origin};
        const flipped = {...frameProps, positionScale: [1, -1] as const};
        const rasterProps = {
          ...frameProps,
          gridSize: [width, height] as const,
          bounds,
          rowOrigin: 'north' as const,
          valueFormat: 'float32' as const,
          opacity: rasterOpacity
        };
        const addRaster = (
          id: string,
          values: Buffer,
          colormap: 'viridis' | 'inferno',
          valueRange: [number, number],
          LayerClass: typeof MapGraphsRasterLayer = MapGraphsRasterLayer
        ) =>
          layers.push(
            new LayerClass({
              ...rasterProps,
              id,
              values,
              colormap,
              valueRange,
              color: [255, 255, 255, 255]
            })
          );
        switch (display) {
          case 'suitable':
            addRaster('suitability-masked', maskedBuffer, 'viridis', [0, 1]);
            break;
          case 'score':
            addRaster('suitability-score', scoreBuffer, 'viridis', [0, 1]);
            break;
          case 'limiting':
            addRaster('suitability-limiting', limitingBuffer, 'viridis', [1, 5]);
            break;
          case 'effect':
            addRaster('suitability-effect', effectBuffer, 'inferno', [-0.25, 0.25]);
            break;
          default: {
            const index = CRITERIA.indexOf(display.replace('rating-', '') as Criterion);
            addRaster(
              `suitability-rating-${CRITERIA[index]}`,
              stackBuffer,
              'viridis',
              [1, 5],
              ratingLayerClasses[index]
            );
          }
        }
        if (showParking) {
          layers.push(
            new MapGraphsPointLayer({
              id: 'suitability-parking',
              ...flipped,
              positions: parkingBuffer,
              instanceCount: parkingCount,
              radiusPixels: 3,
              values: parkingSlopeBuffer,
              valueFormat: 'float32',
              // Bright is flat, dark is steep, so the points stay visible over the raster.
              colormap: 'inferno',
              valueScale: -1,
              valueRange: [-20, 0],
              noDataColor: [120, 130, 150, 180]
            })
          );
        }
        layers.push(
          new MapGraphsSegmentLayer({
            id: 'suitability-profile-line',
            ...frameProps,
            segments: pathSegmentBuffer,
            instanceCount: 1,
            widthPixels: 2.5,
            color: [255, 255, 255, 235]
          }),
          new MapGraphsPointLayer({
            id: 'suitability-endpoints',
            ...flipped,
            positions: endpointBuffer,
            instanceCount: 2,
            radiusPixels: 7,
            color: [255, 255, 255, 255]
          }),
          new MapGraphsPointLayer({
            id: 'suitability-probe-halo',
            ...flipped,
            positions: probeBuffer,
            instanceCount: 1,
            radiusPixels: 9,
            color: [255, 255, 255, 235]
          }),
          new MapGraphsPointLayer({
            id: 'suitability-probe',
            ...flipped,
            positions: probeBuffer,
            instanceCount: 1,
            radiusPixels: 5.5,
            color: [230, 40, 60, 255]
          })
        );
        return layers;
      },
      onClick(event) {
        const point = clampToTerrain(event);
        if (!point) return false;
        if (tool === 'probe') {
          probe = point;
        } else if (clickStage === 0) {
          pathStart = point;
          pathEnd = point;
          clickStage = 1;
          context.setStatus('Click a second point to finish the profile line.');
        } else {
          pathEnd = point;
          clickStage = 0;
          context.setStatus('');
        }
        writePath();
        return true;
      },
      onDragStart(event) {
        const handle = findHandle(event.pixel);
        if (handle < 0) return false;
        draggingHandle = handle;
        context.setMapDragEnabled(false);
        return true;
      },
      onDrag(event) {
        const point = clampToTerrain(event);
        if (!point || draggingHandle < 0) return;
        if (draggingHandle === 0) pathStart = point;
        else pathEnd = point;
        writePath();
      },
      onDragEnd(event) {
        if (draggingHandle < 0) return;
        const point = clampToTerrain(event);
        if (point) {
          if (draggingHandle === 0) pathStart = point;
          else pathEnd = point;
          writePath();
        }
        draggingHandle = -1;
        context.setMapDragEnabled(true);
      },
      destroy() {
        destroyed = true;
        if (draggingHandle >= 0) context.setMapDragEnabled(true);
        chart.remove();
        resources.destroy();
      }
    };
    return instance;
  }
};
