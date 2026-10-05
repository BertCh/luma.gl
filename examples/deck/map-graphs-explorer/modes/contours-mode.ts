// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Raster contrast stretch, filled contour bands and contour lines of one raster, all on the GPU.
 *
 * The raster is switchable between three San Francisco layers that share one 512 x 512 grid
 * (256 x 256 with synthetic data): terrain elevation, slope (`GPUTerrainDerivatives`) and distance
 * to the nearest bike parking (`GPUDistanceField`). A one-time "prepare" graph computes slope and
 * distance; choosing a source is a buffer copy into the raster the analysis graph reads.
 *
 * The analysis graph is `GPURasterStretch` (linear, percentile or histogram-equalized stretch with
 * gamma, sigmoid contrast and a palette, optionally restricted to the visible extent) feeding the
 * raster colors, `GPUIsobands` (filled bands from GPU triangles) and `GPUIsolines` (segments and,
 * in a second compiled graph, stitched polylines). Levels are either an interval in source units
 * or class breaks of the stretch curve (inverse of the readback lookup table, so histogram
 * equalization gives quantile classes). Every control is a buffer write; the graphs are compiled
 * once. They are encoded only when an input changed (the results persist in the output buffers),
 * which includes panning while "stretch to view" is on.
 *
 * Differences from the Terrain mode contours (`GPUTerrainContours`): one ordered segment buffer
 * with level indices instead of one vertex buffer and count per level slot, a 64 level limit in
 * one set of buffers instead of 32 slots, line segments drawn as 6-vertex quads straight from the
 * recipe output (no draw-record repacking), optional stitching into polylines and rings, and
 * matching filled bands that share vertices with the lines bit for bit.
 */

import type {Layer} from '@deck.gl/core';
import type {Buffer, CommandEncoder} from '@luma.gl/core';
import {
  DrawCommandBuffer,
  GPUCommandGraph,
  GPUReadbackRing,
  type CompiledGPUCommandGraph
} from '@luma.gl/gpgpu/gpu-core';
import {getGPUDistanceFieldParameterValues, getGPUIsobandsParameterValues, getGPUIsolinesParameterValues, getGPURasterStretchParameterValues, GPU_DISTANCE_FIELD_PARAMETER_LENGTH, GPU_ISOBANDS_PARAMETER_LENGTH, GPU_ISOLINES_PARAMETER_LENGTH, GPU_RASTER_STRETCH_PARAMETER_LENGTH, GPUDistanceField, GPUIsobands, GPUIsolines, GPURasterStretch, type GPURasterStretchMode} from '@luma.gl/experimental/gpu-raster';
import {getGPUTerrainDerivativesParameterValues, GPU_TERRAIN_DERIVATIVES_PARAMETER_LENGTH, GPUTerrainDerivatives} from '@luma.gl/experimental/gpu-terrain';
import {importGraphBuffer} from '@luma.gl/experimental/UNRESOLVED';
import {LocalMetricProjection} from '../map-graphs-data';
import {MapGraphsSegmentLayer, type MapGraphsColor} from '../map-graphs-layers';
import type {MapGraphsModeDefinition, MapGraphsModeInstance} from '../map-graphs-mode';
import {formatCount, getViewportMetricBounds, MapGraphsResources} from '../map-graphs-resources';
import {IsobandTriangleLayer, PackedColorRasterLayer, PolylineLayer} from './contours-layers';
import {
  formatCompiledGraphTiming,
  measureCompiledGraph,
  type CompiledGraphTiming
} from './vector-timing';

/** Compile-time number of contour level slots (isolines levels and isobands breaks). */
const LEVEL_SLOT_COUNT = 64;
/** Levels written before the first statistics readback of a source tells the real domain. */
const PROVISIONAL_LEVEL_COUNT = 24;
/** Compile-time capacities of the recipe outputs; overflow flags are shown in the panel. */
const SEGMENT_CAPACITY = 400_000;
const TRIANGLE_CAPACITY = 1_000_000;
const HISTOGRAM_BIN_COUNT = 1024;
const LUT_SIZE = 256;
const PALETTE_SIZE = 256;
/** Upper bound on bike-parking seeds for the distance source. */
const SEED_CAPACITY = 4096;
/** Draw records: segments, polylines, bands. */
const RECORD_SEGMENTS = 0;
const RECORD_POLYLINES = 1;
const RECORD_BANDS = 2;
const RECORD_BYTE_LENGTH = 16;

/** Byte layout of the readback ring: statistics, counters, lookup table, histogram. */
const READ_STATISTICS_OFFSET = 0;
const READ_COUNTERS_OFFSET = 32;
const READ_LUT_OFFSET = 64;
const READ_HISTOGRAM_OFFSET = READ_LUT_OFFSET + LUT_SIZE * 4;
const READ_BYTE_LENGTH = READ_HISTOGRAM_OFFSET + HISTOGRAM_BIN_COUNT * 4;
const COUNTER_WORDS = 7;

type Source = 'elevation' | 'slope' | 'distance';
type Classification = 'interval' | 'stretch';
type PaletteName = 'viridis' | 'inferno' | 'terrain' | 'spectral' | 'grayscale';

/** Interval slider unit and value format per source. */
const SOURCE_INFO: Record<
  Source,
  {label: string; unit: string; step: number; maximumSteps: number}
> = {
  elevation: {label: 'Elevation (m)', unit: 'm', step: 5, maximumSteps: 40},
  slope: {label: 'Slope (degrees)', unit: '°', step: 1, maximumSteps: 20},
  distance: {label: 'Distance to bike parking (m)', unit: 'm', step: 100, maximumSteps: 30}
};

const PALETTES: Record<PaletteName, readonly (readonly [number, number, number])[]> = {
  viridis: [
    [68, 1, 84],
    [59, 82, 139],
    [33, 145, 140],
    [94, 201, 98],
    [253, 231, 37]
  ],
  inferno: [
    [0, 0, 4],
    [87, 16, 110],
    [188, 55, 84],
    [249, 142, 9],
    [252, 255, 164]
  ],
  terrain: [
    [38, 70, 140],
    [66, 156, 120],
    [189, 214, 120],
    [201, 160, 90],
    [150, 100, 70],
    [250, 250, 250]
  ],
  spectral: [
    [50, 136, 189],
    [153, 213, 148],
    [255, 255, 191],
    [253, 174, 97],
    [213, 62, 79]
  ],
  grayscale: [
    [15, 15, 15],
    [245, 245, 245]
  ]
};

const LINE_COLOR: MapGraphsColor = [20, 20, 30, 210];
const INDEX_LINE_COLOR: MapGraphsColor = [255, 255, 255, 255];

type Readback = {
  generation: number;
  statistics: Float32Array;
  counters: Uint32Array;
  lut: Float32Array;
  histogram: Uint32Array;
};

export const contoursMode: MapGraphsModeDefinition = {
  id: 'contours',
  title: 'Contours',
  recipes: ['GPURasterStretch', 'GPUIsolines', 'GPUIsobands'],
  description:
    'Stretch a San Francisco raster (elevation, slope or distance to bike parking) with ' +
    'percentile, equalize, gamma and sigmoid controls, then draw filled bands and contour lines ' +
    'from GPU buffers. Levels are an interval or quantile-style classes of the stretch curve.',
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
    const resources = new MapGraphsResources(device, 'contours');
    // The recipes put row 0 at the minimum y. The terrain has row 0 at the north edge, so the
    // analysis runs in a frame with y negated and the layers scale y by -1 to come back.
    const flippedExtent: [number, number, number, number] = [
      bounds[0],
      -bounds[3],
      bounds[2],
      -bounds[1]
    ];

    // --- Buffers -------------------------------------------------------------------------------
    const validityValues = new Uint32Array(cellCount);
    for (let index = 0; index < cellCount; index++) {
      validityValues[index] = terrain.elevation[index] > 0.5 ? 1 : 0;
    }
    const seeds: number[] = [];
    for (
      let index = 0;
      index + 1 < parking.positions.length && seeds.length < SEED_CAPACITY * 2;
    ) {
      const x = parking.positions[index];
      const y = parking.positions[index + 1];
      index += 2;
      if (x >= bounds[0] && x <= bounds[2] && y >= bounds[1] && y <= bounds[3]) {
        seeds.push(x, -y);
      }
    }
    if (seeds.length === 0) seeds.push((bounds[0] + bounds[2]) / 2, -(bounds[1] + bounds[3]) / 2);
    const seedCount = seeds.length / 2;

    const elevationBuffer = resources.createBuffer('elevation', terrain.elevation);
    const slopeBuffer = resources.createBuffer('slope', cellCount * 4);
    const distanceBuffer = resources.createBuffer('distance', cellCount * 4);
    const valuesBuffer = resources.createBuffer('values', cellCount * 4);
    const validityBuffer = resources.createBuffer('validity', validityValues);
    const seedPositionsBuffer = resources.createBuffer('seed-positions', Float32Array.from(seeds));
    const seedIdsBuffer = resources.createBuffer('seed-ids', new Uint32Array(seedCount));
    const seedCountBuffer = resources.createParameterBuffer(
      'seed-count',
      'uint32',
      1,
      Uint32Array.of(seedCount)
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
    const stretchParameters = resources.createParameterBuffer(
      'stretch-parameters',
      'float32',
      GPU_RASTER_STRETCH_PARAMETER_LENGTH
    );
    const isolineParameters = resources.createParameterBuffer(
      'isoline-parameters',
      'float32',
      GPU_ISOLINES_PARAMETER_LENGTH
    );
    const isobandParameters = resources.createParameterBuffer(
      'isoband-parameters',
      'float32',
      GPU_ISOBANDS_PARAMETER_LENGTH
    );
    const levelsBuffer = resources.createBuffer('levels', LEVEL_SLOT_COUNT * 4);
    const levelStyleBuffer = resources.createBuffer('level-style', LEVEL_SLOT_COUNT * 4);
    const paletteBuffer = resources.createBuffer('palette', new Uint32Array(PALETTE_SIZE));
    const colorsBuffer = resources.createBuffer('colors', cellCount * 4);
    const lutBuffer = resources.createBuffer('lut', LUT_SIZE * 4);
    const histogramBuffer = resources.createBuffer('histogram', HISTOGRAM_BIN_COUNT * 4);
    const statisticsBuffer = resources.createBuffer('statistics', 8 * 4);
    const segmentsBuffer = resources.createBuffer('segments', SEGMENT_CAPACITY * 16);
    const segmentLevelsBuffer = resources.createBuffer('segment-levels', SEGMENT_CAPACITY * 4);
    const segmentCountBuffer = resources.createBuffer('segment-count', 4);
    const segmentOverflowBuffer = resources.createBuffer('segment-overflow', 4);
    const trianglesBuffer = resources.createBuffer('triangles', TRIANGLE_CAPACITY * 3 * 8);
    const triangleBandsBuffer = resources.createBuffer('triangle-bands', TRIANGLE_CAPACITY * 4);
    const triangleCountBuffer = resources.createBuffer('triangle-count', 4);
    const triangleOverflowBuffer = resources.createBuffer('triangle-overflow', 4);
    const bandVertexCountBuffer = resources.createBuffer('band-vertex-count', 4);
    const polylineVerticesBuffer = resources.createBuffer(
      'polyline-vertices',
      SEGMENT_CAPACITY * 2 * 8
    );
    const polylineOffsetsBuffer = resources.createBuffer(
      'polyline-offsets',
      (SEGMENT_CAPACITY + 1) * 4
    );
    const polylineLevelsBuffer = resources.createBuffer('polyline-levels', SEGMENT_CAPACITY * 4);
    const polylineClosedBuffer = resources.createBuffer('polyline-closed', SEGMENT_CAPACITY * 4);
    const polylineCountBuffer = resources.createBuffer('polyline-count', 4);
    const polylineVertexCountBuffer = resources.createBuffer('polyline-vertex-count', 4);
    const polylineOverflowBuffer = resources.createBuffer('polyline-overflow', 4);
    const drawCommands = resources.track(
      new DrawCommandBuffer(device, {
        id: 'contours-draw',
        type: 'draw',
        commands: [
          {vertexCount: 6, instanceCount: 0},
          {vertexCount: 6, instanceCount: 0},
          {vertexCount: 0, instanceCount: 1}
        ]
      })
    );
    const readbackRing = resources.track(
      new GPUReadbackRing(device, {id: 'contours-readback', byteLength: READ_BYTE_LENGTH})
    );

    // --- Graphs --------------------------------------------------------------------------------
    const prepareGraph = new GPUCommandGraph<void>(device, {id: 'contours-prepare'});
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
          'seed-positions',
          seedPositionsBuffer,
          'float32x2',
          seedCount
        ),
        seedIds: importGraphBuffer(prepareGraph, 'seed-ids', seedIdsBuffer, 'uint32', seedCount),
        seedCount: seedCountBuffer.importToGraph(prepareGraph),
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

    const compileAnalysis = (withPolylines: boolean): CompiledGPUCommandGraph<void> => {
      const name = withPolylines ? 'contours-stitched' : 'contours-plain';
      const graph = new GPUCommandGraph<void>(device, {id: name});
      const values = importGraphBuffer(graph, 'values', valuesBuffer, 'float32', cellCount);
      const validity = importGraphBuffer(graph, 'validity', validityBuffer, 'uint32', cellCount);
      const levels = importGraphBuffer(graph, 'levels', levelsBuffer, 'float32', LEVEL_SLOT_COUNT);
      graph.add(
        new GPURasterStretch({
          id: 'stretch',
          width,
          height,
          values,
          validity,
          parameters: stretchParameters.importToGraph(graph),
          palette: importGraphBuffer(graph, 'palette', paletteBuffer, 'uint32', PALETTE_SIZE),
          binCount: HISTOGRAM_BIN_COUNT,
          lutSize: LUT_SIZE,
          output: {
            colors: importGraphBuffer(graph, 'colors', colorsBuffer, 'uint32', cellCount),
            lut: importGraphBuffer(graph, 'lut', lutBuffer, 'float32', LUT_SIZE),
            histogram: importGraphBuffer(
              graph,
              'histogram',
              histogramBuffer,
              'uint32',
              HISTOGRAM_BIN_COUNT
            ),
            statistics: importGraphBuffer(graph, 'statistics', statisticsBuffer, 'float32', 8)
          }
        })
      );
      graph.add(
        new GPUIsobands({
          id: 'bands',
          width,
          height,
          values,
          validity,
          breaks: levels,
          parameters: isobandParameters.importToGraph(graph),
          output: {
            triangles: importGraphBuffer(
              graph,
              'triangles',
              trianglesBuffer,
              'float32x2',
              TRIANGLE_CAPACITY * 3
            ),
            triangleBands: importGraphBuffer(
              graph,
              'triangle-bands',
              triangleBandsBuffer,
              'uint32',
              TRIANGLE_CAPACITY
            ),
            count: importGraphBuffer(graph, 'triangle-count', triangleCountBuffer, 'uint32', 1),
            overflow: importGraphBuffer(
              graph,
              'triangle-overflow',
              triangleOverflowBuffer,
              'uint32',
              1
            ),
            vertexCount: importGraphBuffer(
              graph,
              'band-vertex-count',
              bandVertexCountBuffer,
              'uint32',
              1
            )
          }
        })
      );
      graph.add(
        new GPUIsolines({
          id: 'lines',
          width,
          height,
          values,
          validity,
          levels,
          parameters: isolineParameters.importToGraph(graph),
          output: {
            segments: importGraphBuffer(
              graph,
              'segments',
              segmentsBuffer,
              'float32x4',
              SEGMENT_CAPACITY
            ),
            segmentLevels: importGraphBuffer(
              graph,
              'segment-levels',
              segmentLevelsBuffer,
              'uint32',
              SEGMENT_CAPACITY
            ),
            count: importGraphBuffer(graph, 'segment-count', segmentCountBuffer, 'uint32', 1),
            overflow: importGraphBuffer(
              graph,
              'segment-overflow',
              segmentOverflowBuffer,
              'uint32',
              1
            )
          },
          polylines: withPolylines
            ? {
                vertices: importGraphBuffer(
                  graph,
                  'polyline-vertices',
                  polylineVerticesBuffer,
                  'float32x2',
                  SEGMENT_CAPACITY * 2
                ),
                polylineOffsets: importGraphBuffer(
                  graph,
                  'polyline-offsets',
                  polylineOffsetsBuffer,
                  'uint32',
                  SEGMENT_CAPACITY + 1
                ),
                polylineLevels: importGraphBuffer(
                  graph,
                  'polyline-levels',
                  polylineLevelsBuffer,
                  'uint32',
                  SEGMENT_CAPACITY
                ),
                polylineClosed: importGraphBuffer(
                  graph,
                  'polyline-closed',
                  polylineClosedBuffer,
                  'uint32',
                  SEGMENT_CAPACITY
                ),
                polylineCount: importGraphBuffer(
                  graph,
                  'polyline-count',
                  polylineCountBuffer,
                  'uint32',
                  1
                ),
                vertexCount: importGraphBuffer(
                  graph,
                  'polyline-vertex-count',
                  polylineVertexCountBuffer,
                  'uint32',
                  1
                ),
                overflow: importGraphBuffer(
                  graph,
                  'polyline-overflow',
                  polylineOverflowBuffer,
                  'uint32',
                  1
                )
              }
            : undefined
        })
      );
      return resources.track(graph.compile());
    };
    const plainGraph = compileAnalysis(false);
    const stitchedGraph = compileAnalysis(true);

    // --- State ---------------------------------------------------------------------------------
    let source: Source = 'elevation';
    let classification: Classification = 'interval';
    let intervalSteps = 4;
    let classCount = 7;
    let stretchMode: GPURasterStretchMode = 'percentile';
    let percentileLow = 2;
    let percentileHigh = 98;
    let gamma = 1;
    let sigmoidContrast = 0;
    let paletteName: PaletteName = 'viridis';
    let smoothPalette = true;
    let stretchToView = false;
    let showRaster = true;
    let showBands = true;
    let showLines = true;
    let stitchPolylines = false;
    let rasterOpacity = 0.85;
    let bandOpacity = 0.55;
    let dirty = true;
    let destroyed = false;
    let measuring = false;
    let readbackPending = false;
    let prepareEncoded = false;
    /** Incremented whenever the stretch statistics or lookup table can change. */
    let stretchGeneration = 0;
    let lastWindowKey = '';
    let levelCount = 0;
    let levelOffset = 0;
    let levelValues = new Float32Array(LEVEL_SLOT_COUNT);
    let latest: Readback | null = null;
    const fullDomain: Partial<Record<Source, [number, number]>> = {};
    const timings: Record<'plain' | 'stitched', CompiledGraphTiming | null> = {
      plain: null,
      stitched: null
    };

    // --- Parameter writes ----------------------------------------------------------------------
    const getInterval = () => SOURCE_INFO[source].step * intervalSteps;

    function writePalette(): void {
      const stops = PALETTES[paletteName];
      const packed = new Uint32Array(PALETTE_SIZE);
      for (let index = 0; index < PALETTE_SIZE; index++) {
        const position = (index / (PALETTE_SIZE - 1)) * (stops.length - 1);
        const lower = Math.min(Math.floor(position), stops.length - 2);
        const fraction = position - lower;
        const channels = [0, 1, 2].map(channel =>
          Math.round(stops[lower][channel] * (1 - fraction) + stops[lower + 1][channel] * fraction)
        );
        packed[index] =
          (channels[0] | (channels[1] << 8) | (channels[2] << 16) | (255 << 24)) >>> 0;
      }
      paletteBuffer.write(packed);
      dirty = true;
    }

    function writeStretch(): void {
      stretchGeneration++;
      stretchParameters.write(
        getGPURasterStretchParameterValues({
          mode: stretchMode,
          percentiles: [percentileLow, percentileHigh],
          gamma,
          sigmoidContrast,
          paletteInterpolation: smoothPalette ? 'linear' : 'nearest',
          window: getStatisticsWindow(lastViewBounds)
        })
      );
      dirty = true;
    }

    let lastViewBounds: [number, number, number, number] | null = null;
    function getStatisticsWindow(
      viewBounds: [number, number, number, number] | null
    ): [number, number, number, number] | undefined {
      if (!stretchToView || !viewBounds) return undefined;
      return [
        Math.floor((viewBounds[0] - bounds[0]) / cellSize[0]),
        Math.floor((bounds[3] - viewBounds[3]) / cellSize[1]),
        Math.ceil((viewBounds[2] - bounds[0]) / cellSize[0]),
        Math.ceil((bounds[3] - viewBounds[1]) / cellSize[1])
      ];
    }

    /** Interval levels are multiples of the interval spanning the source's full domain. */
    function computeIntervalLevels(): {values: number[]; offset: number} {
      const interval = getInterval();
      const domain = fullDomain[source];
      if (!domain) {
        return {
          values: Array.from(
            {length: PROVISIONAL_LEVEL_COUNT},
            (_, index) => interval * (index + 1)
          ),
          offset: 1
        };
      }
      const first = Math.floor(domain[0] / interval) + 1;
      const last = Math.max(first, Math.floor(domain[1] / interval));
      const count = Math.min(LEVEL_SLOT_COUNT, last - first + 1);
      return {
        values: Array.from({length: count}, (_, index) => interval * (first + index)),
        offset: first
      };
    }

    /** Class breaks at `k / classCount` of the stretch curve, by inverting the lookup table. */
    function computeStretchLevels(readback: Readback): number[] | null {
      const lo = readback.statistics[2];
      const hi = readback.statistics[3];
      if (!(hi > lo)) return null;
      const lut = readback.lut;
      const levels: number[] = [];
      let previous = -Infinity;
      const minimumGap = ((hi - lo) / LUT_SIZE) * 1e-3;
      for (let classIndex = 1; classIndex < classCount; classIndex++) {
        const target = classIndex / classCount;
        let index = 0;
        while (index < LUT_SIZE - 1 && lut[index] < target) index++;
        let level = lo;
        if (index > 0) {
          const span = lut[index] - lut[index - 1];
          const fraction = span > 0 ? (target - lut[index - 1]) / span : 0;
          level = lo + ((index - 1 + fraction) / (LUT_SIZE - 1)) * (hi - lo);
        }
        level = Math.max(level, previous + minimumGap);
        levels.push(level);
        previous = level;
      }
      return levels;
    }

    function writeLevels(values: readonly number[], offset: number): void {
      const count = Math.max(1, Math.min(LEVEL_SLOT_COUNT, values.length));
      const next = new Float32Array(LEVEL_SLOT_COUNT);
      const styles = new Uint32Array(LEVEL_SLOT_COUNT);
      for (let index = 0; index < LEVEL_SLOT_COUNT; index++) {
        // Unused slots stay ascending so the break tables remain sorted.
        next[index] =
          index < count
            ? values[index]
            : values[count - 1] + (index - count + 1) * Math.max(getInterval(), 1);
        styles[index] = (offset + index) % 5 === 0 ? 1 : 0;
      }
      if (
        count === levelCount &&
        offset === levelOffset &&
        next.every((value, index) => value === levelValues[index])
      ) {
        return;
      }
      levelCount = count;
      levelOffset = offset;
      levelValues = next;
      levelsBuffer.write(next);
      levelStyleBuffer.write(styles);
      isolineParameters.write(
        getGPUIsolinesParameterValues({width, height, levelCount: count, extent: flippedExtent})
      );
      isobandParameters.write(
        getGPUIsobandsParameterValues({
          width,
          height,
          breakCount: count,
          extent: flippedExtent
        })
      );
      dirty = true;
    }

    function refreshLevels(): void {
      if (classification === 'interval') {
        const {values, offset} = computeIntervalLevels();
        writeLevels(values, offset);
      } else if (latest && latest.generation === stretchGeneration) {
        const levels = computeStretchLevels(latest);
        if (levels) {
          writeLevels(levels, 1);
          return;
        }
      }
    }

    function writeDerivedSettings(): void {
      derivativesSettings.write(getGPUTerrainDerivativesParameterValues({cellSize}));
      distanceSettings.write(
        getGPUDistanceFieldParameterValues({bounds: flippedExtent, gridSize: [width, height]})
      );
    }

    // --- Controls ------------------------------------------------------------------------------
    const intervalSlider = context.controls.addSlider({
      label: 'Contour interval (per-frame levels)',
      min: 1,
      max: SOURCE_INFO[source].maximumSteps,
      step: 1,
      value: intervalSteps,
      format: value => `${SOURCE_INFO[source].step * value} ${SOURCE_INFO[source].unit}`,
      onChange: value => {
        intervalSteps = value;
        if (classification === 'interval') refreshLevels();
      }
    });
    context.controls.addSelect<Source>({
      label: 'Raster (buffer copy, no recompile)',
      options: (Object.keys(SOURCE_INFO) as Source[]).map(value => ({
        value,
        label: SOURCE_INFO[value].label
      })),
      value: source,
      onChange: value => {
        source = value;
        intervalSteps = Math.min(intervalSteps, SOURCE_INFO[source].maximumSteps);
        intervalSlider.setValue(intervalSteps);
        latest = null;
        writeStretch();
        refreshLevels();
      }
    });
    context.controls.addSelect<Classification>({
      label: 'Levels',
      options: [
        {value: 'interval', label: 'Equal interval in source units'},
        {value: 'stretch', label: 'Classes of the stretch curve (quantiles when equalized)'}
      ],
      value: classification,
      onChange: value => {
        classification = value;
        refreshLevels();
      }
    });
    context.controls.addSlider({
      label: 'Classes (stretch levels)',
      min: 2,
      max: 16,
      step: 1,
      value: classCount,
      onChange: value => {
        classCount = value;
        if (classification === 'stretch') refreshLevels();
      }
    });
    context.controls.addSelect<GPURasterStretchMode>({
      label: 'Stretch method',
      options: [
        {value: 'linear', label: 'Linear min/max'},
        {value: 'percentile', label: 'Percentile clip'},
        {value: 'equalize', label: 'Histogram equalization'}
      ],
      value: stretchMode,
      onChange: value => {
        stretchMode = value;
        writeStretch();
        if (classification === 'stretch') refreshLevels();
      }
    });
    context.controls.addSlider({
      label: 'Low percentile',
      min: 0,
      max: 25,
      step: 0.5,
      value: percentileLow,
      format: value => `${value}%`,
      onChange: value => {
        percentileLow = value;
        writeStretch();
      }
    });
    context.controls.addSlider({
      label: 'High percentile',
      min: 75,
      max: 100,
      step: 0.5,
      value: percentileHigh,
      format: value => `${value}%`,
      onChange: value => {
        percentileHigh = value;
        writeStretch();
      }
    });
    context.controls.addSlider({
      label: 'Gamma',
      min: 0.3,
      max: 3,
      step: 0.05,
      value: gamma,
      format: value => value.toFixed(2),
      onChange: value => {
        gamma = value;
        writeStretch();
      }
    });
    context.controls.addSlider({
      label: 'Sigmoid contrast (0 = off)',
      min: 0,
      max: 20,
      step: 0.5,
      value: sigmoidContrast,
      onChange: value => {
        sigmoidContrast = value;
        writeStretch();
      }
    });
    context.controls.addSelect<PaletteName>({
      label: 'Palette (buffer write)',
      options: (Object.keys(PALETTES) as PaletteName[]).map(value => ({value, label: value})),
      value: paletteName,
      onChange: value => {
        paletteName = value;
        writePalette();
      }
    });
    context.controls.addToggle({
      label: 'Smooth palette interpolation',
      value: smoothPalette,
      onChange: value => {
        smoothPalette = value;
        writeStretch();
      }
    });
    context.controls.addToggle({
      label: 'Stretch to visible extent (statistics window)',
      value: stretchToView,
      onChange: value => {
        stretchToView = value;
        lastWindowKey = '';
        writeStretch();
        if (classification === 'stretch') refreshLevels();
      }
    });
    context.controls.addToggle({
      label: 'Raster',
      value: showRaster,
      onChange: value => {
        showRaster = value;
        context.updateLayers();
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
    context.controls.addToggle({
      label: 'Filled bands (GPUIsobands)',
      value: showBands,
      onChange: value => {
        showBands = value;
        context.updateLayers();
      }
    });
    context.controls.addSlider({
      label: 'Band opacity',
      min: 0,
      max: 1,
      step: 0.05,
      value: bandOpacity,
      format: value => value.toFixed(2),
      onChange: value => {
        bandOpacity = value;
        context.updateLayers();
      }
    });
    context.controls.addToggle({
      label: 'Lines (GPUIsolines)',
      value: showLines,
      onChange: value => {
        showLines = value;
        context.updateLayers();
      }
    });
    context.controls.addToggle({
      label: 'Stitched polylines (second compiled graph)',
      value: stitchPolylines,
      onChange: value => {
        stitchPolylines = value;
        dirty = true;
        context.updateLayers();
      }
    });
    context.controls.addLegend({
      title: 'Bands: class index through the palette (low to high); thick white line = every 5th',
      gradient: {
        colors: PALETTES.viridis,
        minimumLabel: 'lowest class (viridis default)',
        maximumLabel: 'highest class'
      }
    });
    context.controls.addNote(
      'Both the lines and the bands come from the same marching-squares definition, so they ' +
        'coincide exactly. Histogram equalization plus "Classes of the stretch curve" gives ' +
        'equal-count (quantile) classes; compare the class shares below.'
    );
    context.controls.addReadout(
      'Raster',
      `${width} × ${height} cells, ${cellSize[0].toFixed(1)} × ${cellSize[1].toFixed(1)} m`
    );
    const domainReadout = context.controls.addReadout('Domain (exact min, max)');
    const stretchReadout = context.controls.addReadout('Stretch bounds (lo, hi)');
    const validReadout = context.controls.addReadout('Cells in statistics');
    const histogramReadout = context.controls.addReadout('Histogram');
    const levelsReadout = context.controls.addReadout('Levels');
    const sharesReadout = context.controls.addReadout('Class share of cells');
    const lineReadout = context.controls.addReadout('Segments');
    const polylineReadout = context.controls.addReadout('Polylines');
    const bandReadout = context.controls.addReadout('Band triangles');
    const activeReadout = context.controls.addReadout('Active graph');
    const encodeReadout = context.controls.addReadout('Last encode (CPU)');
    const plainReadout = context.controls.addReadout('Plain graph');
    const stitchedReadout = context.controls.addReadout('Stitched graph');
    context.controls.addButton({label: 'Measure both graphs', onClick: () => void measure()});
    context.controls.addReadout('Seeds (distance)', `${formatCount(seedCount)} bike parking sites`);
    context.controls.addReadout('Data', `${terrain.attribution}; ${parking.attribution}`);

    // --- Readback ------------------------------------------------------------------------------
    function describeReadback(readback: Readback): void {
      const [domainMin, domainMax, lo, hi, validCount, binWidth] = readback.statistics;
      const interval = getInterval();
      const unit = SOURCE_INFO[source].unit;
      domainReadout.setValue(
        Number.isFinite(domainMin)
          ? `${domainMin.toFixed(1)} to ${domainMax.toFixed(1)} ${unit}`
          : 'no data'
      );
      stretchReadout.setValue(
        Number.isFinite(lo) ? `${lo.toFixed(1)} to ${hi.toFixed(1)} ${unit}` : 'n/a'
      );
      validReadout.setValue(formatCount(validCount));
      const bars = '▁▂▃▄▅▆▇█';
      const groups = 32;
      const perGroup = HISTOGRAM_BIN_COUNT / groups;
      const sums = Array.from({length: groups}, (_, group) => {
        let sum = 0;
        for (let bin = 0; bin < perGroup; bin++) sum += readback.histogram[group * perGroup + bin];
        return Math.sqrt(sum);
      });
      const peak = Math.max(...sums, 1);
      histogramReadout.setValue(
        sums
          .map(sum => bars[Math.min(bars.length - 1, Math.floor((sum / peak) * bars.length))])
          .join('')
      );
      const counters = readback.counters;
      lineReadout.setValue(
        `${formatCount(counters[0])} of ${formatCount(SEGMENT_CAPACITY)}${counters[1] ? ' OVERFLOW' : ''}`
      );
      bandReadout.setValue(
        `${formatCount(counters[2])} of ${formatCount(TRIANGLE_CAPACITY)}${counters[3] ? ' OVERFLOW' : ''}`
      );
      polylineReadout.setValue(
        stitchPolylines
          ? `${formatCount(counters[4])} (${formatCount(counters[5])} vertices)${counters[6] ? ' OVERFLOW' : ''}`
          : 'off'
      );
      levelsReadout.setValue(
        classification === 'interval'
          ? `${levelCount} every ${interval} ${unit}${levelCount >= LEVEL_SLOT_COUNT ? ' (capped at 64)' : ''}`
          : `${levelCount + 1} classes, ${levelCount} breaks`
      );
      // Class shares from the histogram: each bin goes to the class of its center value.
      if (Number.isFinite(domainMin) && binWidth > 0 && validCount > 0) {
        const classTotals = new Float64Array(levelCount + 1);
        for (let bin = 0; bin < HISTOGRAM_BIN_COUNT; bin++) {
          const center = domainMin + (bin + 0.5) * binWidth;
          let classIndex = 0;
          while (classIndex < levelCount && levelValues[classIndex] <= center) classIndex++;
          classTotals[classIndex] += readback.histogram[bin];
        }
        const total = classTotals.reduce((sum, value) => sum + value, 0) || 1;
        sharesReadout.setValue(
          Array.from(classTotals, value => `${Math.round((value / total) * 100)}`)
            .slice(0, 18)
            .join(' · ') + ' %'
        );
      }
    }

    async function readStatistics(
      commandEncoder: CommandEncoder,
      generation: number,
      withPolylines: boolean
    ): Promise<void> {
      const ticket = readbackRing.tryAcquire();
      if (!ticket) {
        // Ring busy: ask for another encode so the statistics still arrive.
        dirty = true;
        return;
      }
      const copy = (
        sourceBuffer: Buffer,
        sourceOffset: number,
        destinationOffset: number,
        size: number
      ) =>
        commandEncoder.copyBufferToBuffer({
          sourceBuffer,
          sourceOffset,
          destinationBuffer: ticket.buffer,
          destinationOffset,
          size
        });
      copy(statisticsBuffer, 0, READ_STATISTICS_OFFSET, 32);
      const counterBuffers = [
        segmentCountBuffer,
        segmentOverflowBuffer,
        triangleCountBuffer,
        triangleOverflowBuffer,
        polylineCountBuffer,
        polylineVertexCountBuffer,
        polylineOverflowBuffer
      ];
      counterBuffers.forEach((counterBuffer, index) =>
        copy(counterBuffer, 0, READ_COUNTERS_OFFSET + index * 4, 4)
      );
      copy(lutBuffer, 0, READ_LUT_OFFSET, LUT_SIZE * 4);
      copy(histogramBuffer, 0, READ_HISTOGRAM_OFFSET, HISTOGRAM_BIN_COUNT * 4);
      ticket.markEncoded({byteOffset: 0, byteLength: READ_BYTE_LENGTH});
      readbackPending = true;
      try {
        const bytes = await ticket.read();
        if (destroyed) return;
        const copyBytes = bytes.slice(0, READ_BYTE_LENGTH);
        const readback: Readback = {
          generation,
          statistics: new Float32Array(copyBytes.buffer, READ_STATISTICS_OFFSET, 8),
          counters: new Uint32Array(copyBytes.buffer, READ_COUNTERS_OFFSET, COUNTER_WORDS),
          lut: new Float32Array(copyBytes.buffer, READ_LUT_OFFSET, LUT_SIZE),
          histogram: new Uint32Array(copyBytes.buffer, READ_HISTOGRAM_OFFSET, HISTOGRAM_BIN_COUNT)
        };
        if (!withPolylines) readback.counters[4] = readback.counters[5] = readback.counters[6] = 0;
        if (generation !== stretchGeneration) return;
        latest = readback;
        if (!stretchToView && Number.isFinite(readback.statistics[0])) {
          fullDomain[source] = [readback.statistics[0], readback.statistics[1]];
        }
        // Levels derived from this result request one more encode; unchanged levels settle.
        refreshLevels();
        describeReadback(readback);
      } catch {
        // The ring or device was destroyed while the read was in flight.
      } finally {
        readbackPending = false;
      }
    }

    async function measure(): Promise<void> {
      if (measuring || destroyed) return;
      measuring = true;
      try {
        for (const key of ['plain', 'stitched'] as const) {
          timings[key] = await measureCompiledGraph(
            device,
            key === 'plain' ? plainGraph : stitchedGraph,
            {parameters: undefined, completionBuffer: segmentCountBuffer, signal: context.signal}
          );
          if (destroyed) return;
          (key === 'plain' ? plainReadout : stitchedReadout).setValue(
            formatCompiledGraphTiming(timings[key])
          );
        }
      } catch {
        // Aborted or destroyed while measuring.
      } finally {
        measuring = false;
        dirty = true;
      }
    }

    writeDerivedSettings();
    writePalette();
    writeStretch();
    refreshLevels();
    void measure();

    // --- Per-frame encode ----------------------------------------------------------------------
    const getSourceBuffer = (): Buffer =>
      source === 'elevation' ? elevationBuffer : source === 'slope' ? slopeBuffer : distanceBuffer;

    const instance: MapGraphsModeInstance = {
      getCompiledGraphs: () => [prepared, plainGraph, stitchedGraph],
      encode(commandEncoder, frame) {
        if (stretchToView) {
          const viewBounds = getViewportMetricBounds(frame.viewport, projection);
          const window = getStatisticsWindow(viewBounds);
          const key = window ? window.join(',') : '';
          if (key !== lastWindowKey) {
            lastWindowKey = key;
            lastViewBounds = viewBounds;
            writeStretch();
            if (classification === 'stretch') refreshLevels();
          }
        }
        if (!prepareEncoded) {
          prepared.encode(commandEncoder, {parameters: undefined});
          prepareEncoded = true;
          dirty = true;
        }
        if (!dirty || readbackPending) return;
        commandEncoder.copyBufferToBuffer({
          sourceBuffer: getSourceBuffer(),
          sourceOffset: 0,
          destinationBuffer: valuesBuffer,
          destinationOffset: 0,
          size: cellCount * 4
        });
        const compiled = stitchPolylines ? stitchedGraph : plainGraph;
        const encoding = compiled.encode(commandEncoder, {parameters: undefined});
        const copyWord = (sourceBuffer: Buffer, record: number, wordIndex: number) =>
          commandEncoder.copyBufferToBuffer({
            sourceBuffer,
            sourceOffset: 0,
            destinationBuffer: drawCommands.buffer,
            destinationOffset: record * RECORD_BYTE_LENGTH + wordIndex * 4,
            size: 4
          });
        copyWord(segmentCountBuffer, RECORD_SEGMENTS, 1);
        copyWord(polylineVertexCountBuffer, RECORD_POLYLINES, 1);
        copyWord(bandVertexCountBuffer, RECORD_BANDS, 0);
        activeReadout.setValue(stitchPolylines ? 'stitched' : 'plain');
        encodeReadout.setValue(`${encoding.stats.cpuEncodeTimeMilliseconds.toFixed(2)} ms`);
        // Every readback also confirms the result: levels derived from it may request one more
        // encode, which then reads back unchanged levels and settles.
        dirty = false;
        void readStatistics(commandEncoder, stretchGeneration, stitchPolylines);
      },
      getLayers() {
        const layers: Layer[] = [];
        const frameProps = {
          coordinateOrigin: origin,
          positionScale: [1, -1] as const
        };
        if (showRaster) {
          layers.push(
            new PackedColorRasterLayer({
              id: 'contours-raster',
              coordinateOrigin: origin,
              gridSize: [width, height],
              bounds,
              rowOrigin: 'north',
              values: colorsBuffer,
              valueFormat: 'uint32',
              colormap: 'category',
              opacity: rasterOpacity
            })
          );
        }
        if (showBands) {
          layers.push(
            new IsobandTriangleLayer({
              id: 'contours-bands',
              ...frameProps,
              gridSize: [1, 1],
              bounds,
              triangles: trianglesBuffer,
              triangleBands: triangleBandsBuffer,
              values: paletteBuffer,
              valueFormat: 'uint32',
              colormap: 'category',
              extent: isobandParameters.buffer,
              drawCommands,
              drawCommandIndex: RECORD_BANDS,
              opacity: bandOpacity
            })
          );
        }
        if (showLines) {
          const lineStyle = {
            ...frameProps,
            values: levelStyleBuffer,
            valueFormat: 'uint32' as const,
            colormap: 'category' as const,
            palette: [LINE_COLOR, INDEX_LINE_COLOR],
            widthPixels: 1.1
          };
          layers.push(
            stitchPolylines
              ? new PolylineLayer({
                  id: 'contours-polylines',
                  ...lineStyle,
                  segments: polylineVerticesBuffer,
                  polylineOffsets: polylineOffsetsBuffer,
                  valueIndices: polylineLevelsBuffer,
                  extent: polylineCountBuffer,
                  drawCommands,
                  drawCommandIndex: RECORD_POLYLINES
                })
              : new MapGraphsSegmentLayer({
                  id: 'contours-segments',
                  ...lineStyle,
                  segments: segmentsBuffer,
                  valueIndices: segmentLevelsBuffer,
                  drawCommands,
                  drawCommandIndex: RECORD_SEGMENTS
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
