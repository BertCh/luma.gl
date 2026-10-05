// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Kepler-style choropleth classification of New York trip vertices, entirely on the GPU.
 *
 * `GPUPointDensity` bins the vertices into a viewport-following square or hexagon lattice (count
 * and mean speed per cell). `GPUColumnQuantiles` applies a percentile filter to the occupied
 * cells, `GPUClassBreaks` classifies them (any of the eight methods, any class count up to the
 * compiled maximum), and `GPUColorScale` writes packed `rgba8` colors that a deck.gl layer reads
 * straight from the storage buffer. A bivariate toggle adds `GPUBivariateClassification` (count by
 * mean speed) and `GPUColumnProfile` summarizes the filtered columns. Every control writes a
 * parameter or palette buffer; only the square/hexagon switch recompiles. The legend, class counts
 * and profile come back through one small `GPUReadbackRing` read.
 */

import type {Layer} from '@deck.gl/core';
import type {Buffer, CommandEncoder} from '@luma.gl/core';
import {
  GPUCommandGraph,
  GPUReadbackRing,
  createTransientView,
  type CompiledGPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {GPUBivariateClassification, GPUClassBreaks, GPUColorScale, GPUColumnProfile, GPUColumnQuantiles, GPU_COLUMN_PROFILE_STATISTIC, GPU_COLUMN_PROFILE_STATISTIC_COUNT, GPU_CLASS_BREAKS_METHODS, GPU_CLASS_BREAKS_METHOD_CODES, getGPUBivariateClassificationParameterValues, getGPUClassBreaksParameterLength, getGPUClassBreaksParameterValues, getGPUColorScaleParameterValues, getGPUColumnQuantilesParameterLength, getGPUColumnQuantilesParameterValues, GPU_BIVARIATE_CLASSIFICATION_PARAMETER_LENGTH, GPU_COLOR_SCALE_PARAMETER_LENGTH, type GPUClassBreaksMethod, type GPUColorScaleType} from '@luma.gl/experimental/gpu-dataframe';
import {GPUPointDensity} from '@luma.gl/experimental/geospatial';
import {importGraphBuffer} from '@luma.gl/experimental/UNRESOLVED';
import {LocalMetricProjection} from '../map-graphs-data';
import type {MapGraphsModeDefinition, MapGraphsModeInstance} from '../map-graphs-mode';
import {MapGraphsPointLayer} from '../map-graphs-layers';
import {formatCount, getViewportMetricBounds, MapGraphsResources} from '../map-graphs-resources';
import {
  addBreaksSelectPass,
  createPanelBlock,
  escapeHtml,
  formatCompact,
  formatSparkline,
  getPackedColorCss,
  getRampPalette,
  packColor,
  PackedColorRasterLayer
} from './classification-layers';

const GRID_SIZE: readonly [number, number] = [160, 100];
const HEXAGON_GRID_SIZE: readonly [number, number] = [72, 52];
const MAXIMUM_CELL_COUNT = GRID_SIZE[0] * GRID_SIZE[1];
const SQRT3 = Math.sqrt(3);
/** Compile-time class capacity of the main classification (and of the color palette). */
const MAXIMUM_CLASS_COUNT = 9;
/** Compile-time class capacity per bivariate axis. */
const MAXIMUM_BIVARIATE_CLASS_COUNT = 4;
const QUANTILE_PROBABILITIES = [0.25, 0.5, 0.75] as const;
const PROFILE_HISTOGRAM_BINS = 20;
const PROFILE_COLUMN_NAMES = ['Count', 'Speed'] as const;
const READBACK_INTERVAL_FRAMES = 8;
const ENCODED_FRAMES_AFTER_CHANGE = 3;
const NATURAL_BREAKS_BIN_COUNT = 512;
/** Fixed edges of the `custom` method, as a user-defined legend would give them. */
const CUSTOM_EDGES = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256] as const;
const NO_DATA_COLOR = packColor(0, 0, 0, 0);

type Binning = 'grid' | 'hexagon';
type ScaleChoice = 'threshold' | 'linear' | 'sqrt' | 'log' | 'quantize';

const METHOD_LABELS: Record<GPUClassBreaksMethod, string> = {
  quantile: 'Quantile',
  'equal-interval': 'Equal interval',
  'standard-deviation': 'Standard deviation',
  'head-tail': 'Head / tail breaks',
  'box-plot': 'Box plot (6 classes)',
  'maximum-breaks': 'Maximum breaks',
  'natural-breaks': 'Natural breaks (Jenks)',
  custom: 'Custom edges (fixed)'
};

const SCALE_LABELS: Record<ScaleChoice, string> = {
  threshold: 'Classified (threshold)',
  linear: 'Linear',
  sqrt: 'Square root',
  log: 'Log',
  quantize: 'Quantize'
};

const PALETTES: Record<string, {label: string; stops: readonly (readonly number[])[]}> = {
  viridis: {
    label: 'Viridis',
    stops: [
      [68, 1, 84],
      [59, 82, 139],
      [33, 145, 140],
      [94, 201, 98],
      [253, 231, 37]
    ]
  },
  ylorrd: {
    label: 'Yellow orange red',
    stops: [
      [255, 255, 178],
      [254, 204, 92],
      [253, 141, 60],
      [240, 59, 32],
      [189, 0, 38]
    ]
  },
  blues: {
    label: 'Blues',
    stops: [
      [222, 235, 247],
      [158, 202, 225],
      [66, 146, 198],
      [8, 81, 156],
      [8, 48, 107]
    ]
  },
  spectral: {
    label: 'Spectral',
    stops: [
      [94, 79, 162],
      [102, 194, 165],
      [254, 224, 139],
      [244, 109, 67],
      [158, 1, 66]
    ]
  },
  magma: {
    label: 'Magma',
    stops: [
      [0, 0, 4],
      [81, 18, 124],
      [183, 55, 121],
      [252, 137, 97],
      [252, 253, 191]
    ]
  }
};

/** Stevens-style bivariate corners: low/low, high X, high Y, high/high. */
const BIVARIATE_CORNERS = {
  lowLow: [232, 232, 232],
  highX: [200, 90, 90],
  highY: [100, 172, 190],
  highHigh: [87, 66, 73]
} as const;

/** Speed in km/h of each trip segment, assigned to its start vertex; the last vertex repeats it. */
function getVertexSpeeds(
  positions: Float32Array,
  timestamps: Float32Array,
  tripOffsets: Uint32Array
): Float32Array {
  const speeds = new Float32Array(timestamps.length);
  for (let trip = 0; trip + 1 < tripOffsets.length; trip++) {
    const first = tripOffsets[trip];
    const last = tripOffsets[trip + 1];
    let speed = 0;
    for (let vertex = first; vertex < last; vertex++) {
      if (vertex + 1 < last) {
        const seconds = timestamps[vertex + 1] - timestamps[vertex];
        if (seconds > 0) {
          const distance = Math.hypot(
            positions[vertex * 2 + 2] - positions[vertex * 2],
            positions[vertex * 2 + 3] - positions[vertex * 2 + 1]
          );
          speed = Math.min((distance / seconds) * 3.6, 120);
        }
      }
      speeds[vertex] = speed;
    }
  }
  return speeds;
}

type Summary = {
  breaks: Float32Array;
  classCount: number;
  classCounts: Uint32Array;
  filterBounds: Float32Array;
  quantiles: Float32Array;
  validCount: number;
  bivariateCounts: Uint32Array;
  bivariateBreaksX: Float32Array;
  bivariateBreaksY: Float32Array;
  profileStatistics: Float32Array;
  profileHistograms: Uint32Array;
};

export const classificationMode: MapGraphsModeDefinition = {
  id: 'classification',
  title: 'Classes',
  recipes: [
    'GPUPointDensity',
    'GPUColumnQuantiles',
    'GPUClassBreaks',
    'GPUColorScale',
    'GPUBivariateClassification',
    'GPUColumnProfile'
  ],
  description:
    'Kepler-style choropleth of New York trip vertices: bin, percentile-filter, classify and ' +
    'color on the GPU. Change method, class count, palette or percentile range live; the ' +
    'colors are drawn from the GPU buffer with no readback.',
  initialViewState: {longitude: -73.985, latitude: 40.735, zoom: 12.2},

  async create(context) {
    const trips = await context.data.getNewYorkTrips();
    context.signal.throwIfAborted();
    const {device} = context;
    const projection = new LocalMetricProjection(trips.origin);
    const resources = new MapGraphsResources(device, 'classification');
    const pointCount = trips.vertexTimestamps.length;
    const speeds = getVertexSpeeds(
      trips.vertexPositions,
      trips.vertexTimestamps,
      trips.tripOffsets
    );

    const positionsBuffer = resources.createBuffer('positions', trips.vertexPositions);
    const speedsBuffer = resources.createBuffer('speeds', speeds);
    const bounds = resources.createParameterBuffer('bounds', 'float32', 4);
    const hexagonRadius = resources.createParameterBuffer(
      'hexagon-radius',
      'float32',
      1,
      Float32Array.of(50)
    );
    const quantileParameters = resources.createParameterBuffer(
      'quantile-parameters',
      'float32',
      getGPUColumnQuantilesParameterLength(QUANTILE_PROBABILITIES.length)
    );
    const breaksParameters = resources.createParameterBuffer(
      'breaks-parameters',
      'float32',
      getGPUClassBreaksParameterLength(MAXIMUM_CLASS_COUNT)
    );
    const scaleParameters = resources.createParameterBuffer(
      'scale-parameters',
      'float32',
      GPU_COLOR_SCALE_PARAMETER_LENGTH
    );
    const axisParameterLength = getGPUClassBreaksParameterLength(MAXIMUM_BIVARIATE_CLASS_COUNT);
    const axisXParameters = resources.createParameterBuffer(
      'axis-x-parameters',
      'float32',
      axisParameterLength
    );
    const axisYParameters = resources.createParameterBuffer(
      'axis-y-parameters',
      'float32',
      axisParameterLength
    );
    const bivariateParameters = resources.createParameterBuffer(
      'bivariate-parameters',
      'float32',
      GPU_BIVARIATE_CLASSIFICATION_PARAMETER_LENGTH
    );
    const paletteBuffer = resources.createBuffer('palette', MAXIMUM_CLASS_COUNT * 4);
    const bivariatePaletteBuffer = resources.createBuffer(
      'bivariate-palette',
      MAXIMUM_BIVARIATE_CLASS_COUNT ** 2 * 4
    );

    // Caller-owned outputs, sized for the larger (square) lattice; graphs import prefixes of them.
    const cellFloats = (name: string) => resources.createBuffer(name, MAXIMUM_CELL_COUNT * 4);
    const countValues = cellFloats('count-values');
    const cellCounts = cellFloats('cell-counts');
    const meanSpeeds = cellFloats('mean-speeds');
    const filterMask = cellFloats('filter-mask');
    const colors = cellFloats('colors');
    const classIndices = cellFloats('class-indices');
    const bivariateColors = cellFloats('bivariate-colors');
    const filterBounds = resources.createBuffer('filter-bounds', 8);
    const quantiles = resources.createBuffer('quantiles', QUANTILE_PROBABILITIES.length * 4);
    const validCount = resources.createBuffer('valid-count', 4);
    const breaks = resources.createBuffer('breaks', (MAXIMUM_CLASS_COUNT + 1) * 4);
    const classCountBuffer = resources.createBuffer('class-count', 4);
    const scaleClassCounts = resources.createBuffer('scale-class-counts', MAXIMUM_CLASS_COUNT * 4);
    const axisBreaksX = resources.createBuffer(
      'axis-breaks-x',
      (MAXIMUM_BIVARIATE_CLASS_COUNT + 1) * 4
    );
    const axisBreaksY = resources.createBuffer(
      'axis-breaks-y',
      (MAXIMUM_BIVARIATE_CLASS_COUNT + 1) * 4
    );
    const bivariateCounts = resources.createBuffer(
      'bivariate-counts',
      MAXIMUM_BIVARIATE_CLASS_COUNT ** 2 * 4
    );
    const profileStatistics = resources.createBuffer(
      'profile-statistics',
      PROFILE_COLUMN_NAMES.length * GPU_COLUMN_PROFILE_STATISTIC_COUNT * 4
    );
    const profileHistograms = resources.createBuffer(
      'profile-histograms',
      PROFILE_COLUMN_NAMES.length * PROFILE_HISTOGRAM_BINS * 4
    );
    const readbackLayout: {buffer: Buffer; byteLength: number}[] = [
      {buffer: breaks, byteLength: (MAXIMUM_CLASS_COUNT + 1) * 4},
      {buffer: classCountBuffer, byteLength: 4},
      {buffer: scaleClassCounts, byteLength: MAXIMUM_CLASS_COUNT * 4},
      {buffer: filterBounds, byteLength: 8},
      {buffer: quantiles, byteLength: QUANTILE_PROBABILITIES.length * 4},
      {buffer: validCount, byteLength: 4},
      {buffer: bivariateCounts, byteLength: MAXIMUM_BIVARIATE_CLASS_COUNT ** 2 * 4},
      {buffer: axisBreaksX, byteLength: (MAXIMUM_BIVARIATE_CLASS_COUNT + 1) * 4},
      {buffer: axisBreaksY, byteLength: (MAXIMUM_BIVARIATE_CLASS_COUNT + 1) * 4},
      {
        buffer: profileStatistics,
        byteLength: PROFILE_COLUMN_NAMES.length * GPU_COLUMN_PROFILE_STATISTIC_COUNT * 4
      },
      {
        buffer: profileHistograms,
        byteLength: PROFILE_COLUMN_NAMES.length * PROFILE_HISTOGRAM_BINS * 4
      }
    ];
    const readbackByteLength = readbackLayout.reduce((total, part) => total + part.byteLength, 0);
    const readbackRing = resources.track(
      new GPUReadbackRing(device, {id: 'classification-summary', byteLength: readbackByteLength})
    );

    // Control state. Everything but `binning` is a buffer write.
    let binning: Binning = 'grid';
    let method: GPUClassBreaksMethod = 'quantile';
    let requestedClassCount = 6;
    let scale: ScaleChoice = 'threshold';
    let smoothBlend = false;
    let paletteKey = 'viridis';
    let lowerPercentile = 0;
    let upperPercentile = 100;
    let bivariate = false;
    let bivariateClassCount = 3;
    let showPoints = false;
    let writtenClassCount = 0;
    let activePalette: Uint32Array = new Uint32Array(0);
    let latestSummary: Summary | null = null;
    let dirtyFrames = ENCODED_FRAMES_AFTER_CHANGE;
    let lastBoundsKey = '';
    let lastReadbackFrame = -READBACK_INTERVAL_FRAMES;
    let readbackWanted = true;
    let readbackPending = false;
    let destroyed = false;

    let main: CompiledGPUCommandGraph<void> | null = null;
    let bivariateGraph: CompiledGPUCommandGraph<void> | null = null;
    let profileGraph: CompiledGPUCommandGraph<void> | null = null;

    function buildGraphs(): void {
      for (const graph of [main, bivariateGraph, profileGraph]) {
        if (graph) resources.release(graph);
      }
      const gridSize = binning === 'grid' ? GRID_SIZE : HEXAGON_GRID_SIZE;
      const cellCount = gridSize[0] * gridSize[1];
      // A graph rejects two imports that resolve to one physical buffer, so each graph imports a
      // buffer once and every node that reads or writes it shares that view.
      const createImporter = (graph: GPUCommandGraph<void>) => {
        const floatViews = new Map<Buffer, GraphDataView<'float32'>>();
        const wordViews = new Map<Buffer, GraphDataView<'uint32'>>();
        return {
          float: (buffer: Buffer, length: number) => {
            let imported = floatViews.get(buffer);
            if (!imported) {
              imported = importGraphBuffer(graph, buffer.id, buffer, 'float32', length);
              floatViews.set(buffer, imported);
            }
            return imported;
          },
          word: (buffer: Buffer, length: number) => {
            let imported = wordViews.get(buffer);
            if (!imported) {
              imported = importGraphBuffer(graph, buffer.id, buffer, 'uint32', length);
              wordViews.set(buffer, imported);
            }
            return imported;
          }
        };
      };

      // Main graph: density, percentile filter, class breaks, color scale.
      const graph = new GPUCommandGraph<void>(device, {id: `classification-${binning}`});
      const bind = createImporter(graph);
      const cellValues = bind.float(countValues, cellCount);
      const occupied = bind.word(cellCounts, cellCount);
      const filter = bind.word(filterMask, cellCount);
      graph.add(
        new GPUPointDensity({
          id: 'density',
          positions: importGraphBuffer(
            graph,
            'positions',
            positionsBuffer,
            'float32x2',
            pointCount
          ),
          weights: importGraphBuffer(graph, 'speeds', speedsBuffer, 'float32', pointCount),
          bounds: bounds.importToGraph(graph),
          gridSize,
          binning,
          ...(binning === 'hexagon' ? {hexagonRadius: hexagonRadius.importToGraph(graph)} : {}),
          statistic: 'count',
          output: {
            values: cellValues,
            counts: occupied,
            means: bind.float(meanSpeeds, cellCount)
          }
        })
      );
      graph.add(
        new GPUColumnQuantiles({
          id: 'percentile-filter',
          values: cellValues,
          mask: occupied,
          parameters: quantileParameters.importToGraph(graph),
          quantileCount: QUANTILE_PROBABILITIES.length,
          output: {
            quantiles: bind.float(quantiles, QUANTILE_PROBABILITIES.length),
            validCount: bind.word(validCount, 1),
            filterMask: filter,
            filterBounds: bind.float(filterBounds, 2)
          }
        })
      );
      // The finish kernel of one GPUClassBreaks that compiles quantile, standard-deviation and
      // head/tail together binds nine storage buffers, one over the default limit of eight. Head/tail
      // therefore runs in a second instance and a small pass picks the active method's edges.
      const breaksParameterView = breaksParameters.importToGraph(graph);
      const primaryBreaks = createTransientView(
        graph,
        'primary-breaks',
        'float32',
        MAXIMUM_CLASS_COUNT + 1
      );
      const primaryClassCount = createTransientView(graph, 'primary-class-count', 'uint32', 1);
      const headTailBreaks = createTransientView(
        graph,
        'head-tail-breaks',
        'float32',
        MAXIMUM_CLASS_COUNT + 1
      );
      const headTailClassCount = createTransientView(graph, 'head-tail-class-count', 'uint32', 1);
      graph.add(
        new GPUClassBreaks({
          id: 'breaks',
          values: cellValues,
          mask: filter,
          parameters: breaksParameterView,
          maximumClassCount: MAXIMUM_CLASS_COUNT,
          methods: GPU_CLASS_BREAKS_METHODS.filter(value => value !== 'head-tail'),
          naturalBreaksBinCount: NATURAL_BREAKS_BIN_COUNT,
          output: {breaks: primaryBreaks, classCount: primaryClassCount}
        })
      );
      graph.add(
        new GPUClassBreaks({
          id: 'head-tail',
          values: cellValues,
          mask: filter,
          parameters: breaksParameterView,
          maximumClassCount: MAXIMUM_CLASS_COUNT,
          methods: ['head-tail'],
          output: {breaks: headTailBreaks, classCount: headTailClassCount}
        })
      );
      const breaksView = bind.float(breaks, MAXIMUM_CLASS_COUNT + 1);
      const classCountView = bind.word(classCountBuffer, 1);
      addBreaksSelectPass(graph, {
        id: 'select-breaks',
        maximumClassCount: MAXIMUM_CLASS_COUNT,
        parameters: breaksParameterView,
        alternateMethodCode: GPU_CLASS_BREAKS_METHOD_CODES['head-tail'],
        primaryBreaks,
        primaryClassCount,
        alternateBreaks: headTailBreaks,
        alternateClassCount: headTailClassCount,
        breaks: breaksView,
        classCount: classCountView
      });
      graph.add(
        new GPUColorScale({
          id: 'scale',
          values: cellValues,
          mask: filter,
          domain: breaksView,
          domainCount: classCountView,
          palette: bind.word(paletteBuffer, MAXIMUM_CLASS_COUNT),
          parameters: scaleParameters.importToGraph(graph),
          maximumDomainCount: MAXIMUM_CLASS_COUNT + 1,
          maximumPaletteCount: MAXIMUM_CLASS_COUNT,
          output: {
            colors: bind.word(colors, cellCount),
            classIndices: bind.word(classIndices, cellCount),
            classCounts: bind.word(scaleClassCounts, MAXIMUM_CLASS_COUNT)
          }
        })
      );
      main = resources.track(graph.compile());

      // Bivariate graph: count by mean speed, encoded only while the toggle is on.
      const biGraph = new GPUCommandGraph<void>(device, {
        id: `classification-bivariate-${binning}`
      });
      const biBind = createImporter(biGraph);
      const biCount = biBind.float(countValues, cellCount);
      const biSpeed = biBind.float(meanSpeeds, cellCount);
      const biFilter = biBind.word(filterMask, cellCount);
      const biBreaksX = biBind.float(axisBreaksX, MAXIMUM_BIVARIATE_CLASS_COUNT + 1);
      const biBreaksY = biBind.float(axisBreaksY, MAXIMUM_BIVARIATE_CLASS_COUNT + 1);
      for (const [axis, values, axisBreaks, parameters] of [
        ['x', biCount, biBreaksX, axisXParameters],
        ['y', biSpeed, biBreaksY, axisYParameters]
      ] as const) {
        biGraph.add(
          new GPUClassBreaks({
            id: `axis-${axis}`,
            values,
            mask: biFilter,
            parameters: parameters.importToGraph(biGraph),
            maximumClassCount: MAXIMUM_BIVARIATE_CLASS_COUNT,
            methods: ['quantile', 'equal-interval'],
            output: {
              breaks: axisBreaks,
              classCount: createTransientView(biGraph, `axis-${axis}-class-count`, 'uint32', 1)
            }
          })
        );
      }
      biGraph.add(
        new GPUBivariateClassification({
          id: 'bivariate',
          valuesX: biCount,
          valuesY: biSpeed,
          mask: biFilter,
          breaksX: biBreaksX,
          breaksY: biBreaksY,
          palette: biBind.word(bivariatePaletteBuffer, MAXIMUM_BIVARIATE_CLASS_COUNT ** 2),
          parameters: bivariateParameters.importToGraph(biGraph),
          maximumClassCount: MAXIMUM_BIVARIATE_CLASS_COUNT,
          output: {
            colors: biBind.word(bivariateColors, cellCount),
            classCounts: biBind.word(bivariateCounts, MAXIMUM_BIVARIATE_CLASS_COUNT ** 2)
          }
        })
      );
      bivariateGraph = resources.track(biGraph.compile());

      // Profile graph: dataset statistics of the filtered columns.
      const profile = new GPUCommandGraph<void>(device, {id: `classification-profile-${binning}`});
      const profileBind = createImporter(profile);
      profile.add(
        new GPUColumnProfile({
          id: 'profile',
          columns: [
            {values: profileBind.float(countValues, cellCount)},
            {values: profileBind.float(meanSpeeds, cellCount)}
          ],
          mask: profileBind.word(filterMask, cellCount),
          histogramBinCount: PROFILE_HISTOGRAM_BINS,
          hyperLogLogPrecision: 8,
          output: {
            statistics: profileBind.float(
              profileStatistics,
              PROFILE_COLUMN_NAMES.length * GPU_COLUMN_PROFILE_STATISTIC_COUNT
            ),
            histograms: profileBind.word(
              profileHistograms,
              PROFILE_COLUMN_NAMES.length * PROFILE_HISTOGRAM_BINS
            )
          }
        })
      );
      profileGraph = resources.track(profile.compile());
      gridReadout?.setValue(
        `${gridSize[0]} × ${gridSize[1]} ${binning === 'grid' ? 'cells' : 'hexagons'}`
      );
    }

    function writePalette(classCount: number): void {
      writtenClassCount = classCount;
      activePalette = getRampPalette(PALETTES[paletteKey].stops, classCount, 235);
      const padded = new Uint32Array(MAXIMUM_CLASS_COUNT);
      padded.set(activePalette);
      paletteBuffer.write(padded);
    }

    function writeBivariatePalette(): void {
      const palette = new Uint32Array(MAXIMUM_BIVARIATE_CLASS_COUNT ** 2);
      const n = bivariateClassCount;
      for (let y = 0; y < n; y++) {
        for (let x = 0; x < n; x++) {
          const tx = x / (n - 1);
          const ty = y / (n - 1);
          const mix = (channel: 0 | 1 | 2) =>
            Math.round(
              BIVARIATE_CORNERS.lowLow[channel] * (1 - tx) * (1 - ty) +
                BIVARIATE_CORNERS.highX[channel] * tx * (1 - ty) +
                BIVARIATE_CORNERS.highY[channel] * (1 - tx) * ty +
                BIVARIATE_CORNERS.highHigh[channel] * tx * ty
            );
          palette[y * n + x] = packColor(mix(0), mix(1), mix(2), 235);
        }
      }
      bivariatePaletteBuffer.write(palette);
    }

    /** Rewrites every per-frame parameter buffer from the control state. */
    function writeParameters(): void {
      quantileParameters.write(
        getGPUColumnQuantilesParameterValues({
          quantiles: QUANTILE_PROBABILITIES,
          filterRange: [lowerPercentile / 100, upperPercentile / 100]
        })
      );
      breaksParameters.write(
        getGPUClassBreaksParameterValues(
          {
            method,
            classCount: requestedClassCount,
            customEdges: CUSTOM_EDGES.slice(0, requestedClassCount + 1)
          },
          MAXIMUM_CLASS_COUNT
        )
      );
      const classCount = latestSummary?.classCount || requestedClassCount;
      if (classCount !== writtenClassCount) writePalette(Math.min(classCount, MAXIMUM_CLASS_COUNT));
      scaleParameters.write(
        getGPUColorScaleParameterValues({
          scale: (scale === 'threshold' ? 'threshold' : scale) as GPUColorScaleType,
          domainCount: writtenClassCount + 1,
          paletteCount: writtenClassCount,
          interpolation: smoothBlend && scale !== 'threshold' ? 'linear' : 'step',
          clamp: true,
          noDataColor: NO_DATA_COLOR
        })
      );
      axisXParameters.write(
        getGPUClassBreaksParameterValues(
          {method: 'quantile', classCount: bivariateClassCount},
          MAXIMUM_BIVARIATE_CLASS_COUNT
        )
      );
      axisYParameters.write(
        getGPUClassBreaksParameterValues(
          {method: 'quantile', classCount: bivariateClassCount},
          MAXIMUM_BIVARIATE_CLASS_COUNT
        )
      );
      bivariateParameters.write(
        getGPUBivariateClassificationParameterValues({
          classCountX: bivariateClassCount,
          classCountY: bivariateClassCount,
          noDataColor: NO_DATA_COLOR
        })
      );
      writeBivariatePalette();
      dirtyFrames = ENCODED_FRAMES_AFTER_CHANGE;
      readbackWanted = true;
    }

    // Controls.
    context.controls.addSelect<Binning>({
      label: 'Binning (compile-time, rebuilds)',
      options: [
        {value: 'grid', label: `Square grid ${GRID_SIZE[0]} × ${GRID_SIZE[1]}`},
        {value: 'hexagon', label: `Hexagons ${HEXAGON_GRID_SIZE[0]} × ${HEXAGON_GRID_SIZE[1]}`}
      ],
      value: binning,
      onChange: value => {
        binning = value;
        latestSummary = null;
        // Rewrites the bounds and the hexagon radius on the next frame.
        lastBoundsKey = '';
        buildGraphs();
        writeParameters();
        context.updateLayers();
      }
    });
    context.controls.addSelect<GPUClassBreaksMethod>({
      label: 'Class method (per-frame code)',
      options: GPU_CLASS_BREAKS_METHODS.map(value => ({value, label: METHOD_LABELS[value]})),
      value: method,
      onChange: value => {
        method = value;
        writeParameters();
      }
    });
    context.controls.addSlider({
      label: `Classes (up to ${MAXIMUM_CLASS_COUNT} compiled)`,
      min: 2,
      max: MAXIMUM_CLASS_COUNT,
      step: 1,
      value: requestedClassCount,
      onChange: value => {
        requestedClassCount = value;
        latestSummary = null;
        writeParameters();
      }
    });
    context.controls.addSelect<ScaleChoice>({
      label: 'Color scale (per-frame code)',
      options: Object.entries(SCALE_LABELS).map(([value, label]) => ({
        value: value as ScaleChoice,
        label
      })),
      value: scale,
      onChange: value => {
        scale = value;
        writeParameters();
      }
    });
    context.controls.addToggle({
      label: 'Blend between palette colors (continuous scales)',
      value: smoothBlend,
      onChange: value => {
        smoothBlend = value;
        writeParameters();
      }
    });
    context.controls.addSelect<string>({
      label: 'Palette (palette buffer)',
      options: Object.entries(PALETTES).map(([value, {label}]) => ({value, label})),
      value: paletteKey,
      onChange: value => {
        paletteKey = value;
        writePalette(writtenClassCount || requestedClassCount);
        writeParameters();
      }
    });
    context.controls.addSlider({
      label: 'Lower percentile filter',
      min: 0,
      max: 49,
      step: 1,
      value: lowerPercentile,
      format: value => `${value}%`,
      onChange: value => {
        lowerPercentile = value;
        writeParameters();
      }
    });
    context.controls.addSlider({
      label: 'Upper percentile filter',
      min: 51,
      max: 100,
      step: 1,
      value: upperPercentile,
      format: value => `${value}%`,
      onChange: value => {
        upperPercentile = value;
        writeParameters();
      }
    });
    context.controls.addToggle({
      label: 'Bivariate: cell count × mean speed',
      value: bivariate,
      onChange: value => {
        bivariate = value;
        writeParameters();
        context.updateLayers();
      }
    });
    context.controls.addSlider({
      label: 'Bivariate classes per axis',
      min: 2,
      max: MAXIMUM_BIVARIATE_CLASS_COUNT,
      step: 1,
      value: bivariateClassCount,
      onChange: value => {
        bivariateClassCount = value;
        writeParameters();
      }
    });
    context.controls.addToggle({
      label: 'Show source points',
      value: showPoints,
      onChange: value => {
        showPoints = value;
        context.updateLayers();
      }
    });
    context.controls.addNote(
      'Empty cells are masked out. The percentile filter trims occupied cells before the ' +
        'breaks are computed; the custom method uses fixed edges 0, 1, 2, 4 … 256.'
    );
    const legendBlock = createPanelBlock();
    context.controls.addReadout('Points', formatCount(pointCount));
    const gridReadout = context.controls.addReadout('Grid');
    const cellReadout = context.controls.addReadout('Cell size');
    const validReadout = context.controls.addReadout('Occupied cells');
    const filterReadout = context.controls.addReadout('Filter bounds (count)');
    const quartileReadout = context.controls.addReadout('Quartiles (count)');
    const classReadout = context.controls.addReadout('Classes produced');
    const profileReadouts = PROFILE_COLUMN_NAMES.map(name => ({
      summary: context.controls.addReadout(name),
      histogram: context.controls.addReadout(`${name.split(' ')[0]} hist.`)
    }));
    context.controls.addReadout('Data', trips.attribution);

    // The palette depends on the requested count until the first readback reports the real one.
    writePalette(requestedClassCount);
    buildGraphs();
    writeParameters();

    function renderLegend(summary: Summary): void {
      if (!legendBlock) return;
      const html: string[] = [];
      if (bivariate) {
        const n = bivariateClassCount;
        html.push(
          `<div>Bivariate: cell count (→) by mean speed (↑), ${n} × ${n} quantile classes</div>`
        );
        html.push(
          `<div style="display:inline-grid;grid-template-columns:repeat(${n},auto);gap:2px;margin-top:4px">`
        );
        for (let y = n - 1; y >= 0; y--) {
          for (let x = 0; x < n; x++) {
            const tx = x / (n - 1);
            const ty = y / (n - 1);
            const mix = (channel: 0 | 1 | 2) =>
              Math.round(
                BIVARIATE_CORNERS.lowLow[channel] * (1 - tx) * (1 - ty) +
                  BIVARIATE_CORNERS.highX[channel] * tx * (1 - ty) +
                  BIVARIATE_CORNERS.highY[channel] * (1 - tx) * ty +
                  BIVARIATE_CORNERS.highHigh[channel] * tx * ty
              );
            const count = summary.bivariateCounts[y * n + x];
            const dark = mix(0) + mix(1) + mix(2) < 330;
            html.push(
              `<span style="width:44px;height:26px;display:flex;align-items:center;justify-content:center;border-radius:3px;font:10px ui-monospace,monospace;color:${dark ? '#fff' : '#111'};background:rgb(${mix(0)},${mix(1)},${mix(2)})">${formatCount(count)}</span>`
            );
          }
        }
        html.push('</div>');
        html.push(
          `<div style="margin-top:3px">count edges ${Array.from(summary.bivariateBreaksX.subarray(0, n + 1), formatCompact).join(' · ')}</div>`
        );
        html.push(
          `<div>speed edges ${Array.from(summary.bivariateBreaksY.subarray(0, n + 1), formatCompact).join(' · ')} km/h</div>`
        );
      } else {
        const k = Math.min(summary.classCount, MAXIMUM_CLASS_COUNT);
        html.push(
          `<div>${escapeHtml(METHOD_LABELS[method])} · ${k} classes · ${escapeHtml(SCALE_LABELS[scale])}: cells per class</div>`
        );
        const edges = getLegendEdges(summary.breaks, k, scale);
        let total = 0;
        for (let index = 0; index < k; index++) total += summary.classCounts[index];
        for (let index = 0; index < k; index++) {
          const share = total > 0 ? summary.classCounts[index] / total : 0;
          html.push(
            `<div style="display:flex;align-items:center;gap:6px;margin-top:2px">` +
              `<span style="width:12px;height:12px;border-radius:2px;flex:none;background:${getPackedColorCss(activePalette[index] ?? 0)}"></span>` +
              `<span style="width:116px;flex:none;font:10px ui-monospace,monospace">${formatCompact(edges[index])} – ${formatCompact(edges[index + 1])}</span>` +
              `<span style="flex:1;height:6px;background:rgba(255,255,255,.08);border-radius:3px"><span style="display:block;height:6px;border-radius:3px;width:${(share * 100).toFixed(1)}%;background:${getPackedColorCss(activePalette[index] ?? 0)}"></span></span>` +
              `<span style="width:40px;text-align:right;font:10px ui-monospace,monospace">${formatCount(summary.classCounts[index])}</span></div>`
          );
        }
      }
      legendBlock.innerHTML = html.join('');
    }

    function applySummary(summary: Summary): void {
      const previousClassCount = latestSummary?.classCount;
      latestSummary = summary;
      validReadout.setValue(formatCount(summary.validCount));
      filterReadout.setValue(
        `${formatCompact(summary.filterBounds[0])} – ${formatCompact(summary.filterBounds[1])}`
      );
      quartileReadout.setValue(Array.from(summary.quantiles, formatCompact).join(' / '));
      classReadout.setValue(`${summary.classCount} of ${requestedClassCount} requested`);
      PROFILE_COLUMN_NAMES.forEach((_, column) => {
        const statistic = (field: keyof typeof GPU_COLUMN_PROFILE_STATISTIC) =>
          summary.profileStatistics[
            column * GPU_COLUMN_PROFILE_STATISTIC_COUNT + GPU_COLUMN_PROFILE_STATISTIC[field]
          ];
        profileReadouts[column].summary.setValue(
          `μ ${formatCompact(statistic('mean'))} σ ${formatCompact(statistic('standardDeviation'))} · ~${formatCount(statistic('distinctEstimate'))} uniq`
        );
        profileReadouts[column].histogram.setValue(
          `${formatCompact(statistic('minimum'))} ${formatSparkline(
            summary.profileHistograms.subarray(
              column * PROFILE_HISTOGRAM_BINS,
              (column + 1) * PROFILE_HISTOGRAM_BINS
            )
          )} ${formatCompact(statistic('maximum'))}`
        );
      });
      // Methods such as head/tail and maximum breaks can produce fewer classes than requested:
      // recolor with a palette of the real size.
      if (summary.classCount > 0 && summary.classCount !== previousClassCount) {
        if (summary.classCount !== writtenClassCount) {
          writePalette(summary.classCount);
          writeParameters();
        }
      }
      renderLegend(summary);
    }

    async function readSummary(commandEncoder: CommandEncoder): Promise<void> {
      const ticket = readbackRing.tryAcquire();
      if (!ticket) return;
      let offset = 0;
      for (const part of readbackLayout) {
        commandEncoder.copyBufferToBuffer({
          sourceBuffer: part.buffer,
          destinationBuffer: ticket.buffer,
          destinationOffset: offset,
          size: part.byteLength
        });
        offset += part.byteLength;
      }
      ticket.markEncoded({byteOffset: 0, byteLength: readbackByteLength});
      readbackPending = true;
      try {
        const bytes = await ticket.read();
        if (destroyed) return;
        const copy = new ArrayBuffer(bytes.byteLength);
        new Uint8Array(copy).set(bytes);
        let cursor = 0;
        const floats = (length: number) => {
          const values = new Float32Array(copy, cursor, length);
          cursor += length * 4;
          return values;
        };
        const words = (length: number) => {
          const values = new Uint32Array(copy, cursor, length);
          cursor += length * 4;
          return values;
        };
        const breaksValues = floats(MAXIMUM_CLASS_COUNT + 1);
        const classCount = words(1)[0];
        const classCounts = words(MAXIMUM_CLASS_COUNT);
        const filterBoundValues = floats(2);
        const quantileValues = floats(QUANTILE_PROBABILITIES.length);
        const valid = words(1)[0];
        const bivariateCountValues = words(MAXIMUM_BIVARIATE_CLASS_COUNT ** 2);
        const breaksX = floats(MAXIMUM_BIVARIATE_CLASS_COUNT + 1);
        const breaksY = floats(MAXIMUM_BIVARIATE_CLASS_COUNT + 1);
        const profileValues = floats(
          PROFILE_COLUMN_NAMES.length * GPU_COLUMN_PROFILE_STATISTIC_COUNT
        );
        const histogramValues = words(PROFILE_COLUMN_NAMES.length * PROFILE_HISTOGRAM_BINS);
        applySummary({
          breaks: breaksValues,
          classCount,
          classCounts,
          filterBounds: filterBoundValues,
          quantiles: quantileValues,
          validCount: valid,
          bivariateCounts: bivariateCountValues,
          bivariateBreaksX: breaksX,
          bivariateBreaksY: breaksY,
          profileStatistics: profileValues,
          profileHistograms: histogramValues
        });
      } catch {
        // The ring or device was destroyed while the read was in flight.
      } finally {
        readbackPending = false;
      }
    }

    const instance: MapGraphsModeInstance = {
      getCompiledGraphs: () =>
        [main, bivariateGraph, profileGraph].filter(
          (graph): graph is CompiledGPUCommandGraph<void> => graph !== null
        ),
      encode(commandEncoder, frame) {
        if (!main || !bivariateGraph || !profileGraph) return;
        const viewBounds = getViewportMetricBounds(frame.viewport, projection);
        let currentHexagonRadius = 0;
        if (binning === 'hexagon') {
          const [columns, rows] = HEXAGON_GRID_SIZE;
          currentHexagonRadius = Math.max(
            (viewBounds[2] - viewBounds[0]) / (SQRT3 * (columns - 1)),
            (viewBounds[3] - viewBounds[1]) / (1.5 * (rows - 1))
          );
        }
        const boundsKey = `${viewBounds.join(',')}`;
        if (boundsKey !== lastBoundsKey) {
          lastBoundsKey = boundsKey;
          bounds.write(Float32Array.from(viewBounds));
          if (binning === 'hexagon') {
            hexagonRadius.write(Float32Array.of(currentHexagonRadius));
            cellReadout.setValue(`${currentHexagonRadius.toFixed(0)} m radius`);
          } else {
            cellReadout.setValue(
              `${((viewBounds[2] - viewBounds[0]) / GRID_SIZE[0]).toFixed(0)} m`
            );
          }
          dirtyFrames = ENCODED_FRAMES_AFTER_CHANGE;
          readbackWanted = true;
        }
        // The lattice only depends on the camera and the control buffers, so a still map with
        // unchanged controls keeps its outputs and skips the whole graph.
        if (dirtyFrames > 0 || frame.frameIndex < 3) {
          main.encode(commandEncoder, {parameters: undefined});
          if (bivariate) bivariateGraph.encode(commandEncoder, {parameters: undefined});
          profileGraph.encode(commandEncoder, {parameters: undefined});
          dirtyFrames = Math.max(0, dirtyFrames - 1);
        }
        if (
          readbackWanted &&
          !readbackPending &&
          frame.frameIndex - lastReadbackFrame >= READBACK_INTERVAL_FRAMES &&
          dirtyFrames === 0
        ) {
          lastReadbackFrame = frame.frameIndex;
          readbackWanted = false;
          void readSummary(commandEncoder);
        }
      },
      getLayers() {
        const coordinateOrigin: [number, number, number] = [trips.origin[0], trips.origin[1], 0];
        const layers: Layer[] = [
          new PackedColorRasterLayer({
            id: `classification-${binning}-${bivariate ? 'bivariate' : 'classes'}`,
            coordinateOrigin,
            gridSize: binning === 'grid' ? GRID_SIZE : HEXAGON_GRID_SIZE,
            bounds: bounds.buffer,
            binning,
            hexagonRadius: hexagonRadius.buffer,
            values: bivariate ? bivariateColors : colors,
            valueFormat: 'uint32'
          })
        ];
        if (showPoints) {
          layers.push(
            new MapGraphsPointLayer({
              id: 'classification-points',
              coordinateOrigin,
              positions: positionsBuffer,
              instanceCount: pointCount,
              radiusPixels: 1.2,
              color: [120, 220, 255, 120]
            })
          );
        }
        return layers;
      },
      destroy() {
        destroyed = true;
        legendBlock?.remove();
        resources.destroy();
      }
    };
    return instance;
  }
};

/**
 * Legend edges for `classCount` classes. A classified scale uses the breaks themselves. A
 * continuous scale splits the transformed domain `[breaks[0], breaks[last]]` evenly, as
 * `GPUColorScale` does, so the class ranges follow the scale's transform.
 */
function getLegendEdges(breaks: Float32Array, classCount: number, scale: ScaleChoice): number[] {
  const edges: number[] = [];
  const first = breaks[0];
  const last = breaks[classCount];
  const floor = 1e-5;
  for (let index = 0; index <= classCount; index++) {
    const t = index / Math.max(classCount, 1);
    if (scale === 'threshold') {
      edges.push(breaks[index]);
    } else if (scale === 'sqrt') {
      const low = Math.sqrt(Math.max(first, 0));
      const high = Math.sqrt(Math.max(last, 0));
      edges.push((low + (high - low) * t) ** 2);
    } else if (scale === 'log') {
      const low = Math.log(Math.max(first, floor));
      const high = Math.log(Math.max(last, floor));
      edges.push(Math.exp(low + (high - low) * t));
    } else {
      edges.push(first + (last - first) * t);
    }
  }
  return edges;
}
