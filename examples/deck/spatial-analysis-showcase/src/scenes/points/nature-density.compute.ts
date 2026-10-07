// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {
  createGPUPointDensityGaussianKernel,
  createGPUPointDensityGaussianKernel1D,
  GPUPointDensity,
  type GPUParameterBuffer
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {CHICAGO, nearestPlaceLabel} from '../../cartography/gazetteer';
import {hexToRgba, MAP_INK} from '../../cartography/hue-registry';
import {
  formatCount,
  formatDistance,
  formatOrdinal,
  formatPercent,
  formatRate
} from '../../cartography/live-text';
import {createFeatureLocator, getInputPolygons, type PolygonRings} from '../../cartography/picking';
import {getLocalProjector, projectRingsToSegments} from '../../cartography/segments';
import type {LngLat, MapAnnotation} from '../../cartography/types';
import {DENSE_POINT_RADIUS_STOPS} from '../../cartography/zoom';
import {fetchJson, type GeoJsonCollection} from '../../data/loaders';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {
  SpatialAnalysisPointLayer,
  SpatialAnalysisRasterLayer,
  SpatialAnalysisSegmentLayer
} from '../../engine/layers';
import {directionFor, sampleRamp} from '../../engine/ramps';
import {getViewportMetricBounds, SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import {measureCompiledGraph} from '../../engine/vector-timing';
import type {SceneContext, SceneInstance, TooltipRow} from '../scene';
import {
  fillNatureMask,
  formatHourWindow,
  readNatureColumns,
  type NatureColumns
} from './b1-nature-data';

/** Option state of the nature-density scene. */
export type NatureDensityOptions = {
  binning: 'grid' | 'hexagon';
  resolution: 'coarse' | 'medium' | 'fine';
  statistic: 'count' | 'sum' | 'mean';
  weight: 'researchGrade' | 'introduced' | 'animal';
  sumAccumulation: 'workgroup' | 'atomic';
  smoothing: 'off' | 'gaussian-2d' | 'gaussian-separable';
  sigma: number;
  hours: readonly [number, number];
  invertHours: boolean;
  dayType: 'all' | 'weekdays' | 'weekends';
  category: string;
  /** Dark-ground sequential ramps only (rule 3: no viridis or rainbow as a choice). */
  ramp: 'inferno' | 'magma' | 'fire' | 'mako';
  opacity: number;
  /** What the map draws: the raw point process, the density surface, or both. */
  view: 'points' | 'density' | 'both';
  /** How overlapping observation dots combine. */
  blending: 'additive' | 'normal';
  /** Draws the kernel's 3-sigma reach as a ring at the map centre. */
  showKernel: boolean;
  /** Draws park outlines, the lake shore and the city limit (the "data ends here" frame). */
  showContext: boolean;
};

/** Warm "light" ink of one observation dot on the dark ground (registry `nature` dot, `#fec44f`). */
const POINT_COLOR = [254, 196, 79] as const;
/** Cool dots, used when the surface is drawn too so dots and surface do not share a hue. */
const POINT_COLOR_OVER_SURFACE = [140, 215, 255] as const;
/**
 * Dot alpha per blending mode (0-255). Additive stacks add their light, so the alpha is small:
 * about 20 dots overlapping in one pixel reach white. Normal blending needs a stronger alpha so a
 * lone dot is visible.
 */
const DOT_ALPHA = {additive: 20, additiveOverSurface: 14, normal: 64} as const;
/** The brightest colour is the 98th percentile of the non-empty cells; the rest are clipped. */
const CLIP_PERCENTILE = 0.98;
/** Ramp trim: the lowest colour of a dark-ground ramp is not black (rule 3). */
const RAMP_RANGE = [0.15, 1] as const;
/** Peaks must reach this share of the maximum to count as a separate peak. */
const PEAK_COUNT_FLOOR = 0.02;
/** Bars of the legend histogram. */
const LEGEND_BINS = 16;
/** Bars of the distribution chart. */
const CHART_BINS = 24;
/** Context lines (rule 7, tier 3): grey, thin, translucent. */
const PARK_LINE_COLOR = hexToRgba(MAP_INK.dark.context, 84);
const LAKE_LINE_COLOR = hexToRgba(MAP_INK.dark.water, 96);
/** How many local maxima are labelled on the map. */
const PEAK_COUNT = 3;
/** Minimum separation of labelled peaks, in cells (non-maximum suppression). */
const PEAK_SEPARATION_CELLS = 9;

const GRID_SIZES = {coarse: [70, 44], medium: [110, 70], fine: [170, 106]} as const;
const HEXAGON_SIZES = {coarse: [34, 26], medium: [52, 38], fine: [80, 58]} as const;
const SQRT3 = Math.sqrt(3);
/** Compile-time kernel; per-frame weights select the actual sigma. */
const KERNEL_RADIUS = 8;
const KERNEL_WIDTH = KERNEL_RADIUS * 2 + 1;
const HISTOGRAM_BINS = 16;
const SETTLE_MILLISECONDS = 500;

type DensityGraph = {
  compiled: CompiledGPUCommandGraph<void>;
  values: Buffer;
  extent: Buffer;
  histogram: Buffer;
  /** Reads the whole field once the camera settles: clip value, peaks, tooltip, charts. */
  fieldReader: SummaryReader;
};

/** The last settled field and what was derived from it. */
type FieldSnapshot = {
  values: Float32Array;
  gridSize: readonly [number, number];
  /** Bounds (metres) the field was computed for. */
  bounds: Float32Array;
  binning: 'grid' | 'hexagon';
  /** Non-empty cell values, ascending. */
  sorted: Float32Array;
  maximum: number;
  /** Value of the brightest colour (98th percentile), or 0 when not clipped. */
  clip: number;
};

type DensitySet = {
  gridSize: readonly [number, number];
  dense?: DensityGraph;
  separable?: DensityGraph;
  hexagon?: DensityGraph;
  compareReader: SummaryReader | null;
};

/** Loads the park, lake and city-limit geometry of the "dots meet the parks" step. */
type ContextGeometry = {
  parkSegments: Float32Array;
  lakeSegments: Float32Array;
  /** Outer rings of the city limit, largest first, for the "data ends here" frames. */
  cityRings: LngLat[][];
  parks: GeoJsonCollection | null;
};

/** The ring with the most vertices of each polygon of a GeoJSON input, largest polygon first. */
function getOuterRings(input: GeoJsonCollection): LngLat[][] {
  return getInputPolygons(input)
    .map(({polygon}) => polygon[0].map(vertex => [vertex[0], vertex[1]] as LngLat))
    .sort((a, b) => b.length - a.length);
}

/** Flattens every ring of a GeoJSON input (outer rings and holes) to GeoJSON-style rings. */
function getAllRings(input: GeoJsonCollection): PolygonRings {
  return getInputPolygons(input).flatMap(({polygon}) => polygon);
}

/**
 * Viewport-following density of Chicago nature observations. Everything the analyst steers (bounds, hexagon
 * radius, Gaussian sigma, the hour/day/category mask and the weight attribute) is a buffer write.
 * Compile-time choices (lattice, resolution, statistic, accumulation) select between lazily compiled
 * graph sets, so switching back never recompiles.
 */
export async function createNatureDensity(
  ctx: SceneContext<NatureDensityOptions>
): Promise<SceneInstance<NatureDensityOptions>> {
  const observations = ctx.datasets.get('chicago-nature');
  const {device} = ctx;
  const columns: NatureColumns = readNatureColumns(observations);
  const origin = columns.origin;
  const projection = observations.getProjection(origin);
  const pointCount = columns.count;
  const categoryIndex = (name: string) =>
    name === 'all' ? -1 : columns.categoryNames.indexOf(name);

  const resources = new SpatialAnalysisResources(device, 'nature-density');
  const positionsBuffer = resources.createBuffer('positions', columns.positions);
  const maskBuffer = resources.createBuffer('mask', new Uint32Array(pointCount).fill(1));
  const weightsBuffer = resources.createBuffer('weights', columns[ctx.options.weight]);
  const bounds = resources.createParameterBuffer('bounds', 'float32', 4);
  const hexagonRadius = resources.createParameterBuffer(
    'hexagon-radius',
    'float32',
    1,
    Float32Array.of(50)
  );
  const kernel = resources.createParameterBuffer('kernel', 'float32', KERNEL_WIDTH * KERNEL_WIDTH);
  const lineKernel = resources.createParameterBuffer('line-kernel', 'float32', KERNEL_WIDTH);

  // Context geometry (parks, lake shore, city limit), projected with the points' origin.
  const context = await loadContext(ctx, origin);
  const parkBuffer = context.parkSegments.length
    ? resources.createBuffer('park-segments', context.parkSegments)
    : null;
  const lakeBuffer = context.lakeSegments.length
    ? resources.createBuffer('lake-segments', context.lakeSegments)
    : null;

  let destroyed = false;
  let measuring = false;
  let settleStale = true;
  let lastChangeTime = performance.now();
  let lastBounds: Float32Array | null = null;
  /** Bounds of the frame whose field the reader copied. */
  let readBounds: Float32Array | null = null;
  let snapshot: FieldSnapshot | null = null;
  /** Metres per cell of the current frame (hexagons: the hexagon radius). */
  let cellMeters = 0;
  let lastCellText = '';
  let includedCount = pointCount;
  let hoverIndex = -1;
  const sets = new Map<string, DensitySet>();
  let current: DensitySet | null = null;
  let currentKey = '';

  ctx.setReadout('records', formatCount(pointCount));
  ctx.setReadout('taxa', formatCount(new Set(columns.species).size));
  writeKernels(kernel, lineKernel, ctx.options.smoothing === 'off' ? 0 : ctx.options.sigma);
  ctx.setReadout('kernelRadius', `${getKernelRadius(ctx.options.sigma)} cells`);

  const markChanged = () => {
    settleStale = true;
    lastChangeTime = performance.now();
  };

  const getKey = (options: NatureDensityOptions) =>
    `${options.binning}:${options.resolution}:${options.statistic}:${options.sumAccumulation}`;

  const getDisplayed = (): DensityGraph | undefined => {
    if (!current) return undefined;
    if (ctx.options.binning === 'hexagon') return current.hexagon;
    return ctx.options.smoothing === 'gaussian-separable' ? current.separable : current.dense;
  };

  /** Direction of the ramp on the (always dark) night ground, so more is brighter. */
  const getRampDirection = () => directionFor(ctx.ground(), ctx.options.ramp).reverse;

  /** The cost line: records kept and graph nodes encoded per frame. */
  const publishCost = () => {
    const displayed = getDisplayed();
    ctx.setCost({
      records: includedCount,
      passes: displayed ? displayed.compiled.stats.nodeOrder.length : undefined
    });
  };

  /** Colour a value has on the map, for tooltip swatches. */
  const getRampSwatch = (
    value: number,
    clip: number
  ): readonly [number, number, number, number] => {
    const {statistic, ramp} = ctx.options;
    const top = statistic === 'mean' ? 1 : Math.max(clip, 1e-9);
    const [red, green, blue] = sampleRamp(
      ramp,
      Math.sqrt(Math.min(Math.max(value / top, 0), 1)),
      getRampDirection(),
      RAMP_RANGE
    );
    return [red, green, blue, 255];
  };

  function buildDensityGraph(
    id: string,
    options: NatureDensityOptions,
    gridSize: readonly [number, number],
    binning: 'grid' | 'hexagon',
    useSeparable: boolean
  ): DensityGraph {
    const cellCount = gridSize[0] * gridSize[1];
    const values = resources.createBuffer(`${id}-values`, cellCount * 4);
    const extent = resources.createBuffer(`${id}-extent`, 8);
    const histogram = resources.createBuffer(`${id}-histogram`, HISTOGRAM_BINS * 4);
    const graph = new GPUCommandGraph<void>(device, {id: `nature-density-${id}`});
    const lineView = useSeparable ? lineKernel.importToGraph(graph) : undefined;
    const weighted = options.statistic !== 'count';
    graph.add(
      new GPUPointDensity({
        id: 'density',
        positions: importGraphBuffer(graph, 'positions', positionsBuffer, 'float32x2', pointCount),
        mask: importGraphBuffer(graph, 'mask', maskBuffer, 'uint32', pointCount),
        ...(weighted
          ? {weights: importGraphBuffer(graph, 'weights', weightsBuffer, 'float32', pointCount)}
          : {}),
        bounds: bounds.importToGraph(graph),
        gridSize,
        binning,
        ...(binning === 'hexagon'
          ? {hexagonRadius: hexagonRadius.importToGraph(graph)}
          : {
              smoothing: lineView
                ? {
                    separableKernel: {horizontal: lineView, vertical: lineView},
                    kernelWidth: KERNEL_WIDTH,
                    kernelHeight: KERNEL_WIDTH
                  }
                : {
                    kernel: kernel.importToGraph(graph),
                    kernelWidth: KERNEL_WIDTH,
                    kernelHeight: KERNEL_WIDTH,
                    strategy: 'direct'
                  }
            }),
        statistic: options.statistic,
        ...(weighted ? {sumAccumulation: options.sumAccumulation} : {}),
        output: {
          values: importGraphBuffer(graph, 'values', values, 'float32', cellCount),
          extent: importGraphBuffer(graph, 'extent', extent, 'float32', 2),
          histogram: importGraphBuffer(graph, 'histogram', histogram, 'uint32', HISTOGRAM_BINS)
        }
      })
    );
    const fieldReader = new SummaryReader(
      resources,
      `${id}-field`,
      [{buffer: values, size: cellCount * 4}],
      bytes => {
        if (destroyed || getDisplayed()?.values !== values || !readBounds) return;
        handleField(new Float32Array(bytes.slice(0)), gridSize, readBounds, binning);
      }
    );
    return {
      compiled: resources.track(graph.compile()),
      values,
      extent,
      histogram,
      fieldReader
    };
  }

  /** Index of the first element of `sorted` greater than `value` (upper bound). */
  const countAtOrBelow = (sorted: Float32Array, value: number): number => {
    let low = 0;
    let high = sorted.length;
    while (low < high) {
      const middle = (low + high) >> 1;
      if (sorted[middle] <= value) low = middle + 1;
      else high = middle;
    }
    return low;
  };

  /**
   * Everything the settled field teaches: the clip value (98th percentile) of the colour scale,
   * the legend histogram, the cell and bandwidth readouts, the strongest peaks as finding notes,
   * the number of separate peaks and the distribution chart.
   */
  function handleField(
    field: Float32Array,
    gridSize: readonly [number, number],
    fieldBounds: Float32Array,
    binning: 'grid' | 'hexagon'
  ): void {
    const {statistic, view, smoothing, sigma} = ctx.options;
    const positive: number[] = [];
    let maximum = 0;
    for (let index = 0; index < field.length; index++) {
      const value = field[index];
      if (value > 0 && Number.isFinite(value)) {
        positive.push(value);
        maximum = Math.max(maximum, value);
      }
    }
    const sorted = Float32Array.from(positive).sort();
    const clip =
      statistic === 'mean' || sorted.length === 0
        ? 0
        : Math.max(
            1,
            sorted[Math.min(sorted.length - 1, Math.floor(CLIP_PERCENTILE * sorted.length))]
          );
    const clipChanged = !snapshot || snapshot.clip !== clip;
    snapshot = {values: field, gridSize, bounds: fieldBounds, binning, sorted, maximum, clip};

    // Legend: the clipped range, a distribution strip on it and the cell the unit refers to.
    const top = statistic === 'mean' ? 1 : clip || 1;
    ctx.setLegendExtent('density', [0, top]);
    const bins = new Array<number>(LEGEND_BINS).fill(0);
    for (const value of positive) {
      bins[Math.min(LEGEND_BINS - 1, Math.floor((value / top) * LEGEND_BINS))]++;
    }
    ctx.setLegendData('histogram', bins);
    const cellWidth = (fieldBounds[2] - fieldBounds[0]) / gridSize[0];
    const basis =
      binning === 'grid'
        ? `per ${formatDistance(cellWidth)} cell`
        : `per hexagon of ${formatDistance(cellMeters)} radius`;
    ctx.setLegendData('basis', basis);

    const unit = statistic === 'count' ? 'records' : statistic === 'sum' ? 'flagged records' : '';
    ctx.setReadout(
      'peak',
      statistic === 'mean' ? formatPercent(maximum, 0) : `${formatCount(maximum)} ${unit}`
    );
    ctx.setReadout(
      'peakShare',
      statistic === 'count' && includedCount > 0 ? formatPercent(maximum / includedCount, 1) : null
    );
    ctx.setReadout(
      'bandwidth',
      smoothing === 'off' || binning === 'hexagon' ? null : formatDistance(sigma * cellMeters)
    );

    // Strongest local maxima (square grid): finding notes and the separate-peak count.
    let peakCount: number | null = null;
    if (binning === 'grid' && statistic !== 'mean' && view !== 'points') {
      const [columnsCount, rowsCount] = gridSize;
      const floor = maximum * PEAK_COUNT_FLOOR;
      let count = 0;
      const maxima: number[] = [];
      for (let index = 0; index < field.length; index++) {
        const value = field[index];
        if (!(value > floor)) continue;
        const column = index % columnsCount;
        const row = Math.floor(index / columnsCount);
        let isMaximum = true;
        for (let dy = -1; dy <= 1 && isMaximum; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            if (dx === 0 && dy === 0) continue;
            const x = column + dx;
            const y = row + dy;
            if (x < 0 || y < 0 || x >= columnsCount || y >= rowsCount) continue;
            const other = field[y * columnsCount + x];
            // Ties go to the cell with the lower index, so a plateau counts once.
            if (other > value || (other === value && y * columnsCount + x < index)) {
              isMaximum = false;
              break;
            }
          }
        }
        if (isMaximum) {
          count++;
          maxima.push(index);
        }
      }
      peakCount = count;
      const chosen: number[] = [];
      for (const index of maxima.sort((a, b) => field[b] - field[a])) {
        const column = index % columnsCount;
        const row = Math.floor(index / columnsCount);
        const separated = chosen.every(other => {
          const dx = (other % columnsCount) - column;
          const dy = Math.floor(other / columnsCount) - row;
          return dx * dx + dy * dy >= PEAK_SEPARATION_CELLS ** 2;
        });
        if (separated) chosen.push(index);
        if (chosen.length === PEAK_COUNT) break;
      }
      const cellHeight = (fieldBounds[3] - fieldBounds[1]) / rowsCount;
      const coordinates = chosen.map(index => {
        const x = fieldBounds[0] + ((index % columnsCount) + 0.5) * cellWidth;
        const y = fieldBounds[1] + (Math.floor(index / columnsCount) + 0.5) * cellHeight;
        return projection.unproject(x, y) as LngLat;
      });
      const notes: MapAnnotation[] = chosen.map((index, rank) => ({
        kind: 'note',
        id: `peak-${rank}`,
        coordinate: coordinates[rank],
        // The number and its unit come from the readback; the place from the gazetteer.
        title: `${formatCount(field[index])} ${unit} per cell`,
        text: nearestPlaceLabel(CHICAGO, coordinates[rank], {maxDistanceMeters: 8000}) ?? undefined,
        tone: rank === 0 ? 'accent' : 'ink',
        priority: 5 - rank
      }));
      ctx.setAnnotations('peaks', notes.length ? notes : null);
      ctx.setReadout(
        'peakPlace',
        coordinates.length
          ? nearestPlaceLabel(CHICAGO, coordinates[0], {maxDistanceMeters: 8000})
          : null
      );
    } else {
      ctx.setAnnotations('peaks', null);
      ctx.setReadout('peakPlace', null);
    }
    ctx.setReadout('peakCount', peakCount === null ? null : `${formatCount(peakCount)} peaks`);

    publishDistribution(null);
    updateKernelRing();
    updateScaleBarTick();
    updateAnnotationHalo();
    if (clipChanged) ctx.requestLayers();
  }

  /**
   * The distribution chart: non-empty cells binned on a square-root axis (the scale the map
   * uses), with a marker at the hovered cell, or at the peak when nothing is hovered.
   */
  function publishDistribution(hovered: number | null): void {
    if (!snapshot || snapshot.sorted.length === 0) {
      ctx.setChart('distribution', null);
      return;
    }
    const {sorted, maximum} = snapshot;
    const top = Math.sqrt(maximum);
    const bins = new Array<number>(CHART_BINS).fill(0);
    for (const value of sorted) {
      bins[Math.min(CHART_BINS - 1, Math.floor((Math.sqrt(value) / top) * CHART_BINS))]++;
    }
    const marker = hovered ?? maximum;
    ctx.setChart('distribution', {
      kind: 'histogram',
      values: bins,
      xDomain: [0, top],
      title: 'Cells by records (square-root axis)',
      xLabel: 'Records per cell',
      yLabel: 'Cells',
      formatX: value => formatCount(value * value),
      now: Math.sqrt(marker),
      nowLabel: hovered === null ? 'Peak' : 'This cell',
      table: false,
      description:
        'Histogram of non-empty cells by record count: most cells hold a few records and a few hold hundreds.'
    });
  }

  /** The kernel's weights against distance, for the sigma-linked chart. */
  function publishKernelProfile(): void {
    const {sigma} = ctx.options;
    const distances = Array.from({length: 33}, (_, index) => index * 0.25);
    const weights = (width: number) => distances.map(x => Math.exp(-(x * x) / (2 * width * width)));
    ctx.setChart('kernelProfile', {
      kind: 'line',
      title: 'How far a dot reaches',
      xLabel: 'Distance from the dot (cells)',
      yLabel: 'Weight',
      xDomain: [0, 8],
      yDomain: [0, 1],
      series: [
        {label: 'Too narrow', x: distances, y: weights(0.5), ghost: true},
        {label: 'Too wide', x: distances, y: weights(2.5), ghost: true},
        {label: 'This sigma', x: distances, y: weights(sigma)}
      ],
      link: {option: 'sigma', label: value => `σ = ${value} cells`},
      table: false,
      description:
        'A Gaussian bell: the weight a record gives to cells at each distance. The marker is one sigma; beyond three sigma a record no longer counts.'
    });
  }

  /** The bandwidth as dashed rings at the map centre: one sigma and the 3-sigma reach. */
  function updateKernelRing(): void {
    const {showKernel, smoothing, sigma, binning, view} = ctx.options;
    if (
      !showKernel ||
      smoothing === 'off' ||
      binning !== 'grid' ||
      view === 'points' ||
      !cellMeters
    ) {
      ctx.setAnnotations('kernel', null);
      return;
    }
    const sigmaMeters = Math.max(sigma * cellMeters, 1);
    const reachMeters = getKernelRadius(sigma) * cellMeters;
    const {longitude, latitude} = ctx.getViewState();
    ctx.setAnnotations('kernel', [
      {
        kind: 'ring',
        id: 'kernel-sigma',
        coordinate: [longitude, latitude],
        radiusMeters: sigmaMeters,
        text: `σ ${formatDistance(sigmaMeters)}`,
        dashed: true
      },
      {
        kind: 'ring',
        id: 'kernel-reach',
        coordinate: [longitude, latitude],
        radiusMeters: reachMeters,
        text: `reach ${formatDistance(reachMeters)}`,
        dashed: true
      }
    ]);
  }

  // `setFurniture` replaces the previous runtime overrides as a whole, so both live here.
  const runtimeFurniture: {
    title: {sample: string};
    scaleBar: {units: 'metric'; ticks?: number[]};
  } = {
    title: {
      sample: `${formatCount(pointCount)} iNaturalist records, Chicago, ${new Date(
        observations.column<Uint32Array>('timestamp')[0] * 1000
      ).getUTCFullYear()}`
    },
    scaleBar: {units: 'metric'}
  };
  ctx.setFurniture(runtimeFurniture);

  let lastTickMeters: number | null | undefined;
  /**
   * Scale-bar tick at the bandwidth, only while the kernel is shown (the steps that teach it).
   * Rounded to 10 m so the furniture is not rewritten on every sub-pixel camera change.
   */
  function updateScaleBarTick(): void {
    const {showKernel, smoothing, sigma, binning, view} = ctx.options;
    const teaching = showKernel && smoothing !== 'off' && binning === 'grid' && view !== 'points';
    const meters = teaching && cellMeters ? Math.round((sigma * cellMeters) / 10) * 10 : null;
    if (meters === lastTickMeters) return;
    lastTickMeters = meters;
    runtimeFurniture.scaleBar = {units: 'metric', ticks: meters ? [meters] : undefined};
    ctx.setFurniture(runtimeFurniture);
  }

  let lastHalo: 'normal' | 'heavy' | null = null;
  /** Heavy label halos where names sit over the bright density surface. */
  function updateAnnotationHalo(): void {
    const weight = ctx.options.view === 'points' ? 'normal' : 'heavy';
    if (weight === lastHalo) return;
    lastHalo = weight;
    ctx.setAnnotationHalo(weight);
  }

  /** City-limit frames: where the data ends (the edge effect), only with the context on. */
  function updateContextAnnotations(): void {
    if (!ctx.options.showContext) {
      ctx.setAnnotations('city-limit', null);
      return;
    }
    ctx.setAnnotations(
      'city-limit',
      context.cityRings.map((ring, index) => ({
        kind: 'frame',
        id: `city-limit-${index}`,
        ring,
        text: index === 0 ? 'Data ends at the city limit' : undefined
      }))
    );
  }

  let parkShareStarted = false;
  /** Share of records inside mapped green space, computed once when the context first shows. */
  function computeParkShare(): void {
    if (parkShareStarted || !context.parks) return;
    parkShareStarted = true;
    const parks = context.parks;
    setTimeout(() => {
      if (destroyed) return;
      const locator = createFeatureLocator(parks);
      const position = observations.column<Float32Array>('position');
      let inside = 0;
      for (let index = 0; index < pointCount; index++) {
        if (locator.find([position[index * 2], position[index * 2 + 1]])) inside++;
      }
      ctx.setReadout('parkShare', formatPercent(inside / pointCount, 0));
    }, 0);
  }

  function buildSet(options: NatureDensityOptions): DensitySet {
    const key = getKey(options);
    const id = key.replace(/:/g, '-');
    if (options.binning === 'grid') {
      const gridSize = GRID_SIZES[options.resolution];
      const dense = buildDensityGraph(`${id}-dense`, options, gridSize, 'grid', false);
      const separable = buildDensityGraph(`${id}-separable`, options, gridSize, 'grid', true);
      const cellCount = gridSize[0] * gridSize[1];
      const compareReader = new SummaryReader(
        resources,
        `${id}-compare`,
        [
          {buffer: dense.values, size: cellCount * 4},
          {buffer: separable.values, size: cellCount * 4}
        ],
        bytes => {
          if (destroyed) return;
          const denseValues = new Float32Array(bytes, 0, cellCount);
          const separableValues = new Float32Array(bytes, cellCount * 4, cellCount);
          let maximumDifference = 0;
          let maximumValue = 0;
          for (let index = 0; index < cellCount; index++) {
            maximumDifference = Math.max(
              maximumDifference,
              Math.abs(denseValues[index] - separableValues[index])
            );
            maximumValue = Math.max(maximumValue, denseValues[index]);
          }
          ctx.setReadout(
            'difference',
            `${maximumDifference.toExponential(2)} (field max ${maximumValue.toPrecision(3)})`
          );
        }
      );
      return {gridSize, dense, separable, compareReader};
    }
    const gridSize = HEXAGON_SIZES[options.resolution];
    return {
      gridSize,
      hexagon: buildDensityGraph(`${id}-hexagon`, options, gridSize, 'hexagon', false),
      compareReader: null
    };
  }

  function selectSet(): void {
    const key = getKey(ctx.options);
    if (key === currentKey && current) return;
    let next = sets.get(key);
    if (!next) {
      next = buildSet(ctx.options);
      sets.set(key, next);
    }
    current = next;
    currentKey = key;
    snapshot = null;
    const [columnsCount, rowsCount] = next.gridSize;
    ctx.setReadout(
      'grid',
      `${columnsCount} × ${rowsCount} ${ctx.options.binning === 'hexagon' ? 'hexagons' : 'cells'}`
    );
    if (ctx.options.binning === 'hexagon') {
      for (const id of ['difference', 'denseTime', 'separableTime', 'speedup'])
        ctx.setReadout(id, 'n/a (square grid only)');
    } else {
      for (const id of ['difference', 'denseTime', 'separableTime', 'speedup'])
        ctx.setReadout(id, null);
    }
    markChanged();
    publishCost();
  }

  function writeMask(): void {
    const {hours, invertHours, dayType, category} = ctx.options;
    const mask = new Uint32Array(pointCount);
    const included = fillNatureMask(
      columns,
      {hours, invertHours, dayType, category: categoryIndex(category)},
      mask
    );
    maskBuffer.write(mask);
    includedCount = included;
    ctx.setReadout('kept', formatCount(included));
    ctx.setReadout('window', formatHourWindow(hours, invertHours));
    markChanged();
    publishCost();
  }

  function writeWeights(): void {
    weightsBuffer.write(columns[ctx.options.weight]);
    markChanged();
  }

  async function measureSmoothing(): Promise<void> {
    const set = current;
    if (measuring || destroyed || !set?.dense || !set.separable) return;
    measuring = true;
    try {
      const options = {parameters: undefined, completionBuffer: set.dense.extent};
      const denseTiming = await measureCompiledGraph(device, set.dense.compiled, options);
      const separableTiming = await measureCompiledGraph(device, set.separable.compiled, options);
      if (destroyed || current !== set) return;
      ctx.setReadout('denseTime', `${denseTiming.milliseconds.toFixed(2)} ms`);
      ctx.setReadout('separableTime', `${separableTiming.milliseconds.toFixed(2)} ms`);
      const ratio = denseTiming.milliseconds / separableTiming.milliseconds;
      ctx.setReadout('speedup', Number.isFinite(ratio) ? `${ratio.toFixed(2)}x` : 'n/a');
    } catch {
      // Device destroyed or measurement aborted.
    } finally {
      measuring = false;
    }
  }

  writeMask();
  writeWeights();
  selectSet();
  publishKernelProfile();
  updateContextAnnotations();
  if (ctx.options.showContext) computeParkShare();

  return {
    getCompiledGraphs: () => {
      if (!current) return [];
      const graphs = [current.dense, current.separable, current.hexagon]
        .filter((graph): graph is DensityGraph => Boolean(graph))
        .map(graph => graph.compiled);
      return graphs as unknown as CompiledGPUCommandGraph<never>[];
    },

    setOption(id, _value, state) {
      if (['binning', 'resolution', 'statistic', 'sumAccumulation'].includes(id)) {
        selectSet();
        ctx.requestLayers();
      } else if (id === 'smoothing' || id === 'sigma') {
        writeKernels(kernel, lineKernel, state.smoothing === 'off' ? 0 : state.sigma);
        markChanged();
        if (id === 'sigma') {
          ctx.setReadout('kernelRadius', `${getKernelRadius(state.sigma)} cells`);
          publishKernelProfile();
        }
        publishCost();
        ctx.requestLayers();
      } else if (['hours', 'invertHours', 'dayType', 'category'].includes(id)) {
        writeMask();
      } else if (id === 'weight') {
        writeWeights();
        ctx.requestLayers();
      } else if (id === 'showContext') {
        updateContextAnnotations();
        if (state.showContext) computeParkShare();
        ctx.requestLayers();
      } else {
        if (id === 'view') markChanged();
        ctx.requestLayers();
      }
      if (id === 'showKernel' || id === 'view' || id === 'binning') updateKernelRing();
      updateScaleBarTick();
      updateAnnotationHalo();
      if (id === 'binning' || id === 'view') ctx.setAnnotations('peaks', null);
    },

    onAction(id) {
      if (id === 'measure') void measureSmoothing();
    },

    onThemeChange() {
      ctx.requestLayers();
    },

    getTooltip(event) {
      const field = snapshot;
      const {view, statistic, smoothing} = ctx.options;
      const clear = () => {
        if (hoverIndex !== -1) {
          hoverIndex = -1;
          ctx.setLegendData('marker', null);
          publishDistribution(null);
        }
        return null;
      };
      if (!field || !event.coordinate || field.binning !== 'grid' || view === 'points') {
        return clear();
      }
      // The field is only valid for the frame it was read at.
      if (!lastBounds || field.bounds.some((value, index) => value !== lastBounds![index])) {
        return clear();
      }
      const [columnsCount, rowsCount] = field.gridSize;
      const [x, y] = projection.project(event.coordinate[0], event.coordinate[1]);
      const cellWidth = (field.bounds[2] - field.bounds[0]) / columnsCount;
      const cellHeight = (field.bounds[3] - field.bounds[1]) / rowsCount;
      const column = Math.floor((x - field.bounds[0]) / cellWidth);
      const row = Math.floor((y - field.bounds[1]) / cellHeight);
      if (column < 0 || row < 0 || column >= columnsCount || row >= rowsCount) return clear();
      const index = row * columnsCount + column;
      const value = field.values[index];
      if (!(value > 0)) return clear();
      if (index !== hoverIndex) {
        hoverIndex = index;
        ctx.setLegendData('marker', value);
        publishDistribution(value);
      }
      const percentile = countAtOrBelow(field.sorted, value) / field.sorted.length;
      const south = field.bounds[1] + row * cellHeight;
      const west = field.bounds[0] + column * cellWidth;
      const [west0, south0] = projection.unproject(west, south);
      const [east1, north1] = projection.unproject(west + cellWidth, south + cellHeight);
      const areaKm2 = (cellWidth * cellHeight) / 1e6;
      const rows: TooltipRow[] = [
        {
          label:
            statistic === 'count'
              ? 'Records'
              : statistic === 'sum'
                ? 'Flagged records'
                : 'Share flagged',
          value: statistic === 'mean' ? formatPercent(value, 0) : formatCount(value),
          unit:
            statistic === 'mean' ? undefined : smoothing === 'off' ? 'in this cell' : 'smoothed',
          swatch: getRampSwatch(value, field.clip),
          emphasis: true
        },
        {label: 'Rank', value: `${formatOrdinal(percentile * 100)} percentile`}
      ];
      if (statistic === 'count') {
        rows.push(
          {label: 'Density', value: formatRate(value / areaKm2, 'km²')},
          {label: 'Share of kept records', value: formatPercent(value / includedCount, 1)}
        );
      }
      return {
        title: `${formatDistance(cellWidth)} cell`,
        subtitle:
          nearestPlaceLabel(
            CHICAGO,
            projection.unproject(west + cellWidth / 2, south + cellHeight / 2) as LngLat,
            {maxDistanceMeters: 8000}
          ) ?? undefined,
        rows,
        highlight: {kind: 'box', bounds: [west0, south0, east1, north1]}
      };
    },

    encode(commandEncoder, frame) {
      const displayed = getDisplayed();
      if (!displayed || !current) return;
      const {binning} = ctx.options;
      const viewBounds = getViewportMetricBounds(frame.viewport, projection);
      let cellText: string;
      if (binning === 'hexagon') {
        const [columnsCount, rowsCount] = current.gridSize;
        const radius = Math.max(
          (viewBounds[2] - viewBounds[0]) / (SQRT3 * (columnsCount - 1)),
          (viewBounds[3] - viewBounds[1]) / (1.5 * (rowsCount - 1))
        );
        hexagonRadius.write(Float32Array.of(radius));
        cellMeters = radius;
        cellText = `${formatDistance(radius)} hexagon radius`;
      } else {
        cellMeters = (viewBounds[2] - viewBounds[0]) / current.gridSize[0];
        cellText = formatDistance(cellMeters);
      }
      if (cellText !== lastCellText) {
        lastCellText = cellText;
        ctx.setReadout('cellSize', cellText);
      }
      const boundsData = Float32Array.from(viewBounds);
      if (!lastBounds || boundsData.some((value, index) => value !== lastBounds![index]))
        markChanged();
      lastBounds = boundsData;
      bounds.write(boundsData);
      displayed.compiled.encode(commandEncoder, {parameters: undefined});
      const settled = performance.now() - lastChangeTime > SETTLE_MILLISECONDS;
      if (settleStale && settled && !displayed.fieldReader.isPending) {
        readBounds = boundsData;
        displayed.fieldReader.request(commandEncoder);
        const {dense, separable, compareReader} = current;
        if (dense && separable && compareReader && !compareReader.isPending) {
          const other = displayed === dense ? separable : dense;
          other.compiled.encode(commandEncoder, {parameters: undefined});
          compareReader.request(commandEncoder);
        }
        settleStale = false;
      } else {
        displayed.fieldReader.flush(commandEncoder);
        current.compareReader?.flush(commandEncoder);
      }
    },

    getLayers() {
      const displayed = getDisplayed();
      if (!displayed || !current) return [];
      const {binning, ramp, opacity, view, statistic, blending, showContext} = ctx.options;
      const coordinateOrigin: [number, number, number] = [origin[0], origin[1], 0];
      const layers: Layer[] = [];
      // Tier 3, context: park outlines and the lake shore, thin and grey, under everything.
      if (showContext) {
        if (parkBuffer) {
          layers.push(
            new SpatialAnalysisSegmentLayer({
              id: 'nature-density-parks',
              coordinateOrigin,
              segments: parkBuffer,
              instanceCount: context.parkSegments.length / 4,
              widthPixels: 0.8,
              color: PARK_LINE_COLOR
            })
          );
        }
        if (lakeBuffer) {
          layers.push(
            new SpatialAnalysisSegmentLayer({
              id: 'nature-density-lake',
              coordinateOrigin,
              segments: lakeBuffer,
              instanceCount: context.lakeSegments.length / 4,
              widthPixels: 1,
              color: LAKE_LINE_COLOR
            })
          );
        }
      }
      if (view !== 'points') {
        // The brightest colour is the 98th percentile of the non-empty cells (see the legend note).
        const clip = snapshot && snapshot.binning === binning ? snapshot.clip : 0;
        layers.push(
          new SpatialAnalysisRasterLayer({
            id: `nature-density-${binning}`,
            coordinateOrigin,
            gridSize: current.gridSize,
            bounds: bounds.buffer,
            binning,
            hexagonRadius: hexagonRadius.buffer,
            values: displayed.values,
            valueFormat: 'float32',
            ...(statistic === 'mean'
              ? {valueRange: [0, 1] as const}
              : clip > 0
                ? {valueRange: [0, clip] as const}
                : {extent: displayed.extent}),
            colormap: ramp,
            sqrtScale: statistic !== 'mean',
            // Rule 3: dark-ground ramps are trimmed so the lowest colour is not black.
            rampRange: RAMP_RANGE,
            reverseRamp: getRampDirection(),
            discardAtOrBelow:
              binning === 'grid' && ctx.options.smoothing !== 'off'
                ? statistic === 'mean'
                  ? 0.002
                  : 0.3
                : 0,
            color: [255, 255, 255, Math.round(opacity * 255)]
          })
        );
      }
      if (view !== 'density') {
        // One dot per observation, sized by zoom band (rule: dense points), additive so overlaps
        // add up to light. Over the surface the dots turn cool so the two hues stay apart.
        const alone = view === 'points';
        const additive = blending === 'additive';
        const alpha = additive
          ? alone
            ? DOT_ALPHA.additive
            : DOT_ALPHA.additiveOverSurface
          : DOT_ALPHA.normal;
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'nature-density-points',
            coordinateOrigin,
            positions: positionsBuffer,
            instanceCount: pointCount,
            radiusPixels: DENSE_POINT_RADIUS_STOPS,
            blending,
            color: [...(alone ? POINT_COLOR : POINT_COLOR_OVER_SURFACE), alpha]
          })
        );
      }
      return layers;
    },

    destroy() {
      destroyed = true;
      for (const set of sets.values()) {
        set.compareReader?.stop();
        for (const graph of [set.dense, set.separable, set.hexagon]) {
          graph?.fieldReader.stop();
        }
      }
      resources.destroy();
    }
  };
}

/** Reads and projects the context geometry; missing files leave that layer out. */
async function loadContext(
  ctx: SceneContext<NatureDensityOptions>,
  origin: [number, number]
): Promise<ContextGeometry> {
  const project = getLocalProjector(origin);
  const parks = ctx.datasets.get('chicago-parks').geojson;
  const boundary = ctx.datasets.get('chicago-boundary');
  let lake: GeoJsonCollection | null = null;
  try {
    lake = await fetchJson<GeoJsonCollection>(boundary.fileUrl('lake.geojson'), ctx.signal);
  } catch {
    // The lake shore is optional context.
  }
  return {
    parkSegments: parks ? projectRingsToSegments(getAllRings(parks), project) : new Float32Array(0),
    lakeSegments: lake ? projectRingsToSegments(getAllRings(lake), project) : new Float32Array(0),
    cityRings: boundary.geojson ? getOuterRings(boundary.geojson) : [],
    parks
  };
}

function getKernelRadius(sigma: number): number {
  return Math.min(KERNEL_RADIUS, Math.ceil(3 * sigma));
}

/** Writes the 2D and 1D Gaussians for `sigma` (0 writes a unit impulse, smoothing off). */
function writeKernels(
  kernel: GPUParameterBuffer<'float32'>,
  lineKernel: GPUParameterBuffer<'float32'>,
  sigma: number
): void {
  const radius = sigma > 0 ? getKernelRadius(sigma) : 0;
  const small = createGPUPointDensityGaussianKernel(radius, sigma > 0 ? sigma : undefined);
  const smallLine = createGPUPointDensityGaussianKernel1D(radius, sigma > 0 ? sigma : undefined);
  const size = radius * 2 + 1;
  const offset = KERNEL_RADIUS - radius;
  const weights = new Float32Array(KERNEL_WIDTH * KERNEL_WIDTH);
  for (let row = 0; row < size; row++) {
    for (let column = 0; column < size; column++) {
      weights[(row + offset) * KERNEL_WIDTH + column + offset] = small[row * size + column];
    }
  }
  const lineWeights = new Float32Array(KERNEL_WIDTH);
  lineWeights.set(smallLine, offset);
  kernel.write(weights);
  lineKernel.write(lineWeights);
}
