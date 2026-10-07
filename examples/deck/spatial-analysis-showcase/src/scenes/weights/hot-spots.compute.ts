// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import type {Buffer, CommandEncoder} from '@luma.gl/core';
import {
  addHotSpotAnalysisRecipe,
  getGPUNeighborSearchParameterValues,
  getGPUPermutationParameterValues,
  getGPUSpatialAutocorrelationParameterValues,
  GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH,
  GPU_PERMUTATION_PARAMETER_LENGTH,
  GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH,
  type GPUCellFamily,
  type GPUPermutationAlternative
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  CLASSIFICATION_METHOD_INFO,
  getClassBreaks,
  getClassCounts,
  getExtent,
  getGoodnessOfVarianceFit,
  type ClassificationMethod
} from '../../cartography/breaks';
import {
  getClassIndexOf,
  getClassTableLayerProps,
  makeClassTable
} from '../../cartography/class-table';
import {CHICAGO, nearestPlaceLabel} from '../../cartography/gazetteer';
import {getRegistryColors, NO_DATA_COLOR} from '../../cartography/hue-registry';
import {
  formatCount,
  formatDistance,
  formatOrdinal,
  formatPercent,
  formatSigned
} from '../../cartography/live-text';
import {getLocalProjector, projectRingsToSegments} from '../../cartography/segments';
import type {ClassTable, LngLat, MapAnnotation, MapHighlight} from '../../cartography/types';
import {getInputPolygons} from '../../cartography/picking';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {
  SpatialAnalysisPointLayer,
  SpatialAnalysisPolygonLayer,
  SpatialAnalysisSegmentLayer,
  type SpatialAnalysisColor,
  type SpatialAnalysisStyleProps
} from '../../engine/layers';
import {LocalMetricProjection} from '../../engine/projection';
import {addKernelPass} from '../../engine/mode-kernels';
import {sampleRamp, type RampName} from '../../engine/ramps';
import {SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance, TooltipContent, TooltipRow} from '../scene';
import {formatNumber, formatPValue, runGuarded} from './b4-format';
import {
  getVariableInfo,
  loadGeography,
  type Geography,
  type GeographyId,
  type VariableId
} from './b4-geography';
import {
  compileStatGraph,
  createStatBuffers,
  GI_BIN_COUNT,
  MASKED_CLASS,
  MORAN_CLASS_COUNT,
  type StatBuffers,
  type StatVariant
} from './b4-hotspot-stats';
import {createGeographyBuffers, type GeographyBuffers} from './b4-layers';
import {
  getContextLineColor,
  getGiPalette,
  getGiTable,
  GI_BREAKS,
  GI_CRITICAL_Z,
  getHairlineColor,
  getMoranPalette,
  getMoranQuadrantColors,
  getStateLineStyle,
  GI_LEGEND_TO_CATEGORY,
  MORAN_LEGEND_TO_CATEGORY,
  Z_DISPLAY_RANGE
} from './hot-spots.style';
import {
  createWeightsCore,
  type WeightsConfig,
  type WeightsCore,
  type WeightsSource
} from './b4-weights-core';

/** Where the hot spots are computed. */
export type HotSpotSource = 'nature-cells' | 'us-counties' | 'chicago-tracts';

/** Option state of the hot-spots scene. */
export type HotSpotsOptions = {
  source: HotSpotSource;
  category: string;
  hours: readonly [number, number];
  family: GPUCellFamily;
  /** Quadbin zoom level. */
  resolution: number;
  /** H3 resolution. */
  h3Resolution: number;
  radiusCells: number;
  variable: VariableId;
  weights: WeightsSource;
  k: number;
  bandFactor: number;
  weightTransform: 'binary' | 'row';
  statistic: 'gi-star' | 'local-moran';
  selfWeight: 'include' | 'exclude';
  inference: 'analytic' | 'permutation' | 'none';
  falseDiscoveryRate: boolean;
  alternative: GPUPermutationAlternative;
  maximumNeighbors: string;
  permutations: number;
  seed: number;
  significance: number;
  display: 'classes' | 'zscore' | 'values';
  ramp: RampName;
  /** How the analysed values are classed for display (`continuous` = unclassed ramp). */
  classification: ClassificationMethod | 'continuous';
  classCount: number;
  showNotSignificant: boolean;
  showOutlines: boolean;
  /** Labels the strongest hot spots (or High-High clusters) on the map. */
  labelHotSpots: boolean;
  /** Draws the neighbourhood band as a ring around the strongest hot spot (cells only). */
  showBand: boolean;
  /** Swipe-compares equal-interval and natural-breaks classes of the values (rule 9). */
  compareBreaks: boolean;
};

/** One classification of the analysed values, with its goodness of variance fit. */
export type HotSpotValueTable = {
  method: ClassificationMethod;
  table: ClassTable;
  /** Goodness of variance fit of the breaks, 0..1. */
  gvf: number;
};

/**
 * Value classes of the `values` display, computed on the CPU from the last snapshot. `table` is
 * what the legend shows; with the compare toggle `alternate` is the equal-interval version drawn
 * on the left of the divider (the legend stays frozen on `table`).
 */
export type HotSpotValueClasses = HotSpotValueTable & {
  counts: number[];
  alternate: HotSpotValueTable | null;
};

/** Rows of the cell table and slots per row of its neighbour CSR. */
const CELL_CAPACITY = 16_384;
const CELL_NEIGHBORS_PER_ROW = 40;
const HIDDEN: SpatialAnalysisColor = [0, 0, 0, 0];
/** Hot spots labelled on the map. */
const LABELLED_HOT_SPOTS = 3;
/** Bars of the z-score histogram. */
const Z_BIN_COUNT = 32;
/** Ground metres per degree, averaged over east-west and north-south at Chicago's latitude. */
const METERS_PER_DEGREE_MEAN = 97_000;
/** Average H3 edge length in metres per resolution. */
const H3_EDGE_METERS: Record<number, number> = {
  4: 22_606,
  5: 8_544,
  6: 3_229,
  7: 1_220,
  8: 461,
  9: 174
};
const EQUATOR_CIRCUMFERENCE = 40_075_016.68;
const EARTH_RADIUS = 6_378_137;
const DEGREES = Math.PI / 180;

const geographyCache = new Map<GeographyId, Promise<Geography>>();

type StatSnapshot = {
  counts: Uint32Array;
  statistics: Float32Array;
  permutationOverflow: number;
  significantCounts: Uint32Array;
  extent: [number, number];
  weightsOverflow: number;
  tableCount: number;
  tableOverflow: number;
  zScores: Float32Array;
  pValues: Float32Array;
  pseudoPValues: Float32Array;
  classes: Uint32Array;
  neighborCounts: Uint32Array;
  centers: Float32Array | null;
  values: Float32Array | null;
  /** Analysed values (finite, positive for cells), ascending; filled on first use. */
  sortedValues?: Float32Array;
};

/** What the active source (nature cells or one polygon coverage) provides to the scene. */
type HotWorld = {
  rows: number;
  /** Analysed values of the last snapshot (finite only), for classification. */
  getValues: () => Float32Array | null;
  /** `[lng, lat]` and a display name of a row, for hot-spot labels. */
  describeRow: (row: number) => {coordinate: [number, number]; name: string | null; value: number};
  /** Edge of one cell as text ("0.91 km"), or `null` for polygon coverages. */
  getCellText: () => string | null;
  /** Distance in metres below which two labelled hot spots are the same cluster. */
  labelSeparationMeters: () => number;
  /** Planar metres of a row, for label separation. */
  rowMeters: (row: number) => [number, number];
  /** Neighbourhood band in metres around a cell (cells only). */
  bandMeters: () => number | null;
  resources: SpatialAnalysisResources;
  reader: SummaryReader;
  snapshot: StatSnapshot | null;
  prepare: () => void;
  writeParameters: () => void;
  encode: (commandEncoder: CommandEncoder) => void;
  getLayers: () => Layer[];
  getTooltip: (event: {coordinate: readonly [number, number] | null}) => TooltipContent | null;
  getGraphs: () => CompiledGPUCommandGraph<never>[];
  /** GPU graph nodes encoded per frame (the "passes" of the cost line). */
  getPassCount: () => number;
  describe: () => string;
  /** Marks the analysis input buffers stale (the category filter or the variable changed). */
  writeInputs: () => void;
};

/**
 * Hot spots on the GPU: Getis-Ord Gi* hot and cold spots and local Moran clusters of nature observations
 * aggregated into cells (`addHotSpotAnalysisRecipe`) and of county or tract health measures
 * (`GPUHotSpotAnalysis`, `GPULocalMoran`), each confirmed by `GPULocalPermutationTest` on request.
 * Every compile-time variant is compiled on first use and cached; the radius, the seed, the
 * permutation count and the significance level are buffer writes.
 */
export async function createHotSpots(
  ctx: SceneContext<HotSpotsOptions>
): Promise<SceneInstance<HotSpotsOptions>> {
  const {device} = ctx;
  let destroyed = false;
  let world: HotWorld | null = null;
  let switchToken = 0;
  let dirty = true;
  let stale = true;
  /** Classes of the `values` display, or `null` while continuous / not computed yet. */
  let valueClasses: HotSpotValueClasses | null = null;
  /** Category (or class) indices an interactive legend isolates; `null` shows everything. */
  let legendHighlight: number[] | null = null;
  /** Row under the pointer, so the linked chart marker only updates when it changes. */
  let hoverRow = -1;

  /** The standing sample line of the cartouche, counted from the data of the active source. */
  const publishSample = (source: HotSpotSource, geography?: Geography) => {
    if (source === 'nature-cells') {
      const observations = ctx.datasets.get('chicago-nature');
      const year = new Date(
        observations.column<Uint32Array>('timestamp')[0] * 1000
      ).getUTCFullYear();
      ctx.setFurniture({
        title: {sample: `${formatCount(observations.count)} iNaturalist records, Chicago, ${year}`}
      });
    } else if (geography) {
      ctx.setFurniture({
        title: {
          sample: `${formatCount(geography.count)} ${geography.unitPlural}, CDC PLACES model estimates`
        }
      });
    }
  };

  const getGeography = (id: GeographyId) => {
    let promise = geographyCache.get(id);
    if (!promise) {
      promise = loadGeography(id, ctx.datasets, ctx.signal);
      promise.catch(() => geographyCache.delete(id));
      geographyCache.set(id, promise);
    }
    return promise;
  };

  const getStatVariant = (precomputed = false): StatVariant => {
    const o = ctx.options;
    const isGi = o.statistic === 'gi-star';
    return {
      statistic: o.statistic,
      inference: isGi && o.inference === 'none' ? 'analytic' : o.inference,
      selfWeight: isGi && o.selfWeight === 'exclude' ? 0 : 1,
      falseDiscoveryRate: o.falseDiscoveryRate && (isGi || o.inference !== 'none'),
      alternative: o.alternative,
      maximumNeighbors: Number(o.maximumNeighbors),
      precomputed
    };
  };

  const getStatKey = (variant: StatVariant) =>
    [
      variant.statistic,
      variant.inference,
      variant.selfWeight,
      variant.falseDiscoveryRate,
      variant.inference === 'permutation'
        ? `${variant.alternative}-${variant.maximumNeighbors}`
        : '',
      variant.precomputed ? 'pre' : 'full'
    ].join(':');

  const getNoDataLabel = () =>
    ctx.options.source === 'nature-cells' ? 'No observations' : 'No data';
  const getGiTableNow = () => getGiTable(ctx.ground(), getNoDataLabel());

  /** The class table the layer, the legend and the tooltip of the current display share. */
  const publishTables = () => {
    ctx.setLegendData('giTable', getGiTableNow());
  };

  const getPaletteStyle = (
    buffers: StatBuffers,
    noData: SpatialAnalysisColor
  ): SpatialAnalysisStyleProps => {
    const o = ctx.options;
    if (o.display === 'zscore') {
      return {
        values: buffers.zScores,
        valueFormat: 'float32',
        colormap: 'rdbu',
        valueRange: Z_DISPLAY_RANGE,
        noDataColor: noData
      };
    }
    const table = getGiTableNow();
    return {
      values: buffers.classes,
      valueFormat: 'uint32',
      colormap: 'category',
      palette:
        o.statistic === 'gi-star'
          ? getGiPalette(table, o.showNotSignificant)
          : getMoranPalette(table, o.showNotSignificant),
      noDataValue: MASKED_CLASS,
      noDataColor: noData,
      highlightClasses: legendHighlight
    };
  };

  /**
   * Styles of the `values` display, one per layer: classed with the shared class table (two with
   * the compare toggle, equal interval on side a and natural breaks on side b over the same
   * buffers), or a continuous ramp when unclassed.
   */
  const getValueStyles = (
    values: Buffer,
    extent: Buffer
  ): {style: SpatialAnalysisStyleProps; side?: 'a' | 'b'}[] => {
    const o = ctx.options;
    const noDataColor = NO_DATA_COLOR[ctx.ground()];
    if (!valueClasses) {
      return [
        {
          style: {
            values,
            valueFormat: 'float32',
            colormap: o.ramp,
            extent,
            sqrtScale: o.source === 'nature-cells',
            noDataColor
          }
        }
      ];
    }
    const classed = (table: ClassTable, side?: 'a' | 'b') => ({
      style: {
        values,
        valueFormat: 'float32' as const,
        colormap: o.ramp,
        ...getClassTableLayerProps(table),
        highlightClasses: legendHighlight,
        noDataColor
      },
      side
    });
    if (o.compareBreaks && valueClasses.alternate) {
      return [classed(valueClasses.alternate.table, 'a'), classed(valueClasses.table, 'b')];
    }
    return [classed(valueClasses.table)];
  };

  /** Classes the analysed values on the CPU (one pass over at most a few thousand rows). */
  const updateValueClasses = () => {
    const o = ctx.options;
    const values = world?.getValues();
    if (!values || o.classification === 'continuous' || values.length < 2) {
      valueClasses = null;
      ctx.setLegendData('valueClasses', null);
      return;
    }
    const cells = o.source === 'nature-cells';
    const info = getVariableInfo(o.variable);
    const extent = getExtent(values);
    const build = (method: ClassificationMethod): HotSpotValueTable => {
      const breaks = getClassBreaks(values, o.classCount, method);
      const gvf = getGoodnessOfVarianceFit(values, breaks);
      const table = makeClassTable({
        breaks,
        // Registry (rule 2): weights counts are BuPu because the park fill is green; health
        // measures are the "people" hue.
        colors: getRegistryColors(
          cells ? 'natureWeights' : 'people',
          ctx.ground(),
          breaks.length + 1
        ),
        unit: cells ? 'records' : info.unit,
        extent,
        format: cells ? formatCount : (value: number) => value.toFixed(info.digits),
        method: `${CLASSIFICATION_METHOD_INFO[method].label}, GVF ${gvf.toFixed(2)}`,
        noData: {label: getNoDataLabel()}
      });
      return {method, table, gvf};
    };
    // Breaks stay fixed across every toggle that does not change the values (rule 7).
    const primary = build(o.compareBreaks ? 'natural-breaks' : o.classification);
    valueClasses = {
      ...primary,
      counts: getClassCounts(values, primary.table.breaks),
      alternate: o.compareBreaks ? build('equal-interval') : null
    };
    ctx.setLegendData('valueClasses', valueClasses);
    ctx.setReadout('gvfNatural', primary.gvf.toFixed(2));
    ctx.setReadout('gvfEqual', valueClasses.alternate?.gvf.toFixed(2) ?? null);
  };

  /** Clears the hover marker of the z chart; returns `null` for a tooltip. */
  const clearHover = (): null => {
    if (hoverRow !== -1) {
      hoverRow = -1;
      publishZChart(null);
    }
    return null;
  };

  /** Value of `value` among the ascending `sorted` as a 0-1 rank (share at or below). */
  const getRank = (sorted: Float32Array, value: number): number => {
    let low = 0;
    let high = sorted.length;
    while (low < high) {
      const middle = (low + high) >> 1;
      if (sorted[middle] <= value) low = middle + 1;
      else high = middle;
    }
    return sorted.length ? low / sorted.length : Number.NaN;
  };

  /** Ascending analysed values of a snapshot, sorted once per snapshot. */
  const getSortedValues = (target: HotWorld, snapshot: StatSnapshot): Float32Array => {
    snapshot.sortedValues ??= Float32Array.from(target.getValues() ?? []).sort();
    return snapshot.sortedValues;
  };

  /** Class label of a raw class value, capitalised for a tooltip. */
  const describeResult = (value: number): string => {
    const text = describeClass(value);
    return `${text[0].toUpperCase()}${text.slice(1)}`;
  };

  /** Swatch colour of a raw class value (Gi* bin or Moran quadrant code). */
  const getResultSwatch = (value: number) => {
    if (value === MASKED_CLASS) return undefined;
    if (ctx.options.statistic === 'gi-star') {
      const bin = value > 0x7fffffff ? value - 0x100000000 : value;
      return getGiTableNow().colors[bin + 3];
    }
    const quadrants = getMoranQuadrantColors(getGiTableNow());
    return [
      quadrants.notSignificant,
      quadrants.highHigh,
      quadrants.lowHigh,
      quadrants.lowLow,
      quadrants.highLow
    ][value];
  };

  /**
   * The first tooltip row: the mapped value of the current display with its class swatch (the
   * Gi* result, the z-score or the analysed value).
   */
  const describeMappedRow = (
    snapshot: StatSnapshot,
    row: number,
    value: number,
    valueLabel: string,
    valueText: string
  ): TooltipRow => {
    const o = ctx.options;
    const z = snapshot.zScores[row];
    if (o.display === 'zscore') {
      const [red, green, blue] = sampleRamp(
        'rdbu',
        (z - Z_DISPLAY_RANGE[0]) / (Z_DISPLAY_RANGE[1] - Z_DISPLAY_RANGE[0])
      );
      return {
        label: o.statistic === 'gi-star' ? 'Gi* z-score' : 'Local Moran z-score',
        value: formatSigned(z, 2),
        swatch: [red, green, blue, 255],
        emphasis: true
      };
    }
    if (o.display === 'values') {
      const index = valueClasses ? getClassIndexOf(valueClasses.table, value) : -1;
      return {
        label: valueLabel,
        value: valueText,
        swatch: valueClasses && index >= 0 ? valueClasses.table.colors[index] : undefined,
        emphasis: true
      };
    }
    return {
      label: o.statistic === 'gi-star' ? 'Gi* result' : 'Local Moran result',
      value: describeResult(snapshot.classes[row]),
      swatch: getResultSwatch(snapshot.classes[row]),
      emphasis: true
    };
  };

  /** Statistical rows shared by both worlds: z (unless mapped), p-values and neighbours. */
  const describeStatisticRows = (snapshot: StatSnapshot, row: number): TooltipRow[] => {
    const o = ctx.options;
    const rows: TooltipRow[] = [];
    if (o.display !== 'zscore') {
      rows.push({
        label: o.statistic === 'gi-star' ? 'Gi* z-score' : 'Local Moran z-score',
        value: formatSigned(snapshot.zScores[row], 2)
      });
    }
    rows.push({label: 'Analytic p-value', value: formatPValue(snapshot.pValues[row])});
    if (o.inference === 'permutation') {
      rows.push({label: 'Permutation p-value', value: formatPValue(snapshot.pseudoPValues[row])});
    }
    rows.push({label: 'Neighbours', value: formatCount(snapshot.neighborCounts[row])});
    return rows;
  };

  /** Histogram of the z-scores with the 90, 95 and 99 % critical values (the Gi* classes). */
  const publishZChart = (hoveredZ: number | null) => {
    const target = world;
    const snapshot = target?.snapshot;
    if (!target || !snapshot) {
      ctx.setChart('zHistogram', null);
      return;
    }
    const o = ctx.options;
    const bins = new Array<number>(Z_BIN_COUNT).fill(0);
    const [low, high] = Z_DISPLAY_RANGE;
    const rows = Math.min(snapshot.tableCount, snapshot.classes.length);
    for (let row = 0; row < rows; row++) {
      const z = snapshot.zScores[row];
      if (snapshot.classes[row] === MASKED_CLASS || !Number.isFinite(z)) continue;
      bins[
        Math.min(Z_BIN_COUNT - 1, Math.max(0, Math.floor(((z - low) / (high - low)) * Z_BIN_COUNT)))
      ]++;
    }
    const isGi = o.statistic === 'gi-star';
    ctx.setChart('zHistogram', {
      kind: 'histogram',
      values: bins,
      xDomain: Z_DISPLAY_RANGE,
      title: isGi ? 'Gi* z-scores' : 'Local Moran z-scores',
      xLabel: 'z-score (standard deviations)',
      yLabel: 'Places',
      ...(isGi ? {breaks: GI_BREAKS, classColors: getGiTableNow().colors} : {}),
      markers: GI_CRITICAL_Z.flatMap(z => [{x: z, label: formatSigned(z, 2)}, {x: -z}]),
      now: hoveredZ ?? undefined,
      nowLabel: 'This place',
      table: false,
      description:
        'Histogram of z-scores: most places sit near zero; the guides mark the 90, 95 and 99 percent critical values.'
    });
  };

  /**
   * Labels the strongest hot spots (Gi*: highest positive z among significant rows; local Moran:
   * High-High rows by z), one per cluster, and the neighbourhood band around the first.
   */
  const updateHotSpotLabels = () => {
    const o = ctx.options;
    const target = world;
    const snapshot = target?.snapshot;
    if (!target || !snapshot || !o.labelHotSpots || o.display === 'values') {
      ctx.setAnnotations('hot-spots', null);
      ctx.setAnnotations('band', null);
      return;
    }
    const isGi = o.statistic === 'gi-star';
    const candidates: number[] = [];
    const rows = Math.min(snapshot.tableCount, snapshot.classes.length);
    for (let row = 0; row < rows; row++) {
      const value = snapshot.classes[row];
      const hot = isGi ? value >= 1 && value <= 3 : value === 1;
      if (hot && Number.isFinite(snapshot.zScores[row])) candidates.push(row);
    }
    candidates.sort((a, b) => snapshot.zScores[b] - snapshot.zScores[a]);
    const separation = target.labelSeparationMeters();
    const chosen: number[] = [];
    for (const row of candidates) {
      const [x, y] = target.rowMeters(row);
      const apart = chosen.every(other => {
        const [ox, oy] = target.rowMeters(other);
        return (ox - x) ** 2 + (oy - y) ** 2 >= separation ** 2;
      });
      if (apart) chosen.push(row);
      if (chosen.length === LABELLED_HOT_SPOTS) break;
    }
    const unit = o.source === 'nature-cells' ? 'records' : getVariableInfo(o.variable).unit;
    const labels: MapAnnotation[] = chosen.map((row, rank) => {
      const {coordinate, name, value} = target.describeRow(row);
      const z = formatSigned(snapshot.zScores[row], 1);
      // Finding notes: a number with its unit from the readback, the place from the gazetteer.
      const place =
        name ?? nearestPlaceLabel(CHICAGO, coordinate as LngLat, {maxDistanceMeters: 8000});
      return {
        kind: 'note',
        id: `hot-${rank}`,
        coordinate,
        title:
          o.source === 'nature-cells'
            ? `${formatCount(value)} ${unit}, z ${z}`
            : `${formatNumber(value, 1)} ${unit}`,
        text: o.source === 'nature-cells' ? (place ?? undefined) : `${place}, z ${z}`,
        tone: rank === 0 ? 'accent' : 'ink',
        priority: 6 - rank
      };
    });
    ctx.setAnnotations('hot-spots', labels.length ? labels : null);
    const band = target.bandMeters();
    if (o.showBand && band && chosen.length) {
      ctx.setAnnotations('band', [
        {
          kind: 'ring',
          id: 'neighbourhood',
          coordinate: target.describeRow(chosen[0]).coordinate,
          radiusMeters: band,
          text: `neighbourhood ${formatDistance(band)}`,
          dashed: true
        }
      ]);
    } else {
      ctx.setAnnotations('band', null);
    }
  };

  const parseSnapshot = (bytes: ArrayBuffer, rows: number, cells: boolean): StatSnapshot => {
    let offset = 0;
    const floats = (count: number) => {
      const view = new Float32Array(bytes, offset, count);
      offset += count * 4;
      return view;
    };
    const words = (count: number) => {
      const view = new Uint32Array(bytes, offset, count);
      offset += count * 4;
      return view;
    };
    const counts = words(GI_BIN_COUNT);
    const statistics = floats(4);
    const permutationOverflow = words(1)[0];
    const significantCounts = words(2);
    const extent = floats(2);
    const weightsOverflow = words(1)[0];
    const zScores = floats(rows);
    const pValues = floats(rows);
    const pseudoPValues = floats(rows);
    const classes = words(rows);
    const neighborCounts = words(rows);
    let tableCount = rows;
    let tableOverflow = 0;
    let centers: Float32Array | null = null;
    let values: Float32Array | null = null;
    if (cells) {
      tableCount = words(1)[0];
      tableOverflow = words(1)[0];
      centers = floats(rows * 2);
      values = floats(rows);
    }
    return {
      counts,
      statistics,
      permutationOverflow,
      significantCounts,
      extent: [extent[0], extent[1]],
      weightsOverflow,
      tableCount,
      tableOverflow,
      zScores,
      pValues,
      pseudoPValues,
      classes,
      neighborCounts,
      centers,
      values
    };
  };

  const getReaderSources = (
    buffers: StatBuffers,
    rows: number,
    weightsOverflow: Buffer,
    cells: {tableCount: Buffer; tableOverflow: Buffer; centers: Buffer; values: Buffer} | null
  ) => [
    {buffer: buffers.counts, size: GI_BIN_COUNT * 4},
    {buffer: buffers.statistics, size: 16},
    {buffer: buffers.permutationOverflow, size: 4},
    {buffer: buffers.significantCounts, size: 8},
    {buffer: buffers.valuesExtent, size: 8},
    {buffer: weightsOverflow, size: 4},
    {buffer: buffers.zScores, size: rows * 4},
    {buffer: buffers.pValues, size: rows * 4},
    {buffer: buffers.pseudoPValues, size: rows * 4},
    {buffer: buffers.classes, size: rows * 4},
    {buffer: buffers.neighborCounts, size: rows * 4},
    ...(cells
      ? [
          {buffer: cells.tableCount, size: 4},
          {buffer: cells.tableOverflow, size: 4},
          {buffer: cells.centers, size: rows * 8},
          {buffer: cells.values, size: rows * 4}
        ]
      : [])
  ];

  /** Class labels of the active statistic, for tooltips. */
  const describeClass = (value: number): string => {
    const o = ctx.options;
    if (value === MASKED_CLASS) return 'no data';
    if (o.statistic === 'gi-star') {
      const bin = value > 0x7fffffff ? value - 0x100000000 : value;
      if (bin === 0) return 'not significant';
      const confidence = [90, 95, 99][Math.abs(bin) - 1] ?? 0;
      return `${bin > 0 ? 'hot spot' : 'cold spot'}, ${confidence}%`;
    }
    return (
      [
        'not significant',
        'High-High cluster',
        'Low-High outlier',
        'Low-Low cluster',
        'High-Low outlier'
      ][value] ?? 'unknown'
    );
  };

  const updateReadouts = () => {
    const target = world;
    const snapshot = target?.snapshot;
    if (!target || !snapshot) return;
    const o = ctx.options;
    const count = (index: number) => formatCount(snapshot.counts[index]);
    const rows = snapshot.tableCount;
    const analysed = Math.max(0, snapshot.statistics[0]) || rows;
    const isGi = o.statistic === 'gi-star';
    const classCount = isGi ? GI_BIN_COUNT : MORAN_CLASS_COUNT;
    let total = 0;
    for (let index = 0; index < classCount; index++) total += snapshot.counts[index];
    ctx.setReadout('places', formatCount(analysed));
    if (isGi) {
      ctx.setReadout('hot', `${count(6)} / ${count(5)} / ${count(4)}`);
      ctx.setReadout('cold', `${count(0)} / ${count(1)} / ${count(2)}`);
      ctx.setReadout('notSignificant', count(3));
      ctx.setReadout('outliers', 'n/a for Gi*');
      ctx.setReadout(
        'hotTotal',
        formatCount(snapshot.counts[4] + snapshot.counts[5] + snapshot.counts[6])
      );
      ctx.setReadout(
        'coldTotal',
        formatCount(snapshot.counts[0] + snapshot.counts[1] + snapshot.counts[2])
      );
      ctx.setReadout('notSignificantShare', formatPercent(snapshot.counts[3] / Math.max(total, 1)));
    } else {
      ctx.setReadout('hot', count(1));
      ctx.setReadout('cold', count(3));
      ctx.setReadout('outliers', `${count(2)} / ${count(4)}`);
      ctx.setReadout('notSignificant', count(0));
      ctx.setReadout('hotTotal', count(1));
      ctx.setReadout('coldTotal', count(3));
      ctx.setReadout('notSignificantShare', formatPercent(snapshot.counts[0] / Math.max(total, 1)));
    }
    ctx.setReadout(
      'moments',
      `${formatNumber(snapshot.statistics[1], 2)} / ${formatNumber(snapshot.statistics[3], 2)}`
    );
    if (o.inference === 'permutation') {
      const confirmed = formatCount(snapshot.significantCounts[1]);
      ctx.setReadout('confirmed', confirmed);
      ctx.setReadout(
        'permutation',
        `${confirmed} ${target.rows === snapshot.tableCount && !snapshot.centers ? 'places' : 'cells'} confirmed (${o.permutations} permutations)`
      );
    } else {
      ctx.setReadout('confirmed', null);
      ctx.setReadout(
        'permutation',
        o.inference === 'none' ? 'no test: every quadrant shown' : 'analytic p-values'
      );
    }
    const overflow =
      snapshot.weightsOverflow !== 0 || snapshot.tableOverflow !== 0
        ? 'overflow: reduce the radius or the resolution'
        : snapshot.permutationOverflow !== 0
          ? `ok; rows with more than ${o.maximumNeighbors} neighbours are skipped by the permutation test`
          : 'ok';
    ctx.setReadout('capacity', overflow);
    ctx.setReadout('scale', target.describe());
    ctx.setReadout('cellSize', target.getCellText());
    if (Number.isFinite(snapshot.extent[0]) && Number.isFinite(snapshot.extent[1])) {
      ctx.setLegendExtent('display', snapshot.extent);
    }
    // Per-class counts for the classed legends (Gi* bins low to high; Moran quadrant codes).
    ctx.setLegendData(
      'classCounts',
      isGi ? Array.from(snapshot.counts.slice(0, 7)) : Array.from(snapshot.counts.slice(0, 5))
    );
    ctx.setLegendData(
      'basis',
      target.getCellText() ? `per ${target.getCellText()} cell` : undefined
    );
    publishTables();
    updateValueClasses();
    updateHotSpotLabels();
    publishZChart(null);
    ctx.setCost({records: analysed, passes: target.getPassCount()});
  };

  const makeReader = (
    resources: SpatialAnalysisResources,
    id: string,
    sources: ReturnType<typeof getReaderSources>,
    rows: number,
    cells: boolean,
    onWorld: (snapshot: StatSnapshot) => void
  ) =>
    new SummaryReader(resources, id, sources, bytes =>
      runGuarded('hot-spot snapshot', () => onWorld(parseSnapshot(bytes, rows, cells)))
    );

  // ---------------------------------------------------------------------------------------
  // Nature cells: addHotSpotAnalysisRecipe
  // ---------------------------------------------------------------------------------------
  const buildCellsWorld = (tracts: Geography): HotWorld => {
    const observations = ctx.datasets.get('chicago-nature');
    const resources = new SpatialAnalysisResources(device, 'hotspots-cells');
    const create = (name: string, data: number | Float32Array | Uint32Array) =>
      resources.createBuffer(name, data);
    const pointCount = observations.count;
    const positionsLngLat = observations.column<Float32Array>('position');
    const timestamps = observations.column<Uint32Array>('timestamp');
    const categories = observations.column<Uint8Array>('category');
    const positionsBuffer = create('nature-positions', Float32Array.from(positionsLngLat));
    const pointMask = create('nature-mask', new Uint32Array(pointCount).fill(1));
    const hours = new Uint8Array(pointCount);
    for (let index = 0; index < pointCount; index++) {
      hours[index] = Math.floor((timestamps[index] % 86_400) / 3600);
    }
    const rows = CELL_CAPACITY;
    const slots = rows * CELL_NEIGHBORS_PER_ROW;
    const tableCells = create('table-cells', rows * 8);
    const tableCounts = create('table-counts', rows * 4);
    const tableCount = create('table-count', 4);
    const tableOverflow = create('table-overflow', 4);
    const centersDegrees = create('centers-degrees', rows * 8);
    const centersMeters = create('centers-meters', rows * 8);
    const csr = {
      offsets: create('offsets', (rows + 1) * 4),
      neighbors: create('neighbors', slots * 4),
      weights: create('weights', slots * 4)
    };
    const weightsOverflow = create('weights-overflow', 4);
    const values = create('values', rows * 4);
    const mask = create('mask', rows * 4);
    const buffers = createStatBuffers(resources, rows, 'cells');
    const parameters = resources.createParameterBuffer(
      'autocorrelation-parameters',
      'float32',
      GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH
    );
    const permutationParameters = resources.createParameterBuffer(
      'permutation-parameters',
      'uint32',
      GPU_PERMUTATION_PARAMETER_LENGTH
    );
    const searchParameters = resources.createParameterBuffer(
      'search-parameters',
      'float32',
      GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH
    );
    let minimumLng = Infinity;
    let minimumLat = Infinity;
    let maximumLng = -Infinity;
    let maximumLat = -Infinity;
    for (let index = 0; index < pointCount; index++) {
      minimumLng = Math.min(minimumLng, positionsLngLat[index * 2]);
      maximumLng = Math.max(maximumLng, positionsLngLat[index * 2]);
      minimumLat = Math.min(minimumLat, positionsLngLat[index * 2 + 1]);
      maximumLat = Math.max(maximumLat, positionsLngLat[index * 2 + 1]);
    }
    const [originLng, originLat] = tracts.origin;
    const cellProjection = new LocalMetricProjection(tracts.origin);
    const tractBuffers = createGeographyBuffers(resources, tracts, 'cells-tracts');

    const getResolution = () =>
      ctx.options.family === 'quadbin' ? ctx.options.resolution : ctx.options.h3Resolution;

    const getSpacing = (): number => {
      const o = ctx.options;
      return o.family === 'quadbin'
        ? (EQUATOR_CIRCUMFERENCE / 2 ** o.resolution) * Math.cos(originLat * DEGREES)
        : Math.sqrt(3) * (H3_EDGE_METERS[o.h3Resolution] ?? 461);
    };
    const getCellRadius = (): number => {
      const o = ctx.options;
      return o.family === 'quadbin' ? getSpacing() / 2 : (H3_EDGE_METERS[o.h3Resolution] ?? 461);
    };

    const recipes = new Map<string, CompiledGPUCommandGraph<void>>();
    const statGraphs = new Map<string, CompiledGPUCommandGraph<void>>();

    const getRecipeKey = () => {
      const o = ctx.options;
      const variant = getStatVariant();
      return [
        o.family,
        getResolution(),
        variant.selfWeight,
        variant.falseDiscoveryRate,
        o.statistic === 'gi-star' && o.inference === 'permutation'
          ? `perm-${o.alternative}-${o.maximumNeighbors}`
          : 'plain'
      ].join(':');
    };

    const buildRecipe = (): CompiledGPUCommandGraph<void> => {
      const o = ctx.options;
      const variant = getStatVariant();
      const graph = new GPUCommandGraph<void>(device, {id: `hotspots-recipe-${getRecipeKey()}`});
      const view = <Format extends 'float32' | 'uint32' | 'sint32' | 'float32x2' | 'uint32x2'>(
        name: string,
        buffer: Buffer,
        format: Format,
        length: number
      ) => importGraphBuffer(graph, name, buffer, format, length);
      const table = {
        cells: view('table-cells', tableCells, 'uint32x2', rows),
        counts: view('table-counts', tableCounts, 'uint32', rows),
        count: view('table-count', tableCount, 'uint32', 1),
        overflow: view('table-overflow', tableOverflow, 'uint32', 1)
      };
      const centers = view('centers-degrees', centersDegrees, 'float32x2', rows);
      const usePermutation = o.statistic === 'gi-star' && o.inference === 'permutation';
      addHotSpotAnalysisRecipe(graph, {
        id: 'hot-spot',
        source: {
          kind: 'points',
          positions: view('nature-positions', positionsBuffer, 'float32x2', pointCount),
          mask: view('nature-mask', pointMask, 'uint32', pointCount),
          family: o.family,
          resolution: getResolution(),
          tableCapacity: rows,
          neighborCapacity: slots,
          gridSize: [128, 128],
          neighborSearchParameters: searchParameters.importToGraph(graph),
          table,
          centers
        },
        parameters: parameters.importToGraph(graph),
        selfWeight: variant.selfWeight,
        falseDiscoveryRate: variant.falseDiscoveryRate,
        zScores: view('z-scores', buffers.zScores, 'float32', rows),
        bins: view('bins', buffers.base, 'sint32', rows),
        pValues: view('p-values', buffers.pValues, 'float32', rows),
        neighborCounts: view('neighbor-counts', buffers.neighborCounts, 'uint32', rows),
        globalStatistics: view('statistics', buffers.statistics, 'float32', 4),
        weights: {
          offsets: view('offsets', csr.offsets, 'uint32', rows + 1),
          neighbors: view('neighbors', csr.neighbors, 'uint32', slots),
          weights: view('weights', csr.weights, 'float32', slots)
        },
        weightsOverflow: view('weights-overflow', weightsOverflow, 'uint32', 1),
        permutation: usePermutation
          ? {
              parameters: permutationParameters.importToGraph(graph),
              maximumPermutations: 999,
              statistic: variant.selfWeight === 1 ? 'localGStar' : 'localG',
              alternative: o.alternative,
              maximumNeighbors: Number(o.maximumNeighbors),
              falseDiscoveryRate: variant.falseDiscoveryRate,
              exceedances: view('exceedances', buffers.exceedances, 'uint32', rows),
              pseudoPValues: view('pseudo-p-values', buffers.pseudoPValues, 'float32', rows),
              significant: view('significant', buffers.significant, 'uint32', rows)
            }
          : undefined
      });
      // Adapters: counts to analysis values and mask, cell centres to planar metres.
      const valuesView = view('values', values, 'float32', rows);
      const maskView = view('mask', mask, 'uint32', rows);
      addKernelPass(graph, {
        id: 'cells-values',
        invocationCount: rows,
        bindings: [
          {name: 'counts', view: table.counts, type: 'u32', access: 'read'},
          {name: 'values', view: valuesView, type: 'f32', access: 'read_write'},
          {name: 'mask', view: maskView, type: 'u32', access: 'read_write'}
        ],
        body: `let count = counts[countsOffset + index];
  values[valuesOffset + index] = f32(count);
  mask[maskOffset + index] = select(0u, 1u, count > 0u);`
      });
      addKernelPass(graph, {
        id: 'cells-centers',
        invocationCount: rows,
        bindings: [
          {name: 'degrees', view: centers, type: 'f32', access: 'read'},
          {
            name: 'meters',
            view: view('centers-meters', centersMeters, 'float32x2', rows),
            type: 'f32',
            access: 'read_write'
          }
        ],
        // Web Mercator offsets around the origin, as deck.gl's METER_OFFSETS system places them.
        body: `let longitude = degrees[degreesOffset + index * 2u];
  let latitude = degrees[degreesOffset + index * 2u + 1u];
  let scale = ${EARTH_RADIUS.toFixed(1)} * cos(${(originLat * DEGREES).toFixed(12)});
  let projected = log(tan(0.7853981633974483 + latitude * ${DEGREES.toFixed(15)} * 0.5));
  let origin = log(tan(0.7853981633974483 + ${(originLat * DEGREES).toFixed(12)} * 0.5));
  meters[metersOffset + index * 2u] = scale * (longitude - (${originLng.toFixed(9)})) * ${DEGREES.toFixed(15)};
  meters[metersOffset + index * 2u + 1u] = scale * (projected - origin);`
      });
      return resources.track(graph.compile());
    };

    const prepare = () => {
      const recipeKey = getRecipeKey();
      if (!recipes.has(recipeKey)) recipes.set(recipeKey, buildRecipe());
      const variant = getStatVariant(ctx.options.statistic === 'gi-star');
      const statKey = getStatKey(variant);
      if (!statGraphs.has(statKey)) {
        statGraphs.set(
          statKey,
          compileStatGraph({
            device,
            resources,
            id: `hotspots-cells-${statKey}`,
            rows,
            slots,
            weights: csr,
            values,
            mask,
            buffers,
            parameters,
            permutationParameters,
            variant
          })
        );
      }
    };

    const writeInputs = () => {
      const o = ctx.options;
      const wanted = o.category === 'all' ? -1 : Number(o.category);
      const [startHour, endHour] = o.hours;
      pointMask.write(
        Uint32Array.from(hours, (hour, index) =>
          (wanted < 0 || categories[index] === wanted) && hour >= startHour && hour < endHour
            ? 1
            : 0
        )
      );
    };

    const writeParameters = () => {
      const o = ctx.options;
      const padding = (getSpacing() * 4) / METERS_PER_DEGREE_MEAN;
      searchParameters.write(
        getGPUNeighborSearchParameterValues({
          bounds: [
            minimumLng - padding,
            minimumLat - padding,
            maximumLng + padding,
            maximumLat + padding
          ],
          radius: (o.radiusCells * getSpacing()) / METERS_PER_DEGREE_MEAN,
          weightKind: 'binary',
          rowStandardize: o.weightTransform === 'row'
        })
      );
      parameters.write(
        getGPUSpatialAutocorrelationParameterValues({significanceLevel: o.significance})
      );
      permutationParameters.write(
        getGPUPermutationParameterValues({
          seed: o.seed,
          permutations: o.permutations,
          significanceLevel: o.significance
        })
      );
    };

    const reader = makeReader(
      resources,
      'hotspots-cells',
      getReaderSources(buffers, rows, weightsOverflow, {
        tableCount,
        tableOverflow,
        centers: centersMeters,
        values
      }),
      rows,
      true,
      snapshot => {
        created.snapshot = snapshot;
        updateReadouts();
        ctx.requestLayers();
      }
    );

    const created: HotWorld = {
      rows,
      getValues() {
        const snapshot = created.snapshot;
        if (!snapshot?.values) return null;
        const count = Math.min(snapshot.tableCount, snapshot.values.length);
        return snapshot.values.slice(0, count).filter(value => value > 0);
      },
      describeRow(row) {
        const snapshot = created.snapshot;
        const x = snapshot?.centers?.[row * 2] ?? 0;
        const y = snapshot?.centers?.[row * 2 + 1] ?? 0;
        return {
          coordinate: cellProjection.unproject(x, y),
          name: null,
          value: snapshot?.values?.[row] ?? Number.NaN
        };
      },
      getCellText: () => formatDistance(getSpacing()),
      labelSeparationMeters: () => getSpacing() * 6,
      rowMeters(row) {
        const centers = created.snapshot?.centers;
        return [centers?.[row * 2] ?? 0, centers?.[row * 2 + 1] ?? 0];
      },
      bandMeters: () => ctx.options.radiusCells * getSpacing(),
      resources,
      reader,
      snapshot: null,
      prepare,
      writeParameters,
      writeInputs,
      describe() {
        const spacing = getSpacing();
        return `${(spacing / 1000).toFixed(2)} km cells, band ${((ctx.options.radiusCells * spacing) / 1000).toFixed(2)} km`;
      },
      encode(commandEncoder) {
        recipes.get(getRecipeKey())!.encode(commandEncoder, {parameters: undefined});
        const variant = getStatVariant(ctx.options.statistic === 'gi-star');
        statGraphs.get(getStatKey(variant))!.encode(commandEncoder, {parameters: undefined});
      },
      getGraphs: () =>
        [...recipes.values(), ...statGraphs.values()] as CompiledGPUCommandGraph<never>[],
      getPassCount: () =>
        (recipes.get(getRecipeKey())?.stats.nodeOrder.length ?? 0) +
        (statGraphs.get(getStatKey(getStatVariant(ctx.options.statistic === 'gi-star')))?.stats
          .nodeOrder.length ?? 0),
      getLayers() {
        const o = ctx.options;
        const ground = ctx.ground();
        const tractOrigin: [number, number, number] = [originLng, originLat, 0];
        const styles =
          o.display === 'values'
            ? getValueStyles(values, buffers.valuesExtent).map(({style, side}) => ({
                style: {...style, discardAtOrBelow: 0},
                side
              }))
            : [{style: getPaletteStyle(buffers, HIDDEN), side: undefined}];
        // Tier 3 context first: the tract outlines under the cells, thin and grey.
        const layers: Layer[] = [];
        if (o.showOutlines) {
          layers.push(
            new SpatialAnalysisSegmentLayer({
              id: 'hotspot-tract-outline',
              coordinateOrigin: tractOrigin,
              segments: tractBuffers.outline,
              instanceCount: tractBuffers.outlineSegmentCount,
              widthPixels: 0.5,
              color: getContextLineColor(ground)
            })
          );
        }
        for (const {style, side} of styles) {
          // Cells drawn as metre-sized squares or hexagons that tile like a raster; fill 0.88
          // on the paper-city ground so the parks and the lake read through.
          layers.push(
            new SpatialAnalysisPointLayer({
              id: `hotspot-cells-${o.family}-${getResolution()}${side ? `-${side}` : ''}`,
              coordinateOrigin: tractOrigin,
              positions: centersMeters,
              instanceCount: rows,
              shape: o.family === 'quadbin' ? 'square' : 'hexagon',
              radiusMeters: getCellRadius() * 0.98,
              ...style,
              compareSide: side,
              opacity: 0.88
            })
          );
        }
        return layers;
      },
      getTooltip(event) {
        const snapshot = created.snapshot;
        if (!snapshot?.centers || !snapshot.values || !event.coordinate) return clearHover();
        const [x, y] = tracts.project(event.coordinate[0], event.coordinate[1]);
        const o = ctx.options;
        const reach = getCellRadius() * 1.05;
        let best = -1;
        let bestDistance = reach * reach;
        for (let row = 0; row < Math.min(snapshot.tableCount, rows); row++) {
          const dx = snapshot.centers[row * 2] - x;
          const dy = snapshot.centers[row * 2 + 1] - y;
          const distance = dx * dx + dy * dy;
          if (distance < bestDistance) {
            bestDistance = distance;
            best = row;
          }
        }
        if (best < 0) return clearHover();
        const count = snapshot.values[best];
        if (best !== hoverRow) {
          hoverRow = best;
          publishZChart(snapshot.zScores[best]);
        }
        const radius = getCellRadius();
        const centerX = snapshot.centers[best * 2];
        const centerY = snapshot.centers[best * 2 + 1];
        const [west, south] = cellProjection.unproject(centerX - radius, centerY - radius);
        const [east, north] = cellProjection.unproject(centerX + radius, centerY + radius);
        const center = cellProjection.unproject(centerX, centerY) as LngLat;
        const highlight: MapHighlight =
          o.family === 'quadbin'
            ? {kind: 'box', bounds: [west, south, east, north]}
            : {kind: 'circle', coordinate: center, radiusMeters: radius};
        const sorted = getSortedValues(created, snapshot);
        const tooltipRows: TooltipRow[] = [
          describeMappedRow(snapshot, best, count, 'Records', formatCount(count)),
          {
            label: 'Rank',
            value: `${formatOrdinal(getRank(sorted, count) * 100)} percentile`,
            unit: 'of records per cell'
          },
          ...(o.display === 'values' ? [] : [{label: 'Records', value: formatCount(count)}]),
          ...describeStatisticRows(snapshot, best)
        ];
        return {
          title: `Cell of ${formatDistance(getSpacing())}`,
          subtitle: nearestPlaceLabel(CHICAGO, center, {maxDistanceMeters: 8000}) ?? undefined,
          rows: tooltipRows,
          anchor: center,
          highlight
        };
      }
    };
    return created;
  };

  // ---------------------------------------------------------------------------------------
  // Polygons: counties or tracts with a weights core
  // ---------------------------------------------------------------------------------------
  const buildPolygonWorld = (geography: Geography): HotWorld => {
    const resources = new SpatialAnalysisResources(device, `hotspots-${geography.id}`);
    const create = (name: string, data: number | Float32Array | Uint32Array) =>
      resources.createBuffer(name, data);
    const rows = geography.count;
    const core: WeightsCore = createWeightsCore({
      device,
      resources,
      id: 'hotspots-core',
      geography
    });
    const values = create('values', Float32Array.from(geography.getVariable(ctx.options.variable)));
    const mask = create('mask', new Uint32Array(rows));
    const buffers = createStatBuffers(resources, rows, 'polygons');
    const parameters = resources.createParameterBuffer(
      'autocorrelation-parameters',
      'float32',
      GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH
    );
    const permutationParameters = resources.createParameterBuffer(
      'permutation-parameters',
      'uint32',
      GPU_PERMUTATION_PARAMETER_LENGTH
    );
    const geometry: GeographyBuffers = createGeographyBuffers(resources, geography, 'hotspots');
    // The zone-boundary tier of the counties: state lines from the `us-states` dataset, projected
    // around the counties' origin so they sit exactly on the county edges.
    const statesGeojson =
      geography.id === 'us-counties' ? ctx.datasets.get('us-states').geojson : null;
    const stateSegments = statesGeojson
      ? projectRingsToSegments(
          getInputPolygons(statesGeojson).flatMap(({polygon}) => polygon),
          getLocalProjector(geography.origin)
        )
      : new Float32Array(0);
    const stateBuffer = stateSegments.length ? create('state-lines', stateSegments) : null;
    const statGraphs = new Map<string, CompiledGPUCommandGraph<void>>();
    let weightsDirty = true;

    const getConfig = (): WeightsConfig => {
      const o = ctx.options;
      return {
        source: o.weights,
        k: o.k,
        snapTolerance: 0,
        bandFactor: o.bandFactor,
        knnCapFactor: 0,
        weightKind: 'binary',
        kernel: 'triangular',
        power: 1,
        distanceFloor: 0,
        rowStandardize: false,
        transform: o.weightTransform === 'row' ? 'row' : 'none',
        transformKernel: 'bisquare',
        bandwidthFactor: 0,
        doubleSum: 'one'
      };
    };

    const prepare = () => {
      core.prepare(getConfig());
      const variant = getStatVariant();
      const key = getStatKey(variant);
      if (!statGraphs.has(key)) {
        statGraphs.set(
          key,
          compileStatGraph({
            device,
            resources,
            id: `hotspots-${geography.id}-${key}`,
            rows,
            slots: core.slots,
            weights: core.csr,
            values,
            mask,
            buffers,
            parameters,
            permutationParameters,
            variant
          })
        );
      }
    };

    const writeInputs = () => {
      const data = geography.getVariable(ctx.options.variable);
      values.write(Float32Array.from(data));
      mask.write(Uint32Array.from(data, value => (Number.isFinite(value) ? 1 : 0)));
    };

    const writeParameters = () => {
      const o = ctx.options;
      core.writeParameters(getConfig());
      parameters.write(
        getGPUSpatialAutocorrelationParameterValues({significanceLevel: o.significance})
      );
      permutationParameters.write(
        getGPUPermutationParameterValues({
          seed: o.seed,
          permutations: o.permutations,
          significanceLevel: o.significance
        })
      );
      weightsDirty = true;
    };

    const reader = makeReader(
      resources,
      `hotspots-${geography.id}`,
      getReaderSources(buffers, rows, core.overflow, null),
      rows,
      false,
      snapshot => {
        created.snapshot = snapshot;
        updateReadouts();
        ctx.requestLayers();
      }
    );

    writeInputs();
    const polygonProjection = new LocalMetricProjection(geography.origin);

    const created: HotWorld = {
      rows,
      getValues() {
        const data = geography.getVariable(ctx.options.variable);
        return data.filter(value => Number.isFinite(value));
      },
      describeRow(row) {
        const x = geography.centroids[row * 2];
        const y = geography.centroids[row * 2 + 1];
        return {
          coordinate: polygonProjection.unproject(x, y),
          name: `${geography.getName(row)}, ${geography.getGroupName(row)}`,
          value: geography.getVariable(ctx.options.variable)[row]
        };
      },
      labelSeparationMeters: () => geography.medianSpacing * (geography.count > 2000 ? 8 : 4),
      rowMeters: row => [geography.centroids[row * 2], geography.centroids[row * 2 + 1]],
      bandMeters: () => null,
      getCellText: () => null,
      resources,
      reader,
      snapshot: null,
      prepare,
      writeParameters,
      writeInputs,
      describe() {
        const o = ctx.options;
        return o.weights === 'band'
          ? `${(core.getBandMeters(getConfig()) / 1000).toFixed(1)} km band`
          : o.weights === 'knn'
            ? `${o.k} nearest centroids`
            : `${o.weights} contiguity`;
      },
      encode(commandEncoder) {
        if (weightsDirty) {
          core.encode(commandEncoder, getConfig());
          weightsDirty = false;
        }
        statGraphs
          .get(getStatKey(getStatVariant()))!
          .encode(commandEncoder, {parameters: undefined});
      },
      getGraphs: () =>
        [...core.getGraphs(), ...statGraphs.values()] as CompiledGPUCommandGraph<never>[],
      getPassCount: () => statGraphs.get(getStatKey(getStatVariant()))?.stats.nodeOrder.length ?? 0,
      getLayers() {
        const o = ctx.options;
        const ground = ctx.ground();
        const origin: [number, number, number] = [geography.origin[0], geography.origin[1], 0];
        const styles =
          o.display === 'values'
            ? getValueStyles(values, buffers.valuesExtent)
            : [{style: getPaletteStyle(buffers, NO_DATA_COLOR[ground]), side: undefined}];
        // Fill 1.0 on the national paper sheet, 0.88 over the city ground.
        const layers: Layer[] = styles.map(
          ({style, side}) =>
            new SpatialAnalysisPolygonLayer({
              id: `hotspot-fill${side ? `-${side}` : ''}`,
              coordinateOrigin: origin,
              triangles: geometry.triangles,
              features: geometry.features,
              vertexCount: geometry.triangleVertexCount,
              ...style,
              compareSide: side,
              opacity: geography.id === 'us-counties' ? 1 : 0.88
            })
        );
        if (o.showOutlines) {
          layers.push(
            new SpatialAnalysisSegmentLayer({
              id: 'hotspot-outline',
              coordinateOrigin: origin,
              segments: geometry.outline,
              instanceCount: geometry.outlineSegmentCount,
              // Tier 3: a thin hairline in the ground colour, never a dark mesh over thousands
              // of polygons.
              widthPixels: geography.count > 2000 ? 0.4 : 0.5,
              color: getHairlineColor(ground)
            })
          );
        }
        if (o.showOutlines && stateBuffer) {
          // The zone-boundary tier: state lines over a casing.
          const line = getStateLineStyle(ground);
          layers.push(
            new SpatialAnalysisSegmentLayer({
              id: 'hotspot-state-lines',
              coordinateOrigin: origin,
              segments: stateBuffer,
              instanceCount: stateSegments.length / 4,
              widthPixels: line.widthPixels,
              color: line.color,
              outlineColor: line.casing,
              outlineWidthPixels: (line.casingPixels - line.widthPixels) / 2
            })
          );
        }
        return layers;
      },
      getTooltip(event) {
        if (!event.coordinate) return clearHover();
        const row = geography.pick(event.coordinate[0], event.coordinate[1]);
        if (row < 0) return clearHover();
        const o = ctx.options;
        const info = getVariableInfo(o.variable);
        const value = geography.getVariable(o.variable)[row];
        const snapshot = created.snapshot;
        const valueText = Number.isFinite(value) ? value.toFixed(info.digits) : 'no data';
        const tooltipRows: TooltipRow[] = [];
        if (snapshot) {
          if (row !== hoverRow) {
            hoverRow = row;
            publishZChart(snapshot.zScores[row]);
          }
          tooltipRows.push(describeMappedRow(snapshot, row, value, info.label, valueText));
        }
        const sorted = snapshot ? getSortedValues(created, snapshot) : null;
        tooltipRows.push(
          {
            label: 'Rank',
            value: sorted ? `${formatOrdinal(getRank(sorted, value) * 100)} percentile` : '–',
            unit: `of ${geography.unitPlural}`
          },
          ...(o.display === 'values' && snapshot
            ? []
            : [{label: info.label, value: valueText, unit: info.unit}])
        );
        if (snapshot) tooltipRows.push(...describeStatisticRows(snapshot, row));
        // The hovered polygon's rings, projected back to longitude and latitude.
        const rings: LngLat[][] = [];
        for (
          let ring = geography.featureRingOffsets[row];
          ring < geography.featureRingOffsets[row + 1];
          ring++
        ) {
          const points: LngLat[] = [];
          for (
            let vertex = geography.contiguityRingOffsets[ring];
            vertex < geography.contiguityRingOffsets[ring + 1];
            vertex++
          ) {
            points.push(
              polygonProjection.unproject(
                geography.contiguityVertices[vertex * 2],
                geography.contiguityVertices[vertex * 2 + 1]
              ) as LngLat
            );
          }
          rings.push(points);
        }
        return {
          title: geography.getName(row),
          subtitle: geography.getGroupName(row),
          rows: tooltipRows,
          anchor: polygonProjection.unproject(
            geography.centroids[row * 2],
            geography.centroids[row * 2 + 1]
          ) as LngLat,
          highlight: {kind: 'polygon', rings}
        };
      }
    };
    return created;
  };

  // ---------------------------------------------------------------------------------------
  // Source switching
  // ---------------------------------------------------------------------------------------
  const buildWorldFor = async (source: HotSpotSource): Promise<HotWorld> => {
    if (source === 'nature-cells') return buildCellsWorld(await getGeography('chicago-tracts'));
    return buildPolygonWorld(await getGeography(source));
  };

  const activate = async (source: HotSpotSource) => {
    const token = ++switchToken;
    ctx.setStatus('Loading...');
    const next = await buildWorldFor(source);
    if (token !== switchToken || destroyed) {
      next.reader.stop();
      next.resources.destroy();
      return;
    }
    const previous = world;
    world = null;
    ctx.requestLayers();
    if (previous) {
      previous.reader.stop();
      setTimeout(() => previous.resources.destroy(), 200);
    }
    world = next;
    publishSample(source, source === 'nature-cells' ? undefined : await getGeography(source));
    next.writeInputs();
    next.prepare();
    next.writeParameters();
    dirty = true;
    stale = true;
    ctx.setStatus('');
    ctx.requestLayers();
  };

  world = await buildWorldFor(ctx.options.source);
  if (ctx.signal.aborted) throw new Error('aborted');
  publishSample(
    ctx.options.source,
    ctx.options.source === 'nature-cells' ? undefined : await getGeography(ctx.options.source)
  );
  world.writeInputs();
  world.prepare();
  world.writeParameters();
  ctx.setStatus('');

  return {
    getCompiledGraphs: () => world?.getGraphs() ?? [],

    setOption(id, value) {
      // A legend filter belongs to the legend it was made in.
      if (
        [
          'source',
          'statistic',
          'display',
          'classification',
          'classCount',
          'compareBreaks'
        ].includes(id)
      ) {
        legendHighlight = null;
      }
      if (id === 'source') {
        void activate(value as HotSpotSource);
        return;
      }
      const target = world;
      if (!target) return;
      if (
        id === 'classification' ||
        id === 'classCount' ||
        id === 'ramp' ||
        id === 'compareBreaks'
      ) {
        updateValueClasses();
        ctx.requestLayers();
        return;
      }
      if (
        ['display', 'labelHotSpots', 'showBand', 'showNotSignificant', 'showOutlines'].includes(id)
      ) {
        updateHotSpotLabels();
        ctx.requestLayers();
        return;
      }
      if (['category', 'hours', 'variable'].includes(id)) target.writeInputs();
      target.prepare();
      target.writeParameters();
      dirty = true;
      stale = true;
      ctx.requestLayers();
    },

    encode(commandEncoder, frame) {
      const target = world;
      if (!target) return;
      if (dirty || frame.frameIndex < 2) {
        target.encode(commandEncoder);
        dirty = false;
        stale = true;
      }
      if (stale && frame.frameIndex >= 1 && !target.reader.isPending) {
        stale = false;
        target.reader.request(commandEncoder);
      }
    },

    getLayers: () => world?.getLayers() ?? [],

    // The class tables are authored per ground, so a ground flip rebuilds them.
    onGroundChange: () => {
      publishTables();
      updateValueClasses();
      publishZChart(null);
      ctx.requestLayers();
    },

    onLegendFilter(id, classes) {
      legendHighlight =
        classes === null
          ? null
          : id === 'gi-classes'
            ? classes.map(index => GI_LEGEND_TO_CATEGORY[index])
            : id === 'moran-classes'
              ? classes.map(index => MORAN_LEGEND_TO_CATEGORY[index])
              : [...classes];
      ctx.requestLayers();
    },

    getTooltip: event => world?.getTooltip(event) ?? null,

    destroy() {
      destroyed = true;
      switchToken++;
      const target = world;
      world = null;
      if (target) {
        target.reader.stop();
        target.resources.destroy();
      }
    }
  };
}
