// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getClassPalette, makeClassTable} from '../../cartography/class-table';
import {hexToRgba, MAP_INK, OTHER_GREY} from '../../cartography/hue-registry';
import type {ClassTable} from '../../cartography/types';
import {sampleRamp, type PaletteColor} from '../../engine/ramps';
import type {MetricDefinition} from './b5-group-metrics';
import type {Classification, FrozenBreaks} from './group-statistics.stats';

/**
 * Symbolisation of the group-statistics scene that is not a shared table: the class tables of the
 * metrics, the cyclic hour classes, the proportional-circle sizes and the line tiers. Pure data:
 * no luma.gl, so the scene file can import it.
 */

/** The ground a table or line is drawn on. */
export type Ground = 'light' | 'dark';

/** Rates are observations per this many residents. */
export const RATE_PER = 1000;

/** Normalisation basis of every rate legend and tooltip row. */
export const RATE_BASIS = 'per 1,000 residents';

/** The largest proportional circle, in CSS pixels, and the smallest drawn one. */
export const COUNT_RADIUS_PIXELS = 24;
export const COUNT_RADIUS_MINIMUM_PIXELS = 1.5;

/** Fill opacity of classed polygons on the city paper ground. */
export const FILL_OPACITY = 0.88;

/** Interior breaks of the tract table: the empty class first, then the shared manual rate breaks. */
export const TRACT_RATE_BREAKS: readonly number[] = [1e-6, 1, 3, 10, 30];

/** Labels of the six tract classes: the empty class is its own class, not "no data". */
const TRACT_RATE_LABELS: readonly string[] = [
  'None',
  'Under 1',
  '1 to 3',
  '3 to 10',
  '10 to 30',
  '30 or more'
];

/** Three-hour classes of the cyclic hour table. */
export const HOUR_BREAKS: readonly number[] = [3, 6, 9, 12, 15, 18, 21];
const HOUR_LABELS: readonly string[] = [
  '0 to 3',
  '3 to 6',
  '6 to 9',
  '9 to 12',
  '12 to 15',
  '15 to 18',
  '18 to 21',
  '21 to 24'
];

/** One colour per three-hour class, sampled from the cyclic ramp at the class centre. */
export function getHourColors(): PaletteColor[] {
  return HOUR_LABELS.map((_, index) => {
    const [r, g, b] = sampleRamp('romao', (index * 3 + 1.5) / 24);
    return [r, g, b, 255] as const;
  });
}

/** Hour class (`0..7`) of an hour of day. */
export function getHourClass(hour: number): number {
  return Math.min(7, Math.max(0, Math.floor(hour / 3)));
}

/** Short break label: integers as they are, others with one decimal. */
const formatBreak = (value: number) =>
  Number.isInteger(value) ? value.toLocaleString('en-US') : value.toFixed(1);

const formatDollars = (value: number) => `$${Math.round(value / 1000)}k`;

/** Options of {@link getMetricTable}. */
export type MetricTableOptions = {
  metric: MetricDefinition;
  ground: Ground;
  breaks: FrozenBreaks;
  classification: Classification;
  minimumObservations: number;
  /** `'areas'` draws the five-class rate table; tracts and the swipe need the empty class too. */
  zoning: 'areas' | 'tracts' | 'swipe';
};

const NO_OBSERVATIONS_LABEL = 'No data';

/** The "withheld" swatch of a table: hatched, labelled by the threshold, count hidden. */
function getNoData(metric: MetricDefinition, minimumObservations: number): ClassTable['noData'] {
  return {
    label:
      metric.suppressed && minimumObservations > 0
        ? `Fewer than ${minimumObservations} observations`
        : NO_OBSERVATIONS_LABEL,
    hatched: true
  };
}

/**
 * The one class table of a metric: the layer, the legend, the tooltip swatch and the chart colours
 * all read it, so they cannot drift apart. Breaks are frozen from the unfiltered table, so a
 * filter, a swipe or a threshold changes the colour of an area and never the classes.
 */
export function getMetricTable(options: MetricTableOptions): ClassTable {
  const {metric, ground, breaks, classification, minimumObservations, zoning} = options;
  const noData = getNoData(metric, minimumObservations);
  switch (metric.family) {
    case 'rate': {
      if (zoning !== 'areas') return getTractRateTable(ground);
      return makeClassTable({
        breaks: breaks.perThousand[classification],
        scheme: 'YlGnBu',
        ground,
        unit: 'observations',
        extent: breaks.extents.perThousand,
        method: RATE_METHODS[classification],
        noData,
        format: formatBreak
      });
    }
    case 'share':
      return makeClassTable({
        breaks: metric.id === 'researchShare' ? breaks.researchShare : breaks.introducedShare,
        scheme: 'Greens',
        ground,
        unit: '% of records',
        extent:
          metric.id === 'researchShare'
            ? breaks.extents.researchShare
            : breaks.extents.introducedShare,
        method: 'Quantiles of the unfiltered shares (five classes)',
        noData,
        format: value => `${Math.round(value)}`
      });
    case 'richness':
      return makeClassTable({
        breaks: breaks.richness,
        scheme: 'YlGnBu',
        ground,
        unit: 'taxa',
        extent: breaks.extents.richness,
        method: 'Quantiles of the unfiltered counts (five classes)',
        noData,
        format: formatBreak
      });
    case 'income':
      return makeClassTable({
        breaks: breaks.income,
        scheme: 'PuBu',
        ground,
        unit: 'US dollars',
        extent: breaks.extents.income,
        method: 'Quantiles of the population-weighted incomes (five classes)',
        noData,
        format: formatDollars
      });
    default:
      return getHourTable(ground, noData);
  }
}

const RATE_METHODS: Record<Classification, string> = {
  manual: 'Manual breaks, roughly logarithmic: rates are heavy-tailed',
  quantile: 'Quantiles of the unfiltered rates: the same number of areas per class',
  equal: 'Equal intervals of the unfiltered rates'
};

/**
 * The cyclic hour table: eight three-hour classes whose first and last colours meet at midnight.
 * Hatched no-data swatch for the areas withheld.
 */
export function getHourTable(ground: Ground, noData?: ClassTable['noData']): ClassTable {
  return makeClassTable({
    breaks: HOUR_BREAKS,
    colors: getHourColors(),
    ground,
    labels: HOUR_LABELS,
    unit: 'hour of day',
    method: 'Three-hour classes of the median hour; the ring wraps at midnight',
    noData: noData ?? {label: NO_OBSERVATIONS_LABEL, hatched: true}
  });
}

/**
 * The tract rate table: the five manual rate classes of the area map plus an "empty" class below
 * them, so a tract with no record is not mistaken for a tract with no data.
 */
export function getTractRateTable(ground: Ground): ClassTable {
  const colors = [getEmptyClassColor(ground), ...getClassPalette('YlGnBu', 5, {ground})];
  return makeClassTable({
    breaks: TRACT_RATE_BREAKS,
    colors,
    ground,
    labels: TRACT_RATE_LABELS,
    unit: 'observations',
    method: 'Manual breaks shared with the area map; empty tracts are their own class',
    noData: {label: 'No residents', hatched: true}
  });
}

/** Fill of the "no records" class: a neutral grey that reads as zero, never as a data class. */
export function getEmptyClassColor(ground: Ground): PaletteColor {
  return ground === 'dark' ? [58, 66, 78, 255] : [222, 222, 216, 255];
}

/** Legend copy of a metric: the title, the basis shown after the unit and an optional note. */
export function getMetricLegendCopy(metric: MetricDefinition): {
  title: string;
  basis?: string;
  note?: string;
} {
  switch (metric.id) {
    case 'perThousand':
      return {title: 'Observations', basis: RATE_BASIS};
    case 'researchShare':
      return {title: 'Research-grade share'};
    case 'introducedShare':
      return {title: 'Introduced (non-native) share'};
    case 'richness':
      return {
        title: 'Distinct taxa',
        note: 'Climbs with the number of records, so it partly measures effort.'
      };
    case 'income':
      return {title: 'Per-capita income', basis: 'weighted by residents'};
    default:
      return {title: metric.label};
  }
}

/** Ink of selections and rings: achromatic, from the ground. */
export function getInkColor(ground: Ground, alpha = 255): PaletteColor {
  const color = hexToRgba(MAP_INK[ground].ink, alpha);
  return [color[0], color[1], color[2], color[3]];
}

/** Ground-coloured casing under selections and symbol strokes. */
export function getCasingColor(ground: Ground, alpha = 235): PaletteColor {
  const color = hexToRgba(MAP_INK[ground].halo, alpha);
  return [color[0], color[1], color[2], color[3]];
}

/** Stripes of the hatched no-data swatch: ink at about a third on light, a light ink on dark. */
export function getHatchColor(ground: Ground): PaletteColor {
  return ground === 'dark' ? [200, 210, 222, 110] : [31, 41, 51, 110];
}

/** The line passes of an area boundary, drawn in order (the first is the casing). */
export type AreaLinePass = {widthPixels: number; color: PaletteColor};

/**
 * Community-area boundary tier. As context (areas behind tract fills, or between class fills) it is
 * a 0.8 px ink line over a white gap line; as the subject (areas drawn over tract fills) it is
 * 1.8 px at alpha 230 over a 3 px ground casing.
 */
export function getAreaLinePasses(ground: Ground, role: 'context' | 'subject'): AreaLinePass[] {
  if (role === 'subject') {
    return [
      {widthPixels: 3, color: getCasingColor(ground, 235)},
      {widthPixels: 1.8, color: ground === 'dark' ? [190, 200, 212, 230] : [58, 63, 75, 230]}
    ];
  }
  return [
    {
      widthPixels: 1.6,
      color: ground === 'dark' ? [14, 17, 22, 150] : [255, 255, 255, 170]
    },
    {widthPixels: 0.8, color: ground === 'dark' ? [170, 182, 195, 140] : [58, 63, 75, 140]}
  ];
}

/** Census-tract hairline (tier 3): thin, in the ground colour. */
export function getTractHairlineColor(ground: Ground): PaletteColor {
  return ground === 'dark' ? [14, 17, 22, 128] : [255, 255, 255, 161];
}

/** The fill of the unfilled areas under the proportional circles: a faint lighter paper. */
export function getUnfilledAreaColor(ground: Ground): PaletteColor {
  return ground === 'dark' ? [255, 255, 255, 16] : [255, 255, 255, 96];
}

/** Colour of the proportional circles: the darkest class of the nature table. */
export function getCircleColor(ground: Ground): PaletteColor {
  const colors = getClassPalette('YlGnBu', 5, {ground});
  return [colors[4][0], colors[4][1], colors[4][2], 255];
}

/** Colour of a suppressed bar in the run chart: the neutral grey of "other". */
export function getSuppressedBarColor(ground: Ground): PaletteColor {
  const color = hexToRgba(OTHER_GREY[ground], 255);
  return [color[0], color[1], color[2], 255];
}

/** The scatter dot and the unselected bar colour: the nature hue, mid class. */
export function getDotColor(ground: Ground): PaletteColor {
  const colors = getClassPalette('YlGnBu', 5, {ground});
  return [colors[3][0], colors[3][1], colors[3][2], 255];
}

/** Colour of a rose sector outside the percentile range: muted grey. */
export function getMutedSectorColor(ground: Ground): PaletteColor {
  return ground === 'dark' ? [96, 104, 116, 150] : [190, 194, 198, 190];
}
