// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * The statistics half of the lasso-explorer story, in plain TypeScript: the per-point value kinds,
 * the citywide baseline, the peak summary and the linked charts. The GPU gives a histogram of the
 * selection; the chart is only honest next to a baseline, so the citywide shape is counted once on
 * the CPU from the same value columns ("citywide shape from all records") and the charts compare
 * shares, not counts. No DOM and no luma.gl: scene code feeds it a histogram and publishes the
 * result with `ctx.setChart`.
 */

import {getRegistryColors} from '../../cartography/hue-registry';
import {formatCount, formatPercent, formatSigned} from '../../cartography/live-text';
import type {ChartData} from '../chart-types';
import {formatHourWindow} from './b1-nature-data';

/** What is histogrammed over the selected records. */
export type ValueKind = 'hour' | 'weekday' | 'month' | 'category' | 'researchGrade';

/** How the linked chart reads the selection. */
export type Normalisation = 'counts' | 'share' | 'difference';

/** Static facts about one value kind. */
export type ValueKindInfo = {
  binCount: number;
  /** Fixed histogram domain (compile-time on the GPU). */
  domain: readonly [number, number];
  /** Long label of the quantity, for the readout and the option. */
  label: string;
  /** Noun used in chart titles ("hour of day"). */
  noun: string;
  /** True when the last bin joins the first (hours, weekdays, months). */
  cyclic: boolean;
};

/** The five histogram quantities. */
export const VALUE_KINDS: Record<ValueKind, ValueKindInfo> = {
  hour: {binCount: 24, domain: [0, 24], label: 'Hour of day', noun: 'hour of day', cyclic: true},
  weekday: {binCount: 7, domain: [0, 7], label: 'Day of week', noun: 'day of week', cyclic: true},
  month: {binCount: 12, domain: [0, 12], label: 'Month of year', noun: 'month', cyclic: true},
  category: {binCount: 10, domain: [0, 10], label: 'Group', noun: 'group', cyclic: false},
  researchGrade: {
    binCount: 2,
    domain: [0, 1],
    label: 'Identification',
    noun: 'identification',
    cyclic: false
  }
};

/** Weekday names, Sunday first (2023-01-01 was a Sunday). */
export const WEEKDAYS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'] as const;
/** Month names. */
export const MONTHS = [
  'Jan',
  'Feb',
  'Mar',
  'Apr',
  'May',
  'Jun',
  'Jul',
  'Aug',
  'Sep',
  'Oct',
  'Nov',
  'Dec'
] as const;
/** Day of the year on which each month of 2023 starts, plus the year end. */
export const MONTH_STARTS_2023 = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365];
const RESEARCH_GRADE_NAMES = ['Needs an ID', 'Confirmed'] as const;
const HOURS_PER_QUARTER_DAY = 6;
/** Busiest window length for the cyclic kinds (three hours, three months, three days). */
const PEAK_WINDOW_BINS = 3;

/** Chart amber per page theme: the `--chart-5` token, the map's selection hue on a card. */
const CHART_AMBER = {dark: [230, 184, 77, 255], light: [184, 134, 11, 255]} as const;
const CHART_MUTED = {dark: [107, 120, 137, 255], light: [139, 149, 165, 255]} as const;

/** Chart palette slot of the amber series (`--chart-5`). */
const AMBER_SLOT = 4;

/** Bin index of a value (the same floor the GPU histogram takes). */
export function getBinIndex(kind: ValueKind, value: number): number {
  const {binCount} = VALUE_KINDS[kind];
  return Math.max(0, Math.min(binCount - 1, Math.floor(value)));
}

/** Short names of the bins of a kind, in bin order. */
export function getBinNames(kind: ValueKind, categoryNames: readonly string[]): string[] {
  switch (kind) {
    case 'hour':
      return Array.from({length: 24}, (_, hour) => `${String(hour).padStart(2, '0')}:00`);
    case 'weekday':
      return [...WEEKDAYS];
    case 'month':
      return [...MONTHS];
    case 'category':
      return categoryNames.map(shortenCategory);
    default:
      return [...RESEARCH_GRADE_NAMES];
  }
}

/** Short group names for chart rows ("Spiders and kin" stays, "Amphibians and reptiles" shortens). */
function shortenCategory(name: string): string {
  return name === 'Amphibians and reptiles' ? 'Reptiles, amphibians' : name;
}

/** Counts of records per bin of a value column. */
export function countBins(values: Float32Array, kind: ValueKind): Float64Array {
  const counts = new Float64Array(VALUE_KINDS[kind].binCount);
  for (let index = 0; index < values.length; index++) counts[getBinIndex(kind, values[index])]++;
  return counts;
}

/** Counts as shares of their total (zeros when the total is zero). */
export function toShares(counts: ArrayLike<number>): Float64Array {
  let total = 0;
  for (let index = 0; index < counts.length; index++) total += counts[index];
  const shares = new Float64Array(counts.length);
  if (total > 0)
    for (let index = 0; index < counts.length; index++) shares[index] = counts[index] / total;
  return shares;
}

/** What the peak of a histogram says, in words and numbers. */
export type PeakSummary = {
  /** Bin with the most records. */
  peakIndex: number;
  /** Name of that bin ("11:00", "Apr"). */
  peakName: string;
  /** Share of the selection in that bin. */
  peakShare: number;
  /** Busiest window of three bins (cyclic kinds only). */
  window: {startIndex: number; share: number; text: string} | null;
};

/** Peak bin and, for cyclic kinds, the busiest three-bin window (a circular mean would hide a wrap). */
export function summarisePeak(
  counts: ArrayLike<number>,
  kind: ValueKind,
  categoryNames: readonly string[]
): PeakSummary | null {
  let total = 0;
  let peakIndex = 0;
  for (let index = 0; index < counts.length; index++) {
    total += counts[index];
    if (counts[index] > counts[peakIndex]) peakIndex = index;
  }
  if (total <= 0) return null;
  const names = getBinNames(kind, categoryNames);
  const info = VALUE_KINDS[kind];
  let window: PeakSummary['window'] = null;
  if (info.cyclic) {
    let bestStart = 0;
    let bestSum = -1;
    for (let start = 0; start < counts.length; start++) {
      let sum = 0;
      for (let offset = 0; offset < PEAK_WINDOW_BINS; offset++) {
        sum += counts[(start + offset) % counts.length];
      }
      if (sum > bestSum) {
        bestSum = sum;
        bestStart = start;
      }
    }
    const text =
      kind === 'hour'
        ? formatHourWindow([bestStart, bestStart + PEAK_WINDOW_BINS], false)
        : `${names[bestStart]} to ${names[(bestStart + PEAK_WINDOW_BINS - 1) % counts.length]}`;
    window = {startIndex: bestStart, share: bestSum / total, text};
  }
  return {peakIndex, peakName: names[peakIndex], peakShare: counts[peakIndex] / total, window};
}

/** Inputs of {@link buildSelectionChart} and {@link buildClockChart}. */
export type SelectionChartInput = {
  kind: ValueKind;
  /** GPU histogram of the selection, one count per bin. */
  counts: ArrayLike<number>;
  /** Citywide share per bin (the denominator of "more than the city"). */
  citywide: ArrayLike<number>;
  /** The previous selection, kept as a dashed ghost series. */
  pinned: {name: string; shares: ArrayLike<number>} | null;
  normalise: Normalisation;
  /** Name of the current selection ("Uptown", "Your lasso"). */
  selectionName: string;
  /** `'selection'`: the bins stretch over the selected values, so no baseline applies. */
  domainMode: 'fixed' | 'selection';
  categoryNames: readonly string[];
  /** Page theme: the card the chart sits in. */
  theme: 'light' | 'dark';
};

/** The x position of a bin on the line chart (hours run in quarter-days so the ticks read 0 6 12 18). */
function getBinX(kind: ValueKind, index: number): number {
  return kind === 'hour' ? (index + 0.5) / HOURS_PER_QUARTER_DAY : index + 0.5;
}

/** Axis text of an x position on the line chart: a tick (integer) or a bin (hover). */
function formatBinX(kind: ValueKind, categoryNames: readonly string[]) {
  const names = getBinNames(kind, categoryNames);
  return (value: number): string => {
    if (kind === 'hour') {
      return Number.isInteger(value)
        ? String(value * HOURS_PER_QUARTER_DAY)
        : names[Math.max(0, Math.min(23, Math.floor(value * HOURS_PER_QUARTER_DAY)))];
    }
    const index = Math.floor(value + 1e-9);
    return index >= names.length ? '' : (names[Math.max(0, index)] ?? '');
  };
}

const percentTicks = (value: number) => `${Math.round(value * 10) / 10}%`;
const signedPoints = (value: number) => formatSigned(Math.round(value * 10) / 10, 0);

/** Short title of what the chart shows. */
function getTitle(kind: ValueKind, normalise: Normalisation): string {
  const noun = VALUE_KINDS[kind].noun;
  if (normalise === 'counts') return `Records by ${noun}`;
  if (normalise === 'share') return `Share of the selection by ${noun}`;
  return `Selection minus citywide, by ${noun}`;
}

/**
 * The linked chart of the selection: counts, share against the citywide baseline, or the
 * difference from it. Cyclic quantities draw as a line with the baseline and the pinned previous
 * selection as ghost series; nominal groups as paired bars of selection and city. Returns `null`
 * when nothing is selected.
 */
export function buildSelectionChart(input: SelectionChartInput): ChartData | null {
  const {kind, counts, citywide, pinned, selectionName, categoryNames, theme} = input;
  const info = VALUE_KINDS[kind];
  let total = 0;
  for (let index = 0; index < counts.length; index++) total += counts[index];
  if (total <= 0) return null;
  const labels = getBinNames(kind, categoryNames);
  const values = Array.from(counts);
  const shares = toShares(counts);
  const peak = summarisePeak(counts, kind, categoryNames);
  const muted = CHART_MUTED[theme];
  const registry = getRegistryColors('deviation', theme, 5);
  const more = registry[registry.length - 1];
  const fewer = registry[0];
  const normalise = input.domainMode === 'selection' ? 'counts' : input.normalise;

  if (input.domainMode === 'selection') {
    return {
      kind: 'bars',
      title: 'Records by bin, stretched to the selection',
      values,
      labels: labels.map((_, index) => String(index + 1)),
      color: AMBER_SLOT,
      xLabel: 'bin (the range follows the selection)',
      yLabel: 'records',
      formatY: formatCount,
      description:
        'Histogram whose bins span the selected values only, so it cannot be compared with the citywide shape.'
    };
  }

  if (normalise === 'counts') {
    const peakMarker = peak ? {x: peak.peakIndex + 0.5, label: `peak ${peak.peakName}`} : null;
    if (info.cyclic) {
      return {
        kind: 'bars',
        title: getTitle(kind, normalise),
        values,
        labels,
        color: AMBER_SLOT,
        xLabel: info.noun,
        yLabel: 'records',
        formatY: formatCount,
        markers: peakMarker ? [peakMarker] : undefined,
        description: `Bars of the number of selected records in each ${info.noun}.`
      };
    }
    return {
      kind: 'bars',
      horizontal: true,
      title: getTitle(kind, normalise),
      values,
      labels,
      color: AMBER_SLOT,
      formatX: formatCount,
      description: `Bars of the number of selected records in each ${info.noun}.`
    };
  }

  const selectionPercent = Array.from(shares, share => share * 100);
  const cityPercent = Array.from(citywide, share => share * 100);

  if (normalise === 'share') {
    if (info.cyclic) {
      const x = Array.from({length: info.binCount}, (_, index) => getBinX(kind, index));
      const xDomain: [number, number] =
        kind === 'hour' ? [0, 24 / HOURS_PER_QUARTER_DAY] : [0, info.binCount];
      const series = [
        {label: selectionName, x, y: selectionPercent, area: true, color: AMBER_SLOT, width: 2},
        {label: 'Citywide', x, y: cityPercent, ghost: true, width: 1.25},
        ...(pinned
          ? [
              {
                label: pinned.name,
                x,
                y: Array.from(pinned.shares, share => share * 100),
                ghost: true,
                dashed: true,
                width: 1.5
              }
            ]
          : [])
      ];
      return {
        kind: 'line',
        title: getTitle(kind, normalise),
        series,
        xDomain,
        yLabel: 'share of the selection',
        xLabel: info.noun,
        formatX: formatBinX(kind, categoryNames),
        formatY: percentTicks,
        markers: peak
          ? [
              {
                x: getBinX(kind, peak.peakIndex),
                label: `peak ${peak.peakName}, ${formatPercent(peak.peakShare, 0)}`
              }
            ]
          : undefined,
        description: `Share of the selected records in each ${info.noun} against the citywide share${pinned ? ` and ${pinned.name}` : ''}.`
      };
    }
    const maxPercent = Math.max(1, ...selectionPercent, ...cityPercent);
    const domain: [number, number] = [0, Math.ceil(maxPercent / 5) * 5];
    return {
      kind: 'multiples',
      columns: 2,
      shareDomains: false,
      titles: [selectionName, 'Citywide'],
      charts: [
        {
          kind: 'bars',
          horizontal: true,
          values: selectionPercent,
          labels,
          color: AMBER_SLOT,
          xDomain: domain,
          formatX: percentTicks
        },
        {
          kind: 'bars',
          horizontal: true,
          values: cityPercent,
          labels,
          colors: labels.map(() => muted),
          xDomain: domain,
          formatX: percentTicks
        }
      ],
      description: `Share of records in each ${info.noun}, the selection beside the whole city on one scale.`
    };
  }

  // Difference from the citywide shape, in percentage points: orange more, purple fewer.
  const difference = selectionPercent.map((percent, index) => percent - cityPercent[index]);
  const colors = difference.map(value => (value >= 0 ? more : fewer));
  if (info.cyclic) {
    return {
      kind: 'bars',
      title: getTitle(kind, normalise),
      values: difference,
      labels,
      colors,
      xLabel: info.noun,
      yLabel: 'percentage points vs citywide',
      formatY: signedPoints,
      guides: [{y: 0, label: 'same as citywide'}],
      description: `Bars of the selection's share minus the citywide share in each ${info.noun}; above zero is more than the city.`
    };
  }
  return {
    kind: 'bars',
    horizontal: true,
    title: getTitle(kind, normalise),
    values: difference,
    labels,
    colors,
    formatX: signedPoints,
    markers: [{x: 0, label: 'citywide'}],
    description: `Bars of the selection's share minus the citywide share in each ${info.noun}; right of zero is more than the city.`
  };
}

/**
 * The clock view of a cyclic quantity: a rose of the selection's shares with the citywide shape
 * as a dashed baseline ring, so 23:00 joins 00:00 and December joins January. `null` for the
 * nominal kinds or an empty selection.
 */
export function buildClockChart(input: SelectionChartInput): ChartData | null {
  const {kind, counts, citywide, categoryNames, theme} = input;
  const info = VALUE_KINDS[kind];
  if (!info.cyclic || input.domainMode === 'selection') return null;
  const shares = toShares(counts);
  if (!shares.some(share => share > 0)) return null;
  const names = getBinNames(kind, categoryNames);
  const labels =
    kind === 'hour'
      ? names.map((_, index) => (index % HOURS_PER_QUARTER_DAY === 0 ? String(index) : ''))
      : names;
  return {
    kind: 'rose',
    title: `${info.label} as a clock`,
    values: Array.from(shares, share => share * 100),
    baseline: Array.from(citywide, share => share * 100),
    labels,
    colors: names.map(() => CHART_AMBER[theme]),
    description: `Share of the selected records in each ${info.noun}, drawn round so the end joins the start, against the citywide shape.`
  };
}
