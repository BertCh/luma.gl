// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * The charts of the point-patterns story, built from the GPU readbacks: L(r) - r against a
 * Monte Carlo envelope, the G / F / J multiples, the Clark-Evans gauge and the quadrat histogram.
 * Pure functions that return `ChartData`; the scene decides when to call `ctx.setChart`.
 *
 * Series colours are chart palette slots: 4 is the amber of the observations (`--chart-5`), 0 the
 * sky blue of a second population (`--chart-1`).
 */

import {formatSigned} from '../../cartography/live-text';
import type {ClassTable} from '../../cartography/types';
import type {
  BarChartData,
  ChartSeries,
  DiagramChartData,
  LineChartData,
  MultiplesChartData
} from '../chart-types';
import {ENVELOPE_RUNS, type CurveEnvelope} from './point-patterns-envelope';

/** Palette slot of the observed amber series. */
export const OBSERVED_SLOT = 4;
/** Palette slot of a second population (places). */
export const SECOND_SLOT = 0;

/** The curve of the selection that was replaced, drawn dashed and grey for comparison. */
export type PreviousCurve = {label: string; radii: number[]; lMinusR: number[]};

/** Formats a radius for an axis: metres below a kilometre, kilometres above. */
export function formatRadius(value: number): string {
  if (!Number.isFinite(value)) return '';
  if (value < 1000) return `${Math.round(value)} m`;
  const kilometers = value / 1000;
  return `${Number.isInteger(kilometers) ? kilometers : kilometers.toFixed(1)} km`;
}

/** Inputs of {@link buildLCurveChart}. */
export type LCurveInput = {
  /** Name of the selection ("Birds"). */
  label: string;
  radii: ArrayLike<number>;
  lMinusR: ArrayLike<number>;
  /** Palette slot of the observed series. */
  slot: number;
  previous: PreviousCurve | null;
  /** The finished envelope, or `null` while simulations are missing (no partial band is drawn). */
  envelope: CurveEnvelope | null;
  /** Simulations received so far (shown as progress while the band is missing). */
  simulationsDone: number;
  /** Name of the null model ("random on city land"). */
  nullName: string;
  /** Observed curves of the three edge corrections, when the reader asked to compare them. */
  corrections: {
    none: ArrayLike<number>;
    border: ArrayLike<number>;
    isotropic: ArrayLike<number>;
  } | null;
};

/**
 * `L(r) - r` against the envelope of random patterns: the observed curve in the selection colour,
 * the grey band of {@link ENVELOPE_RUNS} random patterns, the dashed zero line ("random, CSR"), the
 * previous selection as a grey dashed ghost, and a marker at the `radius` option that the reader can
 * drag. With `corrections`, three curves (none, border, isotropic) replace the single observed one.
 */
export function buildLCurveChart(input: LCurveInput): LineChartData {
  const series: ChartSeries[] = [];
  if (input.previous && input.previous.label !== input.label) {
    series.push({
      label: input.previous.label,
      x: input.previous.radii,
      y: input.previous.lMinusR,
      ghost: true,
      dashed: true,
      directLabel: true
    });
  }
  if (input.corrections) {
    series.push(
      {
        label: 'No correction',
        x: input.radii,
        y: input.corrections.none,
        color: input.slot,
        dashed: true,
        width: 1,
        directLabel: true
      },
      {
        label: 'Border',
        x: input.radii,
        y: input.corrections.border,
        color: input.slot,
        dashed: true,
        width: 1.75,
        directLabel: true
      },
      {
        label: 'Isotropic',
        x: input.radii,
        y: input.corrections.isotropic,
        color: input.slot,
        width: 2.5,
        directLabel: true
      }
    );
  } else {
    series.push({
      label: input.label,
      x: input.radii,
      y: input.lMinusR,
      color: input.slot,
      width: 2.5,
      directLabel: true
    });
  }
  const complete = input.envelope !== null;
  return {
    kind: 'line',
    height: 156,
    title: complete
      ? undefined
      : `${input.simulationsDone} of ${ENVELOPE_RUNS} random patterns run`,
    xLabel: 'radius r',
    yLabel: 'L(r) − r',
    formatX: formatRadius,
    formatY: value => `${formatSigned(value, 0)} m`,
    series,
    band: input.envelope
      ? {
          x: input.radii,
          low: input.envelope.low,
          high: input.envelope.high,
          label: `${ENVELOPE_RUNS} random patterns (${input.nullName})`
        }
      : undefined,
    guides: [{y: 0, label: 'random (CSR)'}],
    link: {option: 'radius', label: value => `r = ${formatRadius(value)}`},
    description: `Besag L(r) minus r for ${input.label}. Zero is a random pattern; the grey band is the pointwise range of ${ENVELOPE_RUNS} random patterns (${input.nullName}), a 5 % envelope. Above the band the pattern is clustered at that radius.`
  };
}

/** Inputs of {@link buildDistanceChart}. */
export type DistanceChartInput = {
  radii: ArrayLike<number>;
  g: ArrayLike<number>;
  f: ArrayLike<number>;
  j: ArrayLike<number>;
  /** Intensity of the points, per square metre (n over the window area), for the CSR theory. */
  intensity: number;
  slot: number;
  /** Envelopes of the same simulations, or `null` while they are missing. */
  envelopes: {g: CurveEnvelope; f: CurveEnvelope; j: CurveEnvelope} | null;
  nullName: string;
};

const clampJ = (value: number) =>
  Number.isFinite(value) ? Math.min(Math.max(value, 0), 2) : Number.NaN;

/**
 * G, F and J as three small multiples with the CSR theory dashed (G = F = 1 − exp(−λπr²), J = 1) and
 * the envelope of the same random patterns as {@link buildLCurveChart}. J is drawn between 0 and 2.
 */
export function buildDistanceChart(input: DistanceChartInput): MultiplesChartData {
  const radii = Array.from(input.radii);
  const theory = radii.map(radius => 1 - Math.exp(-input.intensity * Math.PI * radius * radius));
  const band = (envelope: CurveEnvelope | undefined, transform?: (value: number) => number) =>
    envelope
      ? {
          x: radii,
          low: transform ? Array.from(envelope.low, transform) : envelope.low,
          high: transform ? Array.from(envelope.high, transform) : envelope.high,
          label: `${ENVELOPE_RUNS} random patterns`
        }
      : undefined;
  const base = {kind: 'line' as const, height: 120, formatX: formatRadius, table: false};
  const g: LineChartData = {
    ...base,
    yDomain: [0, 1],
    series: [
      {label: 'Observed', x: radii, y: Array.from(input.g), color: input.slot, width: 2.25},
      {label: 'CSR', x: radii, y: theory, dashed: true, ghost: true}
    ],
    band: band(input.envelopes?.g)
  };
  const f: LineChartData = {
    ...base,
    yDomain: [0, 1],
    series: [
      {label: 'Observed', x: radii, y: Array.from(input.f), color: input.slot, width: 2.25},
      {label: 'CSR', x: radii, y: theory, dashed: true, ghost: true}
    ],
    band: band(input.envelopes?.f)
  };
  const j: LineChartData = {
    ...base,
    yDomain: [0, 2],
    series: [
      {label: 'Observed', x: radii, y: Array.from(input.j, clampJ), color: input.slot, width: 2.25},
      {label: 'CSR', x: radii, y: radii.map(() => 1), dashed: true, ghost: true}
    ],
    band: band(input.envelopes?.j, clampJ)
  };
  return {
    kind: 'multiples',
    charts: [g, f, j],
    titles: ['G nearest neighbour', 'F empty space', 'J = (1 − G) / (1 − F)'],
    columns: 3,
    shareDomains: false,
    description: `G, F and J of the selection against complete spatial randomness (${input.nullName}). G rising early and F rising late mean clustering; J below 1 agrees.`
  };
}

/** XML-safe text for diagram markup. */
const escapeText = (text: string) =>
  text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

/**
 * The Clark-Evans gauge: a 0 to 2 axis (R = observed over expected nearest-neighbour distance), the
 * shaded "random" band where |z| < 1.96, the words clustered and dispersed on either side, and the
 * observed R with its z under the marker.
 *
 * @param ratio Clark-Evans R.
 * @param zScore Its z-score.
 * @param halfWidth Half-width of the random band in R units (1.96 standard errors over E).
 */
export function buildClarkEvansGauge(
  ratio: number,
  zScore: number,
  halfWidth: number
): DiagramChartData {
  const left = 16;
  const right = 304;
  const toX = (value: number) => left + (Math.min(Math.max(value, 0), 2) / 2) * (right - left);
  const markerX = toX(ratio);
  const bandHalf = Math.max(2, toX(1 + (Number.isFinite(halfWidth) ? halfWidth : 0)) - toX(1));
  const labelX = Math.min(Math.max(markerX, 56), 264);
  const ticks = [0, 0.5, 1, 1.5, 2]
    .map(
      value =>
        `<line class="diagram-muted" x1="${toX(value)}" y1="34" x2="${toX(value)}" y2="46"/><text class="diagram-muted" x="${toX(value)}" y="62" text-anchor="middle">${value}</text>`
    )
    .join('');
  const label = Number.isFinite(ratio)
    ? `R = ${ratio.toFixed(2)}, z = ${formatSigned(zScore, 0)}`
    : 'R undefined';
  const svg = `<rect class="diagram-fill" x="${toX(1) - bandHalf}" y="30" width="${bandHalf * 2}" height="20"/>
<line class="diagram-muted" x1="${left}" y1="40" x2="${right}" y2="40"/>${ticks}
${Number.isFinite(ratio) ? `<line class="diagram-accent" x1="${markerX}" y1="26" x2="${markerX}" y2="54"/><circle class="diagram-accent" cx="${markerX}" cy="40" r="4.5"/>` : ''}
<text class="diagram-accent" x="${labelX}" y="16" text-anchor="middle">${escapeText(label)}</text>
<text class="diagram-muted" x="${toX(0.5)}" y="82" text-anchor="middle">clustered</text>
<text class="diagram-muted" x="${toX(1)}" y="82" text-anchor="middle">random</text>
<text class="diagram-muted" x="${toX(1.5)}" y="82" text-anchor="middle">dispersed</text>`;
  return {
    kind: 'diagram',
    width: 320,
    height: 92,
    svg,
    description: `Clark-Evans gauge: R is ${Number.isFinite(ratio) ? ratio.toFixed(2) : 'undefined'}. Below 1 is clustered, 1 is random, above 1 is dispersed; the shaded band is where a random pattern would fall.`
  };
}

/**
 * Histogram of the quadrat counts with the mean (the expectation of a random pattern) marked. Bars
 * are coloured by the class table of the map; the empty class is grey so the lake does not hide the
 * rest.
 *
 * @param counts Records per quadrat.
 * @param mean Mean count per quadrat.
 * @param varianceToMeanRatio VMR, quoted in the description.
 * @param table The quadrat class table drawn on the map.
 */
export function buildQuadratChart(
  counts: ArrayLike<number>,
  mean: number,
  varianceToMeanRatio: number,
  table: ClassTable
): BarChartData {
  let maximum = 1;
  for (let index = 0; index < counts.length; index++) maximum = Math.max(maximum, counts[index]);
  const integerBins = maximum <= 24;
  const binCount = integerBins ? maximum + 1 : 16;
  const high = integerBins ? maximum + 1 : maximum;
  const values = new Array<number>(binCount).fill(0);
  for (let index = 0; index < counts.length; index++) {
    const bin = Math.min(binCount - 1, Math.floor((counts[index] / high) * binCount));
    values[bin]++;
  }
  return {
    kind: 'histogram',
    height: 120,
    values,
    xDomain: [0, high],
    breaks: table.breaks,
    classColors: table.colors.map((color, index) =>
      index === 0 ? ([128, 136, 148, 150] as const) : color
    ),
    now: mean,
    nowLabel: 'random expects',
    xLabel: 'records per quadrat',
    yLabel: 'quadrats',
    description: `Quadrat counts. A random pattern has a variance equal to its mean (VMR 1); here the variance-to-mean ratio is ${Number.isFinite(varianceToMeanRatio) ? varianceToMeanRatio.toFixed(1) : 'undefined'}.`
  };
}
