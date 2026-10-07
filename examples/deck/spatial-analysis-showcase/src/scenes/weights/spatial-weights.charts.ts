// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  BarChartData,
  ChartColor,
  DiagramChartData,
  LineChartData,
  ScatterChartData
} from '../chart-types';
import type {ClassTable} from '../../cartography/types';
import {CARDINALITY_CHART_LIMIT, getInkColor, type Ground} from './spatial-weights.style';

/** Kernel profiles of `GPUNeighborSearch` and `GPUSpatialWeightsTransform`, as functions of z = d / h. */
export type KernelName = 'gaussian' | 'triangular' | 'epanechnikov' | 'bisquare' | 'uniform';

/** The kernel value K(z) as the contributors define it (see the reference page). */
export function getKernelValue(kernel: KernelName, z: number): number {
  switch (kernel) {
    case 'gaussian':
      return Math.exp((-z * z) / 2) / Math.sqrt(2 * Math.PI);
    case 'triangular':
      return Math.max(1 - z, 0);
    case 'epanechnikov':
      return 0.75 * Math.max(1 - z * z, 0);
    case 'bisquare':
      return (15 / 16) * Math.max(1 - z * z, 0) ** 2;
    default:
      return z <= 1 ? 0.5 : 0;
  }
}

/**
 * Histogram of the neighbour counts 0..14 in the colours of the map's class table (islands in ink,
 * since their class is unfilled on the map), with the rule-of-thumb marker at eight.
 */
export function getCardinalityChart(
  cardinality: ArrayLike<number>,
  table: ClassTable,
  ground: Ground,
  unitPlural: string
): BarChartData {
  const counts = new Array<number>(CARDINALITY_CHART_LIMIT + 1).fill(0);
  for (let row = 0; row < cardinality.length; row++) {
    const value = cardinality[row];
    if (Number.isFinite(value)) counts[Math.min(CARDINALITY_CHART_LIMIT, Math.round(value))]++;
  }
  const colors: ChartColor[] = table.colors.map((color, index) =>
    index === 0 ? getInkColor(ground) : color
  );
  return {
    kind: 'histogram',
    values: counts,
    xDomain: [-0.5, CARDINALITY_CHART_LIMIT + 0.5],
    breaks: table.breaks,
    classColors: colors,
    markers: [{x: 8, label: 'rule of thumb: 8'}],
    xLabel: 'neighbours per place',
    yLabel: unitPlural,
    description: `How many ${unitPlural} list each number of neighbours under the current rule, coloured like the map.`,
    height: 120
  };
}

/**
 * The kernel curve K(d / h), scaled to the weights of the focus row, with the focus row's
 * neighbours as points on it. Distances are ground kilometres at the focus.
 */
export function getKernelChart(props: {
  kernel: KernelName;
  /** Bandwidth h in planar metres. */
  bandwidth: number;
  /** Ground metres per planar metre at the focus. */
  groundFactor: number;
  /** Planar distance of each neighbour of the focus row, in metres. */
  distances: readonly number[];
  /** Weight of each neighbour, aligned with `distances`. */
  weights: readonly number[];
  focusName: string;
}): LineChartData {
  const {kernel, bandwidth, groundFactor, distances, weights} = props;
  const toKilometres = (meters: number) => (meters * groundFactor) / 1000;
  const order = distances.map((_, index) => index).sort((a, b) => distances[a] - distances[b]);
  // Scale the curve so that it passes through the heaviest neighbour (a row-standardised weight
  // is K(d / h) divided by the row sum, so the curve is K up to that constant).
  let scale = 1;
  let heaviest = -1;
  for (const index of order) {
    if (heaviest < 0 || weights[index] > weights[heaviest]) heaviest = index;
  }
  if (heaviest >= 0) {
    const profile = getKernelValue(kernel, distances[heaviest] / bandwidth);
    if (profile > 0 && weights[heaviest] > 0) scale = weights[heaviest] / profile;
  }
  const curveX: number[] = [];
  const curveY: number[] = [];
  for (let step = 0; step <= 50; step++) {
    const z = (step / 50) * 1.05;
    curveX.push(toKilometres(z * bandwidth));
    curveY.push(scale * getKernelValue(kernel, z));
  }
  return {
    kind: 'line',
    series: [
      {label: 'Kernel K(d / h)', x: curveX, y: curveY, color: 0},
      {
        label: 'Neighbours of the focus',
        x: order.map(index => toKilometres(distances[index])),
        y: order.map(index => weights[index]),
        color: 1,
        points: true,
        width: 0.01
      }
    ],
    xLabel: `distance from ${props.focusName} (km on the ground)`,
    yLabel: 'weight',
    xDomain: [0, curveX[curveX.length - 1]],
    description: 'The kernel curve and the weight of each neighbour of the focus place.',
    height: 130
  };
}

/** Ordinary least squares of `y` on `x` over the finite pairs: slope, intercept, means and count. */
export function getRegressionLine(
  x: ArrayLike<number>,
  y: ArrayLike<number>
): {slope: number; intercept: number; meanX: number; meanY: number; count: number} {
  let count = 0;
  let sumX = 0;
  let sumY = 0;
  for (let index = 0; index < x.length; index++) {
    if (!Number.isFinite(x[index]) || !Number.isFinite(y[index])) continue;
    count++;
    sumX += x[index];
    sumY += y[index];
  }
  const meanX = count ? sumX / count : Number.NaN;
  const meanY = count ? sumY / count : Number.NaN;
  let covariance = 0;
  let variance = 0;
  for (let index = 0; index < x.length; index++) {
    if (!Number.isFinite(x[index]) || !Number.isFinite(y[index])) continue;
    covariance += (x[index] - meanX) * (y[index] - meanY);
    variance += (x[index] - meanX) ** 2;
  }
  const slope = variance > 0 ? covariance / variance : Number.NaN;
  return {slope, intercept: meanY - slope * meanX, meanX, meanY, count};
}

/**
 * The lag against the value of each place: the Moran scatterplot's cousin, with the 1:1 line and
 * the fitted slope. Places without a lag (islands, no data) are left out.
 */
export function getLagScatter(props: {
  values: ArrayLike<number>;
  lag: ArrayLike<number>;
  unit: string;
}): ScatterChartData & {slope: number; count: number} {
  const x: number[] = [];
  const y: number[] = [];
  for (let row = 0; row < props.values.length; row++) {
    if (Number.isFinite(props.values[row]) && Number.isFinite(props.lag[row])) {
      x.push(props.values[row]);
      y.push(props.lag[row]);
    }
  }
  const line = getRegressionLine(x, y);
  return {
    kind: 'scatter',
    x,
    y,
    diagonal: true,
    quadrants: {x: line.meanX, y: line.meanY},
    fit: {
      slope: line.slope,
      intercept: line.intercept,
      label: `slope ${line.slope.toFixed(2)}`
    },
    xLabel: `Value (${props.unit})`,
    yLabel: `Neighbourhood lag (${props.unit})`,
    radius: 1.6,
    height: 150,
    description:
      'Each point is a place: its own value against the weighted mean of its neighbours.',
    slope: line.slope,
    count: line.count
  };
}

/** Three polygons that meet at one vertex: one key, three places, six ordered pairs. */
export function getVertexDiagram(): DiagramChartData {
  return {
    kind: 'diagram',
    width: 320,
    height: 120,
    description:
      'Three polygons meeting at one shared vertex: that vertex key lists three polygons, which gives six ordered neighbour pairs.',
    svg: `
<polygon class="diagram-fill" points="20,20 100,12 110,64 70,104 20,84"/>
<polygon class="diagram-fill" points="100,12 170,22 176,64 110,64"/>
<polygon class="diagram-fill" points="70,104 110,64 176,64 168,108"/>
<polygon class="diagram-muted" points="20,20 100,12 110,64 70,104 20,84"/>
<polygon class="diagram-muted" points="100,12 170,22 176,64 110,64"/>
<polygon class="diagram-muted" points="70,104 110,64 176,64 168,108"/>
<circle class="diagram-signal" cx="110" cy="64" r="4.5"/>
<text class="diagram-ink" x="196" y="38" font-size="12">1 vertex key</text>
<text class="diagram-ink" x="196" y="60" font-size="12">3 polygons share it</text>
<text class="diagram-ink" x="196" y="82" font-size="12">3 x 2 = 6 ordered pairs</text>
<text class="diagram-muted" x="196" y="102" font-size="11">sort keys, find runs, emit pairs</text>`
  };
}
