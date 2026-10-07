// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * The shared chart kit. `renderChart(data)` turns a `ChartData` description (see
 * `scenes/chart-types.ts`) into a self-contained element: one SVG (`viewBox` 320 wide, scales to
 * its container) plus an optional legend and a "Show values" table. Colors come from the
 * `--chart-1`..`--chart-6`, `--text`, `--muted` and `--map-signal` tokens (see `chart.css`), so a
 * chart follows the page theme without being redrawn.
 */

import type {ChartData} from '../scenes/chart-types';
import './chart.css';
import {buildBars} from './charts/bars';
import {
  type BuildContext,
  type ChartBuild,
  type ChartRenderOptions,
  CHART_WIDTH,
  formatChartNumber,
  getNiceTicks,
  updateChartLink,
  wrapChart
} from './charts/core';
import {h} from './dom';
import {buildLine} from './charts/line';
import {buildMatrix} from './charts/matrix';
import {buildMultiples} from './charts/multiples';
import {buildScatter} from './charts/scatter';
import {buildDumbbell, buildForest, buildLorenz, buildSlope} from './charts/shapes';
import {buildDiagram, buildRose, buildSparkline, buildStacked} from './charts/simple';
import {buildTimeline} from './charts/timeline';

export {CHART_WIDTH, formatChartNumber, getNiceTicks, updateChartLink};
export type {ChartRenderOptions};

function buildChart(data: ChartData, context: BuildContext): ChartBuild {
  switch (data.kind) {
    case 'line':
      return buildLine(data, context);
    case 'bars':
    case 'histogram':
      return buildBars(data, context);
    case 'sparkline':
      return buildSparkline(data, context);
    case 'scatter':
      return buildScatter(data, context);
    case 'timeline':
      return buildTimeline(data, context);
    case 'matrix':
      return buildMatrix(data, context);
    case 'lorenz':
      return buildLorenz(data, context);
    case 'forest':
      return buildForest(data, context);
    case 'dumbbell':
      return buildDumbbell(data, context);
    case 'slope':
      return buildSlope(data, context);
    case 'rose':
      return buildRose(data, context);
    case 'stacked':
      return buildStacked(data, context);
    case 'multiples':
      return buildMultiples(data, context, buildChart);
    case 'diagram':
      return buildDiagram(data, context);
  }
}

/**
 * Renders a chart as an element. Pure: it keeps no state, so scenes may call `ctx.setChart` on
 * every readout update. With `options`, a chart that declares `link` draws the linked value as a
 * marker and reports pointer input through `onLinkInput`.
 */
export function renderChart(data: ChartData, options?: ChartRenderOptions): HTMLElement {
  try {
    const build = buildChart(data, {width: CHART_WIDTH, compact: false, options});
    return wrapChart(build, data.kind, 'table' in data ? data.table : undefined);
  } catch (error) {
    // A malformed chart must not break the readout update that asked for it.
    return h(
      'div',
      {
        class: 'chart-box chart-failed',
        title: error instanceof Error ? error.message : String(error)
      },
      'Chart unavailable'
    );
  }
}
