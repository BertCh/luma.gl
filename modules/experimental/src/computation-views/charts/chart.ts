// Copied from examples/deck/spatial-analysis-showcase/src/shell/chart.ts; the showcase still owns its copy until it switches to this module.
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

import type {ChartData} from './chart-types';
import {ensureChartStyles} from './chart-styles';
import {buildBars} from './bars';
import {
  type BuildContext,
  type ChartBuild,
  type ChartRenderOptions,
  CHART_WIDTH,
  formatChartNumber,
  getNiceTicks,
  updateChartLink,
  wrapChart
} from './core';
import {h} from './dom';
import {buildLine} from './line';
import {buildMatrix} from './matrix';
import {buildMultiples} from './multiples';
import {buildScatter} from './scatter';
import {buildDumbbell, buildForest, buildLorenz, buildSlope} from './shapes';
import {buildDiagram, buildRose, buildSparkline, buildStacked} from './simple';
import {buildTimeline} from './timeline';

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
  ensureChartStyles(document);
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
