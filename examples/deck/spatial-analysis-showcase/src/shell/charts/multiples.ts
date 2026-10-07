// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {ChartData, MultiplesChartData} from '../../scenes/chart-types';
import {getBarDomains} from './bars';
import {type BuildContext, type ChartBuild, createRoot, svg} from './core';
import {getLineDomains} from './line';
import {getScatterDomains} from './scatter';
import {getTimelineDomains} from './timeline';

type Panel = Exclude<ChartData, MultiplesChartData>;
type Range = [number, number];

type PanelDomains = {x?: Range; y?: Range; y2?: Range};

function getPanelDomains(chart: Panel): PanelDomains | null {
  switch (chart.kind) {
    case 'line':
      return getLineDomains(chart);
    case 'scatter':
      return getScatterDomains(chart);
    case 'timeline':
      return getTimelineDomains(chart);
    case 'bars':
    case 'histogram':
      return chart.horizontal ? null : getBarDomains(chart);
    default:
      return null;
  }
}

const unite = (a: Range | undefined, b: Range | undefined): Range | undefined =>
  a && b ? [Math.min(a[0], b[0]), Math.max(a[1], b[1])] : (a ?? b);

/** Applies the shared domains to a panel (bars only share x when they are histograms). */
function applyDomains(chart: Panel, shared: PanelDomains): Panel {
  switch (chart.kind) {
    case 'line':
      return {
        ...chart,
        xDomain: chart.xDomain ?? shared.x,
        yDomain: chart.yDomain ?? shared.y,
        y2Domain: chart.y2Domain ?? shared.y2
      };
    case 'scatter':
    case 'timeline':
      return {...chart, xDomain: chart.xDomain ?? shared.x, yDomain: chart.yDomain ?? shared.y};
    case 'bars':
    case 'histogram': {
      const histogram = chart.kind === 'histogram' || chart.xDomain !== undefined;
      return {
        ...chart,
        xDomain: histogram ? (chart.xDomain ?? shared.x) : undefined,
        yDomain: chart.yDomain ?? shared.y
      };
    }
    default:
      return chart;
  }
}

/**
 * Builds small multiples: a grid of compact panels in one SVG, sharing x and y domains (unless
 * `shareDomains` is false), with the highlighted panel outlined.
 */
export function buildMultiples(
  data: MultiplesChartData,
  context: BuildContext,
  buildPanel: (chart: Panel, context: BuildContext) => ChartBuild
): ChartBuild {
  const count = data.charts.length;
  const columns = Math.max(1, data.columns ?? (count >= 5 ? 3 : 2));
  const gap = 10;
  const width = context.width;
  const panelWidth = (width - gap * (columns - 1)) / columns;

  // Shared domains per kind group.
  let charts = [...data.charts];
  if (data.shareDomains !== false) {
    const shared = new Map<string, PanelDomains>();
    for (const chart of charts) {
      const domains = getPanelDomains(chart);
      if (!domains) continue;
      const key = chart.kind === 'histogram' ? 'bars' : chart.kind;
      const existing = shared.get(key) ?? {};
      shared.set(key, {
        x: unite(existing.x, domains.x),
        y: unite(existing.y, domains.y),
        y2: unite(existing.y2, domains.y2)
      });
    }
    charts = charts.map(chart => {
      const key = chart.kind === 'histogram' ? 'bars' : chart.kind;
      const domains = shared.get(key);
      return domains ? applyDomains(chart, domains) : chart;
    });
  }

  const builds = charts.map((chart, index) =>
    buildPanel(chart, {
      width: panelWidth,
      compact: true,
      title: data.titles?.[index] ?? ('title' in chart ? chart.title : undefined)
    })
  );
  const heights = builds.map(build => build.svg.viewBox.baseVal.height);
  const rowCount = Math.ceil(count / columns);
  const rowHeights = Array.from({length: rowCount}, (_, row) =>
    Math.max(0, ...heights.slice(row * columns, (row + 1) * columns))
  );
  const rowTops: number[] = [];
  let cursor = 2;
  for (const rowHeight of rowHeights) {
    rowTops.push(cursor);
    cursor += rowHeight + gap;
  }
  const totalHeight = Math.max(20, cursor - gap + 2);
  const root = createRoot(
    'multiples',
    width,
    totalHeight,
    undefined,
    data.description,
    'Small multiples'
  );
  builds.forEach((build, index) => {
    const column = index % columns;
    const row = Math.floor(index / columns);
    const x = column * (panelWidth + gap);
    const y = rowTops[row];
    // Move the panel's drawing into a translated group; the panel's own root is dropped.
    const group = svg('g', {class: 'chart-panel', transform: `translate(${x.toFixed(2)} ${y})`});
    group.append(
      ...Array.from(build.svg.childNodes).filter(
        node => (node as Element).tagName !== 'title' && (node as Element).tagName !== 'desc'
      )
    );
    root.append(group);
    if (data.highlight === index) {
      root.append(
        svg('rect', {
          class: 'chart-panel-highlight',
          x: x - 2,
          y: y - 2,
          width: panelWidth + 4,
          height: heights[index] + 4,
          rx: 3
        })
      );
    }
  });
  return {svg: root, legend: builds[0]?.legend, table: null};
}
