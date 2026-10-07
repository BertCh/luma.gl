// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  DiagramChartData,
  RoseChartData,
  SparklineData,
  StackedBarChartData
} from '../../scenes/chart-types';
import {
  type BuildContext,
  type ChartBuild,
  assignRows,
  colorToCss,
  createRoot,
  createScale,
  estimateTextWidth,
  extent,
  formatChartNumber,
  formatPercent,
  slot,
  svg,
  truncateText
} from './core';

/** Builds a tiny axis-free trend line (with an area wash and an optional current-value dot). */
export function buildSparkline(data: SparklineData, context: BuildContext): ChartBuild {
  const height = data.height ?? 36;
  const width = context.width;
  const [min, max] = extent([data.values]);
  const x = createScale([0, Math.max(1, data.values.length - 1)], [4, width - 4]);
  const y = createScale([min, max], [height - 5, 5]);
  const color = slot(data.color ?? 0);
  let path = '';
  let pen = false;
  let firstX = 0;
  let lastX = 0;
  for (let index = 0; index < data.values.length; index++) {
    const value = data.values[index];
    if (!Number.isFinite(value)) {
      pen = false;
      continue;
    }
    if (!pen) firstX = x(index);
    path += `${pen ? 'L' : 'M'}${x(index).toFixed(1)} ${y(value).toFixed(1)}`;
    lastX = x(index);
    pen = true;
  }
  const root = createRoot(
    'sparkline',
    width,
    height,
    undefined,
    undefined,
    data.description ?? 'Trend'
  );
  root.append(
    svg('path', {
      class: 'chart-area',
      d: path
        ? `${path}L${lastX.toFixed(1)} ${height - 1}L${firstX.toFixed(1)} ${height - 1}Z`
        : '',
      style: `fill:${color}`
    }),
    svg('path', {class: 'chart-line', d: path, style: `stroke:${color};stroke-width:1.5`})
  );
  const highlight = data.highlight;
  if (highlight !== undefined && Number.isFinite(data.values[highlight])) {
    root.append(
      svg('circle', {
        class: 'chart-spark-dot',
        cx: x(highlight),
        cy: y(data.values[highlight]),
        r: 3.2,
        style: `fill:${color}`
      })
    );
  }
  return {svg: root, table: null};
}

const polar = (cx: number, cy: number, radius: number, angle: number): [number, number] => [
  cx + radius * Math.sin(angle),
  cy - radius * Math.cos(angle)
];

/** Builds a rose (polar bar) chart: sectors from the top clockwise, area proportional to value. */
export function buildRose(data: RoseChartData, context: BuildContext): ChartBuild {
  const width = context.width;
  const height = data.height ?? (context.compact ? 150 : 210);
  const count = data.values.length;
  const title = context.title ?? data.title;
  const top = 6 + (title ? 16 : 0);
  const labelPad = context.compact ? 4 : 20;
  const radius = Math.max(20, Math.min(width / 2 - labelPad - 6, (height - top) / 2 - labelPad));
  const cx = width / 2;
  const cy = top + (height - top) / 2;
  const [, max] = extent([data.values, ...(data.baseline ? [data.baseline] : [])], [0]);
  const scale = (value: number) => (max > 0 && value > 0 ? radius * Math.sqrt(value / max) : 0);
  const step = (Math.PI * 2) / Math.max(1, count);
  const half = step / 2;

  const root = createRoot('rose', width, height, title, data.description, 'Rose chart');
  if (title) root.append(svg('text', {class: 'chart-title', x: 2, y: 13}, title));
  for (const fraction of [0.5, 1]) {
    root.append(
      svg('circle', {class: 'chart-grid', cx, cy, r: radius * Math.sqrt(fraction), fill: 'none'})
    );
  }
  const sectors = svg('g', {class: 'chart-sectors'});
  root.append(sectors);
  for (let index = 0; index < count; index++) {
    const value = data.values[index];
    if (!Number.isFinite(value)) continue;
    const a0 = index * step - half * 0.94;
    const a1 = index * step + half * 0.94;
    const r = scale(value);
    const [x0, y0] = polar(cx, cy, r, a0);
    const [x1, y1] = polar(cx, cy, r, a1);
    const color = data.colors?.[index] ? colorToCss(data.colors[index]) : slot(0);
    const dim = data.highlight !== undefined && data.highlight !== index;
    sectors.append(
      svg(
        'path',
        {
          class: `chart-sector${dim ? ' is-dim' : ''}${data.highlight === index ? ' is-highlight' : ''}`,
          d: `M${cx} ${cy}L${x0.toFixed(2)} ${y0.toFixed(2)}A${r.toFixed(2)} ${r.toFixed(2)} 0 ${a1 - a0 > Math.PI ? 1 : 0} 1 ${x1.toFixed(2)} ${y1.toFixed(2)}Z`,
          style: `fill:${color}`
        },
        svg('title', {}, `${data.labels?.[index] ?? index + 1}: ${formatChartNumber(value)}`)
      )
    );
  }
  if (data.baseline) {
    let path = '';
    for (let index = 0; index < count; index++) {
      const value = data.baseline[index];
      if (!Number.isFinite(value)) continue;
      const r = scale(value);
      const [x0, y0] = polar(cx, cy, r, index * step - half * 0.94);
      const [x1, y1] = polar(cx, cy, r, index * step + half * 0.94);
      path += `M${x0.toFixed(2)} ${y0.toFixed(2)}A${r.toFixed(2)} ${r.toFixed(2)} 0 0 1 ${x1.toFixed(2)} ${y1.toFixed(2)}`;
    }
    root.append(svg('path', {class: 'chart-baseline', d: path}));
  }
  if (data.labels && !context.compact) {
    const stride = Math.ceil(count / 12);
    for (let index = 0; index < count; index += stride) {
      const angle = index * step;
      const [lx, ly] = polar(cx, cy, radius + 8, angle);
      const side = Math.sin(angle);
      root.append(
        svg(
          'text',
          {
            class: `chart-tick${data.highlight === index ? ' is-strong' : ''}`,
            x: lx.toFixed(1),
            y: (ly + 3.5 + Math.cos(angle) * -2).toFixed(1),
            'text-anchor': side > 0.3 ? 'start' : side < -0.3 ? 'end' : 'middle'
          },
          data.labels[index] ?? ''
        )
      );
    }
  }
  if (!context.compact) {
    root.append(
      svg(
        'text',
        {class: 'chart-note', x: 2, y: height - 4},
        `Outer ring ${formatChartNumber(max)}`
      )
    );
    if (data.baseline) {
      root.append(
        svg(
          'text',
          {class: 'chart-note', x: width - 2, y: height - 4, 'text-anchor': 'end'},
          'Dashed: baseline'
        )
      );
    }
  }
  const rows = Array.from({length: count}, (_, index) => [
    data.labels?.[index] ?? index + 1,
    formatChartNumber(data.values[index]),
    ...(data.baseline ? [formatChartNumber(data.baseline[index])] : [])
  ]);
  return {
    svg: root,
    table: {headers: ['Sector', 'Value', ...(data.baseline ? ['Baseline'] : [])], rows}
  };
}

/** Builds one stacked share bar with direct labels above it (no legend). */
export function buildStacked(data: StackedBarChartData, context: BuildContext): ChartBuild {
  const width = context.width;
  const title = context.title ?? data.title;
  const segments = data.segments.filter(segment => segment.value > 0);
  const total = segments.reduce((sum, segment) => sum + segment.value, 0) || 1;
  const percent = (data.format ?? 'percent') === 'percent';
  const left = 6;
  const span = width - 12;
  const barHeight = 20;

  const items = segments.map((segment, index) => {
    const start =
      left + (segments.slice(0, index).reduce((sum, s) => sum + s.value, 0) / total) * span;
    const end = start + (segment.value / total) * span;
    const text = `${segment.label} ${percent ? formatPercent(segment.value / total) : formatChartNumber(segment.value)}`;
    const labelWidth = estimateTextWidth(text, 11) + 2;
    const center = (start + end) / 2;
    const anchor: 'start' | 'middle' | 'end' =
      center - labelWidth / 2 < 2
        ? 'start'
        : center + labelWidth / 2 > width - 2
          ? 'end'
          : 'middle';
    const textX =
      anchor === 'start'
        ? Math.max(2, start)
        : anchor === 'end'
          ? Math.min(width - 2, end)
          : center;
    const labelStart =
      anchor === 'start' ? textX : anchor === 'end' ? textX - labelWidth : textX - labelWidth / 2;
    return {
      segment,
      index,
      start,
      end,
      center,
      text,
      textX,
      anchor,
      labelStart,
      labelEnd: labelStart + labelWidth
    };
  });
  const placement = assignRows(
    items.map(item => ({start: item.labelStart, end: item.labelEnd})),
    5
  );
  const rowCount = items.length ? placement.rowCount : 1;
  const top = 6 + (title ? 16 : 0);
  const barTop = top + rowCount * 13 + 4;
  const height = Math.max(data.height ?? 0, barTop + barHeight + 8);

  const root = createRoot('stacked', width, height, title, data.description, 'Composition bar');
  if (title) root.append(svg('text', {class: 'chart-title', x: 2, y: 13}, title));
  for (const item of items) {
    const color = item.segment.color ? colorToCss(item.segment.color) : slot(item.index);
    root.append(
      svg(
        'rect',
        {
          class: 'chart-stack-segment',
          x: item.start.toFixed(2),
          y: barTop,
          width: Math.max(0.5, item.end - item.start).toFixed(2),
          height: barHeight,
          style: `fill:${color}`
        },
        svg('title', {}, item.text)
      )
    );
    const row = placement.rows[item.index];
    const labelY = barTop - 5 - row * 13;
    if (row > 0) {
      root.append(
        svg('line', {
          class: 'chart-stack-leader',
          x1: item.center,
          x2: item.center,
          y1: labelY + 3,
          y2: barTop,
          style: `stroke:${color}`
        })
      );
    }
    root.append(
      svg(
        'text',
        {
          class: 'chart-stack-label',
          x: item.textX.toFixed(1),
          y: labelY,
          'text-anchor': item.anchor
        },
        truncateText(item.text, width - 4, 11)
      )
    );
  }
  return {
    svg: root,
    table: {
      headers: ['Part', 'Value', 'Share'],
      rows: segments.map(segment => [
        segment.label,
        formatChartNumber(segment.value),
        formatPercent(segment.value / total)
      ])
    }
  };
}

/** Removes script-capable content from a parsed diagram node tree. */
function sanitizeDiagram(node: Element): void {
  for (const element of Array.from(node.querySelectorAll('script, foreignObject')))
    element.remove();
  for (const element of [node, ...Array.from(node.querySelectorAll('*'))]) {
    for (const attribute of Array.from(element.attributes)) {
      if (attribute.name.toLowerCase().startsWith('on')) element.removeAttribute(attribute.name);
    }
  }
}

/** Builds an inline SVG diagram from scene-authored markup. */
export function buildDiagram(data: DiagramChartData, _context: BuildContext): ChartBuild {
  const root = createRoot(
    'diagram',
    data.width,
    data.height,
    undefined,
    data.description,
    'Diagram'
  );
  const parsed = new DOMParser().parseFromString(
    `<svg xmlns="http://www.w3.org/2000/svg">${data.svg}</svg>`,
    'image/svg+xml'
  );
  if (parsed.querySelector('parsererror')) {
    root.append(svg('text', {class: 'chart-note', x: 4, y: 14}, 'Diagram could not be drawn'));
  } else {
    sanitizeDiagram(parsed.documentElement);
    for (const child of Array.from(parsed.documentElement.childNodes)) {
      root.append(document.importNode(child, true));
    }
  }
  return {svg: root, table: null};
}
