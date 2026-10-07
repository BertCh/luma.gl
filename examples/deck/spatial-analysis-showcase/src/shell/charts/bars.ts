// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {BarChartData, ChartColor} from '../../scenes/chart-types';
import {
  type AxisTick,
  type BuildContext,
  type ChartBuild,
  attachLink,
  buildXYTable,
  colorToCss,
  createFrame,
  createHitArea,
  drawRules,
  estimateTextWidth,
  extent,
  formatChartNumber,
  niceDomain,
  slot,
  summariseRows,
  svg,
  truncateText
} from './core';

/** Whether the data draws as a histogram (numeric bins) rather than categories. */
const isHistogram = (data: BarChartData) =>
  data.kind === 'histogram' || (data.xDomain !== undefined && !data.horizontal);

/** The x and y domains a bar chart would use (shared by small multiples). */
export function getBarDomains(data: BarChartData): {x: [number, number]; y: [number, number]} {
  const count = data.values.length;
  const x: [number, number] =
    isHistogram(data) && data.xDomain ? [data.xDomain[0], data.xDomain[1]] : [0, count];
  if (data.yDomain) return {x, y: [data.yDomain[0], data.yDomain[1]]};
  if (data.yScale === 'log') return {x, y: extent([data.values], [], true)};
  const [min, max] = extent([data.values], [0]);
  return {x, y: niceDomain(Math.min(0, min), Math.max(0, max))};
}

function getBarColors(data: BarChartData, centers: readonly number[] | null) {
  const highlight = data.highlight ? new Set(data.highlight) : null;
  const base = slot(data.color ?? 0);
  return (index: number): string => {
    const explicit = data.colors?.[index];
    if (explicit) return colorToCss(explicit);
    if (data.breaks && data.classColors?.length && centers) {
      const center = centers[index];
      const classIndex = data.breaks.filter(breakValue => breakValue <= center).length;
      const color = data.classColors[Math.min(classIndex, data.classColors.length - 1)];
      return colorToCss(color as ChartColor);
    }
    if (highlight && highlight.size && !highlight.has(index)) return 'var(--chart-muted)';
    return base;
  };
}

/** Builds a vertical bar chart, a histogram, or (with `horizontal`) a ranked bar list. */
export function buildBars(data: BarChartData, context: BuildContext): ChartBuild {
  return data.horizontal && data.kind === 'bars'
    ? buildHorizontalBars(data, context)
    : buildVerticalBars(data, context);
}

function buildVerticalBars(data: BarChartData, context: BuildContext): ChartBuild {
  const height = data.height ?? (context.compact ? 110 : 140);
  const count = data.values.length;
  const histogram = isHistogram(data);
  const domains = getBarDomains(data);
  const binWidth = (domains.x[1] - domains.x[0]) / Math.max(1, count);
  const centers = Array.from(
    {length: count},
    (_, index) => domains.x[0] + (index + 0.5) * binWidth
  );

  let xTicks: AxisTick[] | undefined;
  const formatX = data.formatX ?? formatChartNumber;
  if (histogram && data.breaks?.length) {
    xTicks = data.breaks.map(value => ({value, label: formatX(value)}));
  } else if (!histogram) {
    xTicks = centers.map((center, index) => ({
      value: center,
      label: data.labels?.[index] ?? String(index + 1)
    }));
  }
  const interactive = !!data.onBarClick;
  const frame = createFrame({
    kind: histogram ? 'histogram' : 'bars',
    width: context.width,
    height,
    compact: context.compact,
    common: data,
    title: context.title,
    xDomain: domains.x,
    yDomain: domains.y,
    xScale: histogram ? data.xScale : 'linear',
    yScale: data.yScale,
    xTicks,
    interactive
  });
  const colorOf = getBarColors(data, histogram ? centers : null);
  const baselineValue = frame.y.isLog
    ? frame.y.domain[0]
    : Math.min(Math.max(0, domains.y[0]), domains.y[1]);
  const baseline = frame.y(baselineValue);
  const showValues =
    !context.compact &&
    !histogram &&
    count <= 24 &&
    (count <= 8 || (data.highlight?.length ?? 0) > 0);
  const highlight = new Set(data.highlight ?? []);

  for (let index = 0; index < count; index++) {
    const value = data.values[index];
    const left = frame.x(domains.x[0] + index * binWidth);
    const right = frame.x(domains.x[0] + (index + 1) * binWidth);
    const columnWidth = right - left;
    const gap = histogram ? Math.min(0.6, columnWidth * 0.1) : Math.min(2, columnWidth * 0.18);
    const label = histogram
      ? `${formatX(domains.x[0] + index * binWidth)} to ${formatX(domains.x[0] + (index + 1) * binWidth)}`
      : (data.labels?.[index] ?? String(index + 1));
    if (Number.isFinite(value)) {
      const top = frame.y(value);
      if (Number.isFinite(top)) {
        const clampedTop = Math.min(Math.max(top, frame.top), frame.bottom);
        frame.layers.main.append(
          svg(
            'rect',
            {
              class: 'chart-bar',
              x: (left + gap / 2).toFixed(2),
              y: Math.min(clampedTop, baseline).toFixed(2),
              width: Math.max(0.5, columnWidth - gap).toFixed(2),
              height: Math.max(0, Math.abs(baseline - clampedTop)).toFixed(2),
              style: `fill:${colorOf(index)}`
            },
            svg('title', {}, `${label}: ${frame.formatY(value)}`)
          )
        );
        if (showValues && (count <= 8 || highlight.has(index))) {
          const text = frame.formatY(value);
          if (estimateTextWidth(text) <= columnWidth + 8) {
            frame.layers.front.append(
              svg(
                'text',
                {
                  class: 'chart-note',
                  x: (left + right) / 2,
                  y: (value >= 0 ? clampedTop - 3 : clampedTop + 11).toFixed(1),
                  'text-anchor': 'middle'
                },
                text
              )
            );
          }
        }
      }
    }
    if (data.onBarClick) {
      const onClick = data.onBarClick;
      const hitRect = svg(
        'rect',
        {
          class: 'chart-hit is-input chart-bar-hit',
          x: left.toFixed(2),
          y: frame.top,
          width: columnWidth.toFixed(2),
          height: frame.bottom - frame.top,
          tabindex: 0,
          role: 'button',
          'aria-label': `${label}: ${Number.isFinite(value) ? frame.formatY(value) : 'no data'}`
        },
        svg('title', {}, label)
      );
      hitRect.addEventListener('click', () => onClick(index));
      hitRect.addEventListener('keydown', event => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault();
          onClick(index);
        }
      });
      frame.layers.overlay.append(hitRect);
    }
  }

  const markers = [
    ...(data.markers ?? []),
    ...(data.now !== undefined ? [{x: data.now, label: data.nowLabel ?? formatX(data.now)}] : [])
  ];
  drawRules(frame, {markers, guides: data.guides});

  const hit = createHitArea(frame, !!context.options?.onLinkInput);
  // The link hit area must sit under the bar hit rects so clicks on bars still select them.
  frame.layers.overlay.prepend(hit);
  const updateLink = attachLink(frame, data.link, context.options, hit);

  const {rows, step} = summariseRows(Array.from({length: count}, (_, index) => index));
  return {
    svg: frame.root,
    table: {
      headers: [data.xLabel ?? (histogram ? 'Bin' : 'Category'), data.yLabel ?? 'Value'],
      rows: rows.map(index => [
        histogram
          ? `${formatX(domains.x[0] + index * binWidth)} to ${formatX(domains.x[0] + (index + 1) * binWidth)}`
          : (data.labels?.[index] ?? index + 1),
        frame.formatY(data.values[index])
      ]),
      note: step > 1 ? `${count} bars, every ${step}th shown` : undefined
    },
    updateLink
  };
}

function buildHorizontalBars(data: BarChartData, context: BuildContext): ChartBuild {
  const count = data.values.length;
  const rowHeight = context.compact ? 13 : 17;
  const height = data.height ?? count * rowHeight + 28;
  const formatValue = data.formatX ?? data.formatY ?? formatChartNumber;
  const [min, max] = extent([data.values], [0]);
  const valueDomain =
    data.xDomain ?? data.yDomain ?? niceDomain(Math.min(0, min), Math.max(0, max));
  const yTicks: AxisTick[] = Array.from({length: count}, (_, index) => ({
    value: index + 0.5,
    label: truncateText(data.labels?.[index] ?? String(index + 1), 110)
  }));
  const frame = createFrame({
    kind: 'bars',
    width: context.width,
    height,
    compact: context.compact,
    common: {...data, formatX: formatValue, yLabel: undefined},
    title: context.title,
    xDomain: valueDomain,
    yDomain: [count, 0],
    xScale: data.xScale,
    yTicks,
    grid: {x: true, y: false},
    pad: {right: 28},
    interactive: !!data.onBarClick
  });
  const colorOf = getBarColors(data, null);
  const baseline = frame.x(Math.min(Math.max(0, valueDomain[0]), valueDomain[1]));
  const rowSpan = (frame.bottom - frame.top) / Math.max(1, count);
  const thickness = Math.min(rowSpan * 0.7, 16);
  for (let index = 0; index < count; index++) {
    const value = data.values[index];
    const center = frame.y(index + 0.5);
    const label = data.labels?.[index] ?? String(index + 1);
    if (Number.isFinite(value)) {
      const end = Math.min(Math.max(frame.x(value), frame.left), frame.right);
      frame.layers.main.append(
        svg(
          'rect',
          {
            class: 'chart-bar',
            x: Math.min(baseline, end).toFixed(2),
            y: (center - thickness / 2).toFixed(2),
            width: Math.max(0.5, Math.abs(end - baseline)).toFixed(2),
            height: thickness.toFixed(2),
            style: `fill:${colorOf(index)}`
          },
          svg('title', {}, `${label}: ${formatValue(value)}`)
        )
      );
      const text = formatValue(value);
      const inside = end + 3 + estimateTextWidth(text) > frame.width - 1;
      frame.layers.front.append(
        svg(
          'text',
          {
            class: inside ? 'chart-bar-value is-inside' : 'chart-bar-value',
            x: inside ? end - 3 : end + 3,
            y: center + 3.5,
            'text-anchor': inside ? 'end' : 'start'
          },
          text
        )
      );
    }
    if (data.onBarClick) {
      const onClick = data.onBarClick;
      const hitRect = svg(
        'rect',
        {
          class: 'chart-hit is-input chart-bar-hit',
          x: 0,
          y: (center - rowSpan / 2).toFixed(2),
          width: frame.width,
          height: rowSpan.toFixed(2),
          tabindex: 0,
          role: 'button',
          'aria-label': `${label}: ${Number.isFinite(value) ? formatValue(value) : 'no data'}`
        },
        svg('title', {}, label)
      );
      hitRect.addEventListener('click', () => onClick(index));
      hitRect.addEventListener('keydown', event => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault();
          onClick(index);
        }
      });
      frame.layers.overlay.append(hitRect);
    }
  }
  const markers = [
    ...(data.markers ?? []),
    ...(data.now !== undefined ? [{x: data.now, label: data.nowLabel}] : [])
  ];
  drawRules(frame, {markers});
  const x = Array.from({length: count}, (_, index) => index);
  const table = buildXYTable(
    [data.yLabel ?? 'Item', data.xLabel ?? 'Value'],
    [x, data.values],
    [index => data.labels?.[index] ?? String(index + 1), formatValue]
  );
  return {svg: frame.root, table};
}
