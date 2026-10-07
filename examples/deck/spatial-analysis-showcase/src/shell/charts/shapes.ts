// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  DumbbellChartData,
  ForestChartData,
  LorenzChartData,
  SlopeChartData
} from '../../scenes/chart-types';
import {
  type AxisTick,
  type BuildContext,
  type ChartBuild,
  type Frame,
  attachLink,
  buildXYTable,
  createFrame,
  createHitArea,
  createRoot,
  createScale,
  drawRules,
  estimateTextWidth,
  extent,
  formatChartNumber,
  formatPercent,
  niceDomain,
  nudgePositions,
  slot,
  svg,
  truncateText
} from './core';

/** Computes the Gini coefficient of a Lorenz curve by the trapezoid rule. */
export function getGini(x: ArrayLike<number>, y: ArrayLike<number>): number {
  let area = 0;
  for (let index = 1; index < Math.min(x.length, y.length); index++) {
    area += ((x[index] - x[index - 1]) * (y[index] + y[index - 1])) / 2;
  }
  return 1 - 2 * area;
}

/** Builds a Lorenz curve with the shaded Gini area. */
export function buildLorenz(data: LorenzChartData, context: BuildContext): ChartBuild {
  const count = Math.min(data.x.length, data.y.length);
  const frame = createFrame({
    kind: 'lorenz',
    width: context.width,
    height: data.height ?? (context.compact ? 130 : 190),
    compact: context.compact,
    common: {
      ...data,
      formatX: data.formatX ?? formatPercent,
      formatY: data.formatY ?? formatPercent
    },
    title: context.title,
    xDomain: data.xDomain ?? [0, 1],
    yDomain: data.yDomain ?? [0, 1],
    grid: {x: true, y: true}
  });
  const {x, y} = frame;
  const curve: string[] = [];
  const diagonal: string[] = [];
  for (let index = 0; index < count; index++) {
    const px = x(data.x[index]);
    const py = y(data.y[index]);
    if (!Number.isFinite(px) || !Number.isFinite(py)) continue;
    curve.push(`${px.toFixed(1)},${py.toFixed(1)}`);
    diagonal.unshift(`${px.toFixed(1)},${y(data.x[index]).toFixed(1)}`);
  }
  frame.layers.main.append(
    svg('polygon', {
      class: 'chart-envelope',
      points: [...curve, ...diagonal].join(' '),
      style: `--band:${slot(0)};--band-opacity:0.2`
    }),
    svg('line', {
      class: 'chart-rule',
      x1: x(Math.max(x.domain[0], y.domain[0])),
      y1: y(Math.max(x.domain[0], y.domain[0])),
      x2: x(Math.min(x.domain[1], y.domain[1])),
      y2: y(Math.min(x.domain[1], y.domain[1]))
    }),
    svg('polyline', {
      class: 'chart-line',
      points: curve.join(' '),
      style: `stroke:${slot(0)};stroke-width:2`
    })
  );
  const gini = data.gini ?? getGini(data.x, data.y);
  if (Number.isFinite(gini)) {
    frame.layers.front.append(
      svg(
        'text',
        {
          class: 'chart-note chart-note-strong',
          x: frame.right - 4,
          y: frame.bottom - 6,
          'text-anchor': 'end'
        },
        `Gini ${gini.toFixed(2)}`
      )
    );
  }
  drawRules(frame, data);
  const hit = createHitArea(frame, !!context.options?.onLinkInput);
  const updateLink = attachLink(frame, data.link, context.options, hit);
  return {
    svg: frame.root,
    table: buildXYTable(
      [data.xLabel ?? 'Population share', data.yLabel ?? 'Quantity share'],
      [Array.from(data.x).slice(0, count), Array.from(data.y).slice(0, count)],
      [frame.formatX, frame.formatY]
    ),
    updateLink
  };
}

type RowChart = {rows: readonly {label: string; highlight?: boolean}[]};

function getRowTicks(rows: RowChart['rows'], maxWidth = 110): AxisTick[] {
  return rows.map((row, index) => ({value: index + 0.5, label: truncateText(row.label, maxWidth)}));
}

/** Faint bands behind highlighted rows. */
function drawRowHighlights(frame: Frame, rows: RowChart['rows']): void {
  const span = (frame.bottom - frame.top) / Math.max(1, rows.length);
  rows.forEach((row, index) => {
    if (!row.highlight) return;
    frame.layers.back.append(
      svg('rect', {
        class: 'chart-row-highlight',
        x: 0,
        y: frame.top + index * span,
        width: frame.width,
        height: span
      })
    );
  });
}

/** Builds a forest plot: estimates with intervals around a reference line. */
export function buildForest(data: ForestChartData, context: BuildContext): ChartBuild {
  const count = data.rows.length;
  const rowHeight = context.compact ? 13 : 18;
  const logX = data.xScale === 'log';
  const values = data.rows.flatMap(row => [row.estimate, row.low, row.high]);
  const raw = extent([values], data.reference !== undefined ? [data.reference] : [], logX);
  const domain = data.xDomain ?? (logX ? raw : niceDomain(raw[0], raw[1]));
  const frame = createFrame({
    kind: 'forest',
    width: context.width,
    height: data.height ?? count * rowHeight + 34,
    compact: context.compact,
    common: {...data, yLabel: undefined},
    title: context.title,
    xDomain: domain,
    yDomain: [count, 0],
    xScale: data.xScale,
    yTicks: getRowTicks(data.rows),
    grid: {x: true, y: false}
  });
  drawRowHighlights(frame, data.rows);
  const {x, y} = frame;
  if (data.reference !== undefined && Number.isFinite(x(data.reference))) {
    const position = x(data.reference);
    frame.layers.back.append(
      svg('line', {
        class: 'chart-rule',
        x1: position,
        x2: position,
        y1: frame.top,
        y2: frame.bottom
      })
    );
    if (data.referenceLabel) {
      frame.layers.front.append(
        svg('text', {class: 'chart-note', x: position + 3, y: frame.top - 1.5}, data.referenceLabel)
      );
    }
  }
  data.rows.forEach((row, index) => {
    const center = y(index + 0.5);
    const low = Math.max(frame.left, Math.min(frame.right, x(row.low)));
    const high = Math.max(frame.left, Math.min(frame.right, x(row.high)));
    const estimate = x(row.estimate);
    if (![low, high, estimate].every(Number.isFinite)) return;
    const tone = row.highlight ? 'is-highlight' : '';
    const label = `${row.label}: ${frame.formatX(row.estimate)} (${frame.formatX(row.low)} to ${frame.formatX(row.high)})`;
    frame.layers.main.append(
      svg(
        'g',
        {class: `chart-interval ${tone}`},
        svg('line', {class: 'chart-interval-line', x1: low, x2: high, y1: center, y2: center}),
        svg('line', {
          class: 'chart-interval-line',
          x1: low,
          x2: low,
          y1: center - 3,
          y2: center + 3
        }),
        svg('line', {
          class: 'chart-interval-line',
          x1: high,
          x2: high,
          y1: center - 3,
          y2: center + 3
        }),
        svg('rect', {
          class: 'chart-interval-point',
          x: estimate - 3,
          y: center - 3,
          width: 6,
          height: 6
        }),
        svg('title', {}, label)
      )
    );
  });
  drawRules(frame, {markers: data.markers});
  const hit = createHitArea(frame, !!context.options?.onLinkInput);
  const updateLink = attachLink(frame, data.link, context.options, hit);
  return {
    svg: frame.root,
    table: {
      headers: ['Row', 'Estimate', 'Low', 'High'],
      rows: data.rows.map(row => [
        row.label,
        frame.formatX(row.estimate),
        frame.formatX(row.low),
        frame.formatX(row.high)
      ])
    },
    updateLink
  };
}

/** Builds a dumbbell chart: two values per row joined by a line. */
export function buildDumbbell(data: DumbbellChartData, context: BuildContext): ChartBuild {
  const count = data.rows.length;
  const rowHeight = context.compact ? 13 : 18;
  const logX = data.xScale === 'log';
  const raw = extent([data.rows.flatMap(row => [row.a, row.b])], [], logX);
  const domain = data.xDomain ?? (logX ? raw : niceDomain(raw[0], raw[1]));
  const frame = createFrame({
    kind: 'dumbbell',
    width: context.width,
    height: (data.height ?? count * rowHeight + 34) + 14,
    compact: context.compact,
    common: {...data, yLabel: undefined},
    title: context.title,
    xDomain: domain,
    yDomain: [count, 0],
    xScale: data.xScale,
    yTicks: getRowTicks(data.rows),
    grid: {x: true, y: false},
    pad: {top: 14}
  });
  drawRowHighlights(frame, data.rows);
  const {x, y} = frame;
  data.rows.forEach((row, index) => {
    const center = y(index + 0.5);
    const a = x(row.a);
    const b = x(row.b);
    if (!Number.isFinite(a) || !Number.isFinite(b)) return;
    frame.layers.main.append(
      svg(
        'g',
        {class: `chart-dumbbell${row.highlight ? ' is-highlight' : ''}`},
        svg('line', {class: 'chart-dumbbell-bar', x1: a, x2: b, y1: center, y2: center}),
        svg('circle', {
          class: 'chart-dumbbell-dot',
          cx: a,
          cy: center,
          r: 3.4,
          style: `fill:${slot(0)}`
        }),
        svg('circle', {
          class: 'chart-dumbbell-dot',
          cx: b,
          cy: center,
          r: 3.4,
          style: `fill:${slot(1)}`
        }),
        svg(
          'title',
          {},
          `${row.label}: ${data.aLabel} ${frame.formatX(row.a)}, ${data.bLabel} ${frame.formatX(row.b)}`
        )
      )
    );
  });
  // Key in the header.
  const keyY = frame.top - 7;
  let cursor = frame.left;
  for (const [label, color] of [
    [data.aLabel, slot(0)],
    [data.bLabel, slot(1)]
  ] as const) {
    frame.root.append(
      svg('circle', {
        class: 'chart-dumbbell-dot',
        cx: cursor + 3,
        cy: keyY - 3.5,
        r: 3.4,
        style: `fill:${color}`
      }),
      svg('text', {class: 'chart-label', x: cursor + 10, y: keyY}, label)
    );
    cursor += 16 + estimateTextWidth(label, 11);
  }
  drawRules(frame, {markers: data.markers});
  const hit = createHitArea(frame, !!context.options?.onLinkInput);
  const updateLink = attachLink(frame, data.link, context.options, hit);
  return {
    svg: frame.root,
    table: {
      headers: ['Row', data.aLabel, data.bLabel],
      rows: data.rows.map(row => [row.label, frame.formatX(row.a), frame.formatX(row.b)])
    },
    updateLink
  };
}

/** Builds a slopegraph: two columns of values joined per row, labelled at both ends. */
export function buildSlope(data: SlopeChartData, context: BuildContext): ChartBuild {
  const count = data.rows.length;
  const width = context.width;
  const title = context.title ?? data.title;
  const format = data.formatY ?? formatChartNumber;
  const height = data.height ?? Math.max(150, count * 15 + 56);
  const top = 6 + (title ? 16 : 0) + 18;
  const bottom = height - 8;
  const logY = data.yScale === 'log';
  const raw = extent([data.rows.flatMap(row => [row.a, row.b])], [], logY);
  const domain = data.yDomain ?? (logY ? raw : niceDomain(raw[0], raw[1], 3));
  const y = createScale(domain, [bottom, top], data.yScale);

  const textWidth = (side: 'a' | 'b') =>
    Math.max(
      0,
      ...data.rows.map(row =>
        estimateTextWidth(`${row.label} ${format(side === 'a' ? row.a : row.b)}`)
      )
    );
  const labelBudget = Math.max(60, (width - 70) / 2);
  const padLeft = Math.min(labelBudget, textWidth('a') + 10);
  const padRight = Math.min(labelBudget, textWidth('b') + 10);
  const xa = padLeft;
  const xb = width - padRight;

  const root = createRoot('slope', width, height, title, data.description, 'Slopegraph');
  if (title) root.append(svg('text', {class: 'chart-title', x: 2, y: 13}, title));
  const headerY = top - 8;
  root.append(
    svg(
      'text',
      {class: 'chart-label is-strong', x: xa, y: headerY, 'text-anchor': 'middle'},
      data.aLabel
    ),
    svg(
      'text',
      {class: 'chart-label is-strong', x: xb, y: headerY, 'text-anchor': 'middle'},
      data.bLabel
    )
  );
  const anyHighlight = data.rows.some(row => row.highlight);
  const leftPositions = nudgePositions(
    data.rows.map(row => y(row.a)),
    top,
    bottom,
    11
  );
  const rightPositions = nudgePositions(
    data.rows.map(row => y(row.b)),
    top,
    bottom,
    11
  );
  const lines = svg('g', {class: 'chart-slope-lines'});
  const labels = svg('g', {class: 'chart-slope-labels'});
  root.append(lines, labels);
  const order = data.rows
    .map((_, index) => index)
    .sort((i, j) => Number(!!data.rows[i].highlight) - Number(!!data.rows[j].highlight));
  for (const index of order) {
    const row = data.rows[index];
    const ya = y(row.a);
    const yb = y(row.b);
    if (!Number.isFinite(ya) || !Number.isFinite(yb)) continue;
    const tone = row.highlight ? 'is-highlight' : anyHighlight ? 'is-quiet' : '';
    lines.append(
      svg(
        'g',
        {class: `chart-slope ${tone}`},
        svg('line', {class: 'chart-slope-line', x1: xa, y1: ya, x2: xb, y2: yb}),
        svg('circle', {class: 'chart-slope-dot', cx: xa, cy: ya, r: 2.6}),
        svg('circle', {class: 'chart-slope-dot', cx: xb, cy: yb, r: 2.6}),
        svg(
          'title',
          {},
          `${row.label}: ${data.aLabel} ${format(row.a)}, ${data.bLabel} ${format(row.b)}`
        )
      )
    );
    const maxLabel = Math.max(20, padLeft - 12);
    labels.append(
      svg(
        'text',
        {
          class: `chart-slope-label ${tone}`,
          x: xa - 7,
          y: leftPositions[index] + 3.5,
          'text-anchor': 'end'
        },
        `${truncateText(row.label, maxLabel - estimateTextWidth(format(row.a)) - 4)} ${format(row.a)}`
      ),
      svg(
        'text',
        {
          class: `chart-slope-label ${tone}`,
          x: xb + 7,
          y: rightPositions[index] + 3.5,
          'text-anchor': 'start'
        },
        `${format(row.b)} ${truncateText(row.label, Math.max(20, padRight - 12) - estimateTextWidth(format(row.b)) - 4)}`
      )
    );
  }
  return {
    svg: root,
    table: {
      headers: ['Row', data.aLabel, data.bLabel],
      rows: data.rows.map(row => [row.label, format(row.a), format(row.b)])
    }
  };
}
