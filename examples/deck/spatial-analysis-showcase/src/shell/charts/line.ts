// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {LineChartData, LineEnvelope} from '../../scenes/chart-types';
import {
  type BuildContext,
  type ChartBuild,
  type LegendItem,
  buildXYTable,
  createFrame,
  createHitArea,
  attachLink,
  drawRules,
  estimateTextWidth,
  extent,
  formatChartNumber,
  indexArray,
  niceDomain,
  nudgePositions,
  slot,
  svg,
  truncateText,
  FONT_TICK
} from './core';

type Run = {x: number; y: number}[];

/** Splits a polyline into runs of finite pixel points (gaps at missing values). */
function getRuns(
  xs: ArrayLike<number>,
  ys: ArrayLike<number>,
  mapX: (value: number) => number,
  mapY: (value: number) => number
): Run[] {
  const runs: Run[] = [];
  let current: Run = [];
  const length = Math.min(xs.length, ys.length);
  for (let index = 0; index < length; index++) {
    const px = mapX(xs[index]);
    const py = mapY(ys[index]);
    if (Number.isFinite(px) && Number.isFinite(py)) current.push({x: px, y: py});
    else if (current.length) {
      runs.push(current);
      current = [];
    }
  }
  if (current.length) runs.push(current);
  return runs;
}

const toPath = (run: Run) =>
  run
    .map((point, index) => `${index ? 'L' : 'M'}${point.x.toFixed(1)} ${point.y.toFixed(1)}`)
    .join('');

/** The x, y and y2 domains a line chart would use (shared by small multiples). */
export function getLineDomains(data: LineChartData): {
  x: [number, number];
  y: [number, number];
  y2?: [number, number];
} {
  const logY = data.yScale === 'log';
  const xArrays: ArrayLike<number>[] = [];
  const yArrays: ArrayLike<number>[] = [];
  const y2Arrays: ArrayLike<number>[] = [];
  let hasArea = false;
  for (const series of data.series) {
    xArrays.push(series.x ?? indexArray(series.y.length));
    if (series.axis === 'y2') y2Arrays.push(series.y);
    else yArrays.push(series.y);
    if (series.area) hasArea = true;
  }
  const envelopes: LineEnvelope[] = [...(data.band ? [data.band] : []), ...(data.envelopes ?? [])];
  for (const envelope of envelopes) {
    xArrays.push(envelope.x ?? indexArray(envelope.low.length));
    yArrays.push(envelope.low, envelope.high);
  }
  const x =
    data.xDomain ??
    (data.xScale === 'log'
      ? extent(
          xArrays,
          (data.markers ?? []).map(marker => marker.x),
          true
        )
      : extent(
          xArrays,
          (data.markers ?? []).map(marker => marker.x)
        ));
  const guideValues = (data.guides ?? []).map(guide => guide.y);
  let y: [number, number];
  if (data.yDomain) y = [data.yDomain[0], data.yDomain[1]];
  else if (logY) y = extent(yArrays, guideValues, true);
  else {
    const raw = extent(yArrays, hasArea ? [0, ...guideValues] : guideValues);
    y = niceDomain(raw[0], raw[1]);
  }
  const result: {x: [number, number]; y: [number, number]; y2?: [number, number]} = {
    x: [x[0], x[1]],
    y
  };
  if (data.y2Domain) result.y2 = [data.y2Domain[0], data.y2Domain[1]];
  else if (y2Arrays.length) {
    const raw = extent(y2Arrays);
    result.y2 = niceDomain(raw[0], raw[1]);
  }
  return result;
}

/** Builds a line chart. */
export function buildLine(data: LineChartData, context: BuildContext): ChartBuild {
  const height = data.height ?? (context.compact ? 110 : 140);
  const domains = getLineDomains(data);
  const hasY2 = domains.y2 !== undefined;
  const series = data.series.map((entry, index) => ({
    entry,
    index,
    xs: entry.x ?? indexArray(entry.y.length),
    color: entry.color ?? index,
    onY2: entry.axis === 'y2' && hasY2
  }));
  const labelled = series.filter(item => item.entry.label);
  const direct = new Set<number>();
  if (!hasY2 && !context.compact) {
    for (const item of labelled) {
      if (item.entry.directLabel ?? labelled.length <= 3) direct.add(item.index);
    }
  }
  const directWidth = direct.size
    ? Math.min(
        66,
        Math.max(...[...direct].map(index => estimateTextWidth(series[index].entry.label ?? '')))
      ) + 8
    : 0;
  const firstY2 = series.find(item => item.onY2);

  const frame = createFrame({
    kind: 'line',
    width: context.width,
    height,
    compact: context.compact,
    common: data,
    title: context.title,
    xDomain: domains.x,
    yDomain: domains.y,
    xScale: data.xScale,
    yScale: data.yScale,
    y2Domain: domains.y2,
    y2Label: data.y2Label,
    formatY2: data.formatY2,
    y2Color: firstY2 ? slot(firstY2.color) : undefined,
    pad: {right: directWidth}
  });
  const {x, y} = frame;
  const scaleOf = (onY2: boolean) => (onY2 && frame.y2 ? frame.y2 : y);
  const main = frame.layers.main;

  // Envelopes (grey band first, then coloured ones) under the lines.
  const envelopes: {envelope: LineEnvelope; style: string}[] = [
    ...(data.band ? [{envelope: data.band, style: '--band:var(--muted);--band-opacity:0.16'}] : []),
    ...(data.envelopes ?? []).map(envelope => ({
      envelope,
      style: `--band:${slot(envelope.color ?? 0)};--band-opacity:0.18`
    }))
  ];
  for (const {envelope, style} of envelopes) {
    const envelopeX = envelope.x ?? indexArray(envelope.low.length);
    const upper: string[] = [];
    const lower: string[] = [];
    for (let index = 0; index < envelope.low.length; index++) {
      const px = x(envelopeX[index]);
      const high = y(envelope.high[index]);
      const low = y(envelope.low[index]);
      if (!Number.isFinite(px) || !Number.isFinite(high) || !Number.isFinite(low)) continue;
      upper.push(`${px.toFixed(1)},${high.toFixed(1)}`);
      lower.unshift(`${px.toFixed(1)},${low.toFixed(1)}`);
    }
    if (upper.length) {
      main.append(
        svg('polygon', {class: 'chart-envelope', points: [...upper, ...lower].join(' '), style})
      );
    }
  }

  // Ghost context lines first, then areas, then the lines on top.
  const order = [...series].sort((a, b) => Number(!!b.entry.ghost) - Number(!!a.entry.ghost));
  for (const item of order) {
    const scale = scaleOf(item.onY2);
    const runs = getRuns(item.xs, item.entry.y, x, scale);
    const ghost = !!item.entry.ghost;
    const color = ghost ? 'var(--chart-muted)' : slot(item.color);
    if (item.entry.area && !ghost) {
      const baseline = Math.min(
        frame.bottom,
        Math.max(frame.top, scale.isLog ? frame.bottom : scale(Math.max(0, scale.domain[0])))
      );
      for (const run of runs) {
        main.append(
          svg('path', {
            class: 'chart-area',
            d: `${toPath(run)}L${run[run.length - 1].x.toFixed(1)} ${baseline}L${run[0].x.toFixed(1)} ${baseline}Z`,
            style: `fill:${color}`
          })
        );
      }
    }
    const linePath = runs.map(toPath).join('');
    main.append(
      svg('path', {
        class: `chart-line${ghost ? ' is-ghost' : ''}`,
        d: linePath,
        style: `stroke:${color};stroke-width:${item.entry.width ?? (ghost ? 1 : 1.75)}`,
        'stroke-dasharray': item.entry.dashed ? '5 3' : undefined
      })
    );
    if (item.entry.points && !ghost) {
      for (const run of runs) {
        if (run.length > 150) continue;
        for (const point of run) {
          main.append(
            svg('circle', {
              class: 'chart-point',
              cx: point.x.toFixed(1),
              cy: point.y.toFixed(1),
              r: 2.2,
              style: `fill:${color}`
            })
          );
        }
      }
    }
  }

  drawRules(frame, data);

  // Direct labels at the line ends, nudged apart.
  if (direct.size) {
    const entries = [...direct].map(index => {
      const item = series[index];
      const scale = scaleOf(item.onY2);
      let lastY = Number.NaN;
      for (let i = item.entry.y.length - 1; i >= 0; i--) {
        const py = scale(item.entry.y[i]);
        if (Number.isFinite(py) && Number.isFinite(x(item.xs[i]))) {
          lastY = py;
          break;
        }
      }
      return {item, py: lastY};
    });
    const valid = entries.filter(entry => Number.isFinite(entry.py));
    const positions = nudgePositions(
      valid.map(entry => entry.py),
      frame.top + 4,
      frame.bottom - 2,
      FONT_TICK + 2
    );
    valid.forEach(({item}, index) => {
      const ghost = !!item.entry.ghost;
      const label = item.entry.label ?? '';
      frame.root.append(
        svg(
          'text',
          {
            class: `chart-direct-label${ghost ? ' is-ghost' : ''}`,
            x: frame.right + 5,
            y: positions[index] + 3.5,
            style: ghost ? undefined : `fill:${slot(item.color)}`
          },
          svg('title', {}, label),
          truncateText(label, directWidth - 8)
        )
      );
    });
  }

  // Hover and link.
  const hasLink = !!data.link && !!context.options?.onLinkInput;
  const hit = createHitArea(frame, hasLink);
  const updateLink = attachLink(frame, data.link, context.options, hit);
  const hoverable = series.filter(item => !item.entry.ghost);
  const first = hoverable[0];
  if (first && !context.compact) {
    const cursor = svg('line', {
      class: 'chart-cursor',
      y1: frame.top,
      y2: frame.bottom,
      visibility: 'hidden'
    });
    const dots = hoverable.map(item =>
      svg('circle', {
        class: 'chart-cursor-dot',
        r: 3,
        visibility: 'hidden',
        style: `fill:${slot(item.color)}`
      })
    );
    const tooltip = svg('text', {class: 'chart-note chart-tooltip', visibility: 'hidden'});
    frame.layers.overlay.append(cursor, ...dots, tooltip);
    const hide = () => {
      cursor.setAttribute('visibility', 'hidden');
      tooltip.setAttribute('visibility', 'hidden');
      for (const dot of dots) dot.setAttribute('visibility', 'hidden');
    };
    const nearest = (xs: ArrayLike<number>, target: number) => {
      let best = -1;
      let bestDistance = Number.POSITIVE_INFINITY;
      for (let index = 0; index < xs.length; index++) {
        const distance = Math.abs(xs[index] - target);
        if (distance < bestDistance) {
          best = index;
          bestDistance = distance;
        }
      }
      return best;
    };
    const show = (event: PointerEvent) => {
      const box = frame.root.getBoundingClientRect();
      if (!box.width) return;
      const viewX = ((event.clientX - box.left) / box.width) * frame.width;
      if (viewX < frame.left - 2 || viewX > frame.right + 2) return hide();
      const target = x.invert(viewX);
      const anchorIndex = nearest(first.xs, target);
      if (anchorIndex < 0) return hide();
      const anchorX = first.xs[anchorIndex];
      const position = x(anchorX);
      if (!Number.isFinite(position)) return hide();
      cursor.setAttribute('x1', String(position));
      cursor.setAttribute('x2', String(position));
      cursor.setAttribute('visibility', 'visible');
      tooltip.replaceChildren();
      const flip = position > (frame.left + frame.right) / 2;
      tooltip.setAttribute('visibility', 'visible');
      tooltip.append(
        svg(
          'tspan',
          {x: position + (flip ? -6 : 6), dy: 0, class: 'chart-tooltip-head'},
          frame.formatX(anchorX)
        )
      );
      tooltip.setAttribute('y', String(frame.top + 9));
      tooltip.setAttribute('text-anchor', flip ? 'end' : 'start');
      hoverable.forEach((item, index) => {
        const sampleIndex = item === first ? anchorIndex : nearest(item.xs, anchorX);
        const value = item.entry.y[sampleIndex];
        const py = scaleOf(item.onY2)(value);
        if (!Number.isFinite(py)) {
          dots[index].setAttribute('visibility', 'hidden');
          return;
        }
        dots[index].setAttribute('cx', String(x(item.xs[sampleIndex])));
        dots[index].setAttribute('cy', String(py));
        dots[index].setAttribute('visibility', 'visible');
        const format = item.onY2 ? (data.formatY2 ?? formatChartNumber) : frame.formatY;
        tooltip.append(
          svg(
            'tspan',
            {x: position + (flip ? -6 : 6), dy: 11},
            svg(
              'tspan',
              {class: 'chart-tooltip-key', style: `fill:${slot(item.color)}`},
              item.entry.label ? `${item.entry.label} ` : ''
            ),
            format(value)
          )
        );
      });
    };
    hit.addEventListener('pointermove', show);
    hit.addEventListener('pointerdown', show);
    hit.addEventListener('pointerleave', hide);
    hit.addEventListener('pointercancel', hide);
  }

  // Legend: series not labelled directly, envelope labels.
  const legend: LegendItem[] = [];
  for (const item of series) {
    if (!item.entry.label || direct.has(item.index)) continue;
    legend.push({
      label: item.entry.label,
      color: item.entry.ghost ? 'var(--chart-muted)' : item.color,
      shape: item.entry.dashed ? 'dashed' : item.entry.area ? 'band' : 'line'
    });
  }
  if (data.band?.label) legend.push({label: data.band.label, color: 'var(--muted)', shape: 'band'});
  for (const envelope of data.envelopes ?? []) {
    if (envelope.label)
      legend.push({label: envelope.label, color: envelope.color ?? 0, shape: 'band'});
  }

  // Table: aligned columns when the series share x, else long format.
  const tableSeries = series.filter(item => item.entry.y.length);
  const aligned = tableSeries.every(item => {
    const reference = tableSeries[0].xs;
    if (item.xs.length !== reference.length) return false;
    return (
      item.xs[0] === reference[0] && item.xs[item.xs.length - 1] === reference[reference.length - 1]
    );
  });
  const table = !tableSeries.length
    ? null
    : aligned
      ? buildXYTable(
          [
            data.xLabel ?? 'x',
            ...tableSeries.map((item, i) => item.entry.label ?? `Series ${i + 1}`)
          ],
          [tableSeries[0].xs, ...tableSeries.map(item => item.entry.y)],
          [
            frame.formatX,
            ...tableSeries.map(item =>
              item.onY2 ? (data.formatY2 ?? formatChartNumber) : frame.formatY
            )
          ]
        )
      : buildXYTable(
          [data.xLabel ?? 'x', data.yLabel ?? 'y'],
          [tableSeries[0].xs, tableSeries[0].entry.y],
          [frame.formatX, frame.formatY]
        );
  return {svg: frame.root, legend, table, updateLink};
}
