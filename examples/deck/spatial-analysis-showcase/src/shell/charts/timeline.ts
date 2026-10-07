// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {TimelineChartData} from '../../scenes/chart-types';
import {
  type BuildContext,
  type ChartBuild,
  assignRows,
  attachLink,
  buildXYTable,
  createFrame,
  createHitArea,
  drawRules,
  estimateTextWidth,
  extent,
  niceDomain,
  onPointerDrag,
  slot,
  svg
} from './core';

function getBinWidth(x: ArrayLike<number>): number {
  const gaps: number[] = [];
  for (let index = 1; index < x.length; index++) {
    const gap = x[index] - x[index - 1];
    if (gap > 0) gaps.push(gap);
  }
  if (!gaps.length) return 1;
  gaps.sort((a, b) => a - b);
  return gaps[Math.floor(gaps.length / 2)];
}

/** The x and y domains a timeline would use (shared by small multiples). */
export function getTimelineDomains(data: TimelineChartData): {
  x: [number, number];
  y: [number, number];
} {
  const count = Math.min(data.x.length, data.y.length);
  const [minX, maxX] = extent([Array.from(data.x).slice(0, count)]);
  const bars = (data.mode ?? 'bars') === 'bars';
  const x: [number, number] = data.xDomain
    ? [data.xDomain[0], data.xDomain[1]]
    : [minX, bars ? maxX + getBinWidth(data.x) : maxX];
  const rawY = extent([data.y], [0]);
  return {
    x,
    y: data.yDomain
      ? [data.yDomain[0], data.yDomain[1]]
      : niceDomain(Math.min(0, rawY[0]), rawY[1], 3)
  };
}

/** Builds a time strip: bars, line or area with window, playhead and event ticks. */
export function buildTimeline(data: TimelineChartData, context: BuildContext): ChartBuild {
  const height = data.height ?? (context.compact ? 78 : 92);
  const count = Math.min(data.x.length, data.y.length);
  const mode = data.mode ?? 'bars';
  const domains = getTimelineDomains(data);
  const binWidth = getBinWidth(data.x);

  // Event labels sit above the plot in as many rows as they need.
  const titleBottom = 6 + ((context.title ?? data.title) ? 16 : 0);
  const events = context.compact ? [] : (data.events ?? []);
  // Estimate the rows in pixels (the final plot is a little narrower) to reserve the space.
  const estimateLeft = 30;
  const estimateSpan = context.width - estimateLeft - 8;
  const eventRows = events.length
    ? assignRows(
        events.map(event => {
          const start =
            estimateLeft +
            ((event.at - domains.x[0]) / (domains.x[1] - domains.x[0] || 1)) * estimateSpan;
          return {start, end: start + estimateTextWidth(event.label ?? '') + 4};
        }),
        3
      )
    : {rows: [] as number[], rowCount: 0};
  const reserved = events.length ? eventRows.rowCount * 11 + 4 : 0;
  const hasInput = !!data.onScrub || !!context.options?.onLinkInput;

  const frame = createFrame({
    kind: 'timeline',
    width: context.width,
    height: height + reserved,
    compact: context.compact,
    common: data,
    title: context.title,
    xDomain: domains.x,
    yDomain: domains.y,
    pad: {top: reserved + (data.playhead !== undefined ? 6 : 0)},
    interactive: false
  });
  const {x, y} = frame;
  const main = frame.layers.main;
  const baseline = y(Math.min(Math.max(0, domains.y[0]), domains.y[1]));
  const pointsX: number[] = [];
  const pointsY: number[] = [];
  for (let index = 0; index < count; index++) {
    pointsX.push(x(data.x[index]));
    pointsY.push(y(data.y[index]));
  }

  if (mode === 'bars') {
    for (let index = 0; index < count; index++) {
      if (!Number.isFinite(pointsY[index])) continue;
      const left = pointsX[index];
      const width = Math.max(0.8, x(data.x[index] + binWidth) - left - (count > 120 ? 0.2 : 0.6));
      const top = Math.min(Math.max(pointsY[index], frame.top), frame.bottom);
      main.append(
        svg(
          'rect',
          {
            class: 'chart-bar',
            x: left.toFixed(2),
            y: Math.min(top, baseline).toFixed(2),
            width: width.toFixed(2),
            height: Math.abs(baseline - top).toFixed(2),
            style: `fill:${slot(0)}`
          },
          svg('title', {}, `${frame.formatX(data.x[index])}: ${frame.formatY(data.y[index])}`)
        )
      );
    }
  } else {
    let path = '';
    let pen = false;
    let firstX = 0;
    let lastX = 0;
    for (let index = 0; index < count; index++) {
      if (!Number.isFinite(pointsX[index]) || !Number.isFinite(pointsY[index])) {
        pen = false;
        continue;
      }
      if (!pen) firstX = pointsX[index];
      path += `${pen ? 'L' : 'M'}${pointsX[index].toFixed(1)} ${pointsY[index].toFixed(1)}`;
      lastX = pointsX[index];
      pen = true;
    }
    if (mode === 'area' && path) {
      main.append(
        svg('path', {
          class: 'chart-area',
          d: `${path}L${lastX.toFixed(1)} ${baseline}L${firstX.toFixed(1)} ${baseline}Z`,
          style: `fill:${slot(0)}`
        })
      );
    }
    main.append(
      svg('path', {class: 'chart-line', d: path, style: `stroke:${slot(0)};stroke-width:1.5`})
    );
  }

  // Window: fade outside and bracket the inside.
  if (data.window) {
    const from = Math.min(Math.max(x(data.window[0]), frame.left), frame.right);
    const to = Math.min(Math.max(x(data.window[1]), frame.left), frame.right);
    if (from > frame.left + 0.5) {
      frame.layers.front.append(
        svg('rect', {
          class: 'chart-fade',
          x: frame.left,
          y: frame.top,
          width: from - frame.left,
          height: frame.bottom - frame.top
        })
      );
    }
    if (to < frame.right - 0.5) {
      frame.layers.front.append(
        svg('rect', {
          class: 'chart-fade',
          x: to,
          y: frame.top,
          width: frame.right - to,
          height: frame.bottom - frame.top
        })
      );
    }
    frame.layers.front.append(
      svg('path', {
        class: 'chart-window',
        d: `M${from + 4} ${frame.top}H${from}V${frame.bottom}H${from + 4}M${to - 4} ${frame.top}H${to}V${frame.bottom}H${to - 4}`
      })
    );
  }

  // Event ticks and labels.
  if (events.length) {
    const pixelItems = events.map(event => {
      const position = x(event.at);
      const width = estimateTextWidth(event.label ?? '') + 4;
      const flip = position + width > frame.right + 6;
      return {
        position,
        flip,
        start: flip ? position - width : position,
        end: flip ? position : position + width
      };
    });
    const placement = assignRows(pixelItems, 3);
    events.forEach((event, index) => {
      const item = pixelItems[index];
      if (
        !Number.isFinite(item.position) ||
        item.position < frame.left - 1 ||
        item.position > frame.right + 1
      )
        return;
      const labelY =
        titleBottom + 7 + Math.min(placement.rows[index], Math.max(0, eventRows.rowCount - 1)) * 11;
      frame.layers.front.append(
        svg('line', {
          class: 'chart-event',
          x1: item.position,
          x2: item.position,
          y1: labelY + 2,
          y2: frame.bottom
        })
      );
      if (event.label) {
        frame.root.append(
          svg(
            'text',
            {
              class: 'chart-note',
              x: item.flip ? item.position - 2 : item.position + 2,
              y: labelY,
              'text-anchor': item.flip ? 'end' : 'start'
            },
            event.label
          )
        );
      }
    });
  }
  drawRules(frame, data);

  // Playhead.
  let setPlayhead: ((time: number) => void) | undefined;
  if (data.playhead !== undefined) {
    const line = svg('line', {
      class: 'chart-rule chart-marker chart-playhead',
      y1: frame.top - 5,
      y2: frame.bottom
    });
    const handle = svg('polygon', {class: 'chart-playhead-handle'});
    const label = svg('text', {class: 'chart-note chart-note-signal'});
    frame.layers.front.append(line, handle, label);
    setPlayhead = time => {
      const position = x(time);
      const visible =
        Number.isFinite(position) && position >= frame.left - 0.5 && position <= frame.right + 0.5;
      for (const element of [line, handle, label])
        element.setAttribute('visibility', visible ? 'visible' : 'hidden');
      if (!visible) return;
      line.setAttribute('x1', String(position));
      line.setAttribute('x2', String(position));
      handle.setAttribute(
        'points',
        `${position - 4},${frame.top - 6} ${position + 4},${frame.top - 6} ${position},${frame.top}`
      );
      const text = frame.formatX(time);
      const flip = position + 6 + estimateTextWidth(text) > frame.right + 4;
      label.textContent = text;
      label.setAttribute('x', String(position + (flip ? -4 : 4)));
      label.setAttribute('y', String(frame.top + 9));
      label.setAttribute('text-anchor', flip ? 'end' : 'start');
    };
    setPlayhead(data.playhead);
  }

  // Pointer: hover readout, scrub and link.
  const hit = createHitArea(frame, hasInput);
  const updateLink = attachLink(frame, data.link, context.options, hit);
  if (data.onScrub) {
    const onScrub = data.onScrub;
    hit.classList.add('is-input');
    onPointerDrag(hit, frame.root, point => {
      const [low, high] = frame.x.domain;
      const time = Math.min(Math.max(frame.x.invert(point.x), low), high);
      setPlayhead?.(time);
      onScrub(time);
    });
  }
  if (!context.compact && count) {
    const cursor = svg('line', {
      class: 'chart-cursor',
      y1: frame.top,
      y2: frame.bottom,
      visibility: 'hidden'
    });
    const tooltip = svg('text', {
      class: 'chart-note chart-tooltip',
      visibility: 'hidden',
      y: frame.top + 9
    });
    frame.layers.overlay.append(cursor, tooltip);
    const hide = () => {
      cursor.setAttribute('visibility', 'hidden');
      tooltip.setAttribute('visibility', 'hidden');
    };
    hit.addEventListener('pointermove', event => {
      const box = frame.root.getBoundingClientRect();
      if (!box.width) return;
      const viewX = ((event.clientX - box.left) / box.width) * frame.width;
      const target = x.invert(viewX);
      let best = 0;
      for (let index = 1; index < count; index++) {
        if (Math.abs(data.x[index] - target) < Math.abs(data.x[best] - target)) best = index;
      }
      const position =
        mode === 'bars' ? (pointsX[best] + x(data.x[best] + binWidth)) / 2 : pointsX[best];
      if (!Number.isFinite(position) || !Number.isFinite(data.y[best])) return hide();
      const flip = position > (frame.left + frame.right) / 2;
      cursor.setAttribute('x1', String(position));
      cursor.setAttribute('x2', String(position));
      cursor.setAttribute('visibility', 'visible');
      tooltip.textContent = `${frame.formatX(data.x[best])}: ${frame.formatY(data.y[best])}`;
      tooltip.setAttribute('x', String(position + (flip ? -5 : 5)));
      tooltip.setAttribute('text-anchor', flip ? 'end' : 'start');
      tooltip.setAttribute('visibility', 'visible');
    });
    hit.addEventListener('pointerleave', hide);
  }

  const table = buildXYTable(
    [data.xLabel ?? 'Time', data.yLabel ?? 'Value'],
    [Array.from(data.x).slice(0, count), Array.from(data.y).slice(0, count)],
    [frame.formatX, frame.formatY]
  );
  return {svg: frame.root, table, updateLink};
}
