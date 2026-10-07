// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {ScatterChartData} from '../../scenes/chart-types';
import {
  type BuildContext,
  type ChartBuild,
  type Frame,
  attachLink,
  buildXYTable,
  colorToCss,
  createFrame,
  createHitArea,
  drawRules,
  estimateTextWidth,
  extent,
  getViewPoint,
  niceDomain,
  slot,
  svg
} from './core';

/** Above this many points the dots are drawn as one path per color instead of circles. */
const CIRCLE_LIMIT = 1500;
/** Pointer movement in viewBox units below which a press is a click, not a brush. */
const DRAG_THRESHOLD = 3;

/** The x and y domains a scatter plot would use (shared by small multiples). */
export function getScatterDomains(data: ScatterChartData): {
  x: [number, number];
  y: [number, number];
} {
  const rect = data.referenceRect;
  const includeX = [
    ...(data.quadrants ? [data.quadrants.x] : []),
    ...(rect ? [rect.x0, rect.x1] : []),
    ...(data.markers ?? []).map(marker => marker.x)
  ];
  const includeY = [
    ...(data.quadrants ? [data.quadrants.y] : []),
    ...(rect ? [rect.y0, rect.y1] : []),
    ...(data.guides ?? []).map(guide => guide.y)
  ];
  const logX = data.xScale === 'log';
  const logY = data.yScale === 'log';
  const rawX = extent([data.x], includeX, logX);
  const rawY = extent([data.y], includeY, logY);
  return {
    x: data.xDomain
      ? [data.xDomain[0], data.xDomain[1]]
      : logX
        ? rawX
        : niceDomain(rawX[0], rawX[1]),
    y: data.yDomain
      ? [data.yDomain[0], data.yDomain[1]]
      : logY
        ? rawY
        : niceDomain(rawY[0], rawY[1])
  };
}

/** Clips the line `y = slope * x + intercept` to a rectangle; `null` when it misses it. */
function clipLine(
  slope: number,
  intercept: number,
  xRange: readonly [number, number],
  yRange: readonly [number, number]
): [[number, number], [number, number]] | null {
  const yLow = Math.min(yRange[0], yRange[1]);
  const yHigh = Math.max(yRange[0], yRange[1]);
  let from = Math.min(xRange[0], xRange[1]);
  let to = Math.max(xRange[0], xRange[1]);
  if (slope !== 0) {
    const a = (yLow - intercept) / slope;
    const b = (yHigh - intercept) / slope;
    from = Math.max(from, Math.min(a, b));
    to = Math.min(to, Math.max(a, b));
  } else if (intercept < yLow || intercept > yHigh) return null;
  if (!(to > from)) return null;
  return [
    [from, slope * from + intercept],
    [to, slope * to + intercept]
  ];
}

/** Builds a scatter plot with optional quadrants, fit, reference box, brush and ringed points. */
export function buildScatter(data: ScatterChartData, context: BuildContext): ChartBuild {
  const height = data.height ?? (context.compact ? 120 : 190);
  const count = Math.min(data.x.length, data.y.length);
  const domains = getScatterDomains(data);
  const brushing = !!data.onBrush;
  const frame = createFrame({
    kind: 'scatter',
    width: context.width,
    height,
    compact: context.compact,
    common: data,
    title: context.title,
    xDomain: domains.x,
    yDomain: domains.y,
    xScale: data.xScale,
    yScale: data.yScale,
    grid: {x: true, y: true},
    interactive: brushing || !!data.onPointClick
  });
  const {x, y} = frame;
  const {back, main, front} = frame.layers;

  // Reference box and decorations under the points.
  if (data.referenceRect) {
    const rect = data.referenceRect;
    const x0 = Math.max(frame.left, Math.min(x(rect.x0), x(rect.x1)));
    const x1 = Math.min(frame.right, Math.max(x(rect.x0), x(rect.x1)));
    const y0 = Math.max(frame.top, Math.min(y(rect.y0), y(rect.y1)));
    const y1 = Math.min(frame.bottom, Math.max(y(rect.y0), y(rect.y1)));
    if (x1 > x0 && y1 > y0) {
      back.append(
        svg('rect', {class: 'chart-refrect', x: x0, y: y0, width: x1 - x0, height: y1 - y0})
      );
      if (rect.label) {
        back.append(svg('text', {class: 'chart-band-label', x: x0 + 3, y: y0 + 10}, rect.label));
      }
    }
  }
  if (data.diagonal) {
    const low = Math.max(Math.min(...x.domain), Math.min(...y.domain));
    const high = Math.min(Math.max(...x.domain), Math.max(...y.domain));
    if (high > low) {
      back.append(
        svg('line', {class: 'chart-rule', x1: x(low), y1: y(low), x2: x(high), y2: y(high)})
      );
    }
  }
  if (data.quadrants) {
    const quadrants = data.quadrants;
    const px = x(quadrants.x);
    const py = y(quadrants.y);
    if (Number.isFinite(px) && px >= frame.left && px <= frame.right) {
      back.append(
        svg('line', {class: 'chart-rule', x1: px, x2: px, y1: frame.top, y2: frame.bottom})
      );
    }
    if (Number.isFinite(py) && py >= frame.top && py <= frame.bottom) {
      back.append(
        svg('line', {class: 'chart-rule', x1: frame.left, x2: frame.right, y1: py, y2: py})
      );
    }
    if (quadrants.labels && !context.compact) {
      // Order: top right, top left, bottom left, bottom right (high-high, low-high, low-low, high-low).
      const [topRight, topLeft, bottomLeft, bottomRight] = quadrants.labels;
      const place = (text: string, cx: number, cy: number, anchor: 'start' | 'end') =>
        back.append(
          svg('text', {class: 'chart-quadrant', x: cx, y: cy, 'text-anchor': anchor}, text)
        );
      place(topRight, frame.right - 3, frame.top + 10, 'end');
      place(topLeft, frame.left + 3, frame.top + 10, 'start');
      place(bottomLeft, frame.left + 3, frame.bottom - 4, 'start');
      place(bottomRight, frame.right - 3, frame.bottom - 4, 'end');
    }
  }

  // Points, as circles (with hover titles) or one dot path per color.
  const pointX = new Float32Array(count);
  const pointY = new Float32Array(count);
  for (let index = 0; index < count; index++) {
    pointX[index] = x(data.x[index]);
    pointY[index] = y(data.y[index]);
  }
  const radius = data.radius ?? (count > 800 ? 1.6 : 2);
  const opacity = data.opacity ?? (count > 2000 ? 0.45 : 0.7);
  const colorOf = (index: number) => {
    const classIndex = data.colorIndex?.[index];
    if (classIndex === undefined) return slot(0);
    const entry = data.palette?.[classIndex];
    return entry ? colorToCss(entry) : slot(classIndex);
  };
  const drawPoints = (layer: SVGGElement, indices: Iterable<number>) => {
    const paths = new Map<string, string[]>();
    for (const index of indices) {
      const px = pointX[index];
      const py = pointY[index];
      if (!Number.isFinite(px) || !Number.isFinite(py)) continue;
      const color = colorOf(index);
      if (count <= CIRCLE_LIMIT) {
        layer.append(
          svg(
            'circle',
            {
              class: 'chart-dot',
              cx: px.toFixed(1),
              cy: py.toFixed(1),
              r: radius,
              style: `fill:${color}`
            },
            count <= 300
              ? svg('title', {}, `${frame.formatX(data.x[index])}, ${frame.formatY(data.y[index])}`)
              : null
          )
        );
      } else {
        const list = paths.get(color) ?? [];
        list.push(`M${px.toFixed(1)} ${py.toFixed(1)}h.01`);
        paths.set(color, list);
      }
    }
    for (const [color, list] of paths) {
      layer.append(
        svg('path', {
          class: 'chart-dots',
          d: list.join(''),
          style: `stroke:${color};stroke-width:${radius * 2}`
        })
      );
    }
  };
  const basePoints = svg('g', {class: 'chart-points', style: `--dot-opacity:${opacity}`});
  const selectedPoints = svg('g', {class: 'chart-points', style: '--dot-opacity:0.95'});
  main.append(basePoints, selectedPoints);
  drawPoints(
    basePoints,
    Array.from({length: count}, (_, index) => index)
  );

  for (const index of data.ringed ?? []) {
    if (!Number.isFinite(pointX[index]) || !Number.isFinite(pointY[index])) continue;
    main.append(
      svg('circle', {
        class: 'chart-ring',
        cx: pointX[index].toFixed(1),
        cy: pointY[index].toFixed(1),
        r: radius + 2.6
      })
    );
  }

  if (data.fit) {
    const segment = clipLine(data.fit.slope, data.fit.intercept, x.domain, y.domain);
    if (segment) {
      front.append(
        svg('line', {
          class: 'chart-fit',
          x1: x(segment[0][0]),
          y1: y(segment[0][1]),
          x2: x(segment[1][0]),
          y2: y(segment[1][1])
        })
      );
      if (data.fit.label && !context.compact) {
        const endX = x(segment[1][0]);
        const endY = y(segment[1][1]);
        const flip = endX + 4 + estimateTextWidth(data.fit.label) > frame.right + 6;
        front.append(
          svg(
            'text',
            {
              class: 'chart-note',
              x: flip ? endX - 3 : endX + 3,
              y: Math.max(frame.top + 9, endY - 5),
              'text-anchor': flip ? 'end' : 'start'
            },
            data.fit.label
          )
        );
      }
    }
  }
  drawRules(frame, data);

  // Pointer: brush, point click, link (a brush owns the drag, so link input is off then).
  const hit = createHitArea(
    frame,
    brushing || !!data.onPointClick || !!context.options?.onLinkInput
  );
  const updateLink = attachLink(
    frame,
    data.link,
    brushing ? {...context.options, onLinkInput: undefined} : context.options,
    hit
  );
  if (data.onBrush || data.onPointClick) {
    attachPointer(frame, data, hit, {
      count,
      pointX,
      pointY,
      drawPoints,
      basePoints,
      selectedPoints
    });
  }

  const table = buildXYTable(
    [data.xLabel ?? 'x', data.yLabel ?? 'y'],
    [Array.from(data.x).slice(0, count), Array.from(data.y).slice(0, count)],
    [frame.formatX, frame.formatY]
  );
  return {svg: frame.root, table, updateLink};
}

type PointerState = {
  count: number;
  pointX: Float32Array;
  pointY: Float32Array;
  drawPoints: (layer: SVGGElement, indices: Iterable<number>) => void;
  basePoints: SVGGElement;
  selectedPoints: SVGGElement;
};

function attachPointer(
  frame: Frame,
  data: ScatterChartData,
  hit: SVGElement,
  state: PointerState
): void {
  const {root} = frame;
  const {count, pointX, pointY} = state;
  const brush = svg('rect', {class: 'chart-brush', visibility: 'hidden'});
  frame.layers.overlay.append(brush);
  let start: {x: number; y: number} | null = null;
  let brushed = false;
  let selection: number[] = [];
  const clampX = (value: number) => Math.min(Math.max(value, frame.left), frame.right);
  const clampY = (value: number) => Math.min(Math.max(value, frame.top), frame.bottom);

  const clear = () => {
    const had = brushed;
    brushed = false;
    selection = [];
    brush.setAttribute('visibility', 'hidden');
    state.basePoints.classList.remove('is-dimmed');
    state.selectedPoints.replaceChildren();
    if (had) data.onBrush?.(null);
  };
  if (data.onBrush) {
    root.setAttribute('tabindex', '0');
    root.addEventListener('keydown', event => {
      if (event.key === 'Escape') clear();
    });
  }

  let dragged = false;
  hit.addEventListener('pointerdown', event => {
    const point = getViewPoint(root, event);
    if (!point) return;
    start = {x: clampX(point.x), y: clampY(point.y)};
    dragged = false;
    hit.setPointerCapture(event.pointerId);
    if (data.onBrush) root.focus({preventScroll: true});
  });
  hit.addEventListener('pointermove', event => {
    if (!start || !data.onBrush) return;
    const point = getViewPoint(root, event);
    if (!point) return;
    const x1 = clampX(point.x);
    const y1 = clampY(point.y);
    if (!dragged && Math.hypot(x1 - start.x, y1 - start.y) < DRAG_THRESHOLD) return;
    dragged = true;
    const left = Math.min(start.x, x1);
    const right = Math.max(start.x, x1);
    const top = Math.min(start.y, y1);
    const bottom = Math.max(start.y, y1);
    brush.setAttribute('x', String(left));
    brush.setAttribute('y', String(top));
    brush.setAttribute('width', String(right - left));
    brush.setAttribute('height', String(bottom - top));
    brush.setAttribute('visibility', 'visible');
    brushed = true;
    selection = [];
    for (let index = 0; index < count; index++) {
      if (
        pointX[index] >= left &&
        pointX[index] <= right &&
        pointY[index] >= top &&
        pointY[index] <= bottom
      ) {
        selection.push(index);
      }
    }
    state.basePoints.classList.add('is-dimmed');
    state.selectedPoints.replaceChildren();
    state.drawPoints(state.selectedPoints, selection);
  });
  hit.addEventListener('pointerup', event => {
    if (!start) return;
    start = null;
    if (dragged) {
      // Report once the drag ends.
      data.onBrush?.(selection);
      return;
    }
    // A click: it clears an existing brush and picks the nearest point.
    if (brushed) clear();
    const point = getViewPoint(root, event);
    if (data.onPointClick && point) {
      let best = -1;
      let bestDistance = 64;
      for (let index = 0; index < count; index++) {
        const distance = (pointX[index] - point.x) ** 2 + (pointY[index] - point.y) ** 2;
        if (distance < bestDistance) {
          best = index;
          bestDistance = distance;
        }
      }
      if (best >= 0) data.onPointClick(best);
    }
  });
  hit.addEventListener('pointercancel', () => {
    start = null;
  });
}
