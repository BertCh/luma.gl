// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  BarChartData,
  ChartData,
  ChartFormatter,
  LineChartData,
  SparklineData
} from '../scenes/chart-types';
import {h} from './dom';

const SVG_NAMESPACE = 'http://www.w3.org/2000/svg';
const WIDTH = 320;
const MARGIN = {left: 38, right: 10, top: 8, bottom: 20};

type Attributes = Record<string, string | number>;

function svg<Tag extends keyof SVGElementTagNameMap>(
  tag: Tag,
  attributes: Attributes = {},
  ...children: (Node | string | null)[]
): SVGElementTagNameMap[Tag] {
  const element = document.createElementNS(SVG_NAMESPACE, tag);
  for (const [name, value] of Object.entries(attributes)) element.setAttribute(name, String(value));
  for (const child of children) if (child !== null) element.append(child);
  return element;
}

const slot = (index: number) => `var(--chart-${(Math.max(0, Math.floor(index)) % 6) + 1})`;

const defaultFormat: ChartFormatter = value => {
  if (!Number.isFinite(value)) return '';
  const magnitude = Math.abs(value);
  if (magnitude !== 0 && (magnitude >= 10000 || magnitude < 0.01)) return value.toPrecision(2);
  return Number(value.toFixed(magnitude >= 100 ? 0 : magnitude >= 10 ? 1 : 2)).toString();
};

/** "Nice" tick values covering `[min, max]`, about `count` of them. */
export function getNiceTicks(min: number, max: number, count = 4): number[] {
  if (!(max > min)) return [min];
  const rough = (max - min) / count;
  const power = 10 ** Math.floor(Math.log10(rough));
  const fraction = rough / power;
  const step = (fraction < 1.5 ? 1 : fraction < 3 ? 2 : fraction < 7 ? 5 : 10) * power;
  const ticks: number[] = [];
  for (let tick = Math.ceil(min / step) * step; tick <= max + step * 1e-9; tick += step) {
    ticks.push(Number(tick.toPrecision(12)));
  }
  return ticks;
}

function extent(
  arrays: readonly ArrayLike<number>[],
  include: readonly number[] = []
): [number, number] {
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  for (const array of arrays) {
    for (let index = 0; index < array.length; index++) {
      const value = array[index];
      if (!Number.isFinite(value)) continue;
      if (value < min) min = value;
      if (value > max) max = value;
    }
  }
  for (const value of include) {
    if (!Number.isFinite(value)) continue;
    if (value < min) min = value;
    if (value > max) max = value;
  }
  if (min === Number.POSITIVE_INFINITY) return [0, 1];
  if (min === max) return [min - 0.5, max + 0.5];
  return [min, max];
}

function scale(domain: readonly [number, number], range: readonly [number, number]) {
  const span = domain[1] - domain[0] || 1;
  return (value: number) => range[0] + ((value - domain[0]) / span) * (range[1] - range[0]);
}

/**
 * Renders a chart as inline SVG. Colors come from the `--chart-1`..`--chart-6` and `--muted`
 * CSS tokens, so charts follow the light and dark theme without being redrawn.
 */
export function renderChart(data: ChartData): HTMLElement {
  if (data.kind === 'sparkline') return renderSparkline(data);
  if (data.kind === 'line') return renderLineChart(data);
  return renderBarChart(data);
}

function renderSparkline(data: SparklineData): HTMLElement {
  const height = data.height ?? 36;
  const [min, max] = extent([data.values]);
  const x = scale([0, Math.max(1, data.values.length - 1)], [3, WIDTH - 3]);
  const y = scale([min, max], [height - 4, 4]);
  const points: string[] = [];
  for (let index = 0; index < data.values.length; index++) {
    if (Number.isFinite(data.values[index])) {
      points.push(`${x(index).toFixed(1)},${y(data.values[index]).toFixed(1)}`);
    }
  }
  const color = slot(data.color ?? 0);
  const marker =
    data.highlight !== undefined && Number.isFinite(data.values[data.highlight])
      ? svg('circle', {
          cx: x(data.highlight),
          cy: y(data.values[data.highlight]),
          r: 3,
          fill: color
        })
      : null;
  const root = svg(
    'svg',
    {
      class: 'chart chart-sparkline',
      viewBox: `0 0 ${WIDTH} ${height}`,
      role: 'img',
      'aria-label': data.description ?? 'Sparkline'
    },
    svg('polyline', {
      points: points.join(' '),
      fill: 'none',
      stroke: color,
      'stroke-width': 1.75,
      'stroke-linejoin': 'round',
      'stroke-linecap': 'round'
    }),
    marker
  );
  return h('div', {class: 'chart-box'}, root);
}

function createFrame(
  height: number,
  description: string | undefined,
  xDomain: readonly [number, number],
  yDomain: readonly [number, number],
  common: {
    xLabel?: string;
    yLabel?: string;
    formatX?: ChartFormatter;
    formatY?: ChartFormatter;
    xTickLabels?: readonly {position: number; label: string}[];
  }
) {
  const left = MARGIN.left + (common.yLabel ? 10 : 0);
  const bottom = MARGIN.bottom + (common.xLabel ? 12 : 0);
  const plotRight = WIDTH - MARGIN.right;
  const plotBottom = height - bottom;
  const x = scale(xDomain, [left, plotRight]);
  const y = scale(yDomain, [plotBottom, MARGIN.top]);
  const formatX = common.formatX ?? defaultFormat;
  const formatY = common.formatY ?? defaultFormat;
  const root = svg('svg', {
    class: 'chart',
    viewBox: `0 0 ${WIDTH} ${height}`,
    role: 'img',
    'aria-label': description ?? 'Chart'
  });
  for (const tick of getNiceTicks(yDomain[0], yDomain[1], 4)) {
    const position = y(tick);
    root.append(
      svg('line', {class: 'chart-grid', x1: left, x2: plotRight, y1: position, y2: position}),
      svg(
        'text',
        {class: 'chart-tick', x: left - 4, y: position + 3, 'text-anchor': 'end'},
        formatY(tick)
      )
    );
  }
  if (common.xTickLabels) {
    for (const {position, label} of common.xTickLabels) {
      root.append(
        svg(
          'text',
          {class: 'chart-tick', x: position, y: plotBottom + 12, 'text-anchor': 'middle'},
          label
        )
      );
    }
  } else {
    for (const tick of getNiceTicks(xDomain[0], xDomain[1], 4)) {
      root.append(
        svg(
          'text',
          {class: 'chart-tick', x: x(tick), y: plotBottom + 12, 'text-anchor': 'middle'},
          formatX(tick)
        )
      );
    }
  }
  root.append(
    svg('line', {class: 'chart-axis', x1: left, x2: plotRight, y1: plotBottom, y2: plotBottom})
  );
  if (common.xLabel) {
    root.append(
      svg(
        'text',
        {class: 'chart-label', x: (left + plotRight) / 2, y: height - 2, 'text-anchor': 'middle'},
        common.xLabel
      )
    );
  }
  if (common.yLabel) {
    root.append(
      svg(
        'text',
        {
          class: 'chart-label',
          transform: `translate(8 ${(MARGIN.top + plotBottom) / 2}) rotate(-90)`,
          'text-anchor': 'middle'
        },
        common.yLabel
      )
    );
  }
  return {root, x, y, left, plotRight, plotBottom, formatX, formatY};
}

function addRules(
  frame: ReturnType<typeof createFrame>,
  data: {
    markers?: readonly {x: number; label?: string}[];
    guides?: readonly {y: number; label?: string}[];
  }
): void {
  for (const guide of data.guides ?? []) {
    const position = frame.y(guide.y);
    if (position < MARGIN.top - 1 || position > frame.plotBottom + 1) continue;
    frame.root.append(
      svg('line', {
        class: 'chart-rule',
        x1: frame.left,
        x2: frame.plotRight,
        y1: position,
        y2: position
      })
    );
    if (guide.label) {
      frame.root.append(
        svg(
          'text',
          {class: 'chart-note', x: frame.plotRight - 2, y: position - 3, 'text-anchor': 'end'},
          guide.label
        )
      );
    }
  }
  for (const marker of data.markers ?? []) {
    const position = frame.x(marker.x);
    if (position < frame.left - 1 || position > frame.plotRight + 1) continue;
    frame.root.append(
      svg('line', {
        class: 'chart-rule chart-marker',
        x1: position,
        x2: position,
        y1: MARGIN.top,
        y2: frame.plotBottom
      })
    );
    if (marker.label) {
      const flip = position > (frame.left + frame.plotRight) / 2;
      frame.root.append(
        svg(
          'text',
          {
            class: 'chart-note',
            x: position + (flip ? -3 : 3),
            y: MARGIN.top + 8,
            'text-anchor': flip ? 'end' : 'start'
          },
          marker.label
        )
      );
    }
  }
}

function renderLineChart(data: LineChartData): HTMLElement {
  const height = data.height ?? 140;
  const xArrays: ArrayLike<number>[] = [];
  const yArrays: ArrayLike<number>[] = [];
  for (const series of data.series) {
    xArrays.push(series.x ?? indexArray(series.y.length));
    yArrays.push(series.y);
  }
  if (data.band) {
    xArrays.push(data.band.x ?? indexArray(data.band.low.length));
    yArrays.push(data.band.low, data.band.high);
  }
  const xDomain =
    data.xDomain ??
    extent(
      xArrays,
      (data.markers ?? []).map(marker => marker.x)
    );
  const yDomain =
    data.yDomain ??
    extent(
      yArrays,
      (data.guides ?? []).map(guide => guide.y)
    );
  const frame = createFrame(height, data.description, xDomain, yDomain, data);
  const {root, x, y} = frame;

  if (data.band) {
    const bandX = data.band.x ?? indexArray(data.band.low.length);
    const upper: string[] = [];
    const lower: string[] = [];
    for (let index = 0; index < data.band.low.length; index++) {
      if (!Number.isFinite(data.band.low[index]) || !Number.isFinite(data.band.high[index]))
        continue;
      upper.push(`${x(bandX[index]).toFixed(1)},${y(data.band.high[index]).toFixed(1)}`);
      lower.unshift(`${x(bandX[index]).toFixed(1)},${y(data.band.low[index]).toFixed(1)}`);
    }
    root.append(svg('polygon', {class: 'chart-band', points: [...upper, ...lower].join(' ')}));
  }
  addRules(frame, data);

  data.series.forEach((series, seriesIndex) => {
    const seriesX = series.x ?? indexArray(series.y.length);
    const color = slot(series.color ?? seriesIndex);
    let path = '';
    let pen = false;
    for (let index = 0; index < series.y.length; index++) {
      if (!Number.isFinite(series.y[index]) || !Number.isFinite(seriesX[index])) {
        pen = false;
        continue;
      }
      path += `${pen ? 'L' : 'M'}${x(seriesX[index]).toFixed(1)} ${y(series.y[index]).toFixed(1)}`;
      pen = true;
    }
    if (series.area && path) {
      root.append(
        svg('path', {
          d: `${path}L${frame.plotRight} ${frame.plotBottom}L${frame.left} ${frame.plotBottom}Z`,
          fill: color,
          'fill-opacity': 0.14,
          stroke: 'none'
        })
      );
    }
    root.append(
      svg('path', {
        d: path,
        fill: 'none',
        stroke: color,
        'stroke-width': 1.75,
        'stroke-linejoin': 'round',
        'stroke-linecap': 'round',
        'stroke-dasharray': series.dashed ? '5 3' : 'none'
      })
    );
  });

  // Hover: snap to the nearest x of the first series and read out every series there.
  const cursor = svg('line', {
    class: 'chart-cursor',
    y1: MARGIN.top,
    y2: frame.plotBottom,
    visibility: 'hidden'
  });
  const readout = svg('text', {
    class: 'chart-note',
    x: frame.plotRight - 2,
    y: MARGIN.top + 8,
    'text-anchor': 'end'
  });
  root.append(cursor, readout);
  const first = data.series[0];
  if (first) {
    const firstX = first.x ?? indexArray(first.y.length);
    root.addEventListener('pointermove', event => {
      const box = root.getBoundingClientRect();
      if (!box.width) return;
      const viewX = ((event.clientX - box.left) / box.width) * WIDTH;
      const target =
        xDomain[0] +
        ((viewX - frame.left) / (frame.plotRight - frame.left)) * (xDomain[1] - xDomain[0]);
      let best = 0;
      for (let index = 1; index < firstX.length; index++) {
        if (Math.abs(firstX[index] - target) < Math.abs(firstX[best] - target)) best = index;
      }
      const position = x(firstX[best]);
      if (!Number.isFinite(position) || viewX < frame.left - 4 || viewX > frame.plotRight + 4) {
        cursor.setAttribute('visibility', 'hidden');
        readout.textContent = '';
        return;
      }
      cursor.setAttribute('x1', String(position));
      cursor.setAttribute('x2', String(position));
      cursor.setAttribute('visibility', 'visible');
      const parts = data.series.map(series => {
        const value = series.y[best];
        return `${series.label ? `${series.label} ` : ''}${frame.formatY(value)}`;
      });
      readout.textContent = `${frame.formatX(firstX[best])}: ${parts.join(', ')}`;
    });
    root.addEventListener('pointerleave', () => {
      cursor.setAttribute('visibility', 'hidden');
      readout.textContent = '';
    });
  }

  const legendItems = [
    ...data.series
      .map((series, index) => ({
        label: series.label,
        color: series.color ?? index,
        dashed: series.dashed,
        band: false
      }))
      .filter(item => item.label),
    ...(data.band?.label ? [{label: data.band.label, color: 0, dashed: false, band: true}] : [])
  ];
  return h('div', {class: 'chart-box'}, root, renderLegendList(legendItems));
}

function renderLegendList(
  items: readonly {label?: string; color: number; dashed?: boolean; band: boolean}[]
): HTMLElement | null {
  if (items.length < 2) return null;
  return h(
    'ul',
    {class: 'chart-legend'},
    items.map(item =>
      h(
        'li',
        {},
        h('span', {
          class: `chart-key${item.band ? ' is-band' : ''}${item.dashed ? ' is-dashed' : ''}`,
          style: `--key:${slot(item.color)}`
        }),
        item.label
      )
    )
  );
}

function renderBarChart(data: BarChartData): HTMLElement {
  const height = data.height ?? 140;
  const count = data.values.length;
  const histogram = data.kind === 'histogram' || data.xDomain !== undefined;
  const xDomain: readonly [number, number] = histogram ? (data.xDomain ?? [0, count]) : [0, count];
  const yMax = data.yDomain?.[1] ?? extent([data.values], [0])[1];
  const yDomain: readonly [number, number] = data.yDomain ?? [
    Math.min(0, extent([data.values])[0]),
    yMax
  ];
  const frame = createFrame(height, data.description, xDomain, yDomain, {
    ...data,
    xTickLabels:
      !histogram && data.labels
        ? chooseLabels(
            data.labels,
            count,
            scale([0, count], [MARGIN.left + (data.yLabel ? 10 : 0), WIDTH - MARGIN.right])
          )
        : undefined
  });
  const baseline = frame.y(Math.max(yDomain[0], 0));
  const highlight = new Set(data.highlight ?? []);
  const color = slot(data.color ?? 0);
  const barWidth = (frame.plotRight - frame.left) / Math.max(1, count);
  for (let index = 0; index < count; index++) {
    const value = data.values[index];
    if (!Number.isFinite(value)) continue;
    const top = frame.y(value);
    const left = frame.left + index * barWidth;
    const bar = svg(
      'rect',
      {
        class: 'chart-bar',
        x: (left + 0.5).toFixed(2),
        y: Math.min(top, baseline).toFixed(2),
        width: Math.max(0.5, barWidth - 1).toFixed(2),
        height: Math.abs(baseline - top).toFixed(2),
        fill: highlight.size === 0 || highlight.has(index) ? color : 'var(--chart-muted)'
      },
      svg('title', {}, `${labelOf(data, index, xDomain, count)}: ${frame.formatY(value)}`)
    );
    frame.root.append(bar);
  }
  addRules(frame, data);
  return h('div', {class: 'chart-box'}, frame.root);
}

function labelOf(
  data: BarChartData,
  index: number,
  xDomain: readonly [number, number],
  count: number
): string {
  if (data.labels?.[index] !== undefined) return data.labels[index];
  if (data.kind === 'histogram' || data.xDomain) {
    const width = (xDomain[1] - xDomain[0]) / count;
    const format = data.formatX ?? defaultFormat;
    return `${format(xDomain[0] + index * width)} to ${format(xDomain[0] + (index + 1) * width)}`;
  }
  return String(index);
}

/** Thins category labels so they do not overlap (about one per 40 view units). */
function chooseLabels(
  labels: readonly string[],
  count: number,
  position: (value: number) => number
): {position: number; label: string}[] {
  const stride = Math.max(1, Math.ceil(count / 8));
  const result: {position: number; label: string}[] = [];
  for (let index = 0; index < count; index += stride) {
    result.push({position: position(index + 0.5), label: labels[index] ?? ''});
  }
  return result;
}

function indexArray(length: number): number[] {
  return Array.from({length}, (_, index) => index);
}
