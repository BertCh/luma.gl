// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Shared building blocks of the chart kit: SVG helpers, scales and ticks, number formatting, the
 * plot frame (axes, grid, bands, rules), the link marker and the chart wrapper (legend, table).
 */

import type {
  ChartBand,
  ChartColor,
  ChartCommon,
  ChartFormatter,
  ChartLink
} from '../../scenes/chart-types';
import {h} from '../dom';

/** Width of every chart's `viewBox`; the SVG scales to its container. */
export const CHART_WIDTH = 320;

/** Tick text size in viewBox units (about 10.5 px in a 340 px panel). */
export const FONT_TICK = 10;
/** Axis label and annotation text size in viewBox units. */
export const FONT_LABEL = 11;

const SVG_NAMESPACE = 'http://www.w3.org/2000/svg';
const MINUS = '−';

/** Attribute values accepted by {@link svg}; `undefined` skips the attribute. */
export type SvgAttributes = Record<string, string | number | undefined>;

/** Creates an SVG element with attributes and children (`null` children are skipped). */
export function svg<Tag extends keyof SVGElementTagNameMap>(
  tag: Tag,
  attributes: SvgAttributes = {},
  ...children: (Node | string | null | undefined)[]
): SVGElementTagNameMap[Tag] {
  const element = document.createElementNS(SVG_NAMESPACE, tag);
  for (const [name, value] of Object.entries(attributes)) {
    if (value !== undefined) element.setAttribute(name, String(value));
  }
  for (const child of children) if (child !== null && child !== undefined) element.append(child);
  return element;
}

/** CSS color of a palette slot `0`-`5` (`--chart-1` to `--chart-6`); wraps around. */
export const slot = (index: number) => `var(--chart-${(Math.max(0, Math.floor(index)) % 6) + 1})`;

/** CSS color of an RGBA palette entry with 0-255 channels. */
export function colorToCss(color: ChartColor): string {
  const alpha = color[3] === undefined ? 1 : color[3] / 255;
  return alpha >= 0.999
    ? `rgb(${color[0]},${color[1]},${color[2]})`
    : `rgba(${color[0]},${color[1]},${color[2]},${alpha.toFixed(3)})`;
}

/** Relative luminance (0-1) of an RGB color, for choosing a legible text color on a fill. */
export function getLuminance(color: ChartColor): number {
  const channel = (value: number) => {
    const unit = value / 255;
    return unit <= 0.03928 ? unit / 12.92 : ((unit + 0.055) / 1.055) ** 2.4;
  };
  return 0.2126 * channel(color[0]) + 0.7152 * channel(color[1]) + 0.0722 * channel(color[2]);
}

/**
 * Formats a number for axes and readouts: thousands separators, compact (`12.4K`, `3.1M`) from
 * 10,000, three significant digits below 1, exponent form below 0.001.
 */
export function formatChartNumber(value: number): string {
  if (!Number.isFinite(value)) return '';
  if (value === 0) return '0';
  const magnitude = Math.abs(value);
  const sign = value < 0 ? MINUS : '';
  let text: string;
  if (magnitude >= 1e9) text = `${trim(magnitude / 1e9, 2)}B`;
  else if (magnitude >= 1e6) text = `${trim(magnitude / 1e6, 2)}M`;
  else if (magnitude >= 1e4) text = `${trim(magnitude / 1e3, 1)}K`;
  else if (magnitude >= 1000) text = Math.round(magnitude).toLocaleString('en-US');
  else if (magnitude >= 100) text = String(Math.round(magnitude));
  else if (magnitude >= 10) text = trim(magnitude, 1);
  else if (magnitude >= 1) text = trim(magnitude, 2);
  else if (magnitude >= 0.001) text = trim(magnitude, 3);
  else text = magnitude.toExponential(1).replace('e-', 'e−');
  return sign + text;
}

function trim(value: number, digits: number): string {
  return String(Number(value.toFixed(digits)));
}

/** Formats a 0-1 share as a whole-number percentage (`42%`). */
export const formatPercent: ChartFormatter = value =>
  Number.isFinite(value) ? `${trim(value * 100, Math.abs(value) < 0.1 ? 1 : 0)}%` : '';

const SUPERSCRIPT: Record<string, string> = {
  '-': '⁻',
  '0': '⁰',
  '1': '¹',
  '2': '²',
  '3': '³',
  '4': '⁴',
  '5': '⁵',
  '6': '⁶',
  '7': '⁷',
  '8': '⁸',
  '9': '⁹'
};

/** Formats a power of ten for log axes: `1`, `10`, `100`, `1,000`, then `10⁵`, `10⁻³`. */
export function formatLogTick(value: number): string {
  if (!(value > 0)) return '';
  const exponent = Math.round(Math.log10(value));
  if (Math.abs(Math.log10(value) - exponent) < 1e-9 && (exponent > 4 || exponent < -2)) {
    return `10${String(exponent).replace(/./g, character => SUPERSCRIPT[character] ?? character)}`;
  }
  return formatChartNumber(value);
}

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

/** Expands `[min, max]` outward to tick boundaries so curves do not touch the plot edge. */
export function niceDomain(min: number, max: number, count = 4): [number, number] {
  const ticks = getNiceTicks(min, max, count);
  if (ticks.length < 2) return [min, max];
  const step = ticks[1] - ticks[0];
  return [
    Number((Math.floor(min / step + 1e-9) * step).toPrecision(12)),
    Number((Math.ceil(max / step - 1e-9) * step).toPrecision(12))
  ];
}

/** Decade ticks (1, 10, 100) of a log axis, with 2 and 5 added when fewer than three decades. */
export function getLogTicks(min: number, max: number): number[] {
  const first = Math.floor(Math.log10(min) + 1e-9);
  const last = Math.ceil(Math.log10(max) - 1e-9);
  const decades: number[] = [];
  for (let exponent = first; exponent <= last; exponent++) decades.push(10 ** exponent);
  let ticks = decades;
  if (decades.length < 4) {
    ticks = decades.flatMap(decade => [decade, decade * 2, decade * 5]);
  }
  return ticks
    .map(tick => Number(tick.toPrecision(12)))
    .filter(tick => tick >= min * (1 - 1e-9) && tick <= max * (1 + 1e-9));
}

/** Finite min and max of arrays plus extra values; `[0, 1]` when nothing is finite. */
export function extent(
  arrays: readonly ArrayLike<number>[],
  include: readonly number[] = [],
  positiveOnly = false
): [number, number] {
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  const visit = (value: number) => {
    if (!Number.isFinite(value) || (positiveOnly && value <= 0)) return;
    if (value < min) min = value;
    if (value > max) max = value;
  };
  for (const array of arrays)
    for (let index = 0; index < array.length; index++) visit(array[index]);
  for (const value of include) visit(value);
  if (min === Number.POSITIVE_INFINITY) return positiveOnly ? [1, 10] : [0, 1];
  if (min === max) return positiveOnly ? [min / 2, max * 2] : [min - 0.5, max + 0.5];
  return [min, max];
}

/** `[0, 1, ..., length - 1]`. */
export function indexArray(length: number): number[] {
  return Array.from({length}, (_, index) => index);
}

/** A mapping between a data domain and a pixel range, with its ticks. */
export type Scale = {
  (value: number): number;
  /** Data value at a pixel position (not clamped). */
  invert: (position: number) => number;
  domain: [number, number];
  range: [number, number];
  isLog: boolean;
  /** Tick values for about `count` ticks. */
  ticks: (count: number) => number[];
  /** Default tick formatter. */
  format: ChartFormatter;
};

/**
 * Creates a linear or log scale. A log scale needs a positive domain (a non-positive end is
 * replaced) and returns `NaN` for non-positive values so callers skip them.
 */
export function createScale(
  domain: readonly [number, number],
  range: readonly [number, number],
  type: 'linear' | 'log' = 'linear'
): Scale {
  if (type === 'log') {
    const high = domain[1] > 0 ? domain[1] : 10;
    const low = domain[0] > 0 && domain[0] < high ? domain[0] : high / 1000;
    const logLow = Math.log10(low);
    const span = Math.log10(high) - logLow || 1;
    const scale = ((value: number) =>
      value > 0
        ? range[0] + ((Math.log10(value) - logLow) / span) * (range[1] - range[0])
        : Number.NaN) as Scale;
    scale.invert = position =>
      10 ** (logLow + ((position - range[0]) / (range[1] - range[0])) * span);
    scale.domain = [low, high];
    scale.range = [range[0], range[1]];
    scale.isLog = true;
    scale.ticks = () => getLogTicks(low, high);
    scale.format = formatLogTick;
    return scale;
  }
  const span = domain[1] - domain[0] || 1;
  const scale = ((value: number) =>
    range[0] + ((value - domain[0]) / span) * (range[1] - range[0])) as Scale;
  scale.invert = position =>
    domain[0] + ((position - range[0]) / (range[1] - range[0] || 1)) * span;
  scale.domain = [domain[0], domain[1]];
  scale.range = [range[0], range[1]];
  scale.isLog = false;
  scale.ticks = count => getNiceTicks(domain[0], domain[1], count);
  scale.format = formatChartNumber;
  return scale;
}

/** Rough text width in viewBox units (tabular numerals, Source Sans proportions). */
export function estimateTextWidth(text: string, fontSize = FONT_TICK): number {
  return text.length * fontSize * 0.55;
}

/** Truncates text with an ellipsis to roughly `maxWidth` viewBox units. */
export function truncateText(text: string, maxWidth: number, fontSize = FONT_TICK): string {
  const maxCharacters = Math.max(3, Math.floor(maxWidth / (fontSize * 0.55)));
  return text.length > maxCharacters ? `${text.slice(0, maxCharacters - 1)}…` : text;
}

/**
 * Spreads label positions apart so none is closer than `gap` to the next, keeping them inside
 * `[min, max]` and as near as possible to where they were asked to be. Returns new positions in
 * the input order.
 */
export function nudgePositions(
  positions: readonly number[],
  min: number,
  max: number,
  gap: number
): number[] {
  const order = positions.map((_, index) => index).sort((a, b) => positions[a] - positions[b]);
  const result = positions.map(position => Math.min(Math.max(position, min), max));
  for (let i = 1; i < order.length; i++) {
    const previous = result[order[i - 1]];
    if (result[order[i]] < previous + gap) result[order[i]] = previous + gap;
  }
  const overflow = order.length ? result[order[order.length - 1]] - max : 0;
  if (overflow > 0) {
    result[order[order.length - 1]] -= overflow;
    for (let i = order.length - 2; i >= 0; i--) {
      const next = result[order[i + 1]];
      if (result[order[i]] > next - gap) result[order[i]] = next - gap;
    }
  }
  return result;
}

/** Greedy row assignment: items (sorted by `start`) go to the first row where they do not overlap. */
export function assignRows(
  items: readonly {start: number; end: number}[],
  gap = 4
): {rows: number[]; rowCount: number} {
  const rowEnds: number[] = [];
  const order = items.map((_, index) => index).sort((a, b) => items[a].start - items[b].start);
  const rows = new Array<number>(items.length).fill(0);
  for (const index of order) {
    let row = rowEnds.findIndex(end => end + gap <= items[index].start);
    if (row < 0) {
      row = rowEnds.length;
      rowEnds.push(0);
    }
    rowEnds[row] = items[index].end;
    rows[index] = row;
  }
  return {rows, rowCount: Math.max(1, rowEnds.length)};
}

/** Options for rendering, passed by the shell. */
export type ChartRenderOptions = {
  /** Current value of `data.link.option` (drawn as the linked marker). */
  linkValue?: number;
  /** Called while the reader clicks or drags on the chart with the value under the pointer (x or y per `link.axis`), unsnapped. */
  onLinkInput?: (value: number) => void;
};

/** What a chart kind builds before the shared wrapper adds legend and table. */
export type ChartBuild = {
  svg: SVGSVGElement;
  /** Legend entries (shown when two or more). */
  legend?: LegendItem[];
  /** Rows of the "Show values" table. */
  table?: TableSpec | null;
  /** Moves the linked marker. */
  updateLink?: (value: number) => void;
};

/** Context a kind builds in. Panels of a multiples chart use a smaller `width`. */
export type BuildContext = {
  /** `viewBox` width in units. */
  width: number;
  /** Small-multiples panel: fewer ticks, no axis labels, no legend or table. */
  compact: boolean;
  /** Replaces the chart's own title (a panel title). */
  title?: string;
  options?: ChartRenderOptions;
};

/** One legend entry. */
export type LegendItem = {
  label: string;
  /** Palette slot, or a CSS color. */
  color: number | string;
  shape?: 'line' | 'dashed' | 'band' | 'dot' | 'bar';
};

/** Rows of the "Show values" disclosure. */
export type TableSpec = {
  headers: readonly string[];
  rows: readonly (string | number)[][];
  /** Note under the table (for summarised data). */
  note?: string;
};

/** Picks about `maxRows` evenly spaced rows (keeping the first and last) from a long list. */
export function summariseRows<T>(rows: readonly T[], maxRows = 50): {rows: T[]; step: number} {
  if (rows.length <= maxRows) return {rows: [...rows], step: 1};
  const step = Math.ceil(rows.length / maxRows);
  const picked: T[] = [];
  for (let index = 0; index < rows.length; index += step) picked.push(rows[index]);
  if ((rows.length - 1) % step !== 0) picked.push(rows[rows.length - 1]);
  return {rows: picked, step};
}

/** Builds table rows from parallel arrays, summarised past 50 rows. */
export function buildXYTable(
  headers: readonly string[],
  columns: readonly ArrayLike<number>[],
  formats: readonly ChartFormatter[]
): TableSpec {
  const length = Math.max(0, ...columns.map(column => column.length));
  const all = Array.from({length}, (_, index) => index);
  const {rows, step} = summariseRows(all);
  return {
    headers,
    rows: rows.map(index =>
      columns.map((column, c) => (index < column.length ? formats[c](column[index]) : ''))
    ),
    note: step > 1 ? `${length} rows, every ${ordinal(step)} shown` : undefined
  };
}

function ordinal(step: number): string {
  if (step === 2) return 'second';
  if (step === 3) return 'third';
  return `${step}th`;
}

function renderTable(spec: TableSpec): HTMLElement {
  return h(
    'details',
    {class: 'chart-table'},
    h('summary', {}, 'Show values'),
    h(
      'div',
      {class: 'chart-table-scroll'},
      h(
        'table',
        {},
        h(
          'thead',
          {},
          h(
            'tr',
            {},
            spec.headers.map(header => h('th', {scope: 'col'}, header))
          )
        ),
        h(
          'tbody',
          {},
          spec.rows.map(row =>
            h(
              'tr',
              {},
              row.map(cell => h('td', {}, String(cell)))
            )
          )
        )
      ),
      spec.note ? h('p', {class: 'chart-table-note'}, spec.note) : null
    )
  );
}

function renderLegend(items: readonly LegendItem[]): HTMLElement | null {
  if (items.length < 2) return null;
  return h(
    'ul',
    {class: 'chart-legend'},
    items.map(item =>
      h(
        'li',
        {},
        h('span', {
          class: `chart-key is-${item.shape ?? 'line'}`,
          style: `--key:${typeof item.color === 'number' ? slot(item.color) : item.color}`
        }),
        item.label
      )
    )
  );
}

const linkUpdaters = new WeakMap<HTMLElement, (value: number) => void>();

/** Wraps a built SVG with the legend and the "Show values" table into the chart element. */
export function wrapChart(
  build: ChartBuild,
  kind: string,
  showTable: boolean | undefined
): HTMLElement {
  const element = h(
    'div',
    {class: 'chart-box', dataset: {chartKind: kind}},
    build.svg,
    build.legend ? renderLegend(build.legend) : null,
    showTable !== false && build.table ? renderTable(build.table) : null
  );
  if (build.updateLink) linkUpdaters.set(element, build.updateLink);
  return element;
}

/** Updates a rendered chart's linked marker without re-rendering. */
export function updateChartLink(element: HTMLElement, value: number): void {
  linkUpdaters.get(element)?.(value);
}

/** The pointer position in viewBox units, or `null` when the chart has no size yet. */
export function getViewPoint(
  root: SVGSVGElement,
  event: {clientX: number; clientY: number}
): {x: number; y: number} | null {
  const box = root.getBoundingClientRect();
  if (!box.width || !box.height) return null;
  const viewBox = root.viewBox.baseVal;
  return {
    x: ((event.clientX - box.left) / box.width) * viewBox.width,
    y: ((event.clientY - box.top) / box.height) * viewBox.height
  };
}

/** Creates the root `<svg>` with `viewBox`, accessible name, `<title>` and `<desc>`. */
export function createRoot(
  kind: string,
  width: number,
  height: number,
  title: string | undefined,
  description: string | undefined,
  fallbackName: string,
  interactive = false
): SVGSVGElement {
  const name = title ?? description ?? fallbackName;
  return svg(
    'svg',
    {
      class: `chart chart-${kind}`,
      viewBox: `0 0 ${width} ${height}`,
      role: interactive ? 'group' : 'img',
      'aria-label': name
    },
    svg('title', {}, name),
    description ? svg('desc', {}, description) : null
  );
}

/** A tick at a data value with its label. */
export type AxisTick = {value: number; label: string};

/** Configuration of {@link createFrame}. */
export type FrameConfig = {
  kind: string;
  width: number;
  height: number;
  compact?: boolean;
  common: Pick<
    ChartCommon,
    'title' | 'description' | 'xLabel' | 'yLabel' | 'formatX' | 'formatY' | 'bands'
  >;
  xDomain: readonly [number, number];
  yDomain: readonly [number, number];
  xScale?: 'linear' | 'log';
  yScale?: 'linear' | 'log';
  /** Explicit x ticks (category or break labels). */
  xTicks?: readonly AxisTick[];
  /** Explicit y ticks (row labels). */
  yTicks?: readonly AxisTick[];
  /** Secondary axis on the right. */
  y2Domain?: readonly [number, number];
  y2Label?: string;
  formatY2?: ChartFormatter;
  y2Color?: string;
  /** Gridlines: horizontal at y ticks (default), vertical at x ticks. */
  grid?: {x?: boolean; y?: boolean};
  /** Extra margin in viewBox units. */
  pad?: {left?: number; right?: number; top?: number; bottom?: number};
  /** Hide the x axis line and tick labels. */
  hideXAxis?: boolean;
  /** Group role (interactive chart) instead of `img`. */
  interactive?: boolean;
  /** Title override (a panel title). */
  title?: string;
};

/** A plot frame: axes, grid, bands and the layers content is drawn into. */
export type Frame = {
  root: SVGSVGElement;
  x: Scale;
  y: Scale;
  y2?: Scale;
  /** Plot rectangle edges in viewBox units. */
  left: number;
  right: number;
  top: number;
  bottom: number;
  width: number;
  height: number;
  formatX: ChartFormatter;
  formatY: ChartFormatter;
  /** Layers in paint order: bands and grid, content, rules and labels, interaction. */
  layers: {back: SVGGElement; main: SVGGElement; front: SVGGElement; overlay: SVGGElement};
};

/** Builds the frame of a cartesian chart: margins sized to the tick labels, axes, grid, bands. */
export function createFrame(config: FrameConfig): Frame {
  const {width, height, common} = config;
  const compact = config.compact ?? false;
  const formatX = common.formatX;
  const formatY = common.formatY;
  const title = config.title ?? common.title;
  const pad = config.pad ?? {};
  const xLabel = compact ? undefined : common.xLabel;
  const yLabel = compact ? undefined : common.yLabel;

  const top = 6 + (title ? 16 : 0) + (pad.top ?? 0);
  const bottom = height - (4 + (config.hideXAxis ? 0 : 14) + (xLabel ? 13 : 0) + (pad.bottom ?? 0));
  const tickCountY = Math.min(5, Math.max(2, Math.floor((bottom - top) / 28)));

  // Provisional y scale for tick labels (the left margin depends on their width).
  const provisionalY = createScale(config.yDomain, [bottom, top], config.yScale);
  const yTicks: AxisTick[] =
    config.yTicks?.map(tick => ({...tick})) ??
    provisionalY
      .ticks(compact ? Math.min(tickCountY, 3) : tickCountY)
      .map(value => ({value, label: (formatY ?? provisionalY.format)(value)}));
  const maxYLabel = Math.max(0, ...yTicks.map(tick => estimateTextWidth(tick.label)));
  const left = Math.max(18, 4 + (yLabel ? 13 : 0) + maxYLabel + 5 + (pad.left ?? 0));

  let right = width - 8 - (pad.right ?? 0);
  let y2: Scale | undefined;
  let y2Ticks: AxisTick[] = [];
  if (config.y2Domain) {
    const provisional = createScale(config.y2Domain, [bottom, top]);
    const format = config.formatY2 ?? provisional.format;
    y2Ticks = provisional.ticks(tickCountY).map(value => ({value, label: format(value)}));
    const y2Width = Math.max(0, ...y2Ticks.map(tick => estimateTextWidth(tick.label)));
    right -= y2Width + 5 + (config.y2Label ? 13 : 0);
    y2 = createScale(config.y2Domain, [bottom, top]);
  }

  const x = createScale(config.xDomain, [left, right], config.xScale);
  const y = createScale(config.yDomain, [bottom, top], config.yScale);

  const root = createRoot(
    config.kind,
    width,
    height,
    title ?? common.title,
    common.description,
    `${config.kind} chart`,
    config.interactive
  );
  const layers = {
    back: svg('g', {class: 'chart-layer-back'}),
    main: svg('g', {class: 'chart-layer-main'}),
    front: svg('g', {class: 'chart-layer-front'}),
    overlay: svg('g', {class: 'chart-layer-overlay'})
  };
  const frame: Frame = {
    root,
    x,
    y,
    y2,
    left,
    right,
    top,
    bottom,
    width,
    height,
    formatX: formatX ?? x.format,
    formatY: formatY ?? y.format,
    layers
  };

  if (title) {
    root.append(svg('text', {class: 'chart-title', x: 2, y: 13}, title));
  }
  drawBands(frame, common.bands);

  // Horizontal grid and y tick labels.
  const gridY = config.grid?.y ?? true;
  for (const tick of yTicks) {
    const position = y(tick.value);
    if (!Number.isFinite(position) || position < top - 0.5 || position > bottom + 0.5) continue;
    if (gridY) {
      layers.back.append(
        svg('line', {class: 'chart-grid', x1: left, x2: right, y1: position, y2: position})
      );
    }
    root.append(
      svg(
        'text',
        {class: 'chart-tick', x: left - 5, y: position + 3.5, 'text-anchor': 'end'},
        tick.label
      )
    );
  }
  for (const tick of y2Ticks) {
    const position = y2?.(tick.value) ?? Number.NaN;
    if (!Number.isFinite(position)) continue;
    root.append(
      svg(
        'text',
        {
          class: 'chart-tick chart-tick-y2',
          x: right + 5,
          y: position + 3.5,
          style: config.y2Color ? `fill:${config.y2Color}` : undefined
        },
        tick.label
      )
    );
  }

  // X axis, ticks and labels.
  if (!config.hideXAxis) {
    layers.back.append(
      svg('line', {class: 'chart-axis', x1: left, x2: right, y1: bottom, y2: bottom})
    );
    const rawTicks: AxisTick[] =
      config.xTicks?.map(tick => ({...tick})) ??
      x
        .ticks(Math.min(6, Math.max(2, Math.floor((right - left) / (compact ? 40 : 52)))))
        .map(value => ({value, label: (formatX ?? x.format)(value)}));
    const ticks = thinTicks(rawTicks, x, 5);
    const gridX = config.grid?.x ?? false;
    for (const tick of ticks) {
      const position = x(tick.value);
      if (!Number.isFinite(position) || position < left - 0.5 || position > right + 0.5) continue;
      if (gridX) {
        layers.back.append(
          svg('line', {class: 'chart-grid', x1: position, x2: position, y1: top, y2: bottom})
        );
      }
      layers.back.append(
        svg('line', {
          class: 'chart-tick-mark',
          x1: position,
          x2: position,
          y1: bottom,
          y2: bottom + 3
        })
      );
      const halfWidth = estimateTextWidth(tick.label) / 2;
      const anchor =
        position - halfWidth < 1 ? 'start' : position + halfWidth > width - 1 ? 'end' : 'middle';
      root.append(
        svg(
          'text',
          {
            class: 'chart-tick',
            x: anchor === 'start' ? Math.max(1, position - 1) : position,
            y: bottom + 13,
            'text-anchor': anchor
          },
          tick.label
        )
      );
    }
  }
  if (xLabel) {
    root.append(
      svg(
        'text',
        {class: 'chart-label', x: (left + right) / 2, y: height - 3, 'text-anchor': 'middle'},
        xLabel
      )
    );
  }
  if (yLabel) {
    root.append(
      svg(
        'text',
        {
          class: 'chart-label',
          transform: `translate(10 ${(top + bottom) / 2}) rotate(-90)`,
          'text-anchor': 'middle'
        },
        yLabel
      )
    );
  }
  if (config.y2Label && !compact) {
    root.append(
      svg(
        'text',
        {
          class: 'chart-label',
          transform: `translate(${width - 4} ${(top + bottom) / 2}) rotate(90)`,
          'text-anchor': 'middle',
          style: config.y2Color ? `fill:${config.y2Color}` : undefined
        },
        config.y2Label
      )
    );
  }
  root.append(layers.back, layers.main, layers.front, layers.overlay);
  return frame;
}

/** Drops ticks so neighbouring labels keep `gap` units apart (first and last preferred). */
function thinTicks(ticks: readonly AxisTick[], scale: Scale, gap: number): AxisTick[] {
  const visible = ticks.filter(tick => Number.isFinite(scale(tick.value)));
  let stride = 1;
  const fits = (candidate: number) => {
    let previousEnd = Number.NEGATIVE_INFINITY;
    for (let index = 0; index < visible.length; index += candidate) {
      const tick = visible[index];
      const center = scale(tick.value);
      const half = estimateTextWidth(tick.label) / 2;
      if (center - half < previousEnd + gap) return false;
      previousEnd = center + half;
    }
    return true;
  };
  while (stride < visible.length && !fits(stride)) stride++;
  return visible.filter((_, index) => index % stride === 0);
}

function getBandStyle(tone: ChartBand['tone']): string {
  if (typeof tone === 'number') return `--band:${slot(tone)};--band-opacity:0.14`;
  if (tone === 'signal') return '--band:var(--map-signal, #D95F0E);--band-opacity:0.14';
  return '--band:var(--muted);--band-opacity:0.12';
}

function drawBands(frame: Frame, bands: readonly ChartBand[] | undefined): void {
  for (const band of bands ?? []) {
    const alongY = band.axis === 'y';
    const scale = alongY ? frame.y : frame.x;
    const a = scale(band.from);
    const b = scale(band.to);
    if (!Number.isFinite(a) || !Number.isFinite(b)) continue;
    const low = Math.min(a, b);
    const high = Math.max(a, b);
    const [x0, x1, y0, y1] = alongY
      ? [frame.left, frame.right, Math.max(frame.top, low), Math.min(frame.bottom, high)]
      : [Math.max(frame.left, low), Math.min(frame.right, high), frame.top, frame.bottom];
    if (x1 <= x0 || y1 <= y0) continue;
    frame.layers.back.append(
      svg('rect', {
        class: 'chart-band',
        x: x0,
        y: y0,
        width: x1 - x0,
        height: y1 - y0,
        style: getBandStyle(band.tone)
      })
    );
    if (band.label) {
      const available = alongY ? frame.right - frame.left : x1 - x0;
      frame.layers.back.append(
        svg(
          'text',
          alongY
            ? {class: 'chart-band-label', x: frame.right - 3, y: y0 + 10, 'text-anchor': 'end'}
            : {class: 'chart-band-label', x: x0 + 3, y: frame.top + 10},
          truncateText(band.label, available - 4)
        )
      );
    }
  }
}

/**
 * Draws `markers` (vertical, signal color: the observed statistic) and `guides` (horizontal,
 * dashed ink: the expectation or a threshold) with labels that avoid each other.
 */
export function drawRules(
  frame: Frame,
  data: Pick<ChartCommon, 'markers' | 'guides'>,
  formatLabelX?: ChartFormatter
): void {
  const front = frame.layers.front;
  for (const guide of data.guides ?? []) {
    const position = frame.y(guide.y);
    if (!Number.isFinite(position) || position < frame.top - 1 || position > frame.bottom + 1)
      continue;
    front.append(
      svg('line', {
        class: 'chart-rule',
        x1: frame.left,
        x2: frame.right,
        y1: position,
        y2: position
      })
    );
    if (guide.label) {
      front.append(
        svg(
          'text',
          {class: 'chart-note', x: frame.right - 2, y: position - 3.5, 'text-anchor': 'end'},
          guide.label
        )
      );
    }
  }
  const markers = (data.markers ?? [])
    .map(marker => ({marker, position: frame.x(marker.x)}))
    .filter(
      ({position}) =>
        Number.isFinite(position) && position >= frame.left - 1 && position <= frame.right + 1
    );
  const middle = (frame.left + frame.right) / 2;
  const labelled = markers.filter(({marker}) => marker.label || formatLabelX);
  const placement = assignRows(
    labelled.map(({marker, position}) => {
      const text = marker.label ?? formatLabelX?.(marker.x) ?? '';
      const width = estimateTextWidth(text, FONT_TICK) + 4;
      return position > middle
        ? {start: position - 3 - width, end: position}
        : {start: position, end: position + 3 + width};
    })
  );
  for (const {marker, position} of markers) {
    front.append(
      svg('line', {
        class: 'chart-rule chart-marker',
        x1: position,
        x2: position,
        y1: frame.top,
        y2: frame.bottom
      })
    );
    const index = labelled.findIndex(item => item.marker === marker);
    if (index < 0) continue;
    const flip = position > middle;
    front.append(
      svg(
        'text',
        {
          class: 'chart-note chart-note-signal',
          x: position + (flip ? -3 : 3),
          y: frame.top + 9 + placement.rows[index] * 11,
          'text-anchor': flip ? 'end' : 'start'
        },
        marker.label ?? formatLabelX?.(marker.x) ?? ''
      )
    );
  }
}

/** An overlay rectangle over the plot that catches pointer events. */
export function createHitArea(frame: Frame, input: boolean): SVGRectElement {
  const rect = svg('rect', {
    class: `chart-hit${input ? ' is-input' : ''}`,
    x: frame.left,
    y: frame.top,
    width: Math.max(0, frame.right - frame.left),
    height: Math.max(0, frame.bottom - frame.top)
  });
  frame.layers.overlay.append(rect);
  return rect;
}

/**
 * Calls `callback` with the pointer's viewBox position on press and while dragging (touch and
 * mouse), capturing the pointer so a drag may leave the plot.
 */
export function onPointerDrag(
  target: SVGElement,
  root: SVGSVGElement,
  callback: (point: {x: number; y: number}, phase: 'start' | 'move' | 'end') => void
): void {
  let active = false;
  target.addEventListener('pointerdown', event => {
    if (event.button !== 0 && event.pointerType === 'mouse') return;
    const point = getViewPoint(root, event);
    if (!point) return;
    active = true;
    target.setPointerCapture(event.pointerId);
    callback(point, 'start');
  });
  target.addEventListener('pointermove', event => {
    if (!active) return;
    const point = getViewPoint(root, event);
    if (point) callback(point, 'move');
  });
  const finish = (event: PointerEvent) => {
    if (!active) return;
    active = false;
    const point = getViewPoint(root, event);
    if (point) callback(point, 'end');
  };
  target.addEventListener('pointerup', finish);
  target.addEventListener('pointercancel', finish);
}

/**
 * Draws the linked marker (a rule with a handle and a label) and wires click and drag on the
 * plot to `onLinkInput`. Returns the function that moves the marker.
 */
export function attachLink(
  frame: Frame,
  link: ChartLink | undefined,
  options: ChartRenderOptions | undefined,
  hit: SVGElement
): ((value: number) => void) | undefined {
  if (!link) return undefined;
  const alongY = link.axis === 'y';
  const scale = alongY ? frame.y : frame.x;
  const format = alongY ? frame.formatY : frame.formatX;
  const line = svg('line', {class: 'chart-rule chart-marker chart-link', visibility: 'hidden'});
  const handle = svg('circle', {class: 'chart-link-handle', r: 3.5, visibility: 'hidden'});
  const label = svg('text', {class: 'chart-note chart-note-signal chart-link-label'});
  frame.layers.front.append(line, handle, label);

  const update = (value: number) => {
    const position = scale(value);
    const inside =
      Number.isFinite(position) &&
      (alongY
        ? position >= frame.top - 0.5 && position <= frame.bottom + 0.5
        : position >= frame.left - 0.5 && position <= frame.right + 0.5);
    for (const element of [line, handle, label]) {
      element.setAttribute('visibility', inside ? 'visible' : 'hidden');
    }
    if (!inside) return;
    const text = link.label?.(value) ?? format(value);
    label.textContent = text;
    if (alongY) {
      line.setAttribute('x1', String(frame.left));
      line.setAttribute('x2', String(frame.right));
      line.setAttribute('y1', String(position));
      line.setAttribute('y2', String(position));
      handle.setAttribute('cx', String(frame.left));
      handle.setAttribute('cy', String(position));
      label.setAttribute('x', String(frame.right - 2));
      label.setAttribute('y', String(position - 4));
      label.setAttribute('text-anchor', 'end');
    } else {
      line.setAttribute('x1', String(position));
      line.setAttribute('x2', String(position));
      line.setAttribute('y1', String(frame.top));
      line.setAttribute('y2', String(frame.bottom));
      handle.setAttribute('cx', String(position));
      handle.setAttribute('cy', String(frame.bottom));
      const flip = position + 6 + estimateTextWidth(text) > frame.right + 4;
      label.setAttribute('x', String(position + (flip ? -4 : 4)));
      label.setAttribute('y', String(frame.top + 9));
      label.setAttribute('text-anchor', flip ? 'end' : 'start');
    }
  };
  if (options?.linkValue !== undefined) update(options.linkValue);

  if (options?.onLinkInput) {
    const onInput = options.onLinkInput;
    hit.classList.add('is-input');
    onPointerDrag(hit, frame.root, point => {
      const raw = scale.invert(alongY ? point.y : point.x);
      const [low, high] = scale.domain;
      const value = Math.min(Math.max(raw, Math.min(low, high)), Math.max(low, high));
      if (!Number.isFinite(value)) return;
      update(value);
      onInput(value);
    });
  }
  return update;
}

/** Adds a legend entry list for series, deduplicated by label. */
export function uniqueLegend(items: readonly LegendItem[]): LegendItem[] {
  const seen = new Set<string>();
  return items.filter(item => {
    if (seen.has(item.label)) return false;
    seen.add(item.label);
    return true;
  });
}
