// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import './legend.css';
import {getClassColors, getRampCssGradient, sampleRamp} from '../engine/ramps';
import type {RampName} from '../engine/ramps';
import type {LegendColor, LegendSpec} from '../scenes/scene';
import {h} from './dom';

type GpuExtents = ReadonlyMap<string, readonly [number, number]>;
type Spec<Kind extends LegendSpec['kind']> = Extract<LegendSpec, {kind: Kind}>;
type Extent = readonly [number, number];

const SVG_NAMESPACE = 'http://www.w3.org/2000/svg';
const FALLBACK_COLOR: LegendColor = [128, 128, 128, 255];
/** Classed bars with more classes than this switch to the list layout unless asked for a bar. */
const MAX_BAR_CLASSES = 8;
/** Approximate width of one 11 px tabular character, as a fraction of the bar (about 6 px of 180). */
const CHARACTER_FRACTION = 0.034;
const MAX_UNIT_LABEL_LENGTH = 9;

const cssColor = (color: LegendColor) =>
  `rgb(${color[0]} ${color[1]} ${color[2]} / ${((color[3] ?? 255) / 255).toFixed(2)})`;

/** Options of {@link formatLegendNumber}. */
export type LegendNumberOptions = {
  /** Significant digits kept (default 4): `12345.6` reads `12,350`, `0.123456` reads `0.1235`. */
  maxSignificantDigits?: number;
  /** Group thousands with a thin space (U+2009) instead of a comma. */
  thinSpace?: boolean;
  /** Text appended after the number, for example a unit. */
  unit?: string;
};

/**
 * Formats a number for legends and tooltips: thousands separators, at most four significant digits
 * (configurable), a true minus sign and no trailing zeros. Non-finite values read `–`.
 */
export function formatLegendNumber(value: number, options: LegendNumberOptions = {}): string {
  if (!Number.isFinite(value)) return '–';
  const formatter = new Intl.NumberFormat('en-US', {
    maximumSignificantDigits: options.maxSignificantDigits ?? 4,
    useGrouping: true
  });
  let text = formatter.format(value).replace('-', '−');
  if (options.thinSpace) text = text.replace(/,/g, ' ');
  return options.unit ? `${text} ${options.unit}` : text;
}

/** Options for {@link renderLegends}. */
export type LegendRenderOptions = {
  /** Interactive legend selection changed (hover isolates temporarily, click locks, Esc clears). */
  onFilter?: (legendId: string, classes: readonly number[] | null) => void;
};

/** Locked selection of one interactive legend, kept across re-renders. */
type SelectionState = {locked: Set<number>; notified: string};

const selectionStates = new WeakMap<HTMLElement, Map<string, SelectionState>>();

type RenderContext = {
  gpuExtents: GpuExtents;
  options: LegendRenderOptions;
  states: Map<string, SelectionState>;
};

/**
 * Renders legends into `container`, replacing its children. Ramp gradients come from the same
 * `RAMP_STOPS` table as the WGSL shader, and `'gpu'` extents show what the scene last reported
 * through `ctx.setLegendExtent`. The locked selection of interactive legends is kept per legend id
 * across calls and dropped when the id disappears.
 */
export function renderLegends(
  container: HTMLElement,
  specs: readonly LegendSpec[],
  gpuExtents: GpuExtents,
  options: LegendRenderOptions = {}
): void {
  let states = selectionStates.get(container);
  if (!states) {
    states = new Map();
    selectionStates.set(container, states);
  }
  const liveIds = new Set(specs.filter(isInteractive).map(getLegendId));
  for (const id of states.keys()) if (!liveIds.has(id)) states.delete(id);
  const context: RenderContext = {gpuExtents, options, states};
  container.replaceChildren(...specs.map(spec => renderLegend(spec, context)));
}

/**
 * A tiny (40 x 8 px) swatch strip summarising a legend, for the collapsed phone pill. Returns null
 * for kinds with no meaningful colour strip (size).
 */
export function renderLegendThumbnail(spec: LegendSpec): HTMLElement | null {
  let background: string | null = null;
  let segments: readonly LegendColor[] | null = null;
  switch (spec.kind) {
    case 'ramp':
      background = getLegendGradient(spec);
      break;
    case 'classes':
      segments = resolveClasses(spec).colors;
      break;
    case 'bivariate': {
      const size = clampBivariateSize(spec.size);
      segments = Array.from(
        {length: size},
        (_, index) => spec.colors[index * size + index] ?? FALLBACK_COLOR
      );
      break;
    }
    case 'categories':
      segments = spec.entries.slice(0, 6).map(entry => entry.color);
      break;
    case 'matrix':
      segments = spec.colors.slice(0, Math.max(1, spec.columns.length));
      break;
    case 'cyclic':
      background = spec.ramp
        ? getRampCssGradient(spec.ramp)
        : spec.colors
          ? getColorsCssGradient(spec.colors)
          : null;
      break;
    case 'alpha': {
      const color = spec.colors[0] ?? FALLBACK_COLOR;
      segments = [0.25, 0.5, 0.75, 1].map(a => [color[0], color[1], color[2], 255 * a]);
      break;
    }
    case 'line':
      segments = spec.entries.slice(0, 6).map(entry => entry.color);
      break;
    case 'size':
      return null;
  }
  if (segments) {
    return h(
      'span',
      {class: 'legend-thumb', 'aria-hidden': 'true'},
      segments.map(color => h('i', {style: {backgroundColor: cssColor(color)}}))
    );
  }
  return background
    ? h('span', {class: 'legend-thumb', 'aria-hidden': 'true', style: {background}})
    : null;
}

function isInteractive(spec: LegendSpec): boolean {
  return (spec.kind === 'classes' || spec.kind === 'categories') && Boolean(spec.interactive);
}

function getLegendId(spec: LegendSpec): string {
  return spec.id ?? spec.title;
}

function renderLegend(spec: LegendSpec, context: RenderContext): HTMLElement {
  switch (spec.kind) {
    case 'ramp':
      return renderRampLegend(spec, context);
    case 'classes':
      return renderClassesLegend(spec, context);
    case 'bivariate':
      return renderBivariateLegend(spec);
    case 'categories':
      return renderCategoriesLegend(spec, context);
    case 'matrix':
      return renderMatrixLegend(spec);
    case 'cyclic':
      return renderCyclicLegend(spec);
    case 'alpha':
      return renderAlphaLegend(spec);
    case 'line':
      return renderLineLegend(spec);
    case 'size':
      return spec.layout === 'nested' ? renderNestedSizeLegend(spec) : renderRowSizeLegend(spec);
  }
}

/** The figure shell: group role, accessible summary, caption (with an optional extra control). */
function renderFigure(
  kind: string,
  spec: LegendSpec,
  summary: string,
  unit: string | undefined,
  extraClass: string,
  headerExtras: HTMLElement | null,
  ...children: (HTMLElement | SVGElement | null)[]
): HTMLElement {
  return h(
    'figure',
    {
      class: `legend legend-${kind}${extraClass ? ` ${extraClass}` : ''}`,
      role: 'group',
      'aria-label': `${spec.title}${unit ? ` (${unit})` : ''}${spec.basis ? ` ${spec.basis}` : ''}: ${summary}`
    },
    renderCaption(spec.title, unit, spec.basis, headerExtras),
    children
  );
}

/** `Name (unit) basis`: unit and basis regular weight and muted. */
function renderCaption(
  title: string,
  unit?: string,
  basis?: string,
  extras: HTMLElement | null = null
): HTMLElement {
  const detail = [unit ? `(${unit})` : '', basis ?? ''].filter(Boolean).join(' ');
  return h(
    'figcaption',
    {},
    h(
      'span',
      {class: 'legend-title'},
      title,
      detail ? h('span', {class: 'muted'}, ` ${detail}`) : null
    ),
    extras
  );
}

function renderNote(...notes: (string | undefined)[]): HTMLElement | null {
  const lines = notes.filter(Boolean) as string[];
  return lines.length
    ? h(
        'div',
        {class: 'legend-note'},
        lines.map(line => h('div', {}, line))
      )
    : null;
}

// --- selection (interactive legends) ---

/**
 * Makes `items` (one `button` per class, index = position) a hover / click / Escape filter. Hover
 * isolates one item, click locks it (Shift adds to the lock), Escape or "Show all" clears.
 */
function bindSelection(
  figure: HTMLElement,
  id: string,
  items: readonly HTMLElement[],
  showAll: HTMLElement,
  context: RenderContext
): void {
  let state = context.states.get(id);
  if (!state) {
    state = {locked: new Set(), notified: 'none'};
    context.states.set(id, state);
  }
  const selection = state;
  // Indices from a previous, longer legend would dim everything.
  for (const index of [...selection.locked])
    if (index >= items.length) selection.locked.delete(index);
  let hovered: number | null = null;

  const apply = () => {
    const effective: number[] | null =
      hovered !== null
        ? [hovered]
        : selection.locked.size
          ? [...selection.locked].sort((a, b) => a - b)
          : null;
    items.forEach((item, index) => {
      item.classList.toggle('is-dim', effective !== null && !effective.includes(index));
      item.setAttribute('aria-pressed', String(selection.locked.has(index)));
    });
    showAll.hidden = selection.locked.size === 0;
    const key = effective ? effective.join(',') : 'none';
    if (key !== selection.notified) {
      selection.notified = key;
      context.options.onFilter?.(id, effective);
    }
  };

  items.forEach((item, index) => {
    item.addEventListener('pointerenter', event => {
      if (event.pointerType === 'mouse') {
        hovered = index;
        apply();
      }
    });
    item.addEventListener('pointerleave', () => {
      if (hovered === index) {
        hovered = null;
        apply();
      }
    });
    item.addEventListener('focus', () => {
      if (item.matches(':focus-visible')) {
        hovered = index;
        apply();
      }
    });
    item.addEventListener('blur', () => {
      if (hovered === index) {
        hovered = null;
        apply();
      }
    });
    item.addEventListener('click', event => {
      hovered = null;
      if (event.shiftKey) {
        if (!selection.locked.delete(index)) selection.locked.add(index);
      } else if (selection.locked.size === 1 && selection.locked.has(index)) {
        selection.locked.clear();
      } else {
        selection.locked.clear();
        selection.locked.add(index);
      }
      apply();
    });
  });
  const clear = () => {
    hovered = null;
    selection.locked.clear();
    apply();
  };
  showAll.addEventListener('click', clear);
  figure.addEventListener('keydown', event => {
    if (event.key === 'Escape' && (selection.locked.size > 0 || hovered !== null)) {
      event.stopPropagation();
      clear();
    }
  });
  // Restore dimming of a selection kept across re-renders. Does not notify: the scene already has it.
  apply();
}

function createShowAllButton(): HTMLElement {
  return h('button', {class: 'legend-show-all', type: 'button', hidden: true}, 'Show all');
}

// --- ramp ---

/** The CSS gradient of a ramp legend, honouring custom colours, `range`, `reverse` and `sqrtScale`. */
function getLegendGradient(spec: Spec<'ramp'>): string {
  if (spec.colors) return getColorsCssGradient(spec.colors);
  if (!spec.range) {
    return getRampCssGradient(spec.ramp, {sqrtScale: spec.sqrtScale, reverse: spec.reverse});
  }
  return getTrimmedRampGradient(spec.ramp, spec.range, spec.sqrtScale, spec.reverse);
}

/** Reverse first, then sample at `t0 + t * (t1 - t0)`, exactly the layer's `rampRange`. */
function getTrimmedRampGradient(
  name: RampName,
  range: Extent,
  sqrtScale = false,
  reverse = false
): string {
  const samples = 24;
  const parts: string[] = [];
  for (let index = 0; index <= samples; index++) {
    const x = index / samples;
    const t = range[0] + (sqrtScale ? Math.sqrt(x) : x) * (range[1] - range[0]);
    const [r, g, b] = sampleRamp(name, t, reverse);
    parts.push(`rgb(${r} ${g} ${b}) ${(x * 100).toFixed(1)}%`);
  }
  return `linear-gradient(to right, ${parts.join(', ')})`;
}

function renderRampLegend(spec: Spec<'ramp'>, context: RenderContext): HTMLElement {
  const extent = spec.extent === 'gpu' ? context.gpuExtents.get(getLegendId(spec)) : spec.extent;
  const format = spec.format ?? formatLegendNumber;
  const isLoading = spec.extent === 'gpu' && !extent && !spec.labels;
  const hasSpan = extent !== undefined && extent[1] > extent[0];
  const isLog = spec.scale === 'log' && hasSpan && extent[0] > 0 && !spec.labels;

  const positionOf = (value: number): number =>
    !hasSpan
      ? 0
      : isLog
        ? (Math.log10(value) - Math.log10(extent[0])) /
          (Math.log10(extent[1]) - Math.log10(extent[0]))
        : (value - extent[0]) / (extent[1] - extent[0]);

  const low = spec.labels?.[0] ?? (extent ? format(extent[0]) : '');
  const high = spec.labels?.[1] ?? (extent ? format(extent[1]) : '');
  let ticks: HTMLElement | null = null;
  if (isLoading) {
    ticks = h(
      'div',
      {class: 'legend-ticks legend-ticks-loading', 'aria-hidden': 'true'},
      h('i'),
      h('i'),
      h('i')
    );
  } else if (isLog) {
    ticks = renderPositionedTicks(getLogTickCandidates(extent, low, high));
  } else if (low || high) {
    const candidates: TickCandidate[] = [
      {fraction: 0, label: low},
      {fraction: 1, label: high}
    ];
    if (hasSpan && spec.midpoint !== undefined) {
      candidates.push({
        fraction: positionOf(spec.midpoint),
        label: spec.midpointLabel ?? format(spec.midpoint),
        tall: true
      });
    }
    if (hasSpan) {
      for (const value of spec.ticks ?? []) {
        candidates.push({fraction: positionOf(value), label: format(value)});
      }
      if (!spec.labels && spec.midpoint === undefined && !spec.ticks?.length) {
        candidates.push({
          fraction: 0.5,
          label: format(extent[0] + (extent[1] - extent[0]) * 0.5)
        });
      }
    }
    ticks = renderPositionedTicks(candidates);
  }

  const markerFraction =
    spec.marker !== undefined && hasSpan && (!isLog || spec.marker > 0)
      ? positionOf(spec.marker)
      : null;
  const notes = [
    spec.note,
    isLoading ? 'Reading range from the GPU' : undefined,
    spec.sqrtScale && !spec.note ? 'Square-root scale' : undefined,
    spec.scale === 'log' && !spec.note ? 'Log scale' : undefined
  ];
  const summary = isLoading
    ? `${spec.ramp} colour ramp, range loading`
    : `${spec.ramp} colour ramp${spec.reverse ? ' reversed' : ''} from ${low || '?'} to ${high || '?'}`;
  return renderFigure(
    'ramp',
    spec,
    summary,
    spec.unit,
    isLoading ? 'is-loading' : '',
    null,
    spec.histogram ? renderHistogram(spec.histogram) : null,
    h(
      'div',
      {class: 'legend-bar-wrap'},
      markerFraction !== null ? renderMarker(markerFraction) : null,
      h('div', {
        class: 'legend-gradient',
        'aria-hidden': 'true',
        style: {background: getLegendGradient(spec)}
      })
    ),
    ticks,
    renderNote(...notes)
  );
}

/** The hovered feature's value: a small triangle above a bar at `fraction` of its width. */
function renderMarker(fraction: number): HTMLElement {
  return h('span', {
    class: 'legend-marker',
    'aria-hidden': 'true',
    style: {left: `${(Math.min(1, Math.max(0, fraction)) * 100).toFixed(2)}%`}
  });
}

/** Small bar chart of bin counts spanning the legend extent, no axes, optional break ticks. */
function renderHistogram(
  bins: readonly number[],
  breakFractions: readonly number[] = []
): HTMLElement {
  const maxCount = Math.max(1e-9, ...bins);
  return h(
    'div',
    {class: 'legend-histogram', 'aria-hidden': 'true'},
    bins.map(count =>
      h('span', {style: {height: `${Math.max(0, Math.min(1, count / maxCount)) * 100}%`}})
    ),
    breakFractions.map(fraction =>
      h('b', {style: {left: `${(Math.min(1, Math.max(0, fraction)) * 100).toFixed(2)}%`}})
    )
  );
}

type TickCandidate = {
  /** Position along the bar, 0 to 1. */
  fraction: number;
  label: string;
  /** Taller tick (the diverging midpoint). */
  tall?: boolean;
};

/**
 * Keeps candidate ticks in priority order, dropping any whose label would overlap one already kept,
 * and returns the survivors positioned along the bar. The ends align inward so labels never leave
 * the legend.
 */
function renderPositionedTicks(candidates: readonly TickCandidate[]): HTMLElement {
  const kept: {candidate: TickCandidate; start: number; end: number}[] = [];
  for (const candidate of candidates) {
    const {fraction} = candidate;
    if (!(fraction >= -1e-9 && fraction <= 1 + 1e-9) || !candidate.label) continue;
    const width = candidate.label.length * CHARACTER_FRACTION + 0.02;
    const alignment = getTickAlignment(fraction);
    const start =
      alignment === 'start'
        ? fraction
        : alignment === 'end'
          ? fraction - width
          : fraction - width / 2;
    const end = start + width;
    if (kept.every(other => end <= other.start || start >= other.end)) {
      kept.push({candidate, start, end});
    }
  }
  kept.sort((a, b) => a.candidate.fraction - b.candidate.fraction);
  return h(
    'div',
    {class: 'legend-ticks', 'aria-hidden': 'true'},
    kept.map(({candidate}) =>
      h(
        'span',
        {
          class: `legend-tick is-${getTickAlignment(candidate.fraction)}${candidate.tall ? ' is-tall' : ''}`,
          style: {left: `${(Math.min(1, Math.max(0, candidate.fraction)) * 100).toFixed(2)}%`}
        },
        candidate.label
      )
    )
  );
}

function getTickAlignment(fraction: number): 'start' | 'center' | 'end' {
  return fraction <= 1e-9 ? 'start' : fraction >= 1 - 1e-9 ? 'end' : 'center';
}

/** Decade ticks (`1`, `10`, `100`, ...) placed by log position inside a positive extent. */
function getLogTickCandidates(extent: Extent, low: string, high: string): TickCandidate[] {
  const lowLog = Math.log10(extent[0]);
  const highLog = Math.log10(extent[1]);
  const candidates: TickCandidate[] = [
    {fraction: 0, label: low},
    {fraction: 1, label: high}
  ];
  for (let power = Math.ceil(lowLog - 1e-9); power <= highLog + 1e-9; power++) {
    candidates.push({
      fraction: (power - lowLog) / (highLog - lowLog),
      label: formatLegendNumber(10 ** power)
    });
  }
  return candidates;
}

/** Evenly spaced CSS gradient through custom legend colours, low value first. */
function getColorsCssGradient(colors: readonly LegendColor[]): string {
  const stops = colors.map(
    (color, index) =>
      `${cssColor(color)} ${((index / Math.max(1, colors.length - 1)) * 100).toFixed(1)}%`
  );
  return `linear-gradient(to right, ${stops.join(', ')})`;
}

// --- classes ---

type ResolvedClasses = {
  breaks: readonly number[];
  colors: readonly LegendColor[];
  labels: readonly string[];
  isGenerated: boolean;
  extent?: Extent;
  unit?: string;
  hatched: ReadonlySet<number>;
  noData?: Spec<'classes'>['noData'];
  format: (value: number) => string;
  method?: string;
};

/** Merges the legend fields over the optional `ClassTable` (the table is the default source). */
function resolveClasses(spec: Spec<'classes'>): ResolvedClasses {
  const table = spec.table;
  const breaks = (spec.breaks?.length ?? 0) > 0 || !table ? (spec.breaks ?? []) : table.breaks;
  const classCount = breaks.length + 1;
  const format = spec.format ?? table?.format ?? formatLegendNumber;
  const extent = spec.extent ?? table?.extent;
  const given = spec.labels ?? table?.labels;
  const colors =
    spec.colors ??
    table?.colors ??
    getClassColors(spec.ramp ?? 'viridis', classCount, spec.reverse);
  const labels = Array.from({length: classCount}, (_, index) => {
    const label = given?.[index];
    if (label !== undefined) return label;
    const lower = index === 0 ? extent?.[0] : breaks[index - 1];
    const upper = index === classCount - 1 ? extent?.[1] : breaks[index];
    if (extent) return `${format(lower as number)}–${format(upper as number)}`;
    if (index === 0) return `< ${format(breaks[0] as number)}`;
    if (index === classCount - 1) return `≥ ${format(breaks[index - 1] as number)}`;
    return `${format(lower as number)}–${format(upper as number)}`;
  });
  return {
    breaks,
    colors: Array.from({length: classCount}, (_, index) => colors[index] ?? FALLBACK_COLOR),
    labels,
    isGenerated: given === undefined,
    extent,
    unit: spec.unit ?? table?.unit,
    hatched: new Set(spec.hatched ?? table?.hatched ?? []),
    noData: spec.noData ?? table?.noData,
    format,
    method: table?.method
  };
}

/** Class index of `value`: the number of breaks at or below it. */
function getClassIndex(breaks: readonly number[], value: number): number {
  let index = 0;
  while (index < breaks.length && value >= (breaks[index] as number)) index++;
  return index;
}

/** Position of `value` on a bar of equal-width class blocks. */
function getClassBarFraction(resolved: ResolvedClasses, value: number): number {
  const count = resolved.breaks.length + 1;
  const index = getClassIndex(resolved.breaks, value);
  const lower = index === 0 ? resolved.extent?.[0] : resolved.breaks[index - 1];
  const upper = index === count - 1 ? resolved.extent?.[1] : resolved.breaks[index];
  const within =
    lower !== undefined && upper !== undefined && upper > lower
      ? Math.min(1, Math.max(0, (value - lower) / (upper - lower)))
      : 0.5;
  return (index + within) / count;
}

function renderSwatchStyle(color: LegendColor): {backgroundColor: string} | undefined {
  return (color[3] ?? 255) === 0 ? undefined : {backgroundColor: cssColor(color)};
}

function swatchClass(base: string, color: LegendColor, isHatched: boolean): string {
  return `${base}${(color[3] ?? 255) === 0 ? ' is-clear' : ''}${isHatched ? ' is-hatched' : ''}`;
}

function renderClassesLegend(spec: Spec<'classes'>, context: RenderContext): HTMLElement {
  const resolved = resolveClasses(spec);
  const classCount = resolved.labels.length;
  const explicitLayout = spec.layout;
  const hasCrowdedLabels = !resolved.isGenerated && classCount > 5;
  const useList =
    (explicitLayout ??
      (spec.counts || classCount > MAX_BAR_CLASSES || hasCrowdedLabels ? 'list' : 'bar')) ===
    'list';
  const interactive = Boolean(spec.interactive);
  const id = getLegendId(spec);
  const showAll = createShowAllButton();
  const marker = spec.marker;
  const body = useList
    ? renderClassList(spec, resolved, interactive, marker)
    : renderClassBar(spec, resolved, interactive, marker);
  const noData = resolved.noData ? renderNoData(resolved.noData) : null;
  const figure = renderFigure(
    'classes',
    spec,
    `${classCount} classes: ${resolved.labels.join(', ')}${resolved.noData ? '; no data' : ''}`,
    resolved.unit,
    useList ? 'legend-classes-list' : 'legend-classes-bar',
    interactive ? showAll : null,
    body.element,
    noData,
    renderNote(spec.note, resolved.method)
  );
  if (interactive) bindSelection(figure, id, body.items, showAll, context);
  return figure;
}

function renderNoData(noData: NonNullable<Spec<'classes'>['noData']>): HTMLElement {
  return h(
    'div',
    {class: 'legend-nodata'},
    h('span', {
      class: `swatch legend-swatch-nodata${noData.hatched ? ' is-hatched' : ''}`,
      'aria-hidden': 'true',
      style: noData.color ? {backgroundColor: cssColor(noData.color)} : undefined
    }),
    h('span', {class: 'legend-class-name'}, noData.label ?? 'No data'),
    noData.count !== undefined
      ? h('span', {class: 'legend-count-text'}, formatLegendNumber(noData.count))
      : null
  );
}

type LegendBody = {element: HTMLElement; items: HTMLElement[]};

function renderClassList(
  spec: Spec<'classes'>,
  resolved: ResolvedClasses,
  interactive: boolean,
  marker: number | undefined
): LegendBody {
  const counts = spec.counts;
  const maxCount = Math.max(1e-9, ...(counts ?? [0]));
  const markerIndex = marker === undefined ? -1 : getClassIndex(resolved.breaks, marker);
  const items: HTMLElement[] = [];
  const rows = resolved.labels.map((label, index) => {
    const color = resolved.colors[index] as LegendColor;
    const count = counts?.[index];
    const cells = [
      h('span', {
        class: swatchClass('swatch', color, resolved.hatched.has(index)),
        'aria-hidden': 'true',
        style: renderSwatchStyle(color)
      }),
      h('span', {class: 'legend-class-name', title: label}, label),
      count === undefined
        ? null
        : h(
            'span',
            {class: 'legend-class-count'},
            h('i', {
              style: {
                width: `${((count / maxCount) * 100).toFixed(1)}%`,
                backgroundColor: cssColor([color[0], color[1], color[2], 255 * 0.35])
              }
            }),
            h('span', {}, formatLegendNumber(count))
          )
    ];
    const rowClass = `legend-row${count === undefined ? '' : ' has-count'}${index === markerIndex ? ' has-marker' : ''}`;
    if (interactive) {
      const button = h(
        'button',
        {class: rowClass, type: 'button', 'aria-pressed': 'false', 'aria-label': label},
        cells
      );
      items.push(button);
      return h('li', {}, button);
    }
    return h('li', {}, h('div', {class: rowClass}, cells));
  });
  // Highest class on top, the map legend convention. The item order stays low class first.
  return {element: h('ul', {class: 'legend-class-list'}, rows.reverse()), items};
}

function renderClassBar(
  spec: Spec<'classes'>,
  resolved: ResolvedClasses,
  interactive: boolean,
  marker: number | undefined
): LegendBody {
  const classCount = resolved.labels.length;
  const {format, extent, breaks} = resolved;
  const candidates: TickCandidate[] = [];
  const unit =
    resolved.unit && resolved.unit.length <= MAX_UNIT_LABEL_LENGTH ? ` ${resolved.unit}` : '';
  if (!resolved.isGenerated) {
    // Custom labels name the classes, so they sit under the swatch they describe.
    resolved.labels.forEach((label, index) => {
      candidates.push({fraction: (index + 0.5) / classCount, label});
    });
  } else {
    // Break labels sit exactly on the boundary between two swatches, never inside one.
    const last = extent ? {fraction: 1, label: format(extent[1])} : null;
    const lastBreak = breaks.length
      ? {fraction: breaks.length / classCount, label: format(breaks[breaks.length - 1] as number)}
      : null;
    const rightmost = last ?? lastBreak;
    if (rightmost) candidates.push({...rightmost, label: rightmost.label + unit});
    if (extent) candidates.push({fraction: 0, label: format(extent[0])});
    breaks.forEach((value, index) => {
      if (!last && index === breaks.length - 1) return;
      candidates.push({fraction: (index + 1) / classCount, label: format(value)});
    });
  }
  const items: HTMLElement[] = [];
  const swatches = resolved.colors.map((color, index) => {
    const className = swatchClass('legend-class-block', color, resolved.hatched.has(index));
    const style = renderSwatchStyle(color);
    if (interactive) {
      const button = h('button', {
        class: className,
        type: 'button',
        'aria-pressed': 'false',
        'aria-label': resolved.labels[index],
        title: resolved.labels[index],
        style
      });
      items.push(button);
      return button;
    }
    return h('span', {class: className, 'aria-hidden': 'true', style});
  });
  const breakFractions =
    extent && extent[1] > extent[0]
      ? breaks.map(value => (value - extent[0]) / (extent[1] - extent[0]))
      : [];
  const element = h(
    'div',
    {class: 'legend-class-bar-wrap'},
    spec.histogram ? renderHistogram(spec.histogram, breakFractions) : null,
    h(
      'div',
      {class: 'legend-bar-wrap'},
      marker !== undefined ? renderMarker(getClassBarFraction(resolved, marker)) : null,
      h('div', {class: 'legend-class-bar'}, swatches)
    ),
    renderPositionedTicks(candidates)
  );
  return {element, items};
}

// --- bivariate and matrix ---

function clampBivariateSize(size: number): number {
  return Math.max(2, Math.min(4, Math.round(size)));
}

function renderBivariateLegend(spec: Spec<'bivariate'>): HTMLElement {
  const size = clampBivariateSize(spec.size);
  const cells: HTMLElement[] = [];
  // Row 0 is the lowest y class and sits at the bottom.
  for (let row = size - 1; row >= 0; row--) {
    for (let column = 0; column < size; column++) {
      const color = spec.colors[row * size + column] ?? FALLBACK_COLOR;
      cells.push(h('span', {style: {backgroundColor: cssColor(color)}}));
    }
  }
  const axis = (axisName: 'x' | 'y', label: string, ends?: readonly [string, string]) =>
    h(
      'div',
      {class: `legend-axis legend-axis-${axisName}${ends ? '' : ' legend-axis-center'}`},
      ends ? h('span', {}, ends[0]) : null,
      h('span', {}, `${label} ${axisName === 'x' ? '→' : '↑'}`),
      ends ? h('span', {}, ends[1]) : null
    );
  return renderFigure(
    'bivariate-figure',
    spec,
    `${size} by ${size} colours: ${spec.xLabel} increases to the right, ${spec.yLabel} increases upward`,
    undefined,
    '',
    null,
    h(
      'div',
      {class: `legend-bivariate legend-bivariate-${size}`},
      axis('y', spec.yLabel, spec.yEnds),
      h(
        'div',
        {
          class: 'legend-bivariate-grid',
          'aria-hidden': 'true',
          style: {gridTemplateColumns: `repeat(${size}, var(--cell))`}
        },
        cells
      ),
      h('span'),
      axis('x', spec.xLabel, spec.xEnds)
    ),
    renderNote(spec.note)
  );
}

function renderMatrixLegend(spec: Spec<'matrix'>): HTMLElement {
  const columnCount = Math.max(1, spec.columns.length);
  const cells: HTMLElement[] = [];
  if (spec.columnTitle) {
    cells.push(
      h('span'),
      h(
        'span',
        {class: 'legend-matrix-title', style: {gridColumn: `2 / span ${columnCount}`}},
        spec.columnTitle
      )
    );
  }
  cells.push(
    h('span', {class: 'legend-matrix-title legend-matrix-row-title'}, spec.rowTitle ?? ''),
    ...spec.columns.map(label => h('span', {class: 'legend-matrix-label'}, label))
  );
  spec.rows.forEach((rowLabel, row) => {
    cells.push(h('span', {class: 'legend-matrix-label legend-matrix-row-label'}, rowLabel));
    for (let column = 0; column < columnCount; column++) {
      const color = spec.colors[row * columnCount + column] ?? FALLBACK_COLOR;
      cells.push(
        h('span', {
          class: 'legend-matrix-cell',
          'aria-hidden': 'true',
          style: {backgroundColor: cssColor(color)}
        })
      );
    }
  });
  return renderFigure(
    'matrix',
    spec,
    `${spec.rows.length} by ${columnCount} key, rows ${spec.rows.join(', ')}, columns ${spec.columns.join(', ')}`,
    undefined,
    '',
    null,
    h(
      'div',
      {
        class: 'legend-matrix-grid',
        style: {gridTemplateColumns: `auto repeat(${columnCount}, minmax(24px, 1fr))`}
      },
      cells
    ),
    renderNote(spec.note)
  );
}

// --- categories ---

function renderCategoriesLegend(spec: Spec<'categories'>, context: RenderContext): HTMLElement {
  const hasColumns = spec.entries.some(
    entry => entry.detail !== undefined || entry.count !== undefined
  );
  const layout = spec.layout ?? (hasColumns ? 'list' : 'grid');
  const interactive = Boolean(spec.interactive);
  const items: HTMLElement[] = [];
  const rows = spec.entries.map(entry => {
    const shape = entry.shape ?? 'swatch';
    const color = cssColor(entry.color);
    const mark = h('span', {
      class: `legend-shape legend-shape-${shape}`,
      'aria-hidden': 'true',
      style:
        shape === 'ring'
          ? {borderColor: color}
          : shape === 'line'
            ? {backgroundColor: color}
            : {backgroundColor: color}
    });
    const cells = [
      mark,
      h('span', {class: 'legend-entry-label', title: entry.label}, entry.label),
      entry.detail !== undefined ? h('span', {class: 'legend-entry-detail'}, entry.detail) : null,
      entry.count !== undefined
        ? h('span', {class: 'legend-count-text'}, formatLegendNumber(entry.count))
        : null
    ];
    const rowClass = `legend-row${entry.detail !== undefined ? ' has-detail' : ''}${entry.count !== undefined ? ' has-entry-count' : ''}`;
    if (interactive) {
      const button = h(
        'button',
        {class: rowClass, type: 'button', 'aria-pressed': 'false', 'aria-label': entry.label},
        cells
      );
      items.push(button);
      return h('li', {}, button);
    }
    return h('li', {}, h('div', {class: rowClass}, cells));
  });
  const showAll = createShowAllButton();
  const figure = renderFigure(
    'categories',
    spec,
    `${spec.entries.length} categories: ${spec.entries.map(entry => entry.label).join(', ')}`,
    undefined,
    `legend-categories-${layout}`,
    interactive ? showAll : null,
    h('ul', {class: 'legend-category-list'}, rows),
    renderNote(spec.note)
  );
  if (interactive) bindSelection(figure, getLegendId(spec), items, showAll, context);
  return figure;
}

// --- cyclic ---

function createSvg(
  tag: string,
  attributes: Record<string, string | number>,
  ...children: SVGElement[]
): SVGElement {
  const element = document.createElementNS(SVG_NAMESPACE, tag);
  for (const [name, value] of Object.entries(attributes)) element.setAttribute(name, String(value));
  element.append(...children);
  return element;
}

function createSvgText(attributes: Record<string, string | number>, text: string): SVGElement {
  const element = createSvg('text', attributes);
  element.textContent = text;
  return element;
}

/** Point on a circle, `fraction` of a turn clockwise from the top. */
function polar(
  centerX: number,
  centerY: number,
  radius: number,
  fraction: number
): [number, number] {
  const angle = fraction * Math.PI * 2;
  return [centerX + radius * Math.sin(angle), centerY - radius * Math.cos(angle)];
}

function renderCyclicLegend(spec: Spec<'cyclic'>): HTMLElement {
  const width = 176;
  const height = 132;
  const centerX = width / 2;
  const centerY = height / 2;
  const outer = 40;
  const inner = 25;
  const labelRadius = outer + 7;
  const segmentCount = spec.colors?.length ?? 72;
  const segments: SVGElement[] = [];
  for (let index = 0; index < segmentCount; index++) {
    const from = index / segmentCount;
    const to = (index + 1) / segmentCount;
    const color = spec.colors
      ? cssColor(spec.colors[index] ?? FALLBACK_COLOR)
      : (() => {
          const [r, g, b] = sampleRamp(spec.ramp ?? 'romao', (index + 0.5) / segmentCount);
          return `rgb(${r} ${g} ${b})`;
        })();
    const [outerFromX, outerFromY] = polar(centerX, centerY, outer, from);
    const [outerToX, outerToY] = polar(centerX, centerY, outer, to);
    const [innerToX, innerToY] = polar(centerX, centerY, inner, to);
    const [innerFromX, innerFromY] = polar(centerX, centerY, inner, from);
    const large = to - from > 0.5 ? 1 : 0;
    segments.push(
      createSvg('path', {
        d: `M${outerFromX} ${outerFromY}A${outer} ${outer} 0 ${large} 1 ${outerToX} ${outerToY}L${innerToX} ${innerToY}A${inner} ${inner} 0 ${large} 0 ${innerFromX} ${innerFromY}Z`,
        fill: color,
        stroke: color,
        'stroke-width': 0.6
      })
    );
  }
  const labelElements = spec.labels.map((label, index) => {
    const fraction = index / spec.labels.length;
    return createCyclicLabel(centerX, centerY, labelRadius, fraction, label, 'legend-cyclic-label');
  });
  const labelFractions = spec.labels.map((_, index) => index / spec.labels.length);
  const markElements = (spec.marks ?? []).flatMap(mark => {
    const [fromX, fromY] = polar(centerX, centerY, inner - 2, mark.at);
    const [toX, toY] = polar(centerX, centerY, outer + 3, mark.at);
    const tick = createSvg('line', {
      x1: fromX,
      y1: fromY,
      x2: toX,
      y2: toY,
      class: 'legend-cyclic-mark'
    });
    const collides = labelFractions.some(
      fraction => Math.min(Math.abs(fraction - mark.at), 1 - Math.abs(fraction - mark.at)) < 0.06
    );
    return collides
      ? [tick]
      : [
          tick,
          createCyclicLabel(
            centerX,
            centerY,
            labelRadius,
            mark.at,
            mark.label,
            'legend-cyclic-label is-mark'
          )
        ];
  });
  return renderFigure(
    'cyclic',
    spec,
    `Cyclic key, starting at the top and running clockwise: ${spec.labels.join(', ')}`,
    undefined,
    '',
    null,
    createSvg(
      'svg',
      {
        class: 'legend-cyclic-ring',
        width,
        height,
        viewBox: `0 0 ${width} ${height}`,
        'aria-hidden': 'true'
      },
      ...segments,
      ...markElements,
      ...labelElements
    ) as unknown as HTMLElement,
    renderNote(spec.note)
  );
}

function createCyclicLabel(
  centerX: number,
  centerY: number,
  radius: number,
  fraction: number,
  label: string,
  className: string
): SVGElement {
  const [x, y] = polar(centerX, centerY, radius, fraction);
  const horizontal = Math.sin(fraction * Math.PI * 2);
  const anchor = horizontal > 0.3 ? 'start' : horizontal < -0.3 ? 'end' : 'middle';
  const vertical = Math.cos(fraction * Math.PI * 2);
  const baselineShift = vertical > 0.5 ? 0 : vertical < -0.5 ? 9 : 4;
  return createSvgText({x, y: y + baselineShift, 'text-anchor': anchor, class: className}, label);
}

// --- alpha ---

function renderAlphaLegend(spec: Spec<'alpha'>): HTMLElement {
  const steps = Math.max(2, Math.round(spec.steps ?? 4));
  const [alphaLow, alphaHigh] = spec.alphaRange ?? [0.25, 1];
  const hasRowLabels = Boolean(spec.rowLabels?.length);
  const rows = spec.colors.map((color, row) =>
    h(
      'div',
      {class: 'legend-alpha-row'},
      hasRowLabels ? h('span', {class: 'legend-alpha-label'}, spec.rowLabels?.[row] ?? '') : null,
      h(
        'div',
        {class: 'legend-alpha-cells', 'aria-hidden': 'true'},
        Array.from({length: steps}, (_, step) => {
          const alpha = alphaLow + ((alphaHigh - alphaLow) * step) / (steps - 1);
          return h(
            'span',
            {class: 'legend-alpha-cell'},
            h('i', {
              style: {
                backgroundColor: cssColor([
                  color[0],
                  color[1],
                  color[2],
                  255 * alpha * ((color[3] ?? 255) / 255)
                ])
              }
            })
          );
        })
      )
    )
  );
  return renderFigure(
    'alpha',
    spec,
    `${spec.colors.length} colours fading from ${spec.ends[0]} to ${spec.ends[1]}`,
    undefined,
    hasRowLabels ? 'has-row-labels' : '',
    null,
    h('div', {class: 'legend-alpha-rows'}, rows),
    h(
      'div',
      {class: `legend-alpha-ends${hasRowLabels ? ' has-row-labels' : ''}`},
      hasRowLabels ? h('span') : null,
      h('span', {}, spec.ends[0]),
      h('span', {}, spec.ends[1])
    ),
    renderNote(spec.note)
  );
}

// --- line and size ---

function renderLineLegend(spec: Spec<'line'>): HTMLElement {
  return renderFigure(
    'lines-figure',
    spec,
    spec.entries
      .map(entry => `${entry.label} ${entry.widthPixels} pixels${entry.dashed ? ', dashed' : ''}`)
      .join('; '),
    undefined,
    '',
    null,
    h(
      'ul',
      {class: 'legend-lines'},
      spec.entries.map(entry => {
        const width = Math.max(1, Math.min(10, entry.widthPixels));
        const sample = createSvg(
          'svg',
          {
            class: 'legend-line-sample',
            width: 28,
            height: 12,
            viewBox: '0 0 28 12',
            'aria-hidden': 'true'
          },
          createSvg('line', {
            x1: 1,
            y1: 6,
            x2: 27,
            y2: 6,
            stroke: cssColor(entry.color),
            'stroke-width': width,
            'stroke-linecap': entry.dashed ? 'butt' : 'round',
            ...(entry.dashed ? {'stroke-dasharray': `${width * 2 + 2} ${width + 2}`} : {})
          })
        );
        return h('li', {}, sample as unknown as HTMLElement, h('span', {}, entry.label));
      })
    ),
    renderNote(spec.note)
  );
}

function renderRowSizeLegend(spec: Spec<'size'>): HTMLElement {
  return renderFigure(
    'size',
    spec,
    spec.entries.map(entry => entry.label).join(', '),
    spec.unit,
    '',
    null,
    h(
      'ul',
      {class: 'legend-size-row'},
      spec.entries.map(entry =>
        h(
          'li',
          {},
          h('span', {
            class: 'dot',
            'aria-hidden': 'true',
            style: {
              width: `${entry.radiusPixels * 2}px`,
              height: `${entry.radiusPixels * 2}px`,
              backgroundColor: cssColor(spec.color ?? [120, 130, 150, 255])
            }
          }),
          h('span', {}, entry.label)
        )
      )
    ),
    renderNote(spec.note)
  );
}

/** Concentric proportional-symbol legend: shared baseline, dashed leaders to right-aligned labels. */
function renderNestedSizeLegend(spec: Spec<'size'>): HTMLElement {
  const entries = [...spec.entries].sort((a, b) => b.radiusPixels - a.radiusPixels);
  const maxRadius = Math.max(1, entries[0]?.radiusPixels ?? 1);
  const diameter = maxRadius * 2;
  const centerX = maxRadius + 1;
  const baseline = diameter + 1;
  const labelWidth = Math.ceil(Math.max(1, ...entries.map(entry => entry.label.length)) * 7) + 2;
  const elbowX = diameter + 6;
  const width = elbowX + 6 + labelWidth;
  const lineHeight = 12;
  const circles: SVGElement[] = [];
  const leaders: SVGElement[] = [];
  const labels: SVGElement[] = [];
  let previousLabelY = Number.NEGATIVE_INFINITY;
  for (const entry of entries) {
    const top = baseline - entry.radiusPixels * 2;
    const labelY = Math.max(top, previousLabelY + lineHeight);
    previousLabelY = labelY;
    circles.push(
      createSvg('circle', {
        cx: centerX,
        cy: baseline - entry.radiusPixels,
        r: entry.radiusPixels,
        fill: spec.color
          ? cssColor([spec.color[0], spec.color[1], spec.color[2], 255 * 0.18])
          : 'none',
        ...(spec.outline ? {style: `stroke:${cssColor(spec.outline)};stroke-opacity:1`} : {})
      })
    );
    const points =
      labelY === top
        ? `${centerX},${top} ${elbowX + 4},${top}`
        : `${centerX},${top} ${elbowX},${top} ${elbowX + 4},${labelY}`;
    leaders.push(createSvg('polyline', {points}));
    labels.push(createSvgText({x: width - 1, y: labelY + 4, 'text-anchor': 'end'}, entry.label));
  }
  const height = Math.ceil(Math.max(baseline, previousLabelY + 6)) + 1;
  return renderFigure(
    'size legend-size-nested',
    spec,
    `Nested proportional symbols: ${entries.map(entry => entry.label).join(', ')}`,
    spec.unit,
    '',
    null,
    createSvg(
      'svg',
      {
        class: 'legend-nested',
        width,
        height,
        viewBox: `0 0 ${width} ${height}`,
        'aria-hidden': 'true'
      },
      ...leaders,
      ...circles,
      ...labels
    ) as unknown as HTMLElement,
    renderNote(spec.note)
  );
}
