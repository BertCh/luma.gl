// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {FurnitureSpec, MapClockSpec, ScaleBarSpec, ScaleBarUnits} from '../cartography/types';
import {h} from './dom';
import './map-furniture.css';

/** What the furniture needs besides the spec. */
export type FurnitureContext = {
  sceneTitle: string;
  stepTitle: string;
  /** Attribution strings of the scene's datasets (may be empty / still loading). */
  credits: readonly string[];
  /**
   * Attribution of the tile basemap when one is shown (for example `'Basemap © CARTO © OpenStreetMap'`).
   * Appended to the credit line automatically; leave it out when no tile basemap is visible.
   */
  basemapCredit?: string;
};

/** Camera the furniture follows. */
export type FurnitureView = {latitude: number; zoom: number; bearing: number; pitch: number};

/** Extra options of {@link measureScaleBar}, taken from the scene's `ScaleBarSpec`. */
export type ScaleBarMeasureOptions = Pick<ScaleBarSpec, 'latitude' | 'minZoom' | 'ticks'>;

/** A parameter distance marked on the bar. */
export type ScaleBarTick = {
  /** Distance in metres. */
  meters: number;
  /** Label with its unit, for example `150 m`. */
  label: string;
  /** Offset from the left end of the bar in pixels. */
  offsetPixels: number;
};

/** One measured scale bar (one entry per unit system). */
export type ScaleBarMeasure = {
  /** Full length label of the bar, for example `5 km`. */
  label: string;
  /** Length of the bar in pixels. */
  widthPixels: number;
  /** Labels under the bar: `0`, the midpoint, and the maximum with its unit (last label only). */
  numerals: readonly [string, string, string];
  /** Parameter ticks that fit within the bar's span, drawn as labelled accent ticks. */
  ticks: readonly ScaleBarTick[];
};

const EARTH_CIRCUMFERENCE_METERS = 40075016.686;
const TILE_SIZE = 512;
const FEET_PER_METER = 3.280839895;
const FEET_PER_MILE = 5280;
const METERS_PER_NAUTICAL_MILE = 1852;
const MAX_BAR_PIXELS = 120;
const MAX_BAR_PIXELS_NARROW = 90;
const MAX_PITCH_FOR_SCALE = 40;
const MIN_TICK_LABEL_GAP_PIXELS = 38;
const NARROW_QUERY = '(max-width: 900px)';
const SVG_NAMESPACE = 'http://www.w3.org/2000/svg';

/** Length unit of a scale bar: its label and size in metres. */
type LengthUnit = {name: string; metersPerUnit: number};

const UNIT_METER: LengthUnit = {name: 'm', metersPerUnit: 1};
const UNIT_KILOMETER: LengthUnit = {name: 'km', metersPerUnit: 1000};
const UNIT_FOOT: LengthUnit = {name: 'ft', metersPerUnit: 1 / FEET_PER_METER};
const UNIT_MILE: LengthUnit = {name: 'mi', metersPerUnit: FEET_PER_MILE / FEET_PER_METER};
const UNIT_NAUTICAL_MILE: LengthUnit = {name: 'nmi', metersPerUnit: METERS_PER_NAUTICAL_MILE};

type UnitSystem = 'metric' | 'imperial' | 'nautical';

/** Largest 1, 2 or 5 times a power of ten that is not greater than `maximum`. */
function niceFloor(maximum: number): number {
  if (!(maximum > 0) || !Number.isFinite(maximum)) return 0;
  const magnitude = 10 ** Math.floor(Math.log10(maximum));
  for (const step of [5, 2, 1]) {
    if (step * magnitude <= maximum * (1 + 1e-9)) return step * magnitude;
  }
  return magnitude;
}

/** Smallest 1, 2 or 5 times a power of ten that is not less than `minimum`. */
function niceCeil(minimum: number): number {
  if (!(minimum > 0) || !Number.isFinite(minimum)) return 0;
  const magnitude = 10 ** Math.floor(Math.log10(minimum));
  for (const step of [1, 2, 5, 10]) {
    if (step * magnitude >= minimum * (1 - 1e-9)) return step * magnitude;
  }
  return 10 * magnitude;
}

function formatNumber(value: number): string {
  return value.toLocaleString('en-US', {maximumFractionDigits: 3});
}

/** The unit a bar of the given length is labelled in, per unit system. */
function pickUnit(system: UnitSystem, barMeters: number): LengthUnit {
  if (system === 'imperial') {
    return barMeters * FEET_PER_METER < FEET_PER_MILE / 4 ? UNIT_FOOT : UNIT_MILE;
  }
  if (system === 'nautical' && barMeters >= METERS_PER_NAUTICAL_MILE / 4) {
    return UNIT_NAUTICAL_MILE;
  }
  // Nautical bars shorter than a quarter mile fall back to metres.
  return barMeters >= 1000 ? UNIT_KILOMETER : UNIT_METER;
}

function measureSystem(
  system: UnitSystem,
  metersPerPixel: number,
  maxWidthPixels: number,
  tickMeters: readonly number[]
): ScaleBarMeasure | null {
  const maxMeters = metersPerPixel * maxWidthPixels;
  let unit = pickUnit(system, maxMeters);
  let value = niceFloor(maxMeters / unit.metersPerUnit);
  if (value <= 0) return null;

  // Extend the bar to the nearest nice value at or above the largest tick when that still fits.
  const largestTick = Math.max(0, ...tickMeters);
  if (largestTick > value * unit.metersPerUnit * (1 + 1e-9)) {
    const extendedMeters = niceCeil(largestTick / unit.metersPerUnit) * unit.metersPerUnit;
    if (extendedMeters / metersPerPixel <= maxWidthPixels * (1 + 1e-9)) {
      unit = pickUnit(system, extendedMeters);
      value = extendedMeters / unit.metersPerUnit;
    }
  }

  const barMeters = value * unit.metersPerUnit;
  const widthPixels = Math.round(barMeters / metersPerPixel);
  const ticks: ScaleBarTick[] = [];
  for (const meters of [...new Set(tickMeters)].sort((a, b) => a - b)) {
    if (!(meters > 0) || meters > barMeters * (1 + 1e-9)) continue;
    ticks.push({
      meters,
      label: `${formatNumber(meters / unit.metersPerUnit)} ${unit.name}`,
      offsetPixels: Math.round(meters / metersPerPixel)
    });
  }
  return {
    label: `${formatNumber(value)} ${unit.name}`,
    widthPixels,
    numerals: ['0', formatNumber(value / 2), `${formatNumber(value)} ${unit.name}`],
    ticks
  };
}

/**
 * Computes the scale bar entries for a camera. Exported for tests and for scenes that draw their
 * own bar. Returns an empty list when the bar would be misleading (tilted view, below
 * `options.minZoom`) or unmeasurable. `options.latitude` measures at that latitude instead of the
 * camera's; `options.ticks` are parameter distances in metres that the bar is extended to cover.
 */
export function measureScaleBar(
  view: FurnitureView,
  units: ScaleBarUnits,
  maxWidthPixels: number,
  options: ScaleBarMeasureOptions = {}
): ScaleBarMeasure[] {
  if (view.pitch > MAX_PITCH_FOR_SCALE) return [];
  if (options.minZoom !== undefined && view.zoom < options.minZoom) return [];
  const latitude = options.latitude ?? view.latitude;
  const metersPerPixel =
    (EARTH_CIRCUMFERENCE_METERS * Math.cos((latitude * Math.PI) / 180)) /
    (TILE_SIZE * 2 ** view.zoom);
  if (!(metersPerPixel > 0) || !Number.isFinite(metersPerPixel)) return [];
  const tickMeters = options.ticks ?? [];
  const systems: UnitSystem[] =
    units === 'both' ? ['metric', 'imperial'] : [units === 'nautical' ? 'nautical' : units];
  return systems
    .map(system => measureSystem(system, metersPerPixel, maxWidthPixels, tickMeters))
    .filter((measure): measure is ScaleBarMeasure => measure !== null);
}

function svgElement(tag: string, attributes: Record<string, string>): SVGElement {
  const element = document.createElementNS(SVG_NAMESPACE, tag);
  for (const [name, value] of Object.entries(attributes)) element.setAttribute(name, value);
  return element;
}

function createNorthArrowSvg(): SVGSVGElement {
  const svg = svgElement('svg', {viewBox: '0 0 30 38', 'aria-hidden': 'true'}) as SVGSVGElement;
  const letter = svgElement('text', {x: '15', y: '10', class: 'map-furniture-north-letter'});
  letter.textContent = 'N';
  svg.append(
    letter,
    // Left half dark, right half light: the classic two-tone needle.
    svgElement('path', {d: 'M15 14 L21 35 L15 30 Z', class: 'map-furniture-north-needle-light'}),
    svgElement('path', {d: 'M15 14 L9 35 L15 30 Z', class: 'map-furniture-north-needle-dark'})
  );
  return svg;
}

/** One zone of the clock with its pre-built formatters. */
type ClockZone = {
  zone: string;
  time: Intl.DateTimeFormat;
  date: Intl.DateTimeFormat;
};

const UNIT_MILLISECONDS = {
  seconds: 1000,
  minutes: 60_000,
  hours: 3_600_000,
  days: 86_400_000
} as const;

function createClockZone(zone: string, hour12 = false): ClockZone | null {
  try {
    return {
      zone,
      time: new Intl.DateTimeFormat('en-GB', {
        timeZone: zone,
        hour: '2-digit',
        minute: '2-digit',
        hour12,
        timeZoneName: 'short'
      }),
      date: new Intl.DateTimeFormat('en-GB', {
        timeZone: zone,
        weekday: 'short',
        day: 'numeric',
        month: 'short',
        year: 'numeric'
      })
    };
  } catch {
    // An unknown IANA zone: skip it rather than break the map.
    return null;
  }
}

/** Splits a formatted time into the digits and the zone abbreviation. */
function formatClockTime(zone: ClockZone, instant: Date): {digits: string; zoneName: string} {
  let digits = '';
  let zoneName = '';
  for (const part of zone.time.formatToParts(instant)) {
    if (part.type === 'timeZoneName') zoneName = part.value;
    else if (part.type === 'hour' || part.type === 'minute' || part.type === 'literal') {
      digits += part.value;
    }
  }
  return {digits: digits.trim(), zoneName: zone.zone === 'UTC' ? 'UTC' : zoneName};
}

/** Converts the clock option value to an instant. `null` when the value is not a finite number. */
function resolveClockInstant(time: MapClockSpec['time'], value: number): Date | null {
  if (!Number.isFinite(value)) return null;
  let milliseconds: number;
  if (time === 'epoch-seconds') milliseconds = value * 1000;
  else if (time === 'epoch-ms') milliseconds = value;
  else {
    const origin = Date.parse(time.origin);
    if (Number.isNaN(origin)) return null;
    milliseconds = origin + value * UNIT_MILLISECONDS[time.unit];
  }
  const instant = new Date(milliseconds);
  return Number.isNaN(instant.getTime()) ? null : instant;
}

/**
 * DOM elements for declarative map furniture. The integrator places the elements into the map
 * corners and feeds them the resolved spec, the camera and the clock option value.
 */
export class MapFurniture {
  /** Title cartouche (placed top-left). Hidden when off. */
  readonly cartouche: HTMLElement;
  /**
   * North arrow (placed top-right). Hidden when off. A `<button>` inside it (44 px target) exists
   * only while {@link onNorthClick} is set.
   */
  readonly northArrow: HTMLElement;
  /** Scale bar (placed bottom-right, above the basemap attribution). Hidden when off. */
  readonly scaleBar: HTMLElement;
  /** Source credit line and caveat (placed bottom-left, under the legend). Hidden when off. */
  readonly credit: HTMLElement;
  /** Clock for playback stories (placed top-right by the shell). Hidden when off or no value. */
  readonly clock: HTMLElement;

  private readonly titleElement: HTMLElement;
  private readonly subtitleElement: HTMLElement;
  private readonly sampleElement: HTMLElement;
  private readonly chipsElement: HTMLElement;
  private readonly creditLine: HTMLElement;
  private readonly caveatLine: HTMLElement;
  private readonly northSvg: SVGSVGElement;
  private readonly narrowQuery: MediaQueryList | null;

  private scaleUnits: ScaleBarUnits | null = null;
  private scaleOptions: ScaleBarMeasureOptions = {};
  private northMode: 'auto' | 'always' | 'never' = 'never';
  private northCallback: (() => void) | undefined;
  private northButton: HTMLButtonElement | null = null;
  private view: FurnitureView | null = null;
  private maxWidthOverride: number | undefined;
  private clockSpec: MapClockSpec | null = null;
  private clockZones: ClockZone[] = [];
  private clockValue: number | null = null;

  // Last values written to the DOM, so repeated calls are cheap no-ops.
  private scaleKey = '';
  private northKey = '';
  private titleKey = '';
  private creditKey = '';
  private clockKey = '';

  constructor() {
    this.titleElement = h('div', {class: 'map-furniture-title'});
    this.subtitleElement = h('div', {class: 'map-furniture-subtitle'});
    this.sampleElement = h('div', {class: 'map-furniture-sample'});
    this.chipsElement = h('div', {class: 'map-furniture-chips'});
    this.cartouche = h(
      'div',
      {class: 'map-furniture-cartouche', hidden: true},
      this.titleElement,
      this.subtitleElement,
      this.sampleElement,
      this.chipsElement
    );
    this.northSvg = createNorthArrowSvg();
    this.northArrow = h(
      'div',
      {class: 'map-furniture-north', role: 'img', 'aria-label': 'North arrow', hidden: true},
      this.northSvg
    );
    this.scaleBar = h('div', {class: 'map-furniture-scale', hidden: true});
    this.creditLine = h('div', {class: 'map-furniture-credit-line'});
    this.caveatLine = h('div', {class: 'map-furniture-caveat'});
    this.credit = h(
      'div',
      {class: 'map-furniture-credit', hidden: true},
      this.creditLine,
      this.caveatLine
    );
    this.clock = h('div', {class: 'map-furniture-clock', role: 'timer', hidden: true});
    this.narrowQuery =
      typeof window !== 'undefined' && window.matchMedia ? window.matchMedia(NARROW_QUERY) : null;
  }

  /**
   * Called when the reader clicks the north arrow (the shell resets the bearing). While set, the
   * arrow is a keyboard-focusable `<button>` with a 44 px touch target; unset, it is a picture.
   */
  get onNorthClick(): (() => void) | undefined {
    return this.northCallback;
  }

  set onNorthClick(callback: (() => void) | undefined) {
    if (callback === this.northCallback) return;
    this.northCallback = callback;
    this.northKey = '';
    if (callback) {
      this.northButton = h('button', {
        class: 'map-furniture-north-button',
        type: 'button',
        'aria-label': 'Reset north',
        on: {click: () => this.northCallback?.()}
      });
      this.northButton.append(this.northSvg);
      this.northArrow.removeAttribute('role');
      this.northArrow.removeAttribute('aria-label');
      this.northArrow.replaceChildren(this.northButton);
      this.northArrow.classList.add('is-interactive');
    } else {
      this.northButton = null;
      this.northArrow.setAttribute('role', 'img');
      this.northArrow.setAttribute('aria-label', 'North arrow');
      this.northArrow.replaceChildren(this.northSvg);
      this.northArrow.classList.remove('is-interactive');
    }
    this.renderNorthArrow();
  }

  /** Applies a resolved spec (`null` = no furniture, everything hidden). */
  setSpec(spec: FurnitureSpec | null, context: FurnitureContext): void {
    this.setCartouche(spec?.title, context);
    this.setCredit(spec, context);

    const scaleBar = spec?.scaleBar;
    this.scaleUnits = !scaleBar
      ? null
      : scaleBar === true
        ? 'metric'
        : (scaleBar.units ?? 'metric');
    this.scaleOptions =
      scaleBar && scaleBar !== true
        ? {latitude: scaleBar.latitude, minZoom: scaleBar.minZoom, ticks: scaleBar.ticks}
        : {};
    this.northMode = spec ? (spec.northArrow ?? 'auto') : 'never';
    this.setClockSpec(spec?.clock || null);

    this.renderScaleBar();
    this.renderNorthArrow();
    this.renderClock();
  }

  /**
   * Updates the scale bar and the north arrow for the camera. Cheap; DOM is written only when a
   * rendered value changed. `maxWidthPixels` caps the bar (default 120, or 90 on narrow screens).
   */
  updateView(view: FurnitureView, maxWidthPixels?: number): void {
    this.view = view;
    this.maxWidthOverride = maxWidthPixels;
    this.renderScaleBar();
    this.renderNorthArrow();
  }

  /**
   * Feeds the clock the current value of its option (`MapClockSpec.option`). Pass `null` to hide
   * the clock, for example while the option has no value yet. Cheap: DOM is written only when the
   * displayed text changed.
   */
  updateClock(value: number | null): void {
    this.clockValue = value;
    this.renderClock();
  }

  destroy(): void {
    this.cartouche.remove();
    this.northArrow.remove();
    this.scaleBar.remove();
    this.credit.remove();
    this.clock.remove();
    this.view = null;
  }

  private setCartouche(
    title: FurnitureSpec['title'],
    {sceneTitle, stepTitle}: FurnitureContext
  ): void {
    let titleText = '';
    let subtitleText = '';
    let sampleText = '';
    let chips: readonly string[] = [];
    if (title === true) {
      titleText = sceneTitle;
      subtitleText = stepTitle;
    } else if (title) {
      titleText = title.title ?? sceneTitle;
      subtitleText = title.subtitle ?? '';
      sampleText = title.sample ?? '';
      chips = title.chips ?? [];
    }
    const key = [titleText, subtitleText, sampleText, ...chips].join('\u0000');
    if (key === this.titleKey) return;
    this.titleKey = key;
    this.cartouche.hidden = !titleText && !subtitleText && !sampleText && chips.length === 0;
    this.titleElement.textContent = titleText;
    this.titleElement.title = titleText;
    this.titleElement.hidden = !titleText;
    this.subtitleElement.textContent = subtitleText;
    this.subtitleElement.title = subtitleText;
    this.subtitleElement.hidden = !subtitleText;
    this.sampleElement.textContent = sampleText;
    this.sampleElement.hidden = !sampleText;
    this.chipsElement.replaceChildren(
      ...chips.map(chip => h('span', {class: 'map-furniture-chip'}, chip))
    );
    this.chipsElement.hidden = chips.length === 0;
  }

  private setCredit(spec: FurnitureSpec | null, {credits, basemapCredit}: FurnitureContext): void {
    const credit = spec?.credit;
    const parts: string[] = [];
    if (typeof credit === 'string') parts.push(credit);
    else if (credit) {
      const unique = [...new Set(credits.map(entry => entry.trim()).filter(Boolean))];
      if (unique.length > 0) parts.push(`Data: ${unique.join(' · ')}`);
    }
    // The basemap credit follows automatically whenever furniture is declared and the credit
    // line is not switched off explicitly.
    if (spec && credit !== false && basemapCredit) parts.push(basemapCredit);
    const text = parts.join(' · ');
    const caveat = spec?.caveat ?? '';
    const key = `${text}\u0000${caveat}`;
    if (key === this.creditKey) return;
    this.creditKey = key;
    this.credit.hidden = !text && !caveat;
    this.creditLine.textContent = text;
    this.creditLine.title = text;
    this.creditLine.hidden = !text;
    this.caveatLine.textContent = caveat;
    this.caveatLine.title = caveat;
    this.caveatLine.hidden = !caveat;
  }

  private setClockSpec(spec: MapClockSpec | null): void {
    this.clockSpec = spec;
    this.clockZones = (spec?.zones?.length ? spec.zones : ['UTC'])
      .map(zone => createClockZone(zone))
      .filter((zone): zone is ClockZone => zone !== null);
    this.clockKey = '';
  }

  private renderScaleBar(): void {
    const measures =
      this.scaleUnits && this.view
        ? measureScaleBar(this.view, this.scaleUnits, this.getMaxBarPixels(), this.scaleOptions)
        : [];
    const latitudeNote =
      measures.length > 0 && this.scaleOptions.latitude !== undefined
        ? `Scale at ${formatLatitude(this.scaleOptions.latitude)}`
        : '';
    const key = `${latitudeNote}|${measures
      .map(
        measure =>
          `${measure.label}@${measure.widthPixels}:${measure.ticks.map(tick => `${tick.label}@${tick.offsetPixels}`).join(',')}`
      )
      .join('|')}`;
    if (key === this.scaleKey) return;
    this.scaleKey = key;
    this.scaleBar.hidden = measures.length === 0;
    replaceWithChildren(
      this.scaleBar,
      measures.map(createScaleRow),
      latitudeNote ? h('div', {class: 'map-furniture-scale-note'}, latitudeNote) : null
    );
    this.scaleBar.setAttribute(
      'aria-label',
      measures.length
        ? `Scale bar: ${measures.map(measure => measure.label).join(' / ')}${latitudeNote ? `, ${latitudeNote.toLowerCase()}` : ''}`
        : ''
    );
  }

  private renderNorthArrow(): void {
    const view = this.view;
    const visible =
      this.northMode === 'always'
        ? true
        : this.northMode === 'auto' && view !== null
          ? Math.abs(view.bearing) > 0.5 || view.pitch > 0.5
          : false;
    const bearing = view?.bearing ?? 0;
    const pitch = view?.pitch ?? 0;
    const key = `${visible}|${bearing.toFixed(1)}|${pitch.toFixed(1)}`;
    if (key === this.northKey) return;
    this.northKey = key;
    this.northArrow.hidden = !visible;
    if (!visible) return;
    const squash = Math.cos((Math.min(pitch, 85) * Math.PI) / 180);
    this.northSvg.style.transform = `rotate(${-bearing}deg) scaleY(${squash.toFixed(3)})`;
    const description = `Bearing ${Math.round(bearing)}°`;
    this.northArrow.title = this.northButton
      ? `${description}. Click to reset north.`
      : description;
    this.northButton?.setAttribute('aria-label', `Reset north (${description.toLowerCase()})`);
  }

  private renderClock(): void {
    const spec = this.clockSpec;
    const instant =
      spec && this.clockValue !== null ? resolveClockInstant(spec.time, this.clockValue) : null;
    if (!spec || !instant || this.clockZones.length === 0) {
      if (this.clockKey !== 'hidden') {
        this.clockKey = 'hidden';
        this.clock.hidden = true;
      }
      return;
    }
    const show = spec.show ?? 'datetime';
    const [primary, ...others] = this.clockZones;
    const primaryTime = formatClockTime(primary, instant);
    const primaryDate = primary.date.format(instant);
    const main = show === 'date' ? primaryDate : primaryTime.digits;

    let fraction = -1;
    if (spec.progress && this.clockValue !== null) {
      const [start, end] = spec.progress;
      fraction =
        end > start ? Math.min(1, Math.max(0, (this.clockValue - start) / (end - start))) : 0;
    }
    const secondaryLines = others.map(zone => {
      const {digits, zoneName} = formatClockTime(zone, instant);
      const date = zone.date.format(instant);
      const text = show === 'date' ? date : show === 'time' ? digits : `${digits} · ${date}`;
      return {zoneName, text};
    });
    const key = [
      main,
      primaryTime.zoneName,
      show === 'datetime' ? primaryDate : '',
      secondaryLines.map(line => `${line.zoneName}${line.text}`).join('|'),
      fraction < 0 ? '' : fraction.toFixed(3)
    ].join('\u0001');
    if (key === this.clockKey) return;
    this.clockKey = key;
    this.clock.hidden = false;
    replaceWithChildren(
      this.clock,
      h(
        'div',
        {class: 'map-furniture-clock-main'},
        h('span', {class: 'map-furniture-clock-digits'}, main),
        show === 'date'
          ? null
          : h('span', {class: 'map-furniture-clock-zone'}, primaryTime.zoneName)
      ),
      show === 'datetime' ? h('div', {class: 'map-furniture-clock-date'}, primaryDate) : null,
      ...secondaryLines.map(line =>
        h(
          'div',
          {class: 'map-furniture-clock-line'},
          h('span', {class: 'map-furniture-clock-line-zone'}, line.zoneName),
          line.text
        )
      ),
      fraction < 0
        ? null
        : h(
            'div',
            {class: 'map-furniture-clock-progress'},
            h('span', {style: {width: `${(fraction * 100).toFixed(1)}%`}})
          )
    );
  }

  private getMaxBarPixels(): number {
    if (this.maxWidthOverride !== undefined) return this.maxWidthOverride;
    return this.narrowQuery?.matches ? MAX_BAR_PIXELS_NARROW : MAX_BAR_PIXELS;
  }
}

/** Replaces the children of `parent`; `null` entries are skipped (as in `h`). */
function replaceWithChildren(
  parent: HTMLElement,
  ...children: (HTMLElement | null | readonly (HTMLElement | null)[])[]
): void {
  parent.replaceChildren(...h('div', null, ...children).childNodes);
}

function formatLatitude(latitude: number): string {
  return `${Math.round(Math.abs(latitude))}° ${latitude < 0 ? 'S' : 'N'}`;
}

/** One scale bar: parameter tick labels above, the bar with ticks, numerals below. */
function createScaleRow(measure: ScaleBarMeasure): HTMLElement {
  const width = Math.max(measure.widthPixels, 8);
  let lastLabelOffset = Number.NEGATIVE_INFINITY;
  const tickLabels: HTMLElement[] = [];
  const tickMarks: HTMLElement[] = [];
  for (const tick of measure.ticks) {
    tickMarks.push(
      h('div', {class: 'map-furniture-scale-tick', style: {left: `${tick.offsetPixels}px`}})
    );
    // Label a tick only when it clears the previous label; the mark is always drawn.
    if (tick.offsetPixels - lastLabelOffset < MIN_TICK_LABEL_GAP_PIXELS) continue;
    lastLabelOffset = tick.offsetPixels;
    tickLabels.push(
      h(
        'div',
        {class: 'map-furniture-scale-tick-label', style: {left: `${tick.offsetPixels}px`}},
        tick.label
      )
    );
  }
  const [zero, middle, maximum] = measure.numerals;
  return h(
    'div',
    {class: 'map-furniture-scale-row', style: {width: `${width}px`}},
    measure.ticks.length > 0 ? h('div', {class: 'map-furniture-scale-ticks'}, tickLabels) : null,
    h('div', {class: 'map-furniture-scale-bar'}, tickMarks),
    h(
      'div',
      {class: 'map-furniture-scale-numerals'},
      h('span', {class: 'is-start'}, zero),
      h('span', {class: 'is-middle'}, middle),
      h('span', {class: 'is-end'}, maximum)
    )
  );
}
