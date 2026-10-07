// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {ChartData, ChartLink} from '../scenes/chart-types';
import type {OptionSpec, OptionValue, ReadoutSpec, StoryStep} from '../scenes/scene';
import {escapeHtml, formatBytes} from './dom';
import {renderMarkdown} from './markdown';

/** Placeholder characters that survive Markdown rendering untouched (private-use code points). */
const TOKEN_OPEN = '';
const TOKEN_CLOSE = '';
const READOUT_TOKEN = /\{\{\s*([\w-]+)\s*\}\}/g;
const READOUT_PLACEHOLDER = new RegExp(`${TOKEN_OPEN}([\\w-]+)${TOKEN_CLOSE}`, 'g');

/** Option and button ids a step shows: its own list, else the options it sets. */
export function getStepControlIds(step: StoryStep): readonly string[] {
  return step.controls ?? Object.keys(step.options ?? {});
}

/** Readout ids a step shows: its own list, else the highlighted readout. */
export function getStepReadoutIds(step: StoryStep): readonly string[] {
  return step.readouts ?? (step.highlight?.readout ? [step.highlight.readout] : []);
}

/** Readout ids a story step mentions as `{{id}}` in its body. */
export function getInterpolatedReadoutIds(body: string): string[] {
  return [...body.matchAll(READOUT_TOKEN)].map(match => match[1]);
}

/** The chart link of a readout chart, when it declares one. */
export function getChartLink(chart: ChartData | null | undefined): ChartLink | undefined {
  return (chart as {link?: ChartLink} | null | undefined)?.link;
}

/**
 * Formats a readout value for display: numbers by `spec.format`, with `spec.unit` appended;
 * strings as given; `null` and non-finite numbers as a dash.
 */
export function formatReadout(spec: ReadoutSpec, value: string | number | null): string {
  if (value === null || value === undefined) return '—';
  if (typeof value === 'string') return value;
  if (!Number.isFinite(value)) return '—';
  const text = formatNumber(spec, value);
  return spec.unit && spec.format !== 'bytes' && spec.format !== 'milliseconds'
    ? `${text} ${spec.unit}`
    : text;
}

function formatNumber(spec: ReadoutSpec, value: number): string {
  switch (spec.format) {
    case 'integer':
      return Math.round(value).toLocaleString('en-US');
    case 'decimal':
      return value.toLocaleString('en-US', {maximumFractionDigits: 3});
    case 'percent':
      return `${(value * 100).toFixed(1)}%`;
    case 'milliseconds':
      return `${value.toFixed(2)} ms`;
    case 'bytes':
      return formatBytes(value);
    case 'meters':
      return value >= 1000 ? `${(value / 1000).toFixed(2)} km` : `${value.toFixed(0)} m`;
    default:
      return String(value);
  }
}

/**
 * Renders step Markdown with live readout slots. Each `{{readoutId}}` becomes
 * `<span class="live-readout" data-readout="readoutId">` holding the current formatted value
 * (from `getText`), so the card updates the span's text when the readout changes instead of
 * re-rendering the whole card.
 */
export function renderStepBody(body: string, getText: (readoutId: string) => string): string {
  const html = renderMarkdown(
    body.replace(READOUT_TOKEN, (_match, id: string) => `${TOKEN_OPEN}${id}${TOKEN_CLOSE}`)
  );
  return html.replace(
    READOUT_PLACEHOLDER,
    (_match, id: string) =>
      `<span class="live-readout" data-readout="${escapeHtml(id)}">${escapeHtml(getText(id))}</span>`
  );
}

/** Reads a text value (action link parameter) as the type of the option it addresses. */
export function parseOptionText(spec: OptionSpec<never>, text: string): OptionValue | undefined {
  switch (spec.kind) {
    case 'toggle':
      return text === '1' || text === 'true';
    case 'slider': {
      const value = Number(text);
      return Number.isFinite(value) ? value : undefined;
    }
    case 'range': {
      const [low, high] = text.split('..').map(Number);
      return Number.isFinite(low) && Number.isFinite(high) ? ([low, high] as const) : undefined;
    }
    case 'select':
      return spec.options.some(option => option.value === text) ? text : undefined;
    default:
      return undefined;
  }
}

/** Snaps a linked chart value to the slider's step and clamps it to its range. */
export function snapToSlider(spec: OptionSpec<never>, value: number): number {
  if (spec.kind !== 'slider') return value;
  const step = spec.step > 0 ? spec.step : 0;
  const snapped = step ? spec.min + Math.round((value - spec.min) / step) * step : value;
  const clamped = Math.min(Math.max(snapped, spec.min), spec.max);
  // Trim float noise (0.1 + 0.2) to the precision of the step.
  const decimals = step ? Math.min(8, (String(step).split('.')[1] ?? '').length) : 6;
  return Number(clamped.toFixed(decimals));
}
