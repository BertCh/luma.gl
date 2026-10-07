// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Live text (rule 8): numbers in prose come from readouts, never from typed constants. A tiny
 * template function plus number formatters with units, thousands separators (`en-US`) and a true
 * minus sign. Pure TypeScript.
 */

/** Placeholder shown for a missing value (an en dash). */
export const MISSING_VALUE = '–';

const MINUS = '−';
const THIN_SPACE = ' ';
const NO_BREAK_SPACE = ' ';

/** Options shared by the formatters. */
export type FormatOptions = {
  /** Group thousands with a thin space (`12 345`) instead of a comma. Default false. */
  thinSpace?: boolean;
};

function group(text: string, options?: FormatOptions): string {
  return options?.thinSpace ? text.replace(/,/g, THIN_SPACE) : text;
}

/** Fixed-digit number with `en-US` grouping and a true minus sign; no unit. */
function formatNumber(value: number, digits: number, options?: FormatOptions): string {
  const text = value.toLocaleString('en-US', {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits
  });
  return group(text.replace('-', MINUS), options);
}

/** Digits that keep about three significant figures: 2 below 10, 1 below 100, else 0. */
function getAutoDigits(value: number): number {
  const magnitude = Math.abs(value);
  return magnitude < 10 ? 2 : magnitude < 100 ? 1 : 0;
}

/**
 * A rounded integer with thousands separators.
 *
 * @example
 * formatCount(1234567); // '1,234,567'
 * formatCount(1234567, {thinSpace: true}); // '1 234 567'
 */
export function formatCount(value: number, options?: FormatOptions): string {
  if (!Number.isFinite(value)) return MISSING_VALUE;
  return formatNumber(Math.round(value), 0, options);
}

/**
 * A share (0 to 1) as a percentage. A non-zero share that would round to zero shows as
 * `<1%` (`<0.1%` with one digit), never as `0%`.
 *
 * @example
 * formatPercent(0.4567); // '46%'
 * formatPercent(0.4567, 1); // '45.7%'
 * formatPercent(0.0004); // '<1%'
 */
export function formatPercent(share: number, digits = 0, options?: FormatOptions): string {
  if (!Number.isFinite(share)) return MISSING_VALUE;
  const percent = share * 100;
  const step = 10 ** -digits;
  if (percent !== 0 && Math.abs(percent) < step / 2) {
    return `<${formatNumber(step, digits, options)}%`;
  }
  return `${formatNumber(percent, digits, options)}%`;
}

/**
 * A distance in metres, as m below 1 km and km above, with digits that suit the size.
 *
 * @example
 * formatDistance(640); // '640 m'
 * formatDistance(1250); // '1.3 km'
 * formatDistance(42_300); // '42.3 km'
 * formatDistance(1_250_000); // '1,250 km'
 */
export function formatDistance(meters: number, options?: FormatOptions): string {
  if (!Number.isFinite(meters)) return MISSING_VALUE;
  const magnitude = Math.abs(meters);
  const metersDigits = magnitude < 10 ? 1 : 0;
  const rounded = Math.round(magnitude * 10 ** metersDigits) / 10 ** metersDigits;
  if (rounded < 1000) {
    return `${formatNumber(meters, metersDigits, options)}${NO_BREAK_SPACE}m`;
  }
  const kilometers = meters / 1000;
  const digits = Math.abs(kilometers) < 100 ? 1 : 0;
  return `${formatNumber(kilometers, digits, options)}${NO_BREAK_SPACE}km`;
}

/**
 * An area in square metres as m², hectares or km² (switching at 1 ha and 1 km²).
 *
 * @example
 * formatArea(850); // '850 m²'
 * formatArea(48_000); // '4.8 ha'
 * formatArea(2_590_000); // '2.6 km²'
 */
export function formatArea(squareMeters: number, options?: FormatOptions): string {
  if (!Number.isFinite(squareMeters)) return MISSING_VALUE;
  const magnitude = Math.abs(squareMeters);
  if (magnitude < 10_000) {
    return `${formatNumber(squareMeters, 0, options)}${NO_BREAK_SPACE}m²`;
  }
  if (magnitude < 1_000_000) {
    const hectares = squareMeters / 10_000;
    return `${formatNumber(hectares, Math.abs(hectares) < 100 ? 1 : 0, options)}${NO_BREAK_SPACE}ha`;
  }
  const squareKilometers = squareMeters / 1_000_000;
  const digits = Math.abs(squareKilometers) < 100 ? 1 : 0;
  return `${formatNumber(squareKilometers, digits, options)}${NO_BREAK_SPACE}km²`;
}

/**
 * A rate as "value per unit" with about three significant figures.
 *
 * @example
 * formatRate(12.345, '1,000 residents'); // '12.3 per 1,000 residents'
 * formatRate(0.8, 'km²'); // '0.80 per km²'
 */
export function formatRate(value: number, perUnit: string, options?: FormatOptions): string {
  if (!Number.isFinite(value)) return MISSING_VALUE;
  return `${formatNumber(value, getAutoDigits(value), options)} per ${perUnit}`;
}

/**
 * A duration in seconds as its two largest units.
 *
 * @example
 * formatDuration(45); // '45 s'
 * formatDuration(750); // '12 min 30 s'
 * formatDuration(7500); // '2 h 5 min'
 * formatDuration(97_200); // '1 d 3 h'
 * formatDuration(0.25); // '250 ms'
 */
export function formatDuration(seconds: number): string {
  if (!Number.isFinite(seconds)) return MISSING_VALUE;
  const sign = seconds < 0 ? MINUS : '';
  const magnitude = Math.abs(seconds);
  if (magnitude < 1) return `${sign}${Math.round(magnitude * 1000)}${NO_BREAK_SPACE}ms`;
  const total = Math.round(magnitude);
  const days = Math.floor(total / 86_400);
  const hours = Math.floor((total % 86_400) / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const rest = total % 60;
  const units: [number, string][] = [
    [days, 'd'],
    [hours, 'h'],
    [minutes, 'min'],
    [rest, 's']
  ];
  const first = units.findIndex(([amount]) => amount > 0);
  if (first < 0) return `0${NO_BREAK_SPACE}s`;
  const parts = units
    .slice(first, first + 2)
    .filter(([amount]) => amount > 0)
    .map(([amount, unit]) => `${amount}${NO_BREAK_SPACE}${unit}`);
  return sign + parts.join(' ');
}

/**
 * A number with an explicit sign and a true minus (U+2212); zero has no sign.
 *
 * @example
 * formatSigned(4.2, 1); // '+4.2'
 * formatSigned(-3); // '−3'
 */
export function formatSigned(value: number, digits = 0, options?: FormatOptions): string {
  if (!Number.isFinite(value)) return MISSING_VALUE;
  const text = formatNumber(Math.abs(value), digits, options);
  if (Number(text.replace(/[^0-9.]/g, '')) === 0) return text;
  return `${value < 0 ? MINUS : '+'}${text}`;
}

/**
 * An integer with its English ordinal suffix.
 *
 * @example
 * formatOrdinal(94); // '94th'
 * formatOrdinal(11); // '11th'
 * `${formatOrdinal(22)} percentile`; // '22nd percentile'
 */
export function formatOrdinal(value: number): string {
  if (!Number.isFinite(value)) return MISSING_VALUE;
  const n = Math.round(value);
  const lastTwo = Math.abs(n) % 100;
  const last = lastTwo % 10;
  const suffix =
    lastTwo >= 11 && lastTwo <= 13
      ? 'th'
      : last === 1
        ? 'st'
        : last === 2
          ? 'nd'
          : last === 3
            ? 'rd'
            : 'th';
  return `${n < 0 ? MINUS : ''}${formatCount(Math.abs(n))}${suffix}`;
}

/** Values a template can use. */
export type LiveTextValues = Record<string, string | number | null | undefined>;

/** Options of {@link liveText}. */
export type LiveTextOptions = FormatOptions & {
  /** Text for a missing value. Default {@link MISSING_VALUE}. */
  missing?: string;
};

function applyFormat(value: number, format: string): string {
  const [name, argument] = format.split(':');
  const digits = argument === undefined ? undefined : Number(argument);
  switch (name) {
    case 'integer':
    case 'count':
      return formatCount(value);
    case 'percent':
      return formatPercent(value, digits ?? 0);
    case 'signed':
      return formatSigned(value, digits ?? 0);
    case 'fixed':
      return formatNumber(value, digits ?? 1);
    case 'ordinal':
      return formatOrdinal(value);
    case 'distance':
      return formatDistance(value);
    case 'area':
      return formatArea(value);
    case 'duration':
      return formatDuration(value);
    case 'km':
      return `${formatNumber(value / 1000, digits ?? (Math.abs(value) < 100_000 ? 1 : 0))}${NO_BREAK_SPACE}km`;
    case 'm':
      return `${formatNumber(value, digits ?? 0)}${NO_BREAK_SPACE}m`;
    default:
      return String(value);
  }
}

/**
 * Fills `{name}` placeholders from `values`. A placeholder may name a formatter, `{name:format}`
 * or `{name:format:digits}`. A missing, null, undefined or non-finite value renders as a visible
 * dash, never as `undefined` or `NaN`. `{{` and `}}` print literal braces.
 *
 * Formats (numbers only; a string value is inserted as it is):
 * `integer` or `count`, `percent` (the value is a share 0-1), `signed`, `fixed` (default 1 digit),
 * `ordinal`, `distance` (metres, m or km), `km` (metres shown as km), `m` (metres), `area`
 * (square metres) and `duration` (seconds). An unknown format inserts the plain number.
 *
 * @example
 * liveText('{n:integer} airports, {share:percent:1} of them above {elev:m}', {
 *   n: 3257, share: 0.1234, elev: 1500
 * }); // '3,257 airports, 12.3% of them above 1,500 m'
 * liveText('Median {value:km}', {value: undefined}); // 'Median –'
 */
export function liveText(
  template: string,
  values: LiveTextValues,
  options: LiveTextOptions = {}
): string {
  const missing = options.missing ?? MISSING_VALUE;
  return template.replace(
    /\{\{|\}\}|\{([A-Za-z_][\w.-]*)(?::([^{}]*))?\}/g,
    (match, name: string | undefined, format: string | undefined) => {
      if (match === '{{') return '{';
      if (match === '}}') return '}';
      const value = values[name as string];
      if (value === null || value === undefined) return missing;
      if (typeof value === 'string') return value;
      if (!Number.isFinite(value)) return missing;
      const text = format
        ? applyFormat(value, format)
        : formatNumber(value, Number.isInteger(value) ? 0 : getAutoDigits(value));
      return options.thinSpace ? text.replace(/,/g, THIN_SPACE) : text;
    }
  );
}
