// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * The CPU side of sun-and-shadow: the float64 sun path of a day, the skyline of one cell (the
 * `horizonAngles` probe of `cpu-dem.ts`), the lit / shadow / night timeline of that cell, and the
 * two SVG diagrams that show them (the polar sun-path and skyline plot, and the 24 hour strip).
 * They answer for one cell what the GPU answers for every cell, with the same definitions, so the
 * reader can check the map against the arithmetic.
 */

import {
  getGPUSolarIrradianceSunTable,
  getSolarPosition,
  GPU_SOLAR_DISK_ANGULAR_RADIUS_DEGREES,
  GPU_SOLAR_IRRADIANCE_SUN_TABLE_STRIDE
} from '@luma.gl/experimental/gpu-terrain';
import type {DiagramChartData} from '../chart-types';
import type {DemProbe} from './cpu-dem';
import {getCompassName} from './b14b-terrain';
import {formatDay, formatHour, getUtcFromZurich, type SunOptions} from './sun-and-shadow.style';

/** Directions of the CPU skyline (the GPU horizon map stores 16 of them by default). */
export const SKYLINE_DIRECTIONS = 32;

/** Sun altitude, degrees, below which it is night (the GPU daylight flag's default threshold). */
const NIGHT_ALTITUDE_DEGREES = -0.833;

/** Minutes between the samples of the CPU day. */
const SAMPLE_MINUTES = 5;

/** The gold of the sun. */
const GOLD = '#F2B134';
/** The indigo of the shadow band: dark enough for paper, light enough for the dark card. */
const SHADOW_BAND = '#3D4C8F';

/** One sample of the sun's path across a local day. */
export type SunSample = {
  /** Local civil hours (Europe/Zurich) at the middle of the sample. */
  hour: number;
  /** Degrees clockwise from north. */
  azimuth: number;
  /** Degrees above the horizon, refracted when asked. */
  altitude: number;
};

/** The sun's path across one local day, one sample per `stepMinutes`, evaluated at each midpoint. */
export function getSunSamples(
  dayOfYear: number,
  lngLat: readonly [number, number],
  refraction: boolean,
  stepMinutes = SAMPLE_MINUTES
): SunSample[] {
  const count = Math.round(1440 / stepMinutes);
  return Array.from({length: count}, (_, index) => {
    const hour = ((index + 0.5) * stepMinutes) / 60;
    const position = getSolarPosition(getUtcFromZurich(dayOfYear, hour), lngLat[0], lngLat[1], {
      refraction
    });
    return {hour, azimuth: position.azimuthDegrees, altitude: position.altitudeDegrees};
  });
}

/** What the date contributes at a place: day length, sunrise and sunset on a flat horizon, direct energy. */
export type DayFacts = {
  dayLengthHours: number;
  sunriseHour: number | null;
  sunsetHour: number | null;
  /** Direct clear-sky energy in kWh/m² on a surface that faces the sun all day. */
  directEnergyKilowattHours: number;
};

/**
 * Day length, sunrise and sunset (sun above a flat horizon) and the direct energy of a day at a
 * place, from the same sun table that `GPUSolarIrradiance` integrates.
 */
export function getDayFacts(
  dayOfYear: number,
  lngLat: readonly [number, number],
  options: Pick<SunOptions, 'refraction' | 'directIrradiance'>
): DayFacts {
  const samples = getSunSamples(dayOfYear, lngLat, options.refraction);
  const up = samples.filter(sample => sample.altitude > 0);
  const start = getUtcFromZurich(dayOfYear, 0);
  const table = getGPUSolarIrradianceSunTable({
    longitude: lngLat[0],
    latitude: lngLat[1],
    start,
    end: start + 24 * 3600 * 1000,
    stepMinutes: 10,
    directNormalIrradiance:
      options.directIrradiance === 'meinel' ? 'meinel' : Number(options.directIrradiance),
    refraction: options.refraction
  });
  let energy = 0;
  const stride = GPU_SOLAR_IRRADIANCE_SUN_TABLE_STRIDE;
  for (let row = 0; row < table.sampleCount; row++) {
    // Rows are [azimuth, altitude, duration hours, direct normal irradiance W/m²].
    energy += table.values[row * stride + 2] * table.values[row * stride + 3];
  }
  return {
    dayLengthHours: (up.length * SAMPLE_MINUTES) / 60,
    sunriseHour: up[0]?.hour ?? null,
    sunsetHour: up[up.length - 1]?.hour ?? null,
    directEnergyKilowattHours: energy / 1000
  };
}

// ---------------------------------------------------------------------------------------------
// Skyline and timeline of one cell
// ---------------------------------------------------------------------------------------------

/**
 * The skyline of a place: the horizon elevation angle in `SKYLINE_DIRECTIONS` evenly spaced
 * directions from north, out to `radiusMeters`, with the earth's curvature and no eye height (the
 * same cell-level horizon the GPU stores per sector).
 */
export function getSkyline(
  probe: DemProbe,
  lngLat: readonly [number, number],
  radiusMeters: number
): Float64Array {
  return probe.horizonAngles(lngLat, SKYLINE_DIRECTIONS, radiusMeters, {eyeHeight: 0});
}

/** The horizon angle at an azimuth, interpolated linearly between the two nearest of `angles`. */
export function interpolateHorizon(angles: ArrayLike<number>, azimuthDegrees: number): number {
  const count = angles.length;
  const position = ((((azimuthDegrees % 360) + 360) % 360) / 360) * count;
  const lower = Math.floor(position) % count;
  const upper = (lower + 1) % count;
  const fraction = position - Math.floor(position);
  return angles[lower] * (1 - fraction) + angles[upper] * fraction;
}

/** Lit, shadowed and night intervals of one cell across a local day. */
export type PixelTimeline = {
  stepHours: number;
  /** Local hours at the middle of each sample. */
  hours: number[];
  /** `0` night, `1` the sun is up but behind the terrain, `2` direct sun. */
  state: number[];
  /** Sun altitude, degrees. */
  altitude: number[];
  /** Hours of direct sun (the visible share of the disc, integrated). */
  sunHours: number;
  /** Local hour of the first and last lit sample, or null on a day without sun. */
  firstSun: number | null;
  lastSun: number | null;
};

/**
 * Computes the timeline of a cell on the CPU: at each sample the share of the sun's disc above the
 * cell's skyline (`GPUSolarShadowMask` does this per pixel with a 16 sector horizon map), summed
 * into sun hours (`GPUSolarIrradiance`).
 *
 * @param skyline Horizon angles from {@link getSkyline}.
 * @param softness The disc radius as a multiple of the real sun's (0 is a point).
 */
export function getPixelTimeline(
  samples: readonly SunSample[],
  skyline: ArrayLike<number>,
  softness: number
): PixelTimeline {
  const stepHours = SAMPLE_MINUTES / 60;
  const radius = GPU_SOLAR_DISK_ANGULAR_RADIUS_DEGREES * softness;
  const hours: number[] = [];
  const state: number[] = [];
  const altitude: number[] = [];
  let sunHours = 0;
  let firstSun: number | null = null;
  let lastSun: number | null = null;
  for (const sample of samples) {
    const horizon = interpolateHorizon(skyline, sample.azimuth);
    const visible =
      radius > 0
        ? Math.min(1, Math.max(0, (sample.altitude + radius - horizon) / (2 * radius)))
        : sample.altitude > horizon
          ? 1
          : 0;
    sunHours += visible * stepHours;
    const lit = visible >= 0.5;
    if (lit) {
      firstSun ??= sample.hour;
      lastSun = sample.hour;
    }
    hours.push(sample.hour);
    altitude.push(sample.altitude);
    state.push(lit ? 2 : sample.altitude > NIGHT_ALTITUDE_DEGREES ? 1 : 0);
  }
  return {stepHours, hours, state, altitude, sunHours, firstSun, lastSun};
}

/** `"09:25"` of a local hour, or `"none"` when there is no sun. */
export function formatClockHour(hour: number | null): string {
  return hour === null ? 'none' : formatHour(hour);
}

// ---------------------------------------------------------------------------------------------
// Diagrams
// ---------------------------------------------------------------------------------------------

const SKY = {width: 320, height: 214, centerX: 104, centerY: 110, radius: 82} as const;

/** Position on the polar plot: azimuth around from the top (north), altitude inward from the rim. */
function toPolar(azimuthDegrees: number, altitudeDegrees: number): [number, number] {
  const distance = SKY.radius * (1 - Math.min(90, Math.max(0, altitudeDegrees)) / 90);
  const angle = (azimuthDegrees * Math.PI) / 180;
  return [SKY.centerX + distance * Math.sin(angle), SKY.centerY - distance * Math.cos(angle)];
}

const fixed = (value: number) => value.toFixed(1);

/** A path through the points; a gap (`null`) lifts the pen. */
function getPolyline(points: readonly ([number, number] | null)[]): string {
  let path = '';
  let pen = false;
  for (const point of points) {
    if (!point) {
      pen = false;
      continue;
    }
    path += `${pen ? 'L' : 'M'}${fixed(point[0])} ${fixed(point[1])}`;
    pen = true;
  }
  return path;
}

/** Input of {@link getSkyDiagram}. */
export type SkyDiagramInput = {
  /** Place the sun path and skyline belong to, shown at the top right. */
  placeName: string;
  /** Sun paths to draw (the three reference dates and the shown date), local day samples each. */
  paths: readonly {dayOfYear: number; samples: readonly SunSample[]; current: boolean}[];
  /** The sun now, or null. */
  now: {azimuth: number; altitude: number} | null;
  /** Local time of `now`. */
  timeLabel: string;
  /** Skyline of the place, or null while it is being computed. */
  skyline: ArrayLike<number> | null;
  /** The horizon angle at each of the GPU's sectors (dots), or null. */
  sectors: ArrayLike<number> | null;
};

/**
 * The polar sun-path and skyline plot. North is up and east is to the right (a plan view of the
 * sky); the rim is the horizon and the centre the zenith. The grey ring is the terrain's skyline:
 * the sun is behind the mountains exactly where its dot sits inside the grey.
 */
export function getSkyDiagram(input: SkyDiagramInput): DiagramChartData {
  const {centerX, centerY, radius} = SKY;
  const parts: string[] = [];
  // Altitude rings (30 and 60 degrees) and the four spokes.
  for (const altitude of [30, 60]) {
    const ring = radius * (1 - altitude / 90);
    parts.push(
      `<circle cx="${centerX}" cy="${centerY}" r="${fixed(ring)}" class="diagram-muted" stroke-opacity="0.45" stroke-dasharray="2 3"/>`,
      `<text x="${centerX + 3}" y="${fixed(centerY - ring + 9)}" class="diagram-muted" font-size="8">${altitude}°</text>`
    );
  }
  parts.push(
    `<path d="M${centerX} ${centerY - radius}V${centerY + radius}M${centerX - radius} ${centerY}H${centerX + radius}" class="diagram-muted" stroke-opacity="0.35"/>`
  );
  // Skyline: the ring between the horizon circle and the terrain's angle, drawn even-odd.
  if (input.skyline) {
    const inner: string[] = [];
    for (let degrees = 0; degrees < 360; degrees += 3) {
      const [x, y] = toPolar(degrees, Math.max(0, interpolateHorizon(input.skyline, degrees)));
      inner.push(`${inner.length ? 'L' : 'M'}${fixed(x)} ${fixed(y)}`);
    }
    parts.push(
      `<path fill-rule="evenodd" fill="currentColor" fill-opacity="0.34" stroke="none" d="M${centerX - radius} ${centerY}a${radius} ${radius} 0 1 0 ${2 * radius} 0a${radius} ${radius} 0 1 0 ${-2 * radius} 0Z${inner.join('')}Z"/>`
    );
  }
  parts.push(
    `<circle cx="${centerX}" cy="${centerY}" r="${radius}" class="diagram-ink" stroke-width="1.1"/>`
  );
  // The values the GPU stores: one dot per sector at the horizon angle.
  if (input.sectors) {
    const count = input.sectors.length;
    for (let sector = 0; sector < count; sector++) {
      const azimuth = (sector * 360) / count;
      const [x, y] = toPolar(azimuth, Math.max(0, input.sectors[sector]));
      parts.push(
        `<circle cx="${fixed(x)}" cy="${fixed(y)}" r="1.9" fill="currentColor" stroke="none"/>`
      );
    }
  }
  // Compass letters.
  const letters: [string, number][] = [
    ['N', 0],
    ['E', 90],
    ['S', 180],
    ['W', 270]
  ];
  for (const [letter, azimuth] of letters) {
    const angle = (azimuth * Math.PI) / 180;
    parts.push(
      `<text x="${fixed(centerX + (radius + 9) * Math.sin(angle))}" y="${fixed(centerY - (radius + 9) * Math.cos(angle) + 3.5)}" text-anchor="middle" class="diagram-muted" font-size="10">${letter}</text>`
    );
  }
  // Sun paths: reference dates thin, the shown date in gold on an ink casing.
  const ordered = [...input.paths].sort((a, b) => Number(a.current) - Number(b.current));
  for (const path of ordered) {
    const points = path.samples.map(sample =>
      sample.altitude > 0 ? toPolar(sample.azimuth, sample.altitude) : null
    );
    const d = getPolyline(points);
    if (!d) continue;
    if (path.current) {
      parts.push(
        `<path d="${d}" fill="none" stroke="currentColor" stroke-opacity="0.35" stroke-width="4.2" stroke-linecap="round" stroke-linejoin="round"/>`,
        `<path d="${d}" fill="none" stroke="${GOLD}" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/>`
      );
    } else {
      parts.push(
        `<path d="${d}" class="diagram-muted" stroke-width="1.3" stroke-linecap="round" stroke-opacity="0.9"/>`
      );
    }
    // The label sits under the highest point of the path.
    let apex = path.samples[0];
    for (const sample of path.samples) if (sample.altitude > apex.altitude) apex = sample;
    if (apex.altitude > 4) {
      const [x, y] = toPolar(apex.azimuth, apex.altitude);
      parts.push(
        `<text x="${fixed(x + 4)}" y="${fixed(y + 10)}" class="diagram-muted" font-size="8.5">${formatDay(path.dayOfYear)}</text>`
      );
    }
  }
  // The sun now.
  let status = 'No sun yet: below the horizon';
  if (input.now && input.now.altitude > NIGHT_ALTITUDE_DEGREES) {
    const [x, y] = toPolar(input.now.azimuth, Math.max(0, input.now.altitude));
    parts.push(
      `<circle cx="${fixed(x)}" cy="${fixed(y)}" r="5.2" fill="${GOLD}" class="diagram-ink" stroke-width="1.6"/>`
    );
    if (input.skyline) {
      status =
        input.now.altitude > interpolateHorizon(input.skyline, input.now.azimuth)
          ? 'In the open sky'
          : 'Behind the mountains';
    } else {
      status = 'Above the horizon';
    }
  }
  // Key on the right.
  const keyX = 206;
  parts.push(
    `<text x="${keyX}" y="22" class="diagram-ink" font-size="11" font-weight="600">${input.placeName}</text>`,
    `<text x="${keyX}" y="38" class="diagram-muted" font-size="9.5">Sun path and skyline</text>`,
    `<circle cx="${keyX + 5}" cy="58" r="4.6" fill="${GOLD}" class="diagram-ink" stroke-width="1.4"/>`,
    `<text x="${keyX + 15}" y="61.5" class="diagram-ink" font-size="10">Sun, ${input.timeLabel}</text>`,
    `<rect x="${keyX}" y="74" width="10" height="9" fill="currentColor" fill-opacity="0.34" stroke="currentColor" stroke-opacity="0.5"/>`,
    `<text x="${keyX + 15}" y="82" class="diagram-ink" font-size="10">Skyline</text>`,
    `<path d="M${keyX} 98H${keyX + 10}" fill="none" stroke="${GOLD}" stroke-width="2.4"/>`,
    `<text x="${keyX + 15}" y="101.5" class="diagram-ink" font-size="10">Shown date</text>`,
    `<path d="M${keyX} 116H${keyX + 10}" class="diagram-muted" stroke-width="1.3"/>`,
    `<text x="${keyX + 15}" y="119.5" class="diagram-ink" font-size="10">Other dates</text>`,
    `<text x="${keyX}" y="146" class="diagram-ink" font-size="10.5" font-weight="600">${status}</text>`
  );
  if (input.now && input.now.altitude > NIGHT_ALTITUDE_DEGREES) {
    parts.push(
      `<text x="${keyX}" y="161" class="diagram-muted" font-size="9.5">${input.now.altitude.toFixed(1)}° high</text>`,
      `<text x="${keyX}" y="174" class="diagram-muted" font-size="9.5">toward ${getCompassName(input.now.azimuth)} (${Math.round(input.now.azimuth)}°)</text>`
    );
  }
  return {
    kind: 'diagram',
    width: SKY.width,
    height: SKY.height,
    description:
      'A polar plot of the sky seen from above, north at the top. Grey marks the terrain skyline; curves are the sun paths on three dates and the shown date; a gold dot is the sun now. The sun is behind the mountains where its dot lies inside the grey.',
    svg: parts.join('\n')
  };
}

const STRIP = {width: 320, height: 112, left: 14, right: 310, bandTop: 28, bandHeight: 18} as const;

/** Input of {@link getTimelineDiagram}. */
export type TimelineDiagramInput = {
  timeline: PixelTimeline;
  /** Local hour of the playhead. */
  hour: number;
  /** Date label of the strip. */
  dayOfYear: number;
};

/**
 * The 24 hour strip of one cell: gold where the sun is on it, indigo where the sun is up behind
 * the terrain, pale where it is night, with the sun's altitude under it and a playhead at the shown
 * time.
 */
export function getTimelineDiagram(input: TimelineDiagramInput): DiagramChartData {
  const {timeline} = input;
  const {left, right, bandTop, bandHeight} = STRIP;
  const x = (hour: number) => left + ((right - left) * hour) / 24;
  const parts: string[] = [];
  // Legend row.
  parts.push(
    `<rect x="${left}" y="5" width="9" height="9" fill="${GOLD}"/>`,
    `<text x="${left + 13}" y="13" class="diagram-ink" font-size="9.5">In sun</text>`,
    `<rect x="${left + 62}" y="5" width="9" height="9" fill="${SHADOW_BAND}"/>`,
    `<text x="${left + 75}" y="13" class="diagram-ink" font-size="9.5">Behind terrain</text>`,
    `<rect x="${left + 160}" y="5" width="9" height="9" fill="currentColor" fill-opacity="0.12" stroke="currentColor" stroke-opacity="0.35"/>`,
    `<text x="${left + 173}" y="13" class="diagram-ink" font-size="9.5">Night</text>`
  );
  // State runs.
  const colours: Record<number, string> = {
    0: 'fill="currentColor" fill-opacity="0.12"',
    1: `fill="${SHADOW_BAND}"`,
    2: `fill="${GOLD}"`
  };
  let runStart = 0;
  for (let index = 1; index <= timeline.state.length; index++) {
    if (index === timeline.state.length || timeline.state[index] !== timeline.state[runStart]) {
      const from = runStart * timeline.stepHours;
      const to = index * timeline.stepHours;
      parts.push(
        `<rect x="${fixed(x(from))}" y="${bandTop}" width="${fixed(x(to) - x(from) + 0.3)}" height="${bandHeight}" ${colours[timeline.state[runStart]]}/>`
      );
      runStart = index;
    }
  }
  parts.push(
    `<rect x="${left}" y="${bandTop}" width="${right - left}" height="${bandHeight}" fill="none" stroke="currentColor" stroke-opacity="0.4"/>`
  );
  // Sun altitude under the band.
  const curveTop = bandTop + bandHeight + 8;
  const curveHeight = 30;
  const maximum = Math.max(10, ...timeline.altitude);
  const points = timeline.hours.map((hour, index) => {
    const altitude = Math.max(0, timeline.altitude[index]);
    return `${fixed(x(hour))} ${fixed(curveTop + curveHeight * (1 - altitude / maximum))}`;
  });
  parts.push(
    `<path d="M${fixed(x(timeline.hours[0]))} ${curveTop + curveHeight}L${points.join('L')}L${fixed(x(timeline.hours[timeline.hours.length - 1]))} ${curveTop + curveHeight}Z" fill="currentColor" fill-opacity="0.1" stroke="none"/>`,
    `<path d="M${points.join('L')}" class="diagram-muted" stroke-width="1.2"/>`,
    `<text x="${right}" y="${curveTop + 8}" text-anchor="end" class="diagram-muted" font-size="8.5">sun altitude</text>`
  );
  // Hour ticks.
  for (let hour = 0; hour <= 24; hour += 3) {
    parts.push(
      `<path d="M${fixed(x(hour))} ${curveTop + curveHeight}v3" class="diagram-muted"/>`,
      `<text x="${fixed(x(hour))}" y="${curveTop + curveHeight + 13}" text-anchor="middle" class="diagram-muted" font-size="9">${hour}</text>`
    );
  }
  // Playhead.
  const head = x(Math.min(24, Math.max(0, input.hour)));
  parts.push(
    `<path d="M${fixed(head)} ${bandTop - 3}V${curveTop + curveHeight}" fill="none" stroke="currentColor" stroke-width="1.6"/>`,
    `<path d="M${fixed(head - 4)} ${bandTop - 8}h8l-4 6z" fill="currentColor" stroke="none"/>`
  );
  const flip = head > right - 40;
  parts.push(
    `<text x="${fixed(head + (flip ? -6 : 6))}" y="${bandTop - 6}" text-anchor="${flip ? 'end' : 'start'}" class="diagram-ink" font-size="9.5" font-weight="600">${formatHour(input.hour)}</text>`
  );
  return {
    kind: 'diagram',
    width: STRIP.width,
    height: STRIP.height,
    description: `A 24 hour strip for ${formatDay(input.dayOfYear)}: gold where the sun reaches the cell, indigo while it is up behind the terrain, pale at night, with the sun's altitude below and a marker at the shown time.`,
    svg: parts.join('\n')
  };
}
