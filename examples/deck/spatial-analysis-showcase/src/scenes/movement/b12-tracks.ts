// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {LoadedDataset} from '../../data/catalog';

/**
 * Shared helpers of the movement chapter: a `TrackSet` (tracks in contributor layout with the
 * derived segment tables the layers need), the vessel category palette and a few formatters.
 */

/** Tracks in the layout `GPUTrajectoryMetrics` and friends expect, plus drawing tables. */
export type TrackSet = {
  trackCount: number;
  vertexCount: number;
  /** `trackCount + 1` row offsets. */
  offsets: Uint32Array;
  /** Planar `x, y` meters, one row per vertex. */
  positions: Float32Array;
  /** Longitude/latitude degrees, one row per vertex (same order as `positions`). */
  lngLat: Float32Array;
  /** Seconds since the dataset epoch, one row per vertex (float32 is exact below 2^24 s). */
  timestamps: Float32Array;
  /** `[first, last]` timestamp of the whole set. */
  timeRange: readonly [number, number];
  /** `[longitude, latitude]` of the planar origin. */
  origin: readonly [number, number];
  /** Lng/lat to planar meters. */
  project: (longitude: number, latitude: number) => [number, number];
  /** Planar meters to lng/lat. */
  unproject: (x: number, y: number) => [number, number];
  /** One row per consecutive vertex pair of one track: `x0, y0, x1, y1` in the set's drawing space. */
  segments: Float32Array;
  /** Track index per segment. */
  segmentTracks: Uint32Array;
  /** End vertex per segment: per-vertex step columns apply to the step that ends at a row. */
  segmentEndVertices: Uint32Array;
  segmentStartTimes: Float32Array;
  segmentEndTimes: Float32Array;
  segmentCount: number;
  /** Vertex count of the longest track. */
  longestTrack: number;
};

type TrackSetInput = {
  offsets: Uint32Array;
  positions: Float32Array;
  lngLat: Float32Array;
  timestamps: Float32Array;
  origin: readonly [number, number];
  project: TrackSet['project'];
  unproject: TrackSet['unproject'];
  /** Draw segments from lng/lat degrees (true) or from planar meters (false). */
  drawInDegrees: boolean;
};

/** Builds the segment tables of a track set. */
export function createTrackSet(input: TrackSetInput): TrackSet {
  const {offsets, positions, lngLat, timestamps} = input;
  const trackCount = offsets.length - 1;
  const vertexCount = timestamps.length;
  let segmentCount = 0;
  let longestTrack = 0;
  for (let track = 0; track < trackCount; track++) {
    const length = offsets[track + 1] - offsets[track];
    segmentCount += Math.max(0, length - 1);
    longestTrack = Math.max(longestTrack, length);
  }
  const source = input.drawInDegrees ? lngLat : positions;
  const segments = new Float32Array(segmentCount * 4);
  const segmentTracks = new Uint32Array(segmentCount);
  const segmentEndVertices = new Uint32Array(segmentCount);
  const segmentStartTimes = new Float32Array(segmentCount);
  const segmentEndTimes = new Float32Array(segmentCount);
  let row = 0;
  for (let track = 0; track < trackCount; track++) {
    const first = offsets[track];
    const last = offsets[track + 1] - 1;
    for (let vertex = first; vertex < last; vertex++, row++) {
      segments.set(source.subarray(vertex * 2, vertex * 2 + 4), row * 4);
      segmentTracks[row] = track;
      segmentEndVertices[row] = vertex + 1;
      segmentStartTimes[row] = timestamps[vertex];
      segmentEndTimes[row] = timestamps[vertex + 1];
    }
  }
  let first = Infinity;
  let last = -Infinity;
  for (let track = 0; track < trackCount; track++) {
    if (offsets[track + 1] > offsets[track]) {
      first = Math.min(first, timestamps[offsets[track]]);
      last = Math.max(last, timestamps[offsets[track + 1] - 1]);
    }
  }
  return {
    trackCount,
    vertexCount,
    offsets,
    positions,
    lngLat,
    timestamps,
    timeRange: [first, last],
    origin: input.origin,
    project: input.project,
    unproject: input.unproject,
    segments,
    segmentTracks,
    segmentEndVertices,
    segmentStartTimes,
    segmentEndTimes,
    segmentCount,
    longestTrack
  };
}

/** The AIS vessel tracks of New York / New Jersey Harbor, in local planar meters. */
export type VesselTrackSet = TrackSet & {
  /** Vessel category index per track (see {@link VESSEL_CATEGORIES}). */
  category: Uint32Array;
  /** MMSI per track. */
  mmsi: Uint32Array;
  /** Dense vessel index per track (a vessel can own several tracks). */
  vesselIndex: Uint32Array;
  /** Length in meters per track, 0 when unknown. */
  length: Float32Array;
  /** Reported speed over ground in knots per vertex. */
  speedKnots: Float32Array;
  /** Reported heading in degrees per vertex. */
  heading: Float32Array;
  vesselCount: number;
};

/** Loads the AIS dataset as a {@link VesselTrackSet}. */
export function loadVesselTracks(dataset: LoadedDataset): VesselTrackSet {
  const origin = dataset.defaultOrigin;
  const projection = dataset.getProjection(origin);
  const offsets = dataset.column<Uint32Array>('pathOffsets');
  const timestamps = Float32Array.from(dataset.column<Uint32Array>('timestamp'));
  const category = Uint32Array.from(dataset.column<Uint8Array>('category'));
  const set = createTrackSet({
    offsets,
    positions: dataset.projectColumn('vertices', origin),
    lngLat: dataset.column<Float32Array>('vertices'),
    timestamps,
    origin,
    project: (longitude, latitude) => projection.project(longitude, latitude),
    unproject: (x, y) => projection.unproject(x, y),
    drawInDegrees: false
  });
  const vesselIndex = dataset.column<Uint32Array>('vesselIndex');
  let vesselCount = 0;
  for (const value of vesselIndex) vesselCount = Math.max(vesselCount, value + 1);
  return {
    ...set,
    category,
    mmsi: dataset.column<Uint32Array>('mmsi'),
    vesselIndex,
    length: dataset.column<Float32Array>('length'),
    speedKnots: dataset.column<Float32Array>('speed'),
    heading: Float32Array.from(dataset.column<Uint16Array>('heading')),
    vesselCount
  };
}

const EARTH_RADIUS_METERS = 6371008.8;

/**
 * Spherical azimuthal-equidistant projection around `center`: distances from the center are exact
 * and distortion stays small across a continent, unlike a flat local projection.
 */
export function createAzimuthalEquidistant(center: readonly [number, number]): {
  project: (longitude: number, latitude: number) => [number, number];
  unproject: (x: number, y: number) => [number, number];
} {
  const lambda0 = (center[0] * Math.PI) / 180;
  const phi0 = (center[1] * Math.PI) / 180;
  const sinPhi0 = Math.sin(phi0);
  const cosPhi0 = Math.cos(phi0);
  return {
    project(longitude, latitude) {
      const lambda = (longitude * Math.PI) / 180;
      const phi = (latitude * Math.PI) / 180;
      const cosC = sinPhi0 * Math.sin(phi) + cosPhi0 * Math.cos(phi) * Math.cos(lambda - lambda0);
      const c = Math.acos(Math.min(1, Math.max(-1, cosC)));
      const k = c < 1e-9 ? 1 : c / Math.sin(c);
      return [
        EARTH_RADIUS_METERS * k * Math.cos(phi) * Math.sin(lambda - lambda0),
        EARTH_RADIUS_METERS *
          k *
          (cosPhi0 * Math.sin(phi) - sinPhi0 * Math.cos(phi) * Math.cos(lambda - lambda0))
      ];
    },
    unproject(x, y) {
      const rho = Math.hypot(x, y);
      if (rho < 1e-9) return [center[0], center[1]];
      const c = rho / EARTH_RADIUS_METERS;
      const phi = Math.asin(Math.cos(c) * sinPhi0 + ((y * Math.sin(c)) / rho) * cosPhi0);
      const lambda =
        lambda0 +
        Math.atan2(x * Math.sin(c), rho * cosPhi0 * Math.cos(c) - y * sinPhi0 * Math.sin(c));
      return [(lambda * 180) / Math.PI, (phi * 180) / Math.PI];
    }
  };
}

/** The gull tracks, in azimuthal-equidistant meters, drawn from longitude/latitude degrees. */
export type GullTrackSet = TrackSet & {
  /** Ground speed in m/s per vertex. */
  speed: Float32Array;
  /** Sex per bird: 0 female, 1 male, 2 unknown. */
  sex: Uint8Array;
  birdIds: readonly string[];
};

/** Loads the gull dataset as a {@link GullTrackSet}. */
export function loadGullTracks(dataset: LoadedDataset): GullTrackSet {
  const lngLat = dataset.column<Float32Array>('vertices');
  const bbox = dataset.manifest.bbox;
  const center: [number, number] = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2];
  const projection = createAzimuthalEquidistant(center);
  const positions = new Float32Array(lngLat.length);
  for (let index = 0; index < lngLat.length; index += 2) {
    const [x, y] = projection.project(lngLat[index], lngLat[index + 1]);
    positions[index] = x;
    positions[index + 1] = y;
  }
  const set = createTrackSet({
    offsets: dataset.column<Uint32Array>('pathOffsets'),
    positions,
    lngLat,
    timestamps: Float32Array.from(dataset.column<Uint32Array>('timestamp')),
    origin: center,
    project: projection.project,
    unproject: projection.unproject,
    drawInDegrees: true
  });
  return {
    ...set,
    speed: dataset.column<Float32Array>('speed'),
    sex: dataset.column<Uint8Array>('sex'),
    birdIds: (dataset.properties.birdIds as string[] | undefined) ?? []
  };
}

/** AIS vessel categories in the dataset's category order. */
export const VESSEL_CATEGORIES = [
  'cargo',
  'tanker',
  'passenger',
  'tug',
  'fishing',
  'pleasure',
  'other'
] as const;

export type VesselCategory = (typeof VESSEL_CATEGORIES)[number];

/** Display labels of the vessel categories. */
export const VESSEL_CATEGORY_LABELS: Record<VesselCategory, string> = {
  cargo: 'Cargo ship',
  tanker: 'Tanker',
  passenger: 'Passenger (ferry, tour boat)',
  tug: 'Tug or tow',
  fishing: 'Fishing',
  pleasure: 'Pleasure craft',
  other: 'Other'
};

/** One color per category (Okabe-Ito based), readable on light and dark basemaps. */
export const VESSEL_CATEGORY_COLORS: readonly (readonly [number, number, number, number])[] = [
  [240, 150, 30, 255],
  [214, 69, 65, 255],
  [86, 180, 233, 255],
  [0, 158, 115, 255],
  [240, 228, 66, 255],
  [170, 120, 220, 255],
  [150, 156, 168, 255]
];

/** Legend entries of the vessel categories. */
export function getVesselLegendEntries(): {
  color: readonly [number, number, number, number];
  label: string;
}[] {
  return VESSEL_CATEGORIES.map((name, index) => ({
    color: VESSEL_CATEGORY_COLORS[index],
    label: VESSEL_CATEGORY_LABELS[name]
  }));
}

/** Meters per second in one knot. */
export const METERS_PER_KNOT_SECOND = 1852 / 3600;
export const KNOTS_PER_METER_SECOND = 3600 / 1852;

/** `HH:MM` of a UTC clock in seconds since midnight (wraps at 24 h). */
export function formatUtcClock(seconds: number): string {
  const total = Math.floor(((seconds % 86400) + 86400) % 86400);
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}`;
}

/** `HH:MM:SS` of a UTC clock in seconds. */
export function formatUtcClockSeconds(seconds: number): string {
  const total = Math.floor(((seconds % 86400) + 86400) % 86400);
  return `${formatUtcClock(total)}:${String(total % 60).padStart(2, '0')}`;
}

/** New York daylight time (UTC-4 on the AIS day) clock for a UTC clock in seconds. */
export function formatEasternClock(seconds: number): string {
  return `${formatUtcClock(seconds - 4 * 3600)} EDT`;
}

/** Human duration: `45 s`, `12 min`, `3.4 h`, `2.1 d`. */
export function formatDuration(seconds: number): string {
  if (!Number.isFinite(seconds)) return 'n/a';
  if (seconds >= 86400 * 2) return `${(seconds / 86400).toFixed(1)} d`;
  if (seconds >= 3600) return `${(seconds / 3600).toFixed(1)} h`;
  if (seconds >= 60) return `${(seconds / 60).toFixed(0)} min`;
  return `${seconds.toFixed(0)} s`;
}

/** Nearest track to a planar point by vertex distance, with that distance in meters. */
export function findNearestTrack(
  set: TrackSet,
  x: number,
  y: number
): {track: number; distance: number} {
  let best = -1;
  let bestSquared = Infinity;
  for (let track = 0; track < set.trackCount; track++) {
    for (let vertex = set.offsets[track]; vertex < set.offsets[track + 1]; vertex++) {
      const dx = set.positions[vertex * 2] - x;
      const dy = set.positions[vertex * 2 + 1] - y;
      const squared = dx * dx + dy * dy;
      if (squared < bestSquared) {
        bestSquared = squared;
        best = track;
      }
    }
  }
  return {track: best, distance: Math.sqrt(bestSquared)};
}
