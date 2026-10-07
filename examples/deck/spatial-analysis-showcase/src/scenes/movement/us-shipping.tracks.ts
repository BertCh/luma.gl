// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {LoadedDataset} from '../../data/catalog';
import {US_TYPE_TO_GROUP} from './movement-style';
import {createAzimuthalEquidistant, createTrackSet, type TrackSet} from './b12-tracks';

/**
 * Data helpers of the `us-shipping-day` scene: the `poopdeck-ais-us` tracks in two planar spaces
 * (azimuthal-equidistant meters for the analysis, Web Mercator meters for the playhead), the
 * vessel-type to vessel-group table, the chokepoint gates and the camera frames of the story.
 */

/** Vessel types of the dataset in display order. */
export const SHIPPING_TYPES = [
  'cargo',
  'tanker',
  'passenger',
  'towing',
  'fishing',
  'special',
  'other'
] as const;

export type ShippingType = (typeof SHIPPING_TYPES)[number];

export const SHIPPING_TYPE_LABELS: Record<ShippingType, string> = {
  cargo: 'Cargo ship',
  tanker: 'Tanker',
  passenger: 'Passenger (ferry, cruise)',
  towing: 'Tug or tow',
  fishing: 'Fishing',
  special: 'Special (pilot, SAR, dredge...)',
  other: 'Other (yachts, sail, unknown)'
};

/** AIS ship-type categories in the manifest's own order (the codes of the `vessel_type` column). */
const MANIFEST_VESSEL_TYPES: readonly ShippingType[] = [
  'towing',
  'passenger',
  'other',
  'cargo',
  'special',
  'tanker',
  'fishing'
];

/**
 * Vessel group of each entry of {@link SHIPPING_TYPES} (index into the four vessel groups of
 * `movement-style`: passenger, cargo and tanker, tug and tow, other).
 */
export const SHIPPING_TYPE_GROUP: readonly number[] = SHIPPING_TYPES.map(
  type => US_TYPE_TO_GROUP[MANIFEST_VESSEL_TYPES.indexOf(type)]
);

/** The camera frames of the story (centre and zoom; pitch and bearing stay 0). */
export const SHIPPING_FRAMES = {
  country: {longitude: -96.5, latitude: 37.5, zoom: 3.6},
  gulf: {longitude: -91.5, latitude: 29.3, zoom: 5.6},
  lowerMississippi: {longitude: -90.6, latitude: 30.5, zoom: 6.4},
  galvestonBay: {longitude: -94.85, latitude: 29.42, zoom: 10.2},
  midAtlantic: {longitude: -73, latitude: 38, zoom: 5.4}
} as const;

/** Region where the clip-and-walk diagram looks for its segment: the open Gulf of Mexico. */
export const CLIP_WALK_REGION: readonly [number, number, number, number] = [
  -97.5, 25.5, -85.5, 30.5
];

/** Center of the azimuthal-equidistant analysis projection (central Kansas). */
export const SHIPPING_PROJECTION_CENTER: readonly [number, number] = [-96, 38];

const EARTH_RADIUS_METERS = 6378137;
const DEGREES = Math.PI / 180;

/** Web Mercator meters of a longitude and latitude (unbounded in latitude below 85 degrees). */
export function projectMercator(longitude: number, latitude: number): [number, number] {
  return [
    EARTH_RADIUS_METERS * longitude * DEGREES,
    EARTH_RADIUS_METERS * Math.log(Math.tan(Math.PI / 4 + (latitude * DEGREES) / 2))
  ];
}

/** Ellipsoid radius used by the azimuthal-equidistant helper of `b12-tracks`. */
export const AZIMUTHAL_EARTH_RADIUS_METERS = 6371008.8;

/** The vessel tracks of the `poopdeck-ais-us` dataset. */
export type ShippingTracks = TrackSet & {
  /** Type index per track (see {@link SHIPPING_TYPES}). */
  category: Uint32Array;
  mmsi: Uint32Array;
  /** Dense vessel index per track; a vessel owns several tracks after reporting gaps. */
  vesselIndex: Uint32Array;
  vesselCount: number;
  /** Length in meters per track, 0 when unknown. */
  length: Float32Array;
  /** Reported speed over ground in knots per vertex, `NaN` when not available. */
  reportedKnots: Float32Array;
  /** Web Mercator meters relative to {@link mercatorOrigin}, one row per vertex (playhead space). */
  mercator: Float32Array;
  mercatorOrigin: readonly [number, number];
  /** First timestamp of each track in seconds. */
  trackStartTimes: Float32Array;
  /** Unix time of second 0, in milliseconds. */
  timeOriginMs: number;
  /** Planar bounds of the dataset in analysis meters. */
  bounds: [number, number, number, number];
};

/** Loads `poopdeck-ais-us` as {@link ShippingTracks}. */
export function loadShippingTracks(dataset: LoadedDataset): ShippingTracks {
  const lngLat = dataset.column<Float32Array>('vertices');
  const projection = createAzimuthalEquidistant(SHIPPING_PROJECTION_CENTER);
  const vertexCount = lngLat.length / 2;
  const positions = new Float32Array(lngLat.length);
  const mercator = new Float32Array(lngLat.length);
  const mercatorOrigin = projectMercator(
    SHIPPING_PROJECTION_CENTER[0],
    SHIPPING_PROJECTION_CENTER[1]
  );
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (let vertex = 0; vertex < vertexCount; vertex++) {
    const longitude = lngLat[vertex * 2];
    const latitude = lngLat[vertex * 2 + 1];
    const [x, y] = projection.project(longitude, latitude);
    positions[vertex * 2] = x;
    positions[vertex * 2 + 1] = y;
    minX = Math.min(minX, x);
    maxX = Math.max(maxX, x);
    minY = Math.min(minY, y);
    maxY = Math.max(maxY, y);
    const [mx, my] = projectMercator(longitude, latitude);
    mercator[vertex * 2] = mx - mercatorOrigin[0];
    mercator[vertex * 2 + 1] = my - mercatorOrigin[1];
  }
  const set = createTrackSet({
    offsets: dataset.column<Uint32Array>('pathOffsets'),
    positions,
    lngLat,
    timestamps: Float32Array.from(dataset.column<Uint32Array>('timestamp')),
    origin: SHIPPING_PROJECTION_CENTER,
    project: projection.project,
    unproject: projection.unproject,
    drawInDegrees: true
  });

  const typeNames = dataset.categories('vessel_type');
  const typeRow = Uint32Array.from(
    typeNames.map(name => {
      const index = SHIPPING_TYPES.indexOf(name as ShippingType);
      return index < 0 ? SHIPPING_TYPES.length - 1 : index;
    })
  );
  const category = Uint32Array.from(
    dataset.column<Uint8Array>('vessel_type'),
    code => typeRow[code]
  );

  const speedColumn = dataset.manifest.columns?.speed as
    | {scale?: number; noData?: number}
    | undefined;
  const speedScale = speedColumn?.scale ?? 0.5;
  const speedNoData = speedColumn?.noData ?? 255;
  const rawSpeed = dataset.column<Uint8Array>('speed');
  const reportedKnots = new Float32Array(rawSpeed.length);
  for (let vertex = 0; vertex < rawSpeed.length; vertex++) {
    reportedKnots[vertex] =
      rawSpeed[vertex] === speedNoData ? Number.NaN : rawSpeed[vertex] * speedScale;
  }

  const vesselIndex = dataset.column<Uint32Array>('vesselIndex');
  let vesselCount = 0;
  for (const value of vesselIndex) vesselCount = Math.max(vesselCount, value + 1);

  const trackStartTimes = new Float32Array(set.trackCount);
  for (let track = 0; track < set.trackCount; track++) {
    trackStartTimes[track] = set.timestamps[set.offsets[track]];
  }

  return {
    ...set,
    category,
    mmsi: dataset.column<Uint32Array>('mmsi'),
    vesselIndex,
    vesselCount,
    length: dataset.column<Float32Array>('length'),
    reportedKnots,
    mercator,
    mercatorOrigin,
    trackStartTimes,
    timeOriginMs: Number(dataset.properties.timeOriginMs ?? Date.UTC(2023, 0, 9)),
    bounds: [minX, minY, maxX, maxY]
  };
}

/** A chokepoint where crossings are counted. Endpoints are approximate shore-to-shore lines. */
export type ShippingGate = {
  id: string;
  name: string;
  /** Short chart label. */
  short: string;
  /** Gate line end points as `[longitude, latitude]`. */
  from: readonly [number, number];
  to: readonly [number, number];
  /** Compass bearing in degrees (0 north, 90 east) of the travel counted as the first direction. */
  firstBearing: number;
  /** Names of the first and second direction. */
  directions: readonly [string, string];
  /** IANA time zone of the gate, for the local-time axis of its crossings chart. */
  timeZone: string;
};

/**
 * Real US chokepoints. The lines are approximate shore-to-shore crossings of the channel, checked
 * against the tracks (every gate is crossed by tracks in the data). The Houston gate sits across
 * the ship channel in the middle of Galveston Bay, not at the bay entrance (Bolivar Roads, in the
 * `US` gazetteer).
 */
export const SHIPPING_GATES: readonly ShippingGate[] = [
  {
    id: 'golden-gate',
    name: 'Golden Gate (San Francisco)',
    short: 'Golden Gate',
    from: [-122.4766, 37.8095],
    to: [-122.48, 37.8325],
    firstBearing: 90,
    directions: ['inbound', 'outbound'],
    timeZone: 'America/Los_Angeles'
  },
  {
    id: 'la-angels',
    name: 'Angels Gate (Los Angeles)',
    short: 'LA Angels Gate',
    from: [-118.256, 33.7],
    to: [-118.24, 33.72],
    firstBearing: 330,
    directions: ['inbound', 'outbound'],
    timeZone: 'America/Los_Angeles'
  },
  {
    id: 'lb-queens',
    name: 'Queens Gate (Long Beach)',
    short: 'LB Queens Gate',
    from: [-118.19, 33.715],
    to: [-118.18, 33.735],
    firstBearing: 340,
    directions: ['inbound', 'outbound'],
    timeZone: 'America/Los_Angeles'
  },
  {
    id: 'san-diego',
    name: 'San Diego Bay entrance',
    short: 'San Diego',
    from: [-117.25, 32.69],
    to: [-117.22, 32.69],
    firstBearing: 20,
    directions: ['inbound', 'outbound'],
    timeZone: 'America/Los_Angeles'
  },
  {
    id: 'juan-de-fuca',
    name: 'Strait of Juan de Fuca',
    short: 'Juan de Fuca',
    from: [-123.55, 48.12],
    to: [-123.55, 48.42],
    firstBearing: 90,
    directions: ['eastbound (inbound)', 'westbound (outbound)'],
    timeZone: 'America/Los_Angeles'
  },
  {
    id: 'seattle-bainbridge',
    name: 'Seattle to Bainbridge ferry lane',
    short: 'Seattle ferries',
    from: [-122.42, 47.58],
    to: [-122.42, 47.66],
    firstBearing: 90,
    directions: ['eastbound', 'westbound'],
    timeZone: 'America/Los_Angeles'
  },
  {
    id: 'houston',
    name: 'Houston Ship Channel, mid Galveston Bay',
    short: 'Houston Ship Channel',
    from: [-94.855, 29.42],
    to: [-94.79, 29.42],
    firstBearing: 0,
    directions: ['northbound (inbound)', 'southbound (outbound)'],
    timeZone: 'America/Chicago'
  },
  {
    id: 'new-orleans',
    name: 'Mississippi River at New Orleans',
    short: 'New Orleans',
    from: [-90.082, 29.935],
    to: [-90.04, 29.935],
    firstBearing: 30,
    directions: ['upriver', 'downriver'],
    timeZone: 'America/Chicago'
  },
  {
    id: 'miami',
    name: 'PortMiami (Government Cut)',
    short: 'PortMiami',
    from: [-80.14, 25.755],
    to: [-80.14, 25.775],
    firstBearing: 270,
    directions: ['inbound', 'outbound'],
    timeZone: 'America/New_York'
  },
  {
    id: 'port-everglades',
    name: 'Port Everglades entrance',
    short: 'Port Everglades',
    from: [-80.105, 26.085],
    to: [-80.105, 26.1],
    firstBearing: 270,
    directions: ['inbound', 'outbound'],
    timeZone: 'America/New_York'
  },
  {
    id: 'chesapeake',
    name: 'Chesapeake Bay entrance (Cape Henry to Cape Charles)',
    short: 'Chesapeake Bay',
    from: [-76.02, 36.93],
    to: [-75.95, 37.11],
    firstBearing: 270,
    directions: ['inbound', 'outbound'],
    timeZone: 'America/New_York'
  },
  {
    id: 'delaware',
    name: 'Delaware Bay entrance (Cape Henlopen to Cape May)',
    short: 'Delaware Bay',
    from: [-75.09, 38.8],
    to: [-74.95, 38.93],
    firstBearing: 315,
    directions: ['inbound', 'outbound'],
    timeZone: 'America/New_York'
  },
  {
    id: 'ambrose',
    name: 'Ambrose Channel (New York)',
    short: 'Ambrose Channel',
    from: [-73.95, 40.49],
    to: [-73.8, 40.49],
    firstBearing: 350,
    directions: ['inbound', 'outbound'],
    timeZone: 'America/New_York'
  }
];

/** Gate edges in analysis meters: one rectangle per gate, `halfWidthMeters` either side of its line. */
export type GateEdges = {
  /** `x0, y0` rows, 4 per gate. */
  starts: Float32Array;
  /** `x1, y1` rows, aligned with `starts`. */
  ends: Float32Array;
  zones: Uint32Array;
  /** Gate centers and unit vectors in analysis meters, for direction tests. */
  centers: Float32Array;
  /** `[ux, uy]` unit vector from the first to the second end point, per gate. */
  axes: Float32Array;
  lengths: Float32Array;
  /** `x0, y0, x1, y1` drawing rows in degrees: the center line of every gate. */
  lines: Float32Array;
  /** `+1` when crossing from the left of the gate axis to the right is the first direction. */
  firstSigns: Int8Array;
};

/** Builds the gate rectangles in the analysis projection. */
export function buildGateEdges(
  project: (longitude: number, latitude: number) => [number, number],
  halfWidthMeters: number
): GateEdges {
  const count = SHIPPING_GATES.length;
  const starts = new Float32Array(count * 4 * 2);
  const ends = new Float32Array(count * 4 * 2);
  const zones = new Uint32Array(count * 4);
  const centers = new Float32Array(count * 2);
  const axes = new Float32Array(count * 2);
  const lengths = new Float32Array(count);
  const lines = new Float32Array(count * 4);
  const firstSigns = new Int8Array(count);
  SHIPPING_GATES.forEach((gate, index) => {
    const [x0, y0] = project(gate.from[0], gate.from[1]);
    const [x1, y1] = project(gate.to[0], gate.to[1]);
    const length = Math.hypot(x1 - x0, y1 - y0);
    const ux = (x1 - x0) / length;
    const uy = (y1 - y0) / length;
    // Left normal of the axis.
    const nx = -uy * halfWidthMeters;
    const ny = ux * halfWidthMeters;
    const corners: [number, number][] = [
      [x0 + nx, y0 + ny],
      [x1 + nx, y1 + ny],
      [x1 - nx, y1 - ny],
      [x0 - nx, y0 - ny]
    ];
    for (let corner = 0; corner < 4; corner++) {
      const next = corners[(corner + 1) % 4];
      const row = index * 4 + corner;
      starts.set(corners[corner], row * 2);
      ends.set(next, row * 2);
      zones[row] = index;
    }
    centers.set([(x0 + x1) / 2, (y0 + y1) / 2], index * 2);
    axes.set([ux, uy], index * 2);
    lengths[index] = length;
    lines.set([gate.from[0], gate.from[1], gate.to[0], gate.to[1]], index * 4);
    // Travel along `bearing` is (sin b, cos b). Moving from the left of the axis to the right
    // means travelling along the right normal (uy, -ux).
    const bearing = gate.firstBearing * DEGREES;
    const travelX = Math.sin(bearing);
    const travelY = Math.cos(bearing);
    firstSigns[index] = travelX * uy - travelY * ux >= 0 ? 1 : -1;
  });
  return {starts, ends, zones, centers, axes, lengths, lines, firstSigns};
}
