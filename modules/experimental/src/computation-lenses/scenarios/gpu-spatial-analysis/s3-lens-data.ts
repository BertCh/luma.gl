// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Loaders for the stream-S3 (spatial geometry) docs datasets in `computation-lenses/data/s3-*.json`,
 * written by `scripts/lens-data/s3-geometry-data.py`. Each loader imports its JSON lazily, so nothing
 * is fetched until a scenario runs. Coordinates come twice: lon/lat (EPSG:4326) and projected metres
 * (UTM minus a stated false origin, so float32 keeps centimetre precision).
 */

/** Dataset identity shown in lens provenance and the data README. */
export type S3DatasetInfo = {id: string; source: string; licence: string; description: string};

/** Projected coordinate frame of a dataset. */
export type S3ProjectedFrame = {
  /** EPSG code of the UTM zone, such as `EPSG:32616`. */
  crs: string;
  unit: 'm';
  /** Subtracted from UTM easting/northing before storage. */
  falseOrigin: [number, number];
};

/** GeoArrow-style polygons: feature → polygon → ring → vertex offsets. */
export type S3Polygons = S3DatasetInfo & {
  featureCount: number;
  positionsLonLat: Float32Array;
  positionsProjected: Float32Array;
  ringOffsets: Uint32Array;
  polygonOffsets: Uint32Array;
  featureOffsets: Uint32Array;
  projected: S3ProjectedFrame;
  names: string[];
};

/** Polylines: line → vertex offsets. */
export type S3Lines = S3DatasetInfo & {
  lineCount: number;
  positionsLonLat: Float32Array;
  positionsProjected: Float32Array;
  lineOffsets: Uint32Array;
  /** Road class index per line into `classNames`. */
  edgeClass: Uint32Array;
  classNames: string[];
  projected: S3ProjectedFrame;
};

/** Points with a nominal category. */
export type S3Points = S3DatasetInfo & {
  pointCount: number;
  positionsLonLat: Float32Array;
  positionsProjected: Float32Array;
  category: Uint32Array;
  categoryNames: string[];
  projected: S3ProjectedFrame;
};

/** Vessel tracks: track → fix offsets, with per-fix timestamps and speeds. */
export type S3Tracks = S3DatasetInfo & {
  trackCount: number;
  positionsLonLat: Float32Array;
  positionsProjected: Float32Array;
  trackOffsets: Uint32Array;
  /** Seconds since `timestampUnit`'s epoch. */
  timestamps: Float32Array;
  timestampUnit: string;
  speeds: Float32Array;
  mmsi: Uint32Array;
  projected: S3ProjectedFrame;
};

/** One hub airport and its busiest destinations (lon/lat only). */
export type S3FlightPairs = S3DatasetInfo & {
  hub: {iata: string; lonLat: [number, number]};
  destinationCount: number;
  iata: string[];
  destinationsLonLat: Float32Array;
};

type JsonRecord = Record<string, unknown>;

function info(json: JsonRecord): S3DatasetInfo {
  return {
    id: String(json['id']),
    source: String(json['source']),
    licence: String(json['licence']),
    description: String(json['description'])
  };
}

function floats(json: JsonRecord, key: string): Float32Array {
  return Float32Array.from(json[key] as number[]);
}

function integers(json: JsonRecord, key: string): Uint32Array {
  return Uint32Array.from(json[key] as number[]);
}

function decodePolygons(json: JsonRecord): S3Polygons {
  return {
    ...info(json),
    featureCount: Number(json['featureCount']),
    positionsLonLat: floats(json, 'positionsLonLat'),
    positionsProjected: floats(json, 'positionsProjected'),
    ringOffsets: integers(json, 'ringOffsets'),
    polygonOffsets: integers(json, 'polygonOffsets'),
    featureOffsets: integers(json, 'featureOffsets'),
    projected: json['projected'] as S3ProjectedFrame,
    names: (json['names'] as string[] | undefined) ?? []
  };
}

/** 51 parks and green spaces in Chicago's North Side (OpenStreetMap, ODbL). */
export async function loadS3ChicagoParks(): Promise<S3Polygons> {
  return decodePolygons((await import('../../data/s3-chicago-parks.json')).default);
}

/** 17 adjacent Chicago community areas: a coverage with shared edges (City of Chicago). */
export async function loadS3ChicagoCommunityAreas(): Promise<S3Polygons> {
  const json = (await import('../../data/s3-chicago-community-areas.json')).default as JsonRecord;
  const polygons = decodePolygons(json);
  return {...polygons, names: (json['areaIds'] as number[]).map(id => `area ${id}`)};
}

/** NOAA anchorage and channel polygons inside Upper New York Bay (public domain). */
export async function loadS3NewYorkHarborZones(): Promise<S3Polygons> {
  return decodePolygons((await import('../../data/s3-ny-harbor-zones.json')).default);
}

/** 223 street centrelines in Wicker Park, Chicago (OpenStreetMap, ODbL). */
export async function loadS3ChicagoRoads(): Promise<S3Lines> {
  const json = (await import('../../data/s3-chicago-roads.json')).default as JsonRecord;
  return {
    ...info(json),
    lineCount: Number(json['lineCount']),
    positionsLonLat: floats(json, 'positionsLonLat'),
    positionsProjected: floats(json, 'positionsProjected'),
    lineOffsets: integers(json, 'lineOffsets'),
    edgeClass: integers(json, 'edgeClass'),
    classNames: json['classNames'] as string[],
    projected: json['projected'] as S3ProjectedFrame
  };
}

/** 1,275 Overture places in Wicker Park, Chicago (CDLA-Permissive-2.0 and others). */
export async function loadS3ChicagoPlaces(): Promise<S3Points> {
  const json = (await import('../../data/s3-chicago-places.json')).default as JsonRecord;
  return {
    ...info(json),
    pointCount: Number(json['pointCount']),
    positionsLonLat: floats(json, 'positionsLonLat'),
    positionsProjected: floats(json, 'positionsProjected'),
    category: integers(json, 'category'),
    categoryNames: json['categoryNames'] as string[],
    projected: json['projected'] as S3ProjectedFrame
  };
}

/** 12 AIS vessel tracks in Upper New York Bay over two hours (NOAA/USCG, CC0). */
export async function loadS3NewYorkHarborTracks(): Promise<S3Tracks> {
  const json = (await import('../../data/s3-ny-harbor-ais.json')).default as JsonRecord;
  return {
    ...info(json),
    trackCount: Number(json['trackCount']),
    positionsLonLat: floats(json, 'positionsLonLat'),
    positionsProjected: floats(json, 'positionsProjected'),
    trackOffsets: integers(json, 'trackOffsets'),
    timestamps: floats(json, 'timestamps'),
    timestampUnit: String(json['timestampUnit']),
    speeds: floats(json, 'speeds'),
    mmsi: integers(json, 'mmsi'),
    projected: json['projected'] as S3ProjectedFrame
  };
}

/** The 40 busiest airport pairs touching Chicago O'Hare (OpenFlights, ODbL). */
export async function loadS3OpenFlightsHub(): Promise<S3FlightPairs> {
  const json = (await import('../../data/s3-openflights-ord.json')).default as JsonRecord;
  return {
    ...info(json),
    hub: json['hub'] as {iata: string; lonLat: [number, number]},
    destinationCount: Number(json['destinationCount']),
    iata: json['iata'] as string[],
    destinationsLonLat: floats(json, 'destinationsLonLat')
  };
}
