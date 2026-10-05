// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {addMetersToLngLat, getDistanceScales} from '@math.gl/web-mercator';

/**
 * Datasets for the map-graphs explorer.
 *
 * Every loader fetches a small open dataset from the deck.gl-data repository (the same files the
 * deck.gl website examples use) and falls back to a deterministic synthetic dataset when the fetch
 * fails or `?data=synthetic` is set, so the demo works offline and in headless tests. Positions are
 * converted once on the CPU into planar meters east/north of a fixed origin; layers render them
 * with deck.gl `COORDINATE_SYSTEM.METER_OFFSETS` around the same origin, so recipe radii,
 * distances, and costs are in meters.
 */

const DECK_DATA_URL = 'https://raw.githubusercontent.com/visgl/deck.gl-data/master';

/** Origin of the New York datasets (lower/mid Manhattan). */
export const NEW_YORK_ORIGIN: readonly [number, number] = [-73.985, 40.745];
/** Origin of the San Francisco datasets. */
export const SAN_FRANCISCO_ORIGIN: readonly [number, number] = [-122.44, 37.76];

/** Where a dataset came from. */
export type MapGraphsDataSource = 'remote' | 'synthetic';

/** Common fields of every loaded dataset. */
export type MapGraphsDataset = {
  /** Whether the dataset was fetched or synthesized. */
  source: MapGraphsDataSource;
  /** Human-readable provenance for the panel. */
  attribution: string;
  /** `[longitude, latitude]` origin of the planar meter coordinates. */
  origin: readonly [number, number];
};

/** Vehicle trips with per-vertex timestamps (deck.gl trips example, New York). */
export type MapGraphsTrips = MapGraphsDataset & {
  /** `x, y` meters per vertex. */
  vertexPositions: Float32Array;
  /** Seconds per vertex. */
  vertexTimestamps: Float32Array;
  /** `tripCount + 1` offsets into the vertex arrays. */
  tripOffsets: Uint32Array;
  /** Vendor (0 or 1) per trip. */
  vendors: Uint32Array;
  /** `[minimum, maximum]` timestamp in seconds. */
  timeRange: readonly [number, number];
};

/** Directed road network in COO form plus drawable polyline segments (deck.gl trips roads). */
export type MapGraphsRoadNetwork = MapGraphsDataset & {
  /** `x, y` meters per node. Nodes are de-duplicated polyline vertices. */
  nodePositions: Float32Array;
  /** Directed edge sources. Every street segment produces both directions. */
  edgeSources: Uint32Array;
  /** Directed edge targets aligned with `edgeSources`. */
  edgeTargets: Uint32Array;
  /** Edge length in meters aligned with `edgeSources`. */
  edgeLengths: Float32Array;
  /**
   * 1 when vehicles may use the edge (it does not run against a one-way street), otherwise 0.
   * Keeping both directions lets walk and drive share one CSR topology and differ only in weights.
   */
  edgeDrivable: Uint32Array;
  /** Undirected drawable segments `x0, y0, x1, y1` in meters (one per polyline segment). */
  segments: Float32Array;
  /** Road class per segment: 0 motorway/trunk, 1 primary/secondary, 2 tertiary, 3 other. */
  segmentClasses: Uint32Array;
  /** First endpoint node of each segment. */
  segmentNodes: Uint32Array;
};

/** Categorized points of interest (deck.gl trips POIs, New York). */
export type MapGraphsPointsOfInterest = MapGraphsDataset & {
  /** `x, y` meters per point. */
  positions: Float32Array;
  /** Dense category index per point into `categoryNames`. */
  categories: Uint32Array;
  /** Category names. */
  categoryNames: readonly string[];
};

/** Bike-parking points with a numeric value (deck.gl website, San Francisco). */
export type MapGraphsBikeParking = MapGraphsDataset & {
  /** `x, y` meters per point. */
  positions: Float32Array;
  /** Parking spaces per location. */
  spaces: Float32Array;
};

/** Polygons in GeoArrow-style offsets (deck.gl website ZIP codes, San Francisco). */
export type MapGraphsPolygons = MapGraphsDataset & {
  /** Flattened ring vertices `x, y` in meters. Rings close implicitly. */
  polygonPositions: Float32Array;
  /** `featureCount + 1` feature-to-polygon offsets. */
  featureOffsets: Uint32Array;
  /** `polygonCount + 1` polygon-to-ring offsets. */
  polygonOffsets: Uint32Array;
  /** `ringCount + 1` ring-to-vertex offsets. */
  ringOffsets: Uint32Array;
  /** Stable feature IDs (ZIP codes). */
  featureIds: Uint32Array;
  /** Per-feature label. */
  featureNames: readonly string[];
  /** Ring outline segments `x0, y0, x1, y1` for drawing. */
  outlineSegments: Float32Array;
  /** Feature row per outline segment. */
  outlineFeatureRows: Uint32Array;
};

/** One elevation raster (deck.gl website terrain.png, San Francisco, USGS DEM). */
export type MapGraphsTerrain = MapGraphsDataset & {
  /** Raster columns. */
  width: number;
  /** Raster rows. */
  height: number;
  /** Elevation meters, row-major, row 0 at the north edge. */
  elevation: Float32Array;
  /** `[minX, minY, maxX, maxY]` meters of the raster's outer cell edges. */
  bounds: readonly [number, number, number, number];
  /** `[cellWidth, cellHeight]` meters. */
  cellSize: readonly [number, number];
  /** `[minimum, maximum]` elevation. */
  elevationRange: readonly [number, number];
};

/** Lazily loaded, memoized datasets shared by every mode in one explorer. */
export type MapGraphsDataCatalog = {
  /** True when `?data=synthetic` forces synthetic data. */
  readonly forceSynthetic: boolean;
  getNewYorkTrips: () => Promise<MapGraphsTrips>;
  getNewYorkRoads: () => Promise<MapGraphsRoadNetwork>;
  getNewYorkPointsOfInterest: () => Promise<MapGraphsPointsOfInterest>;
  getSanFranciscoBikeParking: () => Promise<MapGraphsBikeParking>;
  getSanFranciscoZipCodes: () => Promise<MapGraphsPolygons>;
  getSanFranciscoTerrain: () => Promise<MapGraphsTerrain>;
};

/**
 * Converts between `[longitude, latitude]` and planar meters around one origin using the same
 * high-precision distance scales deck.gl applies to `METER_OFFSETS` coordinates.
 */
export class LocalMetricProjection {
  /** `[longitude, latitude]` origin. */
  readonly origin: readonly [number, number];
  private readonly metersPerDegree: readonly [number, number];

  constructor(origin: readonly [number, number]) {
    this.origin = origin;
    const scales = getDistanceScales({
      longitude: origin[0],
      latitude: origin[1],
      highPrecision: true
    });
    this.metersPerDegree = [
      scales.unitsPerDegree[0] * scales.metersPerUnit[0],
      scales.unitsPerDegree[1] * scales.metersPerUnit[1]
    ];
  }

  /** Returns `[x, y]` meters for one `[longitude, latitude]`. */
  project(longitude: number, latitude: number): [number, number] {
    // Linear estimate refined once against deck.gl's own meters-to-lng/lat conversion.
    let x = (longitude - this.origin[0]) * this.metersPerDegree[0];
    let y = (latitude - this.origin[1]) * this.metersPerDegree[1];
    const [estimatedLongitude, estimatedLatitude] = this.unproject(x, y);
    x += (longitude - estimatedLongitude) * this.metersPerDegree[0];
    y += (latitude - estimatedLatitude) * this.metersPerDegree[1];
    return [x, y];
  }

  /** Returns `[longitude, latitude]` for `[x, y]` meters. */
  unproject(x: number, y: number): [number, number] {
    const [longitude, latitude] = addMetersToLngLat(this.origin as [number, number], [x, y]);
    return [longitude, latitude];
  }
}

/** Creates a catalog. `forceSynthetic` skips every network request. */
export function createMapGraphsDataCatalog(options: {forceSynthetic?: boolean} = {}) {
  const forceSynthetic = Boolean(options.forceSynthetic);
  const memoized = new Map<string, Promise<unknown>>();
  function load<T>(key: string, remote: () => Promise<T>, synthetic: () => T): Promise<T> {
    let promise = memoized.get(key) as Promise<T> | undefined;
    if (!promise) {
      promise = forceSynthetic
        ? Promise.resolve(synthetic())
        : remote().catch(() => {
            // The panel shows the synthetic attribution, so the fallback stays visible.
            return synthetic();
          });
      memoized.set(key, promise);
    }
    return promise;
  }
  const catalog: MapGraphsDataCatalog = {
    forceSynthetic,
    getNewYorkTrips: () => load('trips', loadNewYorkTrips, makeSyntheticTrips),
    getNewYorkRoads: () => load('roads', loadNewYorkRoads, makeSyntheticRoads),
    getNewYorkPointsOfInterest: () =>
      load('points-of-interest', loadNewYorkPointsOfInterest, makeSyntheticPointsOfInterest),
    getSanFranciscoBikeParking: () =>
      load('bike-parking', loadSanFranciscoBikeParking, makeSyntheticBikeParking),
    getSanFranciscoZipCodes: () =>
      load('zip-codes', loadSanFranciscoZipCodes, makeSyntheticZipCodes),
    getSanFranciscoTerrain: () => load('terrain', loadSanFranciscoTerrain, makeSyntheticTerrain)
  };
  return catalog;
}

// ---------------------------------------------------------------------------------------------
// Remote loaders
// ---------------------------------------------------------------------------------------------

async function fetchJson<T>(path: string): Promise<T> {
  const response = await fetch(`${DECK_DATA_URL}/${path}`);
  if (!response.ok) {
    throw new Error(`${path}: HTTP ${response.status}`);
  }
  return (await response.json()) as T;
}

async function loadNewYorkTrips(): Promise<MapGraphsTrips> {
  const trips = await fetchJson<{vendor: number; path: number[][]; timestamps: number[]}[]>(
    'examples/trips/trips-v7.json'
  );
  const projection = new LocalMetricProjection(NEW_YORK_ORIGIN);
  const vertexCount = trips.reduce((total, trip) => total + trip.path.length, 0);
  const vertexPositions = new Float32Array(vertexCount * 2);
  const vertexTimestamps = new Float32Array(vertexCount);
  const tripOffsets = new Uint32Array(trips.length + 1);
  const vendors = new Uint32Array(trips.length);
  let vertex = 0;
  let minimumTime = Infinity;
  let maximumTime = -Infinity;
  trips.forEach((trip, tripIndex) => {
    tripOffsets[tripIndex] = vertex;
    vendors[tripIndex] = trip.vendor;
    trip.path.forEach((coordinate, pathIndex) => {
      const [x, y] = projection.project(coordinate[0], coordinate[1]);
      vertexPositions[vertex * 2] = x;
      vertexPositions[vertex * 2 + 1] = y;
      const time = trip.timestamps[pathIndex];
      vertexTimestamps[vertex] = time;
      minimumTime = Math.min(minimumTime, time);
      maximumTime = Math.max(maximumTime, time);
      vertex++;
    });
  });
  tripOffsets[trips.length] = vertex;
  return {
    source: 'remote',
    attribution: 'deck.gl-data trips-v7 (New York taxi trips)',
    origin: NEW_YORK_ORIGIN,
    vertexPositions,
    vertexTimestamps,
    tripOffsets,
    vendors,
    timeRange: [minimumTime, maximumTime]
  };
}

type RoadFeatureCollection = {
  features: {
    properties: {fclass: string; oneway: string};
    geometry: {type: string; coordinates: number[][]};
  }[];
};

async function loadNewYorkRoads(): Promise<MapGraphsRoadNetwork> {
  const collection = await fetchJson<RoadFeatureCollection>('examples/trips/roads.json');
  const projection = new LocalMetricProjection(NEW_YORK_ORIGIN);
  const builder = new RoadNetworkBuilder();
  for (const feature of collection.features) {
    if (feature.geometry.type !== 'LineString') {
      continue;
    }
    const points = feature.geometry.coordinates.map(
      ([longitude, latitude]) =>
        [longitude, latitude, ...projection.project(longitude, latitude)] as const
    );
    builder.addPolyline(
      points.map(point => ({
        key: `${point[0].toFixed(7)},${point[1].toFixed(7)}`,
        x: point[2],
        y: point[3]
      })),
      getRoadClass(feature.properties.fclass),
      feature.properties.oneway
    );
  }
  return builder.finish('remote', 'deck.gl-data trips roads (OpenStreetMap, New York)');
}

async function loadNewYorkPointsOfInterest(): Promise<MapGraphsPointsOfInterest> {
  const rows = await fetchJson<{fclass: string; coordinates: number[]}[]>(
    'examples/trips/pois.json'
  );
  const projection = new LocalMetricProjection(NEW_YORK_ORIGIN);
  const categoryNames: string[] = [];
  const categoryIndexes = new Map<string, number>();
  const positions = new Float32Array(rows.length * 2);
  const categories = new Uint32Array(rows.length);
  rows.forEach((row, index) => {
    const [x, y] = projection.project(row.coordinates[0], row.coordinates[1]);
    positions[index * 2] = x;
    positions[index * 2 + 1] = y;
    let category = categoryIndexes.get(row.fclass);
    if (category === undefined) {
      category = categoryNames.length;
      categoryNames.push(row.fclass);
      categoryIndexes.set(row.fclass, category);
    }
    categories[index] = category;
  });
  return {
    source: 'remote',
    attribution: 'deck.gl-data trips POIs (OpenStreetMap, New York)',
    origin: NEW_YORK_ORIGIN,
    positions,
    categories,
    categoryNames
  };
}

async function loadSanFranciscoBikeParking(): Promise<MapGraphsBikeParking> {
  const rows = await fetchJson<{SPACES: number; COORDINATES: number[]}[]>(
    'website/sf-bike-parking.json'
  );
  const projection = new LocalMetricProjection(SAN_FRANCISCO_ORIGIN);
  const valid = rows.filter(
    row => Array.isArray(row.COORDINATES) && row.COORDINATES.every(Number.isFinite)
  );
  const positions = new Float32Array(valid.length * 2);
  const spaces = new Float32Array(valid.length);
  valid.forEach((row, index) => {
    const [x, y] = projection.project(row.COORDINATES[0], row.COORDINATES[1]);
    positions[index * 2] = x;
    positions[index * 2 + 1] = y;
    spaces[index] = Number(row.SPACES) || 0;
  });
  return {
    source: 'remote',
    attribution: 'deck.gl-data sf-bike-parking (DataSF)',
    origin: SAN_FRANCISCO_ORIGIN,
    positions,
    spaces
  };
}

async function loadSanFranciscoZipCodes(): Promise<MapGraphsPolygons> {
  const rows = await fetchJson<{zipcode: number; contour: number[][]}[]>(
    'website/sf-zipcodes.json'
  );
  const projection = new LocalMetricProjection(SAN_FRANCISCO_ORIGIN);
  return makePolygons(
    rows.map(row => ({
      id: row.zipcode,
      name: String(row.zipcode),
      rings: [row.contour.map(([longitude, latitude]) => projection.project(longitude, latitude))]
    })),
    'remote',
    'deck.gl-data sf-zipcodes'
  );
}

async function loadSanFranciscoTerrain(): Promise<MapGraphsTerrain> {
  const response = await fetch(`${DECK_DATA_URL}/website/terrain.png`);
  if (!response.ok) {
    throw new Error(`terrain.png: HTTP ${response.status}`);
  }
  const bitmap = await createImageBitmap(await response.blob());
  const size = 512;
  const canvas = new OffscreenCanvas(size, size);
  const context = canvas.getContext('2d', {willReadFrequently: true});
  if (!context) {
    throw new Error('2D canvas unavailable for terrain decoding');
  }
  context.drawImage(bitmap, 0, 0, size, size);
  const pixels = context.getImageData(0, 0, size, size).data;
  const elevation = new Float32Array(size * size);
  for (let index = 0; index < elevation.length; index++) {
    // deck.gl TerrainLayer docs decode this USGS DEM with rScaler 2, offset 0.
    elevation[index] = pixels[index * 4] * 2;
  }
  return makeTerrain(elevation, size, size, 'remote', 'deck.gl-data terrain.png (USGS DEM)');
}

// ---------------------------------------------------------------------------------------------
// Builders shared by remote and synthetic data
// ---------------------------------------------------------------------------------------------

function getRoadClass(featureClass: string): number {
  if (/motorway|trunk/.test(featureClass)) return 0;
  if (/primary|secondary/.test(featureClass)) return 1;
  if (/tertiary/.test(featureClass)) return 2;
  return 3;
}

/** De-duplicates polyline vertices into nodes and emits directed edges and drawable segments. */
class RoadNetworkBuilder {
  private readonly nodeIndexes = new Map<string, number>();
  private readonly nodePositions: number[] = [];
  private readonly edgeSources: number[] = [];
  private readonly edgeTargets: number[] = [];
  private readonly edgeLengths: number[] = [];
  private readonly edgeDrivable: number[] = [];
  private readonly segments: number[] = [];
  private readonly segmentClasses: number[] = [];
  private readonly segmentNodes: number[] = [];

  addPolyline(points: {key: string; x: number; y: number}[], roadClass: number, oneway: string) {
    const nodes = points.map(point => this.getNode(point.key, point.x, point.y));
    for (let index = 0; index + 1 < nodes.length; index++) {
      const from = nodes[index];
      const to = nodes[index + 1];
      if (from === to) {
        continue;
      }
      const x0 = this.nodePositions[from * 2];
      const y0 = this.nodePositions[from * 2 + 1];
      const x1 = this.nodePositions[to * 2];
      const y1 = this.nodePositions[to * 2 + 1];
      const length = Math.hypot(x1 - x0, y1 - y0);
      // OSM `oneway`: B both directions, F forward only, T backward only.
      this.addEdge(from, to, length, oneway !== 'T');
      this.addEdge(to, from, length, oneway !== 'F');
      this.segments.push(x0, y0, x1, y1);
      this.segmentClasses.push(roadClass);
      this.segmentNodes.push(from);
    }
  }

  finish(source: MapGraphsDataSource, attribution: string): MapGraphsRoadNetwork {
    return {
      source,
      attribution,
      origin: NEW_YORK_ORIGIN,
      nodePositions: Float32Array.from(this.nodePositions),
      edgeSources: Uint32Array.from(this.edgeSources),
      edgeTargets: Uint32Array.from(this.edgeTargets),
      edgeLengths: Float32Array.from(this.edgeLengths),
      edgeDrivable: Uint32Array.from(this.edgeDrivable),
      segments: Float32Array.from(this.segments),
      segmentClasses: Uint32Array.from(this.segmentClasses),
      segmentNodes: Uint32Array.from(this.segmentNodes)
    };
  }

  private getNode(key: string, x: number, y: number): number {
    let node = this.nodeIndexes.get(key);
    if (node === undefined) {
      node = this.nodePositions.length / 2;
      this.nodePositions.push(x, y);
      this.nodeIndexes.set(key, node);
    }
    return node;
  }

  private addEdge(from: number, to: number, length: number, drivable: boolean): void {
    this.edgeSources.push(from);
    this.edgeTargets.push(to);
    this.edgeLengths.push(length);
    this.edgeDrivable.push(drivable ? 1 : 0);
  }
}

function makePolygons(
  features: {id: number; name: string; rings: [number, number][][]}[],
  source: MapGraphsDataSource,
  attribution: string
): MapGraphsPolygons {
  const positions: number[] = [];
  const ringOffsets: number[] = [0];
  const polygonOffsets: number[] = [0];
  const featureOffsets: number[] = [0];
  const outlineSegments: number[] = [];
  const outlineFeatureRows: number[] = [];
  features.forEach((feature, featureRow) => {
    for (const ring of feature.rings) {
      // Drop an explicit closing vertex: rings close implicitly.
      const open =
        ring.length > 1 &&
        ring[0][0] === ring[ring.length - 1][0] &&
        ring[0][1] === ring[ring.length - 1][1]
          ? ring.slice(0, -1)
          : ring;
      for (const [x, y] of open) positions.push(x, y);
      ringOffsets.push(positions.length / 2);
      for (let index = 0; index < open.length; index++) {
        const [x0, y0] = open[index];
        const [x1, y1] = open[(index + 1) % open.length];
        outlineSegments.push(x0, y0, x1, y1);
        outlineFeatureRows.push(featureRow);
      }
    }
    // One polygon (shell plus holes) per feature in these datasets.
    polygonOffsets.push(ringOffsets.length - 1);
    featureOffsets.push(polygonOffsets.length - 1);
  });
  return {
    source,
    attribution,
    origin: SAN_FRANCISCO_ORIGIN,
    polygonPositions: Float32Array.from(positions),
    featureOffsets: Uint32Array.from(featureOffsets),
    polygonOffsets: Uint32Array.from(polygonOffsets),
    ringOffsets: Uint32Array.from(ringOffsets),
    featureIds: Uint32Array.from(features.map(feature => feature.id)),
    featureNames: features.map(feature => feature.name),
    outlineSegments: Float32Array.from(outlineSegments),
    outlineFeatureRows: Uint32Array.from(outlineFeatureRows)
  };
}

/** The terrain.png bounds documented for deck.gl's TerrainLayer. */
const SAN_FRANCISCO_TERRAIN_BOUNDS = [-122.5233, 37.6493, -122.3566, 37.8159] as const;

function makeTerrain(
  elevation: Float32Array,
  width: number,
  height: number,
  source: MapGraphsDataSource,
  attribution: string
): MapGraphsTerrain {
  const projection = new LocalMetricProjection(SAN_FRANCISCO_ORIGIN);
  const [west, south, east, north] = SAN_FRANCISCO_TERRAIN_BOUNDS;
  const [minX, minY] = projection.project(west, south);
  const [maxX, maxY] = projection.project(east, north);
  let minimum = Infinity;
  let maximum = -Infinity;
  for (const value of elevation) {
    minimum = Math.min(minimum, value);
    maximum = Math.max(maximum, value);
  }
  return {
    source,
    attribution,
    origin: SAN_FRANCISCO_ORIGIN,
    width,
    height,
    elevation,
    bounds: [minX, minY, maxX, maxY],
    cellSize: [(maxX - minX) / width, (maxY - minY) / height],
    elevationRange: [minimum, maximum]
  };
}

// ---------------------------------------------------------------------------------------------
// Deterministic synthetic fallbacks
// ---------------------------------------------------------------------------------------------

/** Mulberry32: small deterministic PRNG returning values in `[0, 1)`. */
export function createSeededRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

/** Manhattan-like street grid: 12 avenues x 100 streets, about 3 km x 8 km, rotated 29 degrees clockwise. */
const SYNTHETIC_GRID = {columns: 12, rows: 100, spacingX: 260, spacingY: 80, rotation: -0.506};

function getSyntheticGridPoint(column: number, row: number): [number, number] {
  const x = (column - SYNTHETIC_GRID.columns / 2) * SYNTHETIC_GRID.spacingX;
  const y = (row - SYNTHETIC_GRID.rows / 2) * SYNTHETIC_GRID.spacingY;
  const cosine = Math.cos(SYNTHETIC_GRID.rotation);
  const sine = Math.sin(SYNTHETIC_GRID.rotation);
  return [x * cosine - y * sine, x * sine + y * cosine];
}

function makeSyntheticRoads(): MapGraphsRoadNetwork {
  const builder = new RoadNetworkBuilder();
  const {columns, rows} = SYNTHETIC_GRID;
  const point = (column: number, row: number) => {
    const [x, y] = getSyntheticGridPoint(column, row);
    return {key: `${column},${row}`, x, y};
  };
  for (let row = 0; row < rows; row++) {
    const polyline = Array.from({length: columns}, (_, column) => point(column, row));
    // Alternate one-way cross streets like Manhattan; every tenth is a two-way arterial.
    builder.addPolyline(
      polyline,
      row % 10 === 0 ? 1 : 3,
      row % 10 === 0 ? 'B' : row % 2 ? 'F' : 'T'
    );
  }
  for (let column = 0; column < columns; column++) {
    const polyline = Array.from({length: rows}, (_, row) => point(column, row));
    builder.addPolyline(
      polyline,
      column % 4 === 0 ? 1 : 2,
      column % 4 === 0 ? 'B' : column % 2 ? 'F' : 'T'
    );
  }
  return builder.finish('synthetic', 'Synthetic Manhattan-like street grid');
}

function makeSyntheticTrips(): MapGraphsTrips {
  const random = createSeededRandom(7);
  const {columns, rows} = SYNTHETIC_GRID;
  const tripCount = 600;
  const positions: number[] = [];
  const timestamps: number[] = [];
  const tripOffsets = [0];
  const vendors: number[] = [];
  for (let trip = 0; trip < tripCount; trip++) {
    let column = Math.floor(random() * columns);
    let row = Math.floor(random() * rows);
    let time = random() * 1800;
    const steps = 20 + Math.floor(random() * 40);
    for (let step = 0; step < steps; step++) {
      const [x, y] = getSyntheticGridPoint(column, row);
      positions.push(x, y);
      timestamps.push(time);
      if (random() < 0.5)
        column = Math.max(0, Math.min(columns - 1, column + (random() < 0.5 ? -1 : 1)));
      else row = Math.max(0, Math.min(rows - 1, row + (random() < 0.5 ? -1 : 1)));
      time += 6 + random() * 18;
    }
    tripOffsets.push(positions.length / 2);
    vendors.push(trip % 2);
  }
  return {
    source: 'synthetic',
    attribution: 'Synthetic trips on a street grid',
    origin: NEW_YORK_ORIGIN,
    vertexPositions: Float32Array.from(positions),
    vertexTimestamps: Float32Array.from(timestamps),
    tripOffsets: Uint32Array.from(tripOffsets),
    vendors: Uint32Array.from(vendors),
    timeRange: [Math.min(...timestamps), Math.max(...timestamps)]
  };
}

function makeSyntheticPointsOfInterest(): MapGraphsPointsOfInterest {
  const random = createSeededRandom(11);
  const categoryNames = ['restaurant', 'cafe', 'shop', 'school', 'park', 'attraction'];
  const count = 3000;
  const positions = new Float32Array(count * 2);
  const categories = new Uint32Array(count);
  for (let index = 0; index < count; index++) {
    const [x, y] = getSyntheticGridPoint(
      random() * (SYNTHETIC_GRID.columns - 1),
      random() * (SYNTHETIC_GRID.rows - 1)
    );
    positions[index * 2] = x + (random() - 0.5) * 60;
    positions[index * 2 + 1] = y + (random() - 0.5) * 60;
    categories[index] = Math.floor(random() * categoryNames.length);
  }
  return {
    source: 'synthetic',
    attribution: 'Synthetic points of interest',
    origin: NEW_YORK_ORIGIN,
    positions,
    categories,
    categoryNames
  };
}

function makeSyntheticBikeParking(): MapGraphsBikeParking {
  const random = createSeededRandom(13);
  const count = 2500;
  const positions = new Float32Array(count * 2);
  const spaces = new Float32Array(count);
  const hotspots = [
    [2600, 2400],
    [500, 1200],
    [-1500, 0],
    [1800, -1800]
  ];
  for (let index = 0; index < count; index++) {
    const hotspot = hotspots[index % hotspots.length];
    const radius = 2200 * Math.sqrt(random());
    const angle = random() * Math.PI * 2;
    positions[index * 2] = hotspot[0] + Math.cos(angle) * radius * (index % 3 === 0 ? 1.6 : 0.6);
    positions[index * 2 + 1] =
      hotspot[1] + Math.sin(angle) * radius * (index % 3 === 0 ? 1.6 : 0.6);
    spaces[index] = 2 * (1 + Math.floor(random() * random() * 12));
  }
  return {
    source: 'synthetic',
    attribution: 'Synthetic bike parking',
    origin: SAN_FRANCISCO_ORIGIN,
    positions,
    spaces
  };
}

function makeSyntheticZipCodes(): MapGraphsPolygons {
  // A 4 x 3 tiling of jittered convex cells covering the synthetic bike parking.
  const random = createSeededRandom(17);
  const columns = 4;
  const rows = 3;
  const cellWidth = 2400;
  const cellHeight = 2400;
  const corners: [number, number][][] = [];
  for (let row = 0; row <= rows; row++) {
    corners.push([]);
    for (let column = 0; column <= columns; column++) {
      const interior = row > 0 && row < rows && column > 0 && column < columns;
      corners[row].push([
        (column - columns / 2) * cellWidth + (interior ? (random() - 0.5) * 900 : 0),
        (row - rows / 2) * cellHeight + 300 + (interior ? (random() - 0.5) * 900 : 0)
      ]);
    }
  }
  const features: {id: number; name: string; rings: [number, number][][]}[] = [];
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      const id = 94100 + features.length + 1;
      features.push({
        id,
        name: String(id),
        rings: [
          [
            corners[row][column],
            corners[row][column + 1],
            corners[row + 1][column + 1],
            corners[row + 1][column]
          ]
        ]
      });
    }
  }
  return makePolygons(features, 'synthetic', 'Synthetic ZIP-like polygons');
}

function makeSyntheticTerrain(): MapGraphsTerrain {
  const size = 256;
  const elevation = new Float32Array(size * size);
  const random = createSeededRandom(19);
  const hills = Array.from({length: 14}, () => ({
    x: random(),
    y: random(),
    radius: 0.04 + random() * 0.12,
    height: 40 + random() * 240
  }));
  for (let row = 0; row < size; row++) {
    for (let column = 0; column < size; column++) {
      const u = column / (size - 1);
      const v = row / (size - 1);
      let value = 0;
      for (const hill of hills) {
        const distanceSquared = (u - hill.x) ** 2 + (v - hill.y) ** 2;
        value += hill.height * Math.exp(-distanceSquared / (2 * hill.radius ** 2));
      }
      // Sea to the west and north, like the San Francisco peninsula.
      const land =
        Math.min(1, Math.max(0, (u - 0.06) * 12)) * Math.min(1, Math.max(0, (v - 0.04) * 12));
      elevation[row * size + column] = value * land;
    }
  }
  return makeTerrain(elevation, size, size, 'synthetic', 'Synthetic hills');
}
