// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * The numbers behind the bixi-bundles story, as pure functions on typed arrays: the work box the
 * bundling kernel is measured in, the Pareto curve of the busiest pairs, and what a bundled
 * polyline does to the map (stretch, area touched, distance from the streets the routed rides
 * used). Everything here runs on the CPU, once per readback, never per frame.
 */

/** Metres per degree of latitude on a sphere of the mean Earth radius. */
export const METERS_PER_DEGREE = 111_195;
/** `GPUEdgeBundling` pads its square work box by 5 % on each side. */
const WORK_BOX_SCALE = 1.1;
/** Side of a coverage cell in degrees (about 220 x 160 m at Montreal). */
const COVERAGE_CELL_DEGREES = 0.002;
/** Longest stretch of a line tested against the streets at once, in metres. */
const SAMPLE_SPACING_METERS = 100;
/** Spacing of the points a routed ride is densified to before it is indexed, in metres. */
const RIDE_DENSIFY_METERS = 30;

/** Ground distance in kilometres between two lon/lat points (equirectangular, fine at city scale). */
export function getKilometres(
  longitudeA: number,
  latitudeA: number,
  longitudeB: number,
  latitudeB: number
): number {
  const middle = ((latitudeA + latitudeB) / 2) * (Math.PI / 180);
  const dx = (longitudeB - longitudeA) * Math.cos(middle);
  const dy = latitudeB - latitudeA;
  return (Math.hypot(dx, dy) * METERS_PER_DEGREE) / 1000;
}

/**
 * Side of the square work box `GPUEdgeBundling` measures its kernel radius in, in metres: the
 * larger of the east-west extent (scaled by the cosine of the mid-latitude, the `geographic`
 * option) and the north-south extent of the live edges' endpoints, padded by 5 % on each side.
 * It is the GPU's own arithmetic (`getBox` in the contributor), repeated on the CPU so the ring
 * and the readouts can say the radius in metres.
 */
export function getWorkBoxSideMeters(
  lngLat: Float32Array,
  sources: Uint32Array,
  targets: Uint32Array,
  mask: Uint32Array,
  edgeCount: number
): number {
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (let edge = 0; edge < edgeCount; edge++) {
    if (mask[edge] !== 1) continue;
    for (const station of [sources[edge], targets[edge]]) {
      const x = lngLat[station * 2];
      const y = lngLat[station * 2 + 1];
      minX = Math.min(minX, x);
      maxX = Math.max(maxX, x);
      minY = Math.min(minY, y);
      maxY = Math.max(maxY, y);
    }
  }
  if (!Number.isFinite(minX)) return 0;
  const scale = Math.max(Math.cos(0.5 * (minY + maxY) * (Math.PI / 180)), 0.01);
  const extent = Math.max((maxX - minX) * scale, maxY - minY);
  return extent * WORK_BOX_SCALE * METERS_PER_DEGREE;
}

/** Counts of `values` in `bins` equal bins over `[low, high]`; the last bin also holds the overflow. */
export function binValues(
  values: ArrayLike<number>,
  low: number,
  high: number,
  bins: number
): number[] {
  const counts = new Array<number>(bins).fill(0);
  for (let index = 0; index < values.length; index++) {
    const t = (values[index] - low) / (high - low);
    counts[Math.min(bins - 1, Math.max(0, Math.floor(t * bins)))]++;
  }
  return counts;
}

/** Median of `values` (not modified). `NaN` for an empty list. */
export function getMedian(values: ArrayLike<number>): number {
  if (values.length === 0) return Number.NaN;
  const sorted = Float64Array.from(values).sort();
  const middle = sorted.length >> 1;
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

/**
 * The Pareto curve of the pairs ranked by rides: rank on x, the cumulative share of all rides
 * between two stations on y. `cumulative[i]` is the rides on the busiest `i` pairs; ranks are
 * spaced evenly on a log axis, because the curve's whole story is in its first few thousand.
 */
export function getParetoSeries(
  cumulative: Float64Array,
  totalRides: number,
  samples = 160
): {x: number[]; y: number[]} {
  const pairCount = cumulative.length - 1;
  const ranks = new Set<number>([1, pairCount]);
  for (let sample = 0; sample < samples; sample++) {
    ranks.add(Math.round(pairCount ** (sample / (samples - 1))));
  }
  const x = [...ranks].sort((a, b) => a - b);
  return {x, y: x.map(rank => cumulative[rank] / totalRides)};
}

/** A lookup of the streets the routed rides used: is any ride within a distance of a point. */
export type StreetIndex = {
  /** True when a ride passes within `radiusMeters` of `[longitude, latitude]`. */
  isNear: (longitude: number, latitude: number, radiusMeters: number) => boolean;
};

/**
 * Indexes routed rides for nearest-street tests: every ride is densified to a point every 30 m
 * and hashed into 100 m cells, so a test looks at nine cells.
 */
export function createStreetIndex(
  vertices: Float32Array,
  offsets: Uint32Array,
  origin: readonly [number, number],
  cellMeters = 100
): StreetIndex {
  const metersPerLongitude = METERS_PER_DEGREE * Math.cos(origin[1] * (Math.PI / 180));
  const cells = new Map<number, number[]>();
  const keyOf = (cellX: number, cellY: number) => (cellX + 32768) * 65536 + (cellY + 32768);
  const add = (x: number, y: number) => {
    const key = keyOf(Math.floor(x / cellMeters), Math.floor(y / cellMeters));
    const list = cells.get(key);
    if (list) list.push(x, y);
    else cells.set(key, [x, y]);
  };
  const rideCount = offsets.length - 1;
  for (let ride = 0; ride < rideCount; ride++) {
    for (let vertex = offsets[ride]; vertex < offsets[ride + 1]; vertex++) {
      const x = (vertices[vertex * 2] - origin[0]) * metersPerLongitude;
      const y = (vertices[vertex * 2 + 1] - origin[1]) * METERS_PER_DEGREE;
      add(x, y);
      if (vertex + 1 >= offsets[ride + 1]) continue;
      const nextX = (vertices[vertex * 2 + 2] - origin[0]) * metersPerLongitude;
      const nextY = (vertices[vertex * 2 + 3] - origin[1]) * METERS_PER_DEGREE;
      const steps = Math.ceil(Math.hypot(nextX - x, nextY - y) / RIDE_DENSIFY_METERS);
      for (let step = 1; step < steps; step++) {
        add(x + ((nextX - x) * step) / steps, y + ((nextY - y) * step) / steps);
      }
    }
  }
  return {
    isNear(longitude, latitude, radiusMeters) {
      const x = (longitude - origin[0]) * metersPerLongitude;
      const y = (latitude - origin[1]) * METERS_PER_DEGREE;
      const cellX = Math.floor(x / cellMeters);
      const cellY = Math.floor(y / cellMeters);
      const reach = Math.max(1, Math.ceil(radiusMeters / cellMeters));
      const limit = radiusMeters * radiusMeters;
      for (let dx = -reach; dx <= reach; dx++) {
        for (let dy = -reach; dy <= reach; dy++) {
          const list = cells.get(keyOf(cellX + dx, cellY + dy));
          if (!list) continue;
          for (let index = 0; index < list.length; index += 2) {
            if ((list[index] - x) ** 2 + (list[index + 1] - y) ** 2 <= limit) return true;
          }
        }
      }
      return false;
    }
  };
}

/** Input of {@link measureBundles}. */
export type MeasureBundlesInput = {
  /** `float32x2` `[longitude, latitude]` rows, edge-major (the `GPUEdgeBundling` `paths`). */
  paths: Float32Array;
  pointsPerEdge: number;
  edgeCount: number;
  /** One uint32 per edge; only edges set to 1 are measured. */
  mask: Uint32Array;
  /** The routed rides' streets, when the off-street shares are wanted. */
  streets?: StreetIndex | null;
  /** Distance from a ride within which a line counts as on a street, metres. */
  streetMeters?: number;
};

/** What the bundled polylines did to the map, against the straight edges. */
export type BundleMeasures = {
  /** Bundled length over straight length of every live edge longer than 300 m. */
  stretches: number[];
  /** Total bundled length over total straight length, or `null` with no live edge. */
  meanStretch: number | null;
  /** Map area touched by the straight lines and by the bundles, in km2 (0.002 degree cells). */
  straightAreaKm2: number;
  bundledAreaKm2: number;
  /** Share of line length farther than the street distance from every ride; `null` unmeasured. */
  offStreetStraight: number | null;
  offStreetBundled: number | null;
  /** The busiest live edge (the first one: edges are ranked by rides) and its middle control point. */
  busiestEdge: number | null;
  busiestMidpoint: [number, number] | null;
};

/** Area of one 0.002-degree coverage cell at latitude `latitude`, in km2. */
function getCoverageCellKm2(latitude: number): number {
  const height = (COVERAGE_CELL_DEGREES * METERS_PER_DEGREE) / 1000;
  return height * height * Math.cos(latitude * (Math.PI / 180));
}

/**
 * Measures live bundled polylines against the straight line between their pinned ends: path
 * stretch, the map cells each touches, and, with a street index, the share of each that lies
 * farther than a street distance from every routed ride.
 */
export function measureBundles(input: MeasureBundlesInput): BundleMeasures {
  const {paths, pointsPerEdge, edgeCount, mask, streets} = input;
  const streetMeters = input.streetMeters ?? 100;
  const stretches: number[] = [];
  let straightKilometres = 0;
  let bundledKilometres = 0;
  const straightCells = new Set<number>();
  const bundledCells = new Set<number>();
  let latitudeSum = 0;
  let liveEdges = 0;
  let farStraight = 0;
  let totalStraight = 0;
  let farBundled = 0;
  let totalBundled = 0;
  let busiestEdge: number | null = null;
  let busiestMidpoint: [number, number] | null = null;

  const markCells = (cells: Set<number>, ax: number, ay: number, bx: number, by: number) => {
    const steps = Math.max(
      1,
      Math.ceil(Math.max(Math.abs(bx - ax), Math.abs(by - ay)) / (COVERAGE_CELL_DEGREES * 0.5))
    );
    for (let step = 0; step <= steps; step++) {
      const t = step / steps;
      const x = Math.floor((ax + (bx - ax) * t) / COVERAGE_CELL_DEGREES);
      const y = Math.floor((ay + (by - ay) * t) / COVERAGE_CELL_DEGREES);
      cells.add((x + 100000) * 200000 + (y + 100000));
    }
  };
  /** Samples a segment about every 100 m and counts the length that is far from every ride. */
  const sampleStreets = (ax: number, ay: number, bx: number, by: number) => {
    if (!streets) return {far: 0, total: 0};
    const length = getKilometres(ax, ay, bx, by) * 1000;
    const steps = Math.max(1, Math.ceil(length / SAMPLE_SPACING_METERS));
    let far = 0;
    for (let step = 0; step < steps; step++) {
      const t = (step + 0.5) / steps;
      if (!streets.isNear(ax + (bx - ax) * t, ay + (by - ay) * t, streetMeters)) far++;
    }
    return {far: (far * length) / steps, total: length};
  };

  for (let edge = 0; edge < edgeCount; edge++) {
    if (mask[edge] !== 1) continue;
    const first = edge * pointsPerEdge * 2;
    const last = first + (pointsPerEdge - 1) * 2;
    if (Number.isNaN(paths[first]) || Number.isNaN(paths[last])) continue;
    liveEdges++;
    if (busiestMidpoint === null) {
      busiestEdge = edge;
      const middle = first + Math.floor(pointsPerEdge / 2) * 2;
      busiestMidpoint = [paths[middle], paths[middle + 1]];
    }
    latitudeSum += paths[first + 1];
    const direct = getKilometres(paths[first], paths[first + 1], paths[last], paths[last + 1]);
    straightKilometres += direct;
    markCells(straightCells, paths[first], paths[first + 1], paths[last], paths[last + 1]);
    const straightStreets = sampleStreets(
      paths[first],
      paths[first + 1],
      paths[last],
      paths[last + 1]
    );
    farStraight += straightStreets.far;
    totalStraight += straightStreets.total;
    let along = 0;
    for (let point = 0; point < pointsPerEdge - 1; point++) {
      const a = first + point * 2;
      along += getKilometres(paths[a], paths[a + 1], paths[a + 2], paths[a + 3]);
      markCells(bundledCells, paths[a], paths[a + 1], paths[a + 2], paths[a + 3]);
      const bundledStreets = sampleStreets(paths[a], paths[a + 1], paths[a + 2], paths[a + 3]);
      farBundled += bundledStreets.far;
      totalBundled += bundledStreets.total;
    }
    bundledKilometres += along;
    if (direct > 0.3) stretches.push(along / direct);
  }
  const cellKm2 = getCoverageCellKm2(liveEdges ? latitudeSum / liveEdges : 45.5);
  return {
    stretches,
    meanStretch: straightKilometres > 0 ? bundledKilometres / straightKilometres : null,
    straightAreaKm2: straightCells.size * cellKm2,
    bundledAreaKm2: bundledCells.size * cellKm2,
    offStreetStraight: streets && totalStraight > 0 ? farStraight / totalStraight : null,
    offStreetBundled: streets && totalBundled > 0 ? farBundled / totalBundled : null,
    busiestEdge,
    busiestMidpoint
  };
}

/** A station name without its street suffix: "Métro Berri-UQAM (Berri / Sainte-Catherine)" to "Métro Berri-UQAM". */
export function getStationLabel(name: string): string {
  return name.split(' (')[0].trim();
}
