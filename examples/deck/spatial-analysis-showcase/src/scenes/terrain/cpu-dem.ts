// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * CPU probes of a DEM for the terrain chapter: bilinear sampling, Horn's 3 x 3 neighbourhood,
 * a sight line with earth curvature, a horizon fan, geomorphon rays and the summit disc and ring.
 * They answer one cell or one ray at a time (a tooltip, a pinned cell, a drawn profile) with the
 * same definitions as the GPU contributors, so a story can show the algorithm that the GPU runs
 * for every cell. Pure TypeScript, no GPU, no allocation per cell beyond the returned values.
 *
 * Conventions shared by every function:
 * - Pixel coordinates are continuous "edge" coordinates `[x, y]`: pixel `i` spans `[i, i + 1)` and
 *   its centre is `i + 0.5`; row 0 is the north edge. Integer `column` / `row` arguments are pixel
 *   indices.
 * - The DEM rows are Web Mercator, not latitude: every longitude/latitude conversion goes through
 *   {@link TerrainDem.getPixelCoordinates}, never through a linear latitude formula.
 * - Ground distances use the per-row ground cell size (`mercatorCellSize * cos(latitude)`), which
 *   is the same in x and y for a conformal projection.
 */

import type {AlpsGrid} from './b14a-grid';
import type {AlpsTerrain} from './b14b-terrain';

/** Mean Earth radius in metres (the value the viewshed contributors use for curvature). */
export const EARTH_RADIUS_METERS = 6371008.8;

/** Standard atmospheric refraction coefficient `k` of terrestrial sight lines. */
export const DEFAULT_REFRACTION = 0.13;

/**
 * The curvature coefficient `c = (1 - k) / (2 R)` in 1/m: a point at distance `d` metres appears
 * `c d^2` metres lower than the flat-earth line of sight puts it.
 */
export function getCurvatureCoefficient(refraction: number = DEFAULT_REFRACTION): number {
  return (1 - refraction) / (2 * EARTH_RADIUS_METERS);
}

/**
 * How far the earth's surface (with refraction) falls away from a horizontal line at a distance:
 * `c d^2` in metres. At 10 km and `k = 0.13` it is about 6.8 m.
 */
export function earthDrop(distanceMeters: number, refraction: number = DEFAULT_REFRACTION): number {
  return getCurvatureCoefficient(refraction) * distanceMeters * distanceMeters;
}

/** A DEM as the probes and the relief ground read it. Build one with the `createTerrainDem*` helpers. */
export type TerrainDem = {
  /** Columns. */
  width: number;
  /** Rows; row 0 is the north edge. */
  height: number;
  /** Elevation in metres, `width * height`, row-major. Non-finite values are no data. */
  values: ArrayLike<number>;
  /** `[west, south, east, north]` degrees of the outer edges. */
  lngLatBounds: readonly [number, number, number, number];
  /** `[minX, minY, maxX, maxY]` metres around `origin`, the outer edges, for the layers. */
  layerBounds: readonly [number, number, number, number];
  /** `[longitude, latitude]` coordinate origin of the layers. */
  origin: readonly [number, number];
  /** Ground metres per cell at the central row. */
  groundCellSize: number;
  /** Ground metres per cell at a row. */
  getGroundCellSize: (row: number) => number;
  /** Continuous pixel coordinates `[x, y]` of a longitude and latitude (Web Mercator rows). */
  getPixelCoordinates: (longitude: number, latitude: number) => [number, number];
  /** `[longitude, latitude]` of a pixel position: `column`, `row` are centre indices (0.5 = edge). */
  getLongitudeLatitude: (column: number, row: number) => [number, number];
};

/** Describes an {@link AlpsGrid} (the `alps-dem` or `alps-dem-wide` raster) as a {@link TerrainDem}. */
export function createTerrainDemFromGrid(grid: AlpsGrid): TerrainDem {
  return {
    width: grid.width,
    height: grid.height,
    values: grid.cpuElevation,
    lngLatBounds: grid.lngLatBounds,
    layerBounds: grid.bounds,
    origin: grid.origin,
    groundCellSize: grid.groundCellSize,
    getGroundCellSize: grid.getGroundCellSize,
    getPixelCoordinates: grid.getPixelCoordinates,
    getLongitudeLatitude: grid.getLongitudeLatitude
  };
}

/**
 * Describes an {@link AlpsTerrain} (any analysis stride) as a {@link TerrainDem}. The terrain's
 * `getPixel` returns centre indices, so one half pixel is added to get edge coordinates.
 */
export function createTerrainDemFromTerrain(terrain: AlpsTerrain): TerrainDem {
  return {
    width: terrain.width,
    height: terrain.height,
    values: terrain.elevation,
    lngLatBounds: terrain.lngLatBounds,
    layerBounds: terrain.bounds,
    origin: terrain.origin,
    groundCellSize: terrain.groundCellSize,
    getGroundCellSize: terrain.getGroundCellSize,
    getPixelCoordinates(longitude, latitude) {
      const [column, row] = terrain.getPixel(longitude, latitude);
      return [column + 0.5, row + 0.5];
    },
    getLongitudeLatitude: terrain.getLongitudeLatitude
  };
}

// ---------------------------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------------------------

/** Horn's 3 x 3 neighbourhood of a cell. */
export type HornNeighbourhood = {
  column: number;
  row: number;
  /**
   * The nine heights in reading order `a b c / d e f / g h i` (north row first, west to east):
   * `neighbours[4]` is the cell itself. Cells off the raster repeat the nearest edge cell.
   */
  neighbours: number[];
  /** East-positive gradient, metres per metre: `((c + 2f + i) - (a + 2d + g)) / (8 cell)`. */
  dzdx: number;
  /** North-positive gradient, metres per metre: `((a + 2b + c) - (g + 2h + i)) / (8 cell)`. */
  dzdy: number;
  /** Slope in degrees, `atan(hypot(dzdx, dzdy))`. */
  slopeDeg: number;
  /** Direction the slope faces (downhill), degrees clockwise from north; NaN on flat ground. */
  aspectDeg: number;
  /** Ground metres per cell used. */
  cellSizeMeters: number;
};

/** The samples of one sight line, see {@link DemProbe.sampleRay}. All arrays have `count` entries. */
export type RaySamples = {
  count: number;
  /** Ground distance from the eye, metres, accumulated per row cell size. */
  distance: Float64Array;
  /** `[longitude, latitude]` of each sample, interleaved. */
  lngLat: Float64Array;
  /** Terrain height, metres. */
  terrain: Float64Array;
  /** Terrain height minus `c d^2`: the terrain as the curved earth presents it to the eye. */
  loweredTerrain: Float64Array;
  /** The straight sight line in the lowered frame, eye to target. */
  sightLine: Float64Array;
  /** Running maximum of the elevation angle (degrees) from the eye to the lowered terrain. */
  runningMaxAngle: Float64Array;
  /** `sightLine - loweredTerrain`: positive means the line clears the ground. */
  clearance: Float64Array;
  /** Index of the interior sample with the smallest clearance (the controlling obstacle), or -1. */
  blockingIndex: number;
  /** Smallest clearance over the interior samples, metres (Infinity without interior samples). */
  minimumClearance: number;
  /** True when some interior sample has negative clearance: the target is hidden. */
  blocked: boolean;
  eyeElevation: number;
  targetElevation: number;
  totalDistance: number;
};

/** Options of {@link DemProbe.sampleRay}. */
export type RayOptions = {
  /** Refraction coefficient `k`. Defaults to {@link DEFAULT_REFRACTION}. */
  refraction?: number;
  /** Eye height above the ground, metres. Defaults to 2. */
  eyeHeight?: number;
  /** Target height above the ground, metres. Defaults to 0. */
  targetHeight?: number;
};

/** One of the eight geomorphon lines of sight, see {@link DemProbe.geomorphonRays}. */
export type GeomorphonRay = {
  /** Compass name of the direction. */
  name: 'NE' | 'N' | 'NW' | 'W' | 'SW' | 'S' | 'SE' | 'E';
  /** Cell step `[columns, rows]` of the ray (rows grow south). */
  step: readonly [number, number];
  /** Cells examined along the ray (up to the search radius, raster edge or no-data). */
  cellCount: number;
  /** `[column, row]` of the last cell examined (the ray end to draw). */
  endCell: readonly [number, number];
  /** Zenith: the cell with the largest elevation angle; null when no cell was examined. */
  zenith: {cell: readonly [number, number]; count: number; height: number} | null;
  /** Nadir: the cell with the smallest elevation angle; null when no cell was examined. */
  nadir: {cell: readonly [number, number]; count: number; height: number} | null;
  /** +1 the zenith dominates (higher ground), -1 the nadir dominates, 0 flat. */
  sign: -1 | 0 | 1;
};

/** Names of the ten geomorphon forms, index = GRASS / contributor code (0 = invalid). */
export const GEOMORPHON_FORM_NAMES = [
  'invalid',
  'flat',
  'peak',
  'ridge',
  'shoulder',
  'spur',
  'slope',
  'hollow',
  'footslope',
  'valley',
  'pit'
] as const;

/** The eight geomorphons of one cell, see {@link DemProbe.geomorphonRays}. */
export type GeomorphonResult = {
  column: number;
  row: number;
  rays: GeomorphonRay[];
  /** The ternary pattern in GRASS direction order (NE, N, NW, W, SW, S, SE, E), e.g. `+ + 0 - - - 0 +`. */
  pattern: string;
  plusCount: number;
  minusCount: number;
  /** Landform code from the GRASS 9 x 9 table (see {@link GEOMORPHON_FORM_NAMES}); 0 when the cell is too close to the edge. */
  formCode: number;
  formName: (typeof GEOMORPHON_FORM_NAMES)[number];
};

/** The summit disc and ring of one cell, see {@link DemProbe.discAndRing}. */
export type DiscAndRing = {
  /** True when the cell is the strict maximum of its disc under the order (height, -index). */
  isMax: boolean;
  /** Highest ring cell other than the centre, metres; NaN when the ring is empty. */
  ringMax: number;
  /** `[column, row]` of the highest ring cell, or null. */
  ringMaxCell: readonly [number, number] | null;
  /** Centre height minus ring maximum, metres (Infinity for an empty ring, NaN for no data). */
  drop: number;
  /** A disc cell that beats the centre (when `isMax` is false), else null. */
  beatenBy: readonly [number, number] | null;
  discCount: number;
  ringCount: number;
  /** True when part of the disc lies outside the raster or on no-data. */
  incomplete: boolean;
};

/** Options of {@link DemProbe.horizonAngles}. */
export type HorizonOptions = {
  refraction?: number;
  /** Eye height above the ground, metres. Defaults to 2. */
  eyeHeight?: number;
  /** Ground step along each ray, metres. Defaults to the cell size. */
  stepMeters?: number;
};

// ---------------------------------------------------------------------------------------------
// The probe
// ---------------------------------------------------------------------------------------------

const RADIANS = Math.PI / 180;
const DEGREES = 180 / Math.PI;

/** The probes of one DEM; create it with {@link createDemProbe}. */
export type DemProbe = {
  readonly dem: TerrainDem;
  /** Elevation at a pixel index (NaN off the raster). */
  getElevation(column: number, row: number): number;
  /** Ground metres per cell at a row (cached). */
  getGroundCellSize(row: number): number;
  /**
   * Bilinear elevation at a longitude and latitude in metres; NaN outside the raster or when one
   * of the four cells has no data.
   */
  sampleAt(lngLat: readonly [number, number]): number;
  /** Bilinear elevation at pixel-edge coordinates (see the module doc). */
  sampleAtPixel(x: number, y: number): number;
  /** Horn's 3 x 3 neighbourhood, gradients, slope and aspect of a cell. */
  hornAt(column: number, row: number): HornNeighbourhood;
  /**
   * Samples the straight ground line between two points every `stepMeters` and evaluates the
   * line of sight in the curvature-lowered frame (heights minus `c d^2`), the same model as the
   * GPU viewshed: the eye is `eyeHeight` above the terrain at `from`, the target `targetHeight`
   * above the terrain at `to`.
   */
  sampleRay(
    from: readonly [number, number],
    to: readonly [number, number],
    stepMeters: number,
    options?: RayOptions
  ): RaySamples;
  /**
   * The horizon elevation angle (degrees above horizontal) of an eye in each direction, with the
   * curvature term `c d^2`: the maximum over distance of `atan((z - c d^2 - eye) / d)`. `directions`
   * is a count (evenly spaced from north) or azimuths in degrees clockwise from north. A direction
   * that never rises above the eye reports the (negative) best angle it found.
   */
  horizonAngles(
    lngLat: readonly [number, number],
    directions: number | readonly number[],
    maxMeters: number,
    options?: HorizonOptions
  ): Float64Array;
  /**
   * The eight geomorphon lines of sight of a cell out to `searchRadius` cells (the contributor's
   * `searchRadius`), flat threshold in degrees, GRASS `anglev1` comparison, as `GPUGeomorphons`
   * computes them.
   */
  geomorphonRays(
    column: number,
    row: number,
    searchRadius: number,
    flatThresholdDegrees: number
  ): GeomorphonResult;
  /**
   * The summit test of one cell: the disc `(dx cx)^2 + (dy cy)^2 <= r^2` (ground metres), the
   * ring (disc cells with an eight-neighbour outside the disc), and the drop to the ring maximum,
   * as `GPUTerrainSummits` evaluates it. `incomplete` selects `'reject'` (default: a disc that
   * leaves the raster is never a summit) or `'ignore'` (missing cells are absent).
   */
  discAndRing(
    column: number,
    row: number,
    radiusMeters: number,
    incomplete?: 'reject' | 'ignore'
  ): DiscAndRing;
};

/** Directions in the GPU contributor's order: NE, N, NW, W, SW, S, SE, E (column step, north step). */
const GEOMORPHON_DIRECTIONS = [
  {name: 'NE', column: 1, north: 1},
  {name: 'N', column: 0, north: 1},
  {name: 'NW', column: -1, north: 1},
  {name: 'W', column: -1, north: 0},
  {name: 'SW', column: -1, north: -1},
  {name: 'S', column: 0, north: -1},
  {name: 'SE', column: 1, north: -1},
  {name: 'E', column: 1, north: 0}
] as const;

/** GRASS landform codes by minus count (row) and plus count (column); 0 marks impossible pairs. */
const GEOMORPHON_FORM_TABLE: readonly (readonly number[])[] = [
  [1, 1, 1, 8, 8, 9, 9, 9, 10],
  [1, 1, 8, 8, 8, 9, 9, 9, 0],
  [1, 4, 6, 6, 7, 7, 9, 0, 0],
  [4, 4, 6, 6, 6, 7, 0, 0, 0],
  [4, 4, 5, 6, 6, 0, 0, 0, 0],
  [3, 3, 5, 5, 0, 0, 0, 0, 0],
  [3, 3, 3, 0, 0, 0, 0, 0, 0],
  [3, 3, 0, 0, 0, 0, 0, 0, 0],
  [2, 0, 0, 0, 0, 0, 0, 0, 0]
];

/**
 * Creates the probes of a DEM. The per-row ground cell size is computed once per row on first use.
 *
 * @example
 * const probe = createDemProbe(createTerrainDemFromGrid(grid));
 * const {slopeDeg, aspectDeg} = probe.hornAt(column, row);
 */
export function createDemProbe(dem: TerrainDem): DemProbe {
  const {width, height, values} = dem;
  const rowCellSize = new Float64Array(height).fill(Number.NaN);
  const getGroundCellSize = (row: number): number => {
    const clamped = Math.min(height - 1, Math.max(0, Math.round(row)));
    let size = rowCellSize[clamped];
    if (Number.isNaN(size)) {
      size = dem.getGroundCellSize(clamped);
      rowCellSize[clamped] = size;
    }
    return size;
  };
  const getElevation = (column: number, row: number): number =>
    column < 0 || row < 0 || column >= width || row >= height
      ? Number.NaN
      : values[row * width + column];
  const getClamped = (column: number, row: number): number =>
    values[
      Math.min(height - 1, Math.max(0, row)) * width + Math.min(width - 1, Math.max(0, column))
    ];

  const sampleAtPixel = (x: number, y: number): number => {
    if (!(x >= 0 && y >= 0 && x <= width && y <= height)) return Number.NaN;
    // Centre indices: pixel i has its centre at i + 0.5.
    const cx = Math.min(width - 1, Math.max(0, x - 0.5));
    const cy = Math.min(height - 1, Math.max(0, y - 0.5));
    const column = Math.min(width - 2, Math.floor(cx));
    const row = Math.min(height - 2, Math.floor(cy));
    const fractionX = cx - column;
    const fractionY = cy - row;
    const index = row * width + column;
    const top = values[index] * (1 - fractionX) + values[index + 1] * fractionX;
    const bottom = values[index + width] * (1 - fractionX) + values[index + width + 1] * fractionX;
    return top * (1 - fractionY) + bottom * fractionY;
  };
  const sampleAt = (lngLat: readonly [number, number]): number => {
    const [x, y] = dem.getPixelCoordinates(lngLat[0], lngLat[1]);
    return sampleAtPixel(x, y);
  };

  const hornAt = (column: number, row: number): HornNeighbourhood => {
    const neighbours: number[] = [];
    for (let rowOffset = -1; rowOffset <= 1; rowOffset++) {
      for (let columnOffset = -1; columnOffset <= 1; columnOffset++) {
        neighbours.push(getClamped(column + columnOffset, row + rowOffset));
      }
    }
    const [a, b, c, d, , f, g, h, i] = neighbours;
    const cellSizeMeters = getGroundCellSize(row);
    const dzdx = (c + 2 * f + i - (a + 2 * d + g)) / (8 * cellSizeMeters);
    const dzdy = (a + 2 * b + c - (g + 2 * h + i)) / (8 * cellSizeMeters);
    const gradient = Math.hypot(dzdx, dzdy);
    const slopeDeg = Math.atan(gradient) * DEGREES;
    // The slope faces downhill: the vector (-dzdx, -dzdy) in (east, north).
    const aspectDeg = gradient > 0 ? (Math.atan2(-dzdx, -dzdy) * DEGREES + 360) % 360 : Number.NaN;
    return {column, row, neighbours, dzdx, dzdy, slopeDeg, aspectDeg, cellSizeMeters};
  };

  const sampleRay = (
    from: readonly [number, number],
    to: readonly [number, number],
    stepMeters: number,
    options: RayOptions = {}
  ): RaySamples => {
    const coefficient = getCurvatureCoefficient(options.refraction);
    const eyeHeight = options.eyeHeight ?? 2;
    const targetHeight = options.targetHeight ?? 0;
    const [fromX, fromY] = dem.getPixelCoordinates(from[0], from[1]);
    const [toX, toY] = dem.getPixelCoordinates(to[0], to[1]);
    const lengthCells = Math.hypot(toX - fromX, toY - fromY);
    // Walk in cells, converting each step to metres with the cell size of the row it starts in.
    const distances: number[] = [0];
    const positions: number[] = [fromX, fromY];
    let walked = 0;
    let metres = 0;
    while (walked < lengthCells) {
      const t = walked / lengthCells;
      const cell = getGroundCellSize(fromY + (toY - fromY) * t);
      const advance = Math.min(Math.max(stepMeters, 1e-6) / cell, lengthCells - walked);
      walked += advance;
      metres += advance * cell;
      const u = lengthCells > 0 ? walked / lengthCells : 1;
      distances.push(metres);
      positions.push(fromX + (toX - fromX) * u, fromY + (toY - fromY) * u);
      if (advance <= 0) break;
    }
    const count = distances.length;
    const result: RaySamples = {
      count,
      distance: Float64Array.from(distances),
      lngLat: new Float64Array(count * 2),
      terrain: new Float64Array(count),
      loweredTerrain: new Float64Array(count),
      sightLine: new Float64Array(count),
      runningMaxAngle: new Float64Array(count),
      clearance: new Float64Array(count),
      blockingIndex: -1,
      minimumClearance: Number.POSITIVE_INFINITY,
      blocked: false,
      eyeElevation: Number.NaN,
      targetElevation: Number.NaN,
      totalDistance: metres
    };
    for (let index = 0; index < count; index++) {
      const x = positions[index * 2];
      const y = positions[index * 2 + 1];
      const [longitude, latitude] = dem.getLongitudeLatitude(x - 0.5, y - 0.5);
      result.lngLat[index * 2] = longitude;
      result.lngLat[index * 2 + 1] = latitude;
      const distance = result.distance[index];
      const terrain = sampleAtPixel(x, y);
      result.terrain[index] = terrain;
      result.loweredTerrain[index] = terrain - coefficient * distance * distance;
    }
    const eye = result.terrain[0] + eyeHeight;
    const target = result.loweredTerrain[count - 1] + targetHeight;
    result.eyeElevation = eye;
    result.targetElevation = result.terrain[count - 1] + targetHeight;
    let maximumAngle = -90;
    for (let index = 0; index < count; index++) {
      const distance = result.distance[index];
      const fraction = metres > 0 ? distance / metres : 1;
      const sight = eye + (target - eye) * fraction;
      result.sightLine[index] = sight;
      result.clearance[index] = sight - result.loweredTerrain[index];
      if (index > 0) {
        maximumAngle = Math.max(
          maximumAngle,
          Math.atan2(result.loweredTerrain[index] - eye, distance) * DEGREES
        );
      }
      result.runningMaxAngle[index] = maximumAngle;
      if (index > 0 && index < count - 1 && result.clearance[index] < result.minimumClearance) {
        result.minimumClearance = result.clearance[index];
        result.blockingIndex = index;
      }
    }
    result.blocked = result.minimumClearance < 0;
    return result;
  };

  const horizonAngles = (
    lngLat: readonly [number, number],
    directions: number | readonly number[],
    maxMeters: number,
    options: HorizonOptions = {}
  ): Float64Array => {
    const coefficient = getCurvatureCoefficient(options.refraction);
    const azimuths: readonly number[] =
      typeof directions === 'number'
        ? Array.from({length: directions}, (_, index) => (index * 360) / directions)
        : directions;
    const [originX, originY] = dem.getPixelCoordinates(lngLat[0], lngLat[1]);
    const eye = sampleAtPixel(originX, originY) + (options.eyeHeight ?? 2);
    const cell = getGroundCellSize(originY);
    const step = Math.max(options.stepMeters ?? cell, 1e-6);
    const angles = new Float64Array(azimuths.length);
    for (let direction = 0; direction < azimuths.length; direction++) {
      const azimuth = azimuths[direction] * RADIANS;
      // North is up the raster: rows decrease.
      const columnPerMeter = Math.sin(azimuth) / cell;
      const rowPerMeter = -Math.cos(azimuth) / cell;
      let best = -90;
      for (let distance = step; distance <= maxMeters; distance += step) {
        const x = originX + columnPerMeter * distance;
        const y = originY + rowPerMeter * distance;
        if (x < 0 || y < 0 || x > width || y > height) break;
        const terrain = sampleAtPixel(x, y);
        if (!Number.isFinite(terrain)) continue;
        const angle =
          Math.atan2(terrain - coefficient * distance * distance - eye, distance) * DEGREES;
        if (angle > best) best = angle;
      }
      angles[direction] = best;
    }
    return angles;
  };

  const geomorphonRays = (
    column: number,
    row: number,
    searchRadius: number,
    flatThresholdDegrees: number
  ): GeomorphonResult => {
    const centre = getElevation(column, row);
    const cell = getGroundCellSize(row);
    const tangent = Math.tan(flatThresholdDegrees * RADIANS);
    const searchDistance = searchRadius * cell * 0.99999;
    const rays: GeomorphonRay[] = [];
    let plusCount = 0;
    let minusCount = 0;
    // The contributor gives form 0 to cells within one cell of the raster border or on no data.
    const valid =
      Number.isFinite(centre) && column >= 1 && row >= 1 && column < width - 1 && row < height - 1;
    for (const direction of GEOMORPHON_DIRECTIONS) {
      const rowStep = -direction.north;
      const step = direction.column !== 0 && direction.north !== 0 ? Math.SQRT2 * cell : cell;
      let zenith: GeomorphonRay['zenith'] = null;
      let nadir: GeomorphonRay['nadir'] = null;
      let cellCount = 0;
      let endCell: [number, number] = [column, row];
      for (let count = 1; count * step < searchDistance; count++) {
        const sampleColumn = column + count * direction.column;
        const sampleRow = row + count * rowStep;
        if (sampleColumn < 0 || sampleColumn >= width || sampleRow < 0 || sampleRow >= height) {
          break;
        }
        cellCount = count;
        endCell = [sampleColumn, sampleRow];
        const sample = values[sampleRow * width + sampleColumn];
        if (!Number.isFinite(sample)) continue;
        const sampleHeight = sample - centre;
        if (!zenith || sampleHeight * zenith.count > zenith.height * count) {
          zenith = {cell: [sampleColumn, sampleRow], count, height: sampleHeight};
        }
        if (!nadir || sampleHeight * nadir.count < nadir.height * count) {
          nadir = {cell: [sampleColumn, sampleRow], count, height: sampleHeight};
        }
      }
      let sign: -1 | 0 | 1 = 0;
      if (zenith && nadir) {
        const zenithAbsolute = Math.abs(zenith.height);
        const nadirAbsolute = Math.abs(nadir.height);
        const zenithOver = zenithAbsolute > tangent * zenith.count * step;
        const nadirOver = nadirAbsolute > tangent * nadir.count * step;
        if (zenithOver || nadirOver) {
          // Absolute angles compare as |height| / count, cross-multiplied.
          const zenithScaled = zenithAbsolute * nadir.count;
          const nadirScaled = nadirAbsolute * zenith.count;
          sign = nadirScaled < zenithScaled ? 1 : nadirScaled > zenithScaled ? -1 : 0;
        }
      }
      if (sign > 0) plusCount++;
      if (sign < 0) minusCount++;
      rays.push({
        name: direction.name,
        step: [direction.column, rowStep],
        cellCount,
        endCell,
        zenith,
        nadir,
        sign
      });
    }
    const formCode = valid ? GEOMORPHON_FORM_TABLE[minusCount][plusCount] : 0;
    return {
      column,
      row,
      rays,
      pattern: rays.map(ray => (ray.sign > 0 ? '+' : ray.sign < 0 ? '-' : '0')).join(' '),
      plusCount,
      minusCount,
      formCode,
      formName: GEOMORPHON_FORM_NAMES[formCode]
    };
  };

  const discAndRing = (
    column: number,
    row: number,
    radiusMeters: number,
    incompleteNeighborhood: 'reject' | 'ignore' = 'reject'
  ): DiscAndRing => {
    const own = getElevation(column, row);
    const cell = getGroundCellSize(row);
    const reach = Math.ceil(radiusMeters / cell) + 1;
    const radiusSquared = radiusMeters * radiusMeters;
    const inDisc = (offsetColumn: number, offsetRow: number) =>
      (offsetColumn * cell) ** 2 + (offsetRow * cell) ** 2 <= radiusSquared;
    const ownIndex = row * width + column;
    const result: DiscAndRing = {
      isMax: Number.isFinite(own),
      ringMax: Number.NaN,
      ringMaxCell: null,
      drop: Number.NaN,
      beatenBy: null,
      discCount: 0,
      ringCount: 0,
      incomplete: false
    };
    if (!Number.isFinite(own)) {
      result.isMax = false;
      return result;
    }
    let ringHeight = Number.NEGATIVE_INFINITY;
    for (let offsetRow = -reach; offsetRow <= reach; offsetRow++) {
      for (let offsetColumn = -reach; offsetColumn <= reach; offsetColumn++) {
        if (!inDisc(offsetColumn, offsetRow)) continue;
        const sampleColumn = column + offsetColumn;
        const sampleRow = row + offsetRow;
        const neighbour = getElevation(sampleColumn, sampleRow);
        if (!Number.isFinite(neighbour)) {
          result.incomplete = true;
          continue;
        }
        result.discCount++;
        const isCentre = offsetColumn === 0 && offsetRow === 0;
        if (!isCentre) {
          const neighbourIndex = sampleRow * width + sampleColumn;
          // Strict maximum under (height, -index): equal heights go to the lower index.
          if (neighbour > own || (neighbour === own && neighbourIndex < ownIndex)) {
            if (result.isMax) result.beatenBy = [sampleColumn, sampleRow];
            result.isMax = false;
          }
        }
        let onRing = false;
        for (let k = -1; k <= 1 && !onRing; k++) {
          for (let j = -1; j <= 1; j++) {
            if ((j !== 0 || k !== 0) && !inDisc(offsetColumn + j, offsetRow + k)) {
              onRing = true;
              break;
            }
          }
        }
        if (onRing && !isCentre) {
          result.ringCount++;
          if (neighbour > ringHeight) {
            ringHeight = neighbour;
            result.ringMaxCell = [sampleColumn, sampleRow];
          }
        }
      }
    }
    if (incompleteNeighborhood === 'reject' && result.incomplete) result.isMax = false;
    if (result.ringCount > 0) {
      result.ringMax = ringHeight;
      result.drop = own - ringHeight;
    } else {
      result.drop = Number.POSITIVE_INFINITY;
    }
    return result;
  };

  return {
    dem,
    getElevation,
    getGroundCellSize,
    sampleAt,
    sampleAtPixel,
    hornAt,
    sampleRay,
    horizonAngles,
    geomorphonRays,
    discAndRing
  };
}
