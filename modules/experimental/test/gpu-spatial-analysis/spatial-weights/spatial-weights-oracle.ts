// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export {createSeededRandom} from '../neighbor-search/neighbor-search-oracle';

/** Polygon scene: polygons of rings of `[x, y]` vertices. */
export type OraclePolygons = [number, number][][][];

/** Flat GeoArrow-style layout of {@link OraclePolygons}. */
export type PolygonLayout = {
  positions: Float32Array;
  ringOffsets: Uint32Array;
  polygonOffsets: Uint32Array;
};

/** A CPU CSR result. */
export type OracleCSR = {
  offsets: number[];
  neighbors: number[];
  weights: number[];
  distances: number[];
};

/** Flattens polygons into positions, ring offsets and polygon offsets. */
export function flattenPolygons(polygons: OraclePolygons): PolygonLayout {
  const positions: number[] = [];
  const ringOffsets = [0];
  const polygonOffsets = [0];
  let ringCount = 0;
  for (const rings of polygons) {
    for (const ring of rings) {
      for (const [x, y] of ring) {
        positions.push(x, y);
      }
      ringOffsets.push(positions.length / 2);
      ringCount++;
    }
    polygonOffsets.push(ringCount);
  }
  return {
    positions: new Float32Array(positions),
    ringOffsets: new Uint32Array(ringOffsets),
    polygonOffsets: new Uint32Array(polygonOffsets)
  };
}

function quantise(value: number, tolerance: number): number {
  const x = Math.fround(value);
  if (tolerance <= 0) {
    return x === 0 ? 0 : x;
  }
  return Math.floor(Math.fround(Math.fround(x / Math.fround(tolerance)) + 0.5));
}

/**
 * Brute-force contiguity: queen shares a snapped vertex, rook shares a ring edge (both snapped
 * endpoints, either direction). Returns the CSR with binary weights.
 */
export function computeContiguityOracle(
  polygons: OraclePolygons,
  criterion: 'queen' | 'rook',
  snapTolerance = 0
): OracleCSR {
  const owners = new Map<string, Set<number>>();
  const add = (key: string, polygon: number) => {
    let set = owners.get(key);
    if (!set) {
      set = new Set();
      owners.set(key, set);
    }
    set.add(polygon);
  };
  const pointKey = (point: [number, number]) =>
    `${quantise(point[0], snapTolerance)},${quantise(point[1], snapTolerance)}`;
  polygons.forEach((rings, polygon) => {
    for (const ring of rings) {
      ring.forEach((point, index) => {
        if (criterion === 'queen') {
          add(pointKey(point), polygon);
        } else {
          const first = pointKey(point);
          const second = pointKey(ring[(index + 1) % ring.length]);
          if (first !== second) {
            add(first < second ? `${first}|${second}` : `${second}|${first}`, polygon);
          }
        }
      });
    }
  });
  const neighborSets = polygons.map(() => new Set<number>());
  for (const set of owners.values()) {
    for (const first of set) {
      for (const second of set) {
        if (first !== second) {
          neighborSets[first].add(second);
        }
      }
    }
  }
  const csr: OracleCSR = {offsets: [0], neighbors: [], weights: [], distances: []};
  for (const set of neighborSets) {
    for (const neighbor of [...set].sort((a, b) => a - b)) {
      csr.neighbors.push(neighbor);
      csr.weights.push(1);
    }
    csr.offsets.push(csr.neighbors.length);
  }
  return csr;
}

/** CPU lattice weights, ascending IDs. */
export function computeLatticeOracle(options: {
  width: number;
  height: number;
  criterion: 'rook' | 'queen';
  radius?: number;
  mask?: Uint32Array;
  cellSize?: [number, number];
}): OracleCSR {
  const {width, height, criterion, mask} = options;
  const radius = options.radius ?? 1;
  const [cellWidth, cellHeight] = options.cellSize ?? [1, 1];
  const csr: OracleCSR = {offsets: [0], neighbors: [], weights: [], distances: []};
  for (let id = 0; id < width * height; id++) {
    const cellX = id % width;
    const cellY = Math.floor(id / width);
    if (!mask || mask[id]) {
      for (let dy = -radius; dy <= radius; dy++) {
        for (let dx = -radius; dx <= radius; dx++) {
          const x = cellX + dx;
          const y = cellY + dy;
          if (x < 0 || y < 0 || x >= width || y >= height || (dx === 0 && dy === 0)) {
            continue;
          }
          if (criterion === 'rook' && Math.abs(dx) + Math.abs(dy) > radius) {
            continue;
          }
          const neighbor = y * width + x;
          if (mask && !mask[neighbor]) {
            continue;
          }
          csr.neighbors.push(neighbor);
          csr.weights.push(1);
          csr.distances.push(Math.hypot(dx * cellWidth, dy * cellHeight));
        }
      }
    }
    csr.offsets.push(csr.neighbors.length);
  }
  return csr;
}

/** Kernel profile, as PySAL `Kernel` (z = d / h, z = 0 when h <= 0). */
export function getKernelWeight(kernel: string, distance: number, bandwidth: number): number {
  const z = bandwidth > 0 ? distance / bandwidth : 0;
  let weight: number;
  switch (kernel) {
    case 'gaussian':
      weight = Math.exp(-0.5 * z * z) * 0.3989422804014327;
      break;
    case 'triangular':
      weight = Math.max(1 - z, 0);
      break;
    case 'epanechnikov':
      weight = 0.75 * Math.max(1 - z * z, 0);
      break;
    case 'bisquare':
      weight = 0.9375 * Math.max(1 - z * z, 0) ** 2;
      break;
    default:
      weight = z <= 1 ? 0.5 : 0;
  }
  return Number.isFinite(weight) && weight >= 0 ? weight : 0;
}

/** CPU transform of a CSR. Returns new weights aligned with `csr.neighbors`. */
export function computeTransformOracle(
  csr: OracleCSR,
  operation: 'row' | 'binary' | 'kernel' | 'symmetrize' | 'double' | 'variance',
  options: {kernel?: string; bandwidth?: number | 'adaptive'; doubleSum?: 'one' | 'rows'} = {}
): number[] {
  const rows = csr.offsets.length - 1;
  const result = csr.weights.slice();
  if (operation === 'double') {
    const s0 = csr.weights.reduce((sum, weight) => sum + weight, 0);
    const scale = (options.doubleSum === 'rows' ? rows : 1) / s0;
    return csr.weights.map(weight => (s0 > 0 && Number.isFinite(scale) ? weight * scale : 0));
  }
  if (operation === 'variance') {
    // libpysal W.transform = 'V': s_ij = w_ij / sqrt(sum_j w_ij^2); w' = s * n / sum(s).
    const norms: number[] = [];
    let q = 0;
    for (let row = 0; row < rows; row++) {
      let squares = 0;
      for (let slot = csr.offsets[row]; slot < csr.offsets[row + 1]; slot++) {
        squares += csr.weights[slot] ** 2;
      }
      norms.push(Math.sqrt(squares));
      for (let slot = csr.offsets[row]; slot < csr.offsets[row + 1]; slot++) {
        q += norms[row] > 0 ? csr.weights[slot] / norms[row] : 0;
      }
    }
    for (let row = 0; row < rows; row++) {
      for (let slot = csr.offsets[row]; slot < csr.offsets[row + 1]; slot++) {
        result[slot] = norms[row] > 0 && q > 0 ? (csr.weights[slot] / norms[row]) * (rows / q) : 0;
      }
    }
    return result;
  }
  for (let row = 0; row < rows; row++) {
    const begin = csr.offsets[row];
    const end = csr.offsets[row + 1];
    if (operation === 'row') {
      let sum = 0;
      for (let slot = begin; slot < end; slot++) sum += csr.weights[slot];
      for (let slot = begin; slot < end; slot++) {
        result[slot] = sum > 0 && Number.isFinite(sum) ? csr.weights[slot] / sum : 0;
      }
    } else if (operation === 'binary') {
      for (let slot = begin; slot < end; slot++) result[slot] = csr.weights[slot] > 0 ? 1 : 0;
    } else if (operation === 'kernel') {
      let bandwidth = options.bandwidth ?? 'adaptive';
      if (bandwidth === 'adaptive') {
        bandwidth = 0;
        for (let slot = begin; slot < end; slot++) {
          bandwidth = Math.max(bandwidth, csr.distances[slot]);
        }
      }
      for (let slot = begin; slot < end; slot++) {
        result[slot] = getKernelWeight(
          options.kernel ?? 'triangular',
          csr.distances[slot],
          bandwidth
        );
      }
    } else {
      for (let slot = begin; slot < end; slot++) {
        const neighbor = csr.neighbors[slot];
        let reverse = 0;
        for (let other = csr.offsets[neighbor]; other < csr.offsets[neighbor + 1]; other++) {
          if (csr.neighbors[other] === row) reverse = csr.weights[other];
        }
        result[slot] = 0.5 * (csr.weights[slot] + reverse);
      }
    }
  }
  return result;
}

/** CPU spatial lag. */
export function computeLagOracle(
  csr: OracleCSR,
  values: ArrayLike<number>,
  mask?: Uint32Array,
  normalize = false
): number[] {
  const rows = csr.offsets.length - 1;
  const lag: number[] = [];
  for (let row = 0; row < rows; row++) {
    let sum = 0;
    let weightSum = 0;
    if (!mask || mask[row]) {
      for (let slot = csr.offsets[row]; slot < csr.offsets[row + 1]; slot++) {
        const neighbor = csr.neighbors[slot];
        if (neighbor < rows && (!mask || mask[neighbor])) {
          sum += csr.weights[slot] * values[neighbor];
          weightSum += csr.weights[slot];
        }
      }
    }
    lag.push(normalize ? (weightSum > 0 ? sum / weightSum : 0) : sum);
  }
  return lag;
}

/** Throws when a CSR violates the {@link GPUSpatialWeights} invariants. */
export function assertValidCSR(csr: OracleCSR, rows: number, label: string): void {
  if (csr.offsets.length !== rows + 1 || csr.offsets[0] !== 0) {
    throw new Error(`${label}: offsets must hold rows + 1 entries starting at 0`);
  }
  for (let row = 0; row < rows; row++) {
    const begin = csr.offsets[row];
    const end = csr.offsets[row + 1];
    if (end < begin) throw new Error(`${label}: offsets decrease at row ${row}`);
    for (let slot = begin; slot < end; slot++) {
      if (csr.neighbors[slot] === row) throw new Error(`${label}: row ${row} lists itself`);
      if (slot > begin && csr.neighbors[slot] <= csr.neighbors[slot - 1]) {
        throw new Error(`${label}: row ${row} neighbors are not strictly ascending`);
      }
      if (csr.neighbors[slot] >= rows) throw new Error(`${label}: neighbor out of range`);
      if (!(csr.weights[slot] >= 0) || !Number.isFinite(csr.weights[slot])) {
        throw new Error(`${label}: weight at slot ${slot} is not finite and non-negative`);
      }
    }
  }
}

/** Returns whether the CSR pattern is symmetric. */
export function isSymmetricPattern(csr: OracleCSR): boolean {
  const rows = csr.offsets.length - 1;
  const pairs = new Set<number>();
  for (let row = 0; row < rows; row++) {
    for (let slot = csr.offsets[row]; slot < csr.offsets[row + 1]; slot++) {
      pairs.add(row * rows + csr.neighbors[slot]);
    }
  }
  for (const pair of pairs) {
    if (!pairs.has((pair % rows) * rows + Math.floor(pair / rows))) return false;
  }
  return true;
}
