// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPU_RASTER_D8_DIRECTIONS,
  getRasterD8Distance,
  getRasterGroundCellSize,
  type RasterGridSettings
} from '../../../src/gpu-raster/cost-distance/raster-grid-utils';
import type {GPUTerrainCellSizeMode} from '../../../src/gpu-terrain/terrain-analysis';

/** Inputs of {@link accumulateRoutingOnCPU}. */
export type RoutingOracleInput = {
  routing: 'd-infinity' | 'mfd-freeman' | 'mfd-quinn';
  width: number;
  height: number;
  cellSizeMode: GPUTerrainCellSizeMode;
  /** Grid settings holding the float32 values the GPU reads. */
  gridSettings: RasterGridSettings;
  /** Packed float32 `settings[6]`; not finite and positive selects the routing's published exponent. */
  flowExponent: number;
  /** Routing surface, NaN for invalid cells. */
  surface: Float32Array;
  isValid: (cell: number) => boolean;
  /** D8 receivers after optional flat resolution (fallback for cells without a lower neighbor). */
  receivers: Uint32Array;
  /** Per-cell weights (runoff and optional area) as the GPU computes them. */
  weights: Float32Array;
};

const NONE = 0xffffffff;

/** One outgoing flow edge of a cell. */
export type RoutingEdge = {neighbor: number; direction: number; fraction: number};

function getNeighbor(input: RoutingOracleInput, cell: number, direction: number): number {
  const {columnOffset, rowOffset} = GPU_RASTER_D8_DIRECTIONS[direction];
  const column = (cell % input.width) + columnOffset;
  const row = Math.floor(cell / input.width) + rowOffset;
  return column < 0 || row < 0 || column >= input.width || row >= input.height
    ? NONE
    : row * input.width + column;
}

/** D-infinity facets in the documented order: (cardinal, diagonal) E,S,W,N each with its diagonals, lower index first. */
const FACETS: [number, number][] = [];
for (const cardinal of [0, 2, 4, 6]) {
  const diagonals = [(cardinal + 7) & 7, (cardinal + 1) & 7].sort((a, b) => a - b);
  for (const diagonal of diagonals) {
    FACETS.push([cardinal, diagonal]);
  }
}

/**
 * Returns the positive outgoing flow fractions of one valid cell (float64 evaluation of the
 * float32 surface). Cells with no strictly lower usable neighbor return the single fallback edge
 * to `receivers[cell]`, or nothing when that is none.
 */
export function getRoutingEdgesOnCPU(input: RoutingOracleInput, cell: number): RoutingEdge[] {
  const {surface, width, height, cellSizeMode, gridSettings} = input;
  const center = surface[cell];
  const row = Math.floor(cell / width);
  const usable = (neighbor: number) => neighbor !== NONE && Number.isFinite(surface[neighbor]);
  const distance = (direction: number) =>
    getRasterD8Distance(cellSizeMode, direction, row, height, gridSettings);
  const fallback = (): RoutingEdge[] => {
    const receiver = input.receivers[cell];
    if (receiver === NONE) {
      return [];
    }
    for (let direction = 0; direction < 8; direction++) {
      if (getNeighbor(input, cell, direction) === receiver) {
        return [{neighbor: receiver, direction, fraction: 1}];
      }
    }
    return [];
  };
  if (!Number.isFinite(center)) {
    return [];
  }

  if (input.routing === 'd-infinity') {
    let bestSlope = 0;
    let best: {cardinal: number; diagonal: number; fractionDiagonal: number} | null = null;
    for (const [cardinal, diagonal] of FACETS) {
      const n1 = getNeighbor(input, cell, cardinal);
      const n2 = getNeighbor(input, cell, diagonal);
      const cardinalUsable = usable(n1);
      const diagonalUsable = usable(n2);
      if (!cardinalUsable && !diagonalUsable) {
        continue;
      }
      const e1 = cardinalUsable ? surface[n1] : 0;
      const e2 = diagonalUsable ? surface[n2] : 0;
      const d1 = distance(cardinal);
      const dd = distance(diagonal);
      // The other axis' ground size at the diagonal move's midpoint row.
      const diagonalRow = row + GPU_RASTER_D8_DIRECTIONS[diagonal].rowOffset;
      const ground = getRasterGroundCellSize(
        cellSizeMode,
        (row + diagonalRow + 1) * 0.5,
        height,
        gridSettings
      );
      const d2 = cardinal === 0 || cardinal === 4 ? ground[1] : ground[0];
      let slope = 0;
      let fractionDiagonal = 0;
      if (cardinalUsable && diagonalUsable) {
        const s1 = (center - e1) / d1;
        const s2 = (e1 - e2) / d2;
        if (s1 > 0 && s2 === 0) {
          slope = s1;
        } else if (s1 > 0 && s2 > 0) {
          const angle = Math.atan2(s2, s1);
          const maximumAngle = Math.atan2(d2, d1);
          if (angle <= maximumAngle) {
            slope = Math.hypot(s1, s2);
            fractionDiagonal = angle / maximumAngle;
          } else {
            slope = (center - e2) / dd;
            fractionDiagonal = 1;
          }
        } else if (s1 > 0) {
          slope = s1;
        } else if (s2 > 0) {
          slope = (center - e2) / dd;
          fractionDiagonal = 1;
        }
      } else if (cardinalUsable) {
        slope = (center - e1) / d1;
      } else {
        slope = (center - e2) / dd;
        fractionDiagonal = 1;
      }
      if (slope > bestSlope) {
        bestSlope = slope;
        best = {cardinal, diagonal, fractionDiagonal};
      }
    }
    if (!best) {
      return fallback();
    }
    const edges: RoutingEdge[] = [];
    for (const [direction, fraction] of [
      [best.cardinal, 1 - best.fractionDiagonal],
      [best.diagonal, best.fractionDiagonal]
    ]) {
      if (fraction > 0) {
        edges.push({neighbor: getNeighbor(input, cell, direction), direction, fraction});
      }
    }
    return edges.sort((a, b) => a.direction - b.direction);
  }

  const quinn = input.routing === 'mfd-quinn';
  const exponent =
    Number.isFinite(input.flowExponent) && input.flowExponent > 0
      ? input.flowExponent
      : quinn
        ? 1
        : 1.1;
  const [groundX, groundY] = getRasterGroundCellSize(cellSizeMode, row + 0.5, height, gridSettings);
  const lower: {neighbor: number; direction: number; score: number}[] = [];
  for (let direction = 0; direction < 8; direction++) {
    const neighbor = getNeighbor(input, cell, direction);
    if (!usable(neighbor) || !(center - surface[neighbor] > 0)) {
      continue;
    }
    const tangent = (center - surface[neighbor]) / distance(direction);
    let contourLength = 0.25 * distance(direction);
    if (direction === 0 || direction === 4) {
      contourLength = 0.5 * groundY;
    } else if (direction === 2 || direction === 6) {
      contourLength = 0.5 * groundX;
    }
    lower.push({neighbor, direction, score: quinn ? tangent * contourLength : tangent});
  }
  if (!lower.length) {
    return fallback();
  }
  // Normalizing by the largest score keeps large exponents from underflowing.
  const maximumScore = Math.max(...lower.map(edge => edge.score));
  const weights = lower.map(edge => (edge.score / maximumScore) ** exponent);
  const total = weights.reduce((sum, weight) => sum + weight, 0);
  return lower.map((edge, index) => ({
    neighbor: edge.neighbor,
    direction: edge.direction,
    fraction: weights[index] / total
  }));
}

/**
 * Float64 CPU reference of D-infinity / MFD accumulation: topological (Kahn) accumulation over
 * the multiple-receiver DAG with the float32 surface and weights as inputs. Invalid cells and
 * cells that cannot be ordered are NaN.
 */
export function accumulateRoutingOnCPU(input: RoutingOracleInput): Float64Array {
  const cellCount = input.width * input.height;
  const edges: RoutingEdge[][] = [];
  const pendingDonors = new Uint32Array(cellCount);
  for (let cell = 0; cell < cellCount; cell++) {
    edges.push(input.isValid(cell) ? getRoutingEdgesOnCPU(input, cell) : []);
    for (const edge of edges[cell]) {
      pendingDonors[edge.neighbor]++;
    }
  }
  const accumulation = new Float64Array(cellCount).fill(NaN);
  const received = new Float64Array(cellCount);
  const queue: number[] = [];
  for (let cell = 0; cell < cellCount; cell++) {
    if (input.isValid(cell) && pendingDonors[cell] === 0) {
      queue.push(cell);
    }
  }
  for (let head = 0; head < queue.length; head++) {
    const cell = queue[head];
    accumulation[cell] = input.weights[cell] + received[cell];
    for (const edge of edges[cell]) {
      received[edge.neighbor] += edge.fraction * accumulation[cell];
      if (--pendingDonors[edge.neighbor] === 0) {
        queue.push(edge.neighbor);
      }
    }
  }
  return accumulation;
}

/**
 * Sum of accumulation over terminal cells (valid cells without outgoing edges). Equals the sum
 * of the weights of all ordered cells when mass is conserved.
 */
export function sumTerminalAccumulationOnCPU(
  input: RoutingOracleInput,
  accumulation: ArrayLike<number>
): number {
  let total = 0;
  for (let cell = 0; cell < input.width * input.height; cell++) {
    if (input.isValid(cell) && getRoutingEdgesOnCPU(input, cell).length === 0) {
      total += accumulation[cell];
    }
  }
  return total;
}
