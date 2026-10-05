// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPU_RASTER_D8_DIRECTIONS,
  getRasterD8Distance,
  getRasterGroundCellSize
} from '../../../src/map-graphs/cost-distance/raster-grid-utils';
import {
  GPU_TERRAIN_FLOW_CELL_CLASS as CELL_CLASS,
  GPU_TERRAIN_FLOW_NONE,
  getGPUTerrainFlowParameterValues,
  type GPUTerrainFlowSettings
} from '../../../src/map-graphs/hydrology';
import type {GPUTerrainFlowRouting} from '../../../src/map-graphs/hydrology/gpu-terrain-flow';
import {resolveFlatsOnCPU} from './terrain-flow-flats-oracle';
import {accumulateRoutingOnCPU} from './terrain-flow-routing-oracle';
import type {GPUTerrainCellSizeMode} from '../../../src/map-graphs/terrain-analysis';

const fround = Math.fround;

/** Inputs of {@link computeTerrainFlow}. NaN elevation marks an invalid cell. */
export type TerrainFlowOracleInput = {
  elevation: Float32Array;
  width: number;
  height: number;
  settings: GPUTerrainFlowSettings;
  cellSizeMode?: GPUTerrainCellSizeMode;
  fillDepressions?: boolean;
  runoff?: Float32Array;
  area?: boolean;
  /** Mirror of `GPUTerrainFlowProps.resolveFlats`. */
  resolveFlats?: boolean;
  /** Mirror of `GPUTerrainFlowProps.flowRouting`. Defaults to `'d8'`. */
  flowRouting?: GPUTerrainFlowRouting;
};

/** Outputs of {@link computeTerrainFlow}. */
export type TerrainFlowOracleResult = {
  filled: Float32Array;
  directions: Uint32Array;
  classes: Uint32Array;
  receivers: Uint32Array;
  accumulation: Float32Array;
  streams: Uint32Array;
};

/** CPU reference: Planchon-Darboux fill, steepest D8 descent, accumulation, stream mask. */
export function computeTerrainFlow(input: TerrainFlowOracleInput): TerrainFlowOracleResult {
  const {elevation, width, height} = input;
  const cellSizeMode = input.cellSizeMode ?? 'uniform';
  const cellCount = width * height;
  const packed = getGPUTerrainFlowParameterValues(input.settings);
  // Evaluate with the float32 values the GPU reads.
  const gridSettings = {
    cellSize: [packed[0], packed[1]] as [number, number],
    northEdge: packed[2],
    southEdge: packed[3]
  };
  let epsilon = packed[4];
  if (!(epsilon >= 0) || !Number.isFinite(epsilon)) {
    epsilon = 0;
  }
  const streamThreshold = packed[5];
  const isValid = (cell: number) => Number.isFinite(elevation[cell]);
  const getNeighbor = (cell: number, direction: number) => {
    const column = (cell % width) + GPU_RASTER_D8_DIRECTIONS[direction].columnOffset;
    const row = Math.floor(cell / width) + GPU_RASTER_D8_DIRECTIONS[direction].rowOffset;
    return column < 0 || row < 0 || column >= width || row >= height
      ? GPU_TERRAIN_FLOW_NONE
      : row * width + column;
  };
  const isBoundary = (cell: number) => {
    for (let direction = 0; direction < 8; direction++) {
      const neighbor = getNeighbor(cell, direction);
      if (neighbor === GPU_TERRAIN_FLOW_NONE || !isValid(neighbor)) {
        return true;
      }
    }
    return false;
  };

  // Planchon-Darboux: greatest fixpoint below the initial surface (order independent).
  const filled = new Float32Array(cellCount);
  for (let cell = 0; cell < cellCount; cell++) {
    filled[cell] = !isValid(cell) ? NaN : isBoundary(cell) ? elevation[cell] : Infinity;
  }
  if (input.fillDepressions) {
    let changed = true;
    while (changed) {
      changed = false;
      for (let cell = 0; cell < cellCount; cell++) {
        if (!isValid(cell)) {
          continue;
        }
        for (let direction = 0; direction < 8; direction++) {
          const neighbor = getNeighbor(cell, direction);
          if (neighbor === GPU_TERRAIN_FLOW_NONE || !isValid(neighbor)) {
            continue;
          }
          const candidate = Math.max(elevation[cell], fround(filled[neighbor] + epsilon));
          if (candidate < filled[cell]) {
            filled[cell] = candidate;
            changed = true;
          }
        }
      }
    }
  }
  const surface = input.fillDepressions ? filled : elevation;

  const directions = new Uint32Array(cellCount);
  const classes = new Uint32Array(cellCount);
  const receivers = new Uint32Array(cellCount).fill(GPU_TERRAIN_FLOW_NONE);
  for (let cell = 0; cell < cellCount; cell++) {
    if (!isValid(cell)) {
      directions[cell] = GPU_TERRAIN_FLOW_NONE;
      classes[cell] = CELL_CLASS.invalid;
      continue;
    }
    const row = Math.floor(cell / width);
    let bestDrop = 0;
    let bestDirection = -1;
    let hasEqualNeighbor = false;
    let boundary = false;
    for (let direction = 0; direction < 8; direction++) {
      const neighbor = getNeighbor(cell, direction);
      if (neighbor === GPU_TERRAIN_FLOW_NONE || !isValid(neighbor)) {
        boundary = true;
        continue;
      }
      if (surface[neighbor] === surface[cell]) {
        hasEqualNeighbor = true;
      }
      const distance = fround(
        getRasterD8Distance(cellSizeMode, direction, row, height, gridSettings)
      );
      const drop = fround(fround(surface[cell] - surface[neighbor]) / distance);
      if (drop > bestDrop) {
        bestDrop = drop;
        bestDirection = direction;
      }
    }
    if (bestDirection >= 0) {
      directions[cell] = GPU_RASTER_D8_DIRECTIONS[bestDirection].code;
      classes[cell] = CELL_CLASS.draining;
      receivers[cell] = getNeighbor(cell, bestDirection);
    } else {
      classes[cell] = boundary
        ? CELL_CLASS.outlet
        : hasEqualNeighbor
          ? CELL_CLASS.flat
          : CELL_CLASS.pit;
    }
  }

  if (input.resolveFlats) {
    resolveFlatsOnCPU({width, height, surface, isValid, directions, classes, receivers});
  }

  // Accumulation in topological order (Kahn), donors summed in direction order.
  const weights = new Float32Array(cellCount);
  for (let cell = 0; cell < cellCount; cell++) {
    let weight = 1;
    if (input.runoff) {
      const value = input.runoff[cell];
      weight = Number.isFinite(value) && value >= 0 ? value : 0;
    }
    if (input.area) {
      const [groundX, groundY] = getRasterGroundCellSize(
        cellSizeMode,
        Math.floor(cell / width) + 0.5,
        height,
        gridSettings
      );
      weight = fround(weight * fround(groundX * groundY));
    }
    weights[cell] = weight;
  }
  const pendingDonors = new Uint32Array(cellCount);
  for (let cell = 0; cell < cellCount; cell++) {
    if (receivers[cell] !== GPU_TERRAIN_FLOW_NONE) {
      pendingDonors[receivers[cell]]++;
    }
  }
  const accumulation = new Float32Array(cellCount).fill(NaN);
  const queue: number[] = [];
  for (let cell = 0; cell < cellCount; cell++) {
    if (isValid(cell) && pendingDonors[cell] === 0) {
      queue.push(cell);
    }
  }
  for (let head = 0; head < queue.length; head++) {
    const cell = queue[head];
    let sum = weights[cell];
    for (let direction = 0; direction < 8; direction++) {
      const neighbor = getNeighbor(cell, direction);
      if (neighbor !== GPU_TERRAIN_FLOW_NONE && receivers[neighbor] === cell) {
        sum = fround(sum + accumulation[neighbor]);
      }
    }
    accumulation[cell] = sum;
    const receiver = receivers[cell];
    if (receiver !== GPU_TERRAIN_FLOW_NONE && --pendingDonors[receiver] === 0) {
      queue.push(receiver);
    }
  }
  const flowRouting = input.flowRouting ?? 'd8';
  if (flowRouting !== 'd8') {
    // Non-D8 routings use a float64 oracle; GPU results are compared with a tolerance.
    const routed = accumulateRoutingOnCPU({
      routing: flowRouting,
      width,
      height,
      cellSizeMode,
      gridSettings,
      flowExponent: packed[6],
      surface,
      isValid,
      receivers,
      weights
    });
    for (let cell = 0; cell < cellCount; cell++) {
      accumulation[cell] = routed[cell];
    }
  }
  const streams = new Uint32Array(cellCount);
  for (let cell = 0; cell < cellCount; cell++) {
    streams[cell] =
      Number.isFinite(accumulation[cell]) && accumulation[cell] >= streamThreshold ? 1 : 0;
  }
  return {filled, directions, classes, receivers, accumulation, streams};
}

/** Seeded linear congruential generator returning floats in `[0, 1)`. */
export function createSeededRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

/** Named synthetic DEM generators with integer or quarter-step elevations. */
export const TERRAIN_FLOW_DEMS = {
  cone: (width: number, height: number) =>
    Float32Array.from({length: width * height}, (_, cell) => {
      const dx = (cell % width) - (width - 1) / 2;
      const dy = Math.floor(cell / width) - (height - 1) / 2;
      return Math.round(100 - 2 * Math.hypot(dx, dy));
    }),
  invertedCone: (width: number, height: number) =>
    Float32Array.from({length: width * height}, (_, cell) => {
      const dx = (cell % width) - (width - 1) / 2;
      const dy = Math.floor(cell / width) - (height - 1) / 2;
      return Math.round(2 * Math.hypot(dx, dy));
    }),
  valley: (width: number, height: number) =>
    Float32Array.from({length: width * height}, (_, cell) => {
      const column = cell % width;
      const row = Math.floor(cell / width);
      return 3 * Math.abs(column - Math.floor(width / 2)) + (height - 1 - row);
    }),
  tiltedPlane: (width: number, height: number) =>
    Float32Array.from({length: width * height}, (_, cell) => {
      const column = cell % width;
      const row = Math.floor(cell / width);
      return 0.25 * (2 * column + 3 * row);
    }),
  noise: (width: number, height: number, seed = 7) => {
    const random = createSeededRandom(seed);
    return Float32Array.from({length: width * height}, () => Math.floor(random() * 20));
  },
  plateau: (width: number, height: number) =>
    Float32Array.from({length: width * height}, (_, cell) => {
      const column = cell % width;
      const row = Math.floor(cell / width);
      return column < width / 3 || row < height / 3 ? 10 : 5;
    }),
  noDataHoles: (width: number, height: number, seed = 11) => {
    const random = createSeededRandom(seed);
    return Float32Array.from({length: width * height}, () =>
      random() < 0.12 ? NaN : Math.floor(random() * 12)
    );
  }
} as const;
