// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPU_RASTER_D8_DIRECTIONS} from '../../../src/map-graphs/cost-distance/raster-grid-utils';
import {GPU_TERRAIN_FLOW_CELL_CLASS as CELL_CLASS} from '../../../src/map-graphs/hydrology';

/** Inputs of {@link resolveFlatsOnCPU}. `directions`, `classes` and `receivers` are rewritten in place. */
export type FlatResolutionOracleInput = {
  width: number;
  height: number;
  /** Routing surface, NaN for invalid cells. */
  surface: Float32Array;
  isValid: (cell: number) => boolean;
  directions: Uint32Array;
  classes: Uint32Array;
  receivers: Uint32Array;
};

/**
 * CPU reference of Barnes, Lehman and Mulla (2014) flat resolution with sequential BFS queues.
 *
 * Hop counts are integers. Ties between equal masks keep the lowest luma D8 direction index
 * (E, SE, S, SW, W, NW, N, NE), which differs from RichDEM's order.
 */
export function resolveFlatsOnCPU(input: FlatResolutionOracleInput): void {
  const {width, height, surface, isValid, directions, classes, receivers} = input;
  const cellCount = width * height;
  const getNeighbor = (cell: number, direction: number) => {
    const column = (cell % width) + GPU_RASTER_D8_DIRECTIONS[direction].columnOffset;
    const row = Math.floor(cell / width) + GPU_RASTER_D8_DIRECTIONS[direction].rowOffset;
    return column < 0 || row < 0 || column >= width || row >= height ? -1 : row * width + column;
  };
  const isFlat = (cell: number) => isValid(cell) && classes[cell] === CELL_CLASS.flat;
  const hasFlow = (cell: number) =>
    isValid(cell) && (classes[cell] === CELL_CLASS.draining || classes[cell] === CELL_CLASS.outlet);

  const towardLower = new Float64Array(cellCount).fill(Infinity);
  const awayFromHigher = new Float64Array(cellCount);
  const lowEdges: number[] = [];
  const highEdges: number[] = [];
  for (let cell = 0; cell < cellCount; cell++) {
    for (let direction = 0; direction < 8; direction++) {
      const neighbor = getNeighbor(cell, direction);
      if (neighbor < 0 || !isValid(neighbor)) {
        continue;
      }
      if (hasFlow(cell) && isFlat(neighbor) && surface[neighbor] === surface[cell]) {
        if (towardLower[cell] !== 1) {
          towardLower[cell] = 1;
          lowEdges.push(cell);
        }
      }
      if (isFlat(cell) && surface[neighbor] > surface[cell] && awayFromHigher[cell] !== 1) {
        awayFromHigher[cell] = 1;
        highEdges.push(cell);
      }
    }
  }
  // BFS over equal-surface flat cells. Low edges only seed: they are never flat, so never entered.
  const breadthFirst = (queue: number[], distances: Float64Array, unreached: number) => {
    for (let head = 0; head < queue.length; head++) {
      const cell = queue[head];
      for (let direction = 0; direction < 8; direction++) {
        const neighbor = getNeighbor(cell, direction);
        if (
          neighbor >= 0 &&
          isFlat(neighbor) &&
          surface[neighbor] === surface[cell] &&
          distances[neighbor] === unreached
        ) {
          distances[neighbor] = distances[cell] + 1;
          queue.push(neighbor);
        }
      }
    }
  };
  breadthFirst(lowEdges, towardLower, Infinity);
  breadthFirst(highEdges, awayFromHigher, 0);

  // Label equal-surface components of all valid cells and take the maximum awayFromHigher.
  const componentMaximum = new Float64Array(cellCount);
  const visited = new Uint8Array(cellCount);
  for (let seed = 0; seed < cellCount; seed++) {
    if (!isValid(seed) || visited[seed]) {
      continue;
    }
    const component = [seed];
    visited[seed] = 1;
    let maximum = 0;
    for (let head = 0; head < component.length; head++) {
      const cell = component[head];
      maximum = Math.max(maximum, awayFromHigher[cell]);
      for (let direction = 0; direction < 8; direction++) {
        const neighbor = getNeighbor(cell, direction);
        if (
          neighbor >= 0 &&
          isValid(neighbor) &&
          !visited[neighbor] &&
          surface[neighbor] === surface[seed]
        ) {
          visited[neighbor] = 1;
          component.push(neighbor);
        }
      }
    }
    for (const cell of component) {
      componentMaximum[cell] = maximum;
    }
  }

  const mask = new Float64Array(cellCount).fill(Infinity);
  for (let cell = 0; cell < cellCount; cell++) {
    if (Number.isFinite(towardLower[cell])) {
      mask[cell] =
        2 * towardLower[cell] +
        (awayFromHigher[cell] > 0 ? componentMaximum[cell] - awayFromHigher[cell] : 0);
    }
  }
  for (let cell = 0; cell < cellCount; cell++) {
    if (!isFlat(cell) || !Number.isFinite(mask[cell])) {
      continue;
    }
    let bestMask = mask[cell];
    let bestDirection = -1;
    let bestNeighbor = -1;
    for (let direction = 0; direction < 8; direction++) {
      const neighbor = getNeighbor(cell, direction);
      if (
        neighbor >= 0 &&
        isValid(neighbor) &&
        surface[neighbor] === surface[cell] &&
        mask[neighbor] < bestMask
      ) {
        bestMask = mask[neighbor];
        bestDirection = direction;
        bestNeighbor = neighbor;
      }
    }
    if (bestDirection >= 0) {
      classes[cell] = CELL_CLASS.draining;
      directions[cell] = GPU_RASTER_D8_DIRECTIONS[bestDirection].code;
      receivers[cell] = bestNeighbor;
    }
  }
}
