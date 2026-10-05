// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPU_RASTER_D8_DIRECTIONS} from '../../../src/gpu-raster/cost-distance/raster-grid-utils';

/** Receiver and label value meaning "none". */
export const DRAINAGE_ORACLE_NONE = 0xffffffff;

/** Receiver of `cell` for a D8 code: a single bit in 1..128 whose neighbor is inside the grid. */
export function getD8ReceiverOnCPU(
  cell: number,
  code: number,
  width: number,
  height: number
): number {
  const direction = GPU_RASTER_D8_DIRECTIONS.findIndex(entry => entry.code === code);
  if (direction < 0) {
    return DRAINAGE_ORACLE_NONE;
  }
  const column = (cell % width) + GPU_RASTER_D8_DIRECTIONS[direction].columnOffset;
  const row = Math.floor(cell / width) + GPU_RASTER_D8_DIRECTIONS[direction].rowOffset;
  return column < 0 || row < 0 || column >= width || row >= height
    ? DRAINAGE_ORACLE_NONE
    : row * width + column;
}

/** Follows receivers from every cell until a stop cell or a cell without receiver. */
export function followReceiversOnCPU(
  directions: Uint32Array,
  width: number,
  height: number,
  isStop: (cell: number) => boolean
): Uint32Array {
  const cellCount = width * height;
  const fixpoint = new Uint32Array(cellCount);
  for (let start = 0; start < cellCount; start++) {
    let cell = start;
    for (let step = 0; step <= cellCount; step++) {
      if (isStop(cell)) {
        break;
      }
      const receiver = getD8ReceiverOnCPU(cell, directions[cell], width, height);
      if (receiver === DRAINAGE_ORACLE_NONE) {
        break;
      }
      cell = receiver;
    }
    fixpoint[start] = cell;
  }
  return fixpoint;
}

const isInvalidCode = (code: number) => code === DRAINAGE_ORACLE_NONE;

/** Sequential HAND: drainage cells (or NONE) and heights (NaN when undefined). */
export function computeHeightAboveDrainageOnCPU(
  elevation: Float32Array,
  directions: Uint32Array,
  streams: Uint32Array,
  width: number,
  height: number
): {drainageCells: Uint32Array; heightAboveDrainage: Float32Array} {
  const isStream = (cell: number) => streams[cell] !== 0 && !isInvalidCode(directions[cell]);
  const fixpoint = followReceiversOnCPU(directions, width, height, isStream);
  const drainageCells = new Uint32Array(directions.length);
  const heightAboveDrainage = new Float32Array(directions.length);
  for (let cell = 0; cell < directions.length; cell++) {
    const reached = fixpoint[cell];
    const drains = !isInvalidCode(directions[cell]) && isStream(reached);
    drainageCells[cell] = drains ? reached : DRAINAGE_ORACLE_NONE;
    heightAboveDrainage[cell] =
      drains && Number.isFinite(elevation[cell]) && Number.isFinite(elevation[reached])
        ? Math.fround(elevation[cell] - elevation[reached])
        : NaN;
  }
  return {drainageCells, heightAboveDrainage};
}

/** Sequential watershed labels; basins when `pourPoints` is undefined. */
export function computeWatershedsOnCPU(
  directions: Uint32Array,
  width: number,
  height: number,
  pourPoints?: Uint32Array
): Uint32Array {
  const cellCount = width * height;
  const labels = new Uint32Array(cellCount).fill(DRAINAGE_ORACLE_NONE);
  if (!pourPoints) {
    const fixpoint = followReceiversOnCPU(directions, width, height, () => false);
    for (let cell = 0; cell < cellCount; cell++) {
      if (!isInvalidCode(directions[cell]) && !isInvalidCode(directions[fixpoint[cell]])) {
        labels[cell] = fixpoint[cell];
      }
    }
    return labels;
  }
  const markers = new Uint32Array(cellCount).fill(DRAINAGE_ORACLE_NONE);
  for (const [pointIndex, cell] of pourPoints.entries()) {
    if (cell < cellCount && !isInvalidCode(directions[cell]) && pointIndex < markers[cell]) {
      markers[cell] = pointIndex;
    }
  }
  const fixpoint = followReceiversOnCPU(
    directions,
    width,
    height,
    cell => markers[cell] !== DRAINAGE_ORACLE_NONE
  );
  for (let cell = 0; cell < cellCount; cell++) {
    if (!isInvalidCode(directions[cell])) {
      labels[cell] = markers[fixpoint[cell]];
    }
  }
  return labels;
}

/** Sequential Strahler order in Kahn (topological) order. */
export function computeStreamOrderOnCPU(
  directions: Uint32Array,
  streams: Uint32Array,
  width: number,
  height: number
): Uint32Array {
  const cellCount = width * height;
  const isStream = (cell: number) => streams[cell] !== 0 && !isInvalidCode(directions[cell]);
  const receivers = new Uint32Array(cellCount).fill(DRAINAGE_ORACLE_NONE);
  const pending = new Uint32Array(cellCount);
  const donorOrders: number[][] = Array.from({length: cellCount}, () => []);
  for (let cell = 0; cell < cellCount; cell++) {
    if (isStream(cell)) {
      const receiver = getD8ReceiverOnCPU(cell, directions[cell], width, height);
      if (receiver !== DRAINAGE_ORACLE_NONE && isStream(receiver)) {
        receivers[cell] = receiver;
        pending[receiver]++;
      }
    }
  }
  const order = new Uint32Array(cellCount);
  const queue: number[] = [];
  for (let cell = 0; cell < cellCount; cell++) {
    if (isStream(cell) && pending[cell] === 0) {
      queue.push(cell);
    }
  }
  for (let head = 0; head < queue.length; head++) {
    const cell = queue[head];
    const orders = donorOrders[cell];
    const maximum = orders.length ? Math.max(...orders) : 0;
    const count = orders.filter(value => value === maximum).length;
    order[cell] = orders.length === 0 ? 1 : count >= 2 ? maximum + 1 : maximum;
    const receiver = receivers[cell];
    if (receiver !== DRAINAGE_ORACLE_NONE) {
      donorOrders[receiver].push(order[cell]);
      if (--pending[receiver] === 0) {
        queue.push(receiver);
      }
    }
  }
  return order;
}
