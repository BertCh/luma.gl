// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {cellToChildren, cellToParent, getResolution, gridDiskDistances, isValidCell} from 'h3-js';
import type {GPUCellTopologyOperation} from '../../../src/map-graphs/cell-topology/gpu-cell-topology';
import {
  bigIntToH3,
  h3ToBigInt,
  quadbinCellToParent,
  quadbinCellToTile,
  quadbinGetResolution,
  quadbinIsValidCell,
  quadbinTileToCell
} from '../cell-aggregation/cell-aggregation-oracle';

/** One input row's expected output: entries in output order, without padding. */
export type CellTopologyRow = {cells: bigint[]; distances: number[]};

const EMPTY_ROW: CellTopologyRow = {cells: [], distances: []};

function sortByDistanceThenKey(entries: [bigint, number][]): CellTopologyRow {
  entries.sort((a, b) => a[1] - b[1] || (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0));
  return {cells: entries.map(entry => entry[0]), distances: entries.map(entry => entry[1])};
}

/** Quadbin disk or ring by Chebyshev distance, x wrapping and y clipped, one entry per tile. */
function getQuadbinNeighborhood(cell: bigint, k: number, ringOnly: boolean): CellTopologyRow {
  const {x, y, z} = quadbinCellToTile(cell);
  const size = 2 ** z;
  // Smallest circular x distance of every column in the window.
  const columns = new Map<number, number>();
  for (let dx = -k; dx <= k; dx++) {
    const column = (((x + dx) % size) + size) % size;
    const circular = Math.min(Math.abs(column - x), size - Math.abs(column - x));
    columns.set(column, circular);
  }
  const entries: [bigint, number][] = [];
  for (const [column, columnDistance] of columns) {
    for (let dy = -k; dy <= k; dy++) {
      const row = y + dy;
      if (row < 0 || row >= size) {
        continue;
      }
      const distance = Math.max(columnDistance, Math.abs(dy));
      if (!ringOnly || distance === k) {
        entries.push([quadbinTileToCell(column, row, z), distance]);
      }
    }
  }
  return sortByDistanceThenKey(entries);
}

/** H3 disk or ring from h3-js `gridDiskDistances`. */
function getH3Neighborhood(cell: bigint, k: number, ringOnly: boolean): CellTopologyRow {
  const entries: [bigint, number][] = [];
  gridDiskDistances(bigIntToH3(cell), k).forEach((layer, distance) => {
    if (!ringOnly || distance === k) {
      for (const neighbor of layer) {
        entries.push([h3ToBigInt(neighbor), distance]);
      }
    }
  });
  return sortByDistanceThenKey(entries);
}

/**
 * CPU reference of {@link GPUCellTopology} for one input cell: the unpadded output entries.
 * Invalid cells give an empty row.
 */
export function getCellTopologyRowOnCPU(
  family: 'quadbin' | 'h3',
  operation: GPUCellTopologyOperation,
  cell: bigint
): CellTopologyRow {
  const isH3 = family === 'h3';
  const isValid = isH3
    ? cell > 0n && cell < 2n ** 64n && isValidCell(bigIntToH3(cell))
    : quadbinIsValidCell(cell);
  if (!isValid) {
    return EMPTY_ROW;
  }
  const resolution = isH3 ? getResolution(bigIntToH3(cell)) : quadbinGetResolution(cell);
  switch (operation.type) {
    case 'disk':
    case 'ring':
      return isH3
        ? getH3Neighborhood(cell, operation.k, operation.type === 'ring')
        : getQuadbinNeighborhood(cell, operation.k, operation.type === 'ring');
    case 'parent': {
      if (resolution < operation.resolution) {
        return EMPTY_ROW;
      }
      const parent = isH3
        ? h3ToBigInt(cellToParent(bigIntToH3(cell), operation.resolution))
        : quadbinCellToParent(cell, operation.resolution);
      return {cells: [parent], distances: [0]};
    }
    case 'children': {
      if (resolution !== operation.inputResolution) {
        return EMPTY_ROW;
      }
      let children: bigint[];
      if (isH3) {
        children = cellToChildren(bigIntToH3(cell), operation.resolution).map(h3ToBigInt);
      } else {
        const {x, y} = quadbinCellToTile(cell);
        const depth = operation.resolution - resolution;
        children = [];
        for (let dx = 0; dx < 2 ** depth; dx++) {
          for (let dy = 0; dy < 2 ** depth; dy++) {
            children.push(
              quadbinTileToCell(x * 2 ** depth + dx, y * 2 ** depth + dy, operation.resolution)
            );
          }
        }
      }
      children.sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));
      return {cells: children, distances: children.map(() => 0)};
    }
  }
}
