// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// CPU reference for GPUCellCompaction. Quadbin follows CARTO quadbin with BigInt; H3 uses h3-js.

import {cellToChildren, isPentagon} from 'h3-js';
import {
  bigIntToH3,
  getCellParent,
  getCellResolution,
  h3ToBigInt,
  isValidCellKey,
  quadbinCellToTile,
  quadbinTileToCell,
  type CellFamily
} from '../cell-aggregation/cell-aggregation-oracle';

/** Ascending BigInt comparator. */
export function compareBigInt(left: bigint, right: bigint): number {
  return left < right ? -1 : left > right ? 1 : 0;
}

/** Descendants of `cell` at `resolution`, in ascending key order. */
export function getCellDescendants(family: CellFamily, cell: bigint, resolution: number): bigint[] {
  let result: bigint[];
  if (family === 'quadbin') {
    const {x, y, z} = quadbinCellToTile(cell);
    const side = 2 ** (resolution - z);
    result = [];
    for (let dx = 0; dx < side; dx++) {
      for (let dy = 0; dy < side; dy++) {
        result.push(quadbinTileToCell(x * side + dx, y * side + dy, resolution));
      }
    }
  } else {
    result = cellToChildren(bigIntToH3(cell), resolution).map(h3ToBigInt);
  }
  return result.sort(compareBigInt);
}

/** Number of children of a cell: 4 (Quadbin), 7 (H3 hexagon) or 6 (H3 pentagon). */
export function getChildCount(family: CellFamily, cell: bigint): number {
  if (family === 'quadbin') {
    return 4;
  }
  return isPentagon(bigIntToH3(cell)) ? 6 : 7;
}

/**
 * Minimal cover: valid cells at `resolution` are deduplicated, then every complete child set is
 * replaced by its parent down to `minimumResolution`. Returns ascending keys.
 */
export function compactCellsOnCPU(
  family: CellFamily,
  cells: readonly bigint[],
  resolution: number,
  minimumResolution = 0
): bigint[] {
  const result: bigint[] = [];
  let level = [
    ...new Set(
      cells.filter(
        cell => isValidCellKey(family, cell) && getCellResolution(family, cell) === resolution
      )
    )
  ];
  for (let current = resolution; current > minimumResolution; current--) {
    const groups = new Map<bigint, bigint[]>();
    for (const cell of level) {
      const parent = getCellParent(family, cell, current - 1);
      groups.set(parent, [...(groups.get(parent) ?? []), cell]);
    }
    const next: bigint[] = [];
    for (const [parent, children] of groups) {
      if (children.length === getChildCount(family, parent)) {
        next.push(parent);
      } else {
        result.push(...children);
      }
    }
    level = next;
  }
  result.push(...level);
  return result.sort(compareBigInt);
}

/** Expands each valid cell at or above `resolution` (within `maximumDepth`), in input order. */
export function uncompactCellsOnCPU(
  family: CellFamily,
  cells: readonly bigint[],
  resolution: number,
  maximumDepth = 8
): bigint[] {
  const result: bigint[] = [];
  for (const cell of cells) {
    if (!isValidCellKey(family, cell)) {
      continue;
    }
    const cellResolution = getCellResolution(family, cell);
    if (cellResolution <= resolution && resolution - cellResolution <= maximumDepth) {
      result.push(...getCellDescendants(family, cell, resolution));
    }
  }
  return result;
}
