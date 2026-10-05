// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Inputs of {@link selectResidentRowsOracle}. */
export type ResidentRowSelectionOracleInput = {
  liveMask: ArrayLike<number>;
  rowTileSlots?: ArrayLike<number>;
  tileMask?: ArrayLike<number>;
  predicateMasks?: readonly ArrayLike<number>[];
  sourceIds?: ArrayLike<number>;
  capacity: number;
};

/** Result of {@link selectResidentRowsOracle}. */
export type ResidentRowSelectionOracleResult = {
  /** Canonical 0/1 composed mask per row. */
  mask: number[];
  /** Bounded stable IDs, ascending by row. */
  ids: number[];
  /** `min(total, capacity)`. */
  count: number;
  /** Unclamped number of selected rows. */
  total: number;
  /** 1 when `total > capacity`. */
  overflow: number;
};

/** CPU reference for the resident-row selection: live AND tile gate AND predicates, then bounded stable compaction. */
export function selectResidentRowsOracle(
  input: ResidentRowSelectionOracleInput
): ResidentRowSelectionOracleResult {
  const {liveMask, rowTileSlots, tileMask, predicateMasks = [], sourceIds, capacity} = input;
  const mask: number[] = [];
  const accepted: number[] = [];
  for (let row = 0; row < liveMask.length; row++) {
    let pass = liveMask[row] !== 0;
    if (pass && rowTileSlots && tileMask) {
      const slot = rowTileSlots[row];
      pass = slot < tileMask.length && tileMask[slot] !== 0;
    }
    for (const predicateMask of predicateMasks) {
      pass = pass && predicateMask[row] !== 0;
    }
    mask.push(pass ? 1 : 0);
    if (pass) {
      accepted.push(sourceIds ? sourceIds[row] : row);
    }
  }
  const count = Math.min(accepted.length, capacity);
  return {
    mask,
    ids: accepted.slice(0, count),
    count,
    total: accepted.length,
    overflow: accepted.length > capacity ? 1 : 0
  };
}
