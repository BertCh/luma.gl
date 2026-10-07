// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

declare module 'earcut' {
  /** Triangulates a flat polygon (`dimensions` numbers per vertex) with optional hole start indexes. */
  export default function earcut(
    data: ArrayLike<number>,
    holeIndices?: ArrayLike<number> | null,
    dimensions?: number
  ): number[];
}
