// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Shared typings for earcut 3 (the package ships none and `@types/earcut` is not installed).
// Scenes no longer need their own `declare module 'earcut'` file.
declare module 'earcut' {
  /** Triangulates a flat polygon (`dimensions` numbers per vertex) with optional hole start indexes. */
  export default function earcut(
    data: ArrayLike<number>,
    holeIndices?: ArrayLike<number> | null,
    dimensions?: number
  ): number[];
}
