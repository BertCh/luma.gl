declare module 'earcut' {
  /** Triangulates a polygon with optional holes (flat coordinates, hole start vertex indices). */
  export default function earcut(
    data: ArrayLike<number>,
    holeIndices?: ArrayLike<number> | null,
    dimensions?: number
  ): number[];
}
