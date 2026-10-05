// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Properties for {@link makeGPUTileLODQuadtree}. */
export type GPUTileLODQuadtreeProps = {
  /** `[minX, minY, maxX, maxY]` of the root tile in hierarchy space. */
  bounds: readonly [number, number, number, number];
  /** Deepest level (root = 0). Integer in `[0, 12]`. */
  maximumLevel: number;
  /** Minimum Z of tile content, for sphere radii. Defaults to 0. */
  minimumZ?: number;
  /** Maximum Z of tile content, for sphere radii. Defaults to 0. */
  maximumZ?: number;
  /** Tile resolution in pixels; geometric error is `tileWidth / tileSize`. Defaults to 256. */
  tileSize?: number;
};

/** CPU arrays of a full Morton-ordered quadtree, ready to upload for `GPUTileLODSelection`. */
export type GPUTileLODQuadtree = {
  /** Total node count. */
  nodeCount: number;
  /** Breadth-level offsets: `levelOffsets[z] = (4^z - 1) / 3`. */
  levelOffsets: number[];
  /** Four floats per node: sphere center XYZ and radius. */
  sphereBounds: Float32Array;
  /** Geometric error per node. */
  geometricErrors: Float32Array;
  /** Two uint32 per node: `[firstChild, childCount]`; leaves are `[0, 0]`. */
  children: Uint32Array;
  /** Parent node per node; the root is `0xffffffff`. */
  parents: Uint32Array;
};

/** Level and tile coordinates of one quadtree node. */
export type GPUTileLODQuadtreeTile = {
  /** Level, root = 0. */
  level: number;
  /** Tile column. */
  x: number;
  /** Tile row; rows grow from `minY` toward `maxY`. */
  y: number;
};

/**
 * Builds a full quadtree whose levels are Morton (Z-order) sorted, so the four children of each
 * node form one contiguous range in the next level, as `GPUTileLODSelection` requires.
 */
export function makeGPUTileLODQuadtree(props: GPUTileLODQuadtreeProps): GPUTileLODQuadtree {
  const [minX, minY, maxX, maxY] = props.bounds;
  const {maximumLevel} = props;
  const minimumZ = props.minimumZ ?? 0;
  const maximumZ = props.maximumZ ?? 0;
  const tileSize = props.tileSize ?? 256;
  if (!props.bounds.every(Number.isFinite) || maxX <= minX || maxY <= minY) {
    throw new Error('makeGPUTileLODQuadtree bounds must be finite with max > min');
  }
  if (!Number.isInteger(maximumLevel) || maximumLevel < 0 || maximumLevel > 12) {
    throw new Error('makeGPUTileLODQuadtree maximumLevel must be an integer in [0, 12]');
  }
  if (!(tileSize > 0) || !Number.isFinite(tileSize)) {
    throw new Error('makeGPUTileLODQuadtree tileSize must be positive');
  }
  const levelOffsets = Array.from({length: maximumLevel + 2}, (_, level) => (4 ** level - 1) / 3);
  const nodeCount = levelOffsets[maximumLevel + 1];
  const sphereBounds = new Float32Array(nodeCount * 4);
  const geometricErrors = new Float32Array(nodeCount);
  const children = new Uint32Array(nodeCount * 2);
  const parents = new Uint32Array(nodeCount).fill(0xffffffff);
  for (let level = 0; level <= maximumLevel; level++) {
    const width = (maxX - minX) / 2 ** level;
    const height = (maxY - minY) / 2 ** level;
    const radius = 0.5 * Math.hypot(width, height, maximumZ - minimumZ);
    for (let morton = 0; morton < 4 ** level; morton++) {
      const node = levelOffsets[level] + morton;
      const x = deinterleaveBits(morton);
      const y = deinterleaveBits(morton >>> 1);
      sphereBounds.set(
        [minX + (x + 0.5) * width, minY + (y + 0.5) * height, (minimumZ + maximumZ) / 2, radius],
        node * 4
      );
      geometricErrors[node] = width / tileSize;
      if (level < maximumLevel) {
        const firstChild = levelOffsets[level + 1] + 4 * morton;
        children.set([firstChild, 4], node * 2);
        for (let child = 0; child < 4; child++) {
          parents[firstChild + child] = node;
        }
      }
    }
  }
  return {nodeCount, levelOffsets, sphereBounds, geometricErrors, children, parents};
}

/** Decodes a quadtree node index into its level and tile coordinates. */
export function getGPUTileLODQuadtreeTile(
  nodeIndex: number,
  maximumLevel: number
): GPUTileLODQuadtreeTile {
  for (let level = 0; level <= maximumLevel; level++) {
    const first = (4 ** level - 1) / 3;
    const count = 4 ** level;
    if (nodeIndex >= first && nodeIndex < first + count) {
      const morton = nodeIndex - first;
      return {level, x: deinterleaveBits(morton), y: deinterleaveBits(morton >>> 1)};
    }
  }
  throw new Error('getGPUTileLODQuadtreeTile node index is outside the quadtree');
}

/** Collects the even bits of a 32-bit Morton code into a 16-bit coordinate. */
function deinterleaveBits(value: number): number {
  let result = 0;
  for (let bit = 0; bit < 16; bit++) {
    result |= ((value >>> (2 * bit)) & 1) << bit;
  }
  return result;
}
