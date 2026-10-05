// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Feature bounds as `[minX, minY, maxX, maxY]`; empty features have inverted bounds. */
export type FeatureBounds = [number, number, number, number];

const EMPTY_BOUNDS: FeatureBounds = [Infinity, Infinity, -Infinity, -Infinity];

/** Bounds of an invalid feature, matching the inverted bounds written by the GPU bounds kernel. */
export function getEmptyBounds(): FeatureBounds {
  return [...EMPTY_BOUNDS];
}

/** Returns whether bounds describe a nonempty feature. */
export function isValidBounds(bounds: FeatureBounds): boolean {
  return bounds[0] <= bounds[2] && bounds[1] <= bounds[3];
}

/** Complete binary tree over leaf slots, laid out like `GPUBVH` (root 0, children `2n+1`, `2n+2`). */
export type ModelBVH = {
  leafCapacity: number;
  internalNodeCount: number;
  /** `[minX, minY, maxX, maxY]` per node. */
  nodeBounds: FeatureBounds[];
  /** Feature row per leaf slot, or -1 for empty slots. */
  leafRows: number[];
};

/** Builds the tree for features stored in `leafOrder` (feature rows in leaf-slot order). */
export function buildModelBVH(
  featureBounds: readonly FeatureBounds[],
  leafOrder: readonly number[],
  leafCapacity: number
): ModelBVH {
  const internalNodeCount = leafCapacity - 1;
  const nodeBounds: FeatureBounds[] = Array.from({length: 2 * leafCapacity - 1}, getEmptyBounds);
  const leafRows: number[] = new Array(leafCapacity).fill(-1);
  for (let slot = 0; slot < Math.min(leafOrder.length, leafCapacity); slot++) {
    leafRows[slot] = leafOrder[slot];
    nodeBounds[internalNodeCount + slot] = [...featureBounds[leafOrder[slot]]];
  }
  for (let node = internalNodeCount - 1; node >= 0; node--) {
    const left = nodeBounds[2 * node + 1];
    const right = nodeBounds[2 * node + 2];
    nodeBounds[node] = [
      Math.min(left[0], right[0]),
      Math.min(left[1], right[1]),
      Math.max(left[2], right[2]),
      Math.max(left[3], right[3])
    ];
  }
  return {leafCapacity, internalNodeCount, nodeBounds, leafRows};
}

/**
 * Simulates the stackless traversal of `getSpatialJoinProbeNodes` for one query box.
 *
 * Copies the probe's loop: test the node, descend to the left child on overlap, otherwise climb
 * while the node is a right child and step to the right sibling. Counts every node whose bounds
 * were tested (`visited`) and every overlapping leaf that holds a feature (`candidates`).
 */
export function simulateTraversal(
  bvh: ModelBVH,
  queryMinimum: [number, number],
  queryMaximum: [number, number]
): {visited: number; candidates: number} {
  let visited = 0;
  let candidates = 0;
  let node = 0;
  for (;;) {
    visited++;
    const bounds = bvh.nodeBounds[node];
    const overlaps =
      bounds[0] <= queryMaximum[0] &&
      bounds[1] <= queryMaximum[1] &&
      queryMinimum[0] <= bounds[2] &&
      queryMinimum[1] <= bounds[3];
    if (overlaps) {
      if (node < bvh.internalNodeCount) {
        node = node * 2 + 1;
        continue;
      }
      if (bvh.leafRows[node - bvh.internalNodeCount] >= 0) {
        candidates++;
      }
    }
    while (!(node === 0 || (node & 1) === 1)) {
      node = Math.floor((node - 1) / 2);
    }
    if (node === 0) {
      break;
    }
    node = node + 1;
  }
  return {visited, candidates};
}

/** Total visited nodes and candidates over every query point (box half-width `radius`). */
export function measureTraversal(
  bvh: ModelBVH,
  points: readonly [number, number][],
  radius: number
): {visitedPerPoint: number; candidatesPerPoint: number; candidates: number} {
  let visited = 0;
  let candidates = 0;
  for (const [x, y] of points) {
    const result = simulateTraversal(bvh, [x - radius, y - radius], [x + radius, y + radius]);
    visited += result.visited;
    candidates += result.candidates;
  }
  return {
    visitedPerPoint: visited / points.length,
    candidatesPerPoint: candidates / points.length,
    candidates
  };
}

/** Morton key matching the GPU kernel: 16 bits per axis over the valid-center scene bounds. */
export function getMortonKey(
  bounds: FeatureBounds,
  scene: {minimum: [number, number]; maximum: [number, number]}
): number {
  if (!isValidBounds(bounds)) {
    return 0xffffffff;
  }
  const quantize = (value: number, minimum: number, maximum: number) => {
    const extent = maximum - minimum;
    if (!(extent > 0)) {
      return 0;
    }
    const normalized = Math.min(Math.max((value - minimum) / extent, 0), 1);
    return Math.min(Math.floor(normalized * 65535 + 0.5), 65535);
  };
  const spread = (value: number) => {
    let x = value & 0xffff;
    x = (x | (x << 8)) & 0x00ff00ff;
    x = (x | (x << 4)) & 0x0f0f0f0f;
    x = (x | (x << 2)) & 0x33333333;
    x = (x | (x << 1)) & 0x55555555;
    return x;
  };
  const x = quantize((bounds[0] + bounds[2]) / 2, scene.minimum[0], scene.maximum[0]);
  const y = quantize((bounds[1] + bounds[3]) / 2, scene.minimum[1], scene.maximum[1]);
  return (spread(x) | (spread(y) << 1)) >>> 0;
}

/** Feature rows in stable Morton order (invalid features last), as the GPU sort produces. */
export function getMortonLeafOrder(featureBounds: readonly FeatureBounds[]): number[] {
  const minimum: [number, number] = [Infinity, Infinity];
  const maximum: [number, number] = [-Infinity, -Infinity];
  for (const bounds of featureBounds) {
    if (isValidBounds(bounds)) {
      const centerX = (bounds[0] + bounds[2]) / 2;
      const centerY = (bounds[1] + bounds[3]) / 2;
      minimum[0] = Math.min(minimum[0], centerX);
      minimum[1] = Math.min(minimum[1], centerY);
      maximum[0] = Math.max(maximum[0], centerX);
      maximum[1] = Math.max(maximum[1], centerY);
    }
  }
  const keys = featureBounds.map(bounds => getMortonKey(bounds, {minimum, maximum}));
  return featureBounds
    .map((_, row) => row)
    .sort((left, right) => keys[left] - keys[right] || left - right);
}

/** Smallest power of two that is at least `value`. */
export function getLeafCapacity(value: number): number {
  let capacity = 1;
  while (capacity < value) {
    capacity *= 2;
  }
  return capacity;
}
