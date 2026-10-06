// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Deterministic small PRNG (mulberry32). */
export function createRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * A segmentation of touching patches: each pixel takes the label of its nearest seed, so patch
 * sizes vary. Seed `s` has label `s + 1`. A rectangle in one corner is background (label 0).
 */
export function createSegmentation(
  width: number,
  height: number,
  seedCount: number,
  seed: number
): Uint32Array {
  const random = createRandom(seed);
  const seeds = Array.from({length: seedCount}, () => [
    random() * width,
    random() * height,
    0.4 + random() * 1.6
  ]);
  const labels = new Uint32Array(width * height);
  for (let row = 0; row < height; row++) {
    for (let column = 0; column < width; column++) {
      let best = 0;
      let bestDistance = Infinity;
      seeds.forEach(([x, y, weight], index) => {
        const distance = ((column - x) ** 2 + (row - y) ** 2) / weight;
        if (distance < bestDistance) {
          bestDistance = distance;
          best = index;
        }
      });
      labels[row * width + column] = best + 1;
    }
  }
  for (let row = 0; row < 3; row++) {
    for (let column = 0; column < 4; column++) labels[row * width + column] = 0;
  }
  return labels;
}

/** Dense 4-connected clump labels of a random binary mask, numbered by first pixel in row-major order. */
export function createClumps(
  width: number,
  height: number,
  density: number,
  seed: number
): Uint32Array {
  const random = createRandom(seed);
  const mask = Array.from({length: width * height}, () => (random() < density ? 1 : 0));
  const labels = new Uint32Array(width * height);
  let next = 0;
  for (let start = 0; start < labels.length; start++) {
    if (!mask[start] || labels[start]) continue;
    next++;
    const stack = [start];
    labels[start] = next;
    while (stack.length) {
      const pixel = stack.pop()!;
      const column = pixel % width;
      const row = Math.floor(pixel / width);
      for (const [dx, dy] of [
        [1, 0],
        [-1, 0],
        [0, 1],
        [0, -1]
      ]) {
        const nx = column + dx;
        const ny = row + dy;
        if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
        const neighbor = ny * width + nx;
        if (mask[neighbor] && !labels[neighbor]) {
          labels[neighbor] = next;
          stack.push(neighbor);
        }
      }
    }
  }
  return labels;
}

/** Applies the guards of `GPURasterPatchLabels`: returns accepted labels (0 elsewhere). */
export function getAcceptedLabels(
  labels: Uint32Array,
  options: {validity?: Uint32Array; capacity: number; componentCount?: number}
): Uint32Array {
  return labels.map((label, pixel) =>
    label !== 0 &&
    label <= options.capacity &&
    (!options.validity || options.validity[pixel] !== 0) &&
    (options.componentCount === undefined || label <= options.componentCount)
      ? label
      : 0
  );
}

/** Per-patch pixel count, bounding box, exposed faces. Row `r` is label `r + 1`. */
export function computePatchMetricsOracle(
  accepted: Uint32Array,
  width: number,
  height: number,
  capacity: number,
  countBorder: boolean
) {
  const counts = new Array<number>(capacity).fill(0);
  const minColumns = new Array<number>(capacity).fill(0);
  const minRows = new Array<number>(capacity).fill(0);
  const maxColumns = new Array<number>(capacity).fill(0);
  const maxRows = new Array<number>(capacity).fill(0);
  const rowFaces = new Array<number>(capacity).fill(0);
  const columnFaces = new Array<number>(capacity).fill(0);
  for (let row = 0; row < height; row++) {
    for (let column = 0; column < width; column++) {
      const key = accepted[row * width + column];
      if (!key) continue;
      const patch = key - 1;
      if (counts[patch] === 0) {
        minColumns[patch] = maxColumns[patch] = column;
        minRows[patch] = maxRows[patch] = row;
      }
      counts[patch]++;
      minColumns[patch] = Math.min(minColumns[patch], column);
      maxColumns[patch] = Math.max(maxColumns[patch], column);
      minRows[patch] = Math.min(minRows[patch], row);
      maxRows[patch] = Math.max(maxRows[patch], row);
      const exposed = (dx: number, dy: number) => {
        const nx = column + dx;
        const ny = row + dy;
        if (nx < 0 || ny < 0 || nx >= width || ny >= height) return countBorder;
        return accepted[ny * width + nx] !== key;
      };
      rowFaces[patch] += Number(exposed(0, -1)) + Number(exposed(0, 1));
      columnFaces[patch] += Number(exposed(-1, 0)) + Number(exposed(1, 0));
    }
  }
  return {counts, minColumns, minRows, maxColumns, maxRows, rowFaces, columnFaces};
}

/** Sieve oracle with the semantics of `GPURasterSieve`. */
export function computeSieveOracle(
  accepted: Uint32Array,
  width: number,
  height: number,
  capacity: number,
  minimumPixels: number,
  mode: 'remove' | 'merge',
  connectivity: 4 | 8
) {
  const counts = new Array<number>(capacity).fill(0);
  for (const key of accepted) if (key) counts[key - 1]++;
  const small = (label: number) => counts[label - 1] > 0 && counts[label - 1] < minimumPixels;
  const bestCount = new Array<number>(capacity).fill(0);
  const target = new Array<number>(capacity).fill(0xffffffff);
  if (mode === 'merge') {
    for (let row = 0; row < height; row++) {
      for (let column = 0; column < width; column++) {
        const key = accepted[row * width + column];
        if (!key || !small(key)) continue;
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            if ((dx === 0 && dy === 0) || (connectivity === 4 && dx !== 0 && dy !== 0)) continue;
            const nx = column + dx;
            const ny = row + dy;
            if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
            const neighbor = accepted[ny * width + nx];
            if (!neighbor || neighbor === key || counts[neighbor - 1] < minimumPixels) continue;
            bestCount[key - 1] = Math.max(bestCount[key - 1], counts[neighbor - 1]);
          }
        }
      }
    }
    for (let row = 0; row < height; row++) {
      for (let column = 0; column < width; column++) {
        const key = accepted[row * width + column];
        if (!key || !small(key)) continue;
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            if ((dx === 0 && dy === 0) || (connectivity === 4 && dx !== 0 && dy !== 0)) continue;
            const nx = column + dx;
            const ny = row + dy;
            if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
            const neighbor = accepted[ny * width + nx];
            if (!neighbor || neighbor === key || counts[neighbor - 1] < minimumPixels) continue;
            if (counts[neighbor - 1] === bestCount[key - 1]) {
              target[key - 1] = Math.min(target[key - 1], neighbor);
            }
          }
        }
      }
    }
  }
  const patchTargets = counts.map((count, index) => {
    if (count >= minimumPixels) return index + 1;
    if (count > 0 && mode === 'merge' && bestCount[index] > 0) return target[index];
    return 0;
  });
  return {
    counts,
    patchTargets,
    sievedCount: counts.filter(count => count > 0 && count < minimumPixels).length,
    labels: accepted.map(key => (key ? patchTargets[key - 1] : 0))
  };
}
