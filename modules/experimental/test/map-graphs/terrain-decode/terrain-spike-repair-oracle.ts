// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Options of {@link computeTerrainSpikeRepair}; defaults match `GPUTerrainSpikeRepair`. */
export type TerrainSpikeRepairOracleOptions = {
  step?: number;
  jump?: number;
  tolerance?: number;
  agreement?: number;
  maximumComponentFraction?: number;
};

/** Result of {@link computeTerrainSpikeRepair}. */
export type TerrainSpikeRepairOracleResult = {
  /** Repaired heights; canonical NaN where invalid. */
  values: Float32Array;
  /** 1 where valid. */
  validity: Uint32Array;
  /** Lowest member pixel index per pixel; invalid pixels label themselves. */
  labels: Uint32Array;
  /** `[jumpCount, repairedPixelCount, shiftedComponentCount, remainingJumpCount, converged]`. */
  statistics: number[];
};

/**
 * Float64 port of mt-image `validateTile` without the median fill. Invalid pixels are excluded:
 * they are isolated, never seams and never repaired. Votes use a Map mode exactly as mt-image;
 * "too large" uses the valid pixel count instead of all pixels.
 */
export function computeTerrainSpikeRepair(
  input: Float32Array,
  validityInput: Uint32Array | undefined,
  width: number,
  height: number,
  options: TerrainSpikeRepairOracleOptions = {}
): TerrainSpikeRepairOracleResult {
  const step = options.step ?? 256;
  const jump = options.jump ?? 200;
  const tolerance = options.tolerance ?? 40;
  const agreement = options.agreement ?? 0.8;
  const maximumFraction = options.maximumComponentFraction ?? 0.25;
  const count = width * height;
  const valid = Uint8Array.from({length: count}, (_, index) =>
    (validityInput ? validityInput[index] !== 0 : true) && Number.isFinite(input[index]) ? 1 : 0
  );
  const heights = Float32Array.from(input);
  const neighborsOf = (index: number): number[] => {
    const column = index % width;
    const result: number[] = [];
    if (column + 1 < width) result.push(index + 1);
    if (column > 0) result.push(index - 1);
    if (index + width < count) result.push(index + width);
    if (index >= width) result.push(index - width);
    return result;
  };
  const countJumps = (): number => {
    let jumps = 0;
    for (let index = 0; index < count; index++) {
      if (!valid[index]) continue;
      const column = index % width;
      if (
        column + 1 < width &&
        valid[index + 1] &&
        Math.abs(heights[index + 1] - heights[index]) > jump
      )
        jumps++;
      if (
        index + width < count &&
        valid[index + width] &&
        Math.abs(heights[index + width] - heights[index]) > jump
      )
        jumps++;
    }
    return jumps;
  };
  const labels = new Uint32Array(count);
  const finish = (statistics: number[]): TerrainSpikeRepairOracleResult => ({
    values: Float32Array.from(heights, (value, index) => (valid[index] ? value : Number.NaN)),
    validity: Uint32Array.from(valid),
    labels,
    statistics
  });
  const jumps = countJumps();
  // Component labels: lowest member pixel index, BFS in ascending seed order.
  const component = new Int32Array(count).fill(-1);
  const sizes: number[] = [];
  const seeds: number[] = [];
  for (let seed = 0; seed < count; seed++) {
    if (!valid[seed]) {
      labels[seed] = seed;
      continue;
    }
    if (component[seed] >= 0) continue;
    const id = sizes.length;
    const stack = [seed];
    component[seed] = id;
    let size = 0;
    while (stack.length) {
      const index = stack.pop() as number;
      size++;
      labels[index] = seed;
      for (const neighbor of neighborsOf(index)) {
        if (
          valid[neighbor] &&
          component[neighbor] < 0 &&
          Math.abs(heights[neighbor] - heights[index]) <= jump
        ) {
          component[neighbor] = id;
          stack.push(neighbor);
        }
      }
    }
    sizes.push(size);
    seeds.push(seed);
  }
  if (!jumps || sizes.length < 2) {
    return finish([jumps, 0, 0, jumps, 1]);
  }
  let main = 0;
  for (let id = 1; id < sizes.length; id++) if (sizes[id] > sizes[main]) main = id;
  const validCount = valid.reduce((sum, flag) => sum + flag, 0);
  const votes = new Map<number, Map<number, number>>();
  const seams = new Int32Array(sizes.length);
  const vote = (a: number, b: number) => {
    const componentA = component[a];
    if (componentA === component[b]) return;
    seams[componentA]++;
    const delta = heights[a] - heights[b];
    const multiple = Math.round(delta / step);
    if (multiple === 0 || !(Math.abs(delta - step * multiple) < tolerance)) return;
    let histogram = votes.get(componentA);
    if (!histogram) {
      histogram = new Map();
      votes.set(componentA, histogram);
    }
    histogram.set(multiple, (histogram.get(multiple) ?? 0) + 1);
  };
  for (let index = 0; index < count; index++) {
    if (!valid[index]) continue;
    for (const neighbor of neighborsOf(index)) {
      if (valid[neighbor]) vote(index, neighbor);
    }
  }
  const shift = new Float64Array(sizes.length);
  for (const [id, histogram] of votes) {
    if (id === main || sizes[id] > validCount * maximumFraction) continue;
    let bestMultiple = 0;
    let bestCount = 0;
    for (const [multiple, voteCount] of histogram) {
      if (voteCount > bestCount) {
        bestMultiple = multiple;
        bestCount = voteCount;
      }
    }
    if (bestCount >= agreement * seams[id]) shift[id] = step * bestMultiple;
  }
  let repaired = 0;
  for (let index = 0; index < count; index++) {
    if (!valid[index]) continue;
    const amount = shift[component[index]];
    if (amount) {
      heights[index] = heights[index] - amount;
      repaired++;
    }
  }
  const shiftedComponents = shift.reduce((sum, amount) => sum + (amount ? 1 : 0), 0);
  return finish([jumps, repaired, shiftedComponents, repaired ? countJumps() : jumps, 1]);
}
