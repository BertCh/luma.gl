// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getPhilox4x32, PHILOX_PURPOSES} from '../../../src/utils/philox-random';

/**
 * CPU mirror of the `GPUMaxPRegions` parallel multi-seed construction. It executes the same
 * rules as the WGSL kernels in the same pass order, with float32 arithmetic (`Math.fround`), so
 * on data whose distances are exact in float32 (integer-valued columns, `standardize: false`) the
 * labels equal the GPU labels. Concurrent GPU steps are atomic minima or single-writer updates,
 * so executing them one thread after the other gives the same result.
 *
 * Test oracle only: nothing here runs in the library.
 */

const NONE = 0xffffffff;
const PHASE_GROWING = 1;
const PHASE_COMPLETE = 2;
const PHASE_FAILED = 3;
const PHASE_MARKED = 4;
const TIE_MULTIPLIER = 0x9e3779b1;
const WAVE_START_FRACTION = Math.fround(0.4);
const WAVE_GROWTH = Math.fround(1.3);
const MEAN_LANES = 256;
const MAXIMUM_EXPECTED_REGION_SIZE = 1048576;
const MAXIMUM_NEIGHBOR_PRIORITY = 63;

/** Properties of {@link constructMaxPRegionsMirror}. */
export type MaxPConstructionMirrorProps = {
  /** CSR of the symmetric adjacency, built by {@link createSymmetricAdjacencyMirror}. */
  offsets: Uint32Array;
  neighbors: Uint32Array;
  /** Row-major attribute table, already standardized when the contributor standardizes. */
  values: Float32Array;
  columnCount: number;
  extensive: Float32Array;
  threshold: number;
  seed: number;
  startCount: number;
  maximumGrowthSteps: number;
};

/** One start's result. */
export type MaxPStartMirror = {
  /** Region ID (seed row) per row, before the dense relabel. */
  labels: Uint32Array;
  regionCount: number;
  infeasibleRegionCount: number;
  /** Growth-phase within-region sum of squares of the regions that remain. */
  withinSquares: number;
  roundCount: number;
};

/** Result of {@link constructMaxPRegionsMirror}. */
export type MaxPConstructionMirror = {
  /** Dense labels `0..regionCount-1` in ascending region-ID order of the selected start. */
  labels: Uint32Array;
  regionCount: number;
  infeasibleRegionCount: number;
  selectedStart: number;
  starts: MaxPStartMirror[];
};

/**
 * The undirected union of `W` and `W'` as the contributor builds it: each row lists its `W`
 * neighbors (out-of-range IDs replaced by the row itself) followed by the rows that list it.
 */
export function createSymmetricAdjacencyMirror(
  offsets: ArrayLike<number>,
  neighbors: ArrayLike<number>
): {offsets: Uint32Array; neighbors: Uint32Array} {
  const rowCount = offsets.length - 1;
  const lists: number[][] = Array.from({length: rowCount}, () => []);
  for (let row = 0; row < rowCount; row++) {
    for (let slot = offsets[row]; slot < offsets[row + 1]; slot++) {
      lists[row].push(neighbors[slot] >= rowCount ? row : neighbors[slot]);
    }
  }
  for (let row = 0; row < rowCount; row++) {
    for (let slot = offsets[row]; slot < offsets[row + 1]; slot++) {
      const neighbor = neighbors[slot];
      if (neighbor < rowCount) {
        lists[neighbor].push(row);
      }
    }
  }
  const adjacencyOffsets = [0];
  const adjacencyNeighbors: number[] = [];
  for (const list of lists) {
    adjacencyNeighbors.push(...list);
    adjacencyOffsets.push(adjacencyNeighbors.length);
  }
  return {
    offsets: Uint32Array.from(adjacencyOffsets),
    neighbors: Uint32Array.from(adjacencyNeighbors)
  };
}

/** Modular inverse of {@link TIE_MULTIPLIER} modulo 2^32, for the tie order's bijection. */
export const MAX_P_TIE_MULTIPLIER = TIE_MULTIPLIER;
export const MAX_P_TIE_INVERSE = Number(
  (() => {
    const modulus = 1n << 32n;
    let result = 1n;
    // Newton iteration x <- x (2 - a x) doubles the correct low bits each step.
    for (let step = 0; step < 6; step++) {
      result = (result * (2n - ((BigInt(TIE_MULTIPLIER) * result) % modulus) + modulus)) % modulus;
    }
    return result;
  })()
);

function getTie(value: number, salt: number): number {
  return (Math.imul(value, TIE_MULTIPLIER) ^ salt) >>> 0;
}

function getTieValue(tie: number, salt: number): number {
  return Math.imul((tie ^ salt) >>> 0, MAX_P_TIE_INVERSE) >>> 0;
}

/** `ceil(threshold / mean(extensive))` with the kernel's lane-strided float32 sum, at least 1. */
export function getExpectedRegionSizeMirror(extensive: Float32Array, threshold: number): number {
  const lanes = new Float32Array(MEAN_LANES);
  for (let lane = 0; lane < MEAN_LANES; lane++) {
    let sum = 0;
    for (let row = lane; row < extensive.length; row += MEAN_LANES) {
      sum = Math.fround(sum + extensive[row]);
    }
    lanes[lane] = sum;
  }
  for (let stride = MEAN_LANES / 2; stride > 0; stride >>= 1) {
    for (let lane = 0; lane < stride; lane++) {
      lanes[lane] = Math.fround(lanes[lane] + lanes[lane + stride]);
    }
  }
  const mean = Math.fround(lanes[0] / Math.fround(extensive.length));
  if (!(mean > 0)) {
    return 1;
  }
  const quotient = Math.fround(Math.fround(threshold) / mean);
  return Math.max(1, Math.min(Math.ceil(quotient), MAXIMUM_EXPECTED_REGION_SIZE));
}

/** Runs one start of the construction. */
export function constructMaxPStartMirror(
  props: MaxPConstructionMirrorProps,
  start: number
): MaxPStartMirror {
  const {offsets, neighbors, values, columnCount, extensive, maximumGrowthSteps} = props;
  const rowCount = extensive.length;
  // Equals `getGPUMaxPRoundCounts` of the contributor (checked by the node spec).
  const growthRounds = Math.min(maximumGrowthSteps, 2 * rowCount + 2);
  const enclaveRounds = Math.min(maximumGrowthSteps, rowCount + 1);
  const threshold = Math.fround(props.threshold);
  const purpose = PHILOX_PURPOSES.maxPRegionsSeedOrder.id;
  const keys = new Uint32Array(rowCount);
  for (let row = 0; row < rowCount; row++) {
    keys[row] = getPhilox4x32([row, start, 0, 0], [props.seed, purpose])[0];
  }
  const salt = getPhilox4x32([NONE, start, 0, 0], [props.seed, purpose])[1];
  const expectedSize = getExpectedRegionSizeMirror(extensive, props.threshold);

  const labels = new Uint32Array(rowCount).fill(NONE);
  const phase = new Uint8Array(rowCount);
  const regionCountRows = new Float32Array(rowCount);
  const regionExtensive = new Float32Array(rowCount);
  const regionSquares = new Float32Array(rowCount);
  const regionSums = new Float32Array(rowCount * columnCount);

  const getDistance = (row: number, region: number): number => {
    let distance = 0;
    for (let column = 0; column < columnCount; column++) {
      const mean = Math.fround(regionSums[region * columnCount + column] / regionCountRows[region]);
      const difference = Math.fround(values[row * columnCount + column] - mean);
      distance = Math.fround(distance + Math.fround(difference * difference));
    }
    return distance;
  };
  const floatBits = new DataView(new ArrayBuffer(4));
  const getFloatBits = (value: number): number => {
    floatBits.setFloat32(0, value);
    return floatBits.getUint32(0);
  };

  let growing = 0;
  let unassigned = rowCount;
  let round = 0;
  let nextWave = 0;
  const periodRounds = (expectedSize + 1) >> 1;
  let fraction = Math.fround(WAVE_START_FRACTION / Math.fround(expectedSize));
  let roundCount = 0;

  // Claim scratch, one record per row index (region ID for the first two, row for the last two).
  const bestKey = new Uint32Array(rowCount);
  const bestTie = new Uint32Array(rowCount);
  const claimKey = new Uint32Array(rowCount);
  const claimTie = new Uint32Array(rowCount);

  for (let step = 0; step < growthRounds; step++) {
    const active = growing > 0 || unassigned > 0;
    if (!active) {
      break;
    }
    roundCount = step + 1;
    const isWave = unassigned > 0 && (growing === 0 || round >= nextWave);
    let cutKey = 0;
    if (isWave) {
      cutKey = fraction >= 1 ? NONE : Math.floor(Math.fround(fraction * 4294967040)) >>> 0;
      nextWave = round + periodRounds;
      fraction = Math.min(1, Math.fround(fraction * WAVE_GROWTH));
    }
    round += 1;

    // Reset the claim scratch.
    bestKey.fill(NONE);
    bestTie.fill(NONE);
    claimKey.fill(NONE);
    claimTie.fill(NONE);

    if (isWave) {
      const isEligible = (row: number) => labels[row] === NONE && keys[row] <= cutKey;
      for (let row = 0; row < rowCount; row++) {
        if (!isEligible(row)) continue;
        let blocked = false;
        for (let slot = offsets[row]; slot < offsets[row + 1] && !blocked; slot++) {
          const neighbor = neighbors[slot];
          if (neighbor === row) continue;
          const neighborLabel = labels[neighbor];
          if (neighborLabel !== NONE && phase[neighborLabel] === PHASE_GROWING) {
            blocked = true;
          } else if (
            isEligible(neighbor) &&
            (keys[neighbor] < keys[row] || (keys[neighbor] === keys[row] && neighbor < row))
          ) {
            blocked = true;
          }
        }
        if (!blocked) {
          phase[row] = PHASE_MARKED;
        }
      }
      for (let row = 0; row < rowCount; row++) {
        if (phase[row] !== PHASE_MARKED) continue;
        labels[row] = row;
        regionCountRows[row] = 1;
        regionExtensive[row] = extensive[row];
        regionSquares[row] = 0;
        for (let column = 0; column < columnCount; column++) {
          regionSums[row * columnCount + column] = values[row * columnCount + column];
        }
        unassigned -= 1;
        if (extensive[row] >= threshold) {
          phase[row] = PHASE_COMPLETE;
        } else {
          phase[row] = PHASE_GROWING;
          growing += 1;
        }
      }
    }

    // Propose: every unassigned frontier row offers itself to each adjacent growing region.
    const getKey = (row: number, adjacentRegion: number, priority: number): number => {
      const distanceBits = getFloatBits(getDistance(row, adjacentRegion));
      return (((priority << 26) | (distanceBits >>> 6)) >>> 0) as number;
    };
    const propose = (pass: 1 | 2) => {
      for (let row = 0; row < rowCount; row++) {
        if (labels[row] !== NONE) continue;
        let priority = 0;
        for (let slot = offsets[row]; slot < offsets[row + 1]; slot++) {
          const neighbor = neighbors[slot];
          if (neighbor !== row && labels[neighbor] === NONE) priority += 1;
        }
        priority = Math.min(priority, MAXIMUM_NEIGHBOR_PRIORITY);
        for (let slot = offsets[row]; slot < offsets[row + 1]; slot++) {
          const neighborLabel = labels[neighbors[slot]];
          if (neighborLabel === NONE || phase[neighborLabel] !== PHASE_GROWING) continue;
          const key = getKey(row, neighborLabel, priority);
          if (pass === 1) {
            bestKey[neighborLabel] = Math.min(bestKey[neighborLabel], key);
          } else if (key === bestKey[neighborLabel]) {
            bestTie[neighborLabel] = Math.min(bestTie[neighborLabel], getTie(row, salt));
          }
        }
      }
    };
    propose(1);
    propose(2);

    // Claim: a row goes to the adjacent proposing region with the smallest (key, tie).
    for (let pass = 1; pass <= 2; pass++) {
      for (let candidate = 0; candidate < rowCount; candidate++) {
        if (labels[candidate] !== candidate || phase[candidate] !== PHASE_GROWING) continue;
        if (bestKey[candidate] === NONE) continue;
        const row = getTieValue(bestTie[candidate], salt);
        if (pass === 1) {
          claimKey[row] = Math.min(claimKey[row], bestKey[candidate]);
        } else if (bestKey[candidate] === claimKey[row]) {
          claimTie[row] = Math.min(claimTie[row], getTie(candidate, salt));
        }
      }
    }

    // Commit.
    for (let candidate = 0; candidate < rowCount; candidate++) {
      if (labels[candidate] !== candidate || phase[candidate] !== PHASE_GROWING) continue;
      if (bestKey[candidate] === NONE) {
        phase[candidate] = PHASE_FAILED;
        growing -= 1;
        continue;
      }
      const row = getTieValue(bestTie[candidate], salt);
      if (claimTie[row] !== getTie(candidate, salt)) continue;
      const distance = getDistance(row, candidate);
      const count = regionCountRows[candidate];
      labels[row] = candidate;
      regionSquares[candidate] = Math.fround(
        regionSquares[candidate] +
          Math.fround(Math.fround(count / Math.fround(count + 1)) * distance)
      );
      regionCountRows[candidate] = Math.fround(count + 1);
      regionExtensive[candidate] = Math.fround(regionExtensive[candidate] + extensive[row]);
      for (let column = 0; column < columnCount; column++) {
        const index = candidate * columnCount + column;
        regionSums[index] = Math.fround(regionSums[index] + values[row * columnCount + column]);
      }
      unassigned -= 1;
      if (regionExtensive[candidate] >= threshold) {
        phase[candidate] = PHASE_COMPLETE;
        growing -= 1;
      }
    }
  }

  // Enclaves: loose rows join the most similar adjacent completed region, round by round.
  const isLoose = (row: number) => labels[row] === NONE || phase[labels[row]] !== PHASE_COMPLETE;
  for (let step = 0; step < enclaveRounds; step++) {
    const proposals = new Uint32Array(rowCount).fill(NONE);
    for (let row = 0; row < rowCount; row++) {
      if (!isLoose(row)) continue;
      let bestKeyValue = NONE;
      let bestTieValue = NONE;
      let bestRegion = NONE;
      for (let slot = offsets[row]; slot < offsets[row + 1]; slot++) {
        const neighborLabel = labels[neighbors[slot]];
        if (neighborLabel === NONE || phase[neighborLabel] !== PHASE_COMPLETE) continue;
        const key = getFloatBits(getDistance(row, neighborLabel));
        const tie = getTie(neighborLabel, salt);
        if (key < bestKeyValue || (key === bestKeyValue && tie < bestTieValue)) {
          bestKeyValue = key;
          bestTieValue = tie;
          bestRegion = neighborLabel;
        }
      }
      proposals[row] = bestRegion;
    }
    let moved = 0;
    for (let row = 0; row < rowCount; row++) {
      if (proposals[row] !== NONE) {
        labels[row] = proposals[row];
        moved += 1;
      }
    }
    if (moved === 0) break;
  }

  // Rows that no region reached become singleton under-threshold regions.
  for (let row = 0; row < rowCount; row++) {
    if (labels[row] === NONE) {
      labels[row] = row;
      phase[row] = PHASE_FAILED;
      regionSquares[row] = 0;
    }
  }
  const inUse = new Uint8Array(rowCount);
  for (let row = 0; row < rowCount; row++) {
    inUse[labels[row]] = 1;
  }
  let regionCount = 0;
  let infeasibleRegionCount = 0;
  let withinSquares = 0;
  // Fixed lane-strided order with a binary tree, as the kernel's reduction.
  const lanes = new Float32Array(MEAN_LANES);
  for (let lane = 0; lane < MEAN_LANES; lane++) {
    let sum = 0;
    for (let row = lane; row < rowCount; row += MEAN_LANES) {
      if (inUse[row]) {
        sum = Math.fround(sum + regionSquares[row]);
      }
    }
    lanes[lane] = sum;
  }
  for (let stride = MEAN_LANES / 2; stride > 0; stride >>= 1) {
    for (let lane = 0; lane < stride; lane++) {
      lanes[lane] = Math.fround(lanes[lane] + lanes[lane + stride]);
    }
  }
  withinSquares = lanes[0];
  for (let row = 0; row < rowCount; row++) {
    if (!inUse[row]) continue;
    regionCount += 1;
    if (phase[row] !== PHASE_COMPLETE) infeasibleRegionCount += 1;
  }
  return {labels, regionCount, infeasibleRegionCount, withinSquares, roundCount};
}

/** Runs every start and applies the contributor's start selection and dense relabel. */
export function constructMaxPRegionsMirror(
  props: MaxPConstructionMirrorProps
): MaxPConstructionMirror {
  const starts = Array.from({length: props.startCount}, (_, start) =>
    constructMaxPStartMirror(props, start)
  );
  let best = 0;
  for (let start = 1; start < starts.length; start++) {
    const candidate = starts[start];
    const incumbent = starts[best];
    if (
      candidate.infeasibleRegionCount < incumbent.infeasibleRegionCount ||
      (candidate.infeasibleRegionCount === incumbent.infeasibleRegionCount &&
        (candidate.regionCount > incumbent.regionCount ||
          (candidate.regionCount === incumbent.regionCount &&
            candidate.withinSquares < incumbent.withinSquares)))
    ) {
      best = start;
    }
  }
  const selected = starts[best];
  const ranks = new Map<number, number>();
  const sorted = Array.from(new Set(selected.labels)).sort((a, b) => a - b);
  sorted.forEach((label, rank) => ranks.set(label, rank));
  return {
    labels: Uint32Array.from(selected.labels, label => ranks.get(label)!),
    regionCount: selected.regionCount,
    infeasibleRegionCount: selected.infeasibleRegionCount,
    selectedStart: best,
    starts
  };
}

/** Inputs of {@link runMaxPRegionsMirror}: the contributor's props in plain arrays. */
export type MaxPMirrorRunProps = {
  /** CSR of the input weights (directed weights are symmetrized like the contributor does). */
  offsets: ArrayLike<number>;
  neighbors: ArrayLike<number>;
  values: ArrayLike<number>;
  columnCount?: number;
  extensive: ArrayLike<number>;
  threshold: number;
  seed?: number;
  startCount?: number;
  maximumGrowthSteps?: number;
  /** Standardize every column to mean 0 and population deviation 1 first. Defaults to true. */
  standardize?: boolean;
};

/** Region sums of the extensive value, by label. */
export function getRegionSums(labels: ArrayLike<number>, extensive: ArrayLike<number>) {
  const sums = new Map<number, number>();
  for (let row = 0; row < labels.length; row++) {
    sums.set(labels[row], (sums.get(labels[row]) ?? 0) + extensive[row]);
  }
  return sums;
}

/**
 * Runs {@link constructMaxPRegionsMirror} from the contributor's props: symmetrizes the weights,
 * standardizes the columns in float64 (the GPU standardizes in float32; the construction itself
 * runs in float32 on the result) and applies the contributor's defaults.
 */
export function runMaxPRegionsMirror(props: MaxPMirrorRunProps): MaxPConstructionMirror {
  const columnCount = props.columnCount ?? 1;
  const rowCount = props.extensive.length;
  const adjacency = createSymmetricAdjacencyMirror(props.offsets, props.neighbors);
  const table = Float32Array.from(props.values);
  if (props.standardize !== false) {
    for (let column = 0; column < columnCount; column++) {
      let sum = 0;
      for (let row = 0; row < rowCount; row++) sum += props.values[row * columnCount + column];
      const mean = sum / rowCount;
      let squares = 0;
      for (let row = 0; row < rowCount; row++) {
        squares += (props.values[row * columnCount + column] - mean) ** 2;
      }
      const deviation = Math.sqrt(squares / rowCount);
      for (let row = 0; row < rowCount; row++) {
        table[row * columnCount + column] =
          deviation > 1e-20 ? (props.values[row * columnCount + column] - mean) / deviation : 0;
      }
    }
  }
  return constructMaxPRegionsMirror({
    offsets: adjacency.offsets,
    neighbors: adjacency.neighbors,
    values: table,
    columnCount,
    extensive: Float32Array.from(props.extensive),
    threshold: props.threshold,
    seed: props.seed ?? 0,
    startCount: props.startCount ?? 16,
    maximumGrowthSteps: props.maximumGrowthSteps ?? 256
  });
}
