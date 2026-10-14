// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {AnalysisRig, createSeededRandom, type CPUWeights} from '../catchment-accessibility/rig';

/** Events: a space-time clustered core plus uniform background (so interaction exists). */
export function createEvents(count: number, seed: number) {
  const random = createSeededRandom(seed);
  const positions = new Float32Array(count * 2);
  const times = new Float32Array(count);
  for (let index = 0; index < count; index++) {
    const clustered = index % 3 === 0;
    const center = Math.floor(random() * 4);
    positions[index * 2] = Math.fround(clustered ? center * 25 + random() * 6 : random() * 100);
    positions[index * 2 + 1] = Math.fround(clustered ? center * 20 + random() * 6 : random() * 100);
    times[index] = Math.fround(clustered ? center * 30 + random() * 4 : random() * 120);
  }
  return {positions, times};
}

export function buildBand(
  positions: Float32Array,
  radius: number
): CPUWeights & {distances: number[]} {
  const rows = positions.length / 2;
  const csr = {
    offsets: [0],
    neighbors: [] as number[],
    weights: [] as number[],
    distances: [] as number[]
  };
  for (let row = 0; row < rows; row++) {
    for (let other = 0; other < rows; other++) {
      const distance = Math.hypot(
        positions[row * 2] - positions[other * 2],
        positions[row * 2 + 1] - positions[other * 2 + 1]
      );
      if (other !== row && distance <= radius) {
        csr.neighbors.push(other);
        csr.weights.push(1);
        csr.distances.push(Math.fround(distance));
      }
    }
    csr.offsets.push(csr.neighbors.length);
  }
  return csr;
}

export function uploadPairs(rig: AnalysisRig, csr: CPUWeights & {distances: number[]}) {
  const capacity = csr.neighbors.length + 4;
  const distances = new Float32Array(capacity);
  distances.set(csr.distances);
  return {
    ...rig.weights(csr, 4),
    distances: rig.input(distances, 'float32')
  };
}

/** Null scene for sequential stopping: Fisher-Yates destroys the space-time clustering. */
export function createStoppingEvents() {
  const events = createEvents(240, 5);
  let randomState = 4321;
  for (let row = events.times.length - 1; row > 0; row--) {
    randomState = (Math.imul(randomState, 1664525) + 1013904223) >>> 0;
    const other = Math.floor((randomState / 4294967296) * (row + 1));
    [events.times[row], events.times[other]] = [events.times[other], events.times[row]];
  }
  return events;
}
