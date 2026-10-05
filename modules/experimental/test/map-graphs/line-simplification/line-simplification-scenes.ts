// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {LineSimplificationScene} from './line-simplification-oracle';

/** Deterministic xorshift in [0, 1). */
export function createRandom(seed: number): () => number {
  let state = seed >>> 0 || 1;
  return () => {
    state ^= state << 13;
    state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5;
    state >>>= 0;
    return state / 4294967296;
  };
}

/** Builds a scene from per-line `[x, y, t]` samples. */
export function createScene(
  lines: readonly (readonly (readonly number[])[])[],
  metric: LineSimplificationScene['metric'] = 'segment'
): LineSimplificationScene {
  const rowCount = lines.reduce((sum, line) => sum + line.length, 0);
  const positions = new Float32Array(rowCount * 2);
  const timestamps = new Float32Array(rowCount);
  const trackOffsets = new Uint32Array(lines.length + 1);
  let row = 0;
  for (const [lineIndex, line] of lines.entries()) {
    trackOffsets[lineIndex] = row;
    for (const [x, y, t] of line) {
      positions[2 * row] = x;
      positions[2 * row + 1] = y;
      timestamps[row] = t ?? row;
      row++;
    }
  }
  trackOffsets[lines.length] = row;
  return {positions, trackOffsets, timestamps, metric};
}

/**
 * Random-walk GPS-like tracks with varied lengths (including empty, one-, and two-vertex lines),
 * quantized coordinates that create exact distance ties, and repeated samples.
 */
export function createRandomTracksScene(
  seed: number,
  lineCount: number,
  maximumLength: number,
  metric: LineSimplificationScene['metric'] = 'segment'
): LineSimplificationScene {
  const random = createRandom(seed);
  const lines: number[][][] = [];
  for (let line = 0; line < lineCount; line++) {
    const roll = random();
    const length =
      roll < 0.05
        ? 0
        : roll < 0.1
          ? 1
          : roll < 0.15
            ? 2
            : 3 + Math.floor(random() * (maximumLength - 2));
    let x = Math.floor(random() * 200) - 100;
    let y = Math.floor(random() * 200) - 100;
    let t = Math.floor(random() * 50);
    let headingX = random() - 0.5;
    let headingY = random() - 0.5;
    const samples: number[][] = [];
    for (let sample = 0; sample < length; sample++) {
      samples.push([x, y, t]);
      if (random() < 0.08) {
        // Repeated sample (duplicate position, same or later time).
        t += random() < 0.5 ? 0 : 1;
        continue;
      }
      headingX += (random() - 0.5) * 0.6;
      headingY += (random() - 0.5) * 0.6;
      x = Math.round((x + headingX * 4) * 4) / 4;
      y = Math.round((y + headingY * 4) * 4) / 4;
      t += 1 + Math.floor(random() * 4);
    }
    lines.push(samples);
  }
  return createScene(lines, metric);
}

/**
 * An outward Archimedean spiral. Its Douglas-Peucker split tree is deep, so a small round cap
 * leaves rows undecided.
 */
export function createSpiralScene(vertexCount: number): LineSimplificationScene {
  const samples: number[][] = [];
  for (let index = 0; index < vertexCount; index++) {
    const angle = index * 0.35;
    const radius = 1 + index * 0.25;
    samples.push([radius * Math.cos(angle), radius * Math.sin(angle), index]);
  }
  return createScene([samples]);
}

/** Hand-built corner cases: collinear, duplicate, empty, single, two-vertex, and closed lines. */
export function createCornerCaseScene(): LineSimplificationScene {
  return createScene([
    // Collinear, evenly and unevenly spaced, and doubling back along the line.
    [
      [0, 0],
      [1, 0],
      [2, 0],
      [3, 0],
      [4, 0]
    ],
    [
      [0, 1],
      [5, 1],
      [2, 1],
      [7, 1],
      [3, 1],
      [10, 1]
    ],
    // Every vertex identical.
    [
      [4, 4],
      [4, 4],
      [4, 4],
      [4, 4]
    ],
    // Empty line.
    [],
    // Single vertex.
    [[9, 9]],
    // Two vertices.
    [
      [0, 0],
      [3, 4]
    ],
    // Duplicates inside a bent line and a closed ring (first == last).
    [
      [0, 0],
      [1, 2],
      [1, 2],
      [2, 0],
      [2, 0],
      [3, 3]
    ],
    [
      [0, 0],
      [2, 0],
      [2, 2],
      [0, 2],
      [0, 0]
    ],
    // Equal-distance ties: rows 1 and 3 are both 1 from the chord.
    [
      [0, 0],
      [1, 1],
      [2, 0],
      [3, 1],
      [4, 0]
    ]
  ]);
}
