// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {createRandom} from '../outline-geometry/geometry-fixture';
import {getCellReference, getHilbertIndexReference} from './hilbert-oracle';

type Box = [number, number, number, number];

/** Z-order index of a 16-bit cell, as used by the spatial-join Morton sort. */
function getMortonIndex(cellX: number, cellY: number): number {
  const spread = (value: number) => {
    let x = value & 0xffff;
    x = (x | (x << 8)) & 0x00ff00ff;
    x = (x | (x << 4)) & 0x0f0f0f0f;
    x = (x | (x << 2)) & 0x33333333;
    x = (x | (x << 1)) & 0x55555555;
    return x >>> 0;
  };
  return (spread(cellX) | (spread(cellY) << 1)) >>> 0;
}

/**
 * Sum of node perimeters of the complete binary BVH built by pairing adjacent leaves in order
 * (the `GPUBVH` topology), the standard surface-area proxy of tree quality (lower is better).
 */
function getTreeCost(boxes: Box[]): number {
  let level = boxes;
  let cost = 0;
  while (level.length > 1) {
    const next: Box[] = [];
    for (let i = 0; i < level.length; i += 2) {
      const a = level[i];
      const b = level[i + 1] ?? a;
      const merged: Box = [
        Math.min(a[0], b[0]),
        Math.min(a[1], b[1]),
        Math.max(a[2], b[2]),
        Math.max(a[3], b[3])
      ];
      cost += merged[2] - merged[0] + (merged[3] - merged[1]);
      next.push(merged);
    }
    level = next;
  }
  return cost;
}

function sortByKey(boxes: Box[], getKey: (cx: number, cy: number) => number): Box[] {
  const keyed = boxes.map(box => ({
    box,
    key: getKey((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
  }));
  keyed.sort((a, b) => a.key - b.key);
  return keyed.map(entry => entry.box);
}

const SCENES: Record<string, (random: () => number) => Box[]> = {
  uniform: random =>
    Array.from({length: 16384}, () => {
      const x = random() * 1000;
      const y = random() * 1000;
      return [x, y, x + random() * 4, y + random() * 4] as Box;
    }),
  clustered: random => {
    const clusters = Array.from({length: 40}, () => [random() * 1000, random() * 1000]);
    return Array.from({length: 16384}, () => {
      const [cx, cy] = clusters[Math.floor(random() * clusters.length)];
      const x = cx + (random() + random() + random() - 1.5) * 30;
      const y = cy + (random() + random() + random() - 1.5) * 30;
      return [x, y, x + random() * 2, y + random() * 2] as Box;
    });
  },
  roads: random =>
    // Elongated boxes along a grid of streets.
    Array.from({length: 16384}, () => {
      const horizontal = random() < 0.5;
      const lane = Math.floor(random() * 25) * 40;
      const along = random() * 1000;
      return (
        horizontal
          ? [along, lane, along + 10 + random() * 20, lane + 1]
          : [lane, along, lane + 1, along + 10 + random() * 20]
      ) as Box;
    })
};

it('Hilbert order builds a BVH of lower or equal surface cost than Morton order (A/B)', () => {
  const lines: string[] = [];
  for (const [name, createScene] of Object.entries(SCENES)) {
    const boxes = createScene(createRandom(11));
    const centersX = boxes.map(box => (box[0] + box[2]) / 2);
    const centersY = boxes.map(box => (box[1] + box[3]) / 2);
    const [minX, maxX] = [Math.min(...centersX), Math.max(...centersX)];
    const [minY, maxY] = [Math.min(...centersY), Math.max(...centersY)];
    const cellOf = (cx: number, cy: number, order: number) => [
      getCellReference(order, cx, minX, maxX),
      getCellReference(order, cy, minY, maxY)
    ];
    const morton = getTreeCost(
      sortByKey(boxes, (cx, cy) => {
        const [x, y] = cellOf(cx, cy, 16);
        return getMortonIndex(x, y);
      })
    );
    const hilbert = getTreeCost(
      sortByKey(boxes, (cx, cy) => {
        const [x, y] = cellOf(cx, cy, 16);
        return getHilbertIndexReference(16, x, y);
      })
    );
    const unsorted = getTreeCost(boxes);
    lines.push(
      `${name}: tree cost Morton ${morton.toFixed(0)}, Hilbert ${hilbert.toFixed(0)} ` +
        `(${((100 * (hilbert - morton)) / morton).toFixed(1)}%), unsorted ${unsorted.toFixed(0)}`
    );
    expect(hilbert).toBeLessThan(unsorted);
    expect(morton).toBeLessThan(unsorted);
  }
  console.log(`Hilbert versus Morton BVH quality\n${lines.join('\n')}`);
});
