// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createHash} from 'node:crypto';
import {expect, it} from 'vitest';
import {
  getPolygonEdgeSlabQueryWGSL,
  getPolygonEdgeSlabLoopWGSL
} from '../../src/utils/polygon-edge-slabs';

it('keeps the legacy query and loop byte-identical when options are omitted', () => {
  const hash = (source: string) => createHash('sha256').update(source).digest('hex');
  // Captured directly from the original generators before adding storage or dedupe options.
  expect(hash(getPolygonEdgeSlabQueryWGSL({vertexCount: 32, entryCapacity: 1024}))).toBe(
    '56746a9dc7d824baf841f6a8a7b488438ef7a655f4fac541c69caece923afbb8'
  );
  expect(
    hash(getPolygonEdgeSlabLoopWGSL({owner: '0u', lowY: 'low', highY: 'high', body: 'hits++;'}))
  ).toBe('14e5f5bef46c65368e539bf5ae0b5c4d9fc7a38073c3c2fb10244cae32978d30');
});

it('emits dedupe after point loads only on request and consistently remaps all storage', () => {
  const options = {owner: '0u', lowY: 'low', highY: 'high', getVertex: 'vertex', body: 'hits++;'};
  const plain = getPolygonEdgeSlabLoopWGSL(options);
  expect(plain).not.toContain('edgeSlabIndex != max');
  const dedupe = getPolygonEdgeSlabLoopWGSL({...options, dedupe: {edgeLowY: 'min(a.y, b.y)'}});
  const predicate =
    'if (!edgeSlabIsWalked && edgeSlabIndex != max(edgeSlabFirst, edgeSlabOfHeight(edgeSlabOwner, min(a.y, b.y)))) { continue; }';
  expect(dedupe).toContain(predicate);
  expect(dedupe.indexOf(predicate)).toBeGreaterThan(dedupe.indexOf('let b = vertex(edgeEnd);'));
  expect(dedupe.indexOf(predicate)).toBeLessThan(dedupe.indexOf('hits++;'));
  const storage = {
    params: {array: 'packed', offset: '(offset + 8u)'},
    offsets: {array: 'packed', offset: '(offset + 14u)'},
    entries: {array: 'packed', offset: '(offset + 40u)'}
  };
  const source =
    getPolygonEdgeSlabQueryWGSL({vertexCount: 32, entryCapacity: 128}, storage) +
    getPolygonEdgeSlabLoopWGSL({...options, storage});
  expect(source).not.toMatch(/\bedgeSlab(?:Params|Offsets|Entries)(?:Offset)?\b/);
  expect(source).toContain('let ownerBase = (offset + 8u) + owner * EDGE_SLAB_PARAMS_STRIDE;');
  expect(source).toContain('packed[(offset + 14u)');
  expect(source).toContain('packed[(offset + 40u)');
});

function slabOf(height: number, count: number, origin: number, scale: number): number {
  if (count <= 1) {
    return 0;
  }
  return Math.min(
    count - 1,
    Math.max(0, Math.floor(Math.fround(Math.fround(height - origin) * scale)))
  );
}

it('binary32 listing and dedupe visit every overlapping edge exactly once', () => {
  let state = 81623;
  const random = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 2 ** 32;
  };
  for (let polygon = 0; polygon < 60; polygon++) {
    const count = 1 + Math.floor(random() * 128);
    const origin = Math.fround(polygon % 2 ? 1e6 : -1500);
    const scale = Math.fround(count / 2048);
    const heights = Array.from({length: 128}, (_, vertex) =>
      Math.fround(
        origin +
          (vertex % 5 === 0
            ? 1024
            : vertex % 5 === 1
              ? 0
              : vertex % 5 === 2
                ? 2048
                : vertex % 5 === 3
                  ? Math.floor(random() * count) / scale
                  : random() * 2048)
      )
    );
    // Explicit horizontal edges, exact slab boundaries, and full-height edges.
    heights.splice(
      0,
      6,
      origin,
      origin,
      Math.fround(origin + 2048),
      origin,
      Math.fround(origin + 1 / scale),
      Math.fround(origin + 1 / scale)
    );
    const slabs: number[][] = Array.from({length: count}, () => []);
    heights.forEach((height, edge) => {
      const next = heights[(edge + 1) % heights.length];
      for (
        let slab = slabOf(Math.min(height, next), count, origin, scale);
        slab <= slabOf(Math.max(height, next), count, origin, scale);
        slab++
      ) {
        slabs[slab].push(edge);
      }
    });
    for (let query = 0; query < 70; query++) {
      const firstHeight =
        query < heights.length / 2
          ? heights[query]
          : Math.fround(origin + (random() * 1.4 - 0.2) * 2048);
      const secondHeight =
        query % 4 === 0 ? firstHeight : Math.fround(origin + (random() * 1.4 - 0.2) * 2048);
      const low = Math.min(firstHeight, secondHeight);
      const high = Math.max(firstHeight, secondHeight);
      const first = slabOf(low, count, origin, scale);
      const last = slabOf(high, count, origin, scale);
      for (const walked of [false, true]) {
        const visits = new Uint32Array(heights.length);
        for (let slab = walked ? 0 : first; slab <= (walked ? 0 : last); slab++) {
          const entries = walked ? heights.map((_, edge) => edge) : slabs[slab];
          for (const edge of entries) {
            const edgeLow = Math.min(heights[edge], heights[(edge + 1) % heights.length]);
            if (!walked && slab !== Math.max(first, slabOf(edgeLow, count, origin, scale))) {
              continue;
            }
            visits[edge]++;
          }
        }
        heights.forEach((height, edge) => {
          const next = heights[(edge + 1) % heights.length];
          expect(visits[edge]).toBeLessThanOrEqual(1);
          if (Math.min(height, next) <= high && Math.max(height, next) >= low) {
            expect(visits[edge]).toBe(1);
          }
        });
      }
    }
  }
}, 30000);
