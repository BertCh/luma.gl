// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {GPUGroupConvexHull} from '../../../src/gpu-spatial-analysis/group-geometry';
import {
  countDistinctPipelines,
  createNullWebGPUDevice,
  getParsedKernelSources,
  getStorageBindingCount,
  getWorkgroupMemoryBytes
} from '../../utils/gpu-contributor-test-utils';
import {
  computeGroupConvexHullOracle,
  createRandom,
  getOracleGroupKeys,
  quantizeToLattice
} from './group-geometry-oracle';

it('GPUGroupConvexHull prefilter levels beyond the second share pipelines (rule 12)', () => {
  const buildNodes = (prefilterLevels: number) => {
    const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: `hull-${prefilterLevels}`});
    let serial = 0;
    const view = <Format extends 'uint32' | 'float32x2'>(format: Format, length: number) =>
      createTransientView(graph, `hull-${prefilterLevels}-view-${serial++}`, format, length);
    const rows = 600;
    const groupCount = 3;
    return new GPUGroupConvexHull({
      id: 'hull',
      positions: view('float32x2', rows),
      labels: view('uint32', rows),
      groupCount,
      prefilterLevels,
      maximumVerticesPerGroup: 64,
      vertexCapacity: 128,
      output: {
        vertexIndices: view('uint32', 128),
        vertexPositions: view('float32x2', 128),
        offsets: view('uint32', groupCount + 1),
        counts: view('uint32', groupCount),
        sizes: view('uint32', groupCount),
        status: {overflow: view('uint32', 1)}
      }
    }).getCommandNodes(graph);
  };
  // Level 0 reads group keys and later levels group offsets: two shapes, however many levels.
  expect(countDistinctPipelines(buildNodes(3))).toBe(countDistinctPipelines(buildNodes(2)));
  expect(countDistinctPipelines(buildNodes(4))).toBe(countDistinctPipelines(buildNodes(3)));
});

it('GPUGroupConvexHull reads its output row capacity from vertexCapacity', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'hull-vertex-capacity'});
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32x2'>(format: Format, length: number) =>
    createTransientView(graph, `hull-capacity-view-${serial++}`, format, length);
  const create = (vertexCapacity: number) =>
    new GPUGroupConvexHull({
      id: 'hull',
      positions: view('float32x2', 8),
      labels: view('uint32', 8),
      groupCount: 1,
      maximumVerticesPerGroup: 8,
      vertexCapacity,
      output: {
        vertexIndices: view('uint32', 16),
        vertexPositions: view('float32x2', 16),
        offsets: view('uint32', 2),
        counts: view('uint32', 1),
        sizes: view('uint32', 1),
        status: {overflow: view('uint32', 1)}
      }
    });
  expect(() => create(16)).not.toThrow();
  expect(() => create(0)).toThrow(/hull vertexCapacity must be an integer/);
});

function buildHullNodes(options: {
  id: string;
  rows: number;
  groupCount: number;
  extremePointFilter?: boolean;
  prefilterLevels?: number;
}) {
  const {rows, groupCount} = options;
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: options.id});
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32x2'>(format: Format, length: number) =>
    createTransientView(graph, `${options.id}-view-${serial++}`, format, length);
  const hull = new GPUGroupConvexHull({
    id: 'hull',
    positions: view('float32x2', rows),
    labels: view('uint32', rows),
    groupCount,
    extremePointFilter: options.extremePointFilter,
    prefilterLevels: options.prefilterLevels,
    maximumVerticesPerGroup: 64,
    vertexCapacity: 128,
    output: {
      vertexIndices: view('uint32', 128),
      vertexPositions: view('float32x2', 128),
      offsets: view('uint32', groupCount + 1),
      counts: view('uint32', groupCount),
      sizes: view('uint32', groupCount),
      status: {overflow: view('uint32', 1)}
    }
  });
  return {hull, nodes: hull.getCommandNodes(graph)};
}

it('GPUGroupConvexHull runs the extreme-point filter between the lattice and the sorts', () => {
  const {hull, nodes} = buildHullNodes({id: 'hull-filter', rows: 600, groupCount: 3});
  expect(hull.extremePointFilter).toBe(true);
  const ids = nodes.map(node => node.id);
  const order = [
    'hull-quantize',
    'hull-fill-extreme-values',
    'hull-extreme-values',
    'hull-fill-extreme-ties',
    'hull-extreme-ties',
    'hull-extreme-points',
    'hull-filter-rows',
    'hull-sort-y',
    'hull-gather-group'
  ].map(prefix => ids.findIndex(id => id.startsWith(prefix)));
  expect(order.every(position => position >= 0)).toBe(true);
  expect([...order].sort((left, right) => left - right)).toEqual(order);

  const off = buildHullNodes({
    id: 'hull-filter-off',
    rows: 600,
    groupCount: 3,
    extremePointFilter: false
  });
  expect(off.hull.extremePointFilter).toBe(false);
  expect(
    off.nodes.some(node => node.id.includes('extreme') || node.id.includes('filter-rows'))
  ).toBe(false);
  // Four filter kernels and two fills are all the filter adds.
  expect(nodes.length - off.nodes.length).toBe(6);
});

it('GPUGroupConvexHull filter kernels parse and stay inside the core limits', () => {
  // 256 groups use workgroup-private tables (8 KB); 257 use global atomics only.
  for (const groupCount of [3, 256, 257]) {
    const {nodes} = buildHullNodes({id: `hull-limits-${groupCount}`, rows: 9000, groupCount});
    const sources = getParsedKernelSources(nodes);
    for (const [id, source] of sources) {
      expect(getStorageBindingCount(source), id).toBeLessThanOrEqual(8);
    }
    const filterIds = [...sources.keys()].filter(id => /^hull-(extreme-|filter-rows)/.test(id));
    expect(filterIds.length).toBe(4);
    for (const id of filterIds) {
      const memory = getWorkgroupMemoryBytes(sources.get(id)!);
      if (/extreme-(values|ties)$/.test(id)) {
        expect(memory, id).toBe(groupCount <= 256 ? groupCount * 8 * 4 : 0);
      } else {
        expect(memory, id).toBe(0);
      }
    }
  }
});

it('GPUGroupConvexHull filter adds no pipelines per prefilter level (rule 12)', () => {
  const count = (prefilterLevels: number) =>
    countDistinctPipelines(
      buildHullNodes({
        id: `hull-rule12-${prefilterLevels}`,
        rows: 600,
        groupCount: 3,
        prefilterLevels
      }).nodes
    );
  expect(count(3)).toBe(count(2));
  expect(count(4)).toBe(count(3));
});

it('GPUGroupConvexHull sizes the extreme-point filter default by group count', () => {
  const limit = 65536;
  expect(
    buildHullNodes({id: 'hull-many', rows: 10, groupCount: limit}).hull.extremePointFilter
  ).toBe(true);
  expect(
    buildHullNodes({id: 'hull-too-many', rows: 10, groupCount: limit + 1}).hull.extremePointFilter
  ).toBe(false);
  expect(() =>
    buildHullNodes({
      id: 'hull-forced',
      rows: 10,
      groupCount: limit + 1,
      extremePointFilter: true
    })
  ).toThrow(/extremePointFilter supports at most 65536 groups/);
});

/** Exact orientation of three lattice points, positive when counter-clockwise. */
function orient(a: readonly number[], b: readonly number[], c: readonly number[]): bigint {
  return BigInt(b[0] - a[0]) * BigInt(c[1] - a[1]) - BigInt(b[1] - a[1]) * BigInt(c[0] - a[0]);
}

/**
 * CPU model of the filter kernels: per group the eight extreme lattice points (ties to the
 * smallest other coordinate), the octagon in the order S, SE, E, NE, N, NW, W, SW, and the rows
 * strictly left of every non-degenerate edge.
 */
function computeDiscardedRows(
  positions: Float32Array,
  keys: readonly number[],
  groupCount: number
): boolean[] {
  const lattice = quantizeToLattice(
    positions,
    keys.map(key => key < groupCount)
  );
  const bias = 2 ** 29;
  const discarded = keys.map(() => false);
  for (let group = 0; group < groupCount; group++) {
    const rows = keys.flatMap((key, row) => (key === group ? [row] : []));
    if (rows.length === 0) {
      continue;
    }
    const points = rows.map(row => [lattice.x[row], lattice.y[row]]);
    for (const [x, y] of points) {
      expect(x + y).toBeLessThan(2 ** 32);
      expect(x - y + bias).toBeGreaterThanOrEqual(0);
      expect(x).toBeLessThanOrEqual(bias);
    }
    const extreme = (
      value: (point: number[]) => number,
      tieBreak: (point: number[]) => number,
      pickMaximum: boolean
    ) => {
      const best = pickMaximum ? Math.max(...points.map(value)) : Math.min(...points.map(value));
      const tie = Math.min(...points.filter(point => value(point) === best).map(tieBreak));
      return {best, tie};
    };
    const minX = extreme(
      point => point[0],
      point => point[1],
      false
    );
    const maxX = extreme(
      point => point[0],
      point => point[1],
      true
    );
    const minY = extreme(
      point => point[1],
      point => point[0],
      false
    );
    const maxY = extreme(
      point => point[1],
      point => point[0],
      true
    );
    const minSum = extreme(
      point => point[0] + point[1],
      point => point[0],
      false
    );
    const maxSum = extreme(
      point => point[0] + point[1],
      point => point[0],
      true
    );
    const minDifference = extreme(
      point => point[0] - point[1],
      point => point[0],
      false
    );
    const maxDifference = extreme(
      point => point[0] - point[1],
      point => point[0],
      true
    );
    const octagon = [
      [minY.tie, minY.best],
      [maxDifference.tie, maxDifference.tie - maxDifference.best],
      [maxX.best, maxX.tie],
      [maxSum.tie, maxSum.best - maxSum.tie],
      [maxY.tie, maxY.best],
      [minDifference.tie, minDifference.tie - minDifference.best],
      [minX.best, minX.tie],
      [minSum.tie, minSum.best - minSum.tie]
    ];
    for (const point of octagon) {
      // Every octagon vertex is a real lattice point of the group.
      expect(points.some(other => other[0] === point[0] && other[1] === point[1])).toBe(true);
    }
    rows.forEach((row, index) => {
      let edgeCount = 0;
      let isInterior = true;
      for (let edge = 0; edge < 8 && isInterior; edge++) {
        const start = octagon[edge];
        const end = octagon[(edge + 1) % 8];
        if (start[0] === end[0] && start[1] === end[1]) {
          continue;
        }
        edgeCount++;
        isInterior = orient(start, end, points[index]) > 0n;
      }
      discarded[row] = isInterior && edgeCount > 0;
    });
  }
  return discarded;
}

type AdversarialSet = {name: string; positions: number[]; labels: number[]; groupCount: number};

function createAdversarialSets(): AdversarialSet[] {
  const random = createRandom(97);
  const sets: AdversarialSet[] = [];
  const add = (
    name: string,
    groupCount: number,
    build: (add: (x: number, y: number, label: number) => void) => void
  ) => {
    const positions: number[] = [];
    const labels: number[] = [];
    build((x, y, label) => {
      positions.push(x, y);
      labels.push(label);
    });
    sets.push({name, positions, labels, groupCount});
  };
  add('collinear diagonal with duplicates', 1, put => {
    for (let index = 0; index < 40; index++) put(index * 3, index * 3, 0);
    for (let index = 0; index < 10; index++) put(index * 6, index * 6, 0);
  });
  add('collinear horizontal and vertical', 2, put => {
    for (let index = 0; index < 30; index++) put(index, 5, 0);
    for (let index = 0; index < 30; index++) put(5, index, 1);
  });
  add('circle of lattice-rounded points', 1, put => {
    for (let index = 0; index < 400; index++) {
      const angle = (index / 400) * Math.PI * 2;
      put(1000 + Math.round(Math.cos(angle) * 4000) / 8, Math.round(Math.sin(angle) * 4000) / 8, 0);
    }
  });
  add('circle with an interior blob', 1, put => {
    for (let index = 0; index < 200; index++) {
      const angle = (index / 200) * Math.PI * 2;
      put(Math.cos(angle) * 100, Math.sin(angle) * 100, 0);
    }
    for (let index = 0; index < 2000; index++) {
      const radius = Math.sqrt(random()) * 99;
      const angle = random() * Math.PI * 2;
      put(Math.cos(angle) * radius, Math.sin(angle) * radius, 0);
    }
  });
  add('one-point, two-point, repeated-point and empty groups', 5, put => {
    put(3, 4, 0);
    put(1, 1, 1);
    put(2, 5, 1);
    for (let index = 0; index < 6; index++) put(7, 7, 2);
    put(NaN, 1, 3);
    put(1, 1, 4);
    put(9, 9, 4);
    put(5, 5, 4);
  });
  add('integer grid with many collinear boundary points', 1, put => {
    for (let x = 0; x < 20; x++) for (let y = 0; y < 15; y++) put(x, y, 0);
  });
  add('huge coordinates', 2, put => {
    for (let index = 0; index < 500; index++) {
      put((random() * 2 - 1) * 1e30, (random() * 2 - 1) * 1e30, 0);
    }
    for (let index = 0; index < 500; index++) {
      put(1e30 + random() * 1e24, -1e30 + random() * 1e24, 1);
    }
  });
  add('lattice limits: points at the extent corners and edges', 1, put => {
    put(0, 0, 0);
    put(1024, 0, 0);
    put(0, 1024, 0);
    put(1024, 1024, 0);
    put(512, 0, 0);
    put(0, 512, 0);
    put(1024, 512, 0);
    put(512, 1024, 0);
    for (let index = 0; index < 300; index++) put(random() * 1024, random() * 1024, 0);
  });
  add('tiny extent', 1, put => {
    for (let index = 0; index < 300; index++) put(random() * 1e-30, random() * 1e-30, 0);
  });
  add('thin sliver', 1, put => {
    for (let index = 0; index < 500; index++) put(random() * 1000, 5 + random() * 1e-3, 0);
  });
  add('diamond with interior and NaN rows', 2, put => {
    put(-100, 0, 0);
    put(100, 0, 0);
    put(0, 100, 0);
    put(0, -100, 0);
    for (let index = 0; index < 500; index++) {
      put((random() * 2 - 1) * 70, (random() * 2 - 1) * 70, 0);
    }
    put(NaN, NaN, 0);
    put(Infinity, 0, 1);
    put(1, 1, 1);
  });
  add('negative offsets and gaussian blobs', 3, put => {
    for (let group = 0; group < 3; group++) {
      for (let index = 0; index < 1500; index++) {
        const radius = Math.sqrt(-2 * Math.log(1 - random())) * 10;
        const angle = random() * Math.PI * 2;
        put(-5e5 + group * 300 + radius * Math.cos(angle), -3e5 + radius * Math.sin(angle), group);
      }
    }
  });
  return sets;
}

it('the octagon filter never removes a hull vertex or a duplicate of one', () => {
  let discardedTotal = 0;
  let rowTotal = 0;
  for (const set of createAdversarialSets()) {
    const positions = Float32Array.from(set.positions);
    const keys = getOracleGroupKeys(positions, set.labels, set.groupCount);
    const discarded = computeDiscardedRows(positions, keys, set.groupCount);
    const lattice = quantizeToLattice(
      positions,
      keys.map(key => key < set.groupCount)
    );
    const expected = computeGroupConvexHullOracle(positions, keys, set.groupCount);
    expected.hulls.forEach((hull, group) => {
      for (const vertexRow of hull) {
        keys.forEach((key, row) => {
          if (
            key === group &&
            lattice.x[row] === lattice.x[vertexRow] &&
            lattice.y[row] === lattice.y[vertexRow]
          ) {
            expect(discarded[row], `${set.name}: row ${row} of group ${group}`).toBe(false);
          }
        });
      }
    });
    // Excluded rows never count.
    keys.forEach((key, row) => {
      if (key >= set.groupCount) {
        expect(discarded[row], set.name).toBe(false);
      }
    });
    discardedTotal += discarded.filter(Boolean).length;
    rowTotal += keys.length;
  }
  // The filter does real work on the blob-like sets.
  expect(discardedTotal).toBeGreaterThan(rowTotal / 4);
});

it('the octagon filter keeps every row of degenerate groups', () => {
  const sets = createAdversarialSets();
  for (const name of [
    'collinear diagonal with duplicates',
    'collinear horizontal and vertical',
    'one-point, two-point, repeated-point and empty groups'
  ]) {
    const set = sets.find(candidate => candidate.name === name)!;
    const positions = Float32Array.from(set.positions);
    const keys = getOracleGroupKeys(positions, set.labels, set.groupCount);
    expect(computeDiscardedRows(positions, keys, set.groupCount).some(Boolean), name).toBe(false);
  }
});

it('the octagon filter discards most of a gaussian blob', () => {
  const set = createAdversarialSets().find(candidate => candidate.name.startsWith('negative'))!;
  const positions = Float32Array.from(set.positions);
  const keys = getOracleGroupKeys(positions, set.labels, set.groupCount);
  const discarded = computeDiscardedRows(positions, keys, set.groupCount);
  const survivors = discarded.filter(flag => !flag).length;
  expect(survivors / keys.length).toBeLessThan(0.1);
});
