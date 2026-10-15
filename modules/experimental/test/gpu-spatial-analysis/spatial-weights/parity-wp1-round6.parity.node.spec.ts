// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {inflateSync} from 'node:zlib';
import {expect, it} from 'vitest';
import {decodeRound4Fixture, type PackedFixture} from '../../oracles/parity-wp1-round4-fixture';
import {
  sumSparseRows,
  type AlgebraFixture,
  type ForestCase,
  type SparseMatrix,
  type SparseProvenance,
  type TransposeCase
} from '../../oracles/parity-wp1-round6-fixture';

const directory = new URL('../../oracles/fixtures/', import.meta.url);
const sourceDirectory = new URL(
  '../../../../../examples/deck/spatial-analysis-showcase/public/data/',
  import.meta.url
);
function loadFixture(topic: string): PackedFixture {
  return JSON.parse(readFileSync(new URL(`parity-wp1-round6-${topic}.json`, directory), 'utf8'));
}
function hashBytes(bytes: Uint8Array): string {
  return createHash('sha256').update(bytes).digest('hex');
}
function hashFloats(...arrays: number[][]): string {
  return hashBytes(
    Buffer.concat(arrays.map(values => Buffer.from(Float32Array.from(values).buffer)))
  );
}
function checkMatrix(matrix: SparseMatrix, rows: number, columns: number, unique: boolean) {
  expect(matrix.offsets).toHaveLength(rows + 1);
  expect(matrix.offsets[0]).toBe(0);
  expect(matrix.offsets.at(-1)).toBe(matrix.neighbors.length);
  expect(matrix.weights).toHaveLength(matrix.neighbors.length);
  expect(matrix.weights.every(Number.isFinite)).toBe(true);
  expect(matrix.neighbors.every(neighbor => neighbor >= 0 && neighbor < columns)).toBe(true);
  for (let row = 0; row < rows; row++) {
    expect(matrix.offsets[row + 1]).toBeGreaterThanOrEqual(matrix.offsets[row]);
    if (unique) {
      const neighbors = matrix.neighbors.slice(matrix.offsets[row], matrix.offsets[row + 1]);
      expect(neighbors).toEqual([...new Set(neighbors)].sort((first, second) => first - second));
    }
  }
}
for (const topic of ['algebra', 'transpose', 'forest']) {
  it(`Round 6 ${topic} fixture provenance and lossless transport`, async () => {
    const bytes = readFileSync(new URL(`parity-wp1-round6-${topic}.json`, directory));
    expect(bytes.length).toBeLessThanOrEqual(1500000);
    const wrapper = loadFixture(topic);
    expect(wrapper.generator).toBe('parity-wp1-round6.py');
    for (const library of ['python', 'numpy', 'scipy', 'shapely', 'pyproj', 'libpysal'])
      expect(wrapper.versions[library]).toBeTruthy();
    expect(hashBytes(inflateSync(Buffer.from(wrapper.data, 'base64')))).toBe(wrapper.sha256);
    const payload = await decodeRound4Fixture<{
      datasets?: SparseProvenance[];
      cases: SparseProvenance[];
    }>(wrapper);
    for (const testCase of payload.datasets ?? payload.cases) {
      for (const [name, hash] of Object.entries(testCase.provenance.sourceSha256 ?? {})) {
        expect(hashBytes(readFileSync(new URL(name, sourceDirectory))), name).toBe(hash);
      }
    }
  });
}

it('Algebra retains complete tracts and all CSR outputs for every operation', async () => {
  const {datasets, cases} = await decodeRound4Fixture<AlgebraFixture>(loadFixture('algebra'));
  expect(cases).toHaveLength(48);
  expect(datasets[0].n).toBe(791);
  expect(datasets[0].left.neighbors).toHaveLength(5206);
  expect(datasets[0].right.neighbors).toHaveLength(3164);
  for (const dataset of datasets) {
    expect(hashFloats(dataset.left.weights, dataset.right.weights)).toBe(
      dataset.inputFloat32Sha256
    );
    checkMatrix(dataset.left, dataset.n, dataset.n, true);
    checkMatrix(dataset.right, dataset.n, dataset.n, true);
  }
  for (const testCase of cases) {
    checkMatrix(testCase.expected, testCase.n, testCase.n, true);
    expect(sumSparseRows(testCase.expected)).toEqual(testCase.rowSums);
    if (testCase.operation === 'higherOrder')
      expect(testCase.workCapacity).toBeGreaterThan(testCase.expected.neighbors.length);
  }
});

it('Algebra distinguishes structural zeros, replaces self weights and retains empty rows', async () => {
  const {cases} = await decodeRound4Fixture<AlgebraFixture>(loadFixture('algebra'));
  const small = cases.filter(testCase => testCase.datasetIndex === 1);
  const intersection = small.find(testCase => testCase.name.endsWith('/intersection-left'))!;
  expect(intersection.expected.neighbors).toEqual([1, 0, 1]);
  expect(intersection.expected.weights).toEqual([0, 1, 4]);
  const selfWeight = small.find(testCase => testCase.operation === 'selfWeight')!;
  expect(selfWeight.expected.offsets).toEqual([0, 3, 6, 8, 10, 11, 12]);
  expect(selfWeight.expected.weights[9]).toBe(0.5); // replace row 3's old self weight 2
  const empty = small.find(
    testCase => testCase.operation === 'higherOrder' && testCase.order === 3 && !testCase.cumulative
  )!;
  expect(empty.expected.offsets).toEqual([0, 0, 0, 0, 0, 0, 0]);
});

it('Transpose retains all assignments, discards only invalid tract IDs and preserves duplicate slots', async () => {
  const {cases} = await decodeRound4Fixture<{cases: TransposeCase[]}>(loadFixture('transpose'));
  expect(cases).toHaveLength(5);
  for (const testCase of cases) {
    expect(hashFloats(testCase.input.weights, testCase.input.distances)).toBe(
      testCase.inputFloat32Sha256
    );
    checkMatrix(testCase.expected, testCase.columnCount, testCase.n, false);
    expect(sumSparseRows(testCase.expected)).toEqual(testCase.rowSums);
  }
  const places = cases[2];
  expect(places.n).toBe(105808);
  expect(places.input.neighbors).toHaveLength(105808);
  expect(places.expected.neighbors).toHaveLength(105553);
  expect(places.input.neighbors.filter(neighbor => neighbor >= 791)).toHaveLength(255);
  expect(new Set(places.expected.neighbors).size).toBe(105553);
  expect(cases[3].expected).toEqual({
    offsets: [0, 2, 4, 6, 6],
    neighbors: [0, 3, 2, 2, 0, 0],
    weights: [0, 6, 4, 1, 2, 3],
    distances: [0.5, 1.75, 1.25, 1.5, 0.25, 0.75]
  });
  expect(cases[4].asymmetricSlots).toBe(10);
});

it('Transpose symmetry diagnostics are applicable only to square decoded fixtures', async () => {
  const {cases} = await decodeRound4Fixture<{cases: TransposeCase[]}>(loadFixture('transpose'));
  expect(cases.filter(testCase => Number.isFinite(testCase.asymmetricSlots))).toHaveLength(2);
  for (const testCase of cases) {
    expect(Number.isFinite(testCase.asymmetricSlots), testCase.name).toBe(
      testCase.n === testCase.columnCount
    );
    if (testCase.n !== testCase.columnCount) expect(testCase.asymmetricSlots).toBeNaN();
  }
});

it('Forests retain every slot cost, disconnected component, standardization and exact distinct-cost edge', async () => {
  const {cases} = await decodeRound4Fixture<{cases: ForestCase[]}>(loadFixture('forest'));
  expect(cases).toHaveLength(4);
  for (const testCase of cases) {
    expect(hashFloats(testCase.input.weights, testCase.values)).toBe(testCase.inputFloat32Sha256);
    checkMatrix(testCase.input, testCase.n, testCase.n, true);
    expect(testCase.expected.slotCosts).toHaveLength(testCase.input.neighbors.length);
    expect(testCase.expected.componentLabels).toHaveLength(testCase.n);
    expect(new Set(testCase.expected.componentLabels).size).toBe(testCase.expected.componentCount);
    expect(testCase.expected.edgeCount).toBe(testCase.n - testCase.expected.componentCount);
    expect(testCase.expected.flags.filter(Boolean)).toHaveLength(testCase.expected.edgeCount);
    const cost = testCase.expected.flags.reduce(
      (sum, flag, slot) => sum + flag * testCase.expected.slotCosts[slot],
      0
    );
    expect(cost).toBeCloseTo(testCase.expected.totalCost, 6);
  }
  expect(cases[0].n).toBe(791);
  expect(cases[0].expected.componentCount).toBe(2);
  expect(cases[2].exactEdges).toBe(true);
  expect(cases[2].expected.componentLabels).toEqual([0, 0, 0, 0, 4, 4, 6]);
  expect(cases[2].expected.totalCost).toBe(1 + 9 + 36 + 625);
  expect(cases[3].expected.slotCosts.filter(cost => cost === 0).length).toBeGreaterThan(0);
  expect(cases[3].expected.standardizedValues.filter((_, index) => index % 2 === 1)).toEqual([
    0, 0, 0, 0, 0, 0, 0
  ]);
});
