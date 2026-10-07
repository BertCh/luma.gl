// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {
  GPUContiguityWeights,
  GPULatticeWeights,
  GPUSpatialLag,
  GPUSpatialWeightsTransform,
  type GPUSpatialWeightsKernel
} from '../../../src/gpu-spatial-analysis/spatial-weights';
import {readFloat32, readUint32} from '../../utils/gpu-contributor-test-utils';
import {WeightsRig} from './spatial-weights-harness';
import {
  assertValidCSR,
  computeContiguityOracle,
  computeLagOracle,
  computeLatticeOracle,
  computeTransformOracle,
  createSeededRandom,
  flattenPolygons,
  isSymmetricPattern,
  type OracleCSR,
  type OraclePolygons
} from './spatial-weights-oracle';

function expectClose(actual: number[], expected: number[], label: string, relative = 2e-5): void {
  expect(actual.length, `${label} length`).toBe(expected.length);
  for (let index = 0; index < expected.length; index++) {
    if (Math.abs(actual[index] - expected[index]) > 1e-6 + relative * Math.abs(expected[index])) {
      throw new Error(`${label}: [${index}] ${actual[index]} != ${expected[index]}`);
    }
  }
}

/** Squares on a grid; every ring is rotated and half are reversed so vertex order is varied. */
function createSquareGrid(columns: number, rows: number, seed: number): OraclePolygons {
  const random = createSeededRandom(seed);
  const polygons: OraclePolygons = [];
  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < columns; x++) {
      let ring: [number, number][] = [
        [x, y],
        [x + 1, y],
        [x + 1, y + 1],
        [x, y + 1]
      ];
      const rotation = Math.floor(random() * 4);
      ring = [...ring.slice(rotation), ...ring.slice(0, rotation)];
      if (random() < 0.5) ring.reverse();
      polygons.push([ring]);
    }
  }
  return polygons;
}

/** Random axis-aligned integer rectangles: overlapping, T-junctions and partial shared edges. */
function createRandomRectangles(count: number, seed: number): OraclePolygons {
  const random = createSeededRandom(seed);
  const polygons: OraclePolygons = [];
  for (let index = 0; index < count; index++) {
    const x = Math.floor(random() * 16);
    const y = Math.floor(random() * 16);
    const width = 1 + Math.floor(random() * 3);
    const height = 1 + Math.floor(random() * 3);
    polygons.push([
      [
        [x, y],
        [x + width, y],
        [x + width, y + height],
        [x, y + height]
      ]
    ]);
  }
  return polygons;
}

async function runContiguity(
  device: NonNullable<Awaited<ReturnType<typeof getWebGPUTestDevice>>>,
  polygons: OraclePolygons,
  criterion: 'queen' | 'rook',
  options: {snapTolerance?: number; capacity: number; pairCapacity?: number}
): Promise<{csr: OracleCSR; overflow: number; total: number}> {
  const layout = flattenPolygons(polygons);
  const rig = new WeightsRig(device);
  const output = rig.weightsOutput(polygons.length, options.capacity);
  const overflow = rig.output('uint32', 1);
  const total = rig.output('uint32', 1);
  rig.run(
    new GPUContiguityWeights({
      criterion,
      positions: rig.input(layout.positions, 'float32x2', layout.positions.length / 2),
      ringOffsets: rig.input(layout.ringOffsets, 'uint32', layout.ringOffsets.length),
      polygonOffsets: rig.input(layout.polygonOffsets, 'uint32', layout.polygonOffsets.length),
      snapTolerance: options.snapTolerance,
      pairCapacity: options.pairCapacity,
      weights: output.spatialWeights,
      overflow: overflow.view,
      totalNeighbors: total.view
    })
  );
  const result = {
    csr: await output.read(),
    overflow: (await readUint32(overflow.buffer, 1))[0],
    total: (await readUint32(total.buffer, 1))[0]
  };
  rig.destroy();
  return result;
}

it('GPUContiguityWeights matches the oracle on a square grid for queen and rook', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const polygons = createSquareGrid(7, 5, 3);
  for (const criterion of ['queen', 'rook'] as const) {
    const expected = computeContiguityOracle(polygons, criterion);
    // Sanity: the interior cell of a 7x5 grid has 8 (queen) or 4 (rook) neighbors.
    const interior = 2 * 7 + 3;
    expect(expected.offsets[interior + 1] - expected.offsets[interior]).toBe(
      criterion === 'queen' ? 8 : 4
    );
    const {csr, overflow, total} = await runContiguity(device, polygons, criterion, {
      capacity: expected.neighbors.length + 16
    });
    assertValidCSR(csr, polygons.length, criterion);
    expect(csr.offsets, `${criterion} offsets`).toEqual(expected.offsets);
    expect(csr.neighbors, `${criterion} neighbors`).toEqual(expected.neighbors);
    expect(csr.weights.every(weight => weight === 1)).toBe(true);
    expect(overflow).toBe(0);
    expect(total).toBe(expected.neighbors.length);
    expect(isSymmetricPattern(csr)).toBe(true);
  }
});

it('GPUContiguityWeights sorts narrow ID keys correctly at polygon counts around powers of two', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  // The pair and edge sorts compare only getSortKeyBits(count) bits; the invalid key then wraps to
  // the largest key value, so the largest valid ID must stay below it.
  for (const [columns, rows] of [
    [3, 5],
    [4, 4],
    [31, 1],
    [32, 1],
    [1, 33]
  ] as const) {
    const polygons = createSquareGrid(columns, rows, columns * 100 + rows);
    for (const criterion of ['queen', 'rook'] as const) {
      const expected = computeContiguityOracle(polygons, criterion);
      const {csr, overflow} = await runContiguity(device, polygons, criterion, {
        capacity: expected.neighbors.length + 4
      });
      const label = `${columns}x${rows} ${criterion}`;
      assertValidCSR(csr, polygons.length, label);
      expect(csr.offsets, `${label} offsets`).toEqual(expected.offsets);
      expect(csr.neighbors, `${label} neighbors`).toEqual(expected.neighbors);
      expect(overflow).toBe(0);
    }
  }
});

it('GPUContiguityWeights matches the oracle on random rectangles (T-junctions, overlaps)', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  for (const seed of [1, 2, 3]) {
    const polygons = createRandomRectangles(80, seed);
    for (const criterion of ['queen', 'rook'] as const) {
      const expected = computeContiguityOracle(polygons, criterion);
      expect(expected.neighbors.length).toBeGreaterThan(0);
      const {csr} = await runContiguity(device, polygons, criterion, {
        capacity: expected.neighbors.length + 8,
        pairCapacity: 20000
      });
      assertValidCSR(csr, polygons.length, `${criterion} seed ${seed}`);
      expect(csr.offsets, `${criterion} ${seed} offsets`).toEqual(expected.offsets);
      expect(csr.neighbors, `${criterion} ${seed} neighbors`).toEqual(expected.neighbors);
    }
  }
});

it('GPUContiguityWeights handles holes, corner contact and non-finite vertices', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const polygons: OraclePolygons = [
    // 0: square with a hole whose edge is shared with polygon 1
    [
      [
        [0, 0],
        [6, 0],
        [6, 6],
        [0, 6]
      ],
      [
        [2, 2],
        [4, 2],
        [4, 4],
        [2, 4]
      ]
    ],
    // 1: fills the hole (shares the whole hole ring)
    [
      [
        [2, 2],
        [4, 2],
        [4, 4],
        [2, 4]
      ]
    ],
    // 2: touches polygon 0 at the corner (6, 6) only
    [
      [
        [6, 6],
        [8, 6],
        [8, 8],
        [6, 8]
      ]
    ],
    // 3: isolated, with a NaN vertex that must be ignored
    [
      [
        [20, 20],
        [21, 20],
        [Number.NaN, 21],
        [20, 21]
      ]
    ]
  ];
  for (const criterion of ['queen', 'rook'] as const) {
    const expected = computeContiguityOracle(polygons, criterion);
    const {csr} = await runContiguity(device, polygons, criterion, {capacity: 32});
    assertValidCSR(csr, polygons.length, criterion);
    expect(csr.offsets).toEqual(expected.offsets);
    expect(csr.neighbors).toEqual(expected.neighbors);
  }
  const queen = computeContiguityOracle(polygons, 'queen');
  const rook = computeContiguityOracle(polygons, 'rook');
  expect(queen.neighbors.slice(queen.offsets[2], queen.offsets[3])).toEqual([0]);
  expect(rook.neighbors.slice(rook.offsets[2], rook.offsets[3])).toEqual([]);
});

it('GPUContiguityWeights snapTolerance joins near-coincident vertices', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const polygons: OraclePolygons = [
    [
      [
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1]
      ]
    ],
    [
      [
        [1.0004, 0.0003],
        [2, 0],
        [2, 1],
        [1.0004, 0.9997]
      ]
    ]
  ];
  const exact = await runContiguity(device, polygons, 'rook', {capacity: 8});
  expect(exact.csr.neighbors).toEqual([]);
  const snapped = await runContiguity(device, polygons, 'rook', {capacity: 8, snapTolerance: 0.01});
  expect(snapped.csr.neighbors).toEqual([1, 0]);
  expect(snapped.csr.neighbors).toEqual(computeContiguityOracle(polygons, 'rook', 0.01).neighbors);
});

it('GPUContiguityWeights reports capacity and pair overflow with consistent rows', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const polygons = createSquareGrid(6, 6, 9);
  const expected = computeContiguityOracle(polygons, 'queen');
  const capacity = Math.floor(expected.neighbors.length / 2);
  const small = await runContiguity(device, polygons, 'queen', {capacity});
  expect(small.overflow).toBe(1);
  expect(small.total).toBe(expected.neighbors.length);
  expect(small.csr.offsets).toEqual(expected.offsets.map(offset => Math.min(offset, capacity)));
  expect(small.csr.neighbors).toEqual(expected.neighbors.slice(0, capacity));
  const pairs = await runContiguity(device, polygons, 'queen', {
    capacity: expected.neighbors.length,
    pairCapacity: 10
  });
  expect(pairs.overflow).toBe(1);
  for (let row = 0; row < polygons.length; row++) {
    expect(pairs.csr.offsets[row + 1]).toBeGreaterThanOrEqual(pairs.csr.offsets[row]);
  }
});

it('GPULatticeWeights matches the oracle for rook/queen, radius, mask and distances', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const width = 9;
  const height = 6;
  const random = createSeededRandom(11);
  const mask = new Uint32Array(width * height).map(() => (random() < 0.8 ? 1 : 0));
  const cases = [
    {criterion: 'rook', radius: 1},
    {criterion: 'queen', radius: 1},
    {criterion: 'rook', radius: 2},
    {criterion: 'queen', radius: 3, cellSize: [2, 0.5] as [number, number]},
    {criterion: 'queen', radius: 1, mask},
    {criterion: 'rook', radius: 2, mask}
  ] as const;
  for (const options of cases) {
    const expected = computeLatticeOracle({width, height, ...options} as never);
    const rig = new WeightsRig(device);
    const output = rig.weightsOutput(width * height, expected.neighbors.length + 5, true);
    const overflow = rig.output('uint32', 1);
    rig.run(
      new GPULatticeWeights({
        width,
        height,
        criterion: options.criterion,
        radius: options.radius,
        mask: 'mask' in options ? rig.input(options.mask, 'uint32', width * height) : undefined,
        cellSize: 'cellSize' in options ? options.cellSize : undefined,
        weights: output.spatialWeights,
        overflow: overflow.view
      })
    );
    const csr = await output.read();
    const label = JSON.stringify({...options, mask: 'mask' in options});
    assertValidCSR(csr, width * height, label);
    expect(csr.offsets, label).toEqual(expected.offsets);
    expect(csr.neighbors, label).toEqual(expected.neighbors);
    expectClose(csr.distances, expected.distances, `${label} distances`);
    expect((await readUint32(overflow.buffer, 1))[0]).toBe(0);
    rig.destroy();
  }
  // lat2W sanity: interior rook = 4, queen = 8, corner rook = 2.
  const rook = computeLatticeOracle({width, height, criterion: 'rook'});
  expect(rook.offsets[1] - rook.offsets[0]).toBe(2);
  expect(rook.offsets[width + 2] - rook.offsets[width + 1]).toBe(4);
});

it('GPULatticeWeights flags capacity overflow and clamps offsets', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const expected = computeLatticeOracle({width: 5, height: 5, criterion: 'queen'});
  const capacity = 30;
  const rig = new WeightsRig(device);
  const output = rig.weightsOutput(25, capacity);
  const overflow = rig.output('uint32', 1);
  const total = rig.output('uint32', 1);
  rig.run(
    new GPULatticeWeights({
      width: 5,
      height: 5,
      criterion: 'queen',
      weights: output.spatialWeights,
      overflow: overflow.view,
      totalNeighbors: total.view
    })
  );
  const csr = await output.read();
  expect((await readUint32(overflow.buffer, 1))[0]).toBe(1);
  expect((await readUint32(total.buffer, 1))[0]).toBe(expected.neighbors.length);
  expect(csr.offsets).toEqual(expected.offsets.map(offset => Math.min(offset, capacity)));
  rig.destroy();
});

const KERNELS: GPUSpatialWeightsKernel[] = [
  'gaussian',
  'triangular',
  'epanechnikov',
  'bisquare',
  'uniform'
];

it('GPUSpatialWeightsTransform row, binary, kernel and symmetrize match the oracle', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const random = createSeededRandom(5);
  const base = computeLatticeOracle({width: 8, height: 6, criterion: 'queen', radius: 2});
  const weighted: OracleCSR = {
    ...base,
    weights: base.weights.map(() => (random() < 0.15 ? 0 : 0.25 + random() * 4))
  };
  for (const outOfPlace of [true, false]) {
    for (const operation of ['row', 'binary', 'symmetrize'] as const) {
      if (operation === 'symmetrize' && !outOfPlace) continue;
      const rig = new WeightsRig(device);
      const upload = rig.uploadWeights(weighted, 3);
      const output = rig.output('float32', upload.neighbors.length);
      const target = outOfPlace ? output : undefined;
      rig.run(new GPUSpatialWeightsTransform({operation, weights: upload, output: target?.view}));
      const readBuffer = target ? target.buffer : rig.bufferOf(upload.weights);
      const actual = await readFloat32(readBuffer, weighted.neighbors.length);
      const expected = computeTransformOracle(weighted, operation);
      expectClose(actual, expected, `${operation} outOfPlace=${outOfPlace}`);
      if (operation === 'symmetrize') {
        expect(isSymmetricPattern(weighted)).toBe(true);
        // Result is symmetric: w'_ij == w'_ji.
        const symmetric = {...weighted, weights: actual};
        expectClose(computeTransformOracle(symmetric, 'symmetrize'), actual, 'idempotent');
      }
      expect(actual.some(value => value !== 0)).toBe(true);
      rig.destroy();
    }
  }
  for (const kernel of KERNELS) {
    for (const bandwidth of ['adaptive', 2.5] as const) {
      const rig = new WeightsRig(device);
      const upload = rig.uploadWeights(base, 3);
      const output = rig.output('float32', upload.neighbors.length);
      rig.run(
        new GPUSpatialWeightsTransform({
          operation: 'kernel',
          kernel,
          bandwidth,
          weights: upload,
          output: output.view
        })
      );
      const actual = await readFloat32(output.buffer, base.neighbors.length);
      const expected = computeTransformOracle(base, 'kernel', {kernel, bandwidth});
      expectClose(actual, expected, `${kernel} ${bandwidth}`, 1e-4);
      expect(actual.some(value => value > 0)).toBe(true);
      rig.destroy();
    }
  }
});

it('GPUSpatialLag matches the oracle, with mask and normalization', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const random = createSeededRandom(21);
  const csr = computeLatticeOracle({width: 10, height: 8, criterion: 'queen'});
  const rows = 80;
  const values = new Float32Array(rows).map(() => random() * 10 - 3);
  const mask = new Uint32Array(rows).map(() => (random() < 0.75 ? 1 : 0));
  for (const options of [
    {mask: false, normalize: false},
    {mask: true, normalize: false},
    {mask: false, normalize: true},
    {mask: true, normalize: true}
  ]) {
    const rig = new WeightsRig(device);
    const weights = rig.uploadWeights(csr);
    const output = rig.output('float32', rows);
    rig.run(
      new GPUSpatialLag({
        values: rig.input(values, 'float32', rows),
        weights,
        mask: options.mask ? rig.input(mask, 'uint32', rows) : undefined,
        normalize: options.normalize,
        output: output.view
      })
    );
    const actual = await readFloat32(output.buffer, rows);
    const expected = computeLagOracle(
      csr,
      values,
      options.mask ? mask : undefined,
      options.normalize
    );
    expectClose(actual, expected, JSON.stringify(options), 1e-4);
    expect(actual.some(value => value !== 0)).toBe(true);
    rig.destroy();
  }
});

it('contiguity -> transform -> lag chains in one graph', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const polygons = createSquareGrid(6, 4, 17);
  const layout = flattenPolygons(polygons);
  const expected = computeContiguityOracle(polygons, 'rook');
  const rows = polygons.length;
  const values = new Float32Array(rows).map((_, index) => (index * 7) % 5);
  const rig = new WeightsRig(device);
  const output = rig.weightsOutput(rows, expected.neighbors.length + 4);
  const lag = rig.output('float32', rows);
  rig.run(
    new GPUContiguityWeights({
      criterion: 'rook',
      positions: rig.input(layout.positions, 'float32x2', layout.positions.length / 2),
      ringOffsets: rig.input(layout.ringOffsets, 'uint32', layout.ringOffsets.length),
      polygonOffsets: rig.input(layout.polygonOffsets, 'uint32', layout.polygonOffsets.length),
      weights: output.spatialWeights,
      overflow: rig.output('uint32', 1).view
    }),
    new GPUSpatialWeightsTransform({operation: 'row', weights: output.spatialWeights}),
    new GPUSpatialLag({
      values: rig.input(values, 'float32', rows),
      weights: output.spatialWeights,
      output: lag.view
    })
  );
  const csr = await output.read();
  const standardized = computeTransformOracle(expected, 'row');
  expectClose(csr.weights, standardized, 'row standardized');
  expectClose(
    await readFloat32(lag.buffer, rows),
    computeLagOracle({...expected, weights: standardized}, values),
    'lag',
    1e-4
  );
  rig.destroy();
});
