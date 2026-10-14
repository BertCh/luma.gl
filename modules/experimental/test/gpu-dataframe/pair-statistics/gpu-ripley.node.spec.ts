// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  GPURipley,
  type GPURipleyOutput,
  type GPURipleyProps
} from '../../../src/gpu-dataframe/pair-statistics/gpu-ripley';
import {
  getGPURipleyParameterValues,
  GPU_RIPLEY_EDGE_CORRECTION,
  GPU_RIPLEY_PARAMETER_LENGTH
} from '../../../src/gpu-dataframe/pair-statistics/ripley-parameters';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {createRandom} from './pair-statistics-oracle';
import {computeRipleyOnCPU, getIsotropicWeightOnCPU} from './ripley-oracle';

let serial = 0;

type Overrides = Omit<Partial<GPURipleyProps>, 'output'> & {output?: Partial<GPURipleyOutput>};

function createProps(graph: GPUCommandGraph, overrides: Overrides = {}): GPURipleyProps {
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `view-${serial++}`, format, length);
  return {
    positions: view('float32x2', 10),
    parameters: view('float32', GPU_RIPLEY_PARAMETER_LENGTH),
    gridSize: [4, 4],
    radiusCount: 8,
    ...overrides,
    output: {k: view('float32', 8), ...overrides.output}
  };
}

function expectThrows(overrides: (graph: GPUCommandGraph) => Overrides, message: RegExp): void {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  expect(() => new GPURipley(createProps(graph, overrides(graph)))).toThrow(message);
  device.destroy();
}

it('getGPURipleyParameterValues packs and validates the layout', () => {
  expect(
    Array.from(getGPURipleyParameterValues({bounds: [0, 1, 2, 3], maximumDistance: 4}))
  ).toEqual([0, 1, 2, 3, 4, GPU_RIPLEY_EDGE_CORRECTION.isotropic, 0, 0]);
  for (const mode of ['none', 'border', 'isotropic'] as const) {
    expect(
      getGPURipleyParameterValues({
        bounds: [0, 0, 1, 1],
        maximumDistance: 1,
        edgeCorrection: mode
      })[5]
    ).toBe(GPU_RIPLEY_EDGE_CORRECTION[mode]);
  }
  const bounds = [0, 0, 1, 1] as const;
  expect(() => getGPURipleyParameterValues({bounds, maximumDistance: 0})).toThrow(/positive/);
  expect(() => getGPURipleyParameterValues({bounds, maximumDistance: NaN})).toThrow(/finite/);
  expect(() => getGPURipleyParameterValues({bounds: [1, 0, 0, 1], maximumDistance: 1})).toThrow(
    /minX/
  );
  expect(() =>
    getGPURipleyParameterValues({bounds, maximumDistance: 1, edgeCorrection: 'toroidal' as never})
  ).toThrow(/edgeCorrection/);
  expect(() =>
    getGPURipleyParameterValues({bounds, maximumDistance: 1}, new Float32Array(4))
  ).toThrow(/8/);
});

it('GPURipley validates its inputs and outputs', () => {
  const view = <Format extends 'float32' | 'uint32'>(
    graph: GPUCommandGraph,
    format: Format,
    length: number
  ) => createTransientView(graph, `bad-${serial++}`, format, length);
  expectThrows(() => ({radiusCount: 0}), /radiusCount/);
  expectThrows(() => ({radiusCount: 257}), /radiusCount/);
  expectThrows(() => ({radiusCount: 2.5}), /radiusCount/);
  expectThrows(graph => ({output: {k: view(graph, 'float32', 7)}}), /output\.k must hold/);
  expectThrows(graph => ({output: {l: view(graph, 'float32', 7)}}), /output\.l must hold/);
  expectThrows(graph => ({output: {lMinusR: view(graph, 'float32', 7)}}), /lMinusR/);
  expectThrows(
    graph => ({output: {pairCorrelation: view(graph, 'float32', 7)}}),
    /pairCorrelation/
  );
  expectThrows(graph => ({output: {radii: view(graph, 'float32', 7)}}), /radii/);
  expectThrows(graph => ({output: {pairCounts: view(graph, 'uint32', 7)}}), /pairCounts/);
  expectThrows(graph => ({output: {k: view(graph, 'uint32', 8) as never}}), /k/);
  expectThrows(graph => ({parameters: view(graph, 'float32', 4)}), /parameters/);
  expectThrows(graph => ({mask: view(graph, 'uint32', 9)}), /mask length/);
  expectThrows(() => ({gridSize: [0, 4]}), /gridSize/);
  expectThrows(graph => {
    const shared = view(graph, 'float32', 8);
    return {parameters: shared, output: {k: shared}};
  }, /alias|overlap|disjoint|input|share/i);
});

it('GPURipley rejects views from another graph', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const other = new GPUCommandGraph(device);
  const contributor = new GPURipley(
    createProps(graph, {output: {k: createTransientView(other, 'foreign-k', 'float32', 8)}})
  );
  expect(() => contributor.getCommandNodes(graph)).toThrow(/target graph/);
  device.destroy();
});

it('GPURipley emits deterministic node IDs and only the finishing nodes it needs', () => {
  const device = createNullWebGPUDevice();
  const ids = (withDerived: boolean) => {
    const graph = new GPUCommandGraph(device);
    const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
      createTransientView(graph, `r-${serial++}`, format, length);
    const contributor = new GPURipley({
      ...createProps(graph),
      id: 'r',
      output: {
        ...createProps(graph).output,
        pairCounts: view('uint32', 8),
        ...(withDerived
          ? {l: view('float32', 8), pairCorrelation: view('float32', 8), radii: view('float32', 8)}
          : {})
      }
    });
    return contributor.getCommandNodes(graph).map(node => node.id);
  };
  const full = ids(true);
  expect(ids(true)).toEqual(full);
  expect(full.every(id => id.startsWith('r-'))).toBe(true);
  expect(full).toContain('r-histogram-pairs');
  expect(full).toContain('r-finish');
  expect(full).toContain('r-derive');
  expect(ids(false)).not.toContain('r-derive');
  device.destroy();
});

it('getIsotropicWeightOnCPU matches hand-computed circle fractions', () => {
  // Interior circle: weight 1.
  expect(getIsotropicWeightOnCPU(5, 5, 5, 5, 3)).toBe(1);
  // Point (1, 1) of the [0, 10]^2 window, radius 2: each near side cuts arc 2 acos(1/2) = 2 pi / 3,
  // the corner (1, 1) is inside the circle so the arcs overlap by pi / 6: outside = 7 pi / 6 of
  // 2 pi, the fraction inside is 5 / 12 and the weight 12 / 5.
  expect(getIsotropicWeightOnCPU(1, 1, 9, 9, 2)).toBeCloseTo(2.4, 12);
  // Radius 1.2: corner distance sqrt(2) > 1.2, so the arcs are disjoint: fraction 1 - 2 acos(1/1.2) / pi.
  const fraction = 1 - (2 * Math.acos(1 / 1.2)) / Math.PI;
  expect(getIsotropicWeightOnCPU(1, 1, 9, 9, 1.2)).toBeCloseTo(1 / fraction, 12);
  // A point on one side sees half of every circle; in a corner a quarter.
  expect(getIsotropicWeightOnCPU(0, 5, 10, 5, 1)).toBeCloseTo(2, 12);
  expect(getIsotropicWeightOnCPU(0, 0, 10, 10, 3)).toBeCloseTo(4, 12);
  // A circle that swallows the window is capped.
  expect(getIsotropicWeightOnCPU(0.5, 0.5, 0.5, 0.5, 5)).toBe(100);
});

it('getIsotropicWeightOnCPU agrees with a numeric integration of the circle fraction', () => {
  const random = createRandom(31);
  const width = 12;
  const height = 7;
  for (let trial = 0; trial < 60; trial++) {
    const x = random() * width;
    const y = random() * height;
    const distance = 0.2 + random() * 6;
    const samples = 40000;
    let inside = 0;
    for (let sample = 0; sample < samples; sample++) {
      const angle = ((sample + 0.5) / samples) * 2 * Math.PI;
      const pointX = x + distance * Math.cos(angle);
      const pointY = y + distance * Math.sin(angle);
      if (pointX >= 0 && pointX <= width && pointY >= 0 && pointY <= height) {
        inside++;
      }
    }
    const expected = Math.min(samples / Math.max(inside, samples / 100), 100);
    const weight = getIsotropicWeightOnCPU(x, y, width - x, height - y, distance);
    expect(Math.abs(weight - expected), `trial ${trial}`).toBeLessThan(2e-3 * expected + 1e-3);
  }
});

it('computeRipleyOnCPU matches a hand-computed three-point pattern', () => {
  // A = 100, n = 3, one close pair (d = 1, both orders) and a far point; radii 1..5.
  const positions = new Float32Array([1, 1, 2, 1, 9, 9]);
  const scene = {positions};
  const frame = {bounds: [0, 0, 10, 10] as const, maximumDistance: 5};
  const none = computeRipleyOnCPU(scene, {...frame, edgeCorrection: 'none'}, 5);
  expect(none.n).toBe(3);
  expect(none.area).toBe(100);
  expect(none.radii).toEqual([1, 2, 3, 4, 5]);
  expect(none.pairCounts).toEqual([2, 0, 0, 0, 0]);
  for (const value of none.k) {
    expect(value).toBeCloseTo((100 / 6) * 2, 12);
  }
  expect(none.l[0]).toBeCloseTo(Math.sqrt(((100 / 6) * 2) / Math.PI), 12);
  expect(none.lMinusR[0]).toBeCloseTo(none.l[0] - 1, 12);
  // g_0 = K(1) / (pi * 1), g_1 = (K(2) - K(1)) / (pi * 3) = 0.
  expect(none.pairCorrelation[0]).toBeCloseTo(((100 / 6) * 2) / Math.PI, 12);
  expect(none.pairCorrelation[1]).toBeCloseTo(0, 12);

  // Border: every point is exactly 1 from the boundary, so only r_0 = 1 has a border set. With
  // spatstat's lambda = n / A the estimate is A * 2 / (n * 3) = 100 * 2 / 9 there, which is
  // (n - 1) / n of the uncorrected K; larger radii are undefined.
  const border = computeRipleyOnCPU(scene, {...frame, edgeCorrection: 'border'}, 5);
  expect(border.k[0]).toBeCloseTo((100 * 2) / 9, 12);
  expect(border.k.slice(1).every(Number.isNaN)).toBe(true);

  // Isotropic: the d = 1 pair lies at distance exactly 1 from the left and bottom sides, so the
  // circle just touches them and the weight is 1.
  const isotropic = computeRipleyOnCPU(scene, {...frame, edgeCorrection: 'isotropic'}, 5);
  expect(isotropic.k[0]).toBeCloseTo((100 / 6) * 2, 5);
});

it('computeRipleyOnCPU returns NaN for fewer than two points', () => {
  const positions = new Float32Array([1, 1, 50, 50]);
  const result = computeRipleyOnCPU(
    {positions},
    {bounds: [0, 0, 10, 10], maximumDistance: 5, edgeCorrection: 'none'},
    4
  );
  expect(result.n).toBe(1);
  expect(result.k.every(Number.isNaN)).toBe(true);
});

it('computeRipleyOnCPU separates clustered, random and regular patterns by L(r) - r', () => {
  const random = createRandom(77);
  const side = 100;
  const count = 400;
  const uniform = new Float32Array(count * 2).map(() => random() * side);
  const regular = new Float32Array(count * 2);
  for (let index = 0; index < count; index++) {
    regular[index * 2] = ((index % 20) + 0.5) * 5;
    regular[index * 2 + 1] = (Math.floor(index / 20) + 0.5) * 5;
  }
  const clustered = new Float32Array(count * 2);
  for (let index = 0; index < count; index++) {
    const center = (index % 8) * 11 + 6;
    clustered[index * 2] = Math.min(Math.max(center + (random() - 0.5) * 8, 0), side);
    clustered[index * 2 + 1] = Math.min(
      Math.max(((index * 7) % 8) * 11 + 6 + (random() - 0.5) * 8, 0),
      side
    );
  }
  const frame = {
    bounds: [0, 0, side, side] as const,
    maximumDistance: 16.3,
    edgeCorrection: 'isotropic' as const
  };
  const at = (positions: Float32Array) => computeRipleyOnCPU({positions}, frame, 4).lMinusR;
  // Clustered: L - r clearly positive. Regular: negative at short range. CSR: near zero.
  expect(at(clustered)[1]).toBeGreaterThan(2);
  expect(at(regular)[0]).toBeLessThan(-1);
  expect(Math.abs(at(uniform)[2])).toBeLessThan(1.5);
});

it('gpu-ripley declares planar-only spatial support', () => {
  expectThrows(
    () => ({
      spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'} as const
    }),
    /project positions to a planar frame first/
  );
  expectThrows(
    () => ({spatialContext: {coordinateSpace: 'discrete-grid', metric: 'none', gridFamily: 'h3'}}),
    /does not support discrete-grid/
  );
});

// Oracle: <scratchpad>/fix/work/D/point-patterns/border_k.py (numpy 2.5.4, spatstat Kest border
// estimator with lambda = n / A, Baddeley, Rubak and Turner 2015 section 7.4). All coordinates
// and radii are exact binary fractions, so the f32 oracle path equals the f64 numpy values.
it('computeRipleyOnCPU border K uses lambda = n / A like spatstat Kest', () => {
  const positions = new Float32Array([
    1, 1.5, 2, 4, 3.5, 2.5, 4, 5.5, 5, 3, 6, 1, 6.5, 5, 7.5, 3.5, 8, 6, 9, 2, 2.5, 3, 5.5, 4.5
  ]);
  const frame = {bounds: [0, 0, 10, 7] as const, maximumDistance: 3};
  const border = computeRipleyOnCPU({positions}, {...frame, edgeCorrection: 'border'}, 4);
  const expectedBorder = [0, 4.375, 18.666666666666668, 40.833333333333336];
  const expectedNone = [0, 3.1818181818181817, 13.787878787878787, 28.636363636363633];
  const none = computeRipleyOnCPU({positions}, {...frame, edgeCorrection: 'none'}, 4);
  expectedBorder.forEach((value, b) => {
    expect(border.k[b], `border ${b}`).toBeCloseTo(value, 10);
    expect(none.k[b], `none ${b}`).toBeCloseTo(expectedNone[b], 10);
  });
});

it('getIsotropicWeightOnCPU stays exact far beyond half the shorter window side', () => {
  // Brute-force circle sampling (isotropic_validity.py: max fraction error 1.1e-5 up to radius 25
  // in a 12 x 7 window): opposite-side arcs never overlap, so the rectangle formula needs no
  // validity guard on the radius. The weight cap only matters once the fraction drops below 1/100.
  const random = createRandom(5);
  const width = 12;
  const height = 7;
  const samples = 100000;
  for (let trial = 0; trial < 40; trial++) {
    const x = random() * width;
    const y = random() * height;
    const distance = 3.6 + random() * 3.4; // beyond height / 2 = 3.5 and up to ~ width / 2 + 1
    let inside = 0;
    for (let sample = 0; sample < samples; sample++) {
      const angle = ((sample + 0.5) / samples) * 2 * Math.PI;
      const pointX = x + distance * Math.cos(angle);
      const pointY = y + distance * Math.sin(angle);
      if (pointX >= 0 && pointX <= width && pointY >= 0 && pointY <= height) {
        inside++;
      }
    }
    const weight = getIsotropicWeightOnCPU(x, y, width - x, height - y, distance);
    expect(Math.abs(1 / weight - inside / samples), `trial ${trial}`).toBeLessThan(1e-3);
  }
});

it.each([
  'local-azimuthal-equidistant',
  'local-lambert-equal-area'
] as const)('GPURipley accepts the %s frame for combined distance and area measures', kind => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const spatialContext = {
    coordinateSpace: 'planar',
    metric: 'native',
    units: 'meters',
    frame: {id: 'f', kind}
  } as const;
  expect(new GPURipley(createProps(graph, {spatialContext})).spatialContext).toBe(spatialContext);
  device.destroy();
});
