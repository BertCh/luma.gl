// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUPointPatternIndices,
  type GPUPointPatternIndicesProps
} from '../../../src/gpu-dataframe/pair-statistics/gpu-point-pattern-indices';
import {
  getGPUPointPatternIndicesParameterValues,
  GPU_CLARK_EVANS_CORRECTION,
  GPU_CLARK_EVANS_LENGTH,
  GPU_POINT_PATTERN_ALTERNATIVE,
  GPU_POINT_PATTERN_INDICES_PARAMETER_LENGTH
} from '../../../src/gpu-dataframe/pair-statistics/point-pattern-indices-parameters';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  computePointPatternIndicesOnCPU,
  getChiSquarePValueOnCPU,
  getGammaTailsOnCPU,
  getNormalPValueOnCPU
} from './point-pattern-indices-oracle';

let serial = 0;

function createProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUPointPatternIndicesProps> = {}
): GPUPointPatternIndicesProps {
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `view-${serial++}`, format, length);
  return {
    positions: view('float32x2', 10),
    parameters: view('float32', GPU_POINT_PATTERN_INDICES_PARAMETER_LENGTH),
    gridSize: [4, 4],
    clarkEvans: view('float32', GPU_CLARK_EVANS_LENGTH),
    ...overrides
  };
}

function expectThrows(
  overrides: (graph: GPUCommandGraph) => Partial<GPUPointPatternIndicesProps>,
  message: RegExp
): void {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  expect(() => new GPUPointPatternIndices(createProps(graph, overrides(graph)))).toThrow(message);
  device.destroy();
}

it('getGPUPointPatternIndicesParameterValues packs and validates the layout', () => {
  expect(
    Array.from(getGPUPointPatternIndicesParameterValues({bounds: [0, 1, 2, 3], maximumDistance: 4}))
  ).toEqual([
    0,
    1,
    2,
    3,
    4,
    GPU_CLARK_EVANS_CORRECTION.donnelly,
    GPU_POINT_PATTERN_ALTERNATIVE['two-sided'],
    0
  ]);
  const bounds = [0, 0, 1, 1] as const;
  const packed = getGPUPointPatternIndicesParameterValues({
    bounds,
    maximumDistance: 1,
    clarkEvansCorrection: 'none',
    alternative: 'less'
  });
  expect(packed[5]).toBe(GPU_CLARK_EVANS_CORRECTION.none);
  expect(packed[6]).toBe(GPU_POINT_PATTERN_ALTERNATIVE.lesser);
  expect(
    getGPUPointPatternIndicesParameterValues({
      bounds,
      maximumDistance: 1,
      alternative: 'greater'
    })[6]
  ).toBe(GPU_POINT_PATTERN_ALTERNATIVE.greater);
  expect(() =>
    getGPUPointPatternIndicesParameterValues({
      bounds,
      maximumDistance: 1,
      clarkEvansCorrection: 'ripley' as never
    })
  ).toThrow(/clarkEvansCorrection/);
  expect(() =>
    getGPUPointPatternIndicesParameterValues({
      bounds,
      maximumDistance: 1,
      alternative: 'directed' as never
    })
  ).toThrow(/alternative/);
  expect(() => getGPUPointPatternIndicesParameterValues({bounds, maximumDistance: 0})).toThrow(
    /positive/
  );
  expect(() => getGPUPointPatternIndicesParameterValues({bounds, maximumDistance: NaN})).toThrow(
    /finite/
  );
  expect(() =>
    getGPUPointPatternIndicesParameterValues({bounds: [1, 0, 0, 1], maximumDistance: 1})
  ).toThrow(/minX/);
  expect(() =>
    getGPUPointPatternIndicesParameterValues({bounds, maximumDistance: 1}, new Float32Array(4))
  ).toThrow(/8/);
});

it('GPUPointPatternIndices validates its inputs and outputs', () => {
  const view = <Format extends 'float32' | 'uint32'>(
    graph: GPUCommandGraph,
    format: Format,
    length: number
  ) => createTransientView(graph, `bad-${serial++}`, format, length);
  expectThrows(
    graph => ({clarkEvans: undefined, positions: view(graph, 'float32', 3) as never}),
    /positions/
  );
  expectThrows(() => ({clarkEvans: undefined}), /at least one output/);
  expectThrows(graph => ({clarkEvans: view(graph, 'float32', 6)}), /clarkEvans/);
  expectThrows(
    graph => ({nearestNeighborDistances: view(graph, 'float32', 9)}),
    /nearestNeighborDistances/
  );
  expectThrows(graph => ({nearestNeighborIds: view(graph, 'uint32', 11)}), /nearestNeighborIds/);
  expectThrows(graph => ({quadratCounts: view(graph, 'uint32', 6)}), /quadratGrid/);
  expectThrows(
    graph => ({quadratCounts: view(graph, 'uint32', 6), quadratGrid: [0, 3]}),
    /quadratGrid/
  );
  expectThrows(
    graph => ({quadratCounts: view(graph, 'uint32', 65537), quadratGrid: [257, 256]}),
    /65536/
  );
  expectThrows(
    graph => ({quadratCounts: view(graph, 'uint32', 5), quadratGrid: [3, 2]}),
    /quadratCounts/
  );
  expectThrows(
    graph => ({quadratStatistics: view(graph, 'float32', 6), quadratGrid: [3, 2]}),
    /quadratStatistics/
  );
  expectThrows(() => ({quadratGrid: [2, 2]}), /quadratGrid needs/);
  expectThrows(graph => ({parameters: view(graph, 'float32', 4)}), /parameters/);
  expectThrows(graph => ({mask: view(graph, 'uint32', 9)}), /mask length/);
  expectThrows(graph => {
    const shared = view(graph, 'float32', 10);
    return {nearestNeighborDistances: shared, positions: shared as never};
  }, /alias|overlap|disjoint|input|share|positions/i);
});

it('GPUPointPatternIndices builds only the nodes of the requested parts', () => {
  const device = createNullWebGPUDevice();
  const ids = (overrides: (graph: GPUCommandGraph) => Partial<GPUPointPatternIndicesProps>) => {
    const graph = new GPUCommandGraph(device);
    const props = createProps(graph, {id: 'p', ...overrides(graph)});
    return new GPUPointPatternIndices(props).getCommandNodes(graph).map(node => node.id);
  };
  const view = <Format extends 'float32' | 'uint32'>(
    graph: GPUCommandGraph,
    format: Format,
    length: number
  ) => createTransientView(graph, `n-${serial++}`, format, length);
  const nearestOnly = ids(graph => ({
    clarkEvans: undefined,
    nearestNeighborIds: view(graph, 'uint32', 10)
  }));
  expect(nearestOnly).toContain('p-nearest');
  expect(nearestOnly.some(id => id.includes('quadrat') || id.includes('clark'))).toBe(false);
  const quadratOnly = ids(graph => ({
    clarkEvans: undefined,
    quadratGrid: [3, 2],
    quadratStatistics: view(graph, 'float32', 7)
  }));
  expect(quadratOnly).toContain('p-quadrat-counts');
  expect(quadratOnly).toContain('p-quadrat-statistics');
  expect(quadratOnly).not.toContain('p-nearest');
  const both = ids(graph => ({quadratGrid: [3, 2], quadratCounts: view(graph, 'uint32', 6)}));
  expect(both).toContain('p-clark-evans');
  expect(both).toContain('p-quadrat-counts');
  expect(both).not.toContain('p-quadrat-statistics');
  expect(ids(graph => ({quadratGrid: [3, 2], quadratCounts: view(graph, 'uint32', 6)}))).toEqual(
    both
  );
  device.destroy();
});

it('computePointPatternIndicesOnCPU matches hand-computed values', () => {
  // Square window of area 100 with four points: (1,1) (2,1) (8,8) (9,8) form two close pairs.
  const positions = new Float32Array([1, 1, 2, 1, 8, 8, 9, 8]);
  const result = computePointPatternIndicesOnCPU(
    {positions},
    {bounds: [0, 0, 10, 10], maximumDistance: 3, clarkEvansCorrection: 'none'},
    [2, 2]
  );
  expect(result.nearestNeighborDistances).toEqual([1, 1, 1, 1]);
  expect(result.nearestNeighborIds).toEqual([1, 0, 3, 2]);
  const expected = 0.5 / Math.sqrt(4 / 100);
  expect(result.clarkEvans[0]).toBe(4);
  expect(result.clarkEvans[1]).toBe(1);
  expect(result.clarkEvans[2]).toBeCloseTo(expected, 12);
  expect(result.clarkEvans[3]).toBeCloseTo(1 / expected, 12);
  expect(result.clarkEvans[4]).toBeCloseTo(0.26136 / Math.sqrt(16 / 100), 12);
  expect(result.clarkEvans[5]).toBeCloseTo((1 - expected) / result.clarkEvans[4], 12);
  // Quadrats: two points in the lower-left, two in the upper-right: counts [2, 0, 0, 2].
  expect(result.quadratCounts).toEqual([2, 0, 0, 2]);
  // mean 1, sample variance (1 + 1 + 1 + 1) / 3 = 4 / 3, VMR 4 / 3, chi-square 4, 3 degrees.
  expect(result.quadratStatistics[0]).toBe(4);
  expect(result.quadratStatistics[1]).toBe(1);
  expect(result.quadratStatistics[2]).toBeCloseTo(4 / 3, 12);
  expect(result.quadratStatistics[3]).toBeCloseTo(4 / 3, 12);
  expect(result.quadratStatistics[4]).toBeCloseTo(4, 12);
  expect(result.quadratStatistics[5]).toBe(3);
});

it('computePointPatternIndicesOnCPU breaks nearest-neighbor ties to the smallest row and handles n < 2', () => {
  // Row 1 is equidistant from rows 0 and 2.
  const ties = computePointPatternIndicesOnCPU(
    {positions: new Float32Array([0, 0, 1, 0, 2, 0])},
    {bounds: [0, 0, 4, 4], maximumDistance: 2}
  );
  expect(ties.nearestNeighborIds[1]).toBe(0);
  const single = computePointPatternIndicesOnCPU(
    {positions: new Float32Array([1, 1, 20, 20])},
    {bounds: [0, 0, 4, 4], maximumDistance: 2}
  );
  expect(single.clarkEvans[0]).toBe(1);
  expect(single.clarkEvans.slice(1).every(Number.isNaN)).toBe(true);
  expect(single.nearestNeighborDistances[0]).toBeNaN();
  expect(single.nearestNeighborIds[0]).toBe(0xffffffff);
  expect(single.nearestNeighborIds[1]).toBe(0xffffffff);
});

// Oracle constants: <scratchpad>/fix/work/D/point-patterns/clark_evans.py (pointpats 2.6.0 mean
// nearest-neighbor distance; numpy 2.5.4 and scipy 1.18.1 for the published Clark-Evans (1954) and
// Donnelly (1978) formulas and the normal / chi-square p-values).
const ORACLE_POINTS = [
  1, 1.5, 2, 4, 3.5, 2.5, 4, 5.5, 5, 3, 6, 1, 6.5, 5, 7.5, 3.5, 8, 6, 9, 2, 2.5, 3, 5.5, 4.5
];
const ORACLE_FRAME = {bounds: [0, 0, 10, 7] as const, maximumDistance: 3};
const ORACLE_OBSERVED = 1.5881953626373935;
const ORACLE_CLARK_EVANS = {
  none: {
    expected: 1.20761472884912,
    ratio: 1.3151507055159672,
    standardError: 0.18222455377912164,
    z: 2.088525535639855,
    pTwoSided: 0.036750454408795236,
    pGreater: 0.018375227204397618,
    pLesser: 0.9816247727956023
  },
  donnelly: {
    expected: 1.386782490317884,
    ratio: 1.145237536330114,
    standardError: 0.23510279456525304,
    z: 0.8567013109816825,
    pTwoSided: 0.39160997863450153,
    pGreater: 0.19580498931725077,
    pLesser: 0.8041950106827492
  }
} as const;

it('computePointPatternIndicesOnCPU matches pointpats / numpy Clark-Evans with and without Donnelly', () => {
  const scene = {positions: new Float32Array(ORACLE_POINTS)};
  for (const correction of ['none', 'donnelly'] as const) {
    const expected = ORACLE_CLARK_EVANS[correction];
    const cases = [
      ['two-sided', expected.pTwoSided],
      ['greater', expected.pGreater],
      ['lesser', expected.pLesser]
    ] as const;
    for (const [alternative, pValue] of cases) {
      const [n, observed, mean, ratio, standardError, z, p] = computePointPatternIndicesOnCPU(
        scene,
        {...ORACLE_FRAME, clarkEvansCorrection: correction, alternative}
      ).clarkEvans;
      expect(n).toBe(12);
      expect(observed).toBeCloseTo(ORACLE_OBSERVED, 6);
      expect(mean).toBeCloseTo(expected.expected, 12);
      expect(ratio).toBeCloseTo(expected.ratio, 6);
      expect(standardError).toBeCloseTo(expected.standardError, 12);
      expect(z).toBeCloseTo(expected.z, 5);
      expect(p, `${correction} ${alternative}`).toBeCloseTo(pValue, 6);
    }
  }
  // The default is the Donnelly correction.
  expect(computePointPatternIndicesOnCPU(scene, ORACLE_FRAME).clarkEvans[2]).toBeCloseTo(
    ORACLE_CLARK_EVANS.donnelly.expected,
    12
  );
});

it('the oracle incomplete gamma and chi-square p-values match scipy', () => {
  // chi_square cases: [x, degreesOfFreedom, lower, upper, twoSided] from scipy.stats.chi2.
  const cases = [
    [3.2, 5, 0.33081709796675673, 0.6691829020332433, 0.6616341959335135],
    [11.0, 24, 0.010988141058626786, 0.9890118589413732, 0.02197628211725357],
    [30.0, 24, 0.8152482009760686, 0.1847517990239314, 0.3695035980478628],
    [95.0, 99, 0.4049220955704821, 0.5950779044295179, 0.8098441911409642],
    [130.0, 100, 0.9764876021901913, 0.023512397809808714, 0.04702479561961743],
    [160.0, 150, 0.726842194708074, 0.27315780529192607, 0.5463156105838521],
    [480.0, 500, 0.2676500698540158, 0.7323499301459842, 0.5353001397080316]
  ] as const;
  for (const [x, degrees, lower, upper, twoSided] of cases) {
    const [gammaLower, gammaUpper] = getGammaTailsOnCPU(degrees / 2, x / 2);
    expect(gammaLower).toBeCloseTo(lower, 9);
    expect(gammaUpper).toBeCloseTo(upper, 9);
    expect(getChiSquarePValueOnCPU(x, degrees, 'lesser')).toBeCloseTo(lower, 9);
    expect(getChiSquarePValueOnCPU(x, degrees, 'greater')).toBeCloseTo(upper, 9);
    expect(getChiSquarePValueOnCPU(x, degrees, 'two-sided')).toBeCloseTo(twoSided, 9);
  }
  // Normal tails are complementary.
  for (const z of [-2.5, -0.3, 0, 0.8567013109816825, 3.1]) {
    expect(getNormalPValueOnCPU(z, 'greater') + getNormalPValueOnCPU(z, 'lesser')).toBeCloseTo(
      1,
      12
    );
  }
});

it('computePointPatternIndicesOnCPU attaches the chi-square p-value to the quadrat summary', () => {
  // Counts [2, 0, 0, 2]: chi-square 4 on 3 degrees of freedom (clark_evans.py, last line).
  const result = computePointPatternIndicesOnCPU(
    {positions: new Float32Array([1, 1, 2, 1, 8, 8, 9, 8])},
    {bounds: [0, 0, 10, 10], maximumDistance: 3, alternative: 'greater'},
    [2, 2]
  );
  expect(result.quadratStatistics[6]).toBeCloseTo(0.26146412994911034, 9);
});

it('gpu-point-pattern-indices declares planar-only spatial support', () => {
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
