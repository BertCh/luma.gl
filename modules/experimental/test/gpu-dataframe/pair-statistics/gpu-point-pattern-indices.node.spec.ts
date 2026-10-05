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
  GPU_POINT_PATTERN_INDICES_PARAMETER_LENGTH
} from '../../../src/gpu-dataframe/pair-statistics/point-pattern-indices-parameters';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {computePointPatternIndicesOnCPU} from './point-pattern-indices-oracle';

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
    clarkEvans: view('float32', 6),
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
  ).toEqual([0, 1, 2, 3, 4, 0, 0, 0]);
  const bounds = [0, 0, 1, 1] as const;
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
  expectThrows(graph => ({clarkEvans: view(graph, 'float32', 5)}), /clarkEvans/);
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
    graph => ({quadratStatistics: view(graph, 'float32', 5), quadratGrid: [3, 2]}),
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
    quadratStatistics: view(graph, 'float32', 6)
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
    {bounds: [0, 0, 10, 10], maximumDistance: 3},
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
