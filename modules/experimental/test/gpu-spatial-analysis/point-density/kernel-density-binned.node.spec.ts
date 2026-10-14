// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {WgslReflect} from 'wgsl_reflect';
import {
  GPUKernelDensity,
  type GPUKernelDensityProps
} from '../../../src/gpu-spatial-analysis/point-density';
import {
  getGPUKernelDensityBinnedErrorBound,
  type GPUKernelDensityMethod
} from '../../../src/gpu-spatial-analysis/point-density/kernel-density-math';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {makeExtentGrid} from '../spatial-interpolation/extent-grid';
import {
  applyBoxBlur,
  applyExponentialBlur,
  getBinnedAxisPlan,
  getBinnedDensityModel,
  getBoxVariance,
  getExactDensityModel
} from './kernel-density-binned-oracle';

function createNodes(
  method: GPUKernelDensityMethod,
  options: Partial<GPUKernelDensityProps> & {pointCount?: number} = {}
) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const {pointCount = 300, ...props} = options;
  const kde = new GPUKernelDensity({
    positions: createTransientView(graph, 'positions', 'float32x2', pointCount),
    weights: createTransientView(graph, 'weights', 'float32', pointCount),
    parameters: createTransientView(graph, 'parameters', 'float32', 9),
    grid: makeExtentGrid(24, 16, [0, 0, 24, 16]),
    indexGridSize: [4, 4],
    indexBounds: [0, 0, 24, 16],
    method,
    output: {
      values: createTransientView(graph, 'values', 'float32', 24 * 16),
      weightSum: createTransientView(graph, 'weight-sum', 'float32', 1)
    },
    ...props
  });
  return {device, kde, nodes: kde.getCommandNodes(graph)};
}

it('schedules the nodes of each method', () => {
  const exact = createNodes('exact');
  const exactIds = exact.nodes.map(node => node.id);
  expect(exactIds.some(id => id.startsWith('kernel-density-index'))).toBe(true);
  expect(exactIds.at(-1)).toBe('kernel-density-gather');
  expect(exactIds.some(id => id.includes('blur') || id.includes('-bin'))).toBe(false);
  // The default keeps the exact graph.
  expect(createNodes(undefined as never).nodes.map(node => node.id)).toEqual(exactIds);

  const binned = createNodes('binned');
  const binnedIds = binned.nodes.map(node => node.id);
  expect(binnedIds.some(id => id.includes('-index') || id.includes('sort-cells'))).toBe(false);
  expect(binnedIds).not.toContain('kernel-density-gather');
  const order = [
    'kernel-density-point-weights',
    'kernel-density-blur-plan',
    'kernel-density-bin',
    'kernel-density-bin-decode',
    'kernel-density-blur-x-box-0',
    'kernel-density-blur-x-exponential-3',
    'kernel-density-blur-y-box-0',
    'kernel-density-blur-y-exponential-3',
    'kernel-density-finish'
  ].map(id => binnedIds.indexOf(id));
  expect(order.every(position => position >= 0)).toBe(true);
  expect([...order].sort((left, right) => left - right)).toEqual(order);
  expect(binnedIds.at(-1)).toBe('kernel-density-finish');
  expect(new Set(binnedIds).size).toBe(binnedIds.length);
  // Three boxes and one exponential per axis.
  expect(binnedIds.filter(id => id.includes('-blur-x-box-'))).toHaveLength(3);
  expect(binnedIds.filter(id => id.includes('-blur-y-exponential-'))).toHaveLength(1);

  const auto = createNodes('auto');
  const autoIds = auto.nodes.map(node => node.id);
  expect(autoIds.indexOf('kernel-density-blur-plan')).toBeLessThan(
    autoIds.indexOf('kernel-density-gather')
  );
  expect(autoIds.indexOf('kernel-density-gather')).toBeLessThan(
    autoIds.indexOf('kernel-density-bin')
  );
  const gated = auto.nodes.filter(node => node.condition !== undefined).map(node => node.id);
  // Every heavy node of both paths is gated; the plan writer and the O(n) setup are not.
  expect(gated).toContain('kernel-density-gather');
  expect(gated).toContain('kernel-density-finish');
  expect(gated).toContain('kernel-density-bin');
  expect(gated).not.toContain('kernel-density-blur-plan');
  expect(gated).not.toContain('kernel-density-point-weights');
  expect(binned.nodes.some(node => node.condition !== undefined)).toBe(false);
  expect(exact.nodes.some(node => node.condition !== undefined)).toBe(false);
  for (const {device} of [exact, binned, auto]) {
    device.destroy();
  }
});

it('every method resolves kernel support and rejects invalid options', () => {
  const quartic = createNodes('auto', {kernel: 'quartic'});
  expect(quartic.kde.method).toBe('exact');
  expect(quartic.nodes.some(node => node.id.includes('blur'))).toBe(false);
  quartic.device.destroy();
  expect(() => createNodes('binned', {kernel: 'epanechnikov'})).toThrow(
    /kernel-density method binned needs the gaussian kernel/
  );
  expect(() => createNodes('cascade' as never)).toThrow(/method must be/);
  expect(() => createNodes('binned', {binnedPadding: -1})).toThrow(/binnedPadding/);
  expect(() => createNodes('binned', {binnedPadding: 1.5})).toThrow(/binnedPadding/);
  expect(() => createNodes('binned', {binnedPadding: 70000})).toThrow(/too large/);
  // The default pads by one grid side so bandwidths up to 40% of it have their margin.
  expect(createNodes('binned').kde.binnedPadding).toBe(24);
  expect(createNodes('binned', {binnedPadding: 0}).kde.binnedPadding).toBe(0);
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const exactWithStatus = {
    positions: createTransientView(graph, 'positions', 'float32x2', 8),
    parameters: createTransientView(graph, 'parameters', 'float32', 9),
    grid: makeExtentGrid(24, 16, [0, 0, 24, 16]),
    indexGridSize: [4, 4] as const,
    indexBounds: [0, 0, 24, 16] as const,
    output: {
      values: createTransientView(graph, 'values', 'float32', 24 * 16),
      status: {insufficientPadding: createTransientView(graph, 'padding', 'uint32', 1)}
    }
  };
  expect(() => new GPUKernelDensity(exactWithStatus)).toThrow(/needs method binned or auto/);
  expect(() => new GPUKernelDensity({...exactWithStatus, method: 'binned'})).not.toThrow();
  device.destroy();
});

it('binned kernels parse as WGSL, bind at most 8 storage buffers and use bit-test NaN rules', () => {
  for (const method of ['binned', 'auto'] as const) {
    for (const [edgeCorrection, normalization] of [
      ['uniform', 'intensity'],
      ['diggle', 'probability'],
      ['none', 'probability']
    ] as const) {
      const {device, nodes} = createNodes(method, {
        edgeCorrection,
        normalization,
        bandwidthRule: 'scott'
      });
      const sources = getGPUCommandNodeKernelSources(nodes);
      for (const {id, source, constants} of sources) {
        expect(() => new WgslReflect(source), id).not.toThrow();
        // Reserved words, folded self-compares, undeclared calls and the core shader limits.
        expect(auditWGSLModule(source, constants), id).toEqual([]);
        expect(getStorageBindingCount(source), id).toBeLessThanOrEqual(8);
        // The shared reductions are not part of this item; check the kernels it adds.
        if (/-(blur|bin|finish)/.test(id)) {
          expect(source, id).not.toMatch(/(\w+) == \1\b/);
          // WGSL reserved words that a kernel variable could shadow by accident.
          expect(source, id).not.toMatch(/\b(?:let|var) (?:target|filter|length)\b/);
        }
      }
      const byId = (fragment: string) => sources.find(entry => entry.id.includes(fragment))?.source;
      expect(byId('blur-plan')).toContain('bandwidthState[bandwidthStateOffset]');
      expect(byId('-finish')).toContain('bandwidthState[bandwidthStateOffset]');
      device.destroy();
    }
  }
  const auto = createNodes('auto');
  const plan = getGPUCommandNodeKernelSources(auto.nodes).find(entry =>
    entry.id.includes('blur-plan')
  )!;
  // The GPU switches at the documented support radius.
  expect(plan.source).toContain('AUTO_SUPPORT_CELLS: f32 = 32.0');
  expect(plan.source).toContain('dispatch[dispatchOffset + 0u] = 0u;');
  auto.device.destroy();
});

it('bandwidth reaches the binned path only through GPU words (rule 12)', () => {
  // Bandwidth-dependent filter parameters are written by blur-plan at run time, so nothing in a
  // pipeline can depend on them. The same holds for the point count: N and N + 1 points share
  // the number of pipelines and every source.
  for (const method of ['binned', 'auto'] as const) {
    const small = createNodes(method, {pointCount: 300});
    const large = createNodes(method, {pointCount: 301});
    const smallKeys = getGPUCommandNodePipelineKeys(small.nodes);
    const largeKeys = getGPUCommandNodePipelineKeys(large.nodes);
    expect(largeKeys.size).toBe(smallKeys.size);
    // The sources of the kernels this item adds are identical; shared reductions bake the row count.
    const sources = (nodes: typeof small.nodes) =>
      new Set(
        getGPUCommandNodeKernelSources(nodes)
          .filter(entry => /-(blur|bin|finish)/.test(entry.id))
          .map(entry => entry.source)
      );
    expect([...sources(large.nodes)].sort()).toEqual([...sources(small.nodes)].sort());
    // Three identical box passes share one pipeline per axis; the exponential pass is another.
    expect(getGPUCommandNodePipelineKeys(small.nodes, {idPattern: /blur-x-box/}).size).toBe(1);
    expect(getGPUCommandNodePipelineKeys(small.nodes, {idPattern: /blur-x/}).size).toBe(2);
    expect(getGPUCommandNodePipelineKeys(small.nodes, {idPattern: /blur-[xy]-/}).size).toBe(4);
    small.device.destroy();
    large.device.destroy();
  }
});

it('the box plan matches the variance and the passes match direct sums', () => {
  for (const sigma of [3, 7.3, 12, 40, 200]) {
    const plan = getBinnedAxisPlan(sigma, 1000);
    const varianceBox = (sigma * sigma * (1 - 0.32)) / 3;
    expect(getBoxVariance(plan)).toBeCloseTo(varianceBox, 8);
    expect(plan.edge).toBeGreaterThanOrEqual(0);
    expect(plan.edge).toBeLessThanOrEqual(1);
    // The pole gives the variance of the two-sided geometric: 2a / (1 - a)^2.
    expect((2 * plan.pole) / (1 - plan.pole) ** 2).toBeCloseTo(0.32 * sigma * sigma, 6);
  }
  const line = Float64Array.from({length: 400}, (_, index) => ((index * 37) % 11) + (index % 5));
  const plan = getBinnedAxisPlan(20, line.length);
  const running = applyBoxBlur(line, plan);
  // The same pass with a direct window sum at every cell (refresh interval one).
  const direct = applyBoxBlur(line, {...plan, interval: 1});
  for (const [index, value] of running.entries()) {
    expect(Math.abs(value - direct[index])).toBeLessThan(1e-12);
  }
  // Box and exponential passes keep mass that stays inside; zero extension exact at the ends.
  const impulse = new Float64Array(2000);
  impulse[1000] = 1;
  const blurred = applyExponentialBlur(applyBoxBlur(impulse, plan), plan.pole);
  expect(blurred.reduce((sum, value) => sum + value, 0)).toBeCloseTo(1, 10);
  const atEdge = new Float64Array(2000);
  atEdge[0] = 1;
  const edgeBlurred = applyExponentialBlur(applyBoxBlur(atEdge, plan), plan.pole);
  // Brute-force discrete Laplace of the box output with zero extension past both ends.
  const boxed = applyBoxBlur(atEdge, plan);
  const reference = new Float64Array(2000);
  const scale = (1 - plan.pole) / (1 + plan.pole);
  for (let output = 0; output < 2000; output++) {
    for (let input = 0; input < 2000; input++) {
      reference[output] += boxed[input] * scale * plan.pole ** Math.abs(output - input);
    }
  }
  for (let index = 0; index < 2000; index++) {
    expect(Math.abs(edgeBlurred[index] - reference[index])).toBeLessThan(1e-12);
  }
});

it('the binned model stays within the derived bound of the exact estimate', () => {
  // 96 x 64 cells of 0.5 units; clustered points, points on the grid edge and zero weights.
  const width = 96;
  const height = 64;
  const cell = 0.5;
  const start = [10, -4] as const;
  let seed = 12345;
  const random = () => {
    seed = (seed * 1664525 + 1013904223) % 4294967296;
    return seed / 4294967296;
  };
  const points: number[] = [];
  const weights: number[] = [];
  const addCluster = (centerX: number, centerY: number, spread: number, count: number) => {
    for (let index = 0; index < count; index++) {
      const x = centerX + spread * (random() + random() + random() - 1.5) * 2;
      const y = centerY + spread * (random() + random() + random() - 1.5) * 2;
      points.push(
        Math.min(Math.max(x, start[0]), start[0] + width * cell),
        Math.min(Math.max(y, start[1]), start[1] + height * cell)
      );
      weights.push(index % 17 === 0 ? 0 : 0.5 + random() * 2);
    }
  };
  addCluster(20, 6, 2, 150);
  addCluster(48, 24, 6, 100);
  addCluster(10, -4, 0.3, 20); // on the corner
  addCluster(58, 28, 0.2, 20); // on the opposite corner
  for (const bandwidth of [1.5, 3, 6, 12, 25]) {
    const input = {
      points,
      weights,
      width,
      height,
      start,
      cellSize: [cell, cell] as const,
      bandwidth
    };
    const exact = getExactDensityModel(input);
    // Margin of 2.5 bandwidths (the guard cell adds one more) so the cascade drops nothing.
    const binned = getBinnedDensityModel({...input, padding: Math.ceil((2.5 * bandwidth) / cell)});
    const totalWeight = weights.reduce((sum, weight) => sum + weight, 0);
    const scale = totalWeight / (2 * Math.PI * bandwidth * bandwidth);
    const bound = getGPUKernelDensityBinnedErrorBound(cell / bandwidth) * scale;
    let worst = 0;
    for (const [index, value] of exact.entries()) {
      worst = Math.max(worst, Math.abs(binned[index] - value));
    }
    expect(worst, `h = ${bandwidth}`).toBeLessThanOrEqual(bound);
    // The bound is not vacuous: the model errs by more than a thousandth of it.
    expect(worst).toBeGreaterThan(bound * 1e-3);
  }
  // North-up rows (negative y cell size) give the same field mirrored in row order.
  const northUp = getBinnedDensityModel({
    points,
    weights,
    width,
    height,
    start: [start[0], start[1] + height * cell],
    cellSize: [cell, -cell],
    bandwidth: 6,
    padding: 30
  });
  const north = getBinnedDensityModel({
    points,
    weights,
    width,
    height,
    start,
    cellSize: [cell, cell],
    bandwidth: 6,
    padding: 30
  });
  for (let row = 0; row < height; row++) {
    for (let column = 0; column < width; column++) {
      expect(northUp[row * width + column]).toBeCloseTo(
        north[(height - 1 - row) * width + column],
        10
      );
    }
  }
});

it('padding recovers the mass of points outside the grid', () => {
  const width = 40;
  const height = 40;
  const points = [-6, 20, 46, 20, 20, 20]; // two points beyond the grid sides
  const base = {
    points,
    width,
    height,
    start: [0, 0] as const,
    cellSize: [1, 1] as const,
    bandwidth: 5
  };
  const exact = getExactDensityModel(base);
  const unpadded = getBinnedDensityModel(base);
  const padded = getBinnedDensityModel({...base, padding: 30});
  let worstUnpadded = 0;
  let worstPadded = 0;
  for (const [index, value] of exact.entries()) {
    worstUnpadded = Math.max(worstUnpadded, Math.abs(unpadded[index] - value));
    worstPadded = Math.max(worstPadded, Math.abs(padded[index] - value));
  }
  const scale = 3 / (2 * Math.PI * 25);
  expect(worstPadded).toBeLessThanOrEqual(getGPUKernelDensityBinnedErrorBound(0.2) * scale);
  expect(worstUnpadded).toBeGreaterThan(0.05 * scale);
});
