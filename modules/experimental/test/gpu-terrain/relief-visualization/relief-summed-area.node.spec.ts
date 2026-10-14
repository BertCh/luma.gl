// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {describe, expect, it} from 'vitest';
import {makeGPURasterMetadata} from '../../../src/gpu-raster';
import {GPUMultiScaleRelief} from '../../../src/gpu-terrain/relief-visualization/gpu-multi-scale-relief';
import {GPUSimpleLocalRelief} from '../../../src/gpu-terrain/relief-visualization/gpu-simple-local-relief';
import {
  GPU_RELIEF_DEFAULT_QUANTUM,
  GPU_RELIEF_SUMMED_AREA_MAXIMUM_RADIUS
} from '../../../src/gpu-terrain/relief-visualization/relief-visualization-utils';
import {
  createNullWebGPUDevice,
  getGPUCommandNodePipelineKeys,
  getParsedKernelSources,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';
import {createBand} from '../terrain-test-utils';
import {computeMeanFilterRVT} from './relief-visualization-oracle';
import {computeLiteralReliefDeviation, modelReliefDeviation} from './relief-summed-area-model';

function createRandom(seed: number) {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

type Case = {
  name: string;
  width: number;
  height: number;
  radius: number;
  base: number;
  amplitude: number;
  holeFraction: number;
};

const CASES: Case[] = [
  {
    name: 'high positive origin',
    width: 17,
    height: 13,
    radius: 4,
    base: 1000000,
    amplitude: 10,
    holeFraction: 0.1
  },
  {
    name: 'high negative origin',
    width: 17,
    height: 13,
    radius: 4,
    base: -1000000,
    amplitude: 10,
    holeFraction: 0.1
  },

  {
    name: 'interior windows',
    width: 17,
    height: 13,
    radius: 3,
    base: 4000,
    amplitude: 5,
    holeFraction: 0
  },
  {
    name: 'nodata holes',
    width: 17,
    height: 13,
    radius: 4,
    base: 1234.5,
    amplitude: 20,
    holeFraction: 0.3
  },
  {
    name: 'radius larger than grid',
    width: 6,
    height: 5,
    radius: 11,
    base: 50,
    amplitude: 3,
    holeFraction: 0.1
  },
  {name: '1 x N', width: 1, height: 9, radius: 3, base: 800, amplitude: 4, holeFraction: 0.2},
  {name: 'N x 1', width: 9, height: 1, radius: 2, base: -300, amplitude: 4, holeFraction: 0.2},
  {name: '1 x 1', width: 1, height: 1, radius: 5, base: 7.25, amplitude: 1, holeFraction: 0},
  {
    name: 'mostly invalid',
    width: 8,
    height: 7,
    radius: 2,
    base: 10,
    amplitude: 1,
    holeFraction: 0.85
  },
  {
    name: 'sub-metre relief near zero',
    width: 12,
    height: 9,
    radius: 3,
    base: 0,
    amplitude: 0.01,
    holeFraction: 0.1
  },
  {name: 'radius 1', width: 5, height: 5, radius: 1, base: 100, amplitude: 2, holeFraction: 0.2}
];

/**
 * Bound on |model - literal float64|: the window samples are quantized (mean error <= quantum / 2,
 * zero for |z| >= 128 where float32 elevations are multiples of 2^-16), and the float32 chain
 * D -> f32, / count, * quantum, + residual adds <= 3 rounding steps of 2^-24 relative to
 * |deviation| + quantum (the added term is at most |deviation| + quantum / 2).
 */
function getBound(deviation: number, quantum: number): number {
  return quantum / 2 + 4 * 2 ** -24 * (Math.abs(deviation) + quantum);
}

describe('relief summed-area model', () => {
  for (const testCase of CASES) {
    it(`equals the literal RVT mean filter: ${testCase.name}`, () => {
      const {width, height, radius} = testCase;
      const random = createRandom(width * 131 + height * 17 + radius);
      const elevation = new Float32Array(width * height);
      const validity = new Uint8Array(width * height);
      for (let index = 0; index < elevation.length; index++) {
        elevation[index] = testCase.base + (random() - 0.5) * 2 * testCase.amplitude;
        validity[index] = random() < testCase.holeFraction ? 0 : 1;
      }
      const quantum = GPU_RELIEF_DEFAULT_QUANTUM;
      const model = modelReliefDeviation(elevation, validity, width, height, radius, quantum);
      const literal = computeLiteralReliefDeviation(elevation, validity, width, height, radius);
      let worst = 0;
      for (let index = 0; index < elevation.length; index++) {
        // Invalid centres stay NaN; counts are exact integers.
        expect(Number.isNaN(model.deviation[index])).toBe(!validity[index]);
        if (!validity[index]) continue;
        expect(model.count[index]).toBe(literal.count[index]);
        const error = Math.abs(model.deviation[index] - literal.deviation[index]);
        worst = Math.max(worst, error / getBound(literal.deviation[index], quantum));
        expect(error).toBeLessThanOrEqual(getBound(literal.deviation[index], quantum));
      }
      expect(worst).toBeLessThanOrEqual(1);
    });

    it(`keeps the 64-bit difference exact: ${testCase.name}`, () => {
      const {width, height, radius} = testCase;
      const random = createRandom(width * 7 + height * 3 + radius);
      const elevation = new Float32Array(width * height);
      const validity = new Uint8Array(width * height);
      for (let index = 0; index < elevation.length; index++) {
        // Multiples of the quantum, so the quantized values are the elevations.
        elevation[index] =
          Math.round((testCase.base + (random() - 0.5) * testCase.amplitude) * 256) / 256;
        validity[index] = random() < testCase.holeFraction ? 0 : 1;
      }
      const quantum = GPU_RELIEF_DEFAULT_QUANTUM;
      const model = modelReliefDeviation(elevation, validity, width, height, radius, quantum);
      for (let row = 0; row < height; row++) {
        for (let column = 0; column < width; column++) {
          const index = row * width + column;
          if (!validity[index]) continue;
          let difference = 0n;
          for (let dy = -radius; dy <= radius; dy++) {
            for (let dx = -radius; dx <= radius; dx++) {
              const sample =
                Math.min(Math.max(row + dy, 0), height - 1) * width +
                Math.min(Math.max(column + dx, 0), width - 1);
              if (validity[sample]) {
                difference +=
                  BigInt(Math.round(elevation[index] / quantum)) -
                  BigInt(Math.round(elevation[sample] / quantum));
              }
            }
          }
          expect(model.difference[index]).toBe(difference);
        }
      }
    });
  }

  it('literal clamp semantics equal the RVT integral-image oracle', () => {
    const width = 9;
    const height = 7;
    const random = createRandom(3);
    const elevation = Float32Array.from({length: width * height}, () => 100 + random() * 10);
    const validity = new Uint8Array(width * height).fill(1);
    const literal = computeLiteralReliefDeviation(elevation, validity, width, height, 3);
    const oracle = computeMeanFilterRVT(elevation, width, height, 3);
    for (let index = 0; index < elevation.length; index++) {
      expect(literal.deviation[index]).toBeCloseTo(elevation[index] - oracle[index], 9);
    }
  });

  it('treats offsets beyond the quantized relief range as invalid', () => {
    const model = modelReliefDeviation(
      Float32Array.from([1, 40000, 3, 4]),
      [1, 1, 1, 1],
      2,
      2,
      1,
      GPU_RELIEF_DEFAULT_QUANTUM
    );
    expect(Number.isNaN(model.deviation[1])).toBe(true);
    // Edge weights 2 x 2, 1 x 2, 2 x 1, 1 x 1 minus the invalid weight-2 sample.
    expect(model.count[0]).toBe(7);
  });
});

function createSimple(graph: GPUCommandGraph, instance: number, maximumRadius: number, size = 30) {
  return new GPUSimpleLocalRelief({
    grid: makeGPURasterMetadata({width: 6, height: size / 6}),
    maximumRadius,
    elevation: createBand(graph, `elevation-${instance}`, size),
    parameters: createTransientView(graph, `settings-${instance}`, 'float32', 4),
    output: {
      relief: createTransientView(graph, `relief-${instance}`, 'float32', size),
      outputValidity: createTransientView(graph, `validity-${instance}`, 'uint32', size)
    }
  });
}

describe('relief summed-area kernels', () => {
  it('SLRM builds one table and one deviation kernel that parse and fit 8 bindings', () => {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    const nodes = createSimple(graph, 1, 3).getCommandNodes(graph);
    const ids = nodes.map(node => node.id);
    expect(ids[ids.length - 1]).toBe('simple-local-relief-mean-deviation');
    expect(ids.some(id => /summed-area-.*square/.test(id))).toBe(false);
    expect(ids.some(id => /mean-(rows|columns)/.test(id))).toBe(false);
    const sources = getParsedKernelSources(nodes);
    const kernel = sources.get('simple-local-relief-mean-deviation');
    expect(kernel).toBeDefined();
    expect(getStorageBindingCount(kernel as string)).toBeLessThanOrEqual(8);
    expect(kernel).toContain('overhangLeft');
    expect(kernel).toContain('anchor[anchorOffset]');
    expect(ids.indexOf('simple-local-relief-anchor-select')).toBeLessThan(
      ids.indexOf('simple-local-relief-summed-area-quantize')
    );
  });

  it('SLRM keeps one pipeline set for every maximum radius (rule 12)', () => {
    const device = createNullWebGPUDevice();
    const counts = [1, 3, 20, 200].map((maximumRadius, instance) => {
      const graph = new GPUCommandGraph(device);
      return getGPUCommandNodePipelineKeys(
        createSimple(graph, instance, maximumRadius).getCommandNodes(graph)
      ).size;
    });
    expect(new Set(counts).size).toBe(1);
    // The per-frame radius is read from the settings view: no loop index or radius reaches the
    // WGSL of the deviation kernel except the compile-time maximum.
    const graph = new GPUCommandGraph(device);
    const source = getParsedKernelSources(createSimple(graph, 9, 5).getCommandNodes(graph)).get(
      'simple-local-relief-mean-deviation'
    ) as string;
    expect(source).toContain('radiusSettings[radiusSettingsOffset + 1u]');
    expect(source).not.toMatch(/for \(var \w+ = -?(RADIUS|radius)/);
  });

  it('MSRM builds the table once for both filters', () => {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    const contributor = new GPUMultiScaleRelief({
      grid: makeGPURasterMetadata({width: 6, height: 5}),
      resolution: 1,
      featureMinimum: 3,
      featureMaximum: 40,
      scalingFactor: 2,
      elevation: createBand(graph, 'elevation', 30),
      parameters: createTransientView(graph, 'settings', 'float32', 4),
      output: {relief: createTransientView(graph, 'relief', 'float32', 30)}
    });
    const nodes = contributor.getCommandNodes(graph);
    const ids = nodes.map(node => node.id);
    expect(ids.filter(id => /summed-area-quantize$/.test(id))).toHaveLength(1);
    expect(ids.filter(id => /-(first|last)-deviation$/.test(id))).toHaveLength(2);
    const sources = getParsedKernelSources(nodes);
    for (const id of ['multi-scale-relief-last-deviation', 'multi-scale-relief-first-deviation']) {
      expect(sources.has(id)).toBe(true);
    }
  });

  it('validates quantum and radius, and falls back above the table cell cap', () => {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    const create = (overrides: Record<string, unknown>) =>
      new GPUSimpleLocalRelief({
        grid: makeGPURasterMetadata({width: 6, height: 5}),
        maximumRadius: 3,
        elevation: createBand(graph, `e-${Math.random()}`, 30),
        parameters: createTransientView(graph, `s-${Math.random()}`, 'float32', 4),
        output: {relief: createTransientView(graph, `r-${Math.random()}`, 'float32', 30)},
        ...overrides
      });
    expect(() => create({quantum: 0.3})).toThrow(/quantum/);
    expect(() => create({quantum: 0})).toThrow(/quantum/);
    // The radius bound depends on the path, so it is checked at declaration.
    expect(() =>
      create({maximumRadius: GPU_RELIEF_SUMMED_AREA_MAXIMUM_RADIUS + 1}).getCommandNodes(graph)
    ).toThrow(/radius must be at most/);
    expect(() =>
      create({
        maximumRadius: GPU_RELIEF_SUMMED_AREA_MAXIMUM_RADIUS,
        quantum: 0.5
      }).getCommandNodes(graph)
    ).not.toThrow();

    const width = 3400;
    const height = 3400;
    const cellCount = width * height;
    const largeGraph = new GPUCommandGraph(createNullWebGPUDevice());
    const large = new GPUSimpleLocalRelief({
      grid: makeGPURasterMetadata({width, height}),
      maximumRadius: 60000,
      elevation: createBand(largeGraph, 'large-elevation', cellCount),
      parameters: createTransientView(largeGraph, 'large-settings', 'float32', 4),
      output: {relief: createTransientView(largeGraph, 'large-relief', 'float32', cellCount)}
    });
    const ids = large.getCommandNodes(largeGraph).map(node => node.id);
    expect(ids.slice(-2)).toEqual([
      'simple-local-relief-mean-rows',
      'simple-local-relief-mean-columns'
    ]);
  });
});
