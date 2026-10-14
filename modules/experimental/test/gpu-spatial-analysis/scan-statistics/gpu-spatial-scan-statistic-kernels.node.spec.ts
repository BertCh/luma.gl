// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {describe, expect, test} from 'vitest';
import {WgslReflect} from 'wgsl_reflect';
import type {GPUSpatialContext} from '../../../src/analysis-core/contracts/spatial-context';
import {GPUSpatialScanStatistic} from '../../../src/gpu-spatial-analysis/scan-statistics/gpu-spatial-scan-statistic';
import {PhiloxStream} from '../../../src/gpu-spatial-analysis/permutation-inference/permutation-random';
import {PHILOX_PURPOSES} from '../../../src/utils/philox-random';
import {
  getGPUSpatialScanStatisticLayout,
  type GPUSpatialScanModel
} from '../../../src/gpu-spatial-analysis/scan-statistics/scan-statistic-parameters';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  CORE_PRIVATE_MEMORY_LIMIT_BYTES,
  getGPUCommandNodePipelineKeys,
  getPrivateMemoryBytes,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';

let serial = 0;

function createScan(
  graph: GPUCommandGraph,
  model: GPUSpatialScanModel,
  options: {
    spatialContext?: GPUSpatialContext;
    zones?: number;
    timeBuckets?: number;
    maximumClusters?: number;
    maximumWindowZones?: number;
    maximumPermutations?: number;
    batchReplicates?: number;
  } = {}
): GPUSpatialScanStatistic {
  const zones = options.zones ?? 40;
  const timeBuckets = options.timeBuckets ?? (model === 'space-time-permutation' ? 3 : 2);
  const cells = zones * timeBuckets;
  const maximumPermutations = options.maximumPermutations ?? 15;
  const maximumClusters = options.maximumClusters ?? 4;
  const view = <Format extends 'float32' | 'uint32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `scan-view-${serial++}`, format, length);
  return new GPUSpatialScanStatistic({
    positions: view('float32x2', zones),
    spatialContext: options.spatialContext,
    model,
    cases: view('uint32', cells),
    ...(model === 'poisson' ? {baseline: view('float32', cells)} : {}),
    ...(model === 'bernoulli' ? {controls: view('uint32', cells)} : {}),
    timeBuckets,
    maximumWindowZones: options.maximumWindowZones,
    maximumPermutations,
    batchReplicates: options.batchReplicates,
    maximumClusters,
    parameters: view('uint32', 8),
    output: {
      clusterIndices: view('uint32', maximumClusters * 4),
      clusterStatistics: view('float32', maximumClusters * 8),
      statistics: view('float32', maximumPermutations + 1),
      summary: view('uint32', 4),
      zoneStatistics: view('float32', zones)
    }
  });
}

describe('GPUSpatialScanStatistic kernels', () => {
  test.each([
    'meters',
    'kilometers'
  ] as const)('great-circle kernels use haversine keys and decode squared radii in %s', units => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    try {
      const kernels = getGPUCommandNodeKernelSources(
        createScan(graph, 'poisson', {
          spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle', units}
        }).getCommandNodes(graph)
      );
      const nearest = kernels.find(kernel => kernel.id.endsWith('-nearest'))!.source;
      const evaluate = kernels.find(kernel => kernel.id.endsWith('-evaluate'))!.source;
      expect(nearest).toContain('return geodesicHaversineTerm(query, position);');
      expect(nearest).toContain(
        'bitcast<u32>(getSeparationKey(vec2f(zoneX, zoneY), vec2f(centerX, centerY)))'
      );
      expect(evaluate).toContain('bitcast<u32>(getSquaredMeters(distances[');
      expect(evaluate).toContain('return meters * meters;');
      const sphereRadius = Number(nearest.match(/const SPHERE_RADIUS: f32 = ([^;]+);/)![1]);
      expect(sphereRadius).toBe(Math.fround(units === 'kilometers' ? 6371.0088 : 6371008.8));
      for (const kernel of kernels) {
        expect(() => new WgslReflect(kernel.source), kernel.id).not.toThrow();
        expect(
          (kernel.source.match(/fn geodesicHaversineTerm\(/g) ?? []).length
        ).toBeLessThanOrEqual(1);
        expect(getStorageBindingCount(kernel.source), kernel.id).toBeLessThanOrEqual(8);
        expect(getPrivateMemoryBytes(kernel.source), kernel.id).toBeLessThanOrEqual(
          CORE_PRIVATE_MEMORY_LIMIT_BYTES
        );
      }
    } finally {
      device.destroy();
    }
  });

  test.each<GPUSpatialScanModel>([
    'poisson',
    'bernoulli',
    'space-time-permutation'
  ])('%s kernels parse and stay within core private memory and bindings', model => {
    for (const maximumWindowZones of [undefined, 9]) {
      const device = createNullWebGPUDevice();
      const graph = new GPUCommandGraph(device);
      const kernels = getGPUCommandNodeKernelSources(
        createScan(graph, model, {maximumWindowZones}).getCommandNodes(graph)
      );
      expect(kernels.some(kernel => kernel.id.endsWith('-nearest'))).toBe(true);
      for (const kernel of kernels) {
        expect(() => new WgslReflect(kernel.source), kernel.id).not.toThrow();
        // wgsl_reflect accepts WGSL reserved words that Tint rejects ('target' broke `nearest`).
        expect(kernel.source, kernel.id).not.toMatch(
          /\b(let|var)\s+(target|filter|module|self|super)\b/
        );
        expect(getStorageBindingCount(kernel.source), kernel.id).toBeLessThanOrEqual(8);
        expect(getPrivateMemoryBytes(kernel.source), kernel.id).toBeLessThanOrEqual(
          CORE_PRIVATE_MEMORY_LIMIT_BYTES
        );
      }
      device.destroy();
    }
  });

  test('getInferenceMetadata derives the legend from the packed settings (rule 19)', () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const scan = createScan(graph, 'poisson');
    expect(
      scan.getInferenceMetadata({seed: 3, permutationCount: 9, rateDirection: 'low'})
    ).toMatchObject({alternative: 'lesser', permutationCount: 9, seed: 3, multipleTesting: 'none'});
    // The kernels clamp the packed count to maximumPermutations (15).
    expect(scan.getInferenceMetadata({seed: 0, permutationCount: 99}).permutationCount).toBe(15);
    expect(
      scan.getInferenceMetadata({seed: 0, permutationCount: 1, rateDirection: 'both'}).alternative
    ).toBe('two-sided');
    // Validated exactly as the packer validates.
    expect(() => scan.getInferenceMetadata({seed: -1, permutationCount: 9})).toThrow();
    expect(() => scan.getInferenceMetadata({seed: 1, permutationCount: 0})).toThrow();
    device.destroy();
  });

  test('cluster rounds compile the same pipelines at N and N + 1 clusters (rule 12)', () => {
    const count = (maximumClusters: number) => {
      const device = createNullWebGPUDevice();
      const graph = new GPUCommandGraph(device);
      const keys = getGPUCommandNodePipelineKeys(
        createScan(graph, 'poisson', {maximumClusters}).getCommandNodes(graph)
      );
      device.destroy();
      return keys.size;
    };
    expect(count(3)).toBe(count(4));
    expect(count(4)).toBe(count(8));
  });

  test('round kernels after round-select are gated by the GPU-written round dispatch (rule 13)', () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const nodes = createScan(graph, 'poisson', {maximumClusters: 3}).getCommandNodes(graph);
    const gated = nodes.filter(node => node.condition && /-round-/.test(node.id));
    expect(gated.map(node => node.id.replace(/-\d+$/, '')).sort()).toEqual(
      Array.from({length: 3}, () => [
        'scan-statistic-round-mask',
        'scan-statistic-round-members',
        'scan-statistic-round-partials'
      ])
        .flat()
        .sort()
    );
    for (const node of gated) {
      expect(node.condition).toMatchObject({source: 'gpu', mode: 'indirect'});
    }
    const roundGate = (gated[0].condition as {buffer: unknown}).buffer;
    // The writers of the gate (select-init arms, round-select disarms) are never gated themselves.
    const writers = nodes.filter(node =>
      (node.resources ?? []).some(
        resource =>
          'buffer' in resource &&
          resource.usage === 'storage-read-write' &&
          (resource.buffer as {buffer?: unknown}).buffer === roundGate
      )
    );
    expect(writers.map(node => node.id.replace(/-\d+$/, ''))).toEqual([
      'scan-statistic-select-init',
      'scan-statistic-round-select',
      'scan-statistic-round-select',
      'scan-statistic-round-select'
    ]);
    expect(writers.every(node => !node.condition)).toBe(true);
    device.destroy();
  });
});

const DRAW_KERNEL: Record<GPUSpatialScanModel, string> = {
  poisson: 'scatter',
  bernoulli: 'draw-labels',
  'space-time-permutation': 'draw-times'
};

describe('GPUSpatialScanStatistic replicate batches', () => {
  test.each<GPUSpatialScanModel>([
    'poisson',
    'bernoulli',
    'space-time-permutation'
  ])('%s: observed evaluate once, then four nodes per batch in order', model => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const scan = createScan(graph, model, {batchReplicates: 4});
    // P = 15 in batches of 4: ceil(15 / 4) = 4 batches.
    expect(scan.batchReplicates).toBe(4);
    expect(scan.batchCount).toBe(4);
    const ids = scan.getCommandNodes(graph).map(node => node.id.replace('scan-statistic-', ''));
    const evaluate = ids.indexOf('evaluate');
    expect(evaluate).toBeGreaterThan(ids.indexOf('window-limit'));
    expect(ids.filter(id => id === 'evaluate')).toHaveLength(1);
    expect(ids.slice(evaluate + 1, evaluate + 1 + 16)).toEqual(
      Array.from({length: 4}, (_, batch) => [
        `batch-control-${batch}`,
        `batch-clear-${batch}`,
        `${DRAW_KERNEL[model]}-${batch}`,
        `batch-evaluate-${batch}`
      ]).flat()
    );
    expect(ids[evaluate + 17]).toBe('select-init');
    device.destroy();
  });

  test('derives the batch size from the count table budget and caps it', () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    // Few cells: capped at 4096 replicates per batch; 2^25 / 80 words would allow far more.
    const small = createScan(graph, 'poisson', {maximumPermutations: 9999});
    expect(small.batchReplicates).toBe(4096);
    expect(small.batchCount).toBe(3);
    // A count table of (R + 1) * cells words fits 2^25 words: 512 zones * 32 buckets = 2^14 cells.
    const large = createScan(graph, 'poisson', {
      zones: 512,
      timeBuckets: 32,
      maximumPermutations: 20000,
      maximumWindowZones: 16
    });
    expect(large.batchReplicates).toBe(2 ** 11 - 1);
    expect((large.batchReplicates + 1) * 512 * 32).toBeLessThanOrEqual(2 ** 25);
    expect(large.batchCount).toBe(Math.ceil(20000 / large.batchReplicates));
    // A batch never exceeds maximumPermutations: one batch, no wasted slots.
    expect(createScan(graph, 'poisson').batchReplicates).toBe(15);
    device.destroy();
  });

  test('validates the limits: the old replicate-table cap is gone', () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    // (P + 1) * cells = 99_999_001 * ... would have exceeded 2^25 words in one pass.
    expect(createScan(graph, 'poisson', {maximumPermutations: 99999}).batchCount).toBe(25);
    expect(() => createScan(graph, 'poisson', {maximumPermutations: 100000})).toThrow(
      /maximumPermutations/
    );
    expect(() => createScan(graph, 'poisson', {batchReplicates: 0})).toThrow(/batchReplicates/);
    // Too many cells for any batch: 16384 zones * 32 buckets gives batches of 63 replicates, and
    // 99999 of them would need 1588 > 1024 unrolled batches.
    expect(() =>
      createScan(graph, 'poisson', {
        zones: 16384,
        timeBuckets: 32,
        maximumWindowZones: 4,
        maximumPermutations: 99999
      })
    ).toThrow(/1588 batches of 63 replicates, more than 1024/);
    expect(
      createScan(graph, 'poisson', {
        zones: 16384,
        timeBuckets: 32,
        maximumWindowZones: 4,
        maximumPermutations: 60000
      }).batchCount
    ).toBeLessThanOrEqual(1024);
    device.destroy();
  });

  test.each<GPUSpatialScanModel>([
    'poisson',
    'bernoulli',
    'space-time-permutation'
  ])('%s: batches compile the same pipelines at N and N + 1 batches (rule 12)', model => {
    const count = (batchReplicates: number) => {
      const device = createNullWebGPUDevice();
      const graph = new GPUCommandGraph(device);
      const scan = createScan(graph, model, {batchReplicates});
      const nodes = scan.getCommandNodes(graph);
      const batchKeys = getGPUCommandNodePipelineKeys(nodes, {
        idPattern: /-batch-|-scatter-|-draw-/
      });
      const keys = getGPUCommandNodePipelineKeys(nodes);
      device.destroy();
      return {batches: scan.batchCount, size: keys.size, batchSize: batchKeys.size};
    };
    // P = 15: batches of 5, 4 and 3 give 3, 4 and 5 batches.
    const [three, four, five] = [5, 4, 3].map(count);
    expect([three.batches, four.batches, five.batches]).toEqual([3, 4, 5]);
    expect(four.size).toBe(three.size);
    expect(five.size).toBe(four.size);
    // Control, clear, draw and evaluate: one pipeline each, whatever the batch count.
    expect(five.batchSize).toBe(4);
  });

  test('batch kernels are gated by the GPU-written batch dispatch (rule 13)', () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const nodes = createScan(graph, 'poisson', {batchReplicates: 4}).getCommandNodes(graph);
    const gated = nodes.filter(node => /-batch-clear-|-scatter-|-batch-evaluate-/.test(node.id));
    expect(gated).toHaveLength(12);
    for (const node of gated) {
      expect(node.condition, node.id).toMatchObject({source: 'gpu', mode: 'indirect'});
    }
    const gate = (gated[0].condition as {buffer: unknown}).buffer;
    const writers = nodes.filter(node =>
      (node.resources ?? []).some(
        resource =>
          'buffer' in resource &&
          resource.usage === 'storage-read-write' &&
          (resource.buffer as {buffer?: unknown}).buffer === gate
      )
    );
    // The writers of the gate are the ungated controls, one per batch.
    expect(writers.map(node => node.id.replace(/-\d+$/, ''))).toEqual(
      Array.from({length: 4}, () => 'scan-statistic-batch-control')
    );
    expect(writers.every(node => !node.condition)).toBe(true);
    device.destroy();
  });

  test.each<GPUSpatialScanModel>([
    'poisson',
    'bernoulli',
    'space-time-permutation'
  ])('%s: a replicate draws from its global index, whatever the batch size', model => {
    const getDrawKernel = (batchReplicates: number) => {
      const device = createNullWebGPUDevice();
      const graph = new GPUCommandGraph(device);
      const nodes = createScan(graph, model, {batchReplicates}).getCommandNodes(graph);
      const kernel = getGPUCommandNodeKernelSources(nodes).find(candidate =>
        candidate.id.endsWith(`-${DRAW_KERNEL[model]}-0`)
      )!;
      device.destroy();
      return kernel.source;
    };
    // Apart from the batch size constant (used by the control kernel only) the draw kernel text is
    // the same, and the stream is keyed by the replicate: the running batch base plus the local slot.
    const normalize = (source: string) => source.replace(/const BATCH_REPLICATES: u32 = \d+u;/, '');
    const reference = normalize(getDrawKernel(15));
    for (const batchReplicates of [1, 4, 7]) {
      expect(normalize(getDrawKernel(batchReplicates))).toBe(reference);
    }
    expect(reference).toContain('let replicate = totals[totalsOffset + 4u] + localReplicate;');
    expect(reference).toMatch(/createPhiloxStream\(readSeedKey\(\), replicate, /);
    expect(reference).not.toMatch(/BATCH_REPLICATES/);
  });

  test('CPU mirror: every layout draws the same words for the same replicate', () => {
    const permutations = 15;
    const seedKey: [number, number] = [12345, 0];
    const readBatchedDraws = (batchReplicates: number) => {
      // The control kernel's cursor: base = batch * R; the draw kernel's replicate = base + local.
      const draws = new Map<number, number[]>();
      let nextBase = 0;
      for (let batch = 0; batch < Math.ceil(permutations / batchReplicates); batch++) {
        const base = nextBase;
        nextBase += batchReplicates;
        for (let local = 0; local < batchReplicates; local++) {
          const replicate = base + local;
          if (replicate >= permutations) continue;
          const stream = new PhiloxStream(
            seedKey,
            replicate,
            0,
            PHILOX_PURPOSES.spatialScanStatistic.id
          );
          draws.set(replicate, [stream.nextUint32(), stream.nextUint32(), stream.nextUint32()]);
        }
      }
      return draws;
    };
    const reference = readBatchedDraws(permutations);
    expect(reference.size).toBe(permutations);
    for (const batchReplicates of [1, 2, 4, 7, 16]) {
      expect(readBatchedDraws(batchReplicates)).toEqual(reference);
    }
    // Distinct replicates draw distinct streams.
    expect(new Set([...reference.values()].map(words => words[0])).size).toBe(permutations);
  });
});

/**
 * Mirror of the `nearest` kernel: radix select of the LIST-th smallest key, gather in zone order,
 * four stable 8-bit LSD passes. Checked against a direct (key, zone) sort so the digit, rank and
 * tie bookkeeping of the WGSL is exercised on the CPU.
 */
function selectNearestZones(keys: Uint32Array, list: number): number[] {
  const zones = keys.length;
  const histogram = new Uint32Array(256);
  let threshold = 0xffffffff;
  let equalCount = list;
  if (list < zones) {
    let prefix = 0;
    let prefixMask = 0;
    let rank = list - 1;
    for (let level = 0; level < 4; level++) {
      const shift = 24 - 8 * level;
      histogram.fill(0);
      for (let zone = 0; zone < zones; zone++) {
        if ((keys[zone] & prefixMask) >>> 0 === prefix) histogram[(keys[zone] >>> shift) & 255]++;
      }
      let digit = 0;
      while (digit !== 255 && rank >= histogram[digit]) rank -= histogram[digit++];
      prefix = (prefix | (digit << shift)) >>> 0;
      prefixMask = (prefixMask | (255 << shift)) >>> 0;
    }
    threshold = prefix;
    equalCount = rank + 1;
  }
  let rowKeys: number[] = [];
  let rowZones: number[] = [];
  let equalTaken = 0;
  for (let zone = 0; zone < zones && rowKeys.length < list; zone++) {
    let take = keys[zone] < threshold;
    if (!take && keys[zone] === threshold && equalTaken < equalCount) {
      take = true;
      equalTaken++;
    }
    if (take) {
      rowKeys.push(keys[zone]);
      rowZones.push(zone);
    }
  }
  for (let shift = 0; shift < 32; shift += 8) {
    histogram.fill(0);
    for (const key of rowKeys) histogram[(key >>> shift) & 255]++;
    let running = 0;
    for (let bin = 0; bin < 256; bin++) {
      const count = histogram[bin];
      histogram[bin] = running;
      running += count;
    }
    const nextKeys = new Array<number>(rowKeys.length);
    const nextZones = new Array<number>(rowKeys.length);
    for (let slot = 0; slot < rowKeys.length; slot++) {
      const target = histogram[(rowKeys[slot] >>> shift) & 255]++;
      nextKeys[target] = rowKeys[slot];
      nextZones[target] = rowZones[slot];
    }
    rowKeys = nextKeys;
    rowZones = nextZones;
  }
  return rowZones;
}

function getSquaredDistanceKeys(positions: Float32Array, center: number): Uint32Array {
  const zones = positions.length / 2;
  const squared = new Float32Array(zones);
  for (let zone = 0; zone < zones; zone++) {
    const dx = Math.fround(positions[zone * 2] - positions[center * 2]);
    const dy = Math.fround(positions[zone * 2 + 1] - positions[center * 2 + 1]);
    squared[zone] = Math.fround(Math.fround(dx * dx) + Math.fround(dy * dy));
  }
  return new Uint32Array(squared.buffer);
}

describe('GPUSpatialScanStatistic nearest-zone selection', () => {
  const lattice = new Float32Array(
    Array.from({length: 2 * 15 * 15}, (_, index) =>
      index % 2 === 0 ? (Math.floor(index / 2) % 15) * 10 : Math.floor(index / 30) * 10
    )
  );
  let state = 12345;
  const random = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 2 ** 32;
  };
  // Random points with exact duplicates, so equal keys occur at the selection threshold.
  const scattered = new Float32Array(2 * 300);
  for (let zone = 0; zone < 300; zone++) {
    const source = zone % 7 === 3 ? zone - 1 : zone;
    scattered[zone * 2] = source === zone ? Math.fround(random() * 1000) : scattered[source * 2];
    scattered[zone * 2 + 1] =
      source === zone ? Math.fround(random() * 1000) : scattered[source * 2 + 1];
  }
  test.each([
    ['lattice', lattice],
    ['scattered with duplicates', scattered]
  ] as const)('%s: matches the (squared distance, zone) order at every capacity', (_, positions) => {
    const zones = positions.length / 2;
    for (const capacity of [undefined, 1, 7, 8, 33, zones - 2]) {
      const {listLength} = getGPUSpatialScanStatisticLayout(zones, capacity);
      for (const center of [0, 17, Math.floor(zones / 2), zones - 1]) {
        const keys = getSquaredDistanceKeys(positions, center);
        const expected = Array.from({length: zones}, (__, zone) => zone)
          .sort((left, right) => keys[left] - keys[right] || left - right)
          .slice(0, listLength);
        expect(
          selectNearestZones(keys, listLength),
          `capacity ${capacity} center ${center}`
        ).toEqual(expected);
      }
    }
  });
});
