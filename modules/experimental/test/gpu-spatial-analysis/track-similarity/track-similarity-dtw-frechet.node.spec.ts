// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUTrackSimilarity,
  type GPUTrackSimilarityProps
} from '../../../src/gpu-spatial-analysis/track-similarity/index';
import {
  createNullWebGPUDevice,
  getGPUCommandNodePipelineKeys,
  getParsedKernelSources,
  getPrivateMemoryBytes,
  getStorageBindingCount,
  getWorkgroupMemoryBytes
} from '../../utils/gpu-contributor-test-utils';
import {DYNAMIC_PROGRAM_FIXTURE} from './track-similarity-dtw-frechet-harness';

let serial = 0;

function createProps(
  graph: GPUCommandGraph,
  pairCount: number,
  overrides: Partial<GPUTrackSimilarityProps> = {}
): GPUTrackSimilarityProps {
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `view-${serial++}`, format, length);
  return {
    linesA: {kind: 'lines', positions: view('float32x2', 64), lineOffsets: view('uint32', 5)},
    pairA: view('uint32', pairCount),
    pairB: view('uint32', pairCount),
    frechetEpsilon: view('float32', 1),
    dynamicTimeWarpingBandRadius: 3,
    maximumFrechetVertexCount: 2048,
    output: {
      frechet: view('float32', pairCount),
      frechetWithin: view('uint32', pairCount),
      dynamicTimeWarping: view('float32', pairCount)
    },
    ...overrides
  };
}

it('GPUTrackSimilarity decision and DTW kernels fit the core limits at maximal options', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const nodes = new GPUTrackSimilarity(createProps(graph, 16)).getCommandNodes(graph);
  const ids = nodes.map(node => node.id);
  expect(ids).toContain('track-similarity-dynamic-time-warping');
  expect(ids).toContain('track-similarity-pair-ranges');
  expect(ids).toContain('track-similarity-frechet-decision');
  for (const [kernelId, source] of getParsedKernelSources(nodes)) {
    expect(getStorageBindingCount(source), kernelId).toBeLessThanOrEqual(8);
    expect(getWorkgroupMemoryBytes(source), kernelId).toBeLessThanOrEqual(16384);
    expect(getPrivateMemoryBytes(source), kernelId).toBeLessThanOrEqual(8192);
  }
  // Rule 12: the pair capacity is a size; it adds no kernel variant.
  const other = new GPUCommandGraph(createNullWebGPUDevice());
  expect(
    getGPUCommandNodePipelineKeys(
      new GPUTrackSimilarity(createProps(other, 32)).getCommandNodes(other)
    ).size
  ).toBe(getGPUCommandNodePipelineKeys(nodes).size);
});

it('GPUTrackSimilarity validates the decision and band props', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const props = createProps(graph, 4);
  expect(() => new GPUTrackSimilarity({...props, frechetEpsilon: undefined})).toThrow(
    /frechetWithin and frechetEpsilon/
  );
  expect(
    () =>
      new GPUTrackSimilarity({
        ...props,
        output: {...props.output, frechetWithin: undefined}
      })
  ).toThrow(/frechetWithin and frechetEpsilon/);
  expect(() => new GPUTrackSimilarity({...props, dynamicTimeWarpingBandRadius: -1})).toThrow(
    /dynamicTimeWarpingBandRadius/
  );
  expect(() => new GPUTrackSimilarity({...props, dynamicTimeWarpingBandRadius: 1.5})).toThrow(
    /dynamicTimeWarpingBandRadius/
  );
  // Either new output alone is a valid request.
  expect(
    () =>
      new GPUTrackSimilarity({
        ...props,
        frechetEpsilon: undefined,
        output: {dynamicTimeWarping: props.output.dynamicTimeWarping}
      })
  ).not.toThrow();
});

it('track-similarity DTW and Frechet fixture is self-consistent', () => {
  for (const entry of [...DYNAMIC_PROGRAM_FIXTURE.planar, ...DYNAMIC_PROGRAM_FIXTURE.greatCircle]) {
    // The warping cost is at least the Frechet distance (its path holds the bottleneck pair).
    expect(entry.dynamicTimeWarping, entry.name).toBeGreaterThanOrEqual(
      entry.frechet * (1 - 1e-12)
    );
    for (const band of entry.banded) {
      // A band only removes paths.
      expect(band.dynamicTimeWarping, `${entry.name} ${band.radius}`).toBeGreaterThanOrEqual(
        entry.dynamicTimeWarping * (1 - 1e-12)
      );
    }
    for (const decision of entry.decisions) {
      expect(decision.within, `${entry.name} ${decision.epsilon}`).toBe(
        entry.frechet <= decision.epsilon ? 1 : 0
      );
    }
  }
});

it('decision-only work prunes endpoints before coupling and boxes before reachable-cell distances', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const props = createProps(graph, 16);
  const nodes = new GPUTrackSimilarity({
    ...props,
    output: {frechetWithin: props.output.frechetWithin}
  }).getCommandNodes(graph);
  const sources = getParsedKernelSources(nodes);
  expect(sources.has('track-similarity-frechet')).toBe(false);
  expect(sources.has('track-similarity-dynamic-time-warping')).toBe(false);
  const decision = sources.get('track-similarity-frechet-decision')!;
  expect(decision.indexOf('if (decision == 2u)')).toBeLessThan(decision.indexOf('var coupled ='));
  expect(decision.indexOf('let minimumGap')).toBeLessThan(decision.indexOf('var finalValue'));
  expect(decision).toContain('predecessor && coupledMetric');
  expect(decision).toContain('deadDiagonals >= 2u');
  expect(decision).toContain('previousBoundaryCount == 0u');
  expect(getWorkgroupMemoryBytes(decision)).toBeLessThanOrEqual(16384);
});
