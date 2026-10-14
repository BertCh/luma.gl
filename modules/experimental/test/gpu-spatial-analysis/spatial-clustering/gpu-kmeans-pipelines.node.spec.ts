// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {WgslReflect} from 'wgsl_reflect';
import {expect, it} from 'vitest';
import {
  GPUKMeans,
  GPU_KMEANS_PARAMETER_LENGTH,
  type GPUKMeansProps
} from '../../../src/gpu-spatial-analysis/spatial-clustering/index';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys
} from '../../utils/gpu-contributor-test-utils';

const ROWS = 500;

type KMeansShape = Partial<Pick<GPUKMeansProps, 'k' | 'maximumIterations' | 'restartCount'>>;

function getKMeansNodes(shape: KMeansShape) {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `kmeans-view-${serial++}`, format, length);
  const k = shape.k ?? 3;
  return new GPUKMeans({
    positions: view('float32x2', ROWS),
    k,
    maximumIterations: shape.maximumIterations ?? 2,
    restartCount: shape.restartCount ?? 1,
    parameters: view('uint32', GPU_KMEANS_PARAMETER_LENGTH),
    output: {
      labels: view('uint32', ROWS),
      centers: view('float32x2', k),
      inertia: view('float32', 1),
      status: {iterationCount: view('uint32', 1), converged: view('uint32', 1)}
    }
  }).getCommandNodes(graph);
}

it('GPUKMeans pipeline count does not grow with k, maximumIterations or restartCount (rule 12)', () => {
  const distinct = (shape: KMeansShape) =>
    getGPUCommandNodePipelineKeys(getKMeansNodes(shape)).size;
  // k is a size as well as the seeding round count, so compare k at equal size-like constants by
  // checking that one more k adds no per-round variant: the counts at k = 3 and k = 4 are equal.
  expect(distinct({k: 4})).toBe(distinct({k: 3}));
  expect(distinct({maximumIterations: 3})).toBe(distinct({maximumIterations: 2}));
  expect(distinct({restartCount: 3})).toBe(distinct({restartCount: 2}));
  // More rounds still add nodes; only the variant count is fixed.
  expect(getGPUCommandNodeKernelSources(getKMeansNodes({k: 4})).length).toBeGreaterThan(
    getGPUCommandNodeKernelSources(getKMeansNodes({k: 3})).length
  );
});

it('GPUKMeans gates every Lloyd kernel on the convergence word (rule 13)', () => {
  const nodes = getKMeansNodes({maximumIterations: 3}) as readonly {
    id: string;
    condition?: {source: string; mode: string; byteOffset: number};
  }[];
  for (let iteration = 0; iteration < 3; iteration++) {
    const offsets: number[] = [];
    for (const stage of ['assign', 'partial', 'update', 'relocate']) {
      const node = nodes.find(
        candidate => candidate.id === `kmeans-iteration-${iteration}-${stage}`
      );
      expect(node?.condition, stage).toMatchObject({source: 'gpu', mode: 'indirect'});
      offsets.push(node!.condition!.byteOffset);
    }
    expect(new Set(offsets).size).toBe(4);
    // The converge check writes the gate, so it must not read it as an indirect argument.
    const converge = nodes.find(node => node.id === `kmeans-iteration-${iteration}-converge`);
    expect(converge?.condition).toBeUndefined();
  }
  for (const id of ['kmeans-final-assign', 'kmeans-inertia', 'kmeans-begin']) {
    expect(nodes.find(node => node.id === id)?.condition, id).toBeUndefined();
  }
});

it('GPUKMeans scores restarts with a two-stage reduction (rule 14) and parses as WGSL', () => {
  const nodes = getKMeansNodes({restartCount: 2});
  const ids = nodes.map(node => node.id);
  for (const restart of [0, 1]) {
    expect(ids.indexOf(`kmeans-restart-${restart}-score-partial`)).toBeLessThan(
      ids.indexOf(`kmeans-restart-${restart}-score`)
    );
  }
  const sources = getGPUCommandNodeKernelSources(nodes);
  for (const {id, source} of sources) {
    expect(() => new WgslReflect(source), id).not.toThrow();
    // No seeding round, restart index, seed or tolerance is baked into the text: the seed and
    // tolerance are per-frame parameters words, so every seed shares the same pipelines.
    expect(source, id).not.toMatch(
      /ROUND_SEED|FIRST_ROUND|FIRST_RESTART|const ROUND:|const SEED|const TOLERANCE/
    );
  }
});
