// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUKMeans,
  GPUSpatialClustering
} from '../../../src/gpu-spatial-analysis/spatial-clustering/index';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {computeKMeansOracle} from './kmeans-oracle';

it('GPUKMeans validates props and declares nodes', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'kmeans-nodes'});
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `view-${serial++}`, format, length);
  const make = () => ({
    positions: view('float32x2', 20),
    k: 3,
    iterations: 2,
    labels: view('uint32', 20),
    centers: view('float32x2', 3)
  });
  const base = make();
  expect(() => new GPUKMeans({...base, k: 0})).toThrow(/k must/);
  expect(() => new GPUKMeans({...base, k: 300})).toThrow(/k must/);
  expect(() => new GPUKMeans({...base, iterations: 0})).toThrow(/iterations/);
  expect(() => new GPUKMeans({...base, initialization: 'random' as 'kmeans++'})).toThrow(
    /initialization/
  );
  expect(() => new GPUKMeans({...base, seed: -1})).toThrow(/seed/);
  expect(() => new GPUKMeans({...base, centers: view('float32x2', 4)})).toThrow(/centers length/);
  expect(() => new GPUKMeans({...base, sizes: view('uint32', 2)})).toThrow(/sizes length/);
  const first = new GPUKMeans(base).getCommandNodes(graph).map(node => node.id);
  expect(first[0]).toBe('kmeans-prepare');
  expect(first).toContain('kmeans-initial-centers');
  expect(first).toContain('kmeans-iteration-1-update');
  expect(first.at(-1)).toMatch(/kmeans-final-sizes/);
  const seeded = new GPUKMeans({...make(), id: 'seeded', initialization: 'kmeans++', seed: 9})
    .getCommandNodes(graph)
    .map(node => node.id);
  expect(seeded).toContain('seeded-seed-2-update');
  expect(seeded).not.toContain('seeded-initial-centers');
});

it('GPUSpatialClustering declares dense-box nodes only when requested', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'dense-box-nodes'});
  const props = (id: string, denseBoxShortcut?: boolean) => ({
    id,
    denseBoxShortcut,
    positions: createTransientView(graph, `${id}-positions`, 'float32x2', 16),
    parameters: createTransientView(graph, `${id}-parameters`, 'float32', 8),
    gridSize: [8, 8] as const,
    labels: createTransientView(graph, `${id}-labels`, 'uint32', 16)
  });
  const plain = new GPUSpatialClustering(props('plain'))
    .getCommandNodes(graph)
    .map(node => node.id);
  const dense = new GPUSpatialClustering(props('dense', true))
    .getCommandNodes(graph)
    .map(node => node.id);
  expect(plain.some(id => id.includes('box'))).toBe(false);
  expect(dense).toContain('dense-box-keys');
  expect(dense).toContain('dense-box-counts');
  expect(dense.indexOf('dense-box-counts')).toBeLessThan(dense.indexOf('dense-core'));
});

it('computeKMeansOracle sanity: two separated clusters are recovered', () => {
  const positions = new Float32Array([0, 0, 10, 10, 1, 1, 11, 11, 0, 1, 10, 11]);
  const result = computeKMeansOracle(positions, 2, 3);
  expect(result.labels).toEqual([0, 1, 0, 1, 0, 1]);
  expect(result.sizes).toEqual([3, 3]);
});
