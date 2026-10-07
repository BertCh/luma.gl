// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  computeAdjacencyMatrixOrder,
  encodeGPUAdjacencyMatrixWindow,
  getGPUAdjacencyMatrixFixedWeight,
  GPUAdjacencyMatrix,
  GPUAdjacencyMatrixOrder,
  type GPUAdjacencyMatrixProps
} from '../../../src/gpu-network/adjacency-matrix';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

function createContext(device: Device) {
  const graph = new GPUCommandGraph(device);
  const view = <F extends 'uint32' | 'float32'>(name: string, format: F, length: number) => {
    const buffer = device.createBuffer({
      id: `${name}-${Math.random()}`,
      byteLength: Math.max(length, 1) * 4,
      usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
    });
    return importGraphBuffer(graph, `${name}-${buffer.id}`, buffer, format, length);
  };
  const props = (overrides: Partial<GPUAdjacencyMatrixProps> = {}): GPUAdjacencyMatrixProps => ({
    offsets: view('offsets', 'uint32', 7),
    neighbors: view('neighbors', 'uint32', 10),
    resolution: 4,
    output: {counts: view('counts', 'uint32', 16)},
    ...overrides
  });
  return {graph, view, props};
}

it('GPUAdjacencyMatrix validates shapes and options', () => {
  const device = createNullWebGPUDevice();
  const {view, props} = createContext(device);
  expect(() => new GPUAdjacencyMatrix(props())).not.toThrow();
  expect(() => new GPUAdjacencyMatrix(props({resolution: 0}))).toThrow(/resolution/);
  expect(() => new GPUAdjacencyMatrix(props({resolution: 4097}))).toThrow(/resolution/);
  expect(() => new GPUAdjacencyMatrix(props({output: {counts: view('c', 'uint32', 15)}}))).toThrow(
    /counts/
  );
  expect(() => new GPUAdjacencyMatrix(props({order: view('order', 'uint32', 5)}))).toThrow(/order/);
  expect(() => new GPUAdjacencyMatrix(props({edgeMask: view('mask', 'uint32', 9)}))).toThrow(
    /edgeMask/
  );
  expect(() => new GPUAdjacencyMatrix(props({window: view('window', 'uint32', 3)}))).toThrow(
    /window/
  );
  expect(() => new GPUAdjacencyMatrix(props({weightScale: 0}))).toThrow(/weightScale/);
  expect(
    () =>
      new GPUAdjacencyMatrix(
        props({
          output: {
            counts: view('c2', 'uint32', 16),
            weightSums: view('w', 'uint32', 16)
          }
        })
      )
  ).toThrow(/require weights/);
  expect(
    () =>
      new GPUAdjacencyMatrix(
        props({
          weights: view('weights', 'float32', 10),
          output: {
            counts: view('c3', 'uint32', 16),
            maxWeightSum: view('m', 'uint32', 1)
          }
        })
      )
  ).toThrow(/weightSums/);
});

it('GPUAdjacencyMatrix returns deterministic node ids', () => {
  const device = createNullWebGPUDevice();
  const {graph, view, props} = createContext(device);
  const contributor = new GPUAdjacencyMatrix(
    props({
      weights: view('weights', 'float32', 10),
      output: {
        counts: view('c', 'uint32', 16),
        weightSums: view('w', 'uint32', 16),
        maxCount: view('mc', 'uint32', 1),
        maxWeightSum: view('mw', 'uint32', 1)
      }
    })
  );
  const ids = contributor.getCommandNodes(graph).map(node => node.id);
  expect(ids).toEqual([
    'adjacency-matrix-zero-counts',
    'adjacency-matrix-zero-weight-sums',
    'adjacency-matrix-zero-max-count',
    'adjacency-matrix-zero-max-weight-sum',
    // Counts and weight sums share one binning kernel while its bindings fit the device limit.
    'adjacency-matrix-bin',
    'adjacency-matrix-maxima'
  ]);
});

it('window encoding and fixed-point weights follow their documented rules', () => {
  expect(
    Array.from(encodeGPUAdjacencyMatrixWindow({rowStart: 1, rowEnd: 9, colStart: 2, colEnd: 5}, 4))
  ).toEqual([1, 9, 2, 5]);
  expect(() =>
    encodeGPUAdjacencyMatrixWindow({rowStart: 0, rowEnd: 2 ** 31, colStart: 0, colEnd: 1}, 4)
  ).toThrow(/2\^32/);
  expect(() =>
    encodeGPUAdjacencyMatrixWindow({rowStart: -1, rowEnd: 2, colStart: 0, colEnd: 1}, 4)
  ).toThrow(/uint32/);
  expect(getGPUAdjacencyMatrixFixedWeight(1.5, 1024)).toBe(1536);
  expect(getGPUAdjacencyMatrixFixedWeight(-3, 1024)).toBe(0);
  expect(getGPUAdjacencyMatrixFixedWeight(NaN, 1024)).toBe(0);
  expect(getGPUAdjacencyMatrixFixedWeight(1e30, 1024)).toBe(0xffffffff);
});

it('computeAdjacencyMatrixOrder sorts by group, tie key, then vertex', () => {
  expect(Array.from(computeAdjacencyMatrixOrder([1, 0, 1, 0]))).toEqual([2, 0, 3, 1]);
  expect(Array.from(computeAdjacencyMatrixOrder([1, 0, 1, 0], [5, 9, 1, 9]))).toEqual([3, 0, 2, 1]);
});

it('GPUAdjacencyMatrixOrder validates and names its nodes', () => {
  const device = createNullWebGPUDevice();
  const {graph, view} = createContext(device);
  expect(
    () =>
      new GPUAdjacencyMatrixOrder({
        groups: view('g', 'uint32', 4),
        order: view('o', 'uint32', 3)
      })
  ).toThrow(/order/);
  const contributor = new GPUAdjacencyMatrixOrder({
    groups: view('g2', 'uint32', 4),
    tieKeys: view('t', 'uint32', 4),
    order: view('o2', 'uint32', 4)
  });
  const ids = contributor.getCommandNodes(graph).map(node => node.id);
  expect(ids[0]).toBe('adjacency-matrix-order-identity');
  expect(ids.at(-1)).toBe('adjacency-matrix-order-scatter');
  expect(ids).toContain('adjacency-matrix-order-gather-groups');
});
