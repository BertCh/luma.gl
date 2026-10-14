// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {GPUNetworkNoding} from '../../../src/gpu-network/network-noding/index';
import {
  createNullWebGPUDevice,
  getParsedKernelSources
} from '../../utils/gpu-contributor-test-utils';

it('GPUNetworkNoding rejects an empty piece capacity at construction', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    name: string,
    format: Format,
    length: number
  ) => createTransientView(graph, name, format, length);
  // Review B-25: zero pieces used to compile `LAST = -1u` into the endpoint kernels.
  expect(
    () =>
      new GPUNetworkNoding({
        lines: {
          kind: 'lines',
          positions: view('positions', 'float32x2', 4),
          lineOffsets: view('line-offsets', 'uint32', 2)
        },
        intersectionCapacity: 16,
        tolerance: view('tolerance', 'float32', 1),
        output: {
          pieces: {
            geometry: {
              kind: 'lines',
              positions: view('piece-positions', 'float32x2', 0),
              lineOffsets: view('piece-offsets', 'uint32', 1)
            },
            sourceIds: view('piece-lines', 'uint32', 0),
            status: {
              count: view('piece-count', 'uint32', 1),
              overflow: view('piece-overflow', 'uint32', 1)
            }
          },
          nodes: {
            positions: view('node-positions', 'float32x2', 4),
            status: {
              count: view('node-count', 'uint32', 1),
              requiredCount: view('node-required', 'uint32', 1)
            }
          },
          edges: {
            fromNodes: view('from', 'uint32', 0),
            toNodes: view('to', 'uint32', 0),
            lengths: view('lengths', 'float32', 0)
          },
          topology: {
            offsets: view('offsets', 'uint32', 5),
            neighbors: view('neighbors', 'uint32', 0),
            weights: view('weights', 'float32', 0)
          }
        }
      })
  ).toThrow(/output\.pieces\.sourceIds must be non-empty/);
});

it('GPUNetworkNoding folds the snap origin per workgroup', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    name: string,
    format: Format,
    length: number
  ) => createTransientView(graph, name, format, length);
  const pieceCapacity = 16;
  const nodes = new GPUNetworkNoding({
    lines: {
      kind: 'lines',
      positions: view('positions', 'float32x2', 8),
      lineOffsets: view('line-offsets', 'uint32', 3)
    },
    intersectionCapacity: 16,
    tolerance: view('tolerance', 'float32', 1),
    output: {
      pieces: {
        geometry: {
          kind: 'lines',
          positions: view('piece-positions', 'float32x2', 2 * pieceCapacity),
          lineOffsets: view('piece-offsets', 'uint32', pieceCapacity + 1)
        },
        sourceIds: view('piece-lines', 'uint32', pieceCapacity),
        status: {
          count: view('piece-count', 'uint32', 1),
          overflow: view('piece-overflow', 'uint32', 1)
        }
      },
      nodes: {
        positions: view('node-positions', 'float32x2', 2 * pieceCapacity),
        status: {
          count: view('node-count', 'uint32', 1),
          requiredCount: view('node-required', 'uint32', 1)
        }
      },
      edges: {
        fromNodes: view('from', 'uint32', pieceCapacity),
        toNodes: view('to', 'uint32', pieceCapacity),
        lengths: view('lengths', 'float32', pieceCapacity)
      },
      topology: {
        offsets: view('offsets', 'uint32', 2 * pieceCapacity + 1),
        neighbors: view('neighbors', 'uint32', 2 * pieceCapacity),
        weights: view('weights', 'float32', 2 * pieceCapacity)
      }
    }
  }).getCommandNodes(graph);
  const endpoints = nodes.find(node => node.id.endsWith('-endpoints'));
  expect(endpoints).toBeDefined();
  // Rule 14 (review E-14): two atomics per workgroup instead of two per end point.
  const source = getParsedKernelSources([endpoints]).get(endpoints!.id) ?? '';
  expect(source).toContain('minCounter(0u, getOrderedKey(point.x));');
  expect(source).not.toContain('atomicMin(&snapOrigin[snapOriginOffset], getOrderedKey');
});
