// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {GPUNetworkSnapping} from '../../src/gpu-network/network-accessibility/index';
import {GPUNetworkKFunction} from '../../src/gpu-network/network-k-function/index';
import {createNullWebGPUDevice} from '../utils/gpu-contributor-test-utils';

// Review B-03 and B-04: a CSR may be shorter than its compile-time edge capacity (the contents are
// per frame). Kernels that run once per edge row must not trust the rows past offsets[NODE_COUNT],
// which no per-node kernel writes and which may hold another transient's words (buffers alias
// across lifetimes). These specs pin the schedule; the GPU specs run a CSR with tail rows.

it('GPUNetworkSnapping clears every edge-source row before expanding CSR offsets', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'snapping-tail'});
  const offsets = createTransientView(graph, 'offsets', 'uint32', 5);
  const nodes = new GPUNetworkSnapping({
    id: 'snap',
    points: createTransientView(graph, 'points', 'float32x2', 3),
    nodePositions: createTransientView(graph, 'positions', 'float32x2', 4),
    offsets,
    edgeTargets: createTransientView(graph, 'targets', 'uint32', 9),
    snappedEdges: createTransientView(graph, 'snapped', 'uint32', 3)
  }).getCommandNodes(graph);
  const ids = nodes.map(node => node.id);
  const clear = ids.indexOf('snap-edge-sources-clear');
  const expand = ids.indexOf('snap-edge-sources');
  expect(clear).toBeGreaterThanOrEqual(0);
  expect(clear).toBeLessThan(expand);
  // The clear covers the whole edge-source view that the expansion writes and the scan reads.
  const clearBuffer = nodes[clear].resources?.[0]?.buffer;
  const expandBuffers = nodes[expand].resources?.map(resource => resource.buffer) ?? [];
  expect(expandBuffers).toEqual([offsets, clearBuffer]);
  expect(clearBuffer?.length).toBe(9);
  const scanBuffers = nodes[ids.indexOf('snap-scan')].resources?.map(resource => resource.buffer);
  expect(scanBuffers).toContain(clearBuffer);
});

it('GPUNetworkKFunction draws simulated events only from live CSR rows', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'k-function-tail'});
  const offsets = createTransientView(graph, 'offsets', 'uint32', 7);
  const nodes = new GPUNetworkKFunction({
    id: 'k',
    points: createTransientView(graph, 'points', 'float32x2', 10),
    nodePositions: createTransientView(graph, 'positions', 'float32x2', 6),
    offsets,
    neighbors: createTransientView(graph, 'neighbors', 'uint32', 12),
    weights: createTransientView(graph, 'weights', 'float32', 12),
    maxDistance: createTransientView(graph, 'max-distance', 'float32', 1),
    networkLength: createTransientView(graph, 'network-length', 'float32', 1),
    parameters: createTransientView(graph, 'parameters', 'uint32', 4),
    bandCount: 8,
    simulationCount: 3,
    rowsPerBlock: 15,
    kValues: createTransientView(graph, 'k-values', 'float32', 32)
  }).getCommandNodes(graph);
  const quantize = nodes.find(node => node.id === 'k-quantize');
  // The quantize kernel reads offsets to gate on index < offsets[NODE_COUNT]: a tail row with a
  // positive weight gets zero draw weight, so the node search never ends at NODE_COUNT.
  expect(quantize?.resources?.map(resource => resource.buffer)).toContain(offsets);
  // Its snapping expands the same offsets with the clearing edge-source nodes.
  const ids = nodes.map(node => node.id);
  expect(ids.some(id => id.endsWith('-edge-sources-clear'))).toBe(true);
});
