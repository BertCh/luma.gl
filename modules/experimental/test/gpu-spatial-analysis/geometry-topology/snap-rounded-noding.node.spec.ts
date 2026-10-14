// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {describe, expect, it} from 'vitest';
import {
  getGPUSnapRoundedNodingMaximumVertexCount,
  getSnapRoundingGridCell,
  GPUSnapRoundedNoding,
  type GPUSnapRoundedNodingProps
} from '../../../src/gpu-spatial-analysis/geometry-topology/gpu-snap-rounded-noding';
import {
  CORE_PRIVATE_MEMORY_LIMIT_BYTES,
  CORE_WORKGROUP_STORAGE_LIMIT_BYTES,
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getParsedKernelSources,
  getPrivateMemoryBytes,
  getStorageBindingCount,
  getWorkgroupMemoryBytes
} from '../../utils/gpu-contributor-test-utils';
import {getInlinedCallSites, getUndeclaredCalls} from '../segment-intersection/wgsl-call-sites';

// Structure of the snap-rounded noding kernels: every module parses, binds at most eight storage
// buffers, calls only declared functions, stays within the core private and workgroup memory
// limits, and reaches each large exact helper (the limb multiplication, the orientation fallback)
// from at most one inlined call site (library rule: a second copy made join specs 2.5x slower).

const OWN_KERNELS = [
  'clear-state',
  'lattice',
  'snap-vertices',
  'mark-vertices',
  'crossing-terms',
  'round-crossings',
  'row-ids',
  'gather-x',
  'unique-hot',
  'compact-hot',
  'build-diagonals',
  'filter-candidates',
  'scatter-incidences',
  'order-incidences',
  'flag-pieces',
  'initialize',
  'write-geometry',
  'write-provenance',
  'write-parameters',
  'status-candidates',
  'status'
];

type Overrides = {
  bindingBytes?: number;
  vertices?: number;
  lines?: number;
  capacity?: number;
  intersectionCapacity?: number;
  sourceIds?: boolean;
};

function build(overrides: Overrides = {}, props: Partial<GPUSnapRoundedNodingProps> = {}) {
  const device = createNullWebGPUDevice();
  if (overrides.bindingBytes !== undefined) {
    Object.assign(device.limits, {
      maxStorageBufferBindingSize: overrides.bindingBytes,
      maxBufferSize: overrides.bindingBytes
    });
  }
  const graph = new GPUCommandGraph(device, {
    id: 'snap-rounded-noding-structure'
  });
  const view = <Format extends 'uint32' | 'float32' | 'float32x2' | 'float32x4'>(
    name: string,
    format: Format,
    length: number
  ) => createTransientView(graph, name, format, length);
  const capacity = overrides.capacity ?? 64;
  const scalar = (name: string) => view(name, 'uint32', 1);
  const contributor = new GPUSnapRoundedNoding({
    lines: {
      kind: 'lines',
      positions: view('positions', 'float32x2', overrides.vertices ?? 12),
      lineOffsets: view('line-offsets', 'uint32', (overrides.lines ?? 3) + 1),
      sourceIds: overrides.sourceIds
        ? view('source-ids', 'uint32', overrides.lines ?? 3)
        : undefined
    },
    gridSize: 0.25,
    intersectionCapacity: overrides.intersectionCapacity ?? 16,
    output: {
      endpoints: view('endpoints', 'float32x4', capacity),
      latticeEndpoints: view('lattice-endpoints', 'float32x4', capacity),
      lattice: view('lattice', 'float32', 4),
      sourceFeatureIds: view('feature-ids', 'uint32', capacity),
      sourceRingIds: view('ring-ids', 'uint32', capacity),
      sourceEdgeIds: view('edge-ids', 'uint32', capacity),
      sourceStartParameters: view('start-parameters', 'float32', capacity),
      sourceEndParameters: view('end-parameters', 'float32', capacity),
      status: {
        count: scalar('count'),
        requiredCount: scalar('required'),
        overflow: scalar('overflow'),
        candidateOverflow: scalar('candidate-overflow')
      }
    },
    ...props
  });
  return {graph, contributor, nodes: contributor.getCommandNodes(graph)};
}

describe('GPUSnapRoundedNoding structure', () => {
  it.each([
    2 ** 27,
    5 * 2 ** 25,
    2 ** 30
  ])('accepts exactly the hot-pixel maximum with %i-byte bindings', bindingBytes => {
    const intersectionCapacity = 16;
    const vertices = Math.floor(bindingBytes / 128) - intersectionCapacity;
    const {graph, nodes} = build({bindingBytes, vertices, intersectionCapacity});
    expect(getGPUSnapRoundedNodingMaximumVertexCount(graph.device, intersectionCapacity)).toBe(
      vertices
    );
    let largestBindingBytes = 0;
    for (const node of nodes) {
      for (const resource of node.resources ?? []) {
        if (!('buffer' in resource) || !resource.usage.startsWith('storage')) continue;
        const buffer = resource.buffer;
        const bytes =
          'rowByteLength' in buffer
            ? (Math.max(buffer.length, 1) - 1) * buffer.byteStride + buffer.rowByteLength
            : buffer.byteLength;
        largestBindingBytes = Math.max(largestBindingBytes, bytes);
        expect(bytes, node.id).toBeLessThanOrEqual(bindingBytes);
      }
    }
    // The right segment table really reaches the advertised limit; the BVH and all scratch fit.
    expect(largestBindingBytes).toBe(Math.floor(bindingBytes / 128) * 128);
    expect(() => build({bindingBytes, vertices: vertices + 1, intersectionCapacity})).toThrow(
      '4 x (lines.positions rows + intersectionCapacity) must be at most'
    );
  });

  it('accepts more than the old core hot-pixel cap only with a larger device binding', () => {
    const vertices = 2 ** 20;
    expect(() => build({vertices, bindingBytes: 2 ** 30})).not.toThrow();
    expect(() => build({vertices, bindingBytes: 2 ** 27})).toThrow('at most 4194304');
  });

  it('compiles kernels that parse, bind at most eight buffers and call only declared functions', () => {
    for (const sourceIds of [false, true]) {
      const {nodes} = build({sourceIds});
      const sources = getParsedKernelSources(nodes);
      expect(sources.size).toBeGreaterThan(20);
      for (const [id, source] of sources) {
        expect(getStorageBindingCount(source), id).toBeLessThanOrEqual(8);
        expect(getUndeclaredCalls(source), id).toEqual([]);
        expect(getPrivateMemoryBytes(source), id).toBeLessThanOrEqual(
          CORE_PRIVATE_MEMORY_LIMIT_BYTES
        );
        expect(getWorkgroupMemoryBytes(source), id).toBeLessThanOrEqual(
          CORE_WORKGROUP_STORAGE_LIMIT_BYTES
        );
      }
    }
  });

  it('uses one inlined call site of every large exact helper per kernel', () => {
    const {nodes} = build();
    // Own kernels only: the delegated GPUSegmentIntersection kernels carry their own orientSign.
    const ours = getGPUCommandNodeKernelSources(nodes).filter(kernel =>
      OWN_KERNELS.some(name => kernel.id.endsWith(`snap-rounded-noding-${name}`))
    );
    expect(ours.length).toBe(OWN_KERNELS.length);
    let exactKernels = 0;
    for (const kernel of ours) {
      // The exact orientation fallback is not needed: lattice predicates use the limb arithmetic.
      expect(kernel.source, kernel.id).not.toContain('orientSign');
      const multiplySites = getInlinedCallSites(kernel.source, 'latticeBigMultiply');
      if (multiplySites.size > 0) {
        exactKernels++;
      }
      for (const [callee, count] of multiplySites) {
        expect(count, `${kernel.id} ${callee}`).toBeLessThanOrEqual(1);
      }
      // Every kernel that uses an exact lattice routine calls exactly one of them.
      const routines = [
        'latticeCrossingTerms',
        'latticeRoundedCrossing',
        'latticeSegmentMeetsPixel'
      ].filter(name => getInlinedCallSites(kernel.source, name).size > 0);
      expect(routines.length, kernel.id).toBeLessThanOrEqual(1);
    }
    // crossing-terms, round-crossings and filter-candidates.
    expect(exactKernels).toBe(3);
  });

  it('builds the same pipelines on every build', () => {
    const first = getGPUCommandNodeKernelSources(build().nodes).map(kernel => kernel.id);
    const second = getGPUCommandNodeKernelSources(build().nodes).map(kernel => kernel.id);
    expect(second).toEqual(first);
  });

  it('rounds the grid up to a power of two', () => {
    expect(getSnapRoundingGridCell(1)).toBe(1);
    expect(getSnapRoundingGridCell(0.25)).toBe(0.25);
    expect(getSnapRoundingGridCell(0.3)).toBe(0.5);
    expect(getSnapRoundingGridCell(3)).toBe(4);
    expect(getSnapRoundingGridCell(1e-9)).toBe(2 ** -29);
    expect(getSnapRoundingGridCell(2 ** -140)).toBe(2 ** -126);
    expect(getSnapRoundingGridCell(2 ** 126 * 1.5)).toBe(2 ** 127);
    expect(build().contributor.gridCell).toBe(0.25);
  });

  it('validates its props under its own ID', () => {
    expect(() => build({}, {gridSize: 0})).toThrow('snap-rounded-noding gridSize');
    expect(() => build({}, {gridSize: Number.NaN})).toThrow('snap-rounded-noding gridSize');
    expect(() => build({}, {intersectionCapacity: 0})).toThrow('intersectionCapacity');
    expect(() => build({}, {leafCapacity: 3})).toThrow('leafCapacity');
    expect(() => build({}, {id: 'snap'})).not.toThrow();
    expect(() => build({}, {id: 'snap', gridSize: -1})).toThrow('snap gridSize');
    const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'snap-validation'});
    const device = graph.device;
    const limit = getGPUSnapRoundedNodingMaximumVertexCount(device, 16);
    expect(limit).toBeGreaterThan(1_000_000);
    expect(() => build({vertices: limit + 1, lines: 3, intersectionCapacity: 16}).nodes).toThrow(
      'at most'
    );
  });
});
