// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUCommandGraph,
  type GraphBufferHandle,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {describe, expect, it} from 'vitest';
import {
  GPUZonalStatistics,
  type GPUZonalStatisticsExtentStatistic,
  type GPUZonalStatisticsProps
} from '../../../src/gpu-spatial-analysis/zonal-statistics';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  CORE_PRIVATE_MEMORY_LIMIT_BYTES,
  CORE_WORKGROUP_STORAGE_LIMIT_BYTES,
  getPrivateMemoryBytes,
  getStorageBindingCount,
  getWorkgroupMemoryBytes
} from '../../utils/gpu-contributor-test-utils';

type ResourceNode = {id: string; resources?: {buffer?: GraphDataView | GraphBufferHandle}[]};

function getBufferBytes(buffer: GraphDataView | GraphBufferHandle): number {
  return 'rowByteLength' in buffer
    ? Math.max(buffer.length, 1) * buffer.rowByteLength
    : buffer.byteLength;
}

/** Asserts storage-binding count, private and workgroup memory of every generated kernel. */
function expectKernelsWithinCoreLimits(nodes: readonly unknown[]): void {
  const kernels = getGPUCommandNodeKernelSources(nodes);
  expect(kernels.length).toBeGreaterThan(0);
  for (const kernel of kernels) {
    expect(getStorageBindingCount(kernel.source), `${kernel.id} storage`).toBeLessThanOrEqual(8);
    expect(getPrivateMemoryBytes(kernel.source), `${kernel.id} private`).toBeLessThanOrEqual(
      CORE_PRIVATE_MEMORY_LIMIT_BYTES
    );
    expect(getWorkgroupMemoryBytes(kernel.source), `${kernel.id} workgroup`).toBeLessThanOrEqual(
      CORE_WORKGROUP_STORAGE_LIMIT_BYTES
    );
  }
}

/** Asserts every transient and bound view of the nodes fits one storage binding. */
function expectBindingsWithinLimit(nodes: readonly unknown[], limit: number): void {
  let checked = 0;
  for (const node of nodes as ResourceNode[]) {
    for (const use of node.resources ?? []) {
      if (use.buffer) {
        checked++;
        expect(getBufferBytes(use.buffer), `${node.id} binding`).toBeLessThanOrEqual(limit);
      }
    }
  }
  expect(checked).toBeGreaterThan(0);
}

const EXTENT_STATISTICS: GPUZonalStatisticsExtentStatistic[] = [
  'count',
  'sum',
  'mean',
  'minimum',
  'maximum',
  'density'
];

function createMaximalProps(
  graph: GPUCommandGraph,
  options: {
    kind: 'polygons' | 'feature-rows';
    pointCount: number;
    featureCount: number;
    candidateCapacity: number;
    sumOrder: 'atomic' | 'sorted';
    extentStatistic: GPUZonalStatisticsExtentStatistic;
  }
): GPUZonalStatisticsProps {
  const {pointCount, featureCount} = options;
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    format: Format,
    length: number,
    name: string
  ) => createTransientView(graph, name, format, length);
  const polygons = options.kind === 'polygons';
  return {
    id: 'z',
    features: polygons
      ? {
          kind: 'polygons',
          positions: view('float32x2', featureCount * 8, 'polygon-positions'),
          featureOffsets: view('uint32', featureCount + 1, 'feature-offsets'),
          polygonOffsets: view('uint32', featureCount + 1, 'polygon-offsets'),
          ringOffsets: view('uint32', featureCount + 1, 'ring-offsets'),
          candidateCapacity: options.candidateCapacity,
          boundaryRule: 'closed'
        }
      : {
          kind: 'feature-rows',
          pointFeatureRows: view('uint32', pointCount, 'rows'),
          featureCount
        },
    positions: polygons ? view('float32x2', pointCount, 'points') : undefined,
    values: view('float32', pointCount, 'values'),
    weights: view('float32', pointCount, 'weights'),
    areas: polygons ? undefined : view('float32', featureCount, 'areas'),
    sumOrder: options.sumOrder,
    output: {
      counts: view('uint32', featureCount, 'counts'),
      valueCounts: view('uint32', featureCount, 'value-counts'),
      sums: view('float32', featureCount, 'sums'),
      weightSums: view('float32', featureCount, 'weight-sums'),
      means: view('float32', featureCount, 'means'),
      minima: view('float32', featureCount, 'minima'),
      maxima: view('float32', featureCount, 'maxima'),
      variances: view('float32', featureCount, 'variances'),
      standardDeviations: view('float32', featureCount, 'standard-deviations'),
      densities: view('float32', featureCount, 'densities'),
      weightedDensities: view('float32', featureCount, 'weighted-densities'),
      featureAreas: polygons ? view('float32', featureCount, 'feature-areas') : undefined,
      pointFeatureRows: polygons ? view('uint32', pointCount, 'point-rows') : undefined,
      extent: view('float32', 2, 'extent'),
      extentStatistic: options.extentStatistic,
      status: {
        overflow: view('uint32', 1, 'overflow'),
        uncertainCount: view('uint32', 1, 'uncertain')
      }
    }
  };
}

describe('GPUZonalStatistics at the core device limits', () => {
  for (const kind of ['polygons', 'feature-rows'] as const) {
    for (const sumOrder of ['atomic', 'sorted'] as const) {
      it(`${kind} with every output and ${sumOrder} sums`, () => {
        const device = createNullWebGPUDevice();
        const graph = new GPUCommandGraph(device, {id: 'zonal-core-limits'});
        const props = createMaximalProps(graph, {
          kind,
          pointCount: 4096,
          featureCount: 4096,
          candidateCapacity: 1 << 16,
          sumOrder,
          extentStatistic: 'mean'
        });
        const nodes = new GPUZonalStatistics(props).getCommandNodes(graph);
        expectKernelsWithinCoreLimits(nodes);
      });
    }
  }

  for (const extentStatistic of EXTENT_STATISTICS) {
    it(`extent of ${extentStatistic}`, () => {
      const device = createNullWebGPUDevice();
      const graph = new GPUCommandGraph(device, {id: 'zonal-core-limits'});
      const props = createMaximalProps(graph, {
        kind: 'polygons',
        pointCount: 256,
        featureCount: 64,
        candidateCapacity: 1024,
        sumOrder: 'sorted',
        extentStatistic
      });
      const nodes = new GPUZonalStatistics(props).getCommandNodes(graph);
      expectKernelsWithinCoreLimits(nodes);
    });
  }

  it('keeps every transient and bound view within maxStorageBufferBindingSize at scale', () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device, {id: 'zonal-core-limits'});
    const props = createMaximalProps(graph, {
      kind: 'polygons',
      pointCount: 4_000_000,
      featureCount: 65536,
      candidateCapacity: 1 << 22,
      sumOrder: 'sorted',
      extentStatistic: 'density'
    });
    const nodes = new GPUZonalStatistics(props).getCommandNodes(graph);
    expectBindingsWithinLimit(nodes, device.limits.maxStorageBufferBindingSize);
  });
});
