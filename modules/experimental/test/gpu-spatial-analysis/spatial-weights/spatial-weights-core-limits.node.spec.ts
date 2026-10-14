// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {describe, expect, it} from 'vitest';
import {
  GPUContiguityWeights,
  GPULatticeWeights,
  GPUSpatialLag,
  GPUSpatialWeightsAlgebra,
  GPUSpatialWeightsSummary,
  GPUSpatialWeightsTranspose,
  GPUSpatialWeightsTransform,
  GPUSpatialWeightsValidation,
  GPU_LATTICE_WEIGHTS_MAXIMUM_RADIUS,
  GPU_SPATIAL_WEIGHTS_MAXIMUM_ORDER,
  GPU_SPATIAL_WEIGHTS_SUMMARY_LAYOUT,
  type GPUSpatialWeights,
  type GPUSpatialWeightsKernel,
  type GPUSpatialWeightsTransformOperation
} from '../../../src/gpu-spatial-analysis/spatial-weights';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  CORE_PRIVATE_MEMORY_LIMIT_BYTES,
  CORE_WORKGROUP_STORAGE_LIMIT_BYTES,
  getPrivateMemoryBytes,
  getStorageBindingCount,
  getWorkgroupMemoryBytes
} from '../../utils/gpu-contributor-test-utils';

const ROWS = 1000;
const SLOTS = 8;
let serial = 0;

type Resource = {buffer: {byteLength?: number; length?: number; rowByteLength?: number}};

/** Bytes of every buffer or view bound by the nodes. */
function getBoundBytes(nodes: readonly unknown[]): {id: string; bytes: number}[] {
  const result: {id: string; bytes: number}[] = [];
  for (const node of nodes as {
    id: string;
    resources?: Resource[];
    workload?: {resources?: Resource[]};
  }[]) {
    for (const resource of node.resources ?? node.workload?.resources ?? []) {
      const buffer = resource.buffer;
      const bytes =
        buffer.byteLength ??
        (buffer.length ?? 0) * (buffer.rowByteLength ?? Uint32Array.BYTES_PER_ELEMENT);
      result.push({id: node.id, bytes});
    }
  }
  return result;
}

function expectCoreLimits(nodes: readonly unknown[], device = createNullWebGPUDevice()): void {
  const sources = getGPUCommandNodeKernelSources(nodes);
  expect(sources.length).toBeGreaterThan(0);
  for (const {id, source} of sources) {
    expect(getStorageBindingCount(source), `${id} storage bindings`).toBeLessThanOrEqual(8);
    expect(getPrivateMemoryBytes(source), `${id} private bytes`).toBeLessThanOrEqual(
      CORE_PRIVATE_MEMORY_LIMIT_BYTES
    );
    expect(getWorkgroupMemoryBytes(source), `${id} workgroup bytes`).toBeLessThanOrEqual(
      CORE_WORKGROUP_STORAGE_LIMIT_BYTES
    );
  }
  const bound = getBoundBytes(nodes);
  expect(bound.length).toBeGreaterThan(0);
  for (const {id, bytes} of bound) {
    expect(bytes, `${id} bound bytes`).toBeLessThanOrEqual(
      device.limits.maxStorageBufferBindingSize
    );
  }
}

function setup() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device, {id: `weights-limits-${serial++}`});
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `weights-limits-${serial++}`, format, length);
  const weights = (rows: number, capacity: number, distances = true): GPUSpatialWeights => ({
    offsets: view('uint32', rows + 1),
    neighbors: view('uint32', capacity),
    weights: view('float32', capacity),
    distances: distances ? view('float32', capacity) : undefined
  });
  return {device, graph, view, weights};
}

describe('spatial weights at core limits', () => {
  for (const criterion of ['queen', 'rook'] as const) {
    it(`GPUContiguityWeights ${criterion} with every output and a large pair capacity`, () => {
      const {device, graph, view, weights} = setup();
      const nodes = new GPUContiguityWeights({
        criterion,
        polygons: {
          kind: 'polygons',
          positions: view('float32x2', 4000),
          ringOffsets: view('uint32', 101),
          polygonOffsets: view('uint32', 101),
          featureOffsets: view('uint32', 51)
        },
        snapTolerance: 1e-3,
        output: {
          ...weights(50, 50 * 64, false),
          status: {
            overflow: view('uint32', 1),
            requiredCount: view('uint32', 1),
            candidateOverflow: view('uint32', 1)
          }
        },
        pairCapacity: 1 << 22
      }).getCommandNodes(graph);
      expectCoreLimits(nodes, device);
    });
  }

  for (const criterion of ['queen', 'rook'] as const) {
    for (const radius of [1, GPU_LATTICE_WEIGHTS_MAXIMUM_RADIUS]) {
      it(`GPULatticeWeights ${criterion} at radius ${radius} with mask and cell size`, () => {
        const {device, graph, view, weights} = setup();
        const width = 64;
        const height = 64;
        const slots = (2 * radius + 1) ** 2 - 1;
        const nodes = new GPULatticeWeights({
          width,
          height,
          criterion,
          radius,
          mask: view('uint32', width * height),
          cellSize: [10, 20],
          output: {
            ...weights(width * height, width * height * slots),
            status: {overflow: view('uint32', 1), requiredCount: view('uint32', 1)}
          }
        }).getCommandNodes(graph);
        expectCoreLimits(nodes, device);
      });
    }
  }

  const operations: GPUSpatialWeightsTransformOperation[] = [
    'row',
    'binary',
    'symmetrize',
    'double',
    'variance'
  ];
  for (const operation of operations) {
    it(`GPUSpatialWeightsTransform ${operation}`, () => {
      const {device, graph, view, weights} = setup();
      const nodes = new GPUSpatialWeightsTransform({
        operation,
        weights: weights(ROWS, ROWS * SLOTS),
        output: view('float32', ROWS * SLOTS),
        doubleSum: operation === 'double' ? 'rows' : undefined
      }).getCommandNodes(graph);
      expectCoreLimits(nodes, device);
    });
  }
  for (const kernel of [
    'gaussian',
    'triangular',
    'epanechnikov',
    'bisquare',
    'uniform'
  ] as GPUSpatialWeightsKernel[]) {
    for (const bandwidth of [250, 'adaptive'] as const) {
      it(`GPUSpatialWeightsTransform kernel ${kernel} with ${bandwidth} bandwidth`, () => {
        const {device, graph, view, weights} = setup();
        const nodes = new GPUSpatialWeightsTransform({
          operation: 'kernel',
          weights: {
            ...weights(ROWS, ROWS * SLOTS),
            information: {
              rowCount: ROWS,
              transform: 'raw',
              symmetric: 'unknown',
              provenance: {source: 'test', criterion: 'knn', k: SLOTS}
            }
          },
          output: view('float32', ROWS * SLOTS),
          kernel,
          bandwidth
        }).getCommandNodes(graph);
        expectCoreLimits(nodes, device);
      });
    }
  }

  for (const columnCount of [1, 64]) {
    it(`GPUSpatialLag with mask, normalize and ${columnCount} columns`, () => {
      const {device, graph, view, weights} = setup();
      const nodes = new GPUSpatialLag({
        values: view('float32', ROWS * columnCount),
        columnCount,
        weights: weights(ROWS, ROWS * SLOTS),
        mask: view('uint32', ROWS),
        normalize: true,
        output: view('float32', ROWS * columnCount)
      }).getCommandNodes(graph);
      expectCoreLimits(nodes, device);
    });
  }

  it('GPUSpatialWeightsSummary with every output', () => {
    const {device, graph, view, weights} = setup();
    const nodes = new GPUSpatialWeightsSummary({
      weights: weights(ROWS, ROWS * SLOTS),
      output: {
        statistics: view('float32', GPU_SPATIAL_WEIGHTS_SUMMARY_LAYOUT.statistics.length),
        counts: view('uint32', GPU_SPATIAL_WEIGHTS_SUMMARY_LAYOUT.counts.length),
        cardinality: view('uint32', ROWS)
      },
      symmetryTolerance: 1e-6
    }).getCommandNodes(graph);
    expectCoreLimits(nodes, device);
  });

  it('GPUSpatialWeightsTranspose with every output', () => {
    const {device, graph, view, weights} = setup();
    const nodes = new GPUSpatialWeightsTranspose({
      weights: weights(ROWS, ROWS * SLOTS),
      columnCount: ROWS,
      output: {...weights(ROWS, ROWS * SLOTS), asymmetricSlots: view('uint32', 1)},
      symmetryTolerance: 1e-6
    }).getCommandNodes(graph);
    expectCoreLimits(nodes, device);
  });

  it('GPUSpatialWeightsValidation with flags', () => {
    const {device, graph, view, weights} = setup();
    const nodes = new GPUSpatialWeightsValidation({
      weights: weights(ROWS, ROWS * SLOTS),
      output: {flags: view('uint32', ROWS), status: {invalidCount: view('uint32', 1)}}
    }).getCommandNodes(graph);
    expectCoreLimits(nodes, device);
  });

  describe('GPUSpatialWeightsAlgebra', () => {
    for (const operation of [
      'union',
      'intersection',
      'difference',
      'symmetricDifference'
    ] as const) {
      for (const weightRule of [
        'left',
        'right',
        'sum',
        'min',
        'max',
        'product',
        'binary'
      ] as const) {
        it(`${operation} with ${weightRule} rule and distances`, () => {
          const {device, graph, view, weights} = setup();
          const nodes = new GPUSpatialWeightsAlgebra({
            operation,
            weightRule,
            left: weights(ROWS, ROWS * SLOTS),
            right: weights(ROWS, ROWS * SLOTS),
            output: {
              ...weights(ROWS, ROWS * SLOTS * 2),
              status: {overflow: view('uint32', 1), requiredCount: view('uint32', 1)}
            }
          }).getCommandNodes(graph);
          expectCoreLimits(nodes, device);
        });
      }
    }

    for (const cumulative of [false, true]) {
      for (const order of [2, GPU_SPATIAL_WEIGHTS_MAXIMUM_ORDER]) {
        it(`higherOrder order ${order}${cumulative ? ' cumulative' : ''}`, () => {
          const {device, graph, view, weights} = setup();
          const nodes = new GPUSpatialWeightsAlgebra({
            operation: 'higherOrder',
            weights: weights(ROWS, ROWS * SLOTS, false),
            order,
            cumulative,
            workCapacity: 1 << 22,
            output: {
              ...weights(ROWS, ROWS * 64, false),
              status: {overflow: view('uint32', 1), requiredCount: view('uint32', 1)}
            }
          }).getCommandNodes(graph);
          expectCoreLimits(nodes, device);
        });
      }
    }

    for (const selfWeightKind of ['number', 'view'] as const) {
      it(`selfWeight with a ${selfWeightKind}`, () => {
        const {device, graph, view, weights} = setup();
        const nodes = new GPUSpatialWeightsAlgebra({
          operation: 'selfWeight',
          weights: weights(ROWS, ROWS * SLOTS),
          selfWeight: selfWeightKind === 'number' ? 1 : view('float32', ROWS),
          output: {
            ...weights(ROWS, ROWS * (SLOTS + 1)),
            status: {overflow: view('uint32', 1), requiredCount: view('uint32', 1)}
          }
        }).getCommandNodes(graph);
        expectCoreLimits(nodes, device);
      });
    }

    it('subgraph with a mask', () => {
      const {device, graph, view, weights} = setup();
      const nodes = new GPUSpatialWeightsAlgebra({
        operation: 'subgraph',
        weights: weights(ROWS, ROWS * SLOTS),
        mask: view('uint32', ROWS),
        output: {
          ...weights(ROWS, ROWS * SLOTS),
          status: {overflow: view('uint32', 1), requiredCount: view('uint32', 1)}
        }
      }).getCommandNodes(graph);
      expectCoreLimits(nodes, device);
    });

    it('block with many groups', () => {
      const {device, graph, view, weights} = setup();
      const nodes = new GPUSpatialWeightsAlgebra({
        operation: 'block',
        groupIds: view('uint32', ROWS),
        groupCount: ROWS,
        output: {
          ...weights(ROWS, ROWS * 32, false),
          status: {overflow: view('uint32', 1), requiredCount: view('uint32', 1)}
        }
      }).getCommandNodes(graph);
      expectCoreLimits(nodes, device);
    });
  });
});

it('normalized spatial lag fits when its 64-column output reaches the binding limit', () => {
  const {device, graph, view, weights} = setup();
  const columnCount = 64;
  const rows = device.limits.maxStorageBufferBindingSize / (4 * columnCount);
  const nodes = new GPUSpatialLag({
    values: view('float32', rows * columnCount),
    columnCount,
    weights: weights(rows, rows * SLOTS),
    mask: view('uint32', rows),
    normalize: true,
    output: view('float32', rows * columnCount)
  }).getCommandNodes(graph);
  expectCoreLimits(nodes, device);
  expect(nodes.map(node => node.id)).toEqual([
    'spatial-lag-lag',
    'spatial-lag-lag-fixup',
    'spatial-lag-weights',
    'spatial-lag-weights-fixup',
    'spatial-lag-normalize'
  ]);
});
