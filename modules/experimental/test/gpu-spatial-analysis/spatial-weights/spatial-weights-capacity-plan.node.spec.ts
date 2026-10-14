// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {describe, expect, it} from 'vitest';
import {
  GPU_CONTIGUITY_WEIGHTS_PAIR_CAPACITY_FACTOR,
  GPUContiguityWeights,
  GPULatticeWeights,
  type GPUSpatialWeightsOutput
} from '../../../src/gpu-spatial-analysis/spatial-weights';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

// A-15 / E-16: the weights plans are computed from the same constants as the allocation. The spec
// records every transient buffer declared on the graph while the contributor builds its nodes, and
// sets aside the device-dependent scratch the gpgpu sort and scan primitives declare under their
// own IDs (`<sort or scan id>-radix-...`, `-bitonic-...`, `-level-...`, `-chunk-...`), which the
// plans document as excluded. (A module mock is not used: the node project runs with
// `isolate: false`.)
const PRIMITIVE_SCRATCH = /-(?:sort-low|sort-high|scan)-/;

let serial = 0;

function setup() {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: `capacity-plan-${serial++}`});
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    format: Format,
    length: number
  ): GraphDataView<Format> => {
    const buffer = graph.createTransientBuffer({
      id: `input-${serial++}`,
      byteLength: Math.max(length, 1) * (format === 'float32x2' ? 8 : 4),
      usage: Buffer.STORAGE
    });
    return graph.createDataView(buffer, {format, length});
  };
  const output = (rows: number, capacity: number, distances: boolean): GPUSpatialWeightsOutput => ({
    offsets: view('uint32', rows + 1),
    neighbors: view('uint32', capacity),
    weights: view('float32', capacity),
    distances: distances ? view('float32', capacity) : undefined,
    status: {overflow: view('uint32', 1), requiredCount: view('uint32', 1)}
  });
  return {graph, view, output};
}

/** Builds the nodes while recording the contributor's own transients; returns their bytes. */
function getDeclaredTransientBytes(graph: GPUCommandGraph, build: () => unknown): number {
  const original = graph.createTransientBuffer.bind(graph);
  let bytes = 0;
  graph.createTransientBuffer = descriptor => {
    if (!PRIMITIVE_SCRATCH.test(descriptor.id)) {
      bytes += descriptor.byteLength;
    }
    return original(descriptor);
  };
  try {
    build();
  } finally {
    graph.createTransientBuffer = original;
  }
  return bytes;
}

function getOutputBytes(output: GPUSpatialWeightsOutput): number {
  return [output.offsets, output.neighbors, output.weights, output.distances].reduce(
    (sum, view) => sum + (view ? view.length * 4 : 0),
    0
  );
}

describe('weights capacity plans match the allocation (A-15)', () => {
  for (const criterion of ['queen', 'rook'] as const) {
    for (const pairCapacity of [undefined, 777]) {
      it(`GPUContiguityWeights ${criterion} pairCapacity ${pairCapacity ?? 'default'}`, () => {
        const {graph, view, output} = setup();
        const contiguity = new GPUContiguityWeights({
          criterion,
          polygons: {
            kind: 'polygons',
            positions: view('float32x2', 123),
            ringOffsets: view('uint32', 11),
            polygonOffsets: view('uint32', 9),
            featureOffsets: view('uint32', 6)
          },
          pairCapacity,
          output: output(5, 40, false)
        });
        const bytes = getDeclaredTransientBytes(graph, () => contiguity.getCommandNodes(graph));
        const plan = contiguity.getCapacityPlan();
        expect(bytes).toBeGreaterThan(0);
        expect(plan.estimatedPeakTransientBytes).toBe(bytes);
        expect(plan.estimatedOutputBytes).toBe(getOutputBytes(contiguity.props.output));
        expect(plan.slots.capacity).toBe(40);
        expect(plan.pairs.capacity).toBe(
          pairCapacity ?? 40 * GPU_CONTIGUITY_WEIGHTS_PAIR_CAPACITY_FACTOR
        );
      });
    }
  }

  it('GPUContiguityWeights grow step: slots by requiredCount, pairs by candidateOverflow', () => {
    const {view, output} = setup();
    const polygons = {
      kind: 'polygons' as const,
      positions: view('float32x2', 16),
      ringOffsets: view('uint32', 5),
      polygonOffsets: view('uint32', 5),
      featureOffsets: view('uint32', 5)
    };
    const contiguity = new GPUContiguityWeights({
      criterion: 'queen',
      polygons,
      output: output(4, 100, false)
    });
    expect(contiguity.getCapacityPlan({requiredCount: 80}).slots.capacity).toBe(100);
    const grown = contiguity.getCapacityPlan({requiredCount: 300});
    expect(grown.slots.capacity).toBe(Math.ceil(300 * 1.25));
    expect(grown.pairs.capacity).toBe(375 * GPU_CONTIGUITY_WEIGHTS_PAIR_CAPACITY_FACTOR);
    expect(grown.estimatedOutputBytes).toBe(375 * 8 + 5 * 4);
    expect(contiguity.getCapacityPlan({candidateOverflow: true}).pairs.capacity).toBe(800);
    const explicit = new GPUContiguityWeights({
      criterion: 'queen',
      polygons,
      pairCapacity: 50,
      output: output(4, 100, false)
    });
    expect(explicit.getCapacityPlan().pairs.capacity).toBe(50);
    expect(explicit.getCapacityPlan({candidateOverflow: true}).pairs.capacity).toBe(100);
  });

  for (const distances of [false, true]) {
    it(`GPULatticeWeights ${distances ? 'with' : 'without'} distances`, () => {
      const {graph, output} = setup();
      const lattice = new GPULatticeWeights({
        width: 7,
        height: 5,
        criterion: 'queen',
        output: output(35, 200, distances)
      });
      const bytes = getDeclaredTransientBytes(graph, () => lattice.getCommandNodes(graph));
      const plan = lattice.getCapacityPlan();
      expect(plan.estimatedPeakTransientBytes).toBe(bytes);
      expect(plan.estimatedOutputBytes).toBe(getOutputBytes(lattice.props.output));
      expect(lattice.getCapacityPlan({requiredCount: 220}).slots.capacity).toBe(275);
    });
  }
});
