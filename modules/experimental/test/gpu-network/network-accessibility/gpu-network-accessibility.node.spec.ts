// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUNetworkAccessibility,
  GPUNetworkCostMatrix,
  GPUNetworkSnapping,
  encodeGPUNetworkAccessibilityParameters,
  type GPUNetworkAccessibilityProps,
  type GPUNetworkCostMatrixProps,
  type GPUNetworkSnappingProps
} from '../../../src/gpu-network/network-accessibility';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

let viewCount = 0;

/** Creates a transient view with a graph-unique ID. */
function createView<Format extends 'uint32' | 'float32' | 'float32x2'>(
  graph: GPUCommandGraph,
  format: Format,
  length: number
) {
  return createTransientView(graph, `view-${viewCount++}`, format, length);
}

function createMatrixProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUNetworkCostMatrixProps> = {}
): GPUNetworkCostMatrixProps {
  return {
    id: 'matrix',
    offsets: createView(graph, 'uint32', 9),
    neighbors: createView(graph, 'uint32', 12),
    weights: createView(graph, 'float32', 12),
    seedNodes: createView(graph, 'uint32', 10),
    costs: createView(graph, 'float32', 5 * 8),
    ...overrides
  };
}

it('GPUNetworkCostMatrix schedules lane expansion and one reachability per batch', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const contributor = new GPUNetworkCostMatrix({
    ...createMatrixProps(graph),
    seedsPerRow: 2,
    laneCount: 2,
    maxIterations: 3,
    converged: createView(graph, 'uint32', 1)
  });
  expect(contributor.rowCount).toBe(5);
  expect(contributor.nodeCount).toBe(8);
  expect(contributor.batchCount).toBe(3);
  const ids = contributor.getCommandNodes(graph).map(node => node.id);
  expect(ids.slice(0, 3)).toEqual([
    'matrix-expand-offsets',
    'matrix-expand-edges',
    'matrix-expand-seeds'
  ]);
  for (const batch of [0, 1, 2]) {
    expect(ids).toContain(`matrix-batch-${batch}-seed`);
    expect(ids).toContain(`matrix-batch-${batch}-relax-2`);
    expect(ids).not.toContain(`matrix-batch-${batch}-relax-3`);
  }
  expect(ids.at(-1)).toBe('matrix-converged');
  // Deterministic across calls.
  const again = new GPUCommandGraph(createNullWebGPUDevice());
  expect(
    new GPUNetworkCostMatrix({
      ...createMatrixProps(again),
      seedsPerRow: 2,
      laneCount: 2,
      maxIterations: 3,
      converged: createView(again, 'uint32', 1)
    })
      .getCommandNodes(again)
      .map(node => node.id)
  ).toEqual(ids);
});

it('GPUNetworkCostMatrix validates shapes and lane counts', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  expect(new GPUNetworkCostMatrix(createMatrixProps(graph)).laneCount).toBe(5);
  expect(
    () =>
      new GPUNetworkCostMatrix(createMatrixProps(graph, {costs: createView(graph, 'float32', 13)}))
  ).toThrow(/multiple of the node count/);
  expect(() => new GPUNetworkCostMatrix(createMatrixProps(graph, {laneCount: 6}))).toThrow(
    /laneCount/
  );
  expect(
    () =>
      new GPUNetworkCostMatrix(
        createMatrixProps(graph, {
          seedCosts: createView(graph, 'float32', 3)
        })
      )
  ).toThrow(/seedCosts length/);
  expect(
    () =>
      new GPUNetworkCostMatrix(createMatrixProps(graph, {weights: createView(graph, 'float32', 2)}))
  ).toThrow(/weights length/);
  expect(() => new GPUNetworkCostMatrix(createMatrixProps(graph, {seedsPerRow: 0}))).toThrow(
    /seedsPerRow/
  );
});

function createAccessibilityProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUNetworkAccessibilityProps> = {}
): GPUNetworkAccessibilityProps {
  return {
    id: 'access',
    costs: createView(graph, 'float32', 4 * 10),
    opportunityWeights: createView(graph, 'float32', 4),
    parameters: createView(graph, 'float32', 4),
    ...overrides
  };
}

it('GPUNetworkAccessibility schedules scores and catchment nodes', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const contributor = new GPUNetworkAccessibility(
    createAccessibilityProps(graph, {
      cumulative: createView(graph, 'float32', 10),
      catchment: {
        demand: createView(graph, 'float32', 10),
        output: createView(graph, 'float32', 10)
      }
    })
  );
  expect(contributor.rowCount).toBe(4);
  expect(contributor.nodeCount).toBe(10);
  expect(contributor.getCommandNodes(graph).map(node => node.id)).toEqual([
    'access-scores',
    'access-catchment-ratios',
    'access-catchment-scores'
  ]);
  const origin = new GPUNetworkAccessibility(
    createAccessibilityProps(graph, {
      orientation: 'origin-rows',
      opportunityWeights: createView(graph, 'float32', 10),
      gravity: createView(graph, 'float32', 4)
    })
  );
  expect(origin.rowCount).toBe(4);
  expect(origin.nodeCount).toBe(10);
});

it('GPUNetworkAccessibility validates orientation, outputs, and parameters', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  expect(() => new GPUNetworkAccessibility(createAccessibilityProps(graph))).toThrow(
    /at least one/
  );
  expect(
    () =>
      new GPUNetworkAccessibility(
        createAccessibilityProps(graph, {
          gravity: createView(graph, 'float32', 4)
        })
      )
  ).toThrow(/origin count/);
  expect(
    () =>
      new GPUNetworkAccessibility(
        createAccessibilityProps(graph, {
          parameters: createView(graph, 'float32', 2),
          gravity: createView(graph, 'float32', 10)
        })
      )
  ).toThrow(/parameters/);
  expect(
    () =>
      new GPUNetworkAccessibility(
        createAccessibilityProps(graph, {
          orientation: 'origin-rows',
          opportunityWeights: createView(graph, 'float32', 10),
          catchment: {
            demand: createView(graph, 'float32', 10),
            output: createView(graph, 'float32', 10)
          }
        })
      )
  ).toThrow(/opportunity-rows/);
  expect(
    () =>
      new GPUNetworkAccessibility(
        createAccessibilityProps(graph, {
          opportunityWeights: createView(graph, 'float32', 3)
        })
      )
  ).toThrow(/multiple/);
});

it('encodeGPUNetworkAccessibilityParameters packs threshold, decay code, beta, minimum cost', () => {
  expect(Array.from(encodeGPUNetworkAccessibilityParameters({threshold: 30}))).toEqual([
    30, 0, 0, 1
  ]);
  expect(
    Array.from(
      encodeGPUNetworkAccessibilityParameters({
        threshold: Infinity,
        decay: 'power',
        beta: 2,
        minimumCost: 0.5
      })
    )
  ).toEqual([Infinity, 2, 2, 0.5]);
  expect(
    encodeGPUNetworkAccessibilityParameters({
      threshold: 1,
      decay: 'exponential'
    })[1]
  ).toBe(1);
});

function createSnappingProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUNetworkSnappingProps> = {}
): GPUNetworkSnappingProps {
  return {
    id: 'snap',
    points: createView(graph, 'float32x2', 6),
    nodePositions: createView(graph, 'float32x2', 4),
    edgeSources: createView(graph, 'uint32', 5),
    edgeTargets: createView(graph, 'uint32', 5),
    snappedEdges: createView(graph, 'uint32', 6),
    ...overrides
  };
}

it('GPUNetworkSnapping schedules a scan or a BVH join and optional outputs', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const scan = new GPUNetworkSnapping(
    createSnappingProps(graph, {
      seedNodes: createView(graph, 'uint32', 12),
      seedCosts: createView(graph, 'float32', 12),
      snappedPositions: createView(graph, 'float32x2', 6)
    })
  );
  expect(scan.getCommandNodes(graph).map(node => node.id)).toEqual([
    'snap-scan',
    'snap-costs',
    'snap-positions',
    'snap-seeds'
  ]);
  const joinGraph = new GPUCommandGraph(createNullWebGPUDevice());
  const join = new GPUNetworkSnapping(
    createSnappingProps(joinGraph, {
      edgeSources: undefined,
      offsets: createView(joinGraph, 'uint32', 5),
      maxSnapDistance: createView(joinGraph, 'float32', 1),
      candidateCapacity: 64
    })
  );
  const ids = join.getCommandNodes(joinGraph).map(node => node.id);
  expect(ids[0]).toBe('snap-edge-sources');
  expect(ids[1]).toBe('snap-segments');
  expect(ids.some(id => id.startsWith('snap-join-'))).toBe(true);
  expect(ids.at(-1)).toBe('snap-project');
});

it('GPUNetworkSnapping validates edge inputs, capacities, and output lengths', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  expect(
    () =>
      new GPUNetworkSnapping(createSnappingProps(graph, {offsets: createView(graph, 'uint32', 5)}))
  ).toThrow(/exactly one of edgeSources and offsets/);
  expect(() => new GPUNetworkSnapping(createSnappingProps(graph, {candidateCapacity: 8}))).toThrow(
    /requires maxSnapDistance/
  );
  expect(
    () =>
      new GPUNetworkSnapping(
        createSnappingProps(graph, {
          seedNodes: createView(graph, 'uint32', 12)
        })
      )
  ).toThrow(/together/);
  expect(
    () =>
      new GPUNetworkSnapping(
        createSnappingProps(graph, {
          snapFractions: createView(graph, 'float32', 5)
        })
      )
  ).toThrow(/point count/);
  expect(
    () =>
      new GPUNetworkSnapping(
        createSnappingProps(graph, {
          edgeCosts: createView(graph, 'float32', 4)
        })
      )
  ).toThrow(/edgeCosts length/);
});
