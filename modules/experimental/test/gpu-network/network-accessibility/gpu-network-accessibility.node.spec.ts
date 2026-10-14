// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUNetworkAccessibility,
  GPUNetworkCostMatrix,
  GPUNetworkSnapping,
  getGPUNetworkAccessibilityParameterValues,
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
  overrides: Partial<GPUNetworkCostMatrixProps> = {},
  topology: Partial<GPUNetworkCostMatrixProps['topology']> = {},
  output: Partial<GPUNetworkCostMatrixProps['output']> = {}
): GPUNetworkCostMatrixProps {
  return {
    id: 'matrix',
    topology: {
      offsets: createView(graph, 'uint32', 9),
      neighbors: createView(graph, 'uint32', 12),
      weights: createView(graph, 'float32', 12),
      ...topology
    },
    startNodes: createView(graph, 'uint32', 10),
    output: {costs: createView(graph, 'float32', 5 * 8), ...output},
    ...overrides
  };
}

it('GPUNetworkCostMatrix schedules lane expansion and one reachability per batch', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const contributor = new GPUNetworkCostMatrix({
    ...createMatrixProps(graph, {}, {}, {status: {converged: createView(graph, 'uint32', 1)}}),
    startsPerRow: 2,
    laneCount: 2,
    maximumIterations: 3
  });
  expect(contributor.rowCount).toBe(5);
  expect(contributor.nodeCount).toBe(8);
  expect(contributor.batchCount).toBe(3);
  const ids = contributor.getCommandNodes(graph).map(node => node.id);
  // Lanes share the CSR, so only the seeds are expanded.
  expect(ids[0]).toBe('matrix-expand-seeds');
  expect(ids.some(id => id.includes('expand-offsets') || id.includes('expand-edges'))).toBe(false);
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
      ...createMatrixProps(again, {}, {}, {status: {converged: createView(again, 'uint32', 1)}}),
      startsPerRow: 2,
      laneCount: 2,
      maximumIterations: 3
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
      new GPUNetworkCostMatrix(
        createMatrixProps(graph, {}, {}, {costs: createView(graph, 'float32', 13)})
      )
  ).toThrow(/multiple of the node count/);
  expect(() => new GPUNetworkCostMatrix(createMatrixProps(graph, {laneCount: 6}))).toThrow(
    /laneCount/
  );
  expect(
    () =>
      new GPUNetworkCostMatrix(
        createMatrixProps(graph, {
          startCosts: createView(graph, 'float32', 3)
        })
      )
  ).toThrow(/startCosts length/);
  expect(
    () =>
      new GPUNetworkCostMatrix(
        createMatrixProps(graph, {}, {weights: createView(graph, 'float32', 2)})
      )
  ).toThrow(/topology.weights length/);
  expect(() => new GPUNetworkCostMatrix(createMatrixProps(graph, {startsPerRow: 0}))).toThrow(
    /startsPerRow/
  );
  const missingWeights = createMatrixProps(graph);
  expect(
    () =>
      new GPUNetworkCostMatrix({
        ...missingWeights,
        topology: {...missingWeights.topology, weights: undefined as never}
      })
  ).toThrow(/topology.weights is required/);
  expect(
    () =>
      new GPUNetworkCostMatrix(
        createMatrixProps(graph, {}, {offsets: createView(graph, 'float32' as never, 9) as never})
      )
  ).toThrow(/topology.offsets/);
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
  // A-19: the catchment composes GPUNetworkCostWeights (facility rows and the transpose) with the
  // one floating catchment kernel, GPUCatchmentAccessibility.
  const ids = contributor.getCommandNodes(graph).map(node => node.id);
  expect(ids[0]).toBe('access-scores');
  expect(ids.slice(-2)).toEqual(['access-catchment-ratios', 'access-catchment-scores']);
  for (const side of ['facility', 'demand']) {
    for (const stage of ['count', 'fill', 'finalize']) {
      expect(ids).toContain(`access-catchment-${side}-weights-${stage}`);
    }
  }
  expect(ids.indexOf('access-catchment-demand-weights-finalize')).toBeLessThan(
    ids.indexOf('access-catchment-ratios')
  );
  const threeStep = new GPUNetworkAccessibility(
    createAccessibilityProps(graph, {
      id: 'access3',
      catchment: {
        method: '3sfca',
        demand: createView(graph, 'float32', 10),
        output: createView(graph, 'float32', 10)
      }
    })
  );
  expect(threeStep.getCommandNodes(graph).map(node => node.id)).toContain(
    'access3-catchment-selection'
  );
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

it('getGPUNetworkAccessibilityParameterValues packs threshold, decay code, beta, minimum cost', () => {
  expect(Array.from(getGPUNetworkAccessibilityParameterValues({threshold: 30}))).toEqual([
    30, 0, 0, 1
  ]);
  expect(
    Array.from(
      getGPUNetworkAccessibilityParameterValues({
        threshold: Infinity,
        decay: 'power',
        beta: 2,
        minimumCost: 0.5
      })
    )
  ).toEqual([Infinity, 2, 2, 0.5]);
  expect(
    getGPUNetworkAccessibilityParameterValues({
      threshold: 1,
      decay: 'exponential'
    })[1]
  ).toBe(1);
});

function createSnappingProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUNetworkSnappingProps> = {},
  output: Partial<GPUNetworkSnappingProps['output']> = {},
  topology: Partial<GPUNetworkSnappingProps['topology']> = {}
): GPUNetworkSnappingProps {
  return {
    id: 'snap',
    positions: createView(graph, 'float32x2', 6),
    topology: {
      offsets: createView(graph, 'uint32', 5),
      neighbors: createView(graph, 'uint32', 5),
      nodePositions: createView(graph, 'float32x2', 4),
      ...topology
    },
    output: {snappedEdges: createView(graph, 'uint32', 6), ...output},
    ...overrides
  };
}

it('GPUNetworkSnapping schedules a scan or a BVH join and optional outputs', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const scan = new GPUNetworkSnapping(
    createSnappingProps(
      graph,
      {},
      {
        startNodes: createView(graph, 'uint32', 12),
        startCosts: createView(graph, 'float32', 12),
        snappedPositions: createView(graph, 'float32x2', 6)
      }
    )
  );
  expect(scan.getCommandNodes(graph).map(node => node.id)).toEqual([
    'snap-edge-sources-clear',
    'snap-edge-sources',
    'snap-scan',
    'snap-costs',
    'snap-positions',
    'snap-seeds'
  ]);
  const joinGraph = new GPUCommandGraph(createNullWebGPUDevice());
  const join = new GPUNetworkSnapping(
    createSnappingProps(
      joinGraph,
      {
        maximumSnapDistance: createView(joinGraph, 'float32', 1),
        candidateCapacity: 64
      },
      {status: {overflow: createView(joinGraph, 'uint32', 1)}}
    )
  );
  const ids = join.getCommandNodes(joinGraph).map(node => node.id);
  // The edge-source rows are cleared first, so rows past offsets[nodeCount] are never stale.
  expect(ids[0]).toBe('snap-edge-sources-clear');
  expect(ids[1]).toBe('snap-edge-sources');
  expect(ids[2]).toBe('snap-segments');
  expect(ids.some(id => id.startsWith('snap-join-'))).toBe(true);
  expect(ids.at(-1)).toBe('snap-project');
});

it('GPUNetworkSnapping validates topology, capacities, and output lengths', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  expect(
    () =>
      new GPUNetworkSnapping(
        createSnappingProps(graph, {}, {}, {nodePositions: createView(graph, 'float32x2', 3)})
      )
  ).toThrow(/topology.nodePositions must contain one row per node/);
  const missingPositions = createSnappingProps(graph);
  expect(
    () =>
      new GPUNetworkSnapping({
        ...missingPositions,
        topology: {...missingPositions.topology, nodePositions: undefined as never}
      })
  ).toThrow(/topology.nodePositions is required/);
  expect(
    () =>
      new GPUNetworkSnapping(
        createSnappingProps(graph, {}, {}, {weights: createView(graph, 'float32', 4)})
      )
  ).toThrow(/topology.weights length/);
  expect(() => new GPUNetworkSnapping(createSnappingProps(graph, {candidateCapacity: 8}))).toThrow(
    /requires maximumSnapDistance/
  );
  // Rule 9: a candidate capacity can drop candidates, so its overflow flag is required.
  expect(
    () =>
      new GPUNetworkSnapping(
        createSnappingProps(graph, {
          candidateCapacity: 8,
          maximumSnapDistance: createView(graph, 'float32', 1)
        })
      )
  ).toThrow(/candidateCapacity requires output\.status\.overflow/);
  expect(
    () =>
      new GPUNetworkSnapping(
        createSnappingProps(graph, {}, {startNodes: createView(graph, 'uint32', 12)})
      )
  ).toThrow(/together/);
  expect(
    () =>
      new GPUNetworkSnapping(
        createSnappingProps(graph, {}, {snapFractions: createView(graph, 'float32', 5)})
      )
  ).toThrow(/position count/);
});
