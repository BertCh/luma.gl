// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUNetworkCostWeights,
  GPU_NETWORK_COST_WEIGHTS_PARAMETER_LENGTH,
  getGPUNetworkCostWeightsParameterValues,
  type GPUNetworkCostWeightsProps
} from '../../../src/gpu-network/network-cost-weights';
import type {GPUSpatialWeights} from '../../../src/gpu-spatial-analysis/spatial-weights/spatial-weights';
import type {GPUNetworkTopology} from '../../../src/analysis-core/network-topology';
import {
  createNullWebGPUDevice,
  getParsedKernelSources,
  getStorageBindingCount,
  getWorkgroupMemoryBytes,
  CORE_WORKGROUP_STORAGE_LIMIT_BYTES
} from '../../utils/gpu-contributor-test-utils';
import {getNetworkCostWeightsReference} from './network-cost-weights-oracle';

let viewCount = 0;

function createView<Format extends 'uint32' | 'float32'>(
  graph: GPUCommandGraph,
  format: Format,
  length: number
) {
  return createTransientView(graph, `view-${viewCount++}`, format, length);
}

/** Props for a 3 x 5 matrix with `weightsRows` output rows and `capacity` slots. */
function createProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUNetworkCostWeightsProps> = {},
  weightsRows = 3,
  capacity = 15
): GPUNetworkCostWeightsProps {
  return {
    id: 'cost-weights',
    costs: createView(graph, 'float32', 15),
    rowCount: 3,
    parameters: createView(graph, 'float32', GPU_NETWORK_COST_WEIGHTS_PARAMETER_LENGTH),
    output: {
      offsets: createView(graph, 'uint32', weightsRows + 1),
      neighbors: createView(graph, 'uint32', capacity),
      weights: createView(graph, 'float32', capacity),
      distances: createView(graph, 'float32', capacity),
      status: {
        overflow: createView(graph, 'uint32', 1),
        requiredCount: createView(graph, 'uint32', 1)
      }
    },
    ...overrides
  };
}

it('getGPUNetworkCostWeightsParameterValues packs limit, decay, beta and floor', () => {
  expect(Array.from(getGPUNetworkCostWeightsParameterValues({costLimit: 10}))).toEqual([
    10, 0, 0, 1
  ]);
  expect(
    Array.from(
      getGPUNetworkCostWeightsParameterValues({
        costLimit: Infinity,
        decay: 'power',
        beta: 1.5,
        minimumCost: 0.5
      })
    )
  ).toEqual([Infinity, 2, 1.5, 0.5]);
  expect(
    Array.from(getGPUNetworkCostWeightsParameterValues({costLimit: 4, decay: 'gaussian'}))
  ).toEqual([4, 3, 0, 1]);
  expect(() => getGPUNetworkCostWeightsParameterValues({costLimit: Number.NaN})).toThrow(
    /costLimit/
  );
  expect(() =>
    getGPUNetworkCostWeightsParameterValues({costLimit: Infinity, decay: 'gaussian'})
  ).toThrow(/gaussian/);
  expect(() => getGPUNetworkCostWeightsParameterValues({costLimit: 1, beta: -1})).toThrow(/beta/);
  expect(() =>
    getGPUNetworkCostWeightsParameterValues({costLimit: 1, decay: 'power', minimumCost: 0})
  ).toThrow(/minimumCost/);
  expect(() =>
    getGPUNetworkCostWeightsParameterValues({
      costLimit: 1,
      decay: 'linear' as unknown as 'none'
    })
  ).toThrow(/decay/);
});

it('GPUNetworkCostWeights resolves the matrix shape for both orientations', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const rows = new GPUNetworkCostWeights(createProps(graph));
  expect(rows.rowCount).toBe(3);
  expect(rows.columnCount).toBe(5);
  expect(rows.locationCount).toBe(5);
  expect(rows.weightsRowCount).toBe(3);
  expect(rows.capacity).toBe(15);
  expect(rows.outputWeights.information).toMatchObject({
    rowCount: 3,
    transform: 'kernel',
    provenance: {source: 'GPUNetworkCostWeights', criterion: 'network-cost'}
  });
  expect(rows.outputWeights.status?.overflow).toBe(rows.props.output.status?.overflow);
  expect(rows.outputWeights.status?.requiredCount).toBe(rows.props.output.status?.requiredCount);

  const transposed = new GPUNetworkCostWeights(
    createProps(graph, {orientation: 'columns', columns: createView(graph, 'uint32', 4)}, 4)
  );
  expect(transposed.locationCount).toBe(4);
  expect(transposed.weightsRowCount).toBe(4);
});

it('GPUNetworkCostWeights rejects malformed props', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  expect(() => new GPUNetworkCostWeights(createProps(graph, {rowCount: 4}))).toThrow(
    /multiple of rowCount/
  );
  expect(() => new GPUNetworkCostWeights(createProps(graph, {rowCount: 0}))).toThrow(/rowCount/);
  expect(
    () =>
      new GPUNetworkCostWeights(createProps(graph, {orientation: 'diagonal' as unknown as 'rows'}))
  ).toThrow(/orientation/);
  expect(
    () =>
      new GPUNetworkCostWeights(createProps(graph, {parameters: createView(graph, 'float32', 3)}))
  ).toThrow(/parameters/);
  // `'columns'` has one weights row per location.
  expect(() => new GPUNetworkCostWeights(createProps(graph, {orientation: 'columns'}))).toThrow(
    /output.offsets/
  );
  const props = createProps(graph);
  expect(
    () =>
      new GPUNetworkCostWeights({
        ...props,
        output: {...props.output, weights: createView(graph, 'float32', 14)}
      })
  ).toThrow(/output.weights length/);
  expect(
    () =>
      new GPUNetworkCostWeights({
        ...props,
        output: {
          ...props.output,
          neighbors: props.costs as unknown as typeof props.output.neighbors
        }
      })
  ).toThrow();
});

it('GPUNetworkCostWeights schedules count, scan, fill and finalize kernels within core limits', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const contributor = new GPUNetworkCostWeights(
    createProps(graph, {columns: createView(graph, 'uint32', 5)})
  );
  const nodes = contributor.getCommandNodes(graph);
  const ids = nodes.map(node => node.id);
  expect(ids[0]).toBe('cost-weights-count');
  expect(ids.some(id => id.startsWith('cost-weights-offsets'))).toBe(true);
  expect(ids.at(-2)).toBe('cost-weights-fill');
  expect(ids.at(-1)).toBe('cost-weights-finalize');
  const sources = getParsedKernelSources(nodes);
  for (const id of ['cost-weights-count', 'cost-weights-fill', 'cost-weights-finalize']) {
    const source = sources.get(id);
    expect(source, id).toBeDefined();
    expect(getStorageBindingCount(source!)).toBeLessThanOrEqual(8);
    expect(getWorkgroupMemoryBytes(source!)).toBeLessThanOrEqual(
      CORE_WORKGROUP_STORAGE_LIMIT_BYTES
    );
  }
  // The fill kernel reads the column map and writes distances; the finalize writes both status words.
  expect(sources.get('cost-weights-fill')).toContain('columns[columnsOffset + neighbor]');
  expect(sources.get('cost-weights-fill')).toContain('distances[distancesOffset + slot]');
  expect(sources.get('cost-weights-finalize')).toContain('requiredCount[requiredCountOffset]');
});

it('GPUNetworkCostWeights transposes through the location index', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const contributor = new GPUNetworkCostWeights(createProps(graph, {orientation: 'columns'}, 5));
  const sources = getParsedKernelSources(contributor.getCommandNodes(graph));
  // Row = location, neighbor = matrix row.
  expect(sources.get('cost-weights-count')).toContain('let column = row;');
  expect(sources.get('cost-weights-count')).toContain(
    'costs[costsOffset + neighbor * MATRIX_COLUMNS + column]'
  );
});

it('getNetworkCostWeightsReference applies the in-range rule, ordering and capacity', () => {
  const costs = [0, 2, Infinity, -1, 5, Number.NaN, 1, 3];
  const full = getNetworkCostWeightsReference({
    costs,
    rowCount: 2,
    orientation: 'rows',
    settings: {costLimit: 3},
    capacity: 8
  });
  expect(full.offsets).toEqual([0, 2, 4]);
  expect(full.neighbors).toEqual([0, 1, 2, 3]);
  expect(full.distances).toEqual([0, 2, 1, 3]);
  expect(full.weights).toEqual([1, 1, 1, 1]);
  expect(full.requiredCount).toBe(4);
  expect(full.overflow).toBe(0);
  const transposed = getNetworkCostWeightsReference({
    costs,
    rowCount: 2,
    orientation: 'columns',
    settings: {costLimit: 3},
    capacity: 3
  });
  // Location rows 0..3 list the matrix rows that reach them; the fourth slot is dropped.
  expect(transposed.offsets).toEqual([0, 1, 2, 3, 3]);
  expect(transposed.neighbors).toEqual([0, 0, 1]);
  expect(transposed.requiredCount).toBe(4);
  expect(transposed.overflow).toBe(1);
  const mapped = getNetworkCostWeightsReference({
    costs,
    rowCount: 2,
    columns: [3, 9, 1],
    orientation: 'rows',
    settings: {costLimit: 3, decay: 'exponential', beta: 1},
    capacity: 6
  });
  expect(mapped.offsets).toEqual([0, 1, 2]);
  expect(mapped.neighbors).toEqual([2, 0]);
  expect(mapped.weights[0]).toBeCloseTo(Math.exp(-2), 12);
});

it('GPUNetworkTopology is structurally a GPUSpatialWeights CSR', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const topology: GPUNetworkTopology = {
    offsets: createView(graph, 'uint32', 4),
    neighbors: createView(graph, 'uint32', 6),
    weights: createView(graph, 'float32', 6)
  };
  // Compile-time check (the test-tree typecheck): a topology is accepted where weights are read.
  const weights: GPUSpatialWeights = topology;
  expect(weights.neighbors).toBe(topology.neighbors);
});
