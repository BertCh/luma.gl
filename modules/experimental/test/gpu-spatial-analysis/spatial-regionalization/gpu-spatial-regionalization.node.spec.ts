// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  getGPUAZPRegionsParameterValues,
  getGPUMaxPRegionsParameterValues,
  getGPUSkaterRegionsParameterValues,
  getGPUWardRegionsParameterValues,
  GPU_AZP_REGIONS_PARAMETER_LENGTH,
  GPU_MAX_P_REGIONS_PARAMETER_LENGTH,
  GPU_MAX_P_STATUS,
  GPU_REGION_PARTITION_EVALUATION_LAYOUT,
  GPU_SKATER_REGIONS_PARAMETER_LENGTH,
  GPU_WARD_REGIONS_PARAMETER_LENGTH,
  GPUAZPRegions,
  GPUMaxPRegions,
  GPURegionPartitionEvaluation,
  GPUWardRegions,
  GPUSkaterRegions,
  GPUSpatialWeightsMinimumSpanningTree
} from '../../../src/gpu-spatial-analysis/spatial-regionalization/index';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import referenceCases from './skater-reference.json';
import {
  computePseudoF,
  selectMaxPStart,
  standardizeColumns,
  type RegionalizationCase
} from './spatial-regionalization-oracle';

let serial = 0;

function setup(rows: number, capacity: number) {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'regionalization-nodes'});
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `view-${serial++}`, format, length);
  const weights = {
    offsets: view('uint32', rows + 1),
    neighbors: view('uint32', capacity),
    weights: view('float32', capacity)
  };
  return {graph, view, weights};
}

it('GPUSpatialWeightsMinimumSpanningTree validates props and declares nodes', () => {
  const {graph, view, weights} = setup(6, 16);
  const base = {
    weights,
    values: view('float32', 12),
    columnCount: 2,
    output: {treeEdgeFlags: view('uint32', 16), componentLabels: view('uint32', 6)}
  };
  expect(() => new GPUSpatialWeightsMinimumSpanningTree({...base, columnCount: 3})).toThrow(
    /values length/
  );
  expect(
    () =>
      new GPUSpatialWeightsMinimumSpanningTree({
        ...base,
        output: {...base.output, treeEdgeFlags: view('uint32', 15)}
      })
  ).toThrow(/treeEdgeFlags length/);
  expect(
    () =>
      new GPUSpatialWeightsMinimumSpanningTree({
        ...base,
        standardize: false,
        output: {...base.output, standardizedValues: view('float32', 12)}
      })
  ).toThrow(/requires standardize/);
  const nodes = new GPUSpatialWeightsMinimumSpanningTree({
    ...base,
    output: {
      ...base.output,
      edges: {
        ids: view('uint32', 5),
        status: {count: view('uint32', 1), overflow: view('uint32', 1)}
      },
      edgeEndpoints: view('uint32', 10),
      edgeDissimilarities: view('float32', 5)
    }
  }).getCommandNodes(graph);
  expect(nodes[0].id).toBe('minimum-spanning-tree-standardize');
  // ceil(log2(6)) = 3 rounds.
  expect(nodes.filter(node => node.id.endsWith('-hook'))).toHaveLength(3);
  expect(nodes.at(-1)?.id).toBe('minimum-spanning-tree-edge-details');
});

it('GPUSkaterRegions validates props and declares one step per allowed cut', () => {
  const {graph, view, weights} = setup(6, 16);
  const base = {
    weights,
    treeEdgeFlags: view('uint32', 16),
    componentLabels: view('uint32', 6),
    values: view('float32', 12),
    columnCount: 2,
    maximumRegionCount: 4,
    parameters: view('uint32', GPU_SKATER_REGIONS_PARAMETER_LENGTH),
    output: {labels: view('uint32', 6)}
  };
  expect(() => new GPUSkaterRegions({...base, maximumRegionCount: 1})).toThrow(
    /maximumRegionCount/
  );
  expect(() => new GPUSkaterRegions({...base, parameters: view('uint32', 1)})).toThrow(
    /parameters/
  );
  expect(
    () => new GPUSkaterRegions({...base, output: {...base.output, cutEdges: view('uint32', 2)}})
  ).toThrow(/cutEdges/);
  // Every kernel stays within the 8 storage bindings the null device enforces.
  const nodes = new GPUSkaterRegions({
    ...base,
    output: {
      ...base.output,
      regionCount: view('uint32', 1),
      cutEdges: view('uint32', 3),
      cutGains: view('float32', 3)
    }
  }).getCommandNodes(graph);
  expect(nodes.filter(node => node.id.endsWith('-candidates'))).toHaveLength(3);
});

it('GPURegionPartitionEvaluation validates props and declares nodes', () => {
  const {graph, view, weights} = setup(6, 16);
  const base = {
    values: view('float32', 12),
    columnCount: 2,
    labels: view('uint32', 6),
    output: {summary: view('float32', GPU_REGION_PARTITION_EVALUATION_LAYOUT.length)}
  };
  expect(
    () => new GPURegionPartitionEvaluation({...base, output: {summary: view('float32', 4)}})
  ).toThrow(/summary/);
  expect(() => new GPURegionPartitionEvaluation({...base, labelCapacity: 0})).toThrow(
    /labelCapacity/
  );
  const nodes = new GPURegionPartitionEvaluation({...base, weights}).getCommandNodes(graph);
  // The stable label sort contributes its own nodes between the key and region kernels.
  const ids = nodes.map(node => node.id);
  expect(ids[0]).toBe('region-partition-evaluation-sort-keys');
  expect(ids.filter(id => !id.startsWith('region-partition-evaluation-sort-'))).toEqual([
    'region-partition-evaluation-label-offsets',
    'region-partition-evaluation-region-sums',
    'region-partition-evaluation-region-sums-fixup',
    'region-partition-evaluation-region-deviations',
    'region-partition-evaluation-region-deviations-fixup',
    'region-partition-evaluation-region-within',
    'region-partition-evaluation-links',
    'region-partition-evaluation-links-fixup',
    'region-partition-evaluation-row-tiles',
    'region-partition-evaluation-label-tiles',
    'region-partition-evaluation-means',
    'region-partition-evaluation-deviation-tiles',
    'region-partition-evaluation-summary'
  ]);
});

it('standardizeColumns matches sklearn StandardScaler (population standard deviation)', () => {
  // fix/work/C/U6/standard_scaler.py, scikit-learn 1.9.1: StandardScaler().fit_transform(table).
  const table = [1, 10, 2, 20, 4, 60, 7, 0];
  const expected = [
    -1.091089451179962, -0.5488212999484517, -0.6546536707079772, -0.10976425998969035,
    0.2182178902359924, 1.6464638998453551, 1.5275252316519468, -0.9878783399072131
  ];
  standardizeColumns(table, 2).forEach((value, index) => {
    expect(value).toBeCloseTo(expected[index], 12);
  });
  // A constant column is all zero, and a missing entry stays missing without poisoning the rest.
  expect(standardizeColumns([3, 3, 3], 1)).toEqual([0, 0, 0]);
  const withGap = standardizeColumns([1, Number.NaN, 3], 1);
  expect(withGap[0]).toBeCloseTo(-1, 12);
  expect(withGap[2]).toBeCloseTo(1, 12);
  expect(Number.isNaN(withGap[1])).toBe(true);
});

it('computePseudoF matches sklearn calinski_harabasz_score on the SKATER reference runs', () => {
  // pseudoF was written by fix/work/C/U6/pseudo_f.py (scikit-learn 1.9.1).
  let checked = 0;
  for (const data of referenceCases as RegionalizationCase[]) {
    for (const run of data.runs) {
      expect(computePseudoF(data.values, data.columns, run.labels)).toBeCloseTo(run.pseudoF, 9);
      checked++;
    }
  }
  expect(checked).toBeGreaterThan(10);
  // Undefined cases: one region, or as many regions as rows.
  expect(computePseudoF([1, 2, 3], 1, [0, 0, 0])).toBeNaN();
  expect(computePseudoF([1, 2, 3], 1, [0, 1, 2])).toBeNaN();
  // sklearn returns 1.0 when the within-region dispersion is zero.
  expect(computePseudoF([1, 1, 2, 2, 2], 1, [0, 0, 1, 1, 1])).toBe(1);
});

it('selectMaxPStart prefers fewer leftovers, then more regions, then lower SSD, then the lower start', () => {
  const start = (regionCount: number, infeasibleCount: number, withinSquares: number) => ({
    regionCount,
    infeasibleCount,
    withinSquares
  });
  // More regions, but with a leftover below the threshold: the clean construction wins.
  expect(selectMaxPStart([start(5, 1, 1), start(4, 0, 9)])).toBe(1);
  // Among clean constructions, the most regions.
  expect(selectMaxPStart([start(4, 0, 1), start(6, 0, 9), start(5, 0, 0)])).toBe(1);
  // Equal region counts: the smaller within-region sum of squares.
  expect(selectMaxPStart([start(4, 0, 3), start(4, 0, 2), start(4, 0, 2.5)])).toBe(1);
  // Full tie: the lowest start index.
  expect(selectMaxPStart([start(4, 0, 2), start(4, 0, 2)])).toBe(0);
});

function createSolverInputs(rows: number, capacity: number) {
  const {graph, view, weights} = setup(rows, capacity);
  return {graph, view, weights, values: view('float32', rows)};
}

it('GPUMaxPRegions validates props and declares multi-start nodes within the binding limit', () => {
  const {graph, view, weights, values} = createSolverInputs(6, 16);
  const base = {
    weights,
    values,
    extensiveValues: view('float32', 6),
    threshold: view('float32', 1),
    parameters: view('uint32', GPU_MAX_P_REGIONS_PARAMETER_LENGTH),
    output: {
      labels: view('uint32', 6),
      objective: view('float32', 1),
      status: view('uint32', GPU_MAX_P_STATUS.length)
    }
  };
  expect(() => new GPUMaxPRegions({...base, startCount: 0})).toThrow(/startCount/);
  expect(() => new GPUMaxPRegions({...base, startCount: 65})).toThrow(/startCount/);
  expect(() => new GPUMaxPRegions({...base, maximumIterations: 257})).toThrow(/maximumIterations/);
  expect(() => new GPUMaxPRegions({...base, parameters: view('uint32', 0)})).toThrow(
    /parameters must hold 1 words/
  );
  expect(
    () => new GPUMaxPRegions({...base, output: {...base.output, status: view('uint32', 4)}})
  ).toThrow(/status must hold 6 words/);
  expect(() => new GPUMaxPRegions({...base, maximumGrowthSteps: 0})).toThrow(/maximumGrowthSteps/);
  expect(() => new GPUMaxPRegions({...base, maximumGrowthSteps: 4097})).toThrow(
    /maximumGrowthSteps/
  );
  // Default: gated rounds shared by all starts, one statistics dispatch, then selection and copy;
  // no improvement.
  const nodes = new GPUMaxPRegions(base).getCommandNodes(graph);
  expect(nodes.filter(node => node.id.endsWith('-finalize-starts'))).toHaveLength(1);
  expect(nodes.filter(node => node.id.endsWith('-round-0-control'))).toHaveLength(1);
  expect(nodes.some(node => node.id.includes('-improvement-'))).toBe(false);
  // Directed weights are symmetrized through a transpose before the construction.
  expect(nodes.some(node => node.id.includes('adjacency-transpose'))).toBe(true);
  expect(nodes.some(node => node.id.endsWith('-standardize'))).toBe(true);
  // Symmetric weights skip the transpose; standardize false skips the z-scoring node.
  const symmetric = {
    ...weights,
    information: {rowCount: 6, transform: 'raw' as const, symmetric: true}
  };
  const lean = new GPUMaxPRegions({
    ...base,
    weights: symmetric,
    standardize: false,
    id: 'lean-max-p'
  }).getCommandNodes(graph);
  expect(lean.some(node => node.id.includes('adjacency'))).toBe(false);
  expect(lean.some(node => node.id.endsWith('-standardize'))).toBe(false);
  // The improvement phase appends the AZP search under the extensive constraint.
  const improving = new GPUMaxPRegions({
    ...base,
    id: 'improving-max-p',
    maximumIterations: 3
  }).getCommandNodes(graph);
  expect(improving.filter(node => node.id.includes('-improvement-iteration-'))).not.toHaveLength(0);
});

it('GPUAZPRegions validates the extensive constraint and declares the contiguity check', () => {
  const {graph, view, weights, values} = createSolverInputs(6, 16);
  const base = {
    weights,
    values,
    initialLabels: view('uint32', 6),
    regionCapacity: 2,
    maximumIterations: 2,
    parameters: view('uint32', GPU_AZP_REGIONS_PARAMETER_LENGTH),
    output: {labels: view('uint32', 6), objective: view('float32', 1), status: view('uint32', 4)}
  };
  expect(() => new GPUAZPRegions({...base, extensiveValues: view('float32', 6)})).toThrow(
    /required together/
  );
  expect(
    () =>
      new GPUAZPRegions({
        ...base,
        extensiveValues: view('float32', 5),
        extensiveThreshold: view('float32', 1)
      })
  ).toThrow(/extensiveValues length/);
  const nodes = new GPUAZPRegions({
    ...base,
    extensiveValues: view('float32', 6),
    extensiveThreshold: view('float32', 1)
  }).getCommandNodes(graph);
  const ids = nodes.map(node => node.id);
  expect(ids).toContain('azp-regions-initial-contiguity');
  expect(ids).toContain('azp-regions-pack-extensive');
  expect(ids).toContain('azp-regions-standardize');
  expect(ids.indexOf('azp-regions-initial-contiguity')).toBeLessThan(
    ids.indexOf('azp-regions-region-statistics')
  );
});

it('GPUWardRegions declares the standardize and symmetrization nodes', () => {
  const {graph, view, weights, values} = createSolverInputs(6, 16);
  const base = {
    weights,
    values,
    targetRegionCount: 2,
    maximumIterations: 1,
    parameters: view('uint32', GPU_WARD_REGIONS_PARAMETER_LENGTH),
    output: {labels: view('uint32', 6), objective: view('float32', 1), status: view('uint32', 4)}
  };
  const ids = new GPUWardRegions(base).getCommandNodes(graph).map(node => node.id);
  expect(ids).toContain('ward-regions-standardize');
  expect(ids.some(id => id.includes('adjacency-transpose'))).toBe(true);
  const plain = new GPUWardRegions({...base, id: 'plain-ward', standardize: false})
    .getCommandNodes(graph)
    .map(node => node.id);
  expect(plain).not.toContain('plain-ward-standardize');
});

it('regionalization packers write the documented per-frame words', () => {
  // AZP, Ward and Max-P share the `[seed]` layout, so Max-P hands its view to the AZP phase.
  for (const pack of [
    getGPUAZPRegionsParameterValues,
    getGPUWardRegionsParameterValues,
    getGPUMaxPRegionsParameterValues
  ]) {
    expect(Array.from(pack())).toEqual([0]);
    expect(Array.from(pack({seed: 0xffffffff}))).toEqual([0xffffffff]);
    expect(() => pack({seed: -1})).toThrow(/seed/);
    expect(() => pack({seed: 2 ** 32})).toThrow(/seed/);
    expect(() => pack({seed: 1.5})).toThrow(/seed/);
  }
  expect(GPU_AZP_REGIONS_PARAMETER_LENGTH).toBe(GPU_MAX_P_REGIONS_PARAMETER_LENGTH);
  const skater = getGPUSkaterRegionsParameterValues({regionCount: 5, minimumRegionSize: 3});
  expect(skater).toHaveLength(GPU_SKATER_REGIONS_PARAMETER_LENGTH);
  expect(Array.from(skater)).toEqual([5, 3]);
  expect(Array.from(getGPUSkaterRegionsParameterValues({regionCount: 2}))).toEqual([2, 1]);
  expect(() => getGPUSkaterRegionsParameterValues({regionCount: 0})).toThrow(/regionCount/);
  expect(() => getGPUSkaterRegionsParameterValues({regionCount: 2, minimumRegionSize: 0})).toThrow(
    /minimumRegionSize/
  );
});

it('seeded regionalization kernels read the seed from the parameters binding', () => {
  // A seed is per-frame (A-21): no kernel source may depend on it, and every kernel that hashes
  // with it binds `parameters`.
  const {graph, view, weights, values} = createSolverInputs(6, 16);
  const output = () => ({
    labels: view('uint32', 6),
    objective: view('float32', 1),
    status: view('uint32', GPU_MAX_P_STATUS.length)
  });
  const producers = [
    new GPUAZPRegions({
      weights,
      values,
      initialLabels: view('uint32', 6),
      regionCapacity: 2,
      maximumIterations: 2,
      parameters: view('uint32', GPU_AZP_REGIONS_PARAMETER_LENGTH),
      output: output()
    }),
    new GPUWardRegions({
      weights,
      values,
      targetRegionCount: 2,
      maximumIterations: 2,
      parameters: view('uint32', GPU_WARD_REGIONS_PARAMETER_LENGTH),
      output: output()
    }),
    new GPUWardRegions({
      id: 'ward-regions-mutual-nearest',
      weights,
      values,
      targetRegionCount: 2,
      maximumIterations: 2,
      mergeSchedule: 'mutual-nearest',
      parameters: view('uint32', GPU_WARD_REGIONS_PARAMETER_LENGTH),
      output: output()
    }),
    new GPUMaxPRegions({
      weights,
      values,
      extensiveValues: view('float32', 6),
      threshold: view('float32', 1),
      maximumIterations: 2,
      parameters: view('uint32', GPU_MAX_P_REGIONS_PARAMETER_LENGTH),
      output: output()
    })
  ];
  for (const producer of producers) {
    const sources = getGPUCommandNodeKernelSources(producer.getCommandNodes(graph));
    const seeded = sources.filter(({source}) => /tieKey|pairKey\(|philox4x32\(/.test(source));
    expect(seeded.length, producer.id).toBeGreaterThan(0);
    for (const {source} of seeded) {
      expect(source, producer.id).not.toMatch(/SEED/);
      expect(source, producer.id).toMatch(/parameters\[parametersOffset\]/);
    }
  }
});
