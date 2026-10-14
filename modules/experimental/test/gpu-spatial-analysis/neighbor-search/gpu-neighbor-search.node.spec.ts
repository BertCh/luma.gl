// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUSpatialContext} from '../../../src/analysis-core/contracts/spatial-context';
import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  getGPUNeighborSearchParameterValues,
  getGPUNeighborSearchWeightTransform,
  GPUNeighborSearch,
  GPU_NEIGHBOR_SEARCH_GREAT_CIRCLE_SUPPORT,
  GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH,
  type GPUNeighborSearchProps
} from '../../../src/gpu-spatial-analysis/neighbor-search';
import {
  createNullWebGPUDevice,
  getGPUCommandNodePipelineKeys,
  getGPUCommandNodeKernelSources,
  getParsedKernelSources,
  getPrivateMemoryBytes,
  getStorageBindingCount,
  getWorkgroupMemoryBytes,
  CORE_PRIVATE_MEMORY_LIMIT_BYTES,
  CORE_WORKGROUP_STORAGE_LIMIT_BYTES
} from '../../utils/gpu-contributor-test-utils';
import {
  computeNeighborSearchOracle,
  createSeededRandom,
  getNeighborWeight
} from './neighbor-search-oracle';
import {
  computeNeighborSearchBVHModel,
  computeNeighborSearchOccupancy
} from './neighbor-search-bvh-model';
import {GPU_NEIGHBOR_SEARCH_BVH_OCCUPANCY_THRESHOLD} from '../../../src/gpu-spatial-analysis/neighbor-search';

let serial = 0;

function createProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUNeighborSearchProps> = {}
): GPUNeighborSearchProps {
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `view-${serial++}`, format, length);
  return {
    mode: 'knn',
    k: 4,
    positions: view('float32x2', 10),
    parameters: view('float32', GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH),
    gridSize: [4, 4],
    output: {
      offsets: view('uint32', 11),
      neighbors: view('uint32', 40),
      weights: view('float32', 40),
      distances: view('float32', 40),
      status: {overflow: view('uint32', 1)}
    },
    ...overrides
  };
}

it('GPUNeighborSearch validates modes, k, grid, weights and lengths', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'neighbor-search-validation'});
  expect(() => new GPUNeighborSearch(createProps(graph))).not.toThrow();
  expect(
    () => new GPUNeighborSearch(createProps(graph, {mode: 'radius', k: undefined}))
  ).not.toThrow();
  for (const k of [0, 33, 2.5, undefined]) {
    expect(() => new GPUNeighborSearch(createProps(graph, {k})), `k ${k}`).toThrow(/k must/);
  }
  expect(() => new GPUNeighborSearch(createProps(graph, {gridSize: [0, 4]}))).toThrow(/gridSize/);
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `bad-${serial++}`, format, length);
  // Offsets must have queryRows + 1 entries.
  expect(
    () =>
      new GPUNeighborSearch(
        createProps(graph, {
          output: {
            offsets: view('uint32', 10),
            neighbors: view('uint32', 4),
            weights: view('float32', 4),
            status: {overflow: view('uint32', 1)}
          }
        })
      )
  ).toThrow(/offsets length/);
  // Cross joins size the offsets by the query rows.
  expect(
    () =>
      new GPUNeighborSearch(
        createProps(graph, {
          queryPositions: view('float32x2', 3),
          output: {
            offsets: view('uint32', 4),
            neighbors: view('uint32', 4),
            weights: view('float32', 4),
            status: {overflow: view('uint32', 1)}
          }
        })
      )
  ).not.toThrow();
  expect(
    () =>
      new GPUNeighborSearch(
        createProps(graph, {
          output: {
            offsets: view('uint32', 11),
            neighbors: view('uint32', 4),
            weights: view('float32', 5),
            status: {overflow: view('uint32', 1)}
          }
        })
      )
  ).toThrow(/weights length/);
  expect(() => new GPUNeighborSearch(createProps(graph, {mask: view('uint32', 9)}))).toThrow(
    /mask length/
  );
  expect(() => new GPUNeighborSearch(createProps(graph, {queryMask: view('uint32', 9)}))).toThrow(
    /queryMask length/
  );
  expect(() => new GPUNeighborSearch(createProps(graph, {parameters: view('float32', 3)}))).toThrow(
    /parameters/
  );
  const counts = view('uint32', 10);
  const base = createProps(graph);
  expect(
    () =>
      new GPUNeighborSearch(
        createProps(graph, {mask: counts, output: {...base.output, neighborCounts: counts}})
      )
  ).toThrow(/must not share/);
});

it('GPUNeighborSearch emits deterministic node IDs in both modes and every join shape', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'neighbor-search-nodes'});
  const knn = new GPUNeighborSearch({...createProps(graph), id: 'knn'});
  const first = knn.getCommandNodes(graph).map(node => node.id);
  expect(first).toContain('knn-knn');
  expect(first).toContain('knn-emit');
  expect(first.at(-1)).toBe('knn-weights');
  const radius = new GPUNeighborSearch({
    ...createProps(graph, {
      mode: 'radius',
      queryPositions: createTransientView(graph, 'q', 'float32x2', 10)
    }),
    id: 'radius'
  });
  const ids = radius.getCommandNodes(graph).map(node => node.id);
  expect(ids).toContain('radius-count');
  expect(ids).toContain('radius-sort-rows');
  expect(new Set(ids).size).toBe(ids.length);
});

it('getGPUNeighborSearchParameterValues packs the layout and rejects unknown enums', () => {
  const values = getGPUNeighborSearchParameterValues({
    bounds: [1, 2, 3, 4],
    radius: 5,
    weightKind: 'kernel',
    power: 2,
    distanceFloor: 0.25,
    kernel: 'bisquare',
    rowStandardize: true
  });
  expect(Array.from(values)).toEqual([1, 2, 3, 4, 5, 2, 2, 0.25, 3, 1, 0, 0]);
  expect(getGPUNeighborSearchParameterValues({bounds: [0, 0, 1, 1]})[4]).toBe(Infinity);
  expect(() =>
    getGPUNeighborSearchParameterValues({bounds: [0, 0, 1, 1], weightKind: 'nope' as 'binary'})
  ).toThrow(/weightKind/);
});

it('the neighbor-search oracle selects lowest IDs on ties and sorts rows by ID', () => {
  // Four points at distance 1 from the origin point 0, plus a far point.
  const positions = new Float32Array([5, 5, 6, 5, 4, 5, 5, 6, 5, 4, 9, 9]);
  const parameters = {bounds: [0, 0, 10, 10]} as const;
  const result = computeNeighborSearchOracle({mode: 'knn', k: 2, positions, parameters});
  expect(result.neighbors.slice(0, 2)).toEqual([1, 2]);
  const radius = computeNeighborSearchOracle({
    mode: 'radius',
    positions,
    parameters: {...parameters, radius: 1}
  });
  expect(radius.neighbors.slice(radius.offsets[0], radius.offsets[1])).toEqual([1, 2, 3, 4]);
  expect(radius.counts[5]).toBe(0);
  expect(
    getNeighborWeight(0.5, 1, {...parameters, weightKind: 'kernel', kernel: 'triangular'})
  ).toBe(0.5);
  expect(getNeighborWeight(0, 1, {...parameters, weightKind: 'inverseDistance'})).toBe(0);
});

// ---- spatial context, outputWeights and invalidCount (U5) ----

it('GPUNeighborSearch resolves spatial contexts: planar default, great-circle, everything else throws', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'neighbor-search-context'});
  const search = new GPUNeighborSearch(createProps(graph));
  expect(search.spatialContext).toMatchObject({coordinateSpace: 'planar', metric: 'native'});
  const planar = {coordinateSpace: 'planar', metric: 'native', units: 'native'} as const;
  expect(() => new GPUNeighborSearch(createProps(graph, {spatialContext: planar}))).not.toThrow();
  const greatCircle = {coordinateSpace: 'longitude-latitude', metric: 'great-circle'} as const;
  for (const units of [undefined, 'meters', 'kilometers'] as const) {
    expect(
      () => new GPUNeighborSearch(createProps(graph, {spatialContext: {...greatCircle, units}}))
    ).not.toThrow();
  }
  expect(
    () =>
      new GPUNeighborSearch(
        createProps(graph, {spatialContext: {...greatCircle, sphereRadius: 3389500}})
      )
  ).not.toThrow();
  const rejected = [
    {coordinateSpace: 'longitude-latitude', metric: 'none'},
    {coordinateSpace: 'longitude-latitude', metric: 'rhumb'},
    {coordinateSpace: 'longitude-latitude', metric: 'ellipsoidal'},
    {coordinateSpace: 'discrete-grid', metric: 'none', gridFamily: 'h3'},
    {coordinateSpace: 'topology-only', metric: 'none'},
    {coordinateSpace: 'planar', metric: 'none'},
    {...greatCircle, units: 'native'}
  ] as const;
  for (const spatialContext of rejected) {
    expect(
      () => new GPUNeighborSearch(createProps(graph, {spatialContext: spatialContext as never})),
      JSON.stringify(spatialContext)
    ).toThrow();
  }
  expect(GPU_NEIGHBOR_SEARCH_GREAT_CIRCLE_SUPPORT.coordinateSpaces).toEqual([
    'planar',
    'longitude-latitude'
  ]);
});

it('GPUNeighborSearch great-circle node IDs match the planar ones and compile for every join shape', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'neighbor-search-geo-nodes'});
  const geo = {coordinateSpace: 'longitude-latitude', metric: 'great-circle'} as const;
  // The grid structure only, also used by default and by great-circle search.
  const planarIds = new GPUNeighborSearch({
    ...createProps(graph, {searchStructure: 'grid'}),
    id: 'a'
  })
    .getCommandNodes(graph)
    .map(node => node.id.replace(/^a/, ''));
  const geoIds = new GPUNeighborSearch({...createProps(graph, {spatialContext: geo}), id: 'b'})
    .getCommandNodes(graph)
    .map(node => node.id.replace(/^b/, ''));
  expect(geoIds).toEqual(planarIds);
  expect(() =>
    new GPUNeighborSearch({
      ...createProps(graph, {mode: 'radius', spatialContext: geo}),
      id: 'c'
    }).getCommandNodes(graph)
  ).not.toThrow();
});

it('GPUNeighborSearch outputWeights carries rowCount, provenance, symmetry and status', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'neighbor-search-output'});
  const view = <Format extends 'uint32'>(format: Format, length: number) =>
    createTransientView(graph, `status-${serial++}`, format, length);
  const total = view('uint32', 1);
  const invalid = view('uint32', 1);
  const base = createProps(graph);
  const knn = new GPUNeighborSearch(
    createProps(graph, {
      output: {
        ...base.output,
        status: {...base.output.status, requiredCount: total, invalidCount: invalid}
      }
    })
  );
  expect(knn.outputWeights.information).toEqual({
    rowCount: 10,
    transform: 'unknown',
    symmetric: 'unknown',
    hasSelfWeights: false,
    provenance: {source: 'GPUNeighborSearch', criterion: 'knn', k: 4}
  });
  expect(knn.outputWeights.status?.requiredCount).toBe(total);
  expect(knn.outputWeights.status?.invalidCount).toBe(invalid);
  expect(knn.outputWeights.status?.overflow).toBe(knn.props.output.status.overflow);
  expect(knn.outputWeights.offsets).toBe(knn.props.output.offsets);

  const band = new GPUNeighborSearch(
    createProps(graph, {mode: 'radius', k: undefined, weightTransform: 'binary'})
  );
  expect(band.outputWeights.information).toMatchObject({
    transform: 'binary',
    symmetric: true,
    provenance: {criterion: 'distance-band'}
  });
  const rowStandardized = new GPUNeighborSearch(
    createProps(graph, {mode: 'radius', k: undefined, weightTransform: 'row'})
  );
  expect(rowStandardized.outputWeights.information?.symmetric).toBe('unknown');
  const cross = new GPUNeighborSearch(
    createProps(graph, {
      mode: 'radius',
      k: undefined,
      weightTransform: 'kernel',
      queryPositions: createTransientView(graph, `query-${serial++}`, 'float32x2', 3),
      output: {
        offsets: view('uint32', 4),
        neighbors: view('uint32', 4),
        weights: createTransientView(graph, `cw-${serial++}`, 'float32', 4),
        status: {overflow: view('uint32', 1)}
      }
    })
  ).outputWeights.information?.symmetric;
  expect(cross).toBe('unknown');
  expect(
    () => new GPUNeighborSearch(createProps(graph, {weightTransform: 'nope' as 'binary'}))
  ).toThrow(/weightTransform/);
  expect(
    () =>
      new GPUNeighborSearch(
        createProps(graph, {
          output: {...base.output, status: {...base.output.status, invalidCount: view('uint32', 0)}}
        })
      )
  ).toThrow(/invalidCount/);
});

it('getGPUNeighborSearchWeightTransform derives the declared transform from the parameters', () => {
  expect(getGPUNeighborSearchWeightTransform({})).toBe('binary');
  expect(getGPUNeighborSearchWeightTransform({weightKind: 'inverseDistance'})).toBe('raw');
  expect(getGPUNeighborSearchWeightTransform({weightKind: 'kernel'})).toBe('kernel');
  expect(getGPUNeighborSearchWeightTransform({weightKind: 'kernel', rowStandardize: true})).toBe(
    'row'
  );
  expect(() => getGPUNeighborSearchWeightTransform({weightKind: 'x' as 'binary'})).toThrow(
    /weightKind/
  );
});

it('GPUNeighborSearch derives outputWeights.info.transform from the compile-time rowStandardize', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'neighbor-search-row'});
  const transform = (overrides: Partial<GPUNeighborSearchProps>) =>
    new GPUNeighborSearch(createProps(graph, {mode: 'radius', k: undefined, ...overrides}))
      .outputWeights.information;
  expect(transform({rowStandardize: true})?.transform).toBe('row');
  expect(transform({rowStandardize: true})?.symmetric).toBe('unknown');
  expect(transform({rowStandardize: true, weightTransform: 'row'})?.transform).toBe('row');
  expect(transform({rowStandardize: false})?.transform).toBe('unknown');
  expect(transform({rowStandardize: false, weightTransform: 'binary'})?.transform).toBe('binary');
  expect(transform({rowStandardize: false, weightTransform: 'binary'})?.symmetric).toBe(true);
  // Omitted: the caller's declaration, unchanged.
  expect(transform({})?.transform).toBe('unknown');
  expect(transform({weightTransform: 'row'})?.transform).toBe('row');
  expect(transform({weightTransform: 'kernel'})?.transform).toBe('kernel');
  expect(() => transform({rowStandardize: true, weightTransform: 'binary'})).toThrow(
    /rowStandardize/
  );
  expect(() => transform({rowStandardize: true, weightTransform: 'raw'})).toThrow(/rowStandardize/);
  expect(() => transform({rowStandardize: false, weightTransform: 'row'})).toThrow(
    /rowStandardize/
  );
});

// ---- search structure: grid, BVH and the grid-only 'auto' (algorithms roadmap P2) ----

const STRUCTURES = ['grid', 'bvh', 'auto'] as const;

function getStructureNodes(
  structure: (typeof STRUCTURES)[number],
  overrides: Partial<GPUNeighborSearchProps> = {},
  graphId = `neighbor-search-structure-${serial++}`
) {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: graphId});
  const search = new GPUNeighborSearch({
    ...createProps(graph, {searchStructure: structure, ...overrides}),
    id: 'ns'
  });
  return search.getCommandNodes(graph);
}

it('GPUNeighborSearch validates searchStructure and keeps great-circle search on the grid', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'neighbor-search-structure'});
  for (const searchStructure of [undefined, 'auto', 'grid', 'bvh'] as const) {
    expect(
      () => new GPUNeighborSearch(createProps(graph, {searchStructure})),
      `${searchStructure}`
    ).not.toThrow();
  }
  let searchSerial = 0;
  expect(
    () => new GPUNeighborSearch(createProps(graph, {searchStructure: 'tree' as 'bvh'}))
  ).toThrow(/^neighbor-search searchStructure must be/);
  const greatCircle = {coordinateSpace: 'longitude-latitude', metric: 'great-circle'} as const;
  expect(
    () =>
      new GPUNeighborSearch(
        createProps(graph, {id: 'geo', searchStructure: 'bvh', spatialContext: greatCircle})
      )
  ).toThrow(/^geo searchStructure 'bvh' supports planar positions only/);
  for (const searchStructure of [undefined, 'auto', 'grid'] as const) {
    const search = new GPUNeighborSearch({
      ...createProps(graph, {searchStructure, spatialContext: greatCircle}),
      id: `geo-${searchSerial++}`
    });
    expect(search.searchStructure).toBe('grid');
    const ids = search.getCommandNodes(graph).map(node => node.id);
    expect(ids.some(id => id.includes('bvh') || id.includes('structure'))).toBe(false);
  }
  expect(new GPUNeighborSearch(createProps(graph)).searchStructure).toBe('auto');
});

it('GPUNeighborSearch builds only the selected structure, with auto identical to grid', () => {
  for (const overrides of [
    {mode: 'knn' as const, k: 4},
    {mode: 'radius' as const, k: undefined},
    {mode: 'knn' as const, k: 32}
  ]) {
    const idsOf = (structure: (typeof STRUCTURES)[number]) =>
      getStructureNodes(structure, overrides).map(node => node.id);
    const ids = {grid: idsOf('grid'), bvh: idsOf('bvh'), auto: idsOf('auto')};
    for (const structure of STRUCTURES) {
      expect(new Set(ids[structure]).size, structure).toBe(ids[structure].length);
    }
    const searchIds = overrides.mode === 'knn' ? ['ns-knn'] : ['ns-count', 'ns-emit'];
    const bvhSearchIds =
      overrides.mode === 'knn' ? ['ns-bvh-knn'] : ['ns-bvh-count', 'ns-bvh-emit'];
    for (const id of searchIds) {
      expect(ids.grid).toContain(id);
      expect(ids.auto).toContain(id);
      expect(ids.bvh).not.toContain(id);
    }
    for (const id of bvhSearchIds) {
      expect(ids.bvh).toContain(id);
      expect(ids.auto).not.toContain(id);
      expect(ids.grid).not.toContain(id);
    }
    expect(ids.grid.some(id => id.includes('bvh') || id.includes('structure'))).toBe(false);
    expect(ids.bvh).toContain('ns-bvh-node-data');
    expect(ids.bvh.some(id => id.includes('grid-index') || id.includes('structure'))).toBe(false);
    expect(ids.bvh).not.toContain('ns-sorted-targets');
    expect(ids.auto).toEqual(ids.grid);
    expect(ids.grid).not.toContain('ns-structure-gate');
    expect(ids.bvh).not.toContain('ns-structure-gate');
    // Every structure ends with the shared stages.
    for (const structure of STRUCTURES) {
      expect(ids[structure].at(-1)).toBe('ns-weights');
      expect(ids[structure]).toContain('ns-offsets');
    }
  }
});

it('GPUNeighborSearch auto has the same pipelines as grid with no unused build or gate', () => {
  for (const overrides of [
    {mode: 'knn' as const, k: 4},
    {mode: 'radius' as const, k: undefined}
  ]) {
    const automatic = getStructureNodes('auto', overrides);
    const grid = getStructureNodes('grid', overrides);
    expect(automatic.some(node => node.condition)).toBe(false);
    expect(getGPUCommandNodePipelineKeys(automatic)).toEqual(getGPUCommandNodePipelineKeys(grid));
    expect(getGPUCommandNodeKernelSources(automatic)).toEqual(getGPUCommandNodeKernelSources(grid));
  }
});

it('GPUNeighborSearch generated WGSL parses and stays inside the core limits for every structure and join shape', () => {
  const shapes: Partial<GPUNeighborSearchProps>[] = [];
  for (const mode of ['knn', 'radius'] as const) {
    for (const k of mode === 'knn' ? [1, 4, 32] : [undefined]) {
      for (const cross of [false, true]) {
        for (const masked of [false, true]) {
          shapes.push({mode, k, _cross: cross, _masked: masked} as never);
        }
      }
    }
  }
  for (const structure of STRUCTURES) {
    for (const shape of shapes) {
      const {_cross: cross, _masked: masked, ...rest} = shape as never as Record<string, unknown>;
      const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: `ns-limits-${serial++}`});
      const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
        format: Format,
        length: number
      ) => createTransientView(graph, `limits-${serial++}`, format, length);
      const queryRows = cross ? 7 : 10;
      const props = createProps(graph, {
        ...(rest as Partial<GPUNeighborSearchProps>),
        searchStructure: structure,
        queryPositions: cross ? view('float32x2', queryRows) : undefined,
        mask: masked ? view('uint32', 10) : undefined,
        queryMask: masked && cross ? view('uint32', queryRows) : undefined,
        output: {
          offsets: view('uint32', queryRows + 1),
          neighbors: view('uint32', 64),
          weights: view('float32', 64),
          distances: view('float32', 64),
          neighborCounts: view('uint32', queryRows),
          status: {
            overflow: view('uint32', 1),
            requiredCount: view('uint32', 1),
            invalidCount: view('uint32', 1)
          }
        }
      });
      const nodes = new GPUNeighborSearch({...props, id: 'ns'}).getCommandNodes(graph);
      const label = `${structure} ${JSON.stringify(rest)} cross ${cross} masked ${masked}`;
      const sources = getParsedKernelSources(nodes);
      expect(sources.size, label).toBeGreaterThan(0);
      for (const [kernelId, source] of sources) {
        expect(getStorageBindingCount(source), `${label} ${kernelId}`).toBeLessThanOrEqual(8);
        expect(getPrivateMemoryBytes(source), `${label} ${kernelId}`).toBeLessThanOrEqual(
          CORE_PRIVATE_MEMORY_LIMIT_BYTES
        );
        expect(getWorkgroupMemoryBytes(source), `${label} ${kernelId}`).toBeLessThanOrEqual(
          CORE_WORKGROUP_STORAGE_LIMIT_BYTES
        );
      }
      const kernelIds = [...sources.keys()];
      expect(
        kernelIds.some(id => id.includes('bvh')),
        label
      ).toBe(structure === 'bvh');
      expect(
        kernelIds.some(id => id.includes('structure-gate')),
        label
      ).toBe(false);
      // Sanity: BVH kernels bind the whole tree once (one node table) in at most six buffers.
      for (const [kernelId, source] of sources) {
        if (/bvh-(knn|count|emit)/.test(kernelId)) {
          expect(source, kernelId).toContain('var<storage, read> nodeData');
          expect(getStorageBindingCount(source), kernelId).toBeLessThanOrEqual(7);
        }
      }
    }
  }
});

it('GPUNeighborSearch auto falls back to the grid when the BVH node table exceeds one binding', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'neighbor-search-fallback'});
  const device = graph.device;
  // 10 rows -> 16 leaves -> 31 nodes x 20 bytes: shrink the limit below it.
  Object.defineProperty(device, 'limits', {
    value: {...device.limits, maxStorageBufferBindingSize: 31 * 20 - 4}
  });
  const search = new GPUNeighborSearch({
    ...createProps(graph, {searchStructure: 'auto'}),
    id: 'ns'
  });
  const ids = search.getCommandNodes(graph).map(node => node.id);
  expect(ids.some(id => id.includes('bvh') || id.includes('structure'))).toBe(false);
});

// ---- CPU model of the BVH path against the brute-force oracle ----

function createClusteredPoints(count: number, seed: number): Float32Array {
  const random = createSeededRandom(seed);
  const points = new Float32Array(count * 2);
  for (let row = 0; row < count; row++) {
    const clustered = row % 10 !== 0;
    // 90% of the points in 1% of the area (a 100 x 100 box of a 1000 x 1000 domain).
    points[row * 2] = Math.fround(clustered ? 450 + random() * 100 : random() * 1000);
    points[row * 2 + 1] = Math.fround(clustered ? 450 + random() * 100 : random() * 1000);
  }
  return points;
}

function expectModelMatchesOracle(
  input: Parameters<typeof computeNeighborSearchOracle>[0],
  label: string
) {
  const oracle = computeNeighborSearchOracle(input);
  const model = computeNeighborSearchBVHModel(input);
  expect(model.offsets, `${label} offsets`).toEqual(oracle.offsets);
  expect(model.neighbors, `${label} neighbors`).toEqual(oracle.neighbors);
  expect(model.distancesSquared, `${label} d2`).toEqual(oracle.distancesSquared);
}

it('the BVH kNN model equals the brute-force oracle on clustered, duplicate, collinear and invalid points', () => {
  const clustered = createClusteredPoints(1500, 7);
  // Exact duplicates, a NaN row, an out-of-bounds row and a masked row.
  clustered.set([500, 500, 500, 500, 500, 500], 20);
  clustered.set([Number.NaN, 5], 40);
  clustered.set([-10, 5], 42);
  const mask = new Uint32Array(1500).fill(1);
  for (let row = 3; row < 1500; row += 7) mask[row] = 0;
  const parameters = {bounds: [0, 0, 1000, 1000]} as const;
  for (const k of [1, 4, 32]) {
    expectModelMatchesOracle({mode: 'knn', k, positions: clustered, mask, parameters}, `k ${k}`);
    expectModelMatchesOracle(
      {mode: 'knn', k, positions: clustered, parameters: {...parameters, radius: 6}},
      `k ${k} radius-limited`
    );
  }
  const collinear = new Float32Array(600);
  for (let row = 0; row < 300; row++) collinear.set([Math.floor(row / 3), 100], row * 2);
  for (const k of [1, 4, 32]) {
    expectModelMatchesOracle({mode: 'knn', k, positions: collinear, parameters}, `line k ${k}`);
  }
  // A cross join with its own invalid queries.
  const queries = createClusteredPoints(120, 11);
  queries.set([Number.NaN, Number.NaN], 10);
  expectModelMatchesOracle(
    {mode: 'knn', k: 8, positions: clustered, queryPositions: queries, mask, parameters},
    'cross'
  );
  // Fewer valid targets than k, and no valid target at all.
  expectModelMatchesOracle(
    {mode: 'knn', k: 32, positions: new Float32Array([1, 1, 2, 2, 3, 3]), parameters},
    'few'
  );
  expectModelMatchesOracle(
    {
      mode: 'knn',
      k: 4,
      positions: new Float32Array([1, 1, 2, 2]),
      mask: new Uint32Array([0, 0]),
      parameters
    },
    'none'
  );
});

it('the BVH radius model equals the brute-force oracle, including radius boundaries and duplicates', () => {
  const clustered = createClusteredPoints(1200, 9);
  clustered.set([500, 500, 500, 500], 100);
  const parameters = {bounds: [0, 0, 1000, 1000]} as const;
  for (const radius of [0.5, 5, 40, 2000]) {
    expectModelMatchesOracle(
      {mode: 'radius', positions: clustered, parameters: {...parameters, radius}},
      `radius ${radius}`
    );
  }
  // Integer lattice: many exact d^2 == radius^2 hits.
  const lattice = new Float32Array(2 * 400);
  for (let row = 0; row < 400; row++) lattice.set([row % 20, Math.floor(row / 20)], row * 2);
  expectModelMatchesOracle(
    {mode: 'radius', positions: lattice, parameters: {bounds: [0, 0, 19, 19], radius: 2}},
    'lattice'
  );
  // Invalid radius: nothing is a neighbor.
  for (const radius of [0, -1, Number.NaN]) {
    expectModelMatchesOracle(
      {mode: 'radius', positions: lattice, parameters: {bounds: [0, 0, 19, 19], radius}},
      `radius ${radius}`
    );
  }
});

it('the BVH prunes clustered targets, but the former occupancy heuristic is not a timing guarantee', () => {
  const count = 20_000;
  const clustered = createClusteredPoints(count, 3);
  const uniform = new Float32Array(count * 2);
  const random = createSeededRandom(5);
  for (let index = 0; index < uniform.length; index++)
    uniform[index] = Math.fround(random() * 1000);
  const allValid = Array.from({length: count}, () => true);
  const bounds = [0, 0, 1000, 1000] as const;
  const gridSize = [64, 64] as const;
  const clusteredOccupancy = computeNeighborSearchOccupancy(clustered, allValid, bounds, gridSize);
  const uniformOccupancy = computeNeighborSearchOccupancy(uniform, allValid, bounds, gridSize);
  // 90% of the points in one of 4096 cells' worth of area: about 30x more crowded than uniform.
  expect(clusteredOccupancy).toBeGreaterThan(GPU_NEIGHBOR_SEARCH_BVH_OCCUPANCY_THRESHOLD);
  expect(uniformOccupancy).toBeLessThan(GPU_NEIGHBOR_SEARCH_BVH_OCCUPANCY_THRESHOLD);
  // Work per query: the grid ring search scans the whole crowded cell, the BVH a few dozen leaves.
  const work = {nodeBoundTests: 0, leafTests: 0};
  const queryRows = 200;
  const input = {
    mode: 'knn' as const,
    k: 8,
    positions: clustered,
    queryPositions: clustered.slice(0, queryRows * 2),
    parameters: {bounds}
  };
  computeNeighborSearchBVHModel(input, work);
  expect(work.leafTests / queryRows).toBeLessThan(count / 200);
  expect(work.nodeBoundTests / queryRows).toBeLessThan(count / 50);
});

it('the masked radius fixture needs more than 64 neighbors per query on average', () => {
  const random = createSeededRandom(31);
  const count = 900;
  const positions = new Float32Array(count * 2);
  for (let row = 0; row < count; row++) {
    const clustered = row % 5 !== 0;
    positions[row * 2] = clustered ? 500 + random() * 20 : random() * 1000;
    positions[row * 2 + 1] = clustered ? 500 + random() * 20 : random() * 1000;
  }
  const queryMask = Uint32Array.from({length: count}, (_, row) => (row % 4 === 0 ? 0 : 1));
  const result = computeNeighborSearchOracle({
    mode: 'radius',
    positions,
    queryMask,
    parameters: {bounds: [0, 0, 1000, 1000], radius: 6}
  });
  expect(result.neighbors.length).toBeGreaterThan(count * 64);
  expect(result.neighbors.length).toBeLessThanOrEqual(count * (count - 1));
});

it('distance measurements reject Lambert frames and accept AEQD meters', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const lambert = {
    coordinateSpace: 'planar',
    metric: 'native',
    units: 'meters',
    frame: {id: 'local-frame', kind: 'local-lambert-equal-area'}
  } as const;
  const create = (spatialContext: GPUSpatialContext) =>
    new GPUNeighborSearch(createProps(graph, {spatialContext}));
  expect(() => create(lambert)).toThrow(/measures distance/);
  expect(() =>
    create({...lambert, frame: {...lambert.frame, kind: 'local-azimuthal-equidistant'}})
  ).not.toThrow();
  device.destroy();
});

it('GPUNeighborSearch carries the resolved context on kNN and distance-band weights', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  for (const mode of ['knn', 'radius'] as const) {
    for (const spatialContext of [
      undefined,
      {coordinateSpace: 'longitude-latitude', metric: 'great-circle', units: 'kilometers'} as const
    ]) {
      const search = new GPUNeighborSearch(createProps(graph, {mode, spatialContext}));
      expect(search.outputWeights.spatialContext).toBe(search.spatialContext);
      expect(search.spatialContext).toEqual(
        spatialContext ?? {coordinateSpace: 'planar', metric: 'native', units: 'native'}
      );
    }
  }
  expect(
    () =>
      new GPUNeighborSearch(
        createProps(graph, {spatialContext: {coordinateSpace: 'planar', metric: 'none'}})
      )
  ).toThrow('neighbor-search planar positions need the native metric');
  device.destroy();
});
