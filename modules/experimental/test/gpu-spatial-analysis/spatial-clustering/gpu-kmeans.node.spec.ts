// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  getGPUKMeansParameterValues,
  GPUKMeans,
  GPU_KMEANS_PARAMETER_LENGTH,
  GPUSpatialClustering
} from '../../../src/gpu-spatial-analysis/spatial-clustering/index';
import {getKMeansRestartSeed} from '../../../src/gpu-spatial-analysis/spatial-clustering/gpu-kmeans';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  computeKMeansOracle,
  computeKMeansOracleFromCenters,
  createSeparatedBlobScene,
  relocateEmptyClusters,
  selectBestKMeansRestart,
  SKLEARN_RELOCATION_FIXTURES
} from './kmeans-oracle';

it('GPUKMeans validates props and declares nodes', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'kmeans-nodes'});
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `view-${serial++}`, format, length);
  const make = () => ({
    positions: view('float32x2', 20),
    k: 3,
    maximumIterations: 2,
    parameters: view('uint32', GPU_KMEANS_PARAMETER_LENGTH),
    output: {labels: view('uint32', 20), centers: view('float32x2', 3)}
  });
  const base = make();
  const output = (extra: object) => ({...base, output: {...base.output, ...extra}});
  expect(() => new GPUKMeans({...base, k: 0})).toThrow(/k must/);
  expect(() => new GPUKMeans({...base, k: 300})).toThrow(/k must/);
  expect(() => new GPUKMeans({...base, maximumIterations: 0})).toThrow(/maximumIterations/);
  expect(() => new GPUKMeans({...base, initialization: 'random' as 'kmeans++'})).toThrow(
    /initialization/
  );
  expect(() => new GPUKMeans({...base, parameters: view('uint32', 1)})).toThrow(/parameters/);
  expect(() => getGPUKMeansParameterValues({seed: -1})).toThrow(/seed/);
  expect(() => getGPUKMeansParameterValues({tolerance: -1})).toThrow(/tolerance/);
  expect(() => new GPUKMeans(output({status: {converged: view('uint32', 2)}}))).toThrow(
    /converged/
  );
  expect(() => new GPUKMeans(output({centers: view('float32x2', 4)}))).toThrow(/centers length/);
  expect(() => new GPUKMeans(output({sizes: view('uint32', 2)}))).toThrow(/sizes length/);
  const first = new GPUKMeans({...base, initialization: 'first-valid'})
    .getCommandNodes(graph)
    .map(node => node.id);
  expect(first).toContain('kmeans-prepare');
  expect(first).toContain('kmeans-initial-centers');
  expect(first).toContain('kmeans-iteration-1-update');
  expect(first).toContain('kmeans-iteration-1-converge');
  expect(first.at(-1)).toMatch(/kmeans-final-sizes/);
  const seeded = new GPUKMeans({...make(), id: 'seeded', initialization: 'kmeans++'})
    .getCommandNodes(graph)
    .map(node => node.id);
  expect(seeded).toContain('seeded-seed-2-update');
  expect(seeded).not.toContain('seeded-initial-centers');
});

it('GPUSpatialClustering declares dense-box nodes only when requested', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'dense-box-nodes'});
  const props = (id: string, denseBoxShortcut?: boolean) => ({
    id,
    denseBoxShortcut,
    positions: createTransientView(graph, `${id}-positions`, 'float32x2', 16),
    parameters: createTransientView(graph, `${id}-parameters`, 'float32', 8),
    gridSize: [8, 8] as const,
    output: {labels: createTransientView(graph, `${id}-labels`, 'uint32', 16)}
  });
  const plain = new GPUSpatialClustering(props('plain'))
    .getCommandNodes(graph)
    .map(node => node.id);
  const dense = new GPUSpatialClustering(props('dense', true))
    .getCommandNodes(graph)
    .map(node => node.id);
  expect(plain.some(id => id.includes('box'))).toBe(false);
  expect(dense).toContain('dense-box-keys');
  expect(dense).toContain('dense-box-counts');
  expect(dense.indexOf('dense-box-counts')).toBeLessThan(dense.indexOf('dense-core'));
});

it('computeKMeansOracle sanity: two separated clusters are recovered', () => {
  const positions = new Float32Array([0, 0, 10, 10, 1, 1, 11, 11, 0, 1, 10, 11]);
  const result = computeKMeansOracle(positions, 2, 3);
  expect(result.labels).toEqual([0, 1, 0, 1, 0, 1]);
  expect(result.sizes).toEqual([3, 3]);
});

it("GPUKMeans defaults to 'kmeans++' and one restart, and validates restartCount", () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'kmeans-restart-nodes'});
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `view-${serial++}`, format, length);
  const make = (extra = {}, outputs = {}) => ({
    positions: view('float32x2', 20),
    k: 3,
    maximumIterations: 2,
    parameters: view('uint32', GPU_KMEANS_PARAMETER_LENGTH),
    output: {labels: view('uint32', 20), centers: view('float32x2', 3), ...outputs},
    ...extra
  });
  const defaults = new GPUKMeans(make());
  expect(defaults.initialization).toBe('kmeans++');
  expect(defaults.restartCount).toBe(1);
  expect(() => new GPUKMeans(make({restartCount: 0}))).toThrow(/restartCount/);
  expect(() => new GPUKMeans(make({restartCount: 33}))).toThrow(/restartCount/);
  expect(() => new GPUKMeans(make({restartCount: 1.5}))).toThrow(/restartCount/);
  expect(() => new GPUKMeans(make({restartCount: 2, initialization: 'first-valid'}))).toThrow(
    /kmeans\+\+/
  );
  expect(() => new GPUKMeans(make({}, {inertia: view('float32', 2)}))).toThrow(/inertia/);
  expect(
    () =>
      new GPUKMeans(
        make({spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'}})
      )
  ).toThrow(/project positions to a planar frame/);
  const single = defaults.getCommandNodes(graph).map(node => node.id);
  expect(single).toContain('kmeans-seed-2-update');
  expect(single.some(id => id.includes('restart'))).toBe(false);
  const multi = new GPUKMeans(
    make(
      {id: 'multi', restartCount: 3},
      {
        inertia: view('float32', 1),
        status: {iterationCount: view('uint32', 1), converged: view('uint32', 1)}
      }
    )
  )
    .getCommandNodes(graph)
    .map(node => node.id);
  for (let restart = 0; restart < 3; restart++) {
    expect(multi).toContain(`multi-restart-${restart}-seed-2-update`);
    expect(multi).toContain(`multi-restart-${restart}-iteration-1-converge`);
    expect(multi).toContain(`multi-restart-${restart}-score`);
  }
  expect(multi.indexOf('multi-restart-2-score')).toBeLessThan(
    multi.indexOf('multi-publish-centers')
  );
  expect(multi.indexOf('multi-publish-centers')).toBeLessThan(multi.indexOf('multi-final-assign'));
});

it('getGPUKMeansParameterValues packs the seed and the tolerance bits', () => {
  const values = getGPUKMeansParameterValues({seed: 0xfffffffe, tolerance: 0.25});
  expect(values.length).toBe(GPU_KMEANS_PARAMETER_LENGTH);
  expect(values[0]).toBe(0xfffffffe);
  expect(new Float32Array(values.buffer)[1]).toBe(0.25);
  expect(Array.from(getGPUKMeansParameterValues())).toEqual([0, 0]);
});

it('getKMeansRestartSeed is deterministic, keeps seed for restart 0 and separates restarts', () => {
  expect(getKMeansRestartSeed(7, 0)).toBe(7);
  const seeds = Array.from({length: 32}, (_, restart) => getKMeansRestartSeed(7, restart));
  expect(new Set(seeds).size).toBe(32);
  expect(seeds).toEqual(Array.from({length: 32}, (_, restart) => getKMeansRestartSeed(7, restart)));
  expect(getKMeansRestartSeed(8, 1)).not.toBe(getKMeansRestartSeed(7, 1));
  expect(seeds.every(seed => Number.isInteger(seed) && seed >= 0 && seed <= 0xffffffff)).toBe(true);
});

it('selectBestKMeansRestart keeps the lowest inertia, lowest index on ties, finite over NaN', () => {
  expect(selectBestKMeansRestart([5, 3, 4])).toBe(1);
  expect(selectBestKMeansRestart([3, 3, 2, 2])).toBe(2);
  expect(selectBestKMeansRestart([3, 3, 3])).toBe(0);
  expect(selectBestKMeansRestart([NaN, 4, 2])).toBe(2);
  expect(selectBestKMeansRestart([NaN, NaN])).toBe(0);
});

it('best-of-restarts oracle picks a better optimum than a poor start', () => {
  // Two blobs of three points: seeding both centers in one blob converges to a worse local optimum
  // (blob split) than one center per blob.
  const positions = new Float32Array([0, 0, 1, 0, 0, 1, 100, 0, 101, 0, 100, 1, 50, 50]);
  const poor = computeKMeansOracleFromCenters(positions, [0, 0, 1, 0], 20);
  const good = computeKMeansOracleFromCenters(positions, [0, 0, 100, 0], 20);
  const best = selectBestKMeansRestart([poor.inertia, good.inertia]);
  expect(best).toBe(good.inertia < poor.inertia ? 1 : 0);
  expect(Math.min(poor.inertia, good.inertia)).toBeLessThanOrEqual(
    Math.max(poor.inertia, good.inertia)
  );
});

// Oracle: scikit-learn 1.9.1 KMeans(n_clusters=4, n_init=10, random_state=0), run on the f32
// scene dumped by fix/work/C/U7/dump.ts via fix/work/C/U7/kmeans-sklearn.py. Both 'k-means++'
// and 'random' init reach the same inertia (the blobs are 565 apart), so it is init independent.
const SKLEARN_SEPARATED_INERTIA = 3707.8660708989237;

it('f64 Lloyd oracle reaches the scikit-learn inertia on a well-separated scene', () => {
  const positions = createSeparatedBlobScene(4, 60);
  // One initial center per blob (rows 0, 60, 120, 180): the optimum every sensible init finds.
  const centers = [0, 60, 120, 180].flatMap(row => [positions[2 * row], positions[2 * row + 1]]);
  const result = computeKMeansOracleFromCenters(positions, centers, 20);
  expect(result.sizes).toEqual([60, 60, 60, 60]);
  expect(Math.abs(result.inertia - SKLEARN_SEPARATED_INERTIA)).toBeLessThan(
    1e-9 * SKLEARN_SEPARATED_INERTIA
  );
});

for (const fixture of SKLEARN_RELOCATION_FIXTURES) {
  it(`empty-cluster relocation oracle matches scikit-learn (${fixture.name})`, () => {
    const positions = new Float32Array(fixture.positions);
    const initialCenters = fixture.positions.slice(0, 2 * fixture.k);
    for (const [steps, expected] of Object.entries(fixture.steps)) {
      const result = computeKMeansOracleFromCenters(positions, initialCenters, Number(steps));
      expect(result.centers.length).toBe(expected.centers.length);
      for (let index = 0; index < expected.centers.length; index++) {
        expect(result.centers[index]).toBeCloseTo(expected.centers[index], 12);
      }
      expect(result.labels).toEqual(expected.labels);
      expect(Math.abs(result.inertia - expected.inertia)).toBeLessThan(1e-12 * expected.inertia);
    }
    // 'first-valid' initialization produces the same initial centers and result.
    const firstValid = computeKMeansOracle(positions, fixture.k, 50);
    expect(firstValid.labels).toEqual(fixture.steps['50'].labels);
  });
}

it('relocateEmptyClusters: farthest first, lowest row on ties, donors adjusted, zero distance skips', () => {
  // Cluster 0 holds rows 0-3, cluster 1 holds row 4, clusters 2 and 3 are empty.
  const positions = new Float32Array([0, 0, 2, 0, 0, 2, 2, 2, 10, 10]);
  const labels = [0, 0, 0, 0, 1];
  // Rows 1 and 2 tie for the largest distance 4: the lowest row (1) is first.
  const squaredDistances = [1, 4, 4, 2, 0];
  const sums = [4, 4, 10, 10, 0, 0, 0, 0];
  const counts = [4, 1, 0, 0];
  const centers = [1, 1, 10, 10, 5, 5, 6, 6];
  relocateEmptyClusters(positions, centers, labels, squaredDistances, sums, counts);
  expect(counts).toEqual([2, 1, 1, 1]);
  expect(sums).toEqual([4 - 2 - 0, 4 - 0 - 2, 10, 10, 2, 0, 0, 2]);

  // Largest distance 0 (every point on its center): nothing moves.
  const untouchedSums = [4, 4, 10, 10, 0, 0, 0, 0];
  const untouchedCounts = [4, 1, 0, 0];
  relocateEmptyClusters(
    positions,
    centers,
    labels,
    [0, 0, 0, 0, 0],
    untouchedSums,
    untouchedCounts
  );
  expect(untouchedCounts).toEqual([4, 1, 0, 0]);

  // A NaN (never initialized) center is not an empty cluster to relocate.
  const nanCounts = [4, 1, 0, 0];
  const nanSums = [4, 4, 10, 10, 0, 0, 0, 0];
  relocateEmptyClusters(
    positions,
    [1, 1, 10, 10, NaN, NaN, 6, 6],
    labels,
    squaredDistances,
    nanSums,
    nanCounts
  );
  expect(nanCounts).toEqual([3, 1, 0, 1]);
});

it('GPUKMeans relocate kernel binds at most 8 storage buffers and runs once per iteration', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'kmeans-relocate'});
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `view-${format}-${length}`, format, length);
  const nodes = new GPUKMeans({
    positions: view('float32x2', 20),
    k: 3,
    maximumIterations: 4,
    initialization: 'first-valid',
    parameters: view('uint32', GPU_KMEANS_PARAMETER_LENGTH),
    output: {labels: view('uint32', 20), centers: view('float32x2', 3)}
  })
    .getCommandNodes(graph)
    .map(node => node.id);
  for (let iteration = 0; iteration < 4; iteration++) {
    const prefix = `kmeans-iteration-${iteration}`;
    expect(nodes.indexOf(`${prefix}-update`)).toBeLessThan(nodes.indexOf(`${prefix}-relocate`));
    expect(nodes.indexOf(`${prefix}-relocate`)).toBeLessThan(nodes.indexOf(`${prefix}-converge`));
  }
});
