// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {describe, expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {GPUNearestFeatureJoin} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-nearest-feature-join';
import {GPUSpatialPredicateJoin} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-spatial-predicate-join';
import type {GPUNearestFeatureGeometry} from '../../../src/gpu-spatial-analysis/spatial-join/nearest-types';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {buildNearestArrays, type NearestOracleGeometry} from './nearest-feature-oracle';
import {
  NEAREST_OPTIONS_FIXTURES,
  ON_ATTRIBUTE_FIXTURES,
  type NearestOptionsFixture
} from './spatial-join-options-fixtures';

/** Imports a fixture geometry set as graph views. */
function importGeometry(
  device: Device,
  graph: GPUCommandGraph,
  kind: 'points' | 'lines' | 'polygons',
  features: unknown,
  name: string,
  buffers: Buffer[]
): GPUNearestFeatureGeometry & {kind: 'points' | 'lines' | 'polygons'} {
  const arrays = buildNearestArrays({kind, features} as NearestOracleGeometry);
  const make = (
    suffix: string,
    values: Float32Array | Uint32Array,
    format: string,
    rows: number
  ) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return importGraphBuffer(graph, `${name}-${suffix}`, buffer, format as 'uint32', rows);
  };
  const positions = make('positions', arrays.positions, 'float32x2', arrays.positions.length / 2);
  const offsets = (suffix: string, values: Uint32Array) =>
    make(suffix, values, 'uint32', values.length) as GraphDataView<'uint32'>;
  if (kind === 'points') {
    return {kind, positions: positions as GraphDataView<'float32x2'>};
  }
  if (kind === 'lines') {
    return {
      kind,
      positions: positions as GraphDataView<'float32x2'>,
      lineOffsets: offsets('line-offsets', arrays.lineOffsets!)
    };
  }
  return {
    kind,
    positions: positions as GraphDataView<'float32x2'>,
    featureOffsets: offsets('feature-offsets', arrays.featureOffsets!),
    polygonOffsets: offsets('polygon-offsets', arrays.polygonOffsets!),
    ringOffsets: offsets('ring-offsets', arrays.ringOffsets!)
  };
}

type NearestRun = {
  ids: number[];
  distances: number[];
  queryPoints: number[];
  footPoints: number[];
  counts: number[];
  overflow: number;
};

type NearestOptions = {
  exclusive?: boolean;
  keys?: boolean;
  /** Explicit permuted IDs for `featureIds` and `queryIds`. */
  explicitIds?: number[];
  queryPoints?: boolean;
};

async function runNearest(
  device: Device,
  fixture: NearestOptionsFixture,
  options: NearestOptions = {}
): Promise<NearestRun> {
  const graph = new GPUCommandGraph(device, {id: 'nearest-options'});
  const buffers: Buffer[] = [];
  const features = importGeometry(
    device,
    graph,
    fixture.featureKind,
    fixture.features,
    'features',
    buffers
  );
  const queries = importGeometry(
    device,
    graph,
    fixture.queryKind,
    fixture.queries,
    'queries',
    buffers
  );
  const queryCount = fixture.ids.length;
  const featureCount =
    features.kind === 'points' ? features.positions.length : fixture.features.length;
  const uint32 = (name: string, values: number[]) => {
    const buffer = createInputBuffer(device, Uint32Array.from(values));
    buffers.push(buffer);
    return importGraphBuffer(graph, name, buffer, 'uint32', values.length);
  };
  const output = (name: string, length: number, format: 'uint32' | 'float32' | 'float32x2') => {
    const buffer = createOutputBuffer(device, length * (format === 'float32x2' ? 2 : 1));
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, name, buffer, format, length)};
  };
  const ids = output('ids', queryCount, 'uint32');
  const distances = output('distances', queryCount, 'float32');
  const foot = output('foot', queryCount, 'float32x2');
  const queryPoints = output('query-points', queryCount, 'float32x2');
  const counts = output('counts', queryCount, 'uint32');
  const overflow = output('overflow', 1, 'uint32');
  const explicit = options.explicitIds;
  graph.add(
    new GPUNearestFeatureJoin({
      queries,
      features,
      exclusive: options.exclusive,
      featureIds: explicit ? uint32('feature-ids', explicit) : undefined,
      queryIds: explicit ? uint32('query-ids', explicit) : undefined,
      onAttribute:
        options.keys && fixture.queryKeys && fixture.featureKeys
          ? {
              left: uint32('query-keys', fixture.queryKeys),
              right: uint32('feature-keys', fixture.featureKeys)
            }
          : undefined,
      neighborIds: ids.view,
      neighborCounts: counts.view,
      neighborDistances: distances.view,
      neighborFootPoints: foot.view,
      neighborQueryPoints: options.queryPoints === false ? undefined : queryPoints.view,
      overflow: overflow.view
    })
  );
  expect(featureCount).toBeGreaterThan(0);
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const result: NearestRun = {
    ids: await readUint32(ids.buffer, queryCount),
    distances: await readFloat32(distances.buffer, queryCount),
    queryPoints: await readFloat32(queryPoints.buffer, queryCount * 2),
    footPoints: await readFloat32(foot.buffer, queryCount * 2),
    counts: await readUint32(counts.buffer, queryCount),
    overflow: (await readUint32(overflow.buffer, 1))[0]
  };
  compiled.destroy();
  for (const buffer of buffers) buffer.destroy();
  return result;
}

function expectMatchesFixture(run: NearestRun, fixture: NearestOptionsFixture, label: string) {
  expect(run.overflow).toBe(0);
  for (const [query, id] of fixture.ids.entries()) {
    expect(run.ids[query], `${label} id of query ${query}`).toBe(id);
    expect(run.distances[query]).toBeCloseTo(fixture.distances[query], 3);
    // `shapely.shortest_line(query, feature)` is (query point, feature point).
    expect(run.queryPoints[query * 2], `${label} query point x ${query}`).toBeCloseTo(
      fixture.queryPoints[query][0],
      3
    );
    expect(run.queryPoints[query * 2 + 1]).toBeCloseTo(fixture.queryPoints[query][1], 3);
    expect(run.footPoints[query * 2], `${label} foot x ${query}`).toBeCloseTo(
      fixture.footPoints[query][0],
      3
    );
    expect(run.footPoints[query * 2 + 1]).toBeCloseTo(fixture.footPoints[query][1], 3);
  }
}

describe('GPUNearestFeatureJoin options', () => {
  it('neighborQueryPoints equals shapely.shortest_line for every kind pair', async () => {
    const device = await getWebGPUTestDevice('core');
    if (!device) return;
    for (const name of [
      'pointsToLines',
      'linesToPolygons',
      'polygonsToPolygons',
      'polygonsToPoints',
      'linesToLines',
      'pointsToPolygons'
    ]) {
      const fixture = NEAREST_OPTIONS_FIXTURES[name];
      expectMatchesFixture(await runNearest(device, fixture), fixture, name);
    }
  });

  it('exclusive equals STRtree.query_nearest(exclusive=True) on self-joins', async () => {
    const device = await getWebGPUTestDevice('core');
    if (!device) return;
    for (const name of ['pointsSelfExclusive', 'linesSelfExclusive', 'polygonsSelfExclusive']) {
      const fixture = NEAREST_OPTIONS_FIXTURES[name];
      expect(fixture.exclusive).toBe(true);
      expectMatchesFixture(await runNearest(device, fixture, {exclusive: true}), fixture, name);
      // Without it every query finds itself first at distance 0.
      const inclusive = await runNearest(device, fixture);
      expect(inclusive.ids).toEqual(fixture.ids.map((_, row) => row));
      expect(Math.max(...inclusive.distances)).toBe(0);
    }
  });

  it('exclusive compares explicit queryIds and featureIds, not rows', async () => {
    const device = await getWebGPUTestDevice('core');
    if (!device) return;
    const fixture = NEAREST_OPTIONS_FIXTURES.pointsSelfExclusive;
    const count = fixture.ids.length;
    const explicit = Array.from({length: count}, (_, row) => ((row * 5 + 3) % count) + 100);
    const run = await runNearest(device, fixture, {exclusive: true, explicitIds: explicit});
    // Results are feature IDs: map the fixture's feature rows through the same table.
    expect(run.ids).toEqual(fixture.ids.map(row => explicit[row]));
  });

  it('onAttribute keeps only key-equal features, with and without exclusive', async () => {
    const device = await getWebGPUTestDevice('core');
    if (!device) return;
    for (const [name, exclusive] of [
      ['linesToPolygonsKeys', false],
      ['pointsSelfExclusiveKeys', true]
    ] as const) {
      const fixture = NEAREST_OPTIONS_FIXTURES[name];
      const run = await runNearest(device, fixture, {exclusive, keys: true});
      expectMatchesFixture(run, fixture, name);
      for (const [query, id] of run.ids.entries()) {
        expect(fixture.featureKeys![id]).toBe(fixture.queryKeys![query]);
      }
    }
  });

  it('defaults are unchanged when the new outputs are absent', async () => {
    const device = await getWebGPUTestDevice('core');
    if (!device) return;
    const fixture = NEAREST_OPTIONS_FIXTURES.polygonsToPolygons;
    const withQuery = await runNearest(device, fixture);
    const without = await runNearest(device, fixture, {queryPoints: false});
    expect(without.ids).toEqual(withQuery.ids);
    expect(without.distances).toEqual(withQuery.distances);
    expect(without.footPoints).toEqual(withQuery.footPoints);
  });

  it('nearest-feature mode honors exclusive and onAttribute', async () => {
    const device = await getWebGPUTestDevice('core');
    if (!device) return;
    const fixture = NEAREST_OPTIONS_FIXTURES.pointsSelfExclusiveKeys;
    const count = fixture.ids.length;
    for (const withKeys of [false, true]) {
      const reference =
        NEAREST_OPTIONS_FIXTURES[withKeys ? 'pointsSelfExclusiveKeys' : 'pointsSelfExclusive'];
      const graph = new GPUCommandGraph(device, {id: 'nearest-feature-options'});
      const buffers: Buffer[] = [];
      const points = importGeometry(device, graph, 'points', reference.queries, 'points', buffers);
      const make = (
        name: string,
        values: number[] | Float32Array,
        format: 'uint32' | 'float32'
      ) => {
        const buffer = createInputBuffer(
          device,
          format === 'uint32' ? Uint32Array.from(values) : Float32Array.from(values)
        );
        buffers.push(buffer);
        return importGraphBuffer(graph, name, buffer, format, values.length);
      };
      const nearest = createOutputBuffer(device, count);
      const overflow = createOutputBuffer(device, 1);
      buffers.push(nearest, overflow);
      graph.add(
        new GPUNearestFeatureJoin({
          points: points.positions,
          features: {kind: 'points', positions: points.positions},
          radius: make('radius', [1000], 'float32') as GraphDataView<'float32'>,
          candidateCapacity: count * count,
          exclusive: true,
          onAttribute: withKeys
            ? {
                left: make('left', reference.queryKeys!, 'uint32'),
                right: make('right', reference.featureKeys!, 'uint32')
              }
            : undefined,
          nearestFeatureIds: importGraphBuffer(graph, 'nearest', nearest, 'uint32', count),
          overflow: importGraphBuffer(graph, 'overflow', overflow, 'uint32', 1)
        })
      );
      const compiled = graph.compile();
      submitGraph(device, compiled, undefined);
      expect(await readUint32(nearest, count)).toEqual(reference.ids);
      expect((await readUint32(overflow, 1))[0]).toBe(0);
      compiled.destroy();
      for (const buffer of buffers) buffer.destroy();
    }
  });

  it('rejects queryIds without exclusive and misaligned onAttribute keys', async () => {
    const device = await getWebGPUTestDevice('core');
    if (!device) return;
    const fixture = NEAREST_OPTIONS_FIXTURES.pointsToLines;
    const graph = new GPUCommandGraph(device, {id: 'nearest-options-validation'});
    const buffers: Buffer[] = [];
    const features = importGeometry(device, graph, 'lines', fixture.features, 'features', buffers);
    const queries = importGeometry(device, graph, 'points', fixture.queries, 'queries', buffers);
    const buffer = createOutputBuffer(device, 64);
    buffers.push(buffer);
    const view = (name: string, length: number) =>
      importGraphBuffer(graph, name, buffer, 'uint32', length);
    const queryCount = fixture.ids.length;
    const base = {
      queries,
      features,
      neighborIds: view('ids', queryCount),
      neighborCounts: view('counts', queryCount),
      overflow: view('overflow', 1)
    };
    expect(
      () => new GPUNearestFeatureJoin({...base, queryIds: view('query-ids', queryCount)})
    ).toThrow(/queryIds requires exclusive/);
    expect(
      () =>
        new GPUNearestFeatureJoin({
          ...base,
          onAttribute: {left: view('left', queryCount), right: view('right', 1)}
        })
    ).toThrow(/onAttribute keys/);
    for (const item of buffers) item.destroy();
  });
});

describe('GPUSpatialPredicateJoin onAttribute', () => {
  async function runPairs(
    device: Device,
    name: string,
    options: {anti?: boolean; engine?: 'fast' | 'relate'; keys?: boolean}
  ) {
    const fixture = ON_ATTRIBUTE_FIXTURES[name];
    const graph = new GPUCommandGraph(device, {id: 'on-attribute'});
    const buffers: Buffer[] = [];
    const left = importGeometry(device, graph, fixture.leftKind, fixture.left, 'left', buffers);
    const right = importGeometry(device, graph, fixture.rightKind, fixture.right, 'right', buffers);
    const uint32 = (id: string, values: number[]) => {
      const buffer = createInputBuffer(device, Uint32Array.from(values));
      buffers.push(buffer);
      return importGraphBuffer(graph, id, buffer, 'uint32', values.length);
    };
    const output = (id: string, length: number) => {
      const buffer = createOutputBuffer(device, length);
      buffers.push(buffer);
      return {buffer, view: importGraphBuffer(graph, id, buffer, 'uint32', length)};
    };
    const capacity = 100;
    const leftIds = output('left-ids', capacity);
    const rightIds = output('right-ids', capacity);
    const count = output('count', 1);
    const overflow = output('overflow', 1);
    const unmatchedIds = output('unmatched-ids', 10);
    const unmatchedCount = output('unmatched-count', 1);
    const unmatchedOverflow = output('unmatched-overflow', 1);
    const candidateCount = output('candidate-count', 1);
    graph.add(
      new GPUSpatialPredicateJoin({
        left,
        right,
        predicate: fixture.predicate,
        engine: options.engine,
        how: options.anti ? 'anti' : 'inner',
        candidateCapacity: capacity,
        onAttribute:
          options.keys === false
            ? undefined
            : {
                left: uint32('left-keys', fixture.leftKeys),
                right: uint32('right-keys', fixture.rightKeys)
              },
        pairs: options.anti
          ? undefined
          : {
              leftIds: leftIds.view,
              rightIds: rightIds.view,
              count: count.view,
              overflow: overflow.view
            },
        unmatched: options.anti
          ? {
              ids: unmatchedIds.view,
              count: unmatchedCount.view,
              overflow: unmatchedOverflow.view
            }
          : undefined,
        overflow: options.anti ? overflow.view : undefined,
        candidateCount: candidateCount.view
      })
    );
    const compiled = graph.compile();
    submitGraph(device, compiled, undefined);
    const total = (await readUint32(count.buffer, 1))[0];
    const lefts = await readUint32(leftIds.buffer, capacity);
    const rights = await readUint32(rightIds.buffer, capacity);
    const unmatched = (await readUint32(unmatchedIds.buffer, 10)).slice(
      0,
      (await readUint32(unmatchedCount.buffer, 1))[0]
    );
    const candidates = (await readUint32(candidateCount.buffer, 1))[0];
    compiled.destroy();
    for (const buffer of buffers) buffer.destroy();
    return {
      fixture,
      pairs: lefts.slice(0, total).map((row, slot) => [row, rights[slot]]),
      unmatched,
      candidates
    };
  }

  it('matches geopandas sjoin(on_attribute=) for each kind pair', async () => {
    const device = await getWebGPUTestDevice('core');
    if (!device) return;
    for (const [name, engine] of [
      ['pointsPolygons', undefined],
      ['polygonsPolygons', 'fast'],
      ['polygonsPolygons', 'relate'],
      ['linesPolygons', undefined]
    ] as const) {
      const run = await runPairs(device, name, {engine});
      expect(run.pairs, `${name} ${engine}`).toEqual(run.fixture.pairs);
      const plain = await runPairs(device, name, {engine, keys: false});
      // Without keys the join finds every spatial pair, which geopandas also reports.
      expect(plain.pairs.length).toBe(run.fixture.unfilteredPairCount);
    }
  });

  it('anti join treats key-mismatched matches as unmatched', async () => {
    const device = await getWebGPUTestDevice('core');
    if (!device) return;
    const run = await runPairs(device, 'polygonsPolygons', {anti: true});
    expect(run.unmatched).toEqual(run.fixture.unmatched);
  });
});
