// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUZonalStatistics,
  type GPUZonalStatisticsOutput,
  type GPUZonalStatisticsProps
} from '../../../src/gpu-spatial-analysis/zonal-statistics';
import {getPointInPolygonContainmentContext} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-point-in-polygon-join';
import {getZonalStatisticsPlan} from '../../../src/gpu-spatial-analysis/zonal-statistics/zonal-statistics-passes';
import {getSortKeyBits} from '../../../src/utils/sorted-segment-sums';
import {createNullWebGPUDevice, createVectorView} from '../../utils/gpu-contributor-test-utils';
import {POLYGON_FEATURES} from '../spatial-join/spatial-join-oracle';
import {
  computeExtent,
  computePolygonFeatureArea,
  computeZonalStatistics,
  NO_FEATURE
} from './zonal-statistics-oracle';

const POINT_COUNT = 10;
const FEATURE_COUNT = 4;

let viewCounter = 0;

function view<Format extends 'uint32' | 'float32' | 'float32x2'>(
  graph: GPUCommandGraph,
  name: string,
  format: Format,
  length: number
) {
  return createTransientView(graph, `${name}-${viewCounter++}`, format, length);
}

function createProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUZonalStatisticsProps> = {},
  output: GPUZonalStatisticsOutput = {
    counts: view(graph, 'out-counts', 'uint32', FEATURE_COUNT),
    sums: view(graph, 'out-sums', 'float32', FEATURE_COUNT)
  }
): GPUZonalStatisticsProps {
  return {
    id: 'z',
    features: {
      kind: 'polygons',
      polygonPositions: view(graph, 'polygon-positions', 'float32x2', 20),
      featureOffsets: view(graph, 'feature-offsets', 'uint32', FEATURE_COUNT + 1),
      polygonOffsets: view(graph, 'polygon-offsets', 'uint32', 5),
      ringOffsets: view(graph, 'ring-offsets', 'uint32', 6),
      candidateCapacity: 16
    },
    points: view(graph, 'points', 'float32x2', POINT_COUNT),
    values: view(graph, 'values', 'float32', POINT_COUNT),
    output,
    ...overrides
  };
}

function createRowsProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUZonalStatisticsProps> = {},
  output: GPUZonalStatisticsOutput = {counts: view(graph, 'out-counts', 'uint32', FEATURE_COUNT)}
): GPUZonalStatisticsProps {
  return {
    id: 'z',
    features: {
      kind: 'feature-rows',
      pointFeatureRows: view(graph, 'rows', 'uint32', POINT_COUNT),
      featureCount: FEATURE_COUNT
    },
    values: view(graph, 'values', 'float32', POINT_COUNT),
    output,
    ...overrides
  };
}

function getNodeIds(props: GPUZonalStatisticsProps, graph: GPUCommandGraph): string[] {
  return new GPUZonalStatistics(props).getCommandNodes(graph).map(node => node.id);
}

it('GPUZonalStatistics validates props', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const out = <Format extends 'uint32' | 'float32' = 'float32'>(
    name: string,
    format: Format = 'float32' as Format,
    length = FEATURE_COUNT
  ) => view(graph, name, format, length);
  const create = (props: GPUZonalStatisticsProps) => new GPUZonalStatistics(props);

  expect(() => create(createProps(graph, {}, {}))).toThrow(/at least one output/);
  expect(() => create(createProps(graph, {values: undefined}))).toThrow(/require values/);
  expect(() => create(createProps(graph, {}, {weightSums: out('weight-sums')}))).toThrow(
    /weightSums requires weights/
  );
  expect(() =>
    create(createProps(graph, {weights: out('w', 'float32', POINT_COUNT), values: undefined}))
  ).toThrow(/weights require values/);
  expect(() =>
    create(createProps(graph, {values: out('short-values', 'float32', POINT_COUNT - 1)}))
  ).toThrow(/values length/);
  expect(() =>
    create(createProps(graph, {weights: out('short-weights', 'float32', POINT_COUNT - 1)}))
  ).toThrow(/weights length/);
  expect(() => create(createProps(graph, {points: undefined}))).toThrow(/points are required/);
  expect(() => create(createProps(graph, {}, {extent: out('extent', 'float32', 2)}))).toThrow(
    /extentStatistic/
  );
  expect(() =>
    create(createProps(graph, {}, {extent: out('extent2', 'float32', 3), extentStatistic: 'count'}))
  ).toThrow(/two float32 rows/);
  expect(() =>
    create(createProps(graph, {}, {counts: out('short-counts', 'uint32', FEATURE_COUNT - 1)}))
  ).toThrow(/feature count/);
  expect(() =>
    create(createProps(graph, {}, {uncertainCount: out('empty-uncertain', 'uint32', 0)}))
  ).toThrow(/uncertainCount must contain one uint32 row/);
  expect(() =>
    create(createProps(graph, {}, {uncertainCount: out('float-uncertain', 'float32', 1) as never}))
  ).toThrow(/uncertainCount/);
  expect(() =>
    create(createProps(graph, {}, {uncertainCount: out('uncertain', 'uint32', 1)}))
  ).not.toThrow();
  expect(() => create(createProps(graph, {sumOrder: 'nearest' as never}))).toThrow(/sumOrder/);
  expect(() =>
    create(createProps(graph, {areas: out('areas')}, {featureAreas: out('feature-areas')}))
  ).toThrow(/featureAreas/);
  expect(() =>
    create(createProps(graph, {}, {pointFeatureRows: out('short-rows', 'uint32', POINT_COUNT - 1)}))
  ).toThrow(/topology/);

  // Feature rows.
  expect(() => create(createRowsProps(graph, {}, {densities: out('densities')}))).toThrow(
    /require areas/
  );
  expect(() =>
    create(createRowsProps(graph, {}, {pointFeatureRows: out('rows-out', 'uint32', POINT_COUNT)}))
  ).toThrow(/polygon features/);
  expect(() => create(createRowsProps(graph, {}, {featureAreas: out('areas-out')}))).toThrow(
    /polygon features/
  );
  expect(() =>
    create(
      createRowsProps(graph, {
        features: {
          kind: 'feature-rows',
          pointFeatureRows: view(graph, 'rows2', 'uint32', POINT_COUNT),
          featureCount: 0
        }
      })
    )
  ).toThrow(/at least one feature/);
  expect(() =>
    create(
      createRowsProps(graph, {
        values: out('values-b', 'float32', POINT_COUNT + 1)
      })
    )
  ).toThrow(/values length/);

  // Sorted order needs packed inputs.
  const chunkedValues = createVectorView('chunked-values', 'float32', [
    view(graph, 'v0', 'float32', 6),
    view(graph, 'v1', 'float32', 4)
  ]);
  expect(() => create(createRowsProps(graph, {values: chunkedValues, sumOrder: 'sorted'}))).toThrow(
    /packed values/
  );
  expect(() => create(createRowsProps(graph, {values: chunkedValues}))).not.toThrow();
  const chunkedRows = createVectorView('chunked-rows', 'uint32', [
    view(graph, 'r0', 'uint32', 6),
    view(graph, 'r1', 'uint32', 4)
  ]);
  expect(() =>
    create(
      createRowsProps(graph, {
        features: {
          kind: 'feature-rows',
          pointFeatureRows: chunkedRows,
          featureCount: FEATURE_COUNT
        },
        sumOrder: 'sorted'
      })
    )
  ).toThrow(/packed pointFeatureRows/);
  const chunkedWeights = createVectorView('chunked-weights', 'float32', [
    view(graph, 'w0', 'float32', 5),
    view(graph, 'w1', 'float32', 5)
  ]);
  expect(() =>
    create(createRowsProps(graph, {values: chunkedValues, weights: chunkedWeights}))
  ).toThrow(/topology/);

  // Outputs must not overlap inputs or each other.
  const shared = graph.createTransientBuffer({id: 'shared', byteLength: 256, usage: 128});
  expect(() =>
    create(
      createRowsProps(
        graph,
        {values: graph.createDataView(shared, {format: 'float32', length: POINT_COUNT})},
        {sums: graph.createDataView(shared, {format: 'float32', length: FEATURE_COUNT})}
      )
    )
  ).toThrow(/overlap/);
  device.destroy();
});

it('GPUZonalStatistics rejects views from another graph', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const other = new GPUCommandGraph(device);
  const props = createRowsProps(
    graph,
    {},
    {counts: view(other, 'foreign-counts', 'uint32', FEATURE_COUNT)}
  );
  expect(() => new GPUZonalStatistics(props).getCommandNodes(graph)).toThrow(/target graph/);
  device.destroy();
});

it('GPUZonalStatistics schedules the atomic nodes in order', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const f = (name: string) => view(graph, name, 'float32', FEATURE_COUNT);
  const ids = getNodeIds(
    createRowsProps(
      graph,
      {
        weights: view(graph, 'weights', 'float32', POINT_COUNT),
        areas: f('areas'),
        sumOrder: 'atomic'
      },
      {
        counts: view(graph, 'c', 'uint32', FEATURE_COUNT),
        valueCounts: view(graph, 'vc', 'uint32', FEATURE_COUNT),
        sums: f('s'),
        weightSums: f('ws'),
        means: f('m'),
        minima: f('mn'),
        maxima: f('mx'),
        densities: f('d'),
        overflow: view(graph, 'o', 'uint32', 1),
        extent: view(graph, 'e', 'float32', 2),
        extentStatistic: 'density'
      }
    ),
    graph
  );
  const order = [
    'z-count-aggregation-clear',
    'z-overflow',
    'z-prepare',
    'z-value-count-aggregation-clear',
    'z-sum-aggregation-clear',
    'z-weight-sum-aggregation-clear',
    'z-minimum-aggregation-initialize',
    'z-maximum-aggregation-initialize',
    'z-means',
    'z-densities',
    'z-extent-prepare'
  ];
  const positions = order.map(id => ids.indexOf(id));
  expect(
    positions.every(position => position >= 0),
    ids.join(',')
  ).toBe(true);
  expect([...positions].sort((a, b) => a - b)).toEqual(positions);
  expect(ids.some(id => id.startsWith('z-sort'))).toBe(false);
  device.destroy();
});

it('GPUZonalStatistics schedules sort, scan, and segmented sums only for sorted order', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const ids = getNodeIds(
    createRowsProps(
      graph,
      {sumOrder: 'sorted'},
      {sums: view(graph, 's', 'float32', FEATURE_COUNT)}
    ),
    graph
  );
  for (const id of [
    'z-prepare',
    'z-sort-prepare',
    'z-segment-offsets',
    'z-gather-sums',
    'z-reduce-sums'
  ]) {
    expect(ids, id).toContain(id);
  }
  expect(ids.some(id => id.startsWith('z-sort-') && id !== 'z-sort-prepare')).toBe(true);
  expect(ids.some(id => id.includes('aggregation') && id.includes('sum'))).toBe(false);

  // Minima alone never need the sort.
  const minimaIds = getNodeIds(
    createRowsProps(
      graph,
      {sumOrder: 'sorted', id: 'minima-only'},
      {minima: view(graph, 'mn', 'float32', FEATURE_COUNT)}
    ),
    graph
  );
  expect(minimaIds.some(id => id.includes('-sort') || id.includes('-segment'))).toBe(false);
  device.destroy();
});

it('GPUZonalStatistics gives chunked inputs per-chunk prepare nodes and unique IDs', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const values = createVectorView('chunked-values', 'float32', [
    view(graph, 'v0', 'float32', 6),
    view(graph, 'v1', 'float32', 4)
  ]);
  const ids = getNodeIds(
    createRowsProps(graph, {values}, {sums: view(graph, 's', 'float32', FEATURE_COUNT)}),
    graph
  );
  expect(ids).toContain('z-prepare-chunk-0');
  expect(ids).toContain('z-prepare-chunk-1');
  expect(new Set(ids).size).toBe(ids.length);

  const polygonIds = getNodeIds(createProps(graph, {id: 'p'}), graph);
  expect(polygonIds.filter(id => ids.includes(id))).toEqual([]);
  expect(polygonIds).toContain('p-join-bounds');
  device.destroy();
});

it('getZonalStatisticsPlan derives transient statistics', () => {
  const none = {
    counts: false,
    valueCounts: false,
    sums: false,
    weightSums: false,
    means: false,
    minima: false,
    maxima: false,
    densities: false,
    featureAreas: false,
    extent: false
  };
  const meanPlan = getZonalStatisticsPlan({
    outputs: {...none, means: true},
    hasWeights: false,
    hasAreas: false,
    sorted: false
  });
  expect(meanPlan).toMatchObject({needSums: true, needValueCounts: true, needWeightSums: false});
  const weightedMeanPlan = getZonalStatisticsPlan({
    outputs: {...none, means: true},
    hasWeights: true,
    hasAreas: false,
    sorted: false
  });
  expect(weightedMeanPlan).toMatchObject({
    needSums: true,
    needValueCounts: false,
    needWeightSums: true
  });
  const densityPlan = getZonalStatisticsPlan({
    outputs: {...none, extent: true},
    extentStatistic: 'density',
    hasWeights: false,
    hasAreas: false,
    sorted: false
  });
  expect(densityPlan).toMatchObject({
    needDensities: true,
    needCounts: true,
    needGPUAreas: true,
    needValueStatistics: false
  });
  expect(
    getZonalStatisticsPlan({
      outputs: {...none, extent: true},
      extentStatistic: 'density',
      hasWeights: false,
      hasAreas: true,
      sorted: false
    }).needGPUAreas
  ).toBe(false);
  expect(
    getZonalStatisticsPlan({
      outputs: {...none, sums: true},
      hasWeights: false,
      hasAreas: false,
      sorted: true
    }).needCounts
  ).toBe(false);
  expect(
    getZonalStatisticsPlan({
      outputs: {...none, extent: true},
      extentStatistic: 'minimum',
      hasWeights: false,
      hasAreas: false,
      sorted: false
    })
  ).toMatchObject({needMinima: true, needMaxima: false, needSums: false});
});

it('getSortKeyBits counts the bits of the largest key', () => {
  expect(getSortKeyBits(1)).toBe(1);
  expect(getSortKeyBits(2)).toBe(2);
  expect(getSortKeyBits(3)).toBe(2);
  expect(getSortKeyBits(4)).toBe(3);
  expect(getSortKeyBits(600)).toBe(10);
});

it('zonal statistics oracle follows the documented semantics', () => {
  const result = computeZonalStatistics({
    featureRows: [0, 0, 0, 1, 1, 2, NO_FEATURE, 7],
    featureCount: 3,
    values: [1, 3, Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY, 5, 100, 100],
    areas: [2, 0, Number.NaN]
  });
  expect(result.counts).toEqual([3, 2, 1]);
  expect(result.valueCounts).toEqual([2, 0, 1]);
  expect(result.sums).toEqual([4, 0, 5]);
  expect(result.means[0]).toBe(2);
  expect(result.means[1]).toBeNaN();
  expect(result.minima[0]).toBe(1);
  expect(result.maxima[0]).toBe(3);
  expect(result.minima[1]).toBeNaN();
  expect(result.densities[0]).toBe(1.5);
  expect(result.densities[1]).toBeNaN();
  expect(result.densities[2]).toBeNaN();

  const weighted = computeZonalStatistics({
    featureRows: [0, 0, 0, 1],
    featureCount: 3,
    values: [2, 4, 100, 1],
    weights: [1, 3, Number.NaN, 0]
  });
  expect(weighted.sums).toEqual([14, 0, 0]);
  expect(weighted.weightSums).toEqual([4, 0, 0]);
  expect(weighted.means[0]).toBe(3.5);
  expect(weighted.means[1]).toBeNaN();
  expect(weighted.valueCounts).toEqual([2, 1, 0]);
});

it('zonal statistics oracle computes areas and extents', () => {
  expect(POLYGON_FEATURES.map(computePolygonFeatureArea)).toEqual([15, 16, 8, 0]);
  const withNaNVertex = computePolygonFeatureArea([
    [
      [
        [0, 0],
        [Number.NaN, 5],
        [6, 0],
        [6, 4],
        [0, 4]
      ]
    ]
  ]);
  expect(withNaNVertex).toBe(24);
  expect(computeExtent([5, Number.NaN, -2, 9], [1, 1, 3, 0])).toEqual([-2, 5]);
  expect(computeExtent([1, 2], [0, 0])).toEqual([0, 0]);
});

// Oracle: <scratchpad>/fix/work/D/change-of-support/zonal_variance.py (numpy 2.5.4):
// numpy.var(ddof=1) and numpy.cov(v, aweights=w, ddof=1) over the finite values of each feature;
// weighted density is sum(w) / area. null means NaN (fewer than two valid rows, or area <= 0).
const NUMPY_VARIANCE_CASE = {
  rows: [0, 0, 0, 1, 1, 2, 0, 1, NO_FEATURE, 3, 3, 0, 1, 2],
  values: [2.0, 4.5, 7.25, -1.0, 3.0, 9.0, Number.NaN, 0.5, 100.0, 5.0, 5.0, 1.5, 8.0, 2.0],
  weights: [1.0, 2.0, 0.5, 3.0, 1.0, 2.0, 1.0, 0.25, 1.0, 1.0, 4.0, 1.5, 0.75, 1.0],
  areas: [10.0, 4.0, 0.0, 2.0, 8.0],
  unweighted: [6.973958333333333, 15.5625, 24.5, 0.0, null],
  weighted: [4.915178571428571, 18.281521739130437, 24.500000000000004, 0.0, null],
  weightedDensities: [0.5, 1.25, null, 2.5, 0.0]
};

it('zonal statistics oracle variances match numpy.var and numpy.cov', () => {
  const {rows, values, weights, areas, unweighted, weighted, weightedDensities} =
    NUMPY_VARIANCE_CASE;
  const unweightedResult = computeZonalStatistics({
    featureRows: rows,
    featureCount: 5,
    values,
    areas
  });
  const weightedResult = computeZonalStatistics({
    featureRows: rows,
    featureCount: 5,
    values,
    weights,
    areas
  });
  for (let feature = 0; feature < 5; feature++) {
    for (const [actual, expected] of [
      [unweightedResult.variances[feature], unweighted[feature]],
      [weightedResult.variances[feature], weighted[feature]],
      [weightedResult.weightedDensities[feature], weightedDensities[feature]]
    ] as const) {
      if (expected === null) {
        expect(actual).toBeNaN();
      } else {
        expect(actual).toBeCloseTo(expected, 12);
      }
    }
    expect(weightedResult.standardDeviations[feature]).toEqual(
      Math.sqrt(weightedResult.variances[feature])
    );
  }
});

it('getZonalStatisticsPlan derives variance and weighted density needs', () => {
  const none = {
    counts: false,
    valueCounts: false,
    sums: false,
    weightSums: false,
    means: false,
    minima: false,
    maxima: false,
    densities: false,
    featureAreas: false,
    extent: false
  };
  expect(
    getZonalStatisticsPlan({
      outputs: {...none, standardDeviations: true},
      hasWeights: false,
      hasAreas: false,
      sorted: true
    })
  ).toMatchObject({
    needVariances: true,
    needMeans: true,
    needSums: true,
    needValueCounts: true,
    needWeightSums: false,
    needWeightSquareSums: false
  });
  expect(
    getZonalStatisticsPlan({
      outputs: {...none, variances: true},
      hasWeights: true,
      hasAreas: false,
      sorted: true
    })
  ).toMatchObject({needWeightSums: true, needWeightSquareSums: true, needValueCounts: true});
  expect(
    getZonalStatisticsPlan({
      outputs: {...none, weightedDensities: true},
      hasWeights: true,
      hasAreas: false,
      sorted: true
    })
  ).toMatchObject({needWeightedDensities: true, needWeightSums: true, needGPUAreas: true});
});

it('GPUZonalStatistics variances add a second pass and validate their inputs', () => {
  const nodeIds = (output: (graph: GPUCommandGraph) => GPUZonalStatisticsOutput) => {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    return getNodeIds(createProps(graph, {}, output(graph)), graph);
  };
  const base = nodeIds(graph => ({means: view(graph, 'base-means', 'float32', FEATURE_COUNT)}));
  const withVariance = nodeIds(graph => ({
    variances: view(graph, 'variances', 'float32', FEATURE_COUNT),
    standardDeviations: view(graph, 'deviations', 'float32', FEATURE_COUNT)
  }));
  expect(withVariance.some(id => id.includes('deviation'))).toBe(true);
  expect(withVariance.some(id => id.endsWith('-variances'))).toBe(true);
  expect(base.some(id => id.includes('deviation'))).toBe(false);
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const out = (name: string) => view(graph, name, 'float32', FEATURE_COUNT);
  expect(
    () =>
      new GPUZonalStatistics(createProps(graph, {}, {weightedDensities: out('weighted-density')}))
  ).toThrow(/weightedDensities requires weights/);
  expect(
    () =>
      new GPUZonalStatistics(
        createProps(
          graph,
          {
            values: createVectorView('chunked-values', 'float32', [
              view(graph, 'chunk-a', 'float32', 4),
              view(graph, 'chunk-b', 'float32', 6)
            ])
          },
          {variances: out('chunked-variances')}
        )
      )
  ).toThrow(/packed/);
});

it('GPUZonalStatistics spatial context: planar default, longitude-latitude needs areas', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const out = (name: string) => view(graph, name, 'float32', FEATURE_COUNT);
  const longitudeLatitude = {
    coordinateSpace: 'longitude-latitude',
    metric: 'great-circle'
  } as const;
  expect(new GPUZonalStatistics(createProps(graph)).spatialContext.coordinateSpace).toBe('planar');
  // Counts and sums are topological: allowed in degrees.
  expect(
    new GPUZonalStatistics(createProps(graph, {spatialContext: longitudeLatitude})).spatialContext
      .coordinateSpace
  ).toBe('longitude-latitude');
  // Planar shoelace area of degrees is wrong: densities need caller-supplied spherical areas.
  expect(
    () =>
      new GPUZonalStatistics(
        createProps(graph, {spatialContext: longitudeLatitude}, {densities: out('degree-density')})
      )
  ).toThrow(/areas in square meters/);
  expect(
    () =>
      new GPUZonalStatistics(
        createProps(graph, {spatialContext: longitudeLatitude}, {featureAreas: out('degree-areas')})
      )
  ).toThrow(/areas in square meters/);
  expect(
    () =>
      new GPUZonalStatistics(
        createProps(
          graph,
          {spatialContext: longitudeLatitude, areas: out('meter-areas')},
          {densities: out('meter-density')}
        )
      )
  ).not.toThrow();
  expect(
    () =>
      new GPUZonalStatistics(
        createProps(graph, {
          spatialContext: {coordinateSpace: 'discrete-grid', metric: 'none', gridFamily: 'h3'}
        })
      )
  ).toThrow(/does not support discrete-grid/);
});

it('GPUZonalStatistics gives its membership join a containment context, so longitude-latitude builds', () => {
  for (const metric of ['none', 'great-circle'] as const) {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    const zonal = new GPUZonalStatistics(
      createProps(graph, {spatialContext: {coordinateSpace: 'longitude-latitude', metric}})
    );
    // Building the nodes constructs the join with the forwarded context; it used to be planar.
    expect(zonal.getCommandNodes(graph).some(node => node.id.includes('-join'))).toBe(true);
  }
  expect(
    getPointInPolygonContainmentContext({
      coordinateSpace: 'longitude-latitude',
      metric: 'great-circle'
    })
  ).toEqual({coordinateSpace: 'longitude-latitude', metric: 'none'});
  const planar = {coordinateSpace: 'planar', metric: 'native', units: 'meters'} as const;
  expect(getPointInPolygonContainmentContext(planar)).toBe(planar);
});
