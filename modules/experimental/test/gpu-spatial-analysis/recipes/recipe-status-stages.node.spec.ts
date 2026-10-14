// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUCommandGraph,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {describe, expect, it} from 'vitest';
import {GPUNetworkIsochrones} from '../../../src/gpu-network/network-isochrones/index';
import type {GPURecipeStatusPort} from '../../../src/analysis-core/contracts/index';
import {addChangeOfSupportRecipe} from '../../../src/gpu-spatial-analysis/recipes/change-of-support-recipe';
import {addClusterAndOutlineRecipe} from '../../../src/gpu-spatial-analysis/recipes/cluster-and-outline-recipe';
import {addDriveTimeCatchmentRecipe} from '../../../src/gpu-network/recipes/drive-time-catchment-recipe';
import {
  addFleetDwellRecipe,
  addFleetDwellZoneEventsRecipe
} from '../../../src/gpu-spatial-analysis/recipes/fleet-dwell-recipe';
import {addHotSpotAnalysisRecipe} from '../../../src/gpu-spatial-analysis/recipes/hot-spot-analysis-recipe';
import {addPeriodComparisonRecipe} from '../../../src/gpu-spatial-analysis/recipes/period-comparison-recipe';
import {addPointsInPolygonsChoroplethRecipe} from '../../../src/gpu-spatial-analysis/recipes/points-in-polygons-choropleth-recipe';
import {addSpaceTimeHotSpotsRecipe} from '../../../src/gpu-spatial-analysis/recipes/space-time-hot-spots-recipe';
import {addSpatialRegressionRecipe} from '../../../src/gpu-spatial-analysis/recipes/spatial-regression-recipe';
import {addStraightLineCatchmentsRecipe} from '../../../src/gpu-spatial-analysis/recipes/straight-line-catchments-recipe';
import {makeGPURasterMetadata} from '../../../src/gpu-raster/raster-grid/index';
import {GPUParameterBuffer} from '../../../src/utils/gpu-contributor-utils';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

type ViewFormat = 'float32' | 'uint32' | 'sint32' | 'float32x2' | 'float32x4' | 'uint32x2';

type Context = {
  graph: GPUCommandGraph;
  view: <Format extends ViewFormat>(
    format: Format,
    length: number,
    name?: string
  ) => GraphDataView<Format>;
};

let serial = 0;

function createContext(): Context {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: `recipe-status-${serial++}`});
  return {
    graph,
    view: (format, length, name = 'view') =>
      createTransientView(graph, `${name}-${serial++}`, format, length)
  };
}

/** Collects every view reachable through plain objects and arrays, without entering a view. */
function collectViews(value: unknown, views: Set<unknown>, seen: Set<unknown>, depth = 0): void {
  if (!value || typeof value !== 'object' || seen.has(value) || depth > 8) {
    return;
  }
  seen.add(value);
  if ('rowByteLength' in value) {
    views.add(value);
    return;
  }
  const prototype = Object.getPrototypeOf(value);
  if (prototype !== Object.prototype && !Array.isArray(value)) {
    // Class instances such as a GPUSpatialContext helper are not output containers.
    return;
  }
  for (const child of Object.values(value)) {
    collectViews(child, views, seen, depth + 1);
  }
}

/** Views passed to the inner contributors of a recipe, by identity. */
function getContributorViews(contributors: readonly GPUCommandNodeProducer<never>[]) {
  const views = new Set<unknown>();
  const seen = new Set<unknown>();
  for (const contributor of contributors) {
    collectViews((contributor as unknown as {props: unknown}).props, views, seen);
  }
  return views;
}

/** Stages whose status is written by an adapter kernel of the recipe, not an inner contributor. */
const ADAPTER_STAGES = new Set(['minimum-neighbors', 'slice-range']);

/**
 * Asserts the exact stage names and that every status view of every other stage is passed to an
 * inner contributor, so the recipe reports what the contributor writes and nothing invented.
 */
function expectStages(
  recipe: {
    status: GPURecipeStatusPort;
    contributors: readonly GPUCommandNodeProducer<never>[];
  },
  stageNames: string[],
  adapterStages: readonly string[] = []
): Record<string, GPURecipeStatusPort['stages'][number]['status']> {
  expect(recipe.status.stages.map(({stage}) => stage)).toEqual(stageNames);
  const written = getContributorViews(recipe.contributors);
  const byName: Record<string, GPURecipeStatusPort['stages'][number]['status']> = {};
  for (const {stage, status} of recipe.status.stages) {
    byName[stage] = status;
    for (const [name, view] of Object.entries(status)) {
      if (view === undefined) {
        continue;
      }
      expect((view as GraphDataView).format, `${stage}.${name} format`).toBe('uint32');
      if (!ADAPTER_STAGES.has(stage) && !adapterStages.includes(stage)) {
        expect(written.has(view), `${stage}.${name} is written by a contributor`).toBe(true);
      }
    }
  }
  return byName;
}

function createZones({view}: Context, features: number, crossingCapacity: number) {
  return {
    kind: 'polygons',
    positions: view('float32x2', features * 8, 'zone-positions'),
    featureOffsets: view('uint32', features + 1, 'zone-features'),
    polygonOffsets: view('uint32', features + 1, 'zone-polygons'),
    ringOffsets: view('uint32', features + 1, 'zone-rings'),
    crossingCapacity
  };
}

function createStudyArea({view}: Context, cellCapacity: number) {
  return {
    kind: 'polygons' as const,
    positions: view('float32x2', 16, 'polygon'),
    featureOffsets: view('uint32', 3, 'features'),
    polygonOffsets: view('uint32', 3, 'polygons'),
    ringOffsets: view('uint32', 3, 'rings'),
    candidateCapacity: cellCapacity,
    coverCapacity: cellCapacity * 2
  };
}

const CLASS_METHODS = ['equal-interval', 'quantile', 'custom'] as never;

describe('change of support recipe status', () => {
  it('reports both rasterizations (overflow, crossing count) and the areal stage', () => {
    const context = createContext();
    const {view} = context;
    const recipe = addChangeOfSupportRecipe(context.graph, {
      grid: makeGPURasterMetadata({width: 16, height: 16}),
      parameters: view('float32', 4, 'parameters'),
      source: createZones(context, 8, 64),
      target: createZones(context, 8, 64),
      pairCapacity: 64
    } as never);
    const stages = expectStages(recipe, [
      'source-rasterization',
      'target-rasterization',
      'areal-interpolation'
    ]);
    expect(stages['source-rasterization']).toEqual({
      overflow: recipe.status.rasterOverflow[0],
      requiredCount: recipe.status.rasterRequiredCount[0]
    });
    expect(stages['target-rasterization']).toEqual({
      overflow: recipe.status.rasterOverflow[1],
      requiredCount: recipe.status.rasterRequiredCount[1]
    });
    expect(recipe.status.rasterRequiredCount[0]).not.toBe(recipe.status.rasterRequiredCount[1]);
    expect(Object.keys(stages['areal-interpolation']).sort()).toEqual([
      'invalidCount',
      'overflow',
      'requiredCount'
    ]);
  });

  it('uses caller-owned crossing counts', () => {
    const context = createContext();
    const {view} = context;
    const given: [GraphDataView<'uint32'>, GraphDataView<'uint32'>] = [
      view('uint32', 1, 'given-source-crossings'),
      view('uint32', 1, 'given-target-crossings')
    ];
    const recipe = addChangeOfSupportRecipe(context.graph, {
      grid: makeGPURasterMetadata({width: 16, height: 16}),
      parameters: view('float32', 4, 'parameters'),
      source: createZones(context, 8, 64),
      target: createZones(context, 8, 64),
      pairCapacity: 64,
      outputs: {status: {rasterRequiredCount: given}}
    } as never);
    expect(recipe.status.rasterRequiredCount).toEqual(given);
    expect(recipe.status.rasterRequiredCount[0]).toBe(given[0]);
    expectStages(recipe, ['source-rasterization', 'target-rasterization', 'areal-interpolation']);
  });
});

describe('cluster and outline recipe status', () => {
  it('reports the hull measurement invalid count after the hull stage', () => {
    const context = createContext();
    const {view} = context;
    const recipe = addClusterAndOutlineRecipe(context.graph, {
      points: {kind: 'points', positions: view('float32x2', 256, 'positions')},
      spatialContext: {coordinateSpace: 'planar', metric: 'native'} as never,
      clusteringParameters: view('float32', 64, 'clustering-parameters'),
      gridSize: [16, 16],
      geometryParameters: view('float32', 64, 'geometry-parameters'),
      maximumClusterCount: 16,
      maximumVerticesPerHull: 16,
      hullCapacity: 256
    } as never);
    const stages = expectStages(recipe, ['clustering', 'convex-hull', 'hull-measures']);
    expect(stages['hull-measures']).toEqual({invalidCount: recipe.status.invalidCount});
  });
});

describe('drive-time catchment recipe status', () => {
  const createNetwork = ({view}: Context) => ({
    offsets: view('uint32', 65, 'offsets'),
    neighbors: view('uint32', 256, 'neighbors'),
    weights: view('float32', 256, 'weights'),
    nodePositions: view('float32x2', 64, 'node-positions')
  });

  it('reports snapping, service areas and bands', () => {
    const context = createContext();
    const {view} = context;
    const recipe = addDriveTimeCatchmentRecipe(context.graph, {
      topology: createNetwork(context),
      facilities: view('float32x2', 4, 'facilities'),
      demand: view('float32x2', 32, 'demand'),
      maximumSnapDistance: view('float32', 1, 'snap'),
      candidateCapacity: 64,
      costBreaks: view('float32', 4, 'band-breaks')
    } as never);
    const stages = expectStages(recipe, [
      'snap-facilities',
      'snap-demand',
      'service-areas',
      'band-statistics'
    ]);
    expect(stages['service-areas']).toEqual({converged: recipe.status.serviceAreaConverged});
    expect(stages['band-statistics']).toEqual({
      count: recipe.bands.status.count,
      overflow: recipe.bands.status.overflow,
      requiredCount: recipe.bands.status.requiredCount
    });
  });

  it('forwards the service-area assignments to the isochrones and reports their status', () => {
    const context = createContext();
    const {view} = context;
    const capacity = 256;
    const recipe = addDriveTimeCatchmentRecipe(context.graph, {
      topology: createNetwork(context),
      facilities: view('float32x2', 4, 'facilities'),
      demand: view('float32x2', 32, 'demand'),
      costBreaks: view('float32', 4, 'band-breaks'),
      isochrones: {
        costBreaks: view('float32', 4, 'isochrone-breaks'),
        parameters: view('float32', 64, 'isochrone-parameters'),
        raster: {
          width: 16,
          height: 16,
          output: {
            triangles: view('float32x2', capacity * 3, 'triangles'),
            triangleBands: view('uint32', capacity, 'triangle-bands'),
            edges: view('float32x4', capacity, 'edges'),
            edgeBands: view('uint32', capacity, 'edge-bands'),
            status: {
              count: view('uint32', 1, 'triangle-count'),
              overflow: view('uint32', 1, 'triangle-overflow'),
              requiredCount: view('uint32', 1, 'triangle-required'),
              edgeCount: view('uint32', 1, 'edge-count'),
              edgeOverflow: view('uint32', 1, 'edge-overflow'),
              edgeRequiredCount: view('uint32', 1, 'edge-total')
            },
            pixelFacilities: view('uint32', 16 * 16, 'pixel-facilities')
          }
        }
      }
    } as never);
    const isochrones = recipe.contributors.find(
      (contributor): contributor is GPUNetworkIsochrones =>
        contributor instanceof GPUNetworkIsochrones
    )!;
    // The facility-labelled outputs need the allocation: it is the service-area assignments.
    expect(isochrones.props.assignments).toBe(recipe.assignments);
    expect(recipe.outputs['assignments']).toBe(recipe.assignments);
    const stages = expectStages(recipe, [
      'service-areas',
      'isochrone-bands',
      'isochrone-band-edges',
      'band-statistics'
    ]);
    expect(Object.keys(stages['isochrone-bands']).sort()).toEqual([
      'count',
      'overflow',
      'requiredCount'
    ]);
  });

  it('reports the cell outline path: table, outline, rings and the demand join', () => {
    const context = createContext();
    const {view} = context;
    const capacity = 256;
    const table = {
      cells: view('uint32x2', 64, 'table-cells'),
      counts: view('uint32', 64, 'table-counts'),
      status: {
        count: view('uint32', 1, 'table-count'),
        overflow: view('uint32', 1, 'table-overflow'),
        requiredCount: view('uint32', 1, 'table-required')
      }
    };
    const recipe = addDriveTimeCatchmentRecipe(context.graph, {
      topology: createNetwork(context),
      facilities: view('float32x2', 4, 'facilities'),
      demand: view('float32x2', 32, 'demand'),
      costBreaks: view('float32', 4, 'band-breaks'),
      isochrones: {
        costBreaks: view('float32', 4, 'isochrone-breaks'),
        parameters: view('float32', 64, 'isochrone-parameters'),
        cellOutline: {
          family: 'h3',
          resolution: 8,
          table,
          output: {
            rows: view('uint32', capacity, 'outline-rows'),
            cells: view('uint32x2', capacity, 'outline-cells'),
            edgeIndices: view('uint32', capacity, 'outline-edges'),
            endpoints: view('float32x4', capacity, 'outline-endpoints'),
            status: {
              count: view('uint32', 1, 'outline-count'),
              overflow: view('uint32', 1, 'outline-overflow'),
              requiredCount: view('uint32', 1, 'outline-required')
            }
          },
          rings: {
            output: {
              ringOffsets: view('uint32', capacity + 1, 'ring-offsets'),
              positions: view('float32x2', capacity, 'ring-positions'),
              polygons: {
                positions: view('float32x2', capacity, 'polygon-positions'),
                ringOffsets: view('uint32', capacity + 1, 'polygon-ring-offsets'),
                polygonOffsets: view('uint32', capacity + 1, 'polygon-offsets'),
                featureOffsets: view('uint32', capacity + 1, 'polygon-feature-offsets')
              },
              status: {
                count: view('uint32', 1, 'ring-count'),
                overflow: view('uint32', 1, 'ring-overflow'),
                requiredCount: view('uint32', 1, 'ring-required')
              }
            }
          }
        },
        joinDemand: {candidateCapacity: 256}
      }
    } as never);
    const stages = expectStages(recipe, [
      'service-areas',
      'isochrone-cells',
      'isochrone-outline',
      'isochrone-rings',
      'band-statistics',
      'isochrone-demand'
    ]);
    expect(stages['isochrone-cells']).toEqual({
      count: table.status.count,
      overflow: table.status.overflow,
      requiredCount: table.status.requiredCount
    });
    expect(stages['isochrone-demand']).toEqual({
      overflow: recipe.isochroneDemand!.status.overflow,
      uncertainCount: recipe.isochroneDemand!.status.uncertainCount
    });
  });
});

describe('fleet dwell recipes status', () => {
  it('stops variant reports stop count and required count, join uncertainty and zone table', () => {
    const context = createContext();
    const {view} = context;
    const recipe = addFleetDwellRecipe(context.graph, {
      trajectories: {
        kind: 'trajectories',
        positions: view('float32x2', 256, 'positions'),
        timestamps: view('float32', 256, 'timestamps'),
        trackOffsets: view('uint32', 5, 'track-offsets')
      },
      parameters: view('float32', 64, 'parameters'),
      stopCapacity: 32,
      zones: {
        kind: 'polygons',
        positions: view('float32x2', 32, 'zone-positions'),
        featureOffsets: view('uint32', 5, 'zone-features'),
        polygonOffsets: view('uint32', 5, 'zone-polygons'),
        ringOffsets: view('uint32', 5, 'zone-rings'),
        candidateCapacity: 64
      }
    } as never);
    expect(recipe.spatialContext).toEqual({
      coordinateSpace: 'planar',
      metric: 'native',
      units: 'native'
    });
    const stages = expectStages(recipe, ['metrics', 'zone-join', 'zone-statistics']);
    expect(stages['metrics']).toEqual({
      count: recipe.stops.status.count,
      overflow: recipe.stops.status.overflow,
      requiredCount: recipe.stops.status.requiredCount
    });
    expect(stages['zone-join']).toEqual({
      overflow: recipe.status.joinOverflow,
      uncertainCount: recipe.status.joinUncertainCount
    });
    expect(stages['zone-statistics']).toEqual({
      count: recipe.table.status.count,
      overflow: recipe.table.status.overflow,
      requiredCount: recipe.table.status.requiredCount
    });
  });

  it('zone events variant reports event cardinality, candidates, visit table and zone table', () => {
    const create = (withVisitTable: boolean) => {
      const context = createContext();
      const {view} = context;
      const visitTable = withVisitTable
        ? {
            ids: view('uint32', 16, 'visit-ids'),
            status: {
              count: view('uint32', 1, 'visit-count'),
              overflow: view('uint32', 1, 'visit-overflow'),
              requiredCount: view('uint32', 1, 'visit-required')
            },
            zones: view('uint32', 16, 'visit-zones')
          }
        : undefined;
      const recipe = addFleetDwellZoneEventsRecipe(context.graph, {
        trajectories: {
          kind: 'trajectories',
          positions: view('float32x2', 256, 'positions'),
          timestamps: view('float32', 256, 'timestamps'),
          trackOffsets: view('uint32', 5, 'track-offsets')
        },
        zones: {
          kind: 'polygons',
          positions: view('float32x2', 32, 'zone-positions'),
          featureOffsets: view('uint32', 5, 'zone-features'),
          polygonOffsets: view('uint32', 5, 'zone-polygons'),
          ringOffsets: view('uint32', 5, 'zone-rings')
        },
        candidateCapacity: 64,
        trackEventCapacity: 8,
        eventCapacity: 32,
        scratch: visitTable ? {visitTable} : undefined
      } as never);
      return {recipe, visitTable};
    };
    const plain = create(false);
    expect(plain.recipe.spatialContext).toEqual({
      coordinateSpace: 'planar',
      metric: 'native',
      units: 'native'
    });
    const stages = expectStages(plain.recipe, ['zone-events', 'zone-statistics']);
    expect(stages['zone-events']).toEqual({
      count: plain.recipe.events.status.count,
      overflow: plain.recipe.events.status.overflow,
      candidateOverflow: plain.recipe.status.candidateOverflow,
      requiredCount: plain.recipe.events.status.requiredCount
    });
    const withTable = create(true);
    const tableStages = expectStages(withTable.recipe, [
      'zone-events',
      'visit-table',
      'zone-statistics'
    ]);
    expect(tableStages['visit-table']).toEqual({
      count: withTable.visitTable!.status.count,
      overflow: withTable.visitTable!.status.overflow,
      requiredCount: withTable.visitTable!.status.requiredCount
    });
  });
});

describe('hot spot analysis recipe status', () => {
  let parameterSerial = 0;
  const parameters = (context: Context) =>
    new GPUParameterBuffer(context.graph.device, {
      id: `hot-spot-parameters-${parameterSerial++}`,
      format: 'float32',
      length: 8
    });

  it('points source reports the cover, both tables, the weights and the guard', () => {
    const context = createContext();
    const {view} = context;
    const recipe = addHotSpotAnalysisRecipe(context.graph, {
      parameters: parameters(context),
      source: {
        kind: 'points',
        positions: view('float32x2', 64, 'positions'),
        family: 'quadbin',
        resolution: 6,
        studyArea: createStudyArea(context, 32),
        cellCapacity: 32
      }
    });
    const stages = expectStages(
      recipe,
      ['study-area', 'study-table', 'point-aggregation', 'weights', 'minimum-neighbors'],
      // The grid-disk CSR is built by recipe adapter kernels.
      ['weights']
    );
    expect(Object.keys(stages['study-area']).sort()).toEqual([
      'candidateOverflow',
      'count',
      'invalidCount',
      'overflow',
      'requiredCount',
      'uncertainCount'
    ]);
    expect(Object.keys(stages['study-table']).sort()).toEqual([
      'count',
      'overflow',
      'requiredCount'
    ]);
    expect(Object.keys(stages['point-aggregation']).sort()).toEqual([
      'count',
      'overflow',
      'requiredCount'
    ]);
    // The grid-disk weights adapter now reports the unclamped total for the grow loop.
    expect(stages['weights']).toEqual({
      overflow: recipe.status.weightsOverflow,
      requiredCount: recipe.status.weightsRequiredCount
    });
  });

  it('distance band reports the unclamped neighbor total and the dropped rows', () => {
    const context = createContext();
    const {view} = context;
    const recipe = addHotSpotAnalysisRecipe(context.graph, {
      parameters: parameters(context),
      minimumNeighbors: 1,
      source: {
        kind: 'points',
        positions: view('float32x2', 64, 'positions'),
        family: 'h3',
        resolution: 6,
        studyArea: createStudyArea(context, 32),
        cellCapacity: 32,
        distanceBand: {
          spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'},
          neighborSearchParameters: view('float32', 16, 'band-parameters'),
          gridSize: [8, 8],
          neighborCapacity: 256
        }
      }
    });
    const stages = expectStages(recipe, [
      'study-area',
      'study-table',
      'point-aggregation',
      'weights',
      'minimum-neighbors'
    ]);
    expect(stages['weights']).toEqual({
      overflow: recipe.status.weightsOverflow,
      requiredCount: recipe.status.weightsRequiredCount,
      invalidCount: recipe.status.weightsInvalidCount
    });
    expect(recipe.status.weightsRequiredCount).toBeDefined();
  });

  it('lattice source with permutation reports weights, the guard and the permutation test', () => {
    const context = createContext();
    const {view} = context;
    const recipe = addHotSpotAnalysisRecipe(context.graph, {
      parameters: parameters(context),
      permutation: {
        parameters: new GPUParameterBuffer(context.graph.device, {
          id: 'hot-spot-permutation-parameters',
          format: 'uint32',
          length: 8
        }),
        maximumPermutations: 99
      },
      source: {
        kind: 'lattice',
        values: view('float32', 64, 'values'),
        width: 8,
        height: 8,
        neighborCapacity: 512
      }
    });
    const stages = expectStages(recipe, ['weights', 'minimum-neighbors', 'permutation']);
    expect(stages['weights']).toEqual({
      overflow: recipe.status.weightsOverflow,
      requiredCount: recipe.status.weightsRequiredCount
    });
    expect(stages['permutation']).toEqual({overflow: recipe.permutation!.status.overflow});
  });
});

describe('period comparison recipe status', () => {
  it('reports count, overflow and required count of both periods and the union', () => {
    const context = createContext();
    const {view} = context;
    const period = (name: string) => ({
      positions: view('float32x2', 64, `${name}-positions`)
    });
    const recipe = addPeriodComparisonRecipe(context.graph, {
      family: 'quadbin',
      resolution: 7,
      tableCapacity: 64,
      before: period('before'),
      after: period('after'),
      classBreaksParameters: view('float32', 64, 'class-breaks'),
      maximumClassCount: 8,
      methods: CLASS_METHODS,
      palette: view('uint32', 16, 'palette'),
      colorScaleParameters: view('float32', 64, 'color-scale'),
      maximumPaletteCount: 8
    } as never);
    const stages = expectStages(recipe, ['before-aggregation', 'after-aggregation', 'comparison']);
    expect(stages['before-aggregation']).toEqual({
      count: recipe.before.status.count,
      overflow: recipe.before.status.overflow,
      requiredCount: recipe.before.status.requiredCount
    });
    expect(stages['after-aggregation']).toEqual({
      count: recipe.after.status.count,
      overflow: recipe.after.status.overflow,
      requiredCount: recipe.after.status.requiredCount
    });
    expect(stages['comparison']).toEqual({
      count: recipe.comparison.status.count,
      overflow: recipe.comparison.status.overflow,
      requiredCount: recipe.comparison.status.requiredCount
    });
    expect(recipe.comparison.status.requiredCount).toBeDefined();
  });
});

describe('points in polygons choropleth recipe status', () => {
  const create = (backend: 'zonal' | 'group') => {
    const context = createContext();
    const {view} = context;
    return addPointsInPolygonsChoroplethRecipe(context.graph, {
      backend,
      points: {kind: 'points', positions: view('float32x2', 128, 'points')},
      polygons: {
        kind: 'polygons',
        positions: view('float32x2', 64, 'polygon-positions'),
        featureOffsets: view('uint32', 5, 'feature-offsets'),
        polygonOffsets: view('uint32', 5, 'polygon-offsets'),
        ringOffsets: view('uint32', 5, 'ring-offsets'),
        candidateCapacity: 256
      }
    } as never);
  };

  it('zonal backend reports overflow and uncertain pairs', () => {
    const recipe = create('zonal');
    expect(recipe.spatialContext).toEqual({
      coordinateSpace: 'planar',
      metric: 'native',
      units: 'native'
    });
    const stages = expectStages(recipe, ['zonal-statistics']);
    expect(stages['zonal-statistics']).toEqual({
      overflow: recipe.status.overflow,
      uncertainCount: recipe.status.uncertainCount
    });
  });

  it('group backend reports the join and the dense group table', () => {
    const recipe = create('group');
    const stages = expectStages(recipe, ['point-in-polygon', 'group-statistics']);
    expect(stages['point-in-polygon']).toEqual({
      overflow: recipe.status.overflow,
      uncertainCount: recipe.status.uncertainCount
    });
    expect(Object.keys(stages['group-statistics']).sort()).toEqual([
      'count',
      'overflow',
      'requiredCount'
    ]);
  });
});

describe('space-time hot spots recipe status', () => {
  const create = (sorted: boolean) => {
    const context = createContext();
    const {view} = context;
    const groupRequiredCount = view('uint32', 1, 'group-required');
    const recipe = addSpaceTimeHotSpotsRecipe(context.graph, {
      timestamps: view('uint32x2', 64, 'timestamps'),
      calendarParameters: view('sint32', 8, 'calendar-parameters'),
      slices: {step: 'month', origin: {year: 2020}, count: 12},
      cells: {
        kind: 'lattice',
        positions: view('float32x2', 64, 'positions'),
        width: 4,
        height: 4,
        bounds: [0, 0, 1, 1]
      },
      settings: {temporalWindow: 1},
      parameters: new GPUParameterBuffer(context.graph.device, {
        id: 'space-time-parameters',
        format: 'float32',
        length: 64
      }),
      outputs: sorted
        ? {status: {groupOverflow: view('uint32', 1, 'group-overflow'), groupRequiredCount}}
        : undefined
    } as never);
    return {recipe, groupRequiredCount};
  };

  it('counts with atomics and reports only the slice range', () => {
    const {recipe} = create(false);
    const stages = expectStages(recipe, ['slice-range']);
    expect(Object.keys(stages['slice-range'])).toEqual(['invalidCount']);
  });

  it('reports the cube counts of the sorted path with the required count', () => {
    const {recipe, groupRequiredCount} = create(true);
    const stages = expectStages(recipe, ['slice-range', 'cube-counts']);
    expect(stages['cube-counts'].requiredCount).toBe(groupRequiredCount);
    expect(Object.keys(stages['cube-counts']).sort()).toEqual([
      'count',
      'overflow',
      'requiredCount'
    ]);
  });
});

describe('spatial regression recipe status', () => {
  it('reports the solver and diagnostics status words', () => {
    const context = createContext();
    const {view} = context;
    const rows = 32;
    const recipe = addSpatialRegressionRecipe(context.graph, {
      predictors: view('float32', rows * 2, 'predictors'),
      response: view('float32', rows, 'response'),
      predictorCount: 2,
      weights: {
        offsets: view('uint32', rows + 1, 'weights-offsets'),
        neighbors: view('uint32', rows * 4, 'weights-neighbors'),
        weights: view('float32', rows * 4, 'weights-values')
      },
      parameters: new GPUParameterBuffer(context.graph.device, {
        id: 'regression-parameters',
        format: 'float32',
        length: 64
      })
    } as never);
    const stages = expectStages(recipe, ['ordinary-least-squares', 'spatial-diagnostics']);
    expect(stages['ordinary-least-squares']).toEqual({invalidCount: recipe.ols.status.code});
    expect(stages['spatial-diagnostics']).toEqual({invalidCount: recipe.diagnostics.status.code});
  });
});

describe('straight-line catchments recipe status', () => {
  it('reports the zonal statistics overflow', () => {
    const context = createContext();
    const {view} = context;
    const recipe = addStraightLineCatchmentsRecipe(context.graph, {
      grid: makeGPURasterMetadata({width: 16, height: 16}),
      parameters: view('float32', 64, 'parameters'),
      facilities: {kind: 'points', positions: view('float32x2', 8, 'seeds')},
      values: view('float32', 256, 'values')
    } as never);
    expect(recipe.spatialContext).toEqual({
      coordinateSpace: 'planar',
      metric: 'native',
      units: 'native'
    });
    const stages = expectStages(recipe, ['zonal-statistics']);
    expect(stages['zonal-statistics']).toEqual({overflow: recipe.status.overflow});
  });
});
