// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUCommandGraph,
  type GraphBufferHandle,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {describe, expect, it} from 'vitest';
import {addChangeOfSupportRecipe} from '../../../src/gpu-spatial-analysis/recipes/change-of-support-recipe';
import {addClusterAndOutlineRecipe} from '../../../src/gpu-spatial-analysis/recipes/cluster-and-outline-recipe';
import {addDriveTimeCatchmentRecipe} from '../../../src/gpu-network/recipes/drive-time-catchment-recipe';
import {
  addFleetDwellRecipe,
  addFleetDwellZoneEventsRecipe
} from '../../../src/gpu-spatial-analysis/recipes/fleet-dwell-recipe';
import {addPeriodComparisonRecipe} from '../../../src/gpu-spatial-analysis/recipes/period-comparison-recipe';
import {addPointsInPolygonsChoroplethRecipe} from '../../../src/gpu-spatial-analysis/recipes/points-in-polygons-choropleth-recipe';
import {addRateClusterMapRecipe} from '../../../src/gpu-spatial-analysis/recipes/rate-cluster-map-recipe';
import {GPUParameterBuffer} from '../../../src/utils/gpu-contributor-utils';
import {addSpaceTimeHotSpotsRecipe} from '../../../src/gpu-spatial-analysis/recipes/space-time-hot-spots-recipe';
import {addSpatialRegressionRecipe} from '../../../src/gpu-spatial-analysis/recipes/spatial-regression-recipe';
import {addStraightLineCatchmentsRecipe} from '../../../src/gpu-spatial-analysis/recipes/straight-line-catchments-recipe';
import {addHotSpotAnalysisRecipe} from '../../../src/gpu-spatial-analysis/recipes/hot-spot-analysis-recipe';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  CORE_PRIVATE_MEMORY_LIMIT_BYTES,
  CORE_WORKGROUP_STORAGE_LIMIT_BYTES,
  getPrivateMemoryBytes,
  getStorageBindingCount,
  getWorkgroupMemoryBytes
} from '../../utils/gpu-contributor-test-utils';
import {makeGPURasterMetadata} from '../../../src/gpu-raster/raster-grid/index';

type ResourceNode = {id: string; resources?: {buffer?: GraphDataView | GraphBufferHandle}[]};

function getBufferBytes(buffer: GraphDataView | GraphBufferHandle): number {
  return 'rowByteLength' in buffer
    ? Math.max(buffer.length, 1) * buffer.rowByteLength
    : buffer.byteLength;
}

/** Asserts storage-binding count, private and workgroup memory of every generated kernel. */
function expectKernelsWithinCoreLimits(nodes: readonly unknown[]): void {
  const kernels = getGPUCommandNodeKernelSources(nodes);
  expect(kernels.length).toBeGreaterThan(0);
  for (const kernel of kernels) {
    expect(getStorageBindingCount(kernel.source), `${kernel.id} storage`).toBeLessThanOrEqual(8);
    expect(getPrivateMemoryBytes(kernel.source), `${kernel.id} private`).toBeLessThanOrEqual(
      CORE_PRIVATE_MEMORY_LIMIT_BYTES
    );
    expect(getWorkgroupMemoryBytes(kernel.source), `${kernel.id} workgroup`).toBeLessThanOrEqual(
      CORE_WORKGROUP_STORAGE_LIMIT_BYTES
    );
  }
}

/** Asserts every transient and bound view of the nodes fits one storage binding. */
function expectBindingsWithinLimit(nodes: readonly unknown[], limit: number): void {
  for (const node of nodes as ResourceNode[]) {
    for (const use of node.resources ?? []) {
      if (use.buffer) {
        expect(getBufferBytes(use.buffer), `${node.id} binding`).toBeLessThanOrEqual(limit);
      }
    }
  }
}

type ViewFormat =
  | 'float32'
  | 'uint32'
  | 'sint32'
  | 'float32x2'
  | 'float32x4'
  | 'uint32x2'
  | 'uint32x4';

type Context = {
  graph: GPUCommandGraph;
  view: <Format extends ViewFormat>(
    format: Format,
    length: number,
    name?: string
  ) => GraphDataView<Format>;
  nodes: () => unknown[];
};

let serial = 0;

function createContext(): Context {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device, {id: `recipe-core-limits-${serial++}`});
  return {
    graph,
    view: (format, length, name = 'view') =>
      createTransientView(graph, `${name}-${serial++}`, format, length),
    // The graph stores every declared node; recipes add them directly instead of returning them.
    nodes: () => (graph as unknown as {nodes: unknown[]}).nodes
  };
}

function getDevice() {
  return createNullWebGPUDevice();
}

/** Asserts a recipe's kernels and, optionally, every binding at its documented maximum. */
function expectRecipeWithinCoreLimits(context: Context, checkBindings = false): void {
  const nodes = context.nodes();
  expect(nodes.length).toBeGreaterThan(0);
  expectKernelsWithinCoreLimits(nodes);
  if (checkBindings) {
    expectBindingsWithinLimit(nodes, getDevice().limits.maxStorageBufferBindingSize);
  }
}

const GEOGRAPHIC_BAND = {coordinateSpace: 'longitude-latitude', metric: 'great-circle'} as const;

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

describe('addHotSpotAnalysisRecipe at the core device limits', () => {
  const maximalOptions = (context: Context, cellCapacity: number) => {
    const {view} = context;
    return {
      settings: {significanceLevel: 0.01, seed: 3, permutationCount: 499},
      parameters: new GPUParameterBuffer(context.graph.device, {
        id: 'hot-spot-parameters',
        format: 'float32',
        length: 8
      }),
      selfWeight: 1,
      minimumNeighbors: 8,
      multipleTesting: 'benjamini-yekutieli' as const,
      permutation: {
        parameters: new GPUParameterBuffer(context.graph.device, {
          id: 'hot-spot-permutation-parameters',
          format: 'uint32',
          length: 8
        }),
        maximumPermutations: 999,
        statistic: 'localGStar' as const,
        alternative: 'two-sided' as const,
        maximumNeighbors: 64,
        multipleTesting: 'benjamini-hochberg' as const
      },
      color: {
        classBreaksParameters: view('float32', 64, 'class-breaks'),
        maximumClassCount: 16,
        methods: [
          'equal-interval',
          'quantile',
          'standard-deviation',
          'head-tail',
          'box-plot',
          'maximum-breaks',
          'natural-breaks',
          'custom'
        ] as never,
        colorScaleParameters: view('float32', 64, 'color-scale'),
        palette: view('uint32', 64, 'palette'),
        maximumPaletteCount: 16
      },
      cellCapacity
    };
  };
  for (const family of ['quadbin', 'h3'] as const) {
    for (const analyze of ['density', 'count'] as const) {
      if (family === 'quadbin' && analyze === 'count') continue;
      it(`${family} points source, ${analyze}, eight rings, every option`, () => {
        const context = createContext();
        const {view} = context;
        const options = maximalOptions(context, 2048);
        const {cellCapacity, ...recipeOptions} = options;
        addHotSpotAnalysisRecipe(context.graph, {
          ...recipeOptions,
          analyze,
          source: {
            kind: 'points',
            positions: view('float32x2', 4096, 'positions'),
            values: view('float32', 4096, 'values'),
            mask: view('uint32', 4096, 'mask'),
            family,
            resolution: 6,
            studyArea: createStudyArea(context, cellCapacity),
            cellCapacity,
            pointCellCapacity: 4096,
            neighborRings: 8
          }
        });
        expectRecipeWithinCoreLimits(context);
      });
    }
  }

  it('points source with a great-circle distance band', () => {
    const context = createContext();
    const {view} = context;
    const {cellCapacity, ...recipeOptions} = maximalOptions(context, 1024);
    addHotSpotAnalysisRecipe(context.graph, {
      ...recipeOptions,
      source: {
        kind: 'points',
        positions: view('float32x2', 2048, 'positions'),
        values: view('float32', 2048, 'values'),
        mask: view('uint32', 2048, 'mask'),
        family: 'h3',
        resolution: 6,
        studyArea: createStudyArea(context, cellCapacity),
        cellCapacity,
        distanceBand: {
          spatialContext: GEOGRAPHIC_BAND,
          neighborSearchParameters: view('float32', 16, 'band-parameters'),
          gridSize: [64, 64],
          neighborCapacity: 1 << 16,
          rowStandardize: true
        }
      }
    });
    expectRecipeWithinCoreLimits(context);
  });

  for (const criterion of ['rook', 'queen'] as const) {
    it(`lattice source, ${criterion}, with every option`, () => {
      const context = createContext();
      const {view} = context;
      const {cellCapacity: _cellCapacity, ...recipeOptions} = maximalOptions(context, 0);
      addHotSpotAnalysisRecipe(context.graph, {
        ...recipeOptions,
        source: {
          kind: 'lattice',
          values: view('float32', 64 * 64, 'values'),
          mask: view('uint32', 64 * 64, 'mask'),
          width: 64,
          height: 64,
          criterion,
          radius: 4,
          neighborCapacity: 64 * 64 * 80
        }
      });
      expectRecipeWithinCoreLimits(context);
    });
  }

  it('keeps every transient within maxStorageBufferBindingSize at the default neighbor capacity', () => {
    // The default neighbor CSR holds cellCapacity * (disk size - 1) slots of four bytes each.
    // The cell-topology disk output is the widest binding: eight bytes per slot.
    const build = (cellCapacity: number) => {
      const context = createContext();
      const {cellCapacity: _ignored, ...recipeOptions} = maximalOptions(context, cellCapacity);
      addHotSpotAnalysisRecipe(context.graph, {
        ...recipeOptions,
        source: {
          kind: 'points',
          positions: context.view('float32x2', 1_000_000, 'positions'),
          family: 'quadbin',
          resolution: 6,
          studyArea: createStudyArea(context, cellCapacity),
          cellCapacity,
          neighborRings: 8
        }
      });
      return context;
    };
    expectRecipeWithinCoreLimits(build(58_000), true);
    // One more slot than fits is refused at construction, not at encoding.
    expect(() => build(59_000)).toThrow(/in one binding/);
  });
});

const ALL_CLASS_METHODS = [
  'equal-interval',
  'quantile',
  'standard-deviation',
  'head-tail',
  'box-plot',
  'maximum-breaks',
  'natural-breaks',
  'custom'
] as never;

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

describe('addRateClusterMapRecipe at the core device limits', () => {
  for (const analyze of ['standardized', 'smoothed'] as const) {
    for (const criterion of ['rook', 'queen'] as const) {
      it(`${analyze} rates, ${criterion} contiguity, permutations, adjustment and every output`, () => {
        const context = createContext();
        const {view} = context;
        const rows = 512;
        addRateClusterMapRecipe(context.graph, {
          events: view('float32', rows, 'events'),
          populations: view('float32', rows, 'populations'),
          mask: view('uint32', rows, 'mask'),
          polygons: {
            kind: 'polygons',
            positions: view('float32x2', rows * 16, 'positions'),
            ringOffsets: view('uint32', rows + 1, 'ring-offsets'),
            polygonOffsets: view('uint32', rows + 1, 'polygon-offsets'),
            featureOffsets: view('uint32', rows + 1, 'feature-offsets')
          },
          criterion,
          snapTolerance: 1e-6,
          neighborCapacity: rows * 16,
          pairCapacity: rows * 32,
          analyze,
          clampPriorVariance: true,
          multipleTesting: 'holm',
          settings: {significanceLevel: 0.01, permutationCount: 499, seed: 7},
          parameters: new GPUParameterBuffer(context.graph.device, {
            id: 'rate-cluster-parameters',
            format: 'float32',
            length: 64
          }),
          permutation: {
            parameters: new GPUParameterBuffer(context.graph.device, {
              id: 'rate-cluster-permutation-parameters',
              format: 'uint32',
              length: 8
            }),
            maximumPermutations: 999,
            alternative: 'two-sided',
            maximumNeighbors: 64
          },
          palette: view('uint32', 5, 'palette')
        });
        expectRecipeWithinCoreLimits(context);
      });
    }
  }
});

describe('addSpaceTimeHotSpotsRecipe at the core device limits', () => {
  const create = (
    kind: 'lattice' | 'ids',
    step: 'year' | 'quarter' | 'month' | 'week' | 'day' | 'hour',
    cellCount: number,
    sliceCount: number
  ) => {
    const context = createContext();
    const {view} = context;
    const width = 32;
    const height = cellCount / width;
    addSpaceTimeHotSpotsRecipe(context.graph, {
      timestamps: view('uint32x2', 2048, 'timestamps'),
      mask: view('uint32', 2048, 'mask'),
      utcOffsets: view('sint32', 2048, 'utc-offsets'),
      calendarParameters: view('sint32', 8, 'calendar-parameters'),
      slices: {
        step,
        interval: 2,
        origin: {year: 2020, month: 1, day: 1, hour: 0},
        count: sliceCount
      },
      cells:
        kind === 'lattice'
          ? {
              kind: 'lattice',
              positions: view('float32x2', 2048, 'positions'),
              width,
              height,
              bounds: [0, 0, 10, 10],
              maximumRadius: 3
            }
          : {
              kind: 'ids',
              cellIds: view('uint32', 2048, 'cell-ids'),
              cellCount,
              weights: {
                offsets: view('uint32', cellCount + 1, 'weights-offsets'),
                neighbors: view('uint32', cellCount * 8, 'weights-neighbors'),
                weights: view('float32', cellCount * 8, 'weights-values')
              },
              selfWeight: 1
            },
      cellMask: view('uint32', cellCount, 'cell-mask'),
      settings: {radius: 3, temporalWindow: 1},
      parameters: new GPUParameterBuffer(context.graph.device, {
        id: 'space-time-parameters',
        format: 'float32',
        length: 64
      }),
      multipleTesting: 'benjamini-hochberg',
      finalSlice: 'last-valid',
      color: {
        palette: view('uint32', 32, 'palette'),
        parameters: view('float32', 64, 'color-parameters'),
        maximumPaletteCount: 32
      }
    });
    return context;
  };
  for (const kind of ['lattice', 'ids'] as const) {
    for (const step of ['year', 'quarter', 'month', 'week', 'day', 'hour'] as const) {
      it(`${kind} cells, ${step} slices, every option`, () => {
        expectRecipeWithinCoreLimits(create(kind, step, 256, 24));
      });
    }
  }

  it('keeps every transient within maxStorageBufferBindingSize at the largest cube', () => {
    // Multiple testing needs fewer than 2^24 cube bins and sliceCount is at most 256.
    expectRecipeWithinCoreLimits(create('lattice', 'day', 1 << 16, 255), true);
  });
});

describe('addSpatialRegressionRecipe at the core device limits', () => {
  for (const predictorCount of [1, 8, 11, 15]) {
    for (const withLocalFits of [false, true]) {
      // The local fit supports at most 11 predictors, the global fit 15.
      if (withLocalFits && predictorCount > 11) continue;
      it(`${predictorCount} predictors${withLocalFits ? ' with local fits' : ''}`, () => {
        const context = createContext();
        const {view} = context;
        const rows = 256;
        addSpatialRegressionRecipe(context.graph, {
          predictors: view('float32', rows * predictorCount, 'predictors'),
          response: view('float32', rows, 'response'),
          mask: view('uint32', rows, 'mask'),
          restandardizeRows: true,
          settings: {significanceLevel: 0.01},
          predictorCount,
          weights: {
            offsets: view('uint32', rows + 1, 'weights-offsets'),
            neighbors: view('uint32', rows * 16, 'weights-neighbors'),
            weights: view('float32', rows * 16, 'weights-values')
          },
          parameters: new GPUParameterBuffer(context.graph.device, {
            id: 'regression-parameters',
            format: 'float32',
            length: 128
          }),
          olsParameters: view('float32', 128, 'ols-parameters'),
          tileRowCount: 64,
          localFits: withLocalFits
            ? {
                positions: view('float32x2', rows, 'positions'),
                parameters: view('float32', 128, 'local-parameters'),
                maximumBandwidthCount: 16,
                maximumNeighborCount: 64,
                refinement: true
              }
            : undefined
        } as never);
        expectRecipeWithinCoreLimits(context);
      });
    }
  }
});

describe('addPeriodComparisonRecipe at the core device limits', () => {
  for (const family of ['quadbin', 'h3'] as const) {
    for (const classify of ['delta', 'ratio', 'percentChange', 'zScore'] as const) {
      const zScore = classify === 'zScore' ? 'poisson' : 'standardized';
      it(`${family} cells classified by ${classify}`, () => {
        const context = createContext();
        const {view} = context;
        const period = (name: string) => ({
          ...(family === 'quadbin'
            ? {positions: view('float32x2', 2048, `${name}-positions`)}
            : {cells: view('uint32x2', 2048, `${name}-cells`)}),
          values: view('float32', 2048, `${name}-values`),
          mask: view('uint32', 2048, `${name}-mask`)
        });
        addPeriodComparisonRecipe(context.graph, {
          family,
          resolution: 7,
          tableCapacity: 2048,
          unionCapacity: 4096,
          before: period('before'),
          after: period('after'),
          measure: 'sum',
          zScore,
          classify,
          classBreaksParameters: view('float32', 64, 'class-breaks'),
          maximumClassCount: 16,
          methods: ALL_CLASS_METHODS,
          palette: view('uint32', 64, 'palette'),
          colorScaleParameters: view('float32', 64, 'color-scale'),
          maximumPaletteCount: 16
        } as never);
        expectRecipeWithinCoreLimits(context);
      });
    }
  }
});

describe('addPointsInPolygonsChoroplethRecipe at the core device limits', () => {
  for (const backend of ['zonal', 'group'] as const) {
    for (const statistic of ['count', 'sum', 'mean', 'minimum', 'maximum'] as const) {
      it(`${backend} backend, ${statistic}, with colors`, () => {
        const context = createContext();
        const {view} = context;
        addPointsInPolygonsChoroplethRecipe(context.graph, {
          backend,
          points: {kind: 'points', positions: view('float32x2', 4096, 'points')},
          values: view('float32', 4096, 'values'),
          polygons: {
            kind: 'polygons',
            positions: view('float32x2', 4096, 'polygon-positions'),
            featureOffsets: view('uint32', 513, 'feature-offsets'),
            polygonOffsets: view('uint32', 513, 'polygon-offsets'),
            ringOffsets: view('uint32', 513, 'ring-offsets'),
            candidateCapacity: 1 << 14,
            leafCapacity: 512,
            boundaryRule: 'closed'
          },
          statistic,
          color: {
            classBreaksParameters: view('float32', 64, 'class-breaks'),
            maximumClassCount: 16,
            methods: ALL_CLASS_METHODS,
            colorScaleParameters: view('float32', 64, 'color-scale'),
            palette: view('uint32', 64, 'palette'),
            maximumPaletteCount: 16
          }
        } as never);
        expectRecipeWithinCoreLimits(context);
      });
    }
  }
});

describe('addChangeOfSupportRecipe at the core device limits', () => {
  for (const denominator of [
    undefined,
    'zone',
    'overlap',
    {extensive: 'overlap', intensive: 'zone'}
  ]) {
    it(`every option with denominator ${JSON.stringify(denominator)}`, () => {
      const context = createContext();
      const {view} = context;
      addChangeOfSupportRecipe(context.graph, {
        grid: makeGPURasterMetadata({width: 64, height: 64}),
        parameters: view('float32', 4, 'parameters'),
        source: createZones(context, 256, 1024),
        target: createZones(context, 256, 1024),
        cellWeights: view('float32', 64 * 64, 'cell-weights'),
        denominator: denominator as never,
        sourceAnchorCells: view('uint32', 256, 'anchors'),
        pairCapacity: 4096,
        sourceValues: view('float32', 256 * 8, 'source-values'),
        columnCount: 8,
        categories: {
          sourceCategories: view('uint32', 256, 'categories'),
          categoryCount: 16
        },
        outputs: {categoryShares: view('float32', 256 * 16, 'category-output')}
      } as never);
      expectRecipeWithinCoreLimits(context);
    });
  }
});

describe('addClusterAndOutlineRecipe at the core device limits', () => {
  it('clusters and outlines with the largest cluster and hull capacities', () => {
    const context = createContext();
    const {view} = context;
    addClusterAndOutlineRecipe(context.graph, {
      points: {kind: 'points', positions: view('float32x2', 4096, 'positions')},
      spatialContext: {coordinateSpace: 'planar', metric: 'native'} as never,
      clusteringParameters: view('float32', 64, 'clustering-parameters'),
      gridSize: [64, 64],
      geometryParameters: view('float32', 64, 'geometry-parameters'),
      maximumClusterCount: 256,
      maximumVerticesPerHull: 256,
      hullCapacity: 4096
    } as never);
    expectRecipeWithinCoreLimits(context);
  });
});

describe('addStraightLineCatchmentsRecipe at the core device limits', () => {
  for (const mode of ['exact', 'jump-flood'] as const) {
    for (const sumOrder of ['atomic', 'sorted'] as const) {
      it(`${mode} distances with ${sumOrder} sums`, () => {
        const context = createContext();
        const {view} = context;
        addStraightLineCatchmentsRecipe(context.graph, {
          grid: makeGPURasterMetadata({width: 128, height: 128}),
          parameters: view('float32', 64, 'parameters'),
          facilities: {
            kind: 'points',
            positions: view('float32x2', 256, 'seeds'),
            sourceIds: view('uint32', 256, 'seed-ids')
          },
          facilityCount: view('uint32', 1, 'seed-count'),
          values: view('float32', 128 * 128, 'values'),
          mode,
          sumOrder
        } as never);
        expectRecipeWithinCoreLimits(context);
      });
    }
  }
});

describe('addDriveTimeCatchmentRecipe at the core device limits', () => {
  it('snaps, routes and bands with every optional input', () => {
    const context = createContext();
    const {view} = context;
    addDriveTimeCatchmentRecipe(context.graph, {
      topology: {
        offsets: view('uint32', 1025, 'offsets'),
        neighbors: view('uint32', 4096, 'neighbors'),
        weights: view('float32', 4096, 'weights'),
        nodePositions: view('float32x2', 1024, 'node-positions')
      },
      facilities: view('float32x2', 64, 'facilities'),
      demand: view('float32x2', 2048, 'demand'),
      demandValues: view('float32', 2048, 'demand-values'),
      maximumSnapDistance: view('float32', 1, 'snap'),
      candidateCapacity: 4096,
      costLimit: view('float32', 1, 'cost-limit'),
      maxIterations: 64,
      costBreaks: view('float32', 16, 'band-breaks'),
      bandCapacity: 64
    } as never);
    expectRecipeWithinCoreLimits(context);
  });
});

describe('addDriveTimeCatchmentRecipe isochrones at the core device limits', () => {
  for (const family of ['h3', 'quadbin'] as const) {
    it(`raster, ${family} cell outline and demand join together`, () => {
      const context = createContext();
      const {view} = context;
      const capacity = 4096;
      addDriveTimeCatchmentRecipe(context.graph, {
        topology: {
          offsets: view('uint32', 1025, 'offsets'),
          neighbors: view('uint32', 4096, 'neighbors'),
          weights: view('float32', 4096, 'weights'),
          nodePositions: view('float32x2', 1024, 'node-positions')
        },
        facilities: view('float32x2', 64, 'facilities'),
        demand: view('float32x2', 2048, 'demand'),
        demandValues: view('float32', 2048, 'demand-values'),
        maximumSnapDistance: view('float32', 1, 'snap'),
        candidateCapacity: 4096,
        costLimit: view('float32', 1, 'cost-limit'),
        maxIterations: 64,
        costBreaks: view('float32', 16, 'band-breaks'),
        bandCapacity: 64,
        isochrones: {
          costBreaks: view('float32', 16, 'isochrone-breaks'),
          parameters: view('float32', 64, 'isochrone-parameters'),
          raster: {
            width: 128,
            height: 128,
            maximumSamplesPerEdge: 16,
            output: {
              bandClasses: view('uint32', 128 * 128, 'band-classes'),
              triangles: view('float32x2', capacity * 3, 'triangles'),
              triangleBands: view('uint32', capacity, 'triangle-bands'),
              edges: view('float32x4', capacity, 'edges'),
              edgeBands: view('uint32', capacity, 'edge-bands'),
              status: {
                count: view('uint32', 1, 'triangle-count'),
                overflow: view('uint32', 1, 'triangle-overflow'),
                requiredCount: view('uint32', 1, 'required'),
                vertexCount: view('uint32', 1, 'vertex-count'),
                edgeCount: view('uint32', 1, 'edge-count'),
                edgeOverflow: view('uint32', 1, 'edge-overflow'),
                edgeRequiredCount: view('uint32', 1, 'edge-total')
              },
              values: view('float32', 128 * 128, 'raster-values')
            }
          },
          cellOutline: {
            family,
            resolution: 8,
            tableCapacity: 1024,
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
              vertexTolerance: 1e-6,
              normalizeWinding: true,
              output: {
                ringOffsets: view('uint32', capacity + 1, 'ring-offsets'),
                positions: view('float32x2', capacity, 'ring-positions'),
                ringAreas: view('float32', capacity, 'ring-areas'),
                ringIsHole: view('uint32', capacity, 'ring-is-hole'),
                ringShells: view('uint32', capacity, 'ring-shells'),
                segmentRings: view('uint32', capacity, 'segment-rings'),
                segmentVertices: view('uint32', capacity, 'segment-vertices'),
                segmentFlags: view('uint32', capacity, 'segment-flags'),
                polygons: {
                  positions: view('float32x2', capacity, 'polygon-positions'),
                  ringOffsets: view('uint32', capacity + 1, 'polygon-ring-offsets'),
                  polygonOffsets: view('uint32', capacity + 1, 'polygon-offsets'),
                  featureOffsets: view('uint32', capacity + 1, 'polygon-feature-offsets')
                },
                status: {
                  count: view('uint32', 1, 'ring-count'),
                  overflow: view('uint32', 1, 'ring-overflow'),
                  requiredCount: view('uint32', 1, 'ring-required'),
                  openSegmentCount: view('uint32', 1, 'open-segments'),
                  touchingSegmentCount: view('uint32', 1, 'touching-segments')
                }
              }
            }
          },
          joinDemand: {candidateCapacity: 4096}
        }
      } as never);
      expectRecipeWithinCoreLimits(context);
    });
  }
});

describe('fleet dwell recipes at the core device limits', () => {
  it('addFleetDwellRecipe with stops, zones and the zone table', () => {
    const context = createContext();
    const {view} = context;
    addFleetDwellRecipe(context.graph, {
      trajectories: {
        kind: 'trajectories',
        positions: view('float32x2', 4096, 'positions'),
        timestamps: view('float32', 4096, 'timestamps'),
        trackOffsets: view('uint32', 65, 'track-offsets')
      },
      parameters: view('float32', 64, 'parameters'),
      stopCapacity: 512,
      zones: {
        kind: 'polygons',
        positions: view('float32x2', 1024, 'zone-positions'),
        featureOffsets: view('uint32', 65, 'zone-features'),
        polygonOffsets: view('uint32', 65, 'zone-polygons'),
        ringOffsets: view('uint32', 65, 'zone-rings'),
        sourceIds: view('uint32', 64, 'zone-ids'),
        candidateCapacity: 4096
      },
      zoneCapacity: 128
    } as never);
    expectRecipeWithinCoreLimits(context);
  });

  for (const timestampFormat of ['float32', 'uint32x2'] as const) {
    it(`addFleetDwellZoneEventsRecipe with ${timestampFormat} timestamps`, () => {
      const context = createContext();
      const {view} = context;
      addFleetDwellZoneEventsRecipe(context.graph, {
        trajectories: {
          kind: 'trajectories',
          positions: view('float32x2', 4096, 'positions'),
          timestamps: view(timestampFormat, 4096, 'timestamps') as never,
          trackOffsets: view('uint32', 65, 'track-offsets')
        },
        zones: {
          kind: 'polygons',
          positions: view('float32x2', 512, 'zone-positions'),
          featureOffsets: view('uint32', 33, 'zone-features'),
          polygonOffsets: view('uint32', 33, 'zone-polygons'),
          ringOffsets: view('uint32', 33, 'zone-rings')
        },
        candidateCapacity: 4096,
        trackEventCapacity: 16,
        eventCapacity: 1024
      } as never);
      expectRecipeWithinCoreLimits(context);
    });
  }
});
