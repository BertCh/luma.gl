// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  addChangeOfSupportRecipe,
  addClusterAndOutlineRecipe,
  addFleetDwellRecipe,
  addFleetDwellZoneEventsRecipe,
  addHotSpotAnalysisRecipe,
  addPeriodComparisonRecipe,
  addPointsInPolygonsChoroplethRecipe,
  addRateClusterMapRecipe,
  addSpaceTimeHotSpotsRecipe,
  addSpatialRegressionRecipe,
  addStraightLineCatchmentsRecipe
} from '../../../src/gpu-spatial-analysis';
import {makeGPURasterMetadata} from '../../../src/gpu-raster';
import {GPUParameterBuffer} from '../../../src/utils/gpu-contributor-utils';
import type {KernelAuditCase, KernelAuditContext} from './kernel-audit-context';

const GEOGRAPHIC_BAND = {coordinateSpace: 'longitude-latitude', metric: 'great-circle'} as const;

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

function parameters(
  {graph}: KernelAuditContext,
  id: string,
  format: 'float32' | 'uint32',
  length: number
) {
  return new GPUParameterBuffer(graph.device, {id, format, length});
}

function createStudyArea({view}: KernelAuditContext, cellCapacity: number) {
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

function hotSpotOptions(context: KernelAuditContext) {
  const {view} = context;
  return {
    settings: {significanceLevel: 0.01, seed: 3, permutationCount: 499},
    parameters: parameters(context, 'hot-spot-parameters', 'float32', 8),
    selfWeight: 1,
    minimumNeighbors: 8,
    multipleTesting: 'benjamini-yekutieli' as const,
    permutation: {
      parameters: parameters(context, 'hot-spot-permutation-parameters', 'uint32', 8),
      maximumPermutations: 999,
      statistic: 'localGStar' as const,
      alternative: 'two-sided' as const,
      maximumNeighbors: 64,
      multipleTesting: 'benjamini-hochberg' as const
    },
    color: {
      classBreaksParameters: view('float32', 64, 'class-breaks'),
      maximumClassCount: 16,
      methods: ALL_CLASS_METHODS,
      colorScaleParameters: view('float32', 64, 'color-scale'),
      palette: view('uint32', 64, 'palette'),
      maximumPaletteCount: 16
    }
  };
}

const HOT_SPOT_POINT_CASES: KernelAuditCase[] = [];
for (const family of ['quadbin', 'h3'] as const) {
  for (const analyze of ['density', 'count'] as const) {
    if (family === 'quadbin' && analyze === 'count') continue;
    HOT_SPOT_POINT_CASES.push({
      exportName: 'addHotSpotAnalysisRecipe',
      name: `${family} points source, ${analyze}, eight rings, every option`,
      build: context => {
        const {graph, view} = context;
        const cellCapacity = 2048;
        addHotSpotAnalysisRecipe(graph, {
          ...hotSpotOptions(context),
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
        } as never);
      }
    });
  }
}

const HOT_SPOT_LATTICE_CASES: KernelAuditCase[] = (['rook', 'queen'] as const).map(criterion => ({
  exportName: 'addHotSpotAnalysisRecipe',
  name: `lattice source, ${criterion}, every option`,
  build: context => {
    const {graph, view} = context;
    addHotSpotAnalysisRecipe(graph, {
      ...hotSpotOptions(context),
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
    } as never);
  }
}));

const RATE_CLUSTER_CASES: KernelAuditCase[] = [];
for (const analyze of ['standardized', 'smoothed'] as const) {
  for (const criterion of ['rook', 'queen'] as const) {
    RATE_CLUSTER_CASES.push({
      exportName: 'addRateClusterMapRecipe',
      name: `${analyze} rates, ${criterion}, permutations, adjustment, every output`,
      build: context => {
        const {graph, view} = context;
        const rows = 512;
        addRateClusterMapRecipe(graph, {
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
          parameters: parameters(context, 'rate-cluster-parameters', 'float32', 64),
          permutation: {
            parameters: parameters(context, 'rate-cluster-permutation-parameters', 'uint32', 8),
            maximumPermutations: 999,
            alternative: 'two-sided',
            maximumNeighbors: 64
          },
          palette: view('uint32', 5, 'palette')
        } as never);
      }
    });
  }
}

function addSpaceTime(
  context: KernelAuditContext,
  kind: 'lattice' | 'ids',
  step: 'year' | 'quarter' | 'month' | 'week' | 'day' | 'hour',
  cellCount: number,
  sliceCount: number
) {
  const {graph, view} = context;
  const width = 32;
  const height = cellCount / width;
  addSpaceTimeHotSpotsRecipe(graph, {
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
    parameters: parameters(context, 'space-time-parameters', 'float32', 64),
    multipleTesting: 'benjamini-hochberg',
    finalSlice: 'last-valid',
    color: {
      palette: view('uint32', 32, 'palette'),
      parameters: view('float32', 64, 'color-parameters'),
      maximumPaletteCount: 32
    }
  } as never);
}

const SPACE_TIME_CASES: KernelAuditCase[] = [];
for (const kind of ['lattice', 'ids'] as const) {
  for (const step of ['year', 'quarter', 'month', 'week', 'day', 'hour'] as const) {
    SPACE_TIME_CASES.push({
      exportName: 'addSpaceTimeHotSpotsRecipe',
      name: `${kind} cells, ${step} slices, every option`,
      build: context => addSpaceTime(context, kind, step, 256, 24)
    });
  }
}
SPACE_TIME_CASES.push({
  exportName: 'addSpaceTimeHotSpotsRecipe',
  name: 'largest cube (65536 cells, 255 slices)',
  build: context => addSpaceTime(context, 'lattice', 'day', 1 << 16, 255)
});

const REGRESSION_CASES: KernelAuditCase[] = [];
for (const predictorCount of [1, 8, 11, 15]) {
  for (const withLocalFits of [false, true]) {
    // The local fit supports at most 11 predictors, the global fit 15.
    if (withLocalFits && predictorCount > 11) continue;
    REGRESSION_CASES.push({
      exportName: 'addSpatialRegressionRecipe',
      name: `${predictorCount} predictors${withLocalFits ? ' with local fits' : ''}`,
      build: context => {
        const {graph, view} = context;
        const rows = 256;
        addSpatialRegressionRecipe(graph, {
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
          parameters: parameters(context, 'regression-parameters', 'float32', 128),
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
      }
    });
  }
}

const PERIOD_CASES: KernelAuditCase[] = [];
for (const family of ['quadbin', 'h3'] as const) {
  for (const classify of ['delta', 'ratio', 'percentChange', 'zScore'] as const) {
    const zScore = classify === 'zScore' ? 'poisson' : 'standardized';
    PERIOD_CASES.push({
      exportName: 'addPeriodComparisonRecipe',
      name: `${family} cells classified by ${classify}`,
      build: ({graph, view}) => {
        const period = (name: string) => ({
          ...(family === 'quadbin'
            ? {positions: view('float32x2', 2048, `${name}-positions`)}
            : {cells: view('uint32x2', 2048, `${name}-cells`)}),
          values: view('float32', 2048, `${name}-values`),
          mask: view('uint32', 2048, `${name}-mask`)
        });
        addPeriodComparisonRecipe(graph, {
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
      }
    });
  }
}

const CHOROPLETH_CASES: KernelAuditCase[] = [];
for (const backend of ['zonal', 'group'] as const) {
  for (const statistic of ['count', 'sum', 'mean', 'minimum', 'maximum'] as const) {
    CHOROPLETH_CASES.push({
      exportName: 'addPointsInPolygonsChoroplethRecipe',
      name: `${backend} backend, ${statistic}, with colors`,
      build: ({graph, view}) => {
        addPointsInPolygonsChoroplethRecipe(graph, {
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
      }
    });
  }
}

function createZones({view}: KernelAuditContext, features: number, crossingCapacity: number) {
  return {
    kind: 'polygons',
    positions: view('float32x2', features * 8, 'zone-positions'),
    featureOffsets: view('uint32', features + 1, 'zone-features'),
    polygonOffsets: view('uint32', features + 1, 'zone-polygons'),
    ringOffsets: view('uint32', features + 1, 'zone-rings'),
    crossingCapacity
  };
}

const CHANGE_OF_SUPPORT_CASES: KernelAuditCase[] = [
  undefined,
  'zone',
  'overlap',
  {extensive: 'overlap', intensive: 'zone'}
].map(denominator => ({
  exportName: 'addChangeOfSupportRecipe',
  name: `every option with denominator ${JSON.stringify(denominator)}`,
  build: context => {
    const {graph, view} = context;
    addChangeOfSupportRecipe(graph, {
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
  }
}));

const STRAIGHT_LINE_CASES: KernelAuditCase[] = [];
for (const mode of ['exact', 'jump-flood'] as const) {
  for (const sumOrder of ['atomic', 'sorted'] as const) {
    STRAIGHT_LINE_CASES.push({
      exportName: 'addStraightLineCatchmentsRecipe',
      name: `${mode} distances with ${sumOrder} sums`,
      build: ({graph, view}) => {
        addStraightLineCatchmentsRecipe(graph, {
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
      }
    });
  }
}

/** Kernel-audit cases for recipes. */
export const RECIPES_CASES: KernelAuditCase[] = [
  {
    exportName: 'addPointsInPolygonsChoroplethRecipe',
    name: 'prepared great-circle polygons',
    build: context => {
      const {graph, view} = context;
      addPointsInPolygonsChoroplethRecipe(graph, {
        spatialContext: {...GEOGRAPHIC_BAND, edgeModel: 'great-circle'},
        prepare: true,
        preparedPositionCapacity: 64,
        preparationParameters: view('float32', 4, 'preparation'),
        points: {kind: 'points', positions: view('float32x2', 8, 'points')},
        polygons: createStudyArea(context, 64)
      });
    }
  },
  {
    exportName: 'addChangeOfSupportRecipe',
    name: 'prepared source and target zones',
    build: context => {
      const {graph, view} = context;
      const zones = () => ({...createStudyArea(context, 64), crossingCapacity: 64});
      addChangeOfSupportRecipe(graph, {
        spatialContext: {...GEOGRAPHIC_BAND, edgeModel: 'great-circle'},
        prepare: true,
        preparedPositionCapacity: 64,
        preparationParameters: view('float32', 4, 'preparation'),
        grid: makeGPURasterMetadata({width: 4, height: 4}),
        parameters: view('float32', 4, 'raster-parameters'),
        source: zones(),
        target: zones(),
        pairCapacity: 8
      });
    }
  },
  {
    exportName: 'addHotSpotAnalysisRecipe',
    name: 'prepared great-circle study area',
    build: context => {
      const {graph, view} = context;
      addHotSpotAnalysisRecipe(graph, {
        spatialContext: {...GEOGRAPHIC_BAND, edgeModel: 'great-circle'},
        prepare: true,
        preparedPositionCapacity: 64,
        preparationParameters: view('float32', 4, 'preparation'),
        parameters: parameters(context, 'inference', 'float32', 8),
        source: {
          kind: 'points',
          positions: view('float32x2', 8, 'points'),
          family: 'quadbin',
          resolution: 4,
          studyArea: createStudyArea(context, 64),
          cellCapacity: 16
        }
      });
    }
  },
  ...HOT_SPOT_POINT_CASES,
  {
    exportName: 'addHotSpotAnalysisRecipe',
    name: 'points source with a great-circle distance band',
    build: context => {
      const {graph, view} = context;
      const cellCapacity = 1024;
      addHotSpotAnalysisRecipe(graph, {
        ...hotSpotOptions(context),
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
      } as never);
    }
  },
  ...HOT_SPOT_LATTICE_CASES,
  {
    // The default neighbor CSR holds cellCapacity * (disk size - 1) slots; 58000 is the largest
    // cell capacity at eight rings whose widest binding still fits one binding.
    exportName: 'addHotSpotAnalysisRecipe',
    name: 'quadbin points, largest cell capacity at eight rings',
    build: context => {
      const {graph, view} = context;
      const cellCapacity = 58_000;
      addHotSpotAnalysisRecipe(graph, {
        ...hotSpotOptions(context),
        source: {
          kind: 'points',
          positions: view('float32x2', 1_000_000, 'positions'),
          family: 'quadbin',
          resolution: 6,
          studyArea: createStudyArea(context, cellCapacity),
          cellCapacity,
          neighborRings: 8
        }
      } as never);
    }
  },
  ...RATE_CLUSTER_CASES,
  ...REGRESSION_CASES,
  ...SPACE_TIME_CASES,
  ...PERIOD_CASES,
  ...CHOROPLETH_CASES,
  {
    exportName: 'addClusterAndOutlineRecipe',
    name: 'largest cluster and hull capacities',
    build: ({graph, view}) => {
      addClusterAndOutlineRecipe(graph, {
        points: {kind: 'points', positions: view('float32x2', 4096, 'positions')},
        spatialContext: {coordinateSpace: 'planar', metric: 'native'} as never,
        clusteringParameters: view('float32', 64, 'clustering-parameters'),
        gridSize: [64, 64],
        geometryParameters: view('float32', 64, 'geometry-parameters'),
        maximumClusterCount: 256,
        maximumVerticesPerHull: 256,
        hullCapacity: 4096
      } as never);
    }
  },
  ...CHANGE_OF_SUPPORT_CASES,
  ...STRAIGHT_LINE_CASES,
  {
    exportName: 'addFleetDwellRecipe',
    name: 'stops, zones and the zone table',
    build: ({graph, view}) => {
      addFleetDwellRecipe(graph, {
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
    }
  },
  ...(['float32', 'uint32x2'] as const).map(
    (timestampFormat): KernelAuditCase => ({
      exportName: 'addFleetDwellZoneEventsRecipe',
      name: `${timestampFormat} timestamps`,
      build: ({graph, view}) => {
        addFleetDwellZoneEventsRecipe(graph, {
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
      }
    })
  )
];
