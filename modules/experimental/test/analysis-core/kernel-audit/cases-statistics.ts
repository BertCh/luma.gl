// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUGridIndexView} from '@luma.gl/gpgpu/gpu-core';
import {
  GPUAssuncaoReisStandardization,
  GPUCellTableCompare,
  GPUClassAssignment,
  GPUClassificationFit,
  GPUContiguityWeights,
  GPUEmergingHotSpots,
  GPUEmpiricalBayesRates,
  GPUGammaStatistic,
  GPUGeographicallyWeightedRegression,
  GPUGeographicallyWeightedRegressionNonstationarityTest,
  GPUGlobalPermutationTest,
  GPUGlobalSpatialStatistics,
  GPUHotSpotAnalysis,
  GPUKnoxTest,
  GPULISAMarkov,
  GPULatticeWeights,
  GPULocalGeary,
  GPULocalMoran,
  GPULocalPermutationTest,
  GPUMantelTest,
  GPUMultipleTestingAdjustment,
  GPUNeighborhoodSummary,
  GPUOrdinaryLeastSquares,
  GPUPickRegionMask,
  GPURegionMask,
  GPURegionStatistics,
  GPUSpatialEmpiricalBayesRates,
  GPUSpatialErrorGM,
  GPUSpatialErrorMaximumLikelihood,
  GPUSpatialLag,
  GPUSpatialLagImpacts,
  GPUSpatialLagMaximumLikelihood,
  GPUSpatialMarkov,
  GPUSpatialPearson,
  GPUSpatialRegressionDiagnostics,
  GPUSpatialScanStatistic,
  GPUSpatialTwoStageLeastSquares,
  GPUSpatialWeightsAlgebra,
  GPUSpatialWeightsSummary,
  GPUSpatialWeightsTraces,
  GPUSpatialWeightsTranspose,
  GPUSpatialWeightsTransform,
  GPUSpatialWeightsValidation,
  GPUTransitionDiagnostics,
  GPUTransitionMatrix,
  GPUZonalStatistics,
  getGPUGeographicallyWeightedRegressionParameterLength,
  GPU_CLASSIFICATION_FIT_SUMMARY_LENGTH,
  GPU_EMERGING_HOT_SPOTS_MAXIMUM_RADIUS,
  GPU_EMERGING_HOT_SPOTS_MAXIMUM_SLICE_COUNT,
  GPU_EMERGING_HOT_SPOTS_PARAMETER_LENGTH,
  GPU_EMERGING_HOT_SPOTS_STATISTICS_LENGTH,
  GPU_EMPIRICAL_BAYES_SUMMARY,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_CANDIDATE_TABLE_STRIDE,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_INDEXED_ROW_COUNT,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_LADDER_LENGTH,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_NEIGHBOR_COUNT,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_PREDICTOR_COUNT,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_REFINEMENT_PASS_COUNT,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_SEARCH_ITERATIONS,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_RECOMMENDED_GAUSSIAN_CUTOFF,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SEARCH_STATE_LENGTH,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY_LENGTH,
  GPU_GLOBAL_PERMUTATION_SUMMARY,
  GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT,
  GPU_GWR_NONSTATIONARITY_MAXIMUM_PREDICTOR_COUNT,
  GPU_LATTICE_WEIGHTS_MAXIMUM_RADIUS,
  GPU_LOCAL_PERMUTATION_MAXIMUM_NEIGHBORS,
  GPU_MULTIPLE_TESTING_MAXIMUM_ROWS,
  GPU_NEIGHBORHOOD_SUMMARY_MAXIMUM_NEIGHBORS,
  GPU_ORDINARY_LEAST_SQUARES_PARAMETER_LENGTH,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH,
  GPU_PERMUTATION_PARAMETER_LENGTH,
  GPU_RATE_STANDARDIZATION_STATISTICS_LENGTH,
  GPU_SPACE_TIME_PARAMETER_LENGTH,
  GPU_SPACE_TIME_SUMMARY_LENGTH,
  GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH,
  GPU_SPATIAL_AUTOCORRELATION_STATISTICS_LENGTH,
  GPU_SPATIAL_ERROR_GM_MAXIMUM_PREDICTOR_COUNT,
  GPU_SPATIAL_ERROR_GM_SUMMARY_LENGTH,
  GPU_SPATIAL_ERROR_MAXIMUM_LIKELIHOOD_MAXIMUM_PREDICTOR_COUNT,
  GPU_SPATIAL_LAG_MAXIMUM_LIKELIHOOD_MAXIMUM_PREDICTOR_COUNT,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LENGTH,
  getGPUSpatialRegressionDiagnosticsMaximumRowTableValueCount,
  GPU_SPATIAL_REGRESSION_DIAGNOSTICS_SUMMARY_LENGTH,
  GPU_SPATIAL_REGRESSION_DIAGNOSTICS_TESTS_LENGTH,
  GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_MAXIMUM_PREDICTOR_COUNT,
  GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_LENGTH,
  GPU_SPATIAL_WEIGHTS_MAXIMUM_ORDER,
  GPU_SPATIAL_WEIGHTS_SUMMARY_LAYOUT,
  GPU_SPATIAL_WEIGHTS_TRACES_MAXIMUM_POWER,
  GPU_TRANSITION_HOMOGENEITY_LENGTH,
  GPU_TRANSITION_STATE_SPARSITY_LENGTH,
  type GPUGammaOperation,
  type GPUGlobalPermutationStatistic,
  type GPUGlobalSpatialStatistic,
  type GPULocalPermutationStatistic,
  type GPUMultipleTestingMethod,
  type GPUNeighborhoodSummaryStatistic,
  type GPURegionSelection,
  type GPURegionStatisticsProps,
  type GPUSpatialScanModel,
  type GPUSpatialWeights,
  type GPUSpatialWeightsKernel,
  type GPUSpatialWeightsTransformOperation,
  type GPUZonalStatisticsExtentStatistic,
  type GPUZonalStatisticsProps
} from '../../../src/gpu-spatial-analysis';
import type {KernelAuditCase, KernelAuditContext} from './kernel-audit-context';

type View = KernelAuditContext['view'];

const ROWS = 300;
const SLOTS = 8;
const PERIODS = 5;
const METHODS: GPUMultipleTestingMethod[] = [
  'none',
  'bonferroni',
  'holm',
  'benjamini-hochberg',
  'benjamini-yekutieli'
];

/** Weights with an explicit neighbor-slot capacity. */
function capacityWeights(
  view: View,
  rows: number,
  capacity: number,
  distances = true
): GPUSpatialWeights {
  return {
    offsets: view('uint32', rows + 1),
    neighbors: view('uint32', capacity),
    weights: view('float32', capacity),
    distances: distances ? view('float32', capacity) : undefined
  };
}

function transitionOutput(view: View, classCount: number, conditionCount: number) {
  const chains = conditionCount > 1 ? conditionCount + 1 : 1;
  return {
    homogeneity:
      conditionCount > 1 ? view('float32', GPU_TRANSITION_HOMOGENEITY_LENGTH, 'hom') : undefined,
    steadyStates: view('float32', chains * classCount, 'steady'),
    stateSparsity: view('uint32', GPU_TRANSITION_STATE_SPARSITY_LENGTH, 'sparsity'),
    status: {converged: view('uint32', 1, 'converged')},
    counts: view('uint32', conditionCount * classCount * classCount, 'counts'),
    probabilities: view('float32', conditionCount * classCount * classCount, 'probabilities'),
    rowTotals: view('uint32', conditionCount * classCount, 'row-totals'),
    ignored: view('uint32', 1, 'ignored')
  };
}

// ---------------------------------------------------------------------------------------------
// Cell table compare, distribution dynamics
// ---------------------------------------------------------------------------------------------

function buildCompare(
  {graph, view}: KernelAuditContext,
  measure: 'count' | 'sum',
  zScore: 'poisson' | 'standardized',
  capacity: number
) {
  const table = (cells: number) => ({
    cells: view('uint32x2', cells),
    counts: view('uint32', cells),
    sums: view('uint32x2', cells),
    status: {count: view('uint32', 1), overflow: view('uint32', 1)}
  });
  graph.add(
    new GPUCellTableCompare({
      before: table(capacity),
      after: table(capacity),
      measure,
      zScore,
      sumScale: 1024,
      output: {
        cells: view('uint32x2', capacity),
        presence: view('uint32', capacity),
        mask: view('uint32', capacity),
        before: view('float32', capacity),
        after: view('float32', capacity),
        delta: view('float32', capacity),
        ratio: view('float32', capacity),
        percentChange: view('float32', capacity),
        zScore: view('float32', capacity),
        status: {
          count: view('uint32', 1),
          overflow: view('uint32', 1),
          requiredCount: view('uint32', 1)
        }
      }
    })
  );
}

const COMPARE_CASES = (
  [
    ['count', 'poisson'],
    ['count', 'standardized'],
    ['sum', 'poisson'],
    ['sum', 'standardized']
  ] as const
).flatMap(([measure, zScore]): KernelAuditCase[] => [
  {
    exportName: 'GPUCellTableCompare',
    name: `${measure}/${zScore} every output`,
    build: context => buildCompare(context, measure, zScore, 1000)
  },
  {
    exportName: 'GPUCellTableCompare',
    name: `${measure}/${zScore} at 2^24 cells`,
    build: context => buildCompare(context, measure, zScore, 2 ** 24)
  }
]);

function buildClassAssignment({graph, view}: KernelAuditContext, rows: number) {
  graph.add(
    new GPUClassAssignment({
      values: view('float32', rows),
      breaks: view('float32', 65),
      classCount: view('uint32', 1),
      mask: view('uint32', rows),
      output: view('uint32', rows)
    })
  );
}

function buildClassificationFit({graph, view}: KernelAuditContext, rows: number) {
  const classCount = 256;
  graph.add(
    new GPUClassificationFit({
      values: view('float32', rows),
      classes: view('uint32', rows),
      classCount,
      mask: view('uint32', rows),
      output: {
        counts: view('uint32', classCount + 1),
        medians: view('float32', classCount + 1),
        absoluteDeviations: view('float32', classCount + 1),
        squaredDeviations: view('float32', classCount + 1),
        summary: view('float32', GPU_CLASSIFICATION_FIT_SUMMARY_LENGTH)
      }
    })
  );
}

const TRANSITION_SHAPES = [
  [64, 1],
  [64, 8],
  [10, 4]
] as const;

function buildTransitionMatrix(
  {graph, view}: KernelAuditContext,
  rows: number,
  classCount: number,
  conditionCount: number
) {
  graph.add(
    new GPUTransitionMatrix({
      classes: view('uint32', rows * PERIODS),
      periodCount: PERIODS,
      classCount,
      periodLag: 2,
      ...(conditionCount > 1
        ? {conditionClasses: view('uint32', rows * PERIODS), conditionCount}
        : {}),
      mask: view('uint32', rows),
      minimumTransitions: 7,
      output: transitionOutput(view, classCount, conditionCount)
    })
  );
}

function buildSpatialMarkov(
  {graph, view, weights}: KernelAuditContext,
  rows: number,
  classCount: number,
  lagMaximumClassCount: number,
  normalize: boolean
) {
  graph.add(
    new GPUSpatialMarkov({
      values: view('float32', rows * PERIODS),
      classes: view('uint32', rows * PERIODS),
      classCount,
      weights: weights(rows, SLOTS),
      periodCount: PERIODS,
      periodLag: 2,
      normalize,
      maximumLagClassCount: lagMaximumClassCount,
      lagParameters: view('float32', 16 + lagMaximumClassCount),
      lagMethods: ['quantile', 'equal-interval', 'natural-breaks', 'box-plot'],
      mask: view('uint32', rows),
      minimumTransitions: 7,
      output: {
        ...transitionOutput(view, classCount, lagMaximumClassCount),
        lagBreaks: view('float32', lagMaximumClassCount + 1),
        lagClassCount: view('uint32', 1),
        lagValues: view('float32', rows * PERIODS),
        lagClasses: view('uint32', rows * PERIODS)
      }
    })
  );
}

function buildLisaMarkov(
  {graph, view, weights}: KernelAuditContext,
  rows: number,
  gating: 'none' | 'significance'
) {
  graph.add(
    new GPULISAMarkov({
      values: view('float32', rows * PERIODS),
      weights: {...weights(rows, SLOTS), distances: view('float32', rows * SLOTS)},
      periodCount: PERIODS,
      periodLag: 2,
      parameters: view('float32', GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH),
      mask: view('uint32', rows),
      gating,
      minimumTransitions: 7,
      output: {
        ...transitionOutput(view, 5, 1),
        quadrants: view('uint32', rows * PERIODS)
      }
    })
  );
}

const DISTRIBUTION_CASES: KernelAuditCase[] = [
  {
    exportName: 'GPUClassAssignment',
    name: 'mask',
    build: context => buildClassAssignment(context, ROWS)
  },
  {
    exportName: 'GPUClassAssignment',
    name: 'rows at the binding limit',
    build: context => buildClassAssignment(context, context.maximumRows(4))
  },
  {
    exportName: 'GPUClassificationFit',
    name: '256 classes every output',
    build: context => buildClassificationFit(context, ROWS)
  },
  {
    exportName: 'GPUClassificationFit',
    name: '256 classes rows at the binding limit',
    build: context => buildClassificationFit(context, context.maximumRows(4))
  },
  ...TRANSITION_SHAPES.flatMap(([classCount, conditionCount]): KernelAuditCase[] => [
    {
      exportName: 'GPUTransitionMatrix',
      name: `${classCount} classes x ${conditionCount} conditions`,
      build: context => buildTransitionMatrix(context, 200, classCount, conditionCount)
    },
    {
      exportName: 'GPUTransitionMatrix',
      name: `${classCount} classes x ${conditionCount} conditions at the binding limit`,
      build: context =>
        buildTransitionMatrix(
          context,
          Math.floor(context.maximumRows(4) / PERIODS),
          classCount,
          conditionCount
        )
    },
    {
      exportName: 'GPUTransitionDiagnostics',
      name: `${classCount} classes x ${conditionCount} conditions`,
      build: ({graph, view}) =>
        graph.add(
          new GPUTransitionDiagnostics({
            counts: view('uint32', conditionCount * classCount * classCount),
            classCount,
            conditionCount,
            minimumTransitions: 7,
            output: {
              homogeneity:
                conditionCount > 1
                  ? view('float32', GPU_TRANSITION_HOMOGENEITY_LENGTH, 'hom')
                  : undefined,
              steadyStates: view(
                'float32',
                (conditionCount > 1 ? conditionCount + 1 : 1) * classCount,
                'steady'
              ),
              stateSparsity: view('uint32', GPU_TRANSITION_STATE_SPARSITY_LENGTH, 'sparsity'),
              status: {converged: view('uint32', 1, 'converged')}
            }
          })
        )
    }
  ]),
  ...(['none', 'significance'] as const).flatMap((gating): KernelAuditCase[] => [
    {
      exportName: 'GPULISAMarkov',
      name: `gating ${gating} every output`,
      build: context => buildLisaMarkov(context, 200, gating)
    },
    {
      exportName: 'GPULISAMarkov',
      name: `gating ${gating} rows at the binding limit`,
      build: context => buildLisaMarkov(context, Math.floor(context.maximumRows(4) / SLOTS), gating)
    }
  ]),
  ...(
    [
      [16, 16, true],
      [64, 8, false],
      [5, 6, true]
    ] as const
  ).flatMap(([classCount, lagCount, normalize]): KernelAuditCase[] => [
    {
      exportName: 'GPUSpatialMarkov',
      name: `${classCount} classes x ${lagCount} lag classes`,
      build: context => buildSpatialMarkov(context, 200, classCount, lagCount, normalize)
    },
    {
      exportName: 'GPUSpatialMarkov',
      name: `${classCount} classes x ${lagCount} lag classes at the binding limit`,
      build: context =>
        buildSpatialMarkov(
          context,
          Math.min(Math.floor(context.maximumRows(4) / SLOTS), Math.floor(2 ** 24 / PERIODS)),
          classCount,
          lagCount,
          normalize
        )
    }
  ])
];

// ---------------------------------------------------------------------------------------------
// Emerging hot spots, global statistics, multiple testing, neighborhood summary
// ---------------------------------------------------------------------------------------------

type EmergingVariant = {
  gridWidth: number;
  gridHeight: number;
  sliceCount: number;
  multipleTesting: GPUMultipleTestingMethod;
  finalSlice: 'require' | 'last-valid';
  valueFormat: 'float32' | 'uint32';
  weightsMode?: boolean;
  maximumRadius?: number;
  withoutPValues?: boolean;
};

function buildEmerging({graph, view}: KernelAuditContext, variant: EmergingVariant) {
  const cells = variant.gridWidth * variant.gridHeight;
  const bins = cells * variant.sliceCount;
  const spatial = variant.weightsMode
    ? {
        weights: {
          offsets: view('uint32', cells + 1),
          neighbors: view('uint32', cells * 8),
          weights: view('float32', cells * 8)
        },
        selfWeight: 1
      }
    : {gridWidth: variant.gridWidth, gridHeight: variant.gridHeight};
  graph.add(
    new GPUEmergingHotSpots({
      values: variant.valueFormat === 'float32' ? view('float32', bins) : view('uint32', bins),
      ...spatial,
      sliceCount: variant.sliceCount,
      maximumRadius: variant.maximumRadius,
      mask: view('uint32', cells),
      parameters: view('float32', GPU_EMERGING_HOT_SPOTS_PARAMETER_LENGTH),
      multipleTesting: variant.multipleTesting,
      finalSlice: variant.finalSlice,
      output: {
        giZScores: view('float32', bins),
        pValues: variant.withoutPValues ? undefined : view('float32', bins),
        adjustedPValues: variant.withoutPValues ? undefined : view('float32', bins),
        lastValidSlice: view('uint32', cells),
        trendZ: view('float32', cells),
        trendP: view('float32', cells),
        trendS: view('sint32', cells),
        category: view('uint32', cells),
        hotSliceCount: view('uint32', cells),
        coldSliceCount: view('uint32', cells),
        globalStatistics: view('float32', GPU_EMERGING_HOT_SPOTS_STATISTICS_LENGTH)
      }
    })
  );
}

const EMERGING_CASES = METHODS.flatMap((multipleTesting): KernelAuditCase[] => [
  ...(['require', 'last-valid'] as const).map(
    (finalSlice): KernelAuditCase => ({
      exportName: 'GPUEmergingHotSpots',
      name: `lattice ${multipleTesting}/${finalSlice} maximal slices and radius`,
      build: context =>
        buildEmerging(context, {
          gridWidth: 8,
          gridHeight: 8,
          sliceCount: GPU_EMERGING_HOT_SPOTS_MAXIMUM_SLICE_COUNT,
          maximumRadius: GPU_EMERGING_HOT_SPOTS_MAXIMUM_RADIUS,
          multipleTesting,
          finalSlice,
          valueFormat: 'float32'
        })
    })
  ),
  {
    exportName: 'GPUEmergingHotSpots',
    name: `uint32 lattice ${multipleTesting}`,
    build: context =>
      buildEmerging(context, {
        gridWidth: 8,
        gridHeight: 8,
        sliceCount: GPU_EMERGING_HOT_SPOTS_MAXIMUM_SLICE_COUNT,
        maximumRadius: GPU_EMERGING_HOT_SPOTS_MAXIMUM_RADIUS,
        multipleTesting,
        finalSlice: 'last-valid',
        valueFormat: 'uint32'
      })
  },
  {
    exportName: 'GPUEmergingHotSpots',
    name: `weights mode ${multipleTesting} maximal slices`,
    build: context =>
      buildEmerging(context, {
        gridWidth: 64,
        gridHeight: 1,
        sliceCount: GPU_EMERGING_HOT_SPOTS_MAXIMUM_SLICE_COUNT,
        multipleTesting,
        finalSlice: 'last-valid',
        valueFormat: 'float32',
        weightsMode: true
      })
  },
  {
    exportName: 'GPUEmergingHotSpots',
    name: `${multipleTesting} just below 2^24 bins`,
    build: context =>
      buildEmerging(context, {
        gridWidth: 2048,
        gridHeight: 2047,
        sliceCount: 4,
        multipleTesting,
        finalSlice: 'require',
        valueFormat: 'float32'
      })
  }
]).concat({
  exportName: 'GPUEmergingHotSpots',
  name: 'none at 2^25 bins without p-values',
  build: context =>
    buildEmerging(context, {
      gridWidth: 2048,
      gridHeight: 2048,
      sliceCount: 8,
      multipleTesting: 'none',
      finalSlice: 'require',
      valueFormat: 'float32',
      withoutPValues: true
    })
});

const GLOBAL_STATISTICS: GPUGlobalSpatialStatistic[] = [
  'moran',
  'geary',
  'getisOrdG',
  'bivariateMoran',
  'joinCount'
];
const GLOBAL_MAXIMUM_ROWS = Math.floor(134217728 / (27 * 4));

function buildGlobalStatistics(
  {graph, view}: KernelAuditContext,
  rows: number,
  capacity: number,
  islands: 'keep' | 'exclude',
  alternative: 'greater' | 'lesser' | 'two-sided',
  statistics: GPUGlobalSpatialStatistic[] = GLOBAL_STATISTICS
) {
  graph.add(
    new GPUGlobalSpatialStatistics({
      weights: {
        offsets: view('uint32', rows + 1),
        neighbors: view('uint32', capacity),
        weights: view('float32', capacity),
        distances: view('float32', capacity),
        status: {overflow: view('uint32', 1)}
      },
      values: view('float32', rows),
      secondValues: view('float32', rows),
      mask: view('uint32', rows),
      statistics,
      islands,
      alternative,
      output: {
        summary: view('float32', GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.length),
        joinCounts: view('uint32', 3)
      }
    })
  );
}

const GLOBAL_CASES: KernelAuditCase[] = [
  ...(['exclude', 'keep'] as const).flatMap(islands =>
    (['greater', 'lesser', 'two-sided'] as const).map(
      (alternative): KernelAuditCase => ({
        exportName: 'GPUGlobalSpatialStatistics',
        name: `all statistics islands ${islands} ${alternative}`,
        build: context => buildGlobalStatistics(context, 500, 4000, islands, alternative)
      })
    )
  ),
  ...GLOBAL_STATISTICS.map(
    (statistic): KernelAuditCase => ({
      exportName: 'GPUGlobalSpatialStatistics',
      name: `only ${statistic}`,
      build: context => buildGlobalStatistics(context, 500, 4000, 'keep', 'two-sided', [statistic])
    })
  ),
  {
    exportName: 'GPUGlobalSpatialStatistics',
    name: 'documented row maximum',
    build: context =>
      buildGlobalStatistics(
        context,
        GLOBAL_MAXIMUM_ROWS,
        8 * GLOBAL_MAXIMUM_ROWS,
        'exclude',
        'two-sided'
      )
  },
  {
    exportName: 'GPUGlobalSpatialStatistics',
    name: 'neighbor capacity of one binding',
    build: context => buildGlobalStatistics(context, 100000, 2 ** 25, 'keep', 'two-sided')
  }
];

const MULTIPLE_TESTING_CASES = METHODS.flatMap((method): KernelAuditCase[] =>
  [70000, GPU_MULTIPLE_TESTING_MAXIMUM_ROWS - 1].map(rows => ({
    exportName: 'GPUMultipleTestingAdjustment',
    name: `${method} ${rows} rows`,
    build: ({graph, view}) =>
      graph.add(
        new GPUMultipleTestingAdjustment({
          pValues: view('float32', rows),
          method,
          minimumFamilySize: GPU_MULTIPLE_TESTING_MAXIMUM_ROWS - 1,
          output: {
            adjustedPValues: view('float32', rows),
            testedCount: view('uint32', 1)
          }
        })
      )
  }))
);

const ALL_NEIGHBORHOOD_STATISTICS: GPUNeighborhoodSummaryStatistic[] = [
  'count',
  'weightSum',
  'sum',
  'mean',
  'min',
  'max',
  'standardDeviation',
  'median'
];

function buildNeighborhood(
  {graph, view}: KernelAuditContext,
  rows: number,
  slots: number,
  maximumNeighbors: number,
  statistics: GPUNeighborhoodSummaryStatistic[] = ALL_NEIGHBORHOOD_STATISTICS
) {
  graph.add(
    new GPUNeighborhoodSummary({
      weights: {
        offsets: view('uint32', rows + 1),
        neighbors: view('uint32', slots),
        weights: view('float32', slots)
      },
      values: view('float32', rows),
      categories: view('uint32', rows),
      mask: view('uint32', rows),
      includeFocal: true,
      focalWeight: 0.5,
      statistics,
      maximumNeighbors,
      output: {
        table: view('float32', rows * statistics.length),
        modes: view('uint32', rows),
        entropy: view('float32', rows),
        status: {overflow: view('uint32', 1)}
      }
    })
  );
}

const NEIGHBORHOOD_CASES: KernelAuditCase[] = [
  ...[1, 32, GPU_NEIGHBORHOOD_SUMMARY_MAXIMUM_NEIGHBORS].map(
    (maximumNeighbors): KernelAuditCase => ({
      exportName: 'GPUNeighborhoodSummary',
      name: `every statistic maximumNeighbors ${maximumNeighbors}`,
      build: context => buildNeighborhood(context, 100, 800, maximumNeighbors)
    })
  ),
  ...ALL_NEIGHBORHOOD_STATISTICS.map(
    (statistic): KernelAuditCase => ({
      exportName: 'GPUNeighborhoodSummary',
      name: `only ${statistic} at maximum neighbors`,
      build: context =>
        buildNeighborhood(context, 100, 800, GPU_NEIGHBORHOOD_SUMMARY_MAXIMUM_NEIGHBORS, [
          statistic
        ])
    })
  ),
  {
    exportName: 'GPUNeighborhoodSummary',
    name: 'large row count',
    build: context =>
      buildNeighborhood(context, 1_000_000, 8_000_000, GPU_NEIGHBORHOOD_SUMMARY_MAXIMUM_NEIGHBORS)
  },
  {
    exportName: 'GPUNeighborhoodSummary',
    name: 'rows at the binding limit',
    build: context => {
      const rows = Math.floor(context.maximumRows(4 * ALL_NEIGHBORHOOD_STATISTICS.length));
      buildNeighborhood(
        context,
        rows,
        Math.min(rows * 8, context.maximumRows(4)),
        GPU_NEIGHBORHOOD_SUMMARY_MAXIMUM_NEIGHBORS
      );
    }
  }
];

// ---------------------------------------------------------------------------------------------
// Permutation inference
// ---------------------------------------------------------------------------------------------

const LOCAL_STATISTICS: GPULocalPermutationStatistic[] = [
  'localMoran',
  'localGeary',
  'localG',
  'localGStar',
  'localBivariateMoran',
  'localLeeL',
  'localJoinCount',
  'localBivariateJoinCount'
];

function buildLocalPermutation(
  {graph, view, weights}: KernelAuditContext,
  statistic: GPULocalPermutationStatistic,
  rows: number,
  maximumPermutations: number,
  overrides: {
    joinCase?: 'BJC' | 'CLC';
    multipleTesting?: 'bonferroni' | 'holm' | 'benjamini-hochberg' | 'benjamini-yekutieli';
    islands?: 'keep' | 'exclude';
    stopAfterExceedances?: number;
  } = {}
) {
  const bivariate = ['localBivariateMoran', 'localLeeL', 'localBivariateJoinCount'].includes(
    statistic
  );
  graph.add(
    new GPULocalPermutationTest({
      weights: weights(rows, SLOTS),
      values: view('float32', rows),
      mask: view('uint32', rows),
      parameters: view('uint32', GPU_PERMUTATION_PARAMETER_LENGTH),
      maximumPermutations,
      id: `local-${statistic}`,
      statistic,
      secondValues: bivariate ? view('float32', rows) : undefined,
      joinCase: statistic === 'localBivariateJoinCount' ? (overrides.joinCase ?? 'BJC') : undefined,
      maximumNeighbors: GPU_LOCAL_PERMUTATION_MAXIMUM_NEIGHBORS,
      islands: overrides.islands ?? 'keep',
      alternative: 'two-sided',
      multipleTesting: overrides.multipleTesting ?? 'benjamini-hochberg',
      stopAfterExceedances: overrides.stopAfterExceedances,
      output: {
        exceedances: view('uint32', rows),
        permutationCounts:
          overrides.stopAfterExceedances === undefined ? undefined : view('uint32', rows),
        pseudoPValues: view('float32', rows),
        observed: view('float32', rows),
        significant: view('uint32', rows),
        adjustedPValues: view('float32', rows),
        islandCount: view('uint32', 1),
        status: {overflow: view('uint32', 1)},
        classes: statistic === 'localGeary' ? view('uint32', rows) : undefined,
        quadrants: statistic === 'localBivariateMoran' ? view('uint32', rows) : undefined
      }
    })
  );
}

const LOCAL_PERMUTATION_CASES: KernelAuditCase[] = [
  ...LOCAL_STATISTICS.flatMap((statistic): KernelAuditCase[] => [
    {
      exportName: 'GPULocalPermutationTest',
      name: `${statistic} 999 permutations every output`,
      build: context => buildLocalPermutation(context, statistic, 100, 999)
    },
    {
      exportName: 'GPULocalPermutationTest',
      name: `${statistic} 2^20 permutations exclude islands`,
      build: context =>
        buildLocalPermutation(context, statistic, 100, 2 ** 20, {islands: 'exclude'})
    },
    {
      exportName: 'GPULocalPermutationTest',
      name: `${statistic} 2^20 permutations stopAfterExceedances every output`,
      build: context =>
        buildLocalPermutation(context, statistic, 100, 2 ** 20, {stopAfterExceedances: 20})
    }
  ]),
  ...(['BJC', 'CLC'] as const).map(
    (joinCase): KernelAuditCase => ({
      exportName: 'GPULocalPermutationTest',
      name: `localBivariateJoinCount ${joinCase}`,
      build: context =>
        buildLocalPermutation(context, 'localBivariateJoinCount', 100, 999, {joinCase})
    })
  ),
  ...(['bonferroni', 'holm', 'benjamini-hochberg', 'benjamini-yekutieli'] as const).map(
    (multipleTesting): KernelAuditCase => ({
      exportName: 'GPULocalPermutationTest',
      name: `localMoran ${multipleTesting}`,
      build: context => buildLocalPermutation(context, 'localMoran', 100, 999, {multipleTesting})
    })
  ),
  {
    exportName: 'GPULocalPermutationTest',
    name: 'localMoran rows at the binding limit',
    build: context =>
      buildLocalPermutation(context, 'localMoran', context.maximumRows(4 * SLOTS), 2 ** 20)
  }
];

const GLOBAL_PERMUTATION_STATISTICS: GPUGlobalPermutationStatistic[] = [
  'moran',
  'geary',
  'getisOrdG',
  'bivariateMoran',
  'leesL'
];

function buildGlobalPermutation(
  {graph, view, weights}: KernelAuditContext,
  statistic: GPUGlobalPermutationStatistic,
  rows: number,
  maximumPermutations: number,
  islands: 'keep' | 'exclude',
  alternative: 'greater' | 'lesser' | 'two-sided' = 'two-sided',
  stopAfterExceedances?: number
) {
  const needsSecond = statistic === 'bivariateMoran' || statistic === 'leesL';
  graph.add(
    new GPUGlobalPermutationTest({
      weights: weights(rows, SLOTS),
      values: view('float32', rows),
      mask: view('uint32', rows),
      parameters: view('uint32', GPU_PERMUTATION_PARAMETER_LENGTH),
      maximumPermutations,
      id: `global-${statistic}`,
      statistic,
      secondValues: needsSecond ? view('float32', rows) : undefined,
      alternative,
      islands,
      stopAfterExceedances,
      output: {
        summary: view('float32', GPU_GLOBAL_PERMUTATION_SUMMARY.length),
        referenceDistribution: view('float32', maximumPermutations),
        histogram: view('uint32', 64)
      }
    })
  );
}

const GLOBAL_PERMUTATION_CASES = GLOBAL_PERMUTATION_STATISTICS.flatMap(
  (statistic): KernelAuditCase[] => [
    ...(['greater', 'lesser', 'two-sided'] as const).map(
      (alternative): KernelAuditCase => ({
        exportName: 'GPUGlobalPermutationTest',
        name: `${statistic} 999 permutations ${alternative}`,
        build: context => buildGlobalPermutation(context, statistic, 100, 999, 'keep', alternative)
      })
    ),
    {
      exportName: 'GPUGlobalPermutationTest',
      name: `${statistic} 2^20 permutations exclude islands`,
      build: context => buildGlobalPermutation(context, statistic, 100, 2 ** 20, 'exclude')
    },
    {
      exportName: 'GPUGlobalPermutationTest',
      name: `${statistic} rows at the binding limit 999 permutations`,
      build: context =>
        buildGlobalPermutation(context, statistic, context.maximumRows(4 * SLOTS), 999, 'keep')
    },
    {
      exportName: 'GPUGlobalPermutationTest',
      name: `${statistic} 2^20 permutations at 15 row blocks`,
      build: context => buildGlobalPermutation(context, statistic, 15 * 2048, 2 ** 20, 'keep')
    },
    {
      exportName: 'GPUGlobalPermutationTest',
      name: `${statistic} 2^20 permutations at 15 row blocks stopAfterExceedances`,
      build: context =>
        buildGlobalPermutation(context, statistic, 15 * 2048, 2 ** 20, 'keep', 'two-sided', 20)
    },
    {
      exportName: 'GPUGlobalPermutationTest',
      name: `${statistic} 999 permutations stopAfterExceedances`,
      build: context => buildGlobalPermutation(context, statistic, 100, 999, 'keep', 'greater', 10)
    },
    {
      exportName: 'GPUGlobalPermutationTest',
      name: `${statistic} permutations x row blocks at the maximum`,
      // 798915 sums x 21 row blocks is exactly 2^24 - 1, the largest below 2^32 invocations.
      build: context => buildGlobalPermutation(context, statistic, 21 * 2048, 798914, 'keep')
    },
    {
      exportName: 'GPUGlobalPermutationTest',
      name: `${statistic} permutations x row blocks above the maximum`,
      build: context => buildGlobalPermutation(context, statistic, 16 * 2048, 2 ** 20, 'keep')
    }
  ]
);

// ---------------------------------------------------------------------------------------------
// Rate smoothing
// ---------------------------------------------------------------------------------------------

const RATE_SMOOTHING_CASES: KernelAuditCase[] = [
  ...[false, true].map(
    (clampPriorVariance): KernelAuditCase => ({
      exportName: 'GPUEmpiricalBayesRates',
      name: `every output, clamp ${clampPriorVariance}`,
      build: ({graph, view}) =>
        graph.add(
          new GPUEmpiricalBayesRates({
            events: view('uint32', ROWS),
            populations: view('float32', ROWS),
            mask: view('uint32', ROWS),
            clampPriorVariance,
            output: {
              standardizedRates: view('float32', ROWS),
              smoothedRates: view('float32', ROWS),
              rawRates: view('float32', ROWS),
              summary: view('float32', GPU_EMPIRICAL_BAYES_SUMMARY.length)
            }
          })
        )
    })
  ),
  {
    exportName: 'GPUEmpiricalBayesRates',
    name: 'rows at the binding limit',
    build: ({graph, view, maximumRows}) => {
      const rows = maximumRows(4);
      graph.add(
        new GPUEmpiricalBayesRates({
          events: view('float32', rows),
          populations: view('float32', rows),
          output: {
            smoothedRates: view('float32', rows),
            summary: view('float32', GPU_EMPIRICAL_BAYES_SUMMARY.length)
          }
        })
      );
    }
  },
  {
    exportName: 'GPUSpatialEmpiricalBayesRates',
    name: 'every output',
    build: ({graph, view, weights}) =>
      graph.add(
        new GPUSpatialEmpiricalBayesRates({
          events: view('float32', ROWS),
          populations: view('uint32', ROWS),
          weights: weights(ROWS, 6),
          mask: view('uint32', ROWS),
          output: {
            spatialRates: view('float32', ROWS),
            smoothedRates: view('float32', ROWS),
            priorMeans: view('float32', ROWS),
            priorVariances: view('float32', ROWS)
          }
        })
      )
  },
  {
    exportName: 'GPUSpatialEmpiricalBayesRates',
    name: 'rows at the binding limit',
    build: ({graph, view, weights, maximumRows}) => {
      const rows = maximumRows(4 * SLOTS);
      graph.add(
        new GPUSpatialEmpiricalBayesRates({
          events: view('float32', rows),
          populations: view('float32', rows),
          weights: weights(rows, SLOTS),
          mask: view('uint32', rows),
          output: {
            spatialRates: view('float32', rows),
            smoothedRates: view('float32', rows),
            priorMeans: view('float32', rows),
            priorVariances: view('float32', rows)
          }
        })
      );
    }
  }
];

// ---------------------------------------------------------------------------------------------
// Region statistics
// ---------------------------------------------------------------------------------------------

const REGION_ROWS = 4096;
const POLYGON_CAPACITY = 1024;

function regionSelection(
  context: KernelAuditContext,
  kind: GPURegionSelection['kind'],
  screen: boolean,
  rows: number
): GPURegionSelection {
  const {view} = context;
  const screenTransform = screen ? view('float32', 20, 'transform') : undefined;
  switch (kind) {
    case 'rectangle':
      return {kind, bounds: view('float32', 4, 'bounds'), screenTransform};
    case 'polygon':
      return {
        kind,
        vertices: view('float32x2', POLYGON_CAPACITY, 'vertices'),
        vertexCount: view('uint32', 1, 'vertex-count'),
        screenTransform
      };
    case 'radius':
      return {kind, circle: view('float32', 3, 'circle')};
    case 'pick-region':
      return {kind, picks: view('uint32', 2 + 2 * rows, 'pick-result'), batchIndex: 3};
    case 'mask':
      return {kind, mask: view('uint32', rows, 'selection-mask')};
  }
}

function regionProps(
  context: KernelAuditContext,
  rows: number,
  selection: GPURegionSelection,
  overrides: Partial<GPURegionStatisticsProps> = {}
): GPURegionStatisticsProps {
  const {view} = context;
  const binCount = 65536;
  const needsPositions = selection.kind !== 'pick-region' && selection.kind !== 'mask';
  return {
    region: selection,
    positions: needsPositions ? view('float32x2', rows, 'positions') : undefined,
    values: view('float32', rows, 'values'),
    sourceIds: view('uint32', rows, 'source-ids'),
    histogram: {binCount, domain: view('float32', 2, 'domain')},
    output: {
      summary: view('uint32', 8 + binCount, 'summary'),
      mask: selection.kind === 'mask' ? undefined : view('uint32', rows, 'output-mask'),
      selectedRows: {
        ids: view('uint32', rows, 'ids'),
        status: {count: view('uint32', 1, 'count'), overflow: view('uint32', 1, 'overflow')}
      },
      drawInstanceCount: view('uint32', 1, 'draw-count')
    },
    ...overrides
  } as GPURegionStatisticsProps;
}

function gridIndex(view: View, rows: number): GPUGridIndexView {
  return {
    gridSize: [64, 64],
    bounds: [0, 0, 10, 10],
    cellOffsets: view('uint32', 64 * 64 + 1),
    objectIds: view('uint32', rows),
    count: view('uint32', 1),
    overflow: view('uint32', 1)
  };
}

const REGION_VARIANTS: {name: string; kind: GPURegionSelection['kind']; screen: boolean}[] = [
  {name: 'rectangle', kind: 'rectangle', screen: false},
  {name: 'screen rectangle', kind: 'rectangle', screen: true},
  {name: 'polygon', kind: 'polygon', screen: false},
  {name: 'screen polygon', kind: 'polygon', screen: true},
  {name: 'radius', kind: 'radius', screen: false},
  {name: 'pick-region', kind: 'pick-region', screen: false},
  {name: 'mask', kind: 'mask', screen: false}
];

const REGION_CASES: KernelAuditCase[] = [
  ...REGION_VARIANTS.map(
    ({name, kind, screen}): KernelAuditCase => ({
      exportName: 'GPURegionStatistics',
      name: `${name} selection every output 65536 bins`,
      build: context =>
        context.graph.add(
          new GPURegionStatistics(
            regionProps(context, REGION_ROWS, regionSelection(context, kind, screen, REGION_ROWS))
          )
        )
    })
  ),
  ...(['rectangle', 'polygon', 'radius'] as const).map(
    (kind): KernelAuditCase => ({
      exportName: 'GPURegionStatistics',
      name: `${kind} selection through the grid spatialIndex`,
      build: context =>
        context.graph.add(
          new GPURegionStatistics(
            regionProps(context, REGION_ROWS, regionSelection(context, kind, false, REGION_ROWS), {
              spatialIndex: {
                kind: 'grid',
                index: gridIndex(context.view, REGION_ROWS),
                candidateCapacity: REGION_ROWS
              }
            })
          )
        )
    })
  ),
  {
    exportName: 'GPURegionStatistics',
    name: 'rectangle at two million rows',
    build: context =>
      context.graph.add(
        new GPURegionStatistics(
          regionProps(context, 2_000_000, regionSelection(context, 'rectangle', false, 0))
        )
      )
  },
  {
    exportName: 'GPURegionStatistics',
    name: 'rectangle rows at the binding limit',
    build: context => {
      const rows = context.maximumRows(8) - 2;
      context.graph.add(
        new GPURegionStatistics(
          regionProps(context, rows, regionSelection(context, 'rectangle', false, rows))
        )
      );
    }
  },
  {
    exportName: 'GPURegionStatistics',
    name: 'pick-region rows at the binding limit',
    build: context => {
      const rows = context.maximumRows(8) - 2;
      context.graph.add(
        new GPURegionStatistics(
          regionProps(context, rows, regionSelection(context, 'pick-region', false, rows))
        )
      );
    }
  },
  ...(['rectangle', 'polygon'] as const).flatMap(kind =>
    [false, true].map(
      (screen): KernelAuditCase => ({
        exportName: 'GPURegionMask',
        name: `${kind}${screen ? ' in screen space' : ''}`,
        build: context =>
          context.graph.add(
            new GPURegionMask({
              positions: context.view('float32x2', REGION_ROWS),
              region: regionSelection(context, kind, screen, REGION_ROWS) as GPURegionSelection & {
                kind: 'rectangle' | 'polygon';
              },
              output: {
                mask: context.view('uint32', REGION_ROWS),
                status: {overflow: context.view('uint32', 1)}
              }
            })
          )
      })
    )
  ),
  {
    exportName: 'GPURegionMask',
    name: 'polygon rows at the binding limit',
    build: context => {
      const rows = context.maximumRows(8);
      context.graph.add(
        new GPURegionMask({
          positions: context.view('float32x2', rows),
          region: regionSelection(context, 'polygon', true, rows) as GPURegionSelection & {
            kind: 'polygon';
          },
          output: {
            mask: context.view('uint32', rows),
            status: {overflow: context.view('uint32', 1)}
          }
        })
      );
    }
  },
  {
    exportName: 'GPUPickRegionMask',
    name: 'batch filter',
    build: ({graph, view}) =>
      graph.add(
        new GPUPickRegionMask({
          picks: view('uint32', 2 + 2 * REGION_ROWS),
          batchIndex: 0x7fffffff,
          output: {
            mask: view('uint32', REGION_ROWS),
            status: {overflow: view('uint32', 1)}
          }
        })
      )
  },
  {
    exportName: 'GPUPickRegionMask',
    name: 'rows at the binding limit',
    build: ({graph, view, maximumRows}) => {
      const rows = maximumRows(8) - 2;
      graph.add(
        new GPUPickRegionMask({
          picks: view('uint32', 2 + 2 * rows),
          batchIndex: 0x7fffffff,
          output: {
            mask: view('uint32', rows),
            status: {overflow: view('uint32', 1)}
          }
        })
      );
    }
  }
];

// ---------------------------------------------------------------------------------------------
// Scan statistics, space-time tests
// ---------------------------------------------------------------------------------------------

const SCAN_MODELS: GPUSpatialScanModel[] = ['poisson', 'bernoulli', 'space-time-permutation'];

function buildScan(
  {graph, view}: KernelAuditContext,
  model: GPUSpatialScanModel,
  options: {
    zones: number;
    timeBuckets: number;
    maximumPermutations: number;
    maximumClusters: number;
    maximumWindowZones?: number;
  }
) {
  const {zones, timeBuckets, maximumPermutations, maximumClusters} = options;
  const cells = zones * timeBuckets;
  graph.add(
    new GPUSpatialScanStatistic({
      positions: view('float32x2', zones),
      model,
      cases: view('uint32', cells),
      ...(model === 'poisson' ? {baseline: view('float32', cells)} : {}),
      ...(model === 'bernoulli' ? {controls: view('uint32', cells)} : {}),
      timeBuckets,
      maximumWindowZones: options.maximumWindowZones,
      maximumPermutations,
      maximumClusters,
      parameters: view('uint32', 8),
      output: {
        clusterIndices: view('uint32', maximumClusters * 4),
        clusterStatistics: view('float32', maximumClusters * 8),
        statistics: view('float32', maximumPermutations + 1),
        summary: view('uint32', 4),
        zoneStatistics: view('float32', zones)
      }
    })
  );
}

const SCAN_CASES = SCAN_MODELS.flatMap((model): KernelAuditCase[] => [
  {
    exportName: 'GPUSpatialScanStatistic',
    name: `${model} maximal options`,
    build: context =>
      buildScan(context, model, {
        zones: 40,
        timeBuckets: 32,
        maximumPermutations: 999,
        maximumClusters: 64,
        maximumWindowZones: 40
      })
  },
  {
    exportName: 'GPUSpatialScanStatistic',
    name: `${model} count-table maximum, 99999 replicates in batches`,
    build: context =>
      buildScan(context, model, {
        zones: 512,
        timeBuckets: 32,
        maximumPermutations: 99999,
        maximumClusters: 64,
        maximumWindowZones: 32
      })
  },
  {
    exportName: 'GPUSpatialScanStatistic',
    name: `${model} neighbor-table maximum`,
    build: context =>
      buildScan(context, model, {
        zones: 4096,
        timeBuckets: 1,
        maximumPermutations: 2047,
        maximumClusters: 64
      })
  }
]);

const MAXIMUM_SPACE_TIME_PERMUTATIONS = 65536;

function buildSpaceTime(
  {graph, view}: KernelAuditContext,
  kind: 'knox' | 'mantel',
  rows: number,
  maximumPermutations: number,
  stopAfterExceedances?: number
) {
  const slots = 4;
  const common = {
    id: kind,
    ...(stopAfterExceedances === undefined ? {} : {stopAfterExceedances}),
    pairs: {
      offsets: view('uint32', rows + 1),
      neighbors: view('uint32', slots * rows),
      weights: view('float32', slots * rows),
      distances: view('float32', slots * rows)
    },
    times: view('float32', rows),
    parameters: view('uint32', GPU_SPACE_TIME_PARAMETER_LENGTH),
    maximumPermutations
  };
  const summary = view('float32', GPU_SPACE_TIME_SUMMARY_LENGTH);
  const permutationCount = stopAfterExceedances === undefined ? undefined : view('uint32', 1);
  graph.add(
    kind === 'knox'
      ? new GPUKnoxTest({
          ...common,
          output: {statistics: view('uint32', maximumPermutations + 1), summary, permutationCount}
        })
      : new GPUMantelTest({
          ...common,
          output: {statistics: view('float32', maximumPermutations + 1), summary, permutationCount}
        })
  );
}

const SPACE_TIME_CASES = (['knox', 'mantel'] as const).flatMap((kind): KernelAuditCase[] =>
  (
    [
      [300, 99],
      [300, MAXIMUM_SPACE_TIME_PERMUTATIONS],
      [100000, 999],
      [100000, MAXIMUM_SPACE_TIME_PERMUTATIONS]
    ] as const
  )
    .map(
      ([rows, permutations]): KernelAuditCase => ({
        exportName: kind === 'knox' ? 'GPUKnoxTest' : 'GPUMantelTest',
        name: `${rows} rows ${permutations} permutations`,
        build: context => buildSpaceTime(context, kind, rows, permutations)
      })
    )
    .concat(
      // Besag-Clifford stopping at the largest batch count, and with a large h on many rows.
      (
        [
          [300, MAXIMUM_SPACE_TIME_PERMUTATIONS, MAXIMUM_SPACE_TIME_PERMUTATIONS],
          [100000, MAXIMUM_SPACE_TIME_PERMUTATIONS, 10]
        ] as const
      ).map(
        ([rows, permutations, stopAfterExceedances]): KernelAuditCase => ({
          exportName: kind === 'knox' ? 'GPUKnoxTest' : 'GPUMantelTest',
          name: `${rows} rows ${permutations} permutations stopAfterExceedances ${stopAfterExceedances}`,
          build: context => buildSpaceTime(context, kind, rows, permutations, stopAfterExceedances)
        })
      )
    )
);

// ---------------------------------------------------------------------------------------------
// Spatial autocorrelation
// ---------------------------------------------------------------------------------------------

const AUTOCORRELATION_CASES: KernelAuditCase[] = [1000, 2_000_000, -1].flatMap(
  (requestedRows): KernelAuditCase[] => {
    const label = requestedRows < 0 ? 'binding-limit rows' : `${requestedRows} rows`;
    const rowsOf = (context: KernelAuditContext) =>
      requestedRows < 0 ? context.maximumRows(4 * SLOTS) : requestedRows;
    return [
      ...METHODS.flatMap((multipleTesting): KernelAuditCase[] => [
        {
          exportName: 'GPUHotSpotAnalysis',
          name: `${multipleTesting} ${label} every output`,
          build: context => {
            const rows = rowsOf(context);
            const {graph, view, weights} = context;
            graph.add(
              new GPUHotSpotAnalysis({
                weights: weights(rows, SLOTS),
                selfWeight: 1,
                values: view('float32', rows),
                parameters: view('float32', GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH),
                mask: view('uint32', rows),
                islands: 'keep',
                minimumNeighbors: 8,
                multipleTesting,
                output: {
                  zScores: view('float32', rows),
                  bins: view('sint32', rows),
                  pValues: view('float32', rows),
                  adjustedPValues: view('float32', rows),
                  neighborCounts: view('uint32', rows),
                  globalStatistics: view('float32', GPU_SPATIAL_AUTOCORRELATION_STATISTICS_LENGTH)
                }
              })
            );
          }
        },
        ...(multipleTesting === 'none'
          ? (['analytic', 'none'] as const)
          : (['analytic'] as const)
        ).map(
          (quadrantGating): KernelAuditCase => ({
            exportName: 'GPULocalMoran',
            name: `${multipleTesting} ${quadrantGating} gating ${label}`,
            build: context => {
              const rows = rowsOf(context);
              const {graph, view, weights} = context;
              graph.add(
                new GPULocalMoran({
                  weights: weights(rows, SLOTS),
                  values: view('float32', rows),
                  parameters: view('float32', GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH),
                  mask: view('uint32', rows),
                  islands: 'keep',
                  minimumNeighbors: 8,
                  quadrantGating,
                  multipleTesting,
                  output: {
                    zScores: view('float32', rows),
                    localI: view('float32', rows),
                    spatialLag: view('float32', rows),
                    quadrants: view('uint32', rows),
                    pValues: view('float32', rows),
                    adjustedPValues: view('float32', rows),
                    neighborCounts: view('uint32', rows),
                    globalStatistics: view('float32', GPU_SPATIAL_AUTOCORRELATION_STATISTICS_LENGTH)
                  }
                })
              );
            }
          })
        )
      ]),
      {
        exportName: 'GPULocalGeary',
        name: `every output ${label}`,
        build: context => {
          const rows = rowsOf(context);
          const {graph, view, weights} = context;
          graph.add(
            new GPULocalGeary({
              weights: weights(rows, SLOTS),
              values: view('float32', rows),
              parameters: view('float32', GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH),
              mask: view('uint32', rows),
              islands: 'keep',
              output: {
                statistic: view('float32', rows),
                globalStatistics: view('float32', GPU_SPATIAL_AUTOCORRELATION_STATISTICS_LENGTH)
              }
            })
          );
        }
      },
      {
        exportName: 'GPUSpatialPearson',
        name: `every output ${label}`,
        build: context => {
          const rows = rowsOf(context);
          const {graph, view, weights} = context;
          graph.add(
            new GPUSpatialPearson({
              weights: weights(rows, SLOTS),
              values: view('float32', rows),
              secondValues: view('float32', rows),
              mask: view('uint32', rows),
              output: {statistic: view('float32', 4)}
            })
          );
        }
      },
      ...(
        ['cross-product', 'squared-difference', 'absolute-difference'] as GPUGammaOperation[]
      ).map(
        (operation): KernelAuditCase => ({
          exportName: 'GPUGammaStatistic',
          name: `${operation} ${label}`,
          build: context => {
            const rows = rowsOf(context);
            const {graph, view, weights} = context;
            graph.add(
              new GPUGammaStatistic({
                weights: weights(rows, SLOTS),
                operation,
                values: view('float32', rows),
                secondValues: view('float32', rows),
                mask: view('uint32', rows),
                output: {statistic: view('float32', 4)}
              })
            );
          }
        })
      ),
      {
        exportName: 'GPUAssuncaoReisStandardization',
        name: `every output ${label}`,
        build: context => {
          const rows =
            requestedRows < 0 ? Math.min(context.maximumRows(4), 2 ** 24 - 1) : requestedRows;
          const {graph, view} = context;
          graph.add(
            new GPUAssuncaoReisStandardization({
              events: view('float32', rows),
              populations: view('float32', rows),
              mask: view('uint32', rows),
              output: {
                standardizedRates: view('float32', rows),
                statistics: view('float32', GPU_RATE_STANDARDIZATION_STATISTICS_LENGTH),
                status: {invalidCount: view('uint32', 1)}
              }
            })
          );
        }
      }
    ];
  }
);

// ---------------------------------------------------------------------------------------------
// Spatial regression
// ---------------------------------------------------------------------------------------------

const POWER = GPU_SPATIAL_WEIGHTS_TRACES_MAXIMUM_POWER;

/** Row count at the binding limit for a contributor with `predictorCount` float32 predictors. */
function regressionRows(context: KernelAuditContext, predictorCount: number): number {
  return context.maximumRows(4 * Math.max(predictorCount, SLOTS));
}

/** Exact traces support at most 16384 rows; stochastic traces go to the binding limit. */
function traceRows(
  context: KernelAuditContext,
  traceMode: 'exact' | 'stochastic',
  predictorCount: number
): number {
  return traceMode === 'exact' ? 16384 : regressionRows(context, predictorCount);
}

function buildOLS(context: KernelAuditContext, rows: number) {
  const {graph, view} = context;
  const k = 15;
  graph.add(
    new GPUOrdinaryLeastSquares({
      predictors: view('float32', rows * k),
      response: view('float32', rows),
      mask: view('uint32', rows),
      parameters: view('float32', GPU_ORDINARY_LEAST_SQUARES_PARAMETER_LENGTH),
      predictorCount: k,
      tileRowCount: 64,
      output: {
        coefficients: view('float32', k + 1),
        standardErrors: view('float32', k + 1),
        tStatistics: view('float32', k + 1),
        summary: view('float32', GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH),
        status: {code: view('uint32', 1)},
        residuals: view('float32', rows),
        fitted: view('float32', rows)
      }
    })
  );
}

function buildRegressionDiagnostics(
  context: KernelAuditContext,
  rows: number,
  restandardizeRows: boolean
) {
  const {graph, view, weights} = context;
  const k = 15;
  graph.add(
    new GPUSpatialRegressionDiagnostics({
      weights: weights(rows, SLOTS),
      predictors: view('float32', rows * k),
      response: view('float32', rows),
      residuals: view('float32', rows),
      mask: view('uint32', rows),
      restandardizeRows,
      predictorCount: k,
      // Small tiles on small inputs; the default tiling at the row maximum.
      ...(rows <= ROWS ? {tileRowCount: 64} : {}),
      output: {
        tests: view('float32', GPU_SPATIAL_REGRESSION_DIAGNOSTICS_TESTS_LENGTH),
        summary: view('float32', GPU_SPATIAL_REGRESSION_DIAGNOSTICS_SUMMARY_LENGTH),
        status: {code: view('uint32', 1)}
      }
    })
  );
}

function buildTwoStage(context: KernelAuditContext, rows: number, instrumentOrder: 1 | 2) {
  const {graph, view, weights} = context;
  const k = instrumentOrder === 1 ? GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_MAXIMUM_PREDICTOR_COUNT : 5;
  graph.add(
    new GPUSpatialTwoStageLeastSquares({
      weights: weights(rows, SLOTS),
      predictors: view('float32', rows * k),
      response: view('float32', rows),
      predictorCount: k,
      tileRowCount: 64,
      instrumentOrder,
      output: {
        table: view('float32', (k + 3) * 4),
        summary: view('float32', GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_LENGTH),
        status: {code: view('uint32', 1)},
        residuals: view('float32', rows)
      }
    })
  );
}

function buildErrorGM(
  context: KernelAuditContext,
  rows: number,
  estimator: 'homoskedastic' | 'kelejian-prucha',
  traceMode: 'exact' | 'stochastic'
) {
  const {graph, view, weights} = context;
  const k = GPU_SPATIAL_ERROR_GM_MAXIMUM_PREDICTOR_COUNT;
  graph.add(
    new GPUSpatialErrorGM({
      weights: weights(rows, SLOTS),
      predictors: view('float32', rows * k),
      response: view('float32', rows),
      predictorCount: k,
      estimator,
      traceMode,
      traceProbeCount: 64,
      traceParameters: view('uint32', 1),
      tileRowCount: 64,
      output: {
        table: view('float32', (k + 3) * 4),
        summary: view('float32', GPU_SPATIAL_ERROR_GM_SUMMARY_LENGTH),
        status: {code: view('uint32', 1)},
        residuals: view('float32', rows)
      }
    })
  );
}

function buildLagML(context: KernelAuditContext, rows: number, traceMode: 'exact' | 'stochastic') {
  const {graph, view, weights} = context;
  const k = GPU_SPATIAL_LAG_MAXIMUM_LIKELIHOOD_MAXIMUM_PREDICTOR_COUNT;
  graph.add(
    new GPUSpatialLagMaximumLikelihood({
      weights: weights(rows, SLOTS),
      predictors: view('float32', rows * k),
      response: view('float32', rows),
      predictorCount: k,
      maximumPower: POWER,
      traceMode,
      traceProbeCount: 64,
      traceParameters: view('uint32', 1),
      tileRowCount: 64,
      output: {
        table: view('float32', (k + 3) * 4),
        summary: view('float32', GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LENGTH),
        status: {code: view('uint32', 1), converged: view('uint32', 1)},
        residuals: view('float32', rows),
        powerTraces: view('float32', POWER),
        powerSums: view('float32', POWER),
        infinityNorm: view('float32', 1)
      }
    })
  );
}

function buildErrorML(
  context: KernelAuditContext,
  rows: number,
  traceMode: 'exact' | 'stochastic'
) {
  const {graph, view, weights} = context;
  const k = GPU_SPATIAL_ERROR_MAXIMUM_LIKELIHOOD_MAXIMUM_PREDICTOR_COUNT;
  graph.add(
    new GPUSpatialErrorMaximumLikelihood({
      weights: weights(rows, SLOTS),
      predictors: view('float32', rows * k),
      response: view('float32', rows),
      predictorCount: k,
      maximumPower: POWER,
      traceMode,
      traceProbeCount: 64,
      traceParameters: view('uint32', 1),
      tileRowCount: 64,
      output: {
        table: view('float32', (k + 3) * 4),
        summary: view('float32', GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LENGTH),
        status: {code: view('uint32', 1), converged: view('uint32', 1)},
        residuals: view('float32', rows)
      }
    })
  );
}

function buildWeightsTraces(
  context: KernelAuditContext,
  rows: number,
  mode: 'exact' | 'stochastic',
  includeGramTraces: boolean
) {
  const {graph, view, weights} = context;
  graph.add(
    new GPUSpatialWeightsTraces({
      weights: weights(rows, SLOTS),
      maximumPower: POWER,
      mode,
      probeCount: 64,
      parameters: view('uint32', 1),
      includeGramTraces,
      tileRowCount: 64,
      probeBatchSize: 16,
      output: {
        powerTraces: view('float32', POWER),
        powerSums: view('float32', POWER),
        gramTraces: includeGramTraces ? view('float32', 4) : undefined,
        infinityNorm: view('float32', 1),
        status: {approximation: view('uint32', 1)}
      }
    })
  );
}

function buildGWR(
  {graph, view}: KernelAuditContext,
  rows: number,
  options: {
    greatCircle?: boolean;
    ladderLength?: number;
    refinement?: boolean | number;
    indexGridSize?: readonly [number, number] | false;
    bandwidthSearch?: 'ladder' | 'golden-section';
    maximumSearchIterations?: number;
    gaussianCutoff?: number;
  } = {}
) {
  const k = GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_PREDICTOR_COUNT;
  const ladder =
    options.ladderLength ?? GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_LADDER_LENGTH;
  const refinement =
    options.refinement ?? GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_REFINEMENT_PASS_COUNT;
  const passes = typeof refinement === 'number' ? refinement : refinement ? 1 : 0;
  const isGoldenSection = options.bandwidthSearch === 'golden-section';
  const tableRowCount = isGoldenSection
    ? (options.maximumSearchIterations ?? 30) + 1
    : (1 + passes) * ladder;
  const coefficientCount = k + 1;
  graph.add(
    new GPUGeographicallyWeightedRegression({
      positions: view('float32x2', rows),
      predictors: view('float32', rows * k),
      predictorCount: k,
      response: view('float32', rows),
      mask: view('uint32', rows),
      parameters: view('float32', getGPUGeographicallyWeightedRegressionParameterLength(ladder)),
      maximumBandwidthCount: ladder,
      maximumNeighborCount: GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_NEIGHBOR_COUNT,
      refinement,
      indexGridSize: options.indexGridSize,
      bandwidthSearch: options.bandwidthSearch,
      maximumSearchIterations: options.maximumSearchIterations,
      gaussianCutoff: options.gaussianCutoff,
      spatialContext: options.greatCircle
        ? {coordinateSpace: 'longitude-latitude', metric: 'great-circle'}
        : undefined,
      output: {
        coefficients: view('float32', rows * coefficientCount),
        standardErrors: view('float32', rows * coefficientCount),
        tValues: view('float32', rows * coefficientCount),
        significant: view('uint32', rows * coefficientCount),
        localR2: view('float32', rows),
        fitted: view('float32', rows),
        residuals: view('float32', rows),
        hatDiagonal: view('float32', rows),
        localStatus: view('uint32', rows),
        localConditionNumber: view('float32', rows),
        bandwidthScores: view('float32', ladder),
        bandwidthSingularRowCounts: view('uint32', ladder),
        candidateTable: view(
          'float32',
          tableRowCount * GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_CANDIDATE_TABLE_STRIDE
        ),
        selectedBandwidth: view('float32', 2),
        ...(isGoldenSection
          ? {
              searchState: view(
                'float32',
                GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SEARCH_STATE_LENGTH
              )
            }
          : {}),
        summary: view('float32', GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY_LENGTH)
      }
    })
  );
}

function buildNonstationarity(context: KernelAuditContext, rows: number, tileRowCount?: number) {
  const {graph, view} = context;
  const k = GPU_GWR_NONSTATIONARITY_MAXIMUM_PREDICTOR_COUNT;
  graph.add(
    new GPUGeographicallyWeightedRegressionNonstationarityTest({
      positions: view('float32x2', rows),
      predictors: view('float32', rows * k),
      predictorCount: k,
      response: view('float32', rows),
      mask: view('uint32', rows),
      bandwidthParameters: view('float32', getGPUGeographicallyWeightedRegressionParameterLength()),
      selectedBandwidth: view('float32', 2),
      coefficients: view('float32', rows * (k + 1)),
      parameters: view('uint32', 4),
      maximumPermutations: 999,
      maximumBandwidthCount: GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_LADDER_LENGTH,
      maximumNeighborCount: GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_NEIGHBOR_COUNT,
      tileRowCount,
      output: {
        table: view('float32', (k + 1) * 8),
        summary: view('float32', 16),
        standardDeviations: view('float32', 1000 * (k + 1))
      }
    })
  );
}

const REGRESSION_CASES: KernelAuditCase[] = [
  {
    exportName: 'GPUOrdinaryLeastSquares',
    name: 'maximal predictors every output',
    build: context => buildOLS(context, ROWS)
  },
  {
    exportName: 'GPUOrdinaryLeastSquares',
    name: 'rows at the binding limit',
    build: context => buildOLS(context, regressionRows(context, 15))
  },
  ...[false, true].flatMap((restandardizeRows): KernelAuditCase[] => [
    {
      exportName: 'GPUSpatialRegressionDiagnostics',
      name: `maximal predictors restandardize ${restandardizeRows}`,
      build: context => buildRegressionDiagnostics(context, ROWS, restandardizeRows)
    },
    {
      exportName: 'GPUSpatialRegressionDiagnostics',
      name: `restandardize ${restandardizeRows} rows at the maximum`,
      build: context =>
        buildRegressionDiagnostics(
          context,
          Math.floor(
            getGPUSpatialRegressionDiagnosticsMaximumRowTableValueCount(context.device) / 18
          ),
          restandardizeRows
        )
    },
    {
      exportName: 'GPUSpatialRegressionDiagnostics',
      name: `restandardize ${restandardizeRows} rows above the maximum`,
      build: context =>
        buildRegressionDiagnostics(
          context,
          Math.floor(
            getGPUSpatialRegressionDiagnosticsMaximumRowTableValueCount(context.device) / 18
          ) + 1,
          restandardizeRows
        )
    }
  ]),
  ...([1, 2] as const).flatMap((instrumentOrder): KernelAuditCase[] => [
    {
      exportName: 'GPUSpatialTwoStageLeastSquares',
      name: `instrument order ${instrumentOrder}`,
      build: context => buildTwoStage(context, ROWS, instrumentOrder)
    },
    {
      exportName: 'GPUSpatialTwoStageLeastSquares',
      name: `instrument order ${instrumentOrder} rows at the binding limit`,
      build: context =>
        buildTwoStage(
          context,
          regressionRows(context, GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_MAXIMUM_PREDICTOR_COUNT),
          instrumentOrder
        )
    }
  ]),
  ...(['homoskedastic', 'kelejian-prucha'] as const).flatMap(estimator =>
    (['exact', 'stochastic'] as const).flatMap((traceMode): KernelAuditCase[] => [
      {
        exportName: 'GPUSpatialErrorGM',
        name: `${estimator}/${traceMode} maximal predictors`,
        build: context => buildErrorGM(context, ROWS, estimator, traceMode)
      },
      {
        exportName: 'GPUSpatialErrorGM',
        name: `${estimator}/${traceMode} rows at the binding limit`,
        build: context =>
          buildErrorGM(
            context,
            traceRows(context, traceMode, GPU_SPATIAL_ERROR_GM_MAXIMUM_PREDICTOR_COUNT),
            estimator,
            traceMode
          )
      }
    ])
  ),
  ...(['exact', 'stochastic'] as const).flatMap((traceMode): KernelAuditCase[] => [
    {
      exportName: 'GPUSpatialLagMaximumLikelihood',
      name: `${traceMode} maximal predictors and power`,
      build: context => buildLagML(context, ROWS, traceMode)
    },
    {
      exportName: 'GPUSpatialLagMaximumLikelihood',
      name: `${traceMode} rows at the binding limit`,
      build: context =>
        buildLagML(
          context,
          traceRows(context, traceMode, GPU_SPATIAL_LAG_MAXIMUM_LIKELIHOOD_MAXIMUM_PREDICTOR_COUNT),
          traceMode
        )
    },
    {
      exportName: 'GPUSpatialErrorMaximumLikelihood',
      name: `${traceMode} maximal predictors and power`,
      build: context => buildErrorML(context, ROWS, traceMode)
    },
    {
      exportName: 'GPUSpatialErrorMaximumLikelihood',
      name: `${traceMode} rows at the binding limit`,
      build: context =>
        buildErrorML(
          context,
          traceRows(
            context,
            traceMode,
            GPU_SPATIAL_ERROR_MAXIMUM_LIKELIHOOD_MAXIMUM_PREDICTOR_COUNT
          ),
          traceMode
        )
    }
  ]),
  {
    exportName: 'GPUSpatialLagImpacts',
    name: 'maximal power and predictors',
    build: ({graph, view}) => {
      const k = 14;
      graph.add(
        new GPUSpatialLagImpacts({
          coefficients: view('float32', (k + 2) * 4),
          coefficientStride: 4,
          rhoIndex: k + 1,
          firstPredictorIndex: 0,
          predictorCount: k,
          rowCount: ROWS,
          maximumPower: POWER,
          traces: {
            powerTraces: view('float32', POWER),
            powerSums: view('float32', POWER),
            infinityNorm: view('float32', 1)
          },
          status: view('uint32', 1),
          rhoBound: 0.99,
          output: {impacts: view('float32', k * 3)}
        })
      );
    }
  },
  {
    exportName: 'GPUSpatialLagImpacts',
    name: 'row count at the binding limit',
    build: ({graph, view, maximumRows}) => {
      const k = 14;
      graph.add(
        new GPUSpatialLagImpacts({
          coefficients: view('float32', (k + 2) * 4),
          coefficientStride: 4,
          rhoIndex: k + 1,
          firstPredictorIndex: 0,
          predictorCount: k,
          rowCount: maximumRows(4),
          maximumPower: POWER,
          traces: {
            powerTraces: view('float32', POWER),
            powerSums: view('float32', POWER),
            infinityNorm: view('float32', 1)
          },
          status: view('uint32', 1),
          rhoBound: 0.99,
          output: {impacts: view('float32', k * 3)}
        })
      );
    }
  },
  ...(['exact', 'stochastic'] as const).flatMap(mode =>
    [false, true].flatMap((includeGramTraces): KernelAuditCase[] => [
      {
        exportName: 'GPUSpatialWeightsTraces',
        name: `${mode} gram ${includeGramTraces} maximal power`,
        build: context => buildWeightsTraces(context, ROWS, mode, includeGramTraces)
      },
      {
        exportName: 'GPUSpatialWeightsTraces',
        name: `${mode} gram ${includeGramTraces} rows at the binding limit`,
        build: context =>
          buildWeightsTraces(context, traceRows(context, mode, SLOTS), mode, includeGramTraces)
      }
    ])
  ),
  ...[true, false].flatMap((greatCircle): KernelAuditCase[] => [
    {
      exportName: 'GPUGeographicallyWeightedRegression',
      name: `${greatCircle ? 'great-circle' : 'planar'} maximal options`,
      build: context => buildGWR(context, ROWS, {greatCircle})
    }
  ]),
  ...[true, false].flatMap((greatCircle): KernelAuditCase[] => [
    {
      exportName: 'GPUGeographicallyWeightedRegression',
      name: `${greatCircle ? 'great-circle' : 'planar'} golden-section search at 64 iterations with Gaussian cutoff`,
      build: context =>
        buildGWR(context, ROWS, {
          greatCircle,
          bandwidthSearch: 'golden-section',
          maximumSearchIterations: GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_SEARCH_ITERATIONS,
          gaussianCutoff: GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_RECOMMENDED_GAUSSIAN_CUTOFF
        })
    }
  ]),
  {
    exportName: 'GPUGeographicallyWeightedRegression',
    name: 'golden-section search, maximum indexed rows, Gaussian cutoff',
    build: context =>
      buildGWR(context, GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_INDEXED_ROW_COUNT, {
        ladderLength: 2,
        bandwidthSearch: 'golden-section',
        gaussianCutoff: GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_RECOMMENDED_GAUSSIAN_CUTOFF
      })
  },
  {
    exportName: 'GPUGeographicallyWeightedRegression',
    name: 'without index grid or refinement',
    build: context => buildGWR(context, ROWS, {refinement: false, indexGridSize: false})
  },
  {
    exportName: 'GPUGeographicallyWeightedRegression',
    name: 'maximum indexed rows 16 bandwidths',
    build: context =>
      buildGWR(context, GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_INDEXED_ROW_COUNT, {
        ladderLength: 16
      })
  },
  {
    exportName: 'GPUGeographicallyWeightedRegression',
    name: 'maximum rows at the maximum ladder',
    build: context =>
      buildGWR(context, (1 << 24) / GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_LADDER_LENGTH, {
        ladderLength: 32
      })
  },
  {
    exportName: 'GPUGeographicallyWeightedRegressionNonstationarityTest',
    name: 'maximal options',
    build: context => buildNonstationarity(context, ROWS, 64)
  },
  {
    exportName: 'GPUGeographicallyWeightedRegressionNonstationarityTest',
    name: 'rows at the binding limit',
    build: context =>
      buildNonstationarity(
        context,
        Math.min(
          1048576,
          context.maximumRows(4 * (GPU_GWR_NONSTATIONARITY_MAXIMUM_PREDICTOR_COUNT + 1))
        )
      )
  }
];

// ---------------------------------------------------------------------------------------------
// Spatial weights
// ---------------------------------------------------------------------------------------------

const WEIGHT_ROWS = 1000;

const TRANSFORM_OPERATIONS: GPUSpatialWeightsTransformOperation[] = [
  'row',
  'binary',
  'symmetrize',
  'double',
  'variance'
];
const KERNELS: GPUSpatialWeightsKernel[] = [
  'gaussian',
  'triangular',
  'epanechnikov',
  'bisquare',
  'uniform'
];

function buildLattice(
  {graph, view}: KernelAuditContext,
  criterion: 'queen' | 'rook',
  radius: number,
  width: number,
  height: number
) {
  const slots = (2 * radius + 1) ** 2 - 1;
  graph.add(
    new GPULatticeWeights({
      width,
      height,
      criterion,
      radius,
      mask: view('uint32', width * height),
      cellSize: [10, 20],
      output: {
        ...capacityWeights(view, width * height, width * height * slots),
        status: {overflow: view('uint32', 1), requiredCount: view('uint32', 1)}
      }
    })
  );
}

function buildContiguity(
  {graph, view}: KernelAuditContext,
  criterion: 'queen' | 'rook',
  pairCapacity: number
) {
  graph.add(
    new GPUContiguityWeights({
      criterion,
      polygons: {
        kind: 'polygons',
        positions: view('float32x2', 4000),
        ringOffsets: view('uint32', 101),
        polygonOffsets: view('uint32', 101),
        featureOffsets: view('uint32', 51)
      },
      snapTolerance: 1e-3,
      output: {
        ...capacityWeights(view, 50, 50 * 64, false),
        status: {
          overflow: view('uint32', 1),
          requiredCount: view('uint32', 1),
          candidateOverflow: view('uint32', 1)
        }
      },
      pairCapacity
    })
  );
}

const ALGEBRA_STATUS = (view: View) => ({
  status: {overflow: view('uint32', 1), requiredCount: view('uint32', 1)}
});

function buildTransform(
  {graph, view}: KernelAuditContext,
  rows: number,
  operation: GPUSpatialWeightsTransformOperation | 'kernel',
  options: {kernel?: GPUSpatialWeightsKernel; bandwidth?: number | 'adaptive'} = {}
) {
  const slots = rows * SLOTS;
  graph.add(
    new GPUSpatialWeightsTransform(
      operation === 'kernel'
        ? {
            operation,
            weights: {
              ...capacityWeights(view, rows, slots),
              information: {
                rowCount: rows,
                transform: 'raw',
                symmetric: 'unknown',
                provenance: {source: 'test', criterion: 'knn', k: SLOTS}
              }
            },
            output: view('float32', slots),
            kernel: options.kernel,
            bandwidth: options.bandwidth
          }
        : {
            operation,
            weights: capacityWeights(view, rows, slots),
            output: view('float32', slots),
            doubleSum: operation === 'double' ? 'rows' : undefined
          }
    )
  );
}

const WEIGHT_CASES: KernelAuditCase[] = [
  ...(['queen', 'rook'] as const).flatMap((criterion): KernelAuditCase[] => [
    {
      exportName: 'GPUContiguityWeights',
      name: `${criterion} every output large pair capacity`,
      build: context => buildContiguity(context, criterion, 1 << 22)
    },
    {
      exportName: 'GPUContiguityWeights',
      name: `${criterion} pair capacity at the binding limit`,
      build: context => buildContiguity(context, criterion, context.maximumRows(8))
    }
  ]),
  ...(['queen', 'rook'] as const).flatMap(criterion =>
    [1, GPU_LATTICE_WEIGHTS_MAXIMUM_RADIUS].flatMap((radius): KernelAuditCase[] => [
      {
        exportName: 'GPULatticeWeights',
        name: `${criterion} radius ${radius} mask and cell size`,
        build: context => buildLattice(context, criterion, radius, 64, 64)
      },
      {
        exportName: 'GPULatticeWeights',
        name: `${criterion} radius ${radius} slots at the binding limit`,
        build: context => {
          const slots = (2 * radius + 1) ** 2 - 1;
          const cells = Math.min(2 ** 24, Math.floor(context.maximumRows(4) / slots));
          const width = Math.floor(Math.sqrt(cells));
          buildLattice(context, criterion, radius, width, width);
        }
      }
    ])
  ),
  ...TRANSFORM_OPERATIONS.flatMap((operation): KernelAuditCase[] => [
    {
      exportName: 'GPUSpatialWeightsTransform',
      name: operation,
      build: context => buildTransform(context, WEIGHT_ROWS, operation)
    },
    {
      exportName: 'GPUSpatialWeightsTransform',
      name: `${operation} at the binding limit`,
      build: context => buildTransform(context, context.maximumRows(4 * SLOTS), operation)
    }
  ]),
  ...KERNELS.flatMap(kernel =>
    ([250, 'adaptive'] as const).map(
      (bandwidth): KernelAuditCase => ({
        exportName: 'GPUSpatialWeightsTransform',
        name: `kernel ${kernel} ${bandwidth} bandwidth`,
        build: context => buildTransform(context, WEIGHT_ROWS, 'kernel', {kernel, bandwidth})
      })
    )
  ),
  ...[1, 64].flatMap((columnCount): KernelAuditCase[] => [
    {
      exportName: 'GPUSpatialLag',
      name: `mask normalize ${columnCount} columns`,
      build: ({graph, view}) =>
        graph.add(
          new GPUSpatialLag({
            values: view('float32', WEIGHT_ROWS * columnCount),
            columnCount,
            weights: capacityWeights(view, WEIGHT_ROWS, WEIGHT_ROWS * SLOTS),
            mask: view('uint32', WEIGHT_ROWS),
            normalize: true,
            output: view('float32', WEIGHT_ROWS * columnCount)
          })
        )
    },
    {
      exportName: 'GPUSpatialLag',
      name: `${columnCount} columns at the binding limit`,
      build: ({graph, view, maximumRows}) => {
        const rows = Math.floor(maximumRows(4 * Math.max(columnCount, SLOTS)));
        graph.add(
          new GPUSpatialLag({
            values: view('float32', rows * columnCount),
            columnCount,
            weights: capacityWeights(view, rows, rows * SLOTS),
            mask: view('uint32', rows),
            normalize: true,
            output: view('float32', rows * columnCount)
          })
        );
      }
    }
  ]),
  ...[WEIGHT_ROWS, -1].map(
    (requested): KernelAuditCase => ({
      exportName: 'GPUSpatialWeightsSummary',
      name: requested < 0 ? 'rows at the binding limit' : 'every output',
      build: ({graph, view, maximumRows}) => {
        const rows = requested < 0 ? maximumRows(4 * SLOTS) : requested;
        graph.add(
          new GPUSpatialWeightsSummary({
            weights: capacityWeights(view, rows, rows * SLOTS),
            output: {
              statistics: view('float32', GPU_SPATIAL_WEIGHTS_SUMMARY_LAYOUT.statistics.length),
              counts: view('uint32', GPU_SPATIAL_WEIGHTS_SUMMARY_LAYOUT.counts.length),
              cardinality: view('uint32', rows)
            },
            symmetryTolerance: 1e-6
          })
        );
      }
    })
  ),
  ...[WEIGHT_ROWS, -1].map(
    (requested): KernelAuditCase => ({
      exportName: 'GPUSpatialWeightsTranspose',
      name: requested < 0 ? 'rows at the binding limit' : 'every output',
      build: ({graph, view, maximumRows}) => {
        const rows = requested < 0 ? maximumRows(4 * SLOTS) : requested;
        graph.add(
          new GPUSpatialWeightsTranspose({
            weights: capacityWeights(view, rows, rows * SLOTS),
            columnCount: rows,
            output: {
              ...capacityWeights(view, rows, rows * SLOTS),
              asymmetricSlots: view('uint32', 1)
            },
            symmetryTolerance: 1e-6
          })
        );
      }
    })
  ),
  ...[WEIGHT_ROWS, -1].map(
    (requested): KernelAuditCase => ({
      exportName: 'GPUSpatialWeightsValidation',
      name: requested < 0 ? 'rows at the binding limit' : 'flags',
      build: ({graph, view, maximumRows}) => {
        const rows = requested < 0 ? maximumRows(4 * SLOTS) : requested;
        graph.add(
          new GPUSpatialWeightsValidation({
            weights: capacityWeights(view, rows, rows * SLOTS),
            output: {flags: view('uint32', rows), status: {invalidCount: view('uint32', 1)}}
          })
        );
      }
    })
  ),
  ...(['union', 'intersection', 'difference', 'symmetricDifference'] as const).flatMap(operation =>
    (['left', 'right', 'sum', 'min', 'max', 'product', 'binary'] as const).map(
      (weightRule): KernelAuditCase => ({
        exportName: 'GPUSpatialWeightsAlgebra',
        name: `${operation} ${weightRule} rule and distances`,
        build: ({graph, view}) =>
          graph.add(
            new GPUSpatialWeightsAlgebra({
              operation,
              weightRule,
              left: capacityWeights(view, WEIGHT_ROWS, WEIGHT_ROWS * SLOTS),
              right: capacityWeights(view, WEIGHT_ROWS, WEIGHT_ROWS * SLOTS),
              output: {
                ...capacityWeights(view, WEIGHT_ROWS, WEIGHT_ROWS * SLOTS * 2),
                ...ALGEBRA_STATUS(view)
              }
            })
          )
      })
    )
  ),
  {
    exportName: 'GPUSpatialWeightsAlgebra',
    name: 'union at the binding limit',
    build: ({graph, view, maximumRows}) => {
      const rows = maximumRows(4 * SLOTS * 2);
      graph.add(
        new GPUSpatialWeightsAlgebra({
          operation: 'union',
          weightRule: 'sum',
          left: capacityWeights(view, rows, rows * SLOTS),
          right: capacityWeights(view, rows, rows * SLOTS),
          output: {
            ...capacityWeights(view, rows, rows * SLOTS * 2),
            ...ALGEBRA_STATUS(view)
          }
        })
      );
    }
  },
  ...[false, true].flatMap(cumulative =>
    [2, GPU_SPATIAL_WEIGHTS_MAXIMUM_ORDER].map(
      (order): KernelAuditCase => ({
        exportName: 'GPUSpatialWeightsAlgebra',
        name: `higherOrder order ${order}${cumulative ? ' cumulative' : ''}`,
        build: ({graph, view}) =>
          graph.add(
            new GPUSpatialWeightsAlgebra({
              operation: 'higherOrder',
              weights: capacityWeights(view, WEIGHT_ROWS, WEIGHT_ROWS * SLOTS, false),
              order,
              cumulative,
              workCapacity: 1 << 22,
              output: {
                ...capacityWeights(view, WEIGHT_ROWS, WEIGHT_ROWS * 64, false),
                ...ALGEBRA_STATUS(view)
              }
            })
          )
      })
    )
  ),
  ...(['number', 'view'] as const).map(
    (selfWeightKind): KernelAuditCase => ({
      exportName: 'GPUSpatialWeightsAlgebra',
      name: `selfWeight with a ${selfWeightKind}`,
      build: ({graph, view}) =>
        graph.add(
          new GPUSpatialWeightsAlgebra({
            operation: 'selfWeight',
            weights: capacityWeights(view, WEIGHT_ROWS, WEIGHT_ROWS * SLOTS),
            selfWeight: selfWeightKind === 'number' ? 1 : view('float32', WEIGHT_ROWS),
            output: {
              ...capacityWeights(view, WEIGHT_ROWS, WEIGHT_ROWS * (SLOTS + 1)),
              ...ALGEBRA_STATUS(view)
            }
          })
        )
    })
  ),
  {
    exportName: 'GPUSpatialWeightsAlgebra',
    name: 'subgraph with a mask',
    build: ({graph, view}) =>
      graph.add(
        new GPUSpatialWeightsAlgebra({
          operation: 'subgraph',
          weights: capacityWeights(view, WEIGHT_ROWS, WEIGHT_ROWS * SLOTS),
          mask: view('uint32', WEIGHT_ROWS),
          output: {
            ...capacityWeights(view, WEIGHT_ROWS, WEIGHT_ROWS * SLOTS),
            ...ALGEBRA_STATUS(view)
          }
        })
      )
  },
  {
    exportName: 'GPUSpatialWeightsAlgebra',
    name: 'block with many groups',
    build: ({graph, view}) =>
      graph.add(
        new GPUSpatialWeightsAlgebra({
          operation: 'block',
          groupIds: view('uint32', WEIGHT_ROWS),
          groupCount: WEIGHT_ROWS,
          output: {
            ...capacityWeights(view, WEIGHT_ROWS, WEIGHT_ROWS * 32, false),
            ...ALGEBRA_STATUS(view)
          }
        })
      )
  }
];

// ---------------------------------------------------------------------------------------------
// Zonal statistics
// ---------------------------------------------------------------------------------------------

const EXTENT_STATISTICS: GPUZonalStatisticsExtentStatistic[] = [
  'count',
  'sum',
  'mean',
  'minimum',
  'maximum',
  'density'
];

function zonalProps(
  {view}: KernelAuditContext,
  options: {
    kind: 'polygons' | 'feature-rows';
    pointCount: number;
    featureCount: number;
    candidateCapacity: number;
    sumOrder: 'atomic' | 'sorted';
    extentStatistic: GPUZonalStatisticsExtentStatistic;
  }
): GPUZonalStatisticsProps {
  const {pointCount, featureCount} = options;
  const polygons = options.kind === 'polygons';
  return {
    id: 'z',
    features: polygons
      ? {
          kind: 'polygons',
          positions: view('float32x2', featureCount * 8, 'polygon-positions'),
          featureOffsets: view('uint32', featureCount + 1, 'feature-offsets'),
          polygonOffsets: view('uint32', featureCount + 1, 'polygon-offsets'),
          ringOffsets: view('uint32', featureCount + 1, 'ring-offsets'),
          candidateCapacity: options.candidateCapacity,
          boundaryRule: 'closed'
        }
      : {
          kind: 'feature-rows',
          pointFeatureRows: view('uint32', pointCount, 'rows'),
          featureCount
        },
    positions: polygons ? view('float32x2', pointCount, 'points') : undefined,
    values: view('float32', pointCount, 'values'),
    weights: view('float32', pointCount, 'weights'),
    areas: polygons ? undefined : view('float32', featureCount, 'areas'),
    sumOrder: options.sumOrder,
    output: {
      counts: view('uint32', featureCount, 'counts'),
      valueCounts: view('uint32', featureCount, 'value-counts'),
      sums: view('float32', featureCount, 'sums'),
      weightSums: view('float32', featureCount, 'weight-sums'),
      means: view('float32', featureCount, 'means'),
      minima: view('float32', featureCount, 'minima'),
      maxima: view('float32', featureCount, 'maxima'),
      variances: view('float32', featureCount, 'variances'),
      standardDeviations: view('float32', featureCount, 'standard-deviations'),
      densities: view('float32', featureCount, 'densities'),
      weightedDensities: view('float32', featureCount, 'weighted-densities'),
      featureAreas: polygons ? view('float32', featureCount, 'feature-areas') : undefined,
      pointFeatureRows: polygons ? view('uint32', pointCount, 'point-rows') : undefined,
      extent: view('float32', 2, 'extent'),
      extentStatistic: options.extentStatistic,
      status: {
        overflow: view('uint32', 1, 'overflow'),
        uncertainCount: view('uint32', 1, 'uncertain')
      }
    }
  };
}

const ZONAL_CASES: KernelAuditCase[] = [
  ...(['polygons', 'feature-rows'] as const).flatMap(kind =>
    (['atomic', 'sorted'] as const).map(
      (sumOrder): KernelAuditCase => ({
        exportName: 'GPUZonalStatistics',
        name: `${kind} every output ${sumOrder} sums`,
        build: context =>
          context.graph.add(
            new GPUZonalStatistics(
              zonalProps(context, {
                kind,
                pointCount: 4096,
                featureCount: 4096,
                candidateCapacity: 1 << 16,
                sumOrder,
                extentStatistic: 'mean'
              })
            )
          )
      })
    )
  ),
  ...EXTENT_STATISTICS.map(
    (extentStatistic): KernelAuditCase => ({
      exportName: 'GPUZonalStatistics',
      name: `extent of ${extentStatistic}`,
      build: context =>
        context.graph.add(
          new GPUZonalStatistics(
            zonalProps(context, {
              kind: 'polygons',
              pointCount: 256,
              featureCount: 64,
              candidateCapacity: 1024,
              sumOrder: 'sorted',
              extentStatistic
            })
          )
        )
    })
  ),
  {
    exportName: 'GPUZonalStatistics',
    name: 'polygons at scale',
    build: context =>
      context.graph.add(
        new GPUZonalStatistics(
          zonalProps(context, {
            kind: 'polygons',
            pointCount: 4_000_000,
            featureCount: 65536,
            candidateCapacity: 1 << 22,
            sumOrder: 'sorted',
            extentStatistic: 'density'
          })
        )
      )
  },
  {
    exportName: 'GPUZonalStatistics',
    name: 'feature-rows points at the binding limit',
    build: context =>
      context.graph.add(
        new GPUZonalStatistics(
          zonalProps(context, {
            kind: 'feature-rows',
            pointCount: context.maximumRows(4),
            featureCount: 65536,
            candidateCapacity: 1 << 16,
            sumOrder: 'atomic',
            extentStatistic: 'sum'
          })
        )
      )
  }
];

/** Kernel-audit cases for statistics. */
export const STATISTICS_CASES: KernelAuditCase[] = [
  ...COMPARE_CASES,
  ...DISTRIBUTION_CASES,
  ...EMERGING_CASES,
  ...GLOBAL_CASES,
  ...MULTIPLE_TESTING_CASES,
  ...NEIGHBORHOOD_CASES,
  ...LOCAL_PERMUTATION_CASES,
  ...GLOBAL_PERMUTATION_CASES,
  ...RATE_SMOOTHING_CASES,
  ...REGION_CASES,
  ...SCAN_CASES,
  ...SPACE_TIME_CASES,
  ...AUTOCORRELATION_CASES,
  ...REGRESSION_CASES,
  ...WEIGHT_CASES,
  ...ZONAL_CASES
];
