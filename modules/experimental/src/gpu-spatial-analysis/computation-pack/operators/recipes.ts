// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {SpatialFamilyOperators} from '../types';

/**
 * Spatial computation-pack entries for the `recipes` family, keyed by exported name. Every entry
 * must use `family: 'recipes'`. A recipe is a composition of contributors, so its formula is
 * always `none` and the card shows the contributor-grain command graph instead.
 */
export const RECIPES_OPERATORS: SpatialFamilyOperators = {
  addHotSpotAnalysisRecipe: {
    family: 'recipes',
    descriptor: {scope: 'focal', scopeSecondary: 'global', reduction: 'weighted-mean'},
    variables: {
      observations: 'measure',
      values: 'rate',
      zScores: 'score',
      pValues: 'probability',
      bins: 'rank',
      neighborCounts: 'count'
    },
    stages: ['representation', 'support', 'neighbourhood', 'operator', 'inference'],
    datumKinds: [{kind: 'cell', label: 'study cell', indexSpace: '1d'}],
    representation: {from: 'event', to: 'field'},
    guards: ['coordinate-model', 'count-vs-rate', 'uncorrected-p'],
    choices: [
      {stage: 'support', name: 'resolution', label: 'cell resolution', sweepable: true},
      {stage: 'neighbourhood', name: 'neighborRings', label: 'neighbour rings', sweepable: true},
      {stage: 'neighbourhood', name: 'minimumNeighbors', sweepable: false},
      {
        stage: 'inference',
        name: 'multipleTesting',
        alternatives: ['none', 'bonferroni', 'benjamini-hochberg'],
        sweepable: false
      },
      {stage: 'inference', name: 'significanceLevel', sweepable: true}
    ],
    formula: {
      formula: 'none',
      reason:
        'Composition of GPUCellCover, GPUCellAggregation, GPUCellTopology and GPUHotSpotAnalysis; see each contributor card for its formula.'
    },
    metadata: {
      coordinateModel: 'geographic',
      mappedVariableExtent: 'intensive',
      pCorrection: 'benjamini-hochberg'
    }
  },
  addRateClusterMapRecipe: {
    family: 'recipes',
    descriptor: {scope: 'focal', scopeSecondary: 'global', reduction: 'weighted-mean'},
    variables: {
      events: 'count',
      populations: 'count',
      standardizedRates: 'score',
      zScores: 'score',
      adjustedPValues: 'probability',
      quadrants: 'category'
    },
    stages: ['support', 'neighbourhood', 'operator', 'inference', 'presentation'],
    datumKinds: [{kind: 'region', label: 'polygon', indexSpace: '1d'}],
    guards: ['count-vs-rate', 'uncorrected-p', 'row-standardisation'],
    choices: [
      {
        stage: 'neighbourhood',
        name: 'criterion',
        alternatives: ['queen', 'rook'],
        sweepable: false
      },
      {
        stage: 'operator',
        name: 'analyze',
        alternatives: ['standardized', 'smoothed'],
        sweepable: false
      },
      {
        stage: 'inference',
        name: 'multipleTesting',
        alternatives: ['none', 'bonferroni', 'benjamini-hochberg'],
        sweepable: false
      },
      {stage: 'inference', name: 'permutationCount', label: 'permutations', sweepable: true},
      {stage: 'inference', name: 'significanceLevel', sweepable: true}
    ],
    formula: {
      formula: 'none',
      reason:
        'Composition of GPUEmpiricalBayesRates, GPUContiguityWeights, GPULocalMoran and GPULocalPermutationTest; see each contributor card for its formula.'
    },
    metadata: {
      weightsStandardisation: 'row',
      pCorrection: 'benjamini-hochberg',
      mappedVariableExtent: 'intensive'
    }
  },
  addPeriodComparisonRecipe: {
    family: 'recipes',
    descriptor: {scope: 'block', scopeSecondary: 'pairwise', reduction: 'count'},
    variables: {
      before: 'measure',
      after: 'measure',
      delta: 'count',
      zScore: 'score',
      classIndices: 'rank'
    },
    stages: ['representation', 'support', 'operator', 'presentation'],
    datumKinds: [{kind: 'cell', label: 'union cell', indexSpace: '1d'}],
    representation: {from: 'event', to: 'field'},
    guards: ['count-vs-rate'],
    choices: [
      {stage: 'support', name: 'resolution', label: 'cell resolution', sweepable: true},
      {stage: 'operator', name: 'measure', alternatives: ['count', 'sum'], sweepable: false},
      {
        stage: 'presentation',
        name: 'classify',
        alternatives: ['delta', 'ratio', 'percentChange', 'zScore'],
        sweepable: false
      }
    ],
    formula: {
      formula: 'none',
      reason:
        'Composition of GPUCellAggregation, GPUCellTableCompare, GPUClassBreaks and GPUColorScale; see each contributor card for its formula.'
    },
    metadata: {mappedVariableExtent: 'extensive', denominator: 'none: raw count difference'}
  },
  addChangeOfSupportRecipe: {
    family: 'recipes',
    descriptor: {scope: 'zonal', reduction: 'weighted-mean'},
    variables: {
      sourceCounts: 'count',
      extensiveValues: 'count',
      sourceTransferredFractions: 'rate',
      sourceAreas: 'measure'
    },
    stages: ['representation', 'support', 'operator'],
    datumKinds: [{kind: 'region', label: 'target zone', indexSpace: '1d'}],
    guards: ['coordinate-model', 'conservation'],
    choices: [
      {stage: 'support', name: 'cellSize', label: 'raster cell size', unit: 'm', sweepable: true},
      {
        stage: 'operator',
        name: 'denominator',
        alternatives: ['zone', 'overlap'],
        sweepable: false
      }
    ],
    formula: {
      formula: 'none',
      reason:
        'Composition of GPUPolygonRasterization, GPUArealInterpolation and GPUSpatialLag; see each contributor card for its formula.'
    },
    metadata: {coordinateModel: 'projected', metricUnit: 'm', mappedVariableExtent: 'extensive'}
  },
  addPointsInPolygonsChoroplethRecipe: {
    family: 'recipes',
    descriptor: {scope: 'zonal', reduction: 'count'},
    variables: {observations: 'measure', counts: 'count', classIndices: 'rank'},
    stages: ['support', 'operator', 'presentation'],
    datumKinds: [{kind: 'region', label: 'polygon', indexSpace: '1d'}],
    guards: ['coordinate-model', 'count-vs-rate'],
    choices: [
      {stage: 'operator', name: 'backend', alternatives: ['zonal', 'group'], sweepable: false},
      {
        stage: 'operator',
        name: 'statistic',
        alternatives: ['count', 'sum', 'mean', 'minimum', 'maximum'],
        sweepable: false
      },
      {
        stage: 'presentation',
        name: 'classMethod',
        alternatives: ['equal-interval', 'quantile', 'natural-breaks'],
        sweepable: false
      },
      {stage: 'presentation', name: 'classCount', sweepable: true}
    ],
    formula: {
      formula: 'none',
      reason:
        'Composition of GPUZonalStatistics, GPUClassBreaks and GPUColorScale; see each contributor card for its formula.'
    },
    metadata: {coordinateModel: 'projected', metricUnit: 'm', mappedVariableExtent: 'extensive'}
  },
  addClusterAndOutlineRecipe: {
    family: 'recipes',
    descriptor: {scope: 'focal', scopeSecondary: 'zonal', reduction: 'count'},
    variables: {
      observations: 'measure',
      labels: 'category',
      areas: 'measure',
      perimeters: 'measure'
    },
    stages: ['neighbourhood', 'operator', 'presentation'],
    datumKinds: [{kind: 'cluster', label: 'cluster', indexSpace: '1d'}],
    guards: ['coordinate-model'],
    choices: [
      {stage: 'neighbourhood', name: 'epsilon', label: 'epsilon', unit: 'm', sweepable: true},
      {stage: 'neighbourhood', name: 'minimumPoints', sweepable: true}
    ],
    formula: {
      formula: 'none',
      reason:
        'Composition of GPUSpatialClustering, GPUGroupGeometry, GPUGroupConvexHull and GPUGeometryMeasures; see each contributor card for its formula.'
    },
    metadata: {coordinateModel: 'projected', metricUnit: 'm'}
  },
  addFleetDwellRecipe: {
    family: 'recipes',
    descriptor: {scope: 'zonal', scopeSecondary: 'focal', reduction: 'sum'},
    variables: {
      durations: 'measure',
      stopZones: 'category',
      zoneCounts: 'count',
      zoneDwell: 'measure'
    },
    stages: ['support', 'operator'],
    datumKinds: [{kind: 'stop', label: 'stop', indexSpace: '1d'}],
    guards: ['coordinate-model'],
    choices: [
      {
        stage: 'operator',
        name: 'stopSpeedThreshold',
        label: 'stop speed threshold',
        unit: 'm/s',
        sweepable: true
      },
      {
        stage: 'operator',
        name: 'stopMinimumDuration',
        label: 'minimum stop duration',
        unit: 's',
        sweepable: true
      }
    ],
    formula: {
      formula: 'none',
      reason:
        'Composition of GPUTrajectoryMetrics, GPUPointInPolygonJoin and GPUGroupStatistics; see each contributor card for its formula.'
    },
    metadata: {coordinateModel: 'projected', metricUnit: 'm'}
  },
  addSpatialRegressionRecipe: {
    family: 'recipes',
    descriptor: {scope: 'global', scopeSecondary: 'focal', reduction: 'solve'},
    variables: {
      response: 'count',
      fitted: 'count',
      residuals: 'score',
      coefficients: 'score',
      decision: 'category'
    },
    stages: ['neighbourhood', 'operator', 'inference'],
    datumKinds: [{kind: 'region', label: 'polygon', indexSpace: '1d'}],
    guards: ['row-standardisation', 'uncorrected-p'],
    choices: [
      {
        stage: 'neighbourhood',
        name: 'criterion',
        alternatives: ['queen', 'rook'],
        sweepable: false
      },
      {
        stage: 'neighbourhood',
        name: 'transform',
        alternatives: ['binary', 'row'],
        sweepable: false
      },
      {stage: 'inference', name: 'significanceLevel', sweepable: true}
    ],
    formula: {
      formula: 'none',
      reason:
        'Composition of GPUOrdinaryLeastSquares, GPUSpatialRegressionDiagnostics and GPULocalMoran; see each contributor card for its formula.'
    },
    metadata: {weightsStandardisation: 'row', pCorrection: 'none'}
  },
  addStraightLineCatchmentsRecipe: {
    family: 'recipes',
    descriptor: {scope: 'zonal', scopeSecondary: 'pairwise', reduction: 'argmin'},
    variables: {facilities: 'measure', allocation: 'category', distances: 'measure', sums: 'count'},
    stages: ['support', 'operator'],
    datumKinds: [{kind: 'cell', label: 'raster cell', indexSpace: '2d'}],
    representation: {from: 'object', to: 'field'},
    guards: ['coordinate-model'],
    choices: [
      {stage: 'support', name: 'cellSize', label: 'raster cell size', unit: 'm', sweepable: true},
      {stage: 'operator', name: 'mode', alternatives: ['exact', 'jump-flood'], sweepable: false}
    ],
    formula: {
      formula: 'none',
      reason:
        'Composition of GPUDistanceField and GPURasterZonalStatistics; see each contributor card for its formula.'
    },
    metadata: {coordinateModel: 'projected', metricUnit: 'm'}
  }
};
