// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {SpatialFamilyOperators} from '../types';
import {STATISTICS_CLUSTERING_OPERATORS} from './statistics-clustering';
import {STATISTICS_INFERENCE_OPERATORS} from './statistics-inference';
import {STATISTICS_REGION_OPERATORS} from './statistics-regions';
import {STATISTICS_SUMMARY_OPERATORS} from './statistics-summaries';

/**
 * Spatial computation-pack entries for the `statistics` family, keyed by exported name. Every entry
 * must use `family: 'statistics'`. Sub-family tables live in `statistics-*.ts` and are spread in here.
 */
export const STATISTICS_OPERATORS: SpatialFamilyOperators = {
  GPULocalMoran: {
    family: 'statistics',
    descriptor: {scope: 'focal', scopeSecondary: 'global', reduction: 'weighted-mean'},
    variables: {
      values: 'rate',
      localI: 'score',
      zScores: 'score',
      pValues: 'probability',
      adjustedPValues: 'probability',
      quadrants: 'category',
      spatialLag: 'score',
      neighborCounts: 'count'
    },
    stages: ['neighbourhood', 'operator', 'inference'],
    datumKinds: [{kind: 'region', label: 'tract', indexSpace: 'csr'}],
    guards: ['uncorrected-p', 'classification-needs-inference', 'row-standardisation'],
    choices: [
      {
        name: 'transform',
        label: 'Weights transform',
        stage: 'neighbourhood',
        alternatives: ['binary', 'row'],
        sweepable: true
      },
      {name: 'islands', stage: 'neighbourhood', alternatives: ['exclude', 'keep'], sweepable: true},
      {name: 'minimumNeighbors', stage: 'inference', sweepable: true},
      {
        name: 'multipleTesting',
        stage: 'inference',
        alternatives: ['none', 'bonferroni', 'holm', 'benjamini-hochberg', 'benjamini-yekutieli'],
        sweepable: true
      },
      {name: 'significanceLevel', stage: 'inference', sweepable: true},
      {
        name: 'quadrantGating',
        stage: 'presentation',
        alternatives: ['analytic', 'none'],
        sweepable: false
      }
    ],
    formula: {
      id: 'local-moran',
      derivation: 'curated',
      tex: String.raw`\begin{aligned}
\term{localI}{I_i}&=\frac{(\term{count}{n}-1)\term{centered}{c_i}\term{spatialLag}{L_i}}{\term{sumOfSquares}{M_2}},
&\term{spatialLag}{L_i}&=\sum_j\term{weight}{w_{ij}}\term{neighborCentered}{c_j}\\
\term{zScores}{z_i}&=\frac{\operatorname{sign}(\term{centered}{c_i})(\term{spatialLag}{L_i}-\term{weightSum}{W_i}\term{otherMean}{\mu})}{\sqrt{\term{lagVariance}{V_i}}},
&\term{lagVariance}{V_i}&=\term{otherVariance}{\sigma^2}\frac{\term{others}{N}\term{squareSum}{S_{1i}}-\term{weightSum}{W_i}^2}{\term{others}{N}-1}\\
\term{others}{N}&=\term{count}{n}-1,
&\term{otherMean}{\mu}&=-\frac{\term{centered}{c_i}}{\term{others}{N}},
&\term{otherVariance}{\sigma^2}&=\frac{\term{sumOfSquares}{M_2}-\term{centered}{c_i}^2}{\term{others}{N}}-\term{otherMean}{\mu}^2\\
\term{weightSum}{W_i}&=\sum_j\term{weight}{w_{ij}},
&\term{squareSum}{S_{1i}}&=\sum_j\term{weight}{w_{ij}}^2
\end{aligned}`,
      ascii:
        'I_i = (n-1) c_i L_i / M2; L_i = sum_j w_ij c_j; z_i = sign(c_i) (L_i - W_i mu) / sqrt(V_i); V_i = sigma^2 (N S1_i - W_i^2)/(N-1); N = n-1; mu = -c_i/N; sigma^2 = (M2-c_i^2)/N-mu^2; W_i = sum_j w_ij; S1_i = sum_j w_ij^2. c = x - mean; M2 = sum c^2 over analysis rows. Positive weights to other analysis rows only; row weights are re-standardized over unmasked neighbors. Undefined z when n < 3, c_i = 0, too few neighbors, or nonpositive variance.',
      symbols: [
        {id: 'localI', role: 'output', binding: 'wgsl:localI'},
        {id: 'count', role: 'intermediate', label: 'analysis row count', binding: 'wgsl:total'},
        {id: 'centered', role: 'intermediate', binding: 'wgsl:centered'},
        {id: 'spatialLag', role: 'intermediate', binding: 'wgsl:lag'},
        {id: 'sumOfSquares', role: 'intermediate', binding: 'wgsl:sumOfSquares'},
        {id: 'weight', role: 'input', binding: 'wgsl:weight'},
        {
          id: 'neighborCentered',
          role: 'intermediate',
          label: 'centered neighbor value',
          binding: 'wgsl:statistics'
        },
        {id: 'zScores', role: 'output', binding: 'wgsl:zScore'},
        {id: 'weightSum', role: 'intermediate', binding: 'wgsl:weightSum'},
        {id: 'otherMean', role: 'intermediate', binding: 'wgsl:otherMean'},
        {id: 'lagVariance', role: 'intermediate', binding: 'wgsl:lagVariance'},
        {id: 'otherVariance', role: 'intermediate', binding: 'wgsl:otherVariance'},
        {id: 'others', role: 'intermediate', binding: 'wgsl:others'},
        {id: 'squareSum', role: 'intermediate', binding: 'wgsl:squareSum'}
      ]
    }
  },
  ...STATISTICS_INFERENCE_OPERATORS,
  ...STATISTICS_SUMMARY_OPERATORS,
  ...STATISTICS_REGION_OPERATORS,
  ...STATISTICS_CLUSTERING_OPERATORS
};
