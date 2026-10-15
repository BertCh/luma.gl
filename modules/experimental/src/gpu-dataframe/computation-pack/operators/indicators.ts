// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {DataFrameFamilyOperators} from '../types';

/** Curated composite and inequality formulas, with per-column and per-zone normalization. */
export const INDICATORS_OPERATORS: DataFrameFamilyOperators = {
  /**
   * Primary: min-max scaler and weighted-sum aggregation. Constant columns scale to zero before
   * direction reversal (negative direction gives 1-s). Exclude masked rows and any row with a
   * nonfinite indicator from both scores and column statistics; a zero absolute weight sum is NaN.
   * Other scalers: z-score with population standard deviation (constant=0, reversal=-s), and
   * rank with averaged ties divided by n-1 (a singleton gets 0.5). Geometric aggregation is exp(sum(w*ln(s+epsilon))/
   * sum(w)) for positive weights; invalid shifted values or zero total weight give NaN.
   * Principal-component aggregation projects direction-adjusted z-scores on the correlation
   * matrix's first eigenvector: 64 power iterations starting at 1/sqrt(d), signed to nonnegative
   * loading sum, ignoring weights and scaler. Rank/PCA require their compile-time enable flags.
   */
  GPUCompositeScore: {
    family: 'indicators',
    descriptor: {scope: 'local', scopeSecondary: 'global', reduction: 'weighted-mean'},
    formula: {
      id: 'gpu-dataframe:GPUCompositeScore',
      derivation: 'curated',
      tex: String.raw`\term{scaled}{s_j}=\begin{cases}(\term{indicators}{x_j}-\term{minimum}{a_j})/\term{range}{(b_j-a_j)}&\term{range}{b_j-a_j}>0\\0&\text{otherwise}\end{cases},\quad \term{scaled}{s_j}\leftarrow1-\term{scaled}{s_j}\ \text{if }\term{isReversed}{\mathrm{reversed}_j},\quad \term{score}{C}=\frac{\sum_{\term{column}{j}}\term{weights}{w_j}\term{scaled}{s_j}}{\sum_{\term{column}{j}}|\term{weights}{w_j}|}`,
      ascii:
        'min-max s_j=(x_j-min_j)/(max_j-min_j), constant=0; reverse s_j to 1-s_j for negative direction; score=sum(w_j*s_j)/sum(abs(w_j)).',
      symbols: [
        {id: 'scaled', role: 'intermediate', binding: 'wgsl:scaled'},
        {id: 'indicators', role: 'input', binding: 'wgsl:indicators'},
        {id: 'minimum', role: 'intermediate', binding: 'wgsl:minimum'},
        {id: 'range', role: 'intermediate', binding: 'wgsl:range'},
        {id: 'isReversed', role: 'parameter', binding: 'wgsl:isReversed'},
        {id: 'score', role: 'output', binding: 'wgsl:score'},
        {id: 'column', role: 'index', binding: 'wgsl:column'},
        {id: 'weights', role: 'parameter', binding: 'wgsl:weight'}
      ]
    },
    stages: ['filter', 'summarise', 'derive', 'aggregate'],
    datumKinds: [
      {kind: 'row', label: 'Scored rows', indexSpace: '1d'},
      {kind: 'column', label: 'Indicator columns', indexSpace: '1d'}
    ],
    guards: ['count-vs-rate'],
    choices: [
      'scaler',
      'aggregation',
      'weights',
      'directions',
      'epsilon',
      'indicatorCount',
      'enableRank',
      'enablePrincipalComponent'
    ],
    variables: {
      indicators: {level: 'ratio', extent: 'intensive'},
      score: {level: 'interval', extent: 'intensive'},
      scaled: {level: 'interval', extent: 'intensive'}
    },
    sourceFiles: [
      'modules/experimental/src/gpu-dataframe/composite-indicators/gpu-composite-score.ts'
    ]
  },
  /**
   * Primary: weighted empirical Gini from the sorted Lorenz trapezoids (no n/(n-1) correction).
   * Nonnegative finite values and positive finite weights only, valid zones and nonzero mask;
   * default weight=1. P and L are cumulative population/income shares. Every cut interpolates
   * within a row; knots are L(k/(K-1)). Empty/zero-income zones give NaN indices.
   * With r=x/mu, Theil T=mean_w(r*ln r), Theil L=-mean_w(ln r), Atkinson at epsilon!=1 is
   * 1-mean_w(r^(1-epsilon))^(1/(1-epsilon)); at epsilon=1 it is 1-exp(mean_w(ln r)).
   * Zero values contribute zero to T and to power sums for epsilon<1; any zero makes L and
   * Atkinson for epsilon>=1 NaN. Palma=(1-L(1-palmaTopShare))/L(palmaBottomShare), NaN for a
   * zero denominator. Hoover=sum(w*abs(x-mu))/(2*totalIncome). Global Theil decomposition and
   * pooled Gini are additional outputs. All sums follow the kernel's fixed-order f32 arithmetic.
   */
  GPUInequality: {
    family: 'indicators',
    descriptor: {scope: 'zonal', scopeSecondary: 'global', reduction: 'scan-sort'},
    formula: {
      id: 'gpu-dataframe:GPUInequality',
      derivation: 'curated',
      tex: String.raw`\term{populationShare}{P_i}=\frac{\term{cumulativeWeight}{\sum_{j\leq i}w_j}}{\term{totalWeight}{W}},\quad \term{incomeShare}{L_i}=\frac{\term{cumulativeIncome}{\sum_{j\leq i}w_jx_j}}{\term{totalIncome}{S}},\quad \term{gini}{G}=1-\sum_{\text{value-sorted rows}}(\term{populationShare}{P_i}-\term{previousP}{P_{i-1}})(\term{incomeShare}{L_i}+\term{previousL}{L_{i-1}})`,
      ascii:
        'P_i=cumulativeWeight_i/W; L_i=cumulativeIncome_i/S; Gini=1-sum_sorted((P_i-P_(i-1))*(L_i+L_(i-1))); no finite-sample correction; Lorenz cuts interpolate within rows.',
      symbols: [
        {id: 'populationShare', role: 'intermediate', binding: 'wgsl:nextP'},
        {id: 'cumulativeWeight', role: 'intermediate', binding: 'wgsl:cumulativeWeight'},
        {id: 'totalWeight', role: 'intermediate', binding: 'wgsl:totalWeight'},
        {id: 'incomeShare', role: 'intermediate', binding: 'wgsl:nextL'},
        {id: 'cumulativeIncome', role: 'intermediate', binding: 'wgsl:cumulativeIncome'},
        {id: 'totalIncome', role: 'intermediate', binding: 'wgsl:totalIncome'},
        {id: 'gini', role: 'output', binding: 'wgsl:gini'},
        {id: 'previousP', role: 'intermediate', binding: 'wgsl:previousP'},
        {id: 'previousL', role: 'intermediate', binding: 'wgsl:previousL'}
      ]
    },
    stages: ['filter', 'group', 'order', 'aggregate', 'summarise'],
    datumKinds: [
      {kind: 'row', label: 'Weighted values', indexSpace: '1d'},
      {kind: 'zone', label: 'Income zones', indexSpace: '1d'},
      {kind: 'quantile', label: 'Lorenz population knots', indexSpace: '1d'}
    ],
    guards: [],
    choices: [
      'epsilon',
      'palmaTopShare',
      'palmaBottomShare',
      'zoneCount',
      'lorenzKnotCount',
      'weights'
    ],
    variables: {
      values: {level: 'ratio', extent: 'intensive'},
      weights: {level: 'ratio', extent: 'extensive'},
      zoneIds: {level: 'nominal'},
      gini: {level: 'ratio', extent: 'intensive'},
      lorenzKnots: {level: 'ratio', extent: 'intensive'}
    },
    sourceFiles: [
      'modules/experimental/src/gpu-dataframe/composite-indicators/inequality-wgsl.ts',
      'modules/experimental/src/gpu-dataframe/composite-indicators/gpu-inequality.ts'
    ]
  }
};
