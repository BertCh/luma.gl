// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {DataFrameFamilyOperators} from '../types';

/** Curated grouped statistics and key lookup formulas with the actual missing-key policy. */
export const GROUPS_JOINS_OPERATORS: DataFrameFamilyOperators = {
  /**
   * Primary: corrected two-pass variance; mean0 is the signed 64-bit fixed-point sum divided by
   * sumScale*n. For d=x-mean0, S1=sum(d), S2=sum(d*d), M2=max(S2-S1*S1/n,0), snapped to zero
   * when M2<=2^-19*S2. Sample variance divides by n-1 (NaN for n<2); population divides by n.
   * Finite values only; keys/masks/capacity select the group. Other statistics include fixed-point
   * sums/means, extrema, Fisher-Pearson skewness, excess kurtosis, linear R-7 median/percentiles,
   * smallest-value mode on ties, unique counts and z-scores. Order statistics use the shared
   * separately rounded finite-value policy, including the overflow-safe interpolation fallback.
   * No conservation guard: the generic input column is not declared to be an extensive amount.
   */
  GPUGroupStatistics: {
    family: 'groups-joins',
    // Primary moments sum within keyed groups; the optional order statistics instead select.
    descriptor: {scope: 'zonal', reduction: 'sum'},
    formula: {
      id: 'gpu-dataframe:GPUGroupStatistics',
      derivation: 'curated',
      tex: String.raw`\term{m2}{M_2}=\max\left(\term{momentSums}{S_2-S_1^2/n},0\right),\quad \term{m2}{M_2}\leftarrow0\ \text{if }\term{m2}{M_2}\leq2^{-19}\term{momentSums}{S_2},\quad \term{variances}{v}=\frac{\term{m2}{M_2}}{\term{counts}{n}-\mathbf{1}[\term{sampleVariance}{\mathrm{sample}}]}`,
      ascii:
        'd=x-mean0; M2=max(sum(d*d)-sum(d)^2/n,0), snapped to 0 if <=2^-19*sum(d*d); variance=M2/(n-(variance==sample)).',
      symbols: [
        {id: 'm2', role: 'intermediate', binding: 'wgsl:m2'},
        {id: 'momentSums', role: 'intermediate', binding: 'wgsl:momentSums'},
        {id: 'variances', role: 'output', binding: 'wgsl:variancesOut'},
        {id: 'counts', role: 'output', binding: 'wgsl:countsOut'},
        {id: 'sampleVariance', role: 'constant', binding: 'wgsl:SAMPLE_VARIANCE'}
      ]
    },
    stages: ['filter', 'group', 'aggregate', 'order', 'summarise'],
    datumKinds: [
      {kind: 'row', label: 'Rows', indexSpace: '1d'},
      {kind: 'group', label: 'Key groups', indexSpace: '1d'},
      {kind: 'quantile', label: 'Group percentiles', indexSpace: '1d'}
    ],
    guards: [],
    choices: ['statistics', 'variance', 'percentiles', 'sumScale', 'keyCount'],
    variables: {
      keys: {level: 'nominal'},
      values: {level: 'ratio', extent: 'intensive'},
      counts: {level: 'ratio', extent: 'extensive'},
      means: {level: 'ratio', extent: 'intensive'}
    },
    sourceFiles: [
      'modules/experimental/src/gpu-dataframe/group-statistics/group-statistics-moments.ts',
      'modules/experimental/src/gpu-dataframe/group-statistics/group-statistics-order.ts',
      'modules/experimental/src/gpu-dataframe/column-classification/order-statistic-wgsl.ts'
    ]
  },
  /**
   * Primary: smallest matching right row. Right keys need NOT be unique: the index stores unique
   * keys plus ranges, gather picks the first row, and aggregates consume the whole range.
   * All-ones keys (both words for uint64) and masked rows never match. Empty matches write
   * 0xffffffff; gathered values write NaN/all-ones. Inner joins also compact matched left IDs.
   */
  GPUKeyJoin: {
    family: 'groups-joins',
    descriptor: {scope: 'pairwise', scopeSecondary: 'zonal', reduction: 'select'},
    formula: {
      id: 'gpu-dataframe:GPUKeyJoin',
      derivation: 'curated',
      tex: String.raw`\term{rightRows}{m_i}=\min\{\term{rightRow}{j}:\term{rightKeys}{k^R_j}=\term{leftKeys}{k^L_i},\ \text{both rows valid}\},\quad \min\varnothing=\mathtt{0xffffffff}`,
      ascii:
        'rightRows[i]=smallest valid right row j with rightKeys[j]==leftKeys[i], else 0xffffffff; null keys never match; duplicate right keys are supported.',
      symbols: [
        {id: 'rightRows', role: 'output', binding: 'wgsl:firstOut'},
        {id: 'rightRow', role: 'index', binding: 'wgsl:sortedRows'},
        {id: 'rightKeys', role: 'input', binding: 'wgsl:rightKeys'},
        {id: 'leftKeys', role: 'input', binding: 'wgsl:leftKeys'}
      ]
    },
    stages: ['filter', 'order', 'group', 'join', 'select'],
    datumKinds: [
      {kind: 'row', label: 'Left rows', indexSpace: '1d'},
      {kind: 'group', label: 'Unique right keys', indexSpace: '1d'},
      {kind: 'pair', label: 'Matching rows', indexSpace: '1d'}
    ],
    guards: [],
    choices: ['kind', 'gather', 'aggregates', 'operation', 'sumScale'],
    variables: {
      leftKeys: {level: 'nominal'},
      rightKeys: {level: 'nominal'},
      rightRows: {level: 'nominal'},
      matched: {level: 'nominal'},
      matchCounts: {level: 'ratio', extent: 'extensive'}
    },
    sourceFiles: ['modules/experimental/src/gpu-dataframe/key-join/gpu-key-join.ts']
  }
};
