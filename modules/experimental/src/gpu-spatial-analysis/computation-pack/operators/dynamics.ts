// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {SpatialChoiceTemplate} from '../choices';
import type {SpatialFamilyOperators} from '../types';

const CLASSIFICATION_METHODS = [
  'equal-interval',
  'quantile',
  'standard-deviation',
  'head-tail',
  'box-plot',
  'maximum-breaks',
  'natural-breaks',
  'custom'
] as const;

const CLASS_CHOICES: readonly SpatialChoiceTemplate[] = [
  {stage: 'support', name: 'method', alternatives: CLASSIFICATION_METHODS, sweepable: false},
  {stage: 'support', name: 'classCount', label: 'classes k', sweepable: true}
];

const WEIGHTS_STANDARDISATION_CHOICE: SpatialChoiceTemplate = {
  stage: 'neighbourhood',
  name: 'weightsStandardisation',
  alternatives: ['row', 'binary', 'other'],
  sweepable: false
};

/**
 * Spatial computation-pack entries for the `dynamics` family, keyed by exported name. Every entry
 * must use `family: 'dynamics'`.
 */
export const DYNAMICS_OPERATORS: SpatialFamilyOperators = {
  GPUClassAssignment: {
    family: 'dynamics',
    descriptor: {scope: 'local', reduction: 'count'},
    variables: {values: 'rate', breaks: 'rate', classes: 'rank'},
    stages: ['support', 'presentation'],
    datumKinds: [{kind: 'region', label: 'county', indexSpace: '1d'}],
    guards: ['classification-needs-inference'],
    choices: CLASS_CHOICES,
    // Class rule of gpu-class-assignment.ts: `below` counts inner edges breaks[1..count-1] that are <= value.
    formula: {
      id: 'gpu-spatial-analysis.class-assignment',
      tex: String.raw`\term{class}{c(x)} = \#\{\, k \in \{1,\dots,\term{count}{K}-1\} : \term{edge}{b_k} \le \term{value}{x} \,\}`,
      ascii:
        'class(x) = #{ k in 1..K-1 : b_k <= x }. Ties go to the upper class (the comparison is <=). ' +
        'A NaN or masked value gets class 0xffffffff (no class).',
      symbols: [
        {id: 'class', role: 'output', label: 'class', binding: 'wgsl:below'},
        {id: 'count', role: 'parameter', label: 'K', binding: 'wgsl:count'},
        {id: 'edge', role: 'input', label: 'b_k', binding: 'wgsl:breaks'},
        {id: 'value', role: 'input', label: 'x', binding: 'wgsl:value'}
      ],
      derivation: 'curated'
    },
    // A class map carries no significance test: the classification-needs-inference guard stays failed.
    metadata: {significanceAvailable: false}
  },
  GPUClassificationFit: {
    family: 'dynamics',
    descriptor: {scope: 'zonal', reduction: 'sum'},
    variables: {
      values: 'rate',
      classes: 'rank',
      counts: 'count',
      medians: 'rate',
      absoluteDeviations: 'rate',
      squaredDeviations: 'rate',
      summary: 'score'
    },
    stages: ['operator', 'presentation'],
    datumKinds: [{kind: 'class', label: 'class', indexSpace: '1d'}],
    guards: ['classification-needs-inference'],
    choices: CLASS_CHOICES,
    // gpu-classification-fit.ts `summary` kernel: GADF = 1 - ADCM / ADAM (1 when ADAM is 0), medians from the `stats` kernel.
    formula: {
      id: 'gpu-spatial-analysis.classification-fit',
      tex: String.raw`\term{gadf}{\mathrm{GADF}} = 1 - \frac{\term{adcm}{\mathrm{ADCM}}}{\term{adam}{\mathrm{ADAM}}}, \quad \mathrm{ADCM} = \sum_{c}\sum_{i \in c} \lvert \term{value}{x_i} - \term{classMedian}{m_c} \rvert, \quad \mathrm{ADAM} = \sum_{i} \lvert x_i - m \rvert`,
      ascii:
        'GADF = 1 - ADCM / ADAM; ADCM = sum over classes c of sum over rows i in c of |x_i - median_c|; ' +
        'ADAM = sum over all counted rows of |x_i - median of all rows|. GADF = 1 when ADAM = 0. ' +
        'Only rows with a class below k, an unmasked flag and a finite value are counted; medians are exact (mean of the two middle values).',
      symbols: [
        {id: 'gadf', role: 'output', label: 'GADF', binding: 'wgsl:summary'},
        {id: 'adcm', role: 'intermediate', label: 'ADCM', binding: 'wgsl:adcm'},
        {id: 'adam', role: 'intermediate', label: 'ADAM', binding: 'wgsl:adam'},
        {id: 'value', role: 'input', label: 'x_i', binding: 'wgsl:values'},
        {id: 'classMedian', role: 'intermediate', label: 'm_c', binding: 'wgsl:median'}
      ],
      derivation: 'curated'
    },
    metadata: {significanceAvailable: false}
  },
  GPUTransitionMatrix: {
    family: 'dynamics',
    descriptor: {scope: 'global', reduction: 'histogram'},
    variables: {
      classes: 'rank',
      counts: 'count',
      probabilities: 'probability',
      rowTotals: 'count',
      ignored: 'count'
    },
    stages: ['support', 'operator'],
    datumKinds: [{kind: 'transition', label: 'transition cell', indexSpace: '1d'}],
    guards: [],
    choices: [
      {stage: 'support', name: 'classCount', label: 'classes K', sweepable: true},
      {stage: 'operator', name: 'periodLag', label: 'period lag', sweepable: true}
    ],
    // gpu-transition-matrix.ts `count` and `normalize` kernels: exact integer counts, then n_ab / n_a (0 for an empty row).
    formula: {
      id: 'gpu-spatial-analysis.transition-matrix',
      tex: String.raw`\term{probability}{p_{ab}} = \frac{\term{count}{n_{ab}}}{\term{rowTotal}{n_{a}}}, \quad n_{ab} = \#\{(t, i) : \term{fromClass}{x_{t,i}} = a,\ \term{toClass}{x_{t+\ell,i}} = b\}, \quad n_a = \sum_b n_{ab}`,
      ascii:
        'p_ab = n_ab / n_a; n_ab = #{(t, i) : class(t, i) = a and class(t + lag, i) = b}, t = 0 .. T - lag - 1; ' +
        'n_a = sum over b of n_ab. p_ab = 0 when n_a = 0. Pairs with a masked row or a class >= K are not counted (see ignored).',
      symbols: [
        {id: 'probability', role: 'output', label: 'p_ab', binding: 'wgsl:probabilities'},
        {id: 'count', role: 'intermediate', label: 'n_ab', binding: 'wgsl:counts'},
        {id: 'rowTotal', role: 'intermediate', label: 'n_a', binding: 'wgsl:total'},
        {id: 'fromClass', role: 'input', label: 'class at t', binding: 'wgsl:fromClass'},
        {id: 'toClass', role: 'input', label: 'class at t + lag', binding: 'wgsl:toClass'}
      ],
      derivation: 'curated'
    }
  },
  GPUTransitionDiagnostics: {
    family: 'dynamics',
    descriptor: {scope: 'global', reduction: 'sum'},
    variables: {
      counts: 'count',
      homogeneity: 'score',
      steadyStates: 'probability',
      stateSparsity: 'count',
      converged: 'category'
    },
    stages: ['operator', 'inference'],
    datumKinds: [{kind: 'transition', label: 'transition cell', indexSpace: '1d'}],
    guards: ['uncorrected-p'],
    choices: [
      {stage: 'support', name: 'conditionCount', label: 'conditions C', sweepable: false},
      {stage: 'inference', name: 'minimumTransitions', sweepable: true}
    ],
    // gpu-transition-diagnostics.ts `summary` kernel: the Q branch of `homogeneity`; the p-value is the upper chi-square tail.
    formula: {
      id: 'gpu-spatial-analysis.transition-diagnostics',
      tex: String.raw`\term{q}{Q} = \sum_{c,a,b} \frac{(\term{observed}{n_{cab}} - \term{expected}{e_{cab}})^2}{e_{cab}}, \quad \term{expected}{e_{cab}} = \term{regimeTotal}{n_{ca}}\,\frac{\term{pooled}{n_{ab}}}{\term{pooledTotal}{n_{a}}}, \quad p = 1 - F_{\chi^2}(Q;\ \term{dof}{\nu})`,
      ascii:
        'Q = sum over conditions c, from-classes a, to-classes b of (n_cab - e_cab)^2 / e_cab, e_cab = n_ca * n_ab / n_a, ' +
        'n_ab pooled over conditions. Terms with n_ab = 0 are skipped; a from-class without transitions is skipped. ' +
        'nu = sum over a of (regimes_a - 1)(nonZeroCells_a - 1). p = upper chi-square tail of Q (one-sided, greater); no multiple-testing correction.',
      symbols: [
        {id: 'q', role: 'output', label: 'Q', binding: 'wgsl:q'},
        {id: 'observed', role: 'input', label: 'n_cab', binding: 'wgsl:observed'},
        {id: 'expected', role: 'intermediate', label: 'e_cab', binding: 'wgsl:expected'},
        {id: 'regimeTotal', role: 'intermediate', label: 'n_ca', binding: 'wgsl:regimeTotal'},
        {id: 'pooled', role: 'intermediate', label: 'n_ab', binding: 'wgsl:pooledRow'},
        {id: 'pooledTotal', role: 'intermediate', label: 'n_a', binding: 'wgsl:pooledTotal'},
        {id: 'dof', role: 'intermediate', label: 'nu', binding: 'wgsl:degreesOfFreedom'}
      ],
      derivation: 'curated'
    },
    metadata: {pCorrection: 'none'}
  },
  GPUSpatialMarkov: {
    family: 'dynamics',
    descriptor: {scope: 'focal', reduction: 'histogram'},
    variables: {
      values: 'rate',
      classes: 'rank',
      lagValues: 'rate',
      lagClasses: 'rank',
      lagBreaks: 'rate',
      counts: 'count',
      probabilities: 'probability',
      rowTotals: 'count',
      homogeneity: 'score',
      stateSparsity: 'count'
    },
    stages: ['neighbourhood', 'operator', 'inference'],
    datumKinds: [{kind: 'transition', label: 'conditional transition cell', indexSpace: '1d'}],
    guards: ['row-standardisation', 'uncorrected-p'],
    choices: [
      WEIGHTS_STANDARDISATION_CHOICE,
      {stage: 'support', name: 'classCount', label: 'classes K', sweepable: true},
      {stage: 'support', name: 'lagClassCount', label: 'lag classes', sweepable: true},
      {stage: 'support', name: 'lagMethod', alternatives: CLASSIFICATION_METHODS, sweepable: false}
    ],
    // gpu-spatial-markov.ts chain: GPUSpatialLag (normalize) -> pooled GPUClassBreaks -> GPUClassAssignment -> GPUTransitionMatrix with the lag class as condition.
    formula: {
      id: 'gpu-spatial-analysis.spatial-markov',
      tex: String.raw`\term{probability}{p_{ab\mid c}} = \frac{\term{count}{n_{abc}}}{\term{rowTotal}{n_{ac}}}, \quad n_{abc} = \#\{(t, i) : x_{t,i} = a,\ x_{t+1,i} = b,\ \term{lagClass}{\mathrm{class}(L_{t,i})} = c\}, \quad \term{lag}{L_{t,i}} = \frac{\sum_j w_{ij}\, y_{t,j}}{\sum_j w_{ij}}`,
      ascii:
        'p_ab|c = n_abc / n_ac; n_abc = #{(t, i): class(t, i) = a, class(t + 1, i) = b, lag class of L(t, i) = c}; ' +
        'L(t, i) = sum_j w_ij y(t, j) / sum_j w_ij over the included neighbours (normalize = true). ' +
        'Lag classes use one set of breaks pooled over all periods. p = 0 for a (c, a) state without transitions.',
      symbols: [
        {id: 'probability', role: 'output', label: 'p_ab|c', binding: 'wgsl:probabilities'},
        {id: 'count', role: 'intermediate', label: 'n_abc', binding: 'wgsl:counts'},
        {id: 'rowTotal', role: 'intermediate', label: 'n_ac', binding: 'wgsl:rowTotals'},
        {id: 'lagClass', role: 'intermediate', label: 'lag class', binding: 'wgsl:lagClasses'},
        {id: 'lag', role: 'intermediate', label: 'L', binding: 'wgsl:lagValues'}
      ],
      derivation: 'curated'
    },
    // The homogeneity p-values are single analytic chi-square tests, uncorrected.
    metadata: {pCorrection: 'none'}
  },
  GPULISAMarkov: {
    family: 'dynamics',
    descriptor: {scope: 'focal', reduction: 'histogram'},
    variables: {
      values: 'rate',
      quadrants: 'category',
      counts: 'count',
      probabilities: 'probability',
      rowTotals: 'count',
      ignored: 'count'
    },
    stages: ['neighbourhood', 'operator'],
    datumKinds: [{kind: 'transition', label: 'LISA transition cell', indexSpace: '1d'}],
    guards: ['classification-needs-inference', 'row-standardisation'],
    choices: [
      WEIGHTS_STANDARDISATION_CHOICE,
      {stage: 'support', name: 'gating', alternatives: ['none', 'significance'], sweepable: false},
      {
        stage: 'inference',
        name: 'significanceAvailable',
        alternatives: [true, false],
        sweepable: false
      }
    ],
    // gpu-lisa-markov.ts: `states` kernel (GPULocalMoran quadrant per period) feeding GPUTransitionMatrix with 5 states.
    formula: {
      id: 'gpu-spatial-analysis.lisa-markov',
      tex: String.raw`\term{probability}{p_{ab}} = \frac{\term{count}{n_{ab}}}{n_a}, \quad n_{ab} = \#\{(t, i) : \term{state}{s_{t,i}} = a,\ s_{t+1,i} = b\}, \quad s_{t,i} \in \{\mathrm{HH}, \mathrm{LH}, \mathrm{LL}, \mathrm{HL}\}`,
      ascii:
        'p_ab = n_ab / n_a; n_ab = #{(t, i): s(t, i) = a, s(t + 1, i) = b}. s(t, i) is the local Moran quadrant of county i in period t ' +
        '(signs of the centred value and of the spatial lag; HH=1, LH=2, LL=3, HL=4; state 0 stays empty without significance gating). ' +
        'Rows without a finite value or a neighbour have no state and are counted in ignored.',
      symbols: [
        {id: 'probability', role: 'output', label: 'p_ab', binding: 'wgsl:probabilities'},
        {id: 'count', role: 'intermediate', label: 'n_ab', binding: 'wgsl:counts'},
        {id: 'state', role: 'intermediate', label: 's', binding: 'wgsl:quadrants'}
      ],
      derivation: 'curated'
    }
  },
  GPUKnoxTest: {
    family: 'dynamics',
    descriptor: {scope: 'pairwise', reduction: 'count'},
    variables: {times: 'measure', statistics: 'count', summary: 'probability'},
    stages: ['neighbourhood', 'operator', 'inference'],
    datumKinds: [{kind: 'permutation', label: 'permutation (0 = observed)', indexSpace: '1d'}],
    guards: ['coordinate-model', 'uncorrected-p'],
    choices: [
      {stage: 'neighbourhood', name: 'radius', label: 'spatial radius', unit: 'm', sweepable: true},
      {
        stage: 'neighbourhood',
        name: 'timeThreshold',
        label: 'time threshold',
        unit: 'days',
        sweepable: true
      },
      {stage: 'inference', name: 'permutationCount', label: 'permutations P', sweepable: true},
      {stage: 'inference', name: 'seed', sweepable: false}
    ],
    // gpu-knox-test.ts `partials`/`reduce` kernels and space-time-kernels.ts getSpaceTimeSummaryBodyWGSL (pGreater, one-sided).
    formula: {
      id: 'gpu-spatial-analysis.knox-test',
      tex: String.raw`\term{x}{X} = \sum_{(i,j) \in \term{pairs}{S}} \mathbf{1}\big[\,\lvert t_i - t_j \rvert \le \term{threshold}{\tau}\,\big], \qquad \term{p}{p} = \frac{1 + \term{greater}{\#\{s : X_s \ge X\}}}{\term{permutations}{P} + 1}, \qquad \term{expected}{E[X]} = \frac{\lvert S \rvert\, T}{n(n-1)/2}`,
      ascii:
        'X = number of listed spatial pairs (i < j) with |t_i - t_j| <= tau. p = (1 + #{permutations s with X_s >= X}) / (P + 1): ' +
        'one-sided (excess of close pairs), the +1 counts the observed value, ties count as exceedances. ' +
        'E[X] = |S| T / (n (n - 1) / 2) with T the number of time-close pairs among all n (n - 1) / 2 pairs. Times are permuted over fixed locations (swap-or-not shuffle, fixed seed).',
      symbols: [
        {id: 'x', role: 'output', label: 'X', binding: 'wgsl:statistics'},
        {id: 'pairs', role: 'input', label: 'S', binding: 'wgsl:pairMask'},
        {id: 'threshold', role: 'parameter', label: 'tau', binding: 'wgsl:threshold'},
        {id: 'p', role: 'output', label: 'p', binding: 'wgsl:pGreater'},
        {id: 'greater', role: 'intermediate', label: '#{X_s >= X}', binding: 'wgsl:greater'},
        {id: 'permutations', role: 'parameter', label: 'P', binding: 'wgsl:permutations'},
        {id: 'expected', role: 'intermediate', label: 'E[X]', binding: 'wgsl:expected'}
      ],
      derivation: 'curated'
    },
    // Monte Carlo pseudo p-value of one test, no multiple-testing correction; pairs come from a planar radius search.
    metadata: {pCorrection: 'none', coordinateModel: 'projected', metricUnit: 'm'}
  },
  GPUMantelTest: {
    family: 'dynamics',
    descriptor: {scope: 'pairwise', reduction: 'sum'},
    variables: {times: 'measure', statistics: 'score', summary: 'probability'},
    stages: ['neighbourhood', 'operator', 'inference'],
    datumKinds: [{kind: 'permutation', label: 'permutation (0 = observed)', indexSpace: '1d'}],
    guards: ['coordinate-model', 'uncorrected-p'],
    choices: [
      {stage: 'neighbourhood', name: 'radius', label: 'distance band', unit: 'm', sweepable: true},
      {stage: 'inference', name: 'permutationCount', label: 'permutations P', sweepable: true},
      {stage: 'inference', name: 'seed', sweepable: false}
    ],
    // gpu-mantel-test.ts `reduce` kernel (r from centred sums) and getSpaceTimeSummaryBodyWGSL (pGreater, one-sided).
    formula: {
      id: 'gpu-spatial-analysis.mantel-test',
      tex: String.raw`\term{r}{r} = \frac{\term{covariance}{S_{xy}}}{\sqrt{\term{varianceX}{S_{xx}}\,\term{varianceY}{S_{yy}}}}, \quad S_{xy} = \sum xy - \frac{\sum x \sum y}{\term{pairCount}{N}}, \qquad \term{p}{p} = \frac{1 + \#\{s : r_s \ge r\}}{P + 1}`,
      ascii:
        'r = S_xy / sqrt(S_xx S_yy) over the N listed pairs i < j, x = spatial distance, y = |t_i - t_j|; S_xy = sum xy - sum x sum y / N, ' +
        'S_xx = sum xx - (sum x)^2 / N, S_yy likewise. r = 0 when N <= 1 or a variance is 0. ' +
        'p = (1 + #{permutations s with r_s >= r}) / (P + 1): one-sided (positive space-time correlation), +1 counts the observed value, ties count as exceedances.',
      symbols: [
        {id: 'r', role: 'output', label: 'r', binding: 'wgsl:result'},
        {id: 'covariance', role: 'intermediate', label: 'S_xy', binding: 'wgsl:covariance'},
        {id: 'varianceX', role: 'intermediate', label: 'S_xx', binding: 'wgsl:varianceX'},
        {id: 'varianceY', role: 'intermediate', label: 'S_yy', binding: 'wgsl:varianceY'},
        {id: 'pairCount', role: 'intermediate', label: 'N', binding: 'wgsl:count'},
        {id: 'p', role: 'output', label: 'p', binding: 'wgsl:pGreater'}
      ],
      derivation: 'curated'
    },
    metadata: {pCorrection: 'none', coordinateModel: 'projected', metricUnit: 'm'}
  },
  GPUSpatialScanStatistic: {
    family: 'dynamics',
    descriptor: {scope: 'focal', reduction: 'select'},
    variables: {
      positions: 'measure',
      cases: 'count',
      baseline: 'count',
      zoneStatistics: 'score',
      clusterStatistics: 'score',
      statistics: 'score'
    },
    stages: ['support', 'operator', 'inference'],
    datumKinds: [{kind: 'zone', label: 'zone (county)', indexSpace: '1d'}],
    guards: ['coordinate-model', 'uncorrected-p'],
    choices: [
      {
        stage: 'support',
        name: 'maximumPopulationFraction',
        label: 'largest window share',
        sweepable: true
      },
      {
        stage: 'support',
        name: 'windowShape',
        alternatives: ['circle', 'nearest'],
        sweepable: false
      },
      {
        stage: 'operator',
        name: 'rateDirection',
        alternatives: ['high', 'low', 'both'],
        sweepable: false
      },
      {stage: 'inference', name: 'permutationCount', label: 'replicates P', sweepable: true},
      {stage: 'inference', name: 'seed', sweepable: false}
    ],
    // gpu-spatial-scan-statistic.ts getLogLikelihoodRatio (Poisson) and the cluster p-value (exceeding + 1) / (permutations + 1).
    formula: {
      id: 'gpu-spatial-analysis.spatial-scan-statistic',
      tex: String.raw`\term{llr}{\Lambda} = c \ln\frac{c}{\term{expected}{e}} + (\term{totalCases}{C} - c)\ln\frac{C - c}{C - e}, \quad e = \term{windowWeight}{W}\,\frac{C}{\term{totalWeight}{B}}, \quad p = \frac{1 + \term{exceeding}{\#\{s : \Lambda_s^{\max} \ge \Lambda\}}}{P + 1}`,
      ascii:
        'Poisson model, high rates: for a window with c observed cases and baseline weight W, e = W C / B, ' +
        'LLR = c ln(c / e) + (C - c) ln((C - c) / (C - e)) when c > e, else 0. The cluster is the window with the largest LLR. ' +
        'p = (1 + #{replicates s with maximum LLR_s >= LLR}) / (P + 1): one-sided, the maximum over all windows already accounts for the search, ties count as exceedances.',
      symbols: [
        {id: 'llr', role: 'output', label: 'LLR', binding: 'wgsl:getLogLikelihoodRatio'},
        {id: 'expected', role: 'intermediate', label: 'e', binding: 'wgsl:expected'},
        {id: 'totalCases', role: 'parameter', label: 'C', binding: 'wgsl:totalCases'},
        {id: 'windowWeight', role: 'intermediate', label: 'W', binding: 'wgsl:windowWeight'},
        {id: 'totalWeight', role: 'parameter', label: 'B', binding: 'wgsl:totalWeight'},
        {
          id: 'exceeding',
          role: 'intermediate',
          label: '#{max LLR_s >= LLR}',
          binding: 'wgsl:exceeding'
        }
      ],
      derivation: 'curated'
    },
    metadata: {pCorrection: 'none', coordinateModel: 'projected', metricUnit: 'm'}
  }
};
