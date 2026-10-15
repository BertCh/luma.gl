// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {SpatialFamilyOperators} from '../types';

/**
 * Spatial computation-pack entries for the `statistics` family (autocorrelation and inference), keyed by exported name.
 * Merged into `STATISTICS_OPERATORS`. Every entry must use `family: 'statistics'`.
 */
export const STATISTICS_INFERENCE_OPERATORS: SpatialFamilyOperators = {
  GPUGlobalSpatialStatistics: {
    family: 'statistics',
    descriptor: {scope: 'global', scopeSecondary: 'pairwise', reduction: 'sum'},
    variables: {values: 'rate', summary: 'score'},
    stages: ['neighbourhood', 'operator', 'inference'],
    datumKinds: [{kind: 'region', label: 'tract', indexSpace: 'csr'}],
    guards: ['uncorrected-p'],
    choices: [
      {
        name: 'transform',
        label: 'Weights transform',
        stage: 'neighbourhood',
        alternatives: ['binary', 'row'],
        sweepable: true
      },
      {name: 'islands', stage: 'neighbourhood', alternatives: ['exclude', 'keep'], sweepable: true},
      {
        name: 'alternative',
        label: 'Tail of the analytic p-value',
        stage: 'inference',
        alternatives: ['two-sided', 'greater', 'lesser'],
        sweepable: true
      }
    ],
    formula: {
      id: 'global-moran',
      derivation: 'curated',
      tex: String.raw`\begin{aligned}
\term{statistic}{I}&=\frac{\term{count}{n}\,\term{crossSum}{\sum_{ij}w_{ij}c_ic_j}}{\term{weightSum}{S_0}\,\term{sumSquares}{\sum_i c_i^2}},
&\term{expected}{E[I]}&=-\frac{1}{\term{count}{n}-1}\\
\term{zRandomization}{z}&=\frac{\term{statistic}{I}-\term{expected}{E[I]}}{\sqrt{\term{varianceRandomization}{V_{\mathrm{rand}}}}},
&\term{pValue}{p}&=\operatorname{erfc}\left(\frac{|\term{zRandomization}{z}|}{\sqrt{2}}\right)
\end{aligned}`,
      ascii:
        'I = n * sum_ij w_ij c_i c_j / (S0 * sum c_i^2); E[I] = -1/(n-1); z = (I - E[I]) / sqrt(V_rand); p = erfc(|z| / sqrt(2)) for the default two-sided tail. c = x - mean over the n analysis rows; S0 = sum w_ij over included pairs; V_rand is esda VI_rand (sample kurtosis). Geary C and General G use the same z and p with their own moments. For G, E[G] = S0/(n(n-1)); both inference triples report the analytical VG and normal approximation, evaluated in centered form. Binary weights and islands = keep match esda.G; row weights give a weighted generalization. Inference is undefined for n < 4 or S0 = 0. Constant positive data give G variance zero (z and p undefined); G also requires non-negative values and a positive pair-product denominator.',
      symbols: [
        {id: 'statistic', role: 'output', label: 'Moran I', binding: 'wgsl:statistic'},
        {id: 'count', role: 'intermediate', label: 'analysis rows', binding: 'wgsl:n'},
        {id: 'crossSum', role: 'intermediate', binding: 'wgsl:COLUMN_MORAN_CROSS'},
        {id: 'weightSum', role: 'intermediate', binding: 'wgsl:s0'},
        {id: 'sumSquares', role: 'intermediate', binding: 'wgsl:sumSquares'},
        {id: 'expected', role: 'intermediate', binding: 'wgsl:expected'},
        {
          id: 'varianceRandomization',
          role: 'intermediate',
          binding: 'wgsl:varianceRandomization'
        },
        {id: 'zRandomization', role: 'output', binding: 'wgsl:zRandomization'},
        {id: 'pValue', role: 'output', binding: 'wgsl:getPValue'}
      ]
    }
  },
  GPUHotSpotAnalysis: {
    family: 'statistics',
    descriptor: {scope: 'focal', scopeSecondary: 'global', reduction: 'weighted-mean'},
    variables: {
      values: 'rate',
      zScores: 'score',
      bins: 'rank',
      pValues: 'probability',
      adjustedPValues: 'probability',
      neighborCounts: 'count',
      globalStatistics: 'score'
    },
    stages: ['neighbourhood', 'operator', 'inference'],
    datumKinds: [{kind: 'region', label: 'tract', indexSpace: 'csr'}],
    guards: ['uncorrected-p', 'classification-needs-inference'],
    choices: [
      {
        name: 'transform',
        label: 'Weights transform',
        stage: 'neighbourhood',
        alternatives: ['binary', 'row'],
        sweepable: true
      },
      {name: 'selfWeight', stage: 'neighbourhood', alternatives: [0, 1], sweepable: true},
      {name: 'islands', stage: 'neighbourhood', alternatives: ['exclude', 'keep'], sweepable: true},
      {name: 'minimumNeighbors', stage: 'inference', sweepable: true},
      {
        name: 'multipleTesting',
        stage: 'inference',
        alternatives: ['none', 'bonferroni', 'holm', 'benjamini-hochberg', 'benjamini-yekutieli'],
        sweepable: true
      }
    ],
    formula: {
      id: 'hot-spot-gi-star',
      derivation: 'curated',
      tex: String.raw`\begin{aligned}
\term{zScores}{z_i}&=\frac{\term{weightedSum}{\sum_j w_{ij}c_j}}{\sqrt{\term{variance}{S^2}}\sqrt{\term{spread}{\Delta_i}}},
&\term{spread}{\Delta_i}&=\frac{\term{count}{n}\,\term{squareSum}{S_{1i}}-\term{weightSum}{W_i}^2}{\term{count}{n}-1}\\
\term{weightSum}{W_i}&=\sum_j w_{ij}+\term{selfWeight}{w_{ii}},
&\term{squareSum}{S_{1i}}&=\sum_j w_{ij}^2+\term{selfWeight}{w_{ii}}^2\\
\term{centered}{c_j}&=x_j-\bar{x},
&\term{weightedSum}{\sum_j w_{ij}c_j}&=\sum_{j\ne i}w_{ij}c_j+\term{selfWeight}{w_{ii}}c_i
\end{aligned}`,
      ascii:
        'z_i = sum_j w_ij c_j / (S sqrt(Delta_i)); Delta_i = (n S1_i - W_i^2)/(n-1); W_i = sum_j w_ij + w_ii; S1_i = sum_j w_ij^2 + w_ii^2; c_j = x_j - mean; j runs over active neighbors, w_ii = selfWeight (1 = Gi*). S is the population standard deviation of the n analysis rows. Row-standardized weights are re-divided by what remains of the row sum. With selfWeight = 0 (Gi) the mean and variance of the n-1 other rows replace X and S^2. NaN when n < 2, S = 0, n S1_i <= W_i^2 or fewer than minimumNeighbors neighbors. Two-sided normal p-value; bins use p (or the adjusted p) at 0.10, 0.05, 0.01.',
      symbols: [
        {id: 'zScores', role: 'output', binding: 'wgsl:zScores'},
        {id: 'weightedSum', role: 'intermediate', binding: 'wgsl:weightedSum'},
        {id: 'variance', role: 'intermediate', binding: 'wgsl:variance'},
        {id: 'spread', role: 'intermediate', binding: 'wgsl:spread'},
        {id: 'count', role: 'intermediate', label: 'analysis rows', binding: 'wgsl:count'},
        {id: 'squareSum', role: 'intermediate', binding: 'wgsl:squareSum'},
        {id: 'weightSum', role: 'intermediate', binding: 'wgsl:weightSum'},
        {id: 'selfWeight', role: 'parameter', binding: 'wgsl:SELF_WEIGHT'},
        {id: 'centered', role: 'intermediate', binding: 'wgsl:centered'}
      ]
    }
  },
  GPULocalGeary: {
    family: 'statistics',
    descriptor: {scope: 'focal', scopeSecondary: 'global', reduction: 'weighted-mean'},
    variables: {values: 'rate', statistic: 'score', globalStatistics: 'score'},
    stages: ['neighbourhood', 'operator'],
    datumKinds: [{kind: 'region', label: 'tract', indexSpace: 'csr'}],
    guards: ['classification-needs-inference'],
    choices: [
      {
        name: 'transform',
        label: 'Weights transform',
        stage: 'neighbourhood',
        alternatives: ['binary', 'row'],
        sweepable: true
      },
      {name: 'islands', stage: 'neighbourhood', alternatives: ['exclude', 'keep'], sweepable: true}
    ],
    formula: {
      id: 'local-geary',
      derivation: 'curated',
      tex: String.raw`\begin{aligned}
\term{result}{c_i}&=\frac{\term{sum}{\sum_j w_{ij}(x_i-x_j)^2}}{\term{variance}{\sigma^2}},
&\term{sum}{\sum_j w_{ij}(x_i-x_j)^2}&=\sum_j\term{weight}{w_{ij}}\left(\term{focus}{x_i}-\term{other}{x_j}\right)^2
\end{aligned}`,
      ascii:
        'c_i = sum_j w_ij (x_i - x_j)^2 / sigma^2, over active neighbors j; sigma^2 is the population variance of the analysis rows, so this is the sum of squared differences of standardized values. Row-standardized weights are divided by what remains of the row sum, so masked neighbors drop out. Small c_i means similar neighbors. NaN for excluded rows, for islands under islands = exclude (0 under keep) and for a constant variable. There is no analytic p-value or class.',
      symbols: [
        {id: 'result', role: 'output', binding: 'wgsl:result'},
        {id: 'sum', role: 'intermediate', binding: 'wgsl:sum'},
        {id: 'variance', role: 'intermediate', binding: 'wgsl:variance'},
        {id: 'weight', role: 'input', binding: 'wgsl:weight'},
        {id: 'focus', role: 'intermediate', binding: 'wgsl:focus'},
        {id: 'other', role: 'intermediate', binding: 'wgsl:other'}
      ]
    }
  },
  GPUSpatialPearson: {
    family: 'statistics',
    descriptor: {scope: 'pairwise', scopeSecondary: 'global', reduction: 'weighted-mean'},
    variables: {values: 'rate', secondValues: 'rate', statistic: 'score'},
    stages: ['neighbourhood', 'operator'],
    datumKinds: [{kind: 'region', label: 'tract', indexSpace: 'csr'}],
    guards: [],
    choices: [
      {
        name: 'transform',
        label: 'Weights transform',
        stage: 'neighbourhood',
        alternatives: ['binary', 'row'],
        sweepable: true
      }
    ],
    formula: {
      id: 'spatial-pearson',
      derivation: 'curated',
      tex: String.raw`\begin{aligned}
\term{statistic}{r}&=\frac{\term{numerator}{\sum_{ij}w_{ij}(x_i-\bar{x})(y_j-\bar{y})}}{\sqrt{\term{squareX}{\sum_{ij}w_{ij}(x_i-\bar{x})^2}\;\term{squareY}{\sum_{ij}w_{ij}(y_j-\bar{y})^2}}}\\
\term{meanX}{\bar{x}}&=\frac{\sum_{ij}w_{ij}x_i}{\term{weightTotal}{\sum_{ij}w_{ij}}},
\qquad\term{meanY}{\bar{y}}=\frac{\sum_{ij}w_{ij}y_j}{\term{weightTotal}{\sum_{ij}w_{ij}}}
\end{aligned}`,
      ascii:
        'r = sum_ij w_ij (x_i - mx)(y_j - my) / sqrt(sum_ij w_ij (x_i - mx)^2 * sum_ij w_ij (y_j - my)^2); mx = sum_ij w_ij x_i / sum_ij w_ij, my = sum_ij w_ij y_j / sum_ij w_ij. Sums run over directed edges (i, j) with positive weight to another included row; the weights are used as given and the means are edge-weighted, not the plain column means. Clamped to [-1, 1]; NaN for zero weight or zero variance. There is no analytic p-value.',
      symbols: [
        {id: 'statistic', role: 'output', binding: 'wgsl:answer'},
        {id: 'numerator', role: 'intermediate', binding: 'wgsl:numerator'},
        {id: 'squareX', role: 'intermediate', binding: 'wgsl:squareX'},
        {id: 'squareY', role: 'intermediate', binding: 'wgsl:squareY'},
        {id: 'meanX', role: 'intermediate', binding: 'wgsl:meanX'},
        {id: 'meanY', role: 'intermediate', binding: 'wgsl:meanY'},
        {id: 'weightTotal', role: 'intermediate', binding: 'wgsl:weightTotal'}
      ]
    }
  },
  GPUGammaStatistic: {
    family: 'statistics',
    descriptor: {scope: 'pairwise', scopeSecondary: 'global', reduction: 'weighted-mean'},
    variables: {values: 'rate', statistic: 'score'},
    stages: ['neighbourhood', 'operator'],
    datumKinds: [{kind: 'region', label: 'tract', indexSpace: 'csr'}],
    guards: [],
    choices: [
      {
        name: 'transform',
        label: 'Weights transform',
        stage: 'neighbourhood',
        alternatives: ['binary', 'row'],
        sweepable: true
      },
      {
        name: 'operation',
        label: 'Pair term',
        stage: 'operator',
        alternatives: ['cross-product', 'squared-difference', 'absolute-difference'],
        sweepable: true
      }
    ],
    formula: {
      id: 'gamma-statistic',
      derivation: 'curated',
      tex: String.raw`\begin{aligned}
\term{statistic}{\Gamma}&=\frac{\term{numerator}{\sum_{ij}w_{ij}\,t(x_i,y_j)}}{\term{weightTotal}{\sum_{ij}w_{ij}}},
&t(x,y)&=\begin{cases}(x-\term{meanX}{\bar{x}})(y-\term{meanY}{\bar{y}})&\text{cross-product}\\(x-y)^2&\text{squared difference}\\|x-y|&\text{absolute difference}\end{cases}
\end{aligned}`,
      ascii:
        'Gamma = sum_ij w_ij t(x_i, y_j) / sum_ij w_ij over directed edges with positive weight to another included row; t = (x - mx)(y - my) for cross-product (edge-weighted means mx, my), (x - y)^2 for squared-difference, |x - y| for absolute-difference. y is the second variable, or x itself when none is given. NaN when the total edge weight is not positive. There is no analytic p-value.',
      symbols: [
        {id: 'statistic', role: 'output', binding: 'wgsl:answer'},
        {id: 'numerator', role: 'intermediate', binding: 'wgsl:numerator'},
        {id: 'weightTotal', role: 'intermediate', binding: 'wgsl:weightTotal'},
        {id: 'meanX', role: 'intermediate', binding: 'wgsl:meanX'},
        {id: 'meanY', role: 'intermediate', binding: 'wgsl:meanY'}
      ]
    }
  },
  GPULocalPermutationTest: {
    family: 'statistics',
    descriptor: {scope: 'focal', scopeSecondary: 'global', reduction: 'count'},
    variables: {
      values: 'rate',
      observed: 'score',
      exceedances: 'count',
      pValues: 'probability',
      adjustedPValues: 'probability',
      significant: 'category',
      nullDistribution: 'score'
    },
    stages: ['neighbourhood', 'operator', 'inference'],
    datumKinds: [{kind: 'region', label: 'tract', indexSpace: 'csr'}],
    guards: ['uncorrected-p', 'pseudo-p-grid'],
    choices: [
      {
        name: 'transform',
        label: 'Weights transform',
        stage: 'neighbourhood',
        alternatives: ['binary', 'row'],
        sweepable: true
      },
      {name: 'islands', stage: 'neighbourhood', alternatives: ['exclude', 'keep'], sweepable: true},
      {
        name: 'statistic',
        label: 'Local statistic',
        stage: 'operator',
        alternatives: [
          'localMoran',
          'localGeary',
          'localG',
          'localGStar',
          'localBivariateMoran',
          'localLeeL',
          'localJoinCount',
          'localBivariateJoinCount'
        ],
        sweepable: false
      },
      {
        name: 'alternative',
        label: 'Tail of the pseudo p-value',
        stage: 'inference',
        alternatives: ['two-sided', 'greater', 'lesser', 'folded', 'directed'],
        sweepable: true
      },
      {name: 'permutations', label: 'Permutations P', stage: 'inference', sweepable: true},
      {
        name: 'multipleTesting',
        stage: 'inference',
        alternatives: ['none', 'bonferroni', 'holm', 'benjamini-hochberg', 'benjamini-yekutieli'],
        sweepable: true
      },
      {name: 'significanceLevel', stage: 'inference', sweepable: true}
    ],
    formula: {
      id: 'local-permutation-pseudo-p',
      derivation: 'curated',
      tex: String.raw`\begin{aligned}
\term{pValue}{p_i}&=\frac{\min\left(m\,(\term{count}{M_i}+1),\;\term{permutations}{P}+1\right)}{\term{permutations}{P}+1},
&\term{count}{M_i}&=\min(\term{greater}{g_i},\term{lesser}{l_i})\\
\term{greater}{g_i}&=\#\{\term{simulated}{s_{ip}}\ge\term{observed}{s_i}\},
&\term{lesser}{l_i}&=\#\{\term{simulated}{s_{ip}}\le\term{observed}{s_i}\}
\end{aligned}`,
      ascii:
        'p_i = min(m (M_i + 1), P + 1) / (P + 1); g_i = #{p : s_ip >= s_i}, l_i = #{p : s_ip <= s_i}; the default two-sided tail has m = 2 and M_i = min(g_i, l_i), so a row whose simulations all tie the observed value has p = 1. greater: m = 1, M = g; lesser: m = 1, M = l; folded: m = 1, M = #{|s_ip - mean| >= |s_i - mean|}; directed: m = 1, M = min(g, l). Each simulation s_ip assigns k_i distinct other rows (Philox draw, partial Fisher-Yates) to the focus row neighbor slots; the observed and simulated terms use the same f32 expression. The p-values lie on the grid k/(P+1).',
      symbols: [
        {id: 'pValue', role: 'output', binding: 'wgsl:pValue'},
        {id: 'count', role: 'intermediate', label: 'exceedance count', binding: 'wgsl:count'},
        {id: 'permutations', role: 'parameter', binding: 'wgsl:permutations'},
        {id: 'greater', role: 'intermediate', binding: 'wgsl:greater'},
        {id: 'lesser', role: 'intermediate', binding: 'wgsl:lesser'},
        {id: 'simulated', role: 'intermediate', binding: 'wgsl:simulated'},
        {id: 'observed', role: 'intermediate', binding: 'wgsl:observed'}
      ]
    }
  },
  GPUGlobalPermutationTest: {
    family: 'statistics',
    descriptor: {scope: 'global', scopeSecondary: 'pairwise', reduction: 'count'},
    variables: {
      values: 'rate',
      summary: 'score',
      referenceDistribution: 'score',
      histogram: 'count'
    },
    stages: ['neighbourhood', 'operator', 'inference'],
    datumKinds: [{kind: 'region', label: 'tract', indexSpace: 'csr'}],
    guards: ['uncorrected-p', 'pseudo-p-grid'],
    choices: [
      {
        name: 'transform',
        label: 'Weights transform',
        stage: 'neighbourhood',
        alternatives: ['binary', 'row'],
        sweepable: true
      },
      {name: 'islands', stage: 'neighbourhood', alternatives: ['exclude', 'keep'], sweepable: true},
      {
        name: 'statistic',
        label: 'Global statistic',
        stage: 'operator',
        alternatives: ['moran', 'geary', 'getisOrdG', 'bivariateMoran', 'leesL'],
        sweepable: false
      },
      {
        name: 'alternative',
        label: 'Tail of the pseudo p-value',
        stage: 'inference',
        alternatives: ['two-sided', 'greater', 'lesser', 'folded', 'directed'],
        sweepable: true
      },
      {name: 'permutations', label: 'Permutations P', stage: 'inference', sweepable: true}
    ],
    formula: {
      id: 'global-permutation-pseudo-p',
      derivation: 'curated',
      tex: String.raw`\begin{aligned}
\term{pValue}{p}&=\frac{\min\left(m\,(\term{larger}{M}+1),\;\term{permutations}{P}+1\right)}{\term{permutations}{P}+1},
&\term{larger}{M}&=\min(\term{greater}{g},\term{lesser}{l})\\
\term{greater}{g}&=\#\{\term{pairSum}{q_p}\ge\term{observedSum}{q_0}\},
&\term{lesser}{l}&=\#\{\term{pairSum}{q_p}\le\term{observedSum}{q_0}\}\\
\term{observed}{s_0}&=\term{scale}{\kappa}\,\term{observedSum}{q_0},
&\term{zSimulated}{z}&=\frac{\term{observed}{s_0}-\term{mean}{\bar{s}}}{\term{standardDeviation}{\sigma_s}}
\end{aligned}`,
      ascii:
        'p = min(m (M + 1), P + 1) / (P + 1); g = #{p in 1..P : q_p >= q_0}, l = #{q_p <= q_0}, where q_0 is the pair sum of the identity permutation and q_p that of permutation p; the default two-sided tail has m = 2 and M = min(g, l) (greater: m = 1, M = g; lesser: m = 1, M = l; folded: m = 1, M = #{|s_p - mean| >= |s_0 - mean|}; directed: m = 1, M = min(g, l)). The statistic is s = kappa q with a positive constant kappa, so the counts compare identical f32 pair sums. z = (s_0 - mean) / sd of the P simulated statistics (esda z_sim). The p-value lies on the grid k/(P+1).',
      symbols: [
        {id: 'pValue', role: 'output', binding: 'wgsl:results'},
        {id: 'larger', role: 'intermediate', label: 'exceedance count', binding: 'wgsl:larger'},
        {id: 'permutations', role: 'parameter', binding: 'wgsl:permutations'},
        {id: 'greater', role: 'intermediate', binding: 'wgsl:greater'},
        {id: 'lesser', role: 'intermediate', binding: 'wgsl:lesser'},
        {id: 'pairSum', role: 'intermediate', binding: 'wgsl:pairSum'},
        {id: 'observedSum', role: 'intermediate', binding: 'wgsl:observedSum'},
        {id: 'observed', role: 'output', binding: 'wgsl:observed'},
        {id: 'scale', role: 'intermediate', binding: 'wgsl:scale'},
        {id: 'zSimulated', role: 'output', binding: 'wgsl:zSimulated'},
        {id: 'mean', role: 'intermediate', binding: 'wgsl:mean'},
        {id: 'standardDeviation', role: 'intermediate', binding: 'wgsl:standardDeviation'}
      ]
    }
  },
  GPUMultipleTestingAdjustment: {
    family: 'statistics',
    descriptor: {scope: 'global', reduction: 'scan-sort'},
    variables: {
      pValues: 'probability',
      adjustedPValues: 'probability',
      testedCount: 'count'
    },
    stages: ['inference'],
    datumKinds: [{kind: 'region', label: 'tract', indexSpace: 'csr'}],
    guards: ['uncorrected-p'],
    choices: [
      {
        name: 'method',
        label: 'Correction method',
        stage: 'inference',
        alternatives: ['none', 'bonferroni', 'holm', 'benjamini-hochberg', 'benjamini-yekutieli'],
        sweepable: true
      },
      {name: 'minimumFamilySize', stage: 'inference', sweepable: false},
      {name: 'significanceLevel', stage: 'inference', sweepable: true}
    ],
    formula: {
      id: 'multiple-testing-adjustment',
      derivation: 'curated',
      tex: String.raw`\begin{aligned}
\term{adjusted}{\tilde{p}_{(k)}}&=\min\Bigl(1,\;\min_{l\ge k}\;\term{stepValue}{\frac{p_{(l)}\,m\,c}{l}}\Bigr),
&c&=\begin{cases}1&\text{Benjamini-Hochberg}\\\term{scale}{\textstyle\sum_{i=1}^{m}1/i}&\text{Benjamini-Yekutieli}\end{cases}\\
\term{adjusted}{\tilde{p}_{(k)}}&=\min\Bigl(1,\;\max_{l\le k}\,(\term{family}{m}-l+1)\,p_{(l)}\Bigr),
&\term{adjusted}{\tilde{p}}&=\min(1,\;\term{family}{m}\,\term{pValues}{p})
\end{aligned}`,
      ascii:
        'Bonferroni: min(1, m p). Holm: min(1, max_{l<=k} (m - l + 1) p_(l)). Benjamini-Hochberg: min(1, min_{l>=k} p_(l) m / l). Benjamini-Yekutieli: BH times c(m) = sum_{i=1..m} 1/i. p_(k) is the k-th smallest tested p-value (finite and in [0, 1]); the family size is m = max(tested rows, minimumFamilySize). A row is significant at level alpha exactly when its adjusted p-value is <= alpha; ties get identical adjusted values; untested rows are NaN. Equal to statsmodels multipletests.',
      symbols: [
        {id: 'adjusted', role: 'output', binding: 'wgsl:adjusted'},
        {id: 'stepValue', role: 'intermediate', binding: 'wgsl:getStepValue'},
        {id: 'scale', role: 'intermediate', label: 'c(m)', binding: 'wgsl:familyScale'},
        {id: 'family', role: 'intermediate', label: 'family size m', binding: 'wgsl:family'},
        {id: 'pValues', role: 'input', binding: 'wgsl:pValues'}
      ]
    }
  }
};
