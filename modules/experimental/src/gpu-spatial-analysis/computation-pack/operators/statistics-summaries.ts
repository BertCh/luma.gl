// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {SpatialFamilyOperators} from '../types';

/**
 * Spatial computation-pack entries for the `statistics` family (rates, summaries, zonal and focal statistics), keyed by exported name.
 * Merged into `STATISTICS_OPERATORS`. Every entry must use `family: 'statistics'`.
 */
export const STATISTICS_SUMMARY_OPERATORS: SpatialFamilyOperators = {
  GPUAssuncaoReisStandardization: {
    family: 'statistics',
    descriptor: {scope: 'global', scopeSecondary: 'local', reduction: 'weighted-mean'},
    variables: {
      events: 'count',
      populations: 'count',
      standardizedRates: 'score',
      statistics: 'score',
      invalidCount: 'count'
    },
    stages: ['operator'],
    datumKinds: [{kind: 'region', label: 'tract', indexSpace: 'csr'}],
    guards: ['eb-reference-rate', 'count-vs-rate'],
    choices: [],
    formula: {
      id: 'assuncao-reis-standardization',
      derivation: 'curated',
      tex: String.raw`\begin{aligned}
\term{standardizedRates}{z_i}&=\frac{\term{rate}{p_i}-\term{mean}{b}}{\sqrt{\term{variance}{v_i}}},
&\term{rate}{p_i}&=\frac{e_i}{\term{population}{n_i}},
&\term{mean}{b}&=\frac{\term{eventSum}{\sum e}}{\term{populationSum}{\sum n}}\\
\term{variance}{v_i}&=\begin{cases}\term{rawVariance}{a+b/n_i}&\text{if }a+b/n_i\ge 0\\\term{poissonVariance}{b/n_i}&\text{otherwise}\end{cases},
&\term{priorVariance}{a}&=\frac{\term{squareSum}{\sum n_i(p_i-b)^2}}{\sum n}-\frac{b}{\sum n/\term{count}{m}}
\end{aligned}`,
      ascii:
        'z_i = (p_i - b) / sqrt(v_i); p_i = e_i / n_i; b = sum e / sum n; a = sum n_i (p_i - b)^2 / sum n - b / (sum n / m); v_i = a + b / n_i, or b / n_i where that is negative. Sums run over the m valid rows only (unmasked, finite e >= 0, finite n > 0). a is not clamped: a negative a is kept wherever v_i stays non-negative. Rows that are not valid, or whose v_i is not positive, are NaN.',
      symbols: [
        {id: 'standardizedRates', role: 'output', binding: 'wgsl:standardizedRates'},
        {id: 'rate', role: 'intermediate', binding: 'wgsl:rate'},
        {id: 'mean', role: 'intermediate', label: 'pooled rate', binding: 'wgsl:mean'},
        {id: 'variance', role: 'intermediate', binding: 'wgsl:variance'},
        {id: 'population', role: 'input', binding: 'wgsl:population'},
        {id: 'eventSum', role: 'intermediate', binding: 'wgsl:total_events'},
        {id: 'populationSum', role: 'intermediate', binding: 'wgsl:total_population'},
        {id: 'rawVariance', role: 'intermediate', binding: 'wgsl:rawVariance'},
        {id: 'poissonVariance', role: 'intermediate', binding: 'wgsl:poissonVariance'},
        {id: 'priorVariance', role: 'intermediate', binding: 'wgsl:priorVariance'},
        {id: 'squareSum', role: 'intermediate', binding: 'wgsl:total_square'},
        {id: 'count', role: 'intermediate', label: 'valid row count', binding: 'wgsl:count'}
      ]
    }
  },
  GPUEmpiricalBayesRates: {
    family: 'statistics',
    descriptor: {scope: 'global', scopeSecondary: 'local', reduction: 'weighted-mean'},
    variables: {
      events: 'count',
      populations: 'count',
      rawRates: 'rate',
      smoothedRates: 'rate',
      standardizedRates: 'score',
      summary: 'rate'
    },
    stages: ['operator'],
    datumKinds: [{kind: 'region', label: 'tract', indexSpace: 'csr'}],
    guards: ['eb-reference-rate', 'count-vs-rate'],
    choices: [
      {
        name: 'clampPriorVariance',
        label: 'Clamp prior variance at zero',
        stage: 'operator',
        alternatives: [true, false],
        sweepable: true
      }
    ],
    formula: {
      id: 'empirical-bayes-rates',
      derivation: 'curated',
      tex: String.raw`\begin{aligned}
\term{smoothed}{r_i}&=\term{weight}{w_i}\,\term{rate}{y_i}+(1-w_i)\,\term{pooledRate}{m},
&\term{weight}{w_i}&=\frac{\term{clampedVariance}{a^{+}}}{a^{+}+\term{noise}{m/b_i}},
&\term{rate}{y_i}&=\frac{e_i}{b_i}\\
\term{pooledRate}{m}&=\frac{\term{eventSum}{\sum e}}{\term{populationSum}{\sum b}},
&\term{clampedVariance}{a^{+}}&=\max(\term{priorVariance}{a},0),
&\term{priorVariance}{a}&=\term{weightedVariance}{s^2}-\frac{m}{\sum b/\term{count}{n}}
\end{aligned}`,
      ascii:
        'r_i = w_i y_i + (1 - w_i) m; w_i = a+ / (a+ + m / b_i) (0 where the denominator is not positive); y_i = e_i / b_i; m = sum e / sum b (pooled, not the mean of rates); a+ = max(a, 0) by default (clampPriorVariance true; false uses w = a / (a + m / b_i), esda parity); a = s2 - m / (sum b / n); s2 = sum b_i (y_i - m)^2 / sum b. Sums over the n included rows (unmasked, finite e, finite b > 0); other rows are NaN.',
      symbols: [
        {id: 'smoothed', role: 'output', binding: 'wgsl:smoothed'},
        {id: 'weight', role: 'intermediate', binding: 'wgsl:weight'},
        {id: 'rate', role: 'intermediate', binding: 'wgsl:rate'},
        {id: 'pooledRate', role: 'intermediate', binding: 'wgsl:pooledRate'},
        {id: 'clampedVariance', role: 'intermediate', binding: 'wgsl:clampedVariance'},
        {id: 'noise', role: 'intermediate', binding: 'wgsl:noise'},
        {id: 'eventSum', role: 'intermediate', binding: 'wgsl:eventSum'},
        {id: 'populationSum', role: 'intermediate', binding: 'wgsl:populationSum'},
        {id: 'priorVariance', role: 'intermediate', binding: 'wgsl:priorVariance'},
        {id: 'weightedVariance', role: 'intermediate', binding: 'wgsl:weightedVariance'},
        {id: 'count', role: 'intermediate', label: 'included row count', binding: 'wgsl:count'}
      ]
    }
  },
  GPUSpatialEmpiricalBayesRates: {
    family: 'statistics',
    descriptor: {scope: 'focal', scopeSecondary: 'local', reduction: 'weighted-mean'},
    variables: {
      events: 'count',
      populations: 'count',
      spatialRates: 'rate',
      smoothedRates: 'rate',
      priorMeans: 'rate',
      priorVariances: 'score'
    },
    stages: ['neighbourhood', 'operator'],
    datumKinds: [{kind: 'region', label: 'tract', indexSpace: 'csr'}],
    guards: ['eb-reference-rate', 'count-vs-rate'],
    choices: [
      {
        name: 'criterion',
        label: 'Contiguity criterion',
        stage: 'neighbourhood',
        alternatives: ['queen', 'rook'],
        sweepable: true
      }
    ],
    formula: {
      id: 'spatial-empirical-bayes-rates',
      derivation: 'curated',
      tex: String.raw`\begin{aligned}
\term{smoothed}{r_i}&=\term{weight}{w_i}\,\term{ownRate}{y_i}+(1-w_i)\,\term{priorMean}{m_i},
&\term{weight}{w_i}&=\frac{\term{priorVariance}{a_i}}{a_i+m_i/\term{ownPopulation}{b_i}}\\
\term{priorMean}{m_i}&=\frac{\term{eventSum}{E_i}}{\term{populationSum}{B_i}},
&\term{priorVariance}{a_i}&=\max\!\left(\frac{\term{deviationSum}{\sum_j b_j(y_j-m_i)^2}}{B_i}-\frac{m_i}{B_i/\term{count}{n_i}},0\right)
\end{aligned}`,
      ascii:
        'r_i = w_i y_i + (1 - w_i) m_i; w_i = a_i / (a_i + m_i / b_i), 0 where the denominator is not positive; m_i = E_i / B_i; a_i = max(sum_j b_j (y_j - m_i)^2 / B_i - m_i / (B_i / n_i), 0). The neighbourhood j is row i plus its listed neighbours that are included (unmasked, finite e, finite b > 0); only the structure of the weights counts. E_i, B_i = sums of e and b over the n_i members. The spatial rate is m_i itself. Excluded rows are NaN.',
      symbols: [
        {id: 'smoothed', role: 'output', binding: 'wgsl:smoothed'},
        {id: 'weight', role: 'intermediate', binding: 'wgsl:weight'},
        {id: 'ownRate', role: 'intermediate', binding: 'wgsl:ownRate'},
        {id: 'priorMean', role: 'intermediate', binding: 'wgsl:priorMean'},
        {id: 'priorVariance', role: 'intermediate', binding: 'wgsl:priorVariance'},
        {id: 'ownPopulation', role: 'input', binding: 'wgsl:ownPopulation'},
        {id: 'eventSum', role: 'intermediate', binding: 'wgsl:eventSum'},
        {id: 'populationSum', role: 'intermediate', binding: 'wgsl:populationSum'},
        {id: 'deviationSum', role: 'intermediate', binding: 'wgsl:deviationSum'},
        {id: 'count', role: 'intermediate', label: 'neighbourhood size', binding: 'wgsl:count'}
      ]
    }
  },
  GPUNeighborhoodSummary: {
    family: 'statistics',
    descriptor: {scope: 'focal', reduction: 'weighted-mean'},
    variables: {values: 'rate', table: 'rate', categories: 'category', entropy: 'score'},
    stages: ['neighbourhood', 'operator'],
    datumKinds: [{kind: 'region', label: 'tract', indexSpace: 'csr'}],
    guards: ['row-standardisation'],
    choices: [
      {
        name: 'transform',
        label: 'Weights transform',
        stage: 'neighbourhood',
        alternatives: ['binary', 'row'],
        sweepable: true
      },
      {
        name: 'includeFocal',
        label: 'Include focal row',
        stage: 'neighbourhood',
        alternatives: [true, false],
        sweepable: true
      },
      {name: 'statistics', label: 'Numeric statistics', stage: 'operator', sweepable: false}
    ],
    formula: {
      id: 'neighborhood-summary',
      derivation: 'curated',
      tex: String.raw`\begin{aligned}
\term{mean}{\bar{x}_i}&=\frac{\term{weightedSum}{\sum_s w_s x_s}}{\term{weightSum}{W_i}},
&\term{weightSum}{W_i}&=\sum_s \term{weight}{w_s}\\
\term{entropy}{H_i}&=-\sum_c \term{share}{p_c}\ln p_c,
&\term{share}{p_c}&=\frac{\term{frequency}{f_c}}{W_i},
&\term{frequency}{f_c}&=\sum_{s:\,c_s=c} w_s
\end{aligned}`,
      ascii:
        'mean_i = sum_s w_s x_s / W_i; W_i = sum_s w_s; H_i = -sum_c p_c ln p_c; p_c = f_c / W_i; f_c = sum of w_s over members with category c. Members s of row i are the focal row (only with includeFocal) and the CSR slots with positive weight to another unmasked row (numeric statistics also need a finite value). With row-standardised weights the slot weights are re-divided by the sum of the active slot weights. NaN when W_i = 0; zero-weight slots are not members.',
      symbols: [
        {id: 'mean', role: 'output', binding: 'wgsl:mean'},
        {id: 'weightedSum', role: 'intermediate', binding: 'wgsl:weightedSum'},
        {id: 'weightSum', role: 'intermediate', binding: 'wgsl:weightSum'},
        {id: 'weight', role: 'input', binding: 'wgsl:weight'},
        {id: 'entropy', role: 'output', binding: 'wgsl:entropyTotal'},
        {id: 'share', role: 'intermediate', binding: 'wgsl:share'},
        {id: 'frequency', role: 'intermediate', binding: 'wgsl:frequency'}
      ]
    }
  },
  GPUGeographicDistribution: {
    family: 'statistics',
    descriptor: {scope: 'global', reduction: 'weighted-mean'},
    variables: {
      weights: 'count',
      counts: 'count',
      weightSums: 'count',
      meanCenters: 'coordinate',
      standardDistances: 'measure',
      ellipses: 'measure',
      ellipseVertices: 'coordinate'
    },
    stages: ['support', 'operator'],
    datumKinds: [{kind: 'group', label: 'group', indexSpace: '1d'}],
    guards: ['coordinate-model'],
    choices: [
      {
        name: 'standardDeviations',
        label: 'Standard deviations',
        stage: 'operator',
        alternatives: [1, 2, 3],
        sweepable: true
      },
      {
        name: 'ellipseConvention',
        label: 'Ellipse convention',
        stage: 'operator',
        alternatives: ['arcgis', 'standard'],
        sweepable: true
      }
    ],
    formula: {
      id: 'geographic-distribution',
      derivation: 'curated',
      tex: String.raw`\begin{aligned}
\term{meanCenters}{M}&=\frac{\sum_i w_i p_i}{\term{total}{W}},
&\term{standardDistances}{SD}&=\term{multiplier}{k}\sqrt{\term{varianceX}{S_{xx}}+\term{varianceY}{S_{yy}}}\\
\term{halfSum}{h}&=\tfrac{1}{2}(S_{xx}+S_{yy}),
&\term{halfDifference}{d}&=\tfrac{1}{2}(S_{xx}-S_{yy}),
&\term{radius}{\rho}&=\sqrt{d^2+\term{covariance}{S_{xy}}^2}\\
\term{ellipses}{(\sigma_x,\sigma_y)}&=\term{scale}{k\,c}\left(\sqrt{h-\rho},\ \sqrt{h+\rho}\right)
\end{aligned}`,
      ascii:
        'M = sum w p / W; Sxx, Syy, Sxy = weighted second moments about M divided by W (not W - 1); SD = k sqrt(Sxx + Syy); h = (Sxx + Syy)/2; d = (Sxx - Syy)/2; rho = sqrt(d^2 + Sxy^2); ellipse semi-axes (sigmaX, sigmaY) = k c (sqrt(h - rho), sqrt(h + rho)) with c = sqrt(2) for the arcgis convention and 1 for standard; k = standardDeviations. Planar coordinates only; rows with a non-finite position or a non-positive weight are excluded.',
      symbols: [
        {id: 'meanCenters', role: 'output', binding: 'wgsl:meanCentersOut'},
        {id: 'total', role: 'intermediate', label: 'weight sum', binding: 'wgsl:total'},
        {id: 'standardDistances', role: 'output', binding: 'wgsl:standardDistances'},
        {id: 'multiplier', role: 'parameter', binding: 'wgsl:multiplier'},
        {id: 'varianceX', role: 'intermediate', binding: 'wgsl:varianceX'},
        {id: 'varianceY', role: 'intermediate', binding: 'wgsl:varianceY'},
        {id: 'covariance', role: 'intermediate', binding: 'wgsl:covariance'},
        {id: 'halfSum', role: 'intermediate', binding: 'wgsl:halfSum'},
        {id: 'halfDifference', role: 'intermediate', binding: 'wgsl:halfDifference'},
        {id: 'radius', role: 'intermediate', binding: 'wgsl:radius'},
        {id: 'ellipses', role: 'output', binding: 'wgsl:ellipses'},
        {id: 'scale', role: 'parameter', binding: 'wgsl:scale'}
      ]
    }
  },
  GPURegionStatistics: {
    family: 'statistics',
    descriptor: {scope: 'zonal', reduction: 'sum'},
    variables: {positions: 'coordinate', values: 'count', mask: 'category', summary: 'count'},
    stages: ['support', 'operator'],
    datumKinds: [{kind: 'region', label: 'tract', indexSpace: '1d'}],
    guards: ['coordinate-model'],
    choices: [
      {
        name: 'regionKind',
        label: 'Region shape',
        stage: 'support',
        alternatives: ['rectangle', 'polygon', 'radius', 'pick-region', 'mask'],
        sweepable: true
      },
      {name: 'binCount', label: 'Histogram bins', stage: 'operator', sweepable: true}
    ],
    formula: {
      id: 'region-statistics',
      derivation: 'curated',
      tex: String.raw`\begin{aligned}
\term{selected}{s_i}&=[\,p_i\in R\,],
&\term{valueCount}{N}&=\sum_i s_i\,[\,v_i\ \text{finite}\,]\\
\term{sum}{S}&=\sum_i s_i\,[\,v_i\ \text{finite}\,]\,v_i,
&\term{mean}{\bar{v}}&=\frac{S}{N}
\end{aligned}`,
      ascii:
        's_i = [p_i in R]; N = sum_i s_i [v_i finite]; S = sum_i s_i [v_i finite] v_i; mean = S / N, reported as 0 when N = 0 (not NaN), as are the sum, minimum and maximum. R is the rectangle, lasso polygon (even-odd), circle, pick result or caller mask. Points with a non-finite position are never selected. Counts are exact; the sum is a GPU reduction.',
      symbols: [
        {id: 'selected', role: 'intermediate', binding: 'wgsl:selected'},
        {id: 'valueCount', role: 'intermediate', binding: 'wgsl:valueCount'},
        {id: 'sum', role: 'output', binding: 'wgsl:sum'},
        {id: 'mean', role: 'output', binding: 'wgsl:mean'}
      ]
    }
  },
  GPURegionMask: {
    family: 'statistics',
    descriptor: {scope: 'local', reduction: 'select'},
    variables: {positions: 'coordinate', mask: 'category'},
    stages: ['support'],
    datumKinds: [{kind: 'region', label: 'tract', indexSpace: '1d'}],
    guards: ['coordinate-model'],
    choices: [
      {
        name: 'regionKind',
        label: 'Region shape',
        stage: 'support',
        alternatives: ['rectangle', 'polygon'],
        sweepable: true
      }
    ],
    formula: {
      id: 'region-mask',
      derivation: 'curated',
      tex: String.raw`\term{mask}{m_i}=\begin{cases}1&\term{valid}{\text{bounds valid}}\ \wedge\ \term{minimum}{p_{\min}}\le p_i\le\term{maximum}{p_{\max}}\\0&\text{otherwise}\end{cases}`,
      ascii:
        'm_i = 1 when the bounds are finite with min <= max, the point is finite and min <= p_i <= max on both axes (inclusive); otherwise 0. The lasso polygon uses an even-odd crossing test instead. The rectangle form is shown.',
      symbols: [
        {id: 'mask', role: 'output', binding: 'wgsl:outputMask'},
        {id: 'valid', role: 'intermediate', binding: 'wgsl:validBounds'},
        {id: 'minimum', role: 'parameter', binding: 'wgsl:minimum'},
        {id: 'maximum', role: 'parameter', binding: 'wgsl:maximum'}
      ]
    }
  },
  GPUPickRegionMask: {
    family: 'statistics',
    descriptor: {scope: 'local', reduction: 'select'},
    variables: {picks: 'category', mask: 'category'},
    stages: ['support'],
    datumKinds: [{kind: 'region', label: 'tract', indexSpace: '1d'}],
    guards: [],
    choices: [{name: 'batchIndex', label: 'Batch filter', stage: 'support', sweepable: false}],
    formula: {
      id: 'pick-region-mask',
      derivation: 'curated',
      tex: String.raw`\term{outputMask}{m_r}=\begin{cases}1&\exists\,k<\min(\term{count}{c},\term{pairCapacity}{K}):\ \term{objectIndex}{o_k}=r\ \wedge\ \term{batchIndex}{b_k}=b\\0&\text{otherwise}\end{cases}`,
      ascii:
        'm_r = 1 when some pair k < min(count, K) of the pick result has objectIndex o_k = r and batchIndex b_k = b (any batch when no filter is set); otherwise 0. Duplicate picks write the same 1, negative or out-of-range indices are ignored, and the overflow word is copied as 0 or 1.',
      symbols: [
        {id: 'outputMask', role: 'output', binding: 'wgsl:outputMask'},
        {id: 'count', role: 'input', label: 'pick count', binding: 'wgsl:result'},
        {id: 'pairCapacity', role: 'parameter', binding: 'wgsl:PAIR_CAPACITY'},
        {id: 'objectIndex', role: 'input', binding: 'wgsl:objectIndex'},
        {id: 'batchIndex', role: 'parameter', binding: 'wgsl:batchIndex'}
      ]
    }
  },
  GPUZonalStatistics: {
    family: 'statistics',
    descriptor: {scope: 'zonal', reduction: 'sum'},
    variables: {
      pointFeatureRows: 'category',
      values: 'count',
      counts: 'count',
      valueCounts: 'count',
      sums: 'count',
      means: 'rate',
      standardDeviations: 'rate'
    },
    stages: ['support', 'operator'],
    datumKinds: [{kind: 'zone', label: 'community area', indexSpace: '1d'}],
    guards: ['conservation', 'count-vs-rate'],
    choices: [
      {
        name: 'sumOrder',
        label: 'Sum order',
        stage: 'operator',
        alternatives: ['sorted', 'atomic'],
        sweepable: false
      },
      {
        name: 'membership',
        label: 'Zone membership',
        stage: 'support',
        alternatives: ['feature-rows', 'polygons'],
        sweepable: false
      }
    ],
    formula: {
      id: 'zonal-statistics',
      derivation: 'curated',
      tex: String.raw`\begin{aligned}
\term{sums}{S_f}&=\sum_{i:\ \term{row}{z_i}=f,\ \term{valid}{v_i\ \text{finite}}}\term{values}{v_i},
&\term{means}{\bar{v}_f}&=\frac{S_f}{\term{valueCounts}{N_f}}\\
\term{standardDeviations}{s_f}&=\sqrt{\frac{\term{squaredDeviationSums}{\sum_i(v_i-\bar{v}_f)^2}}{N_f-1}}
\end{aligned}`,
      ascii:
        'S_f = sum of v_i over points i with zone z_i = f and v_i finite; N_f = their count; mean_f = S_f / N_f (NaN when N_f = 0); s_f = sqrt(sum (v_i - mean_f)^2 / (N_f - 1)) (NaN when N_f < 2). Unweighted form. The deviation sum is a second pass about the mean. Points whose zone row is not below featureCount are unassigned and contribute to no zone.',
      symbols: [
        {id: 'sums', role: 'output', binding: 'wgsl:sums'},
        {id: 'row', role: 'input', binding: 'wgsl:row'},
        {id: 'valid', role: 'intermediate', binding: 'wgsl:valid'},
        {id: 'values', role: 'input', binding: 'wgsl:values'},
        {id: 'means', role: 'output', binding: 'wgsl:means'},
        {id: 'valueCounts', role: 'intermediate', binding: 'wgsl:valueCounts'},
        {id: 'standardDeviations', role: 'output', binding: 'wgsl:standardDeviations'},
        {id: 'squaredDeviationSums', role: 'intermediate', binding: 'wgsl:squaredDeviationSums'}
      ]
    }
  },
  GPUFocalStatistics: {
    family: 'statistics',
    descriptor: {scope: 'focal', reduction: 'weighted-mean'},
    variables: {
      values: 'measure',
      mean: 'measure',
      standardDeviation: 'measure',
      windowCounts: 'count'
    },
    stages: ['neighbourhood', 'operator'],
    datumKinds: [{kind: 'cell', label: 'grid cell', indexSpace: '2d'}],
    guards: [],
    choices: [
      {
        name: 'radius',
        label: 'Window radius',
        unit: 'cells',
        stage: 'neighbourhood',
        sweepable: true
      },
      {
        name: 'shape',
        label: 'Window shape',
        stage: 'neighbourhood',
        alternatives: ['square', 'circle'],
        sweepable: true
      },
      {name: 'minimumCount', label: 'Minimum valid cells', stage: 'neighbourhood', sweepable: true}
    ],
    formula: {
      id: 'focal-statistics',
      derivation: 'curated',
      tex: String.raw`\begin{aligned}
\term{mean}{\bar{x}_c}&=\frac{\term{sum}{\sum_{k\in W_c}x_k}}{\term{count}{n_c}},
&\term{standardDeviation}{s_c}&=\sqrt{\frac{\term{squaredDeviationSum}{\sum_{k\in W_c}(x_k-\bar{x}_c)^2}}{n_c}}
\end{aligned}`,
      ascii:
        'mean_c = sum_{k in W_c} x_k / n_c; s_c = sqrt(sum_{k in W_c} (x_k - mean_c)^2 / n_c) (population deviation, centred second pass). W_c = the valid cells of the square or circular window around cell c, clipped at the raster edge; n_c = their count. The result is nodata (NaN) when n_c < max(minimumCount, 1), or when the cell itself is nodata and propagateCenterNoData is set.',
      symbols: [
        {id: 'mean', role: 'output', binding: 'wgsl:meanOut'},
        {id: 'sum', role: 'intermediate', binding: 'wgsl:sum'},
        {id: 'count', role: 'intermediate', binding: 'wgsl:count'},
        {id: 'standardDeviation', role: 'output', binding: 'wgsl:deviationOut'},
        {
          id: 'squaredDeviationSum',
          role: 'intermediate',
          binding: 'wgsl:squaredDeviationSum'
        }
      ]
    }
  }
};
