// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {SpatialChoiceTemplate} from '../choices';
import type {SpatialFamilyOperators} from '../types';

/** Weights choices shared by every regression entry that builds weights beside the estimator. */
const WEIGHTS_CHOICES: readonly SpatialChoiceTemplate[] = [
  {
    name: 'weightsCriterion',
    label: 'Contiguity criterion',
    stage: 'neighbourhood',
    alternatives: ['queen', 'rook'],
    sweepable: true
  },
  {
    name: 'weightsTransform',
    label: 'Weights transform',
    stage: 'neighbourhood',
    alternatives: ['binary', 'row'],
    sweepable: true
  }
];

/** Coefficient-table columns written by the lag, error and two-stage estimators. */
const COEFFICIENT_TABLE_VARIABLES = {
  estimates: 'score',
  standardErrors: 'score',
  zStatistics: 'score',
  pValues: 'probability'
} as const;

/**
 * Spatial computation-pack entries for the `regression` family, keyed by exported name. Every entry
 * must use `family: 'regression'`.
 */
export const REGRESSION_OPERATORS: SpatialFamilyOperators = {
  GPUOrdinaryLeastSquares: {
    family: 'regression',
    descriptor: {scope: 'global', reduction: 'solve'},
    variables: {
      coefficients: 'score',
      standardErrors: 'score',
      tStatistics: 'score',
      residuals: 'score',
      fitted: 'rate'
    },
    stages: ['operator', 'inference'],
    datumKinds: [{kind: 'coefficient', label: 'coefficient', indexSpace: '1d'}],
    guards: [],
    choices: [
      {
        name: 'ridgeLambda',
        label: 'Ridge penalty',
        unit: 'squared predictor units',
        stage: 'operator',
        sweepable: true
      }
    ],
    formula: {
      id: 'ordinary-least-squares',
      derivation: 'curated',
      tex: String.raw`\begin{aligned}
\term{coefficients}{b}&=\term{moments}{S_{xx}}^{-1}s_{xy},
&\term{intercept}{b_0}&=\bar y-\sum_j b_j\bar x_j\\
\term{sigmaSquared}{\hat\sigma^2}&=\frac{\term{residualSumSquares}{\mathrm{RSS}}}{\term{freedom}{n-p}},
&\term{standardErrors}{\mathrm{se}_j}&=\sqrt{\hat\sigma^2\,[S_{xx}^{-1}]_{jj}}\\
\term{tStatistics}{t_j}&=\frac{b_j}{\mathrm{se}_j},
&\term{rSquared}{R^2}&=1-\frac{\mathrm{RSS}}{\term{totalSumSquares}{\mathrm{TSS}}}
\end{aligned}`,
      ascii:
        'b = Sxx^-1 sxy; b0 = ybar - sum_j b_j xbar_j; sigma2 = RSS/(n-p); se_j = sqrt(sigma2 [Sxx^-1]_jj); t_j = b_j/se_j; R2 = 1 - RSS/TSS. S are centred cross-products of the finite rows; the unit-scaled normal equations are solved by Cholesky, optionally with ridge lambda added to the predictor diagonal. p counts the intercept; the intercept variance is sigma2 (1/n + xbar Sxx^-1 xbar).',
      symbols: [
        {
          id: 'coefficients',
          role: 'output',
          label: 'coefficients (intercept first)',
          binding: 'wgsl:coefficientsOut'
        },
        {id: 'moments', role: 'intermediate', label: 'centred moments', binding: 'wgsl:momentAt'},
        {id: 'intercept', role: 'intermediate', binding: 'wgsl:intercept'},
        {id: 'sigmaSquared', role: 'intermediate', binding: 'wgsl:sigmaSquared'},
        {id: 'residualSumSquares', role: 'intermediate', binding: 'wgsl:residualSumSquares'},
        {
          id: 'freedom',
          role: 'intermediate',
          label: 'residual degrees of freedom',
          binding: 'wgsl:freedom'
        },
        {id: 'standardErrors', role: 'output', binding: 'wgsl:standardErrorsOut'},
        {id: 'tStatistics', role: 'output', binding: 'wgsl:tStatisticsOut'},
        {id: 'rSquared', role: 'output', binding: 'wgsl:rSquared'},
        {id: 'totalSumSquares', role: 'intermediate', binding: 'wgsl:totalSumSquares'}
      ]
    }
  },
  GPUGeographicallyWeightedRegression: {
    family: 'regression',
    descriptor: {scope: 'focal', scopeSecondary: 'solver', reduction: 'solve'},
    variables: {
      response: 'rate',
      coefficients: 'score',
      standardErrors: 'score',
      tValues: 'score',
      localR2: 'rate',
      residuals: 'score',
      hatDiagonal: 'rate',
      significant: 'category',
      localStatus: 'category'
    },
    stages: ['support', 'neighbourhood', 'operator', 'inference'],
    datumKinds: [{kind: 'region', label: 'tract', indexSpace: '1d'}],
    guards: ['coordinate-model', 'count-vs-rate', 'uncorrected-p'],
    choices: [
      {
        name: 'kernel',
        label: 'Weight kernel',
        stage: 'neighbourhood',
        alternatives: ['bisquare', 'gaussian'],
        sweepable: true
      },
      {
        name: 'bandwidthMode',
        label: 'Bandwidth mode',
        stage: 'neighbourhood',
        alternatives: ['adaptive', 'fixed'],
        sweepable: true
      },
      {name: 'bandwidthUnit', label: 'Bandwidth unit', stage: 'neighbourhood', sweepable: false},
      {
        name: 'bandwidthSearch',
        label: 'Bandwidth search',
        stage: 'neighbourhood',
        alternatives: ['golden-section', 'ladder'],
        sweepable: false
      },
      {name: 'significanceLevel', stage: 'inference', sweepable: true}
    ],
    formula: {
      id: 'geographically-weighted-regression',
      derivation: 'curated',
      tex: String.raw`\begin{aligned}
\term{weight}{w_{ij}}&=\left(1-\frac{d_{ij}^2}{\term{bandwidth}{h_i}^2}\right)^2\ (d_{ij}<h_i),\ \text{else }0\\
\term{coefficients}{\beta_i}&=(Z_i^\top W_i Z_i)^{-1}Z_i^\top W_i\,y,
&\term{hat}{S_{ii}}&=\left[(Z_i^\top W_i Z_i)^{-1}\right]_{00}\\
\term{aicc}{\mathrm{AICc}}&=n\ln\frac{\term{residualSum}{\mathrm{RSS}}}{n}+n\ln 2\pi+\frac{n\,(n+\term{traceSum}{\operatorname{tr}S})}{n-2-\operatorname{tr}S},
&\term{standardErrors}{\mathrm{se}_{ik}}&=\sqrt{\hat\sigma^2\,[C_iC_i^\top]_{kk}}\\
\term{tValues}{t_{ik}}&=\frac{\beta_{ik}}{\mathrm{se}_{ik}}
\end{aligned}`,
      ascii:
        "w_ij = (1 - (d_ij/h_i)^2)^2 for d_ij < h_i, else 0 (bisquare); h_i = 1.00001 x distance to the k-th nearest included tract (adaptive). beta_i = (Z_i' W_i Z_i)^-1 Z_i' W_i y with z_j = (1, x_j - x_i), so the intercept is the fit at tract i and S_ii = [(Z'WZ)^-1]_00. AICc = n ln(RSS/n) + n ln 2pi + n (n + tr S)/(n - 2 - tr S) is minimised over k by golden-section search. se_ik = sqrt(sigma2 [C C']_kk) with C = (Z'WZ)^-1 Z'W and sigma2 = RSS/(n - tr S); t_ik = beta_ik/se_ik. Singular tracts give NaN.",
      symbols: [
        {id: 'weight', role: 'intermediate', binding: 'wgsl:weight'},
        {id: 'bandwidth', role: 'parameter', binding: 'wgsl:bandwidth'},
        {
          id: 'coefficients',
          role: 'output',
          label: 'local coefficients',
          binding: 'wgsl:beta'
        },
        {id: 'hat', role: 'intermediate', binding: 'wgsl:hat'},
        {id: 'aicc', role: 'intermediate', binding: 'wgsl:candidateScore'},
        {id: 'residualSum', role: 'intermediate', binding: 'wgsl:residualSum'},
        {
          id: 'traceSum',
          role: 'intermediate',
          label: 'effective parameters',
          binding: 'wgsl:traceSum'
        },
        {id: 'standardErrors', role: 'output', binding: 'wgsl:standardError'},
        {id: 'tValues', role: 'output', binding: 'wgsl:tValue'}
      ]
    }
  },
  GPUGeographicallyWeightedRegressionNonstationarityTest: {
    family: 'regression',
    descriptor: {scope: 'global', scopeSecondary: 'focal', reduction: 'count'},
    variables: {
      observedStandardDeviation: 'score',
      pseudoPValue: 'probability',
      simulatedMean: 'score',
      zSimulated: 'score'
    },
    stages: ['neighbourhood', 'operator', 'inference'],
    datumKinds: [{kind: 'coefficient', label: 'coefficient', indexSpace: '1d'}],
    guards: ['coordinate-model', 'pseudo-p-grid', 'uncorrected-p'],
    choices: [
      {name: 'permutations', stage: 'inference', sweepable: true},
      {name: 'seed', stage: 'inference', sweepable: false},
      {
        name: 'bandwidthPolicy',
        label: 'Bandwidth in permutations',
        stage: 'neighbourhood',
        sweepable: false
      }
    ],
    formula: {
      id: 'gwr-nonstationarity-test',
      derivation: 'curated',
      tex: String.raw`\begin{aligned}
\term{observedStandardDeviation}{s_k}&=\operatorname{sd}_i\,\beta_{ik},
&\term{simulatedMean}{\bar s_k}&=\frac{1}{P}\sum_{p=1}^{P}s_k^{(p)}\\
\term{exceedances}{g_k}&=\#\{p:\ s_k^{(p)}\ge s_k\},
&\term{pseudoPValue}{p_k}&=\frac{g_k+1}{\term{permutationCount}{P}+1}\\
\term{zSimulated}{z_k}&=\frac{s_k-\bar s_k}{\term{simulatedDeviation}{\sigma_{s_k}}}
\end{aligned}`,
      ascii:
        's_k = population standard deviation of the local estimates beta_ik over the tracts with a finite fit. Permutation p relabels (X, y) among the tracts with a keyed swap-or-not shuffle, refits every tract at the bandwidth already selected on the observed data, and records s_k^(p). g_k = number of permutations with s_k^(p) >= s_k; pseudo p = (g_k + 1)/(P + 1); z_k = (s_k - mean of s_k^(p))/sd of s_k^(p). Failed permuted fits are left out and counted.',
      symbols: [
        {id: 'observedStandardDeviation', role: 'output', binding: 'wgsl:observed'},
        {id: 'simulatedMean', role: 'output', binding: 'wgsl:mean'},
        {id: 'exceedances', role: 'intermediate', binding: 'wgsl:greater'},
        {id: 'pseudoPValue', role: 'output', binding: 'wgsl:table'},
        {
          id: 'permutationCount',
          role: 'parameter',
          label: 'permutations P',
          binding: 'wgsl:permutationCount'
        },
        {id: 'zSimulated', role: 'output', binding: 'wgsl:simulatedDeviation'},
        {id: 'simulatedDeviation', role: 'intermediate', binding: 'wgsl:simulatedDeviation'}
      ]
    }
  },
  GPUSpatialRegressionDiagnostics: {
    family: 'regression',
    descriptor: {scope: 'global', reduction: 'sum'},
    variables: {statistics: 'score', degreesOfFreedom: 'count', pValues: 'probability'},
    stages: ['neighbourhood', 'operator', 'inference'],
    datumKinds: [{kind: 'test', label: 'diagnostic test', indexSpace: '1d'}],
    guards: ['row-standardisation', 'uncorrected-p'],
    choices: [
      ...WEIGHTS_CHOICES,
      {
        name: 'restandardizeRows',
        label: 'Re-standardise rows after dropping excluded tracts',
        stage: 'neighbourhood',
        alternatives: [true, false],
        sweepable: true
      }
    ],
    formula: {
      id: 'spatial-regression-diagnostics',
      derivation: 'curated',
      tex: String.raw`\begin{aligned}
\term{errorGradient}{d_e}&=\frac{e^\top We}{\term{sigmaSquared}{\hat\sigma^2}},
&\term{lagGradient}{d_l}&=\frac{e^\top Wy}{\hat\sigma^2},
&\term{traceT}{T}&=\operatorname{tr}(W^\top W+W^2)\\
\term{information}{J}&=\frac{(WZb)^\top M\,(WZb)}{\hat\sigma^2}+T\\
\term{errorStatistic}{\mathrm{LM}_{\mathrm{err}}}&=\frac{d_e^2}{T},
&\term{lagStatistic}{\mathrm{LM}_{\mathrm{lag}}}&=\frac{d_l^2}{J}\\
\term{robustError}{\mathrm{LM}^{*}_{\mathrm{err}}}&=\frac{(d_e-T d_l/J)^2}{T-T^2/J},
&\term{robustLag}{\mathrm{LM}^{*}_{\mathrm{lag}}}&=\frac{(d_l-d_e)^2}{J-T}\\
\term{sarma}{\mathrm{SARMA}}&=\mathrm{LM}_{\mathrm{lag}}+\mathrm{LM}^{*}_{\mathrm{err}},
&\term{moranI}{I}&=\frac{n}{S_0}\,\frac{e^\top We}{e^\top e},
&\term{moranZ}{z_I}&=\frac{I-\mathrm{E}[I]}{\sqrt{\mathrm{Var}[I]}}
\end{aligned}`,
      ascii:
        "sigma2 = e'e/n; d_e = e'We/sigma2; d_l = e'Wy/sigma2; T = tr(W'W + WW); J = (W Zb)' M (W Zb)/sigma2 + T with Z = [1, X], M = I - Z(Z'Z)^-1 Z'. LM-error = d_e^2/T; LM-lag = d_l^2/J (df 1); robust LM-error = (d_e - T d_l/J)^2/(T - T^2/J); robust LM-lag = (d_l - d_e)^2/(J - T); LM-SARMA = LM-lag + robust LM-error (df 2). Moran I of the residuals = (n/S0) e'We/e'e with the regression-residual mean and variance, two-sided normal p. Tracts with a non-finite residual are dropped with their links.",
      symbols: [
        {id: 'errorGradient', role: 'intermediate', binding: 'wgsl:errorGradient'},
        {id: 'sigmaSquared', role: 'intermediate', binding: 'wgsl:sigmaSquared'},
        {id: 'lagGradient', role: 'intermediate', binding: 'wgsl:lagGradient'},
        {id: 'traceT', role: 'intermediate', binding: 'wgsl:traceT'},
        {id: 'information', role: 'intermediate', binding: 'wgsl:information'},
        {id: 'errorStatistic', role: 'output', binding: 'wgsl:errorStatistic'},
        {id: 'lagStatistic', role: 'output', binding: 'wgsl:lagStatistic'},
        {id: 'robustError', role: 'output', binding: 'wgsl:robustError'},
        {id: 'robustLag', role: 'output', binding: 'wgsl:robustLag'},
        {id: 'sarma', role: 'output', binding: 'wgsl:sarma'},
        {id: 'moranI', role: 'output', binding: 'wgsl:moranI'},
        {id: 'moranZ', role: 'output', binding: 'wgsl:moranZ'}
      ]
    }
  },
  GPUSpatialTwoStageLeastSquares: {
    family: 'regression',
    descriptor: {scope: 'global', scopeSecondary: 'solver', reduction: 'solve'},
    variables: COEFFICIENT_TABLE_VARIABLES,
    stages: ['neighbourhood', 'operator', 'inference'],
    datumKinds: [{kind: 'coefficient', label: 'coefficient', indexSpace: '1d'}],
    guards: ['row-standardisation'],
    choices: [
      ...WEIGHTS_CHOICES,
      {
        name: 'instrumentOrder',
        label: 'Instrument order',
        stage: 'operator',
        alternatives: [1, 2],
        sweepable: true
      }
    ],
    formula: {
      id: 'spatial-two-stage-least-squares',
      derivation: 'curated',
      tex: String.raw`\begin{aligned}
\term{delta}{\hat\delta}&=(Z^\top PZ)^{-1}Z^\top P\,y,
&Z&=[1,X,Wy],\ P=H(H^\top H)^{-1}H^\top,\ H=[1,X,WX]\\
\term{sigmaSquared}{\hat\sigma^2}&=\frac{u^\top u}{n},\ u=y-Z\hat\delta,
&\term{zStatistic}{z_j}&=\frac{\hat\delta_j}{\term{standardError}{\mathrm{se}_j}},\ \mathrm{se}_j^2=\hat\sigma^2[(Z^\top PZ)^{-1}]_{jj}\\
\term{moranI}{I_u}&=\frac{n}{S_0}\,\frac{u^\top Wu}{u^\top u},
&\term{anselinKelejian}{\mathrm{AK}}&=\frac{n\,I_u^2}{\term{phiSquared}{\varphi^2}}
\end{aligned}`,
      ascii:
        "Z = [1, X, Wy]; H = [1, X, WX] (instrumentOrder 2 adds W^2 X); P = H (H'H)^-1 H'. delta = (Z'PZ)^-1 Z'P y; u = y - Z delta; sigma2 = u'u/n; se_j = sqrt(sigma2 [(Z'PZ)^-1]_jj); z_j = delta_j/se_j with a standard-normal p-value. The constant is centred out; the intercept is ybar - mean(Z) delta. Anselin-Kelejian AK = n I_u^2/phi2 with I_u = (n/S0) u'Wu/u'u tests the residuals for spatial autocorrelation. Classical (non-robust) standard errors.",
      symbols: [
        {id: 'delta', role: 'output', label: 'coefficients with rho last', binding: 'wgsl:delta'},
        {id: 'sigmaSquared', role: 'intermediate', binding: 'wgsl:sigmaSquared'},
        {id: 'zStatistic', role: 'output', binding: 'wgsl:z'},
        {id: 'standardError', role: 'output', binding: 'wgsl:standardError'},
        {id: 'moranI', role: 'intermediate', binding: 'wgsl:moranI'},
        {id: 'anselinKelejian', role: 'output', binding: 'wgsl:anselinKelejian'},
        {id: 'phiSquared', role: 'intermediate', binding: 'wgsl:phiSquared'}
      ]
    }
  },
  GPUSpatialWeightsTraces: {
    family: 'regression',
    descriptor: {scope: 'global', scopeSecondary: 'pairwise', reduction: 'sum'},
    variables: {powerTraces: 'score', powerSums: 'score'},
    stages: ['neighbourhood', 'operator'],
    datumKinds: [{kind: 'power', label: 'matrix power q', indexSpace: '1d'}],
    guards: ['row-standardisation'],
    choices: [
      ...WEIGHTS_CHOICES,
      {
        name: 'mode',
        label: 'Trace mode',
        stage: 'operator',
        alternatives: ['exact', 'stochastic'],
        sweepable: false
      },
      {name: 'maximumPower', label: 'Highest power Q', stage: 'operator', sweepable: true}
    ],
    formula: {
      id: 'spatial-weights-traces',
      derivation: 'curated',
      tex: String.raw`\begin{aligned}
\term{infinityNorm}{s}&=\max_i\sum_j|w_{ij}|\\
\term{powerTraces}{c_q}&=\operatorname{tr}\!\big((W/s)^q\big)=\sum_i e_i^\top (W/s)^q e_i\\
\term{powerSums}{m_q}&=\mathbf 1^\top (W/s)^q\,\mathbf 1
\end{aligned}`,
      ascii:
        "s = max_i sum_j |w_ij| (largest absolute row sum; 1 for row-standardised W). c_q = tr((W/s)^q) = sum_i e_i'(W/s)^q e_i from one unit-vector chain per row (exact mode; stochastic mode averages Rademacher probes z'(W/s)^q z). m_q = 1'(W/s)^q 1 from one chain started at the ones vector. Series downstream run in rho*s.",
      symbols: [
        {id: 'infinityNorm', role: 'output', binding: 'wgsl:infinityNorm'},
        {id: 'powerTraces', role: 'output', binding: 'wgsl:powerTraces'},
        {id: 'powerSums', role: 'output', binding: 'wgsl:powerSums'}
      ]
    }
  },
  GPUSpatialLagImpacts: {
    family: 'regression',
    descriptor: {scope: 'global', reduction: 'sum'},
    variables: {direct: 'score', indirect: 'score', total: 'score'},
    stages: ['operator', 'presentation'],
    datumKinds: [{kind: 'predictor', label: 'predictor', indexSpace: '1d'}],
    guards: ['row-standardisation'],
    choices: [
      {name: 'maximumPower', label: 'Series length Q', stage: 'operator', sweepable: true},
      {name: 'rhoBound', label: 'Largest |rho| s', stage: 'inference', sweepable: false}
    ],
    formula: {
      id: 'spatial-lag-impacts',
      derivation: 'curated',
      tex: String.raw`\begin{aligned}
\term{direct}{D_k}&=\frac{\term{beta}{\beta_k}}{n}\Big(n+\sum_{q=1}^{Q}\term{rho}{\rho}^q\operatorname{tr}W^q\Big)=\frac{\beta_k}{n}\operatorname{tr}(I-\rho W)^{-1}\\
\term{total}{T_k}&=\frac{\beta_k}{n}\Big(n+\sum_{q=1}^{Q}\rho^q\,\mathbf 1^\top W^q\mathbf 1\Big)=\frac{\beta_k}{n}\mathbf 1^\top(I-\rho W)^{-1}\mathbf 1\\
\term{indirect}{N_k}&=T_k-D_k
\end{aligned}`,
      ascii:
        "D_k = beta_k tr((I - rho W)^-1)/n; T_k = beta_k 1'(I - rho W)^-1 1/n; indirect = T_k - D_k. Both inverses are the truncated series n + sum_{q=1..Q} (rho s)^q c_q with the stored powers of W/s (Horner), valid for |rho| s < 1; otherwise NaN, as for a failed fit.",
      symbols: [
        {id: 'direct', role: 'output', binding: 'wgsl:direct'},
        {id: 'beta', role: 'input', label: 'coefficient', binding: 'wgsl:beta'},
        {id: 'rho', role: 'input', label: 'spatial lag parameter', binding: 'wgsl:rho'},
        {id: 'total', role: 'output', binding: 'wgsl:total'},
        {id: 'indirect', role: 'output', binding: 'wgsl:impacts'}
      ]
    }
  },
  GPUSpatialErrorGM: {
    family: 'regression',
    descriptor: {scope: 'global', scopeSecondary: 'solver', reduction: 'solve'},
    variables: COEFFICIENT_TABLE_VARIABLES,
    stages: ['neighbourhood', 'operator', 'inference'],
    datumKinds: [{kind: 'coefficient', label: 'coefficient', indexSpace: '1d'}],
    guards: ['row-standardisation'],
    choices: [
      ...WEIGHTS_CHOICES,
      {
        name: 'estimator',
        label: 'Moment estimator',
        stage: 'operator',
        alternatives: ['homoskedastic', 'kelejian-prucha'],
        sweepable: true
      },
      {
        name: 'traceMode',
        label: 'Trace mode',
        stage: 'inference',
        alternatives: ['exact', 'stochastic'],
        sweepable: false
      }
    ],
    formula: {
      id: 'spatial-error-gm',
      derivation: 'curated',
      tex: String.raw`\begin{aligned}
\term{moments}{r(\lambda)}&=\mathbf A\lambda+\mathbf B\lambda^2-\mathbf g,
&\term{lambda}{\hat\lambda}&=\arg\min_{|\lambda|\le 0.99/\|W\|_\infty}\ r(\lambda)^\top\Psi^{-1}r(\lambda)\\
\term{filteredResponse}{y_s}&=y-\lambda Wy,\quad X_s=X-\lambda WX,
&\hat\beta&=(X_s^\top X_s)^{-1}X_s^\top y_s\\
\term{sigmaSquared}{\hat\sigma^2}&=\frac{u_s^\top u_s}{n},
&\mathrm{Var}(\hat\beta)&=\hat\sigma^2(X_s^\top X_s)^{-1}
\end{aligned}`,
      ascii:
        "Spatial error model y = Xb + u, u = lambda W u + e (spreg GM_Error_Hom, A1 = hom_sc, one iteration; estimator kelejian-prucha is GM_Error). Step 1: OLS residuals u give moments u'A1u, u'A2u with A1 = (W'W - (t/n) I)/(1 + (t/n)^2), t = tr(W'W), A2 = (W + W')/2; r(lambda) = A lambda + B lambda^2 - g is minimised unweighted on a grid in [-0.99/||W||_inf, 0.99/||W||_inf] refined by bisection. Step 2: FGLS betas at that lambda, moments recomputed on their residuals and weighted by psi^-1. Final: y_s = y - lambda W y, X_s = X - lambda W X (filtered constant included); beta = (Xs'Xs)^-1 Xs'ys, sigma2 = u_s'u_s/n. Standard errors treat lambda as known; lambda has its own standard error.",
      symbols: [
        {
          id: 'moments',
          role: 'intermediate',
          label: 'moment residual',
          binding: 'wgsl:getMomentResidual'
        },
        {id: 'lambda', role: 'output', binding: 'wgsl:lambda'},
        {id: 'filteredResponse', role: 'intermediate', binding: 'wgsl:ys'},
        {id: 'sigmaSquared', role: 'intermediate', binding: 'wgsl:sigmaSquared'}
      ]
    }
  },
  GPUSpatialLagMaximumLikelihood: {
    family: 'regression',
    descriptor: {scope: 'global', scopeSecondary: 'solver', reduction: 'solve'},
    variables: COEFFICIENT_TABLE_VARIABLES,
    stages: ['neighbourhood', 'operator', 'inference'],
    datumKinds: [{kind: 'coefficient', label: 'coefficient', indexSpace: '1d'}],
    guards: ['row-standardisation'],
    choices: [
      ...WEIGHTS_CHOICES,
      {name: 'maximumPower', label: 'Series length Q', stage: 'operator', sweepable: true},
      {
        name: 'traceMode',
        label: 'Trace mode',
        stage: 'operator',
        alternatives: ['exact', 'stochastic'],
        sweepable: false
      }
    ],
    formula: {
      id: 'spatial-lag-maximum-likelihood',
      derivation: 'curated',
      tex: String.raw`\begin{aligned}
\term{objective}{\ell(\rho)}&=-\frac n2\ln\frac{\term{sumOfSquares}{\mathrm{SSR}(\rho)}}{n}+\term{logDeterminant}{\ln|I-\rho W|}\\
\mathrm{SSR}(\rho)&=e_0^\top e_0-2\rho\,e_0^\top e_1+\rho^2e_1^\top e_1,
&\ln|I-\rho W|&=-\sum_{q=1}^{Q}\frac{(\rho s)^q}{q}\operatorname{tr}\!\big((W/s)^q\big)\\
\term{rho}{\hat\rho}&=\arg\max_{|\rho|\le B}\ \ell(\rho),
&\hat\beta&=\hat\beta_0-\hat\rho\,\hat\beta_1
\end{aligned}`,
      ascii:
        "Spatial lag model y = rho W y + X b + e (spreg ML_Lag, exact log-determinant replaced by a truncated trace series). e0, e1 are the OLS residuals of y and W y on [1, X]; SSR(rho) = e0'e0 - 2 rho e0'e1 + rho^2 e1'e1; concentrated log-likelihood l(rho) = -(n/2) ln(SSR(rho)/n) + ln|I - rho W| with ln|I - rho W| = -sum_{q=1..Q} (rho s)^q tr((W/s)^q)/q (Horner). rho = global maximum on [-B, B], B = min(0.99, series limit)/||W||_inf: analytic derivative on 1025 grid points, sign changes refined by 32 bisection steps. b = b0 - rho b1; sigma2 = SSR/n; variance = inverse information matrix of (b, rho, sigma2).",
      symbols: [
        {id: 'objective', role: 'output', binding: 'wgsl:getObjective'},
        {id: 'sumOfSquares', role: 'intermediate', binding: 'wgsl:getSumOfSquares'},
        {id: 'logDeterminant', role: 'intermediate', binding: 'wgsl:getSpatialLogDeterminant'},
        {id: 'rho', role: 'output', binding: 'wgsl:rho'}
      ]
    }
  },
  GPUSpatialErrorMaximumLikelihood: {
    family: 'regression',
    descriptor: {scope: 'global', scopeSecondary: 'solver', reduction: 'solve'},
    variables: COEFFICIENT_TABLE_VARIABLES,
    stages: ['neighbourhood', 'operator', 'inference'],
    datumKinds: [{kind: 'coefficient', label: 'coefficient', indexSpace: '1d'}],
    guards: ['row-standardisation'],
    choices: [
      ...WEIGHTS_CHOICES,
      {name: 'maximumPower', label: 'Series length Q', stage: 'operator', sweepable: true},
      {
        name: 'traceMode',
        label: 'Trace mode',
        stage: 'operator',
        alternatives: ['exact', 'stochastic'],
        sweepable: false
      }
    ],
    formula: {
      id: 'spatial-error-maximum-likelihood',
      derivation: 'curated',
      tex: String.raw`\begin{aligned}
\term{objective}{\ell(\lambda)}&=-\frac n2\ln\frac{\term{errorSumOfSquares}{\mathrm{EE}(\lambda)}}{n}+\term{logDeterminant}{\ln|I-\lambda W|}\\
\mathrm{EE}(\lambda)&=y_s^\top y_s-2\hat\beta^\top r+\hat\beta^\top A\hat\beta,
&A(\lambda)&=G_0-\lambda(G_1+G_1^\top)+\lambda^2G_2\\
\term{lambda}{\hat\lambda}&=\arg\max_{|\lambda|\le B}\ \ell(\lambda)
\end{aligned}`,
      ascii:
        "Spatial error model y = Xb + u, u = lambda W u + e (spreg ML_Error, exact log-determinant replaced by a truncated trace series). With Z = [1, X - mean, y - mean], G0 = Z'Z, G1 + G1' = Z'WZ + (Z'WZ)', G2 = (WZ)'(WZ) the Gram matrix of the filtered columns is A(lambda) = G0 - lambda (G1 + G1') + lambda^2 G2; one unit-scaled Cholesky solve per lambda gives beta and EE(lambda) = yy - 2 beta'r + beta'A beta. l(lambda) = -(n/2) ln(EE/n) + ln|I - lambda W|; lambda = global maximum on [-B, B] (1025-point derivative scan, 32 bisection steps). sigma2 = EE/n; Var(b) = sigma2 (Xs'Xs)^-1 treats lambda as known.",
      symbols: [
        {id: 'objective', role: 'output', binding: 'wgsl:getObjective'},
        {id: 'errorSumOfSquares', role: 'intermediate', binding: 'wgsl:errorSumOfSquares'},
        {id: 'logDeterminant', role: 'intermediate', binding: 'wgsl:getSpatialLogDeterminant'},
        {id: 'lambda', role: 'output', binding: 'wgsl:lambda'}
      ]
    }
  }
};
