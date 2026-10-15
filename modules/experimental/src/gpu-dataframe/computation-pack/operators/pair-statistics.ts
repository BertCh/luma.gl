// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {DataFrameFamilyOperators} from '../types';

/** Curated planar pair statistics; formulas state the selected estimator and edge convention. */
export const PAIR_STATISTICS_OPERATORS: DataFrameFamilyOperators = {
  /**
   * Primary: Matheron semivariance, each unordered pair once in cell order, finite selected
   * values and positions inside the window. Lag=min(floor(d/maximumDistance*lagCount),lagCount-1)
   * for d<=maximumDistance; internal edges belong to the higher lag, the outer edge to the last.
   * The i<j sum denotes unordered pairs; focus need not be the lower row, but directional deltas
   * are oriented from lower to higher row ID. Sectors use atan2 modulo pi relative to azimuthOffset.
   * The mathematical sum is
   * realized by quantized 64-bit accumulators (2^24 units per robust variance, large terms at
   * 2^12); empty bins are NaN and overflow is reported, not silently interpreted as exact data.
   * Optional robustSemivariances use (mean sqrt(abs(difference)))^4/(2*(0.457+0.494/N)).
   */
  GPUVariogram: {
    family: 'pair-statistics',
    descriptor: {scope: 'pairwise', reduction: 'sum'},
    formula: {
      id: 'gpu-dataframe:GPUVariogram',
      derivation: 'curated',
      tex: String.raw`\term{semivariances}{\gamma_h}=\frac{1}{2\term{pairCounts}{N_h}}\sum_{\substack{\term{focus}{i}<\term{neighbor}{j}\\\text{pair in lag }\term{lag}{h}}}(\term{values}{z_i-z_j})^2,\quad \term{lag}{h}=\min\left(\left\lfloor\frac{\term{pairDistance}{d}}{\term{maximumDistance}{D}}\term{lagCount}{H}\right\rfloor,\term{lagCount}{H}-1\right)`,
      ascii:
        'gamma[h]=sum_{i<j in h}(z_i-z_j)^2/(2*N[h]); h=min(floor(d/maximumDistance*lagCount),lagCount-1), d<=maximumDistance; fixed-point accumulation.',
      symbols: [
        {id: 'semivariances', role: 'output', binding: 'wgsl:semivariances'},
        {id: 'pairCounts', role: 'output', binding: 'wgsl:pairCounts'},
        {id: 'focus', role: 'index', binding: 'wgsl:focus'},
        {id: 'neighbor', role: 'index', binding: 'wgsl:neighbor'},
        {id: 'lag', role: 'index', binding: 'wgsl:lag'},
        {
          id: 'values',
          role: 'intermediate',
          binding: 'wgsl:difference',
          label: 'included value difference'
        },
        {id: 'pairDistance', role: 'intermediate', binding: 'wgsl:pairDistance'},
        {id: 'maximumDistance', role: 'parameter', binding: 'wgsl:maximumDistance'},
        {id: 'lagCount', role: 'constant', binding: 'wgsl:LAG_COUNT'}
      ]
    },
    stages: ['filter', 'group', 'aggregate', 'summarise'],
    datumKinds: [
      {kind: 'row', label: 'Observations', indexSpace: '1d'},
      {kind: 'pair', label: 'Unordered pairs', indexSpace: '1d'},
      {kind: 'lag', label: 'Directional lags', indexSpace: '1d'}
    ],
    guards: ['coordinate-model'],
    choices: [
      'bounds',
      'maximumDistance',
      'lagCount',
      'directionCount',
      'azimuthOffset',
      'gridSize',
      'studyWindow'
    ],
    variables: {
      positions: {level: 'interval', extent: 'intensive'},
      values: {level: 'ratio', extent: 'intensive'},
      semivariances: {level: 'ratio', extent: 'intensive'},
      pairCounts: {level: 'ratio', extent: 'extensive'}
    },
    sourceFiles: [
      'modules/experimental/src/gpu-dataframe/pair-statistics/gpu-variogram.ts',
      'modules/experimental/src/gpu-dataframe/pair-statistics/pair-histogram.ts'
    ]
  },
  /**
   * Primary: Moran's I, binary symmetric band weights with zero diagonal, no row normalization.
   * Annulus=min(floor(d/D*bandCount),bandCount-1); cumulative mode prefix-sums annuli.
   * productSum is a signed fixed-point sum over ordered pairs; sumOfSquares is the population
   * centered sum (variance=M2/n). I needs n>=3, M2>0 and S0>0; E[I]=-1/(n-1).
   * Analytic z/p use varianceAssumption=randomization by default (fourth moment, n>=4), or
   * normality; variance below 1e-4 of term magnitude is NaN. Published pairCounts are unordered.
   */
  GPUSpatialCorrelogram: {
    family: 'pair-statistics',
    // The mean of ordered neighbor products is normalized by the population variance.
    descriptor: {scope: 'pairwise', reduction: 'weighted-mean'},
    formula: {
      id: 'gpu-dataframe:GPUSpatialCorrelogram',
      derivation: 'curated',
      tex: String.raw`\term{moransI}{I_h}=\frac{\term{count}{n}}{\term{weightSum}{S_{0,h}}}\frac{\term{productSum}{\sum_{i\ne j}w_{ij,h}(x_i-\bar x)(x_j-\bar x)}}{\term{sumOfSquares}{\sum_i(x_i-\bar x)^2}},\quad \term{expectedI}{E[I]}=-\frac{1}{\term{count}{n}-1}`,
      ascii:
        'I[h]=(n/S0[h])*sum_ordered(w_ij*(x_i-mean)*(x_j-mean))/M2; E[I]=-1/(n-1); binary bands, population M2/n, fixed-point products.',
      symbols: [
        {id: 'moransI', role: 'output', binding: 'wgsl:moransI'},
        {id: 'count', role: 'intermediate', binding: 'wgsl:count'},
        {id: 'weightSum', role: 'intermediate', binding: 'wgsl:weightSum'},
        {id: 'productSum', role: 'intermediate', binding: 'wgsl:productSum'},
        {id: 'sumOfSquares', role: 'intermediate', binding: 'wgsl:sumOfSquares'},
        {id: 'expectedI', role: 'output', binding: 'wgsl:expected'}
      ]
    },
    stages: ['filter', 'group', 'aggregate', 'infer'],
    datumKinds: [
      {kind: 'row', label: 'Observations', indexSpace: '1d'},
      {kind: 'pair', label: 'Ordered neighbors', indexSpace: '1d'},
      {kind: 'lag', label: 'Distance bands', indexSpace: '1d'}
    ],
    guards: ['coordinate-model', 'uncorrected-p'],
    choices: [
      'bounds',
      'maximumDistance',
      'bandCount',
      'bandMode',
      'varianceAssumption',
      'gridSize',
      'studyWindow'
    ],
    variables: {
      values: {level: 'ratio', extent: 'intensive'},
      moransI: {level: 'interval', extent: 'intensive'},
      pairCounts: {level: 'ratio', extent: 'extensive'}
    },
    metadata: {pCorrection: 'none'},
    sourceFiles: [
      'modules/experimental/src/gpu-dataframe/pair-statistics/gpu-spatial-correlogram.ts'
    ]
  },
  /**
   * Primary: default isotropic K and derived L. Ordered non-self pairs at d<=r; annulus is
   * clamp(ceil(d/D*radiusCount)-1,0,radiusCount-1), so radius equality is included.
   * f is the fraction of the focus-centered circle within the window. WEIGHT_CAP is the WGSL
   * spelling of GPU_RIPLEY_WEIGHT_CAP=100; weights are quantized as round(w*2^17)/2^17.
   * 'none' uses w=1. 'border' instead uses A*eligiblePairs/(n*eligibleFocusCount), where eligible
   * focuses are at least r from the boundary; it does NOT use n*(n-1). Undefined K is NaN.
   */
  GPURipley: {
    family: 'pair-statistics',
    descriptor: {scope: 'pairwise', reduction: 'count'},
    formula: {
      id: 'gpu-dataframe:GPURipley',
      derivation: 'curated',
      tex: String.raw`\term{weight}{w_{ij}}=\frac{1}{\max(\term{circleFraction}{f_{ij}},1/\term{weightCap}{100})},\quad \term{k}{\widehat K(r)}=\frac{\term{area}{A}}{\term{count}{n}(\term{count}{n}-1)}\sum_{\text{ordered }i\ne j}\term{weight}{w_{ij}}\mathbf{1}[\term{pairDistance}{d_{ij}}\leq\term{radii}{r}],\quad \term{l}{L(r)}=\sqrt{\term{k}{\widehat K(r)}/\pi}`,
      ascii:
        'isotropic: w=min(1/circleFraction,100); K(r)=A*sum_{i!=j}(w_ij*1[d_ij<=r])/(n*(n-1)); L=sqrt(K/pi); weights quantized at 2^17.',
      symbols: [
        {id: 'weight', role: 'intermediate', binding: 'wgsl:weight'},
        {id: 'circleFraction', role: 'intermediate', binding: 'wgsl:inside'},
        {id: 'weightCap', role: 'constant', binding: 'wgsl:WEIGHT_CAP'},
        {id: 'k', role: 'output', binding: 'wgsl:k'},
        {id: 'area', role: 'intermediate', binding: 'wgsl:area'},
        {id: 'count', role: 'intermediate', binding: 'wgsl:n'},
        {id: 'pairDistance', role: 'intermediate', binding: 'wgsl:pairDistance'},
        {id: 'radii', role: 'output', binding: 'wgsl:radii'},
        {id: 'l', role: 'output', binding: 'wgsl:l'}
      ]
    },
    stages: ['filter', 'aggregate', 'summarise'],
    datumKinds: [
      {kind: 'row', label: 'Events', indexSpace: '1d'},
      {kind: 'pair', label: 'Ordered event pairs', indexSpace: '1d'},
      {kind: 'lag', label: 'Radii', indexSpace: '1d'}
    ],
    guards: ['coordinate-model'],
    choices: [
      'bounds',
      'maximumDistance',
      'edgeCorrection',
      'radiusCount',
      'gridSize',
      'studyWindow'
    ],
    variables: {
      positions: {level: 'interval', extent: 'intensive'},
      k: {level: 'ratio', extent: 'intensive'},
      l: {level: 'ratio', extent: 'intensive'},
      pairCounts: {level: 'ratio', extent: 'extensive'}
    },
    sourceFiles: ['modules/experimental/src/gpu-dataframe/pair-statistics/gpu-ripley.ts']
  },
  /**
   * Primary: default border-corrected G/F/J. G uses nearest OTHER event; F uses nearest event
   * from referenceGrid cell centers (only inside centers for polygons). At radius r, retain
   * focuses whose boundary distance is >=r, and count nearest distances <=r. Empty retained
   * sets give NaN; G also needs n>=2, F n>=1; J is NaN if 1-F<=0.
   * Alternatives: none (empirical CDF), kaplan-meier (binned hazards), hanisch (inverse eroded
   * area, 14-bit weights capped at 1024; polygon area uses the sampled reference lattice).
   */
  GPURipleyDistanceFunctions: {
    family: 'pair-statistics',
    descriptor: {scope: 'pairwise', reduction: 'argmin'},
    formula: {
      id: 'gpu-dataframe:GPURipleyDistanceFunctions',
      derivation: 'curated',
      tex: String.raw`\term{nearestDistance}{d_i}=\min_{j\ne i}\term{pairDistance}{d_{ij}},\quad \term{g}{G(r)}=\frac{\sum_{\text{events}}\mathbf{1}[\term{borderDistance}{b_i}\geq\term{radii}{r}]\mathbf{1}[\term{nearestDistance}{d_i}\leq\term{radii}{r}]}{\sum_{\text{events}}\mathbf{1}[\term{borderDistance}{b_i}\geq\term{radii}{r}]},\quad \term{f}{F(r)}=\frac{\term{referenceHits}{\#\{u:b_u\geq r,d_u\leq r\}}}{\term{referenceEligible}{\#\{u:b_u\geq r\}}},\quad \term{j}{J(r)}=\frac{1-\term{g}{G(r)}}{1-\term{f}{F(r)}}`,
      ascii:
        'border: G=#events(b>=r and NN<=r)/#events(b>=r); F=#references(b>=r and NN<=r)/#references(b>=r); J=(1-G)/(1-F).',
      symbols: [
        {id: 'nearestDistance', role: 'intermediate', binding: 'wgsl:nearestDistance'},
        {id: 'pairDistance', role: 'intermediate', binding: 'wgsl:pairDistance'},
        {id: 'borderDistance', role: 'intermediate', binding: 'wgsl:borderDistance'},
        {id: 'radii', role: 'output', binding: 'wgsl:radii'},
        {id: 'g', role: 'output', binding: 'wgsl:g'},
        {id: 'f', role: 'output', binding: 'wgsl:f'},
        {id: 'j', role: 'output', binding: 'wgsl:j'},
        {id: 'referenceHits', role: 'intermediate', binding: 'wgsl:referenceAccumulators'},
        {id: 'referenceEligible', role: 'intermediate', binding: 'wgsl:denominator'}
      ]
    },
    stages: ['filter', 'select', 'aggregate', 'summarise'],
    datumKinds: [
      {kind: 'row', label: 'Events and reference centers', indexSpace: '1d'},
      {kind: 'pair', label: 'Nearest event candidates', indexSpace: '1d'},
      {kind: 'lag', label: 'Radii', indexSpace: '1d'}
    ],
    guards: ['coordinate-model'],
    choices: [
      'bounds',
      'maximumDistance',
      'edgeCorrection',
      'radiusCount',
      'referenceGrid',
      'gridSize',
      'studyWindow'
    ],
    variables: {
      positions: {level: 'interval', extent: 'intensive'},
      g: {level: 'ratio', extent: 'intensive'},
      f: {level: 'ratio', extent: 'intensive'},
      j: {level: 'ratio', extent: 'intensive'}
    },
    sourceFiles: [
      'modules/experimental/src/gpu-dataframe/pair-statistics/gpu-ripley-distance-functions.ts'
    ]
  },
  /**
   * Primary: rank=1 pointwise envelope (finite simulation min/max), transformed from K to the
   * selected statistic (default L-r). Higher rank selects the rank-th smallest/largest;
   * fewer than 2*rank finite simulations at a radius gives NaN bounds, with ties by simulation ID.
   * CSR fixes the included n, samples uniformly with Philox4x32-10 keyed by seed and PURPOSE,
   * counter (row,simulationIndex,0,0); polygons add rejection attempt in counter word 2 and
   * report exhausted draws. Each simulation runs the same Ripley correction. With observedK,
   * T=max_r|sqrt(K/pi)-r| over finite radii and p=(1+#{T_sim>=T_obs})/(s_valid+1).
   * Global bands use the rank-th largest T; pointwise bounds have no multiplicity correction.
   */
  GPURipleyEnvelope: {
    family: 'pair-statistics',
    // Simulations repeatedly count Ripley pairs, followed by rank selection and inference.
    descriptor: {scope: 'pairwise', reduction: 'count'},
    formula: {
      id: 'gpu-dataframe:GPURipleyEnvelope',
      derivation: 'curated',
      tex: String.raw`\term{lower}{E_-(r)}=\min_{\text{finite simulations}}\term{allK}{T(K_s(r))},\quad \term{upper}{E_+(r)}=\max_{\text{finite simulations}}\term{allK}{T(K_s(r))},\quad \term{pValue}{p}=\frac{1+\term{atLeast}{\#\{s:D_s\geq D_{\rm obs}\}}}{\term{validCount}{s_{\rm valid}}+1}`,
      ascii:
        'rank=1: lower(r)=min_sim transform(K_sim(r)), upper(r)=max_sim transform(K_sim(r)); MAD p=(1+#(D_sim>=D_obs))/(validSimulations+1).',
      symbols: [
        {id: 'lower', role: 'output', binding: 'wgsl:lower'},
        {id: 'upper', role: 'output', binding: 'wgsl:upper'},
        {
          id: 'allK',
          role: 'intermediate',
          binding: 'wgsl:allK',
          label: 'simulation K, transformed by fromK'
        },
        {id: 'pValue', role: 'output', binding: 'wgsl:pValue'},
        {id: 'atLeast', role: 'intermediate', binding: 'wgsl:atLeast'},
        {id: 'validCount', role: 'intermediate', binding: 'wgsl:validCount'}
      ]
    },
    stages: ['filter', 'derive', 'aggregate', 'order', 'infer'],
    datumKinds: [
      {kind: 'row', label: 'CSR events', indexSpace: '1d'},
      {kind: 'pair', label: 'Simulated pairs', indexSpace: '1d'},
      {kind: 'lag', label: 'Radii', indexSpace: '1d'},
      {kind: 'simulation', label: 'CSR simulations', indexSpace: '1d'}
    ],
    guards: ['coordinate-model', 'uncorrected-p'],
    choices: [
      'bounds',
      'maximumDistance',
      'edgeCorrection',
      'seed',
      'simulationCount',
      'rank',
      'statistic',
      'radiusCount',
      'gridSize',
      'studyWindow',
      'maximumAttempts'
    ],
    variables: {
      lower: {level: 'interval', extent: 'intensive'},
      upper: {level: 'interval', extent: 'intensive'},
      pValue: {level: 'ratio', extent: 'intensive'}
    },
    metadata: {pCorrection: 'none'},
    sourceFiles: [
      'modules/experimental/src/gpu-dataframe/pair-statistics/gpu-ripley-envelope.ts',
      'modules/experimental/src/gpu-dataframe/pair-statistics/gpu-ripley.ts'
    ]
  },
  /**
   * Primary: Clark-Evans with default rectangular Donnelly correction (0.041, not 0.0412).
   * NN ties choose the smallest row; search is not truncated at maximumDistance. 'none' removes
   * the perimeter term; 'guard' averages boundary-distant focuses but searches all events and
   * uses 0.5*sqrt(A/n), SE=0.26136*sqrt(A/(n*nFocus)). Donnelly is invalid for polygons.
   * Quadrat chi-square=sum((count_q-expected_q)^2/expected_q), expected_q=n*area_q/A; drop
   * zero-area quadrats, df=mEffective-1. Rectangles have equal expected n/m. Polygon areas are
   * sampled and normalized. Clark-Evans uses normal tails; quadrats use gamma tails (Wilson-
   * Hilferty above df=100); alternative selects two-sided/greater/lesser without correction.
   */
  GPUPointPatternIndices: {
    family: 'pair-statistics',
    // NN drives the primary index; the optional quadrat branch instead performs block counts.
    descriptor: {scope: 'pairwise', scopeSecondary: 'block', reduction: 'argmin'},
    formula: {
      id: 'gpu-dataframe:GPUPointPatternIndices',
      derivation: 'curated',
      tex: String.raw`\term{expected}{e}=0.5\sqrt{\term{area}{A}/\term{count}{n}}+(0.0514+0.041/\sqrt{\term{count}{n}})\term{perimeter}{P}/\term{count}{n},\quad \term{ratio}{R}=\frac{\term{observed}{\operatorname{mean}(d_{\rm NN})}}{\term{expected}{e}},\quad \term{chiSquare}{\chi^2}=\sum_{\text{positive-area quadrats}}\frac{(\term{quadratCounts}{c_q}-\term{quadratExpected}{e_q})^2}{\term{quadratExpected}{e_q}}`,
      ascii:
        'Donnelly: R=meanNN/(0.5*sqrt(A/n)+(0.0514+0.041/sqrt(n))*P/n); quadrat chiSquare=sum((c_q-e_q)^2/e_q), e_q=n*a_q/A.',
      symbols: [
        {id: 'expected', role: 'intermediate', binding: 'wgsl:expected'},
        {id: 'area', role: 'intermediate', binding: 'wgsl:area'},
        {id: 'count', role: 'intermediate', binding: 'wgsl:n'},
        {id: 'perimeter', role: 'intermediate', binding: 'wgsl:perimeter'},
        {id: 'ratio', role: 'output', binding: 'wgsl:ratio'},
        {id: 'observed', role: 'intermediate', binding: 'wgsl:observed'},
        {id: 'chiSquare', role: 'output', binding: 'wgsl:chiSquare'},
        {id: 'quadratCounts', role: 'output', binding: 'wgsl:quadratCounts'},
        {id: 'quadratExpected', role: 'intermediate', binding: 'wgsl:expected'}
      ]
    },
    stages: ['filter', 'select', 'aggregate', 'infer'],
    datumKinds: [
      {kind: 'row', label: 'Events', indexSpace: '1d'},
      {kind: 'pair', label: 'Nearest neighbors', indexSpace: '1d'},
      {kind: 'bin', label: 'Quadrats', indexSpace: '1d'}
    ],
    guards: ['coordinate-model', 'uncorrected-p'],
    choices: [
      'bounds',
      'maximumDistance',
      'clarkEvansCorrection',
      'guardDistance',
      'alternative',
      'gridSize',
      'quadratGrid',
      'quadratSamples',
      'studyWindow'
    ],
    variables: {
      positions: {level: 'interval', extent: 'intensive'},
      nearestNeighborDistances: {level: 'ratio', extent: 'intensive'},
      quadratCounts: {level: 'ratio', extent: 'extensive'},
      pValue: {level: 'ratio', extent: 'intensive'}
    },
    metadata: {pCorrection: 'none'},
    sourceFiles: [
      'modules/experimental/src/gpu-dataframe/pair-statistics/gpu-point-pattern-indices.ts'
    ]
  }
};
