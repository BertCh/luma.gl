// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {DataFrameFamilyOperators} from '../types';

/** Curated classification formulas; each entry documents its primary output and method. */
export const CLASSIFICATION_OPERATORS: DataFrameFamilyOperators = {
  /**
   * Primary: linear (R-7) quantiles of finite, unmasked rows, with signed zero canonicalized.
   * R denotes a separate f32 rounding, including subtraction and multiplication (no FMA).
   * The upper neighbor is clamped at n-1. A zero fraction or equal endpoints returns low.
   * An overflowing high-low uses R(R(low * R(1-t)) + R(high*t)); empty samples and invalid
   * probabilities return NaN. Other interpolation choices: lower, higher, nearest, midpoint.
   */
  GPUColumnQuantiles: {
    family: 'classification',
    descriptor: {scope: 'global', reduction: 'select'},
    formula: {
      id: 'gpu-dataframe:GPUColumnQuantiles',
      derivation: 'curated',
      tex: String.raw`\term{rank}{r}=\operatorname{R}(\operatorname{R}(\term{validCount}{n}-1)\term{probability}{p}),\quad \term{fraction}{t}=\operatorname{R}(\term{rank}{r}-\lfloor\term{rank}{r}\rfloor),\quad \term{quantiles}{Q}=\operatorname{R}(\term{lower}{x_{\lfloor r\rfloor}}+\operatorname{R}(\term{fraction}{t}\operatorname{R}(\term{upper}{x_{\min(\lfloor r\rfloor+1,n-1)}}-\term{lower}{x_{\lfloor r\rfloor}})))`,
      ascii:
        'r=fround(fround(n-1)*p); t=fround(r-floor(r)); Q=fround(low+fround(t*fround(high-low))) on finite unmasked rows; convex fallback on difference overflow.',
      symbols: [
        {id: 'quantiles', role: 'output', binding: 'wgsl:quantilesOut'},
        {id: 'validCount', role: 'output', binding: 'wgsl:validCountOut'},
        {id: 'probability', role: 'parameter', binding: 'wgsl:p'},
        {id: 'rank', role: 'intermediate', binding: 'wgsl:h'},
        {id: 'fraction', role: 'intermediate', binding: 'wgsl:fraction'},
        {id: 'lower', role: 'intermediate', binding: 'wgsl:low'},
        {id: 'upper', role: 'intermediate', binding: 'wgsl:high'}
      ]
    },
    stages: ['filter', 'order', 'select'],
    datumKinds: [
      {kind: 'row', label: 'Selected rows', indexSpace: '1d'},
      {kind: 'quantile', label: 'Quantile probabilities', indexSpace: '1d'}
    ],
    guards: ['count-vs-rate'],
    choices: ['quantiles', 'interpolation', 'filterRange', 'quantileCount'],
    variables: {
      values: {level: 'ratio', extent: 'intensive'},
      quantiles: {level: 'ratio', extent: 'intensive'}
    },
    sourceFiles: [
      'modules/experimental/src/gpu-dataframe/column-classification/column-quantiles-nodes.ts',
      'modules/experimental/src/gpu-dataframe/column-classification/order-statistic-wgsl.ts'
    ]
  },
  /**
   * Primary: equal-interval edges, bounded by the finite sample extremes. Count is zero for an
   * empty sample and one for a constant sample. The class count is the number of inner edges
   * at or below the value (bisect-right); out-of-domain values stay in the end classes.
   * Other method choices: quantile, standard-deviation, head-tail, box-plot, maximum-breaks,
   * natural-breaks and custom. Finite unmasked values alone contribute to class counts.
   */
  GPUClassBreaks: {
    family: 'classification',
    descriptor: {scope: 'global', scopeSecondary: 'local', reduction: 'select'},
    formula: {
      id: 'gpu-dataframe:GPUClassBreaks',
      derivation: 'curated',
      tex: String.raw`\term{breaks}{e_i}=\term{minimum}{a}+\frac{\term{maximum}{b}-\term{minimum}{a}}{\term{classCount}{k}}\term{edge}{i},\quad \term{classIndex}{c(x)}=\sum_{\term{edge}{i}=1}^{\term{classCount}{k}-1}\mathbf{1}[\term{breaks}{e_i}\leq\term{values}{x}]`,
      ascii:
        'equal-interval: e[i]=min+((max-min)/k)*i; class(x)=number of inner edges <= x (bisect-right, end-clamped).',
      symbols: [
        {id: 'breaks', role: 'output', binding: 'wgsl:breaksOut'},
        {id: 'minimum', role: 'intermediate', binding: 'wgsl:minimum'},
        {id: 'maximum', role: 'intermediate', binding: 'wgsl:maximum'},
        {id: 'classCount', role: 'output', binding: 'wgsl:classCountOut'},
        {id: 'edge', role: 'index', binding: 'wgsl:edge'},
        {
          id: 'classIndex',
          role: 'intermediate',
          binding: 'wgsl:low',
          label: 'bisect-right low minus one'
        },
        {id: 'values', role: 'input', binding: 'wgsl:values'}
      ]
    },
    stages: ['filter', 'summarise', 'classify'],
    datumKinds: [
      {kind: 'row', label: 'Rows', indexSpace: '1d'},
      {kind: 'class', label: 'Classes', indexSpace: '1d'}
    ],
    guards: ['count-vs-rate'],
    choices: [
      'method',
      'classCount',
      'methods',
      'maximumClassCount',
      'standardDeviationInterval',
      'headTailRatio',
      'boxPlotHinge',
      'customEdges',
      'naturalBreaksBinCount'
    ],
    variables: {
      values: {level: 'ratio', extent: 'intensive'},
      classCounts: {level: 'ratio', extent: 'extensive'}
    },
    sourceFiles: [
      'modules/experimental/src/gpu-dataframe/column-classification/gpu-class-breaks.ts'
    ]
  },
  /**
   * Primary: threshold/quantile scale with step colors. Comparison keys equate signed zeros;
   * NaN/masked rows get NO_CLASS and noDataColor. The palette clamps the class if shorter.
   * Continuous linear/sqrt/pow/log/symlog scales may instead blend each RGBA channel as
   * floor(a+(b-a)*fraction+0.5), clamped to [0,255], between adjacent ramp stops. Threshold,
   * quantile, quantize and ordinal modes remain step colors even with interpolation=linear.
   */
  GPUColorScale: {
    family: 'classification',
    descriptor: {scope: 'local', reduction: 'none'},
    formula: {
      id: 'gpu-dataframe:GPUColorScale',
      derivation: 'curated',
      tex: String.raw`\term{classIndices}{c}=\min\left(\sum_{\term{edge}{i}=1}^{\term{domainCount}{D}-2}\mathbf{1}[\term{domain}{e_i}\leq\term{values}{x}],\term{paletteCount}{P}-1\right),\quad \term{colors}{C}=\term{palette}{\mathrm{palette}}[\term{classIndices}{c}]`,
      ascii:
        'threshold: class=min(number of inner domain edges <= value, paletteCount-1); color=palette[class]; continuous linear interpolation blends RGBA channels half-up.',
      symbols: [
        {id: 'classIndices', role: 'output', binding: 'wgsl:classes'},
        {id: 'colors', role: 'output', binding: 'wgsl:colors'},
        {id: 'edge', role: 'index', binding: 'wgsl:middle'},
        {id: 'domainCount', role: 'parameter', binding: 'wgsl:domainCount'},
        {id: 'paletteCount', role: 'parameter', binding: 'wgsl:paletteCount'},
        {id: 'domain', role: 'input', binding: 'wgsl:domain'},
        {id: 'values', role: 'input', binding: 'wgsl:values'},
        {id: 'palette', role: 'input', binding: 'wgsl:palette'}
      ]
    },
    stages: ['classify', 'derive'],
    datumKinds: [
      {kind: 'row', label: 'Rows', indexSpace: '1d'},
      {kind: 'class', label: 'Palette classes', indexSpace: '1d'}
    ],
    guards: ['count-vs-rate'],
    choices: [
      'scale',
      'domainCount',
      'paletteCount',
      'interpolation',
      'clamp',
      'noDataColor',
      'logFloor',
      'exponent'
    ],
    variables: {
      values: {level: 'ratio', extent: 'intensive'},
      classIndices: {level: 'ordinal'},
      colors: {level: 'nominal'}
    },
    sourceFiles: [
      'modules/experimental/src/gpu-dataframe/column-classification/gpu-color-scale.ts',
      'modules/experimental/src/gpu-dataframe/column-classification/color-scale-wgsl.ts'
    ]
  },
  /** Primary: joint class id; each axis uses bisect-right inner edges, clamped to end classes. */
  GPUBivariateClassification: {
    family: 'classification',
    descriptor: {scope: 'local', reduction: 'none'},
    formula: {
      id: 'gpu-dataframe:GPUBivariateClassification',
      derivation: 'curated',
      tex: String.raw`\term{classIds}{c}=\term{classY}{c_y}\term{classCountX}{k_x}+\term{classX}{c_x}`,
      ascii:
        'classIds=classY*classCountX+classX; each axis counts inner edges <= value; NaN/masked rows are 0xffffffff.',
      symbols: [
        {id: 'classIds', role: 'output', binding: 'wgsl:classes'},
        {id: 'classY', role: 'intermediate', binding: 'wgsl:classY'},
        {id: 'classCountX', role: 'parameter', binding: 'wgsl:classCountX'},
        {id: 'classX', role: 'intermediate', binding: 'wgsl:classX'}
      ]
    },
    stages: ['classify', 'derive'],
    datumKinds: [
      {kind: 'row', label: 'Rows', indexSpace: '1d'},
      {kind: 'class', label: 'Joint classes', indexSpace: '1d'}
    ],
    guards: ['count-vs-rate'],
    choices: ['classCountX', 'classCountY', 'maximumClassCount', 'noDataColor', 'valueByAlpha'],
    variables: {
      valuesX: {level: 'ratio', extent: 'intensive'},
      valuesY: {level: 'ratio', extent: 'intensive'},
      classIds: {level: 'nominal'}
    },
    sourceFiles: [
      'modules/experimental/src/gpu-dataframe/column-classification/gpu-bivariate-classification.ts'
    ]
  },
  /**
   * Primary: numeric profile moments. Masked rows contribute nothing; NaNs are nulls. Infinities
   * count and enter min/max but not finiteCount, sum or moments. Welford/Chan merges produce
   * M2=sum_finite((x-mean)^2), without deriving it by subtracting raw squared sums;
   * variance=M2/n, sampleVariance=M2/(n-1), standardDeviation=sqrt(variance). Empty moments are
   * NaN (sample variance also for n<2). The other outputs are histogram, HLL distinct estimate,
   * category overflow and top categories. Category moments are NaN, null code is 0xffffffff.
   */
  GPUColumnProfile: {
    family: 'classification',
    descriptor: {scope: 'global', reduction: 'histogram'},
    formula: {
      id: 'gpu-dataframe:GPUColumnProfile',
      derivation: 'curated',
      tex: String.raw`\term{mean}{\mu}=\frac{\term{sum}{S}}{\term{finiteCount}{n_f}},\quad \term{variance}{v}=\frac{\term{m2}{M_2}}{\term{finiteCount}{n_f}},\quad \term{statistics}{\mathrm{statistics}}=(\term{count}{n},\term{nullCount}{n_{\rm null}},\term{minimum}{\min x},\term{maximum}{\max x},\term{sum}{S},\term{mean}{\mu},\term{variance}{v},\frac{\term{m2}{M_2}}{\term{finiteCount}{n_f}-1},\sqrt{\term{variance}{v}},\term{distinctEstimate}{\widehat D},0)`,
      ascii:
        'numeric statistics=(count,nullCount,min,max,sum,mean,M2/nFinite,M2/(nFinite-1),sqrt(M2/nFinite),HLL,0); infinities only enter count/extrema.',
      symbols: [
        {id: 'statistics', role: 'output', binding: 'wgsl:statistics'},
        {id: 'count', role: 'intermediate', binding: 'wgsl:count'},
        {id: 'nullCount', role: 'intermediate', binding: 'wgsl:nullCount'},
        {
          id: 'minimum',
          role: 'intermediate',
          binding: 'wgsl:minimumKey',
          label: 'decoded minimum key'
        },
        {
          id: 'maximum',
          role: 'intermediate',
          binding: 'wgsl:maximumKey',
          label: 'decoded maximum key'
        },
        {id: 'sum', role: 'intermediate', binding: 'wgsl:sum'},
        {id: 'mean', role: 'intermediate', binding: 'wgsl:mean'},
        {id: 'variance', role: 'intermediate', binding: 'wgsl:variance'},
        {id: 'm2', role: 'intermediate', binding: 'wgsl:m2'},
        {id: 'finiteCount', role: 'intermediate', binding: 'wgsl:finiteCount'},
        {id: 'distinctEstimate', role: 'intermediate', binding: 'wgsl:estimate'}
      ]
    },
    stages: ['filter', 'aggregate', 'summarise'],
    datumKinds: [
      {kind: 'row', label: 'Rows', indexSpace: '1d'},
      {kind: 'column', label: 'Columns', indexSpace: '1d'},
      {kind: 'bin', label: 'Histogram bins', indexSpace: '1d'}
    ],
    guards: ['count-vs-rate'],
    choices: [
      'histogramBinCount',
      'hyperLogLogPrecision',
      'topCategoryCount',
      'kind',
      'categoryCount',
      'minimum',
      'maximum'
    ],
    variables: {
      values: {level: 'ratio', extent: 'intensive'},
      counts: {level: 'ratio', extent: 'extensive'},
      histograms: {level: 'ratio', extent: 'extensive'}
    },
    sourceFiles: ['modules/experimental/src/gpu-dataframe/column-profile/column-profile-kernels.ts']
  }
};
