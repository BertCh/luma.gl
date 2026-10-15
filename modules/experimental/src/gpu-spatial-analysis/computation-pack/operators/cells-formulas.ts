// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUComputationFormulaRef} from '@luma.gl/gpgpu';

/** Curated cell reductions and measure components, bound to the contributing WGSL identifiers. */
export const CELLS_FORMULAS = {
  aggregation: {
    id: 's3.cell-aggregation',
    derivation: 'curated',
    tex: String.raw`\begin{aligned}\term{counts}{n_c}&=|\term{members}{I_c}|,\quad \term{sums}{S_c}=\sum_{\term{row}{i}\in\term{members}{I_c}}\operatorname{clamp}(\operatorname{roundEven}(\operatorname{f32}(\term{values}{x_i}\term{scale}{s})),-\term{limit}{M},\term{limit}{M})\pmod{2^{64}}\\\term{decoded}{V_c}&=\operatorname{f32}(\term{sums}{S_c})/\term{scale}{s}\end{aligned}`,
    ascii:
      'For each retained cell c, I[c] contains its accepted input rows: counts[c] = |I[c]|; sums[c] = sum(i in I[c]) clamp(roundHalfEven(f32(values[i] * s)), -M, M), modulo 2^64; sumValues[c] = f32(signed sums[c]) / s. M is the largest f32 below 2^62. Masked, invalid-key and non-finite-value rows are excluded; only NaN coordinates are excluded by the Quadbin aggregation keyer (infinite coordinates are clipped). Capacity retains the smallest keys. Optional minimums/maximums select the original accepted values, not quantized sums.',
    symbols: [
      {id: 'counts', role: 'output', binding: 'wgsl:countsOut'},
      {
        id: 'members',
        role: 'intermediate',
        label: 'Accepted rows assigned to retained cell c',
        binding: 'wgsl:rowCells'
      },
      {id: 'sums', role: 'output', label: 'Signed 64-bit fixed-point sum', binding: 'wgsl:sums'},
      {id: 'values', role: 'input', binding: 'wgsl:values'},
      {id: 'scale', role: 'parameter', label: 'sumScale', binding: 'wgsl:SUM_SCALE'},
      {id: 'limit', role: 'constant', binding: 'wgsl:MAXIMUM_SCALED_VALUE'},
      {id: 'decoded', role: 'output', binding: 'wgsl:sumValuesOut'},
      {id: 'row', role: 'index'}
    ]
  },
  rollup: {
    id: 's3.cell-rollup',
    derivation: 'curated',
    tex: String.raw`\term{counts}{n_p}=\sum_{\term{child}{c}\in\term{members}{C_p}}\term{childCounts}{n_c}\pmod{2^{32}},\qquad \term{sums}{S_p}=\sum_{\term{child}{c}\in\term{members}{C_p}}\term{childSums}{S_c}\pmod{2^{64}}`,
    ascii:
      'For each retained parent p, C[p] is the source table rows assigned to p by truncated keys: counts[p] = sum(c in C[p]) childCounts[c] modulo 2^32; sums[p] = sum(c in C[p]) childSums[c] modulo 2^64. No re-quantization occurs; sumValues decodes the signed sum with the existing sumScale. Minimums/maximums reduce the matching source extremes. Source rows must be at sourceResolution; capacity drops largest parent keys and source overflow propagates.',
    symbols: [
      {id: 'counts', role: 'output', binding: 'wgsl:countsOut'},
      {id: 'sums', role: 'output', binding: 'wgsl:sums'},
      {id: 'childCounts', role: 'input', binding: 'wgsl:childCounts'},
      {id: 'childSums', role: 'input', binding: 'wgsl:childSums'},
      {
        id: 'members',
        role: 'intermediate',
        label: 'Source rows mapped to parent p',
        binding: 'wgsl:rowCells'
      },
      {id: 'child', role: 'index'}
    ]
  },
  pyramid: {
    id: 's3.cell-pyramid',
    derivation: 'curated',
    tex: String.raw`\term{counts}{n_{\ell,p}}=\sum_{\term{child}{c}\in\term{members}{C_{\ell,p}}}\term{childCounts}{n_{\ell-1,c}}\pmod{2^{32}},\qquad \term{sums}{S_{\ell,p}}=\sum_{\term{child}{c}\in\term{members}{C_{\ell,p}}}\term{childSums}{S_{\ell-1,c}}\pmod{2^{64}}`,
    ascii:
      'Coarser levels ell >= 1: counts[ell,p] = sum(c in C[ell,p]) counts[ell-1,c] modulo 2^32; sums[ell,p] = sum(c in C[ell,p]) sums[ell-1,c] modulo 2^64. C[ell,p] groups finer rows by the parent key at resolutions[ell]. Level 0 uses GPUCellAggregation: count accepted rows and sum saturated roundHalfEven(f32(value * sumScale)). Intermediate sums/extremes are retained when coarser levels need them. Conservation compares individual level totals, with no filtering or capacity losses between them; levelCounts counts occupied cells, not input events.',
    symbols: [
      {id: 'counts', role: 'output', binding: 'wgsl:countsOut'},
      {id: 'sums', role: 'output', binding: 'wgsl:sums'},
      {id: 'childCounts', role: 'input', binding: 'wgsl:childCounts'},
      {id: 'childSums', role: 'input', binding: 'wgsl:childSums'},
      {
        id: 'members',
        role: 'intermediate',
        label: 'Finer rows assigned to this parent and level',
        binding: 'wgsl:rowCells'
      },
      {id: 'child', role: 'index'}
    ]
  },
  selection: {
    id: 's3.cell-level-selection',
    derivation: 'curated',
    tex: String.raw`\term{level}{\ell}=\min(\term{active}{a},|\term{counts}{N}|-1),\qquad \term{count}{n}=\term{counts}{N}[\term{level}{\ell}],\quad \term{first}{f}=\term{firstRows}{F}[\term{level}{\ell}]`,
    ascii:
      'ell = min(activeLevel[0], levelCounts.length - 1); status.count = levelCounts[ell]; firstRow = levelFirstRows[ell] (zero offsets by default). Optional drawArguments[1] = status.count and drawArguments[3] = firstRow; the other draw words are untouched.',
    symbols: [
      {id: 'level', role: 'intermediate', binding: 'wgsl:level'},
      {id: 'active', role: 'input', binding: 'wgsl:activeLevel'},
      {id: 'counts', role: 'input', binding: 'wgsl:levelCounts'},
      {id: 'count', role: 'output', binding: 'wgsl:countOut'},
      {id: 'first', role: 'output', binding: 'wgsl:firstRowOut'},
      {
        id: 'firstRows',
        role: 'parameter',
        label: 'levelFirstRows',
        binding: 'wgsl:LEVEL_FIRST_ROWS'
      }
    ]
  },
  gridPath: {
    id: 's3.cell-grid-path',
    derivation: 'curated',
    tex: String.raw`\term{difference}{\Delta}=\term{originLocal}{u}-\term{destinationLocal}{v},\qquad \term{distance}{d}=\max(\term{difference}{\Delta_x},\term{difference}{\Delta_y},\term{difference}{\Delta_z})-\min(\term{difference}{\Delta_x},\term{difference}{\Delta_y},\term{difference}{\Delta_z})`,
    ascii:
      'Distance component for valid origin-anchored local IJK coordinates u and v: Delta = u - v; d = max(Delta.x, Delta.y, Delta.z) - min(Delta.x, Delta.y, Delta.z). This is the maximum component after IJK normalization. Invalid/masked local conversions write 0xffffffff. Optional paths use sequential integer cube rounding and unique-neighbor lookup, not this closed-form distance alone; failed path lookup writes counts=0 while retaining distance. Successful stored prefixes report distance+1 even when capacity truncates them.',
    symbols: [
      {
        id: 'difference',
        role: 'intermediate',
        label: 'IJK difference before normalization',
        binding: 'wgsl:difference'
      },
      {
        id: 'originLocal',
        role: 'intermediate',
        label: 'Origin local IJK',
        binding: 'wgsl:originLocal'
      },
      {
        id: 'destinationLocal',
        role: 'intermediate',
        label: 'Destination local IJK',
        binding: 'wgsl:destinationLocal'
      },
      {id: 'distance', role: 'output', binding: 'wgsl:outDistances'}
    ]
  },
  measures: {
    id: 's3.cell-measures',
    derivation: 'curated',
    tex: String.raw`\begin{aligned}\term{edgeLength}{L_e}&=\term{radius}{R}\operatorname{atan2}(\lVert\term{first}{a_e}\times\term{second}{b_e}\rVert,\term{first}{a_e}\cdot\term{second}{b_e}),\quad \term{perimeter}{P}=\sum_{\term{edge}{e}}\term{edgeLength}{L_e}\\\term{area}{A}&=2\term{radius}{R}^{2}\sum_{\term{edge}{e}}\left|\operatorname{atan2}\left(\term{center}{c}\cdot(\term{first}{a_e}\times\term{second}{b_e}),1+\term{first}{a_e}\cdot\term{second}{b_e}+\term{second}{b_e}\cdot\term{center}{c}+\term{center}{c}\cdot\term{first}{a_e}\right)\right|\end{aligned}`,
    ascii:
      'For valid H3 cells, c is the center unit vector and a[e], b[e] are consecutive boundary unit vectors. L[e] = R * atan2(length(cross(a[e], b[e])), dot(a[e], b[e])); P = sum(e) L[e]; A = 2*R^2 * sum(e) abs(atan2(dot(c,cross(a[e],b[e])), 1+dot(a[e],b[e])+dot(b[e],c)+dot(c,a[e]))). R is the WGS84 authalic radius in the selected meters/kilometers unit. Single-face hexagons evaluate equivalent gnomonic expressions using integer lattice offsets to avoid cancellation; pentagons and face-crossing cells use f32 unit vectors directly. Totals use all boundary edges even if edgeLengths storage is truncated. Invalid cells give zero.',
    symbols: [
      {id: 'edgeLength', role: 'output', binding: 'wgsl:edgeLengthsOut'},
      {id: 'perimeter', role: 'output', binding: 'wgsl:perimetersOut'},
      {id: 'area', role: 'output', binding: 'wgsl:areasOut'},
      {
        id: 'radius',
        role: 'parameter',
        label: 'Authalic radius in selected units',
        binding: 'wgsl:EARTH_RADIUS'
      },
      {
        id: 'first',
        role: 'intermediate',
        label: 'First decoded boundary unit vector',
        binding: 'wgsl:a'
      },
      {
        id: 'second',
        role: 'intermediate',
        label: 'Next decoded boundary unit vector',
        binding: 'wgsl:b'
      },
      {id: 'center', role: 'intermediate', binding: 'wgsl:center'},
      {id: 'edge', role: 'index'}
    ]
  },
  compare: {
    id: 's3.cell-table-compare',
    derivation: 'curated',
    tex: String.raw`\begin{aligned}\term{delta}{\delta}&=\term{decode}{D}(\term{afterInteger}{A}-\term{beforeInteger}{B})/\term{scale}{s},\quad \term{ratio}{r}=\term{after}{a}/\term{before}{b},\quad \term{percent}{p}=100\term{delta}{\delta}/\term{before}{b}\\\term{score}{z}&=\begin{cases}\term{delta}{\delta}/\sqrt{\term{before}{b}+\term{after}{a}}&\text{Poisson}\\(\term{delta}{\delta}-\term{mean}{\mu})/\term{deviation}{\sigma}&\text{standardized}\end{cases}\end{aligned}`,
    ascii:
      'On retained union rows, A/B are after/before integer measures (zero when absent), D decodes a signed 64-bit word pair to f32, and s is sumScale for sums or 1 for counts. a=D(A)/s, b=D(B)/s; delta=D(A-B)/s, with integer subtraction modulo 2^64 before decoding; ratio=a/b and percentChange=100*delta/b (both NaN for b=0). Poisson z=delta/sqrt(b+a), zero for b+a=0. Standardized z=(delta-mu)/sigma, zero for sigma=0; mu and population sigma are reduced in fixed order over retained union deltas, not dropped rows. Empty tail derived values are NaN.',
    symbols: [
      {id: 'delta', role: 'output', binding: 'wgsl:deltaOut'},
      {id: 'decode', role: 'intermediate', binding: 'wgsl:cellI64ToF32'},
      {id: 'afterInteger', role: 'input', binding: 'wgsl:afterMeasure'},
      {id: 'beforeInteger', role: 'input', binding: 'wgsl:beforeMeasure'},
      {
        id: 'scale',
        role: 'parameter',
        label: 'sumScale for sums, 1 for counts',
        binding: 'wgsl:SUM_SCALE'
      },
      {id: 'ratio', role: 'output', binding: 'wgsl:ratioOut'},
      {id: 'percent', role: 'output', binding: 'wgsl:percentOut'},
      {id: 'after', role: 'intermediate', binding: 'wgsl:afterValues'},
      {id: 'before', role: 'intermediate', binding: 'wgsl:beforeValues'},
      {id: 'score', role: 'output', binding: 'wgsl:zScoreOut'},
      {id: 'mean', role: 'intermediate', binding: 'wgsl:mean'},
      {id: 'deviation', role: 'intermediate', binding: 'wgsl:deviation'}
    ]
  }
} as const satisfies Readonly<Record<string, GPUComputationFormulaRef>>;
