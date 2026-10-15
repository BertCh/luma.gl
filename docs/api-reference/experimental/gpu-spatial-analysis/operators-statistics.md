---
title: "GPU Spatial Analysis: Spatial statistics"
sidebar_label: "Spatial statistics"
description: "Spatial statistics contributors in @luma.gl/experimental/gpu-spatial-analysis."
---

import {ExperimentalDocsTabs} from '@site/src/components/docs/experimental-docs-tabs';

# Spatial statistics

<ExperimentalDocsTabs active="gpu-spatial-analysis-operators" />

### `GPUGlobalSpatialStatistics`

Summarise a tract column with global Moran’s I, Geary’s C and General G from queen weights, each with an analytic z-score and p-value.

::lens{scenario=gpu-spatial-analysis.global-spatial-statistics rung=result caption="Global Moran I = 0.60 (z 29.8), Geary and G for 791 Chicago tracts, with p-values."}

Global spatial autocorrelation from a `GPUSpatialWeights` CSR and a value column, each statistic with
its analytic expectation, variance, z-score and normal p-value (`alternative`, default two-sided).
Pick the statistics with `statistics` (compile-time). The `output.summary` layout (`GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT`) is fixed,
and blocks that were not requested hold NaN. `values` and `secondValues` may be `float32` or
`uint32` (for example the `uint32` counts of `addPointsInPolygonsChoroplethRecipe`); counts are
converted to f32 by one node, exact below 2^24, so the results equal those of an f32 copy bit for bit.

- `moran`: Moran's I with esda's `VI_norm` and `VI_rand` (sample kurtosis).
- `geary`: Geary's C with `VC_norm` and `VC_rand`.
- `getisOrdG`: Getis-Ord General G, which needs non-negative ratio-scale data: when any analysis
  value is negative the whole G block is NaN. Its randomization variance
  uses the Getis–Ord analytical moments, evaluated in centered form to avoid f32 cancellation in
  `E[G^2] - E[G]^2`. Both `varianceNormality` and `varianceRandomization` report this variance;
  their z and p fields use the same normal approximation. These are analytical results, not
  sampled permutations. [esda.G](https://pysal.org/esda/_modules/esda/getisord.html) calls these
  `VG` and `z_norm` and binarizes its weights. Match it with binary weights and `islands: 'keep'`.
  This card retains row-standardized weights for all three statistics, so its G is the weighted
  generalization. The former NaN normality fields were unpopulated slots, not failed arithmetic.
  Zeros are allowed when the pair-product denominator is positive; a zero denominator makes G
  and its variance undefined. Positive constant values give G = E[G], variance zero and NaN z/p.
- `bivariateMoran`: `(n / S0) sum w_ij zx_i zy_j / sqrt(sum zx^2 sum zy^2)`, which equals esda's
  `Moran_BV` for row-standardized weights. The exact variance under randomization of `y` is
  `E = 0`. esda itself offers only permutation inference.
- `joinCount`: BB, BW and WW joins of a binary column (nonzero is black) over binarized weights.
  Counts are exact integers, also available as ordered-pair `output.joinCounts`. Cliff-Ord randomization
  moments are given for BB and BW.

```ts
graph.add(new GPUGlobalSpatialStatistics({
  weights, values, statistics: ['moran', 'geary', 'getisOrdG', 'joinCount'],
  output: {summary, joinCounts}
}));
// summary[GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.moran + GPU_GLOBAL_SPATIAL_STATISTIC_FIELD.zRandomization]
```

Neighbors, islands and re-standardization follow the shared rules of the local statistics (see
`GPUHotSpotAnalysis` and `GPULocalMoran`):

- **Neighbors.** A neighbor is a slot with a positive weight to another analysis row. Zero-weight
  slots, the row's own slot and out-of-range IDs are not neighbors.
- **Re-standardization.** Weights are used as given, except when `weights.information.transform` is `'row'`:
  each row is then divided by the sum of its active weights everywhere (`S0`, `S1`, `S2`, the reverse
  weights `w'_ji` and the cross products), so masked, missing and island neighbors drop out and the
  row still sums to 1 (esda on the subset `W`, re-row-standardized).
- **Islands.** An island is an included row (mask nonzero, every value finite) with no active slot.
  The rule looks at out-links only; for symmetric weights it equals the old in-or-out rule.
  `islands: 'exclude'` (default, compile-time) removes islands from `n`, the moments, `S0`, `S1`,
  `S2` and from the neighbors of other rows, in one pass with no cascade (a row whose only neighbors
  were islands keeps its place in `n` and adds no weight). `islands: 'keep'` is esda's convention.
  The summary `islandCount` reports the islands under either policy. esda parity needs `'keep'`.
- **Truncated weights.** If `weights.status.overflow` is nonzero the weights are incomplete, so the
  weight sums, the mean, the variance and every statistic block are NaN (`n` and `islandCount` stay;
  the `output.joinCounts` view holds 0).
- **`alternative`** (`'two-sided'` default, `'greater'`, `'lesser'`; compile-time) selects
  the tail of every analytic p field, including the join counts: `'greater'` is the upper tail of z
  (for example positive autocorrelation of Moran's I), `'lesser'` the lower tail. `'directed'` and
  `'folded'` are permutation conventions and throw. `getInferenceMetadata()` returns a
  `GPUInferenceMetadata` for a legend. Under non-Gaussian data prefer the randomization fields.

The summary block reports `n`, `S0`, `S1`, `S2`, the mean, the variance, the black count and the
island count. `n` is counted in `u32` and the mean gets a second-pass correction
(`mean += sum(x - mean) / n`), so a large offset does not bias the variance; the row count must be
below `2^24`, the limit of the local statistics, because degrees and counts are f32 columns.
`w_ji` comes from a binary search of row `j`. Column sums come from a deterministic transpose: a
stable sort of slots by neighbor. Per-row partials are reduced by fixed-shape workgroup trees, and
join counts by u32 atomics. Results are bitwise reproducible, and values, weights and the mask
change per frame without a rebuild. Inference is NaN when `n < 4`, `S0 = 0`, or the values are
constant (G reports zero variance for positive constant values, with NaN z/p). Constant values give NaN Moran's I and Geary's C: the centering corrects the f32 mean
quotient with one `fma` remainder step, so constant data centers to exactly 0 on every adapter.
`E[I]` and `E[C]` are NaN below two rows (join counts of an empty input stay 0).

#### How it computes

{/* lens:GPUGlobalSpatialStatistics */}

::lens{scenario=gpu-spatial-analysis.global-spatial-statistics rung=datum bind=page datum=420 caption="Inspect tract 420: its poverty value is one of 791 rows entering the global sums."}
::lens{scenario=gpu-spatial-analysis.global-spatial-statistics rung=formula caption="Follow 4 equations: Moran’s I, its expectation, the randomisation z-score and the two-sided p-value."}
::lens{scenario=gpu-spatial-analysis.global-spatial-statistics rung=kernel caption="Inspect the row-gather, column-sum and finalize kernels with their retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.global-spatial-statistics rung=graph caption="Trace queen weights, row standardisation and the fixed-order reductions into 52 summary values."}

### `GPUHotSpotAnalysis` and `GPULocalMoran`

Score tract values against neighbouring values with local Moran’s I, analytic conditional-randomisation z-scores, and corrected significance.

::lens{scenario=gpu-spatial-analysis.local-moran rung=result caption="Compare poverty percentages and local I across 791 Chicago tracts; significance is reported separately."}

`GPUHotSpotAnalysis`: Score each tract with Getis-Ord Gi* from binary queen weights, returning z-scores, 90, 95 and 99% confidence bins and corrected p-values.

::lens{scenario=gpu-spatial-analysis.hot-spot-analysis rung=result caption="Gi* z-scores for 791 Chicago tracts; 265 are significant at 95% after Benjamini-Hochberg correction."}

Local spatial autocorrelation on caller-supplied square `weights`. Rows count when their `mask` is
nonzero and their value is finite; excluded rows are neither foci nor neighbors. `n`, the mean `X`
and the population standard deviation `S` are taken over the analysis rows (see islands below), or
from per-frame `fixedMoments`. The mean gets a second-pass correction (`mean += sum(x - mean) / n`),
so a large offset does not bias the variance, and `n` is counted in `u32`. Sums run in slot order
with fixed-order reductions, so outputs are bitwise reproducible.

Shared semantics (also `GPULocalGeary`):

- **Inputs and outputs.** `values` is `float32` or `uint32` (counts are converted to f32 by one
  node, exact below 2^24), so the `uint32` counts of a points-in-polygons aggregation chain without a
  cast. Every caller-owned result view lives in one `output` bag (`GPUHotSpotAnalysisOutput`,
  `GPULocalMoranOutput`, `GPULocalGearyOutput`); the CPU settings type is
  `GPUSpatialAutocorrelationSettings`.
- **Neighbors.** A neighbor is a slot with a positive weight that points at another included row.
  Zero-weight slots, the focal row's own slot and out-of-range IDs are not neighbors: they are not
  counted in `neighborCounts` or `minimumNeighbors`. If `weights.status.overflow` is nonzero the
  weights were truncated and every output is NaN.
- **Re-standardization.** Weights are used as given, except when `weights.information.transform` is
  `'row'`: then each row's weights are divided by their sum over the active neighbors, so masked or
  missing neighbors drop out and the row still sums to 1 (esda on the subset `W`, re-row-standardized,
  not a lag that shrinks toward 0). Without masking this changes nothing beyond rounding. Weights
  with no declared transform are never rescaled.
- **Islands.** An island is an included row with no neighbor among the included rows.
  `islands: 'exclude'` (default, compile-time) removes islands from `n`, the mean and the variance and
  writes NaN for every output. `islands: 'keep'` is the esda convention: islands stay in the moments;
  their local statistic is the formula value (local I and the spatial lag are 0, local Geary is 0),
  and z and p are NaN because they are not tested. The decision is one pass with no cascade: a row
  whose only neighbors were islands has no neighbor among the analysis rows, gets NaN, and stays in
  `n`. `globalStatistics` is `[n, mean, variance, sd, islandCount]` under either policy. esda
  parity specs use `'keep'`.
- **`minimumNeighbors`** (default 1, compile-time integer). Rows with fewer neighbors get NaN z and p
  and bin or quadrant 0; the moments are unchanged. The default only rejects rows without a
  neighbor, since no row can have a z-score from its own value. Esri recommends at least 8
  neighbors for Gi* (pass `minimumNeighbors: 8`); the normal approximation of local Moran is poor
  below about 4.
- **`multipleTesting`** (`'none'` default, `'bonferroni'`, `'holm'`, `'benjamini-hochberg'`,
  `'benjamini-yekutieli'`, compile-time) adjusts the two-sided p-values through
  `GPUMultipleTestingAdjustment`; the family is every row with a defined z-score, independent of any
  viewport. The optional `output.adjustedPValues` holds the adjusted values (equal to `pValues` for
  `'none'`).
  `getInferenceMetadata(settings)` takes the same `GPUSpatialAutocorrelationSettings` that are packed
  into `parameters` (validated identically) and returns a `GPUInferenceMetadata` (two-sided,
  analytic, the method, the significance level, the tested-rows definition) for building a legend.
- **Legend contract.** Bins and quadrants mean the same thing for every correction method. A Gi*
  bin of level `L` means the adjusted p-value (the plain p-value for `'none'`) is at most `0.10`,
  `0.05` or `0.01` for `L = 1, 2, 3`, and the sign is the sign of z. A local Moran quadrant is the
  sign pair of the centered value and the lag, reported only where the adjusted p-value is at most
  `significanceLevel`. With `'none'`, Gi* bins use the critical values `1.645, 1.960, 2.576`,
  which is the same rule.

- `GPUHotSpotAnalysis` writes the Getis-Ord Gi* z-score per row for arbitrary weights:
  `z_i = sum_j w_ij (x_j - X) / (S sqrt((n S1_i - W_i^2) / (n - 1)))`, with `W_i = sum_j w_ij` and
  `S1_i = sum_j w_ij^2`. The focal row enters through the compile-time `selfWeight` (default 1,
  classic Gi*), never through the matrix; a row's own value alone never produces a z-score. The
  z-score is unchanged when every value is shifted by a constant. It equals esda's
  `G_Local(star=True)` analytic `Zs` for positive data (esda normalizes by the sum of values, which
  flips the sign when the mean is negative). `selfWeight: 0` is the exact Ord-Getis Gi (esda
  `G_Local(star=False)`), which standardizes with the mean and variance of the `n - 1` other
  observations: `z_i = (sum_j w_ij x_j - W_i mu_i) / (s_i sqrt(((n - 1) S1_i - W_i^2) / (n - 2)))`.
  With row-standardized weights `selfWeight: 1` weighs the focal row as much as all its neighbors,
  so prefer `0` there. Optional outputs: the two-sided p-value, the adjusted p-value, the neighbor
  count (the focal row not included) and a `sint32` confidence bin in `-3..3` (99/95/90% hot or cold).
- `GPULocalMoran` writes the z-score of local Moran's I with the analytic moments of conditional
  randomization, the null esda's permutation test samples: `E[L_i] = W_i mu`,
  `Var[L_i] = sigma^2 (N S1_i - W_i^2) / (N - 1)` over the other `N = n - 1` rows. Optional outputs
  are `localI` (`(n - 1) c_i L_i / sum c^2`, esda's `Is`; NaN when the values are constant, 0 for
  kept islands), the spatial lag `L_i = sum_j w_ij c_j`,
  the p-value, the adjusted p-value, the neighbor count and the esda quadrant code (1 HH, 2 LH, 3 LL,
  4 HL; 0 when not significant at the per-frame `significanceLevel`).
  `quadrantGating` (`'analytic'` default or `'none'`, compile-time) chooses what `quadrants` holds. With
  `'analytic'` it is nonzero only where the analytic p-value (adjusted when `multipleTesting` is set) is
  significant. With `'none'` it
  is the ungated quadrant from the signs of the centered value and the spatial lag, for every included
  row with a nonzero centered value and lag, and 0 otherwise, so another test can gate it, for example the
  `significant` mask of `GPULocalPermutationTest` (esda's `q` with `p_sim`). `zScores` and `pValues` do
  not change. `'none'` cannot be combined with `multipleTesting` (the constructor throws).
  Local Moran significance does not depend on `|c_i|`: a row near the mean can be significant.

Checked against esda 2.10 on 200 points with 6-nearest-neighbor weights: `localI` matches `Is` to
3e-7, the z-score tracks esda's 99,999-permutation `z_sim` (correlation 0.999997, largest gap 0.03),
and Gi* matches `Zs` to 1e-7. A 60-row fixture with an island and masked rows pins `G_Local` (star
and non-star), `Moran_Local`, `Geary_Local` and the statsmodels adjustments for `islands: 'keep'`,
`'exclude'` and the re-standardized masked case.

```ts
graph.add(new GPUNeighborSearch({
  mode: 'radius', gridSize: [256, 256], positions,
  parameters: searchParameters.importToGraph(graph), output: {...weights, status: {overflow}}
}));
graph.add(new GPUHotSpotAnalysis({weights, values, parameters, output: {zScores, bins, pValues}}));
graph.add(new GPULocalMoran({weights, values, parameters, output: {zScores, quadrants}}));
searchParameters.write(getGPUNeighborSearchParameterValues({bounds, radius: 500}));
```

{/* parameters: GPUHotSpotAnalysis generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUHotSpotAnalysis` parameters

- Packer: `getGPUSpatialAutocorrelationParameterValues(settings?)`, writes `GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH` (8) `float32` values.

Settings of `getGPUSpatialAutocorrelationParameterValues`:

| Field | Type | Description |
|---|---|---|
| `significanceLevel` (optional) | `number` | Two-sided significance level of `GPULocalMoran` quadrants, in `(0, 1)`, applied to the adjusted p-value when `multipleTesting` is set. Defaults to `0.05`. `GPUHotSpotAnalysis` ignores it and always reports 90/95/99% bins. |
| `fixedMoments` (optional) | `GPUSpatialAutocorrelationFixedMoments` | Optional moments that replace the moments of the current rows. |

{/* /parameters */}

{/* parameters: GPULocalMoran generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPULocalMoran` parameters

- Packer: `getGPUSpatialAutocorrelationParameterValues(settings?)`, writes `GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH` (8) `float32` values.

Settings of `getGPUSpatialAutocorrelationParameterValues`:

| Field | Type | Description |
|---|---|---|
| `significanceLevel` (optional) | `number` | Two-sided significance level of `GPULocalMoran` quadrants, in `(0, 1)`, applied to the adjusted p-value when `multipleTesting` is set. Defaults to `0.05`. `GPUHotSpotAnalysis` ignores it and always reports 90/95/99% bins. |
| `fixedMoments` (optional) | `GPUSpatialAutocorrelationFixedMoments` | Optional moments that replace the moments of the current rows. |

{/* /parameters */}

#### How it computes

{/* lens:GPUHotSpotAnalysis */}
{/* lens:GPULocalMoran */}

`GPULocalMoran`:

::lens{scenario=gpu-spatial-analysis.local-moran rung=datum bind=page datum=420 caption="Inspect tract 420: its value, neighbour lag, local I, and corrected p-value."}
::lens{scenario=gpu-spatial-analysis.local-moran rung=formula caption="Follow 2 equations: local I and its conditional-randomisation z-score."}
::lens{scenario=gpu-spatial-analysis.local-moran rung=kernel caption="Inspect 791 focal invocations and the retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.local-moran rung=graph caption="Trace queen weights, row standardisation, local moments, and FDR across 791 tracts."}

`GPUHotSpotAnalysis`:

::lens{scenario=gpu-spatial-analysis.hot-spot-analysis rung=datum bind=page datum=420 view=neighbourhood-view caption="Inspect tract 420: its queen neighbours, Gi* z-score, corrected p-value and confidence bin."}
::lens{scenario=gpu-spatial-analysis.hot-spot-analysis rung=formula caption="Follow the Gi* z-score: a weighted sum of centred values over its neighbour-count spread."}
::lens{scenario=gpu-spatial-analysis.hot-spot-analysis rung=kernel caption="Inspect 791 focal invocations, the moment passes, FDR adjustment and classify kernel."}
::lens{scenario=gpu-spatial-analysis.hot-spot-analysis rung=graph caption="Trace queen weights, moments, Gi* gather, Benjamini-Hochberg adjustment and confidence bins."}

### `GPULocalGeary`

Measure how much each tract differs from its neighbours with local Geary c, where small values mean similar surroundings.

::lens{scenario=gpu-spatial-analysis.local-geary rung=result caption="Local Geary c across 791 Chicago tracts, mean 0.81; small values mark tracts like their neighbours."}

Neighbor, row re-standardization and `islands` semantics follow `GPULocalMoran`. The output is
`output.statistic`; the parameters and the permutation adapter are described under
Additional spatial statistics below.

#### How it computes

{/* lens:GPULocalGeary */}

::lens{scenario=gpu-spatial-analysis.local-geary rung=datum bind=page datum=420 caption="Inspect tract 420: its value, its queen neighbours’ values and the resulting c."}
::lens{scenario=gpu-spatial-analysis.local-geary rung=formula caption="Follow the rule: squared differences to each neighbour, weighted, divided by the variance."}
::lens{scenario=gpu-spatial-analysis.local-geary rung=kernel caption="Inspect 791 focal invocations over the retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.local-geary rung=graph caption="Trace queen weights, row standardisation, moments and the local Geary kernel."}

### `GPUAssuncaoReisStandardization`

Standardize each tract's rate against the pooled rate and Poisson noise, turning event counts and populations into one z-score per tract.

::lens{scenario=gpu-spatial-analysis.assuncao-reis-standardization rung=result caption="Standardized poverty rates of 791 tracts range from -1.5 to 3.3; the 3 zero-population tracts stay missing."}

Empirical-Bayes standardized rates of Assuncao and Reis (1999), the transformation behind
`esda.Moran_Rate` and `esda.Moran_Local_Rate`. It writes one standardized value per row and no
spatial statistic; Moran statistics of rate data come from composing it with the existing nodes.
Lives in `spatial-autocorrelation/gpu-rate-standardization`.

```ts
const standardization = new GPUAssuncaoReisStandardization({
  events, // float32 or uint32 view (counts are converted to f32, exact below 2^24)
  populations, // float32 or uint32 view, population at risk
  mask, // optional uint32
  output: {
    standardizedRates, // float32 view, NaN for rows outside the analysis
    statistics, // optional, [n, b, s2, a, totalPopulation]
    status: {invalidCount} // optional uint32 scalar
  }
});
```

Definition (identical to `esda.smoothing.assuncao_rate`), over the `m` valid rows with rates
`p_i = e_i / n_i`: `b = sum e / sum n`, `s2 = sum n_i (p_i - b)^2 / sum n`,
`a = s2 - b / (sum n / m)`, `v_i = a + b / n_i` (replaced by `b / n_i` where `v_i < 0`), and
`z_i = (p_i - b) / sqrt(v_i)`.

- **Negative prior variance.** `a` is negative when the rates vary less than Poisson noise predicts.
  esda does not clamp `a`: it keeps the negative `a` wherever `v_i` stays non-negative and falls
  back to the pure Poisson variance `b / n_i` only for rows where `v_i < 0`. This node does the
  same, for parity, and `statistics[3]` reports the unclamped `a`. The EB smoothing in
  `rate-smoothing` clamps the prior variance at zero instead (a different, also standard
  convention for shrinkage); the two are not meant to agree when `a < 0`.
- **Missing rows.** A row is valid when it is unmasked, its events are finite and non-negative, and
  its population is finite and positive. Other rows write NaN, are excluded from `b`, `s2`, `m`
  and the mean population, and are excluded by the downstream statistic. `output.status.invalidCount`
  counts the unmasked rows that were rejected. Rows where `v_i` is not positive (for example `b = 0`)
  write NaN, as esda's 0 / 0 does.
- **Precision.** `m` is a u32; sums are fixed-order two-level workgroup trees (no float atomics,
  bitwise reproducible); `b` gets a second-pass correction and `s2` is formed from centered
  deviations. Fewer than 2^24 rows.

Composition (no statistic code is duplicated):

```ts
const standardizedRates = createTransientView(graph, 'rates', 'float32', rows);
graph.add(new GPUAssuncaoReisStandardization({events, populations, output: {standardizedRates}}));

// EB Moran's I (esda.Moran_Rate, transformation 'r'): row-standardized weights.
graph.add(
  new GPUGlobalSpatialStatistics({
    weights, // weights.information.transform === 'row'
    values: standardizedRates,
    statistics: ['moran'],
    islands: 'keep', // esda's convention
    output: {summary}
  })
);
// EB local Moran (esda.Moran_Local_Rate): GPULocalMoran, or GPULocalPermutationTest('localMoran').
graph.add(
  new GPULocalMoran({
    weights,
    values: standardizedRates,
    parameters,
    islands: 'keep',
    output: {zScores, localI}
  })
);
```

`summary[moran + statistic]`, `summary[moran + expected]` and `summary[moran + zRandomization]`
equal `Moran_Rate.I`, `.EI` and `.z_rand` (esda recomputes the Moran of the standardized values and
ignores the EB sampling variance); `localI` equals `Moran_Local_Rate.Is`. Row-standardized weights
are re-standardized over the active neighbors, so NaN rows drop out of the neighborhoods of the
rows that remain.

#### How it computes

{/* lens:GPUAssuncaoReisStandardization */}

::lens{scenario=gpu-spatial-analysis.assuncao-reis-standardization rung=datum bind=page datum=420 caption="Inspect tract 420: its events, population, crude rate and standardized z."}
::lens{scenario=gpu-spatial-analysis.assuncao-reis-standardization rung=formula caption="Follow z: the tract rate minus the pooled rate, over its Poisson-plus-prior spread."}
::lens{scenario=gpu-spatial-analysis.assuncao-reis-standardization rung=kernel caption="Inspect the three block-sum passes, the mean correction and the standardize kernel."}
::lens{scenario=gpu-spatial-analysis.assuncao-reis-standardization rung=graph caption="Trace the totals, mean correction, variance and per-tract standardization passes."}

### `GPUMultipleTestingAdjustment`

Adjust a column of per-tract p-values for multiple testing, returning corrected values that are significant when at most alpha.

::lens{scenario=gpu-spatial-analysis.multiple-testing-adjustment rung=result caption="Hot spot p-values of 754 tested tracts before and after Benjamini-Hochberg: 369 pass 0.05 unadjusted, 265 adjusted."}

Adjusts a column of per-row p-values for multiple testing on the GPU. `method` is `'none'`,
`'bonferroni'`, `'holm'`, `'benjamini-hochberg'` or `'benjamini-yekutieli'`, and
`output.adjustedPValues` holds the same numbers as statsmodels `multipletests` (`bonferroni`, `holm`, `fdr_bh`, `fdr_by`), in
`[0, 1]`. `getMultipleTestingAdjustedPValues` is the float64 CPU reference with the same semantics.

- **Family.** The family is every tested row: a row whose p-value is finite and in `[0, 1]`. NaN
  rows are not tested and get NaN. There is no mask prop by design: run the statistic and this node
  on the full analysis set and apply a viewport or display mask only when rendering, otherwise
  panning changes the family size `m` and makes the visible rows look more or less significant.
- **`minimumFamilySize`.** `m = max(testedRows, minimumFamilySize)`; rows without a p-value count as
  `p = 1`. Use it to pin `m` when the tested set can shrink (for example an analysis mask that
  follows the viewport). The optional `output.testedCount` reports the `m` actually used.
- **Which method.** Local statistics of neighboring rows share observations, so their p-values are
  dependent. Benjamini-Hochberg only controls the false discovery rate under independence or
  positive regression dependence; use `'benjamini-yekutieli'` (Benjamini-Hochberg times
  `c(m) = sum 1/i`, a cost of about `ln m` in power) for controlled error under arbitrary
  dependence. Holm controls the family-wise error rate and is uniformly more powerful than
  Bonferroni.
- **Significance.** For every method a row is significant at level `alpha` exactly when
  `adjustedPValue <= alpha`, so 0.10 / 0.05 / 0.01 bins are consistent by construction. Tied
  p-values get identical adjusted values.
- **Cost.** `none` and Bonferroni are O(n). The step procedures add a 30-bit `GPUSort` plus a
  deterministic cumulative maximum (Holm) or minimum (Benjamini) scan in workgroup memory, a few
  linear kernels in all. Fewer than 2^24 rows (ranks are exact f32 integers); results are
  independent of scheduling.

```ts
graph.add(new GPUMultipleTestingAdjustment({
  pValues, method: 'benjamini-yekutieli', output: {adjustedPValues}
}));
// significant at 5%: adjustedPValues[i] <= 0.05
```

#### How it computes

{/* lens:GPUMultipleTestingAdjustment */}

::lens{scenario=gpu-spatial-analysis.multiple-testing-adjustment rung=datum bind=page datum=420 view=fdr-staircase caption="Inspect tract 420: its p-value, rank in the sorted family and adjusted p-value."}
::lens{scenario=gpu-spatial-analysis.multiple-testing-adjustment rung=formula caption="Follow the step-up rule: a running minimum of p times m over rank."}
::lens{scenario=gpu-spatial-analysis.multiple-testing-adjustment rung=kernel caption="Inspect the count, 30-bit sort, scan and scatter kernels with their retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.multiple-testing-adjustment rung=graph caption="Trace the hot spot p-values through counting, sorting, the cumulative minimum and scatter."}

### `GPULocalPermutationTest` and `GPUGlobalPermutationTest`

`GPULocalPermutationTest`: Test each tract’s local Moran I against 99 conditional permutations, returning pseudo p-values, adjusted values and a significance mask.

::lens{scenario=gpu-spatial-analysis.local-permutation-test rung=result caption="Two-sided pseudo p-values of 791 tracts from 99 permutations; 303 are at most 0.05, none after adjustment."}

`GPUGlobalPermutationTest`: Test global Moran’s I against 99 relabelings of the tracts, returning the pseudo p-value and the simulated reference distribution.

::lens{scenario=gpu-spatial-analysis.global-permutation-test rung=result caption="Observed Moran I = 0.60 against 99 simulated values near 0; pseudo p = 0.02, the smallest possible."}

Maximum (`GPUGlobalPermutationTest`): `(maximumPermutations + 1) * ceil(rows / 2048)` is at most `GPU_GLOBAL_PERMUTATION_TEST_MAXIMUM_PAIR_BLOCK_COUNT` (2^24 - 1), so 2^20 permutations allow 15 row blocks (30720 rows). Without `stopAfterExceedances` the permutations are not chunked (one dispatch); with it they run in batches (see below).

Permutation inference (pseudo p-values) over a `GPUSpatialWeights` CSR, following PySAL esda's
`p_sim` with the corrections below. Both contributors read a uint32 parameter view written with
`getGPUPermutationParameterValues` from a `GPUPermutationSettings` object
`{seed, permutationCount, significanceLevel}`. Changing the seed, the permutation count (up to the
compile-time `maximumPermutations`), the values or the weights never rebuilds the graph. `values`
and `secondValues` are `float32` or `uint32` (converted to f32, exact below 2^24), and every result
view lives in the `output` bag (`GPULocalPermutationTestOutput`, `GPUGlobalPermutationTestOutput`).

- `GPULocalPermutationTest` does conditional randomization for `localMoran`, `localGeary`, `localG`,
  `localGStar` (a self weight of 1), `localBivariateMoran`, `localLeeL`, `localJoinCount` and
  `localBivariateJoinCount` (see **Bivariate, join-count and Lee statistics** below). For each tested row and permutation it draws an ordered sample
  of `k_i` distinct other analysis rows (a partial Fisher-Yates over their compacted positions) and
  assigns them to the row's weight slots. It writes `exceedances`, `pseudoPValues`, an optional
  `observed` statistic, an optional `adjustedPValues`, and a `significant` mask. Rows with more than
  `maximumNeighbors` (64 at most) neighbors are not tested and set `output.status.overflow`.
- `GPUGlobalPermutationTest` handles `moran`, `geary`, `getisOrdG`, `bivariateMoran` (`y`
  permuted) and `leesL` (Lee's spatial Pearson, `x` and `y` permuted together).
  `output.summary` (`GPU_GLOBAL_PERMUTATION_SUMMARY`) holds the observed statistic, `p_sim`, the
  exceedance count, the permutation count used, the simulated mean and standard deviation, `z_sim`,
  `p_z_sim`, the minimum and maximum, and the island and analysis row counts (`islandCount`,
  `analysisCount`). Optional outputs are the `referenceDistribution` and its `histogram` (GeoDa's
  permutation plot).

**Sequential stopping (`stopAfterExceedances`, opt-in).** Besag and Clifford (1991, "Sequential Monte
Carlo p-values", Biometrika 78(2)): set the compile-time integer `h` in `[1, maximumPermutations]`
and each test consumes its permutations in index order, stopping at the first `L` whose exceedance
count (the count the p-value uses for the chosen `alternative`: `g`, `l` or `min(g, l)`) equals `h`.
Then `p = min(1, m h / L)` (`m` is 2 for `'two-sided'`, else 1), `exceedances` is `h` and `L` is
reported: `GPULocalPermutationTest` writes it per row to the optional `output.permutationCounts`
(uint32; `P` for a row that never reached `h`, `GPU_LOCAL_PERMUTATION_NOT_TESTED` for an untested
row; the output requires `stopAfterExceedances`), and `GPUGlobalPermutationTest` writes it to the
`permutationCount` field of `output.summary`. A row or test that never reaches `h` keeps the
standard `min(m (M + 1), P + 1) / (P + 1)`, so with `h > P` the results equal the unstopped run bit
for bit; reaching `h` exactly at `L = P` counts as a stop (`p = m h / P`). The Philox stream of
permutation `p` is unchanged, so the first `L` permutations are those of the full run and `L` does not
depend on the dispatch shape. The `'folded'` alternative is rejected (its statistic needs the mean
of the whole run). Omitting the prop keeps today's behavior and kernels unchanged.

- Local: each row loops over permutations in its own invocation and `break`s at `L`, so a clearly
  non-extreme row costs about `h / (tail probability)` permutations instead of `P`. `significant`,
  `adjustedPValues` and the Geary `classes` use the stopped p-value.
- Global: the permutations run in fixed batches of `getGPUGlobalPermutationBatchSize(rows,
  maximumPermutations)` (about 128 row-block workgroups per batch so the GPU stays occupied, at most 64
  batches; the batch is also the most permutations computed past `L`). After each batch a one-workgroup
  node scans that batch's tail flags in permutation order, keeps the running counts in a GPU state
  word and zeroes the indirect dispatches of the remaining batches, so the stopping index does not
  depend on the batch size. `referenceDistribution`, `histogram`, `simulatedMean`,
  `simulatedStandardDeviation`, `zSimulated`, `minimum` and `maximum` use the first `L` permutations
  only (`referenceDistribution` is NaN past `L`).
- Typical use is `h` between 10 and 20 with large `P` (a LISA map with `P = 999`, `h = 10`): rows
  with `p` far above 0.05 stop after tens of permutations while the significant rows run to `P`.

**Tail (`alternative`).** The type is the shared `GPUInferenceAlternative` and the default is now `'two-sided'`: with `g = #{sim >= observed}`
and `l = #{sim <= observed}`, `p = min(2 (min(g, l) + 1), P + 1) / (P + 1)`. A row (or a global
statistic) whose simulations all tie the observed value has `g = l = P` and `p = 1`. `'directed'`
(`p = (min(g, l) + 1) / (P + 1)`) is esda's legacy default and is anti-conservative: the tail is
chosen from the data and not doubled, so under the null `p` is about twice too small. Its tie
handling deviates from esda on purpose: esda uses `min(g, P - g)`, and `P - g` counts only
simulations strictly below the observed value, so when simulations tie the observed value (discrete
data, zero-inflated counts, a constant neighborhood) esda reports the smallest attainable
`p = 1/(P+1)` (for 20 equal simulations: 0.0476 for a tied observed value), while `min(g, l)` reports
1. The two agree whenever no simulation equals the observed value (pinned against esda 2.10.0).
`'folded'` matches esda's undirected test exactly; `'greater'` and `'lesser'`
match esda's one-sided tests. Do not combine `'directed'` with `multipleTesting`.

**Neighbors, islands and re-standardization.** One rule, shared with the other Stream A statistics:
a neighbor slot is *active* when its neighbor is a valid row other than the row itself, its weight is
positive, and the neighbor is an included row (mask nonzero, finite value). Zero-weight slots are not
neighbors: they are not counted in `k_i` and are never drawn. An *island* is an included row with no
active slot. `islands: 'exclude'` (default) drops islands from the analysis: NaN outputs, and they
are not counted in `n`, the mean, the variance or `S0`, and not neighbors of other rows. There is no
cascade: a row whose only neighbors were islands stays in `n` with no neighbor, so it is not tested
(global tests add 0 to `S0` for it). `islands: 'keep'` reproduces esda: islands stay in `n` and the
moments, their local statistic is the formula value with a zero lag (`observed`, for example local
I = 0), but they are never tested (`exceedances` is `GPU_LOCAL_PERMUTATION_NOT_TESTED`, p-value NaN).
Parity specs against esda use `islands: 'keep'`. Island counts are reported in the optional
one-row `output.islandCount` (local) or `summary[GPU_GLOBAL_PERMUTATION_SUMMARY.islandCount]` (global).
When `weights.information.transform` is `'row'`, each analysis row's active weights are re-standardized to
sum to 1, in the lags and in `S0`, so a masked or missing neighbor drops out instead of shrinking
the row sum below 1 (numerically identical when nothing is dropped); other transforms are used as
given. If `weights.status.overflow` is nonzero the weights are truncated: no row is analyzed
(statistics NaN), and the local test sets `output.status.overflow`. The mean is a fixed-order tree sum followed by a
second-pass correction of the deviations, and `n` is an exact uint32, so large-offset data keep
`sum(z) = 0` to f32 accuracy.

**Permutation count.** The count is never clamped. A per-frame `permutationCount` outside
`[1, maximumPermutations]` writes NaN p-values: the local test leaves every row untested and sets
`output.status.overflow`; the global test writes NaN for the `permutationCount` field of the summary,
the p-values and the simulated summaries (the observed statistic is still reported). Validate on the
CPU with `getGPUPermutationParameterValues(settings, {maximumPermutations})` or the contributor's
`getParameterValues(settings)`, both of which throw.

**Multiple testing (local).** `multipleTesting` is `'none'` (default), `'bonferroni'`, `'holm'`,
`'benjamini-hochberg'` or `'benjamini-yekutieli'`, run through `GPUMultipleTestingAdjustment` over the
pseudo p-values of all tested rows (never a display mask). `output.adjustedPValues` receives the
adjusted values, and `output.significant` is `adjustedPValue <= level`; with `'none'` it is the exact
cross-multiplied integer comparison `p <= level`.
`getInferenceMetadata(settings)` on either contributor (the same `GPUPermutationSettings` that are
packed, validated identically)
returns a `GPUInferenceMetadata` (alternative, permutation count, seed, multiple-testing method,
significance level and a `testedRowsDefinition` sentence) for legends.

**Bivariate, join-count and Lee statistics.** Definitions below use `c = x - mean` over the analysis
rows, `SS = sum c^2`, `n` analysis rows and the weights as given (re-standardized for `'row'`
weights). Parity: statistics against esda 2.10 at f32 accuracy, p-values against the twin of the
Philox draws, plus a Monte Carlo bound against esda `p_sim` (R = 9999).

- `localBivariateMoran` (esda `Moran_Local_BV`, needs `secondValues`): `I_i = (n - 1) c^x_i sum_j
  w_ij c^y_j / sqrt(SS_x SS_y)`; the focus keeps `x`, the drawn neighbors supply `y`. The optional
  `quadrants` output is esda's `q` in the PySAL scheme (1 high-high, 2 low-high, 3 low-low, 4
  high-low in `x_i` and the lag of `y`); it is not gated by significance.
- `localLeeL` (esda `Spatial_Pearson_Local`, needs `secondValues`): `L_i = n (sum_j w_ij c^x_j)(sum_j
  w_ij c^y_j) / sqrt(SS_x SS_y)`. A drawn row supplies both its `x` and its `y` (the pair stays
  together), the focus value does not enter. For row-standardized weights the mean of the local
  values is the global `L`. esda's own p-value is the `'directed'` one (anti-conservative); the
  default here is `'two-sided'`.
- `leesL` in `GPUGlobalPermutationTest` (esda `Spatial_Pearson`, element `[0, 1]`): `L = z'W'Wz /
  1'W'W1 = n sum_i (sum_j w_ij c^x_j)(sum_j w_ij c^y_j) / (sqrt(SS_x SS_y) sum_i (sum_j w_ij)^2)`,
  with the rows of `(x, y)` permuted together by the swap-or-not shuffle (esda permutes the rows of
  `Z`). Note that esda's global version does not row-standardize `W`; here `'row'` weights are
  re-standardized over active slots, others are used as given. The smoothing factors esda reports
  at `[0, 0]` and `[1, 1]` are not exposed. (`GPUSpatialPearson` is a different statistic: the
  weighted correlation across edges.)
- `localJoinCount` (esda `Join_Counts_Local`, binary `values`): `LJC_i = x_i sum_j x_j` over the active
  neighbors with binary weights (the weight values are ignored), where a value is a join only when
  it is exactly 1. Only rows with `LJC_i > 0` are tested; the others get NaN p-values and
  `GPU_LOCAL_PERMUTATION_NOT_TESTED`, as esda sets their `p_sim` to NaN. The tail that detects
  clustering is `'greater'`. With ties (discrete statistics) esda's `'two-sided'` p-value
  (percentile based) is not `2 min(g, l)`; the parity specs therefore use `'greater'`.
- `localBivariateJoinCount` (esda `Join_Counts_Local_BV`, `joinCase`): `'CLC'` (co-location) counts
  neighbors with `x_j = y_j = 1` for a focus with `x_i = y_i = 1`; `'BJC'` (no co-location) counts
  neighbors with `x_j = 0, y_j = 1` for a focus with `x_i = 1, y_i = 0`. A drawn row supplies its `x`
  and `y` together. **Deviation from esda for `'BJC'`:** esda observes the count above but
  simulates `x_i sum_j y_j`, which ignores `y_i = 0` and `x_j = 0`, so its p-values do not test the
  statistic it reports (on the specs' fixture they differ by up to 0.09 at R = 9999). This
  implementation simulates the observed statistic; `'CLC'` is identical to esda.
- `localGeary` classification (`classes` output, `GPU_LOCAL_GEARY_CLASS`; Anselin 2019, esda
  `Geary_Local(labels=True)`): with `m` the mean of the local Geary values over the analysis rows, a
  significant row (`p <= level`, or `adjustedPValue <= level` with `multipleTesting`) is 1
  (high-high, positive association: `c_i < m` and `x_i` above the mean), 2 (low-low: `c_i < m`, `x_i`
  below the mean) or 3 (negative association: `c_i > m`); an insignificant row is 4 and a row that
  is not tested (NaN p-value), or exactly at a mean, is 0 (esda: NaN). esda's own docstring calls
  codes 1 and 2 "outlier" and "cluster"; the numeric rule is ported, not the words. Small `c_i` means
  similar neighbors, so the tail of clustering is `'lesser'` (also the tail declared by
  `GPU_LOCAL_GEARY_PERMUTATION_ADAPTER`): with `'lesser'` class 3 is empty because negative
  association never has a small lower-tail p-value, with `'greater'` classes 1 and 2 are empty, and
  with the default `'two-sided'` both appear. The classification is not available analytically.

**Global shuffle.** Each permutation relabels the analysis values through a keyed swap-or-not
shuffle (Hoang, Morris and Rogaway 2012, `6 ceil(log2 n) + 6` rounds, round keys and salts from
Philox 4x32-10 on the seed and the permutation index), so no sort runs per permutation and nothing
of size `P x n` is stored. The earlier four-round Feistel network is no longer used by the global
test because it is not uniform on small domains: with 400000 draws the marginal of `pi(0)` and the
joint placement of fixed pairs reject uniformity far below the 0.1 percent level for domains of
10 to 50 elements (joint chi-square p-values below 1e-100; marginal p-values from 4e-4 at `n = 50`
down to 1e-139 at `n = 10`), while swap-or-not passes (all 150 chi-square p-values above 0.0023,
none below 1e-3). At 20000 draws on rook lattices and random kNN graphs with `n` = 30 to 5000
both shuffles reproduce the analytic randomization mean and variance of Moran's I (`E = -1/(n-1)`,
esda `VI_rand`) to within about 2.7 standard errors, so the Feistel bias is a small-domain,
joint-placement effect that the variance check alone does not see; it is removed rather than
tolerated. The Knox and Mantel tests and the GWR non-stationarity test use the same swap-or-not
shuffle.

```ts
const parameters = new GPUParameterBuffer(device, {
  id: 'permutations', format: 'uint32', length: GPU_PERMUTATION_PARAMETER_LENGTH,
  values: getGPUPermutationParameterValues(
    {seed: 1, permutationCount: 99},
    {maximumPermutations: 999}
  )
});
const test = new GPULocalPermutationTest({
  weights, values, statistic: 'localMoran', parameters: parameters.importToGraph(graph),
  maximumPermutations: 999, multipleTesting: 'benjamini-yekutieli', islands: 'exclude',
  output: {
    exceedances, pseudoPValues, significant, adjustedPValues, islandCount, status: {overflow}
  }
});
graph.add(test);
// Preview with 99 permutations, then refine on idle:
const settings = {seed: 1, permutationCount: 999};
parameters.write(test.getParameterValues(settings));
const metadata = test.getInferenceMetadata(settings);
```

{/* parameters: GPUGlobalPermutationTest generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUGlobalPermutationTest` parameters

- Packer: `getGPUPermutationParameterValues(settings, options?)`, writes `GPU_PERMUTATION_PARAMETER_LENGTH` (4) `uint32` values.
  - `options` (`{maximumPermutations?: number}`)
- Packer: `instance.getParameterValues(settings)`, writes the `uint32` parameter view.

Settings of `getGPUPermutationParameterValues`:

| Field | Type | Description |
|---|---|---|
| `seed` | `number` | Random seed, an integer in `[0, 2^53)`. Results are a pure function of the seed, the inputs and `permutationCount`, independent of the dispatch shape. Per frame. |
| `permutationCount` | `number` | Number of permutations `P`, in `[1, maximumPermutations]` of the contributor. The count is never clamped: a value outside that range makes the kernels write NaN p-values (local: rows are not tested and `output.status.overflow` is set; global: the `permutationCount` field of `output.summary` is NaN). Validate on the CPU with the contributor's `getParameterValues` or the `maximumPermutations` option of `getGPUPermutationParameterValues`. |
| `significanceLevel` (optional) | `number` | Significance level for the local `significant` mask. Defaults to 0.05. Ignored by the global test. |

{/* /parameters */}

{/* parameters: GPULocalPermutationTest generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPULocalPermutationTest` parameters

- Packer: `getGPUPermutationParameterValues(settings, options?)`, writes `GPU_PERMUTATION_PARAMETER_LENGTH` (4) `uint32` values.
  - `options` (`{maximumPermutations?: number}`)
- Packer: `instance.getParameterValues(settings)`, writes the `uint32` parameter view.

Settings of `getGPUPermutationParameterValues`:

| Field | Type | Description |
|---|---|---|
| `seed` | `number` | Random seed, an integer in `[0, 2^53)`. Results are a pure function of the seed, the inputs and `permutationCount`, independent of the dispatch shape. Per frame. |
| `permutationCount` | `number` | Number of permutations `P`, in `[1, maximumPermutations]` of the contributor. The count is never clamped: a value outside that range makes the kernels write NaN p-values (local: rows are not tested and `output.status.overflow` is set; global: the `permutationCount` field of `output.summary` is NaN). Validate on the CPU with the contributor's `getParameterValues` or the `maximumPermutations` option of `getGPUPermutationParameterValues`. |
| `significanceLevel` (optional) | `number` | Significance level for the local `significant` mask. Defaults to 0.05. Ignored by the global test. |

{/* /parameters */}

#### How it computes

{/* lens:GPULocalPermutationTest */}
{/* lens:GPUGlobalPermutationTest */}

`GPULocalPermutationTest`:

::lens{scenario=gpu-spatial-analysis.local-permutation-test rung=datum bind=page datum=420 view=fdr-staircase caption="Inspect tract 420: its observed I, exceedance count, pseudo p-value and 99-draw null distribution."}
::lens{scenario=gpu-spatial-analysis.local-permutation-test rung=formula caption="Follow the pseudo p-value: doubled smaller tail count plus one, over permutations plus one."}
::lens{scenario=gpu-spatial-analysis.local-permutation-test rung=kernel caption="Inspect the per-tract Philox draw loop and tail counting with retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.local-permutation-test rung=graph caption="Trace weights, moments, 99 permutation draws, classification and Benjamini-Hochberg adjustment."}

`GPUGlobalPermutationTest`:

::lens{scenario=gpu-spatial-analysis.global-permutation-test rung=datum bind=page datum=420 caption="Inspect tract 420: its poverty value, one of 791 relabeled across 99 permutations."}
::lens{scenario=gpu-spatial-analysis.global-permutation-test rung=formula caption="Follow the pseudo p-value and the simulated z-score from the 99 pair sums."}
::lens{scenario=gpu-spatial-analysis.global-permutation-test rung=kernel caption="Inspect the pair-sum, shuffle and finalize kernels with their retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.global-permutation-test rung=graph caption="Trace weights, 100 pair sums, reduction over row blocks and the finalize summary."}

### `GPUSpatialPearson` and `GPUGammaStatistic`

`GPUSpatialPearson`: Correlate a tract variable with a second variable at its neighbours, returning one edge-weighted Pearson coefficient.

::lens{scenario=gpu-spatial-analysis.spatial-pearson rung=result caption="Edge-weighted Pearson r = 0.30 between poverty and neighbouring uninsured rates over 791 tracts."}

`GPUGammaStatistic`: Reduce neighbour pairs to one Gamma value: the weighted mean cross-product, squared difference or absolute difference.

::lens{scenario=gpu-spatial-analysis.gamma-statistic rung=result caption="Weighted mean squared poverty difference across the directed queen pairs of 791 tracts is 214 points squared."}

Both statistics sum over directed edges with positive weight to another included row, in three
fixed-order stages. Neither reports an analytic p-value; their permutation adapters and the other
details are described under Additional spatial statistics below.

#### How it computes

{/* lens:GPUSpatialPearson */}
{/* lens:GPUGammaStatistic */}

`GPUSpatialPearson`:

::lens{scenario=gpu-spatial-analysis.spatial-pearson rung=datum bind=page datum=420 caption="Inspect tract 420: its poverty value is one of 791 foci whose neighbours supply y."}
::lens{scenario=gpu-spatial-analysis.spatial-pearson rung=formula caption="Follow the edge-weighted correlation: covariance over the product of weighted standard deviations."}
::lens{scenario=gpu-spatial-analysis.spatial-pearson rung=kernel caption="Inspect the moments, centered and statistic kernels with their retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.spatial-pearson rung=graph caption="Trace queen weights, row standardisation and the three-stage edge reduction."}

`GPUGammaStatistic`:

::lens{scenario=gpu-spatial-analysis.gamma-statistic rung=datum bind=page datum=420 caption="Inspect tract 420: its poverty value enters every directed edge to its neighbours."}
::lens{scenario=gpu-spatial-analysis.gamma-statistic rung=formula caption="Follow the Gamma ratio: weighted pair terms over total edge weight."}
::lens{scenario=gpu-spatial-analysis.gamma-statistic rung=kernel caption="Inspect the moments, centered and statistic kernels with their retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.gamma-statistic rung=graph caption="Trace queen weights, row standardisation and the three-stage edge reduction."}

### Additional spatial statistics and bounded optimization

`GPULocalGeary` writes source-aligned `sum_j w_ij (z_i - z_j)^2` values to `output.statistic`
(`GPULocalGearyOutput`, with the optional `globalStatistics`), with the neighbor, re-standardization and `islands` semantics of
`GPULocalMoran` (an island is NaN under `'exclude'`, the default, and 0 under `'keep'`, esda's
convention; a constant variable gives NaN). It has no analytic classification: use permutation
inference (`GPULocalPermutationTest` with `statistic: 'localGeary'` and `classes` gives the Anselin
2019 positive / negative association classes, see above). `GPU_LOCAL_GEARY_PERMUTATION_ADAPTER` declares the `'lesser'` tail, because small `c_i`
means similar neighbors (a cluster). `GPUSpatialPearson` is the weighted Pearson correlation across
directed spatial edges `(x_i, y_j)`. `GPUGammaStatistic` reports the weighted mean centered
cross-product, squared difference, or absolute difference over those edges, into the one-row
`output.statistic` (`GPUEdgeAssociationOutput`; `float32` or `uint32` values). Both skip the focal
row's own slot and zero-weight slots, and reduce the edges in three fixed-order stages (per-workgroup
moments, per-workgroup centered sums around the combined means, one combining workgroup), so they use
the whole device and stay deterministic. The local Geary, Pearson, and the two Gamma forms with an
order-equivalent pair sum expose `GPUPermutationStatisticAdapter` descriptors for the existing
permutation contributors; they do not duplicate permutation loops.

`GPURoseStatistic` consumes planar line starts and ends. Its required
`GPUCircularStatisticsContract` makes directional versus axial interpretation, radians, angular
origin, and equal-width bin count explicit. It reports count, mean angle, resultant length,
circular variance, Rayleigh z, and the weighted Rose bins for every group, in
`output: {summary, bins}`. The descriptive mean, resultant length, variance and bins use the
positive finite `weights`; invalid lines or weights are excluded. Rayleigh z uses **each valid
segment once**, with unweighted resultant $\bar R_0$ and valid segment count $n$:

$$z = n\bar R_0^2 = \frac\{(\sum_i\cos\phi_i)^2 + (\sum_i\sin\phi_i)^2\}\{n\}.$$

Here $\phi_i=\theta_i$ for directional data and $2\theta_i$ for axial data. This corrects the
previous use of total descriptive weight as sample size: metres of street are not independent
observations, and changing their units must not change z. Weighted descriptive `resultantLength`
is distinct from $\bar R_0$. No p-value is returned. Under independent uniform directions the
usual [Rayleigh approximation](https://github.com/circstat/circstat-matlab/blob/master/circ_rtest.m)
is $p\approx e^\{-z\}[1+(2z-z^2)/(4n)-(24z-132z^2+76z^3-9z^4)/(288n^2)]$;
spatially dependent street segments do not establish that null assumption. This planar rose is
separate from [giddy Rose](https://pysal.org/giddy/generated/giddy.directional.Rose.html), which
forms directional LISA vectors and uses permutations of sector counts for inference.

`GPUAZPRegions` is a greedy AZP with batched moves: a bounded, deterministic local search from
caller-supplied contiguous labels with no tabu list, annealing or reactive escape, so it stops at the
first local optimum and the result depends on `initialLabels` (for several starts, run several
instances from different initial partitions and keep the smallest `objective`). Each round every row
finds its best improving move (the largest reduction of the within-region sum of squares from leaving
its region for an adjacent one) and the round applies an independent set of them at once: every region
takes part in at most one selected move, chosen by gain and then by a stable seed-keyed row order
(two atomic-max claim passes over the regions), so the selected gains add up exactly and the objective
falls strictly every round. The best move of a round is always selected, so a round without a move is
the local optimum. Contiguity is kept by a **local articulation test** instead of a search of the whole
source region: a row may leave region A only when its A-neighbors stay connected to each other through
rows of A within two steps of it (at most `GPU_AZP_LOCAL_NEIGHBORHOOD_CAPACITY` rows, scanning neighbor
lists of at most `GPU_AZP_MAXIMUM_SCAN_DEGREE` entries; a larger neighborhood or a hub refuses the
move). The test is sufficient, never necessary, so a region never becomes disconnected, but a safe
move whose reconnecting path is longer is refused: the result is a local optimum with respect to the
moves that pass the test, not necessarily to every contiguity-preserving move (on random lattice
partitions the test refuses about 1% of the safe moves). A row only moves to the region of one of its
neighbors, which keeps the destination connected. The initial labels are checked on the GPU as well: a
region that is not connected adds its extra components to `invalidCount` and the search does not run.
The operator publishes `output: {labels, objective, status}`: the final within-region sum of squares
(the same quantity as `withinSumOfSquares` of `GPURegionPartitionEvaluation` on the same table,
recomputed from the returned labels) plus packed iteration, convergence, iteration-limit, and
invalid-count status. `iterationCount` counts the rounds that applied at least one move, not single
moves; `maximumIterations` is a hard compile-time bound on rounds (a round applies many moves, so far
fewer are needed than the single moves of earlier versions). The seed is per frame: the one-word
`parameters` view written with `getGPUAZPRegionsParameterValues({seed})` fixes tie order, so a new seed
is a buffer write. Optional `extensiveValues` and `extensiveThreshold` forbid a move that would leave
the source region below the threshold (the max-p constraint). A multi-start AZP is deferred: one
instance already costs `maximumIterations` rounds of row-parallel work, so parallel starts would
multiply the dispatch count rather than fill idle lanes. There is no row cap: private memory is
`O(GPU_AZP_LOCAL_NEIGHBORHOOD_CAPACITY)` and the work per row and round is bounded by the row's
neighbor lists, so only the `rows x columnCount` attribute table and the region record table must each
fit one 128 MiB binding (`getGPUAZPMaximumTableWordCount(device)` words, 33,554,432 on a default
device). After the local optimum the one-thread round-finish kernel zeroes a GPU-written indirect
dispatch, so the remaining rounds launch only that kernel.

`GPUAZPRegions`, `GPUMaxPRegions` and `GPUWardRegions` share two input rules. **Attributes are
standardized by default** (`standardize`, default true): each column becomes a z-score over its finite
rows with the population standard deviation (divisor `n`, the `sklearn` `StandardScaler` convention),
a constant column becomes zero, and `objective` is in z-score units. Pass `standardize: false` for the
raw-unit behavior of earlier versions (a default change). **Adjacency is undirected**: row `i` is
adjacent to `j` when either row lists the other, so directed (for example k-nearest-neighbor) weights
are symmetrized inside the operator with `GPUSpatialWeightsTranspose`; weights whose
`information.symmetric` is `true` skip that pass. All three take a per-frame `uint32` `parameters`
view holding `[seed]` (`getGPUAZPRegionsParameterValues`, `getGPUWardRegionsParameterValues`,
`getGPUMaxPRegionsParameterValues`) and write `output: {labels, objective, status}`.

`GPUWardRegions` begins with one region per row and repeatedly merges the adjacent pair with the
smallest Ward increase in within-region squared error. `targetRegionCount` and `maximumIterations`
bound the search; disconnected weights can stop it before the target, which is reported through
`output.status` (`GPU_WARD_STATUS`).

`GPUWardRegions` has a compile-time `mergeSchedule`. The default `'sequential'` applies one merge per
unrolled iteration (always the globally cheapest adjacent pair), so reaching `k` regions from `n` rows
needs `n - k` iterations. `'mutual-nearest'` makes every iteration a round: each region finds its cheapest
adjacent region (Ward increase, ties by the same seed-keyed pair order), and every pair that chose each
other merges at once, which roughly halves the region count per round, so about `log2(n)` rounds suffice
and `maximumIterations` is an upper bound on rounds. When a round would pass `targetRegionCount`, a
one-workgroup radix select keeps only the cheapest pairs in the total order. `status[iterationCount]`
still counts merges, `objective` is still the sum of the merges' Ward increases (the within-region sum of
squares of `labels`), and a `status` view of five words receives the number of rounds in word
`GPU_WARD_ROUND_COUNT_STATUS_WORD`. It is opt-in because the result can differ from `'sequential'`. Without
contiguity constraints (complete adjacency) Ward's linkage is reducible, so both schedules build the same
dendrogram and agree whenever the target is a natural cut (every merge below it is cheaper than every
merge above it, as for well separated clusters), which the `scipy.cluster.hierarchy.ward` fixture checks.
At other targets a round can merge a costly reciprocal pair before a cheaper pair that exists only after
other merges of the same round, so the partitions differ; with contiguity constraints Ward can also have
inversions (a merged region can be cheaper to join than its parts), which makes them differ further.
Either result is a valid contiguous partition with an exact objective.

`GPUMaxPRegions` is a parallel multi-seed Max-P: `startCount` (compile time, default 16, at most 64)
independent constructions share every dispatch. Each start plants many region seeds at once and grows all
regions together in gated rounds, so there is no row cap. A *seed wave* plants a seed at every
unassigned row that is a local minimum of its Philox 4x32-10 key of `(row, start; seed)` (seed read from
`parameters`) among the unassigned rows next to it, is not next to a growing region and is below the
wave's key cut; the first wave cuts to about `0.4 / s` of the rows, with `s = ceil(threshold /
mean(extensiveValues))` the expected region size, later waves cut 1.3 times as many, and a wave starts
when no region is growing or `ceil(s / 2)` rounds after the previous one. Each growing region then claims
one frontier row per round: rows offer themselves with a key made of their number of unassigned
neighbors (fewest first, which keeps regions compact) and their squared distance to the region mean, a
region keeps its smallest key and a row offered by several regions joins the smallest. Region sums are
updated by their single owner, so no float atomics are involved. A region that reaches the threshold
stops, and a region with no frontier left under the threshold is marked failed. The rows of failed
regions and rows that no region reached then join the most similar adjacent completed region in
label-propagation rounds; a failed region with no completed neighbor stays and is reported in the
status. The returned start has the fewest under-threshold regions, then the most regions, then the
smallest within-region sum of squares of the grown regions (before enclave rows join), then the lowest
start index; the status reports the region count, any infeasible remainder, the `selectedStart` and the
`improvementMoveCount`, and `objective` is the exact within-region sum of squares of the returned
labels. With `maximumIterations > 0` (default 0) the selected construction is refined by greedy AZP
moves that keep every region at or above the threshold, which lowers `objective` at the same region
count (the AZP phase reads the same `parameters` view); spopt's `MaxPHeuristic` anneals instead.

The rounds are unrolled to `maximumGrowthSteps` (compile time, default 256, at most 4096) and gated on
the GPU, so rounds after the construction has settled launch nothing and every dispatch is
`O(startCount * (rowCount + neighborCount))`. A region claims one row per round, so a region needing more
rows than `maximumGrowthSteps` cannot complete and is reported under the threshold; seed waves are spaced
half an expected region apart, so raise `maximumGrowthSteps` (nine command nodes per round) when the
threshold needs regions of more than about `maximumGrowthSteps / 4` rows. Scratch memory is
`startCount * rows * (columnCount + 3) * 4` bytes for the region table and must fit one storage binding
(`startCount * rows * 16` bytes for the claim records), which is the only size limit. Against `spopt`'s
`MaxPHeuristic` (test/oracles/statistics-max-p.py, spopt 0.8.0, lattices and Delaunay graphs of 144 to
900 rows) the construction reaches 0.975 to 1.04 of spopt's best region count with every region at the
threshold and contiguous; its objective before the optional AZP improvement is up to 1.55 times spopt's
annealed objective.

`GPULocationAllocation` consumes a row-major demand-by-facility cost matrix and implements bounded
greedy p-median, maximum-coverage, and set-covering selection. Pair it with
`GPUNetworkCostMatrix` for network costs, or provide any dense cost matrix (`costs`) or a sparse one
(`sparseCosts`, below). Outputs include selected facility flags, nearest assignments, the objective, uncovered demand count, completion, and invalid
input count. The contributor keeps selection on the GPU, but it does not claim the integer-programming
optimum or application-specific facility constraints of a packaged solver.

A `+Infinity` cost marks an unreachable demand-facility pair. Unreachable demand is never summed
into the p-median objective: the objective is the weighted cost of the reachable rows, the
unreachable rows are counted in `uncoveredDemandCount` (not `invalidCount`, which marks invalid input
and stops the run), their assignment is `0xffffffff` and their `assignedCosts` entry is `+Infinity`.
Greedy and swap steps compare p-median candidates lexicographically, first by the demand weight left
unreachable and then by the reachable cost, so one unreachable row cannot swamp the float32 cost
differences of the others.

After the greedy construction, p-median and maximum coverage run a Teitz-Bart vertex-substitution
(swap) phase, bounded by `maximumSwapIterations` (compile-time, integer 0 to 1000, default 100; 0 is
pure greedy; set covering has no swap phase). Each iteration scores every (selected facility, unselected
candidate) swap in parallel, applies the single best strictly improving swap (ties go to the lowest
selected facility, then the lowest candidate) and stops when none improves. A swap must gain more than
1e-6 of the objective scale, which keeps f32 summation noise from cycling. It needs
`maximumFacilityCount * facilityCount <= 2^24` (scores are kept per pair). The `status` output now has seven words: after the four earlier ones,
`swapIterationCount` (swaps applied), `converged` (1 when no improving swap remains, so the result is a
swap-local optimum) and `iterationLimitReached` (1 when the budget was used up, so local optimality is
not verified); all three are 0 when the phase is off. Behavior change: swaps are on by default, so
p-median and coverage results can differ from (and are never worse than) the old greedy result, and
`output.status` must hold seven words. Greedy maximum coverage alone has the `(1 - 1/e)` guarantee
relative to the optimum (Nemhauser, Wolsey and Fisher), which the swap phase preserves because it never
lowers the covered weight; greedy p-median has no constant-factor guarantee. Either way the result is a
local optimum, not the integer-programming optimum. It is checked against a brute-force optimum on small
instances, and p-median against a randomized weighted fixture with `+Infinity` entries whose exact optimum
comes from the integer program solved by `scipy.optimize.milp` (generator and package versions in
`test/oracles/p-median-oracle.py`).

Swap scoring uses the Whitaker fast-interchange gain decomposition in the form of Resende and Werneck
("A fast swap-based local search procedure for location problems", Annals of Operations Research,
2007). With `d1` and `d2` the cost of a demand row to its nearest and second-nearest selected facility,
`w` its weight and `c(i, j)` the cost of candidate `i` to row `j`, swapping out `r` for `i` changes the
p-median cost by `gain(i) - loss(r) + extra(i, r)`, where `gain(i)` sums `w * max(0, d1 - c(i, j))` over
all rows, `loss(r)` sums `w * (d2 - d1)` over the rows served by `r`, and `extra(i, r)` sums
`w * (d2 - max(c(i, j), d1))` over the rows served by `r` with `c(i, j) < d2`. Unreachable rows keep the
lexicographic rule: a separate reachable-weight term (which `d2 = +Infinity` and a newly reached row
feed) takes precedence over the cost term. Maximum coverage has the same shape with `gain(i)` the weight
of the uncovered rows `i` covers, `loss(r)` the weight covered only by `r`, and `extra(i, r)` the part of
it that `i` also covers. One thread scores all `maximumFacilityCount` swaps of one candidate in a single
ascending pass over that candidate's costs, writing only its own pair entries (no atomics), and a
fixed workgroup tree sums the objective scale (sparse input: `loss(r)` too), so an iteration costs `O(facilityCount *
demandCount)` work (dense) instead of `O(maximumFacilityCount * facilityCount * demandCount)` and the
sums do not depend on scheduling. The earlier per-pair kernels summed per-row differences; the
decomposition sums the same terms grouped differently, so the applied swap is the same except where
float32 summation order changes a near-tie, which the 1e-6 threshold guards (integer-valued data, as in
the specs, is exact, and the specs require identical swaps). The dense kernel visits every row of a
candidate, so each row served by `r` adds its own net term `w * (min(max(c, d1), d2) - d1)` (and the
reachable weight it loses if neither `d2` nor `c` reaches it) instead of a separate `loss(r)` and
`extra(i, r)`; no large `d2` terms cancel, which a spec with `d2` about `1e7 * d1` checks in float32.
The sparse kernel cannot visit unlisted pairs, so it sums `loss(r)` per selected facility (one
workgroup each) and adds `extra(i, r)`; there a second-nearest cost many orders of magnitude above the
nearest can add rounding noise of that size to the pair score.

`sparseCosts` replaces the dense matrix with CSR rows: `offsets` (`demandCount + 1` non-decreasing
`uint32`, from 0 to the entry count), `facilities` (strictly ascending and below `facilityCount` within a
row) and `costs` (`float32`), with at least one entry; a pair that is not listed is unreachable
(`+Infinity`), and a coverage distance that is `+Infinity` does not cover unlisted pairs. Exactly one of
`costs` and `sparseCosts` is given. Every operation (greedy p-median, maximum coverage, set covering and
the swap phase) supports it and gives the result of the equivalent dense matrix. Violations of the CSR
structure, NaN or negative costs count in `invalidCount`. The GPU builds the facility-major transpose
once (a stable radix sort of the entries by facility and a binary search of the segment offsets), which
the facility-side loops read, so greedy scores and swap candidates scan only the listed pairs and the
work scales with the entry count; hub rows are validated with the shared load-balanced loop, and the
demand loops scan the shorter of the row and the selected-facility list. The dense `demand rows x
facilityCount` limit no longer applies: the entry count is at most
`getGPULocationAllocationMaximumSparseEntryCount(device)` (16,777,216 on a default device, the 8-byte
transpose entry in one 128 MiB binding), the demand row count `offsets.length - 1` at most
`getGPULocationAllocationMaximumDemandCount(device)`, and the facility count and swap pair limits are
unchanged.

Once no swap improves, the one-thread apply kernel zeroes a GPU-written indirect dispatch, so a
converged run launches only that kernel per remaining iteration instead of six; maximum coverage and
set covering do the same for the greedy steps after one that chose nothing.

`alternative` (compile-time; a change recompiles) selects the tail, with esda's definitions. With `R`
permutations, `g` is the number of simulated values `>= observed` and `l` the number `<= observed`:

| `alternative` | pseudo p-value |
| --- | --- |
| `'directed'` (default) | `(min(g, R - g) + 1) / (R + 1)`, the previous behavior |
| `'greater'` | `(g + 1) / (R + 1)` |
| `'lesser'` | `(l + 1) / (R + 1)` |
| `'two-sided'` | `min(2 (min(g, l) + 1) / (R + 1), 1)` |
| `'folded'` | `(f + 1) / (R + 1)`, `f` = number of simulated values with `abs(sim - mean) >= abs(observed - mean)` |

`exceedances` holds the numerator count of the chosen tail, and `significant` and the Benjamini-Hochberg
path use the same p-value. `'two-sided'` doubles the p-value against `'directed'` and cannot go below
`2 / (R + 1)`, so raise `permutations` when testing a small level.

`'folded'` matches esda 2.10 exactly: `mean` is the mean of the `R` simulated statistics (the observed
value is excluded, as esda computes it from the reference distribution), ties count, and the test is
symmetric about that mean. The kernels run two sweeps over the same deterministic stream (a sum, then
the count); the other tails keep one. `'two-sided'` is not esda's definition: esda 2.10 places the
observed value at a percentile of the simulated statistics and counts simulations beyond the
interpolated percentiles, which would need every simulated value of a row stored and sorted. The two
can differ by a few counts (a 20-permutation fixture gives 3/21 in esda and 2/21 here). Use `'folded'`
when an undirected test must agree with esda.

On skewed data `'folded'` can flag far fewer rows than `'two-sided'`. With local Moran's I over a
right-skewed variable (hot spots on a low background), a low-value row's conditional distribution is
left-skewed, so its observed value sits in the short tail and the long tail holds simulations just as
far from the mean. Low-low rows that `'greater'` flags then mostly fail `'folded'`, while high-high rows
pass. esda 2.10 behaves the same: on a 1,500-point hot-spot field it flags 211 rows with `'folded'` and
551 with `'two-sided'`.

Random numbers come from Philox 4x32-10 counter streams, keyed by the 64-bit seed with the counter
`(block, row, permutation, tag)`. Results are therefore a pure function of the seed and inputs,
independent of the dispatch shape. The observed and simulated statistics run through the same f32
code, positive constant factors are dropped from the comparisons, and exceedances are integers. The
significance and BH tests compare correctly rounded f32 products, never quotients. Global pair sums
use fixed-shape workgroup trees. Every output is bitwise reproducible for a seed.

Maximum: all limits derive from the device's `maxStorageBufferBindingSize`; the values below are for a default device (128 MiB). `GPUAZPRegions` and `GPUWardRegions` take at most `getGPURegionalizationMaximumSlotCount(device)` (`2^24`) `weights.neighbors` slots (the symmetrized adjacency holds 8 bytes per slot in one binding) and `GPUMaxPRegions` at most `getGPUMaxPMaximumSlotCount(device)` (`2^23`, its AZP refinement symmetrizes again); `GPUAZPRegions` tables are at most `getGPUAZPMaximumTableWordCount(device)` (`2^25`) words and Ward's `rows x (columnCount + 1)` region table is at most `getGPUWardMaximumRegionTableWordCount(device)` (`2^25`). `GPULocationAllocation` takes at most `getGPULocationAllocationMaximumFacilityCount(device)` (`2^24`) facilities, `getGPULocationAllocationMaximumDemandCount(device)` (`11184810`) demand rows and `getGPULocationAllocationMaximumCostCount(device)` (`2^25`) cost entries.

{/* parameters: GPUAZPRegions generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUAZPRegions` parameters

- Packer: `getGPUAZPRegionsParameterValues(settings?)`, writes `GPU_AZP_REGIONS_PARAMETER_LENGTH` (1) `uint32` values.

Settings of `getGPUAZPRegionsParameterValues`:

| Field | Type | Description |
|---|---|---|
| `seed` (optional) | `number` | Reproducible tie-order seed, an integer in `[0, 2^32 - 1]`. Defaults to 0. Record the seed used to create `initialLabels` next to it. |

{/* /parameters */}

{/* parameters: GPUMaxPRegions generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUMaxPRegions` parameters

- Packer: `getGPUMaxPRegionsParameterValues(settings?)`, writes `GPU_MAX_P_REGIONS_PARAMETER_LENGTH` (1) `uint32` values.

Settings of `getGPUMaxPRegionsParameterValues`:

| Field | Type | Description |
|---|---|---|
| `seed` (optional) | `number` | Reproducible seed of the random seed-row order of every start (and of the improvement phase's tie order), an integer in `[0, 2^32 - 1]`. Defaults to 0. |

{/* /parameters */}

{/* parameters: GPUWardRegions generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUWardRegions` parameters

- Packer: `getGPUWardRegionsParameterValues(settings?)`, writes `GPU_WARD_REGIONS_PARAMETER_LENGTH` (1) `uint32` values.

Settings of `getGPUWardRegionsParameterValues`:

| Field | Type | Description |
|---|---|---|
| `seed` (optional) | `number` | Reproducible tie-order seed, an integer in `[0, 2^32 - 1]`. Defaults to 0. |

{/* /parameters */}

`GPUAZPRegions`: Improve contiguous starting regions by moving border tracts, returning region labels, the within-region sum of squares and status words.

::lens{scenario=gpu-spatial-analysis.azp-regions rung=result caption="Eight community areas regrouped by poverty; a 64-round budget allows longer local searches."}

The former 16-round budget stopped while improving moves remained; 32 rounds were also insufficient.
The example now allows 64 rounds, including a no-move round to establish convergence. A round
can move only one tract per participating region; the budget is not a count of full boundary sweeps.
Check `status[GPU_AZP_STATUS.converged]` before interpreting the result as a local optimum under the bounded
connectivity test.

`GPUMaxPRegions`: Grow as many contiguous regions as possible that each reach a minimum extensive total, returning labels, the within-region sum of squares and status words.

::lens{scenario=gpu-spatial-analysis.max-p-regions rung=result caption="11 regions of at least 40,000 residents grown over 136 tracts; compare poverty with region labels."}

`GPUWardRegions`: Merge adjacent tracts by smallest Ward cost into a target region count, returning labels and the within-region sum of squares.

::lens{scenario=gpu-spatial-analysis.ward-regions rung=result caption="Ward merges 136 tracts into five contiguous regions in 131 merges; the largest holds 91 tracts."}

`GPULocationAllocation`: Choose p facilities from candidate tracts to minimize population-weighted cost, returning selected flags, nearest-facility assignments and the objective.

::lens{scenario=gpu-spatial-analysis.location-allocation rung=result caption="Five facilities chosen from 136 tracts, each tract coloured by its nearest; weighted mean distance 1.24 km."}

`GPURoseStatistic`: Summarize planar line directions as a weighted rose, returning count, mean angle, resultant length, circular variance, Rayleigh z and sector bins.

::lens{scenario=gpu-spatial-analysis.rose-statistic rung=result caption="Axial rose of 894 street segments in 18 sectors; east carries 42.5 km, north 41.3 km."}

The regionalization and location-allocation cards use community areas 1 to 8 (136 contiguous north-side
tracts) of the Chicago tract data, so the unrolled algorithms stay small. Their label values are
nominal identifiers, not scores. The location-allocation cost matrix is straight-line kilometres between
tract vertex-mean centres, prepared on the CPU as the input; the figure does not draw it.

#### How it computes

{/* lens:GPUAZPRegions */}
{/* lens:GPULocationAllocation */}
{/* lens:GPUMaxPRegions */}

`GPUAZPRegions`:

::lens{scenario=gpu-spatial-analysis.azp-regions rung=datum bind=page datum=40 caption="Inspect tract 40: its poverty value, starting community area and AZP region."}
::lens{scenario=gpu-spatial-analysis.azp-regions rung=formula caption="Follow the AZP objective and the gain of moving one tract between regions."}
::lens{scenario=gpu-spatial-analysis.azp-regions rung=kernel caption="Inspect the standardize, candidate-move, claim and apply kernels with their retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.azp-regions rung=graph caption="Trace queen weights, standardization, up to 64 rounds of batched moves and the final objective."}

`GPUMaxPRegions`:

::lens{scenario=gpu-spatial-analysis.max-p-regions rung=datum bind=page datum=40 caption="Inspect tract 40: its poverty, population and the max-p region that claimed it."}
::lens{scenario=gpu-spatial-analysis.max-p-regions rung=formula caption="Follow the max-p rule: regions meet the population threshold, then the sum of squares decides."}
::lens{scenario=gpu-spatial-analysis.max-p-regions rung=kernel caption="Inspect the growth, enclave and selection kernels with their retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.max-p-regions rung=graph caption="Trace queen weights, eight parallel starts, growth rounds and the selection of the best start."}

`GPULocationAllocation`:

::lens{scenario=gpu-spatial-analysis.location-allocation rung=datum bind=page datum=40 caption="Inspect tract 40: its population weight, assigned facility and straight-line distance to it."}
::lens{scenario=gpu-spatial-analysis.location-allocation rung=formula caption="Follow the p-median objective: weight times distance to the nearest selected facility."}
::lens{scenario=gpu-spatial-analysis.location-allocation rung=kernel caption="Inspect the greedy choice, swap scoring and finalize kernels with their retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.location-allocation rung=graph caption="Trace the greedy construction, up to eight swap iterations and the final objective reduction."}

{/* lens:GPURoseStatistic */}
{/* lens:GPUWardRegions */}

`GPUWardRegions`:

::lens{scenario=gpu-spatial-analysis.ward-regions rung=datum bind=page datum=40 caption="Inspect tract 40: its poverty value and the Ward region it merged into."}
::lens{scenario=gpu-spatial-analysis.ward-regions rung=formula caption="Follow the Ward merge cost: size-weighted squared distance between adjacent region means."}
::lens{scenario=gpu-spatial-analysis.ward-regions rung=kernel caption="Inspect the candidate, merge and relabel kernels with their retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.ward-regions rung=graph caption="Trace queen weights, standardization and up to 32 mutual-nearest merge rounds (23 ran) toward five regions."}

`GPURoseStatistic`:

::lens{scenario=gpu-spatial-analysis.rose-statistic rung=datum bind=page datum=100 caption="Inspect segment 100: its endpoints and length weight, one of 894 rows entering the rose."}
::lens{scenario=gpu-spatial-analysis.rose-statistic rung=formula caption="Follow the circular moments: weighted moments describe street length; unweighted directions give standard Rayleigh z."}
::lens{scenario=gpu-spatial-analysis.rose-statistic rung=kernel caption="Inspect the tiled bin and moment reduction kernel with its retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.rose-statistic rung=graph caption="Trace the single-pass reduction from 894 segments into five summary words and 18 sector bins."}

### `GPUSpatialClustering`

Cluster planar points by density (DBSCAN), returning compact cluster labels, core flags and the cluster count.

::lens{scenario=gpu-spatial-analysis.spatial-clustering rung=result caption="DBSCAN with a 250 m radius and six-point cores finds 34 clusters among 2,344 observations; noise stays unclustered."}

Maximum: `gridSize` has at most `getGPUGridIndexMaximumCellCount(device)` (`2^25 - 1` on a default device: `cells + 1` uint32 cell offsets in one 128 MiB binding) cells; with `output.clusterCentroids` and sorted sums the cluster capacity is at most `GPU_SPATIAL_CLUSTERING_MAXIMUM_SORTED_CLUSTER_COUNT` (`2^24 - 1`, one 256-thread workgroup per cluster).

DBSCAN density clustering of planar points. Bounds, `epsilon`, and `minimumPoints` are per-frame
parameters (`getGPUSpatialClusteringParameterValues`). `gridSize` is only a compile-time maximum
lattice. Each encoding derives an active lattice whose cells are at least `epsilon` wide, so the
3×3 neighborhood search is always exact and `epsilon` changes never recompile. The neighbor scans
read positions and core flags from copies in grid-cell order, so the points of one cell are read
from consecutive memory whatever the input order. Clusters are joined
by a lock-free union-find that hooks larger roots under smaller ones. Labels are therefore
canonical: compact cluster IDs follow the smallest core row of each cluster, and a border point
joins the adjacent cluster with the smallest root. Outputs are per-point labels, roots, and core
flags, the cluster count, a compact list of cluster representatives, and cluster sizes and
centroids, all in the `output` bag. The settings type is `GPUSpatialClusteringSettings`.

```ts
graph.add(new GPUSpatialClustering({
  positions, parameters: parameters.importToGraph(graph), gridSize: [128, 128],
  output: {labels, clusterCount, clusters: {ids: clusterRoots, status: {count, overflow}}, clusterSizes, clusterCentroids}
}));
parameters.write(getGPUSpatialClusteringParameterValues({bounds, epsilon: 25, minimumPoints: 8}));
```

With `minimumPoints = 1` every point is a core point, so clusters are the connected components of the
graph that joins points within `epsilon`, with no noise. That is PostGIS `ST_ClusterWithin(geom,
distance)` (single-linkage clustering by distance, with the same transitive chaining); larger values
are `ST_ClusterDBSCAN(geom, eps, minpoints)`. `GPUSpatialClustering` writes labels only: pass them to
[`GPUGroupGeometry`](#gpugroupgeometry) and [`GPUGroupConvexHull`](#gpugroupconvexhull) for cluster
bounds, centers and outlines.

`denseBoxShortcut: true` enables the FDBSCAN dense-box shortcut. Points are binned into boxes of side
`0.7 * epsilon`, and a box with at least `minimumPoints` points makes all of its points core without a
neighbor scan. Labels, roots and core flags are identical to the plain algorithm. The union kernel
still scans neighbors, so the measured gain on 200,000 clustered points is only 3 to 9 percent, and
the option is off by default. Points whose box index exceeds 2^16 skip the shortcut.

`output.drawInstanceCount` (a one-row `uint32`, needs `output.clusters`) receives the clamped cluster count, so a layer
can draw one instance per cluster: import an indirect draw record's instance count with
`graph.importGPUData(id, drawCommands.getInstanceCountData(0))` and pass the view.

`sumOrder` defaults to `'sorted'` (stable sort of members by cluster, then a fixed-order segmented sum):
because labels are canonical, `clusterCentroids` is bitwise reproducible across encodings on one device,
and cluster sizes are exact either way. `'atomic'` skips the sorts but uses float atomics, so last bits vary
between runs. Behavior change: last-bit differences versus the previous atomic default.

`spatialContext` (optional) declares the coordinate space of `positions`. Omitted or planar, the search is
Euclidean in position units. It measures distance, so `local-lambert-equal-area` frames throw. `{coordinateSpace: 'longitude-latitude', metric: 'great-circle'}` clusters
`[longitude, latitude]` degrees with `epsilon` a great-circle distance in meters (kilometers with
`units: 'kilometers'`, or any `sphereRadius`; the mean Earth radius by default). Then `bounds` are
`[minLon, minLat, maxLon, maxLat]` degrees inside `[-180, 180] x [-90, 90]`. The neighbor test is the
haversine term against `sin^2(epsilon / (2 R))`, and each encoding sizes its cells to
`epsilon / R` of latitude by `2 asin(sin(epsilon / (2 R)) / cos(maxAbsLatitude))` of longitude (the whole
extent in one column near a pole), so the 3x3 search stays exact. The search does not wrap at the
antimeridian: a cluster straddling +-180 is split, so keep `bounds` on one side of it. `denseBoxShortcut`
is planar only, and other metrics (rhumb, ellipsoidal) throw. Geographic results match scikit-learn
`DBSCAN(metric='haversine')` on the test scene.
Planar positions are float32, so the representable spacing near a coordinate of magnitude `M` is about
`M * 1.2e-7`: UTM northings near 5e6 are only 0.5 m apart, and an `epsilon` below a few such spacings
(a few metres) is unreliable. Subtract a local origin from the coordinates (and from `bounds`) before
upload so the spacing stays far below `epsilon`.
Pass optional `spatialContext` to `getGPUSpatialClusteringParameterValues(settings)` to warn on short `epsilon` values; see [Precision of longitude/latitude positions](./concepts.md#precision-of-longitudelatitude-positions).

{/* parameters: GPUSpatialClustering generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUSpatialClustering` parameters

- Packer: `getGPUSpatialClusteringParameterValues(settings)`, writes `GPU_SPATIAL_CLUSTERING_PARAMETER_LENGTH` (8) `float32` values.

Settings of `getGPUSpatialClusteringParameterValues`:

| Field | Type | Description |
|---|---|---|
| `spatialContext` (optional) | `GPUSpatialContext` | Spatial context of the positions; when it is longitude/latitude, a distance under 10 float32 quanta (about 17 m) logs a warning (see Precision of longitude/latitude positions). Does not change the packed values. |
| `bounds` | `readonly [number, number, number, number]` | Inclusive `[minX, minY, maxX, maxY]` extent of the data. Points outside it, or with a non-finite coordinate, are excluded and labeled noise. Bounds should cover the data: they size the active cell lattice, and a tighter extent makes the neighbor search cheaper. Bounds with `maxX < minX` or `maxY < minY` exclude every point. |
| `epsilon` | `number` | Neighbor radius. Two points are neighbors when their distance is `<= epsilon`. Positive. |
| `minimumPoints` | `number` | Integer `>= 1`. A point is a core point when its epsilon-neighborhood, including the point itself, holds at least this many points. |

{/* /parameters */}

#### How it computes

{/* lens:GPUSpatialClustering */}

::lens{scenario=gpu-spatial-analysis.spatial-clustering rung=datum bind=page datum=1000 caption="Inspect observation 1000: its position, core flag and cluster label."}
::lens{scenario=gpu-spatial-analysis.spatial-clustering rung=formula caption="Follow the core test: observations within epsilon, counted against the minimum points."}
::lens{scenario=gpu-spatial-analysis.spatial-clustering rung=kernel caption="Inspect the grid-binned core, union and label kernels with their retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.spatial-clustering rung=graph caption="Trace the cell grid, core flags, lock-free union-find and compact labels over 2,344 observations."}

### `GPUKMeans`

Partition planar points into k clusters by Lloyd iterations, returning labels, centers, sizes, squared distances, inertia and status.

::lens{scenario=gpu-spatial-analysis.k-means rung=result caption="k-means assigns 2,344 observations to six clusters; inertia 3.1e9 m² after 14 iterations, converged."}

Deterministic k-means over planar points (scikit-learn `KMeans`, PostGIS `ST_ClusterKMeans`). Props:
`positions`, optional planar `spatialContext` (longitude/latitude throws; project first; it measures
distance, so `local-lambert-equal-area` frames also throw), `k`
(1 to 256), `maximumIterations` (1 to 64), `initialization` (`'kmeans++'` by default, or `'first-valid'`),
`restartCount` (1 to 32, default 1), the per-frame `parameters` view
(`getGPUKMeansParameterValues({seed, tolerance})`, `GPU_KMEANS_PARAMETER_LENGTH` `uint32` words) and the
`output` bag: `labels` (cluster or noise), `centers` (`k` rows), optional `sizes`, `squaredDistances` (the
explain column), `inertia` (a one-row `float32`: the total inertia, the sum of `squaredDistances`),
`status.iterationCount` and `status.converged`.
Behavior change: the default `initialization` was `'first-valid'`, which is order dependent and can place
centers next to each other; `'kmeans++'` matches scikit-learn. Each point goes to the nearest center with lowest-ID ties, means are fixed-order chunked
reductions (no sort). Results are bitwise reproducible per seed. `'kmeans++'` uses `log`, which WGSL does not
pin down across devices.

Behavior change (empty clusters): a center that receives no point is no longer left in place. After every
assignment step the empty clusters (lowest ID first) are relocated like scikit-learn's
`_relocate_empty_clusters_dense`: they take the points with the largest squared distance to their assigned
center, farthest first, and the donor clusters' counts and sums lose those points before the means are
taken. Ties in distance go to the lowest row. scikit-learn's `argpartition` leaves the order among the
relocated points unspecified; here it is always descending distance. If every point sits on its center
nothing is relocated, and a cluster that is still empty (or whose only member was taken) moves to the mean of
the largest cluster (lowest ID on ties). A center that is NaN because it could not be initialized (fewer valid
points than `k`) is never relocated. The selection is a single-workgroup iterated argmax with no atomics and
no readback; it costs `O(n E)` for `E` empty clusters and nothing when none is empty.

`restartCount` runs that many independent seeded initializations, each with its own Lloyd iterations, and
returns the one with the lowest total inertia (a fixed-order compensated sum; a tie goes to the lowest
restart index). Restart `r` uses the seed `getKMeansRestartSeed(seed, r)` (restart 0 uses `seed`), so any
single restart can be reproduced with `restartCount: 1` and that seed. The default of 1 matches
scikit-learn's `n_init='auto'` for k-means++. Restarts are separate node chains in the graph, so graph size
and GPU time grow linearly with `restartCount`; `status.iterationCount`, `status.converged` and `inertia` describe the
returned restart.
`restartCount` above 1 needs `'kmeans++'` (`'first-valid'` has no randomness).

```ts
graph.add(new GPUKMeans({positions, k: 8, maximumIterations: 20, initialization: 'kmeans++',
  parameters, output: {labels, centers, sizes}}));
parameters.write(getGPUKMeansParameterValues({seed: 3}));
```

`maximumIterations` is a cap: a cap that is too low reports `status.converged` 0 (first-valid initialization needs 20 to 30
iterations on taxi data at `k = 16`), and the largest shift usually falls straight to exactly 0, a true
Lloyd fixed point, so a tolerance relative to the data extent is enough. `tolerance` (per frame, default 0) is a convergence threshold on the largest center shift. After an
iteration whose shift is at most `tolerance`, the one-thread convergence check zeroes a GPU-written
indirect dispatch, so the later iterations' assign, reduction and relocation kernels do not launch; the
result equals running every iteration when `tolerance` is 0 (an exact fixed point). The optional one-row
`status.iterationCount` and `status.converged` outputs report the iterations used and the flag. The seed and tolerance
are `parameters` words and the k-means++ round and the restart index come from a GPU cursor word, so the graph compiles one pipeline per kernel at any `k` and
`restartCount`, and restart scoring is a two-stage (per-chunk, then fixed-order) reduction.

{/* parameters: GPUKMeans generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUKMeans` parameters

- Packer: `getGPUKMeansParameterValues(settings?)`, writes `GPU_KMEANS_PARAMETER_LENGTH` (2) `uint32` values.

Settings of `getGPUKMeansParameterValues`:

| Field | Type | Description |
|---|---|---|
| `seed` (optional) | `number` | Seed of the `'kmeans++'` hash, an integer in `[0, 2^32 - 1]`. Default 0. Restart `r` is seeded with `getKMeansRestartSeed(seed, r)`. |
| `tolerance` (optional) | `number` | Convergence tolerance: an iteration converges when the largest center shift (Euclidean, in position units) is at most this value. Finite and `>= 0`; default `0`, which stops only at an exact fixed point and so gives the same result as running every iteration. A positive value trades exactness for fewer iterations. After convergence the remaining iterations' assign, reduction and relocation kernels are skipped through a GPU-written indirect dispatch, so each one costs only a one-thread convergence check. |

{/* /parameters */}

#### How it computes

{/* lens:GPUKMeans */}

::lens{scenario=gpu-spatial-analysis.k-means rung=datum bind=page datum=1000 caption="Inspect observation 1000: its cluster label and squared distance to that center."}
::lens{scenario=gpu-spatial-analysis.k-means rung=formula caption="Follow the inertia and the center update: squared distance to the nearest center, then member means."}
::lens{scenario=gpu-spatial-analysis.k-means rung=kernel caption="Inspect the assignment, chunked reduction and center-update kernels with their retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.k-means rung=graph caption="Trace k-means++ seeding and 20 unrolled Lloyd iterations down to the inertia reduction."}

### `GPUHierarchicalDensityClustering`

Cluster planar points by density at every scale (HDBSCAN), returning compact cluster labels, core distances and the cluster count.

::lens{scenario=gpu-spatial-analysis.hierarchical-density-clustering rung=result caption="HDBSCAN finds 19 clusters of different density among 782 observations without a radius; the rest is noise."}

HDBSCAN density clustering of planar points, fully on the GPU, matching scikit-learn's `HDBSCAN` with its
defaults (`cluster_selection_method='eom'`, `allow_single_cluster=False`, `cluster_selection_epsilon=0`,
Euclidean) up to a relabeling of the clusters. Props: `positions` (at most
`GPU_HIERARCHICAL_DENSITY_CLUSTERING_MAXIMUM_ROW_COUNT`, 2^20 rows), the compile-time `minimumClusterSize`
(`min_cluster_size`, at least 2) and `minimumSamples` (`min_samples`, the point itself counted, 1 to 32,
default `min(minimumClusterSize, 32)`), and `output: {labels, coreDistances?, status?: {count}}`. Labels are
compact cluster IDs or `0xffffffff` for noise; non-finite points are noise.

The pipeline: core distances from a k-nearest search; the exact minimum spanning tree over the mutual
reachability distance by Boruvka (unrolled `ceil(log2 rows) + 1` rounds of the same kernels, gated on the GPU
component count; each point walks a Hilbert-sorted BVH and skips nodes that hold only its own component);
the single-linkage dendrogram by parallel parent splicing over the sorted tree, with Euler-tour pointer
doubling for subtree sizes; condensing by pointer jumping; stability as a fixed-order segmented sum; and
excess-of-mass selection by heavy-path tree contraction. The tree passes use GPU convergence flags and
indirect dispatch gates, including on deep chains; the row maximum bounds transient storage.
Where mutual reachability distances tie, several minimum spanning trees exist and the GPU breaks ties by point
rows, which can condense a few points differently from scikit-learn: on the oracle scenes
(`test/oracles/spatial-clustering-hdbscan.py`) the spanning-tree weights are identical and the adjusted Rand
index is at least 0.98. Membership probabilities, outlier scores, `allow_single_cluster` and `'leaf'`
selection are not provided.

```ts
graph.add(new GPUHierarchicalDensityClustering({
  positions, minimumClusterSize: 15, minimumSamples: 5,
  output: {labels, coreDistances, status: {count: clusterCount}}
}));
```

#### How it computes

{/* lens:GPUHierarchicalDensityClustering */}

::lens{scenario=gpu-spatial-analysis.hierarchical-density-clustering rung=datum bind=page datum=300 caption="Inspect observation 300: its core distance and cluster label."}
::lens{scenario=gpu-spatial-analysis.hierarchical-density-clustering rung=formula caption="Follow core distance, mutual reachability and cluster stability, which decide the surviving clusters."}
::lens{scenario=gpu-spatial-analysis.hierarchical-density-clustering rung=kernel caption="Inspect the k-nearest, Boruvka, dendrogram, condensing and selection kernels with retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.hierarchical-density-clustering rung=graph caption="Trace core distances, the spanning tree, the dendrogram and excess-of-mass selection over 782 observations."}

### `GPUSpatialWeightsMinimumSpanningTree`, `GPUSkaterRegions`, and `GPURegionPartitionEvaluation`

`GPUSpatialWeightsMinimumSpanningTree`: Find the minimum spanning forest of a contiguity graph by attribute dissimilarity, returning tree-edge flags, component labels and edge costs.

::lens{scenario=gpu-spatial-analysis.spatial-weights-minimum-spanning-tree rung=result caption="The 135 tree edges joining 136 tracts, drawn between centroids over poverty and coloured by dissimilarity; total 29.6."}

`GPUSkaterRegions`: Cut a spanning tree greedily where the within-region sum of squares falls most, returning region labels, cut edges and cut gains.

::lens{scenario=gpu-spatial-analysis.skater-regions rung=result caption="Target five regions, with at least five tracts per cut side to exclude singleton regions."}

This example sets `minimumRegionSize: 5` (spopt's `floor`) because the default of one permits
singleton regions. Each cut maximizes SSD reduction among cuts satisfying that floor; it does
not balance region sizes and may stop below the target if no feasible cut remains.

`GPURegionPartitionEvaluation`: Score any label column by within-region, between-region and total sum of squares, region sizes, boundary share and pseudo-F.

::lens{scenario=gpu-spatial-analysis.region-partition-evaluation rung=result caption="Nine summary words for the eight community areas as a partition of poverty; pseudo-F is 13.7."}

These cards use community areas 1 to 8 (136 contiguous north-side Chicago tracts). The spanning-tree card
draws tree edges as segments between GPU-computed tract centroids.

Maximum: SKATER keeps `weights.neighbors rows x (2 + columnCount)` arc statistics, at most `getGPUSkaterMaximumStatisticCount(device)` (`2^25` on a default device); `GPURegionPartitionEvaluation` takes at most `getGPURegionPartitionEvaluationMaximumRowCount(device)` (`2^24`) rows, `getGPURegionPartitionEvaluationMaximumLabelCapacity(device)` (`2^25 - 1`) labels and `labelCapacity x columnCount` at most `getGPURegionPartitionEvaluationMaximumRegionValueCount(device)` (`2^25`).

Spatially constrained regionalization over a contiguity graph (PySAL spopt `SpanningForest` and SKATER,
Assuncao et al. 2006).

- `GPUSpatialWeightsMinimumSpanningTree`: Boruvka minimum spanning forest over a symmetric
  `GPUSpatialWeights` (every link in both rows; for directed weights build the union with
  `GPUSpatialWeightsTranspose` first, unlike the AZP, Max-P and Ward operators, which symmetrize
  themselves because the tree's edge IDs are the caller's CSR slots). The edge cost is the squared Euclidean distance between (optionally standardized)
  attribute rows. A total order of `(cost, lower-row CSR slot)` makes the forest unique and equal to
  Kruskal under that order, with integer atomics and no float races. `output` holds `treeEdgeFlags`
  (per CSR slot), `componentLabels` (the minimum row of each tree) and, optionally, a compact `edges`
  list (`ids` plus `status: {count, overflow, requiredCount?}`) with `edgeEndpoints`, `edgeDissimilarities` and `standardizedValues`. It unrolls `ceil(log2 rows)` rounds. Tied costs
  can pick different edges than scipy's traversal order, with the same total cost.
- `GPUSkaterRegions`: greedy SKATER cuts of that tree. Each step evaluates the sum-of-squared-deviation
  reduction of every tree edge in parallel (Euler-tour side sums) and applies the best one, ties to the
  lowest slot, while honoring `minimumRegionSize`. The target region count and the minimum size are
  per-frame `parameters` words written with
  `getGPUSkaterRegionsParameterValues({regionCount, minimumRegionSize})`
  (`GPU_SKATER_REGIONS_PARAMETER_LENGTH` words); `maximumRegionCount - 1` steps are compile-time, so the
  partition with k regions is a prefix of the one with k + 1. `output` holds `labels` (the row index of
  each region's top), `regionCount`, `cutEdges` and `cutGains`. The partitions equal spopt `SpanningForest` (squared-Euclidean metric) exactly, including
  floors and islands; spopt's `islands='increase'` adds the tree count to `n_clusters`, while here the
  target counts every region. SKATER does not center attributes itself: the tree's `standardize`
  (default true) does, and `standardizedValues` should be passed as the SKATER `values` so the cuts and
  the tree share a metric and the f32 subtree-sum subtraction stays well conditioned.
- `GPURegionPartitionEvaluation`: scores any label column (SKATER, `GPUSpatialClustering`, `GPUKMeans`):
  region count, within, between and total sum of squared deviations, smallest and largest region, the
  fraction of weights-graph links that cross regions (a compactness proxy) and the number of ignored
  rows, plus per-label sizes and within-region deviations. The summary also holds `pseudoF`, the
  Calinski-Harabasz statistic `(between / (k - 1)) / (within / (n - k))`, pinned against
  `sklearn.metrics.calinski_harabasz_score` (NaN where undefined, 1 when the within-region dispersion
  is zero, as sklearn). `output` holds `summary` (`GPU_REGION_PARTITION_EVALUATION_LAYOUT`: `withinSumOfSquares`,
  `betweenSumOfSquares`, `totalSumOfSquares`, ...) and the optional `regionSizes` and
  `regionWithinSumOfSquares`. The cost is `O(labelCapacity * rows)`.

```ts
graph.add(new GPUSpatialWeightsMinimumSpanningTree({weights, values, columnCount: 3,
  output: {treeEdgeFlags, componentLabels, standardizedValues}}));
// skater: a GPUParameterBuffer written with getGPUSkaterRegionsParameterValues({regionCount: 8})
graph.add(new GPUSkaterRegions({weights, treeEdgeFlags, componentLabels, values: standardizedValues,
  columnCount: 3, parameters: skater.importToGraph(graph), maximumRegionCount: 12,
  output: {labels, regionCount}}));
```

Cost: the publish and apply kernels are single-threaded and each SKATER step is
`O(sum of side sizes)`, which suits thousands of units and is quadratic on path-like trees of 100,000
rows or more. Once a step finds no cut or the target region count is reached, the one-thread kernels
zero a GPU-written indirect dispatch, so the remaining steps launch only their one-thread kernels. The
step index comes from a GPU cursor word, so every step shares one pipeline per kernel.

{/* parameters: GPUSkaterRegions generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUSkaterRegions` parameters

- Packer: `getGPUSkaterRegionsParameterValues(settings)`, writes `GPU_SKATER_REGIONS_PARAMETER_LENGTH` (2) `uint32` values.

Settings of `getGPUSkaterRegionsParameterValues`:

| Field | Type | Description |
|---|---|---|
| `regionCount` | `number` | Target region count, clamped to `maximumRegionCount` on the GPU. A value at or below the number of forest trees cuts nothing. |
| `minimumRegionSize` (optional) | `number` | Minimum region size in rows. Defaults to 1. |

{/* /parameters */}

#### How it computes

{/* lens:GPUSpatialWeightsMinimumSpanningTree */}
{/* lens:GPUSkaterRegions */}
{/* lens:GPURegionPartitionEvaluation */}

`GPUSpatialWeightsMinimumSpanningTree`:

::lens{scenario=gpu-spatial-analysis.spatial-weights-minimum-spanning-tree rung=datum bind=page datum=40 caption="Inspect tract 40: its poverty value, standardized value and component label in the tree."}
::lens{scenario=gpu-spatial-analysis.spatial-weights-minimum-spanning-tree rung=formula caption="Follow the edge cost: squared difference of standardized attribute rows across each link."}
::lens{scenario=gpu-spatial-analysis.spatial-weights-minimum-spanning-tree rung=kernel caption="Inspect the standardize, edge-cost, cheapest-edge and relabel kernels with retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.spatial-weights-minimum-spanning-tree rung=graph caption="Trace queen weights, edge costs and eight Boruvka rounds that merge trees."}

`GPUSkaterRegions`:

::lens{scenario=gpu-spatial-analysis.skater-regions rung=datum bind=page datum=40 caption="Inspect tract 40: its poverty value, tree component and SKATER region."}
::lens{scenario=gpu-spatial-analysis.skater-regions rung=formula caption="Follow the cut gain: region sum of squares minus the sums of its two sides."}
::lens{scenario=gpu-spatial-analysis.skater-regions rung=kernel caption="Inspect the Euler-tour, cut-gain, best-cut and apply kernels with their retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.skater-regions rung=graph caption="Trace queen weights, the spanning tree and up to seven greedy cut steps."}

`GPURegionPartitionEvaluation`:

::lens{scenario=gpu-spatial-analysis.region-partition-evaluation rung=datum bind=page datum=40 caption="Inspect tract 40: its poverty value and community-area label in the evaluated partition."}
::lens{scenario=gpu-spatial-analysis.region-partition-evaluation rung=formula caption="Follow within, between and total sums of squares into the Calinski-Harabasz pseudo-F."}
::lens{scenario=gpu-spatial-analysis.region-partition-evaluation rung=kernel caption="Inspect the sort, region-sum, deviation and summary kernels with their retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.region-partition-evaluation rung=graph caption="Trace queen weights, the label sort, per-region sums and the nine-word summary."}

### `GPUGroupGeometry`

Summarize the geometry of each label group, returning counts, bounds, centres, medoids, standard distances and ellipses.

::lens{scenario=gpu-spatial-analysis.group-geometry rung=result caption="Mean centres of the 34 DBSCAN clusters of 2,344 observations, sized by member count; the largest has 553."}

Planar only: `spatialContext` defaults to native planar coordinates and a longitude/latitude context
throws at construction (project first with `createGPULocalFrame` and `GPULocalFrameProjection`).
Geographic group centroids live in `GPUGeometryMeasures`.
Declares `measure: 'distance'`: a Lambert equal-area local frame throws; use an
azimuthal-equidistant frame.

Per-label summaries over a `uint32` label column, for example the output of `GPUSpatialClustering`:
count, bounds `[minX, minY, maxX, maxY]`, mean center, weighted center, weight sum, medoid, standard
deviational ellipse and standard distance.

- Props: `positions` (`float32x2`), `labels`, `groupCount`, optional `noiseLabel` and `weights`,
  `parameters` (the `getGPUGeographicDistributionParameterValues` layout: origin, standard deviations,
  ellipse convention), `medoidMaximumGroupSize` (default 4096) and `output`.
- Outputs (all optional, at least one): `counts`, `bounds` (`float32x4`), `meanCenters`,
  `weightedCenters` and `weightSums` (need `weights`), `medoidIndices`, `ellipses`
  (`[angle, sigmaX, sigmaY]`, weighted when `weights` is given), `standardDistances` and `status.overflow`.
- Labels at or above `groupCount` (for example DBSCAN noise, `0xffffffff`), a label equal to
  `noiseLabel`, and rows with a non-finite coordinate are excluded. Empty groups give count 0, NaN
  floats and `GPU_GROUP_GEOMETRY_NO_MEDOID`.
- The medoid is the included row with the smallest summed Euclidean distance to the group, lowest row
  on ties, unweighted. It is quadratic per group, so a group larger than `medoidMaximumGroupSize` gets
  no medoid and sets `output.status.overflow`.
- Sums are fixed-order, so results are bitwise reproducible.

```ts
graph.add(new GPUGroupGeometry({
  positions, labels: clusterLabels, groupCount: 64,
  parameters: parameterBuffer.importToGraph(graph),
  output: {counts, bounds, meanCenters, medoidIndices, ellipses}
}));
```

{/* parameters: GPUGroupGeometry generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUGroupGeometry` parameters

- Packer: `getGPUGeographicDistributionParameterValues(settings?)`, writes `GPU_GEOGRAPHIC_DISTRIBUTION_PARAMETER_LENGTH` (8) `float32` values.

Settings of `getGPUGeographicDistributionParameterValues`:

| Field | Type | Description |
|---|---|---|
| `origin` (optional) | `readonly [number, number]` | Local origin `[x, y]` subtracted from every position before any sum, so float32 sums keep precision when the planar coordinates are large (for example Web Mercator meters). Choose a point near the data. Defaults to `[0, 0]`. |
| `standardDeviations` (optional) | `number` | Number of standard deviations that scale `standardDistances`, `ellipses` and the vertex rings (ArcGIS offers 1, 2 and 3). Any positive number is accepted. Defaults to `1`. |
| `ellipseConvention` (optional) | `GPUGeographicDistributionEllipseConvention` | Axis scaling of the ellipse: `'arcgis'` scales the axes by `sqrt(2)` as the ArcGIS Directional Distribution tool does, `'standard'` uses the plain standard deviation along each principal axis. Defaults to `'arcgis'`. |
| `orientationOnly` (optional) | `boolean` | Treat lines as undirected (angles are doubled before averaging, so a line and its reverse agree). Only affects `directionalMeans`. Defaults to `false`. |
| `medianTolerance` (optional) | `number` | Absolute tolerance, in position units, under which the last Weiszfeld step counts as converged for `medianConverged`. The iteration stops once every non-empty group's last step is at most this value. Non-negative. Defaults to `GPU_GEOGRAPHIC_DISTRIBUTION_DEFAULT_MEDIAN_TOLERANCE`. |

{/* /parameters */}

#### How it computes

{/* lens:GPUGroupGeometry */}

::lens{scenario=gpu-spatial-analysis.group-geometry rung=datum bind=page datum=1000 caption="Inspect observation 1000: its cluster label, the group it adds to."}
::lens{scenario=gpu-spatial-analysis.group-geometry rung=formula caption="Follow the group mean centre and the medoid, the member with the smallest summed distance."}
::lens{scenario=gpu-spatial-analysis.group-geometry rung=kernel caption="Inspect the sort, segmented bounds, centre, ellipse and medoid kernels with their retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.group-geometry rung=graph caption="Trace DBSCAN labels, the sort by group, segmented reductions and the medoid search."}

### `GPUGroupConvexHull`

Outline each label group by its convex hull, returning hull vertices in CSR layout with emitted counts and true sizes.

::lens{scenario=gpu-spatial-analysis.group-convex-hull rung=result caption="Convex hulls of the 34 DBSCAN clusters, one outline per cluster; the largest has 15 vertices."}

Planar only: a longitude/latitude `spatialContext` throws at construction; project first with
`createGPULocalFrame` and `GPULocalFrameProjection`.

Convex hull per label (Andrew monotone chain), for cluster and hot-region outlines and trajectory
footprints (PostGIS `ST_ConvexHull` per group, `GeoSeries.convex_hull`). Props: `positions`, `labels`,
`groupCount`, optional `noiseLabel`, `maximumVerticesPerGroup` (at least 3), `vertexCapacity` and
`output`: `vertexIndices`, optional `vertexPositions`, `offsets` (`groupCount + 1`), `counts`, optional
`sizes` (true hull sizes) and `status.overflow` (bit 1: a hull exceeded the per-group cap; bit 2: the total
capacity ran out).

- Hulls are counter-clockwise, not closed, and start at the smallest `(x, y)` vertex. Collinear points
  are excluded, equal points collapse to the lowest row, one distinct point gives one vertex, and
  all-collinear points give the two extremes.
- Orientation is exact: coordinates are snapped to a 2^29 lattice with one global power-of-two scale,
  and the sign is computed with 64-bit integer arithmetic. Output positions are the original floats.
- A hull over the per-group cap is dropped whole. Groups are admitted in group order until the total
  capacity runs out, and the first group that does not fit and all later ones are dropped (`counts` 0,
  `sizes` still true).
- `extremePointFilter` (default `true` up to 65536 groups) is an Akl-Toussaint filter before the sorts:
  per group it finds the eight extreme lattice points (min and max of x, y, x + y and x - y) and gives
  every row strictly inside their octagon (exact test) the excluded-row key. Boundary and collinear
  rows are kept, so the hull is bit for bit the same with `true` or `false`. `GPUSort` has a fixed
  length and cannot be skipped from the GPU, so the three sorts still cover every row; the saving is in
  the prefilter and chain, which skip the discarded run. The extreme tables hold 16 words per group, so
  forcing it on above 65536 groups throws.
- `prefilterLevels` (0 to 4, `GPU_GROUP_CONVEX_HULL_MAXIMUM_PREFILTER_LEVELS`; default 2 from 8192 rows)
  discards points inside nested octagon-like filters before the chain. For one 500k-point disk the median
  went from 551 ms (level 0) to 36, 8.8 and 7.6 ms (levels 1 to 3); 64 groups of 500k: 33 to 7.3 ms. With
  4096 small groups the levels cost 2 to 3 ms (3.0 ms at level 0), so set 0 there.
- The chain is one invocation per group, so a group whose points are all on the hull stays serial. For a
  concave outline, close a raster `GPUDistanceField`; exact alpha shapes stay on the CPU.

#### How it computes

{/* lens:GPUGroupConvexHull */}

::lens{scenario=gpu-spatial-analysis.group-convex-hull rung=datum bind=page datum=1000 caption="Inspect observation 1000: its cluster label, the group whose hull it may bound."}
::lens{scenario=gpu-spatial-analysis.group-convex-hull rung=formula caption="Follow the exact lattice orientation test that keeps only counter-clockwise hull turns."}
::lens{scenario=gpu-spatial-analysis.group-convex-hull rung=kernel caption="Inspect the lattice, extreme-point filter, sort and monotone-chain kernels with their retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.group-convex-hull rung=graph caption="Trace DBSCAN labels, three sort passes, the per-group chain and the hull vertex emission."}

### `GPUNeighborhoodSummary`

Summarise each tract's neighbourhood from a spatial weights table, returning a row-standardised mean of values and the entropy of neighbour labels.

::lens{scenario=gpu-spatial-analysis.neighborhood-summary rung=result caption="Mean poverty of each tract's queen neighbours across 791 tracts, from row-standardised weights; range 4.2 to 57.5%."}

Per-row summaries of a neighborhood over a `GPUSpatialWeights` CSR (momepy `describe`, generalized and
categorical spatial lag). The members of row `i`, in fixed order, are the focal row (with
`includeFocal`, weight `focalWeight`, default 1) and then the active CSR slots in slot order. A slot
is active when its weight is positive; zero-weight slots, a listed self slot, out-of-range slots,
zero-mask neighbors and, for numeric statistics, non-finite values are skipped. Zero-weight slots are
therefore not members (`count`, `min`, `max`, `median` and the categorical outputs ignore them),
which differs from libpysal `lag_spatial` and momepy when zero weights are listed. When
`weights.information.transform` is `'row'` the slot weights are divided by the sum of the active slot
weights, so `sum` (the lag) stays a weighted average when neighbors are masked or non-finite (the
focal weight is unchanged). If `weights.status.overflow` is nonzero every output is NaN. A masked-out focal row gives `count` 0 and NaN elsewhere. With `k` members, weights `w` and
values `x`:

- `count` is `k`, `weightSum` is `W`, `sum` is `sum w x` (the lag) and `mean` is `sum / W` (NaN when
  `W = 0`).
- `min` and `max` are unweighted. `standardDeviation` is the weighted population deviation about the
  mean, NaN when `W = 0`.
- `median` is unweighted (the mean of the two middle values for even `k`) for rows with at most
  `maximumNeighbors` members (default 32, at most 64). Larger rows give NaN and set
  `output.status.overflow`, which is required when `median` is requested.
- `output.modes` (`uint32`) is the category with the largest weighted frequency, ties to the lowest category
  (`GPU_NEIGHBORHOOD_SUMMARY_NO_MODE` for none). `output.entropy` is the Shannon entropy (natural log) of the
  weighted category shares. Categorical cost is `O(k^2)` per row.

`statistics` fixes the order of the `output.table` columns (no repeats): column `c` of row `i` is at
`i * statistics.length + c`. `values` may be `float32` or `uint32` counts (one conversion node,
exact below 2^24). The numeric and categorical parts are two kernels, each below the eight
storage buffer limit. Sums run in member order in f32, so results are deterministic.

```ts
graph.add(new GPUNeighborhoodSummary({
  weights, values, categories, includeFocal: true,
  statistics: ['count', 'mean', 'standardDeviation', 'median'],
  output: {table, modes, entropy, status: {overflow}}
}));
```

#### How it computes

{/* lens:GPUNeighborhoodSummary */}

::lens{scenario=gpu-spatial-analysis.neighborhood-summary rung=datum bind=page datum=420 caption="Inspect tract 420: its neighbours, their weights and the weighted mean."}
::lens{scenario=gpu-spatial-analysis.neighborhood-summary rung=formula caption="Follow the neighbour mean and label entropy: weighted sums over the active slots."}
::lens{scenario=gpu-spatial-analysis.neighborhood-summary rung=kernel caption="Inspect the numeric and categorical kernels with their retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.neighborhood-summary rung=graph caption="Trace queen weights, row standardisation and the neighbourhood summary kernels."}

### `GPUEmpiricalBayesRates`

Shrink each tract's crude rate toward the pooled rate according to its population, returning smoothed rates, raw rates and moments.

::lens{scenario=gpu-spatial-analysis.empirical-bayes-rates rung=result caption="Crude and empirical-Bayes poverty rates of 791 tracts around a pooled 25.3%; small populations shrink the most."}

Global empirical-Bayes standardization and smoothing of rates (esda `assuncao_rate`, as used by
`Moran_Rate` and `Moran_Local_Rate` with `adjusted=True`, and `Empirical_Bayes`). Over the included rows
(nonzero `mask`, finite event count `e`, finite population `b > 0`; esda gives NaN for `b = 0`, here the
row is excluded): `y = e / b`, `m = sum e / sum b`, `s2 = sum b (y - m)^2 / sum b`, `a = s2 - m / mean(b)`.

`events` and `populations` are `float32` or `uint32` (for example the `uint32` `counts` of
`addPointsInPolygonsChoroplethRecipe`; one conversion node, exact below 2^24). Outputs go in `output`:

- `output.standardizedRates`: `z = (y - m) / sqrt(v)` with `v = a + m / b`, replaced by `m / b` when negative
  (esda's guard).
- `output.smoothedRates`: `r = w y + (1 - w) m` with `w = a / (a + m / b)`. By default (`clampPriorVariance: true`)
  `a` is clamped at 0 (Marshall 1991, GeoDa), so `w` stays in `[0, 1]` and `a <= 0` (counts no more dispersed
  than Poisson, common for small `n` or low counts) shrinks every rate fully to `m`. esda `Empirical_Bayes`
  does not clamp, so with `a < 0` its `w` leaves `[0, 1]` and rates are pushed away from `m`, possibly below
  zero. Pass `clampPriorVariance: false` for exact esda parity. This is a default change.
- `standardizedRates` never clamps `a`; it keeps esda's `assuncao_rate` guard (`v = m / b` when `a + m / b < 0`).
- Optional `output.rawRates` and `output.summary` (`GPU_EMPIRICAL_BAYES_SUMMARY`, 8 floats: count, event sum,
  population sum, pooled rate, weighted rate variance, prior variance, clamped prior variance). The
  `priorVariance` slot stays unclamped (esda `ebi_a`); `clampedPriorVariance` is `max(a, 0)`.

Sums are fixed-order two-level trees, so results are bitwise reproducible. Excluded rows are NaN, which
`GPULocalMoran`, `GPUGlobalSpatialStatistics` and the permutation tests exclude. Write
`standardizedRates` to a transient view and pass it as `values`. For the neighborhood-pooled variant see
[`GPUSpatialEmpiricalBayesRates`](#gpuspatialempiricalbayesrates).

```ts
graph.add(new GPUEmpiricalBayesRates({events, populations, output: {standardizedRates: z}}));
graph.add(
  new GPUGlobalSpatialStatistics({weights, values: z, statistics: ['moran'], output: {summary}})
);
```

#### How it computes

{/* lens:GPUEmpiricalBayesRates */}

::lens{scenario=gpu-spatial-analysis.empirical-bayes-rates rung=datum bind=page datum=420 caption="Inspect tract 420: its crude rate, shrinkage weight and smoothed rate."}
::lens{scenario=gpu-spatial-analysis.empirical-bayes-rates rung=formula caption="Follow the shrinkage: a weighted mix of the tract rate and the pooled rate."}
::lens{scenario=gpu-spatial-analysis.empirical-bayes-rates rung=kernel caption="Inspect the partial sums, totals, variance and per-tract rates kernels."}
::lens{scenario=gpu-spatial-analysis.empirical-bayes-rates rung=graph caption="Trace the two-pass reduction into the summary and the per-tract rates."}

### `GPUSpatialEmpiricalBayesRates`

Pool each tract with its queen neighbours into a local prior, returning spatial rates and shrunken rates per tract.

::lens{scenario=gpu-spatial-analysis.spatial-empirical-bayes-rates rung=result caption="Spatial empirical-Bayes poverty rates of 791 tracts pooled over queen neighbourhoods, spanning 2.8% to 75.9%."}

Neighborhood-pooled rate smoothing (esda 2.10 `Spatial_Rate` and `Spatial_Empirical_Bayes`, libpysal 4.15
weights). The neighborhood of an included row `i` is `i` plus its listed neighbors that are included
(nonzero `mask`, finite `events`, finite `populations > 0`); only the weights' structure counts (esda sets
`transform = 'b'`). With `E = sum e`, `B = sum b` and `n` members: the spatial rate is `E / B`
(`Spatial_Rate.r`), the local prior mean is `m = E / B`, the local variance is
`s2 = sum b (e / b - m)^2 / B`, `a = max(s2 - m / (B / n), 0)`, and the smoothed rate is
`w e_i / b_i + (1 - w) m` with `w = a / (a + m / b_i)`. `events` and `populations` are `float32` or
`uint32`. Outputs are the optional `output.spatialRates`, `output.smoothedRates`, `output.priorMeans` and
`output.priorVariances` (NaN for excluded rows). One invocation reduces each row
over its ascending slots, so results are bitwise reproducible. Isolates smooth to their own rate.

esda 2.10.0 `Spatial_Empirical_Bayes` itself sets `r_var[r_var < 0] = 0`, so the clamp matches esda and needs
no option (unlike the global `Empirical_Bayes`, which does not clamp; see
[`GPUEmpiricalBayesRates`](#gpuempiricalbayesrates)). The spec pins esda 2.10.0 on a 6x6 queen lattice (spatial
rate within 1e-5 relative, smoothed within 2e-4 in f32) and on low-dispersion counts where the local `a`
clamps. Deviations: a neighborhood with no events gives `0 / 0 = NaN` in esda and 0 here; rows excluded by the
mask or a non-positive population are skipped as neighbors instead of raising; a neighbor ID equal to the
focus row is skipped, where esda would count it twice.

```ts
graph.add(new GPUSpatialEmpiricalBayesRates({events, populations, weights, output: {smoothedRates}}));
```

#### How it computes

{/* lens:GPUSpatialEmpiricalBayesRates */}

::lens{scenario=gpu-spatial-analysis.spatial-empirical-bayes-rates rung=datum bind=page datum=420 caption="Inspect tract 420: its neighbourhood totals, local prior mean and smoothed rate."}
::lens{scenario=gpu-spatial-analysis.spatial-empirical-bayes-rates rung=formula caption="Follow the local shrinkage: the tract rate mixed with its neighbourhood prior mean."}
::lens{scenario=gpu-spatial-analysis.spatial-empirical-bayes-rates rung=kernel caption="Inspect 791 invocations, each summing its own and its neighbours' events and populations."}
::lens{scenario=gpu-spatial-analysis.spatial-empirical-bayes-rates rung=graph caption="Trace queen weights into the neighbourhood-pooled smoothing kernel."}

### `GPUEmergingHotSpots`

Classify a space-time cube into ArcGIS emerging hot and cold spot categories, returning Gi* z-scores, Mann-Kendall trends and categories.

::lens{scenario=gpu-spatial-analysis.emerging-hot-spots rung=result caption="Emerging hot spot category of 224 cells over twelve months; 35 are sporadic hot, zero is no pattern."}

ArcGIS-style emerging hot spot analysis over a dense space-time cube, on a regular lattice or on any
spatial weights.
`values` is indexed `cell * sliceCount + slice` (the `GPUTemporalReduction` layout, `cell = row *
gridWidth + column`); a `float32` view treats NaN as a missing bin, a `uint32` view (for example
`GPUTemporalReduction` `counts`) is read as `f32(count)`. An optional per-cell `mask` excludes cells.
`cellCount * sliceCount` must fit one storage binding (2^25 bins at the default 128 MiB
`maxStorageBufferBindingSize`); `getCommandNodes` throws above it.

Pass exactly one neighborhood: `gridWidth` and `gridHeight` for the lattice, with a per-frame
`radius` in cells, or `weights` over the cells (rows are cells) plus a compile-time `selfWeight`,
which works on H3 cells, polygons or points. With binary weights equal to the lattice disc, weights
mode reproduces lattice mode.

```ts
const hotSpots = new GPUEmergingHotSpots({
  values, gridWidth: 64, gridHeight: 64, sliceCount: 24, maximumRadius: 4, parameters,
  output: {giZScores, trendZ, trendP, trendS, category, hotSliceCount, coldSliceCount}
});
graph.add(hotSpots);
const settings = {radius: 2, temporalWindow: 3, confidenceLevel: 0.95};
parameterBuffer.write(hotSpots.getParameterValues(settings));
console.log(hotSpots.getInferenceMetadata(settings));
// {alternative: 'two-sided', permutationCount: 0, multipleTesting: 'benjamini-hochberg',
//  significanceLevel: 0.05, testedRowsDefinition: 'valid space-time bins'}
```

Defaults changed: `multipleTesting` is `'benjamini-hochberg'` (it was no correction), `confidenceLevel` is 0.95
(it was 0.9), the trend level is `1 - confidenceLevel`, `minimumTrendSliceCount` is 10, and
`finalSlice` is `'require'`. Pass `multipleTesting: 'none'` for the previous rule.

Passes (deterministic, no float atomics):

1. Global `n`, mean and standard deviation of valid bins: fixed 1024-bin block sums added in order.
2. Space-time Gi* per bin: the spatial neighbors (lattice cells with `dx^2 + dy^2 <= radius^2`, or
   the cell's weights row plus the cell itself at `selfWeight`) over the current and `temporalWindow`
   previous slices, `z = sum w (x - X) / (S sqrt((n S1 - W^2) / (n - 1)))` over valid bins.
3. Two-sided normal p-values `erfc(|z| / sqrt(2))` of every valid Gi* bin, adjusted by
   `GPUMultipleTestingAdjustment` over the whole family of valid bins of the cube (never a display
   mask). A bin is hot or cold when its adjusted p-value is at most `1 - confidenceLevel`, with the
   sign of z. With `multipleTesting: 'none'` the rule is `|z| >= criticalZ` with the same two-sided
   critical values, and the p-value passes are skipped unless `output.pValues` or
   `output.adjustedPValues` is requested. Bins within a temporal window overlap and neighboring cells share observations, so the
   tests are dependent: Benjamini-Hochberg is only guaranteed under positive dependence, and
   `'benjamini-yekutieli'` controls the false discovery rate under arbitrary dependence at a cost in
   power of about `ln m`. The correction needs fewer than 2^24 bins.
4. Per-cell Mann-Kendall on the finite z series: exact integer `S`, tie-corrected variance
   `[n(n-1)(2n+5) - sum t(t-1)(2t+5)] / 18` (ties by exact f32 equality), continuity-corrected z, and a
   two-sided p-value from the Numerical Recipes `erfcc` fit. Consecutive Gi* values share
   `temporalWindow` slices, so the series is serially correlated and this p-value is anti-conservative
   (ESRI has the same property): treat the trend level as a screening threshold. Cells with fewer
   than `minimumTrendSliceCount` finite slices (default 10, ESRI guidance) get NaN `trendZ` and
   `trendP`.
5. Classification into the 17 `GPU_EMERGING_HOT_SPOTS_CATEGORIES` (0 = no pattern, hot 1..8, cold 9..16:
   new, consecutive, intensifying, persistent, diminishing, sporadic, oscillating, historical) plus
   `UNDETERMINED` (17).

`UNDETERMINED` marks a cell whose category cannot be decided: with `finalSlice: 'require'` (the
default) a cell with data whose last slice is missing, and a cell that is persistent in one direction
(Intensifying, Persistent or Diminishing, hot or cold) but has fewer finite slices than
`minimumTrendSliceCount`. It is never silently reported as Persistent. `finalSlice: 'last-valid'`
keeps the old behavior, where the last finite slice stands in for the final slice, so a stale cell can
be labelled New or Persistent for a period it has no data in. Cells without any finite slice stay
`NO_PATTERN`. The optional `output.lastValidSlice` gives each cell's last finite slice index
(`GPU_EMERGING_HOT_SPOTS_NO_SLICE`, `0xffffffff`, for none).

Legend contract: category codes never change meaning with `multipleTesting`, `confidenceLevel` or
`finalSlice`; those options only change which bins are hot or cold and which cells are undetermined.
`getInferenceMetadata(settings)` returns a plain `GPUInferenceMetadata` for the legend, derived from the
same `GPUEmergingHotSpotsSettings` that `getParameterValues(settings)` packs (`significanceLevel` is
`1 - confidenceLevel`). `getParameterValues` takes `multipleTesting` from the node and rejects a
different `settings.multipleTesting`, so the packed values and the legend cannot disagree.

`criticalZ` is only accepted with `multipleTesting: 'none'` (the parameter packer throws otherwise).
Without a correction `confidenceLevel` must be 0.9, 0.95 or 0.99 or `criticalZ` must be given; with a
correction any level in `(0, 1)` works.

Per-frame parameters (radius, temporal window, confidence level, critical z, trend significance,
persistence fraction, minimum trend length) live in a parameter buffer and never rebuild the graph.
Compile-time: `gridWidth`, `gridHeight`, `sliceCount` (at most 256), `maximumRadius` (at most 32),
`multipleTesting`, `finalSlice`, value format, optional views. Outputs are caller-owned fields of
`output` (`GPUEmergingHotSpotsOutput`): `giZScores` (per bin), `trendZ`, `trendP`, `trendS`, `category`,
`hotSliceCount`, `coldSliceCount` (per cell), optional `pValues` and `adjustedPValues` (per bin),
optional `lastValidSlice` (per cell) and optional `globalStatistics`
`[n, mean, variance, standardDeviation]`.

{/* parameters: GPUEmergingHotSpots generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUEmergingHotSpots` parameters

- Packer: `getGPUEmergingHotSpotsParameterValues(settings)`, writes `GPU_EMERGING_HOT_SPOTS_PARAMETER_LENGTH` (8) `float32` values.
- Packer: `instance.getParameterValues(settings)`, writes the `float32` parameter view.

Settings of `getGPUEmergingHotSpotsParameterValues`:

| Field | Type | Description |
|---|---|---|
| `radius` (optional) | `number` | Lattice mode: spatial neighborhood radius in lattice cells. Cell `(dx, dy)` offsets with `dx * dx + dy * dy <= radius * radius` are neighbors (the focal cell included, so `0` is the focal cell alone). Clamped to the compile-time `maximumRadius`. Ignored in weights mode. Defaults to `0`. |
| `temporalWindow` | `number` | Temporal window `k >= 0`: neighbors span the current slice and the `k` previous slices. |
| `confidenceLevel` (optional) | `number` | Confidence level of hot and cold bins, in `(0, 1)`. A bin is hot or cold when its adjusted two-sided p-value is at most `1 - confidenceLevel`. With `multipleTesting: 'none'` only `0.9`, `0.95` and `0.99` (the tabulated critical z values) or an explicit `criticalZ` are accepted. Defaults to `0.95` (it was `0.9`). |
| `criticalZ` (optional) | `number` | Explicit critical z that overrides `confidenceLevel` for hot and cold bins. Only meaningful without a correction: it requires `multipleTesting: 'none'` and throws otherwise, because a fixed critical z cannot express a family-wise or false discovery rate threshold. |
| `multipleTesting` (optional) | `GPUMultipleTestingMethod` | The compile-time `multipleTesting` of the node these settings feed. It does not change the packed values; it makes the packer reject a `criticalZ` combined with a correction and decide whether `confidenceLevel` must be tabulated. Defaults to `'benjamini-hochberg'`, like the node. `GPUEmergingHotSpots.getParameterValues` fills it from the node and rejects a different value. |
| `trendSignificanceLevel` (optional) | `number` | Two-sided p-value at or below which the Mann-Kendall trend is significant, in `(0, 1)`. Defaults to `1 - confidenceLevel`. Consecutive Gi* z values share `temporalWindow` slices, so the series is serially correlated and this p-value is anti-conservative (ESRI has the same property); treat it as a screening threshold. |
| `minimumTrendSliceCount` (optional) | `number` | Cells with fewer finite slices than this get NaN `trendZ` and `trendP`, and a persistent cell that would need the trend decision gets `UNDETERMINED`. A non-negative integer; `0` disables the guard. Defaults to 10, ESRI's guidance for Mann-Kendall. |
| `persistentFraction` (optional) | `number` | Fraction of slices that makes a spot persistent or historical, in `(0.5, 1]`. Default `0.9`. |

{/* /parameters */}

#### How it computes

{/* lens:GPUEmergingHotSpots */}

::lens{scenario=gpu-spatial-analysis.emerging-hot-spots rung=datum bind=page datum=100 caption="Inspect cell 100: its trend statistics, hot and cold month counts and category."}
::lens{scenario=gpu-spatial-analysis.emerging-hot-spots rung=formula caption="Follow space-time Gi*, the Mann-Kendall trend and the adjusted p-value rule for hot and cold months."}
::lens{scenario=gpu-spatial-analysis.emerging-hot-spots rung=kernel caption="Inspect the moment, Gi*, adjustment, Mann-Kendall and classification kernels with their retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.emerging-hot-spots rung=graph caption="Trace the global moments, Gi* per bin, Benjamini-Hochberg adjustment and per-cell classification."}

### `addSpaceTimeHotSpotsRecipe`

Turn timestamped events into an emerging hot spot map, returning the month-by-cell cube, Gi* z-scores, trends and categories.

::lens{scenario=gpu-spatial-analysis.space-time-hot-spots-recipe rung=result caption="Hot spot categories of 224 cells from 2,344 timestamped observations over twelve months; 35 are sporadic hot."}

Timestamped events to an emerging hot spot map: `GPUCalendarBuckets` decodes the event times, an
adapter bins every event into a dense `cell * sliceCount + slice` cube, `GPUEmergingHotSpots`
classifies it and an ordinal `GPUColorScale` colors the categories (18 palette entries, 0 to 17).

For `cells.kind: 'lattice'`, optional `cells.spatialContext` describes positions and bounds, defaults to
native planar, and rejects longitude/latitude. `result.spatialContext` returns this resolved context;
with caller-supplied cell IDs it carries `cells.weights.spatialContext` when declared.

Slices are absolute time steps, the ArcGIS "time step interval", so the Mann-Kendall trend runs over
a monotone axis:

```ts
const spaceTime = addSpaceTimeHotSpotsRecipe(graph, {
  timestamps, calendarParameters, cells,
  settings: {temporalWindow: 1, radius: 1.5},
  parameters: emergingParameters, // GPUParameterBuffer the recipe fills from settings
  slices: {step: 'month', interval: 3, origin: {year: 2022, month: 11}, count: 12},
  multipleTesting: 'benjamini-yekutieli'
});
spaceTime.setSettings({temporalWindow: 2, radius: 1.5}); // next frame, no rebuild
```

Slice `floor((stepIndex(t) - stepIndex(origin)) / (interval * stepsPerUnit))` holds the events at
local time `t` (the same UTC-offset semantics as `GPUCalendarBuckets`). `stepIndex` counts whole
steps: `year`, `year * 4 + quarter - 1`, `year * 12 + month - 1`, days since the epoch for `'day'` and
`'week'` (a week is 7 days anchored to the origin day), hours since the epoch for `'hour'`. It is
integer arithmetic on the GPU (Hinnant's `days_from_civil`) and on the CPU
(`getSpaceTimeSliceIndex`). Events before the origin or past the last slice are skipped and counted
in the `slice-range` status stage (`invalidCount`, or the `outputs.status.skippedEventCount` override). The sorted path's
`cube-counts` stage also reports `requiredCount` (the `outputs.status.groupRequiredCount` output).

Optional `timeZone: GPUCalendarBucketsTimeZone` forwards a transition table from
`getGPUTimeZoneTransitionTable` to `GPUCalendarBuckets`, which searches it per row on the GPU.
It is mutually exclusive with `utcOffsets`; setting both throws. With an `'hour'` step, local
slices on transition days are 0 h or 2 h long, so use UTC for hourly cubes.

Cyclic fields (month of year, weekday, hour of day, day of year) are no longer accepted: `month = 1`
would pool every January of a multi-year range, and a trend over hour or weekday is meaningless.
Cyclic profiles belong in a non-trend summary such as `GPUCalendarBuckets` `hourWeekdayCounts`.

`multipleTesting` (default `'benjamini-hochberg'`) and `finalSlice` (default `'require'`) pass through
to `GPUEmergingHotSpots`. The per-frame `settings` (`GPUSpaceTimeHotSpotsRecipeSettings`: the emerging
hot spot settings without `multipleTesting`) are packed by the recipe into its `parameters`
`GPUParameterBuffer`, and `result.inference` (a plain `GPUInferenceMetadata` object) is
`GPUEmergingHotSpots.getInferenceMetadata` of the same object; `result.setSettings` repacks both. The
result also has optional `pValues`, `adjustedPValues` and `lastValidSlice` outputs.

#### How it computes

{/* lens:addSpaceTimeHotSpotsRecipe */}

::lens{scenario=gpu-spatial-analysis.space-time-hot-spots-recipe rung=formula caption="Follow the binning: event time to month slice, position to lattice cell, counts per cell and slice."}
::lens{scenario=gpu-spatial-analysis.space-time-hot-spots-recipe rung=kernel caption="Inspect the calendar decode, cube binning and emerging hot spot kernels with their retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.space-time-hot-spots-recipe rung=graph caption="Trace calendar decoding, the atomic cube count and the emerging hot spot chain."}

### `GPUGeographicDistribution`

Measure the weighted centre, spread and ellipse of tract centroids, returning mean centre, standard distance and ellipse per group.

::lens{scenario=gpu-spatial-analysis.geographic-distribution rung=result caption="Population-weighted standard distance of 791 Chicago tract centroids: 11,343 m in the local frame."}

`GPUGeographicDistribution` computes the ArcGIS "Measuring Geographic Distributions" statistics
over planar points, per group or per masked selection: counts, weight sums, mean centre, Weiszfeld
median centre, standard distance, standard deviational ellipse, linear directional mean, and
polygon rings for the ellipse and the standard-distance circle. Every statistic is Euclidean:
`spatialContext` (default native planar) rejects `longitude-latitude` at construction with a
"project positions to a planar frame first" error, and `GPURoseStatistic` does the same. A
spherical mean centre (normalised mean of 3D unit vectors) and a great-circle standard distance are
deferred, because they need a second implementation of every reduction pass in both the chunked and
the sorted path.

Declares `measure: 'distance'`: a Lambert equal-area local frame throws; use an
azimuthal-equidistant frame.

```ts
const parameters = new GPUParameterBuffer(device, {
  id: 'distribution-parameters',
  format: 'float32',
  length: GPU_GEOGRAPHIC_DISTRIBUTION_PARAMETER_LENGTH
});
graph.add(
  new GPUGeographicDistribution({
    positions, weights, groupIds, groupCount: 8, mask,
    parameters: parameters.importToGraph(graph),
    output: {meanCenters, ellipses, ellipseVertices}
  })
);
parameters.write(
  getGPUGeographicDistributionParameterValues({origin: [x0, y0], standardDeviations: 2})
);
```

- Inputs: `positions` (`float32x2`, fewer than 2^24 rows), optional `weights` (a non-finite,
  negative or zero weight excludes the row), `groupIds` with compile-time `groupCount` (ids at or
  above it are excluded), `mask`, and `lineEnds` for the directional mean.
- Per-frame parameters (no recompile): `origin` (local origin subtracted before every sum for
  float32 stability), `standardDeviations`, `ellipseConvention` (`'arcgis'` scales the axes by
  `sqrt(2)`, `'standard'` does not), `orientationOnly`, `medianTolerance`.
- Median centre: Weiszfeld steps from the mean centre, all groups together, for at most the
  compile-time `maximumMedianIterations` (default 24). Behavior change: the steps stop after the first one
  in which every non-empty group moved at most `medianTolerance` (a GPU-written indirect dispatch skips the
  rest), instead of always running `maximumMedianIterations` steps; `medianConverged` is then 1 for every
  non-empty group.
- Divisors are the weight sum, as in ArcGIS (not `n - 1`). Second moments are central (a mean
  pass, then a pass about the mean), never `E[x^2] - E[x]^2`. No small-sample degrees-of-freedom
  correction is applied, so for small `n` the spread is biased low by `sqrt((n - 1) / n)`; there is
  no confidence interval or significance test.
- `ellipses` holds `[angle, sigmaX, sigmaY]` per group. As in ArcGIS the "y" axis is the long one
  (`sigmaY >= sigmaX`). `angle` is counter-clockwise from +x of the short x axis, so the long axis
  points at `angle + pi / 2`; a ring vertex is `M + R(angle) (sigmaX cos t, sigmaY sin t)`.
- `directionalMeans` holds `[meanAngle, circularVariance, meanLength]` (angles counter-clockwise
  from +x). With `orientationOnly`, angles are doubled before averaging and the mean angle lies in
  `(-pi/2, pi/2]`.
- `ellipseVertices` and `circleVertices` are counter-clockwise rings of `polygonVertexCount`
  (default 64) vertices per group, not closed, ready for a deck.gl `PolygonLayer` or `PathLayer`.
- The median centre runs at most `maximumMedianIterations` (default 24) Weiszfeld steps from the mean
  centre; `medianConverged` is 1 when the last step moved at most `medianTolerance`.
- Empty groups report 0 counts and weight sums and NaN in every other output.
- Determinism: rows are sorted by group once and every sum, including each Weiszfeld step, is a
  fixed-order segmented tree sum. No float atomics; repeated encodings are bitwise identical.

{/* parameters: GPUGeographicDistribution generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUGeographicDistribution` parameters

- Packer: `getGPUGeographicDistributionParameterValues(settings?)`, writes `GPU_GEOGRAPHIC_DISTRIBUTION_PARAMETER_LENGTH` (8) `float32` values.

Settings of `getGPUGeographicDistributionParameterValues`:

| Field | Type | Description |
|---|---|---|
| `origin` (optional) | `readonly [number, number]` | Local origin `[x, y]` subtracted from every position before any sum, so float32 sums keep precision when the planar coordinates are large (for example Web Mercator meters). Choose a point near the data. Defaults to `[0, 0]`. |
| `standardDeviations` (optional) | `number` | Number of standard deviations that scale `standardDistances`, `ellipses` and the vertex rings (ArcGIS offers 1, 2 and 3). Any positive number is accepted. Defaults to `1`. |
| `ellipseConvention` (optional) | `GPUGeographicDistributionEllipseConvention` | Axis scaling of the ellipse: `'arcgis'` scales the axes by `sqrt(2)` as the ArcGIS Directional Distribution tool does, `'standard'` uses the plain standard deviation along each principal axis. Defaults to `'arcgis'`. |
| `orientationOnly` (optional) | `boolean` | Treat lines as undirected (angles are doubled before averaging, so a line and its reverse agree). Only affects `directionalMeans`. Defaults to `false`. |
| `medianTolerance` (optional) | `number` | Absolute tolerance, in position units, under which the last Weiszfeld step counts as converged for `medianConverged`. The iteration stops once every non-empty group's last step is at most this value. Non-negative. Defaults to `GPU_GEOGRAPHIC_DISTRIBUTION_DEFAULT_MEDIAN_TOLERANCE`. |

{/* /parameters */}

#### How it computes

{/* lens:GPUGeographicDistribution */}

::lens{scenario=gpu-spatial-analysis.geographic-distribution rung=datum bind=page datum=420 caption="Inspect tract 420: its centroid and population weight enter every group sum."}
::lens{scenario=gpu-spatial-analysis.geographic-distribution rung=formula caption="Follow the standard distance and ellipse: weighted second moments about the mean centre."}
::lens{scenario=gpu-spatial-analysis.geographic-distribution rung=kernel caption="Inspect the moment passes, the spread kernel and the ellipse ring kernel."}
::lens{scenario=gpu-spatial-analysis.geographic-distribution rung=graph caption="Trace the weighted sums into the mean centre, spread and ellipse vertices."}

### `GPURegionStatistics`

`GPURegionStatistics`: Select tract centroids inside a lasso and reduce them to counts, sum, mean, extremes and a histogram.

::lens{scenario=gpu-spatial-analysis.region-statistics rung=result caption="Tracts whose centroids fall inside a lasso: 240 of 791 Chicago tracts."}

`GPUPickRegionMask`: Convert a region-pick result into a deduplicated 0/1 mask over the source rows.

::lens{scenario=gpu-spatial-analysis.pick-region-mask rung=result caption="20 tracts of community area 8, picked by a supplied region-pick result."}

`GPURegionMask`: Test every point against a rectangle or lasso, writing one 0/1 mask value per row.

::lens{scenario=gpu-spatial-analysis.region-mask rung=result caption="Tracts whose centroids fall inside a rectangle: 180 of 791 Chicago tracts."}

Selects points inside a per-frame rectangle, circle, lasso polygon (world space or screen pixels
through a view-projection transform), pick region (`GPUIndexPickingTarget.addRegionPass` result),
or caller mask, and reduces them into selected count, value count, sum, mean, minimum, maximum, and
a fixed-bin histogram packed into one small `output.summary` buffer. Optional stable selected IDs
(`output.selectedRows`, a `GPUCompactOutput`) and a 0/1 `output.mask` stay on the GPU for
highlighting. `GPURegionStatisticsReadback` reads the summary through a
`GPUReadbackRing` without stalling. `GPURegionMask` and `GPUPickRegionMask` are exported as
standalone building blocks.

```ts
graph.add(new GPURegionStatistics({
  region: {kind: 'polygon', vertices: lasso, vertexCount: lassoCount.importToGraph(graph),
    screenTransform: transform.importToGraph(graph)},
  positions, values, histogram: {binCount: 16},
  output: {summary: graph.importGPUData('summary', summaryData), selectedRows}
}));
```

Lasso membership is even-odd with an exact side test: a straddling edge counts when the exact
orientation sign puts the point strictly on its inner side, so points on an edge are outside and no
f32 crossing abscissa decides the result.

`GPURegionMask` and `GPUPickRegionMask` write `output: {mask, status: {overflow}}`
(`GPURegionMaskOutput`); `GPUPickRegionMask` reads the packed pick result from `picks`, as does a
`{kind: 'pick-region', picks}` region.

Coordinates: `spatialContext` defaults to native planar. Rectangle, lasso and pick regions are
topological tests and also accept `longitude-latitude` (metric `'none'`): edges are straight lines
in degree space and require linear edges. Prepare polygons that cross ±180 or enclose a pole with
`GPULonLatPolygonPreparation`. Optional `output.status.antimeridianUncertainCount` and
`output.status.poleUncertainCount` report unprepared rings in `GPURegionMask` and
`GPURegionStatistics`. These one-row `uint32` counters are cleared and rewritten every encoding
for world-space lon/lat lassos; planar contexts, screen-space lassos and other region kinds leave
them untouched. A lasso consumes one ring at a time, including an empty prepared copy through a
zero active vertex count. A `'radius'` region is a planar distance in
position units, so it throws at construction in `longitude-latitude` (the region kind is
compile-time); project positions to a planar frame first, or use a rectangle or lasso. Geographic
metrics are rejected. Radius selections measure distance and reject `local-lambert-equal-area` frames. `GPURegionMask` takes the same `spatialContext`; `GPUPickRegionMask` reads no
positions and has none.

An optional `spatialIndex: {kind: 'grid', index, candidateCapacity}` takes a caller-built
`GPUGridIndex` over the same positions and reduces only candidates from the cells a world-space
rectangle, circle, or lasso touches. Results match the full scan except for float reassociation in
`sum` and `mean`; the `candidatesTruncated` flag reports capacity overflow or rows missing from the
index.

An optional `output.drawInstanceCount` (one-row `uint32` view, for example the instance-count word
of an indirect draw record) receives the same capacity-clamped selected count as
`output.selectedRows.status.count` on every encoding, so selected rows can be drawn instanced through
`output.selectedRows.ids` without a readback. It requires `output.selectedRows`; statistics-only
graphs read the count from `output.summary`.

#### How it computes

{/* lens:GPURegionStatistics */}
{/* lens:GPUPickRegionMask */}
{/* lens:GPURegionMask */}

`GPURegionStatistics`:

::lens{scenario=gpu-spatial-analysis.region-statistics rung=datum bind=page datum=420 caption="Inspect tract 420: its centroid and value, and whether the lasso selects it."}
::lens{scenario=gpu-spatial-analysis.region-statistics rung=formula caption="Follow the summary: selected finite values give the count, sum and mean."}
::lens{scenario=gpu-spatial-analysis.region-statistics rung=kernel caption="Inspect the lasso mask, count, sum, extent and summary kernels."}
::lens{scenario=gpu-spatial-analysis.region-statistics rung=graph caption="Trace the mask, count and reduction nodes into the summary buffer."}

`GPUPickRegionMask`:

::lens{scenario=gpu-spatial-analysis.pick-region-mask rung=datum bind=page datum=420 caption="Inspect tract 420: which pick pairs name it and the mask value they set."}
::lens{scenario=gpu-spatial-analysis.pick-region-mask rung=formula caption="Follow the rule: a row is 1 when a pick pair of the chosen batch names it."}
::lens{scenario=gpu-spatial-analysis.pick-region-mask rung=kernel caption="Inspect the clear, scatter and overflow kernels with their retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.pick-region-mask rung=graph caption="Trace the mask clear, the pick scatter and the overflow copy."}

`GPURegionMask`:

::lens{scenario=gpu-spatial-analysis.region-mask rung=datum bind=page datum=420 caption="Inspect tract 420: its centroid against the rectangle bounds."}
::lens{scenario=gpu-spatial-analysis.region-mask rung=formula caption="Follow the rule: a point is 1 inside the inclusive, valid rectangle on both axes."}
::lens{scenario=gpu-spatial-analysis.region-mask rung=kernel caption="Inspect 791 point tests with the retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.region-mask rung=graph caption="Trace the overflow flag and the single region mask kernel."}

### `GPUZonalStatistics`

Reduce tract values to one count, sum, mean and deviation per community area, using per-row zone membership.

::lens{scenario=gpu-spatial-analysis.zonal-statistics rung=result caption="People below the poverty line summed over 77 community areas from 791 tracts; total 688,314."}

Choropleth aggregation. Each point is joined to the polygon feature that contains it (an internal
`GPUPointInPolygonJoin`), or the caller supplies per-point feature rows. The points are then
reduced per feature to count, valid-value count, sum, weighted sum and weight sum, mean, minimum,
maximum, density (count divided by a caller area or a GPU shoelace area), and a feature-value
`[min, max]` extent for color scales.

Boundary points follow `boundaryRule` (passed to the join). The default `'half-open'` counts a point
on an edge or vertex shared by coverage features exactly once, whatever the feature row order; points
on the outer right or top edge of the coverage are counted by no feature. `'closed'` counts every
boundary point (smallest containing row wins) and `'open'` only strict interiors.

Polygon features use the four-level `GPUPolygonGeometryPort` layout (`positions`, `featureOffsets`,
`polygonOffsets`, `ringOffsets`; one output row per feature, multipart features included) with the
join options; their `sourceIds` are not read. Point positions are the top-level `positions`.

The output views you pass select the statistics at compile time. Empty features give a sum of 0 and
NaN for the mean, minimum and maximum. `sumOrder: 'sorted'` stably sorts points by feature and
reduces each segment in a fixed tree order, so sums are bitwise reproducible at a bounded cost. It is
the default when every point-rate input (positions, values, weights, `pointFeatureRows`) is packed.
Chunked inputs fall back to `sumOrder: 'atomic'`, whose float atomics can vary in the last bits
between runs and serialize when many points share a feature.

```ts
graph.add(new GPUZonalStatistics({
  features: {kind: 'polygons', positions: polygonPositions, featureOffsets, polygonOffsets,
    ringOffsets, candidateCapacity: positions.length * 2},
  positions, values: incomes, weights: households,
  output: {counts, means, densities, extent, extentStatistic: 'mean', status: {overflow}}
}));
```

`output.status.uncertainCount` (one `uint32` row) reports the join's uncertain candidate pairs. A nonzero
value means some points could not be classified and the per-feature statistics may be low. It is
always 0 for caller-supplied feature rows.

- Variance. `output.variances` and `output.standardDeviations` are the unbiased estimates in two
  passes (mean first, then `sum w (v - m)^2`), so there is no `E[v^2] - E[v]^2` cancellation. Unweighted it
  is `ddof = 1` (`numpy.var(v, ddof=1)`). With `weights`, which act as reliability weights, it is
  `sum w (v - m)^2 / (V1 - V2 / V1)` with `V1 = sum w`, `V2 = sum w^2`, equal to
  `numpy.cov(v, aweights=w, ddof=1)`. A feature with fewer than two valid rows, or a non-positive
  denominator, gives NaN, never 0. Variances need packed point views (not chunked vectors) and follow
  `sumOrder`. Checked against numpy in `zonal_variance.py`.
- Weighted density. `output.weightedDensities` is `weightSums / area` (needs `weights`); the area rules
  of `densities` apply (finite and positive, else NaN).
- Coordinates. `spatialContext` accepts planar (default) or `longitude-latitude` with metric `'none'`.
  Membership runs `GPUPointInPolygonJoin` with a containment-only context derived from `spatialContext`
  (metric `'none'`), so in degrees it is topological with straight edges (not geodesics) and polygons
  require linear edges. Prepare polygons that cross ±180 or enclose a pole with
  `GPULonLatPolygonPreparation`; optional one-row `uint32`
  `output.status.antimeridianUncertainCount` / `output.status.poleUncertainCount` report unprepared
  rings and are cleared every encoding. Planar contexts and caller-supplied feature rows leave
  those counters untouched. The shoelace area of degrees is
  wrong, so a longitude-latitude context rejects GPU areas (`densities`, `weightedDensities`,
  `featureAreas` without `areas`) at construction: pass spherical `areas` in square meters, for example
  from `GPUGeometryMeasures`. GPU-computed planar areas and densities measure area and reject
  `local-azimuthal-equidistant` frames; supplied `areas` and means-only configurations do not trigger
  this area-frame restriction. Counts, sums, means, extrema and variances need no area.
- Not provided: median and quantiles (they need a per-feature sort of the values), weighted minima and
  maxima, fractional polygon coverage (a point belongs to a feature or it does not).

#### How it computes

{/* lens:GPUZonalStatistics */}

::lens{scenario=gpu-spatial-analysis.zonal-statistics rung=datum bind=page datum=420 caption="Inspect tract 420: its zone row, value and share of the zone sum."}
::lens{scenario=gpu-spatial-analysis.zonal-statistics rung=formula caption="Follow the zone sum, mean and deviation: finite values grouped by zone row."}
::lens{scenario=gpu-spatial-analysis.zonal-statistics rung=kernel caption="Inspect the sorted sums and the mean and deviation kernels."}
::lens{scenario=gpu-spatial-analysis.zonal-statistics rung=graph caption="Trace the stable sort by zone, segmented sums, means and deviations."}

### `GPUFocalStatistics`

Summarise each raster cell's moving window, writing the window mean, deviation and valid-cell count per cell.

::lens{scenario=gpu-spatial-analysis.focal-statistics rung=result caption="Window mean elevation (radius 3) on a 120 m terrain lattice; means span 1,083 to 1,531 m."}

Moving-window statistics over a float32 raster, like ArcGIS `FocalStatistics` and GRASS
`r.neighbors`: mean, sum, min, max, range, population standard deviation, and the count of valid
cells. The radius, window shape, minimum count, and center nodata policy are per-frame.
The operator is raster-only: it reads a lattice of values and never positions, so it takes no
`spatialContext`, its `grid` (`GPURasterMetadata`) only sizes the raster, and its radius is in
cells, not ground units. Cell size and shape (for example
lon/lat cells that shrink with latitude) are the caller's concern.

```ts
graph.add(
  new GPUFocalStatistics({
    values: raster, // row-major float32, NaN = nodata
    validity, // optional uint32, 0 = nodata
    noDataValue: -9999, // optional sentinel
    grid, // GPURasterMetadata: width * height cells, any affine
    maximumRadius: 16, // compile-time cap, at most 64
    parameters: focalParameters.importToGraph(graph), // 4 float32
    output: {mean, standardDeviation, min, max, counts} // any non-empty subset
  })
);
focalParameters.write(
  getGPUFocalStatisticsParameterValues({radius: 3, shape: 'circle', minimumCount: 5})
); // no recompile
```

- Square windows cover `|dx|, |dy| <= floor(radius)`; circles keep offsets with
  `dx^2 + dy^2 <= radius^2`. The radius is clamped to `maximumRadius`.
- Window cells outside the raster and nodata cells are skipped, so edge windows are clipped.
  A cell is nodata (NaN) with fewer than `max(minimumCount, 1)` valid cells, or when its own value
  is nodata and `propagateCenterNoData` is set. `count` is written for every cell.
- Direct gather in fixed row-major window order: deterministic, no atomics. Standard deviation
  subtracts the window mean in a second pass. Moments and extremes are separate kernels, each
  scheduled only when one of its outputs is requested.
- Cost is `(2r + 1)^2` reads per cell; see Status and limitations for the still-open separable and
  summed-area variants.

{/* parameters: GPUFocalStatistics generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUFocalStatistics` parameters

- Packer: `getGPUFocalStatisticsParameterValues(settings)`, writes `GPU_FOCAL_STATISTICS_PARAMETER_LENGTH` (4) `float32` values.

Settings of `getGPUFocalStatisticsParameterValues`:

| Field | Type | Description |
|---|---|---|
| `radius` | `number` | Window radius in cells. A square window covers `\|dx\|, \|dy\| <= floor(radius)`; a circle covers offsets with `dx^2 + dy^2 <= radius^2` inside that square. Clamped to the contributor's compile-time `maximumRadius`. |
| `shape` (optional) | `GPUFocalStatisticsShape` | Window shape. Defaults to `'square'`. |
| `minimumCount` (optional) | `number` | Cells whose window has fewer valid cells are nodata. Defaults to 1. |
| `propagateCenterNoData` (optional) | `boolean` | When true, a nodata center cell yields nodata statistics. Defaults to false. |

{/* /parameters */}

#### How it computes

{/* lens:GPUFocalStatistics */}

::lens{scenario=gpu-spatial-analysis.focal-statistics rung=datum bind=page datum=528 caption="Inspect cell 528: its window, mean, deviation and valid-cell count."}
::lens{scenario=gpu-spatial-analysis.focal-statistics rung=formula caption="Follow the window mean and population deviation over the valid cells of each window."}
::lens{scenario=gpu-spatial-analysis.focal-statistics rung=kernel caption="Inspect 1,024 gather invocations reading each window in row-major order."}
::lens{scenario=gpu-spatial-analysis.focal-statistics rung=graph caption="Trace the single moments kernel from the elevation raster to the window statistics."}

### `GPUSimilarLocations`

Rank rows by weighted distance in standardized attribute space to a reference, returning ranks, distances and the best row ids.

::lens{scenario=gpu-spatial-analysis.similar-locations rung=result caption="Distance of 136 north-side tracts to one reference tract in five indicators; nearest 0.44, reference blank."}

Maximum: `columnCount` is at most `GPU_SIMILAR_LOCATIONS_MAXIMUM_COLUMN_COUNT` (one 256-lane reduction per column within a uint32 dispatch) and the value table at most `getGPUSimilarLocationsMaximumValueCount(device)` (`2^25` on a default device) float32 values.

Ranks rows by weighted Euclidean distance, in standardized attribute space, to a reference row or the mean
of several selected rows (the ArcGIS Similarity Search task; checked against NumPy, SciPy and scikit-learn distances). Props: `values` (row-major `rowCount * columnCount`), `columnCount`,
`referenceMask` (a `uint32` mask of the reference rows), optional `mask`, per-frame `parameters`
(`getGPUSimilarLocationsParameterValues`: result count, most or least similar, reference exclusion and
attribute weights), `standardization` (`'zscore'`, population, or `'rank'`, percentile; compile-time) and
`resultCapacity` (compile-time size of `output.topIds`).

- Outputs: `ranks` (0 is the most similar; unranked rows hold `GPU_SIMILAR_LOCATIONS_NO_RANK`),
  `distances` (NaN when unranked), `topIds` (the best rows in rank order) and `status.count`.
- Rows with a non-finite attribute or a zero mask are never ranked. A stable `GPUSort` on the distance bits
  keeps ties on the lowest ID.
- `'rank'` standardization uses a quadratic scan, so it suits tens of thousands of rows, not millions.

{/* parameters: GPUSimilarLocations generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUSimilarLocations` parameters

- Packer: `getGPUSimilarLocationsParameterValues(settings, columnCount)`, writes `getGPUSimilarLocationsParameterLength(...)` `float32` values.
  - `columnCount` (`number`): Number of attributes of the contributor.

Settings of `getGPUSimilarLocationsParameterValues`:

| Field | Type | Description |
|---|---|---|
| `resultCount` | `number` | Number of top-ranked rows reported in `output.topIds`, clamped to the compile-time capacity. |
| `direction` (optional) | `'most' \| 'least'` | `'most'` ranks the nearest rows first; `'least'` ranks the farthest first. Defaults to `'most'`. |
| `excludeReference` (optional) | `boolean` | Excludes the reference (selected) rows from the ranking. Defaults to true. |
| `weights` (optional) | `ArrayLike<number>` | Non-negative weight per attribute. Defaults to 1 for every attribute. |

{/* /parameters */}

#### How it computes

{/* lens:GPUSimilarLocations */}

::lens{scenario=gpu-spatial-analysis.similar-locations rung=datum bind=page datum=60 caption="Inspect tract 60: its distance to the reference tract and its similarity rank."}
::lens{scenario=gpu-spatial-analysis.similar-locations rung=formula caption="Follow the z-scores, the reference vector and the weighted Euclidean distance to it."}
::lens{scenario=gpu-spatial-analysis.similar-locations rung=kernel caption="Inspect the column statistics, standardize, reference, distance, sort and rank kernels with retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.similar-locations rung=graph caption="Trace validation, z-scores, the reference mean, distances and the stable sort into ranks."}

### `GPULocalOutlierFactor`

Score each point's local density against its neighbours' over a kNN table, returning k-distance, density, LOF and an outlier mask.

::lens{scenario=gpu-spatial-analysis.local-outlier-factor rung=result caption="Local outlier factor of 2,344 observations from 10 neighbours; 75 exceed 3 and mark sparse points."}

Local Outlier Factor (Breunig 2000; geo `OutlierDetection`, scikit-learn `LocalOutlierFactor`, Sedona
`ST_LocalOutlierFactor`) over a kNN table. `neighbors` is a `GPUSpatialWeights` with `distances`, typically
`GPUNeighborSearch` in `'knn'` mode as a self join; the weight values are unused. Per row it writes
`kDistance` (the largest neighbor distance), `localReachabilityDensity = 1 / (mean max(kDistance_j, d_ij) +
densityFloor)` and `lof = mean(lrd_j) / lrd_i`, which equals `-negative_outlier_factor_`, all in the `output` bag. Optional
outputs are an `outlier` mask (`lof > threshold`) and a one-row `outlierCount` (integer atomics, deterministic).
`parameters` (`getGPULocalOutlierFactorParameterValues({threshold, densityFloor = 1e-10})`) change per
frame without recompiling. There is no default `threshold`: LOF has no null distribution, so no cut is
calibrated (the former hidden 1.5 is gone). Omit it and the mask and count are all 0 while `lof` is
unaffected; otherwise pick a cut from the score distribution. LOF is exploratory, not inferential: no
p-value, nothing "significant". Choose `k` (the neighbor count of the table) of at least 10 and larger than
the smallest group that should count as a cluster rather than as outliers (Breunig 2000), and compare
scores only at the same `k`. The input must be a true kNN list; with ties at the k-th distance a list cut
at exactly `k` entries changes the k-distance neighborhood (as in scikit-learn). The class reads only
distances; `neighbors.spatialContext` describes their length unit, which also applies to `densityFloor`.
Optional `spatialContext` declares the floor's context and is checked against `neighbors.spatialContext`
with `assertCompatibleGPUSpatialContexts` when both are supplied. LOF itself is a unit-free ratio. The floor keeps coincident clusters finite at density `1 / densityFloor`, and
points whose neighbors are such duplicates score enormous values, as in scikit-learn, so use a `k` above
the duplicate count or a larger floor. A row with no neighbors gets kDistance 0, density 0 and lof 1 and is
never flagged. A radius-mode table works, but the k-distance is then the largest in-band distance. Against
a pinned scikit-learn run (24 points with `k = 5`, and a duplicate case) kDistance was within 2e-6 relative
and lof within 1e-4, with the mask and count exact.

{/* parameters: GPULocalOutlierFactor generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPULocalOutlierFactor` parameters

- Packer: `getGPULocalOutlierFactorParameterValues(settings?)`, writes `GPU_LOCAL_OUTLIER_FACTOR_PARAMETER_LENGTH` (4) `float32` values.

Settings of `getGPULocalOutlierFactorParameterValues`:

| Field | Type | Description |
|---|---|---|
| `threshold` (optional) | `number` | Rows with `lof > threshold` are flagged in the `outlier` mask and counted. There is no default: LOF has no null distribution, so no cut is calibrated (Breunig et al. 2000 give none; values near 1 are as dense as their neighbors, well above 1 locally sparse, and the scale depends on `k` and the data). Choose it from the score distribution, for example a top fraction. When omitted the packed threshold is NaN, no row is flagged, the `outlier` mask is all 0 and `outlierCount` is 0; the `lof` scores are unaffected. The flag is exploratory, not a significance statement. |
| `densityFloor` (optional) | `number` | Added to each row's mean reachability distance before inverting it, so a cluster of coincident points (mean reachability 0) gets a large finite density `1 / densityFloor` instead of infinity. Must be positive and finite. Defaults to scikit-learn's `1e-10`. Raise it (for example to the coordinate resolution) to make duplicate-heavy data less extreme. Its unit is the length unit of `neighbors.spatialContext`; declare the context of this value through `GPULocalOutlierFactorProps.spatialContext` to check compatibility. |

{/* /parameters */}

#### How it computes

{/* lens:GPULocalOutlierFactor */}

::lens{scenario=gpu-spatial-analysis.local-outlier-factor rung=datum bind=page datum=1000 caption="Inspect observation 1000: its k-distance, density, LOF and outlier flag."}
::lens{scenario=gpu-spatial-analysis.local-outlier-factor rung=formula caption="Follow the reachability distance, local density and the ratio of neighbour density to its own."}
::lens{scenario=gpu-spatial-analysis.local-outlier-factor rung=kernel caption="Inspect the k-distance, density, score and outlier-mask kernels with their retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.local-outlier-factor rung=graph caption="Trace the ten-nearest-neighbour search into k-distance, density, LOF and the thresholded mask."}
