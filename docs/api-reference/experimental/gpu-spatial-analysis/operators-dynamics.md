---
title: "GPU Spatial Analysis: Distribution dynamics & space-time tests"
sidebar_label: "Distribution dynamics & space-time tests"
description: "Distribution dynamics & space-time tests contributors in @luma.gl/experimental/gpu-spatial-analysis."
---

import {ExperimentalDocsTabs} from '@site/src/components/docs/experimental-docs-tabs';

# Distribution dynamics & space-time tests

<ExperimentalDocsTabs active="gpu-spatial-analysis-operators" />

Distribution dynamics contributors use period-major columns: period `t` of row `i` is at
`t * rowCount + i`, with `periodCount` periods.

### `GPUClassAssignment`

Assigns each county's 2023 unemployment rate to one of five quantile classes using shared class breaks.

::lens{scenario=gpu-spatial-analysis.class-assignment rung=result caption="437 Midwest counties coloured by quantile class of 2023 unemployment; higher class means higher rate."}

`new GPUClassAssignment({values, breaks, classCount, mask, output})` assigns each value the number of
inner edges `e[1..k-1]` that are `<=` the value, clamped, the same rule as `GPUClassBreaks.classCounts`.
NaN and masked rows get `GPU_CLASS_ASSIGNMENT_NO_CLASS` (`0xffffffff`). Run one `GPUClassBreaks` over the
pooled `n * T` column and assign every period with its breaks to get one legend for a time slider.

#### How it computes

{/* lens:GPUClassAssignment */}

::lens{scenario=gpu-spatial-analysis.class-assignment rung=datum datum=15 caption="Cook County, Illinois (row 15 of the county table): its rate and class."}

::lens{scenario=gpu-spatial-analysis.class-assignment rung=formula caption="A county's class is the number of inner breaks at or below its rate."}

::lens{scenario=gpu-spatial-analysis.class-assignment rung=kernel caption="Inspect the recorded shader and its input and output bindings."}

::lens{scenario=gpu-spatial-analysis.class-assignment rung=graph caption="Follow the recorded dependencies from uploaded inputs to captured outputs."}

### `GPUClassificationFit`

Scores how well five quantile classes describe 2023 county unemployment: per-class counts, medians, deviations and GADF.

::lens{scenario=gpu-spatial-analysis.classification-fit rung=result caption="Quantile classes of 437 counties; the table lists counts, medians and fit statistics per class."}

Goodness of fit of a classification (mapclassify 2.11 `adcm`, `get_gadf()`, `get_tss()`): pass `values`, the
`classes` from `GPUClassAssignment`, `classCount` (1 to 256) and an optional `mask`. Outputs are `counts`,
exact per-class `medians` (numpy convention, mean of the two middle values), `absoluteDeviations` and
`squaredDeviations` (one row per class plus a last row for all counted rows) and an optional five-row
`summary` `[GADF, ADCM, ADAM, GVF, TSS]` (`GPU_CLASSIFICATION_FIT_*`). `ADCM` is the sum of absolute
deviations from the class medians, `ADAM` the same from the overall median, `GADF = 1 - ADCM / ADAM` (1 when
`ADAM` is 0), and `TSS` is mapclassify's name for the within-class sum of squares about the class means, not a
total sum of squares. `GVF = 1 - TSS / total sum of squares` is an extension, because mapclassify 2.11 has no
`gvf`. Empty classes add 0, as in mapclassify. Two stable radix sorts (value, then class) leave each class a
sorted segment, so medians are read directly and sums run in sorted order: no atomics, bit-reproducible, no
per-class search passes.

The spec pins mapclassify 2.11.0 Quantiles, NaturalBreaks and EqualInterval fits (about 1e-4 relative).
Deviations: sums run in float32 and in sorted order rather than index order, non-finite values are skipped, and
at least one row is required.

```ts
graph.add(new GPUClassificationFit({values, classes, classCount: 5, output: {counts, medians, absoluteDeviations, squaredDeviations, summary}}));
```

#### How it computes

{/* lens:GPUClassificationFit */}

::lens{scenario=gpu-spatial-analysis.classification-fit rung=datum datum=4 caption="Class 4, the highest class of 2023 rates: its county count and median."}

::lens{scenario=gpu-spatial-analysis.classification-fit rung=formula caption="GADF compares absolute deviations around class medians with those around the overall median."}

::lens{scenario=gpu-spatial-analysis.classification-fit rung=kernel caption="Inspect the recorded shader and its input and output bindings."}

::lens{scenario=gpu-spatial-analysis.classification-fit rung=graph caption="Follow the recorded dependencies from uploaded inputs to captured outputs."}

### `GPUTransitionMatrix`

**`GPUTransitionMatrix`**: Counts how many counties move between unemployment classes from one year to the next, with row-normalised probabilities.

::lens{scenario=gpu-spatial-analysis.transition-matrix rung=result caption="Counts of 437 counties moving between five classes over 15 yearly steps, 2008 to 2023."}

**`GPUTransitionDiagnostics`**: Tests whether five states share one transition matrix and solves each chain's steady state, flagging sparse states.

::lens{scenario=gpu-spatial-analysis.transition-diagnostics rung=result caption="Per-state transition counts of Illinois, Indiana, Michigan, Ohio and Wisconsin feed Q, LR, steady states and sparsity."}

Markov transition counts between classes of consecutive periods (giddy `Markov`).

- Props: `classes` (`rowCount * periodCount` rows; the row count is derived from it), `periodCount`,
  `classCount`, optional `periodLag` (default 1),
  `conditionClasses` with `conditionCount`, `mask`, and `output: {counts, probabilities?, rowTotals?, ignored?}`.
- `counts` holds exact `uint32` counts at `(c * K + from) * K + to` for `t = 0 .. T - lag - 1`
  (integer atomics, independent of order). `probabilities` are the row-normalized floats, 0 for empty
  rows. `rowTotals` and `ignored` are explain columns.
- Optional diagnostics (also available on their own as `GPUTransitionDiagnostics`, taking the `counts`
  tensor): `output.homogeneity` (needs `conditionCount > 1`; five rows `Q`, `Q_p_value`, `LR`,
  `LR_p_value`, `dof`, indexed by `GPU_TRANSITION_HOMOGENEITY`), `output.steadyStates` (ergodic
  distribution of the pooled chain, then one per condition, `chains * K` rows), `output.stateSparsity`
  (two rows, `GPU_TRANSITION_STATE_SPARSITY`: states with 0 < n < `minimumTransitions`, and states with
  n = 0) and `output.status.converged`. `minimumTransitions` defaults to 5. The steady-state solver keeps
  its elimination matrix in a storage scratch buffer the class allocates (`chains * K^2` floats), so 64
  classes fit the 8,192-byte private memory limit of core WebGPU.
- Homogeneity follows giddy's `homogeneity` (Bickenbach and Bode 2003): Q and LR compare each conditional
  matrix with the pooled one and are chi-square with `sum_i (b_i - 1)(A_i - 1)` degrees of freedom. A
  from-state with no transitions adds 0 degrees of freedom; giddy adds 1 for it, so `dof` differs from
  `dof_hom` by the number of empty pooled rows. The p-values are upper-tail chi-square (`'greater'`) from
  an f32 incomplete gamma (relative error below 6e-5 for p above 1e-6); `dof = 0` gives NaN. The test is
  asymptotic: with sparse rows (see `stateSparsity`) it is unreliable.
- Steady states use a direct Grassmann-Taksar-Heyman solve per chain. States with no transition in or out
  are dropped (probability 0, as giddy's `Markov` drops classes absent from the data); the remaining
  states must have an outgoing transition and form one communicating class, otherwise the whole chain is
  NaN (giddy raises or returns several distributions). `converged` is one when every chain that has
  transitions got a steady state. The solver is direct, so `iterationLimitReached` is never written.
  A periodic chain is solved (its stationary distribution, not a limit of powers).

```ts
// `classCount` is the one-row uint32 view written by GPUClassBreaks.
graph.add(new GPUClassAssignment({values, breaks, classCount, output: classes}));
graph.add(new GPUTransitionMatrix({classes, periodCount, classCount: 5, output: {counts, probabilities}}));
```

#### How it computes

{/* lens:GPUTransitionMatrix */}

::lens{scenario=gpu-spatial-analysis.transition-matrix rung=datum datum=0 caption="Cell 0: counties in class 0, the lowest, that are still in class 0 a year later."}

::lens{scenario=gpu-spatial-analysis.transition-matrix rung=formula caption="Each probability is a transition count divided by the count leaving that class."}

::lens{scenario=gpu-spatial-analysis.transition-matrix rung=kernel caption="Inspect the recorded shader and its input and output bindings."}

::lens{scenario=gpu-spatial-analysis.transition-matrix rung=graph caption="Follow the recorded dependencies from uploaded inputs to captured outputs."}

#### How it computes

{/* lens:GPUTransitionDiagnostics */}

::lens{scenario=gpu-spatial-analysis.transition-diagnostics rung=datum datum=0 caption="Cell 0: Illinois counties in class 0 that are still in class 0 a year later."}

::lens{scenario=gpu-spatial-analysis.transition-diagnostics rung=formula caption="Q sums squared gaps between each state's counts and the pooled expectation."}

::lens{scenario=gpu-spatial-analysis.transition-diagnostics rung=kernel caption="Inspect the recorded shader and its input and output bindings."}

::lens{scenario=gpu-spatial-analysis.transition-diagnostics rung=graph caption="Follow the recorded dependencies from uploaded inputs to captured outputs."}

### `GPUSpatialMarkov` and `GPULISAMarkov`

**`GPUSpatialMarkov`**: Splits class transitions by the quantile class of the queen-neighbour mean rate and tests whether the conditional matrices differ.

::lens{scenario=gpu-spatial-analysis.spatial-markov rung=result caption="Transition counts for 437 counties split by neighbour lag class: 5 lag classes by 5 by 5 cells."}

**`GPULISAMarkov`**: Counts yearly moves of counties between local Moran quadrants HH, LH, LL and HL using queen weights.

::lens{scenario=gpu-spatial-analysis.lisa-markov rung=result caption="Transitions among LISA states, 2008 to 2023; without significance gating state 0 stays empty."}

- `GPUSpatialMarkov` (giddy `Spatial_Markov`): props `values` (`float32`, `rowCount * periodCount`, period-major),
  `classes` (`uint32`, same layout, for example from `GPUClassAssignment`), `classCount` (`K`, 1 to 64),
  `weights` (square self-join `GPUSpatialWeights`, constant across periods), `periodCount` (at least 2),
  optional `periodLag` (default 1), `normalize` (default true), `mask`, `maximumLagClassCount`,
  `lagParameters` (`getGPUClassBreaksParameterValues` view), optional `lagMethods` (default
  `['quantile']`) and an `output` of `counts`, `lagBreaks`, `lagClassCount` and the optional
  `probabilities`, `rowTotals`, `ignored`, `lagValues` and `lagClasses`. For each period it runs `GPUSpatialLag` (row-standardized by default,
  `normalize`), classifies the lag with one pooled `GPUClassBreaks`, assigns the lag classes, and builds a
  transition matrix conditioned on the lag class at the start period. `counts` is
  `maximumLagClassCount x K x K`; slices past the produced `lagClassCount` stay zero. The `output` also
  takes the `GPUTransitionMatrix` diagnostics (`homogeneity`, `steadyStates`, `stateSparsity`,
  `status.converged`) and `minimumTransitions`: `homogeneity` is giddy's `Q`, `Q_p_value`, `LR`,
  `LR_p_value`, `dof_hom` across the lag classes; `steadyStates` is giddy's `s` (rows `0..K-1`, pooled) and
  `S` (rows `(1 + c) * K ..`), NaN where a chain is not ergodic. The pooled chain here sums the counted
  transitions over the lag classes, so it excludes rows without a lag class. A small `Q_p_value` says the
  spatial lag matters; a large one says the conditional matrices cannot be told apart from the pooled one.
  `getInferenceMetadata()` describes the tests (analytic, `'greater'`, no correction).
- `GPULISAMarkov` (giddy `LISA_Markov`): props `values`, `weights`, `periodCount` (at least `periodLag + 1`),
  optional `periodLag`, `parameters` (the `GPULocalMoran` view from
  `getGPUSpatialAutocorrelationParameterValues`, which carries the significance level), optional `mask`,
  `gating` and an `output` of `counts` and the optional `probabilities`, `rowTotals`, `ignored`,
  `steadyStates`, `stateSparsity` and `quadrants`. It packs the values into a 64-aligned period stride, so
  every period compiles the same pipelines, runs `GPULocalMoran` per period and counts moves
  between the five states (`GPU_LISA_MARKOV_STATE_COUNT`: 0 not significant, 1 HH, 2 LH, 3 LL, 4 HL). The
  layout is the same for both `gating` modes, so a legend never changes with the option.
  - `gating: 'none'` (default, giddy `LISA_Markov` parity): every valid row is in its quadrant, the signs of
    the centered value and of the lag; state 0 stays empty and the chain is a four-state chain. A row at
    exactly the mean or with an exactly zero lag has no quadrant (esda would call it low) and is ignored.
  - `gating: 'significance'`: rows whose analytic local Moran p-value is above the per-frame significance
    level are state 0. The test is the per-period analytic z at an uncorrected level, not esda
    permutations, and each period is gated on its own, so a transition into or out of state 0 can be a
    threshold crossing rather than a change of the local pattern. Treat it as a map of significant
    clusters, not as a test of dynamics. `getInferenceMetadata(settings)` takes the same
    `GPUSpatialAutocorrelationSettings` packed into `parameters`, describes the test and returns
    `undefined` for `'none'`.
  - Quadrant numbering is esda's default (`geoda_quads=False`): 1 HH, 2 LH, 3 LL, 4 HL. With
    `geoda_quads=True` esda swaps 2 and 3 (1 HH, 2 LL, 3 LH, 4 HL).
  - Missing data: a row-period that is masked, non-finite, or has no neighbor among the finite unmasked
    rows of that period is not a state. It is written as `0xffffffff` (`GPU_CLASS_ASSIGNMENT_NO_CLASS`) in
    `quadrants`, and every transition with such an end is counted in `ignored`, never as state 0. Each
    period excludes its invalid rows from the mean and the neighbor sums (`islands: 'exclude'`).
    giddy has no missing-data handling; parity is on the complete rows. `quadrants` (`rowCount * periodCount`)
    is optional.

Both emit one node chain per period, which suits periods in the tens. giddy `relative` rescaling, mean first
passage times and the permutation test `x2_rpvalue` are not built in: divide each period before
classifying.

{/* parameters: GPULISAMarkov generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPULISAMarkov` parameters

- Packer: `getGPUSpatialAutocorrelationParameterValues(settings?)`, writes `GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH` (8) `float32` values.

Settings of `getGPUSpatialAutocorrelationParameterValues`:

| Field | Type | Description |
|---|---|---|
| `significanceLevel` (optional) | `number` | Two-sided significance level of `GPULocalMoran` quadrants, in `(0, 1)`, applied to the adjusted p-value when `multipleTesting` is set. Defaults to `0.05`. `GPUHotSpotAnalysis` ignores it and always reports 90/95/99% bins. |
| `fixedMoments` (optional) | `GPUSpatialAutocorrelationFixedMoments` | Optional moments that replace the moments of the current rows. |

{/* /parameters */}

#### How it computes

{/* lens:GPUSpatialMarkov */}

::lens{scenario=gpu-spatial-analysis.spatial-markov rung=datum datum=0 caption="Cell 0: lag class 0, class 0 staying in class 0 a year later."}

::lens{scenario=gpu-spatial-analysis.spatial-markov rung=formula caption="Each conditional probability divides a count by the transitions out of its lag class and origin class."}

::lens{scenario=gpu-spatial-analysis.spatial-markov rung=kernel caption="Inspect the recorded shader and its input and output bindings."}

::lens{scenario=gpu-spatial-analysis.spatial-markov rung=graph caption="Follow the recorded dependencies from uploaded inputs to captured outputs."}

#### How it computes

{/* lens:GPULISAMarkov */}

::lens{scenario=gpu-spatial-analysis.lisa-markov rung=datum datum=6 caption="Cell 6: a county in the high-high quadrant that is still high-high a year later."}

::lens{scenario=gpu-spatial-analysis.lisa-markov rung=formula caption="Probabilities divide state-to-state counts by the transitions leaving each quadrant."}

::lens{scenario=gpu-spatial-analysis.lisa-markov rung=kernel caption="Inspect the recorded shader and its input and output bindings."}

::lens{scenario=gpu-spatial-analysis.lisa-markov rung=graph caption="Follow the recorded dependencies from uploaded inputs to captured outputs."}

### `GPUKnoxTest` and `GPUMantelTest`

**`GPUKnoxTest`**: Counts observation pairs within 250 m and 7 days of each other and compares the count with 199 time permutations.

::lens{scenario=gpu-spatial-analysis.knox-test rung=result caption="2,344 observations; the strip compares the observed pair count (row 0) with 199 permuted counts."}

**`GPUMantelTest`**: Correlates pairwise distance with pairwise day difference over 293 sampled observations and compares r with 199 permutations.

::lens{scenario=gpu-spatial-analysis.mantel-test rung=result caption="293 sampled observations (every 8th); row 0 of the strip is the observed correlation r."}

Space-time interaction tests over the pairs of a self-join `GPUSpatialWeights` (typically radius-mode
`GPUNeighborSearch`): every unordered pair listed in either direction is a spatial pair and is counted
exactly once (a symmetric list takes it from the lower row, a pair listed only from the higher row, as in a
kNN list, is taken from that row; duplicate and self entries are ignored; lists need not be sorted). Results
are only as complete as the pair list; the capacity overflow flag lives on `GPUNeighborSearch`.
Pairs carry their own `spatialContext`; Knox needs no context check because it only counts spatial pairs.
Mantel needs no context check because its correlation is unit-free and it takes no spatial distance parameter.

- `GPUKnoxTest({pairs, times, parameters, maximumPermutations, output: {statistics, summary}})` counts the pairs
  that are close in space and have `|t_i - t_j| <= timeThreshold`, then runs a Monte Carlo (modified Knox)
  test over `P` permuted time assignments. Counts are exact integers, and the expected value comes
  from the exact count of time-close pairs. `output.statistics` is `uint32` with `P + 1` entries. Knox's
  hypothesis is one-sided (an excess of close pairs), so the Knox p-value is `pseudoPGreater`.
  `getInferenceMetadata(settings)` takes the same `GPUSpaceTimeSettings` that are packed into
  `parameters` and returns the CPU-side `GPUInferenceMetadata` (alternative `'greater'`, Monte Carlo,
  the clamped permutation count and the seed, no multiple-testing correction). `getKnoxPoissonPValue(observed, expected)` gives the classic Poisson
  upper tail, but it assumes Poisson variance, ignores ties and ignores the Mantel-Kulldorff variance
  inflation, so it is anti-conservative; prefer the Monte Carlo p-value.
- `GPUMantelTest({pairs, ...})` (the weights need `distances`) is the Pearson correlation between pair
  distances and `|t_i - t_j|`, with permuted times. Sums are f32 shifted partials, within about 1e-4 of a
  double-precision oracle. Above about 43,000 permutations the rows are split into fewer than 256 blocks so
  the `(permutation, block)` partials table stays within one storage binding (`maxStorageBufferBindingSize`). The statistic runs over the listed pairs only, so with a distance band it is a
  band-truncated Mantel `r` (pairs beyond the band are absent) that differs from the full Mantel `r`;
  report the band with the result. `getInferenceMetadata(settings)` states alternative `'greater'`: the interaction
  hypothesis is a positive space-time correlation, the upper-tail convention of pointpats `mantel`
  (`pseudoPTwoSided` is also in the summary if repulsion matters).

`timeThreshold`, `seed` and `permutationCount` are per-frame settings (`GPUSpaceTimeSettings`), written with
`getGPUSpaceTimeParameterValues({seed, permutationCount, timeThreshold})` (`uint32[4]`). `times` must be
finite and are float32, so raw epoch seconds (about 1.7e9) are spaced 128 s apart and epoch milliseconds
are unusable. Subtract an origin on the CPU with `getGPUSpaceTimeRelativeTimes(times, origin?,
requiredResolution?)`, which subtracts a float64 origin (default the minimum), returns
`{times, origin, resolution}` and throws if the span is too coarse for `requiredResolution`; the float32
spacing at a span `S` is `2^(floor(log2 S) - 23)` (about 0.5 ms for an hour, 1 s for 4.6 months of
seconds). `timeThreshold` is in the same relative units. `output.summary` (`float32`, 12 entries, `GPU_SPACE_TIME_SUMMARY`) holds the observed statistic, pair
count, time-close pair count, expected value, permutation mean and variance, greater and lesser counts,
the pseudo p-values (greater, lesser, two-sided, `(1 + count) / (P + 1)`) and the z-score. The Kulldorff
space-time scan is `GPUSpatialScanStatistic` with `timeBuckets > 1`.

Both tests permute the times with the keyed swap-or-not shuffle of `GPUGlobalPermutationTest`, which is
uniform on small domains (permutation 0 is the identity, permutation `k` is slot `k`). Permuted counts and
p-values for a given seed differ from earlier versions, which used a Feistel bijection that was measurably
non-uniform on small domains. At most `2^24 - 1` events are supported.

**Sequential stopping.** The compile-time `stopAfterExceedances` prop (an integer `h` in `[1,
maximumPermutations]`, default omitted) enables Besag-Clifford sequential Monte Carlo p-values
(Besag and Clifford 1991) for both tests, with the greater alternative that `pseudoPGreater` reports. The
run stops at the first permutation `L` whose running exceedance count (permuted statistic `>=` observed)
equals `h`, and `pseudoPGreater = min(1, h / L)`; with fewer than `h` exceedances in all `P` it is the usual
`(count + 1) / (P + 1)`. Omitted, nothing changes. Permutations run in batches of 64 slots (the batch doubles
when `maximumPermutations` exceeds 16,383, so there are at most 256 batches; batch 0 also computes the
observed statistic). After each batch a one-thread node scans its statistics in permutation order and
disarms the remaining batches through a GPU dispatch gate, so `L` and the p-value do not depend on the batch
size, and the first `L` permutations are those of the full run (same keyed shuffle stream). Work saved is
the batches after the one holding `L`, and the batches past the per-frame `permutationCount`; the stopping
batch finishes whole, so expect savings in steps of one batch, and each batch is a few more graph nodes
(`batch-<n>-partials`, `batch-<n>-reduce`, `batch-<n>-scan`, between `stop-reset` and `stop-finalize`).
`output.permutationCount` (one `uint32`, required with `stopAfterExceedances` and rejected without it)
receives `L`, or `P` when the rule did not fire. `output.statistics` rows past `L` are 0, and the summary
permutation mean, variance, greater and lesser counts and z-score cover the first `L` permutations.
`pseudoPLesser` and `pseudoPTwoSided` keep the fixed-length formulas over those `L` permutations and are not
sequentially valid; use `pseudoPGreater`. The `parameters` layout is unchanged, and `h` is baked into the kernels, so a different `h`
is a rebuild.

{/* parameters: GPUKnoxTest generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUKnoxTest` parameters

- Packer: `getGPUSpaceTimeParameterValues(settings)`, writes `GPU_SPACE_TIME_PARAMETER_LENGTH` (4) `uint32` values.

Settings of `getGPUSpaceTimeParameterValues`:

| Field | Type | Description |
|---|---|---|
| `seed` | `number` | Random seed, an integer in `[0, 2^53)`. The result is a pure function of the seed and inputs. |
| `permutationCount` | `number` | Number of time permutations `P`, clamped to the contributor's `maximumPermutations`. |
| `timeThreshold` (optional) | `number` | Knox only: two rows are close in time when `\|t_i - t_j\| <= timeThreshold` (f32), in the same units as the relative `times` (see `getGPUSpaceTimeRelativeTimes`). Ignored by `GPUMantelTest`. |

{/* /parameters */}

{/* parameters: GPUMantelTest generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUMantelTest` parameters

- Packer: `getGPUSpaceTimeParameterValues(settings)`, writes `GPU_SPACE_TIME_PARAMETER_LENGTH` (4) `uint32` values.

Settings of `getGPUSpaceTimeParameterValues`:

| Field | Type | Description |
|---|---|---|
| `seed` | `number` | Random seed, an integer in `[0, 2^53)`. The result is a pure function of the seed and inputs. |
| `permutationCount` | `number` | Number of time permutations `P`, clamped to the contributor's `maximumPermutations`. |
| `timeThreshold` (optional) | `number` | Knox only: two rows are close in time when `\|t_i - t_j\| <= timeThreshold` (f32), in the same units as the relative `times` (see `getGPUSpaceTimeRelativeTimes`). Ignored by `GPUMantelTest`. |

{/* /parameters */}

#### How it computes

{/* lens:GPUKnoxTest */}

::lens{scenario=gpu-spatial-analysis.knox-test rung=datum datum=0 caption="Row 0: the observed Knox count, before any permutation of days."}

::lens{scenario=gpu-spatial-analysis.knox-test rung=formula caption="The p-value is one plus the permutations at or above the observed count, over P plus one."}

::lens{scenario=gpu-spatial-analysis.knox-test rung=kernel caption="Inspect the recorded shader and its input and output bindings."}

::lens{scenario=gpu-spatial-analysis.knox-test rung=graph caption="Follow the recorded dependencies from uploaded inputs to captured outputs."}

#### How it computes

{/* lens:GPUMantelTest */}

::lens{scenario=gpu-spatial-analysis.mantel-test rung=datum datum=0 caption="Row 0: the observed correlation r, before any permutation of days."}

::lens{scenario=gpu-spatial-analysis.mantel-test rung=formula caption="The correlation runs over all listed pairs; p counts permutations at or above the observed r."}

::lens{scenario=gpu-spatial-analysis.mantel-test rung=kernel caption="Inspect the recorded shader and its input and output bindings."}

::lens{scenario=gpu-spatial-analysis.mantel-test rung=graph caption="Follow the recorded dependencies from uploaded inputs to captured outputs."}

### `GPUSpatialScanStatistic`

Finds circular windows of counties with more traffic deaths than their person-years imply, with Monte Carlo p-values.

::lens{scenario=gpu-spatial-analysis.spatial-scan-statistic rung=result caption="437 counties coloured by the best window log-likelihood ratio centred on each; 199 replicates, seed 1."}

Kulldorff's scan statistic (SaTScan, CARTO `DETECT_SPACETIME_ANOMALIES`) over zones, which are cells or
points, with `cases` (`uint32`) per zone and optional time bucket (`zone * timeBuckets + bucket`). The
compile-time `model` selects the null:

- `'poisson'` (default): needs `baseline` (population or expected counts); `E[cases] = baseline * C /
  sum(baseline)`. LLR `c ln(c/e) + (C - c) ln((C - c)/(C - e))`, evaluated per cell as
  `e ((1 + d) ln(1 + d) - d)` with `d = (o - e) / e` (a series for small `d`; the linear parts cancel
  exactly), so windows with `c` close to `e` keep f32 precision.
- `'bernoulli'`: needs `controls` (`uint32`, same layout); the population of a cell is `cases + controls`.
  LLR of Kulldorff (1997), computed as the sum of `o ln(o/e)` over the inside/outside x case/control
  table. Monte Carlo permutes the case labels over the fixed population (conditioning on the total
  cases), by drawing individuals without replacement. Window populations are f32 (exact below 2^24).
- `'space-time-permutation'` (Kulldorff et al. 2005): no baseline or population. The expectation of a
  cylinder is `(sum of its zones' totals over all buckets) * (sum of its buckets' totals over all zones) /
  C`, the Poisson LLR is applied to it, and Monte Carlo permutes the case times over the case locations
  (both margins fixed). A trend or seasonal cycle shared by every zone is therefore not a cluster, which
  the Poisson model with a static population would report as an "emerging cluster". Needs
  `timeBuckets > 1` to be meaningful.

`spatialContext` is optional and defaults to native planar coordinates. Longitude/latitude positions
require metric `'great-circle'`: windows are ranked by great-circle distance using the shared haversine
separation key, rather than distances in degrees. Cluster `radius` uses the context's length unit
(meters by default for great-circle; kilometers with `units: 'kilometers'`), with `sphereRadius` or the
mean Earth radius in that unit. This operator measures distance, so `local-lambert-equal-area` frames throw.

Windows and defaults:

- Spatial: the `k` nearest zones of every center (the `'circle'` shape never splits equidistant zones;
  `'nearest'` is plain k-nearest), kept while they hold at most `maximumPopulationFraction` of the total
  population (default 0.5, as in SaTScan). The population share governs: the per-center neighbor
  tables hold up to the compile-time `maximumWindowZones` zones, which defaults to every zone while
  `zones^2 <= 2^24` (4096 zones) and otherwise to the largest capacity that fits
  (`GPU_SPATIAL_SCAN_STATISTIC_MAXIMUM_NEIGHBOR_WORDS`; use `getGPUSpatialScanStatisticLayout` to see it). Each center's
  table is a radix select (when the capacity is below the zone count) and an LSD radix sort of its zones by
  (separation key, zone index), with squared distance for planar and the haversine term for great-circle,
  about `zones * (5 * zones + 12 * capacity)` operations per frame, so the
  neighbor step grows with `zones^2`, not `zones^2 * capacity`.
  A window the capacity cut short while the population fraction would still allow it is never silent:
  `output.summary` word `overflow` counts the affected centers and the cluster record carries
  `truncatedByZoneCap` (0 or 1). The per-frame `maximumWindowZones` may lower the capacity.
- Temporal: with `timeBuckets > 1` each window is a cylinder over every run of up to
  `maximumTimeBuckets` buckets. The default is 50% of the study period, `max(1, floor(timeBuckets / 2))`
  (SaTScan's default), no longer all buckets: with a global trend, pure-spatial windows over the whole
  period would otherwise dominate.
- Direction: per-frame `rateDirection` `'high'` (default, `c > e`), `'low'` (`c < e`) or `'both'`.
  Windows of the other direction score 0.
- Defaults: `maximumPopulationFraction` 0.5, `maximumWindowZones` every zone (bounded), `maximumTimeBuckets`
  50% of the study period, `windowShape` `'circle'`, `rateDirection` `'high'`, `maximumClusters` 8.

Outputs: the most likely cluster plus greedy, non-overlapping secondary clusters (no shared zone, time
ignored, same maximum distribution, as SaTScan), all in the caller-owned `output` bag:
`output.clusterIndices` (`GPU_SPATIAL_SCAN_STATISTIC_CLUSTER_INDEX`) and `output.clusterStatistics`
(`GPU_SPATIAL_SCAN_STATISTIC_CLUSTER`: LLR, observed, expected, p, radius, ratio,
`truncatedByZoneCap`); `output.statistics` (the observed maximum then the maximum of every Monte Carlo
replicate), `output.summary` (`GPU_SPATIAL_SCAN_STATISTIC_SUMMARY`: cluster count, total cases,
`permutationCount`, `overflow`) and an optional per-zone best LLR `output.zoneStatistics`.

Inference: `p = (1 + #{max_rep >= LLR}) / (P + 1)`, a Monte Carlo maximum over every window, so it already
accounts for the search; no further correction applies. `getInferenceMetadata(settings)` takes the same
`GPUSpatialScanStatisticSettings` that are packed into `parameters` and returns the CPU-side
`GPUInferenceMetadata` (alternative `'greater'` for high rates, `'lesser'` for low and `'two-sided'` for
both, the permutation count clamped to `maximumPermutations` as the kernels do, the seed,
`multipleTesting: 'none'`, the tested-window definition).
Poisson replicates are multinomial draws from an integer cumulative table (Philox streams of the
permutation contributors); the Bernoulli and space-time permutation draws run one invocation per
replicate (about `C` sequential draws, times `timeBuckets` for the latter). The result is deterministic
(fixed-order reductions, integer atomics, ties to the lowest index). Limits, the seed and the replicate
count are per-frame settings (`getGPUSpatialScanStatisticParameterValues(settings)`), up to the compile-time
`maximumPermutations` (at most 99999, SaTScan-style runs).

Replicate batches: the observed data is evaluated once, then the replicates run in `batchCount` batches
of `batchReplicates` (`R`). A batch draws replicates `b * R + 1 .. b * R + R` into an `R`-slot count
table, evaluates them and folds each replicate's largest LLR into its row of `output.statistics`, which
is the running maximum, so no per-replicate table is kept. `R` is the largest count with
`(R + 1) * zones * timeBuckets <= 2^25` words (the 128 MiB binding limit), at most 4096 and at most
`maximumPermutations`; both are readable properties of the contributor. Philox streams are keyed by the
global replicate index and every reduction is a fixed-order sum or an order-independent maximum, so the
results do not depend on `R` and equal the former single pass bit for bit. A batch whose first replicate
is past the per-frame `permutationCount` is disarmed on the GPU (indirect dispatch gate) and launches
nothing. Each unrolled batch adds four graph nodes (`batch-control`, `batch-clear`, the draw kernel and
`batch-evaluate`) and no pipelines; `maximumPermutations` may need at most 1024 batches, which is only
binding for very large `zones * timeBuckets` (above about 340,000 cells, 99999 replicates need `R >= 98`).

Limits: ordinal and normal models, elliptic windows and Gumbel p-values are not provided. The count
table needs `(R + 1) * zones * timeBuckets <= 2^25` words (so `zones * timeBuckets <= 2^24`), and the
cost is about `(P + 1) * zones * windowZones * timeBuckets^2`. The two neighbor tables have two sort scratch tables of
the same size beside them while they are built. The absolute LLR error is about `|c - e|` times the relative
f32 rounding of `e`; `baseline` must be finite and non-negative. Cluster rounds after the last cluster are
GPU-gated and launch nothing. No kernel binds more than 8 storage buffers, so the scan runs on `'core'`
WebGPU devices.

{/* parameters: GPUSpatialScanStatistic generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUSpatialScanStatistic` parameters

- Packer: `getGPUSpatialScanStatisticParameterValues(settings)`, writes `GPU_SPATIAL_SCAN_STATISTIC_PARAMETER_LENGTH` (8) `uint32` values.

Settings of `getGPUSpatialScanStatisticParameterValues`:

| Field | Type | Description |
|---|---|---|
| `seed` | `number` | Random seed, an integer in `[0, 2^53)`. The result is a pure function of seed and inputs. |
| `permutationCount` | `number` | Monte Carlo replicates `P`, at most the contributor's `maximumPermutations`. |
| `maximumPopulationFraction` (optional) | `number` | Largest spatial window as a fraction of the total baseline, in `(0, 1]`. A window grows until adding the next zone would exceed it. Defaults to 0.5. |
| `maximumWindowZones` (optional) | `number` | Largest number of zones per window, clamped to the contributor's compile-time capacity. Defaults to the capacity (every zone unless the neighbor tables are bounded), so the population fraction governs. |
| `maximumTimeBuckets` (optional) | `number` | Longest cylinder in time buckets, clamped to `timeBuckets`. Defaults to 50% of the study period, `max(1, floor(timeBuckets / 2))`, as in SaTScan. |
| `windowShape` (optional) | `GPUSpatialScanWindowShape` | Window shape. Defaults to `'circle'`. |
| `rateDirection` (optional) | `GPUSpatialScanRateDirection` | Which anomalies count as clusters. Defaults to `'high'`. |

{/* /parameters */}

#### How it computes

{/* lens:GPUSpatialScanStatistic */}

::lens{scenario=gpu-spatial-analysis.spatial-scan-statistic rung=datum datum=15 caption="Cook County, Illinois (row 15 of the county table): the best window centred there."}

::lens{scenario=gpu-spatial-analysis.spatial-scan-statistic rung=formula caption="Poisson log-likelihood ratio of a window; p counts replicate maxima at or above it."}

::lens{scenario=gpu-spatial-analysis.spatial-scan-statistic rung=kernel caption="Inspect the recorded shader and its input and output bindings."}

::lens{scenario=gpu-spatial-analysis.spatial-scan-statistic rung=graph caption="Follow the recorded dependencies from uploaded inputs to captured outputs."}
