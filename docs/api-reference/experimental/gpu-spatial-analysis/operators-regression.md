---
title: "GPU Spatial Analysis: Regression"
sidebar_label: "Regression"
description: "Regression contributors in @luma.gl/experimental/gpu-spatial-analysis."
---

import {ExperimentalDocsTabs} from '@site/src/components/docs/experimental-docs-tabs';

# Regression

<ExperimentalDocsTabs active="gpu-spatial-analysis-operators" />

### `GPUOrdinaryLeastSquares`

Fits a linear model of one response on predictor columns by least squares and reports coefficients, standard errors, t statistics and diagnostics.

::lens{scenario=gpu-spatial-analysis.ordinary-least-squares rung=result caption="Diabetes on three predictors across 788 tracts, R squared 0.67; compare each estimate with its standard error."}

Ordinary least squares with an intercept and a full regression report, in one prebuilt graph:
coefficients, standard errors, t statistics, R2, adjusted R2, sigma squared, log-likelihood, AIC,
BIC, optional per-row residual and fitted columns, the Jarque-Bera normality test, the
Koenker-Bassett and Breusch-Pagan heteroskedasticity tests, and collinearity and precision
diagnostics (Belsley condition number, correlation condition number, maximum VIF, precision
warning). `getGPUOrdinaryLeastSquaresModelResult(readback, {predictorNames})` turns the read-back
arrays into the shared `GPURegressionModelResult` (coefficient table with Student t p-values on
`n - p` degrees of freedom, `sigmaSquared = RSS / (n - p)` and `sigmaSquaredMaximumLikelihood = RSS / n`,
R2, log-likelihood, AIC and BIC equal to spreg, conditioning, and the tests `jarqueBera`,
`breuschPagan`, `koenkerBassett`).

```ts
const graph = new GPUCommandGraph(device);
graph.add(
  new GPUOrdinaryLeastSquares({
    predictors, // GraphDataView<'float32'>, row-major row * predictorCount + column
    response, // GraphDataView<'float32'>
    mask, // optional GraphDataView<'uint32'>; zero excludes the row
    parameters: ridgeBuffer.importToGraph(graph), // optional: getGPUOrdinaryLeastSquaresParameterValues({ridgeLambda})
    predictorCount: 3, // compile-time, 1 to 15 (16-wide Cholesky); the intercept is added internally
    output: {coefficients, standardErrors, tStatistics, summary, status: {code: status}, residuals, fitted}
  })
);
```

- Rows with any non-finite predictor or response value, or a zero mask, are excluded.
- `coefficients`, `standardErrors`, `tStatistics`: `predictorCount + 1` floats, intercept first.
- `summary` (22 floats, `GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH`), indexed by
  `GPU_ORDINARY_LEAST_SQUARES_SUMMARY_*`: row count, R2, adjusted R2, sigma squared, log-likelihood,
  AIC, BIC, Jarque-Bera, its p-value, Koenker-Bassett, its p-value, RSS, TSS, skewness, kurtosis,
  ridge lambda, Breusch-Pagan, its p-value, `CONDITION_NUMBER`, `CORRELATION_CONDITION_NUMBER`,
  `MAXIMUM_VARIANCE_INFLATION_FACTOR`, `PRECISION_WARNING`. The slots 9 and 10 used to be called
  `BREUSCH_PAGAN`; they were always the Koenker-Bassett statistic, and `BREUSCH_PAGAN` now names the
  classic test in slots 16 and 17.
- The predictor count is capped at 15 (16 coefficients with the intercept) by the 16-wide
  function-local Cholesky factorization of the solve; a larger design needs a tiled solver.
- `status.code` (uint32): 0 ok, 1 singular or ill-conditioned (a predictor within `1 - 1e-6` of the span
  of the others, or constant), 2 too few rows (`n <= predictorCount + 1`). A failed fit leaves every
  statistic and per-row column NaN (the row count is still written).
- `residuals` and `fitted` are optional per-row columns (NaN for excluded rows).
- Conditioning never changes `status`: a warning is not a failure and the fit is still reported.
  `CONDITION_NUMBER` is the Belsley number (spreg `OLS.mulColli`): `sqrt(lambdaMax / lambdaMin)` of the
  raw `[1, X]` Gram matrix scaled to unit diagonal (rebuilt from the centered moments and means; values
  above about 30 mean harmful collinearity; ridge is not included). `CORRELATION_CONDITION_NUMBER` is
  `lambdaMax / lambdaMin` of the predictor correlation matrix the float32 Cholesky factors (with the
  ridge penalty), and `MAXIMUM_VARIANCE_INFLATION_FACTOR` the largest diagonal of its inverse. Float32
  coefficients lose about `correlationConditionNumber * 6e-8` relative accuracy, so
  `PRECISION_WARNING` is 1 when the correlation condition number exceeds 1e4
  (`GPU_REGRESSION_PRECISION_CONDITION_LIMIT`, about three of seven digits lost). Eigenvalues come from a
  cyclic Jacobi routine in float32, so the condition numbers are indicative for very ill-conditioned
  designs (the GPU specs allow 2e-2 to 1e-1 relative against the float64 oracle and spreg). They match spreg `mulColli` and `diagnostics.vif` (max
  over the predictor columns, computed on the centered design) in the float64 oracle.
- The optional ridge penalty `lambda >= 0` is added to the centered normal matrix of the
  predictors (never the intercept) and changes between frames without rebuilding the graph. Ridge
  standard errors are `sigma2 (X'X + lambda I)^-1` and ignore the shrinkage bias.

Numerics: no float atomics. Per-tile partial sums (one invocation per tile of consecutive rows, in
row order) are merged in tile order with Kahan compensation; means first, then centered
cross-products about those means; the solve runs on the correlation-scaled normal matrix with a
single-invocation Cholesky. The residual sum of squares comes from the residual pass, not from
`Syy - b'Sxy`, so it stays accurate when R2 is close to 1. Results are bitwise reproducible for a
given `tileRowCount` (default `max(64, ceil(rows / 4096))`).

AIC and BIC count `predictorCount + 1` parameters (GeoDa convention); the log-likelihood is
Gaussian with `sigma2 = RSS / n`. Jarque-Bera is `n/6 (S^2 + (K - 3)^2 / 4)` with p-value
`exp(-JB / 2)`. Both heteroskedasticity tests use the auxiliary regression of `e^2` on `[1, X]` with
`R2_aux`, and a chi-square p-value on `predictorCount` degrees of freedom computed by the regularized
upper incomplete gamma with a fixed iteration cap. Koenker-Bassett (studentized, valid without
normality) is `n R2_aux`; the classic Breusch-Pagan (assumes normal errors) is
`0.5 R2_aux (sum e^4 / s2^2 - n) = 0.5 R2_aux n (kurtosis - 1)` with `s2 = RSS / n`, which is spreg
`breusch_pagan` with `z = [1, X]`. Both match spreg 1.9.1 (`breusch_pagan`, `koenker_bassett` with
`z = reg.x`; spreg's default `z` is the squared design). With skewed errors they differ several
times over, and Jarque-Bera rejecting normality is the cue to read Koenker-Bassett instead.
`getChiSquareSurvival(value, degreesOfFreedom)` is the float64 CPU mirror, and
`getStudentTTwoSidedPValue` / `getStudentTSurvival` (float64 regularized incomplete beta, pinned to
`scipy.stats.t.sf`) back the model-result p-values.

{/* parameters: GPUOrdinaryLeastSquares generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUOrdinaryLeastSquares` parameters

- Packer: `getGPUOrdinaryLeastSquaresParameterValues(settings?)`, writes `GPU_ORDINARY_LEAST_SQUARES_PARAMETER_LENGTH` (1) `float32` values.

Settings of `getGPUOrdinaryLeastSquaresParameterValues`:

| Field | Type | Description |
|---|---|---|
| `ridgeLambda` (optional) | `number` | Ridge penalty in squared predictor units, added as `lambda * I` to the centered normal matrix of the predictors (never to the intercept). A value that is not finite and positive means no penalty. Defaults to 0. |

{/* /parameters */}

#### How it computes

{/* lens:GPUOrdinaryLeastSquares */}

::lens{scenario=gpu-spatial-analysis.ordinary-least-squares rung=formula caption="Follow 6 equations: normal equations, intercept, sigma squared, standard errors, t statistics and R squared."}
::lens{scenario=gpu-spatial-analysis.ordinary-least-squares rung=kernel caption="Inspect the tile, merge, solve, residual and finish kernels with their retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.ordinary-least-squares rung=graph caption="Trace the accumulation, solve, residual and report nodes of one fit on 788 tracts."}

### `GPUGeographicallyWeightedRegression`

Fits a separate weighted regression at every location and writes local coefficients, standard errors, t values and fit statistics, choosing the bandwidth by AICc.

::lens{scenario=gpu-spatial-analysis.geographically-weighted-regression rung=result caption="Diabetes beside the local poverty coefficient, 0.002 to 0.34 across 788 tracts; compare where it is strongest."}

Pass optional `spatialContext` to `getGPUGeographicallyWeightedRegressionParameterValues(settings)` to warn on short fixed `bandwidths`; see [Precision of longitude/latitude positions](./concepts.md#precision-of-longitudelatitude-positions).

Geographically weighted regression with AICc bandwidth selection (Fotheringham, Brunsdon and
Charlton 2002; mgwr, ArcGIS GWR). For every included row `i` it fits a weighted least squares
`y ~ 1 + x` over all included rows `j` with weights `K(d_ij / h)` and writes local coefficients,
local standard errors, t values and a significance mask, local R-squared, fitted values, residuals and
the hat-matrix diagonal.

```ts
const parameters = new GPUParameterBuffer(device, {
  id: 'gwr-parameters',
  format: 'float32',
  length: getGPUGeographicallyWeightedRegressionParameterLength(16)
});
graph.add(
  new GPUGeographicallyWeightedRegression({
    positions, // GraphDataView<'float32x2'>, planar (or lon/lat degrees with spatialContext)
    predictors, // row-major row * predictorCount + column
    predictorCount: 2,
    response,
    parameters: parameterBuffer.importToGraph(graph),
    maximumBandwidthCount: 16,
    output: {coefficients, standardErrors, tValues, significant, selectedBandwidth, summary}
  })
);
parameters.write(
  getGPUGeographicallyWeightedRegressionParameterValues(
    {kernel: 'gaussian', bandwidthMode: 'adaptive', bandwidths: [20, 40, 60, 80]},
    16
  )
);
```

- Kernels: `'gaussian'` `exp(-0.5 (d/h)^2)` (unbounded unless `gaussianCutoff` is set) and `'bisquare'`
  `(1 - (d/h)^2)^2` for `d < h`.
- Bandwidths: `'fixed'` (ladder values are distances) or `'adaptive'` (ladder values are neighbour
  counts `k`, `h` is 1.00001 times the distance to the k-th nearest included row, the row itself
  counted as in mgwr; `k` in `[2, maximumNeighborCount]`, at most 128).
- Bandwidth ladder: up to `maximumBandwidthCount` (at most 32) candidates in the parameter buffer.
  Each candidate's AICc, `n ln(RSS/n) + n ln(2 pi) + n (n + tr S) / (n - 2 - tr S)`, is evaluated on
  the GPU and the outputs are written at the selected bandwidth. A ladder of one value is a fixed
  bandwidth. Candidates with a non-positive bandwidth, invalid `k` or `n - 2 - tr S <= 0` score NaN.
- Singular locations (changed): a location whose local system is singular (for example a rare dummy
  that is constant in its neighbourhood) makes only that row NaN (coefficients, standard errors,
  fitted value, residual, hat; `localStatus` 1). It no longer voids the whole candidate. A candidate's
  `n`, RSS, `tr S` and AICc are taken over its non-singular rows, and `bandwidthSingularRowCounts`
  reports the count per candidate. Selection: among the candidates with the fewest singular rows the
  lowest AICc wins (ties to the lowest index), so a bandwidth with singular rows is chosen only when
  every valid candidate has some. This is a deliberate deviation from mgwr, which raises on a singular
  local system.
- Refinement (new, default on; `refinement: false` keeps the fixed ladder, a number from 1 to 4 sets
  the pass count, default 2): after the caller ladder, a GPU kernel writes a new ladder that zooms
  around the best candidate between its nearest evaluated neighbours (it never extrapolates beyond the
  caller's range), the same kernels evaluate it, and the overall best wins. Adaptive ladders hold
  integers: when the integers inside the bracket number at most `maximumBandwidthCount` all of them
  are tried, so the result is the exact integer optimum of the bracket; a wider bracket is sampled at
  equal spacing and the next pass narrows it. Fixed ladders are `maximumBandwidthCount` equally spaced
  distances per pass. Each pass costs up to one more ladder evaluation, so the default is about three
  times the work of the bare ladder; shorten the ladder rather than turning refinement off when cost
  matters. `selectedBandwidth` is `[candidate index, value]`; indices from `maximumBandwidthCount` up
  are refined candidates, and `candidateTable` lists every candidate of every pass. Every pass reads its
  ladder through a GPU pass cursor, so N and N + 1 passes compile the same pipelines. On the
  144-row test scene (adaptive bisquare) mgwr's golden section with its default bounds `[40 + 2 p, n]`
  stops at `k = 47`, while its exhaustive integer AICc optimum, which the refinement finds from a
  five-value ladder, is `k = 26` (and mgwr's golden section with bounds widened to 6..144 agrees).
  The AICc curve in `k` is not unimodal there (local minimum near 30), which is why a bracket that
  fits the ladder is enumerated rather than searched.
- Golden-section search (new, opt-in `bandwidthSearch: 'golden-section'`; the default `'ladder'` is
  unchanged): mgwr's `Sel_BW(...).search('golden_section')` as a fixed-depth loop on the GPU, about
  12 to 20 candidate evaluations instead of a ladder of up to 32 per pass, no read-back. The first two
  `bandwidths` values of the parameters are the bounds `[minimum, maximum]`; a value of 0 or less
  takes mgwr's default for that bound, so `[0, 0]` is mgwr's default search, and a single value is a
  fixed, unselected bandwidth. Defaults are computed on the GPU from the included rows: adaptive
  `[40 + 2 (predictorCount + 1), n]` with `n` capped at `maximumNeighborCount` (mgwr's `n` cannot be
  evaluated beyond the neighbour list; identical for `n <= maximumNeighborCount`), fixed
  `[min_dist / 2, 2 max_dist]` with `min_dist` the smallest positive row-to-row distance and
  `max_dist` the largest pairwise distance (exact without a grid index; with one, the largest
  distance from a row to a corner of the grid extent, an upper bound that costs at most an
  iteration or two). The interior points are mgwr's `a + 0.38197 (c - a)` and `c - 0.38197 (c - a)`,
  rounded half to even to integers for adaptive bandwidths. Iteration 1 evaluates both points, every
  later iteration compares the two scores, shrinks the bracket and evaluates the one new interior
  point with the same candidate kernels (the other is reused). The search stops when the two
  interior AICc values differ by at most `max(1e-6, 4e-6 |AICc|)`, mgwr's `1e-6` with a float32 noise
  floor (the AICc sums `n` float32 terms; `1e-6` alone never fires at AICc values of hundreds), or
  after `maximumSearchIterations` (default 30, at most 64, compile-time). The selected bandwidth is
  the winner of the last comparison, as mgwr returns it; fewest singular rows first, then the lower
  AICc, an invalid score loses to a valid one and two invalid scores stop the search. The iteration
  index is a GPU cursor and the loop is gated by an indirect dispatch that the one-thread iteration
  kernel zeroes on convergence, so N and N + 1 iterations compile the same pipelines and converged
  iterations launch nothing. `candidateTable` then has `maximumSearchIterations + 1` rows in
  evaluation order (rows 0 and 1 are the first interior points, row `i + 1` the point of iteration
  `i`; unreached rows are NaN), `selectedBandwidth[0]` is the table row of the selection,
  `bandwidthScores` and `bandwidthSingularRowCounts` hold its first `maximumBandwidthCount` rows, and
  the optional `searchState` output reports the bracket, iteration count and whether the tolerance
  (not the cap) stopped it (`GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SEARCH_STATE`). `refinement` is
  ignored. Needs `maximumBandwidthCount >= 2` and `maximumNeighborCount >= 2`. Pinned to mgwr 2.2.1
  on the 144-row scene (bisquare and Gaussian, adaptive and fixed): the adaptive `k` is mgwr's exactly,
  fixed bandwidths agree within the AICc-flatness tolerance of the float32 search, and the float64 CPU
  mirror of the state machine reproduces mgwr's evaluated bandwidths in order. mgwr rounds its fixed
  answer to two decimals; this contributor returns the unrounded value.
- Gaussian cutoff (new, `gaussianCutoff` in bandwidth multiples, compile-time, default exact
  unbounded Gaussian): the weight of a row farther than `gaussianCutoff * h` is exactly 0 (fixed and
  adaptive), and the Gaussian kernel then uses the grid index like bisquare, `O(n * neighbours)`
  instead of `O(n^2)` (mgwr has no cutoff). The omitted weight of a row is at most `exp(-c^2 / 2)` of
  the self weight: 1.1e-2 at `c = 3`, 3.4e-4 at 4, 3.7e-6 at 5, 1.5e-8 at 6. For a locally uniform
  density the omitted share of the weight sum is that value and of the weighted second moment
  `(1 + c^2 / 2) exp(-c^2 / 2)` (3.0e-3 at 4, 5.0e-5 at 5, 2.8e-7 at 6), which bounds the relative
  effect on the local sums up to the local conditioning; on the test scene the coefficient error stays
  below 20 times that share. Recommended: 5 (`GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_RECOMMENDED_GAUSSIAN_CUTOFF`),
  where the effect is below the float32 accumulation error of the fits; use 4 when 3e-3 is acceptable
  and 6 only to match the exact Gaussian to float32. The cutoff does not change the bisquare kernel.
- Local inference (new): `standardErrors`, `tValues` and `significant` (all laid out like
  `coefficients`) are mgwr `bse`, `tvalues` and `filter_tvals != 0`. `bse = sqrt(diag(C C') sigma2)`
  with `C C' = (X'WX)^-1 X'W^2 X (X'WX)^-1` (one more weighted Gram matrix in the same neighbourhood
  pass) and `sigma2 = RSS / (n - tr S)` (mgwr `sigma2_v1`). The summary carries `ENP = tr S`,
  `SIGMA_SQUARED`, `LOG_LIKELIHOOD`, the Da Silva-Fotheringham adjusted alpha `alpha * k / tr S`
  (`ADJUSTED_ALPHA`, `k` design columns, `alpha` 0.05 unless the per-frame `significanceLevel` says
  otherwise) and the critical value `t.ppf(1 - adjusted / 2, n - 1)` (`CRITICAL_T`). The Student t
  quantile is inverted on the GPU (continued-fraction incomplete beta with a bisection; the float64
  CPU mirror `getStudentTCriticalValue` is pinned to `scipy.stats.t.ppf` to 1e-9 and the GPU summary is
  checked against mgwr), because the mask needs it there.
  `significant` is 1 where `|t| > CRITICAL_T`. The bandwidth is selected on the observed data and the
  inference is conditional on it, as in mgwr. Pinned to mgwr 2.2.1 (fixed and adaptive, planar and
  spherical); adaptive fits agree to about 5e-4 because mgwr scales the k-th neighbour distance by
  1.0000001 and this contributor by 1.00001.
- `getGPUGeographicallyWeightedRegressionModelResult({summary, selectedBandwidth})` returns the shared
  `GPURegressionModelResult` (`kind: 'geographically-weighted-regression'`, empty `coefficients`,
  `parameterCount = tr S`, `rowCount = n`, `sigmaSquared = RSS / (n - tr S)`, `logLikelihood`,
  `akaikeInformationCriterion` `= -2 llf + 2 (tr S + 1)` and `correctedAkaikeInformationCriterion`
  (mgwr `llf`, `aic`, `aicc`), `rSquared`) plus `effectiveParameterCount`, `singularRowCount`,
  `includedRowCount`, `adjustedAlpha`, `criticalT` and `selectedBandwidth`.
- `spatialContext` (new): omitted is planar. It measures distance, so `local-lambert-equal-area` frames throw. `{coordinateSpace: 'longitude-latitude', metric:
  'great-circle', units: 'meters'}` makes `positions` `(longitude, latitude)` degrees and every
  distance (kernels, fixed bandwidths, the adaptive k-th neighbour) a haversine distance on a sphere of
  `sphereRadius` (default the mean Earth radius 6,371,008.8 m); bandwidths are then metres. Rows with a
  latitude outside [-90, 90] are excluded. The grid index is planar-only and is not built for a
  great-circle context, so every location scans every included row and the 65,536-row scan limit
  applies; the cell walk is not made exact for longitude wrap-around. mgwr `spherical=True` uses a
  6371.0 km radius and kilometres, so pass `sphereRadius: 6371000` and metres to match it.
- Predictors: 1 to 11 (12 design columns; it was 7). No kernel binds more than eight storage buffers,
  because the per-row results (including standard errors) share one row-state buffer; the limit is
  per-thread private memory (`2 P^2 + 6 P` floats in the fit). The 11-predictor shader has not been run
  on every adapter: lower `GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_PREDICTOR_COUNT` if one
  rejects it.
- Outputs: `coefficients` (`row * (predictorCount + 1) + column`, column 0 intercept), optional
  `standardErrors`, `tValues`, `significant`, `localR2`, `fitted`, `residuals`, `hatDiagonal`,
  `localStatus` (0 ok, 1 singular, 2 excluded), `bandwidthScores` (AICc of the caller ladder),
  `bandwidthSingularRowCounts`, `candidateTable`, `selectedBandwidth` (`[index, value]`),
  `localConditionNumber` and `summary` (`GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY`: RSS, `tr S`,
  AICc, global R-squared, n, has-valid-candidate flag, singular rows, sigma2, adjusted alpha, critical
  t, log-likelihood, included rows; 12 floats, it was 6).
- `localConditionNumber` is the mgwr `local_collinearity` local condition number: the singular-value
  ratio of the kernel-weighted, column-normalized design (above 30 flags collinearity). It is pinned to
  mgwr 2.2.1 (bisquare and gaussian, fixed and adaptive) and is f32-accurate. Its eigenvalues come
  from the shared `getSymmetricEigenvalueRangeWGSL` Jacobi helper of the regression solvers.
- Rows with a zero mask or a non-finite position, predictor or response are excluded as
  calibration locations and as neighbours.
- Numerics: the design is centred on the focal row and Jacobi-equilibrated before the shared
  Cholesky solve, so large coordinates or badly scaled predictors stay well conditioned in f32.
  A predictor that is constant over a neighbourhood makes that location singular.
- Cost: one thread per location scanning every row per candidate: `O(n^2 * ladder * p^2)`, at most
  65536 rows without the grid index (see `indexGridSize` below) (about 80 ms for 4096 rows and 8 candidates on the
  test adapter). Reductions use fixed-order 256-row tiles, so results are bitwise reproducible.
- Grid index: `indexGridSize` (default: automatic from 1024 rows; `false` disables) bins rows into a uniform grid so
  each location scans nearby cells only. The row cap rises from 65,536 to 1,048,576 with the grid, and
  `'bisquare'` fixed-bandwidth fits sum in cell order (about 1e-5 relative to the earlier row-order sums, still
  deterministic). Gaussian kernels use the grid only with a `gaussianCutoff`.

{/* parameters: GPUGeographicallyWeightedRegression generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUGeographicallyWeightedRegression` parameters

- Packer: `getGPUGeographicallyWeightedRegressionParameterValues(settings, maximumBandwidthCount?)`, writes `getGPUGeographicallyWeightedRegressionParameterLength(...)` `float32` values.
  - `maximumBandwidthCount` (`number`): Ladder capacity of the contributor (its `maximumBandwidthCount`).

Settings of `getGPUGeographicallyWeightedRegressionParameterValues`:

| Field | Type | Description |
|---|---|---|
| `spatialContext` (optional) | `GPUSpatialContext` | Spatial context of the positions; when it is longitude/latitude, a distance under 10 float32 quanta (about 17 m) logs a warning (see Precision of longitude/latitude positions). Does not change the packed values. |
| `kernel` (optional) | `GPUGeographicallyWeightedRegressionKernel` | Weight kernel. Defaults to `'bisquare'`. |
| `bandwidthMode` (optional) | `GPUGeographicallyWeightedRegressionBandwidthMode` | Meaning of the ladder values. Defaults to `'fixed'`. |
| `bandwidths` | `ArrayLike<number>` | Candidate bandwidths: distances (`'fixed'`) or neighbour counts `k` (`'adaptive'`, rounded to integers). One value is a fixed bandwidth; several are searched by AICc (and refined, see `refinementPassCount`). With `bandwidthSearch: 'golden-section'` the first two values are the search bounds `[minimum, maximum]` (a value of 0 or less takes mgwr's default for that bound); further values are ignored. |
| `significanceLevel` (optional) | `number` | Two-sided significance level of the adjusted alpha and critical t value, in (0, 1). Defaults to 0.05. Per frame. |

{/* /parameters */}

#### How it computes

{/* lens:GPUGeographicallyWeightedRegression */}

::lens{scenario=gpu-spatial-analysis.geographically-weighted-regression rung=datum bind=page datum=300 caption="Inspect tract 300: its local coefficients, t value and local R squared."}
::lens{scenario=gpu-spatial-analysis.geographically-weighted-regression rung=formula caption="Follow the bisquare weight, local fit, hat value, AICc and local t equations."}
::lens{scenario=gpu-spatial-analysis.geographically-weighted-regression rung=kernel caption="Inspect the grid, candidate, golden-section and local-fit kernels with retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.geographically-weighted-regression rung=graph caption="Trace the golden-section search over 10 candidate bandwidths and the final local fits over 788 tracts."}

### `GPUGeographicallyWeightedRegressionNonstationarityTest`

Tests whether each GWR coefficient varies across space more than permuted data would, with a Monte Carlo pseudo p-value per coefficient.

::lens{scenario=gpu-spatial-analysis.geographically-weighted-regression-nonstationarity-test rung=result caption="Observed against permuted spread of four coefficients; poverty pseudo p = 0.03, the other three 0.01."}

Monte Carlo test of coefficient non-stationarity (mgwr `spatial_variability`, GWR4). It permutes the
observations among the locations with the Philox-keyed swap-or-not shuffle of `GPUGlobalPermutationTest`
(the earlier Feistel bijection was non-uniform on small row counts), refits
at the selected bandwidth and compares the standard deviation of each local coefficient with the observed
one. The pseudo p-value is `(g + 1) / (P + 1)`. Outputs are `table` (six floats per coefficient), `summary`
(`GPU_GWR_NONSTATIONARITY_SUMMARY`, `GPU_GWR_NONSTATIONARITY_TABLE`, `GPU_GWR_NONSTATIONARITY_TABLE_STRIDE`)
and an optional `standardDeviations` (the reference distribution). The per-permutation standard deviations
match mgwr 2.2.1 to 3e-15 (CPU oracle) and 2e-3 (GPU). The bandwidth stays at the value the regression
selected (mgwr re-selects it for every permutation), and that value was selected on the observed data, so
the test is conditional on it, as mgwr's is. `selectedBandwidth` is read from slot 1 (the selected value),
so it also works when the regression refined its bandwidth beyond the caller ladder. It is planar-only
(a great-circle `spatialContext` throws, with a hint to project first) and takes 1 to 7 predictors even
though the regression accepts 11. The cost is `O(P * n^2 * p^2)` with a full scan (no grid index).

{/* parameters: GPUGeographicallyWeightedRegressionNonstationarityTest generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUGeographicallyWeightedRegressionNonstationarityTest` parameters

- Packer: `getGPUGeographicallyWeightedRegressionParameterValues(settings, maximumBandwidthCount?)`, writes `getGPUGeographicallyWeightedRegressionParameterLength(...)` `float32` values.
  - `maximumBandwidthCount` (`number`): Ladder capacity of the contributor (its `maximumBandwidthCount`).
- Packer: `getGPUPermutationParameterValues(settings, options?)`, writes `GPU_PERMUTATION_PARAMETER_LENGTH` (4) `uint32` values.
  - `options` (`{maximumPermutations?: number}`)

Settings of `getGPUGeographicallyWeightedRegressionParameterValues`:

| Field | Type | Description |
|---|---|---|
| `spatialContext` (optional) | `GPUSpatialContext` | Spatial context of the positions; when it is longitude/latitude, a distance under 10 float32 quanta (about 17 m) logs a warning (see Precision of longitude/latitude positions). Does not change the packed values. |
| `kernel` (optional) | `GPUGeographicallyWeightedRegressionKernel` | Weight kernel. Defaults to `'bisquare'`. |
| `bandwidthMode` (optional) | `GPUGeographicallyWeightedRegressionBandwidthMode` | Meaning of the ladder values. Defaults to `'fixed'`. |
| `bandwidths` | `ArrayLike<number>` | Candidate bandwidths: distances (`'fixed'`) or neighbour counts `k` (`'adaptive'`, rounded to integers). One value is a fixed bandwidth; several are searched by AICc (and refined, see `refinementPassCount`). With `bandwidthSearch: 'golden-section'` the first two values are the search bounds `[minimum, maximum]` (a value of 0 or less takes mgwr's default for that bound); further values are ignored. |
| `significanceLevel` (optional) | `number` | Two-sided significance level of the adjusted alpha and critical t value, in (0, 1). Defaults to 0.05. Per frame. |

Settings of `getGPUPermutationParameterValues`:

| Field | Type | Description |
|---|---|---|
| `seed` | `number` | Random seed, an integer in `[0, 2^53)`. Results are a pure function of the seed, the inputs and `permutationCount`, independent of the dispatch shape. Per frame. |
| `permutationCount` | `number` | Number of permutations `P`, in `[1, maximumPermutations]` of the contributor. The count is never clamped: a value outside that range makes the kernels write NaN p-values (local: rows are not tested and `output.status.overflow` is set; global: the `permutationCount` field of `output.summary` is NaN). Validate on the CPU with the contributor's `getParameterValues` or the `maximumPermutations` option of `getGPUPermutationParameterValues`. |
| `significanceLevel` (optional) | `number` | Significance level for the local `significant` mask. Defaults to 0.05. Ignored by the global test. |

{/* /parameters */}

#### How it computes

{/* lens:GPUGeographicallyWeightedRegressionNonstationarityTest */}

::lens{scenario=gpu-spatial-analysis.geographically-weighted-regression-nonstationarity-test rung=formula caption="Follow the observed spread, exceedance count, pseudo p-value and permuted z score."}
::lens{scenario=gpu-spatial-analysis.geographically-weighted-regression-nonstationarity-test rung=kernel caption="Inspect the permuted refit, spread merge and finish kernels with retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.geographically-weighted-regression-nonstationarity-test rung=graph caption="Trace the GWR fit, 99 permuted refits and the finish step across 788 tracts."}

### `GPUSpatialRegressionDiagnostics`

Tests an ordinary least squares fit for spatial dependence with LM-lag, LM-error, robust, SARMA and residual Moran's I tests.

::lens{scenario=gpu-spatial-analysis.spatial-regression-diagnostics rung=result caption="Six tests on the diabetes OLS residuals: LM-lag 638 against LM-error 595, then robust 170 against 127."}

Maximum: `rows * (predictorCount + 3)` is at most `getGPUSpatialRegressionDiagnosticsMaximumRowTableValueCount(device)` (2^25 on a default device); the per-tile moment table is at most `getGPUSpatialRegressionDiagnosticsMaximumTileMomentCount(device)` values.

Spatial diagnostics for an OLS fit (spreg `LMtests` and `MoranRes`): LM-lag, LM-error, robust LM-lag,
robust LM-error, LM-SARMA, and Moran's I of the residuals (mean, variance, z, two-sided normal p).

- Props: `weights` (a square `GPUSpatialWeights`; any sparsity pattern is accepted, directed kNN included,
  and row-standardized weights work; `W'` is formed with `GPUSpatialWeightsTranspose`),
  `predictors` (row-major `n x k`, no intercept), `response`, `residuals` (OLS residuals with intercept,
  for example `GPUOrdinaryLeastSquares` `output.residuals`), `predictorCount` (1 to 15), optional `mask`,
  `restandardizeRows`, `tileRowCount` and `output: {tests, summary, status: {code}}`.
- Masked and missing fits: a row is analyzed when its `mask` entry is not zero and its response, residual
  and predictors are finite; pass the OLS `mask` and the NaN residuals of excluded rows do the rest. Excluded
  rows are removed from the whole analysis: `n`, `S0`, `T`, every Gram matrix and every `W`-product run over
  the fitted rows, and links from fitted rows to excluded rows are dropped. That is `libpysal.weights.w_subset`
  on the fitted ids. With `restandardizeRows: true` each fitted row's remaining weights are rescaled to sum
  to 1 (`w_subset` then `transform = 'r'`; use it after masking row-standardized weights); the default
  keeps the surviving weights as given. Rows left with no fitted neighbor keep an all-zero row. Results
  match spreg 1.9.1 `OLS(..., spat_diag=True, moran=True)` on the fitted subset in both modes.
- `tests` (18 f32): six rows of `[statistic, df, p]`: LM_LAG, LM_ERROR, ROBUST_LM_LAG, ROBUST_LM_ERROR
  (df 1), LM_SARMA (df 2) and MORAN_RESIDUALS (the statistic is z, the df slot is 0, p is two-sided). The
  row index constants are `GPU_SPATIAL_REGRESSION_DIAGNOSTICS_TEST_*`.
- `summary` (13 f32): `n` (analyzed rows), `sigma2 = e'e / n`, `T = tr(W'W + WW)`, `J`, `S0`, Moran `I`,
  `E[I]`, `Var[I]`, `z`, `p`, `e'We`, `e'Wy` and `EXCLUDED_ROW_COUNT` (`..._SUMMARY_*`). The excluded
  count is a summary slot, not a status port, because `status.code` is this operator's single result code; it is
  also written when the status is not 0. Before this change a masked fit reported status 4 for the whole
  result.
- `status.code`: 0 ok, 1 singular, 2 too few analyzed rows (`n <= predictorCount + 1`, also a failed OLS fit),
  3 degenerate weights, 4 non-finite moments or exactly zero residuals. Every output is NaN unless the
  status is 0. Conditioning slots are not reported here (see the OLS fit).

Sums are fixed-order tile sums with a Kahan merge on centered columns, and the solve is a unit-diagonal
scaled Cholesky, so there are no atomics and no readback. Moran's variance and `J` are f32 with
cancellation (about 1e-2 relative on the variance and z). The statistics match spreg 1.9.1 (`OLS` with
`spat_diag` and `moran`) on a lattice, a symmetrized kNN scene and two directed row-standardized kNN scenes
(`k = 3`, 60 and 90 rows, asymmetric patterns) to 2e-3 relative. Earlier versions accumulated `W'Z` over each
row's own neighbors only, which biased the Moran variance for asymmetric patterns; that is fixed.

```ts
graph.add(new GPUOrdinaryLeastSquares({predictors, response, mask, predictorCount: 2, output: {
  coefficients, standardErrors, tStatistics, summary: olsSummary, residuals, status: {code: olsCode}
}}));
graph.add(new GPUSpatialRegressionDiagnostics({
  weights, predictors, response, residuals, mask, restandardizeRows: true, predictorCount: 2,
  output: {tests, summary, status: {code}}
}));
```

#### How it computes

{/* lens:GPUSpatialRegressionDiagnostics */}

::lens{scenario=gpu-spatial-analysis.spatial-regression-diagnostics rung=formula caption="Follow the LM-error, LM-lag, robust, SARMA and Moran's I equations."}
::lens{scenario=gpu-spatial-analysis.spatial-regression-diagnostics rung=kernel caption="Inspect the tile, merge and finish kernels with retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.spatial-regression-diagnostics rung=graph caption="Trace queen weights, the OLS fit and the six diagnostics over 788 tracts."}

### `GPUSpatialTwoStageLeastSquares`

Estimates a spatial lag model by two-stage least squares, instrumenting the lagged response with neighbouring predictors, and tests the residuals.

::lens{scenario=gpu-spatial-analysis.spatial-two-stage-least-squares rung=result caption="Five rows: intercept, three predictors and rho = 0.55 (se 0.027); compare each with its standard error."}

The spatial lag model `y = rho W y + X b + e` by two-stage least squares (spreg `GM_Lag` with `w_lags = 1`),
with instruments `[1, X, WX]`. Props: `weights`, `predictors`, `response`, `predictorCount` (1 to 8 with
`instrumentOrder: 1`, 1 to 5 with `instrumentOrder: 2`: the constant is centered out, so the solve factors `(order + 1) k`
instrument columns in a 16-wide Cholesky; the order-2 cap was previously 8 in the check but failed when building the
graph), optional `tileRowCount`, `instrumentOrder` and `output: {table, summary, status: {code}, residuals?}`. No kernel
binds more than 8 storage buffers.
`instrumentOrder: 2` (compile-time, default 1) adds `W^2 X` (spreg `w_lags=2`, instruments `[1, X, WX,
W^2 X]`), evaluated per row from the two-hop neighborhood with no extra buffers; it is pinned to spreg
1.9.1 `GM_Lag(w_lags=2)` on three scenes.

- `table`: `(k + 2) x 4` f32 rows of `[coefficient, se, z, two-sided normal p]` in spreg order: the
  intercept, then `X`, then `rho` last.
- `summary` (16, `GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_*`): `n`, `sigma2 = u'u / n` (spreg `sig2n`, the `GM_Lag`
  default), `u'u`, the pseudo R2 (squared correlation of `y` and `y - u`), Moran's `I` of `u`, the
  Anselin-Kelejian statistic and its p-value (slots 0 to 6). The statistic follows spreg `akTest(case='gen')`; any
  sparsity pattern is accepted (`W'u` comes from `GPUSpatialWeightsTranspose`), and it matches spreg 1.9.1 on
  directed kNN scenes. Diagnostics (slots 7 to 15; none changes `status`):
  - Conditioning of the second stage: the Belsley condition number of the raw `[1, PZ]` Gram matrix (`Z = [X, Wy]`,
    `P` the projection on the instruments), the correlation condition number of `Z'PZ` and its largest variance
    inflation factor, the correlation condition number of the instrument system `H'H` (the instrument-projection
    system), and a precision warning (1.0) when either correlation condition number exceeds 1e4 (float32
    normal equations may have lost three of seven digits). Computed by the Jacobi eigenvalue routine of
    `spatial-regression-solve`.
  - First-stage strength: the classical joint F statistic and the partial R2 of the excluded instruments (`WX`, and
    `W^2 X` for order 2) in the regression of `Wy` on `[1, X, WX, ...]`; `F = ((ESS_full - ESS_restricted) / (m - k)) /
    (RSS_full / (n - m - 1))`, partial R2 `= (ESS_full - ESS_restricted) / (TSS - ESS_restricted)`. The usual
    weak-instrument rule of thumb is `F < 10`. Both are the non-robust (homoskedastic) forms and are pinned to a numpy
    computation of the same definitions.
  - Admissibility of `rho`: `||W||_inf` (largest absolute row sum) and a flag that is 1.0 when `|rho| ||W||_inf >= 1`,
    that is when `I - rho W` is not guaranteed invertible and the reduced form and impacts are unreliable
    (`(-1, 1)` for a row-standardized `W`; sufficient, not necessary, for other `W`). It is a flag, not a failure:
    2SLS is unconstrained, the estimate is still reported, and the reader decides what a boundary or explosive `rho`
    means for the model.
- `status.code`: 0 ok, 1 singular or ill-conditioned (including non-finite inputs), 2 too few rows, 3 non-finite
  sums.
- `residuals`: the structural residuals `u = y - Z delta`, which can be passed to later diagnostics.

The coefficients, standard errors, `sigma2`, pseudo R2 and Anselin-Kelejian statistic match spreg on a
144-row scene within 5e-3 (coefficients) to 2e-2 (z, AK). The standard errors are the classical
`sigma2 (Z'PZ)^-1`: heteroskedasticity-robust and HAC standard errors are not provided (spreg `robust`),
nor are `GM_Combo` and maximum-likelihood models. For the spatial error model see
[`GPUSpatialErrorGM`](#gpuspatialerrorgm).

`getGPUSpatialTwoStageLeastSquaresModelResult(readback, {predictorNames?})` turns the read-back `table`, `summary`
and `status` into the shared `GPURegressionModelResult`: rows `CONSTANT`, the predictors and `W_y` (role
`'spatial-lag'`), z statistics with float64 normal p-values, `sigmaSquaredMaximumLikelihood = u'u / n` (the scale
of the standard errors, `standardErrorSigmaSquared: 'maximum-likelihood'`), `pseudoRSquared`, no likelihood,
`tests.anselinKelejian` and conditioning (the correlation condition number is the larger of the second-stage
and instrument values, so `illConditioned` equals the precision warning).
`getGPUSpatialTwoStageLeastSquaresDiagnostics(summary)` returns the first-stage F, partial R2, `||W||_inf` and the
`rho` admissibility flag.

#### How it computes

{/* lens:GPUSpatialTwoStageLeastSquares */}

::lens{scenario=gpu-spatial-analysis.spatial-two-stage-least-squares rung=formula caption="Follow the instrumented estimator, residual variance, z statistics and Anselin-Kelejian test."}
::lens{scenario=gpu-spatial-analysis.spatial-two-stage-least-squares rung=kernel caption="Inspect the moment, solve, residual and finish kernels with retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.spatial-two-stage-least-squares rung=graph caption="Trace queen weights, the two stages and the Anselin-Kelejian test over 788 tracts."}

### `GPUSpatialWeightsTraces`

Computes the traces and sums of powers of a weights matrix, the inputs of spatial log-determinants and impacts.

::lens{scenario=gpu-spatial-analysis.spatial-weights-traces rung=result caption="Traces of 12 queen-weight powers fall from 122 at q = 2 to 19; sums stay at 788."}

Powers of a spatial weights matrix: `powerTraces[q-1] = tr((W/s)^q)` and `powerSums[q-1] = 1' (W/s)^q 1` with `s = ||W||_inf` (largest absolute row sum, written to `output.infinityNorm`; series then run in `rho * s`, which keeps the float32 powers bounded for raw weights) for
`q = 1..Q`. These are the inputs of the spatial-lag log-determinant `ln|I - rho W| = -sum_q rho^q tr(W^q) / q`
and of the impact multipliers (see `GPUSpatialLagImpacts`). Props: `weights` (any square CSR: asymmetric,
not standardized, self-loops allowed), `maximumPower` (`Q`, compile-time, 1 to 256), optional `mode`,
`probeCount`, `parameters` (per-frame `[seed]`, `getGPUSpatialWeightsTracesParameterValues({seed})`; omitted means seed 0),
`includeGramTraces`, `tileRowCount`, `probeBatchSize` and
`output: {powerTraces, powerSums, gramTraces?, status?: {approximation?}, infinityNorm?}` (float32 views of at least `Q`, `Q`, 4 values
and a uint32 scalar).

- `mode: 'exact'` runs one unit-vector chain `v_q = W v_{q-1}` per row and sums the diagonal entries
  `v_q[i]`: exact up to float32 rounding. It is the default for at most 4096 rows
  (`GPU_SPATIAL_WEIGHTS_TRACES_MAXIMUM_EXACT_ROW_COUNT`): the work is `rows * Q * nnz`, which at 4096 rows,
  `Q = 256` and 8 neighbors is about 3e10 multiply-adds in about 520 dispatches, while 16384 rows (the limit
  of an explicit `'exact'`) already costs 8200 dispatches and 5e11 multiply-adds.
- `mode: 'stochastic'` is Hutchinson estimation with `probeCount` (default 64) Rademacher probes from Philox
  (counter `(probe, row / 4, 0, 'PROB')`, key `(seed, 'TRAC')`, the seed read on the GPU, so changing it never
  recompiles; a seeded run is reproducible and
  `getSpatialWeightsTracesOnCPU` replays the same probes). The standard error of `tr(A)` is at most
  `sqrt(2 / probeCount) ||A||_F`. `output.status.approximation` is 1 (it is 0 in exact mode).
- `powerSums` is always exact: one chain from the ones vector.
- `gramTraces` (when `includeGramTraces`, 4 values) are `[tr(W'W), tr(W'W W), tr((W'W)^2), sum_i c_i^2]` with
  `c_i = sum_l w_li^2`: the moments of the Kelejian-Prucha and Anselin-Kelejian variance formulas. `tr(W'W)` and
  `sum c_i^2` are exact sums (the column sums use `GPUSpatialWeightsTranspose`); the other two are
  `E[(Wz)'(WWz)]` and `E[||W'(Wz)||^2]` over the same probes, exact in exact mode.

Probes run in batches whose two ping-pong vectors stay within 64 MiB. Every dot product is a fixed-order tile
sum with a Kahan merge, so there are no float atomics and results are bit-identical across batch sizes and
runs. The shaders do not depend on the power, the batch or the probe index: per-power and per-batch tables
are 64-element aligned sub-views and a GPU batch table holds each batch's first probe and probe count, so
the pipeline count does not grow with `maximumPower` or the number of batches. A `W` with spectral radius above 1 grows as `radius^q` and can overflow float32 for large `Q`.

CPU float64 mirrors for specs and later consumers: `getSpatialWeightsTracesOnCPU(weights, Q, options?)`,
`getSpatialLogDeterminantFromTraces(rho, powerTraces)`, `getSpatialMultiplierTraceFromTraces(rho, powerTraces,
n)` (`= tr((I - rho W)^-1) = n + sum_q rho^q tr(W^q)`) and the truncation bounds
`getSpatialLogDeterminantTruncationBound` and `getSpatialMultiplierTruncationBound`. WGSL generators
`getSpatialLogDeterminantWGSL(Q, bindingName)`, `getSpatialMultiplierTraceWGSL(Q, bindingName, n)` and
`getSpatialMultiplierTotalWGSL(Q, bindingName, n)` define `getSpatialLogDeterminant(rho)` and friends by
Horner's rule over an f32 storage binding. The series require `|rho| * spectralRadius(W) < 1` (`|rho| < 1` for
a row-standardized `W`) and are truncated: the log-determinant error is below
`n |rho|^(Q+1) / ((Q+1)(1 - |rho|))` and the multiplier error below `n |rho|^(Q+1) / (1 - |rho|)`, so choose `Q`
from `|rho|^Q` (`Q = 30` is accurate to 1e-9 at `|rho| = 0.5` but needs about 300 terms at `|rho| = 0.9`).

The traces match `np.trace(np.linalg.matrix_power(W, q))` for `q = 1..12` on a row-standardized lattice, a
binary lattice and two directed kNN scenes (numpy 2.5.4), and the log-determinant series matches
`np.linalg.slogdet` within the stated bound.

```ts
graph.add(new GPUSpatialWeightsTraces({
  weights, maximumPower: 30, output: {powerTraces, powerSums, infinityNorm}   // exact for <= 4096 rows
}));
```

{/* parameters: GPUSpatialWeightsTraces generated by modules/experimental/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUSpatialWeightsTraces` parameters

- Packer: `getGPUSpatialWeightsTracesParameterValues(settings?)`, writes `GPU_SPATIAL_WEIGHTS_TRACES_PARAMETER_LENGTH` (1) `uint32` values.

Settings of `getGPUSpatialWeightsTracesParameterValues`:

| Field | Type | Description |
|---|---|---|
| `seed` (optional) | `number` | Philox seed of the stochastic probes, an integer in `[0, 2^32)`. Defaults to 0. |

{/* /parameters */}

#### How it computes

{/* lens:GPUSpatialWeightsTraces */}

::lens{scenario=gpu-spatial-analysis.spatial-weights-traces rung=formula caption="Follow the infinity norm, the power traces and the power sums."}
::lens{scenario=gpu-spatial-analysis.spatial-weights-traces rung=kernel caption="Inspect the norm, scaling, probe-chain and merge kernels with retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.spatial-weights-traces rung=graph caption="Trace queen weights, 788 unit-vector chains and 12 power steps."}

### `GPUSpatialLagImpacts`

Turns spatial lag coefficients and weights traces into average direct, indirect and total effects of each predictor.

::lens{scenario=gpu-spatial-analysis.spatial-lag-impacts rung=result caption="Direct, indirect and total effects of three predictors; poverty's total is 0.42, 61% of it indirect."}

Average direct, indirect and total impacts of the predictors of a spatial lag model (LeSage and Pace 2009):
`direct = b_k tr((I - rho W)^-1) / n`, `total = b_k 1'(I - rho W)^-1 1 / n`, `indirect = total - direct`, from the
truncated series `n + sum_{q<=Q} rho^q tr(W^q)` and `n + sum_{q<=Q} rho^q 1'W^q 1` evaluated in one kernel. Props:
`coefficients` and `coefficientStride` (the `table` of `GPUSpatialTwoStageLeastSquares` with stride 4, or any
fit with a coefficient per row), `rhoIndex` (`predictorCount + 1` for the 2SLS table), optional
`firstPredictorIndex` (default 1, so the intercept is excluded), `predictorCount`, `rowCount`, `maximumPower`,
`traces: {powerTraces, powerSums, infinityNorm}` (the outputs of `GPUSpatialWeightsTraces`), optional `status` (a fit's `output.status.code`) and `rhoBound`
(default 1), and `output: {impacts}`: `predictorCount * 3` f32 values `[direct, indirect, total]` per
predictor.

Impacts are NaN when the fit `status` is nonzero, `rho` is not finite or `|rho| >= rhoBound`. For a `W` that
is not row-standardized set `rhoBound = 1 / spectralRadius`, so the series cannot diverge silently.

The direct impact matches spreg 1.9.1 `GM_Lag(..., spat_impacts='full')`, and so does the total for a
row-standardized `W` (spreg uses `1 / (1 - rho)`, which is exact only there; this node uses the exact
`1'W^q1` sums, so it is also right for other weights). spreg's `'simple'` method fixes the direct multiplier at 1
and is not reproduced; its `'power'` method stops after the first term below tolerance, which is wrong on
bipartite lattices where odd traces vanish. Standard errors of the impacts (simulation of `(rho, b)` from the
coefficient covariance) are not provided: the 2SLS contributor does not output the covariance matrix.

```ts
graph.add(new GPUSpatialTwoStageLeastSquares({
  weights, predictors, response, predictorCount: 2, output: {table, summary, status: {code: status}}
}));
graph.add(new GPUSpatialWeightsTraces({weights, maximumPower: 30, output: {powerTraces, powerSums, infinityNorm}}));
graph.add(new GPUSpatialLagImpacts({
  coefficients: table, coefficientStride: 4, rhoIndex: 3, predictorCount: 2,
  rowCount, maximumPower: 30, traces: {powerTraces, powerSums, infinityNorm}, status, output: {impacts}
}));
```

#### How it computes

{/* lens:GPUSpatialLagImpacts */}

::lens{scenario=gpu-spatial-analysis.spatial-lag-impacts rung=formula caption="Follow the direct, total and indirect effect series in rho."}
::lens{scenario=gpu-spatial-analysis.spatial-lag-impacts rung=kernel caption="Inspect the impacts kernel with its retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.spatial-lag-impacts rung=graph caption="Trace the likelihood fit, its power traces and the impacts over 788 tracts."}

### `GPUSpatialErrorGM`

Estimates a spatial error model by generalized moments, then a feasible GLS regression on spatially filtered variables.

::lens{scenario=gpu-spatial-analysis.spatial-error-gm rung=result caption="Five rows: intercept, three predictors and lambda; lambda = 0.83 with standard error 0.015."}

The spatial error model `y = X b + u`, `u = lambda W u + e` by generalized moments. Props: `weights` (any square
CSR; no symmetry or standardization needed), `predictors`, `response`, `predictorCount` (1 to 15, the OLS limit and
the 16-wide Cholesky; no kernel binds more than 8 storage buffers, so the previous cap of 8 was not needed),
`estimator`, optional `tileRowCount`, `traceMode`, `traceProbeCount`, `traceParameters` (per-frame seed, as `GPUSpatialWeightsTraces` `parameters`) and
`output: {table, summary, status: {code}, residuals?}`.

**Default change.** `estimator` is `'homoskedastic'` by default (spreg 1.9.1 `GM_Error_Hom`, `A1 = 'hom_sc'`,
one iteration: Drukker, Egger and Prucha 2013, Anselin 2011), which gives `lambda` a standard error, z and p-value.
The previous behavior is `estimator: 'kelejian-prucha'` (spreg `GM_Error`, Kelejian and Prucha 1998 and 1999), whose
`lambda` row keeps NaN for se, z and p; in both cases the standard errors of the betas condition on the
estimated `lambda`.

- Steps (both): OLS of `y` on `[1, X]` gives `u`; moments of `u` give `lambda`, minimized globally on a 1024-interval
  grid with bisection of every bracketed minimum (spreg runs a local L-BFGS-B from zero; the two agree whenever the
  objective has a single basin); OLS of `y - lambda W y` on the filtered `[1, X]` (the constant is filtered too, as
  `get_spFilter` does) gives the coefficients.
- `lambda` bound: `0.99 / ||W||_inf` with `||W||_inf` the largest absolute row sum, computed on the GPU. It equals
  spreg's fixed 0.99 for a row-standardized `W` and keeps `I - lambda W` invertible for any `W`; a fixed 0.99
  admitted explosive `lambda` for raw weights. A solution on the bound is flagged
  (`..._SUMMARY_LAMBDA_AT_BOUND`) and `..._SUMMARY_LAMBDA_BOUND` reports the bound.
- `homoskedastic` replicates spreg step for step: (a) `lambda1` minimizes the unweighted
  `|G [l, l^2]' - g|^2` of the moments `u'A1u`, `u'A2u` with `A1 = (W'W - (t/n) I) / (1 + (t/n)^2)`, `t = tr(W'W)`,
  `A2 = (W + W') / 2`; (b) the FGLS betas at `lambda1` are the reported betas and give `u1 = y - X b1`; (c) the
  moments are recomputed on `u1` and `psi`, the variance of the moments (from `sigma2`, `mu3`, `mu4` of
  `u1 - lambda1 W u1` and the traces `tr W`, `tr W^2`, `tr W'W`, `tr W'WW`, `tr (W'W)^2`, `sum_i c_i^2` of
  [`GPUSpatialWeightsTraces`](#gpuspatialweightstraces)), weights the second minimization
  `(G p - g)' psi^-1 (G p - g)` giving `lambda`; (d) `Var(b) = sigma2 (Xs'Xs)^-1` with `Xs` and `u_s` filtered at
  `lambda` (`sigma2 = u_s'u_s / n`) and `Var(lambda) = (j' psi^-1 j)^-1 / n`, `j = G [1, 2 lambda]'`. The
  closed forms replace spreg's `n x n` `A1`, `A2` and are verified against spreg's own `psi`, `G`, `lambda` and
  standard errors (the betas-`lambda` covariance of spreg is not formed; it changes no reported number).
  Pinned to spreg on a row-standardized lattice, an asymmetric kNN scene and a non-row-standardized (binary) lattice.
- Traces are exact up to 4096 rows and Hutchinson estimates above (`traceMode` overrides). With stochastic traces
  the standard error of `lambda` is approximate and `..._SUMMARY_APPROXIMATION` is 1.0 (betas, `lambda` and the
  betas' standard errors still use the exact moment sums).
- `table`: `(k + 2) x 4` f32 rows of `[coefficient, se, z, two-sided normal p]`: the intercept, `X`, then
  `lambda` last.
- `summary` (13, `GPU_SPATIAL_ERROR_GM_SUMMARY_*`): `n`, `lambda`, `sigma2 = e'e / n` (spreg `sig2n`), pseudo R2,
  the residual sum of squares `u'u` and the moment objective at the solution, then conditioning of the final
  filtered regression (Belsley condition number of the raw filtered `[1, X]`, correlation condition number of the
  factored Gram matrix, largest predictor variance inflation factor, and a precision warning above a correlation
  condition number of 1e4 that never changes `status`), the `lambda` bound, the approximation flag and the
  at-bound flag.
- `status.code`: 0 ok, 1 singular (OLS or filtered fit), 2 too few rows (`n <= k + 2`), 3 non-finite (including `W`
  without links).
- `residuals`: `u = y - X b`.

`getGPUSpatialErrorGMModelResult(readback, {predictorNames?})` turns the read-back `table`, `summary` and `status`
into the shared `GPURegressionModelResult`: rows `CONSTANT`, the predictors and `lambda` (role `'spatial-error'`),
z statistics with float64 normal p-values, `sigmaSquaredMaximumLikelihood = e'e / n` (and
`residualSumOfSquares = n sigma2` of the filtered residuals, the scale of the standard errors), `pseudoRSquared`,
no likelihood and the conditioning above.

Accumulations use the fixed-order tile and Kahan scheme of `GPUOrdinaryLeastSquares`, so results are
deterministic. The spec matches spreg on a 10x10 lattice and an asymmetric kNN scene (`k = 4`, 80 rows):
`lambda` and the coefficients within 5e-3, standard errors within 1e-2 and z within 2e-2 (relative), for both
estimators. The `GM_Error_Het` variant and the Kelejian-Prucha weighted estimator are not provided.

```ts
graph.add(new GPUSpatialErrorGM({weights, predictors, response, predictorCount: 2, output: {table, summary, status: {code: status}}}));
```

#### How it computes

{/* lens:GPUSpatialErrorGM */}

::lens{scenario=gpu-spatial-analysis.spatial-error-gm rung=formula caption="Follow the moment system, the lambda search and the filtered regression."}
::lens{scenario=gpu-spatial-analysis.spatial-error-gm rung=kernel caption="Inspect the moment, lambda scan, filtered regression and finish kernels with retained WGSL."}
::lens{scenario=gpu-spatial-analysis.spatial-error-gm rung=graph caption="Trace the OLS stage, traces, moment steps and filtered fit over 788 tracts."}

### `GPUSpatialLagMaximumLikelihood`

Estimates a spatial lag model by maximum likelihood, searching rho on the concentrated likelihood with a trace-series log-determinant.

::lens{scenario=gpu-spatial-analysis.spatial-lag-maximum-likelihood rung=result caption="Five rows: intercept, three predictors and rho; rho = 0.65 with standard error 0.021."}

Maximum-likelihood estimator of the spatial lag model `y = rho W y + X b + e` (spreg 1.9.1 `ML_Lag`, `method='full'`),
with the exact log-determinant replaced by a truncated trace series. Nothing is read back. Props: `weights` (any
square CSR), `predictors`, `response`, `predictorCount` (1 to 14), `maximumPower` (series length `Q`, 2 to 256,
default 128), `traceMode`, `traceProbeCount`, `traceParameters` (per-frame seed), optional `tileRowCount` and
`output: {table, summary, status: {code, converged?}, residuals?, powerTraces?, powerSums?, infinityNorm?}`. Provide `powerTraces`,
`powerSums` and `infinityNorm` together to share the traces with a downstream [`GPUSpatialLagImpacts`](#gpuspatiallagimpacts).

- Steps: the OLS residuals `e0`, `e1` of `y` and `W y` on `[1, X]` (two `GPUOrdinaryLeastSquares` compositions) make
  the concentrated likelihood `-n/2 ln(s2(rho)) + ln|I - rho W|`, `s2 = (e0 - rho e1)'(e0 - rho e1) / n`, depend on
  three sums. The log-determinant is `-sum_{q<=Q} rho^q tr(W^q) / q` from
  [`GPUSpatialWeightsTraces`](#gpuspatialweightstraces) (exact traces up to 4096 rows, Hutchinson above).
  `rho` is the global maximum on `[-B, B]`: one kernel evaluates the analytic derivative on 1025 grid points,
  bisects (32 steps) every positive-to-non-positive sign change on the derivative, and keeps the best candidate
  (an end point counts when the derivative points outward). It is deterministic and has no iteration budget.
  `b = b0 - rho b1`, `s2 = u'u / n`.
- Variance: spreg's information matrix of `(b, rho, s2)` with `tr1 = tr(W A)` and `tr2 = tr((W A)^2)` as series in
  `rho` of the power traces, and `tr3 = tr((W A)' W A)`, `A = (I - rho W)^-1`, from a second probe pass at the
  estimated `rho` (Horner `u <- z + rho W u` per probe, exact unit probes up to 4096 rows, Rademacher above; the
  probes of `GPUSpatialWeightsTraces`). The `s2` row is eliminated by its Schur complement, the matrix is built on
  centered predictors, unit-scaled and factored with the shared Cholesky, and the intercept variance is mapped back
  to the uncentered model. A positive definite failure leaves the standard errors NaN
  (`..._SUMMARY_VARIANCE_DEFINED` 0) without changing `status`. The coefficient block equals spreg's `vm`.
- Search interval and truncation. `B = min(0.99, seriesLimit) / ||W||_inf` with `||W||_inf` the largest absolute row
  sum computed on the GPU (0.99 for a row-standardized `W`; keeps `I - rho W` invertible for any `W`, whereas spreg's
  fixed `(-1, 1)` admits explosive `rho` for raw weights). The series omits a tail below
  `n s^(Q+1) / ((Q+1) (1 - s))`, `s = |rho| ||W||_inf` (worst case, `|tr(W^q)| <= n s^q`;
  `getSpatialLogDeterminantTruncationBound`), and omitting it biases the likelihood toward large positive `rho`, so
  the interval also stops at the `seriesLimit` where the worst-case tail reaches `1e-5 n`
  (`getSpatialMaximumLikelihoodSeriesLimit`: about 0.93 for `Q = 128`, 0.96 for `Q = 256`). Beyond it the fit is a
  boundary solution (`..._SUMMARY_AT_BOUNDARY`, `..._SERIES_LIMITED`, `converged` 0); raise `maximumPower` when
  `rho` is near the bound. The default `Q = 128` keeps the worst-case truncation below `1e-7 n` log-likelihood units
  at `|rho| <= 0.9` (below `1e-6 |LL|`) at the cost of about `Q (1 + probe batches)` dispatches; the probe pass costs
  `rows x Q x nnz` multiply-adds in exact mode. `..._SUMMARY_TRUNCATION_BOUND` reports the remainder at the estimate.
- Approximation. Above 4096 rows the traces and the `tr3` pass are Hutchinson estimates (`..._SUMMARY_APPROXIMATION`
  1.0): `rho` itself then carries stochastic error (about 1e-3 with 256 probes on a 144-row lattice), unlike
  `GPUSpatialErrorGM` where only the standard error of `lambda` is approximate.
- `table`: `(k + 2) x 4` f32 rows of `[coefficient, se, z, two-sided normal p]`: the intercept, `X`, then `rho` last
  (spreg `W_y`). It is the layout `GPUSpatialLagImpacts` reads (`coefficientStride: 4`, `rhoIndex: predictorCount + 1`).
- `summary` (22, `GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_*`): `n`, `rho`, `sigma2 = u'u / n`, full log-likelihood
  `-n/2 ln s2 + ln|I - rho W| - n/2 (ln 2 pi + 1)` (spreg `logll`), AIC and Schwarz with spreg's `k = predictorCount + 2`
  (`rho` counted), `pr2` (squared correlation of `y` and `y - u`), `pr2_e` (of `y` and `(I - rho W)^-1 X b`), `u'u`,
  conditioning of the design (Belsley number, correlation condition number, largest VIF, precision warning above
  1e4; never changes `status`), the bound, the approximation, converged, at-boundary and series-limited flags, the
  truncation bound, variance-defined, the number of local maxima found on the grid (more than 1 is multimodal) and
  `ln|I - rho W|`.
- `status.code`: 0 ok, 1 singular design, 2 too few rows (`n <= k + 2`), 3 non-finite predictor, response or weight (or no
  admissible interval); the outputs are then NaN. `status.converged` (optional) is 1 for an interior maximum.
- `residuals`: `u = y - rho W y - X b`.

`getGPUSpatialLagMaximumLikelihoodModelResult(readback, {predictorNames?})` builds the shared
`GPURegressionModelResult` (`kind: 'spatial-lag-maximum-likelihood'`, rows `CONSTANT`, predictors and `W_y` with role
`'spatial-lag'`, z statistics with float64 normal p-values, `standardErrorSigmaSquared: 'maximum-likelihood'`,
`logLikelihood`, and AIC and Schwarz equal to spreg's); `getGPUSpatialMaximumLikelihoodDiagnostics(summary)` returns
the flags and `pr2_e`. At most 14 predictors: the `(predictorCount + 2)`-wide information matrix must fit the
16-wide Cholesky. Inputs must be finite.

Verified against spreg `ML_Lag(method='full', spat_impacts='full')` on a 12x12 lattice, a binary (raw) lattice, a
six-predictor lattice and two kNN scenes by a float64 mirror (`rho` to 4e-8, coefficients 3e-7, standard errors 1e-7,
z, log-likelihood, AIC, Schwarz, `pr2`, `pr2_e`, conditioning and `tr1`, `tr2`, `tr3`); the GPU spec checks the float32
output against the same pins at 2e-3 (`rho`, coefficients) and 1e-2 (standard errors), and chains the fit into
`GPUSpatialLagImpacts`. spreg's total multiplier `1 / (1 - rho)` is exact only for a row-standardized `W`; for other
weights the impacts here use the exact `1'W^q 1` sums.

```ts
graph.add(new GPUSpatialLagMaximumLikelihood({weights, predictors, response, predictorCount: 2,
  output: {table, summary, status: {code: status}, powerTraces, powerSums, infinityNorm}}));
graph.add(new GPUSpatialLagImpacts({coefficients: table, coefficientStride: 4, rhoIndex: 3, predictorCount: 2,
  rowCount, maximumPower: 128, traces: {powerTraces, powerSums, infinityNorm}, status, output: {impacts}}));
```

#### How it computes

{/* lens:GPUSpatialLagMaximumLikelihood */}

::lens{scenario=gpu-spatial-analysis.spatial-lag-maximum-likelihood rung=formula caption="Follow the concentrated likelihood, its log-determinant series and the rho search."}
::lens{scenario=gpu-spatial-analysis.spatial-lag-maximum-likelihood rung=kernel caption="Inspect the OLS, trace, rho search and variance kernels with retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.spatial-lag-maximum-likelihood rung=graph caption="Trace the two OLS fits, 128 trace powers and the rho search over 788 tracts."}

### `GPUSpatialErrorMaximumLikelihood`

Estimates a spatial error model by maximum likelihood, searching lambda on the concentrated likelihood with a trace-series log-determinant.

::lens{scenario=gpu-spatial-analysis.spatial-error-maximum-likelihood rung=result caption="Five rows: intercept, three predictors and lambda; lambda = 0.89 with standard error 0.018."}

Maximum-likelihood estimator of the spatial error model `y = X b + u`, `u = lambda W u + e` (spreg 1.9.1
`ML_Error`, `method='full'`), with the same truncated-series log-determinant, search and approximation rules as
`GPUSpatialLagMaximumLikelihood`. Props: `weights`, `predictors`, `response`, `predictorCount` (1 to 15),
`maximumPower`, `traceMode`, `traceProbeCount`, `traceParameters` (per-frame seed), optional `tileRowCount` and
`output: {table, summary, status: {code, converged?}, residuals?}`.

- Steps: with `Z = [1, X - mean, y - mean]`, three tile passes (Kahan-merged, fixed order) accumulate once
  `G0 = Z'Z`, `G1 + G1' = Z'WZ + (Z'WZ)'` and `G2 = (WZ)'(WZ)`. The Gram matrix of the filtered columns
  `Z - lambda W Z` (the filtered constant `1 - lambda sum_j w_ij` is column 0, as in spreg) is then exactly
  `G0 - lambda (G1 + G1') + lambda^2 G2`, so scanning `lambda` needs no pass over the rows. Each grid point solves
  one unit-scaled Cholesky system for `ee(lambda)`, and the derivative follows from the envelope theorem
  (`ee' = yy' - 2 b'r' + b'A'b`); the search is the one of the lag model on `-n/2 ln(ee / n) + ln|I - lambda W|`.
- Variance: `Var(b) = s2 (Xs'Xs)^-1` (centered form mapped back to the uncentered intercept) and
  `Var(lambda) = 1 / (tr2 + tr3 - 2 tr1^2 / n)`, the Schur complement of spreg's 2 x 2 information matrix of
  `(lambda, s2)`, with the traces of `W (I - lambda W)^-1` (`tr3` from the probe pass at the estimated `lambda`). As in
  spreg the coefficient standard errors condition on `lambda`.
- `table`: `(k + 2) x 4` rows, `lambda` last. `summary`: the 22 slots of the lag model; `sigma2 = e'e / n` of the
  filtered residuals, `pr2` of `y` and `X b`, `pr2_e` NaN, `u'u` of `u = y - X b`, and the conditioning of the final
  filtered regression (Belsley number of the raw `[c, X - lambda W X]`, correlation condition number and largest
  predictor VIF of the unit-scaled filtered system). AIC and Schwarz use spreg's `k = predictorCount + 1`: spreg's
  `ML_Error` does not count `lambda` (`k = x.shape[1]`), unlike `ML_Lag`, which counts `rho`.
- `status.code` as the lag model (the cap is 15 predictors, the OLS and Cholesky limit).

`getGPUSpatialErrorMaximumLikelihoodModelResult(readback, {predictorNames?})` builds the shared model result
(`kind: 'spatial-error-maximum-likelihood'`, `lambda` role `'spatial-error'`, `residualSumOfSquares = n sigma2` of the
filtered residuals, AIC and Schwarz equal to spreg's). Verified against `ML_Error(method='full')` on a
row-standardized lattice, an asymmetric kNN scene and a binary lattice (float64 mirror to 1e-7; GPU spec at 2e-3 and
1e-2, not yet measured).

```ts
graph.add(new GPUSpatialErrorMaximumLikelihood({weights, predictors, response, predictorCount: 2,
  output: {table, summary, status: {code: status}}}));
```

#### How it computes

{/* lens:GPUSpatialErrorMaximumLikelihood */}

::lens{scenario=gpu-spatial-analysis.spatial-error-maximum-likelihood rung=formula caption="Follow the filtered Gram matrix, the concentrated likelihood and the lambda search."}
::lens{scenario=gpu-spatial-analysis.spatial-error-maximum-likelihood rung=kernel caption="Inspect the Gram, trace, lambda search and variance kernels with retained WGSL bindings."}
::lens{scenario=gpu-spatial-analysis.spatial-error-maximum-likelihood rung=graph caption="Trace the Gram passes, 128 trace powers and the lambda search over 788 tracts."}
