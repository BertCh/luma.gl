// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Reference Matheron semivariances of the variogram outlier scenes (see
 * `variogram-outlier-fixture.ts`; maximum distance 10, 8 even lags), pinned from scikit-gstat 1.0.24
 * `Variogram(coordinates, values, estimator='matheron', n_lags=8, maxlag=10, bin_func='even',
 * normalize=False)`.experimental, which agrees with a direct numpy sum `0.5 * mean((z_i - z_j)^2)`
 * per lag (relative 1e-12) as asserted by the script.
 * Oracle script: /private/tmp/claude-501/-Users-robertchristie-Documents-GitHub-deck-gl-master/04829ece-fd0e-40ba-860a-0759cbf76955/scratchpad/fix/work/W2A/pairstats/oracle.py
 * (reads the scenes dumped from the fixture; output in the sibling oracle.json).
 */
export type VariogramOutlierReference = {
  /** scikit-gstat `Variogram.experimental` per lag. */
  semivariances: number[];
  /** Pairs per lag (numpy, f64 distances). */
  pairCounts: number[];
  /** Population variance of the scene values (numpy). */
  variance: number;
};

export const VARIOGRAM_OUTLIER_REFERENCES: Record<string, VariogramOutlierReference> = {
  clean: {
    semivariances: [
      0.9959115156817024, 0.9825185019735775, 0.9771117745085447, 0.9685524998262328,
      0.9926251759955264, 0.9780776374134468, 0.9755460804946733, 0.9774014236889484
    ],
    pairCounts: [3959, 11381, 18567, 25934, 32977, 39462, 45498, 51736],
    variance: 0.9805594486578746
  },
  'isolated-2e4': {
    semivariances: [
      0.9962288826138443, 0.9822786045879904, 0.976984207403112, 0.9684143671614173,
      0.9925900385359837, 0.9780337671614615, 0.9754357485908762, 0.9770633628438364
    ],
    pairCounts: [3957, 11377, 18559, 25920, 32965, 39447, 45475, 51705],
    variance: 99976.19730020572
  },
  'isolated-1e5': {
    semivariances: [
      0.9962288826138443, 0.9822786045879904, 0.976984207403112, 0.9684143671614173,
      0.9925900385359837, 0.9780337671614615, 0.9754357485908762, 0.9770633628438364
    ],
    pairCounts: [3957, 11377, 18559, 25920, 32965, 39447, 45475, 51705],
    variance: 2499377.0660483716
  },
  'embedded-2e4': {
    semivariances: [
      101041.67015060269, 70291.55540873784, 86174.06053604899, 107970.84923550821,
      72779.47619087335, 76024.72372919733, 101103.62357037488, 119839.20876112375
    ],
    pairCounts: [3959, 11381, 18567, 25934, 32977, 39462, 45498, 51736],
    variance: 99976.19730020572
  },
  'embedded-1e5': {
    semivariances: [
      2525916.668682519, 1757305.7071752942, 2154354.304834137, 2699177.8967211302,
      1819453.3473765722, 1900569.7606157954, 2527581.1217614813, 2995975.807613769
    ],
    pairCounts: [3959, 11381, 18567, 25934, 32977, 39462, 45498, 51736],
    variance: 2499377.0660483716
  }
};
