// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import fixtureJson from '../../oracles/fixtures/statistics-gwr.json';
import {decodeOracleFixture} from '../../oracles/oracle-fixture';

/**
 * Reference values from mgwr 2.2.1 (float64, numpy 2.5.4) for GWR local inference,
 * bandwidth search and spherical distances. GENERATED: do not edit by hand.
 *
 * Generator: `test/oracles/statistics-gwr.py` runs mgwr GWR / Sel_BW on the `planar_12_77`
 * (`createNonstationarityScene(12, 77)`) and `geographic_12_77` scenes of
 * `test/oracles/fixtures/statistics-regression-scenes.json` and writes
 * `test/oracles/fixtures/statistics-gwr.json`, imported here.
 *
 * All fits are `constant = True` bisquare unless stated. Planar scene: 144 rows, 2 predictors, so
 * `k = 3` columns. `significant` is `filter_tvals() != 0`, that is `|t| >= critical_tval()` with
 * mgwr's adjusted alpha at 0.05 (`adj_alpha[1]`). mgwr `sigma2` is `sigma2_v1`
 * (`RSS / (n - tr(S))`). Spherical fits use mgwr's haversine radius of 6371.0 km, so bandwidths
 * are kilometres there (the GPU specs pass `sphereRadius = 6371000` and bandwidths in metres).
 */

/** mgwr summary of one fit; the per-row arrays are present on the full fits only. */
export type MgwrFitReference = {
  enp: number;
  trS: number;
  rss: number;
  sigma2: number;
  aicc: number;
  aic: number;
  bic: number;
  llf: number;
  r2: number;
  adjAlpha05: number;
  criticalT: number;
  n: number;
  k: number;
  significantCount: number;
  /** Row-major `n x k`. */
  coefficients: number[];
  standardErrors: number[];
  tValues: number[];
  /** 1 where `filter_tvals() != 0`. */
  significant: number[];
};

/**
 * One mgwr golden-section search (`Sel_BW(...).search('golden_section')`) with its trace, from
 * `golden_trace` of the generator (a port of `mgwr.search.golden_section` that records the calls).
 */
export type MgwrGoldenSectionReference = {
  label: string;
  kernel: 'bisquare' | 'gaussian';
  fixed: boolean;
  /** Bounds the search started from. */
  lower: number;
  upper: number;
  /** What mgwr returns: the optimum rounded to two decimals (integers for adaptive). */
  bandwidth: number;
  /** The unrounded optimum, whose AICc is `aicc`. */
  unroundedBandwidth: number;
  aicc: number;
  /** mgwr iterations (comparisons of the two interior points). */
  iterationCount: number;
  /** mgwr tolerance on the interior AICc difference. */
  tolerance: number;
  /** Every distinct bandwidth mgwr evaluated, in call order, with its AICc. */
  evaluated: [number, number][];
};

const fixture = decodeOracleFixture<{
  generator: string;
  versions: Record<string, string>;
  inference: {
    fixedBisquare22: MgwrFitReference;
    adaptiveBisquare20: MgwrFitReference;
    fixedGaussian15: Omit<
      MgwrFitReference,
      'coefficients' | 'standardErrors' | 'tValues' | 'significant'
    >;
    adjustedAlpha10: number;
    /** Exact (untruncated) Gaussian fits with per-row arrays: the `gaussianCutoff` reference. */
    fixedGaussian15Full: MgwrFitReference;
    adaptiveGaussian20Full: MgwrFitReference;
  };
  bandwidthSearch: {
    adaptive: {
      ks: number[];
      aicc: number[];
      argminK: number;
      goldenDefaultBounds: number;
      goldenBounds6to144: number;
    };
    fixed: {
      distances: number[];
      aicc: number[];
      argminDistance: number;
      goldenBounds1to6: number;
      goldenAicc: number;
    };
    goldenSection: MgwrGoldenSectionReference[];
  };
  spherical: {
    radiusKilometers: number;
    fixedBisquare25km: MgwrFitReference;
    adaptiveBisquare20: MgwrFitReference;
  };
}>(fixtureJson);

/**
 * Local inference of fixed bandwidth 2.2, adaptive k = 20 and Gaussian fixed 1.5 on the planar
 * scene; `adjustedAlpha10` is mgwr `adj_alpha[0]`, the 0.1 level, of the fixed 2.2 fit.
 */
export const MGWR_INFERENCE_REFERENCE = fixture.inference;

/**
 * Bandwidth search on the planar scene, bisquare kernel. `adaptive.aicc[i]` is the mgwr AICc of
 * adaptive `k = adaptive.ks[i]` (every integer 6..144); `goldenBounds6to144` is
 * `Sel_BW(...).search(bw_min=6, bw_max=144)` and `goldenDefaultBounds` its default bounds
 * (`[40 + 2 p, n]`), which exclude the optimum. `fixed.aicc[i]` is the AICc at distance
 * `fixed.distances[i]` and `goldenBounds1to6` the golden-section result over `[1, 6]`.
 */
export const MGWR_BANDWIDTH_SEARCH_REFERENCE = fixture.bandwidthSearch;

/**
 * mgwr golden-section searches on the planar scene: adaptive (bounds `[46, 128]` and `[6, 128]`, the
 * default upper bound `n = 144` capped at the GPU neighbour list) and fixed (default bounds
 * `[min_dist / 2, 2 max_dist]` and explicit ones), bisquare and Gaussian.
 */
export const MGWR_GOLDEN_SECTION_REFERENCE = fixture.bandwidthSearch.goldenSection;

/** Spherical (haversine, kilometres) fits of the geographic scene: fixed 25 km and adaptive k = 20. */
export const MGWR_SPHERICAL_REFERENCE = fixture.spherical;
