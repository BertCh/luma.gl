// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Reference global Moran's I of the variogram outlier scenes (see `variogram-outlier-fixture.ts`)
 * at the cumulative correlogram bands with upper distances 2.5, 5, 7.5 and 10: esda 2.10.0
 * `Moran(values, w, permutations=0, transformation='O')` with libpysal 4.15.0
 * `DistanceBand(coordinates, threshold, binary=True, alpha=-1)` weights.
 * Oracle script: /private/tmp/claude-501/-Users-robertchristie-Documents-GitHub-deck-gl-master/04829ece-fd0e-40ba-860a-0759cbf76955/scratchpad/fix/work/W2A/pairstats/moran.py
 * (reads the scenes dumped from the fixture; output in the sibling moran.json).
 */
export type CorrelogramOutlierReference = {
  /** Moran's I per band. */
  moransI: number[];
  /** `S0` per band: ordered neighbor pairs. */
  weightSums: number[];
};

export const CORRELOGRAM_OUTLIER_REFERENCES: Record<string, CorrelogramOutlierReference> = {
  clean: {
    moransI: [
      0.001254828787709608, 0.0041637329051520035, 0.001975496675569744, 0.0018424253284275048
    ],
    weightSums: [30680.0, 119682.0, 264560.0, 459028.0]
  },
  'isolated-2e4': {
    moransI: [
      0.00025021263472931987, 0.0002506265595054778, 0.0002500544595225976, 0.00024996135370212514
    ],
    weightSums: [30668.0, 119626.0, 264450.0, 458810.0]
  },
  'isolated-1e5': {
    moransI: [
      0.0002500909216677268, 0.00025016926679691844, 0.000250058280119305, 0.0002500397838985624
    ],
    weightSums: [30668.0, 119626.0, 264450.0, 458810.0]
  }
};
