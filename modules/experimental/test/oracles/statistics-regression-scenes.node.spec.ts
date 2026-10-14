// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Input scenes of the spreg / mgwr regression generators (`statistics-spatial-regression.py`,
 * `statistics-gwr.py`). The scenes are built by the TypeScript generators the GPU specs use; this
 * spec checks that `fixtures/statistics-regression-scenes.json` still equals them, so a changed
 * scene generator cannot silently drift away from its pinned reference values. To regenerate after
 * an intentional scene change, run
 *     LUMA_WRITE_ORACLE_SCENES=1 npx vitest run --project node <this file>
 * then rerun both python generators. `frozen` scenes (the two `GM_Error` scenes whose original
 * generator was not kept) are data, preserved as they are.
 */

import {existsSync, readFileSync, writeFileSync} from 'node:fs';
import {dirname, join} from 'node:path';
import {fileURLToPath} from 'node:url';
import {expect, it} from 'vitest';
import {
  createCollinearOrdinaryLeastSquaresScene,
  createHeteroskedasticOrdinaryLeastSquaresScene,
  createOrdinaryLeastSquaresScene
} from '../gpu-spatial-analysis/spatial-regression/ordinary-least-squares-oracle';
import {createErrorScene} from '../gpu-spatial-analysis/spatial-regression/spatial-error-gm-oracle';
import {
  createDiagnosticsExclusionMask,
  createDiagnosticsScene,
  createNearestNeighborScene
} from '../gpu-spatial-analysis/spatial-regression/spatial-regression-diagnostics-oracle';
import {
  createIllConditionedLagScene,
  createLagScene,
  createWeakInstrumentScene
} from '../gpu-spatial-analysis/spatial-regression/spatial-two-stage-least-squares-oracle';
import {
  createGeographicScene,
  createNonstationarityScene,
  createSingularRowScene
} from '../gpu-spatial-analysis/spatial-regression/geographically-weighted-regression-scenes';

const FIXTURE_PATH = join(
  dirname(fileURLToPath(import.meta.url)),
  'fixtures',
  'statistics-regression-scenes.json'
);

type RegressionSceneInput = {
  predictorCount: number;
  predictors: ArrayLike<number>;
  response: ArrayLike<number>;
  weights?: {offsets: ArrayLike<number>; neighbors: ArrayLike<number>; weights: ArrayLike<number>};
  positions?: ArrayLike<number>;
  mask?: ArrayLike<number>;
};

function toJsonScene(scene: RegressionSceneInput) {
  return {
    k: scene.predictorCount,
    predictors: Array.from(scene.predictors),
    response: Array.from(scene.response),
    ...(scene.weights
      ? {
          offsets: Array.from(scene.weights.offsets),
          neighbors: Array.from(scene.weights.neighbors),
          weights: Array.from(scene.weights.weights)
        }
      : {}),
    ...(scene.positions ? {positions: Array.from(scene.positions)} : {}),
    ...(scene.mask ? {mask: Array.from(scene.mask)} : {})
  };
}

function createScenes() {
  const nonstationarity = createNonstationarityScene(12, 77);
  const scenes: Record<string, RegressionSceneInput> = {
    // spreg.OLS
    hetero_41_300: createHeteroskedasticOrdinaryLeastSquaresScene(41, 300),
    collinear_moderate_17_200: createCollinearOrdinaryLeastSquaresScene(17, 200, 0.3),
    collinear_severe_19_200: createCollinearOrdinaryLeastSquaresScene(19, 200, 0.03),
    plain_2_400: createOrdinaryLeastSquaresScene(2, 400, 2, [3, -1], 0.5),
    // spreg.ML_Lag
    lag_12_21: createLagScene(12, 21, 0.5, true),
    lag_binary_10_5: createLagScene(10, 5, 0.1, false),
    lag_6pred_10_7: createLagScene(10, 7, 0.4, true, 6),
    knn_60_13: createNearestNeighborScene(60, 3, 13, 0.6, false),
    knn_90_29: createNearestNeighborScene(90, 3, 29, 0.6, false),
    // spreg.ML_Error, spreg.GM_Error and GM_Error_Hom (err_binary_324_9)
    err_lattice_400_31: createErrorScene(400, 31, 0.6, 'lattice-row-standardized'),
    err_knn_400_31: createErrorScene(400, 31, 0.6, 'knn'),
    err_binary_324_9: createErrorScene(324, 9, 0.5, 'lattice-binary'),
    // spreg.OLS(spat_diag=True, moran=True), GM_Lag and the 2SLS diagnostics
    diagnostics_lattice_8_11: createDiagnosticsScene(8, 11, 0.6, true),
    diagnostics_knn_80_7: createNearestNeighborScene(80, 4, 7, 0.6),
    masked_lattice_8_11: {
      ...createDiagnosticsScene(8, 11, 0.6, true),
      mask: createDiagnosticsExclusionMask(64, 7, 3)
    },
    masked_knn_60_13: {
      ...createNearestNeighborScene(60, 3, 13, 0.6, false),
      mask: createDiagnosticsExclusionMask(60, 6, 2)
    },
    lag_binary_10_5_high: createLagScene(10, 5, 0.255, false),
    weak_80_7: createWeakInstrumentScene(80, 7),
    ill_conditioned_12_21: createIllConditionedLagScene(12, 21, 0.5),
    // mgwr.GWR
    planar_12_77: nonstationarity,
    geographic_12_77: createGeographicScene(nonstationarity),
    singular_12_5_4: createSingularRowScene(12, 5, 4)
  };
  return Object.fromEntries(
    Object.entries(scenes).map(([name, scene]) => [name, toJsonScene(scene)])
  );
}

it('the regression oracle scenes equal the TypeScript scene generators', () => {
  const scenes = createScenes();
  const existing = existsSync(FIXTURE_PATH)
    ? (JSON.parse(readFileSync(FIXTURE_PATH, 'utf8')) as {
        frozen: Record<string, unknown>;
        scenes: Record<string, unknown>;
      })
    : {frozen: {}, scenes: {}};
  if (process.env['LUMA_WRITE_ORACLE_SCENES']) {
    writeFileSync(
      FIXTURE_PATH,
      `${JSON.stringify({
        generator: 'test/oracles/statistics-regression-scenes.node.spec.ts',
        frozen: existing.frozen,
        scenes
      })}\n`
    );
  }
  const fixture = JSON.parse(readFileSync(FIXTURE_PATH, 'utf8'));
  // JSON keeps every float32 (and float64) value exactly, so the comparison is exact.
  expect(fixture.scenes).toEqual(JSON.parse(JSON.stringify(scenes)));
  expect(Object.keys(fixture.frozen).sort()).toEqual(['knn', 'lattice']);
});
