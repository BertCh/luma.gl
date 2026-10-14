// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Input scenes of the traces / lag-impacts generator (`test/oracles/statistics-spatial-weights-traces.py`).
 * The scenes are built by the TypeScript generators the GPU specs use; this spec checks that
 * `test/oracles/fixtures/statistics-spatial-weights-traces-scenes.json` still equals them, so a
 * changed scene generator cannot silently drift away from its pinned reference values. To
 * regenerate after an intentional scene change, run
 *     LUMA_WRITE_ORACLE_SCENES=1 npx vitest run --project node <this file>
 * then rerun the python generator.
 */

import {readFileSync, writeFileSync} from 'node:fs';
import {dirname, join} from 'node:path';
import {fileURLToPath} from 'node:url';
import {expect, it} from 'vitest';
import {createNearestNeighborScene} from './spatial-regression-diagnostics-oracle';
import {createLagScene} from './spatial-two-stage-least-squares-oracle';

const FIXTURE_PATH = join(
  dirname(fileURLToPath(import.meta.url)),
  '..',
  '..',
  'oracles',
  'fixtures',
  'statistics-spatial-weights-traces-scenes.json'
);

type TracesSceneInput = {
  predictorCount: number;
  predictors: ArrayLike<number>;
  response: ArrayLike<number>;
  weights: {offsets: ArrayLike<number>; neighbors: ArrayLike<number>; weights: ArrayLike<number>};
};

function toJsonScene(scene: TracesSceneInput) {
  return {
    k: scene.predictorCount,
    predictors: Array.from(scene.predictors),
    response: Array.from(scene.response),
    offsets: Array.from(scene.weights.offsets),
    neighbors: Array.from(scene.weights.neighbors),
    weights: Array.from(scene.weights.weights)
  };
}

function createScenes() {
  const scenes: Record<string, TracesSceneInput> = {
    lattice: createLagScene(12, 21, 0.5, true),
    rawLattice: createLagScene(10, 8, 0.12, false),
    knn60: createNearestNeighborScene(60, 3, 13, 0.6, false),
    knn90: createNearestNeighborScene(90, 3, 29, 0.6, false)
  };
  return Object.fromEntries(
    Object.entries(scenes).map(([name, scene]) => [name, toJsonScene(scene)])
  );
}

it('the traces oracle scenes equal the TypeScript scene generators', () => {
  const scenes = createScenes();
  if (process.env['LUMA_WRITE_ORACLE_SCENES']) {
    writeFileSync(
      FIXTURE_PATH,
      `${JSON.stringify({
        generator:
          'test/gpu-spatial-analysis/spatial-regression/spatial-weights-traces-scenes.node.spec.ts',
        scenes
      })}\n`
    );
  }
  const fixture = JSON.parse(readFileSync(FIXTURE_PATH, 'utf8'));
  // JSON keeps every float32 (and float64) value exactly, so the comparison is exact.
  expect(fixture.scenes).toEqual(JSON.parse(JSON.stringify(scenes)));
});
