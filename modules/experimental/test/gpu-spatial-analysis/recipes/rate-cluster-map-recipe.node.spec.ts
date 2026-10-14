// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {describe, expect, test} from 'vitest';
import {GPULocalPermutationTest} from '../../../src/gpu-spatial-analysis/permutation-inference/index';
import {
  addRateClusterMapRecipe,
  type GPURateClusterMapRecipeProps
} from '../../../src/gpu-spatial-analysis/recipes/rate-cluster-map-recipe';
import {GPULocalMoran} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/index';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

const ROWS = 4;

function createProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPURateClusterMapRecipeProps> = {}
): GPURateClusterMapRecipeProps {
  const view = <Format extends 'float32' | 'uint32' | 'float32x2'>(
    id: string,
    format: Format,
    length: number
  ) => createTransientView(graph, `test-${id}`, format, length);
  return {
    events: view('events', 'float32', ROWS),
    populations: view('populations', 'float32', ROWS),
    positions: view('positions', 'float32x2', 16),
    ringOffsets: view('ring-offsets', 'uint32', ROWS + 1),
    polygonOffsets: view('polygon-offsets', 'uint32', ROWS + 1),
    neighborCapacity: 16,
    parameters: view('parameters', 'float32', 8),
    palette: view('palette', 'uint32', 5),
    ...overrides
  };
}

describe('addRateClusterMapRecipe inference metadata', () => {
  test('permutation metadata is the permutation node getInferenceMetadata, not a copy', () => {
    const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'rate-cluster-permutation'});
    const result = addRateClusterMapRecipe(
      graph,
      createProps(graph, {
        multipleTesting: 'holm',
        significanceLevel: 0.01,
        permutation: {
          parameters: createTransientView(graph, 'permutation-parameters', 'uint32', 4),
          maximumPermutations: 99,
          alternative: 'greater',
          permutations: 49,
          seed: 7
        }
      })
    );
    const node = result.contributors.find(
      contributor => contributor instanceof GPULocalPermutationTest
    ) as GPULocalPermutationTest;
    expect(node).toBeDefined();
    const expected = node.getInferenceMetadata({
      seed: 7,
      permutations: 49,
      significanceLevel: 0.01
    });
    expect(result.inference).toEqual(expected);
    expect(result.inference).toMatchObject({
      alternative: 'greater',
      permutationCount: 49,
      seed: 7,
      multipleTesting: 'holm',
      significanceLevel: 0.01
    });
  });

  test('permutation metadata defaults: two-sided, maximumPermutations, no seed', () => {
    const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'rate-cluster-defaults'});
    const result = addRateClusterMapRecipe(
      graph,
      createProps(graph, {
        permutation: {
          parameters: createTransientView(graph, 'permutation-parameters', 'uint32', 4),
          maximumPermutations: 99
        }
      })
    );
    expect(result.inference).toMatchObject({
      alternative: 'two-sided',
      permutationCount: 99,
      multipleTesting: 'benjamini-hochberg'
    });
    expect(result.inference.seed).toBeUndefined();
    expect('seed' in result.inference).toBe(false);
  });

  test('a permutation count above maximumPermutations is refused', () => {
    const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'rate-cluster-too-many'});
    expect(() =>
      addRateClusterMapRecipe(
        graph,
        createProps(graph, {
          permutation: {
            parameters: createTransientView(graph, 'permutation-parameters', 'uint32', 4),
            maximumPermutations: 99,
            permutations: 100,
            seed: 1
          }
        })
      )
    ).toThrow();
  });

  test('analytic metadata is the local Moran getInferenceMetadata', () => {
    const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'rate-cluster-analytic'});
    const result = addRateClusterMapRecipe(
      graph,
      createProps(graph, {multipleTesting: 'benjamini-yekutieli', significanceLevel: 0.1})
    );
    const node = result.contributors.find(
      contributor => contributor instanceof GPULocalMoran
    ) as GPULocalMoran;
    expect(result.inference).toEqual(node.getInferenceMetadata(0.1));
    expect(result.inference).toMatchObject({
      alternative: 'two-sided',
      permutationCount: 0,
      multipleTesting: 'benjamini-yekutieli',
      significanceLevel: 0.1
    });
  });
});
