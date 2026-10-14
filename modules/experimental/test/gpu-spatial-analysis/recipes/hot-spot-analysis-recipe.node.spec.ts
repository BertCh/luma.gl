// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {addHotSpotAnalysisRecipe} from '../../../src/gpu-spatial-analysis/recipes/index';
import {GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/index';
import {GPUParameterBuffer} from '../../../src/utils/gpu-contributor-utils';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {computeHotSpotOracle} from '../spatial-autocorrelation/spatial-autocorrelation-oracle';
import {ESDA_GI_STAR_LATTICE} from './hot-spot-esda-fixture';

it('the Gi* oracle reproduces esda G_Local(star=True) including a cold spot of zeros', () => {
  const fixture = ESDA_GI_STAR_LATTICE;
  const oracle = computeHotSpotOracle({
    weights: {
      offsets: Uint32Array.from(fixture.offsets),
      neighbors: Uint32Array.from(fixture.neighbors),
      weights: new Float32Array(fixture.neighbors.length).fill(1)
    },
    values: Float64Array.from(fixture.values) as unknown as Float32Array
  });
  for (const [row, expected] of fixture.zScores.entries()) {
    expect(oracle.zScores[row], `row ${row}`).toBeCloseTo(expected, 6);
  }
  expect(Math.min(...oracle.zScores)).toBeLessThan(-2.576);
});

function createProps(graph: GPUCommandGraph, overrides: Record<string, unknown> = {}) {
  const view = <Format extends 'float32' | 'uint32' | 'float32x2'>(
    id: string,
    format: Format,
    length: number
  ) => createTransientView(graph, id, format, length);
  return {
    source: {
      kind: 'points' as const,
      positions: view('positions', 'float32x2', 10),
      family: 'quadbin' as const,
      resolution: 6,
      studyArea: {
        kind: 'polygons' as const,
        positions: view('polygon', 'float32x2', 4),
        featureOffsets: view('features', 'uint32', 2),
        polygonOffsets: view('polygons', 'uint32', 2),
        ringOffsets: view('rings', 'uint32', 2),
        candidateCapacity: 64
      },
      cellCapacity: 32,
      ...overrides
    },
    parameters: new GPUParameterBuffer(graph.device, {
      id: 'hot-spot-parameters',
      format: 'float32',
      length: GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH
    })
  };
}

it('addHotSpotAnalysisRecipe validates the study-area points source', () => {
  const graph = () => new GPUCommandGraph(createNullWebGPUDevice(), {id: 'hot-spot-validation'});
  const g1 = graph();
  expect(() => addHotSpotAnalysisRecipe(g1, {...createProps(g1), analyze: 'count'})).toThrow(
    /quadbin/
  );
  const g2 = graph();
  expect(() =>
    addHotSpotAnalysisRecipe(g2, {...createProps(g2), multipleTesting: 'sidak' as never})
  ).toThrow();
  const g3 = graph();
  expect(() => addHotSpotAnalysisRecipe(g3, {...createProps(g3), minimumNeighbors: 9})).toThrow(
    /exceeds the 8 neighbors/
  );
  const g4 = graph();
  expect(() => addHotSpotAnalysisRecipe(g4, {...createProps(g4, {neighborRings: 0})})).toThrow(
    /neighborRings/
  );
});

it('addHotSpotAnalysisRecipe states its inference and exposes the guard and study stages', () => {
  const g = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'hot-spot-metadata'});
  const result = addHotSpotAnalysisRecipe(g, createProps(g));
  expect(result.inference).toEqual({
    alternative: 'two-sided',
    permutationCount: 0,
    multipleTesting: 'benjamini-hochberg',
    testedRowsDefinition: 'study-area cells with at least 8 neighbors'
  });
  const stages = result.status.stages.map(stage => stage.stage);
  expect(stages).toEqual([
    'study-area',
    'study-table',
    'point-aggregation',
    'weights',
    'minimum-neighbors'
  ]);
  expect(result.status.stages.at(-1)!.status.invalidCount).toBeDefined();
  expect(result.weights.information?.provenance?.criterion).toBe('disk');
  expect(result.rowCount).toBe(32);
});

it('addHotSpotAnalysisRecipe accepts only a great-circle distance band over cell centers', () => {
  const graph = () => new GPUCommandGraph(createNullWebGPUDevice(), {id: 'hot-spot-band'});
  const band = (g: GPUCommandGraph, spatialContext?: unknown) => ({
    spatialContext,
    neighborSearchParameters: createTransientView(g, 'band-parameters', 'float32', 16),
    gridSize: [8, 8] as const,
    neighborCapacity: 256
  });
  // Omitted (planar) context: a radius in degrees is refused.
  const g1 = graph();
  expect(() =>
    addHotSpotAnalysisRecipe(g1, createProps(g1, {distanceBand: band(g1)}) as never)
  ).toThrow(/planar/);
  // Longitude/latitude with a planar-degree metric is refused too.
  const g2 = graph();
  expect(() =>
    addHotSpotAnalysisRecipe(
      g2,
      createProps(g2, {
        distanceBand: band(g2, {coordinateSpace: 'longitude-latitude', metric: 'native'})
      }) as never
    )
  ).toThrow(/metric/);
  // Exclusive with grid rings.
  const g3 = graph();
  expect(() =>
    addHotSpotAnalysisRecipe(
      g3,
      createProps(g3, {
        neighborRings: 2,
        distanceBand: band(g3, {coordinateSpace: 'longitude-latitude', metric: 'great-circle'})
      }) as never
    )
  ).toThrow(/exclusive/);
});

it('addHotSpotAnalysisRecipe labels distance-band weights binary or row', () => {
  const context = {coordinateSpace: 'longitude-latitude', metric: 'great-circle'} as const;
  const build = (rowStandardize?: boolean) => {
    const g = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'hot-spot-band-transform'});
    return addHotSpotAnalysisRecipe(
      g,
      createProps(g, {
        distanceBand: {
          spatialContext: context,
          neighborSearchParameters: createTransientView(g, 'band-parameters', 'float32', 16),
          gridSize: [8, 8] as const,
          neighborCapacity: 256,
          rowStandardize
        }
      }) as never
    );
  };
  expect(build().weights.information?.transform).toBe('binary');
  expect(build(false).weights.information?.transform).toBe('binary');
  const row = build(true);
  expect(row.weights.information?.transform).toBe('row');
  expect(row.weights.information?.provenance?.criterion).toBe('distance-band');
  expect(row.weights.status?.overflow).toBeDefined();
  expect((row.intermediates['weights'] as typeof row.weights).information?.transform).toBe('row');
});
