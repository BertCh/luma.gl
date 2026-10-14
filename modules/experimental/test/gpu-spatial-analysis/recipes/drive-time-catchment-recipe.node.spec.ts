// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {GPUNetworkSnapping} from '../../../src/gpu-network/network-accessibility/index';
import type {GPUSpatialContext} from '../../../src/gpu-spatial-analysis/contracts/spatial-context';
import {addDriveTimeCatchmentRecipe} from '../../../src/gpu-spatial-analysis/recipes/drive-time-catchment-recipe';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

let serial = 0;

function buildRecipe(spatialContext: GPUSpatialContext | undefined, candidateCapacity?: number) {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: `drive-time-${serial++}`});
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `drive-time-view-${serial++}`, format, length);
  const recipe = addDriveTimeCatchmentRecipe(graph, {
    spatialContext,
    network: {
      offsets: view('uint32', 5),
      neighbors: view('uint32', 8),
      weights: view('float32', 8),
      nodePositions: view('float32x2', 4)
    },
    facilities: view('float32x2', 2),
    demand: view('float32x2', 6),
    maxSnapDistance: view('float32', 1),
    candidateCapacity,
    bandBreaks: view('float32', 3)
  });
  const snappings = recipe.contributors.filter(
    (contributor): contributor is GPUNetworkSnapping => contributor instanceof GPUNetworkSnapping
  );
  return {recipe, snappings};
}

it('drive-time recipe snaps longitude/latitude positions great-circle, in meters', () => {
  // The explorer reported raw-degree snapping: Euclidean on lon/lat, about a 24% east-west bias at
  // its latitude. Either lon/lat metric now snaps great-circle; planar stays planar.
  for (const metric of ['none', 'great-circle'] as const) {
    const {recipe, snappings} = buildRecipe({coordinateSpace: 'longitude-latitude', metric});
    expect(recipe.spatialContext.coordinateSpace).toBe('longitude-latitude');
    expect(snappings).toHaveLength(2);
    for (const snapping of snappings) {
      expect(snapping.spatialContext).toMatchObject({
        coordinateSpace: 'longitude-latitude',
        metric: 'great-circle'
      });
    }
  }
  const {snappings: planar} = buildRecipe(undefined);
  for (const snapping of planar) {
    expect(snapping.spatialContext.coordinateSpace).toBe('planar');
  }
  // A great-circle context keeps its sphere radius.
  const {snappings: custom} = buildRecipe({
    coordinateSpace: 'longitude-latitude',
    metric: 'great-circle',
    sphereRadius: 6378137
  });
  expect(custom[0].spatialContext.sphereRadius).toBe(6378137);
  expect(() => buildRecipe({coordinateSpace: 'longitude-latitude', metric: 'rhumb'})).toThrow(
    /does not support the rhumb metric/
  );
});

it('drive-time recipe reports BVH snapping overflow per stage', () => {
  // Review B-08: both snappings took candidateCapacity but no overflow view, so a candidate
  // overflow mis-snapped facilities or demand with no signal.
  const {recipe, snappings} = buildRecipe(undefined, 16);
  expect(recipe.snapping).toBeDefined();
  const [facilities, demand] = snappings;
  expect(facilities.props.overflow).toBe(recipe.snapping!.facilitiesOverflow);
  expect(demand.props.overflow).toBe(recipe.snapping!.demandOverflow);
  expect(recipe.snapping!.facilitiesOverflow).not.toBe(recipe.snapping!.demandOverflow);
  const stages = Object.fromEntries(recipe.status.stages.map(stage => [stage.stage, stage.status]));
  expect(stages['snap-facilities']).toEqual({overflow: recipe.snapping!.facilitiesOverflow});
  expect(stages['snap-demand']).toEqual({overflow: recipe.snapping!.demandOverflow});

  // The exact all-edges scan cannot overflow, so it reports no snapping stage.
  const exact = buildRecipe(undefined);
  expect(exact.recipe.snapping).toBeUndefined();
  expect(exact.recipe.status.stages.map(stage => stage.stage)).not.toContain('snap-demand');
});

it('drive-time recipe shares one edge-source view across both snappings', () => {
  // Review B-04: the recipe's private copy of the edge-source kernel is gone; it uses the
  // gpu-network nodes that clear every row first.
  const {snappings} = buildRecipe(undefined);
  const [facilities, demand] = snappings;
  expect(facilities.props.edgeSources).toBeDefined();
  expect(facilities.props.edgeSources).toBe(demand.props.edgeSources);
  expect(facilities.props.edgeSources?.length).toBe(8);
});
