// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {addDriveTimeCatchmentRecipe} from '../../../src/gpu-network/recipes/drive-time-catchment-recipe';
import {
  addFleetDwellRecipe,
  addFleetDwellZoneEventsRecipe
} from '../../../src/gpu-spatial-analysis/recipes/fleet-dwell-recipe';
import {addStraightLineCatchmentsRecipe} from '../../../src/gpu-spatial-analysis/recipes/straight-line-catchments-recipe';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

const LONGITUDE_LATITUDE = {coordinateSpace: 'longitude-latitude', metric: 'none'} as const;
const GREAT_CIRCLE = {coordinateSpace: 'longitude-latitude', metric: 'great-circle'} as const;

// The context is resolved before any other prop is read, so the unsupported cases throw without
// building a graph.
const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'movement-recipes-context'});
const noProps = {} as never;

it('planar-only movement recipes reject longitude-latitude contexts at construction', () => {
  for (const spatialContext of [LONGITUDE_LATITUDE, GREAT_CIRCLE]) {
    expect(() =>
      addStraightLineCatchmentsRecipe(graph, {spatialContext, ...(noProps as object)} as never)
    ).toThrow(/does not support longitude-latitude coordinates; project positions/);
    expect(() =>
      addFleetDwellRecipe(graph, {spatialContext, ...(noProps as object)} as never)
    ).toThrow(/does not support longitude-latitude coordinates; project positions/);
    expect(() =>
      addFleetDwellZoneEventsRecipe(graph, {spatialContext, ...(noProps as object)} as never)
    ).toThrow(/does not support longitude-latitude coordinates; project positions/);
  }
});

it('the drive-time recipe accepts longitude-latitude degrees but no rhumb or ellipsoidal metric', () => {
  // Great-circle is accepted (both snappings run great-circle on lon/lat); see
  // drive-time-catchment-recipe.node.spec.ts.
  expect(() =>
    addDriveTimeCatchmentRecipe(graph, {
      spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'rhumb'},
      ...(noProps as object)
    } as never)
  ).toThrow(/does not support the rhumb metric/);
  expect(() =>
    addDriveTimeCatchmentRecipe(graph, {
      spatialContext: {coordinateSpace: 'discrete-grid', metric: 'none', gridFamily: 'h3'},
      ...(noProps as object)
    } as never)
  ).toThrow(/discrete-grid/);
});
