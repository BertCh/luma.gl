// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, expectTypeOf, it} from 'vitest';
import * as terrain from '../../src/gpu-terrain/index';
import * as terrainFeatures from '../../src/gpu-terrain/terrain-features/index';
import * as prominence from '../../src/gpu-terrain/terrain-features/gpu-terrain-peak-prominence';

it('exports the public prominence contributor and helpers through both terrain barrels', () => {
  expect(prominence.GPUTerrainPeakProminence).toBeTypeOf('function');
  expect(prominence.getGPUTerrainPeakProminenceMaximumCellCount).toBeTypeOf('function');
  for (const [exportName, value] of Object.entries(prominence)) {
    expect(terrainFeatures).toHaveProperty(exportName, value);
    expect(terrain).toHaveProperty(exportName, value);
  }
});

it('exports prominence props and output types through both terrain barrels', () => {
  expectTypeOf<terrain.GPUTerrainPeakProminenceProps>().toEqualTypeOf<prominence.GPUTerrainPeakProminenceProps>();
  expectTypeOf<terrain.GPUTerrainPeakProminenceOutput>().toEqualTypeOf<prominence.GPUTerrainPeakProminenceOutput>();
  expectTypeOf<terrainFeatures.GPUTerrainPeakProminenceProps>().toEqualTypeOf<prominence.GPUTerrainPeakProminenceProps>();
  expectTypeOf<terrainFeatures.GPUTerrainPeakProminenceOutput>().toEqualTypeOf<prominence.GPUTerrainPeakProminenceOutput>();
});
