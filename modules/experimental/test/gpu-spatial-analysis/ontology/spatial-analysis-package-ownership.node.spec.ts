// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import * as spatialAnalysis from '../../../src/gpu-spatial-analysis/index';
import * as terrain from '../../../src/gpu-terrain/index';

it('resolves every ontology operator from its declared public entry', () => {
  const entryModules: Record<string, Record<string, unknown>> = {
    '@luma.gl/experimental/gpu-spatial-analysis': spatialAnalysis,
    '@luma.gl/experimental/gpu-terrain': terrain
  };

  for (const operator of spatialAnalysis.GPU_SPATIAL_ANALYSIS_OPERATORS) {
    expect(
      entryModules[operator.package]?.[operator.exportName],
      `${operator.package}: ${operator.exportName}`
    ).toBeTypeOf('function');
  }
});

it('discovers terrain prominence without exporting it from spatial analysis', () => {
  const operator = spatialAnalysis.getGPUSpatialAnalysisOperator('GPUTerrainPeakProminence');
  expect(operator?.package).toBe('@luma.gl/experimental/gpu-terrain');
  expect(terrain.GPUTerrainPeakProminence).toBeTypeOf('function');
  expect('GPUTerrainPeakProminence' in spatialAnalysis).toBe(false);
});
