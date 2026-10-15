// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {registerLensScenario} from '../../registry';
import type {LensScenarioResult} from '../../types';
import {T_DIXIE_TERRARIUM} from '../gpu-raster/raster-lens-datasets';

const SCENARIO_ID = 'gpu-terrain.terrain-rgb-decode';

/** Runs GPUTerrainRGBDecode on the recorded Dixie terrain inputs. */
export async function run(device: Device): Promise<LensScenarioResult> {
  const [
    {runExample},
    {runRasterLens},
    {loadDixieTerrarium, T_DATA_VERSION},
    {getTerrainOperatorEntries}
  ] = await Promise.all([
    import('./terrain-rgb-decode.example'),
    import('../gpu-raster/raster-lens-harness'),
    import('../gpu-raster/raster-lens-data'),
    import('../../../gpu-terrain/computation-pack/index')
  ]);
  const data = await loadDixieTerrarium();
  const entry = getTerrainOperatorEntries()['GPUTerrainRGBDecode'];
  return runRasterLens({
    device,
    scenarioId: SCENARIO_ID,
    domain: 'gpu-terrain',
    label: 'Terrarium bytes to elevation in metres',
    grid: data.grid,
    coordinateModel: data.coordinateModel,
    unit: data.unit,
    dataset: {id: T_DIXIE_TERRARIUM.id, version: T_DATA_VERSION},
    formulas: 'tex' in entry.formula ? [entry.formula] : [],
    choices: [
      {
        stage: 'representation',
        name: 'encoding',
        value: 'terrarium',
        alternatives: ['terrarium', 'mapbox'],
        sweepable: false
      },
      {stage: 'operator', name: 'alphaNoData', value: true, sweepable: false},
      {stage: 'operator', name: 'clampBathymetry', value: false, sweepable: false}
    ],
    build: async session => {
      // Explicit little-endian RGBA packing preserves the raw bytes on every host.
      const words = Uint32Array.from({length: data.grid.width * data.grid.height}, (_, index) => {
        const offset = index * 4;
        return (
          (data.rgba[offset] |
            (data.rgba[offset + 1] << 8) |
            (data.rgba[offset + 2] << 16) |
            (data.rgba[offset + 3] << 24)) >>>
          0
        );
      });
      const terrariumRGBA = session.input('terrariumRGBA', 'uint32', words, {
        role: 'input',
        geometry: 'raster',
        meaning: 'raw Terrarium RGBA packed little-endian (four bytes per cell)'
      });
      const decodedElevation = session.output('decodedElevation', 'float32', {
        role: 'output',
        geometry: 'raster',
        meaning: 'decoded elevation (m) on the 32×32, 20 m Terrarium grid'
      });
      const decodedValidity = session.output('decodedValidity', 'uint32', {
        role: 'intermediate',
        geometry: 'raster',
        meaning: 'decoded elevation validity (0 or 1)'
      });
      await runExample(device, {
        graph: session.graph,
        grid: data.grid,
        terrariumRGBA,
        decodedElevation,
        decodedValidity
      });
    }
  });
}

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-terrain',
  element: 'GPUTerrainRGBDecode',
  title: 'Terrarium bytes to elevation in metres',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph', 'table'],
  defaultRung: 'result',
  dataset: T_DIXIE_TERRARIUM,
  load: async () => ({run})
});
