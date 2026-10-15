// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {registerLensScenario} from '../../registry';
import type {LensScenarioResult} from '../../types';
import {T_DIXIE_BANDS} from '../gpu-raster/raster-lens-datasets';

const SCENARIO_ID = 'gpu-raster.raster-sieve';

/** USGS threshold of moderate-low burn severity; cells above it form the burn mask. */
const BURN_DNBR_THRESHOLD = 0.27;
/** Patch rows: the mask has 44 four-connected patches. */
const PATCH_CAPACITY = 48;

/**
 * Records GPURasterSieve on the Dixie Fire window: patches of the burn mask (dNBR > 0.27) smaller
 * than 12 cells are removed, which keeps 7 of 44 patches. The example file adds the labelling
 * nodes that precede the sieve (4-connected components, dense labels) before its shown region.
 */
export async function run(device: Device): Promise<LensScenarioResult> {
  const [
    {runExample},
    {runRasterLens},
    {loadDixieBands, T_DATA_VERSION},
    {getRasterOperatorEntries}
  ] = await Promise.all([
    import('./raster-sieve.example'),
    import('./raster-lens-harness'),
    import('./raster-lens-data'),
    import('../../../gpu-raster/computation-pack/index')
  ]);
  const bands = await loadDixieBands();
  const entry = getRasterOperatorEntries()['GPURasterSieve'];
  const getRatio = (nir: number, swir: number) =>
    nir > bands.noData && swir > bands.noData ? (nir - swir) / (nir + swir) : NaN;
  const burnMask = Uint32Array.from(bands.bands.nirBefore, (_, cell) => {
    const dnbr =
      getRatio(bands.bands.nirBefore[cell], bands.bands.swirBefore[cell]) -
      getRatio(bands.bands.nirAfter[cell], bands.bands.swirAfter[cell]);
    return dnbr > BURN_DNBR_THRESHOLD ? 1 : 0;
  });
  return runRasterLens({
    device,
    scenarioId: SCENARIO_ID,
    domain: 'gpu-raster',
    label: 'GPURasterSieve on the Dixie Fire window',
    grid: bands.grid,
    coordinateModel: bands.coordinateModel,
    unit: bands.unit,
    dataset: {id: T_DIXIE_BANDS.id, version: T_DATA_VERSION},
    formulas: 'tex' in entry.formula ? [entry.formula] : [],
    keySpaces: [{id: 'patches', label: 'burn patches', size: PATCH_CAPACITY, indexSpace: '1d'}],
    choices: [
      {stage: 'operator', name: 'minimumPixels', value: 12, sweepable: true},
      {stage: 'operator', name: 'mode', value: 'remove', sweepable: false},
      {stage: 'neighbourhood', name: 'connectivity', value: 4, sweepable: false}
    ],
    build: async session => {
      // The patch column is registered first so the datum rung pins a patch.
      const patchTargets = session.output(
        'sievePatchTargets',
        'uint32',
        {
          role: 'intermediate',
          keySpace: 'patches',
          meaning: 'column final label of each patch: itself when kept, 0 when removed'
        },
        PATCH_CAPACITY
      );
      const sievedCount = session.output(
        'sieveSievedCount',
        'uint32',
        {role: 'intermediate', meaning: 'patches smaller than minimumPixels'},
        1
      );
      const converged = session.output(
        'sieveConverged',
        'uint32',
        {role: 'intermediate', meaning: 'converged: raster connected components'},
        1
      );
      const burnMaskView = session.input('sieveBurnMask', 'uint32', burnMask, {
        role: 'context',
        geometry: 'raster',
        meaning: 'burn mask: 1 where dNBR > 0.27'
      });
      const labels = session.output('sieveLabels', 'uint32', {
        role: 'input',
        geometry: 'raster',
        meaning: 'dense patch label before sieving, 0 = background'
      });
      const sievedLabels = session.output('sieveOutputLabels', 'uint32', {
        role: 'output',
        geometry: 'raster',
        meaning: 'patch label after removing patches under 12 cells, 0 = background'
      });
      const sparseLabels = session.output('sieveSparseLabels', 'uint32', false);
      const sparseValidity = session.output('sieveSparseValidity', 'uint32', false);
      const labelValidity = session.output('sieveLabelValidity', 'uint32', false);
      const componentCount = session.output('sieveComponentCount', 'uint32', false, 1);
      const overflow = session.output('sieveOverflow', 'uint32', false, 1);
      const {grid} = bands;
      const {parameters} = await runExample(device, {
        graph: session.graph,
        grid,
        burnMask: burnMaskView,
        sparseLabels,
        sparseValidity,
        labelValidity,
        converged,
        labels,
        componentCount,
        overflow,
        sievedLabels,
        patchTargets,
        sievedCount
      });
      session.track(parameters);
    }
  });
}

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-raster',
  element: 'GPURasterSieve',
  title: 'RasterSieve',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph', 'table'],
  defaultRung: 'result',
  dataset: T_DIXIE_BANDS,
  load: async () => ({run})
});
