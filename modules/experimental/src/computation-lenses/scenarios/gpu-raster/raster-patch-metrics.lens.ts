// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {registerLensScenario} from '../../registry';
import type {LensScenarioResult} from '../../types';
import {T_DIXIE_BANDS} from '../gpu-raster/raster-lens-datasets';

const SCENARIO_ID = 'gpu-raster.raster-patch-metrics';

/** USGS threshold of moderate-low burn severity; cells above it form the burn mask. */
const BURN_DNBR_THRESHOLD = 0.27;
/** Patch rows: the mask has 44 four-connected patches. */
const PATCH_CAPACITY = 48;

/**
 * Records GPURasterPatchMetrics on the Dixie Fire window. The input is the burn mask
 * dNBR > 0.27 (NBR = (NIR - SWIR) / (NIR + SWIR), before minus after), derived on the CPU and
 * labelled on the GPU with 4-connected components and dense labels.
 */
export async function run(device: Device): Promise<LensScenarioResult> {
  const [
    {runExample},
    {runRasterLens},
    {loadDixieBands, T_DATA_VERSION},
    {getRasterOperatorEntries}
  ] = await Promise.all([
    import('./raster-patch-metrics.example'),
    import('./raster-lens-harness'),
    import('./raster-lens-data'),
    import('../../../gpu-raster/computation-pack/index')
  ]);
  const bands = await loadDixieBands();
  const entry = getRasterOperatorEntries()['GPURasterPatchMetrics'];
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
    label: 'GPURasterPatchMetrics on the Dixie Fire window',
    grid: bands.grid,
    coordinateModel: bands.coordinateModel,
    unit: bands.unit,
    dataset: {id: T_DIXIE_BANDS.id, version: T_DATA_VERSION},
    formulas: 'tex' in entry.formula ? [entry.formula] : [],
    keySpaces: [{id: 'patches', label: 'burn patches', size: PATCH_CAPACITY, indexSpace: '1d'}],
    choices: [
      {stage: 'neighbourhood', name: 'connectivity', value: 4, sweepable: false},
      {stage: 'operator', name: 'countRasterBorder', value: true, sweepable: false}
    ],
    build: async session => {
      // Patch columns first so the datum rung pins a patch.
      const column = <Format extends 'uint32' | 'float32'>(
        name: string,
        format: Format,
        meaning: string
      ) =>
        session.output(
          name,
          format,
          {role: 'output', keySpace: 'patches', meaning: `column ${meaning}`},
          PATCH_CAPACITY
        );
      const pixelCounts = column('patchPixelCounts', 'uint32', 'pixels per patch');
      const areas = column('patchAreas', 'float32', 'area per patch (m2)');
      const perimeters = column('patchPerimeters', 'float32', 'perimeter per patch (m)');
      const perimeterFaceCounts = column(
        'patchPerimeterFaces',
        'uint32',
        'exposed pixel faces per patch'
      );
      const converged = session.output(
        'patchConverged',
        'uint32',
        {role: 'intermediate', meaning: 'converged: raster connected components'},
        1
      );
      const burnMaskView = session.input('patchBurnMask', 'uint32', burnMask, {
        role: 'context',
        geometry: 'raster',
        meaning: 'burn mask: 1 where dNBR > 0.27'
      });
      const labels = session.output('patchLabels', 'uint32', {
        role: 'input',
        geometry: 'raster',
        meaning: 'dense patch label, 0 = background (4-connected clumps of the burn mask)'
      });
      const sparseLabels = session.output('patchSparseLabels', 'uint32', false);
      const sparseValidity = session.output('patchSparseValidity', 'uint32', false);
      const labelValidity = session.output('patchLabelValidity', 'uint32', false);
      const componentCount = session.output('patchComponentCount', 'uint32', false, 1);
      const overflow = session.output('patchOverflow', 'uint32', false, 1);
      await runExample(device, {
        graph: session.graph,
        grid: bands.grid,
        burnMask: burnMaskView,
        sparseLabels,
        sparseValidity,
        converged,
        labels,
        labelValidity,
        componentCount,
        overflow,
        pixelCounts,
        areas,
        perimeters,
        perimeterFaceCounts
      });
    }
  });
}

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-raster',
  element: 'GPURasterPatchMetrics',
  title: 'RasterPatchMetrics',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph', 'table'],
  defaultRung: 'result',
  dataset: T_DIXIE_BANDS,
  load: async () => ({run})
});
