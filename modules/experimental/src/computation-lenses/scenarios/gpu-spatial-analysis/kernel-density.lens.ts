// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';
import {S2_DATASETS} from './s2-datasets';

registerLensScenario({
  id: 'gpu-spatial-analysis.kernel-density',
  entry: 'gpu-spatial-analysis',
  element: 'GPUKernelDensity',
  title: 'Kernel density of iNaturalist observations, Chicago North Side',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: S2_DATASETS.nature,
  load: async () => {
    const [
      {runS2Lens, createS2ChicagoFrame, projectS2Positions, snapS2Extent},
      {runExample},
      data
    ] = await Promise.all([
      import('./s2-lens-run'),
      import('./kernel-density.example'),
      import('../../data/s2-chicago-north-nature.json').then(module => module.default)
    ]);
    return {
      run: device => {
        // Density is per unit area, so the frame must preserve area (the library guard requires it).
        const frame = createS2ChicagoFrame('lambert-equal-area');
        const projected = projectS2Positions(frame, data.positions, 500);
        const extent = snapS2Extent(projected.extent, 125);
        const count = projected.positions.length / 2;
        const width = Math.round((extent[2] - extent[0]) / 125);
        const height = Math.round((extent[3] - extent[1]) / 125);
        return runS2Lens(device, {
          scenarioId: 'gpu-spatial-analysis.kernel-density',
          element: 'GPUKernelDensity',
          datasets: ['s2-chicago-north-nature'],
          retain: ['density'],
          keySpaces: [
            {
              id: 'cell',
              label: 'grid cell',
              size: width * height,
              indexSpace: '2d',
              shape: [width, height]
            },
            {id: 'observation', label: 'observation', size: count, indexSpace: '1d'}
          ],
          choices: {bandwidth: 400, kernel: 'gaussian', edgeCorrection: 'uniform', cellSize: 125},
          async build(graph, uploads) {
            const observations = uploads.view('observations', projected.positions, 'float32x2');
            const {density, parameters} = await runExample(graph, {
              observations,
              spatialContext: frame.spatialContext,
              extent
            });
            uploads.own(parameters);
            return {
              observations: {
                view: observations,
                role: 'input',
                keySpace: 'observation',
                geometry: {kind: 'points', coordinateModel: 'projected', unit: 'm'}
              },
              density: {
                view: density,
                role: 'output',
                keySpace: 'cell',
                geometry: {
                  kind: 'raster',
                  shape: [width, height],
                  coordinateModel: 'projected',
                  cellSize: 125,
                  unit: 'm',
                  origin: [extent[0], extent[3]],
                  rowDirection: 'south'
                }
              }
            };
          }
        });
      }
    };
  }
});
