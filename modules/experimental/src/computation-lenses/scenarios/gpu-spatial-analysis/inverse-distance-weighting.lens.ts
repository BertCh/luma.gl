// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';
import {S2_DATASETS} from './s2-datasets';

registerLensScenario({
  id: 'gpu-spatial-analysis.inverse-distance-weighting',
  entry: 'gpu-spatial-analysis',
  element: 'GPUInverseDistanceWeighting',
  title: 'Helene rainfall inverse distance weighting',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: S2_DATASETS.heleneRain,
  load: async () => {
    const [
      {runS2Lens, projectS2Positions, snapS2Extent},
      {runExample},
      {createGPULocalFrame},
      data
    ] = await Promise.all([
      import('./s2-lens-run'),
      import('./inverse-distance-weighting.example'),
      import('@luma.gl/experimental/gpu-spatial-analysis'),
      import('../../data/s2-helene-rain.json').then(module => module.default)
    ]);
    return {
      run: device => {
        const frame = createGPULocalFrame({origin: [-82, 34.75]});
        const projected = projectS2Positions(frame, data.positions);
        const extent = snapS2Extent(projected.extent, 10000);
        const width = Math.round((extent[2] - extent[0]) / 10000);
        const height = Math.round((extent[3] - extent[1]) / 10000);
        return runS2Lens(device, {
          scenarioId: 'gpu-spatial-analysis.inverse-distance-weighting',
          element: 'GPUInverseDistanceWeighting',
          datasets: ['s2-helene-rain'],
          retain: ['rainfall', 'counts'],
          keySpaces: [
            {
              id: 'cell',
              label: '10 km cell',
              size: width * height,
              indexSpace: '2d',
              shape: [width, height]
            },
            {id: 'station', label: 'rainfall station', size: data.prcpMm.length, indexSpace: '1d'},
            {id: 'run', label: 'whole raster', size: 1, indexSpace: '1d'}
          ],
          choices: {
            searchRadius: 150000,
            neighborCount: 8,
            minimumNeighborCount: 3,
            power: 2,
            taper: 'none'
          },
          async build(graph, uploads) {
            const stations = uploads.view('stations', projected.positions, 'float32x2');
            const precipitation = uploads.view(
              'precipitation',
              Float32Array.from(data.prcpMm),
              'float32'
            );
            const result = await runExample(graph, {
              stations,
              precipitation,
              extent,
              spatialContext: frame.spatialContext
            });
            uploads.own(result.parameters);
            return {
              stations: {
                view: stations,
                role: 'input',
                keySpace: 'station',
                geometry: {
                  kind: 'points',
                  coordinateModel: 'projected',
                  unit: 'm',
                  valuesCapture: 'precipitation'
                }
              },
              precipitation: {
                view: precipitation,
                role: 'input',
                keySpace: 'station',
                meaning: 'daily precipitation in mm'
              },
              rainfall: {
                view: result.rainfall,
                role: 'output',
                keySpace: 'cell',
                geometry: {
                  kind: 'raster',
                  shape: [width, height],
                  coordinateModel: 'projected',
                  cellSize: 10000,
                  unit: 'm',
                  origin: [extent[0], extent[3]],
                  rowDirection: 'south'
                }
              },
              counts: {view: result.counts, role: 'output', keySpace: 'cell'}
            };
          }
        });
      }
    };
  }
});
