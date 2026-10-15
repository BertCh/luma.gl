// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';
import {S2_DATASETS} from './s2-datasets';

registerLensScenario({
  id: 'gpu-spatial-analysis.dot-density',
  entry: 'gpu-spatial-analysis',
  element: 'GPUDotDensity',
  title: 'Seeded population dots in Chicago tracts',
  audience: 'analyst',
  rungs: ['result', 'datum', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: S2_DATASETS.tracts,
  load: async () => {
    const [
      {runS2Lens, createS2ChicagoFrame, projectS2Positions, closeS2Rings},
      {runExample},
      data
    ] = await Promise.all([
      import('./s2-lens-run'),
      import('./dot-density.example'),
      import('../../data/s2-chicago-north-tracts.json').then(module => module.default)
    ]);
    return {
      run: device => {
        const frame = createS2ChicagoFrame('lambert-equal-area');
        const projected = projectS2Positions(frame, data.polygons.vertices);
        const geometry = closeS2Rings(projected.positions, data.polygons.ringOffsets);
        const featureCount = data.polygons.featureOffsets.length - 1;
        const capacity = data.population.reduce(
          (total, value) => total + Math.ceil(value / 500),
          0
        );
        return runS2Lens(device, {
          scenarioId: 'gpu-spatial-analysis.dot-density',
          element: 'GPUDotDensity',
          datasets: ['s2-chicago-north-tracts'],
          seed: 2026,
          retain: [
            'points',
            'source-ids',
            'count',
            'overflow',
            'required-count',
            'slot-counts',
            'failed-count'
          ],
          keySpaces: [
            {id: 'feature', label: 'census tract', size: featureCount, indexSpace: '1d'},
            {id: 'point', label: 'sample slot', size: capacity, indexSpace: '1d'},
            {
              id: 'vertex',
              label: 'source vertex',
              size: geometry.positions.length / 2,
              indexSpace: '1d'
            },
            {id: 'boundary', label: 'path boundary', size: featureCount + 1, indexSpace: '1d'},
            {id: 'run', label: 'sampling status', size: 1, indexSpace: '1d'}
          ],
          choices: {seed: 2026, dotsPerUnit: 1 / 500},
          async build(graph, uploads) {
            const positions = uploads.view('positions', geometry.positions, 'float32x2');
            const ringOffsets = uploads.view('ring-offsets', geometry.ringOffsets, 'uint32');
            const featureOffsets = uploads.view(
              'feature-offsets',
              Uint32Array.from(data.polygons.featureOffsets),
              'uint32'
            );
            const polygonOffsets = uploads.view(
              'polygon-offsets',
              Uint32Array.from({length: featureCount + 1}, (_, index) => index),
              'uint32'
            );
            const population = uploads.view(
              'population',
              Float32Array.from(data.population),
              'float32'
            );
            const result = await runExample(graph, {
              geometry: {positions, ringOffsets, featureOffsets, polygonOffsets},
              population,
              spatialContext: frame.spatialContext,
              capacity
            });
            uploads.own(result.parameters);
            return {
              positions: {
                view: positions,
                role: 'input',
                keySpace: 'vertex',
                geometry: {
                  kind: 'polygons',
                  coordinateModel: 'projected',
                  unit: 'm',
                  offsetsCapture: 'ringOffsets',
                  valuesCapture: 'population'
                }
              },
              ringOffsets: {view: ringOffsets, role: 'input', keySpace: 'boundary'},
              featureOffsets: {view: featureOffsets, role: 'input', keySpace: 'boundary'},
              polygonOffsets: {view: polygonOffsets, role: 'input', keySpace: 'boundary'},
              population: {view: population, role: 'input', keySpace: 'feature'},
              points: {
                view: result.points,
                role: 'output',
                keySpace: 'point',
                geometry: {kind: 'points', coordinateModel: 'projected', unit: 'm'}
              },
              sourceIds: {view: result.sourceIds, role: 'output', keySpace: 'point'},
              count: {view: result.count, role: 'output', keySpace: 'run'},
              overflow: {view: result.overflow, role: 'output', keySpace: 'run'},
              requiredCount: {view: result.requiredCount, role: 'output', keySpace: 'run'},
              slotCounts: {view: result.slotCounts, role: 'output', keySpace: 'feature'},
              failedCount: {view: result.failedCount, role: 'output', keySpace: 'run'}
            };
          }
        });
      }
    };
  }
});
