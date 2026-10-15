// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';
import {S2_DATASETS} from './s2-datasets';

registerLensScenario({
  id: 'gpu-spatial-analysis.random-points-in-polygon',
  entry: 'gpu-spatial-analysis',
  element: 'GPURandomPointsInPolygon',
  title: 'Seeded points in Chicago community areas',
  audience: 'analyst',
  rungs: ['result', 'datum', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: S2_DATASETS.areas,
  load: async () => {
    const [
      {runS2Lens, createS2ChicagoFrame, projectS2Positions, closeS2Rings},
      {runExample},
      data
    ] = await Promise.all([
      import('./s2-lens-run'),
      import('./random-points-in-polygon.example'),
      import('../../data/s2-chicago-north-areas.json').then(module => module.default)
    ]);
    return {
      run: device => {
        const frame = createS2ChicagoFrame('lambert-equal-area');
        const projected = projectS2Positions(frame, data.polygons.vertices);
        const geometry = closeS2Rings(projected.positions, data.polygons.ringOffsets);
        const featureCount = data.polygons.featureOffsets.length - 1;
        const capacity = featureCount * 20;
        return runS2Lens(device, {
          scenarioId: 'gpu-spatial-analysis.random-points-in-polygon',
          element: 'GPURandomPointsInPolygon',
          datasets: ['s2-chicago-north-areas'],
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
            {id: 'feature', label: 'community area', size: featureCount, indexSpace: '1d'},
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
          choices: {seed: 2026},
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
            const counts = uploads.view('counts', new Uint32Array(featureCount).fill(20), 'uint32');
            const result = await runExample(graph, {
              geometry: {positions, ringOffsets, featureOffsets, polygonOffsets},
              counts,
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
                  offsetsCapture: 'ringOffsets'
                }
              },
              ringOffsets: {view: ringOffsets, role: 'input', keySpace: 'boundary'},
              featureOffsets: {view: featureOffsets, role: 'input', keySpace: 'boundary'},
              polygonOffsets: {view: polygonOffsets, role: 'input', keySpace: 'boundary'},
              counts: {view: counts, role: 'input', keySpace: 'feature'},
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
