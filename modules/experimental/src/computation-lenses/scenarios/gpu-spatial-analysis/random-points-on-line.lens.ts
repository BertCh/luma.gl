// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';
import {S2_DATASETS} from './s2-datasets';

registerLensScenario({
  id: 'gpu-spatial-analysis.random-points-on-line',
  entry: 'gpu-spatial-analysis',
  element: 'GPURandomPointsOnLine',
  title: 'Seeded points on Chicago road edges',
  audience: 'analyst',
  rungs: ['result', 'datum', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: S2_DATASETS.roads,
  load: async () => {
    const [{runS2Lens, createS2ChicagoFrame, projectS2Positions}, {runExample}, data] =
      await Promise.all([
        import('./s2-lens-run'),
        import('./random-points-on-line.example'),
        import('../../data/s2-chicago-north-roads.json').then(module => module.default)
      ]);
    return {
      run: device => {
        const frame = createS2ChicagoFrame('azimuthal-equidistant');
        const projected = projectS2Positions(frame, data.lines.vertices);
        const geometry = {
          positions: projected.positions,
          ringOffsets: Uint32Array.from(data.lines.offsets)
        };
        const featureCount = data.lines.offsets.length - 1;
        const capacity = featureCount * 2;
        return runS2Lens(device, {
          scenarioId: 'gpu-spatial-analysis.random-points-on-line',
          element: 'GPURandomPointsOnLine',
          datasets: ['s2-chicago-north-roads'],
          seed: 2026,
          retain: ['points', 'source-ids', 'count', 'overflow', 'required-count', 'fractions'],
          keySpaces: [
            {id: 'point', label: 'sample slot', size: capacity, indexSpace: '1d'},
            {id: 'feature', label: 'road edge', size: featureCount, indexSpace: '1d'},
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

            const counts = uploads.view('counts', new Uint32Array(featureCount).fill(2), 'uint32');
            const result = await runExample(graph, {
              geometry: {positions, lineOffsets: ringOffsets},
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
                  kind: 'lines',
                  coordinateModel: 'projected',
                  unit: 'm',
                  offsetsCapture: 'ringOffsets'
                }
              },
              ringOffsets: {view: ringOffsets, role: 'input', keySpace: 'boundary'},

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
              fractions: {view: result.fractions, role: 'output', keySpace: 'point'}
            };
          }
        });
      }
    };
  }
});
