// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';
import {S2_DATASETS} from './s2-datasets';

registerLensScenario({
  id: 'gpu-spatial-analysis.knox-test',
  entry: 'gpu-spatial-analysis',
  element: 'GPUKnoxTest',
  title: 'Knox test of iNaturalist observations, Chicago North Side 2023',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: S2_DATASETS.nature,
  load: async () => {
    const [{runS2Lens, createS2ChicagoFrame, projectS2Positions}, {runExample}, data] =
      await Promise.all([
        import('./s2-lens-run'),
        import('./knox-test.example'),
        import('../../data/s2-chicago-north-nature.json').then(module => module.default)
      ]);
    return {
      run: device => {
        const radius = 250;
        const frame = createS2ChicagoFrame();
        const projected = projectS2Positions(frame, data.positions, 500);
        const count = projected.positions.length / 2;
        // Directed pair slots within the radius, counted on the CPU to size the CSR (plus slack for f32 band edges).
        let slots = 0;
        for (let row = 0; row < count; row++) {
          for (let other = 0; other < count; other++) {
            const dx = projected.positions[2 * row] - projected.positions[2 * other];
            const dy = projected.positions[2 * row + 1] - projected.positions[2 * other + 1];
            if (other !== row && dx * dx + dy * dy <= radius * radius) slots++;
          }
        }
        const pairCapacity = slots + 1024;
        return runS2Lens(device, {
          scenarioId: 'gpu-spatial-analysis.knox-test',
          element: 'GPUKnoxTest',
          datasets: ['s2-chicago-north-nature'],
          // The ~460k close-pair neighbour slots are not captured: no view of this element reads
          // them (result, datum and formula read the permutation counts and summary), and they
          // alone were 1.8 MB of the record. Their row offsets (pairs per observation) are kept.
          retain: ['pairOffsets', 'statistics', 'summary'],
          seed: 1,
          keySpaces: [
            {id: 'permutation', label: 'permutation (0: observed)', size: 200, indexSpace: '1d'},
            {id: 'summary', label: 'summary statistic', size: 12, indexSpace: '1d'},
            {id: 'observation', label: 'observation', size: count, indexSpace: '1d'},
            {id: 'pair-offset', label: 'pair row offset', size: count + 1, indexSpace: '1d'},
            {
              id: 'pair-slot',
              label: 'close-in-space pair slot',
              size: pairCapacity,
              indexSpace: '1d'
            }
          ],
          choices: {radius, timeThreshold: 7, permutationCount: 199, seed: 1},
          async build(graph, uploads) {
            const observations = uploads.view('observations', projected.positions, 'float32x2');
            const days = uploads.view('days', Float32Array.from(data.day), 'float32');
            const result = await runExample(graph, {
              observations,
              days,
              spatialContext: frame.spatialContext,
              bounds: projected.extent,
              pairCapacity
            });
            result.parameterBuffers.forEach(buffer => uploads.own(buffer));
            return {
              observations: {
                view: observations,
                role: 'input',
                keySpace: 'observation',
                geometry: {kind: 'points', coordinateModel: 'projected', unit: 'm'}
              },
              days: {
                view: days,
                role: 'input',
                keySpace: 'observation',
                meaning: 'Day of 2023 (0 = 1 January)'
              },
              pairOffsets: {
                view: result.closeInSpace.offsets,
                role: 'intermediate',
                keySpace: 'pair-offset'
              },
              statistics: {view: result.statistics, role: 'output', keySpace: 'permutation'},
              summary: {
                view: result.summary,
                role: 'output',
                keySpace: 'summary',
                meaning:
                  'observed, spatial pairs, time-close pairs, expected, permutation mean and variance, greater and lesser counts, p greater, p lesser, p two-sided, z'
              }
            };
          }
        });
      }
    };
  }
});
