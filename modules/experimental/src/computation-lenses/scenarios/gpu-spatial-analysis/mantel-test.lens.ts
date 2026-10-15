// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';
import {S2_DATASETS} from './s2-datasets';

/** Every k-th observation forms the deterministic subsample (2344 / 8 = 293 observations). */
const SUBSAMPLE_STRIDE = 8;

registerLensScenario({
  id: 'gpu-spatial-analysis.mantel-test',
  entry: 'gpu-spatial-analysis',
  element: 'GPUMantelTest',
  title: 'Mantel test of an iNaturalist subsample, Chicago North Side 2023',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: S2_DATASETS.nature,
  load: async () => {
    const [{runS2Lens, createS2ChicagoFrame, projectS2Positions}, {runExample}, data] =
      await Promise.all([
        import('./s2-lens-run'),
        import('./mantel-test.example'),
        import('../../data/s2-chicago-north-nature.json').then(module => module.default)
      ]);
    return {
      run: device => {
        const frame = createS2ChicagoFrame();
        const lngLats: number[] = [];
        const sampledDays: number[] = [];
        for (let row = 0; row < data.day.length; row += SUBSAMPLE_STRIDE) {
          lngLats.push(data.positions[2 * row], data.positions[2 * row + 1]);
          sampledDays.push(data.day[row]);
        }
        const projected = projectS2Positions(frame, lngLats, 500);
        const count = sampledDays.length;
        const [minX, minY, maxX, maxY] = projected.extent;
        // A band as long as the diagonal of the padded bounds lists every pair.
        const band = Math.ceil(Math.hypot(maxX - minX, maxY - minY)) + 1;
        const pairSlots = count * (count - 1);
        return runS2Lens(device, {
          scenarioId: 'gpu-spatial-analysis.mantel-test',
          element: 'GPUMantelTest',
          datasets: ['s2-chicago-north-nature'],
          // The all-pairs neighbour and distance slots (85k each) are not captured: no view of this
          // element reads them (result, datum and formula read the permutation statistics and
          // summary), and they were 0.9 MB of the record. Their row offsets are kept.
          retain: ['pairOffsets', 'statistics', 'summary'],
          seed: 1,
          keySpaces: [
            {id: 'permutation', label: 'permutation (0: observed)', size: 200, indexSpace: '1d'},
            {id: 'summary', label: 'summary statistic', size: 12, indexSpace: '1d'},
            {id: 'observation', label: 'sampled observation', size: count, indexSpace: '1d'},
            {id: 'pair-offset', label: 'pair row offset', size: count + 1, indexSpace: '1d'},
            {id: 'pair-slot', label: 'pair slot', size: pairSlots, indexSpace: '1d'}
          ],
          choices: {radius: band, permutationCount: 199, seed: 1},
          async build(graph, uploads) {
            const observations = uploads.view('observations', projected.positions, 'float32x2');
            const days = uploads.view('days', Float32Array.from(sampledDays), 'float32');
            const result = await runExample(graph, {
              observations,
              days,
              spatialContext: frame.spatialContext,
              bounds: projected.extent,
              band
            });
            result.parameterBuffers.forEach(buffer => uploads.own(buffer));
            return {
              observations: {
                view: observations,
                role: 'input',
                keySpace: 'observation',
                geometry: {kind: 'points', coordinateModel: 'projected', unit: 'm'},
                meaning: `Every ${SUBSAMPLE_STRIDE}th observation (${count} of ${data.day.length})`
              },
              days: {
                view: days,
                role: 'input',
                keySpace: 'observation',
                meaning: 'Day of 2023 (0 = 1 January)'
              },
              pairOffsets: {
                view: result.everyPair.offsets,
                role: 'intermediate',
                keySpace: 'pair-offset'
              },
              statistics: {view: result.statistics, role: 'output', keySpace: 'permutation'},
              summary: {
                view: result.summary,
                role: 'output',
                keySpace: 'summary',
                meaning:
                  'observed r, pair count, (unused), (unused), permutation mean and variance, greater and lesser counts, p greater, p lesser, p two-sided, z'
              }
            };
          }
        });
      }
    };
  }
});
