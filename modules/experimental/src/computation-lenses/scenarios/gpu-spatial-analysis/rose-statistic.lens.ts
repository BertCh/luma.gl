// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';
import {S2_DATASETS} from './s2-datasets';

registerLensScenario({
  id: 'gpu-spatial-analysis.rose-statistic',
  entry: 'gpu-spatial-analysis',
  element: 'GPURoseStatistic',
  title: 'Street directions of Lincoln Park and Lake View as an axial rose',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: S2_DATASETS.roads,
  load: async () => {
    const [{runS2Lens, createS2ChicagoFrame}, {runExample, ROSE_BIN_COUNT}, roads] =
      await Promise.all([
        import('./s2-lens-run'),
        import('./rose-statistic.example'),
        import('../../data/s2-chicago-north-roads.json').then(module => module.default)
      ]);
    // Straight segments between consecutive polyline vertices, projected to local metres.
    const frame = createS2ChicagoFrame();
    const projected = frame.projectArray(Float64Array.from(roads.lines.vertices));
    const startList: number[] = [];
    const endList: number[] = [];
    const lengthList: number[] = [];
    for (let line = 0; line + 1 < roads.lines.offsets.length; line++) {
      for (
        let vertex = roads.lines.offsets[line];
        vertex + 1 < roads.lines.offsets[line + 1];
        vertex++
      ) {
        const [startX, startY, endX, endY] = [
          projected[2 * vertex],
          projected[2 * vertex + 1],
          projected[2 * vertex + 2],
          projected[2 * vertex + 3]
        ];
        startList.push(startX, startY);
        endList.push(endX, endY);
        lengthList.push(Math.hypot(endX - startX, endY - startY));
      }
    }
    const count = lengthList.length;
    return {
      run: device =>
        runS2Lens(device, {
          scenarioId: 'gpu-spatial-analysis.rose-statistic',
          element: 'GPURoseStatistic',
          datasets: ['s2-chicago-north-roads'],
          retain: ['summary', 'bins'],
          keySpaces: [
            {id: 'segment', label: 'street segment', size: count, indexSpace: '1d'},
            {id: 'summary', label: 'rose summary word', size: 5, indexSpace: '1d'},
            {id: 'sector', label: '10 degree sector', size: ROSE_BIN_COUNT, indexSpace: '1d'}
          ],
          choices: {mode: 'axial', binCount: ROSE_BIN_COUNT, rayleighSampling: 'one-per-segment'},
          async build(graph, uploads) {
            const starts = uploads.view('starts', Float32Array.from(startList), 'float32x2');
            const ends = uploads.view('ends', Float32Array.from(endList), 'float32x2');
            const weights = uploads.view('weights', Float32Array.from(lengthList), 'float32');
            const {summary, bins} = await runExample(graph, {
              starts,
              ends,
              weights,
              spatialContext: frame.spatialContext
            });
            return {
              starts: {view: starts, role: 'intermediate', keySpace: 'segment'},
              ends: {view: ends, role: 'intermediate', keySpace: 'segment'},
              weights: {
                view: weights,
                role: 'intermediate',
                keySpace: 'segment',
                meaning: 'Segment length (m), the weight of each direction'
              },
              summary: {
                view: summary,
                role: 'output',
                keySpace: 'summary',
                meaning:
                  'Count, length-weighted mean axial angle (rad from east), weighted resultant length and circular variance, unweighted segment Rayleigh z'
              },
              bins: {
                view: bins,
                role: 'output',
                keySpace: 'sector',
                meaning: 'Street length (m) per 10 degree sector, counter-clockwise from east'
              }
            };
          }
        })
    };
  }
});
