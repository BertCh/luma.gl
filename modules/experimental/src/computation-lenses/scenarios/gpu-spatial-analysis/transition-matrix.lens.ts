// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';
import {S2_DATASETS} from './s2-datasets';

registerLensScenario({
  id: 'gpu-spatial-analysis.transition-matrix',
  entry: 'gpu-spatial-analysis',
  element: 'GPUTransitionMatrix',
  title: 'Transitions of county unemployment classes, Midwest 2008-2023',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: S2_DATASETS.counties,
  load: async () => {
    const [{runS2Lens, getS2PeriodMajorColumn}, {runExample}, counties] = await Promise.all([
      import('./s2-lens-run'),
      import('./transition-matrix.example'),
      import('../../data/s2-midwest-counties.json').then(module => module.default)
    ]);
    return {
      run: device => {
        const periodCount = counties.unemploymentRate.years.length;
        const unemploymentByYear = getS2PeriodMajorColumn(counties.unemploymentRate.values);
        return runS2Lens(device, {
          scenarioId: 'gpu-spatial-analysis.transition-matrix',
          element: 'GPUTransitionMatrix',
          datasets: ['s2-midwest-counties'],
          retain: ['breaks', 'classes', 'counts', 'probabilities', 'rowTotals', 'ignored'],
          keySpaces: [
            {id: 'transition', label: 'from class x to class', size: 25, indexSpace: '1d'},
            {id: 'class', label: 'class (from)', size: 5, indexSpace: '1d'},
            {id: 'status', label: 'ignored transitions', size: 1, indexSpace: '1d'},
            {
              id: 'county-year',
              label: 'county and year (period-major)',
              size: unemploymentByYear.length,
              indexSpace: '1d'
            },
            {id: 'edge', label: 'class edge', size: 6, indexSpace: '1d'}
          ],
          choices: {classCount: 5, periodLag: 1},
          async build(graph, uploads) {
            const unemployment = uploads.view('unemployment', unemploymentByYear, 'float32');
            const result = await runExample(graph, {unemployment, periodCount});
            uploads.own(result.breakParameters);
            return {
              unemployment: {
                view: unemployment,
                role: 'input',
                keySpace: 'county-year',
                meaning: 'Unemployment rate (%), 2008-2023, period-major'
              },
              breaks: {view: result.breaks, role: 'intermediate', keySpace: 'edge'},
              classes: {view: result.classes, role: 'intermediate', keySpace: 'county-year'},
              counts: {view: result.counts, role: 'output', keySpace: 'transition'},
              probabilities: {view: result.probabilities, role: 'output', keySpace: 'transition'},
              rowTotals: {view: result.rowTotals, role: 'output', keySpace: 'class'},
              ignored: {view: result.ignored, role: 'output', keySpace: 'status'}
            };
          }
        });
      }
    };
  }
});
