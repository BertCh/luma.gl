// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';
import {S2_DATASETS} from './s2-datasets';

registerLensScenario({
  id: 'gpu-spatial-analysis.lisa-markov',
  entry: 'gpu-spatial-analysis',
  element: 'GPULISAMarkov',
  title: 'LISA Markov of county unemployment, Midwest 2008-2023',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: S2_DATASETS.counties,
  load: async () => {
    const [{runS2Lens, getS2PeriodMajorColumn, uploadS2QueenWeights}, {runExample}, counties] =
      await Promise.all([
        import('./s2-lens-run'),
        import('./lisa-markov.example'),
        import('../../data/s2-midwest-counties.json').then(module => module.default)
      ]);
    return {
      run: device => {
        const periodCount = counties.unemploymentRate.years.length;
        const countyCount = counties.state.length;
        const unemploymentByYear = getS2PeriodMajorColumn(counties.unemploymentRate.values);
        return runS2Lens(device, {
          scenarioId: 'gpu-spatial-analysis.lisa-markov',
          element: 'GPULISAMarkov',
          datasets: ['s2-midwest-counties'],
          retain: ['counts', 'probabilities', 'rowTotals', 'ignored', 'quadrants'],
          keySpaces: [
            {id: 'transition', label: 'from state x to state', size: 25, indexSpace: '1d'},
            {id: 'state', label: 'LISA state (from)', size: 5, indexSpace: '1d'},
            {id: 'status', label: 'ignored transitions', size: 1, indexSpace: '1d'},
            {
              id: 'queen-offset',
              label: 'queen row offset',
              size: countyCount + 1,
              indexSpace: '1d'
            },
            {
              id: 'county-year',
              label: 'county and year (period-major)',
              size: unemploymentByYear.length,
              indexSpace: '1d'
            }
          ],
          // Quadrants without significance gating: no significance test enters the states.
          choices: {weightsStandardisation: 'row', gating: 'none', significanceAvailable: false},
          async build(graph, uploads) {
            const unemployment = uploads.view('unemployment', unemploymentByYear, 'float32');
            const queen = uploadS2QueenWeights(uploads, counties.queen);
            const result = await runExample(graph, {unemployment, queen, periodCount});
            uploads.own(result.moranParameters);
            return {
              unemployment: {
                view: unemployment,
                role: 'input',
                keySpace: 'county-year',
                meaning: 'Unemployment rate (%), 2008-2023, period-major'
              },
              queenOffsets: {
                view: queen.offsets,
                role: 'input',
                keySpace: 'queen-offset',
                meaning: 'Queen contiguity CSR row offsets (weights row-standardised)'
              },
              quadrants: {
                view: result.quadrants,
                role: 'intermediate',
                keySpace: 'county-year',
                meaning: 'LISA state per county and year (0xffffffff: no state)'
              },
              counts: {view: result.counts, role: 'output', keySpace: 'transition'},
              probabilities: {view: result.probabilities, role: 'output', keySpace: 'transition'},
              rowTotals: {view: result.rowTotals, role: 'output', keySpace: 'state'},
              ignored: {view: result.ignored, role: 'output', keySpace: 'status'}
            };
          }
        });
      }
    };
  }
});
