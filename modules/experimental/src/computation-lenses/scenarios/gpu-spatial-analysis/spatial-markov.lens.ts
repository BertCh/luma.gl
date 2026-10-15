// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';
import {S2_DATASETS} from './s2-datasets';

registerLensScenario({
  id: 'gpu-spatial-analysis.spatial-markov',
  entry: 'gpu-spatial-analysis',
  element: 'GPUSpatialMarkov',
  title: 'Spatial Markov of county unemployment classes, Midwest 2008-2023',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: S2_DATASETS.counties,
  load: async () => {
    const [{runS2Lens, getS2PeriodMajorColumn, uploadS2QueenWeights}, {runExample}, counties] =
      await Promise.all([
        import('./s2-lens-run'),
        import('./spatial-markov.example'),
        import('../../data/s2-midwest-counties.json').then(module => module.default)
      ]);
    return {
      run: device => {
        const periodCount = counties.unemploymentRate.years.length;
        const countyCount = counties.state.length;
        const unemploymentByYear = getS2PeriodMajorColumn(counties.unemploymentRate.values);
        return runS2Lens(device, {
          scenarioId: 'gpu-spatial-analysis.spatial-markov',
          element: 'GPUSpatialMarkov',
          datasets: ['s2-midwest-counties'],
          retain: [
            'breaks',
            'classes',
            'counts',
            'probabilities',
            'rowTotals',
            'lagBreaks',
            'lagClassCount',
            'homogeneity',
            'stateSparsity'
          ],
          keySpaces: [
            {
              id: 'transition',
              label: 'lag class x from class x to class',
              size: 125,
              indexSpace: '1d'
            },
            {id: 'state', label: 'lag class x from class', size: 25, indexSpace: '1d'},
            {id: 'homogeneity', label: 'homogeneity statistic', size: 5, indexSpace: '1d'},
            {id: 'sparsity', label: 'sparsity count', size: 2, indexSpace: '1d'},
            {id: 'edge', label: 'class edge', size: 6, indexSpace: '1d'},
            {id: 'status', label: 'lag class count', size: 1, indexSpace: '1d'},
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
          choices: {
            weightsStandardisation: 'row',
            classCount: 5,
            lagClassCount: 5,
            lagMethod: 'quantile'
          },
          async build(graph, uploads) {
            const unemployment = uploads.view('unemployment', unemploymentByYear, 'float32');
            const queen = uploadS2QueenWeights(uploads, counties.queen);
            const result = await runExample(graph, {unemployment, queen, periodCount});
            uploads.own(result.breakParameters);
            uploads.own(result.lagParameters);
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
              breaks: {view: result.breaks, role: 'intermediate', keySpace: 'edge'},
              classes: {view: result.classes, role: 'intermediate', keySpace: 'county-year'},
              lagBreaks: {view: result.lagBreaks, role: 'intermediate', keySpace: 'edge'},
              lagClassCount: {view: result.lagClassCount, role: 'intermediate', keySpace: 'status'},
              counts: {view: result.counts, role: 'output', keySpace: 'transition'},
              probabilities: {view: result.probabilities, role: 'output', keySpace: 'transition'},
              rowTotals: {view: result.rowTotals, role: 'output', keySpace: 'state'},
              homogeneity: {view: result.homogeneity, role: 'output', keySpace: 'homogeneity'},
              stateSparsity: {view: result.stateSparsity, role: 'output', keySpace: 'sparsity'}
            };
          }
        });
      }
    };
  }
});
