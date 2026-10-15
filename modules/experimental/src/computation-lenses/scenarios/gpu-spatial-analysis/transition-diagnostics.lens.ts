// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';
import {S2_DATASETS} from './s2-datasets';

/** State order of the condition classes: the five states of the county table, in FIPS order. */
const STATE_ORDER = ['IL', 'IN', 'MI', 'OH', 'WI'];

registerLensScenario({
  id: 'gpu-spatial-analysis.transition-diagnostics',
  entry: 'gpu-spatial-analysis',
  element: 'GPUTransitionDiagnostics',
  title: 'Diagnostics of state-wise unemployment class transitions, Midwest 2008-2023',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: S2_DATASETS.counties,
  load: async () => {
    const [{runS2Lens, getS2PeriodMajorColumn}, {runExample}, counties] = await Promise.all([
      import('./s2-lens-run'),
      import('./transition-diagnostics.example'),
      import('../../data/s2-midwest-counties.json').then(module => module.default)
    ]);
    return {
      run: device => {
        const periodCount = counties.unemploymentRate.years.length;
        const unemploymentByYear = getS2PeriodMajorColumn(counties.unemploymentRate.values);
        const stateOfCounty = counties.state.map(state => {
          const index = STATE_ORDER.indexOf(state);
          // The county table holds exactly these five states.
          if (index < 0) throw new Error(`unexpected state ${state}`);
          return index;
        });
        const stateByYear = new Uint32Array(unemploymentByYear.length);
        for (let period = 0; period < periodCount; period++) {
          stateByYear.set(stateOfCounty, period * stateOfCounty.length);
        }
        return runS2Lens(device, {
          scenarioId: 'gpu-spatial-analysis.transition-diagnostics',
          element: 'GPUTransitionDiagnostics',
          datasets: ['s2-midwest-counties'],
          retain: [
            'breaks',
            'classes',
            'counts',
            'homogeneity',
            'steadyStates',
            'stateSparsity',
            'converged'
          ],
          keySpaces: [
            {id: 'transition', label: 'state x from class x to class', size: 125, indexSpace: '1d'},
            {id: 'homogeneity', label: 'homogeneity statistic', size: 5, indexSpace: '1d'},
            {id: 'steady', label: 'chain x class (chain 0: pooled)', size: 30, indexSpace: '1d'},
            {id: 'sparsity', label: 'sparsity count', size: 2, indexSpace: '1d'},
            {id: 'status', label: 'converged flag', size: 1, indexSpace: '1d'},
            {
              id: 'county-year',
              label: 'county and year (period-major)',
              size: unemploymentByYear.length,
              indexSpace: '1d'
            },
            {id: 'edge', label: 'class edge', size: 6, indexSpace: '1d'}
          ],
          choices: {conditionCount: 5, minimumTransitions: 5},
          async build(graph, uploads) {
            const unemployment = uploads.view('unemployment', unemploymentByYear, 'float32');
            const states = uploads.view('states', stateByYear, 'uint32');
            const result = await runExample(graph, {unemployment, states, periodCount});
            uploads.own(result.breakParameters);
            return {
              unemployment: {
                view: unemployment,
                role: 'input',
                keySpace: 'county-year',
                meaning: 'Unemployment rate (%), 2008-2023, period-major'
              },
              states: {
                view: states,
                role: 'input',
                keySpace: 'county-year',
                meaning: `Condition: state ${STATE_ORDER.join(', ')} as 0 to 4`
              },
              breaks: {view: result.breaks, role: 'intermediate', keySpace: 'edge'},
              classes: {view: result.classes, role: 'intermediate', keySpace: 'county-year'},
              counts: {
                view: result.counts,
                role: 'input',
                keySpace: 'transition',
                meaning: 'Transition counts from GPUTransitionMatrix, one matrix per state'
              },
              homogeneity: {view: result.homogeneity, role: 'output', keySpace: 'homogeneity'},
              steadyStates: {view: result.steadyStates, role: 'output', keySpace: 'steady'},
              stateSparsity: {view: result.stateSparsity, role: 'output', keySpace: 'sparsity'},
              converged: {view: result.converged, role: 'output', keySpace: 'status'}
            };
          }
        });
      }
    };
  }
});
