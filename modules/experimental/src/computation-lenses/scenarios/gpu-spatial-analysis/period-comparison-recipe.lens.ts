// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';
import {S2_DATASETS} from './s2-datasets';

/** Zero-based day-of-year ranges: spring is 1 March to 31 May, autumn 1 September to 30 November. */
const SPRING_DAYS = [59, 151] as const;
const AUTUMN_DAYS = [243, 334] as const;

registerLensScenario({
  id: 'gpu-spatial-analysis.period-comparison-recipe',
  entry: 'gpu-spatial-analysis',
  element: 'addPeriodComparisonRecipe',
  title: 'Spring and autumn observations, Chicago North Side',
  audience: 'analyst',
  rungs: ['result', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: S2_DATASETS.nature,
  load: async () => {
    const [{runS2Lens}, {runExample}, nature] = await Promise.all([
      import('./s2-lens-run'),
      import('./period-comparison-recipe.example'),
      import('../../data/s2-chicago-north-nature.json').then(module => module.default)
    ]);
    return {
      run: device => {
        const spring: number[] = [];
        const autumn: number[] = [];
        nature.day.forEach((day, index) => {
          const target =
            day >= SPRING_DAYS[0] && day < SPRING_DAYS[1]
              ? spring
              : day >= AUTUMN_DAYS[0] && day < AUTUMN_DAYS[1]
                ? autumn
                : null;
          target?.push(nature.positions[2 * index], nature.positions[2 * index + 1]);
        });
        return runS2Lens(device, {
          scenarioId: 'gpu-spatial-analysis.period-comparison-recipe',
          element: 'addPeriodComparisonRecipe',
          datasets: ['s2-chicago-north-nature'],
          retain: ['delta', 'zScore', 'classIndices', 'colors', 'count', 'overflow'],
          keySpaces: [
            {id: 'cell', label: 'union cell row', size: 2048, indexSpace: '1d'},
            {id: 'spring', label: 'spring observation', size: spring.length / 2, indexSpace: '1d'},
            {id: 'autumn', label: 'autumn observation', size: autumn.length / 2, indexSpace: '1d'},
            {id: 'status', label: 'run status', size: 1, indexSpace: '1d'}
          ],
          choices: {resolution: 16, measure: 'count', classify: 'delta'},
          async build(graph, uploads) {
            const springView = uploads.view('spring', Float32Array.from(spring), 'float32x2');
            const autumnView = uploads.view('autumn', Float32Array.from(autumn), 'float32x2');
            const result = await runExample(graph, {spring: springView, autumn: autumnView});
            result.parameters.forEach(parameters => uploads.own(parameters));
            return {
              spring: {
                view: springView,
                role: 'input',
                keySpace: 'spring',
                geometry: {kind: 'points', coordinateModel: 'geographic'}
              },
              autumn: {
                view: autumnView,
                role: 'input',
                keySpace: 'autumn',
                geometry: {kind: 'points', coordinateModel: 'geographic'}
              },
              delta: {view: result.delta, role: 'output', keySpace: 'cell'},
              zScore: {view: result.zScore, role: 'output', keySpace: 'cell'},
              classIndices: {view: result.classIndices, role: 'output', keySpace: 'cell'},
              count: {view: result.count, role: 'output', keySpace: 'status'},
              overflow: {view: result.overflow, role: 'output', keySpace: 'status'}
            };
          }
        });
      }
    };
  }
});
