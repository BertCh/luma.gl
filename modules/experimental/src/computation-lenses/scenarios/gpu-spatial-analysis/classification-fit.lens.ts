// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUComputationKeySpace} from '@luma.gl/gpgpu';
import {registerLensScenario} from '../../registry';
import {S2_DATASETS} from './s2-datasets';

registerLensScenario({
  id: 'gpu-spatial-analysis.classification-fit',
  entry: 'gpu-spatial-analysis',
  element: 'GPUClassificationFit',
  title: 'Goodness of fit of a quintile classification, Midwest county unemployment',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: S2_DATASETS.counties,
  load: async () => {
    const [{runS2Lens, uploadS2CountyShapes}, {runExample}, counties, shapes] = await Promise.all([
      import('./s2-lens-run'),
      import('./classification-fit.example'),
      import('../../data/s2-midwest-counties.json').then(module => module.default),
      import('../../data/s2-midwest-county-shapes.json').then(module => module.default)
    ]);
    return {
      run: device => {
        const years = counties.unemploymentRate.years;
        const last = years.length - 1;
        const unemployment2023 = Float32Array.from(
          counties.unemploymentRate.values,
          row => row[last]
        );
        const keySpaces: GPUComputationKeySpace[] = [
          {id: 'class', label: 'class (last row: all counties)', size: 6, indexSpace: '1d'},
          {id: 'summary', label: 'fit statistic', size: 5, indexSpace: '1d'},
          {id: 'county', label: 'Midwest county', size: unemployment2023.length, indexSpace: '1d'},
          {id: 'edge', label: 'class edge', size: 6, indexSpace: '1d'}
        ];
        return runS2Lens(device, {
          scenarioId: 'gpu-spatial-analysis.classification-fit',
          element: 'GPUClassificationFit',
          datasets: ['s2-midwest-counties', 's2-midwest-county-shapes'],
          retain: [
            'breaks',
            'classes',
            'counts',
            'medians',
            'absoluteDeviations',
            'squaredDeviations',
            'summary'
          ],
          keySpaces,
          choices: {method: 'quantile', classCount: 5},
          async build(graph, uploads) {
            const unemployment = uploads.view('unemployment', unemployment2023, 'float32');
            const outline = uploadS2CountyShapes(uploads, shapes.polygons);
            keySpaces.push(...outline.keySpaces);
            const result = await runExample(graph, {unemployment});
            uploads.own(result.breakParameters);
            return {
              ...outline.captures,
              unemployment: {
                view: unemployment,
                role: 'input',
                keySpace: 'county',
                meaning: `Unemployment rate ${years[last]} (%)`
              },
              breaks: {view: result.breaks, role: 'intermediate', keySpace: 'edge'},
              classes: {view: result.classes, role: 'input', keySpace: 'county'},
              classMap: {
                ...outline.paint('classes'),
                role: 'input',
                meaning: 'Quantile classes assigned upstream by GPUClassAssignment'
              },
              counts: {view: result.counts, role: 'output', keySpace: 'class'},
              medians: {view: result.medians, role: 'output', keySpace: 'class'},
              absoluteDeviations: {
                view: result.absoluteDeviations,
                role: 'output',
                keySpace: 'class'
              },
              squaredDeviations: {
                view: result.squaredDeviations,
                role: 'output',
                keySpace: 'class'
              },
              summary: {
                view: result.summary,
                role: 'output',
                keySpace: 'summary',
                meaning: 'GADF, ADCM, ADAM, GVF, TSS'
              }
            };
          }
        });
      }
    };
  }
});
