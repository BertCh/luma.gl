// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUComputationKeySpace} from '@luma.gl/gpgpu';
import {registerLensScenario} from '../../registry';
import {S2_DATASETS} from './s2-datasets';

registerLensScenario({
  id: 'gpu-spatial-analysis.class-assignment',
  entry: 'gpu-spatial-analysis',
  element: 'GPUClassAssignment',
  title: 'Class assignment of 2023 county unemployment, Midwest',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: S2_DATASETS.counties,
  load: async () => {
    const [{runS2Lens, uploadS2CountyShapes}, {runExample}, counties, shapes] = await Promise.all([
      import('./s2-lens-run'),
      import('./class-assignment.example'),
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
          {id: 'county', label: 'Midwest county', size: unemployment2023.length, indexSpace: '1d'},
          {id: 'edge', label: 'class edge', size: 6, indexSpace: '1d'},
          {id: 'status', label: 'class count', size: 1, indexSpace: '1d'}
        ];
        return runS2Lens(device, {
          scenarioId: 'gpu-spatial-analysis.class-assignment',
          element: 'GPUClassAssignment',
          datasets: ['s2-midwest-counties', 's2-midwest-county-shapes'],
          retain: ['breaks', 'classCount', 'classes'],
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
              classCount: {view: result.classCount, role: 'intermediate', keySpace: 'status'},
              classes: {view: result.classes, role: 'output', keySpace: 'county'},
              classMap: outline.paint('classes')
            };
          }
        });
      }
    };
  }
});
