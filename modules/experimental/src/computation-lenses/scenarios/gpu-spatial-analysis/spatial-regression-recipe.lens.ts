// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUComputationKeySpace} from '@luma.gl/gpgpu';
import {registerLensScenario} from '../../registry';
import {S2_DATASETS} from './s2-datasets';
import type {S2CaptureSpec} from './s2-lens-run';

registerLensScenario({
  id: 'gpu-spatial-analysis.spatial-regression-recipe',
  entry: 'gpu-spatial-analysis',
  element: 'addSpatialRegressionRecipe',
  title: 'Observations per tract against income and car-free households, Chicago North Side',
  audience: 'analyst',
  rungs: ['result', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: S2_DATASETS.tracts,
  load: async () => {
    const [{runS2Lens, createS2ChicagoFrame, uploadS2JoinPolygons}, {runExample}, tracts] =
      await Promise.all([
        import('./s2-lens-run'),
        import('./spatial-regression-recipe.example'),
        import('../../data/s2-chicago-north-tracts.json').then(module => module.default)
      ]);
    return {
      run: device => {
        const frame = createS2ChicagoFrame();
        const keySpaces: GPUComputationKeySpace[] = [
          {id: 'coefficient', label: 'intercept and slopes', size: 3, indexSpace: '1d'},
          {
            id: 'test',
            label: 'diagnostic value (statistic, df, p per test)',
            size: 18,
            indexSpace: '1d'
          },
          {id: 'status', label: 'run status', size: 1, indexSpace: '1d'}
        ];
        return runS2Lens(device, {
          scenarioId: 'gpu-spatial-analysis.spatial-regression-recipe',
          element: 'addSpatialRegressionRecipe',
          datasets: ['s2-chicago-north-tracts'],
          retain: ['coefficients', 'residuals', 'fitted', 'tests', 'decision', 'moranZScores'],
          keySpaces,
          choices: {criterion: 'queen', transform: 'row', significanceLevel: 0.05},
          async build(graph, uploads) {
            const tractGeometry = uploadS2JoinPolygons(uploads, frame, 'tracts', tracts.polygons);
            keySpaces.unshift(...tractGeometry.keySpaces);
            const count = tracts.natureObs2023.length;
            // Predictors are rescaled to $10,000 and hundreds of households for a well-conditioned fit.
            const predictorValues = new Float32Array(count * 2);
            for (let tract = 0; tract < count; tract++) {
              predictorValues[2 * tract] = tracts.medianHouseholdIncome[tract] / 10000;
              predictorValues[2 * tract + 1] = tracts.noVehicle[tract] / 100;
            }
            const observations = uploads.view(
              'observations',
              Float32Array.from(tracts.natureObs2023),
              'float32'
            );
            const predictors = uploads.view('predictors', predictorValues, 'float32');
            const result = await runExample(graph, {
              tracts: tractGeometry.geometry,
              observations,
              predictors
            });
            uploads.own(result.parameters);
            const polygonGeometry = (valuesCapture: string) => ({
              kind: 'polygons' as const,
              offsetsCapture: 'tractsDisplayOffsets',
              valuesCapture,
              coordinateModel: 'projected' as const,
              unit: 'm'
            });
            const captures: Record<string, S2CaptureSpec> = {
              ...tractGeometry.captures,
              observations: {view: observations, role: 'input', keySpace: 'tracts'},
              predictors: {view: predictors, role: 'input', keySpace: 'tracts'},
              resultTracts: {
                view: tractGeometry.display,
                role: 'output',
                keySpace: 'tracts',
                geometry: polygonGeometry('residuals')
              },
              residuals: {view: result.residuals, role: 'output', keySpace: 'tracts'},
              fitted: {view: result.fitted, role: 'output', keySpace: 'tracts'},
              moranZScores: {view: result.moranZScores, role: 'output', keySpace: 'tracts'},
              coefficients: {view: result.coefficients, role: 'output', keySpace: 'coefficient'},
              tests: {view: result.tests, role: 'output', keySpace: 'test'},
              decision: {view: result.decision, role: 'output', keySpace: 'status'}
            };
            captures['tracts'].geometry = polygonGeometry('observations');
            return captures;
          }
        });
      }
    };
  }
});
