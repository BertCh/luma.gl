// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUComputationKeySpace} from '@luma.gl/gpgpu';
import {registerLensScenario} from '../../registry';
import {S2_DATASETS} from './s2-datasets';
import type {S2CaptureSpec} from './s2-lens-run';

registerLensScenario({
  id: 'gpu-spatial-analysis.rate-cluster-map-recipe',
  entry: 'gpu-spatial-analysis',
  element: 'addRateClusterMapRecipe',
  title: 'Bird share of iNaturalist observations by tract, Chicago North Side',
  audience: 'analyst',
  rungs: ['result', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: S2_DATASETS.tracts,
  load: async () => {
    const [{runS2Lens, createS2ChicagoFrame, uploadS2JoinPolygons}, {runExample}, tracts] =
      await Promise.all([
        import('./s2-lens-run'),
        import('./rate-cluster-map-recipe.example'),
        import('../../data/s2-chicago-north-tracts.json').then(module => module.default)
      ]);
    return {
      run: device => {
        const frame = createS2ChicagoFrame();
        const keySpaces: GPUComputationKeySpace[] = [];
        return runS2Lens(device, {
          scenarioId: 'gpu-spatial-analysis.rate-cluster-map-recipe',
          element: 'addRateClusterMapRecipe',
          datasets: ['s2-chicago-north-tracts'],
          retain: ['standardizedRates', 'zScores', 'quadrants', 'adjustedPValues', 'significant'],
          keySpaces,
          choices: {
            criterion: 'queen',
            analyze: 'standardized',
            multipleTesting: 'benjamini-hochberg',
            permutationCount: 199,
            significanceLevel: 0.05
          },
          async build(graph, uploads) {
            const tractGeometry = uploadS2JoinPolygons(uploads, frame, 'tracts', tracts.polygons);
            keySpaces.push(...tractGeometry.keySpaces);
            const birdObservations = uploads.view(
              'birdObservations',
              Uint32Array.from(tracts.birdObs2023),
              'uint32'
            );
            const natureObservations = uploads.view(
              'natureObservations',
              Uint32Array.from(tracts.natureObs2023),
              'uint32'
            );
            const result = await runExample(graph, {
              tracts: tractGeometry.geometry,
              birdObservations,
              natureObservations
            });
            result.parameters.forEach(parameters => uploads.own(parameters));
            const polygonGeometry = (valuesCapture: string) => ({
              kind: 'polygons' as const,
              offsetsCapture: 'tractsDisplayOffsets',
              valuesCapture,
              coordinateModel: 'projected' as const,
              unit: 'm'
            });
            const captures: Record<string, S2CaptureSpec> = {
              ...tractGeometry.captures,
              birdObservations: {view: birdObservations, role: 'input', keySpace: 'tracts'},
              natureObservations: {view: natureObservations, role: 'input', keySpace: 'tracts'},
              resultTracts: {
                view: tractGeometry.display,
                role: 'output',
                keySpace: 'tracts',
                geometry: polygonGeometry('quadrants')
              },
              quadrants: {view: result.quadrants, role: 'output', keySpace: 'tracts'},
              standardizedRates: {
                view: result.standardizedRates,
                role: 'output',
                keySpace: 'tracts'
              },
              zScores: {view: result.zScores, role: 'output', keySpace: 'tracts'},
              adjustedPValues: {
                view: result.permutation.adjustedPValues,
                role: 'output',
                keySpace: 'tracts'
              },
              significant: {
                view: result.permutation.significant,
                role: 'output',
                keySpace: 'tracts'
              }
            };
            captures['tracts'].geometry = polygonGeometry('birdObservations');
            return captures;
          }
        });
      }
    };
  }
});
