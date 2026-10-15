// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';
import {S2_DATASETS} from './s2-datasets';
import type {S2CaptureSpec} from './s2-lens-run';
import type {GPUComputationKeySpace} from '@luma.gl/gpgpu';

registerLensScenario({
  id: 'gpu-spatial-analysis.coverage-validity',
  entry: 'gpu-spatial-analysis',
  element: 'GPUCoverageValidity',
  title: 'Coverage validity of Chicago North Side census tracts',
  audience: 'analyst',
  rungs: ['result', 'datum', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: S2_DATASETS.tracts,
  load: async () => {
    const [{runS2Lens, createS2ChicagoFrame, uploadS2JoinPolygons}, {runExample}, tracts] =
      await Promise.all([
        import('./s2-lens-run'),
        import('./coverage-validity.example'),
        import('../../data/s2-chicago-north-tracts.json').then(module => module.default)
      ]);
    return {
      run: device => {
        const frame = createS2ChicagoFrame();
        const keySpaces: GPUComputationKeySpace[] = [];
        return runS2Lens(device, {
          scenarioId: 'gpu-spatial-analysis.coverage-validity',
          element: 'GPUCoverageValidity',
          datasets: ['s2-chicago-north-tracts'],
          retain: ['segmentFlags', 'featureInvalidCounts', 'isValid', 'invalidCount'],
          keySpaces,
          choices: {gapWidth: 1},
          async build(graph, uploads) {
            const tractGeometry = uploadS2JoinPolygons(uploads, frame, 'tracts', tracts.polygons);
            keySpaces.push(...tractGeometry.keySpaces, {
              id: 'status',
              label: 'run status',
              size: 1,
              indexSpace: '1d'
            });
            const result = await runExample(graph, {
              tracts: tractGeometry.geometry,
              spatialContext: frame.spatialContext
            });
            uploads.own(result.parameters);
            const captures: Record<string, S2CaptureSpec> = {...tractGeometry.captures};
            captures['resultTracts'] = {
              view: tractGeometry.display,
              role: 'output',
              keySpace: 'tracts',
              geometry: {
                kind: 'polygons',
                offsetsCapture: 'tractsDisplayOffsets',
                valuesCapture: 'featureInvalidCounts',
                coordinateModel: 'projected',
                unit: 'm'
              }
            };
            captures['featureInvalidCounts'] = {
              view: result.featureInvalidCounts,
              role: 'output',
              keySpace: 'tracts',
              meaning: 'invalid ring edges per tract'
            };
            captures['segmentFlags'] = {
              view: result.segmentFlags,
              role: 'output',
              keySpace: 'tracts-vertex',
              meaning: 'GPU_COVERAGE_VALIDITY_FLAG bits for the edge starting at each vertex'
            };
            captures['isValid'] = {view: result.isValid, role: 'output', keySpace: 'status'};
            captures['invalidCount'] = {
              view: result.invalidCount,
              role: 'output',
              keySpace: 'status'
            };
            return captures;
          }
        });
      }
    };
  }
});
