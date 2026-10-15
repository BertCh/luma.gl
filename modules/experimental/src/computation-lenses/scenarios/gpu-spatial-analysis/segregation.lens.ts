// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';
import {S2_DATASETS} from './s2-datasets';
import type {S2CaptureSpec} from './s2-lens-run';
import type {GPUComputationKeySpace} from '@luma.gl/gpgpu';

const GROUP_FIELDS = ['nhWhite', 'nhBlack', 'hispanic', 'nhAsian', 'nhOther'] as const;

registerLensScenario({
  id: 'gpu-spatial-analysis.segregation',
  entry: 'gpu-spatial-analysis',
  element: 'GPUSegregation',
  title: 'Residential segregation of Chicago North Side tracts, five groups',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: S2_DATASETS.tracts,
  load: async () => {
    const [{runS2Lens, createS2ChicagoFrame, uploadS2JoinPolygons}, {runExample}, tracts] =
      await Promise.all([
        import('./s2-lens-run'),
        import('./segregation.example'),
        import('../../data/s2-chicago-north-tracts.json').then(module => module.default)
      ]);
    return {
      run: device => {
        const frame = createS2ChicagoFrame();
        const tractCount = tracts.population.length;
        const counts = new Float32Array(tractCount * GROUP_FIELDS.length);
        for (let tract = 0; tract < tractCount; tract++) {
          GROUP_FIELDS.forEach((field, group) => {
            counts[tract * GROUP_FIELDS.length + group] = tracts[field][tract];
          });
        }
        const keySpaces: GPUComputationKeySpace[] = [];
        return runS2Lens(device, {
          scenarioId: 'gpu-spatial-analysis.segregation',
          element: 'GPUSegregation',
          datasets: ['s2-chicago-north-tracts'],
          retain: ['indices', 'localTheil', 'localEntropy'],
          keySpaces,
          choices: {groupCount: GROUP_FIELDS.length, spatialScales: 'aspatial'},
          async build(graph, uploads) {
            const tractGeometry = uploadS2JoinPolygons(uploads, frame, 'tracts', tracts.polygons);
            const groupCounts = uploads.view('groupCounts', counts, 'float32');
            const result = await runExample(graph, {groupCounts});
            keySpaces.push(
              ...tractGeometry.keySpaces,
              {
                id: 'group-count',
                label: 'tract-group count',
                size: counts.length,
                indexSpace: '1d'
              },
              {
                id: 'index',
                label: 'segregation index',
                size: result.indices.length,
                indexSpace: '1d'
              }
            );
            const captures: Record<string, S2CaptureSpec> = {...tractGeometry.captures};
            captures['tracts'].meaning = 'context';
            captures['groupCounts'] = {
              view: groupCounts,
              role: 'input',
              keySpace: 'group-count',
              meaning: 'row-major tracts x groups: White, Black, Hispanic, Asian, Other'
            };
            captures['resultTracts'] = {
              view: tractGeometry.display,
              role: 'output',
              keySpace: 'tracts',
              geometry: {
                kind: 'polygons',
                offsetsCapture: 'tractsDisplayOffsets',
                valuesCapture: 'localTheil',
                coordinateModel: 'projected',
                unit: 'm'
              }
            };
            captures['localTheil'] = {
              view: result.localTheil,
              role: 'output',
              keySpace: 'tracts',
              meaning: 'tract contribution to the entropy index H; NaN where undefined'
            };
            captures['localEntropy'] = {
              view: result.localEntropy,
              role: 'output',
              keySpace: 'tracts'
            };
            captures['indices'] = {
              view: result.indices,
              role: 'output',
              keySpace: 'index',
              meaning:
                'entropy H, multigroup dissimilarity D, diversity E, then dissimilarity[5], isolation[5], Atkinson[5], interaction[25]'
            };
            return captures;
          }
        });
      }
    };
  }
});
