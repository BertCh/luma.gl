// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';
import {S2_DATASETS} from './s2-datasets';
import type {S2CaptureSpec} from './s2-lens-run';
import type {GPUComputationKeySpace} from '@luma.gl/gpgpu';

registerLensScenario({
  id: 'gpu-spatial-analysis.point-in-polygon-join',
  entry: 'gpu-spatial-analysis',
  element: 'GPUPointInPolygonJoin',
  title: 'Point In Polygon Join, Chicago North Side',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: S2_DATASETS.nature,
  load: async () => {
    const [
      {runS2Lens, createS2ChicagoFrame, projectS2Positions, uploadS2JoinPolygons},
      {runExample},
      nature,
      tracts
    ] = await Promise.all([
      import('./s2-lens-run'),
      import('./point-in-polygon-join.example'),
      import('../../data/s2-chicago-north-nature.json').then(module => module.default),
      import('../../data/s2-chicago-north-tracts.json').then(module => module.default)
    ]);
    return {
      run: device => {
        const frame = createS2ChicagoFrame();
        const projected = projectS2Positions(frame, nature.positions).positions;
        const observationCount = nature.tract.length;
        const keySpaces: GPUComputationKeySpace[] = [
          {id: 'observation', label: 'observation', size: observationCount, indexSpace: '1d'},
          {id: 'status', label: 'run status', size: 1, indexSpace: '1d'}
        ];
        return runS2Lens(device, {
          scenarioId: 'gpu-spatial-analysis.point-in-polygon-join',
          element: 'GPUPointInPolygonJoin',
          datasets: ['s2-chicago-north-nature', 's2-chicago-north-tracts'],
          retain: ['tractIds', 'featureCounts', 'overflow', 'uncertainCount'],
          keySpaces,
          relations: [
            {
              id: 'observation-tract',
              from: 'observation',
              to: 'tracts',
              cardinality: 'N:1',
              kind: 'buffer',
              timeDependent: false,
              ref: {layout: 'index', resource: 'tractIds', capture: 'tractIds'}
            }
          ],
          choices: {boundaryRule: 'half-open', edgeSlabs: false},
          async build(graph, uploads) {
            const observations = uploads.view('observations', projected, 'float32x2');
            const captures: Record<string, S2CaptureSpec> = {
              observations: {
                view: observations,
                role: 'input',
                keySpace: 'observation',
                geometry: {kind: 'points', coordinateModel: 'projected', unit: 'm'}
              }
            };
            const tractGeometry = uploadS2JoinPolygons(uploads, frame, 'tracts', tracts.polygons);
            keySpaces.push(...tractGeometry.keySpaces);
            Object.assign(captures, tractGeometry.captures);
            captures['tracts'].meaning = 'context';
            captures['referenceTract'] = {
              view: uploads.view('referenceTract', new Int32Array(nature.tract), 'sint32'),
              role: 'input',
              keySpace: 'observation',
              meaning:
                'Precomputed input tract reference; agreement has not been measured for this projected run'
            };
            const result = await runExample(graph, {
              observations,
              spatialContext: frame.spatialContext,
              tracts: tractGeometry.geometry
            });
            captures['resultObservations'] = {
              view: observations,
              role: 'output',
              keySpace: 'observation',
              geometry: {
                kind: 'points',
                valuesCapture: 'tractIds',
                coordinateModel: 'projected',
                unit: 'm'
              }
            };
            captures['tractIds'] = {view: result.tractIds, role: 'output', keySpace: 'observation'};
            captures['featureCounts'] = {
              view: result.featureCounts,
              role: 'output',
              keySpace: 'tracts'
            };
            captures['overflow'] = {view: result.overflow, role: 'output', keySpace: 'status'};
            captures['uncertainCount'] = {
              view: result.uncertainCount,
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
