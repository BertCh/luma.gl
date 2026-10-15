// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';
import {S2_DATASETS} from './s2-datasets';
import type {S2CaptureSpec} from './s2-lens-run';
import type {GPUComputationKeySpace} from '@luma.gl/gpgpu';

registerLensScenario({
  id: 'gpu-spatial-analysis.spatial-predicate-join',
  entry: 'gpu-spatial-analysis',
  element: 'GPUSpatialPredicateJoin',
  title: 'Spatial Predicate Join, Chicago North Side',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: S2_DATASETS.parks,
  load: async () => {
    const [{runS2Lens, createS2ChicagoFrame, uploadS2JoinPolygons}, {runExample}, parks, tracts] =
      await Promise.all([
        import('./s2-lens-run'),
        import('./spatial-predicate-join.example'),
        import('../../data/s2-chicago-north-parks.json').then(module => module.default),
        import('../../data/s2-chicago-north-tracts.json').then(module => module.default)
      ]);
    return {
      run: device => {
        const frame = createS2ChicagoFrame();
        const capacity = parks.name.length * tracts.geoid.length;
        const keySpaces: GPUComputationKeySpace[] = [
          {id: 'tracts', label: 'tract', size: tracts.geoid.length, indexSpace: '1d'},
          {
            id: 'pair',
            label: 'pair capacity; read count for active rows',
            size: capacity,
            indexSpace: '1d'
          },
          {id: 'status', label: 'run status', size: 1, indexSpace: '1d'}
        ];
        return runS2Lens(device, {
          scenarioId: 'gpu-spatial-analysis.spatial-predicate-join',
          element: 'GPUSpatialPredicateJoin',
          datasets: ['s2-chicago-north-parks', 's2-chicago-north-tracts'],
          retain: ['tractMatches'],
          keySpaces,
          relations: [
            {
              id: 'pair-parks',
              from: 'pair',
              to: 'parks',
              cardinality: 'N:1',
              kind: 'buffer',
              timeDependent: false,
              ref: {layout: 'index', resource: 'leftIds', capture: 'leftIds'}
            },
            {
              id: 'pair-tracts',
              from: 'pair',
              to: 'tracts',
              cardinality: 'N:1',
              kind: 'buffer',
              timeDependent: false,
              ref: {layout: 'index', resource: 'rightIds', capture: 'rightIds'}
            }
          ],
          choices: {predicate: 'intersects', engine: 'relate'},
          async build(graph, uploads) {
            const parkGeometry = uploadS2JoinPolygons(uploads, frame, 'parks', parks.polygons);
            const tractGeometry = uploadS2JoinPolygons(uploads, frame, 'tracts', tracts.polygons);
            keySpaces.push(
              ...parkGeometry.keySpaces,
              ...tractGeometry.keySpaces.filter(space => space.id !== 'tracts')
            );
            const pairs = {
              leftIds: uploads.view(
                'leftIds',
                new Uint32Array(capacity).fill(0xffffffff),
                'uint32'
              ),
              rightIds: uploads.view(
                'rightIds',
                new Uint32Array(capacity).fill(0xffffffff),
                'uint32'
              ),
              status: {
                count: uploads.view('pairCount', new Uint32Array(1), 'uint32'),
                overflow: uploads.view('pairOverflow', new Uint32Array(1), 'uint32')
              }
            };
            const result = await runExample(graph, {
              parks: parkGeometry.geometry,
              tracts: tractGeometry.geometry,
              pairs,
              spatialContext: frame.spatialContext
            });
            const captures: Record<string, S2CaptureSpec> = {
              ...parkGeometry.captures,
              ...tractGeometry.captures,
              leftIds: {view: pairs.leftIds, role: 'output', keySpace: 'pair'},
              rightIds: {view: pairs.rightIds, role: 'output', keySpace: 'pair'},
              pairCount: {view: pairs.status.count, role: 'output', keySpace: 'status'},
              pairOverflow: {view: pairs.status.overflow, role: 'output', keySpace: 'status'},
              joinedTracts: {
                view: tractGeometry.display,
                role: 'output',
                keySpace: 'tracts',
                geometry: {
                  kind: 'polygons',
                  offsetsCapture: 'tractsDisplayOffsets',
                  valuesCapture: 'tractMatches',
                  coordinateModel: 'projected',
                  unit: 'm'
                }
              }
            };
            captures['tractMatches'] = {
              view: result.tractMatches,
              role: 'output',
              keySpace: 'tracts'
            };
            return captures;
          }
        });
      }
    };
  }
});
