// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';
import {S2_DATASETS} from './s2-datasets';
import type {S2CaptureSpec} from './s2-lens-run';
import type {GPUComputationKeySpace} from '@luma.gl/gpgpu';

registerLensScenario({
  id: 'gpu-spatial-analysis.nearest-feature-weights',
  entry: 'gpu-spatial-analysis',
  element: 'GPUNearestFeatureWeights',
  title: 'Nearest Feature Weights, Chicago North Side',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: S2_DATASETS.nature,
  load: async () => {
    const [
      {runS2Lens, createS2ChicagoFrame, projectS2Positions},
      {runExample},
      nature,
      facilities
    ] = await Promise.all([
      import('./s2-lens-run'),
      import('./nearest-feature-weights.example'),
      import('../../data/s2-chicago-north-nature.json').then(module => module.default),
      import('../../data/s2-chicago-north-facilities.json').then(module => module.default)
    ]);
    return {
      run: device => {
        const frame = createS2ChicagoFrame();
        const projected = projectS2Positions(frame, nature.positions).positions;
        const observationCount = nature.tract.length;
        const libraryCategory = facilities.categories.indexOf('library');
        const libraryRows = facilities.category.flatMap((category, index) =>
          category === libraryCategory ? [index] : []
        );
        const libraryPositions = projectS2Positions(
          frame,
          libraryRows.flatMap(index => facilities.positions.slice(index * 2, index * 2 + 2))
        ).positions;
        const keySpaces: GPUComputationKeySpace[] = [
          {id: 'observation', label: 'observation', size: observationCount, indexSpace: '1d'},
          {id: 'library', label: 'library', size: libraryRows.length, indexSpace: '1d'},
          {
            id: 'slot',
            label: 'query-library weight slot',
            size: observationCount * 3,
            indexSpace: '1d'
          },
          {id: 'offset', label: 'query CSR boundary', size: observationCount + 1, indexSpace: '1d'},
          {id: 'status', label: 'run status', size: 1, indexSpace: '1d'}
        ];
        return runS2Lens(device, {
          scenarioId: 'gpu-spatial-analysis.nearest-feature-weights',
          element: 'GPUNearestFeatureWeights',
          datasets: ['s2-chicago-north-nature', 's2-chicago-north-facilities'],
          retain: [
            'neighborIds',
            'neighborDistances',
            'neighborCounts',
            'joinOverflow',
            'offsets',
            'neighbors',
            'weights',
            'distances',
            'overflow'
          ],
          keySpaces,
          relations: [
            {
              id: 'observation-libraries',
              from: 'observation',
              to: 'library',
              cardinality: 'N:M',
              kind: 'buffer',
              timeDependent: false,
              ref: {
                layout: 'csr',
                resource: 'neighbors',
                capture: 'neighbors',
                offsetsResource: 'offsets'
              }
            }
          ],
          choices: {
            k: 3,
            weightType: 'inverse-distance',
            power: 1,
            distanceFloor: 1,
            excludeSelf: false
          },
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
            const libraries = uploads.view('libraries', libraryPositions, 'float32x2');
            captures['libraries'] = {
              view: libraries,
              role: 'input',
              keySpace: 'library',
              geometry: {kind: 'points', coordinateModel: 'projected', unit: 'm'}
            };
            captures['librarySourceRows'] = {
              view: uploads.view('librarySourceRows', new Uint32Array(libraryRows), 'uint32'),
              role: 'input',
              keySpace: 'library'
            };
            const result = await runExample(graph, {
              observations,
              spatialContext: frame.spatialContext,
              libraries
            });
            captures['neighborIds'] = {view: result.neighborIds, role: 'output', keySpace: 'slot'};
            captures['neighborDistances'] = {
              view: result.neighborDistances,
              role: 'output',
              keySpace: 'slot'
            };
            captures['neighborCounts'] = {
              view: result.neighborCounts,
              role: 'output',
              keySpace: 'observation'
            };
            captures['joinOverflow'] = {
              view: result.joinOverflow,
              role: 'output',
              keySpace: 'status'
            };
            captures['offsets'] = {view: result.offsets, role: 'output', keySpace: 'offset'};
            captures['neighbors'] = {view: result.neighbors, role: 'output', keySpace: 'slot'};
            captures['weights'] = {view: result.weights, role: 'output', keySpace: 'slot'};
            captures['distances'] = {view: result.distances, role: 'output', keySpace: 'slot'};
            captures['overflow'] = {view: result.overflow, role: 'output', keySpace: 'status'};
            captures['weights'].geometry = {
              kind: 'graph',
              offsetsCapture: 'offsets',
              targetsCapture: 'neighbors'
            };
            captures['weights'].meaning =
              'Cross weights: query rows index observations; targets index libraries; no shared node layout';
            return captures;
          }
        });
      }
    };
  }
});
