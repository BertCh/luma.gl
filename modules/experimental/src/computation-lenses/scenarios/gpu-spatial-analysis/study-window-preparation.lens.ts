// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';
import {S2_DATASETS} from './s2-datasets';

registerLensScenario({
  id: 'gpu-spatial-analysis.study-window-preparation',
  entry: 'gpu-spatial-analysis',
  element: 'GPUStudyWindowPreparation',
  title: 'Prepare the Chicago North Side study window',
  audience: 'analyst',
  rungs: ['result', 'datum', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: S2_DATASETS.areas,
  load: async () => {
    const [
      {runS2Lens, createS2ChicagoFrame, projectS2Positions, closeS2Rings},
      {runExample},
      data
    ] = await Promise.all([
      import('./s2-lens-run'),
      import('./study-window-preparation.example'),
      import('../../data/s2-chicago-north-areas.json').then(module => module.default)
    ]);
    return {
      run: device => {
        const frame = createS2ChicagoFrame('lambert-equal-area');
        const projected = projectS2Positions(frame, data.polygons.vertices);
        const geometry = closeS2Rings(projected.positions, data.polygons.ringOffsets);
        const polygonCount = data.polygons.featureOffsets.length - 1;
        return runS2Lens(device, {
          scenarioId: 'gpu-spatial-analysis.study-window-preparation',
          element: 'GPUStudyWindowPreparation',
          datasets: ['s2-chicago-north-areas'],
          retain: ['summary'],
          keySpaces: [
            {
              id: 'summaryField',
              label: 'bounds, area, perimeter, edge count, flags',
              size: 8,
              indexSpace: '1d'
            },
            {
              id: 'vertex',
              label: 'window boundary vertex',
              size: geometry.positions.length / 2,
              indexSpace: '1d'
            },
            {id: 'boundary', label: 'polygon boundary', size: polygonCount + 1, indexSpace: '1d'},
            {id: 'featureBoundary', label: 'single window boundaries', size: 2, indexSpace: '1d'}
          ],
          choices: {samplingGrid: '64 × 64'},
          async build(graph, uploads) {
            const positions = uploads.view('positions', geometry.positions, 'float32x2');
            const ringOffsets = uploads.view('ring-offsets', geometry.ringOffsets, 'uint32');
            const polygonOffsets = uploads.view(
              'polygon-offsets',
              Uint32Array.from({length: polygonCount + 1}, (_, index) => index),
              'uint32'
            );
            const featureOffsets = uploads.view(
              'feature-offsets',
              Uint32Array.from([0, polygonCount]),
              'uint32'
            );
            const {studyWindow, summary} = await runExample(graph, {
              communityAreas: {positions, ringOffsets, polygonOffsets, featureOffsets},
              spatialContext: frame.spatialContext
            });
            uploads.own(studyWindow);
            return {
              // The prepared handle preserves these polygon boundaries; preparation does not move vertices.
              preparedWindow: {
                view: positions,
                role: 'output',
                keySpace: 'vertex',
                meaning: 'polygon support of the prepared study window',
                geometry: {
                  kind: 'polygons',
                  coordinateModel: 'projected',
                  unit: 'm',
                  offsetsCapture: 'ringOffsets'
                }
              },
              ringOffsets: {view: ringOffsets, role: 'input', keySpace: 'boundary'},
              polygonOffsets: {view: polygonOffsets, role: 'input', keySpace: 'boundary'},
              featureOffsets: {view: featureOffsets, role: 'input', keySpace: 'featureBoundary'},
              summary: {
                view: summary,
                role: 'output',
                keySpace: 'summaryField',
                meaning: 'minX, minY, maxX, maxY, area m², perimeter m, edges, flags'
              }
            };
          }
        });
      }
    };
  }
});
