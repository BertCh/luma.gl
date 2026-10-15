// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';
import {S2_DATASETS} from './s2-datasets';
import type {S2CaptureSpec} from './s2-lens-run';
import type {GPUComputationKeySpace} from '@luma.gl/gpgpu';

registerLensScenario({
  id: 'gpu-spatial-analysis.buffer-selection',
  entry: 'gpu-spatial-analysis',
  element: 'GPUBufferSelection',
  title: 'Buffer Selection, Chicago North Side',
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
      import('./buffer-selection.example'),
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
          {id: 'status', label: 'run status', size: 1, indexSpace: '1d'}
        ];
        return runS2Lens(device, {
          scenarioId: 'gpu-spatial-analysis.buffer-selection',
          element: 'GPUBufferSelection',
          datasets: ['s2-chicago-north-nature', 's2-chicago-north-facilities'],
          retain: ['selected', 'distances', 'overflow'],
          keySpaces,
          choices: {distance: 300},
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
            const distance = uploads.view('distance', new Float32Array([300]), 'float32');
            captures['distance'] = {view: distance, role: 'input', keySpace: 'status'};
            const result = await runExample(graph, {
              observations,
              spatialContext: frame.spatialContext,
              libraries,
              distance
            });
            captures['resultObservations'] = {
              view: observations,
              role: 'output',
              keySpace: 'observation',
              geometry: {
                kind: 'points',
                valuesCapture: 'selected',
                coordinateModel: 'projected',
                unit: 'm'
              }
            };
            captures['selected'] = {view: result.selected, role: 'output', keySpace: 'observation'};
            captures['distances'] = {
              view: result.distances,
              role: 'output',
              keySpace: 'observation'
            };
            captures['overflow'] = {view: result.overflow, role: 'output', keySpace: 'status'};
            return captures;
          }
        });
      }
    };
  }
});
