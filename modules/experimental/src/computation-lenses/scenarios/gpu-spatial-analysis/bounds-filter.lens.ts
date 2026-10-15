// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';
import {S2_DATASETS} from './s2-datasets';
import type {S2CaptureSpec} from './s2-lens-run';
import type {GPUComputationKeySpace} from '@luma.gl/gpgpu';

registerLensScenario({
  id: 'gpu-spatial-analysis.bounds-filter',
  entry: 'gpu-spatial-analysis',
  element: 'GPUBoundsFilter',
  title: 'Bounds Filter, Chicago North Side',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: S2_DATASETS.nature,
  load: async () => {
    const [{runS2Lens, createS2ChicagoFrame, projectS2Positions}, {runExample}, nature, parks] =
      await Promise.all([
        import('./s2-lens-run'),
        import('./bounds-filter.example'),
        import('../../data/s2-chicago-north-nature.json').then(module => module.default),
        import('../../data/s2-chicago-north-parks.json').then(module => module.default)
      ]);
    return {
      run: device => {
        const frame = createS2ChicagoFrame();
        const projected = projectS2Positions(frame, nature.positions).positions;
        const observationCount = nature.tract.length;
        const lincolnPark = parks.name.indexOf('Lincoln (Abraham) Park');
        const ring = parks.polygons.featureOffsets[lincolnPark];
        const start = parks.polygons.ringOffsets[ring];
        const end = parks.polygons.ringOffsets[ring + 1];
        const {extent} = projectS2Positions(
          frame,
          parks.polygons.vertices.slice(start * 2, end * 2)
        );
        const [minX, minY, maxX, maxY] = extent;
        const keySpaces: GPUComputationKeySpace[] = [
          {id: 'observation', label: 'observation', size: observationCount, indexSpace: '1d'},
          {id: 'parameter', label: 'box coordinate', size: 4, indexSpace: '1d'},
          {id: 'box', label: 'Lincoln Park envelope', size: 1, indexSpace: '1d'},
          {id: 'box-offset', label: 'box ring boundary', size: 2, indexSpace: '1d'},
          {id: 'status', label: 'run status', size: 1, indexSpace: '1d'}
        ];
        return runS2Lens(device, {
          scenarioId: 'gpu-spatial-analysis.bounds-filter',
          element: 'GPUBoundsFilter',
          datasets: ['s2-chicago-north-nature', 's2-chicago-north-parks'],
          retain: ['selected', 'ids', 'count', 'overflow'],
          keySpaces,
          choices: {mode: 'within'},
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
            const bounds = uploads.view(
              'bounds',
              new Float32Array(
                nature.tract.flatMap((_, index) => [
                  projected[index * 2],
                  projected[index * 2 + 1],
                  projected[index * 2],
                  projected[index * 2 + 1]
                ])
              ),
              'float32x4'
            );
            const parameters = uploads.view('parameters', new Float32Array(extent), 'float32');
            captures['bounds'] = {view: bounds, role: 'input', keySpace: 'observation'};
            captures['parameters'] = {view: parameters, role: 'input', keySpace: 'parameter'};
            captures['boxOffsets'] = {
              view: uploads.view('boxOffsets', new Uint32Array([0, 5]), 'uint32'),
              role: 'input',
              keySpace: 'box-offset'
            };
            captures['box'] = {
              view: uploads.view(
                'box',
                new Float32Array([minX, minY, maxX, minY, maxX, maxY, minX, maxY, minX, minY]),
                'float32x2'
              ),
              role: 'input',
              keySpace: 'box',
              geometry: {
                kind: 'polygons',
                offsetsCapture: 'boxOffsets',
                coordinateModel: 'projected',
                unit: 'm'
              }
            };
            const result = await runExample(graph, {
              bounds,
              parameters,
              spatialContext: frame.spatialContext
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
            captures['ids'] = {view: result.ids, role: 'output', keySpace: 'observation'};
            captures['count'] = {view: result.count, role: 'output', keySpace: 'status'};
            captures['overflow'] = {view: result.overflow, role: 'output', keySpace: 'status'};
            return captures;
          }
        });
      }
    };
  }
});
