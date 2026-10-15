// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';
import {S2_DATASETS} from './s2-datasets';
import type {S2CaptureSpec} from './s2-lens-run';
import type {GPUComputationKeySpace} from '@luma.gl/gpgpu';

registerLensScenario({
  id: 'gpu-spatial-analysis.offset-expansion',
  entry: 'gpu-spatial-analysis',
  element: 'GPUOffsetExpansion',
  title: 'Offset Expansion, Chicago North Side',
  audience: 'analyst',
  rungs: ['result', 'datum', 'kernel', 'graph'],
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
      import('./offset-expansion.example'),
      import('../../data/s2-chicago-north-nature.json').then(module => module.default),
      import('../../data/s2-chicago-north-tracts.json').then(module => module.default)
    ]);
    return {
      run: device => {
        const frame = createS2ChicagoFrame();
        const projected = projectS2Positions(frame, nature.positions).positions;
        // Group only observations whose supplied tract reference lies in this dataset.
        const order = nature.tract
          .map((tract, index) => ({tract, index}))
          .filter(row => row.tract >= 0 && row.tract < tracts.geoid.length)
          .sort((left, right) => left.tract - right.tract || left.index - right.index);
        const grouped = new Float32Array(
          order.flatMap(row => [projected[row.index * 2], projected[row.index * 2 + 1]])
        );
        const offsets = new Uint32Array(tracts.geoid.length + 1);
        for (const row of order) offsets[row.tract + 1]++;
        for (let index = 1; index < offsets.length; index++) offsets[index] += offsets[index - 1];
        const observationCount = order.length;
        const keySpaces: GPUComputationKeySpace[] = [
          {
            id: 'observation',
            label: 'observation grouped by supplied tract',
            size: observationCount,
            indexSpace: '1d'
          },
          {id: 'offset', label: 'tract range boundary', size: offsets.length, indexSpace: '1d'},
          {id: 'status', label: 'run status', size: 1, indexSpace: '1d'}
        ];
        return runS2Lens(device, {
          scenarioId: 'gpu-spatial-analysis.offset-expansion',
          element: 'GPUOffsetExpansion',
          datasets: ['s2-chicago-north-nature', 's2-chicago-north-tracts'],
          retain: ['owners', 'localIndex', 'count', 'overflow'],
          keySpaces,
          relations: [
            {
              id: 'observation-owner',
              from: 'observation',
              to: 'tracts',
              cardinality: 'N:1',
              kind: 'buffer',
              timeDependent: false,
              ref: {layout: 'index', resource: 'owners', capture: 'owners'}
            }
          ],
          choices: {},
          async build(graph, uploads) {
            const observations = uploads.view('observations', grouped, 'float32x2');
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
            const offsetView = uploads.view('offsets', offsets, 'uint32');
            captures['offsets'] = {view: offsetView, role: 'input', keySpace: 'offset'};
            captures['sourceRows'] = {
              view: uploads.view(
                'sourceRows',
                new Uint32Array(order.map(row => row.index)),
                'uint32'
              ),
              role: 'input',
              keySpace: 'observation'
            };
            const result = await runExample(graph, {offsets: offsetView, observationCount});
            captures['resultObservations'] = {
              view: observations,
              role: 'output',
              keySpace: 'observation',
              geometry: {
                kind: 'points',
                valuesCapture: 'owners',
                coordinateModel: 'projected',
                unit: 'm'
              }
            };
            captures['owners'] = {view: result.owners, role: 'output', keySpace: 'observation'};
            captures['localIndex'] = {
              view: result.localIndex,
              role: 'output',
              keySpace: 'observation'
            };
            captures['count'] = {view: result.count, role: 'output', keySpace: 'status'};
            captures['overflow'] = {view: result.overflow, role: 'output', keySpace: 'status'};
            return captures;
          }
        });
      }
    };
  }
});
