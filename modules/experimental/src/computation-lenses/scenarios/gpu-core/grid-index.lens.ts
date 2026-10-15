// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';
import {CORE_DATASET} from './k-core-data';
import type {CoreLensRunProps} from './core-lens-run';

registerLensScenario({
  id: 'gpu-core.grid-index',
  entry: 'gpu-core',
  element: 'GPUGridIndex',
  title: 'Stations indexed on a 1° grid',
  audience: 'engineer',
  defaultRung: 'result',
  rungs: ['result', 'kernel', 'graph', 'datum', 'table'],
  dataset: CORE_DATASET,

  load: async () => {
    const [{runCoreLens, describeCoreLens}, {runExample}, data] = await Promise.all([
      import('./core-lens-run'),
      import('./grid-index.example'),
      import('../../data/k-core.json').then(module => module.default)
    ]);
    const {lon, lat} = data.stations;
    const positions = new Float32Array(lon.length * 2);
    lon.forEach((value, index) => {
      positions[index * 2] = value;
      positions[index * 2 + 1] = lat[index];
    });
    const props: CoreLensRunProps = {
      scenarioId: 'gpu-core.grid-index',
      element: 'GPUGridIndex',
      label: 'Stations indexed on a 1° grid',
      keySpaces: [
        {id: 'stations', label: 'Station', size: lon.length, indexSpace: '1d'},
        {id: 'cells', label: '1° cell', size: 18 * 16, shape: [18, 16], indexSpace: '2d'}
      ],
      choices: [
        {stage: 'grid', name: 'gridOriginX', value: -92, unit: 'degree', sweepable: false},
        {stage: 'grid', name: 'gridOriginY', value: 24, unit: 'degree', sweepable: false},
        {stage: 'grid', name: 'gridCellSize', value: 1, unit: 'degree', sweepable: false}
      ],

      async build(graph, uploads) {
        const stationPositions = uploads.view('stationPositions', positions, 'float32x2');
        const cellOffsetsBuffer = uploads.view(
          'cellOffsets',
          new Uint32Array(18 * 16 + 1),
          'uint32'
        ).buffer;
        const stationIdsBuffer = uploads.view(
          'stationIds',
          new Uint32Array(lon.length),
          'uint32'
        ).buffer;
        const indexedCount = uploads.view('indexedCount', new Uint32Array(1), 'uint32');
        const overflow = uploads.view('overflow', new Uint32Array(1), 'uint32');
        const {cellOffsets} = await graph.withRecordScope({datum: {keySpace: 'stations'}}, () =>
          runExample(graph, {
            stationPositions,
            cellOffsetsBuffer,
            stationIdsBuffer,
            indexedCount,
            overflow
          })
        );
        return {
          stationPositions: {
            view: stationPositions,
            role: 'input',
            keySpace: 'stations',
            geometry: {kind: 'points', coordinateModel: 'geographic'}
          },
          cellOffsets: {
            view: cellOffsets,
            role: 'output',
            keySpace: 'cells',
            meaning: 'grid csr-offsets',
            highlight: (offsets: ArrayLike<number>) => {
              let key = 0;
              for (let cell = 1; cell < offsets.length - 1; cell++) {
                if (offsets[cell + 1] - offsets[cell] > offsets[key + 1] - offsets[key]) key = cell;
              }
              return {
                key,
                label: `densest cell (${key % 18}, ${Math.floor(key / 18)}): ${offsets[key + 1] - offsets[key]} stations`
              };
            }
          }
        };
      }
    };
    return {
      run: device => runCoreLens(device, props),
      describeStructure: device => describeCoreLens(device, props)
    };
  }
});
