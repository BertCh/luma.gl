// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';
import {CORE_DATASET} from './k-core-data';
import type {CoreLensRunProps} from './core-lens-run';

registerLensScenario({
  id: 'gpu-core.grid-binning',
  entry: 'gpu-core',
  element: 'GPUGridBinning',
  title: 'Stations per 1° grid cell',
  audience: 'engineer',
  defaultRung: 'result',
  rungs: ['result', 'kernel', 'graph', 'datum', 'formula', 'table'],
  dataset: CORE_DATASET,

  load: async () => {
    const [{runCoreLens, describeCoreLens}, {runExample}, data] = await Promise.all([
      import('./core-lens-run'),
      import('./grid-binning.example'),
      import('../../data/k-core.json').then(module => module.default)
    ]);
    const {lon, lat} = data.stations;
    const positions = new Float32Array(lon.length * 2);
    lon.forEach((value, index) => {
      positions[index * 2] = value;
      positions[index * 2 + 1] = lat[index];
    });
    const props: CoreLensRunProps = {
      scenarioId: 'gpu-core.grid-binning',
      element: 'GPUGridBinning',
      label: 'Stations per 1° grid cell',
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
        const stationsPerCellBuffer = uploads.view(
          'stationsPerCell',
          new Uint32Array(18 * 16),
          'uint32'
        ).buffer;
        const {stationsPerCell} = await graph.withRecordScope({datum: {keySpace: 'stations'}}, () =>
          runExample(graph, {stationPositions, stationsPerCellBuffer})
        );
        return {
          stationPositions: {
            view: stationPositions,
            role: 'input',
            keySpace: 'stations',
            geometry: {kind: 'points', coordinateModel: 'geographic'}
          },
          stationsPerCell: {
            view: stationsPerCell,
            role: 'output',
            keySpace: 'cells',
            meaning: 'grid',
            highlight: (counts: ArrayLike<number>) => {
              const key = Array.from(counts).indexOf(Math.max(...Array.from(counts)));
              return {
                key,
                label: `densest cell (${key % 18}, ${Math.floor(key / 18)}): ${counts[key]} stations`
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
