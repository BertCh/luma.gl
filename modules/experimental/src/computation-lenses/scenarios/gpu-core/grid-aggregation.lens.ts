// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';
import {CORE_DATASET} from './k-core-data';
import type {CoreLensRunProps} from './core-lens-run';

registerLensScenario({
  id: 'gpu-core.grid-aggregation',
  entry: 'gpu-core',
  element: 'GPUGridAggregation',
  title: 'Rainfall total per 1° grid cell',
  audience: 'engineer',
  defaultRung: 'result',
  rungs: ['result', 'kernel', 'graph', 'datum', 'formula', 'table'],
  dataset: CORE_DATASET,

  load: async () => {
    const [{runCoreLens, describeCoreLens}, {runExample}, data] = await Promise.all([
      import('./core-lens-run'),
      import('./grid-aggregation.example'),
      import('../../data/k-core.json').then(module => module.default)
    ]);
    const {lon, lat, prcpMm} = data.stations;
    const positions = new Float32Array(lon.length * 2);
    lon.forEach((value, index) => {
      positions[index * 2] = value;
      positions[index * 2 + 1] = lat[index];
    });
    const props: CoreLensRunProps = {
      scenarioId: 'gpu-core.grid-aggregation',
      element: 'GPUGridAggregation',
      label: 'Rainfall total per 1° grid cell',
      keySpaces: [
        {id: 'stations', label: 'Station', size: lon.length, indexSpace: '1d'},
        {id: 'cells', label: '1° cell', size: 18 * 16, shape: [18, 16], indexSpace: '2d'}
      ],
      choices: [
        {stage: 'grid', name: 'gridOriginX', value: -92, unit: 'degree', sweepable: false},
        {stage: 'grid', name: 'gridOriginY', value: 24, unit: 'degree', sweepable: false},
        {stage: 'grid', name: 'gridCellSize', value: 1, unit: 'degree', sweepable: false},
        {
          stage: 'aggregation',
          name: 'operation',
          value: 'sum',
          alternatives: ['min', 'max', 'mean'],
          sweepable: false
        }
      ],

      async build(graph, uploads) {
        const stationPositions = uploads.view('stationPositions', positions, 'float32x2');
        const rainfall = uploads.view('rainfall', Float32Array.from(prcpMm), 'float32');
        const rainfallPerCellBuffer = uploads.view(
          'rainfallPerCell',
          new Float32Array(18 * 16),
          'float32'
        ).buffer;
        const {rainfallPerCell} = await graph.withRecordScope({datum: {keySpace: 'stations'}}, () =>
          runExample(graph, {stationPositions, rainfall, rainfallPerCellBuffer})
        );
        return {
          stationPositions: {
            view: stationPositions,
            role: 'input',
            keySpace: 'stations',
            geometry: {kind: 'points', coordinateModel: 'geographic'}
          },
          rainfallPerCell: {
            view: rainfallPerCell,
            role: 'output',
            keySpace: 'cells',
            meaning: 'grid',
            highlight: (totals: ArrayLike<number>) => {
              const key = Array.from(totals).indexOf(Math.max(...Array.from(totals)));
              return {
                key,
                label: `wettest cell (${key % 18}, ${Math.floor(key / 18)}): ${totals[key].toFixed(1)} mm`
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
