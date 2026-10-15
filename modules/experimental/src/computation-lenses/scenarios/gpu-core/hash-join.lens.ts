// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';
import {CORE_DATASET} from './k-core-data';
import type {CoreLensRunProps} from './core-lens-run';

const CAPACITY = 2048;

registerLensScenario({
  id: 'gpu-core.hash-join',
  entry: 'gpu-core',
  element: 'GPUHashJoin',
  title: 'Wet stations joined back to their station index',
  audience: 'engineer',
  defaultRung: 'result',
  rungs: ['result', 'kernel', 'graph', 'datum', 'table'],
  dataset: CORE_DATASET,

  load: async () => {
    const [{runCoreLens, describeCoreLens}, {runExample}, data, {getCoreWindow}] =
      await Promise.all([
        import('./core-lens-run'),
        import('./hash-join.example'),
        import('../../data/k-core.json').then(module => module.default),
        import('./k-core-data')
      ]);
    const {row, prcpMm} = data.stations;
    // The left side keeps the original row numbers of stations with at least 100 mm of rain.
    const wetRows = row.filter((_, index) => prcpMm[index] >= 100);
    const props: CoreLensRunProps = {
      scenarioId: 'gpu-core.hash-join',
      element: 'GPUHashJoin',
      label: 'Wet stations joined back to their station index',
      keySpaces: [
        {id: 'stations', label: 'Station', size: row.length, indexSpace: '1d'},
        {id: 'wet-stations', label: 'Wet station', size: wetRows.length, indexSpace: '1d'},
        {id: 'pairs', label: 'Joined pair', size: wetRows.length, indexSpace: '1d'},
        {id: 'scalar', label: 'Scalar', size: 1, indexSpace: '1d'}
      ],

      async build(graph, uploads) {
        const stationRows = uploads.view('stationRows', Uint32Array.from(row), 'uint32');
        const wetStationRows = uploads.view('wetStationRows', Uint32Array.from(wetRows), 'uint32');
        const tableKeys = uploads.view('tableKeys', new Uint32Array(CAPACITY), 'uint32');
        const tableValues = uploads.view('tableValues', new Uint32Array(CAPACITY), 'uint32');
        const indexStatistics = uploads.view('indexStatistics', new Uint32Array(6), 'uint32');
        const lookupStatistics = uploads.view('lookupStatistics', new Uint32Array(4), 'uint32');
        const pairCount = uploads.view('pairCount', new Uint32Array(1), 'uint32');
        const overflow = uploads.view('overflow', new Uint32Array(1), 'uint32');
        const wetPositions = uploads.view(
          'wetPositions',
          new Uint32Array(wetRows.length),
          'uint32'
        );
        const wetStationIndices = uploads.view(
          'wetStationIndices',
          new Uint32Array(wetRows.length),
          'uint32'
        );
        await graph.withRecordScope({datum: {keySpace: 'stations'}}, () =>
          runExample(graph, {
            stationRows,
            wetStationRows,
            tableKeys,
            tableValues,
            indexStatistics,
            lookupStatistics,
            wetPositions,
            wetStationIndices,
            pairCount,
            overflow
          })
        );
        return {
          wetStationRows: {
            view: getCoreWindow(graph, wetStationRows),
            role: 'input',
            keySpace: 'wet-stations',
            meaning: 'column:wet_station_row'
          },
          wetPositions: {
            view: getCoreWindow(graph, wetPositions),
            role: 'output',
            keySpace: 'pairs',
            meaning: 'column:left'
          },
          wetStationIndices: {
            view: getCoreWindow(graph, wetStationIndices),
            role: 'output',
            keySpace: 'pairs',
            meaning: 'column:right'
          },
          pairCount: {view: pairCount, role: 'output', keySpace: 'scalar', meaning: 'count'}
        };
      }
    };
    return {
      run: device => runCoreLens(device, props),
      describeStructure: device => describeCoreLens(device, props)
    };
  }
});
