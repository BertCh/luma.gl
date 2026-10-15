// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';
import {CORE_DATASET} from './k-core-data';
import type {CoreLensRunProps} from './core-lens-run';

const CAPACITY = 2048;
const WINDOW_LENGTH = 16;
const VISIBLE_ROWS = 6;

/** Chooses where the captured table window starts: the first run of occupied slots. */
function getFirstOccupiedRun(keys: readonly number[]): number {
  // Linear probing without deletion fills the same slots in any insertion order, so this model
  // only places the window; every drawn value is read back from the GPU table.
  const occupied = new Uint8Array(CAPACITY);
  for (const key of keys) {
    let value = key >>> 0;
    value = Math.imul(value ^ (value >>> 16), 0x7feb352d) >>> 0;
    value = Math.imul(value ^ (value >>> 15), 0x846ca68b) >>> 0;
    let slot = (value ^ (value >>> 16)) & (CAPACITY - 1);
    while (occupied[slot]) slot = (slot + 1) & (CAPACITY - 1);
    occupied[slot] = 1;
  }
  for (let start = 0; start + WINDOW_LENGTH <= CAPACITY; start++) {
    if (occupied.subarray(start, start + VISIBLE_ROWS).every(Boolean)) return start;
  }
  return 0;
}

registerLensScenario({
  id: 'gpu-core.hash-index',
  entry: 'gpu-core',
  element: 'GPUHashIndex',
  title: 'Station rows indexed in a hash table',
  audience: 'engineer',
  defaultRung: 'result',
  rungs: ['result', 'kernel', 'graph', 'datum', 'table'],
  dataset: CORE_DATASET,

  load: async () => {
    const [{runCoreLens, describeCoreLens}, {runExample}, data, {getCoreWindow}] =
      await Promise.all([
        import('./core-lens-run'),
        import('./hash-index.example'),
        import('../../data/k-core.json').then(module => module.default),
        import('./k-core-data')
      ]);
    const rows = data.stations.row;
    const first = getFirstOccupiedRun(rows);
    const props: CoreLensRunProps = {
      scenarioId: 'gpu-core.hash-index',
      element: 'GPUHashIndex',
      label: 'Station rows indexed in a hash table',
      keySpaces: [
        {id: 'stations', label: 'Station', size: rows.length, indexSpace: '1d'},
        {id: 'slots', label: 'Table slot', size: CAPACITY, indexSpace: '1d'},
        {id: 'scalar', label: 'Scalar', size: 1, indexSpace: '1d'},
        {id: 'statistics', label: 'Build statistic', size: 6, indexSpace: '1d'}
      ],
      choices: [
        {stage: 'table', name: 'capacity', value: CAPACITY, sweepable: false},
        {stage: 'table', name: 'loadFactor', value: rows.length / CAPACITY, sweepable: false}
      ],

      async build(graph, uploads) {
        const stationRows = uploads.view('stationRows', Uint32Array.from(rows), 'uint32');
        const slotIds = uploads.view(
          'slotIds',
          Uint32Array.from({length: CAPACITY}, (_, slot) => slot),
          'uint32'
        );
        const tableKeysBuffer = uploads.view(
          'tableKeys',
          new Uint32Array(CAPACITY),
          'uint32'
        ).buffer;
        const tableValuesBuffer = uploads.view(
          'tableValues',
          new Uint32Array(CAPACITY),
          'uint32'
        ).buffer;
        const statistics = uploads.view('statistics', new Uint32Array(6), 'uint32');
        const {tableKeys, tableValues} = await graph.withRecordScope(
          {datum: {keySpace: 'stations'}},
          () => runExample(graph, {stationRows, tableKeysBuffer, tableValuesBuffer, statistics})
        );
        const slotWindow = `first:${first}`;
        return {
          stationRows: {
            view: getCoreWindow(graph, stationRows),
            role: 'input',
            keySpace: 'stations',
            meaning: 'column:station_row'
          },
          slotIds: {
            view: getCoreWindow(graph, slotIds, first),
            role: 'output',
            keySpace: 'slots',
            meaning: `column:slot ${slotWindow}`
          },
          tableKeys: {
            view: getCoreWindow(graph, tableKeys, first),
            role: 'output',
            keySpace: 'slots',
            meaning: `column:station_row ${slotWindow}`
          },
          tableValues: {
            view: getCoreWindow(graph, tableValues, first),
            role: 'output',
            keySpace: 'slots',
            meaning: `column:station ${slotWindow}`
          },
          maxProbes: {
            view: graph.createDataView(statistics.buffer, {
              format: 'uint32',
              length: 1,
              byteOffset: statistics.byteOffset + 5 * 4
            }),
            role: 'output',
            keySpace: 'scalar',
            meaning: 'count'
          },
          statistics: {
            view: statistics,
            role: 'intermediate',
            keySpace: 'statistics',
            meaning: 'unique duplicates overflow invalid total-probes max-probes'
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
