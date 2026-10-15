// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';
import {CORE_DATASET} from './k-core-data';
import type {CoreLensRunProps} from './core-lens-run';

registerLensScenario({
  id: 'gpu-core.segmented-sort',
  entry: 'gpu-core',
  element: 'GPUSegmentedSort',
  title: 'Stations ranked by rainfall within each state',
  audience: 'engineer',
  defaultRung: 'result',
  rungs: ['result', 'kernel', 'graph', 'datum', 'table'],
  dataset: CORE_DATASET,

  load: async () => {
    const [
      {runCoreLens, describeCoreLens},
      {runExample},
      data,
      {getCoreStationInputs, getCoreWindow}
    ] = await Promise.all([
      import('./core-lens-run'),
      import('./segmented-sort.example'),
      import('../../data/k-core.json').then(module => module.default),
      import('./k-core-data')
    ]);
    const input = getCoreStationInputs(data);
    const {stateOffsets: offsets} = data.stations;
    // Each state is one independent sort domain; the sort supports domains of at most 256 rows.
    const rainfallTenths = Uint32Array.from(data.stations.prcpMm, value => Math.round(value * 10));
    const firstStateCode = data.stations.states[offsets.indexOf(input.first)];
    const props: CoreLensRunProps = {
      scenarioId: 'gpu-core.segmented-sort',
      element: 'GPUSegmentedSort',
      label: 'Stations ranked by rainfall within each state',
      keySpaces: [
        {id: 'stations', label: 'Station', size: 1024, indexSpace: '1d'},
        {id: 'state-offsets', label: 'State start', size: offsets.length, indexSpace: '1d'}
      ],

      async build(graph, uploads) {
        const rainfallTenthsMm = uploads.view('rainfallTenthsMm', rainfallTenths, 'uint32');
        const stationIndex = uploads.view(
          'stationIndex',
          Uint32Array.from({length: 1024}, (_, index) => index),
          'uint32'
        );
        const stateOffsets = uploads.view('stateOffsets', Uint32Array.from(offsets), 'uint32');
        const sortedRainfallBuffer = uploads.view(
          'sortedRainfall',
          new Uint32Array(1024),
          'uint32'
        ).buffer;
        const sortedStationsBuffer = uploads.view(
          'sortedStations',
          new Uint32Array(1024),
          'uint32'
        ).buffer;
        const {sortedRainfall} = await graph.withRecordScope({datum: {keySpace: 'stations'}}, () =>
          runExample(graph, {
            rainfallTenthsMm,
            stationIndex,
            stateOffsets: offsets,
            sortedRainfallBuffer,
            sortedStationsBuffer
          })
        );
        return {
          rainfallTenthsMm: {
            view: getCoreWindow(graph, rainfallTenthsMm, input.first),
            role: 'input',
            keySpace: 'stations',
            meaning: `bars first:${input.first}`
          },
          stateOffsets: {
            view: stateOffsets,
            role: 'context',
            keySpace: 'state-offsets',
            meaning: 'segments-offsets'
          },
          sortedRainfall: {
            view: getCoreWindow(graph, sortedRainfall, input.first),
            role: 'output',
            keySpace: 'stations',
            meaning: `bars first:${input.first}`,
            highlight: (values: ArrayLike<number>) => ({
              key: input.first,
              label: `${firstStateCode} wettest = ${(values[0] / 10).toFixed(1)} mm`
            })
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
