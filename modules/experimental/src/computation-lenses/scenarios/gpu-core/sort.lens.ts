// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';
import {CORE_DATASET} from './k-core-data';
import type {CoreLensRunProps} from './core-lens-run';

registerLensScenario({
  id: 'gpu-core.sort',
  entry: 'gpu-core',
  element: 'GPUSort',
  title: 'Stations ranked by rainfall',
  audience: 'engineer',
  defaultRung: 'result',
  rungs: ['result', 'kernel', 'graph', 'datum', 'table'],
  dataset: CORE_DATASET,

  load: async () => {
    const [{runCoreLens, describeCoreLens}, {runExample}, data, {getCoreWindow}] =
      await Promise.all([
        import('./core-lens-run'),
        import('./sort.example'),
        import('../../data/k-core.json').then(module => module.default),
        import('./k-core-data')
      ]);
    // Radix sorts u32 keys: one-decimal millimetres are exact integers in tenths of a millimetre.
    const rainfallTenths = Uint32Array.from(data.stations.prcpMm, value => Math.round(value * 10));
    const keyBits = Math.max(1, Math.ceil(Math.log2(Math.max(...rainfallTenths) + 1)));
    // The input window is centred on the wettest station so its travel to rank 0 is drawn.
    const wettest = rainfallTenths.indexOf(Math.max(...rainfallTenths));
    const windowStart = Math.min(wettest - 7, 1024 - 16);
    const props: CoreLensRunProps = {
      scenarioId: 'gpu-core.sort',
      element: 'GPUSort',
      label: 'Stations ranked by rainfall',
      keySpaces: [
        {id: 'stations', label: 'Station', size: 1024, indexSpace: '1d'},
        {id: 'ranks', label: 'Rank', size: 1024, indexSpace: '1d'}
      ],

      async build(graph, uploads) {
        const rainfallTenthsMm = uploads.view('rainfallTenthsMm', rainfallTenths, 'uint32');
        const stationIndex = uploads.view(
          'stationIndex',
          Uint32Array.from({length: 1024}, (_, index) => index),
          'uint32'
        );
        const rankedRainfallBuffer = uploads.view(
          'rankedRainfall',
          new Uint32Array(1024),
          'uint32'
        ).buffer;
        const rankedStationsBuffer = uploads.view(
          'rankedStations',
          new Uint32Array(1024),
          'uint32'
        ).buffer;
        const {rankedRainfall, rankedStations} = await graph.withRecordScope(
          {datum: {keySpace: 'stations'}},
          () =>
            runExample(graph, {
              rainfallTenthsMm,
              stationIndex,
              rankedRainfallBuffer,
              rankedStationsBuffer,
              keyBits
            })
        );
        return {
          rainfallTenthsMm: {
            view: getCoreWindow(graph, rainfallTenthsMm, windowStart),
            role: 'input',
            keySpace: 'stations',
            meaning: `bars first:${windowStart}`
          },
          rankedRainfall: {
            view: getCoreWindow(graph, rankedRainfall),
            role: 'output',
            keySpace: 'ranks',
            meaning: 'bars',
            highlight: (values: ArrayLike<number>) => ({
              key: 0,
              label: `rank 0 = ${(values[0] / 10).toFixed(1)} mm`
            })
          },
          rankedStations: {
            view: getCoreWindow(graph, rankedStations),
            role: 'context',
            keySpace: 'ranks',
            meaning: 'index-gather'
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
