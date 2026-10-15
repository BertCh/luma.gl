// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';
import {registerLensScenario} from '../../registry';
import {CORE_DATASET} from './k-core-data';
import type {CoreLensRunProps} from './core-lens-run';

registerLensScenario({
  id: 'gpu-core.gather',
  entry: 'gpu-core',
  element: 'GPUGather',
  title: 'Previous-day rainfall of one state',
  audience: 'engineer',
  defaultRung: 'result',
  rungs: ['result', 'kernel', 'graph', 'datum', 'formula', 'table'],
  dataset: CORE_DATASET,

  load: async () => {
    const [{runCoreLens, describeCoreLens}, {runExample}, data, {getCoreWindow}] =
      await Promise.all([
        import('./core-lens-run'),
        import('./gather.example'),
        import('../../data/k-core.json').then(module => module.default),
        import('./k-core-data')
      ]);
    const {states, stateOffsets: offsets} = data.stations;
    // The first state that fits one 16-row source window and had real previous-day rain (≥ 50 mm
    // somewhere), so the gathered output is not all zeros.
    const stateIndex = states.findIndex(
      (_, index) =>
        index < offsets.length - 1 &&
        offsets[index + 1] - offsets[index] <= 16 &&
        data.stations.prcpPrevDayMm
          .slice(offsets[index], offsets[index + 1])
          .some(rainfall => rainfall >= 50)
    );
    // The selected dataset has such a state (WV, 12 stations).
    assert(stateIndex >= 0);
    const first = offsets[stateIndex];
    const stateStationIndices = Uint32Array.from(
      {length: offsets[stateIndex + 1] - first},
      (_, index) => first + index
    );
    const props: CoreLensRunProps = {
      scenarioId: 'gpu-core.gather',
      element: 'GPUGather',
      label: 'Previous-day rainfall of one state',
      keySpaces: [
        {id: 'stations', label: 'Station', size: 1024, indexSpace: '1d'},
        {
          id: 'state-stations',
          label: `${states[stateIndex]} station`,
          size: stateStationIndices.length,
          indexSpace: '1d'
        }
      ],

      async build(graph, uploads) {
        const prevDayRainfall = uploads.view(
          'prevDayRainfall',
          Float32Array.from(data.stations.prcpPrevDayMm),
          'float32'
        );
        const stateStations = uploads.view('stateStations', stateStationIndices, 'uint32');
        const statePrevDayRainfallBuffer = uploads.view(
          'statePrevDayRainfall',
          new Float32Array(stateStationIndices.length),
          'float32'
        ).buffer;
        const {statePrevDayRainfall} = await graph.withRecordScope(
          {datum: {keySpace: 'state-stations'}},
          () => runExample(graph, {prevDayRainfall, stateStations, statePrevDayRainfallBuffer})
        );
        return {
          prevDayRainfall: {
            view: getCoreWindow(graph, prevDayRainfall, first),
            role: 'input',
            keySpace: 'stations',
            meaning: `first:${first}`
          },
          stateStations: {
            view: stateStations,
            role: 'context',
            keySpace: 'state-stations',
            meaning: 'index-gather'
          },
          statePrevDayRainfall: {
            view: statePrevDayRainfall,
            role: 'output',
            keySpace: 'state-stations',
            highlight: (values: ArrayLike<number>) => {
              const rainfall = Array.from(values);
              const key = rainfall.indexOf(Math.max(...rainfall));
              return {key, label: `wettest = ${values[key].toFixed(1)} mm`};
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
