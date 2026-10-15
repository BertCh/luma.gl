// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';
import {CORE_DATASET} from './k-core-data';
import type {CoreLensRunProps} from './core-lens-run';

registerLensScenario({
  id: 'gpu-core.group-aggregation',
  entry: 'gpu-core',
  element: 'GPUGroupAggregation',
  title: 'Station count, rainfall total and mean per state',
  audience: 'engineer',
  defaultRung: 'result',
  rungs: ['result', 'kernel', 'graph', 'datum', 'formula', 'table'],
  dataset: CORE_DATASET,

  load: async () => {
    const [{runCoreLens, describeCoreLens}, {runExample}, data, {getCoreWindow}] =
      await Promise.all([
        import('./core-lens-run'),
        import('./group-aggregation.example'),
        import('../../data/k-core.json').then(module => module.default),
        import('./k-core-data')
      ]);
    const {state, prcpMm, states} = data.stations;
    const props: CoreLensRunProps = {
      scenarioId: 'gpu-core.group-aggregation',
      element: 'GPUGroupAggregation',
      label: 'Station count, rainfall total and mean per state',
      keySpaces: [
        {id: 'stations', label: 'Station', size: state.length, indexSpace: '1d'},
        {id: 'states', label: 'State', size: states.length, indexSpace: '1d'}
      ],

      async build(graph, uploads) {
        const stationState = uploads.view('stationState', Uint32Array.from(state), 'uint32');
        const rainfall = uploads.view('rainfall', Float32Array.from(prcpMm), 'float32');
        const stationCountsBuffer = uploads.view(
          'stationCounts',
          new Uint32Array(states.length),
          'uint32'
        ).buffer;
        const rainTotalsBuffer = uploads.view(
          'rainTotals',
          new Float32Array(states.length),
          'float32'
        ).buffer;
        const rainMeansBuffer = uploads.view(
          'rainMeans',
          new Float32Array(states.length),
          'float32'
        ).buffer;
        const {stationCounts, rainTotals, rainMeans} = await graph.withRecordScope(
          {datum: {keySpace: 'stations'}},
          () =>
            runExample(graph, {
              stationState,
              rainfall,
              stateCount: states.length,
              stationCountsBuffer,
              rainTotalsBuffer,
              rainMeansBuffer
            })
        );
        return {
          stationState: {
            view: getCoreWindow(graph, stationState),
            role: 'input',
            keySpace: 'stations',
            meaning: 'column:state'
          },
          rainfall: {
            view: getCoreWindow(graph, rainfall),
            role: 'input',
            keySpace: 'stations',
            meaning: 'column:rainfall_(mm)'
          },
          stationCounts: {
            view: stationCounts,
            role: 'output',
            keySpace: 'states',
            meaning: 'column:stations'
          },
          rainTotals: {
            view: rainTotals,
            role: 'output',
            keySpace: 'states',
            meaning: 'column:total_(mm)'
          },
          rainMeans: {
            view: rainMeans,
            role: 'output',
            keySpace: 'states',
            meaning: 'column:mean_(mm)'
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
