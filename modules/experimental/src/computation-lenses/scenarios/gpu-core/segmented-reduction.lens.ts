// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';
import {CORE_DATASET} from './k-core-data';
import type {CoreLensRunProps} from './core-lens-run';

registerLensScenario({
  id: 'gpu-core.segmented-reduction',
  entry: 'gpu-core',
  element: 'GPUSegmentedReduction',
  title: 'Rainfall total per state',
  audience: 'engineer',
  defaultRung: 'result',
  rungs: ['result', 'kernel', 'graph', 'datum', 'formula', 'table'],
  dataset: CORE_DATASET,

  load: async () => {
    const [
      {runCoreLens, describeCoreLens},
      {runExample},
      data,
      {getCoreStationInputs, getCoreWindow}
    ] = await Promise.all([
      import('./core-lens-run'),
      import('./segmented-reduction.example'),
      import('../../data/k-core.json').then(module => module.default),
      import('./k-core-data')
    ]);
    const input = getCoreStationInputs(data);
    const {states, stateOffsets: offsets} = data.stations;
    // The output window starts at the first state of the input window.
    const firstState = offsets.findIndex(offset => offset === input.first);
    const props: CoreLensRunProps = {
      scenarioId: 'gpu-core.segmented-reduction',
      element: 'GPUSegmentedReduction',
      label: 'Rainfall total per state',
      keySpaces: [
        {id: 'stations', label: 'Station', size: 1024, indexSpace: '1d'},
        {id: 'states', label: 'State', size: states.length, indexSpace: '1d'},
        {id: 'state-offsets', label: 'State start', size: offsets.length, indexSpace: '1d'}
      ],

      async build(graph, uploads) {
        const rainfall = uploads.view('rainfall', input.rainfallMm, 'float32');
        const stateOffsets = uploads.view('stateOffsets', Uint32Array.from(offsets), 'uint32');
        const stateTotalsBuffer = uploads.view(
          'stateTotals',
          new Float32Array(states.length),
          'float32'
        ).buffer;
        const {stateTotals} = await graph.withRecordScope({datum: {keySpace: 'states'}}, () =>
          runExample(graph, {rainfall, stateOffsets, stateTotalsBuffer})
        );
        return {
          rainfall: {
            view: getCoreWindow(graph, rainfall, input.first),
            role: 'input',
            keySpace: 'stations',
            meaning: `first:${input.first}`
          },
          stateOffsets: {
            view: stateOffsets,
            role: 'context',
            keySpace: 'state-offsets',
            meaning: 'segments-offsets'
          },
          stateTotals: {
            view: getCoreWindow(graph, stateTotals, firstState),
            role: 'output',
            keySpace: 'states',
            meaning: `first:${firstState}`,
            highlight: (values: ArrayLike<number>) => ({
              key: firstState,
              label: `${states[firstState]} total = ${values[0].toFixed(1)} mm`
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
