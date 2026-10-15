// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';
import {CORE_DATASET} from './k-core-data';
import type {CoreLensRunProps} from './core-lens-run';

registerLensScenario({
  id: 'gpu-core.mask',
  entry: 'gpu-core',
  element: 'GPUMask',
  title: 'Wet stations above 500 metres',
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
      import('./mask.example'),
      import('../../data/k-core.json').then(module => module.default),
      import('./k-core-data')
    ]);
    const input = getCoreStationInputs(data);
    const props: CoreLensRunProps = {
      scenarioId: 'gpu-core.mask',
      element: 'GPUMask',
      label: 'Wet stations above 500 metres',

      keySpaces: [
        {id: 'stations', label: 'Station', size: 1024, indexSpace: '1d'},
        {id: 'scalar', label: 'Scalar', size: 1, indexSpace: '1d'},
        {id: 'selected-stations', label: 'Selected station', indexSpace: '1d'}
      ],

      async build(graph, uploads) {
        const wet = uploads.view('wet', input.wet, 'uint32');
        const high = uploads.view('high', input.high, 'uint32');
        const heavyRainHighStationsBuffer = uploads.view(
          'heavyRainHighStations',
          new Uint32Array(1024),
          'uint32'
        ).buffer;
        const {heavyRainHighStations} = await graph.withRecordScope(
          {datum: {keySpace: 'stations'}},
          () => runExample(graph, {wet, high, heavyRainHighStationsBuffer})
        );
        return {
          wet: {
            view: getCoreWindow(graph, wet),
            role: 'input',
            keySpace: 'stations',
            meaning: 'flags'
          },
          high: {
            view: getCoreWindow(graph, high),
            role: 'input',
            keySpace: 'stations',
            meaning: 'flags'
          },
          heavyRainHighStations: {
            view: getCoreWindow(graph, heavyRainHighStations),
            role: 'output',
            keySpace: 'stations',
            meaning: 'flags'
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
