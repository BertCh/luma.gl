// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';
import {CORE_DATASET} from './k-core-data';
import type {CoreLensRunProps} from './core-lens-run';

registerLensScenario({
  id: 'gpu-core.reduction',
  entry: 'gpu-core',
  element: 'GPUReduction',
  title: 'Wettest station rainfall',
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
      import('./reduction.example'),
      import('../../data/k-core.json').then(module => module.default),
      import('./k-core-data')
    ]);
    const input = getCoreStationInputs(data);
    const props: CoreLensRunProps = {
      scenarioId: 'gpu-core.reduction',
      element: 'GPUReduction',
      label: 'Wettest station rainfall',
      keySpaces: [
        {id: 'stations', label: 'Station', size: 1024, indexSpace: '1d'},
        {id: 'scalar', label: 'Scalar', size: 1, indexSpace: '1d'}
      ],

      async build(graph, uploads) {
        const rainfall = uploads.view('rainfall', input.rainfallMm, 'float32');
        const maxRainfallBuffer = uploads.view(
          'maxRainfall',
          new Float32Array(1),
          'float32'
        ).buffer;
        const {maxRainfall} = await graph.withRecordScope({datum: {keySpace: 'stations'}}, () =>
          runExample(graph, {rainfall, maxRainfallBuffer})
        );
        return {
          rainfall: {view: getCoreWindow(graph, rainfall), role: 'input', keySpace: 'stations'},
          maxRainfall: {
            view: maxRainfall,
            role: 'output',
            keySpace: 'scalar',
            highlight: (values: ArrayLike<number>) => ({
              key: 0,
              label: `max = ${values[0].toFixed(1)} mm`
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
