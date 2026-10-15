// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';
import {CORE_DATASET} from './k-core-data';
import type {CoreLensRunProps} from './core-lens-run';

registerLensScenario({
  id: 'gpu-core.histogram',
  entry: 'gpu-core',
  element: 'GPUHistogram',
  title: 'Stations per 20 mm rainfall bin',
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
      import('./histogram.example'),
      import('../../data/k-core.json').then(module => module.default),
      import('./k-core-data')
    ]);
    const input = getCoreStationInputs(data);
    const edges = Array.from({length: 17}, (_, index) => index * 20).join(',');
    const props: CoreLensRunProps = {
      scenarioId: 'gpu-core.histogram',
      element: 'GPUHistogram',
      label: 'Stations per 20 mm rainfall bin',
      keySpaces: [
        {id: 'stations', label: 'Station', size: 1024, indexSpace: '1d'},
        {id: 'bins', label: '20 mm bin', size: 16, indexSpace: '1d'}
      ],

      async build(graph, uploads) {
        const rainfall = uploads.view('rainfall', input.rainfallMm, 'float32');
        const stationsPerBinBuffer = uploads.view(
          'stationsPerBin',
          new Uint32Array(16),
          'uint32'
        ).buffer;
        const {stationsPerBin} = await graph.withRecordScope({datum: {keySpace: 'stations'}}, () =>
          runExample(graph, {rainfall, stationsPerBinBuffer})
        );
        return {
          rainfall: {view: getCoreWindow(graph, rainfall), role: 'input', keySpace: 'stations'},
          stationsPerBin: {
            view: stationsPerBin,
            role: 'output',
            keySpace: 'bins',
            meaning: `bars edges:${edges}`,
            highlight: (values: ArrayLike<number>) => {
              const counts = Array.from(values);
              const key = counts.indexOf(Math.max(...counts));
              return {
                key,
                label: `modal bin ${key * 20}–${key * 20 + 20} mm: ${values[key]} stations`
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
