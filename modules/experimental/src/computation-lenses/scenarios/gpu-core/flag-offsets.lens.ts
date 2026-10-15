// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';
import {CORE_DATASET} from './k-core-data';
import type {CoreLensRunProps} from './core-lens-run';

registerLensScenario({
  id: 'gpu-core.flag-offsets',
  entry: 'gpu-core',
  element: 'GPUFlagOffsets',
  title: 'Dense offsets for wet stations',
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
      import('./flag-offsets.example'),
      import('../../data/k-core.json').then(module => module.default),
      import('./k-core-data')
    ]);
    const input = getCoreStationInputs(data);
    const props: CoreLensRunProps = {
      scenarioId: 'gpu-core.flag-offsets',
      element: 'GPUFlagOffsets',
      label: 'Dense offsets for wet stations',

      keySpaces: [
        {id: 'stations', label: 'Station', size: 1024, indexSpace: '1d'},
        {id: 'scalar', label: 'Scalar', size: 1, indexSpace: '1d'},
        {id: 'selected-stations', label: 'Selected station', indexSpace: '1d'}
      ],

      async build(graph, uploads) {
        const wet = uploads.view('wet', input.wet, 'uint32');
        const wetOffsetsBuffer = uploads.view('wetOffsets', new Uint32Array(1024), 'uint32').buffer;
        const wetCountBuffer = uploads.view('wetCount', new Uint32Array(1), 'uint32').buffer;
        const {wetOffsets, wetCount} = await graph.withRecordScope(
          {datum: {keySpace: 'stations'}},
          () => runExample(graph, {wet, wetOffsetsBuffer, wetCountBuffer})
        );
        return {
          wet: {
            view: getCoreWindow(graph, wet),
            role: 'input',
            keySpace: 'stations',
            meaning: 'flags'
          },
          wetOffsets: {
            view: getCoreWindow(graph, wetOffsets),
            role: 'output',
            keySpace: 'stations'
          },
          wetCount: {view: wetCount, role: 'output', keySpace: 'scalar', meaning: 'count'}
        };
      }
    };
    return {
      run: device => runCoreLens(device, props),
      describeStructure: device => describeCoreLens(device, props)
    };
  }
});
