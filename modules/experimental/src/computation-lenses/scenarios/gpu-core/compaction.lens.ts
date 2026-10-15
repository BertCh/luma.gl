// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';
import {CORE_DATASET} from './k-core-data';
import type {CoreLensRunProps} from './core-lens-run';

registerLensScenario({
  id: 'gpu-core.compaction',
  entry: 'gpu-core',
  element: 'GPUCompaction',
  title: 'Original row identifiers of wet stations',
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
      import('./compaction.example'),
      import('../../data/k-core.json').then(module => module.default),
      import('./k-core-data')
    ]);
    const input = getCoreStationInputs(data);
    const props: CoreLensRunProps = {
      scenarioId: 'gpu-core.compaction',
      element: 'GPUCompaction',
      label: 'Original row identifiers of wet stations',

      keySpaces: [
        {id: 'stations', label: 'Station', size: 1024, indexSpace: '1d'},
        {id: 'scalar', label: 'Scalar', size: 1, indexSpace: '1d'},
        {id: 'selected-stations', label: 'Selected station', indexSpace: '1d'}
      ],

      async build(graph, uploads) {
        const stationRows = uploads.view('stationRows', input.stationRows, 'uint32');
        const wet = uploads.view('wet', input.wet, 'uint32');
        const wetStationsBuffer = uploads.view(
          'wetStations',
          new Uint32Array(1024),
          'uint32'
        ).buffer;
        const wetCountBuffer = uploads.view('wetCount', new Uint32Array(1), 'uint32').buffer;
        const {wetStations, wetCount} = await graph.withRecordScope(
          {datum: {keySpace: 'stations'}},
          () => runExample(graph, {stationRows, wet, wetStationsBuffer, wetCountBuffer})
        );
        return {
          stationRows: {
            view: getCoreWindow(graph, stationRows),
            role: 'input',
            keySpace: 'stations'
          },
          wet: {
            view: getCoreWindow(graph, wet),
            role: 'context',
            keySpace: 'stations',
            meaning: 'flags'
          },
          wetStations: {
            view: getCoreWindow(graph, wetStations),
            role: 'output',
            keySpace: 'selected-stations',
            meaning: 'compact'
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
