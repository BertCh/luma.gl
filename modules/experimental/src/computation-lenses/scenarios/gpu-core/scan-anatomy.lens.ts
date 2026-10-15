// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';
import {CORE_DATASET} from './k-core-data';
import type {CoreLensRunProps} from './core-lens-run';

registerLensScenario({
  id: 'gpu-core.scan.anatomy',
  entry: 'gpu-core',
  element: 'GPUScan',
  title: 'Mask, compaction scan, and rainfall reduction',
  audience: 'engineer',
  defaultRung: 'result',
  rungs: ['result', 'kernel', 'iterations', 'graph', 'datum', 'formula', 'table'],
  dataset: CORE_DATASET,
  example: {path: 'scenarios/gpu-core/scan-anatomy.example.ts'},
  load: async () => {
    const [
      {runCoreLens, describeCoreLens},
      {runExample},
      data,
      {getCoreStationInputs, getCoreWindow}
    ] = await Promise.all([
      import('./core-lens-run'),
      import('./scan-anatomy.example'),
      import('../../data/k-core.json').then(module => module.default),
      import('./k-core-data')
    ]);
    const input = getCoreStationInputs(data);
    const props: CoreLensRunProps = {
      scenarioId: 'gpu-core.scan.anatomy',
      element: 'GPUScan',
      label: 'Mask, compaction scan, and rainfall reduction',

      keySpaces: [
        {id: 'stations', label: 'Station', size: 1024, indexSpace: '1d'},
        {id: 'scalar', label: 'Scalar', size: 1, indexSpace: '1d'},
        {id: 'selected-stations', label: 'Selected station', indexSpace: '1d'}
      ],
      series: compiled =>
        [512, 2048, 8192].map(budget => {
          const steps = compiled.createExecution({maximumInvocationCount: budget}).plan.steps;
          return {
            name: `slices@${budget}`,
            axis: 'node',
            values: compiled.stats.nodeOrder.map((_, index) =>
              steps.findIndex(step => index >= step.firstNodeIndex && index < step.nextNodeIndex)
            )
          };
        }),
      async build(graph, uploads) {
        const wet = uploads.view('wet', input.wet, 'uint32');
        const high = uploads.view('high', input.high, 'uint32');
        const rainfallMm = uploads.view('rainfallMm', input.rainfallMm, 'float32');
        const stationRows = uploads.view('stationRows', input.stationRows, 'uint32');
        const idsBuffer = uploads.view('ids', new Uint32Array(1024), 'uint32').buffer;
        const countBuffer = uploads.view('count', new Uint32Array(1), 'uint32').buffer;
        const totalRainBuffer = uploads.view('totalRain', new Float32Array(1), 'float32').buffer;
        const {ids, count, totalRain} = await graph.withRecordScope(
          {datum: {keySpace: 'stations'}},
          () =>
            runExample(graph, {
              wet,
              high,
              rainfallMm,
              stationRows,
              idsBuffer,
              countBuffer,
              totalRainBuffer
            })
        );
        return {
          stationRows: {
            view: getCoreWindow(graph, stationRows),
            role: 'input',
            keySpace: 'stations'
          },
          rainfallMm: {view: getCoreWindow(graph, rainfallMm), role: 'input', keySpace: 'stations'},
          wet: {
            view: getCoreWindow(graph, wet),
            role: 'context',
            keySpace: 'stations',
            meaning: 'flags'
          },
          high: {
            view: getCoreWindow(graph, high),
            role: 'context',
            keySpace: 'stations',
            meaning: 'flags'
          },
          ids: {view: getCoreWindow(graph, ids), role: 'output', keySpace: 'selected-stations'},
          count: {view: count, role: 'output', keySpace: 'scalar', meaning: 'count'},
          totalRain: {view: totalRain, role: 'output', keySpace: 'scalar'}
        };
      }
    };
    return {
      run: device => runCoreLens(device, props),
      describeStructure: device => describeCoreLens(device, props)
    };
  }
});
