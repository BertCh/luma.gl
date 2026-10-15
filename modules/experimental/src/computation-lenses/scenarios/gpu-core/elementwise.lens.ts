// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';
import {CORE_DATASET} from './k-core-data';
import type {CoreLensRunProps} from './core-lens-run';

/** Registers the elementwise scenario, or its DispatchView id-probe variant. */
export function registerElementwiseScenario(idProbe: boolean): void {
  registerLensScenario({
    id: idProbe ? 'gpu-core.elementwise.id-probe' : 'gpu-core.elementwise',
    entry: 'gpu-core',
    element: 'GPUElementwise',
    title: 'Two-day rainfall per station',
    audience: 'engineer',
    defaultRung: 'result',
    rungs: idProbe
      ? ['result', 'kernel', 'graph']
      : ['result', 'kernel', 'graph', 'datum', 'formula', 'table'],
    dataset: CORE_DATASET,

    load: async () => {
      const [{runCoreLens, describeCoreLens}, {runExample}, data, {getCoreWindow}] =
        await Promise.all([
          import('./core-lens-run'),
          import('./elementwise.example'),
          import('../../data/k-core.json').then(module => module.default),
          import('./k-core-data')
        ]);
      // The probe variant uses 1,000 rows so the last 256-lane workgroup has idle lanes.
      const count = idProbe ? 1000 : 1024;
      const rainfallMm = Float32Array.from(data.stations.prcpMm).subarray(0, count);
      const prevDayMm = Float32Array.from(data.stations.prcpPrevDayMm).subarray(0, count);
      const props: CoreLensRunProps = {
        scenarioId: idProbe ? 'gpu-core.elementwise.id-probe' : 'gpu-core.elementwise',
        element: 'GPUElementwise',
        label: 'Two-day rainfall per station',
        keySpaces: [{id: 'stations', label: 'Station', size: count, indexSpace: '1d'}],
        ...(idProbe && {idProbe: {op: 'GPUElementwise'}}),

        async build(graph, uploads) {
          const rainfall = uploads.view('rainfall', rainfallMm, 'float32');
          const prevDayRainfall = uploads.view('prevDayRainfall', prevDayMm, 'float32');
          const twoDayRainfallBuffer = uploads.view(
            'twoDayRainfall',
            new Float32Array(count),
            'float32'
          ).buffer;
          const {twoDayRainfall} = await graph.withRecordScope(
            {datum: {keySpace: 'stations'}},
            () => runExample(graph, {rainfall, prevDayRainfall, twoDayRainfallBuffer})
          );
          return {
            rainfall: {view: getCoreWindow(graph, rainfall), role: 'input', keySpace: 'stations'},
            prevDayRainfall: {
              view: getCoreWindow(graph, prevDayRainfall),
              role: 'input',
              keySpace: 'stations',
              meaning: 'operator:+'
            },
            twoDayRainfall: {
              view: getCoreWindow(graph, twoDayRainfall),
              role: 'output',
              keySpace: 'stations',
              meaning: 'operator:='
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
}

registerElementwiseScenario(false);
