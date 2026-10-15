// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';
import {CORE_DATASET} from './k-core-data';
import type {CoreLensRunProps} from './core-lens-run';

for (const segmented of [false, true]) {
  registerLensScenario({
    id: segmented ? 'gpu-core.scan.segmented' : 'gpu-core.scan',
    entry: 'gpu-core',
    element: 'GPUScan',
    title: 'Inclusive rainfall totals',
    audience: 'engineer',
    defaultRung: 'result',
    rungs: ['result', 'kernel', 'iterations', 'graph', 'datum', 'formula', 'table'],
    dataset: CORE_DATASET,

    load: async () => {
      const [
        {runCoreLens, describeCoreLens},
        {runExample},
        data,
        {getCoreStationInputs, getCoreWindow}
      ] = await Promise.all([
        import('./core-lens-run'),
        import('./scan.example'),
        import('../../data/k-core.json').then(module => module.default),
        import('./k-core-data')
      ]);
      const input = getCoreStationInputs(data);
      const props: CoreLensRunProps = {
        scenarioId: segmented ? 'gpu-core.scan.segmented' : 'gpu-core.scan',
        element: 'GPUScan',
        label: 'Inclusive rainfall totals',
        at: 'step',
        keySpaces: [
          {id: 'stations', label: 'Station', size: 1024, indexSpace: '1d'},
          {id: 'scalar', label: 'Scalar', size: 1, indexSpace: '1d'},
          {id: 'selected-stations', label: 'Selected station', indexSpace: '1d'}
        ],

        async build(graph, uploads) {
          const rainfall = uploads.view('rainfall', input.rainfall, 'uint32');
          const rainTotalsBuffer = uploads.view(
            'rainTotals',
            new Uint32Array(1024),
            'uint32'
          ).buffer;
          const segmentFlags = segmented
            ? uploads.view('stateStarts', input.segmentFlags, 'uint32')
            : undefined;
          const {rainTotals} = await graph.withRecordScope({datum: {keySpace: 'stations'}}, () =>
            runExample(graph, {rainfall, rainTotalsBuffer, segmentFlags})
          );
          const first = segmented ? input.first : 0;
          const origin = segmented ? `first:${first}` : '';
          return {
            rainfall: {
              view: getCoreWindow(graph, rainfall, first),
              role: 'input',
              keySpace: 'stations',
              meaning: origin
            },
            rainTotals: {
              view: getCoreWindow(graph, rainTotals, first),
              role: 'output',
              keySpace: 'stations',
              meaning: `prefix ${origin}`.trim(),
              ...(!segmented && {
                highlight: (values: ArrayLike<number>) => ({
                  key: 9,
                  label: `output[9] = input[0] + … + input[9] = ${values[9]}`
                })
              })
            },
            ...(segmentFlags && {
              stateStarts: {
                view: getCoreWindow(graph, segmentFlags, first),
                role: 'context' as const,
                keySpace: 'stations',
                meaning: `segments-flags first:${first}`
              }
            })
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
