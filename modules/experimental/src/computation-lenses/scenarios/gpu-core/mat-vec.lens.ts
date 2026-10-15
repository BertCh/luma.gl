// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';
import {CORE_DATASET} from './k-core-data';
import type {CoreLensRunProps} from './core-lens-run';

registerLensScenario({
  id: 'gpu-core.mat-vec',
  entry: 'gpu-core',
  element: 'GPUMatVec',
  title: 'Row means of an 8 × 8 Alps elevation block',
  audience: 'engineer',
  defaultRung: 'result',
  rungs: ['result', 'kernel', 'graph', 'datum', 'formula', 'table'],
  dataset: CORE_DATASET,

  load: async () => {
    const [{runCoreLens, describeCoreLens}, {runExample}, data] = await Promise.all([
      import('./core-lens-run'),
      import('./mat-vec.example'),
      import('../../data/k-core.json').then(module => module.default)
    ]);
    // Tile rows and columns 24–31 of the 64 × 64 DEM, row-major from the north-west, in m.
    const block = Float32Array.from(
      {length: 64},
      (_, index) =>
        data.dem.elevationM[(24 + Math.floor(index / 8)) * data.dem.width + 24 + (index % 8)]
    );
    const props: CoreLensRunProps = {
      scenarioId: 'gpu-core.mat-vec',
      element: 'GPUMatVec',
      label: 'Row means of an 8 × 8 Alps elevation block',
      keySpaces: [
        {id: 'block-cells', label: 'Block cell', size: 64, shape: [8, 8], indexSpace: '2d'},
        {id: 'block-columns', label: 'Block column', size: 8, shape: [1, 8], indexSpace: '1d'},
        {id: 'block-rows', label: 'Block row', size: 8, shape: [1, 8], indexSpace: '1d'}
      ],

      async build(graph, uploads) {
        const elevation = uploads.view('elevation', block, 'float32');
        const weights = uploads.view('weights', new Float32Array(8).fill(1 / 8), 'float32');
        const rowMeansBuffer = uploads.view('rowMeans', new Float32Array(8), 'float32').buffer;
        // Each workgroup of the kernel reduces one matrix row, so the invocation key is the block row.
        const {rowMeans} = await graph.withRecordScope({datum: {keySpace: 'block-rows'}}, () =>
          runExample(graph, {elevation, weights, rowMeansBuffer})
        );
        return {
          elevation: {view: elevation, role: 'input', keySpace: 'block-cells'},
          weights: {view: weights, role: 'input', keySpace: 'block-columns', meaning: 'operator:×'},
          rowMeans: {
            view: rowMeans,
            role: 'output',
            keySpace: 'block-rows',
            meaning: 'operator:=',
            highlight: (values: ArrayLike<number>) => {
              const key = Array.from(values).indexOf(Math.max(...Array.from(values)));
              return {key, label: `highest row mean: row ${key}, ${values[key].toFixed(1)} m`};
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
