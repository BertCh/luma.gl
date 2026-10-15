// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';
import {CORE_DATASET} from './k-core-data';
import type {CoreLensRunProps} from './core-lens-run';

registerLensScenario({
  id: 'gpu-core.mat-mul',
  entry: 'gpu-core',
  element: 'GPUMatMul',
  title: 'East-west elevation steps of an 8 × 8 Alps block',
  audience: 'engineer',
  defaultRung: 'result',
  rungs: ['result', 'kernel', 'graph', 'datum', 'formula', 'table'],
  dataset: CORE_DATASET,

  load: async () => {
    const [{runCoreLens, describeCoreLens}, {runExample}, data] = await Promise.all([
      import('./core-lens-run'),
      import('./mat-mul.example'),
      import('../../data/k-core.json').then(module => module.default)
    ]);
    // Tile rows and columns 24–31 of the 64 × 64 DEM, row-major from the north-west, in m.
    const block = Float32Array.from(
      {length: 64},
      (_, index) =>
        data.dem.elevationM[(24 + Math.floor(index / 8)) * data.dem.width + 24 + (index % 8)]
    );
    // B[j][j] = -1 and B[j + 1][j] = 1, so (A · B)[r][c] = A[r][c + 1] - A[r][c]; the last column is zero.
    const difference = new Float32Array(64);
    for (let column = 0; column < 7; column++) {
      difference[column * 8 + column] = -1;
      difference[(column + 1) * 8 + column] = 1;
    }
    const props: CoreLensRunProps = {
      scenarioId: 'gpu-core.mat-mul',
      element: 'GPUMatMul',
      label: 'East-west elevation steps of an 8 × 8 Alps block',
      keySpaces: [
        {id: 'left-cells', label: 'Left cell', size: 64, shape: [8, 8], indexSpace: '2d'},
        {id: 'right-cells', label: 'Right cell', size: 64, shape: [8, 8], indexSpace: '2d'},
        {id: 'product-cells', label: 'Product cell', size: 64, shape: [8, 8], indexSpace: '2d'}
      ],

      async build(graph, uploads) {
        const elevation = uploads.view('elevation', block, 'float32');
        const forwardDifference = uploads.view('forwardDifference', difference, 'float32');
        const eastWestStepsBuffer = uploads.view(
          'eastWestSteps',
          new Float32Array(64),
          'float32'
        ).buffer;
        // One 16 × 16 workgroup tile covers the block; the datum is the output cell.
        const {eastWestSteps} = await graph.withRecordScope(
          {datum: {keySpace: 'product-cells'}},
          () => runExample(graph, {elevation, forwardDifference, eastWestStepsBuffer})
        );
        return {
          elevation: {view: elevation, role: 'input', keySpace: 'left-cells'},
          forwardDifference: {
            view: forwardDifference,
            role: 'input',
            keySpace: 'right-cells',
            meaning: 'operator:×'
          },
          eastWestSteps: {
            view: eastWestSteps,
            role: 'output',
            keySpace: 'product-cells',
            meaning: 'operator:=',
            highlight: (values: ArrayLike<number>) => {
              const steps = Array.from(values);
              const key = steps.indexOf(
                steps.reduce((a, b) => (Math.abs(b) > Math.abs(a) ? b : a))
              );
              return {
                key,
                label: `steepest step: row ${Math.floor(key / 8)}, column ${key % 8}, ${values[key] > 0 ? '+' : ''}${values[key]} m`
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
