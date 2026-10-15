// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';
import {CORE_DATASET} from './k-core-data';
import type {CoreLensRunProps} from './core-lens-run';

registerLensScenario({
  id: 'gpu-core.convolution',
  entry: 'gpu-core',
  element: 'GPUConvolution',
  title: 'Laplacian curvature of a 64 × 64 Alps elevation tile',
  audience: 'engineer',
  defaultRung: 'result',
  rungs: ['result', 'kernel', 'graph', 'datum', 'formula', 'table'],
  dataset: CORE_DATASET,

  load: async () => {
    const [{runCoreLens, describeCoreLens}, {runExample}, data] = await Promise.all([
      import('./core-lens-run'),
      import('./convolution.example'),
      import('../../data/k-core.json').then(module => module.default)
    ]);
    const {width, height} = data.dem;
    // Four-neighbour Laplacian; it is symmetric, so convolution and correlation agree.
    const stencil = Float32Array.of(0, 1, 0, 1, -4, 1, 0, 1, 0);
    const props: CoreLensRunProps = {
      scenarioId: 'gpu-core.convolution',
      element: 'GPUConvolution',
      label: 'Laplacian curvature of a 64 × 64 Alps elevation tile',
      keySpaces: [
        {
          id: 'tile-cells',
          label: 'Tile cell',
          size: width * height,
          shape: [width, height],
          indexSpace: '2d'
        },
        {id: 'stencil-cells', label: 'Stencil cell', size: 9, shape: [3, 3], indexSpace: '2d'}
      ],

      async build(graph, uploads) {
        const elevation = uploads.view(
          'elevation',
          Float32Array.from(data.dem.elevationM),
          'float32'
        );
        const laplacian = uploads.view('laplacian', stencil, 'float32');
        const curvatureBuffer = uploads.view(
          'curvature',
          new Float32Array(width * height),
          'float32'
        ).buffer;
        // The direct kernel runs one invocation per output cell.
        const {curvature} = await graph.withRecordScope({datum: {keySpace: 'tile-cells'}}, () =>
          runExample(graph, {elevation, laplacian, curvatureBuffer})
        );
        return {
          elevation: {view: elevation, role: 'input', keySpace: 'tile-cells'},
          laplacian: {
            view: laplacian,
            role: 'context',
            keySpace: 'stencil-cells',
            meaning: 'stencil operator:→'
          },
          curvature: {
            view: curvature,
            role: 'output',
            keySpace: 'tile-cells',
            meaning: 'operator:→',
            highlight: (values: ArrayLike<number>) => {
              // Interior cells only: the outer ring reads zero outside the tile.
              let key = width + 1;
              for (let row = 1; row < height - 1; row++)
                for (let column = 1; column < width - 1; column++)
                  if (values[row * width + column] > values[key]) key = row * width + column;
              return {
                key,
                label: `sharpest hollow: row ${Math.floor(key / width)}, column ${key % width}, +${values[key]} m`
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
