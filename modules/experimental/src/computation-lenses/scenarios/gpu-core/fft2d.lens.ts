// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';
import {CORE_DATASET} from './k-core-data';
import type {CoreLensRunProps} from './core-lens-run';

registerLensScenario({
  id: 'gpu-core.fft2d',
  entry: 'gpu-core',
  element: 'GPUFFT2D',
  title: 'Spatial frequencies of a 64 × 64 Alps elevation tile',
  audience: 'engineer',
  defaultRung: 'result',
  rungs: ['result', 'kernel', 'graph', 'datum', 'formula', 'table'],
  dataset: CORE_DATASET,

  load: async () => {
    const [{runCoreLens, describeCoreLens}, {runExample}, data] = await Promise.all([
      import('./core-lens-run'),
      import('./fft2d.example'),
      import('../../data/k-core.json').then(module => module.default)
    ]);
    const {width, height} = data.dem;
    // The tile as a complex field: real part = elevation (mean NOT removed), imaginary part = 0.
    const tile = new Float32Array(width * height * 2);
    data.dem.elevationM.forEach((value, index) => {
      tile[index * 2] = value;
    });
    const props: CoreLensRunProps = {
      scenarioId: 'gpu-core.fft2d',
      element: 'GPUFFT2D',
      label: 'Spatial frequencies of a 64 × 64 Alps elevation tile',
      keySpaces: [
        {
          id: 'tile-cells',
          label: 'Tile cell',
          size: width * height,
          shape: [width, height],
          indexSpace: '2d'
        },
        {
          id: 'frequency-cells',
          label: 'Frequency (u, v)',
          size: width * height,
          shape: [width, height],
          indexSpace: '2d'
        }
      ],

      async build(graph, uploads) {
        const elevationTile = uploads.view('elevationTile', tile, 'float32x2');
        const spectrumBuffer = uploads.view(
          'spectrum',
          new Float32Array(width * height * 2),
          'float32x2'
        ).buffer;
        // Each pass runs one invocation per tile cell.
        const {spectrum} = await graph.withRecordScope({datum: {keySpace: 'tile-cells'}}, () =>
          runExample(graph, {elevationTile, spectrumBuffer})
        );
        return {
          elevationTile: {view: elevationTile, role: 'input', keySpace: 'tile-cells'},
          spectrum: {
            view: spectrum,
            role: 'output',
            keySpace: 'frequency-cells',
            meaning: 'complex-magnitude log operator:→',
            highlight: (values: ArrayLike<number>) => {
              // The DC term (0, 0) is the sum of all elevations; name the strongest other frequency.
              let key = 1;
              for (let cell = 2; cell < values.length / 2; cell++)
                if (
                  Math.hypot(values[cell * 2], values[cell * 2 + 1]) >
                  Math.hypot(values[key * 2], values[key * 2 + 1])
                )
                  key = cell;
              const magnitude = Math.hypot(values[key * 2], values[key * 2 + 1]);
              return {
                key,
                label: `strongest non-DC frequency (u, v) = (${key % width}, ${Math.floor(key / width)}): |Y| = ${Math.round(magnitude)} m`
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
