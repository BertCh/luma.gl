// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';
import {CORE_DATASET} from './k-core-data';
import type {CoreLensRunProps} from './core-lens-run';

registerLensScenario({
  id: 'gpu-core.fft1d',
  entry: 'gpu-core',
  element: 'GPUFFT1D',
  title: 'Spectrum of one Alps elevation profile',
  audience: 'engineer',
  defaultRung: 'result',
  rungs: ['result', 'kernel', 'graph', 'datum', 'formula', 'table'],
  dataset: CORE_DATASET,

  load: async () => {
    const [{runCoreLens, describeCoreLens}, {runExample}, data] = await Promise.all([
      import('./core-lens-run'),
      import('./fft1d.example'),
      import('../../data/k-core.json').then(module => module.default)
    ]);
    // Tile row 32, mean removed on the CPU before upload so that k = 0 does not dominate; the
    // imaginary parts are zero. This is input preparation only: the spectrum is a GPU readback.
    const row = data.dem.elevationM.slice(32 * data.dem.width, 33 * data.dem.width);
    const mean = row.reduce((sum, value) => sum + value, 0) / row.length;
    const profile = new Float32Array(row.length * 2);
    row.forEach((value, index) => {
      profile[index * 2] = value - mean;
    });
    const props: CoreLensRunProps = {
      scenarioId: 'gpu-core.fft1d',
      element: 'GPUFFT1D',
      label: 'Spectrum of one Alps elevation profile',
      keySpaces: [
        {id: 'samples', label: 'Profile sample', size: row.length, indexSpace: '1d'},
        {id: 'frequencies', label: 'Frequency k', size: row.length, indexSpace: '1d'}
      ],

      async build(graph, uploads) {
        const elevationProfile = uploads.view('elevationProfile', profile, 'float32x2');
        const spectrumBuffer = uploads.view(
          'spectrum',
          new Float32Array(row.length * 2),
          'float32x2'
        ).buffer;
        // Each butterfly pass runs one invocation per sample.
        const {spectrum} = await graph.withRecordScope({datum: {keySpace: 'samples'}}, () =>
          runExample(graph, {elevationProfile, spectrumBuffer})
        );
        return {
          elevationProfile: {
            view: elevationProfile,
            role: 'input',
            keySpace: 'samples',
            meaning: 'signal'
          },
          spectrum: {
            // k = 0..31, the non-redundant half of a real input's spectrum.
            view: graph.createDataView(spectrum.buffer, {
              format: 'float32x2',
              length: row.length / 2,
              byteOffset: spectrum.byteOffset
            }),
            role: 'output',
            keySpace: 'frequencies',
            meaning: 'complex-magnitude bars operator:→',
            highlight: (values: ArrayLike<number>) => {
              let key = 1;
              for (let k = 2; k < values.length / 2; k++)
                if (
                  Math.hypot(values[k * 2], values[k * 2 + 1]) >
                  Math.hypot(values[key * 2], values[key * 2 + 1])
                )
                  key = k;
              const magnitude = Math.hypot(values[key * 2], values[key * 2 + 1]);
              return {key, label: `strongest wave k = ${key}: |Y| = ${Math.round(magnitude)} m`};
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
