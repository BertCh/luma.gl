// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';

/** Tract whose permutation null distribution is replayed on the GPU (a well-connected tract). */
const REPLAY_ROW = 420;
const PERMUTATIONS = 99;

registerLensScenario({
  id: 'gpu-spatial-analysis.local-permutation-test',
  entry: 'gpu-spatial-analysis',
  element: 'GPULocalPermutationTest',
  title: 'Local Moran permutation test: Chicago poverty percentages',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: {
    id: 's1-chicago-tracts/1',
    source: 'Census cb_2022_17_tract_500k; CDC/ATSDR SVI 2022 (ACS 2018-2022)',
    licence:
      'US Census Bureau cartographic boundary file cb_2022_17_tract_500k, public domain; CDC/ATSDR SVI 2022, public domain'
  },
  async load() {
    const [
      {runExample},
      {loadChicagoTracts},
      utilities,
      spatial,
      gpuCore,
      {GPULocalPermutationReplay},
      {STATISTICS_OPERATORS},
      {Buffer}
    ] = await Promise.all([
      import('./local-permutation-test.example'),
      import('./spatial-lens-data'),
      import('./spatial-scenario-utils'),
      import('@luma.gl/experimental/gpu-spatial-analysis'),
      import('@luma.gl/gpgpu/gpu-core'),
      import('../../../gpu-spatial-analysis/permutation-inference/gpu-local-permutation-replay'),
      import('../../../gpu-spatial-analysis/computation-pack/operators/statistics'),
      import('@luma.gl/core')
    ]);
    const tracts = await loadChicagoTracts();
    return {
      run: device =>
        utilities.runTractStatisticsScenario({
          device,
          id: 'gpu-spatial-analysis.local-permutation-test',
          tracts,
          entry: STATISTICS_OPERATORS['GPULocalPermutationTest'],
          columns: {values: 'poverty150Pct'},
          choices: {
            transform: 'row',
            islands: 'exclude',
            statistic: 'localMoran',
            alternative: 'two-sided',
            permutations: PERMUTATIONS,
            multipleTesting: 'benjamini-hochberg',
            significanceLevel: 0.05
          },
          metadata: {
            pCorrection: 'benjamini-hochberg',
            significanceAvailable: true,
            weightsStandardisation: 'row',
            valueColumn: 'poverty150Pct',
            mappedVariableExtent: 'intensive',
            seed: 1
          },
          async execute({graph, polygons, columns, upload}) {
            const parameters = upload(
              'parameters',
              spatial.getGPUPermutationParameterValues(
                {seed: 1, permutationCount: PERMUTATIONS, significanceLevel: 0.05},
                {maximumPermutations: PERMUTATIONS}
              ),
              'uint32'
            );
            const execution = await runExample(device, {
              graph,
              tracts: polygons,
              values: columns['values'],
              parameters
            });
            // Recording-only instrument: replays the test's own Philox draws for one pinned tract.
            const usage = Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST;
            const nullDistribution = gpuCore.createTransientView(
              graph,
              'nullDistribution',
              'float32',
              PERMUTATIONS,
              usage
            );
            graph.add(
              new GPULocalPermutationReplay({
                test: execution.test,
                keySpace: 'tract',
                row: upload('replayRow', Uint32Array.of(REPLAY_ROW), 'uint32'),
                output: {
                  nullDistribution,
                  replayInfo: gpuCore.createTransientView(graph, 'replayInfo', 'float32', 3, usage)
                }
              })
            );
            return {
              ...execution,
              views: {...execution.views, nullDistribution},
              replays: {
                nullDistribution: {
                  keySpace: 'permutation',
                  meaning: `null distribution (raw local term) of row ${REPLAY_ROW}`
                }
              },
              keySpaces: [
                {
                  id: 'permutation',
                  label: 'Permutation',
                  size: PERMUTATIONS,
                  indexSpace: '1d' as const
                }
              ],
              tags: {
                values: {
                  role: 'input',
                  polygons: true,
                  meaning: 'Population below 150% poverty line (%)'
                },
                pValues: {
                  role: 'output',
                  polygons: true,
                  meaning: 'Two-sided pseudo p-value on the grid k/100 (unadjusted)'
                },
                observed: {role: 'intermediate', meaning: 'Observed local Moran I (esda scaling)'},
                exceedances: {
                  role: 'intermediate',
                  meaning: 'Exceedance count M of the two-sided tail; 4294967295 = not tested'
                },
                adjustedPValues: {
                  role: 'intermediate',
                  meaning: 'Benjamini-Hochberg adjusted pseudo p-value'
                }
              }
            };
          }
        })
    };
  }
});
