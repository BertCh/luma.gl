// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';

registerLensScenario({
  id: 'gpu-spatial-analysis.max-p-regions',
  entry: 'gpu-spatial-analysis',
  element: 'GPUMaxPRegions',
  title: 'Max-p regions: north-side Chicago tracts of at least 40,000 residents',
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
    const [{runExample}, support, utilities, spatial, {STATISTICS_OPERATORS}] = await Promise.all([
      import('./max-p-regions.example'),
      import('./s1-regions-support'),
      import('./spatial-scenario-utils'),
      import('@luma.gl/experimental/gpu-spatial-analysis'),
      import('../../../gpu-spatial-analysis/computation-pack/operators/statistics')
    ]);
    const tracts = await support.loadRegionTracts();
    return {
      run: async device =>
        support.assertRegionalizationValid(
          await utilities.runTractStatisticsScenario({
            device,
            id: 'gpu-spatial-analysis.max-p-regions',
            tracts,
            entry: STATISTICS_OPERATORS['GPUMaxPRegions'],
            columns: {values: 'poverty150Pct', population: 'population'},
            choices: {
              criterion: 'queen',
              standardize: true,
              threshold: 40000,
              startCount: 8,
              maximumGrowthSteps: 48,
              seed: 0
            },
            async execute({graph, polygons, columns, upload}) {
              const parameters = upload(
                'parameters',
                spatial.getGPUMaxPRegionsParameterValues({seed: 0}),
                'uint32'
              );
              const threshold = upload('threshold', Float32Array.of(40000), 'float32');
              const execution = await runExample(device, {
                graph,
                tracts: polygons,
                values: columns['values'],
                population: columns['population'],
                threshold,
                parameters
              });
              return {
                ...execution,
                tags: {
                  values: {
                    role: 'input',
                    polygons: true,
                    meaning: 'Population below 150% poverty line (%)'
                  },
                  population: {
                    role: 'intermediate',
                    meaning: 'Residents per tract: the extensive total each region must reach'
                  },
                  labels: {
                    role: 'output',
                    polygons: true,
                    meaning: 'Max-p region (nominal, dense label)'
                  },
                  objective: {
                    role: 'intermediate',
                    keySpace: 'status',
                    meaning: 'Within-region sum of squares (standardized units)'
                  },
                  status: {
                    role: 'intermediate',
                    keySpace: 'status',
                    meaning:
                      'Regions, feasible, under-threshold regions, invalid count, start, improvement moves'
                  }
                },
                keySpaces: [{id: 'status', label: 'Scalar and status words', indexSpace: '1d'}]
              };
            }
          }),
          'status',
          3
        )
    };
  }
});
