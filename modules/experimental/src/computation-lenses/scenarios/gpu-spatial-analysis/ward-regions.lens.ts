// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';

registerLensScenario({
  id: 'gpu-spatial-analysis.ward-regions',
  entry: 'gpu-spatial-analysis',
  element: 'GPUWardRegions',
  title: 'Ward regions: north-side Chicago tracts by poverty',
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
      import('./ward-regions.example'),
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
            id: 'gpu-spatial-analysis.ward-regions',
            tracts,
            entry: STATISTICS_OPERATORS['GPUWardRegions'],
            columns: {values: 'poverty150Pct'},
            choices: {
              criterion: 'queen',
              standardize: true,
              targetRegionCount: 5,
              mergeSchedule: 'mutual-nearest',
              seed: 0
            },
            async execute({graph, polygons, columns, upload}) {
              const parameters = upload(
                'parameters',
                spatial.getGPUWardRegionsParameterValues({seed: 0}),
                'uint32'
              );
              const execution = await runExample(device, {
                graph,
                tracts: polygons,
                values: columns['values'],
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
                  labels: {
                    role: 'output',
                    polygons: true,
                    meaning: 'Ward region (nominal; label is the smallest tract row)'
                  },
                  objective: {
                    role: 'intermediate',
                    keySpace: 'status',
                    meaning: 'Within-region sum of squares (standardized units)'
                  },
                  status: {
                    role: 'intermediate',
                    keySpace: 'status',
                    meaning: 'Merges, converged, limit reached, invalid count, merge rounds'
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
