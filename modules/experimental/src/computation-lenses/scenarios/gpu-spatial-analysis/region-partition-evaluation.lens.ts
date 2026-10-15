// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';

registerLensScenario({
  id: 'gpu-spatial-analysis.region-partition-evaluation',
  entry: 'gpu-spatial-analysis',
  element: 'GPURegionPartitionEvaluation',
  title: 'Scoring community areas as a partition of poverty',
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
    const [{runExample}, support, utilities, {STATISTICS_OPERATORS}] = await Promise.all([
      import('./region-partition-evaluation.example'),
      import('./s1-regions-support'),
      import('./spatial-scenario-utils'),
      import('../../../gpu-spatial-analysis/computation-pack/operators/statistics')
    ]);
    const tracts = await support.loadRegionTracts();
    return {
      run: device =>
        utilities.runTractStatisticsScenario({
          device,
          id: 'gpu-spatial-analysis.region-partition-evaluation',
          tracts,
          entry: STATISTICS_OPERATORS['GPURegionPartitionEvaluation'],
          columns: {values: 'poverty150Pct'},
          choices: {labelCapacity: 8},
          async execute({graph, polygons, columns, upload}) {
            const labels = upload('labels', support.getCommunityAreaLabels(tracts), 'uint32');
            const execution = await runExample(device, {
              graph,
              tracts: polygons,
              values: columns['values'],
              labels
            });
            return {
              ...execution,
              tags: {
                summary: {
                  role: 'output',
                  keySpace: 'summary',
                  meaning:
                    'Regions, within SS, between SS, total SS, smallest, largest, cross-link fraction, ignored, pseudo-F'
                },
                regionSizes: {role: 'intermediate', keySpace: 'label', meaning: 'Tracts per label'},
                regionWithinSumOfSquares: {
                  role: 'intermediate',
                  keySpace: 'label',
                  meaning: 'Within-region sum of squares per label'
                }
              },
              keySpaces: [
                {id: 'summary', label: 'Partition summary words', size: 9, indexSpace: '1d'},
                {id: 'label', label: 'Community area label', size: 8, indexSpace: '1d'}
              ]
            };
          }
        })
    };
  }
});
