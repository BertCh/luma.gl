// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';

registerLensScenario({
  id: 'gpu-spatial-analysis.location-allocation',
  entry: 'gpu-spatial-analysis',
  element: 'GPULocationAllocation',
  title: 'P-median: five facilities for north-side Chicago tracts',
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
      import('./location-allocation.example'),
      import('./s1-regions-support'),
      import('./spatial-scenario-utils'),
      import('../../../gpu-spatial-analysis/computation-pack/operators/statistics')
    ]);
    const tracts = await support.loadRegionTracts();
    return {
      run: async device =>
        support.assertRegionalizationValid(
          await utilities.runTractStatisticsScenario({
            device,
            id: 'gpu-spatial-analysis.location-allocation',
            tracts,
            entry: STATISTICS_OPERATORS['GPULocationAllocation'],
            columns: {demandWeights: 'population'},
            choices: {
              costMetric: 'straight-line-km',
              operation: 'p-median',
              maximumFacilityCount: 5,
              maximumSwapIterations: 8
            },
            async execute({graph, columns, upload}) {
              // Prepared on the CPU as the caller-supplied cost matrix; the figure does not draw it.
              const costs = upload('costs', support.getCentreCostMatrix(tracts), 'float32');
              const execution = await runExample(device, {
                graph,
                costs,
                demandWeights: columns['demandWeights']
              });
              return {
                ...execution,
                tags: {
                  demandWeights: {
                    role: 'input',
                    polygons: true,
                    meaning: 'Residents per tract: the demand weight'
                  },
                  assignments: {
                    role: 'output',
                    polygons: true,
                    meaning:
                      'Selected facility serving each tract (nominal; label is its tract row)'
                  },
                  selectedFacilities: {
                    role: 'intermediate',
                    meaning: 'One flag per candidate tract: 1 when a facility is placed there'
                  },
                  assignedCosts: {
                    role: 'intermediate',
                    meaning: 'Straight-line kilometres from each tract to its facility'
                  },
                  objective: {
                    role: 'intermediate',
                    keySpace: 'status',
                    meaning: 'Population-weighted kilometres (p-median objective)'
                  },
                  status: {
                    role: 'intermediate',
                    keySpace: 'status',
                    meaning:
                      'Selected, uncovered demand, complete, invalid count, swaps, converged, limit reached'
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
