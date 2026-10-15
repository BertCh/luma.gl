// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';

registerLensScenario({
  id: 'gpu-spatial-analysis.zonal-statistics',
  entry: 'gpu-spatial-analysis',
  element: 'GPUZonalStatistics',
  title: 'Zonal statistics: poverty counts by community area',
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
      {STATISTICS_OPERATORS},
      {createSpatialChoice},
      gpgpu
    ] = await Promise.all([
      import('./zonal-statistics.example'),
      import('./spatial-lens-data'),
      import('./spatial-scenario-utils'),
      import('../../../gpu-spatial-analysis/computation-pack/operators/statistics'),
      import('../../../gpu-spatial-analysis/computation-pack/choices'),
      import('@luma.gl/gpgpu')
    ]);
    const tracts = await loadChicagoTracts();
    // Dense zone rows in ascending community-area order: the membership is the tract's own label.
    const areas = [...new Set(tracts.columns['communityArea'])].sort((a, b) => a - b);
    const rowOfArea = new Map(areas.map((area, row) => [area, row]));
    const zoneRows = Uint32Array.from(
      tracts.columns['communityArea'],
      area => rowOfArea.get(area)!
    );
    return {
      run: async device => {
        const record = await utilities.runTractStatisticsScenario({
          device,
          id: 'gpu-spatial-analysis.zonal-statistics',
          tracts,
          entry: STATISTICS_OPERATORS['GPUZonalStatistics'],
          columns: {values: 'poverty150'},
          choices: {sumOrder: 'sorted', membership: 'feature-rows'},
          metadata: {
            // The mapped quantity is a count; no denominator is applied to the zone sums.
            mappedVariableExtent: 'extensive',
            valueColumn: 'poverty150'
          },
          async execute({graph, columns, upload}) {
            const pointFeatureRows = upload('pointFeatureRows', zoneRows, 'uint32');
            const execution = await runExample(device, {
              graph,
              pointFeatureRows,
              values: columns['values'],
              zoneCount: areas.length
            });
            return {
              ...execution,
              keySpaces: [
                {
                  id: 'zone',
                  label: 'Chicago community area',
                  size: areas.length,
                  indexSpace: '1d' as const
                }
              ],
              tags: {
                pointFeatureRows: {
                  role: 'input',
                  polygons: true,
                  meaning: 'Community-area row of each tract (nominal)'
                },
                values: {role: 'input', meaning: 'People below 150% of the poverty line (count)'},
                counts: {role: 'intermediate', keySpace: 'zone', meaning: 'Tracts per zone'},
                valueCounts: {
                  role: 'intermediate',
                  keySpace: 'zone',
                  meaning: 'Tracts with a finite value per zone'
                },
                sums: {
                  role: 'output',
                  keySpace: 'zone',
                  meaning: 'People below 150% of the poverty line per community area'
                },
                means: {
                  role: 'intermediate',
                  keySpace: 'zone',
                  meaning: 'Mean tract count per community area'
                },
                standardDeviations: {
                  role: 'intermediate',
                  keySpace: 'zone',
                  meaning: 'Sample deviation of the tract counts per community area'
                }
              }
            };
          }
        });
        // Conservation (M12): both totals come from the GPU readbacks of this run.
        const snapshots = record.captures!.snapshots;
        const total = (name: string) =>
          Array.from(snapshots.find(capture => capture.name === name)!.values).reduce(
            (sum, value) => (Number.isFinite(value) ? sum + value : sum),
            0
          );
        const result = {
          ...record,
          choices: [
            ...(record.choices ?? []),
            createSpatialChoice(
              {name: 'sumBefore', stage: 'operator', sweepable: false},
              total('values')
            ),
            createSpatialChoice(
              {name: 'sumAfter', stage: 'operator', sweepable: false},
              total('sums')
            )
          ]
        };
        if (!gpgpu.validateGPUComputationRecord(result).valid) {
          throw new Error('Zonal conservation choices made the record invalid');
        }
        return result;
      }
    };
  }
});
