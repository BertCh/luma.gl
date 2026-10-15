// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';

/** Community area whose tracts the supplied pick result lists. */
const PICKED_COMMUNITY_AREA = 8;

registerLensScenario({
  id: 'gpu-spatial-analysis.pick-region-mask',
  entry: 'gpu-spatial-analysis',
  element: 'GPUPickRegionMask',
  title: 'Pick region mask: tracts of one community area',
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
    const [{runExample}, {loadChicagoTracts}, utilities, {STATISTICS_OPERATORS}] =
      await Promise.all([
        import('./pick-region-mask.example'),
        import('./spatial-lens-data'),
        import('./spatial-scenario-utils'),
        import('../../../gpu-spatial-analysis/computation-pack/operators/statistics')
      ]);
    const tracts = await loadChicagoTracts();
    // A pick result as GPUIndexPickingTarget.addRegionPass writes it: the tracts of one community
    // area in batch 0, three of them picked twice, one pick of another tract in batch 1 (filtered
    // out) and spare capacity. This is an input to the element, not an output of it.
    const members: number[] = [];
    tracts.columns['communityArea'].forEach((area, row) => {
      if (area === PICKED_COMMUNITY_AREA) members.push(row);
    });
    const outsider = tracts.columns['communityArea'].findIndex(
      area => area !== PICKED_COMMUNITY_AREA
    );
    const pairs = [
      ...members.map(row => [row, 0]),
      ...members.slice(0, 3).map(row => [row, 0]),
      [outsider, 1]
    ];
    const capacity = pairs.length + 4;
    const picks = new Uint32Array(2 + 2 * capacity);
    picks[0] = pairs.length;
    pairs.forEach(([row, batch], index) => {
      picks[2 + 2 * index] = row;
      picks[3 + 2 * index] = batch;
    });
    return {
      run: async device => {
        const extra: {destroy(): void}[] = [];
        try {
          return await utilities.runTractStatisticsScenario({
            device,
            id: 'gpu-spatial-analysis.pick-region-mask',
            tracts,
            entry: STATISTICS_OPERATORS['GPUPickRegionMask'],
            columns: {},
            choices: {batchIndex: 0},
            metadata: {pickedCommunityArea: PICKED_COMMUNITY_AREA, pickPairs: pairs.length},
            async execute({graph}) {
              const pickResult = utilities.uploadSpatialView(graph, 'picks', picks, 'uint32');
              extra.push(pickResult);
              const execution = await runExample(device, {
                graph,
                picks: pickResult.view,
                rowCount: tracts.count
              });
              return {
                ...execution,
                keySpaces: [
                  {
                    id: 'picks',
                    label: 'Pick result words: count, overflow, then object and batch pairs',
                    size: picks.length,
                    indexSpace: '1d' as const
                  },
                  {id: 'status', label: 'Overflow flag', size: 1, indexSpace: '1d' as const}
                ],
                tags: {
                  picks: {
                    role: 'input',
                    keySpace: 'picks',
                    meaning: `Supplied region-pick result: ${members.length} tracts of community area ${PICKED_COMMUNITY_AREA}, duplicates and one other-batch pick`
                  },
                  mask: {
                    role: 'output',
                    polygons: true,
                    meaning: 'Picked tracts of batch 0 as a 0/1 mask'
                  },
                  overflow: {
                    role: 'intermediate',
                    keySpace: 'status',
                    meaning: 'The pick result’s own overflow word'
                  }
                }
              };
            }
          });
        } finally {
          extra.forEach(resource => resource.destroy());
        }
      }
    };
  }
});
