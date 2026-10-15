// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';

registerLensScenario({
  id: 'gpu-spatial-analysis.spatial-weights-minimum-spanning-tree',
  entry: 'gpu-spatial-analysis',
  element: 'GPUSpatialWeightsMinimumSpanningTree',
  title: 'Minimum spanning tree of north-side Chicago tracts by poverty',
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
      import('./spatial-weights-minimum-spanning-tree.example'),
      import('./s1-regions-support'),
      import('./spatial-scenario-utils'),
      import('../../../gpu-spatial-analysis/computation-pack/operators/statistics')
    ]);
    const tracts = await support.loadRegionTracts();
    return {
      run: async device => {
        const record = await utilities.runTractStatisticsScenario({
          device,
          id: 'gpu-spatial-analysis.spatial-weights-minimum-spanning-tree',
          tracts,
          entry: STATISTICS_OPERATORS['GPUSpatialWeightsMinimumSpanningTree'],
          columns: {values: 'poverty150Pct'},
          choices: {criterion: 'queen', standardize: true},
          async execute({graph, polygons, columns, upload}) {
            const execution = await runExample(device, {
              graph,
              tracts: polygons,
              values: columns['values']
            });
            // One path of two vertices per tree edge: offsets 0, 2, 4, ... are structure, not a result.
            const treeSegmentOffsets = upload(
              'treeSegmentOffsets',
              Uint32Array.from({length: tracts.count}, (_, index) => index * 2),
              'uint32'
            );
            const edgeCount = tracts.count - 1;
            return {
              ...execution,
              views: {...execution.views, treeSegmentOffsets},
              tags: {
                values: {
                  role: 'input',
                  polygons: true,
                  meaning: 'Population below 150% poverty line (%)'
                },
                treeSegments: {
                  role: 'intermediate',
                  keySpace: 'edgeVertex',
                  meaning: 'Tree edge as a segment between the centroids of its two tracts'
                },
                treeSegmentOffsets: {
                  role: 'intermediate',
                  keySpace: 'edgeOffset',
                  meaning: 'Segment start offsets: two vertices per tree edge'
                },
                edgeEndpoints: {
                  role: 'intermediate',
                  keySpace: 'edgeVertex',
                  meaning: 'Tract row and neighbor row of each tree edge'
                },
                edgeDissimilarities: {
                  role: 'intermediate',
                  keySpace: 'edge',
                  meaning: 'Squared difference of standardized poverty across the edge'
                },
                edgeOverflow: {role: 'intermediate', keySpace: 'status'}
              },
              keySpaces: [
                {id: 'edge', label: 'Tree edge', size: edgeCount, indexSpace: '1d'},
                {
                  id: 'edgeVertex',
                  label: 'Tree edge endpoint',
                  size: 2 * edgeCount,
                  indexSpace: '1d'
                },
                {
                  id: 'edgeOffset',
                  label: 'Tree segment offset',
                  size: tracts.count,
                  indexSpace: '1d'
                },
                {id: 'status', label: 'Scalar and status words', indexSpace: '1d'}
              ]
            };
          }
        });
        // The tree draws as segments coloured by edge dissimilarity; the geometry is attached to the
        // captured GPU readback, not recomputed.
        return utilities.finishSpatialWeightsRecord(record, {
          treeSegments: {
            role: 'output',
            geometry: {
              kind: 'lines',
              coordinateModel: 'geographic',
              offsetsCapture: 'treeSegmentOffsets',
              valuesCapture: 'edgeDissimilarities'
            }
          }
        });
      }
    };
  }
});
