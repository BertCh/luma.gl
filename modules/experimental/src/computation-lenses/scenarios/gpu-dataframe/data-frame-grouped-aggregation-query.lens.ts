// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';

const DATASET = {
  id: 'f-chicago-nature',
  source: 'iNaturalist observations in Chicago, 2023 (showcase chicago-nature v1)',
  licence:
    'iNaturalist contributors; per-observation CC0 / CC BY 4.0 / CC BY-NC 4.0; non-commercial'
} as const;

registerLensScenario({
  id: 'gpu-dataframe.data-frame-grouped-aggregation-query',
  entry: 'gpu-dataframe',
  element: 'GPUDataFrameGroupedAggregationQuery',
  title: 'Chicago nature observations grouped by category',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: DATASET,
  load: async () => import('./data-frame-grouped-aggregation-query.run')
});

// `groupBy()` returns a GPUDataFrameGroupByQuery that `aggregate()` turns into the compiled
// grouped query, so one example and one GPU run cover both documented classes.
registerLensScenario({
  id: 'gpu-dataframe.data-frame-group-by-query',
  entry: 'gpu-dataframe',
  element: 'GPUDataFrameGroupByQuery',
  title: 'Chicago nature observations grouped by category',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
  defaultRung: 'result',
  example: {path: 'scenarios/gpu-dataframe/data-frame-grouped-aggregation-query.example.ts'},
  dataset: DATASET,
  load: async () => {
    const module = await import('./data-frame-grouped-aggregation-query.run');
    return {run: module.runGroupBy};
  }
});
