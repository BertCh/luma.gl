// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {runExample} from './group-statistics.example';
import {
  createDataFrameChoice,
  createDataFrameLensGraph,
  loadDataFrameLensData,
  runDataFrameContributorLens
} from './dataframe-lens-support';

const SCENARIO_ID = 'gpu-dataframe.group-statistics';
/** Transient outputs the capture reads; they must be retained because transients can alias. */
const RETAINED = ['groupKeys', 'groupRows', 'counts', 'means', 'variances', 'medians'];

/** Executes the recorded grouped statistics once and reads inputs and outputs from GPU buffers. */
export async function run(device: Device, options?: {signal?: AbortSignal}) {
  options?.signal?.throwIfAborted();
  const dataset = await loadDataFrameLensData('f-chicago-tracts');
  const communityArea = Uint32Array.from(dataset.columns.communityArea);
  const poverty = Float32Array.from(dataset.columns.poverty150Pct, value => value ?? NaN);
  const groupCount = new Set(communityArea).size;
  const graph = createDataFrameLensGraph<undefined>(
    device,
    'gpu-dataframe-group-statistics',
    RETAINED
  );
  const example = await runExample(device, {
    graph,
    tracts: {communityArea, poverty150Pct: poverty},
    groupCount
  });
  try {
    options?.signal?.throwIfAborted();
    return await runDataFrameContributorLens(device, {
      scenarioId: SCENARIO_ID,
      element: 'GPUGroupStatistics',
      graph,
      specs: {
        communityArea: {
          view: example.keys,
          role: 'input',
          keySpace: 'row',
          meaning: 'Community area (1 to 77) of each tract; the group key.'
        },
        poverty150Pct: {
          view: example.poverty,
          role: 'input',
          keySpace: 'row',
          meaning:
            'Share of residents below 150% of the poverty line; suppressed estimates are NaN.'
        },
        groupKeys: {
          view: example.groupKeys,
          role: 'output',
          keySpace: 'group',
          meaning: 'Distinct community areas in ascending order.'
        },
        groupRows: {
          view: example.groupRows,
          role: 'output',
          keySpace: 'group',
          meaning: 'Tracts per community area, including tracts with a suppressed estimate.'
        },
        counts: {
          view: example.counts,
          role: 'output',
          keySpace: 'group',
          meaning: 'Tracts with a finite poverty estimate per community area.'
        },
        means: {view: example.means, role: 'output', keySpace: 'group'},
        variances: {
          view: example.variances,
          role: 'output',
          keySpace: 'group',
          meaning: 'Sample variance (n - 1); NaN for a group with a single finite value.'
        },
        medians: {view: example.medians, role: 'output', keySpace: 'group'}
      },
      keySpaces: [
        {id: 'row', label: 'Census tract', size: poverty.length, indexSpace: '1d'},
        {id: 'group', label: 'Community area', size: groupCount, indexSpace: '1d'}
      ],
      datumKeySpace: 'group',
      choices: [
        createDataFrameChoice('aggregate', 'statistics', 'count, mean, variance, median'),
        createDataFrameChoice('aggregate', 'variance', 'sample', {
          alternatives: ['sample', 'population']
        }),
        createDataFrameChoice('aggregate', 'sumScale', 65536),
        createDataFrameChoice('group', 'groupCount', groupCount)
      ],
      datasetVersion: `${dataset.id}:${dataset.rowCount}`
    });
  } finally {
    example.buffers.forEach(buffer => buffer.destroy());
  }
}
