// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {runExample} from './temporal-reduction.example';
import {
  createDataFrameChoice,
  createDataFrameLensGraph,
  loadDataFrameLensData,
  runDataFrameContributorLens
} from './dataframe-lens-support';

const SCENARIO_ID = 'gpu-dataframe.temporal-reduction';
/** Transient outputs the capture reads; they must be retained because transients can alias. */
const RETAINED = [
  'counts',
  'minimums',
  'maximums',
  'first',
  'last',
  'occupiedIds',
  'occupiedCount',
  'occupiedOverflow'
];

/** Executes the recorded temporal reduction once and reads inputs and outputs from GPU buffers. */
export async function run(device: Device, options?: {signal?: AbortSignal}) {
  options?.signal?.throwIfAborted();
  const dataset = await loadDataFrameLensData('f-chicago-nature');
  const columns = dataset.columns;
  const graph = createDataFrameLensGraph<undefined>(
    device,
    'gpu-dataframe-temporal-reduction',
    RETAINED
  );
  const example = await runExample(device, {
    graph,
    observations: {
      timestamp: Float32Array.from(columns.timestamp),
      category: Uint32Array.from(columns.category),
      researchGrade: Float32Array.from(columns.researchGrade)
    }
  });
  try {
    options?.signal?.throwIfAborted();
    const rowCount = dataset.rowCount;
    return await runDataFrameContributorLens(device, {
      scenarioId: SCENARIO_ID,
      element: 'GPUTemporalReduction',
      graph,
      specs: {
        timestamp: {
          view: example.timestamps,
          role: 'input',
          keySpace: 'row',
          meaning: 'Seconds since 2023-01-01 on the Chicago local clock, as float32.'
        },
        category: {
          view: example.cellIds,
          role: 'input',
          keySpace: 'row',
          meaning: 'Taxon group, used as the cell id.'
        },
        researchGrade: {view: example.values, role: 'input', keySpace: 'row'},
        counts: {
          view: example.counts,
          role: 'output',
          keySpace: 'slot',
          meaning: `Observations per slot, slot = category * ${example.bucket.bucketCount} + week.`
        },
        minimums: {view: example.minimums, role: 'output', keySpace: 'slot'},
        maximums: {view: example.maximums, role: 'output', keySpace: 'slot'},
        first: {
          view: example.first,
          role: 'output',
          keySpace: 'slot',
          meaning: 'researchGrade at the earliest timestamp of the slot; NaN for an empty slot.'
        },
        last: {
          view: example.last,
          role: 'output',
          keySpace: 'slot',
          meaning: 'researchGrade at the latest timestamp of the slot; NaN for an empty slot.'
        },
        occupiedCount: {view: example.occupiedCount, role: 'output', keySpace: 'summary'},
        occupiedIds: {
          view: example.occupiedIds,
          role: 'output',
          keySpace: 'compactSlot',
          meaning:
            'Only the first occupiedCount entries are slot ids; remaining entries are unused.'
        }
      },
      keySpaces: [
        {id: 'row', label: 'Observation', size: rowCount, indexSpace: '1d'},
        {id: 'invocation', label: 'Compute invocation', size: rowCount, indexSpace: '1d'},
        {id: 'slot', label: 'Taxon group and week slot', size: example.slotCount, indexSpace: '1d'},
        {id: 'summary', label: 'Whole-column summary', size: 1, indexSpace: '1d'},
        {id: 'compactSlot', label: 'Compacted slot id', size: example.slotCount, indexSpace: '1d'}
      ],
      relations: [
        {
          id: 'invocation-row',
          from: 'invocation',
          to: 'row',
          cardinality: '1:1',
          kind: 'function',
          ref: {function: 'identity'},
          timeDependent: false
        }
      ],
      datumKeySpace: 'slot',
      choices: [
        createDataFrameChoice('group', 'origin', example.bucket.origin, {unit: 's'}),
        createDataFrameChoice('group', 'width', example.bucket.width, {
          unit: 's',
          sweepable: true,
          alternatives: [86_400, 7 * 86_400, 30 * 86_400]
        }),
        createDataFrameChoice('group', 'cellCount', example.bucket.cellCount),
        createDataFrameChoice('group', 'bucketCount', example.bucket.bucketCount),
        createDataFrameChoice('group', 'sorted', false, {alternatives: [false, true]})
      ],
      datasetVersion: `${dataset.id}:${dataset.rowCount}`
    });
  } finally {
    example.destroy();
  }
}
