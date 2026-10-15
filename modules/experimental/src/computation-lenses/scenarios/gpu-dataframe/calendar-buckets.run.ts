// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {GPU_CALENDAR_BUCKETS_MATRIX_LENGTH} from '@luma.gl/experimental/gpu-dataframe';
import {runExample} from './calendar-buckets.example';
import {
  createDataFrameChoice,
  createDataFrameLensGraph,
  loadDataFrameLensData,
  runDataFrameContributorLens
} from './dataframe-lens-support';

const SCENARIO_ID = 'gpu-dataframe.calendar-buckets';
/** Transient outputs the capture reads; they must be retained because transients can alias. */
const RETAINED = ['month', 'weekday', 'hour', 'hourWeekdayCounts'];
/** 2023-01-01T00:00 in epoch milliseconds, the origin of the dataset's `timestamp` seconds. */
const EPOCH_2023_MILLISECONDS = Date.UTC(2023, 0, 1);

/** Executes the recorded calendar decode once and reads inputs and outputs from GPU buffers. */
export async function run(device: Device, options?: {signal?: AbortSignal}) {
  options?.signal?.throwIfAborted();
  const dataset = await loadDataFrameLensData('f-chicago-nature');
  // The dataset stores Chicago local clock seconds as if they were UTC, so the offset is zero.
  const epochMilliseconds = BigInt64Array.from(dataset.columns.timestamp, seconds =>
    BigInt(EPOCH_2023_MILLISECONDS + seconds * 1000)
  );
  const graph = createDataFrameLensGraph<undefined>(
    device,
    'gpu-dataframe-calendar-buckets',
    RETAINED
  );
  const example = await runExample(device, {graph, observations: {epochMilliseconds}});
  try {
    options?.signal?.throwIfAborted();
    const rowCount = dataset.rowCount;
    return await runDataFrameContributorLens(device, {
      scenarioId: SCENARIO_ID,
      element: 'GPUCalendarBuckets',
      graph,
      specs: {
        timestampWords: {
          view: example.timestamps,
          role: 'input',
          keySpace: 'row',
          meaning: 'Epoch milliseconds as signed 64-bit (low, high) words of the local clock time.'
        },
        month: {
          view: example.month,
          role: 'output',
          keySpace: 'row',
          meaning: 'Month of the local time, 1 to 12.'
        },
        weekday: {
          view: example.weekday,
          role: 'output',
          keySpace: 'row',
          meaning: 'Weekday of the local time, 0 is Monday.'
        },
        hour: {
          view: example.hour,
          role: 'output',
          keySpace: 'row',
          meaning: 'Hour of the local time, 0 to 23.'
        },
        weekdayHourCounts: {
          view: example.hourWeekdayCounts,
          role: 'output',
          keySpace: 'weekdayHour',
          meaning: 'Observations per bucket, index weekday * 24 + hour.'
        }
      },
      keySpaces: [
        {id: 'row', label: 'Observation', size: rowCount, indexSpace: '1d'},
        {id: 'invocation', label: 'Compute invocation', size: rowCount, indexSpace: '1d'},
        {
          id: 'weekdayHour',
          label: 'Weekday and hour bucket',
          size: GPU_CALENDAR_BUCKETS_MATRIX_LENGTH,
          indexSpace: '1d'
        }
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
      datumKeySpace: 'row',
      choices: [
        createDataFrameChoice('derive', 'utcOffsetMinutes', 0, {unit: 'min'}),
        createDataFrameChoice('derive', 'firstDayOfWeek', 0, {
          alternatives: [0, 1, 2, 3, 4, 5, 6]
        })
      ],
      datasetVersion: `${dataset.id}:${dataset.rowCount}`
    });
  } finally {
    example.destroy();
  }
}
