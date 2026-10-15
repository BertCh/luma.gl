// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GPUComputationCapture} from '@luma.gl/gpgpu';
import {
  GPU_COMPOSITE_SCORE_AGGREGATION,
  GPU_COMPOSITE_SCORE_COLUMN_STATISTICS_STRIDE,
  GPU_COMPOSITE_SCORE_SCALER
} from '@luma.gl/experimental/gpu-dataframe';
import {runExample} from './composite-score.example';
import {
  createDataFrameChoice,
  createDataFrameLensGraph,
  loadDataFrameLensData,
  runDataFrameContributorLens
} from './dataframe-lens-support';

const SCENARIO_ID = 'gpu-dataframe.composite-score';
/** Transient outputs the capture reads; they must be retained because transients can alias. */
const RETAINED = ['score', 'columnStatistics'];
const INDICATOR_NAMES = ['poverty150Pct', 'unemployedPct', 'uninsuredPct', 'noVehiclePct'] as const;
const STATISTIC_NAMES = ['columnMinimum', 'columnMaximum', 'columnMean', 'columnStandardDeviation'];

/**
 * Splits the row-major packed `indicators` readback into one input column per indicator, and the
 * flat `columnStatistics` readback into one column per statistic. Both only reshape values the
 * GPU holds (`indicators[row * 4 + column]`, `columnStatistics[column * 4 + field]`).
 */
function splitPackedViews(captures: readonly GPUComputationCapture[]): GPUComputationCapture[] {
  const indicators = captures.find(capture => capture.name === 'indicators');
  const statistics = captures.find(capture => capture.name === 'columnStatistics');
  // The scenario always captures both views it just built.
  if (!indicators || !statistics) throw new Error('composite score captures are missing');
  const count = INDICATOR_NAMES.length;
  const rowCount = indicators.values.length / count;
  const columns = INDICATOR_NAMES.map((name, column) => ({
    ...indicators,
    id: `${indicators.id}-${name}`,
    name,
    keySpace: 'row',
    role: 'input' as const,
    values: Array.from({length: rowCount}, (_, row) => indicators.values[row * count + column]),
    meaning: `Tract ${name}; NaN where an estimate is suppressed (column ${column} of the packed indicators).`
  }));
  const fields = STATISTIC_NAMES.map((name, field) => ({
    ...statistics,
    id: `${statistics.id}-${name}`,
    name,
    keySpace: 'indicator',
    role: 'output' as const,
    values: Array.from(
      {length: count},
      (_, column) =>
        statistics.values[column * GPU_COMPOSITE_SCORE_COLUMN_STATISTICS_STRIDE + field]
    ),
    meaning: `Per-indicator ${name} over the scored tracts (field ${field} of each block of 4).`
  }));
  return [
    ...captures.filter(capture => capture !== indicators && capture !== statistics),
    ...columns,
    ...fields
  ];
}

/** Executes the recorded composite score once and reads inputs and outputs from GPU buffers. */
export async function run(device: Device, options?: {signal?: AbortSignal}) {
  options?.signal?.throwIfAborted();
  const dataset = await loadDataFrameLensData('f-chicago-tracts');
  const {poverty150Pct, unemployedPct, uninsuredPct, noVehicle, households} = dataset.columns;
  const rowCount = dataset.rowCount;
  // Count columns would rank big tracts, so the vehicle indicator is a rate.
  const indicators = new Float32Array(rowCount * INDICATOR_NAMES.length);
  for (let row = 0; row < rowCount; row++) {
    indicators.set(
      [
        poverty150Pct[row] ?? NaN,
        unemployedPct[row] ?? NaN,
        uninsuredPct[row] ?? NaN,
        households[row] > 0 ? (100 * noVehicle[row]) / households[row] : NaN
      ],
      row * INDICATOR_NAMES.length
    );
  }
  const graph = createDataFrameLensGraph<undefined>(
    device,
    'gpu-dataframe-composite-score',
    RETAINED
  );
  const example = await runExample(device, {graph, tracts: {indicators}});
  try {
    options?.signal?.throwIfAborted();
    return await runDataFrameContributorLens(device, {
      scenarioId: SCENARIO_ID,
      element: 'GPUCompositeScore',
      graph,
      specs: {
        indicators: {view: example.indicators, role: 'input', keySpace: 'row'},
        score: {
          view: example.score,
          role: 'output',
          keySpace: 'row',
          meaning: 'Mean of the four min-max scaled indicators; NaN for excluded tracts.'
        },
        columnStatistics: {
          view: example.columnStatistics,
          role: 'output',
          keySpace: 'indicator'
        }
      },
      keySpaces: [
        {id: 'row', label: 'Census tract', size: rowCount, indexSpace: '1d'},
        {
          id: 'indicator',
          label: 'Indicator',
          size: INDICATOR_NAMES.length,
          indexSpace: '1d'
        }
      ],
      datumKeySpace: 'row',
      choices: [
        createDataFrameChoice('derive', 'scaler', 'min-max', {
          alternatives: Object.keys(GPU_COMPOSITE_SCORE_SCALER)
        }),
        createDataFrameChoice('aggregate', 'aggregation', 'weighted-sum', {
          alternatives: Object.keys(GPU_COMPOSITE_SCORE_AGGREGATION)
        }),
        createDataFrameChoice('aggregate', 'weights', '1, 1, 1, 1', {sweepable: true}),
        createDataFrameChoice('derive', 'directions', '+1, +1, +1, +1'),
        createDataFrameChoice('aggregate', 'indicatorCount', INDICATOR_NAMES.length)
      ],
      datasetVersion: `${dataset.id}:${dataset.rowCount}`,
      deriveCaptures: splitPackedViews
    });
  } finally {
    example.parameters.destroy();
    example.indicatorBuffer.destroy();
  }
}
