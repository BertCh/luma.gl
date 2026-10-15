// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GPUComputationCapture} from '@luma.gl/gpgpu';
import {
  GPU_COLUMN_PROFILE_STATISTIC,
  GPU_COLUMN_PROFILE_STATISTIC_COUNT
} from '@luma.gl/experimental/gpu-dataframe';
import {runExample} from './column-profile.example';
import {
  createDataFrameChoice,
  createDataFrameLensGraph,
  loadDataFrameLensData,
  runDataFrameContributorLens
} from './dataframe-lens-support';

const SCENARIO_ID = 'gpu-dataframe.column-profile';
/** Transient outputs the capture reads; they must be retained because transients can alias. */
const RETAINED = ['statistics', 'histograms'];
const COLUMN_NAMES = [
  'poverty150Pct',
  'uninsuredPct',
  'age65Pct',
  'medianHouseholdIncome'
] as const;
/** Statistic fields shown as columns of the profile table, in the order of the kernel layout. */
const SHOWN_STATISTICS = [
  'count',
  'nullCount',
  'minimum',
  'maximum',
  'sum',
  'mean',
  'variance',
  'sampleVariance',
  'standardDeviation',
  'distinctEstimate'
] as const;

/**
 * Splits the flat column-major `statistics` readback into one capture per statistic. This only
 * reshapes values the GPU wrote (`statistics[column * 11 + field]`); nothing is recomputed.
 */
function splitStatistics(captures: readonly GPUComputationCapture[]): GPUComputationCapture[] {
  const statistics = captures.find(capture => capture.name === 'statistics');
  // The scenario always captures the statistics view it just built.
  if (!statistics) throw new Error('column profile statistics were not captured');
  const split = SHOWN_STATISTICS.map(field => ({
    ...statistics,
    id: `${statistics.id}-${field}`,
    name: field,
    keySpace: 'column',
    role: 'output' as const,
    values: Array.from(
      {length: COLUMN_NAMES.length},
      (_, column) =>
        statistics.values[
          column * GPU_COLUMN_PROFILE_STATISTIC_COUNT + GPU_COLUMN_PROFILE_STATISTIC[field]
        ]
    ),
    meaning: `column statistic ${field}, field ${GPU_COLUMN_PROFILE_STATISTIC[field]} of each 11-float block of statistics`
  }));
  return [...captures.filter(capture => capture.name !== 'statistics'), ...split];
}

/** Executes the recorded column profile once and reads inputs and outputs from GPU buffers. */
export async function run(device: Device, options?: {signal?: AbortSignal}) {
  options?.signal?.throwIfAborted();
  const dataset = await loadDataFrameLensData('f-chicago-tracts');
  const toFloat32 = (values: readonly (number | null)[]) =>
    Float32Array.from(values, value => value ?? NaN);
  const graph = createDataFrameLensGraph<undefined>(
    device,
    'gpu-dataframe-column-profile',
    RETAINED
  );
  const example = await runExample(device, {
    graph,
    tracts: {
      poverty150Pct: toFloat32(dataset.columns.poverty150Pct),
      uninsuredPct: toFloat32(dataset.columns.uninsuredPct),
      age65Pct: toFloat32(dataset.columns.age65Pct),
      medianHouseholdIncome: toFloat32(dataset.columns.medianHouseholdIncome)
    }
  });
  try {
    options?.signal?.throwIfAborted();
    const rowCount = dataset.rowCount;
    return await runDataFrameContributorLens(device, {
      scenarioId: SCENARIO_ID,
      element: 'GPUColumnProfile',
      graph,
      specs: {
        ...Object.fromEntries(
          example.columns.map(({name, values}) => [
            name,
            {
              view: values,
              role: 'input' as const,
              keySpace: 'row',
              meaning: `Census column ${name}; suppressed estimates are NaN (null).`
            }
          ])
        ),
        statistics: {view: example.statistics, role: 'intermediate', keySpace: 'statistic'},
        histograms: {
          view: example.histograms,
          role: 'output',
          keySpace: 'bin',
          meaning: `column-major equal-width histogram: column * ${example.histogramBinCount} + bin`
        }
      },
      keySpaces: [
        {id: 'row', label: 'Census tract', size: rowCount, indexSpace: '1d'},
        {id: 'column', label: 'Profiled column', size: COLUMN_NAMES.length, indexSpace: '1d'},
        {
          id: 'statistic',
          label: 'Statistic of a column',
          size: COLUMN_NAMES.length * GPU_COLUMN_PROFILE_STATISTIC_COUNT,
          indexSpace: '1d'
        },
        {
          id: 'bin',
          label: 'Histogram bin of a column',
          size: COLUMN_NAMES.length * example.histogramBinCount,
          indexSpace: '1d'
        }
      ],
      datumKeySpace: 'column',
      choices: [
        createDataFrameChoice('summarise', 'kind', 'numeric', {
          alternatives: ['numeric', 'category']
        }),
        createDataFrameChoice('aggregate', 'histogramBinCount', example.histogramBinCount, {
          sweepable: true
        }),
        createDataFrameChoice('aggregate', 'hyperLogLogPrecision', 12)
      ],
      datasetVersion: `${dataset.id}:${dataset.rowCount}`,
      deriveCaptures: splitStatistics
    });
  } finally {
    example.destroy();
  }
}
