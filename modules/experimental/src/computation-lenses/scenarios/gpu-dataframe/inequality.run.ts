// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  GPU_INEQUALITY_DEFAULT_PALMA_BOTTOM_SHARE,
  GPU_INEQUALITY_DEFAULT_PALMA_TOP_SHARE
} from '@luma.gl/experimental/gpu-dataframe';
import {runExample} from './inequality.example';
import {
  createDataFrameChoice,
  createDataFrameLensGraph,
  loadDataFrameLensData,
  runDataFrameContributorLens
} from './dataframe-lens-support';

const SCENARIO_ID = 'gpu-dataframe.inequality';
/** Transient outputs the capture reads; they must be retained because transients can alias. */
const RETAINED = ['gini', 'counts', 'mean', 'lorenzKnots', 'globalSummary'];

/** Executes the recorded inequality indices once and reads inputs and outputs from GPU buffers. */
export async function run(device: Device, options?: {signal?: AbortSignal}) {
  options?.signal?.throwIfAborted();
  const dataset = await loadDataFrameLensData('f-chicago-tracts');
  const {perCapitaIncome, population, communityArea} = dataset.columns;
  const income = Float32Array.from(perCapitaIncome, value => value ?? NaN);
  // Zone ids are zero based, so community area 1 is zone 0.
  const zoneIds = Uint32Array.from(communityArea, area => area - 1);
  const zoneCount = Math.max(...communityArea);
  const graph = createDataFrameLensGraph<undefined>(device, 'gpu-dataframe-inequality', RETAINED);
  const example = await runExample(device, {
    graph,
    tracts: {perCapitaIncome: income, population: Float32Array.from(population), zoneIds},
    zoneCount
  });
  try {
    options?.signal?.throwIfAborted();
    return await runDataFrameContributorLens(device, {
      scenarioId: SCENARIO_ID,
      element: 'GPUInequality',
      graph,
      specs: {
        values: {
          view: example.values,
          role: 'input',
          keySpace: 'row',
          meaning: 'Per-capita income in dollars; suppressed estimates are NaN and excluded.'
        },
        weights: {
          view: example.weights,
          role: 'input',
          keySpace: 'row',
          meaning: 'Residents of the tract; tracts with zero population are excluded.'
        },
        zoneIds: {
          view: example.zoneIds,
          role: 'input',
          keySpace: 'row',
          meaning: 'Community area minus one.'
        },
        gini: {
          view: example.gini,
          role: 'output',
          keySpace: 'zone',
          meaning: 'Population-weighted Gini of per-capita income; NaN for an empty zone.'
        },
        counts: {view: example.counts, role: 'output', keySpace: 'zone'},
        mean: {view: example.mean, role: 'output', keySpace: 'zone'},
        lorenzKnots: {
          view: example.lorenzKnots,
          role: 'output',
          keySpace: 'knot',
          meaning: `Lorenz curve L(k / ${example.lorenzKnotCount - 1}) at index zone * ${example.lorenzKnotCount} + k.`
        },
        globalSummary: {
          view: example.globalSummary,
          role: 'output',
          keySpace: 'summary',
          meaning:
            'Pooled Theil T, its between and within terms, pooled Gini, count, mean, weight and income.'
        }
      },
      keySpaces: [
        {id: 'row', label: 'Census tract', size: income.length, indexSpace: '1d'},
        {id: 'zone', label: 'Community area', size: zoneCount, indexSpace: '1d'},
        {
          id: 'knot',
          label: 'Lorenz knot of a zone',
          size: zoneCount * example.lorenzKnotCount,
          indexSpace: '1d'
        },
        {id: 'summary', label: 'Pooled summary', size: 8, indexSpace: '1d'}
      ],
      datumKeySpace: 'zone',
      choices: [
        createDataFrameChoice('group', 'zoneCount', zoneCount),
        createDataFrameChoice('aggregate', 'lorenzKnotCount', example.lorenzKnotCount, {
          sweepable: true
        }),
        createDataFrameChoice('aggregate', 'weights', 'population'),
        createDataFrameChoice('aggregate', 'epsilon', 1),
        createDataFrameChoice('aggregate', 'palmaTopShare', GPU_INEQUALITY_DEFAULT_PALMA_TOP_SHARE),
        createDataFrameChoice(
          'aggregate',
          'palmaBottomShare',
          GPU_INEQUALITY_DEFAULT_PALMA_BOTTOM_SHARE
        )
      ],
      datasetVersion: `${dataset.id}:${dataset.rowCount}`
    });
  } finally {
    example.parameters.destroy();
    example.buffers.forEach(buffer => buffer.destroy());
  }
}
