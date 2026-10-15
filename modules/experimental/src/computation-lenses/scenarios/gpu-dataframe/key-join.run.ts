// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GPUComputationCapture} from '@luma.gl/gpgpu';
import {runExample} from './key-join.example';
import {
  createDataFrameChoice,
  createDataFrameLensGraph,
  loadDataFrameLensData,
  runDataFrameContributorLens
} from './dataframe-lens-support';

const SCENARIO_ID = 'gpu-dataframe.key-join';
/** Transient outputs the capture reads; they must be retained because transients can alias. */
const RETAINED = ['joinedPoverty', 'rightRows', 'matchCounts', 'matched'];
const NO_MATCH = 0xffffffff;

/**
 * Adds the right key each left row matched, read through the GPU's own `rightRows`. This only
 * gathers values the GPU wrote (`tractKeys[rightRows[i]]`); NaN marks an unmatched row.
 */
function addMatchedRightKeys(captures: readonly GPUComputationCapture[]): GPUComputationCapture[] {
  const rightRows = captures.find(capture => capture.name === 'rightRows');
  const tractKeys = captures.find(capture => capture.name === 'tractKeys');
  // The scenario always captures both views it just built.
  if (!rightRows || !tractKeys) throw new Error('key join captures are missing');
  const matchedKeys: GPUComputationCapture = {
    ...rightRows,
    id: `${rightRows.id}-right-key`,
    name: 'rightKeys',
    keySpace: 'row',
    role: 'output',
    elementType: 'f32',
    values: Array.from(rightRows.values, row => (row === NO_MATCH ? NaN : tractKeys.values[row])),
    meaning: 'Key of the right row each observation matched (tractKeys[rightRows[i]]); NaN if none.'
  };
  return [...captures, matchedKeys];
}

/** Executes the recorded key join once and reads inputs and outputs from GPU buffers. */
export async function run(device: Device, options?: {signal?: AbortSignal}) {
  options?.signal?.throwIfAborted();
  const [tracts, nature] = await Promise.all([
    loadDataFrameLensData('f-chicago-tracts'),
    loadDataFrameLensData('f-chicago-nature')
  ]);
  const poverty = Float32Array.from(tracts.columns.poverty150Pct, value => value ?? NaN);
  const graph = createDataFrameLensGraph<undefined>(device, 'gpu-dataframe-key-join', RETAINED);
  const example = await runExample(device, {
    graph,
    observations: {tract: Uint32Array.from(nature.columns.tract)},
    tracts: {
      tract: Uint32Array.from(tracts.columns.tract),
      poverty150Pct: poverty,
      hasEstimate: Uint32Array.from(poverty, value => (Number.isFinite(value) ? 1 : 0))
    }
  });
  try {
    options?.signal?.throwIfAborted();
    return await runDataFrameContributorLens(device, {
      scenarioId: SCENARIO_ID,
      element: 'GPUKeyJoin',
      graph,
      specs: {
        leftKeys: {
          view: example.leftKeys,
          role: 'input',
          keySpace: 'row',
          meaning: 'Tract id of each iNaturalist observation.'
        },
        tractKeys: {
          view: example.rightKeys,
          role: 'input',
          keySpace: 'right',
          meaning: 'Tract id of each tract; the right keys are unique.'
        },
        hasEstimate: {
          view: example.rightMask,
          role: 'input',
          keySpace: 'right',
          meaning: 'Right mask: 1 when the tract has a poverty estimate.'
        },
        poverty150Pct: {
          view: example.rightPoverty,
          role: 'input',
          keySpace: 'right',
          meaning: 'Share of residents below 150% of the poverty line.'
        },
        rightRows: {
          view: example.rightRows,
          role: 'output',
          keySpace: 'row',
          meaning: 'Smallest matching right row; 0xffffffff when unmatched.'
        },
        matchCounts: {view: example.matchCounts, role: 'output', keySpace: 'row'},
        matched: {view: example.matched, role: 'output', keySpace: 'row'},
        joinedPoverty: {
          view: example.joinedPoverty,
          role: 'output',
          keySpace: 'row',
          meaning: 'Gathered poverty of the matching tract; NaN when unmatched.'
        }
      },
      keySpaces: [
        {id: 'row', label: 'Observation', size: nature.rowCount, indexSpace: '1d'},
        {id: 'right', label: 'Census tract', size: poverty.length, indexSpace: '1d'}
      ],
      datumKeySpace: 'row',
      choices: [
        createDataFrameChoice('join', 'kind', 'left', {alternatives: ['left', 'inner']}),
        createDataFrameChoice('join', 'gather', 'poverty150Pct'),
        createDataFrameChoice('filter', 'rightMask', 'tracts with a poverty estimate')
      ],
      datasetVersion: `${tracts.id}+${nature.id}:${tracts.rowCount}:${nature.rowCount}`,
      deriveCaptures: addMatchedRightKeys
    });
  } finally {
    example.buffers.forEach(buffer => buffer.destroy());
  }
}
