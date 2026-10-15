// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';
import type {
  LensRecord,
  ViewContext,
  ViewSatisfaction
} from '../../../computation-views/registry/view-registration';
import {
  getAllCaptures,
  getCaptureValue,
  getDatumKeySpace
} from '../../../computation-views/views/record-queries';
import {getSpatialGuardMetadata} from '../guards';

/** Finds a named contributor output in the datum key space, or a scalar/CSR companion capture. */
export function findSpatialCapture(context: ViewContext, name: string) {
  const captures = getAllCaptures(context.record).filter(capture => capture.name === name);
  return captures.find(capture => capture.keySpace === getDatumKeySpace(context)) ?? captures[0];
}

/** Requires named captures; CSR edge arrays may be empty for an all-island dataset. */
export function requireSpatialCaptures(
  context: ViewContext,
  names: readonly string[]
): ViewSatisfaction {
  for (const name of names) {
    const capture = findSpatialCapture(context, name);
    if (
      !capture ||
      (!capture.values.length && name !== 'weightsNeighbors' && name !== 'weightsValues')
    )
      return {ok: false, reason: `missing or empty ${name} capture`};
  }
  return {ok: true};
}

/** Reads scalar components of a required capture after satisfies has succeeded. */
export function getSpatialValues(context: ViewContext, name: string): number[] {
  const capture = findSpatialCapture(context, name);
  // Required captures were checked by satisfies.
  assert(capture);
  return Array.from(
    {length: Math.floor(capture.values.length / (capture.components ?? 1))},
    (_, key) => getCaptureValue(capture, key)!
  );
}

/** Reads scalar choices without inventing defaults for missing evidence. */
export function getSpatialChoices(record: LensRecord) {
  return getSpatialGuardMetadata(record);
}

/** Significance level recorded for this run, never an implicit 0.05. */
export function getSpatialSignificanceLevel(context: ViewContext): number | undefined {
  const level = getSpatialChoices(context.record)['significanceLevel'];
  return typeof level === 'number' && level > 0 && level < 1 ? level : undefined;
}

/** Reads the pinned row's CSR star; an unpinned view has no invented core. */
export function getSpatialNeighbors(
  context: ViewContext,
  key: number | undefined
): {key: number; weight: number}[] {
  if (key === undefined) return [];
  const offsets = getSpatialValues(context, 'weightsRowOffsets');
  if (key + 1 >= offsets.length) return [];
  const neighbors = getSpatialValues(context, 'weightsNeighbors');
  const weights = getSpatialValues(context, 'weightsValues');
  return neighbors
    .slice(offsets[key], offsets[key + 1])
    .map((neighbor, index) => ({key: neighbor, weight: weights[offsets[key] + index]}));
}

/** Captured CSR arrays required to disclose a neighbourhood. */
export const SPATIAL_CSR_CAPTURES = [
  'weightsRowOffsets',
  'weightsNeighbors',
  'weightsValues'
] as const;
