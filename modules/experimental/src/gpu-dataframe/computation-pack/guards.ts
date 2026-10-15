// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GuardSpec} from '../../computation-views/guards/guards';
import {
  CLASSIFICATION_NEEDS_INFERENCE_GUARD,
  CONSERVATION_GUARD,
  COORDINATE_MODEL_GUARD,
  COUNT_VS_RATE_GUARD,
  UNCORRECTED_P_GUARD
} from '../../computation-views/guards/standard-guards';
import type {LensRecord} from '../../computation-views/registry/view-registration';

/** Converts scalar analysis choices of a record into guard metadata; later choices win. */
export function getDataFrameGuardMetadata(
  record?: LensRecord
): Record<string, string | number | boolean> {
  const metadata: Record<string, string | number | boolean> = {};
  for (const choice of record?.choices ?? []) {
    if (['string', 'number', 'boolean'].includes(typeof choice.value)) {
      metadata[choice.name] = choice.value as string | number | boolean;
    }
  }
  return metadata;
}

/** Adds record-choice fallback while keeping explicitly supplied metadata. */
function withDataFrameRecordChoices(guard: GuardSpec): GuardSpec {
  return {
    ...guard,
    evaluate: context =>
      guard.evaluate({
        ...context,
        metadata: {...getDataFrameGuardMetadata(context.record), ...context.metadata}
      })
  };
}

/**
 * Premises that apply to dataframe elements: metric distances of pair statistics, count-vs-rate
 * and conservation of classified or aggregated columns, and multiple-testing disclosure.
 */
export const DATAFRAME_GUARDS = {
  'coordinate-model': withDataFrameRecordChoices(COORDINATE_MODEL_GUARD),
  'count-vs-rate': withDataFrameRecordChoices(COUNT_VS_RATE_GUARD),
  conservation: withDataFrameRecordChoices(CONSERVATION_GUARD),
  'uncorrected-p': withDataFrameRecordChoices(UNCORRECTED_P_GUARD),
  'classification-needs-inference': withDataFrameRecordChoices(CLASSIFICATION_NEEDS_INFERENCE_GUARD)
} as const satisfies Readonly<Record<string, GuardSpec>>;

/** Identifier of a dataframe guard. */
export type DataFrameGuardId = keyof typeof DATAFRAME_GUARDS;

/** Resolves guard ids to specs, in the given order. */
export function getDataFrameGuards(ids: readonly DataFrameGuardId[]): GuardSpec[] {
  return ids.map(id => DATAFRAME_GUARDS[id]);
}
