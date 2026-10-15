// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  CLASSIFICATION_NEEDS_INFERENCE_GUARD,
  CONSERVATION_GUARD,
  COORDINATE_MODEL_GUARD,
  COUNT_VS_RATE_GUARD,
  ROW_STANDARDISATION_GUARD,
  UNCORRECTED_P_GUARD
} from '../../computation-views/guards/standard-guards';
import type {LensRecord} from '../../computation-views/registry/view-registration';
import type {GuardSpec} from '../../computation-views/guards/guards';

/** Converts scalar analysis choices into guard metadata; later choices win. */
export function getSpatialGuardMetadata(
  record?: LensRecord
): Record<string, string | number | boolean> {
  return Object.fromEntries(
    (record?.choices ?? [])
      .filter(choice => ['string', 'number', 'boolean'].includes(typeof choice.value))
      .map(choice => [choice.name, choice.value])
  ) as Record<string, string | number | boolean>;
}

/** Adds record-choice fallback while preserving explicitly supplied metadata, including unknowns. */
export function withSpatialRecordChoices(guard: GuardSpec): GuardSpec {
  return {
    ...guard,
    evaluate: context =>
      guard.evaluate({
        ...context,
        metadata: {...getSpatialGuardMetadata(context.record), ...context.metadata}
      })
  };
}

/** Spatial premises, all evaluated against record choices plus explicit metadata. */
export const SPATIAL_GUARDS = {
  'coordinate-model': withSpatialRecordChoices(COORDINATE_MODEL_GUARD),
  'count-vs-rate': withSpatialRecordChoices(COUNT_VS_RATE_GUARD),
  conservation: withSpatialRecordChoices(CONSERVATION_GUARD),
  'uncorrected-p': withSpatialRecordChoices(UNCORRECTED_P_GUARD),
  'classification-needs-inference': withSpatialRecordChoices(CLASSIFICATION_NEEDS_INFERENCE_GUARD),
  'row-standardisation': withSpatialRecordChoices(ROW_STANDARDISATION_GUARD),
  'pseudo-p-grid': withSpatialRecordChoices({
    id: 'pseudo-p-grid',
    label: 'Pseudo-p grid',
    blocks: ['exact-p'],
    evaluate: ({metadata}) => {
      const permutations = metadata['permutations'];
      return typeof permutations === 'number' && Number.isInteger(permutations) && permutations > 0
        ? {
            status: 'pass',
            reason: `achievable grid k/(${permutations}+1) = k/${permutations + 1}; floor 1/${permutations + 1} = ${1 / (permutations + 1)}`
          }
        : {status: 'unknown', reason: 'missing or invalid permutations'};
    }
  }),
  'islands-disclosed': withSpatialRecordChoices({
    id: 'islands-disclosed',
    label: 'Islands disclosed',
    blocks: ['neighbourhood-map'],
    evaluate: ({metadata}) => {
      const count = metadata['islandCount'];
      return typeof count === 'number' && Number.isFinite(count) && count >= 0
        ? {status: 'pass', reason: `${count} islands, hatched`}
        : {status: 'unknown', reason: 'missing or invalid islandCount'};
    }
  }),
  'eb-reference-rate': withSpatialRecordChoices({
    id: 'eb-reference-rate',
    label: 'EB reference rate',
    blocks: ['eb-reference'],
    evaluate: ({metadata}) =>
      metadata['ebReferenceRate'] === 'pooled'
        ? {status: 'pass', reason: 'pooled μ = ΣO/ΣP'}
        : metadata['ebReferenceRate'] === 'mean-of-rates'
          ? {status: 'fail', reason: 'EB reference requires pooled μ = ΣO/ΣP, not mean-of-rates'}
          : {status: 'unknown', reason: 'missing or invalid ebReferenceRate'}
  })
} as const satisfies Readonly<Record<string, GuardSpec>>;

/** Misconceptions addressed by the guards (roadmap M1–M16). */
export const SPATIAL_GUARD_MISCONCEPTIONS: Readonly<Record<SpatialGuardId, string>> = {
  'coordinate-model': 'M8',
  'count-vs-rate': 'M9',
  conservation: 'M12',
  'uncorrected-p': 'M3',
  'classification-needs-inference': 'M1',
  'row-standardisation': 'M7',
  'pseudo-p-grid': 'M2',
  'islands-disclosed': 'M6',
  'eb-reference-rate': 'M10'
};

/** Id of a guard the spatial pack can apply. */
export type SpatialGuardId = keyof typeof SPATIAL_GUARDS;

/** Resolves guard ids to specs in the declared order. */
export function getSpatialGuards(ids: readonly SpatialGuardId[]): GuardSpec[] {
  return ids.map(id => SPATIAL_GUARDS[id]);
}
