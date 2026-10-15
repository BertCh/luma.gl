// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GuardSpec} from './guards';

const unknown = (key: string) => ({
  status: 'unknown' as const,
  reason: `missing or invalid ${key}`
});
const verdict = (passes: boolean, reason: string, success: string) => ({
  status: passes ? ('pass' as const) : ('fail' as const),
  reason: passes ? success : reason
});

/**
 * Metric geometry requires projected coordinates, or geographic coordinates measured with a
 * geodesic (ellipsoidal or spherical) metric (`metric: 'geodesic'`), which is metric on lon/lat.
 */
export const COORDINATE_MODEL_GUARD: GuardSpec = {
  id: 'coordinate-model',
  label: 'Coordinate model',
  blocks: ['metric-distance', 'metric-buffer', 'support-ring'],
  evaluate: ({metadata}) => {
    if (metadata['coordinateModel'] === 'projected')
      return verdict(true, '', 'projected coordinates');
    if (metadata['coordinateModel'] !== 'geographic') return unknown('coordinateModel');
    return verdict(
      metadata['metric'] === 'geodesic',
      'metric glyphs need projected coordinates or a geodesic metric',
      'geographic coordinates with a geodesic metric'
    );
  }
};

/** Extensive variables require a denominator for choropleth mapping. */
export const COUNT_VS_RATE_GUARD: GuardSpec = {
  id: 'count-vs-rate',
  label: 'Count versus rate',
  blocks: ['choropleth'],
  evaluate: ({metadata}) => {
    if (
      metadata['mappedVariableExtent'] !== 'extensive' &&
      metadata['mappedVariableExtent'] !== 'intensive'
    )
      return unknown('mappedVariableExtent');
    return verdict(
      metadata['mappedVariableExtent'] === 'intensive' ||
        (typeof metadata['denominator'] === 'string' && metadata['denominator'].trim().length > 0),
      'extensive choropleths need a denominator',
      'intensive variable or explicit denominator'
    );
  }
};

/** Aggregation preserves an extensive sum within an explicit relative tolerance. */
export const CONSERVATION_GUARD: GuardSpec = {
  id: 'conservation',
  label: 'Conservation',
  blocks: ['aggregate-total'],
  evaluate: ({metadata}) => {
    const before = metadata['sumBefore'];
    const after = metadata['sumAfter'];
    if (typeof before !== 'number' || !Number.isFinite(before)) return unknown('sumBefore');
    if (typeof after !== 'number' || !Number.isFinite(after)) return unknown('sumAfter');
    const tolerance = metadata['conservationTolerance'] ?? 1e-6;
    if (typeof tolerance !== 'number' || !Number.isFinite(tolerance) || tolerance < 0)
      return unknown('conservationTolerance');
    return verdict(
      Math.abs(before - after) <= tolerance * Math.max(Math.abs(before), Math.abs(after)),
      'sum changed beyond conservation tolerance',
      'sum conserved'
    );
  }
};

/** Significance classes disclose and require a multiple-testing correction. */
export const UNCORRECTED_P_GUARD: GuardSpec = {
  id: 'uncorrected-p',
  label: 'P correction',
  blocks: ['significance-class'],
  evaluate: ({metadata}) =>
    typeof metadata['pCorrection'] === 'string' && metadata['pCorrection'].length > 0
      ? verdict(
          metadata['pCorrection'] !== 'none',
          'uncorrected p-values',
          `correction: ${metadata['pCorrection']}`
        )
      : unknown('pCorrection')
};

/** Cluster classification needs significance evidence. */
export const CLASSIFICATION_NEEDS_INFERENCE_GUARD: GuardSpec = {
  id: 'classification-needs-inference',
  label: 'Classification inference',
  blocks: ['cluster-class'],
  evaluate: ({metadata}) =>
    typeof metadata['significanceAvailable'] === 'boolean'
      ? verdict(
          metadata['significanceAvailable'],
          'classification needs significance inference',
          'significance available'
        )
      : unknown('significanceAvailable')
};

/** The Moran scatter slope equals I only for row-standardised weights. */
export const ROW_STANDARDISATION_GUARD: GuardSpec = {
  id: 'row-standardisation',
  label: 'Row standardisation',
  blocks: ['slope-equals-i'],
  evaluate: ({metadata}) =>
    typeof metadata['weightsStandardisation'] === 'string' &&
    metadata['weightsStandardisation'].length > 0
      ? verdict(
          metadata['weightsStandardisation'] === 'row',
          'slope = I holds only for row-standardised weights',
          'row-standardised weights'
        )
      : unknown('weightsStandardisation')
};

/** Standard premises, in their presentation order. */
export const STANDARD_GUARDS: readonly GuardSpec[] = [
  COORDINATE_MODEL_GUARD,
  COUNT_VS_RATE_GUARD,
  CONSERVATION_GUARD,
  UNCORRECTED_P_GUARD,
  CLASSIFICATION_NEEDS_INFERENCE_GUARD,
  ROW_STANDARDISATION_GUARD
];
