// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  GPUInferenceMetadata,
  GPUMultipleTestingMethod
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Emerging-hot-spot category colors, indexed by category code (0 to 17); code 0 is transparent. */
export const EMERGING_CATEGORY_COLORS: readonly (readonly [number, number, number])[] = [
  [0, 0, 0],
  [255, 224, 102],
  [255, 160, 60],
  [165, 0, 38],
  [215, 48, 39],
  [244, 160, 150],
  [253, 208, 162],
  [200, 100, 200],
  [170, 140, 140],
  [160, 230, 255],
  [90, 170, 255],
  [8, 48, 107],
  [33, 102, 172],
  [150, 190, 230],
  [190, 230, 215],
  [120, 100, 220],
  [130, 150, 170],
  [235, 235, 235]
];

/**
 * Emerging-hot-spot category names in code order. Code 17 is `UNDETERMINED`: the final slice is missing,
 * or a persistent cell has too few finite slices for a trend decision.
 */
export const EMERGING_CATEGORY_NAMES = [
  'No pattern',
  'New hot spot',
  'Consecutive hot spot',
  'Intensifying hot spot',
  'Persistent hot spot',
  'Diminishing hot spot',
  'Sporadic hot spot',
  'Oscillating hot spot',
  'Historical hot spot',
  'New cold spot',
  'Consecutive cold spot',
  'Intensifying cold spot',
  'Persistent cold spot',
  'Diminishing cold spot',
  'Sporadic cold spot',
  'Oscillating cold spot',
  'Historical cold spot',
  'Undetermined'
] as const;

/** Slices of each cube source of the emerging-hot-spots scene (months, 52 weeks, 73 five-day periods). */
export const EMERGING_CUBE_SLICE_COUNTS = {months: 12, weeks: 52, days: 73};

/** Plain-language names of the multiple-testing methods, for readouts and legends. */
export const CORRECTION_NAMES: Record<GPUMultipleTestingMethod, string> = {
  none: 'no correction',
  bonferroni: 'Bonferroni',
  holm: 'Holm',
  'benjamini-hochberg': 'Benjamini-Hochberg FDR',
  'benjamini-yekutieli': 'Benjamini-Yekutieli FDR'
};

/** One-sentence description of the inference behind the hot and cold bins, for legends. */
export function describeEmergingInference(
  inference: GPUInferenceMetadata | undefined,
  confidence: string
): string {
  const method = inference?.multipleTesting ?? 'benjamini-hochberg';
  const level = (1 - Number(confidence)).toFixed(2);
  return method === 'none'
    ? `Two-sided Gi* p <= ${level} per bin, no correction.`
    : `Two-sided Gi* p adjusted by ${CORRECTION_NAMES[method]} over all valid bins, adjusted p <= ${level}.`;
}
