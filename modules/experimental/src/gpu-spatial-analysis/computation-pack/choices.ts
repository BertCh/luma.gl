// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUAnalysisChoice, GPUComputationAttributeValue} from '@luma.gl/gpgpu';

import type {SpatialAnalysisStage} from './stages';

/**
 * A choice an operator exposes (weights type, support, CRS, correction method, permutations,
 * bandwidth), declared once per operator; scenarios turn it into a {@link GPUAnalysisChoice}
 * with the value they actually ran.
 */
export type SpatialChoiceTemplate = {
  stage: SpatialAnalysisStage;
  /** Parameter name as in the contributor's props or parameter schema. */
  name: string;
  label?: string;
  unit?: string;
  alternatives?: readonly GPUComputationAttributeValue[];
  /** Whether a stability sweep over this choice is meaningful. */
  sweepable: boolean;
};

/** Builds the record's choice entry for a template and the value the run used. */
export function createSpatialChoice(
  template: SpatialChoiceTemplate,
  value: GPUComputationAttributeValue
): GPUAnalysisChoice {
  return {...template, value};
}

/** Record-choice names consumed by spatial guards; scalar values become guard metadata.
 * weightsStandardisation: row/binary/other; pCorrection: method/none;
 * coordinateModel: projected/geographic; mappedVariableExtent: intensive/extensive;
 * denominator: field name; significanceAvailable: boolean; permutations: positive integer;
 * islandCount: nonnegative count; ebReferenceRate: pooled/mean-of-rates.
 */
export const SPATIAL_GUARD_CHOICE_NAMES = [
  'weightsStandardisation',
  'pCorrection',
  'coordinateModel',
  'mappedVariableExtent',
  'denominator',
  'significanceAvailable',
  'permutations',
  'islandCount',
  'ebReferenceRate',
  'sumBefore',
  'sumAfter',
  'conservationTolerance'
] as const;
