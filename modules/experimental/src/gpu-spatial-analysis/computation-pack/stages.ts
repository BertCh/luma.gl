// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * The six spatial analysis stages (roadmap §B, theory table "Analysis stage"). Receipt rows,
 * choice records and guards are ordered by these. Every hard misconception sits at a stage.
 */
export const SPATIAL_ANALYSIS_STAGES = [
  'representation',
  'support',
  'neighbourhood',
  'operator',
  'inference',
  'presentation'
] as const;

/** One spatial analysis stage. */
export type SpatialAnalysisStage = (typeof SPATIAL_ANALYSIS_STAGES)[number];

/**
 * Representation facet of a spatial quantity. Contributors convert between these (KDE turns
 * events into a field; terrain flow turns a field into a network).
 */
export type SpatialRepresentation = 'field' | 'object' | 'network' | 'event';

/** Whether a string is one of {@link SPATIAL_ANALYSIS_STAGES}. */
export function isSpatialAnalysisStage(value: string): value is SpatialAnalysisStage {
  return (SPATIAL_ANALYSIS_STAGES as readonly string[]).includes(value);
}
