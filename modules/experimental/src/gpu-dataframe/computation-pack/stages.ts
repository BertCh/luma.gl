// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Dataframe analysis stages, in the order a query or contributor applies them. Receipt rows and
 * choice records are ordered by these.
 */
export const DATAFRAME_ANALYSIS_STAGES = [
  'select',
  'filter',
  'derive',
  'join',
  'group',
  'aggregate',
  'order',
  'classify',
  'summarise',
  'infer'
] as const;

/** One dataframe analysis stage. */
export type DataFrameAnalysisStage = (typeof DATAFRAME_ANALYSIS_STAGES)[number];

/** Whether a string is one of {@link DATAFRAME_ANALYSIS_STAGES}. */
export function isDataFrameAnalysisStage(value: string): value is DataFrameAnalysisStage {
  return (DATAFRAME_ANALYSIS_STAGES as readonly string[]).includes(value);
}
