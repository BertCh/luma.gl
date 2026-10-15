// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUComputationFormulaRef} from '@luma.gl/gpgpu';
import {registerDefaultViewPack} from '../../computation-views/registry/default-registry';
import type {
  ViewPackData,
  ViewPackNoFormula,
  ViewPackRegistration
} from '../../computation-views/registry/view-registration';
import type {OperatorDescriptor} from '../../computation-views/types';
import {getDataFrameGuards} from './guards';
import {DATAFRAME_ANALYSIS_STAGES} from './stages';
import type {DataFrameFamilyOperators, DataFrameOperatorEntry} from './types';
import {QUERIES_OPERATORS} from './operators/queries';
import {CLASSIFICATION_OPERATORS} from './operators/classification';
import {TEMPORAL_OPERATORS} from './operators/temporal';
import {GROUPS_JOINS_OPERATORS} from './operators/groups-joins';
import {PAIR_STATISTICS_OPERATORS} from './operators/pair-statistics';
import {INDICATORS_OPERATORS} from './operators/indicators';

export * from './types';
export * from './stages';
export * from './guards';

/** Domain pack id shared by dataframe scenarios. */
export const DATAFRAME_COMPUTATION_PACK_ID = 'gpu-dataframe';

/** All family tables, in documentation order. */
export const DATAFRAME_FAMILY_OPERATORS: readonly DataFrameFamilyOperators[] = [
  QUERIES_OPERATORS,
  CLASSIFICATION_OPERATORS,
  TEMPORAL_OPERATORS,
  GROUPS_JOINS_OPERATORS,
  PAIR_STATISTICS_OPERATORS,
  INDICATORS_OPERATORS
];

/** Returns every entry, rejecting duplicate exported element names. */
export function getDataFrameOperatorEntries(): Readonly<Record<string, DataFrameOperatorEntry>> {
  const entries: Record<string, DataFrameOperatorEntry> = {};
  for (const family of DATAFRAME_FAMILY_OPERATORS) {
    for (const [name, entry] of Object.entries(family)) {
      if (entries[name]) throw new Error(`dataframe pack: ${name} declared twice`);
      entries[name] = entry;
    }
  }
  return entries;
}

/** Resolves one exported element without importing GPU code. */
export function getDataFrameOperatorEntry(element: string): DataFrameOperatorEntry | undefined {
  return getDataFrameOperatorEntries()[element];
}

/** Resolves the guards for an element; unknown elements have no guards. */
export function getDataFrameElementGuards(element: string) {
  return getDataFrameGuards(getDataFrameOperatorEntry(element)?.guards ?? []);
}

/** Plain operator descriptors, curated formulas and analysis stages. */
export function createDataFramePackData(): ViewPackData {
  const operators: Record<string, OperatorDescriptor> = {};
  const formulas: Record<string, GPUComputationFormulaRef | ViewPackNoFormula> = {};
  for (const [name, entry] of Object.entries(getDataFrameOperatorEntries())) {
    operators[name] = {...entry.descriptor, variableKinds: entry.variables ?? {}};
    formulas[name] = entry.formula;
  }
  return {
    id: DATAFRAME_COMPUTATION_PACK_ID,
    operators,
    formulas,
    analysisStages: DATAFRAME_ANALYSIS_STAGES
  };
}

/** Element-specific variable kinds and fixed metadata, with run choices taking precedence. */
export function createDataFrameElementPackData(
  element: string,
  runMetadata: Readonly<Record<string, string | number | boolean>> = {}
): ViewPackData {
  const entry = getDataFrameOperatorEntry(element);
  return {
    ...createDataFramePackData(),
    variableKinds: entry?.variables ?? {},
    metadata: {...entry?.metadata, ...runMetadata}
  };
}

/** Registers dataframe semantics with the default computation-view registry. */
export function createDataFrameComputationPack(): ViewPackRegistration {
  return {
    ...createDataFramePackData(),
    views: [],
    guardsFor: ({element}) => (element ? getDataFrameElementGuards(element) : [])
  };
}

registerDefaultViewPack(createDataFrameComputationPack());
