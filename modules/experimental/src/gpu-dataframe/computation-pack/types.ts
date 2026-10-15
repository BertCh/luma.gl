// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUComputationFormulaRef} from '@luma.gl/gpgpu';

import type {ViewPackNoFormula} from '../../computation-views/registry/view-registration';
import type {OperatorDescriptor, VariableKind} from '../../computation-views/types';
import type {DataFrameGuardId} from './guards';
import type {DataFrameAnalysisStage} from './stages';

/** Operator family; one `operators/<family>.ts` file each. */
export type DataFrameFamily =
  | 'queries'
  | 'classification'
  | 'temporal'
  | 'groups-joins'
  | 'pair-statistics'
  | 'indicators';

/** Kind of datum a dataframe element's invocations map to. */
export type DataFrameDatumKind = {
  /** Key space id, such as `row`, `group`, `bin` or `lag`. */
  kind: string;
  /** Display label. */
  label: string;
  /** Index shape of the key space. */
  indexSpace: '1d' | '2d' | 'csr' | 'batch';
};

/** Computation-pack entry of one exported dataframe element, keyed by its exported name. */
export type DataFrameOperatorEntry = {
  /** Operator family (docs grouping and owning file). */
  family: DataFrameFamily;
  /** Generic `scope × reduction` descriptor. */
  descriptor: OperatorDescriptor;
  /** Curated formula of the primary output, or an explicit absence with a reason. */
  formula: GPUComputationFormulaRef | ViewPackNoFormula;
  /** Analysis stages the element performs, in order. */
  stages: readonly DataFrameAnalysisStage[];
  /** Datum kinds of the element's kernels. */
  datumKinds: readonly DataFrameDatumKind[];
  /** Guards that apply to the element's figures. */
  guards: readonly DataFrameGuardId[];
  /** Analyst choice names recorded by its lens scenarios (method, class count, lag width...). */
  choices: readonly string[];
  /** Measurement level of named captures (inputs and outputs). */
  variables?: Readonly<Record<string, VariableKind>>;
  /** Fixed guard metadata. */
  metadata?: Readonly<Record<string, string | number | boolean>>;
  /**
   * Repo-relative source files whose generated WGSL contains every `wgsl:` binding of the formula;
   * the pack spec checks each binding against their text.
   */
  sourceFiles: readonly string[];
};

/** One family table keyed by exported element name. */
export type DataFrameFamilyOperators = Readonly<Record<string, DataFrameOperatorEntry>>;
