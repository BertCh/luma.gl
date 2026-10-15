// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUComputationFormulaRef} from '@luma.gl/gpgpu';

import type {ViewPackNoFormula} from '../../computation-views/registry/view-registration';
import type {OperatorDescriptor} from '../../computation-views/types';
import type {SpatialChoiceTemplate} from './choices';
import type {SpatialGuardId} from './guards';
import type {SpatialAnalysisStage, SpatialRepresentation} from './stages';
import type {SpatialVariableKindName} from './variable-kinds';

/** The eleven operator families (one docs page and one `operators/<family>.ts` file each) plus recipes. */
export type SpatialOperatorFamily =
  | 'weights'
  | 'statistics'
  | 'dynamics'
  | 'accessibility'
  | 'joins'
  | 'validity'
  | 'density'
  | 'cells'
  | 'lines'
  | 'geometry'
  | 'regression'
  | 'recipes';

/** A datum kind an operator's lens can pin, with its index space. */
export type SpatialDatumKind = {
  /** Kind id, such as `region`, `point`, `pair`, `cell`. */
  kind: string;
  label: string;
  indexSpace: '1d' | '2d' | 'csr' | 'batch';
};

/**
 * One exported contributor (or `add*Recipe` function) in the spatial computation pack.
 * Keyed by the exported name, which is also the lens scenario's `element`.
 */
export type SpatialOperatorEntry = {
  family: SpatialOperatorFamily;
  /** `scope × reduction` (+ `scopeSecondary`); variable kinds go in {@link variables}. */
  descriptor: Omit<OperatorDescriptor, 'variableKinds'>;
  /** Variable kind per output/input capture name, by shared kind name. */
  variables: Readonly<Record<string, SpatialVariableKindName>>;
  /** Analysis stages the operator acts at, in order. */
  stages: readonly SpatialAnalysisStage[];
  datumKinds: readonly SpatialDatumKind[];
  /** Representation conversion, when the operator changes it (KDE: event → field). */
  representation?: {from: SpatialRepresentation; to: SpatialRepresentation};
  /** Guards that apply to this operator's glyphs. */
  guards: readonly SpatialGuardId[];
  /** Analyst choices the operator exposes. */
  choices: readonly SpatialChoiceTemplate[];
  /** Curated formula bound to real buffer/WGSL names, or an explicit none with a reason. */
  formula: GPUComputationFormulaRef | ViewPackNoFormula;
  /** Guard metadata that is fixed for the operator (scenarios add run-specific metadata). */
  metadata?: Readonly<Record<string, string | number | boolean>>;
};

/** Operator entries of one family, keyed by exported class or recipe function name. */
export type SpatialFamilyOperators = Readonly<Record<string, SpatialOperatorEntry>>;
