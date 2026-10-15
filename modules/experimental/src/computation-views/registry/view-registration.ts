// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUComputationFormulaRef, GPUComputationRecordWithSidecars} from '@luma.gl/gpgpu';

import type {
  AxisBindings,
  AxisId,
  ComputationGrain,
  ComputationLevel,
  DescriptionFacts,
  LensRungId,
  MarkProvenance,
  OperatorDescriptor,
  RenderTeX,
  RungDescriptor,
  VariableKind,
  ViewGround,
  ViewSignature
} from '../types';
import type {GuardSpec} from '../guards/guards';
import type {ScaleRegistry} from '../scales/scale-registry';
import type {ViewElementNode} from '../view-node';

/** The record a view projects: one recorded real execution plus its metrics and captures sidecars. */
export type LensRecord = GPUComputationRecordWithSidecars;

/**
 * Record type a view inspects (moldable-inspector style). `record` is the whole record; the table
 * fallback accepts every type.
 */
export type ViewRecordType =
  | 'record'
  | 'graph'
  | 'node'
  | 'formula'
  | 'datum'
  | 'buffer'
  | 'dispatch'
  | 'loop'
  | 'timing'
  | 'comparison'
  | 'operator'
  | (string & {});

/**
 * Plain-data domain-pack inputs a view may consult. The core never knows a domain: packs pass
 * these through the view context (wave-2 packs register them with {@link ViewPackRegistration}).
 */
export type ViewPackData = {
  /** Pack id, such as `gpu-spatial-analysis`. */
  id: string;
  /**
   * Operator table keyed by node `op`. For contributors the key is the exported class name
   * (`GPULocalMoran`); for recipes the function name. Lens scenarios use the same name as
   * `LensScenarioDescriptor.element`, so the docs coverage contract joins on it.
   */
  operators?: Readonly<Record<string, OperatorDescriptor>>;
  /**
   * Formula per operator, keyed like {@link ViewPackData.operators}: a curated or derived formula
   * ref, or an explicit `{formula: 'none', reason}` for operators without a closed form (those
   * document a SkeletonView instead).
   */
  formulas?: Readonly<Record<string, GPUComputationFormulaRef | ViewPackNoFormula>>;
  /** Analysis stages in order, used to order receipt rows. */
  analysisStages?: readonly string[];
  /** Capture name → analysis stage, for receipt rows. */
  captureStages?: Readonly<Record<string, string>>;
  /** Variable kinds by capture name, used by guards and scales. */
  variableKinds?: Readonly<Record<string, VariableKind>>;
  /** Neighbourhood slot family of each key space, for the datum receipt. */
  neighbourhoods?: Readonly<Record<string, ViewNeighbourhoodFamily>>;
  /** Free-form guard metadata (CRS model, weights standardisation, correction method). */
  metadata?: Readonly<Record<string, string | number | boolean>>;
};

/** Explicit absence of a closed-form formula for an operator. */
export type ViewPackNoFormula = {formula: 'none'; reason: string};

/** Neighbourhood slot family drawn in a datum receipt. */
export type ViewNeighbourhoodFamily =
  | {kind: 'csr-star'; relation: string}
  | {kind: 'stencil-3x3'; keySpace: string}
  | {kind: 'candidate-list'; relation: string}
  | {kind: 'flag-offset-slot'; captures: readonly string[]}
  | {kind: 'none'};

/** Live selection state handed to a view render (absent for static twins). */
export type ViewSelectionState = {
  /** Highlighted marks with partial coverage, keyed by mark key. */
  highlighted?: ReadonlyMap<number, {covered: number; total: number}>;
  /** Hovered mark keys. */
  hovered?: ReadonlySet<number>;
  /** Keys selected elsewhere that this view cannot show (off-view cue). */
  offViewCount?: number;
  /** Whether the selection could not be mapped into this view (*unknown*, never guessed). */
  unknownReason?: string;
};

/** Everything a view needs to build and render. Pure data; no DOM. */
export type ViewContext = {
  record: LensRecord;
  /** Axis bindings (`entity` = datum pin, `iteration`, `node`, `workgroup`...). */
  bindings: AxisBindings;
  /** Page ground (§E rule 8). */
  ground: ViewGround;
  /** Unique prefix for SVG ids in this lens. */
  idPrefix: string;
  /** Under reduced motion views use stepped frames. */
  reducedMotion: boolean;
  /** Host TeX renderer; without it formulas print as TeX source plus ASCII. */
  renderTeX?: RenderTeX;
  /** Domain-pack data for this record's domains. */
  packs?: readonly ViewPackData[];
  /** A second record or capture set for comparison views (k vs k', run A vs B, GPU vs oracle). */
  compare?: {label: string; record: LensRecord; referenceLabel: string};
  /** Live selection state; absent in static twins. */
  selection?: ViewSelectionState;
  /**
   * The run's frozen scales (§E rule 1: one scale per variable, shared by every view of the lens
   * page). Views resolve value colours through it and never build private scales.
   */
  scales?: ScaleRegistry;
  /** Preferred width in CSS pixels for SVG layouts (twins default to 640). */
  width?: number;
  /** View-specific options from the directive or host. */
  options?: Readonly<Record<string, string | number | boolean>>;
};

/** Whether a record satisfies a view's signature. A disabled rung shows the reason, never a fake. */
export type ViewSatisfaction = {ok: true} | {ok: false; reason: string};

/** One chip in a view's axis header (`row 17 pinned · iteration free (0–11 of 64) · Apple M3`). */
export type AxisHeaderChip = {
  axis: AxisId;
  label: string;
  state: 'bound' | 'free' | 'context';
  /** Value text, such as `17`, `0–11 of 64` or `all 64 iterations`. */
  valueText: string;
  /** Action when clicked: bound chips free their axis; free chips do nothing until a mark binds. */
  action?: 'free';
};

/** Axis header model of one view. */
export type AxisHeaderModel = {
  chips: readonly AxisHeaderChip[];
  /** Adapter and variant label, always shown. */
  adapterLabel: string;
};

/** Receipt table printed in the static twin and the table fallback. */
export type LensReceiptTable = {
  columns: readonly string[];
  rows: readonly (readonly string[])[];
  /** Provenance per row. */
  provenance?: readonly MarkProvenance[];
};

/** Formula payload printed in the static twin (the host renders TeX + MathML). */
export type LensFormulaPayload = {
  tex: string;
  ascii: string;
  derivation: 'derived' | 'curated';
};

/**
 * A view registration: about ten lines per view. `build` projects the record into a model, `render`
 * draws it as a view tree (string for twins, DOM for live), `describe` feeds the one description
 * generator.
 */
export type ViewRegistration<TModel = unknown> = {
  /** Registry-unique id, such as `formula-card`. */
  id: string;
  /** Display name. */
  label: string;
  /** Rung this view serves. */
  rung: LensRungId;
  /** Record type inspected. */
  recordType: ViewRecordType;
  /** Lower is preferred within a rung. Domain views use 0–49, generic views 50–99, table 1000. */
  order: number;
  /** Bound and free axes with layout roles. */
  signature: ViewSignature;
  /** Audience, task and interaction. */
  descriptor: RungDescriptor;
  /** Level answered. */
  level: ComputationLevel;
  /** Grain. */
  grain: ComputationGrain;
  /** Activation predicate on the selection path; default always active. */
  isActive?(context: ViewContext): boolean;
  /** Whether this record satisfies the view (else the rung is disabled with the reason). */
  satisfies(context: ViewContext): ViewSatisfaction;
  /** Projects the record into the view model. Only called when `satisfies` is ok. */
  build(context: ViewContext): TModel;
  /** Draws the model. Marks carry `data-action`/`data-axis`/`data-key`; no closures. */
  render(model: TModel, context: ViewContext): ViewElementNode;
  /** Facts for the description generator (alt text, live region, llms twin). */
  describe(model: TModel, context: ViewContext): DescriptionFacts;
  /** Axis header chips; default derives them from the signature and bindings. */
  axisHeader?(model: TModel, context: ViewContext): AxisHeaderModel;
  /** Receipt table for the static twin, if the view has one. */
  receipt?(model: TModel, context: ViewContext): LensReceiptTable | undefined;
  /** Formula for the static twin, if the view shows one. */
  formula?(model: TModel, context: ViewContext): LensFormulaPayload | undefined;
};

/** What a pack is asked guards for: one recorded run of one documented element. */
export type ViewPackGuardTarget = {
  record: LensRecord;
  /** Exported element name (`GPULocalMoran`), when the host knows the scenario. */
  element?: string;
  /** Scenario id, such as `gpu-spatial-analysis.local-moran`. */
  scenarioId?: string;
};

/** A domain pack's registration: plain data plus optional views. */
export type ViewPackRegistration = ViewPackData & {
  /** Domain-specific views (order 0–49 so they win over generic views). */
  views?: readonly ViewRegistration<unknown>[];
  /**
   * Guards that apply to one element's record. Static twins and live lenses evaluate them, so a
   * card shows only the premises that matter for that element (never six *unknown* chips).
   */
  guardsFor?(target: ViewPackGuardTarget): readonly GuardSpec[];
};
