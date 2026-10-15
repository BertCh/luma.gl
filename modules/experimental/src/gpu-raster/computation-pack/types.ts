// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUComputationFormulaRef} from '@luma.gl/gpgpu';

import type {ViewPackNoFormula} from '../../computation-views/registry/view-registration';
import type {OperatorDescriptor, VariableKind} from '../../computation-views/types';

/**
 * Raster analysis stages, in receipt order. `neighbourhood` is the stencil or window a cell reads;
 * `iteration` is the gated fixpoint loop of propagating operators (cost distance, fill, flats,
 * accumulation).
 */
export const RASTER_ANALYSIS_STAGES = [
  'representation',
  'support',
  'neighbourhood',
  'operator',
  'iteration',
  'presentation'
] as const;

/** One raster analysis stage. */
export type RasterAnalysisStage = (typeof RASTER_ANALYSIS_STAGES)[number];

/**
 * Named variable kinds shared by the raster and terrain packs. `measure` quantities (elevation,
 * distance, cost) are interval or ratio; class rasters are nominal; angles are cyclic.
 */
export const RASTER_VARIABLE_KINDS = {
  /** Elevations, heights above a datum, relief: interval quantities in length units. */
  elevation: {level: 'interval', extent: 'intensive'},
  /** Lengths, accumulated costs, areas and counts of cells: ratio quantities. */
  measure: {level: 'ratio', extent: 'extensive'},
  /** Per-cell rates, gradients, indices and reflectances: ratio, not summable over cells. */
  rate: {level: 'ratio', extent: 'intensive'},
  /** Signed scores and differences with a stated zero (curvature, dNBR, relief anomaly). */
  score: {level: 'interval', extent: 'intensive'},
  /** Fractions in [0, 1] such as hillshade, sky view, visibility or irradiance ratios. */
  fraction: {level: 'ratio', extent: 'intensive'},
  /** Directions such as aspect or flow direction angles. */
  direction: {level: 'cyclic'},
  /** Categories: land cover, landform class, basin or component id, D8 code. */
  category: {level: 'nominal'},
  /** Ordered classes: bands, stream order, rank. */
  rank: {level: 'ordinal'},
  /** Iteration stamps (round in which a cell settled). */
  round: {level: 'ordinal'}
} as const satisfies Readonly<Record<string, VariableKind>>;

/** Name of a shared raster variable kind. */
export type RasterVariableKindName = keyof typeof RASTER_VARIABLE_KINDS;

/** A datum kind an operator's lens can pin, with its index space. */
export type RasterDatumKind = {
  /** Kind id, such as `cell`, `observer`, `vertex`, `particle`, `zone`. */
  kind: string;
  label: string;
  indexSpace: '1d' | '2d' | 'csr' | 'batch';
};

/** An analyst choice the operator exposes, turned into a record choice by the scenario. */
export type RasterChoiceTemplate = {
  stage: RasterAnalysisStage;
  /** Parameter name as in the contributor props or settings packer. */
  name: string;
  label?: string;
  unit?: string;
  alternatives?: readonly (string | number | boolean)[];
  /** Whether a stability sweep over this choice is meaningful. */
  sweepable: boolean;
};

/**
 * One exported raster or terrain contributor in its computation pack, keyed by the exported class
 * name (also the lens scenario's `element`).
 */
export type RasterOperatorEntry = {
  /** Docs section family, such as `derivatives`, `hydrology`, `algebra`. */
  family: string;
  /** `scope × reduction` (+ `scopeSecondary`). */
  descriptor: Omit<OperatorDescriptor, 'variableKinds'>;
  /** Variable kind per input/output capture name. */
  variables: Readonly<Record<string, RasterVariableKindName>>;
  /** Analysis stages the operator acts at, in order. */
  stages: readonly RasterAnalysisStage[];
  datumKinds: readonly RasterDatumKind[];
  /** Neighbourhood a cell reads, for the datum receipt. */
  neighbourhood: 'stencil-3x3' | 'window' | 'ray' | 'path' | 'zone' | 'none';
  /** Guard ids that apply to this operator's glyphs (see the pack's guard table). */
  guards: readonly string[];
  /** Analyst choices the operator exposes. */
  choices: readonly RasterChoiceTemplate[];
  /**
   * Curated formula bound to real WGSL names (`wgsl:<binding or identifier>`), checked against
   * {@link RasterOperatorEntry.sources}, or an explicit none with a reason (procedural algorithms
   * document a SkeletonView instead).
   */
  formula: GPUComputationFormulaRef | ViewPackNoFormula;
  /** Repo-relative source files whose text must contain every `wgsl:` symbol binding. */
  sources: readonly string[];
  /** Guard metadata fixed for the operator. */
  metadata?: Readonly<Record<string, string | number | boolean>>;
};

/** Operator entries of one pack, keyed by exported class name. */
export type RasterOperatorTable = Readonly<Record<string, RasterOperatorEntry>>;
