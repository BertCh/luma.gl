// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUComputationFormulaRef} from '@luma.gl/gpgpu';

import type {
  ViewPackNoFormula,
  ViewPackRegistration,
  ViewRegistration
} from '../../computation-views/registry/view-registration';
import type {OperatorDescriptor, VariableKind} from '../../computation-views/types';
import {withGuardMetadata} from '../../computation-views/guards/guards';
import {getRasterGuards} from './guards';
import {
  RASTER_ANALYSIS_STAGES,
  RASTER_VARIABLE_KINDS,
  type RasterOperatorEntry,
  type RasterOperatorTable
} from './types';

/** Merges family tables keyed by exported class name; a name declared twice throws. */
export function mergeRasterOperatorTables(
  packId: string,
  tables: readonly RasterOperatorTable[]
): Readonly<Record<string, RasterOperatorEntry>> {
  const entries: Record<string, RasterOperatorEntry> = {};
  for (const table of tables) {
    for (const [name, entry] of Object.entries(table)) {
      if (entries[name]) throw new Error(`${packId} pack: ${name} declared twice`);
      entries[name] = entry;
    }
  }
  return entries;
}

/** Resolves an entry's named variable kinds into {@link VariableKind}s. */
export function getRasterVariableKinds(
  entry: RasterOperatorEntry
): Readonly<Record<string, VariableKind>> {
  return Object.fromEntries(
    Object.entries(entry.variables).map(([capture, kind]) => [capture, RASTER_VARIABLE_KINDS[kind]])
  );
}

/**
 * Builds the plain-data view pack of a raster-family entry: operators with variable kinds,
 * formulas, stages, capture stages, variable kinds by capture name and 3×3 stencil neighbourhoods.
 */
export function createRasterViewPack(
  packId: string,
  entries: Readonly<Record<string, RasterOperatorEntry>>,
  views: readonly ViewRegistration<unknown>[] = []
): ViewPackRegistration {
  const operators: Record<string, OperatorDescriptor> = {};
  const formulas: Record<string, GPUComputationFormulaRef | ViewPackNoFormula> = {};
  const variableKinds: Record<string, VariableKind> = {};
  for (const [name, entry] of Object.entries(entries)) {
    const kinds = getRasterVariableKinds(entry);
    operators[name] = {...entry.descriptor, variableKinds: kinds};
    formulas[name] = entry.formula;
    Object.assign(variableKinds, kinds);
  }
  return {
    id: packId,
    operators,
    formulas,
    analysisStages: RASTER_ANALYSIS_STAGES,
    variableKinds,
    neighbourhoods: {cells: {kind: 'stencil-3x3', keySpace: 'cells'}},
    metadata: {},
    views,
    guardsFor: ({element}) => {
      const entry = element ? entries[element] : undefined;
      return entry
        ? getRasterGuards(entry.guards).map(guard => withGuardMetadata(guard, entry.metadata))
        : [];
    }
  };
}
