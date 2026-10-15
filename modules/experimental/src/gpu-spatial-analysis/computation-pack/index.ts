// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUComputationFormulaRef} from '@luma.gl/gpgpu';

import {withGuardMetadata, type GuardSpec} from '../../computation-views/guards/guards';
import {registerDefaultViewPack} from '../../computation-views/registry/default-registry';
import type {
  ViewPackData,
  ViewPackNoFormula,
  ViewPackRegistration,
  ViewRegistration
} from '../../computation-views/registry/view-registration';
import type {OperatorDescriptor, VariableKind} from '../../computation-views/types';
import {getSpatialGuards} from './guards';
import {ACCESSIBILITY_OPERATORS} from './operators/accessibility';
import {CELLS_OPERATORS} from './operators/cells';
import {DENSITY_OPERATORS} from './operators/density';
import {DYNAMICS_OPERATORS} from './operators/dynamics';
import {GEOMETRY_OPERATORS} from './operators/geometry';
import {JOINS_OPERATORS} from './operators/joins';
import {LINES_OPERATORS} from './operators/lines';
import {RECIPES_OPERATORS} from './operators/recipes';
import {REGRESSION_OPERATORS} from './operators/regression';
import {STATISTICS_OPERATORS} from './operators/statistics';
import {VALIDITY_OPERATORS} from './operators/validity';
import {WEIGHTS_OPERATORS} from './operators/weights';
import {SPATIAL_ANALYSIS_STAGES} from './stages';
import type {SpatialFamilyOperators, SpatialOperatorEntry} from './types';
import {SPATIAL_VARIABLE_KINDS} from './variable-kinds';

import {NEIGHBOURHOOD_VIEW, MORAN_SCATTER_VIEW, FDR_STAIRCASE_VIEW} from './views/index';

export * from './views/index';
export * from './choices';
export * from './guards';
export * from './stages';
export * from './types';
export * from './variable-kinds';

/** Pack id; also the lens scenario entry. */
export const SPATIAL_COMPUTATION_PACK_ID = 'gpu-spatial-analysis';

/** Every family table, in docs page order. Each family file is owned by one stream. */
export const SPATIAL_FAMILY_OPERATORS: readonly SpatialFamilyOperators[] = [
  WEIGHTS_OPERATORS,
  STATISTICS_OPERATORS,
  REGRESSION_OPERATORS,
  DYNAMICS_OPERATORS,
  DENSITY_OPERATORS,
  ACCESSIBILITY_OPERATORS,
  JOINS_OPERATORS,
  VALIDITY_OPERATORS,
  CELLS_OPERATORS,
  GEOMETRY_OPERATORS,
  LINES_OPERATORS,
  RECIPES_OPERATORS
];

/** Domain views of the spatial pack (NeighbourhoodView, MoranScatter, FDR staircase). */
const SPATIAL_VIEWS: ViewRegistration<unknown>[] = [
  NEIGHBOURHOOD_VIEW,
  MORAN_SCATTER_VIEW,
  FDR_STAIRCASE_VIEW
];

/** All operator entries keyed by exported name. Throws on a name declared by two families. */
export function getSpatialOperatorEntries(): Readonly<Record<string, SpatialOperatorEntry>> {
  const entries: Record<string, SpatialOperatorEntry> = {};
  for (const table of SPATIAL_FAMILY_OPERATORS) {
    for (const [name, entry] of Object.entries(table)) {
      if (entries[name]) {
        throw new Error(`spatial pack: ${name} declared twice`);
      }
      entries[name] = entry;
    }
  }
  return entries;
}

/** Looks up one operator entry by exported name. */
export function getSpatialOperatorEntry(element: string): SpatialOperatorEntry | undefined {
  return getSpatialOperatorEntries()[element];
}

/** Resolves an entry's named variable kinds. */
export function getSpatialVariableKinds(
  entry: SpatialOperatorEntry
): Readonly<Record<string, VariableKind>> {
  return Object.fromEntries(
    Object.entries(entry.variables).map(([capture, kind]) => [
      capture,
      SPATIAL_VARIABLE_KINDS[kind]
    ])
  );
}

/** Guards that apply to one element (empty when the element is unknown). */
export function getSpatialElementGuards(element: string): GuardSpec[] {
  const entry = getSpatialOperatorEntry(element);
  return entry ? getSpatialGuards(entry.guards) : [];
}

/**
 * Pack data scoped to one element and run: that operator's variable kinds and its fixed metadata
 * merged with the run's metadata (CRS model, weights standardisation, correction method...).
 * Lenses pass this together with {@link getSpatialElementGuards} so verdicts are not *unknown*.
 */
export function createSpatialElementPackData(
  element: string,
  runMetadata: Readonly<Record<string, string | number | boolean>> = {}
): ViewPackData {
  const entry = getSpatialOperatorEntry(element);
  return {
    ...createSpatialPackData(),
    variableKinds: entry ? getSpatialVariableKinds(entry) : {},
    metadata: {...entry?.metadata, ...runMetadata}
  };
}

/** The pack-wide plain data: operators, formulas and analysis stages. */
export function createSpatialPackData(): ViewPackData {
  const operators: Record<string, OperatorDescriptor> = {};
  const formulas: Record<string, GPUComputationFormulaRef | ViewPackNoFormula> = {};
  for (const [name, entry] of Object.entries(getSpatialOperatorEntries())) {
    operators[name] = {...entry.descriptor, variableKinds: getSpatialVariableKinds(entry)};
    formulas[name] = entry.formula;
  }
  return {
    id: SPATIAL_COMPUTATION_PACK_ID,
    operators,
    formulas,
    analysisStages: SPATIAL_ANALYSIS_STAGES
  };
}

/** The spatial pack registration (plain data plus domain views). */
export function createSpatialComputationPack(): ViewPackRegistration {
  return {
    ...createSpatialPackData(),
    views: SPATIAL_VIEWS,
    // The element's guards; its fixed metadata is the fallback under record choices.
    guardsFor: ({element}) => {
      const entry = element ? getSpatialOperatorEntry(element) : undefined;
      return entry
        ? getSpatialGuards(entry.guards).map(guard => withGuardMetadata(guard, entry.metadata))
        : [];
    }
  };
}

registerDefaultViewPack(createSpatialComputationPack());
