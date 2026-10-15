// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {BUFFER_STRIP_VIEW} from '../views/buffer-strip';
import {COMMAND_GRAPH_VIEW} from '../views/command-graph-view';
import {COMPARE_VIEW} from '../views/compare-view';
import {DATUM_RECEIPT_VIEW} from '../views/datum-receipt';
import {DISPATCH_VIEW} from '../views/dispatch-view';
import {FORMULA_CARD_VIEW} from '../views/formula-card';
import {RESULT_VIEW} from '../views/result-view';
import {REALIZED_VIEW} from '../views/realized-view';
import {SCOPE_DIAGRAM_VIEW} from '../views/scope-diagram';
import {SKELETON_VIEW} from '../views/skeleton-view';
import {TABLE_VIEW} from '../views/table-view';
import {TIMING_VIEW} from '../views/timing-view';
import type {ViewPackRegistration, ViewRegistration} from './view-registration';
import {ViewRegistry} from './view-registry';

/** Generic views every default registry contains (the table view is added by the registry itself). */
const BUILT_IN_VIEWS: readonly ViewRegistration<unknown>[] = [
  RESULT_VIEW,
  FORMULA_CARD_VIEW,
  DATUM_RECEIPT_VIEW,
  BUFFER_STRIP_VIEW,
  DISPATCH_VIEW,
  REALIZED_VIEW,
  SKELETON_VIEW,
  COMMAND_GRAPH_VIEW,
  TIMING_VIEW,
  COMPARE_VIEW,
  SCOPE_DIAGRAM_VIEW,
  TABLE_VIEW
];

/** Domain packs registered for every default registry, in registration order. */
const DEFAULT_VIEW_PACKS = new Map<string, ViewPackRegistration>();

/**
 * Registers a domain pack (plain data plus optional views) into every registry created by
 * {@link createDefaultViewRegistry} from now on. Pack modules call this once at import, so a
 * static-twin build that imports the pack sees its views. Registering the same id again replaces
 * the earlier registration (hot reload); registries already created are unaffected.
 */
export function registerDefaultViewPack(pack: ViewPackRegistration): void {
  DEFAULT_VIEW_PACKS.set(pack.id, pack);
}

/** Packs that {@link createDefaultViewRegistry} will include. */
export function getDefaultViewPacks(): ViewPackRegistration[] {
  return [...DEFAULT_VIEW_PACKS.values()];
}

/** Creates a registry with the generic built-in views and every registered default pack. */
export function createDefaultViewRegistry(): ViewRegistry {
  const registry = new ViewRegistry();
  for (const view of BUILT_IN_VIEWS) {
    if (view.id !== registry.getTableView().id) {
      registry.register(view);
    }
  }
  for (const pack of DEFAULT_VIEW_PACKS.values()) {
    registry.registerPack(pack);
  }
  return registry;
}
