// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';
import {LENS_RUNG_IDS} from '../types';
import type {LensRungId} from '../types';
import type {GuardSpec} from '../guards/guards';
import type {
  LensRecord,
  ViewContext,
  ViewRegistration,
  ViewPackData,
  ViewPackGuardTarget,
  ViewPackRegistration
} from './view-registration';
import {TABLE_VIEW} from '../views/table-view';

/** A satisfied rung or an honest explanation with a table fallback. */
export type RungResolution =
  | {
      status: 'enabled';
      rung: LensRungId;
      view: ViewRegistration;
      alternatives: readonly ViewRegistration[];
    }
  | {status: 'disabled'; rung: LensRungId; reason: string; fallback: ViewRegistration};

/** Ordered, record-driven view and domain-pack registry. */
export class ViewRegistry {
  private readonly views = new Map<string, ViewRegistration>();
  private readonly packs = new Map<string, ViewPackData>();
  private readonly packGuards = new Map<string, NonNullable<ViewPackRegistration['guardsFor']>>();
  private readonly tableView: ViewRegistration;

  constructor(props: {tableView?: ViewRegistration} = {}) {
    this.tableView = props.tableView ?? TABLE_VIEW;
    this.register(this.tableView);
  }

  /** Registers a unique view id. */
  register<Model>(view: ViewRegistration<Model>): void {
    // View ids must be unique within a registry.
    assert(!this.views.has(view.id));
    this.views.set(view.id, view);
  }

  /** Registers domain metadata and its optional views. */
  registerPack(pack: ViewPackRegistration): void {
    // Pack ids must be unique within a registry.
    assert(!this.packs.has(pack.id));
    const {views = [], guardsFor, ...data} = pack;
    const identifiers = new Set(this.views.keys());
    for (const view of views) {
      // Validate the whole pack before changing the registry.
      assert(!identifiers.has(view.id));
      identifiers.add(view.id);
    }
    for (const view of views) this.register(view);
    this.packs.set(pack.id, data);
    if (guardsFor) this.packGuards.set(pack.id, guardsFor);
  }

  /**
   * Guards the record's domain packs apply to this element, in domain order, first id wins.
   * Without an element a pack may still return record-wide guards, or none.
   */
  getGuardsFor(target: ViewPackGuardTarget): GuardSpec[] {
    const guards = new Map<string, GuardSpec>();
    for (const id of new Set(target.record.domains)) {
      for (const guard of this.packGuards.get(id)?.(target) ?? []) {
        if (!guards.has(guard.id)) guards.set(guard.id, guard);
      }
    }
    return [...guards.values()];
  }

  /** Looks up domain metadata. */
  getPack(id: string): ViewPackData | undefined {
    return this.packs.get(id);
  }

  /** Returns packs declared by this record, in domain order. */
  getPacksFor(record: LensRecord): ViewPackData[] {
    return [...new Set(record.domains)].flatMap(id =>
      this.packs.has(id) ? [this.packs.get(id)!] : []
    );
  }

  /** Looks up a view by id. */
  get(viewId: string): ViewRegistration | undefined {
    return this.views.get(viewId);
  }

  /** Lists registered views in registration order. */
  list(): ViewRegistration[] {
    return [...this.views.values()];
  }

  /** Lists candidates in priority order, with a stable id tie-break. */
  viewsForRung(rung: LensRungId): ViewRegistration[] {
    return this.list()
      .filter(view => view.rung === rung)
      .sort(
        (first, second) =>
          first.order - second.order || (first.id < second.id ? -1 : first.id > second.id ? 1 : 0)
      );
  }

  /** Resolves only active, satisfied views; preferences never bypass satisfaction. */
  resolveRung(context: ViewContext, rung: LensRungId, preferredViewId?: string): RungResolution {
    const candidates = this.viewsForRung(rung).filter(view => view.isActive?.(context) ?? true);
    const enabled: ViewRegistration[] = [];
    let reason: string | undefined;
    for (const view of candidates) {
      const satisfaction = view.satisfies(context);
      if (!('reason' in satisfaction)) enabled.push(view);
      else reason ??= satisfaction.reason;
    }
    if (!enabled.length)
      return {
        status: 'disabled',
        rung,
        reason: reason ?? `no view registered for rung ${rung}`,
        fallback: this.tableView
      };
    const view = enabled.find(candidate => candidate.id === preferredViewId) ?? enabled[0];
    return {
      status: 'enabled',
      rung,
      view,
      alternatives: enabled.filter(candidate => candidate !== view)
    };
  }

  /** Status of every rung for the ladder rail. */
  listRungs(
    context: ViewContext
  ): Array<{rung: LensRungId; status: 'enabled' | 'disabled'; reason?: string; viewId?: string}> {
    return LENS_RUNG_IDS.map(rung => {
      const resolution = this.resolveRung(context, rung);
      return resolution.status === 'enabled'
        ? {rung, status: resolution.status, viewId: resolution.view.id}
        : {rung, status: resolution.status, reason: resolution.reason};
    });
  }

  /** Gets the complete inspector available to every record. */
  getTableView(): ViewRegistration {
    return this.tableView;
  }
}

/** Creates a pure view context with host-independent defaults. */
export function createViewContext(
  record: LensRecord,
  partial: Partial<Omit<ViewContext, 'record'>> = {}
): ViewContext {
  return {
    ...partial,
    record,
    bindings: partial.bindings ?? {},
    ground: partial.ground ?? 'light',
    idPrefix: partial.idPrefix ?? 'lens',
    reducedMotion: partial.reducedMotion ?? false
  };
}
