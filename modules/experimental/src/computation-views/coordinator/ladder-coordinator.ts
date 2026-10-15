// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';
import type {
  CoordinatorWriter,
  KeySpaceId,
  LinkHeat,
  SelectionClause,
  SelectionResolution,
  ViewCoupling
} from '../types';
import {KeyRelationGraph, type KeyMappingResult, type KeyRelation} from './key-relations';
import {CPUSelectionBackend, type SelectionBackend} from './selection-backend';

/** Single-valued navigation variables stored independently of selection sets. */
export type CoordinatorParams = Record<string, string | number | boolean>;

/** Synchronous notifications; writer switches precede their changes. */
export type CoordinatorEvent =
  | {
      type: 'param';
      name: string;
      value: unknown;
      writer: CoordinatorWriter;
      origin?: string;
      heat: LinkHeat;
    }
  | {
      type: 'selection';
      selectionId: string;
      writer: CoordinatorWriter;
      origin?: string;
      heat: LinkHeat;
    }
  | {type: 'writer'; writer: CoordinatorWriter}
  | {type: 'coupling'; viewId: string; coupling: ViewCoupling}
  | {type: 'record'; runId: string};

/** A labelled semantic action, excluding transient previews. */
export type CoordinatorAction = {
  sequence: number;
  label: string;
  writer: CoordinatorWriter;
  kind: 'param' | 'clause' | 'clear' | 'record';
  time: number;
};

/** Ownership and attribution of one change; ifWriter guards delayed gestures. */
export type CoordinatorChangeOptions = {
  writer?: CoordinatorWriter;
  ifWriter?: CoordinatorWriter;
  origin?: string;
  heat?: LinkHeat;
  label?: string;
};

/** An empty selection is distinct from an active selection of zero keys. */
export type SelectionResolutionResult = KeyMappingResult | {status: 'empty'};

/** Mark coverage and an explicit count of selected keys that this view cannot show. */
export type HighlightResult =
  | {status: 'ok'; marks: Map<number, {covered: number; total: number}>; offView: {count: number}}
  | Exclude<SelectionResolutionResult, {status: 'ok'}>;

/** Synchronous CPU coordinator. GPU masks are a separate, deferred backend contract. */
export class LadderCoordinator {
  readonly id: string;
  private relations: KeyRelationGraph;
  private backends: SelectionBackend[];
  private now: () => number;
  private decoupleThresholdMs: number;
  private params = new Map<string, unknown>();
  private selections = new Map<
    string,
    {resolution: SelectionResolution; clauses: SelectionClause[]}
  >();
  private keySpaces = new Map<
    KeySpaceId,
    {keyCount: number; fieldValues?: (field: string) => ArrayLike<number> | undefined}
  >();
  private writer: CoordinatorWriter = 'narrative';
  private listeners = new Set<{listener: (event: CoordinatorEvent) => void; viewId?: string}>();
  private couplings = new Map<string, ViewCoupling>();
  private staleViews = new Set<string>();
  private previews = new Set<string>();
  private actions: CoordinatorAction[] = [];
  private sequence = 0;
  private destroyed = false;

  constructor(
    props: {
      id?: string;
      relations?: KeyRelationGraph;
      backends?: SelectionBackend[];
      now?: () => number;
      decoupleThresholdMs?: number;
    } = {}
  ) {
    this.id = props.id ?? 'ladder';
    this.relations = props.relations ?? new KeyRelationGraph();
    this.backends = props.backends ?? [new CPUSelectionBackend()];
    this.now = props.now ?? Date.now;
    this.decoupleThresholdMs = props.decoupleThresholdMs ?? 100;
    this.defineSelection('hover', 'single');
    this.defineSelection('selection', 'single');
    this.defineSelection('filter', 'intersect');
  }

  /**
   * Adds declared key-space relations (from a record) to the page's relation graph, so clauses in
   * one key space link into views of another. Relations with an existing id are replaced.
   */
  addRelations(relations: Iterable<KeyRelation>): void {
    this.assertAvailable();
    for (const relation of relations) this.relations.add(relation);
  }

  /** Removes relations by id, for example when a lens's record is replaced by a re-run. */
  removeRelations(relationIds: Iterable<string>): void {
    this.assertAvailable();
    for (const relationId of relationIds) this.relations.remove(relationId);
  }

  /** Supplies source-domain metadata for all/range clauses, including cross-space resolution. */
  registerKeySpace(
    keySpace: KeySpaceId,
    keyCount: number,
    fieldValues?: (field: string) => ArrayLike<number> | undefined
  ): void {
    this.assertAvailable();
    this.keySpaces.set(keySpace, {keyCount, fieldValues});
  }

  /** Reads a navigation variable. */
  getParam(name: string): unknown {
    return this.params.get(name);
  }

  /** Writes a navigation variable, returning false if a gesture no longer owns the store. */
  setParam(name: string, value: unknown, options: CoordinatorChangeOptions = {}): boolean {
    if (!this.acceptWriter(options)) return false;
    this.previews.delete(name);
    this.params.set(name, value);
    this.recordAction('param', options.label ?? `Set ${name}`);
    this.emitChange({
      type: 'param',
      name,
      value,
      writer: this.writer,
      origin: options.origin,
      heat: options.heat ?? 'warm'
    });
    return true;
  }

  /** Defines combination semantics, preserving existing source clauses. */
  defineSelection(selectionId: string, resolution: SelectionResolution): void {
    this.assertAvailable();
    this.selections.set(selectionId, {
      resolution,
      clauses: this.selections.get(selectionId)?.clauses ?? []
    });
  }

  /** Replaces a source's clause and makes it the most recent for single resolution. */
  addClause(
    selectionId: string,
    clause: SelectionClause,
    options: CoordinatorChangeOptions = {}
  ): boolean {
    if (!this.acceptWriter(options)) return false;
    const selection = this.getSelection(selectionId);
    selection.clauses = [
      ...selection.clauses.filter(previous => previous.source !== clause.source),
      cloneClause(clause)
    ];
    this.recordAction('clause', options.label ?? clause.label);
    this.emitChange({
      type: 'selection',
      selectionId,
      writer: this.writer,
      origin: options.origin,
      heat: options.heat ?? 'hot'
    });
    return true;
  }

  /** Clears one source, or every clause of this selection. */
  clearClauses(
    selectionId: string,
    source?: string,
    options: CoordinatorChangeOptions = {}
  ): boolean {
    if (!this.acceptWriter(options)) return false;
    const selection = this.getSelection(selectionId);
    selection.clauses =
      source === undefined ? [] : selection.clauses.filter(clause => clause.source !== source);
    this.recordAction(
      'clear',
      options.label ?? `Clear ${selectionId}${source === undefined ? '' : ` from ${source}`}`
    );
    this.emitChange({
      type: 'selection',
      selectionId,
      writer: this.writer,
      origin: options.origin,
      heat: options.heat ?? 'hot'
    });
    return true;
  }

  /** Returns a detached snapshot so clauses cannot change without an event. */
  getClauses(selectionId: string): readonly SelectionClause[] {
    return this.getSelection(selectionId).clauses.map(cloneClause);
  }

  /** Resolves applicable clauses, excluding a view's own source before composition. */
  resolve(
    selectionId: string,
    keySpace: KeySpaceId,
    options: {viewId?: string; keyCount: number}
  ): SelectionResolutionResult {
    this.assertAvailable();
    const selection = this.getSelection(selectionId);
    let clauses = selection.clauses.filter(
      clause =>
        clause.source !== options.viewId &&
        (!clause.clients || options.viewId === undefined || clause.clients.includes(options.viewId))
    );
    if (selection.resolution === 'single') clauses = clauses.slice(-1);
    if (!clauses.length) return {status: 'empty'};
    let result: Set<number> | undefined;
    for (const clause of clauses) {
      const metadata = this.keySpaces.get(clause.keySpace);
      const keyCount =
        metadata?.keyCount ?? (clause.keySpace === keySpace ? options.keyCount : undefined);
      let keys: Set<number> | 'unknown';
      if (keyCount === undefined && clause.predicate.kind === 'keys') {
        // Explicit keys need no guessed source-domain extent; only small sets are CPU-owned.
        const backend = this.backends.find(candidate =>
          candidate.accepts(
            clause.keySpace,
            clause.predicate.kind === 'keys' ? clause.predicate.keys.length : 0
          )
        );
        keys = backend instanceof CPUSelectionBackend ? new Set(clause.predicate.keys) : 'unknown';
      } else {
        const backend =
          keyCount === undefined
            ? undefined
            : this.backends.find(candidate => candidate.accepts(clause.keySpace, keyCount));
        keys =
          backend instanceof CPUSelectionBackend && keyCount !== undefined
            ? backend.evaluate(clause, keyCount, metadata?.fieldValues)
            : 'unknown';
      }
      if (keys === 'unknown')
        return {status: 'unknown', reason: `CPU selection data unavailable for ${clause.keySpace}`};
      const mapped = this.relations.mapKeys(keys, clause.keySpace, keySpace);
      if (mapped.status !== 'ok') return mapped;
      if (!result) result = mapped.keys;
      else if (selection.resolution === 'union') for (const key of mapped.keys) result.add(key);
      else result = new Set([...result].filter(key => mapped.keys.has(key)));
    }
    return {status: 'ok', keys: result ?? new Set()};
  }

  /** Highlights the image of selected keys, retaining partial coverage and off-view counts. */
  highlight(
    selectionId: string,
    view: {
      viewId: string;
      keySpace: KeySpaceId;
      keyCount: number;
      markRelation?: KeyRelation;
      visibleKeys?: ReadonlySet<number> | readonly number[];
    }
  ): HighlightResult {
    const resolved = this.resolve(selectionId, view.keySpace, view);
    if (resolved.status !== 'ok') return resolved;
    const relation = view.markRelation;
    if (relation && relation.from !== view.keySpace) return {status: 'unlinked'};
    if (relation && !relation.available)
      return {
        status: 'unknown',
        reason: relation.reason ?? `Relation ${relation.id} was not captured`
      };
    const visibleKeys = view.visibleKeys === undefined ? undefined : new Set(view.visibleKeys);
    const marks = new Map<number, {covered: number; total: number}>();
    let offViewCount = 0;
    for (const key of resolved.keys) {
      if (visibleKeys && !visibleKeys.has(key)) {
        offViewCount++;
        continue;
      }
      const image = relation
        ? relation.forward([key])
        : new Set(key >= 0 && key < view.keyCount ? [key] : []);
      if (!image.size) offViewCount++;
      for (const mark of image) {
        const coverage = marks.get(mark) ?? {
          covered: 0,
          total: relation ? (relation.preimageSize?.(mark) ?? relation.inverse([mark]).size) : 1
        };
        coverage.covered++;
        marks.set(mark, coverage);
      }
    }
    return {status: 'ok', marks, offView: {count: offViewCount}};
  }

  /** Adds the preimage of picked marks as a key clause owned by the view. */
  pick(
    selectionId: string,
    view: {viewId: string; keySpace: KeySpaceId; markRelation?: KeyRelation},
    markKeys: readonly number[],
    options: CoordinatorChangeOptions & {additive?: boolean} = {}
  ): SelectionClause {
    const relation = view.markRelation;
    // A pick cannot claim keys from an uncaptured relation or a different domain.
    assert(!relation || (relation.available && relation.from === view.keySpace));
    const keys = relation ? relation.inverse(markKeys) : new Set(markKeys);
    if (options.additive) {
      const previous = this.getClauses(selectionId).find(
        clause => clause.source === view.viewId && clause.keySpace === view.keySpace
      );
      if (previous?.predicate.kind === 'keys')
        for (const key of previous.predicate.keys) keys.add(key);
    }
    const clause: SelectionClause = {
      source: view.viewId,
      keySpace: view.keySpace,
      predicate: {kind: 'keys', keys: [...keys]},
      label: options.label ?? `Pick ${markKeys.join(', ')}`
    };
    this.addClause(selectionId, clause, {...options, origin: options.origin ?? view.viewId});
    return clause;
  }

  /** Reads the current gesture owner. */
  getWriter(): CoordinatorWriter {
    return this.writer;
  }

  /** Switches gesture owner and emits the switch before any subsequent change. */
  setWriter(writer: CoordinatorWriter): void {
    this.assertAvailable();
    if (this.writer === writer) return;
    this.writer = writer;
    this.emit({type: 'writer', writer});
  }

  /** Subscribes without echoing changes back to their originating view. */
  subscribe(
    listener: (event: CoordinatorEvent) => void,
    options: {viewId?: string} = {}
  ): () => void {
    this.assertAvailable();
    const subscription = {listener, viewId: options.viewId};
    this.listeners.add(subscription);
    return () => {
      this.listeners.delete(subscription);
    };
  }

  /** Marks dependent views stale until a new record arrives. */
  markStale(viewIds: readonly string[]): void {
    this.assertAvailable();
    for (const viewId of viewIds) {
      this.staleViews.add(viewId);
      if (this.getCoupling(viewId) !== 'decoupled') this.setCoupling(viewId, 'stale');
    }
  }

  /** Installs the new record and clears stale state, retaining explicit decoupling. */
  setRecord(runId: string): void {
    this.assertAvailable();
    this.staleViews.clear();
    for (const [viewId, coupling] of this.couplings)
      if (coupling === 'stale') this.setCoupling(viewId, 'linked');
    this.recordAction('record', `Record ${runId}`);
    this.emit({type: 'record', runId});
  }

  /** Decouples slow hot updates; fast reports do not silently relink a decoupled view. */
  reportUpdate(viewId: string, durationMs: number): ViewCoupling {
    this.assertAvailable();
    if (durationMs > this.decoupleThresholdMs) this.setCoupling(viewId, 'decoupled');
    else if (!this.couplings.has(viewId)) this.couplings.set(viewId, 'linked');
    return this.getCoupling(viewId);
  }

  /** Reads a view's coupling status. */
  getCoupling(viewId: string): ViewCoupling {
    return this.couplings.get(viewId) ?? 'linked';
  }

  /** Relinks after an explicit update request; stale evidence still requires a new record. */
  requestUpdate(viewId: string): void {
    this.assertAvailable();
    this.setCoupling(viewId, this.staleViews.has(viewId) ? 'stale' : 'linked');
  }

  /** Publishes a hot preview without adding an action-log entry. */
  previewParam(name: string, value: unknown, options: CoordinatorChangeOptions = {}): boolean {
    if (!this.acceptWriter(options)) return false;
    this.params.set(name, value);
    this.previews.add(name);
    this.emitChange({
      type: 'param',
      name,
      value,
      writer: this.writer,
      origin: options.origin,
      heat: 'hot'
    });
    return true;
  }

  /** Commits the most recent preview as one warm semantic action. */
  commitParam(name: string, options: CoordinatorChangeOptions = {}): boolean {
    if (!this.previews.has(name)) return false;
    return this.setParam(name, this.params.get(name), {...options, heat: 'warm'});
  }

  /** Returns a detached snapshot of the last 500 semantic actions. */
  getActionLog(): readonly CoordinatorAction[] {
    return this.actions.map(action => ({...action}));
  }

  /** Releases subscriptions and CPU state. */
  destroy(): void {
    this.listeners.clear();
    this.params.clear();
    this.selections.clear();
    this.keySpaces.clear();
    this.couplings.clear();
    this.staleViews.clear();
    this.previews.clear();
    this.actions = [];
    this.destroyed = true;
  }

  private assertAvailable(): void {
    // Destroyed coordinators cannot participate in later gestures.
    assert(!this.destroyed);
  }

  private getSelection(selectionId: string) {
    const selection = this.selections.get(selectionId);
    // Custom selections must declare their resolution before use.
    assert(selection);
    return selection;
  }

  private acceptWriter(options: CoordinatorChangeOptions): boolean {
    this.assertAvailable();
    if (options.ifWriter !== undefined && options.ifWriter !== this.writer) return false;
    if (options.writer !== undefined) this.setWriter(options.writer);
    return true;
  }

  private recordAction(kind: CoordinatorAction['kind'], label: string): void {
    this.actions.push({
      sequence: ++this.sequence,
      kind,
      label,
      writer: this.writer,
      time: this.now()
    });
    if (this.actions.length > 500) this.actions.shift();
  }

  private emitChange(event: Extract<CoordinatorEvent, {type: 'param' | 'selection'}>): void {
    if (event.heat === 'warm') {
      const dependentViews = new Set(this.couplings.keys());
      for (const {viewId} of this.listeners) if (viewId !== undefined) dependentViews.add(viewId);
      if (event.origin !== undefined) dependentViews.delete(event.origin);
      this.markStale([...dependentViews]);
    }
    this.emit(event);
  }

  private setCoupling(viewId: string, coupling: ViewCoupling): void {
    if (this.getCoupling(viewId) === coupling) return;
    this.couplings.set(viewId, coupling);
    this.emit({type: 'coupling', viewId, coupling});
  }

  private emit(event: CoordinatorEvent): void {
    for (const {listener, viewId} of [...this.listeners]) {
      if ('origin' in event && event.origin !== undefined && event.origin === viewId) continue;
      listener(event);
    }
  }
}

const pageCoordinators = new Map<string, LadderCoordinator>();

/** Shares one coordinator per bind name in this JavaScript realm. */
export function getPageCoordinator(bindName: string): LadderCoordinator {
  let coordinator = pageCoordinators.get(bindName);
  if (!coordinator) {
    coordinator = new LadderCoordinator({id: bindName});
    pageCoordinators.set(bindName, coordinator);
  }
  return coordinator;
}

/** Destroys shared coordinators and resets the realm cache for tests. */
export function resetPageCoordinators(): void {
  for (const coordinator of pageCoordinators.values()) coordinator.destroy();
  pageCoordinators.clear();
}

function cloneClause(clause: SelectionClause): SelectionClause {
  const predicate = clause.predicate;
  return {
    ...clause,
    clients: clause.clients && [...clause.clients],
    predicate:
      predicate.kind === 'keys'
        ? {...predicate, keys: [...predicate.keys]}
        : predicate.kind === 'range'
          ? {...predicate, range: [...predicate.range]}
          : {...predicate}
  };
}
