// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {ProductBuilder, Stage, TerrainSession} from './b14a-session';

/** A built stage and the builder that owns its buffers. */
export type StageEntry = {key: string; stage: Stage; builder: ProductBuilder};

/**
 * Keeps the stages of a scene (compiled graph groups with dependencies between them). A stage is
 * rebuilt only when its own compile-time key, or the key of a stage it reads, changes; dependents
 * of a rebuilt stage are rebuilt with it, because their graphs import its buffers.
 */
export class StageRegistry<Id extends string, State> {
  private readonly entries = new Map<Id, StageEntry>();

  constructor(
    private readonly session: TerrainSession,
    private readonly dependencies: Record<Id, readonly Id[]>,
    private readonly getOwnKey: (id: Id, state: State) => string,
    private readonly build: (id: Id, state: State, registry: StageRegistry<Id, State>) => StageEntry
  ) {}

  /** Key of a stage including the keys of everything it depends on. */
  getFullKey(id: Id, state: State): string {
    const dependencies = this.dependencies[id].map(dependency =>
      this.getFullKey(dependency, state)
    );
    return `${this.getOwnKey(id, state)}<${dependencies.join(',')}>`;
  }

  /** The built stage, which must exist (its dependents are built after it). */
  require(id: Id): StageEntry {
    const entry = this.entries.get(id);
    if (!entry) throw new Error(`Stage ${id} is not built`);
    return entry;
  }

  /** The built stage if there is one. */
  get(id: Id): StageEntry | undefined {
    return this.entries.get(id);
  }

  /** Builds (or rebuilds) a stage and its dependencies when their keys changed. */
  ensure(id: Id, state: State): StageEntry {
    const key = this.getFullKey(id, state);
    const existing = this.entries.get(id);
    if (existing && existing.key === key) return existing;
    for (const dependency of this.dependencies[id]) this.ensure(dependency, state);
    const entry = this.build(id, state, this);
    entry.key = key;
    if (existing) {
      this.session.unregisterStage(existing.stage);
      existing.stage.release();
    }
    this.entries.set(id, entry);
    return entry;
  }

  /** Marks stages dirty (those that are built). */
  markDirty(ids: readonly Id[]): void {
    for (const id of ids) this.entries.get(id)?.stage.markDirty();
  }
}
