// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {ProductBuilder, Stage, TerrainSession} from './b14a-session';

/** A built stage and the builder that owns its buffers. */
export type StageEntry = {key: string; stage: Stage; builder: ProductBuilder};

/** Lookup of the stages a stage build may read: those built earlier in a plan, then live ones. */
export type StageLookup<Id extends string> = {
  /** The stage, which must exist (its dependencies are built before it). */
  require(id: Id): StageEntry;
  /** The stage if there is one. */
  get(id: Id): StageEntry | undefined;
};

/** Stages built by {@link StageRegistry.plan} that are not part of the registry yet. */
export type StagePlan<Id extends string> = StageLookup<Id> & {
  /** Stages built for this plan, dependencies first. */
  readonly built: Map<Id, StageEntry>;
};

/**
 * Keeps the stages of a scene (compiled graph groups with dependencies between them). A stage is
 * rebuilt only when its own compile-time key, or the key of a stage it reads, changes; dependents
 * of a rebuilt stage are rebuilt with it, because their graphs import its buffers.
 *
 * Rebuilding is asynchronous: {@link StageRegistry.plan} builds and compiles the changed stages
 * without touching the live ones, then {@link StageRegistry.commit} swaps them in synchronously
 * and releases the stages they replace. {@link StageRegistry.discard} frees a superseded plan.
 */
export class StageRegistry<Id extends string, State> {
  private readonly entries = new Map<Id, StageEntry>();

  constructor(
    private readonly session: TerrainSession,
    private readonly dependencies: Record<Id, readonly Id[]>,
    private readonly getOwnKey: (id: Id, state: State) => string,
    private readonly build: (id: Id, state: State, lookup: StageLookup<Id>) => Promise<StageEntry>
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

  /**
   * Builds and compiles a stage and its dependencies when their keys changed. Nothing is
   * registered: the live stages keep running until {@link StageRegistry.commit}.
   */
  async plan(id: Id, state: State): Promise<StagePlan<Id>> {
    const built = new Map<Id, StageEntry>();
    const plan: StagePlan<Id> = {
      built,
      get: stage => built.get(stage) ?? this.entries.get(stage),
      require: stage => {
        const entry = plan.get(stage);
        if (!entry) throw new Error(`Stage ${stage} is not built`);
        return entry;
      }
    };
    try {
      await this.planStage(plan, id, state);
    } catch (error) {
      this.discard(plan);
      throw error;
    }
    return plan;
  }

  private async planStage(plan: StagePlan<Id>, id: Id, state: State): Promise<void> {
    const key = this.getFullKey(id, state);
    const existing = this.entries.get(id);
    if ((existing && existing.key === key) || plan.built.has(id)) return;
    for (const dependency of this.dependencies[id]) await this.planStage(plan, dependency, state);
    const entry = await this.build(id, state, plan);
    entry.key = key;
    plan.built.set(id, entry);
  }

  /** Swaps the stages of a plan in, then releases the stages they replace. Synchronous. */
  commit(plan: StagePlan<Id>): void {
    for (const [id, entry] of plan.built) {
      const existing = this.entries.get(id);
      this.entries.set(id, entry);
      if (existing) this.releaseEntry(existing);
    }
  }

  /** Frees the stages of a plan that was superseded before {@link StageRegistry.commit}. */
  discard(plan: StagePlan<Id>): void {
    for (const entry of plan.built.values()) this.releaseEntry(entry);
    plan.built.clear();
  }

  private releaseEntry(entry: StageEntry): void {
    this.session.unregisterStage(entry.stage);
    entry.stage.release();
  }

  /** Marks stages dirty (those that are built). */
  markDirty(ids: readonly Id[]): void {
    for (const id of ids) this.entries.get(id)?.stage.markDirty();
  }
}
