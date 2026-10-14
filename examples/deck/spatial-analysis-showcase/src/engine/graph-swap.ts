// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** One rebuild handed to {@link GraphSwap.request}. */
export type GraphSwapRequest<T> = {
  /**
   * Builds the next graphs and awaits `graph.compileAsync()`. Must not change state that
   * `encode`, layers or readbacks use: the current graphs keep running until `commit`.
   */
  build: () => Promise<T>;
  /** Swaps `next` in and only then releases the graphs it replaces. Runs synchronously. */
  commit: (next: T) => void;
  /**
   * Releases `next` when a later request or `destroy()` superseded it. Omit when `next` is kept
   * elsewhere, for example in a variant cache that `SpatialAnalysisResources` destroys.
   */
  discard?: (next: T) => void;
};

/**
 * Serializes asynchronous graph rebuilds so that only the latest request is swapped in.
 *
 * At most one `build` runs at a time. A request made while one is running replaces any queued
 * request; when the running build resolves it is discarded and the queued one starts. A scene
 * therefore never shows a half-built state: the old graphs encode until `commit` swaps the new
 * ones in, and the old graphs are released after the swap. Track the swap in the scene's
 * resources (`resources.track(new GraphSwap())`) so destroying the scene discards a pending
 * build instead of committing it.
 */
export class GraphSwap {
  private running = false;
  private queued: {request: GraphSwapRequest<unknown>; settle: Settle} | null = null;
  private destroyed = false;

  /** True while a build runs or waits. */
  get pending(): boolean {
    return this.running || this.queued !== null;
  }

  /**
   * Requests a rebuild.
   *
   * @returns `true` once `request` was committed, `false` when a later request or `destroy()`
   *   superseded it. Rejects with the error of `build` or `commit`; the current graphs stay.
   */
  request<T>(request: GraphSwapRequest<T>): Promise<boolean> {
    if (this.destroyed) return Promise.resolve(false);
    return new Promise<boolean>((resolve, reject) => {
      this.queued?.settle.resolve(false);
      this.queued = {request: request as GraphSwapRequest<unknown>, settle: {resolve, reject}};
      if (!this.running) void this.runQueued();
    });
  }

  /** Discards pending requests; a running build is discarded when it resolves. Idempotent. */
  destroy(): void {
    this.destroyed = true;
    this.queued?.settle.resolve(false);
    this.queued = null;
  }

  private async runQueued(): Promise<void> {
    while (this.queued && !this.destroyed) {
      const {request, settle} = this.queued;
      this.queued = null;
      this.running = true;
      let next: unknown;
      try {
        next = await request.build();
      } catch (error) {
        this.running = false;
        settle.reject(error);
        continue;
      }
      this.running = false;
      if (this.queued || this.destroyed) {
        request.discard?.(next);
        settle.resolve(false);
        continue;
      }
      try {
        request.commit(next);
        settle.resolve(true);
      } catch (error) {
        settle.reject(error);
      }
    }
  }
}

type Settle = {resolve: (committed: boolean) => void; reject: (error: unknown) => void};
