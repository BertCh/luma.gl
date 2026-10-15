// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Per-view and page-wide construction, visibility and memory budgets. */
export type LensBudget = {
  maxLiveVisible: number;
  maxConstructed: number;
  maxBytesPerView: number;
  maxBytesPerPage: number;
};

/** Starting judgement, to be checked once with an A/B, not benchmarked. */
export const DEFAULT_LENS_BUDGET: LensBudget = {
  maxLiveVisible: 4,
  maxConstructed: 8,
  maxBytesPerView: 16 * 2 ** 20,
  maxBytesPerPage: 96 * 2 ** 20
};

/** Budgets live lenses and serializes GPU work FIFO, suspending queued work while hidden. */
export class PageScheduler {
  private budget: LensBudget;
  private lenses = new Map<string, {bytes: number; visible: boolean; visibleSequence: number}>();
  private pauseListeners = new Set<(lensId: string) => void>();
  private queue: {lensId: string; execute: () => Promise<void>}[] = [];
  private running = false;
  private hidden = false;
  private sequence = 0;

  constructor(props: {budget?: Partial<LensBudget>} = {}) {
    this.budget = {...DEFAULT_LENS_BUDGET, ...props.budget};
  }

  /** Reserves a live lens, or updates its reservation atomically if every budget permits it. */
  requestLive(
    lensId: string,
    estimatedBytes: number
  ): {granted: true} | {granted: false; reason: string} {
    if (!Number.isFinite(estimatedBytes) || estimatedBytes < 0)
      return {granted: false, reason: 'Invalid byte estimate'};
    if (estimatedBytes > this.budget.maxBytesPerView)
      return {granted: false, reason: 'Per-view memory budget exceeded'};
    const existing = this.lenses.get(lensId);
    const usage = this.getUsage();
    if (!existing && usage.constructed >= this.budget.maxConstructed)
      return {granted: false, reason: 'Constructed lens budget exceeded'};
    if (usage.bytes - (existing?.bytes ?? 0) + estimatedBytes > this.budget.maxBytesPerPage)
      return {granted: false, reason: 'Page memory budget exceeded'};
    this.lenses.set(lensId, {
      ...(existing ?? {visible: false, visibleSequence: 0}),
      bytes: estimatedBytes
    });
    return {granted: true};
  }

  /** Releases construction and memory reservations; already submitted runs complete normally. */
  release(lensId: string): void {
    this.lenses.delete(lensId);
  }

  /** Marks visibility, pausing the least recently visible lenses when the live budget is full. */
  setVisible(lensId: string, visible: boolean): void {
    const lens = this.lenses.get(lensId);
    if (!lens || lens.visible === visible) return;
    lens.visible = visible;
    if (!visible) {
      this.emitPause(lensId);
      return;
    }
    lens.visibleSequence = ++this.sequence;
    const visibleLenses = [...this.lenses.entries()]
      .filter(([, entry]) => entry.visible)
      .sort(([, left], [, right]) => left.visibleSequence - right.visibleSequence);
    while (visibleLenses.length > this.budget.maxLiveVisible && visibleLenses.length) {
      const oldest = visibleLenses.shift()!;
      oldest[1].visible = false;
      this.emitPause(oldest[0]);
    }
    if (this.hidden && lens.visible) this.emitPause(lensId);
  }

  /** Enqueues one run. Rejections propagate to its caller without blocking later work. */
  enqueueRun<T>(lensId: string, run: () => Promise<T>): Promise<T> {
    return new Promise<T>((resolve, reject) => {
      this.queue.push({
        lensId,
        execute: async () => {
          try {
            resolve(await run());
          } catch (error) {
            reject(error);
          }
        }
      });
      void this.drain();
    });
  }

  /** Pauses visible lenses and queued runs; in-flight GPU work is not interrupted. */
  setPageHidden(hidden: boolean): void {
    if (this.hidden === hidden) return;
    this.hidden = hidden;
    if (hidden) {
      for (const [lensId, lens] of this.lenses) if (lens.visible) this.emitPause(lensId);
    } else void this.drain();
  }

  /** Subscribes to pause requests from visibility or page lifecycle changes. */
  onPause(listener: (lensId: string) => void): () => void {
    this.pauseListeners.add(listener);
    return () => {
      this.pauseListeners.delete(listener);
    };
  }

  /** Returns current reservations; hidden pages have no actively visible lenses. */
  getUsage(): {constructed: number; visible: number; bytes: number} {
    let visible = 0;
    let bytes = 0;
    for (const lens of this.lenses.values()) {
      bytes += lens.bytes;
      if (lens.visible && !this.hidden) visible++;
    }
    return {constructed: this.lenses.size, visible, bytes};
  }

  private emitPause(lensId: string): void {
    for (const listener of [...this.pauseListeners]) listener(lensId);
  }

  private async drain(): Promise<void> {
    if (this.running || this.hidden) return;
    this.running = true;
    try {
      while (!this.hidden && this.queue.length) await this.queue.shift()!.execute();
    } finally {
      this.running = false;
    }
  }
}

let pageScheduler: PageScheduler | undefined;

/** Returns the shared scheduler in this JavaScript realm. */
export function getPageScheduler(): PageScheduler {
  pageScheduler ??= new PageScheduler();
  return pageScheduler;
}
