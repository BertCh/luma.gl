// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test} from 'vitest';
import {
  PageScheduler,
  getPageScheduler,
  DEFAULT_LENS_BUDGET
} from '../../../src/computation-views/hosts/page-scheduler';

function deferred() {
  let resolve!: () => void;
  const promise = new Promise<void>(complete => {
    resolve = complete;
  });
  return {promise, resolve};
}

describe('PageScheduler', () => {
  test('enforces construction and byte budgets atomically and releases reservations', () => {
    const scheduler = new PageScheduler({
      budget: {maxConstructed: 2, maxBytesPerView: 10, maxBytesPerPage: 15}
    });
    expect(scheduler.requestLive('a', 10)).toEqual({granted: true});
    expect(scheduler.requestLive('oversize', 11).granted).toBe(false);
    expect(scheduler.requestLive('b', 6).granted).toBe(false);
    expect(scheduler.requestLive('b', 5).granted).toBe(true);
    expect(scheduler.requestLive('c', 0).granted).toBe(false);
    expect(scheduler.requestLive('b', 6).granted).toBe(false);
    expect(scheduler.getUsage()).toEqual({constructed: 2, visible: 0, bytes: 15});
    expect(scheduler.requestLive('a', 8).granted).toBe(true);
    expect(scheduler.getUsage().bytes).toBe(13);
    scheduler.release('a');
    expect(scheduler.requestLive('c', 1).granted).toBe(true);
    expect(scheduler.requestLive('c', NaN).granted).toBe(false);
    expect(scheduler.requestLive('c', -1).granted).toBe(false);
    expect(DEFAULT_LENS_BUDGET.maxLiveVisible).toBe(4);
    expect(getPageScheduler()).toBe(getPageScheduler());
  });
  test('pauses the least recently visible lens and notifies hidden-page changes', () => {
    const scheduler = new PageScheduler({budget: {maxLiveVisible: 2}});
    const paused: string[] = [];
    const unsubscribe = scheduler.onPause(lensId => paused.push(lensId));
    for (const lensId of ['a', 'b', 'c']) {
      scheduler.requestLive(lensId, 1);
      scheduler.setVisible(lensId, true);
    }
    expect(paused).toEqual(['a']);
    expect(scheduler.getUsage().visible).toBe(2);
    scheduler.setVisible('a', true);
    expect(paused).toEqual(['a', 'b']);
    scheduler.setPageHidden(true);
    expect(paused).toEqual(['a', 'b', 'a', 'c']);
    expect(scheduler.getUsage().visible).toBe(0);
    scheduler.setPageHidden(false);
    expect(scheduler.getUsage().visible).toBe(2);
    unsubscribe();
    scheduler.setVisible('c', false);
    expect(paused).toHaveLength(4);
  });
  test('serializes FIFO without overlap and advances after rejection or synchronous throw', async () => {
    const scheduler = new PageScheduler();
    const gate = deferred();
    const order: string[] = [];
    const first = scheduler.enqueueRun('a', async () => {
      order.push('a-start');
      await gate.promise;
      order.push('a-end');
      return 1;
    });
    const second = scheduler.enqueueRun('b', async () => {
      order.push('b');
      throw Error('expected failure');
    });
    const rejected = expect(second).rejects.toThrow('expected failure');
    const third = scheduler.enqueueRun('c', () => {
      order.push('c');
      throw Error('sync failure');
    });
    const syncRejected = expect(third).rejects.toThrow('sync failure');
    const fourth = scheduler.enqueueRun('d', async () => {
      order.push('d');
      return 4;
    });
    expect(order).toEqual(['a-start']);
    gate.resolve();
    expect(await first).toBe(1);
    await rejected;
    await syncRejected;
    expect(await fourth).toBe(4);
    expect(order).toEqual(['a-start', 'a-end', 'b', 'c', 'd']);
  });
  test('holds queued runs while hidden, allowing only in-flight work to finish', async () => {
    const scheduler = new PageScheduler();
    const gate = deferred();
    const order: number[] = [];
    scheduler.setPageHidden(true);
    const first = scheduler.enqueueRun('a', async () => {
      order.push(1);
      await gate.promise;
    });
    const second = scheduler.enqueueRun('b', async () => {
      order.push(2);
    });
    await Promise.resolve();
    expect(order).toEqual([]);
    scheduler.setPageHidden(false);
    expect(order).toEqual([1]);
    scheduler.setPageHidden(true);
    gate.resolve();
    await first;
    await Promise.resolve();
    expect(order).toEqual([1]);
    scheduler.setPageHidden(false);
    await second;
    expect(order).toEqual([1, 2]);
  });
});
