// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test, vi} from 'vitest';
import {createDebouncedAnnouncer} from '../../../src/computation-views/hosts/live-region';

describe('debounced live announcements', () => {
  test('announces only the last scrub frame after 600 ms quiet, and suppresses repeats', () => {
    vi.useFakeTimers();
    try {
      const emitted: string[] = [];
      const announcer = createDebouncedAnnouncer(text => emitted.push(text));
      announcer.announce('Iteration 1');
      vi.advanceTimersByTime(500);
      announcer.announce('Iteration 2');
      vi.advanceTimersByTime(599);
      expect(emitted).toEqual([]);
      vi.advanceTimersByTime(1);
      expect(emitted).toEqual(['Iteration 2']);
      announcer.announce('Iteration 2');
      vi.runAllTimers();
      expect(emitted).toEqual(['Iteration 2']);
      announcer.announce('Iteration 3');
      announcer.announce('Iteration 2');
      vi.runAllTimers();
      expect(emitted).toEqual(['Iteration 2']);
      announcer.destroy();
    } finally {
      vi.useRealTimers();
    }
  });
  test('flushes immediately, cancels on destroy, and accepts injected timers', () => {
    vi.useFakeTimers();
    try {
      const emitted: string[] = [];
      const announcer = createDebouncedAnnouncer(text => emitted.push(text), {
        debounceMs: 20,
        setTimeout: (callback, delay) => setTimeout(callback, delay),
        clearTimeout: handle => clearTimeout(handle as ReturnType<typeof setTimeout>)
      });
      announcer.announce('A');
      announcer.flush();
      announcer.flush();
      expect(emitted).toEqual(['A']);
      announcer.announce('B');
      vi.advanceTimersByTime(20);
      announcer.announce('A');
      announcer.flush();
      expect(emitted).toEqual(['A', 'B', 'A']);
      announcer.announce('cancelled');
      announcer.destroy();
      announcer.announce('after destroy');
      announcer.flush();
      vi.runAllTimers();
      expect(emitted).toEqual(['A', 'B', 'A']);
      expect(vi.getTimerCount()).toBe(0);
    } finally {
      vi.useRealTimers();
    }
  });
});
