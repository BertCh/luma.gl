// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test} from 'vitest';
import {
  createLadderNavigation,
  navigateLadder,
  isLadderNavigationKey
} from '../../../src/computation-views/hosts/navigation-graph';
import {LENS_RUNG_IDS} from '../../../src/computation-views/types';

describe('ladder navigation graph', () => {
  test('moves rung and entity axes independently, clamping each', () => {
    const start = createLadderNavigation({
      rungs: LENS_RUNG_IDS,
      rung: 'iterations',
      entityCount: 4096
    });
    const down = navigateLadder(start, 'ArrowDown');
    expect(down.announcement).toBe('Rung kernel, 5 of 10');
    expect(down.state.rungIndex).toBe(4);
    expect(start.rungIndex).toBe(3);
    expect(navigateLadder(down.state, 'ArrowUp').state.rungIndex).toBe(3);
    const entity = navigateLadder({...down.state, entityIndex: 16}, 'ArrowRight');
    expect(entity.announcement).toBe('Entity 18 of 4096');
    expect(entity.state.rungIndex).toBe(4);
    expect(navigateLadder(entity.state, 'ArrowLeft').state.entityIndex).toBe(16);
    expect(navigateLadder({...start, rungIndex: 0}, 'ArrowUp').changed).toBe(false);
    expect(navigateLadder({...start, rungIndex: 9}, 'ArrowDown').changed).toBe(false);
  });
  test('supports page jumps, endpoints and pin/unpin with no wrap', () => {
    const start = createLadderNavigation({rungs: LENS_RUNG_IDS, entityCount: 19});
    expect(start.pageSize).toBe(10);
    expect(navigateLadder(start, 'ArrowLeft').changed).toBe(false);
    const page = navigateLadder(start, 'PageDown').state;
    expect(page.entityIndex).toBe(10);
    expect(navigateLadder(page, 'PageDown').state.entityIndex).toBe(18);
    expect(navigateLadder(page, 'PageUp').state.entityIndex).toBe(0);
    expect(navigateLadder(page, 'Home').state.entityIndex).toBe(0);
    expect(navigateLadder(page, 'End').state.entityIndex).toBe(18);
    const pinned = navigateLadder({...page, entityIndex: 17}, 'Enter');
    expect(pinned.announcement).toBe('Pinned entity 17');
    expect(navigateLadder(pinned.state, 'Enter').state.pinned).toBe(false);
    expect(navigateLadder(pinned.state, 'Escape').state.pinned).toBe(false);
    expect(navigateLadder(pinned.state, 'ArrowDown').state.pinned).toBe(true);
  });
  test('handles empty entity domains and recognizes only supported keys', () => {
    const empty = createLadderNavigation({rungs: ['datum'], entityCount: 0});
    expect(navigateLadder(empty, 'End')).toEqual({
      state: empty,
      changed: false,
      announcement: 'No entities'
    });
    expect(navigateLadder(empty, 'Enter').state.pinned).toBe(false);
    expect(isLadderNavigationKey('ArrowDown')).toBe(true);
    expect(isLadderNavigationKey('Tab')).toBe(false);
    expect(() => createLadderNavigation({rungs: [], entityCount: 1})).toThrow();
  });
});
