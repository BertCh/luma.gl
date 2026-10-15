// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';
import type {LensRungId} from '../types';

/** Position and pin in a rung by entity navigation graph. Empty entity domains use index zero. */
export type LadderNavigationState = {
  rungs: readonly LensRungId[];
  rungIndex: number;
  entityCount: number;
  entityIndex: number;
  pinned: boolean;
  pageSize: number;
};

/** Keyboard commands supported by the ladder navigation graph. */
export type LadderNavigationKey =
  | 'ArrowUp'
  | 'ArrowDown'
  | 'ArrowLeft'
  | 'ArrowRight'
  | 'Home'
  | 'End'
  | 'PageUp'
  | 'PageDown'
  | 'Enter'
  | 'Escape';

const NAVIGATION_KEYS: readonly string[] = [
  'ArrowUp',
  'ArrowDown',
  'ArrowLeft',
  'ArrowRight',
  'Home',
  'End',
  'PageUp',
  'PageDown',
  'Enter',
  'Escape'
];

/** Creates navigation state at the requested rung and first entity. */
export function createLadderNavigation(props: {
  rungs: readonly LensRungId[];
  rung?: LensRungId;
  entityCount: number;
  pageSize?: number;
}): LadderNavigationState {
  // Navigation requires at least one rung and a finite, integral domain and page size.
  assert(
    props.rungs.length > 0 &&
      Number.isInteger(props.entityCount) &&
      props.entityCount >= 0 &&
      Number.isInteger(props.pageSize ?? 10) &&
      (props.pageSize ?? 10) > 0
  );
  return {
    rungs: [...props.rungs],
    rungIndex: Math.max(0, props.rungs.indexOf(props.rung ?? props.rungs[0])),
    entityCount: props.entityCount,
    entityIndex: 0,
    pinned: false,
    pageSize: props.pageSize ?? 10
  };
}

/** Applies a clamped command without mutating the input state. */
export function navigateLadder(
  state: LadderNavigationState,
  key: LadderNavigationKey
): {state: LadderNavigationState; changed: boolean; announcement: string} {
  const next = {...state};
  switch (key) {
    case 'ArrowUp':
      next.rungIndex--;
      break;
    case 'ArrowDown':
      next.rungIndex++;
      break;
    case 'ArrowLeft':
      next.entityIndex--;
      break;
    case 'ArrowRight':
      next.entityIndex++;
      break;
    case 'Home':
      next.entityIndex = 0;
      break;
    case 'End':
      next.entityIndex = state.entityCount - 1;
      break;
    case 'PageUp':
      next.entityIndex -= state.pageSize;
      break;
    case 'PageDown':
      next.entityIndex += state.pageSize;
      break;
    case 'Enter':
      next.pinned = state.entityCount > 0 && !state.pinned;
      break;
    case 'Escape':
      next.pinned = false;
      break;
  }
  next.rungIndex = Math.max(0, Math.min(next.rungs.length - 1, next.rungIndex));
  next.entityIndex = Math.max(0, Math.min(next.entityCount - 1, next.entityIndex));
  const changed =
    next.rungIndex !== state.rungIndex ||
    next.entityIndex !== state.entityIndex ||
    next.pinned !== state.pinned;
  const announcement =
    key === 'ArrowUp' || key === 'ArrowDown'
      ? `Rung ${next.rungs[next.rungIndex]}, ${next.rungIndex + 1} of ${next.rungs.length}`
      : next.entityCount === 0
        ? 'No entities'
        : key === 'Enter' || key === 'Escape'
          ? `${next.pinned ? 'Pinned' : 'Unpinned'} entity ${next.entityIndex}`
          : `Entity ${next.entityIndex + 1} of ${next.entityCount}`;
  return {state: changed ? next : state, changed, announcement};
}

/** Recognizes keyboard commands without requiring a DOM event. */
export function isLadderNavigationKey(key: string): key is LadderNavigationKey {
  return NAVIGATION_KEYS.includes(key);
}
