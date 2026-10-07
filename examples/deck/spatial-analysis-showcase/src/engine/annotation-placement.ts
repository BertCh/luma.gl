// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {LabelAnchor} from '../cartography/types';
import {type Box, boxInside, boxesOverlap} from './annotation-geometry';

/** Measured label size in CSS pixels. */
export type LabelSize = {width: number; height: number};

/**
 * One place a label may go: its collision `box`, the position its element is translated to and,
 * for notes and offset points, the leader path (local to the feature position).
 */
export type LabelCandidate = {
  box: Box;
  labelX: number;
  labelY: number;
  /** SVG path data of the leader, relative to the feature (`M0 0 L...`). */
  leader?: string;
  /** Write the label position without rounding (the callout pointer tip). */
  exact?: boolean;
};

/** Gap between a feature and its label box (Imhof), in CSS pixels. */
export const LABEL_GAP = 6;
/** Distance kept free between a movable label and the map edge, in CSS pixels. */
export const EDGE_INSET = 12;
/** Padding added around boxes when testing label collisions, in CSS pixels. */
export const COLLISION_PADDING = 3;

type Compass = Exclude<LabelAnchor, 'auto'>;

const AUTO_ORDER: readonly Compass[] = ['ne', 'se', 'nw', 'sw'];
const DIAGONAL = Math.SQRT1_2;

/** Unit direction (screen space, y down) of each compass anchor. */
const DIRECTIONS: Record<Compass, readonly [number, number]> = {
  ne: [1, -1],
  se: [1, 1],
  nw: [-1, -1],
  sw: [-1, 1],
  n: [0, -1],
  s: [0, 1],
  e: [1, 0],
  w: [-1, 0]
};

/** The candidate anchors for an `anchor` option: NE, SE, NW, SW for `'auto'`, else just one. */
export function anchorOrder(anchor: LabelAnchor | undefined): readonly Compass[] {
  return anchor === undefined || anchor === 'auto' ? AUTO_ORDER : [anchor];
}

/** A candidate whose label sits exactly at `(x, y)`. */
export function plainCandidate(x: number, y: number, size: LabelSize): LabelCandidate {
  const left = Math.round(x);
  const top = Math.round(y);
  return {
    box: {x: left, y: top, width: size.width, height: size.height},
    labelX: left,
    labelY: top
  };
}

/**
 * Candidates around a feature at `(x, y)`: the label box touches a point `distance` away along
 * each anchor direction (the diagonal distance is Euclidean). With `leader`, the candidate carries
 * a straight leader from the feature to that point.
 */
export function directionalCandidates(
  anchor: LabelAnchor | undefined,
  x: number,
  y: number,
  distance: number,
  size: LabelSize,
  leader: boolean
): LabelCandidate[] {
  return anchorOrder(anchor).map(name => {
    const [directionX, directionY] = DIRECTIONS[name];
    const scale = directionX !== 0 && directionY !== 0 ? DIAGONAL : 1;
    const offsetX = directionX * distance * scale;
    const offsetY = directionY * distance * scale;
    const left =
      directionX > 0 ? x + offsetX : directionX < 0 ? x + offsetX - size.width : x - size.width / 2;
    const top =
      directionY > 0
        ? y + offsetY
        : directionY < 0
          ? y + offsetY - size.height
          : y - size.height / 2;
    const candidate = plainCandidate(left, top, size);
    if (leader) candidate.leader = `M0 0L${offsetX.toFixed(1)} ${offsetY.toFixed(1)}`;
    return candidate;
  });
}

/** Leader path data for a point label offset by `(dx, dy)`: straight, or one elbow when both are large. */
export function offsetLeader(dx: number, dy: number): string {
  if (Math.abs(dx) >= 14 && Math.abs(dy) >= 14) {
    const run = Math.min(10, Math.abs(dx) / 2);
    return `M0 0L${(dx - Math.sign(dx) * run).toFixed(1)} ${dy}L${dx} ${dy}`;
  }
  return `M0 0L${dx} ${dy}`;
}

/** The single candidate of a point with an explicit pixel `offset` from its feature. */
export function offsetCandidate(
  x: number,
  y: number,
  dx: number,
  dy: number,
  size: LabelSize
): LabelCandidate {
  const anchorX = x + dx;
  const anchorY = y + dy;
  let left: number;
  let top: number;
  if (dx > 0) {
    left = anchorX + 3;
    top = anchorY - size.height / 2;
  } else if (dx < 0) {
    left = anchorX - size.width - 3;
    top = anchorY - size.height / 2;
  } else {
    left = anchorX - size.width / 2;
    top = dy < 0 ? anchorY - size.height : anchorY;
  }
  return plainCandidate(left, top, size);
}

/** `true` when the box overlaps any placed box (padded by {@link COLLISION_PADDING}). */
export function collides(box: Box, placed: readonly Box[]): boolean {
  for (const other of placed) {
    if (boxesOverlap(box, other, COLLISION_PADDING)) return true;
  }
  return false;
}

/**
 * Picks the first usable candidate: the previously used one first (hysteresis), then in order.
 * A candidate is usable when it is inside `edge` (when given) and free of `placed`. Returns its
 * index, or -1 when none is usable.
 */
export function chooseCandidate(
  candidates: readonly LabelCandidate[],
  previousIndex: number,
  placed: readonly Box[],
  edge: Box | null
): number {
  const usable = (index: number): boolean => {
    const box = candidates[index].box;
    return (!edge || boxInside(box, edge)) && !collides(box, placed);
  };
  if (previousIndex >= 0 && previousIndex < candidates.length && usable(previousIndex)) {
    return previousIndex;
  }
  for (let index = 0; index < candidates.length; index++) {
    if (index !== previousIndex && usable(index)) return index;
  }
  return -1;
}
