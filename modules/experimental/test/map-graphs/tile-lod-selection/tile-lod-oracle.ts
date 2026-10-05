// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPU_TILE_LOD_VIEW_OFFSETS as VIEW,
  getGPUTileLODViewParameterValues,
  type GPUTileLODViewProps
} from '../../../src/map-graphs/tile-lod-selection';

export const UNLIMITED = 0xffffffff;
export const INVALID = 0xffffffff;

/** CPU-side tile hierarchy fixture. */
export type TileLODFixture = {
  sphereBounds: Float32Array;
  geometricErrors: Float32Array;
  children: Uint32Array;
  levelOffsets: number[];
  tileIds?: Uint32Array;
  nodeCosts?: Uint32Array;
  residency?: Uint32Array;
  enabledNodes?: Uint32Array;
  parents?: Uint32Array;
};

/** CPU reference outputs. */
export type TileLODOracleResult = {
  drawnIds: number[];
  desiredIds: number[];
  requestedIds: number[];
  requestPriorities: number[];
  drawMask: number[];
  drawnAncestors: number[];
  statistics: number[];
};

const addSaturated = (left: number, right: number) => Math.min(left + right, UNLIMITED);

/** Port of the level-synchronous GPU selection algorithm. */
export function selectTileLODOnCPU(
  fixture: TileLODFixture,
  view: Float32Array,
  options: {refinement?: 'replace' | 'add'; budget?: readonly [number, number]} = {}
): TileLODOracleResult {
  const add = options.refinement === 'add';
  const {levelOffsets} = fixture;
  const nodeCount = levelOffsets[levelOffsets.length - 1];
  const levelCount = levelOffsets.length - 1;
  const cost = (node: number) => fixture.nodeCosts?.[node] ?? 0;
  const resident = (node: number) => !fixture.residency || fixture.residency[node] !== 0;
  const enabled = (node: number) => !fixture.enabledNodes || fixture.enabledNodes[node] !== 0;
  const center = (node: number) => [
    fixture.sphereBounds[node * 4],
    fixture.sphereBounds[node * 4 + 1],
    fixture.sphereBounds[node * 4 + 2]
  ];
  const radius = (node: number) => fixture.sphereBounds[node * 4 + 3];
  const camera = [
    view[VIEW.cameraPosition],
    view[VIEW.cameraPosition + 1],
    view[VIEW.cameraPosition + 2]
  ];
  const scale = view[VIEW.pixelProjectionScale];
  const threshold = view[VIEW.maximumScreenSpaceError];
  const distanceTo = (point: number[]) =>
    Math.hypot(point[0] - camera[0], point[1] - camera[1], point[2] - camera[2]);
  const isVisible = (node: number) => {
    if (!enabled(node)) return false;
    const c = center(node);
    const r = radius(node);
    for (let plane = 0; plane < 6; plane++) {
      const p = Array.from(view.slice(plane * 4, plane * 4 + 4));
      if (p[0] * c[0] + p[1] * c[1] + p[2] * c[2] + p[3] < -r) return false;
    }
    return true;
  };
  const weightedError = (node: number): number => {
    const c = center(node);
    const r = radius(node);
    const surface = distanceTo(c) - r;
    if (surface <= 1e-6) return Infinity;
    let error = (fixture.geometricErrors[node] * scale) / surface;
    const strength = view[VIEW.foveationStrength];
    if (strength > 0) {
      const cameraDistance = distanceTo(c);
      const m = Array.from(view.slice(VIEW.viewProjectionMatrix, VIEW.viewProjectionMatrix + 16));
      const clip = [0, 1, 2, 3].map(
        k => m[k] * c[0] + m[4 + k] * c[1] + m[8 + k] * c[2] + m[12 + k]
      );
      if (cameraDistance > r && clip[3] > 0) {
        const width = Math.max(view[VIEW.viewportSize], 1);
        const height = Math.max(view[VIEW.viewportSize + 1], 1);
        const screen = [(clip[0] / clip[3]) * 0.5 + 0.5, 0.5 - (clip[1] / clip[3]) * 0.5];
        const offset = [
          screen[0] - view[VIEW.foveationCenter],
          screen[1] - view[VIEW.foveationCenter + 1]
        ];
        const gaze = Math.hypot(offset[0], offset[1]);
        const projectedRadius = (r * scale) / (cameraDistance * height);
        const aspect = width / height;
        const toward =
          gaze > 0
            ? projectedRadius / Math.hypot((offset[0] / gaze) * aspect, offset[1] / gaze)
            : projectedRadius;
        const peripheral = Math.max(gaze - toward - Math.max(view[VIEW.foveationRadius], 0), 0);
        error /= 1 + peripheral * strength;
      }
    }
    const focus = view[VIEW.focusDistance];
    const falloff = view[VIEW.distanceFalloff];
    if (falloff > 0 && focus > 0) {
      error /= Math.max(surface / focus, 1) ** falloff;
    }
    return error;
  };
  const bucketOf = (error: number) =>
    Number.isFinite(error)
      ? Math.min(Math.max(Math.floor(Math.log2(Math.max(error, 2 ** -8))) + 8, 0), 31)
      : 31;

  const state = new Array<number>(nodeCount).fill(0);
  for (let node = 0; node < levelOffsets[1]; node++) state[node] = 2;
  const visible = new Array<boolean>(nodeCount).fill(false);
  const refined = new Array<boolean>(nodeCount).fill(false);
  const childrenResidentFlag = new Array<boolean>(nodeCount).fill(false);
  const priority = new Array<number>(nodeCount).fill(0);
  let committedCost = 0;
  let committedCount = 0;
  let exhausted = 0;

  for (let level = 0; level < levelCount; level++) {
    const first = levelOffsets[level];
    const end = levelOffsets[level + 1];
    const hasNext = level + 1 < levelCount;
    const nextFirst = hasNext ? levelOffsets[level + 1] : 0;
    const nextEnd = hasNext ? levelOffsets[level + 2] : 0;
    type Evaluation = {
      node: number;
      wantsRefine: boolean;
      childrenResident: boolean;
      childCount: number;
      childCost: number;
      bucket: number;
      firstChild: number;
      count: number;
    };
    const evaluations: Evaluation[] = [];
    let levelCost = 0;
    let levelNodeCount = 0;
    const buckets = Array.from({length: 32}, () => [0, 0, 0, 0]);
    for (let node = first; node < end; node++) {
      if (state[node] === 0 || !isVisible(node)) continue;
      visible[node] = true;
      const firstChild = fixture.children[node * 2];
      const count = fixture.children[node * 2 + 1];
      const validRange =
        hasNext &&
        count > 0 &&
        firstChild >= nextFirst &&
        firstChild <= nextEnd &&
        count <= nextEnd - firstChild;
      const error = weightedError(node);
      const wantsRefine = validRange && (!Number.isFinite(error) || error > threshold);
      let childrenResident = true;
      let childCount = 0;
      let childCost = 0;
      if (validRange) {
        for (let child = firstChild; child < firstChild + count; child++) {
          if (isVisible(child)) {
            childCount++;
            childCost = addSaturated(childCost, cost(child));
            if (!resident(child)) childrenResident = false;
          }
        }
      }
      childrenResidentFlag[node] = childrenResident;
      priority[node] = Number.isFinite(error) ? Math.fround(error) : 3.4028234663852886e38;
      const bucket = bucketOf(error);
      evaluations.push({
        node,
        wantsRefine,
        childrenResident,
        childCount,
        childCost,
        bucket,
        firstChild,
        count
      });
      levelCost += cost(node);
      levelNodeCount++;
      if (wantsRefine) {
        buckets[bucket][0] += childCost;
        buckets[bucket][2] += childCount;
        if (!add) {
          buckets[bucket][1] += cost(node);
          buckets[bucket][3] += 1;
        }
      }
    }
    let acceptedBucket = 0;
    if (options.budget) {
      const baseCost = addSaturated(committedCost, levelCost);
      const baseCount = addSaturated(committedCount, levelNodeCount);
      let accepted = [0, 0, 0, 0];
      acceptedBucket = 32;
      for (let bucket = 31; bucket >= 0; bucket--) {
        const entry = buckets[bucket];
        if (entry.every(value => value === 0)) continue;
        const next = accepted.map((value, index) => addSaturated(value, entry[index]));
        const fitsCost =
          addSaturated(baseCost, next[0]) <= addSaturated(options.budget[0], next[1]);
        const fitsCount =
          addSaturated(baseCount, next[2]) <= addSaturated(options.budget[1], next[3]);
        if (fitsCost && fitsCount) {
          accepted = next;
          acceptedBucket = bucket;
        } else {
          exhausted = 1;
          break;
        }
      }
    }
    for (const evaluation of evaluations) {
      const refine = evaluation.wantsRefine && evaluation.bucket >= acceptedBucket;
      if (refine) {
        refined[evaluation.node] = true;
        const childState =
          state[evaluation.node] === 2 && (add || evaluation.childrenResident) ? 2 : 1;
        for (
          let child = evaluation.firstChild;
          child < evaluation.firstChild + evaluation.count;
          child++
        ) {
          state[child] = Math.max(state[child], childState);
        }
      }
      if (options.budget && (!refine || add)) {
        committedCount++;
        committedCost += cost(evaluation.node);
      }
    }
  }

  const tileId = (node: number) => fixture.tileIds?.[node] ?? node;
  const result: TileLODOracleResult = {
    drawnIds: [],
    desiredIds: [],
    requestedIds: [],
    requestPriorities: [],
    drawMask: [],
    drawnAncestors: [],
    statistics: [0, 0, 0, 0, 0, exhausted, 0, 0]
  };
  for (let node = 0; node < nodeCount; node++) {
    const isNodeVisible = state[node] > 0 && visible[node];
    const desired = isNodeVisible && (add || !refined[node]);
    const drawn =
      isNodeVisible &&
      resident(node) &&
      state[node] === 2 &&
      (add || !refined[node] || !childrenResidentFlag[node]);
    const requested = isNodeVisible && !resident(node);
    result.drawMask.push(drawn ? 1 : 0);
    if (isNodeVisible) result.statistics[6]++;
    if (desired) {
      result.desiredIds.push(tileId(node));
      result.statistics[0]++;
      result.statistics[1] += cost(node);
    }
    if (drawn) {
      result.drawnIds.push(tileId(node));
      result.statistics[2]++;
      result.statistics[3] += cost(node);
    }
    if (requested) {
      result.requestedIds.push(tileId(node));
      result.requestPriorities.push(priority[node]);
      result.statistics[4]++;
    }
  }
  for (let node = 0; node < nodeCount; node++) {
    let current = node;
    let ancestor = INVALID;
    for (let depth = 0; depth <= levelCount && current < nodeCount; depth++) {
      if (result.drawMask[current]) {
        ancestor = current;
        break;
      }
      current = fixture.parents?.[current] ?? INVALID;
    }
    result.drawnAncestors.push(ancestor);
  }
  return result;
}

/** Six axis-aligned inward planes bounding `minimumX <= x <= maximumX` and `|y|, |z| <= extent`. */
export function makeBoxFrustum(minimumX: number, maximumX: number, extent: number): Float32Array {
  return Float32Array.from([
    1,
    0,
    0,
    -minimumX,
    -1,
    0,
    0,
    maximumX,
    0,
    1,
    0,
    extent,
    0,
    -1,
    0,
    extent,
    0,
    0,
    1,
    extent,
    0,
    0,
    -1,
    extent
  ]);
}

/** Fixture view: wide box frustum, camera at the origin, 100 px per unit, 100x100 viewport. */
export function makeFixtureView(overrides: Partial<GPUTileLODViewProps> = {}): Float32Array {
  return getGPUTileLODViewParameterValues({
    viewProjectionMatrix: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    frustumPlanes: Array.from(makeBoxFrustum(-1000, 1000, 1000)),
    cameraPosition: [0, 0, 0],
    pixelProjectionScale: 100,
    viewportSize: [100, 100],
    maximumScreenSpaceError: 10,
    ...overrides
  });
}

/** Fixture A: a 7-node, 3-level binary tile tree. */
export const FIXTURE_A: TileLODFixture = {
  levelOffsets: [0, 1, 3, 7],
  sphereBounds: Float32Array.from([
    0, 0, 20, 10, -5, 0, 20, 5, 5, 0, 20, 5, -7.5, 0, 20, 2.5, -2.5, 0, 20, 2.5, 2.5, 0, 20, 2.5,
    7.5, 0, 20, 2.5
  ]),
  geometricErrors: Float32Array.from([8, 2, 1, 0.5, 0.5, 0.5, 0.5]),
  children: Uint32Array.from([1, 2, 3, 2, 5, 2, 0, 0, 0, 0, 0, 0, 0, 0]),
  nodeCosts: Uint32Array.from([100, 60, 60, 40, 40, 40, 40]),
  tileIds: Uint32Array.from([1000, 1100, 1101, 1200, 1201, 1202, 1203]),
  parents: Uint32Array.from([INVALID, 0, 0, 1, 1, 2, 2])
};
