// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  GPUCommandGraph,
  GPUCommandGraphNodeCondition,
  GPUCommandNode,
  GraphBufferUse,
  GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';

/** Distance measured by the Douglas-Peucker importance pass. */
export type GPULineSimplificationMetric = 'segment' | 'time-ratio';

const OPERATION = 'GPULineSimplification';

/** Bit pattern of `+Infinity`, the importance of a line endpoint. @internal */
export const LINE_ENDPOINT_IMPORTANCE_BITS = 0x7f800000;

/**
 * WGSL metric helpers. Only correctly rounded `+`, `-`, and `*` decide a result, so a CPU oracle
 * that mirrors each operation with `Math.fround` reproduces it bit for bit. Division and square
 * root (which WGSL allows to be a few ULP off) only seed a bounded correction to a defined value:
 *
 * - `lineDivideFloor(n, d)` is the largest f32 `q >= 0` with `fround(q * d) <= n`;
 * - `lineSqrtFloor(x)` is the largest f32 `s >= 0` with `fround(s * s) <= x`.
 *
 * A zero (or negative) operand returns 0, so the subnormal range is never searched. Both exist because `q -> fround(q * d)` is monotone, so the set of admissible values has a
 * unique maximum that every starting estimate within 16 ULP reaches.
 *
 * @internal
 */
export const LINE_SIMPLIFICATION_METRIC_WGSL = /* wgsl */ `
const LINE_MAXIMUM_FINITE: f32 = bitcast<f32>(0x7f7fffffu);

// Runtime zero: the kernel stores \`leftAnchor >> 31u\`, a row index loaded from memory (rows are
// below 2^31), which the compiler cannot fold, unlike thread IDs or \`arrayLength\` whose ranges it
// knows. ORing it into a
// product's bits keeps the compiler from contracting \`a * b + c\` or \`a * b > c\` into a fused
// multiply-add (an unrounded product), which Metal otherwise does, changing results by a few ULP.
var<private> lineOpaqueZero: u32;

fn lineProduct(left: f32, right: f32) -> f32 {
  return bitcast<f32>(bitcast<u32>(left * right) | lineOpaqueZero);
}

fn lineNextUp(value: f32) -> f32 {
  return bitcast<f32>(bitcast<u32>(value) + 1u);
}

fn lineNextDown(value: f32) -> f32 {
  return bitcast<f32>(bitcast<u32>(value) - 1u);
}

fn lineDivideFloor(numerator: f32, denominator: f32) -> f32 {
  if (!(numerator > 0.0)) {
    return 0.0;
  }
  var quotient = numerator / denominator;
  if (!(quotient >= 0.0)) {
    quotient = 0.0;
  }
  quotient = min(quotient, LINE_MAXIMUM_FINITE);
  for (var step = 0u; step < 16u; step++) {
    if (quotient > 0.0 && lineProduct(quotient, denominator) > numerator) {
      quotient = lineNextDown(quotient);
    } else {
      break;
    }
  }
  for (var step = 0u; step < 16u; step++) {
    let next = lineNextUp(quotient);
    if (lineProduct(next, denominator) <= numerator) {
      quotient = next;
    } else {
      break;
    }
  }
  return quotient;
}

fn lineSqrtFloor(value: f32) -> f32 {
  if (!(value > 0.0)) {
    return 0.0;
  }
  var root = sqrt(value);
  if (!(root >= 0.0)) {
    root = 0.0;
  }
  root = min(root, LINE_MAXIMUM_FINITE);
  for (var step = 0u; step < 16u; step++) {
    if (root > 0.0 && lineProduct(root, root) > value) {
      root = lineNextDown(root);
    } else {
      break;
    }
  }
  for (var step = 0u; step < 16u; step++) {
    let next = lineNextUp(root);
    if (lineProduct(next, next) <= value) {
      root = next;
    } else {
      break;
    }
  }
  return root;
}

fn linePointDistance(px: f32, py: f32, ax: f32, ay: f32) -> f32 {
  let deltaX = px - ax;
  let deltaY = py - ay;
  return lineSqrtFloor(lineProduct(deltaX, deltaX) + lineProduct(deltaY, deltaY));
}

fn lineSegmentDistance(px: f32, py: f32, ax: f32, ay: f32, bx: f32, by: f32) -> f32 {
  let segmentX = bx - ax;
  let segmentY = by - ay;
  let offsetX = px - ax;
  let offsetY = py - ay;
  let lengthSquared = lineProduct(segmentX, segmentX) + lineProduct(segmentY, segmentY);
  let projection = lineProduct(offsetX, segmentX) + lineProduct(offsetY, segmentY);
  if (lengthSquared <= 0.0 || projection <= 0.0) {
    return linePointDistance(px, py, ax, ay);
  }
  if (projection >= lengthSquared) {
    return linePointDistance(px, py, bx, by);
  }
  let segmentLength = lineSqrtFloor(lengthSquared);
  if (segmentLength <= 0.0) {
    return linePointDistance(px, py, ax, ay);
  }
  let cross = abs(lineProduct(segmentX, offsetY) - lineProduct(segmentY, offsetX));
  return lineDivideFloor(cross, segmentLength);
}

fn lineTimeRatioDistance(
  px: f32, py: f32, pt: f32, ax: f32, ay: f32, at: f32, bx: f32, by: f32, bt: f32
) -> f32 {
  let duration = bt - at;
  if (!(duration > 0.0)) {
    return linePointDistance(px, py, ax, ay);
  }
  let elapsed = pt - at;
  var ratio = lineDivideFloor(abs(elapsed), duration);
  if (elapsed < 0.0) {
    ratio = -ratio;
  }
  let expectedX = ax + lineProduct(ratio, bx - ax);
  let expectedY = ay + lineProduct(ratio, by - ay);
  return linePointDistance(px, py, expectedX, expectedY);
}
`;

/** Graph-owned scratch columns of the importance loop. @internal */
export type LineImportanceScratch = {
  /** Left anchor row of the open interval holding each row; the row itself once decided. */
  leftAnchors: GraphDataView<'uint32'>;
  /** Right anchor row of the open interval holding each row; the row itself once decided. */
  rightAnchors: GraphDataView<'uint32'>;
  /** f32 bits of the row's distance in the current round. */
  distanceKeys: GraphDataView<'uint32'>;
  /** Per-interval (indexed by left anchor) maximum distance bits. */
  bestKeys: GraphDataView<'uint32'>;
  /** Per-interval (indexed by left anchor) smallest row with the maximum distance. */
  bestRows: GraphDataView<'uint32'>;
};

/** Gate shared by every node of one round. @internal */
export type LineRoundGate<Parameters> = {
  condition: GPUCommandGraphNodeCondition<Parameters>;
  extraResources: GraphBufferUse[];
};

const UNDECIDED_WGSL = (
  index: string
) => /* wgsl */ `let leftAnchor = leftAnchors[leftAnchorsOffset + ${index}];
  let rightAnchor = rightAnchors[rightAnchorsOffset + ${index}];
  if (!(leftAnchor < ${index} && ${index} < rightAnchor)) {
    return;
  }`;

/**
 * Initializes anchors and importance: endpoints get `+Infinity`, interior rows of a line get the
 * whole line as their open interval, and rows outside every line are decided with importance 0.
 *
 * @internal
 */
export function createLineImportanceInitNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    trackOffsets: GraphDataView<'uint32'>;
    scratch: LineImportanceScratch;
    importance: GraphDataView<'float32'>;
  }
): GPUCommandNode<Parameters> {
  const lineCount = props.trackOffsets.length - 1;
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'importance-init',
    bindings: [
      {
        name: 'trackOffsets',
        view: props.trackOffsets,
        type: 'u32',
        access: 'read'
      },
      {
        name: 'leftAnchors',
        view: props.scratch.leftAnchors,
        type: 'u32',
        access: 'read_write'
      },
      {
        name: 'rightAnchors',
        view: props.scratch.rightAnchors,
        type: 'u32',
        access: 'read_write'
      },
      {
        name: 'importance',
        view: props.importance,
        type: 'u32',
        access: 'read_write'
      }
    ],
    invocationCount: props.importance.length,
    declarations: `const LINE_COUNT: u32 = ${lineCount}u;
const ENDPOINT_IMPORTANCE: u32 = ${LINE_ENDPOINT_IMPORTANCE_BITS}u;`,
    body: /* wgsl */ `let firstRow = trackOffsets[trackOffsetsOffset];
  let endRow = trackOffsets[trackOffsetsOffset + LINE_COUNT];
  if (index < firstRow || index >= endRow) {
    leftAnchors[leftAnchorsOffset + index] = index;
    rightAnchors[rightAnchorsOffset + index] = index;
    importance[importanceOffset + index] = 0u;
    return;
  }
  // Last line whose start is <= index (empty lines share a start with the next line).
  var low = 0u;
  var high = LINE_COUNT + 1u;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (trackOffsets[trackOffsetsOffset + middle] <= index) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  let line = low - 1u;
  let lineStart = trackOffsets[trackOffsetsOffset + line];
  let lineLast = trackOffsets[trackOffsetsOffset + line + 1u] - 1u;
  if (index == lineStart || index >= lineLast) {
    leftAnchors[leftAnchorsOffset + index] = index;
    rightAnchors[rightAnchorsOffset + index] = index;
    importance[importanceOffset + index] = ENDPOINT_IMPORTANCE;
    return;
  }
  leftAnchors[leftAnchorsOffset + index] = lineStart;
  rightAnchors[rightAnchorsOffset + index] = lineLast;
  importance[importanceOffset + index] = 0u;`
  });
}

/**
 * Returns the four gated nodes of one level-synchronous Douglas-Peucker round:
 *
 * 1. `reset`: every undecided row clears its interval's best key and best row;
 * 2. `distance`: every undecided row measures its distance to the interval chord and raises the
 *    interval's best key with `atomicMax` on the f32 bits (non-negative, so bit order is value
 *    order);
 * 3. `argmax`: rows whose key equals the maximum lower the interval's best row with `atomicMin`, so
 *    ties go to the smallest row;
 * 4. `split`: the winner is decided with importance `min(distance, min(importance[a], importance[b]))`,
 *    which is the importance of the split that created the interval; other rows move to the half
 *    that holds them and raise the "rows remain" flag. An interval whose maximum distance is 0 is
 *    decided at once with importance 0: every descendant would be clamped to 0 anyway.
 *
 * @internal
 */
export function createLineImportanceRoundNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    metric: GPULineSimplificationMetric;
    positions: GraphDataView<'float32x2'>;
    timestamps?: GraphDataView<'float32'>;
    scratch: LineImportanceScratch;
    importance: GraphDataView<'float32'>;
    status: GraphDataView<'uint32'>;
    gate: LineRoundGate<Parameters>;
    /** Open intervals with at most this many interior rows are solved whole; 0 disables. */
    finishSpanLimit?: number;
  }
): GPUCommandNode<Parameters>[] {
  const {scratch, gate} = props;
  const finishSpanLimit = props.finishSpanLimit ?? 0;
  const rowCount = props.importance.length;
  const anchors: WGSLKernelBinding[] = [
    {
      name: 'leftAnchors',
      view: scratch.leftAnchors,
      type: 'u32',
      access: 'read'
    },
    {
      name: 'rightAnchors',
      view: scratch.rightAnchors,
      type: 'u32',
      access: 'read'
    }
  ];
  const bestKeys: WGSLKernelBinding = {
    name: 'bestKeys',
    view: scratch.bestKeys,
    type: 'atomic<u32>',
    access: 'read_write'
  };
  const bestRows: WGSLKernelBinding = {
    name: 'bestRows',
    view: scratch.bestRows,
    type: 'atomic<u32>',
    access: 'read_write'
  };
  const gated = {
    operation: OPERATION,
    invocationCount: rowCount,
    condition: gate.condition,
    extraResources: gate.extraResources
  };
  const isTimeRatio = props.metric === 'time-ratio';
  const distanceBindings: WGSLKernelBinding[] = [
    {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
    ...anchors,
    {
      name: 'distanceKeys',
      view: scratch.distanceKeys,
      type: 'u32',
      access: 'read_write'
    },
    bestKeys
  ];
  if (isTimeRatio && props.timestamps) {
    distanceBindings.push({
      name: 'timestamps',
      view: props.timestamps,
      type: 'f32',
      access: 'read'
    });
  }
  const position = (row: string, component: 0 | 1) =>
    `positions[positionsOffset + 2u * ${row} + ${component}u]`;
  const time = (row: string) => `timestamps[timestampsOffset + ${row}]`;
  const getDistanceExpression = (row: string, left: string, right: string) =>
    isTimeRatio
      ? `lineTimeRatioDistance(${position(row, 0)}, ${position(row, 1)}, ${time(row)},
      ${position(left, 0)}, ${position(left, 1)}, ${time(left)},
      ${position(right, 0)}, ${position(right, 1)}, ${time(right)})`
      : `lineSegmentDistance(${position(row, 0)}, ${position(row, 1)},
      ${position(left, 0)}, ${position(left, 1)},
      ${position(right, 0)}, ${position(right, 1)})`;
  const distanceExpression = getDistanceExpression('index', 'leftAnchor', 'rightAnchor');

  const resetNode =
    finishSpanLimit > 0
      ? createWGSLKernelNode<Parameters>(graph, {
          ...gated,
          id: `${props.id}-reset`,
          variant: `importance-reset-finish-${props.metric}`,
          bindings: [
            {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
            {
              name: 'leftAnchors',
              view: scratch.leftAnchors,
              type: 'atomic<u32>',
              access: 'read_write'
            },
            {
              name: 'rightAnchors',
              view: scratch.rightAnchors,
              type: 'atomic<u32>',
              access: 'read_write'
            },
            bestKeys,
            bestRows,
            {name: 'importance', view: props.importance, type: 'u32', access: 'read_write'},
            ...(isTimeRatio && props.timestamps
              ? [{name: 'timestamps', view: props.timestamps, type: 'f32', access: 'read'} as const]
              : [])
          ],
          declarations: `${LINE_SIMPLIFICATION_METRIC_WGSL}
const FINISH_SPAN_LIMIT: u32 = ${finishSpanLimit}u;

// Distance key of a row against the chord (left, right), as in the distance node.
fn getDistanceKey(row: u32, leftAnchor: u32, rightAnchor: u32) -> u32 {
  let distance = ${getDistanceExpression('row', 'leftAnchor', 'rightAnchor')};
  return select(bitcast<u32>(distance), 0u, distance <= 0.0);
}

// Decides every interior row of the open interval (first, last) with the recursion the rounds
// would run, depth first from one lane. Importance is min(key, parent) where parent is the
// importance of the split that created the interval; both children of a split row inherit that
// row's importance as parent because it never exceeds either anchor's.
fn finishInterval(first: u32, last: u32) {
  lineOpaqueZero = first >> 31u;
  var stackLeft: array<u32, ${finishSpanLimit}>;
  var stackRight: array<u32, ${finishSpanLimit}>;
  var stackParent: array<u32, ${finishSpanLimit}>;
  stackLeft[0] = first;
  stackRight[0] = last;
  stackParent[0] = min(importance[importanceOffset + first], importance[importanceOffset + last]);
  var depth = 1u;
  while (depth > 0u) {
    depth--;
    let left = stackLeft[depth];
    let right = stackRight[depth];
    let parent = stackParent[depth];
    var bestKey = 0u;
    var bestRow = 0xffffffffu;
    for (var row = left + 1u; row < right; row++) {
      let key = getDistanceKey(row, left, right);
      // Strict comparison keeps the smallest row on ties.
      if (key > bestKey) {
        bestKey = key;
        bestRow = row;
      }
    }
    if (bestKey == 0u) {
      for (var row = left + 1u; row < right; row++) {
        importance[importanceOffset + row] = 0u;
        atomicStore(&leftAnchors[leftAnchorsOffset + row], row);
        atomicStore(&rightAnchors[rightAnchorsOffset + row], row);
      }
      continue;
    }
    let splitImportance = min(bestKey, parent);
    importance[importanceOffset + bestRow] = splitImportance;
    atomicStore(&leftAnchors[leftAnchorsOffset + bestRow], bestRow);
    atomicStore(&rightAnchors[rightAnchorsOffset + bestRow], bestRow);
    if (bestRow > left + 1u) {
      stackLeft[depth] = left;
      stackRight[depth] = bestRow;
      stackParent[depth] = splitImportance;
      depth++;
    }
    if (right > bestRow + 1u) {
      stackLeft[depth] = bestRow;
      stackRight[depth] = right;
      stackParent[depth] = splitImportance;
      depth++;
    }
  }
}`,
          // Atomic loads and stores make the finishing lane's writes to the anchors of its own
          // interval defined for the other lanes of that interval, which read them concurrently.
          body: `let leftAnchor = atomicLoad(&leftAnchors[leftAnchorsOffset + index]);
  let rightAnchor = atomicLoad(&rightAnchors[rightAnchorsOffset + index]);
  if (!(leftAnchor < index && index < rightAnchor)) {
    return;
  }
  if (rightAnchor - leftAnchor - 1u <= FINISH_SPAN_LIMIT) {
    // The first interior row finishes the whole interval; its siblings have nothing to reset.
    if (index == leftAnchor + 1u) {
      finishInterval(leftAnchor, rightAnchor);
    }
    return;
  }
  atomicStore(&bestKeys[bestKeysOffset + leftAnchor], 0u);
  atomicStore(&bestRows[bestRowsOffset + leftAnchor], 0xffffffffu);`
        })
      : createWGSLKernelNode<Parameters>(graph, {
          ...gated,
          id: `${props.id}-reset`,
          variant: 'importance-reset',
          bindings: [...anchors, bestKeys, bestRows],
          body: `${UNDECIDED_WGSL('index')}
  atomicStore(&bestKeys[bestKeysOffset + leftAnchor], 0u);
  atomicStore(&bestRows[bestRowsOffset + leftAnchor], 0xffffffffu);`
        });

  return [
    resetNode,
    createWGSLKernelNode<Parameters>(graph, {
      ...gated,
      id: `${props.id}-distance`,
      variant: `importance-distance-${props.metric}`,
      bindings: distanceBindings,
      declarations: LINE_SIMPLIFICATION_METRIC_WGSL,
      body: `${UNDECIDED_WGSL('index')}
  lineOpaqueZero = leftAnchor >> 31u;
  let distance = ${distanceExpression};
  // Canonical +0 so equal distances always have equal bits.
  let key = select(bitcast<u32>(distance), 0u, distance <= 0.0);
  distanceKeys[distanceKeysOffset + index] = key;
  atomicMax(&bestKeys[bestKeysOffset + leftAnchor], key);`
    }),
    createWGSLKernelNode<Parameters>(graph, {
      ...gated,
      id: `${props.id}-argmax`,
      variant: 'importance-argmax',
      bindings: [
        ...anchors,
        {
          name: 'distanceKeys',
          view: scratch.distanceKeys,
          type: 'u32',
          access: 'read'
        },
        bestKeys,
        bestRows
      ],
      body: `${UNDECIDED_WGSL('index')}
  if (distanceKeys[distanceKeysOffset + index] == atomicLoad(&bestKeys[bestKeysOffset + leftAnchor])) {
    atomicMin(&bestRows[bestRowsOffset + leftAnchor], index);
  }`
    }),
    createWGSLKernelNode<Parameters>(graph, {
      ...gated,
      id: `${props.id}-split`,
      variant: 'importance-split',
      bindings: [
        {
          name: 'leftAnchors',
          view: scratch.leftAnchors,
          type: 'u32',
          access: 'read_write'
        },
        {
          name: 'rightAnchors',
          view: scratch.rightAnchors,
          type: 'u32',
          access: 'read_write'
        },
        {
          name: 'distanceKeys',
          view: scratch.distanceKeys,
          type: 'u32',
          access: 'read'
        },
        bestKeys,
        bestRows,
        {
          name: 'importance',
          view: props.importance,
          type: 'u32',
          access: 'read_write'
        },
        {
          name: 'status',
          view: props.status,
          type: 'atomic<u32>',
          access: 'read_write'
        }
      ],
      body: `${UNDECIDED_WGSL('index')}
  if (atomicLoad(&bestKeys[bestKeysOffset + leftAnchor]) == 0u) {
    importance[importanceOffset + index] = 0u;
    leftAnchors[leftAnchorsOffset + index] = index;
    rightAnchors[rightAnchorsOffset + index] = index;
    return;
  }
  let splitRow = atomicLoad(&bestRows[bestRowsOffset + leftAnchor]);
  if (index == splitRow) {
    // Non-negative f32 bits order like their values, so u32 min is f32 min.
    let parentImportance = min(
      importance[importanceOffset + leftAnchor],
      importance[importanceOffset + rightAnchor]
    );
    importance[importanceOffset + index] = min(distanceKeys[distanceKeysOffset + index], parentImportance);
    leftAnchors[leftAnchorsOffset + index] = index;
    rightAnchors[rightAnchorsOffset + index] = index;
    return;
  }
  if (index < splitRow) {
    rightAnchors[rightAnchorsOffset + index] = splitRow;
  } else {
    leftAnchors[leftAnchorsOffset + index] = splitRow;
  }
  atomicStore(&status[statusOffset], 1u);`
    })
  ];
}

/**
 * Assigns every row still undecided after the round cap the importance of the split that created
 * its interval, so the kept set at any tolerance is a superset of Douglas-Peucker's.
 *
 * @internal
 */
export function createLineImportanceUnresolvedNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    scratch: LineImportanceScratch;
    importance: GraphDataView<'float32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'importance-unresolved',
    bindings: [
      {
        name: 'leftAnchors',
        view: props.scratch.leftAnchors,
        type: 'u32',
        access: 'read'
      },
      {
        name: 'rightAnchors',
        view: props.scratch.rightAnchors,
        type: 'u32',
        access: 'read'
      },
      {
        name: 'importance',
        view: props.importance,
        type: 'u32',
        access: 'read_write'
      }
    ],
    invocationCount: props.importance.length,
    body: `${UNDECIDED_WGSL('index')}
  importance[importanceOffset + index] = min(
    importance[importanceOffset + leftAnchor],
    importance[importanceOffset + rightAnchor]
  );`
  });
}

/** Writes the per-frame keep mask and the row-ID column compacted by `GPUCompaction`. @internal */
export function createLineKeepMaskNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    trackOffsets: GraphDataView<'uint32'>;
    importance: GraphDataView<'float32'>;
    parameters: GraphDataView<'float32'>;
    keepMask: GraphDataView<'uint32'>;
    rowIds: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  const lineCount = props.trackOffsets.length - 1;
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'keep-mask',
    bindings: [
      {
        name: 'trackOffsets',
        view: props.trackOffsets,
        type: 'u32',
        access: 'read'
      },
      {
        name: 'importance',
        view: props.importance,
        type: 'f32',
        access: 'read'
      },
      {
        name: 'parameters',
        view: props.parameters,
        type: 'f32',
        access: 'read'
      },
      {
        name: 'keepMask',
        view: props.keepMask,
        type: 'u32',
        access: 'read_write'
      },
      {name: 'rowIds', view: props.rowIds, type: 'u32', access: 'read_write'}
    ],
    invocationCount: props.importance.length,
    declarations: `const LINE_COUNT: u32 = ${lineCount}u;`,
    body: `rowIds[rowIdsOffset + index] = index;
  let tolerance = parameters[parametersOffset];
  let inLine = index >= trackOffsets[trackOffsetsOffset] &&
    index < trackOffsets[trackOffsetsOffset + LINE_COUNT];
  let keep = inLine && importance[importanceOffset + index] > tolerance;
  keepMask[keepMaskOffset + index] = select(0u, 1u, keep);`
  });
}

/**
 * Per line, finds the kept rows in the ascending compacted row list by binary search and writes
 * their count and first position. Uses the full-size compaction, so counts are not clamped.
 *
 * @internal
 */
export function createLineKeptRangesNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    trackOffsets: GraphDataView<'uint32'>;
    compactRows: GraphDataView<'uint32'>;
    totalCount: GraphDataView<'uint32'>;
    lineCounts?: GraphDataView<'uint32'>;
    lineStarts?: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  const lineCount = props.trackOffsets.length - 1;
  const bindings: WGSLKernelBinding[] = [
    {
      name: 'trackOffsets',
      view: props.trackOffsets,
      type: 'u32',
      access: 'read'
    },
    {
      name: 'compactRows',
      view: props.compactRows,
      type: 'u32',
      access: 'read'
    },
    {name: 'totalCount', view: props.totalCount, type: 'u32', access: 'read'}
  ];
  if (props.lineCounts) {
    bindings.push({
      name: 'lineCounts',
      view: props.lineCounts,
      type: 'u32',
      access: 'read_write'
    });
  }
  if (props.lineStarts) {
    bindings.push({
      name: 'lineStarts',
      view: props.lineStarts,
      type: 'u32',
      access: 'read_write'
    });
  }
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'kept-ranges',
    bindings,
    invocationCount: lineCount,
    declarations: `const COMPACT_CAPACITY: u32 = ${props.compactRows.length}u;

fn lowerBoundKeptRow(row: u32, count: u32) -> u32 {
  var low = 0u;
  var high = count;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (compactRows[compactRowsOffset + middle] < row) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  return low;
}`,
    body: `let keptCount = min(totalCount[totalCountOffset], COMPACT_CAPACITY);
  let lineStart = trackOffsets[trackOffsetsOffset + index];
  let lineEnd = max(trackOffsets[trackOffsetsOffset + index + 1u], lineStart);
  let first = lowerBoundKeptRow(lineStart, keptCount);
  let last = lowerBoundKeptRow(lineEnd, keptCount);
  ${props.lineCounts ? 'lineCounts[lineCountsOffset + index] = last - first;' : ''}
  ${props.lineStarts ? 'lineStarts[lineStartsOffset + index] = first;' : ''}`
  });
}
