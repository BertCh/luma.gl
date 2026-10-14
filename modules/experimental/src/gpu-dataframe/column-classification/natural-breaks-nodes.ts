// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {
  createWGSLKernelNode,
  getAutomaticWGSLKernelWorkgroupSize,
  getWGSLFloatLiteral,
  type WGSLKernelBinding
} from '../../utils/wgsl-kernel-nodes';
import {
  CLASS_BREAKS_BIN_WGSL,
  CLASS_BREAKS_EXTREMES,
  CLASS_BREAKS_STATE,
  CLASS_BREAKS_WIDE_WGSL,
  getClassBreaksDispatchWGSL,
  getClassBreaksIndirectGate,
  NATURAL_BREAKS_CELL_COUNT,
  NATURAL_BREAKS_ELEMENT_STRIDE,
  NATURAL_BREAKS_MAXIMUM_ROUND_COUNT,
  NATURAL_BREAKS_QUANTIZATION_BITS,
  NATURAL_BREAKS_SETTLED_ZOOM_LEVEL,
  NATURAL_BREAKS_SUBBIN_COUNT,
  NATURAL_BREAKS_ZOOM_CELLS
} from './class-breaks-kernels';
import {GPU_CLASS_BREAKS_METHOD_CODES} from './class-breaks-parameters';
import {COLUMN_ORDERED_KEY_WGSL} from './column-classification-shared';

/*
 * Histogram approximation to Fisher-Jenks on the GPU, without readback or a B x B table.
 *
 * Fisher-Jenks cuts the sorted values into k classes of least total within-class squared
 * deviation SSD = sum(x^2) - sum(x)^2 / n. The pipeline is a coarse exact dynamic program on a
 * histogram, then a local refinement on the data itself:
 *
 * 1. COARSE HISTOGRAM. B equal-width bins of [min, max]. A row at bin b and fraction f of it gets
 *    the integer position b * 256 + floor(f * 256) (below 2^19 for B <= 2048); each bin sums
 *    count, position and position^2 as 64-bit integers (two u32 atomics each, carry by compare),
 *    so the bin carries its true spread, not a point mass at its centre.
 *
 * 2. PREFIX SUMS AND COSTS. Exact 64-bit prefix sums of those three quantities give the cost of
 *    any run of bins in O(1) by `getClassCost` (see class-breaks-kernels.ts). SSD is a difference
 *    of large sums, which cancels catastrophically in f32 (and an f64 emulation by error-free
 *    transforms is unsafe where Metal reassociates), so the cost is formed from exact integers:
 *    recentre on the integer nearest the mean, which is exact modulo 2^64, and subtract a small
 *    f32 correction. There is no cost table.
 *
 * 3. MONOTONE DIVIDE AND CONQUER. With T_c(j) the best cost of bins 0..j in c + 1 classes, the
 *    last class starts at opt_c(j), which is non-decreasing in j because the SSD of runs satisfies
 *    the quadrangle inequality. One workgroup computes a whole layer in log2(B) barrier-separated
 *    levels of the implicit split tree: at level l the mids are the odd multiples of the stride
 *    2^(L-1-l), and a mid's candidates are bounded by the optima of its two already solved
 *    neighbours. A level has O(B) candidate evaluations in all, shared by teams of lanes (a team
 *    per node while there are at most 256 nodes, a thread per node after), so a layer is one
 *    dispatch of O(B log B) work: O(k B log B) in all. Ties keep the lowest start, so plateaus
 *    of empty bins resolve to the same data partition.
 *
 * 3b. COLLAPSED BINS. Equal-width bins cannot tell structure apart that is narrower than a bin:
 *    with one extreme outlier the whole bulk sits in one bin, the zero-cost partitions of the
 *    dynamic program are a plateau, and its cuts land in empty bins. Empty classes after the
 *    first stage are that symptom. The class with most rows then takes the classes the empty ones
 *    would have held: the whole coarse stage (histogram, prefix sums, layers, backtrack) runs a
 *    second time over the range of that class, on the same pipelines (a switch kernel points the
 *    shared state words and extremes at the class, and switches the stage off when there is
 *    nothing to split), and an assemble kernel restores them and merges the cuts. Without it the
 *    refinement alone settles in a local optimum for the long chains of cuts that a collapsed
 *    bulk leaves (seven classes and an outlier of 1e5 times the spread).
 *
 * 4. REFINEMENT. The coarse cuts are bin edges and the bins cannot see structure finer than
 *    range / B. Each break is therefore moved onto the data: red-black coordinate descent in which
 *    every break, with its two neighbours fixed, picks the best split of the values in its two
 *    adjacent classes. Odd breaks update together and then even breaks, so simultaneous updates
 *    never interact (a break's cost involves only its two adjacent classes) and the total cost can
 *    only fall; a plain Jacobi sweep gives no such guarantee. One half-round is one pass over the
 *    rows:
 *    - The window of a break is centred on its cut (clamped into the extent of its two classes,
 *      which the pass tracks per class) and holds 1024 cells. Rows outside the window are
 *      accumulated in two blocks, left and right. The first window covers the whole extent, so the
 *      window can see every single-cut move, but the result still depends on the coarse cuts:
 *      coordinate descent cannot escape a strict local minimum requiring joint cut moves; once the break and its neighbours stop moving, the window
 *      shrinks by 128 per round around the chosen boundary until a cell is below an f32 ulp.
 *    - Each cell sums count, q and q^2 of the fixed-point offset q = round(2^18 * (x - anchor) /
 *      span) as 64-bit integers, and the smallest and largest ordered key. Costs of all 1025 cell
 *      boundaries come from running sums and the same exact `getClassCost`.
 *    - The best boundary decides the partition; the cut becomes the smallest data value on its
 *      right (the minimum ordered key of the right cells), so every break ends on a data value.
 *      A boundary only replaces the current one when its cost is strictly lower, so ties never
 *      oscillate.
 *    - Rounds stop (a GPU state word disarms the remaining indirect dispatches) after a round in
 *      which no cut moved while every break had zoomed twice, or after the eighth round.
 *
 * 5. FINAL CLASSES. A cut whose class holds no value is dropped, so a class count above the
 *    number of distinct values yields one class per distinct value (mapclassify and jenkspy
 *    refuse that request).
 *
 * Edge convention: mapclassify bins are upper bounds b[i], the largest value of class i; edges
 * here are lower edges e[i], the smallest value of class i, i.e. the smallest data value above
 * b[i-1]. The refinement produces e[i] directly.
 *
 * Rules 12 and 13: no loop index reaches WGSL text. The DP layer and the refinement half-round
 * are GPU state words, advanced by one-thread nodes that also write the indirect dispatches of
 * the gated nodes, so every layer and every round shares its pipelines and launches nothing once
 * done.
 */

const CODES = GPU_CLASS_BREAKS_METHOD_CODES;
const LARGE_COST = '3.0e38';
const NO_MINIMUM_KEY = '0xffffffffu';
const SETTLED_UNZOOMED_LEVEL = 99;
/** Half-rounds: odd breaks and even breaks of every round. */
const MAXIMUM_HALF_ROUND_COUNT = 2 * NATURAL_BREAKS_MAXIMUM_ROUND_COUNT;
const WORKGROUP_SIZE = 256;

/** Statements adding a 64-bit `(low, high)` value to two atomic words, carrying by comparison. */
function getAtomicAddWideWGSL(
  array: string,
  lowIndex: string,
  highIndex: string,
  low: string,
  high: string
): string {
  return `{
    let previousLow = atomicAdd(&${array}[${lowIndex}], ${low});
    let carry = select(0u, 1u, previousLow + ${low} < previousLow);
    let highPart = ${high} + carry;
    if (highPart != 0u) {
      atomicAdd(&${array}[${highIndex}], highPart);
    }
  }`;
}

/** Views and ids shared with the rest of `GPUClassBreaks`. */
export type NaturalBreaksNodesProps = {
  id: string;
  operation: string;
  /** The float32 view of the values (converted when the column is integer). */
  values: GraphDataView<'float32'>;
  mask?: GraphDataView<'uint32'>;
  state: GraphDataView<'uint32'>;
  extremes: GraphDataView<'uint32'>;
  edges: GraphDataView<'float32'>;
  edgeState: GraphDataView<'uint32'>;
  maximumClassCount: number;
  binCount: number;
};

/**
 * Returns the nodes of the natural-breaks method: histogram, prefix sums, dynamic program layers,
 * backtrack, refinement rounds, and the removal of empty classes. Every node is a no-op unless
 * the per-frame method is `'natural-breaks'`.
 *
 * @internal
 */
export function getNaturalBreaksNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: NaturalBreaksNodesProps
): GPUCommandNode<Parameters>[] {
  const {operation, values, mask, maximumClassCount, binCount} = props;
  const prefix = `${props.id}-natural-breaks`;
  const rowCount = values.length;
  const u32 = (name: string, length: number) =>
    createTransientView(graph, `${prefix}-${name}`, 'uint32', Math.max(length, 1));
  const f32 = (name: string, length: number) =>
    createTransientView(graph, `${prefix}-${name}`, 'float32', Math.max(length, 1));
  const histogram = u32('histogram', 5 * binCount);
  const prefixSums = u32('prefix', 5 * (binCount + 1));
  const totals = f32('totals', maximumClassCount * binCount);
  const starts = u32('starts', maximumClassCount * binCount);
  const stateRead = {name: 'state', view: props.state, type: 'u32', access: 'read'} as const;
  const maskBinding = mask
    ? ([{name: 'rowMask', view: mask, type: 'u32', access: 'read'}] as const)
    : [];
  const maskTest = mask ? 'rowMask[rowMaskOffset + index] != 0u' : 'true';
  const gate = `state[stateOffset + ${CLASS_BREAKS_STATE.method}u] != ${CODES['natural-breaks']}u`;
  const binDeclarations = `${COLUMN_ORDERED_KEY_WGSL}
const BIN_COUNT: u32 = ${binCount}u;
const INVERSE_BIN_COUNT: f32 = ${getWGSLFloatLiteral(1 / binCount)};
${CLASS_BREAKS_BIN_WGSL}`;
  const extremesRead = `let minimum = decodeOrderedKey(extremes[extremesOffset + ${CLASS_BREAKS_EXTREMES.minimumKey}u]);
  let maximum = decodeOrderedKey(extremes[extremesOffset + ${CLASS_BREAKS_EXTREMES.maximumKey}u]);
  let width = (maximum - minimum) * INVERSE_BIN_COUNT;`;
  const nodes: GPUCommandNode<Parameters>[] = [];

  /**
   * The coarse stage: histogram, prefix sums and dynamic program layers, ending with the class
   * starts of the partition in `starts`. Run once over the whole column and, when empty classes
   * show that the bins collapsed the data, once more over the range of the most populated class
   * (same pipelines, other state words and extremes), so `stage` only names the nodes.
   */
  const getCoarseNodes = (stage: string): GPUCommandNode<Parameters>[] => {
    const stagePrefix = `${prefix}${stage}`;
    const nodes: GPUCommandNode<Parameters>[] = [];
    // Coarse histogram: count, and 64-bit position sums and square sums, of every bin.
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${stagePrefix}-clear`,
        operation,
        variant: 'natural-breaks-clear',
        bindings: [
          stateRead,
          {name: 'histogram', view: histogram, type: 'u32', access: 'read_write'}
        ],
        invocationCount: 5 * binCount,
        body: `if (${gate}) {
      return;
    }
    histogram[histogramOffset + index] = 0u;`
      })
    );
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${stagePrefix}-histogram`,
        operation,
        variant: 'natural-breaks-histogram',
        bindings: [
          {name: 'values', view: values, type: 'f32', access: 'read'},
          ...maskBinding,
          stateRead,
          {name: 'extremes', view: props.extremes, type: 'u32', access: 'read'},
          {name: 'histogram', view: histogram, type: 'atomic<u32>', access: 'read_write'}
        ],
        invocationCount: rowCount,
        declarations: `${binDeclarations}
  ${CLASS_BREAKS_WIDE_WGSL}`,
        body: `if (${gate}) {
      return;
    }
    let value = values[valuesOffset + index];
    if ((bitcast<u32>(value) & 0x7f800000u) == 0x7f800000u) {
      return;
    }
    ${mask ? 'if (rowMask[rowMaskOffset + index] == 0u) {\n    return;\n  }' : ''}
    // Rows outside [minimum, maximum] belong to other classes when the stage runs over one.
    let key = getOrderedKey(value);
    if (key < extremes[extremesOffset + ${CLASS_BREAKS_EXTREMES.minimumKey}u] || key > extremes[extremesOffset + ${CLASS_BREAKS_EXTREMES.maximumKey}u]) {
      return;
    }
    ${extremesRead}
    var bin = 0u;
    var offset = 0u;
    if (width > 0.0) {
      bin = getBinIndex(value, minimum, width);
      // Position inside the bin in 1/${NATURAL_BREAKS_SUBBIN_COUNT} bins.
      let remainder = (value - minimum) - f32(bin) * width;
      offset = u32(clamp(floor(remainder * (${NATURAL_BREAKS_SUBBIN_COUNT}.0 / width)), 0.0, ${NATURAL_BREAKS_SUBBIN_COUNT - 1}.0));
    }
    let position = bin * ${NATURAL_BREAKS_SUBBIN_COUNT}u + offset;
    let square = multiplyWide(position, position);
    atomicAdd(&histogram[histogramOffset + bin], 1u);
    ${getAtomicAddWideWGSL('histogram', `histogramOffset + ${binCount}u + bin`, `histogramOffset + ${2 * binCount}u + bin`, 'position', '0u')}
    ${getAtomicAddWideWGSL('histogram', `histogramOffset + ${3 * binCount}u + bin`, `histogramOffset + ${4 * binCount}u + bin`, 'square.x', 'square.y')}`
      })
    );

    // Exact 64-bit prefix sums, one thread: B steps of a few integer operations are far cheaper
    // than the launch of a scan, and the sums must be exact, so there is no float reduction tree.
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${stagePrefix}-prefix`,
        operation,
        variant: 'natural-breaks-prefix',
        bindings: [
          stateRead,
          {name: 'histogram', view: histogram, type: 'u32', access: 'read'},
          {name: 'prefix', view: prefixSums, type: 'u32', access: 'read_write'}
        ],
        invocationCount: 1,
        declarations: `const BIN_COUNT: u32 = ${binCount}u;
  const PREFIX_STRIDE: u32 = ${binCount + 1}u;
  ${CLASS_BREAKS_WIDE_WGSL}`,
        body: `if (${gate}) {
      return;
    }
    var count = 0u;
    var sum = vec2<u32>(0u, 0u);
    var squares = vec2<u32>(0u, 0u);
    for (var bin = 0u; bin < BIN_COUNT; bin = bin + 1u) {
      prefix[prefixOffset + bin] = count;
      prefix[prefixOffset + PREFIX_STRIDE + bin] = sum.x;
      prefix[prefixOffset + 2u * PREFIX_STRIDE + bin] = sum.y;
      prefix[prefixOffset + 3u * PREFIX_STRIDE + bin] = squares.x;
      prefix[prefixOffset + 4u * PREFIX_STRIDE + bin] = squares.y;
      count = count + histogram[histogramOffset + bin];
      sum = addWide(sum, vec2<u32>(histogram[histogramOffset + BIN_COUNT + bin], histogram[histogramOffset + 2u * BIN_COUNT + bin]));
      squares = addWide(squares, vec2<u32>(histogram[histogramOffset + 3u * BIN_COUNT + bin], histogram[histogramOffset + 4u * BIN_COUNT + bin]));
    }
    prefix[prefixOffset + BIN_COUNT] = count;
    prefix[prefixOffset + PREFIX_STRIDE + BIN_COUNT] = sum.x;
    prefix[prefixOffset + 2u * PREFIX_STRIDE + BIN_COUNT] = sum.y;
    prefix[prefixOffset + 3u * PREFIX_STRIDE + BIN_COUNT] = squares.x;
    prefix[prefixOffset + 4u * PREFIX_STRIDE + BIN_COUNT] = squares.y;`
      })
    );

    // Cost of the bins [first, last] from the prefix sums, shared by the layer kernels.
    const rangeCostWGSL = `const BIN_COUNT: u32 = ${binCount}u;
  const PREFIX_STRIDE: u32 = ${binCount + 1}u;
  ${CLASS_BREAKS_WIDE_WGSL}

  fn getRangeCost(first: u32, last: u32) -> f32 {
    let end = last + 1u;
    let count = prefix[prefixOffset + end] - prefix[prefixOffset + first];
    let sum = subtractWide(
      vec2<u32>(prefix[prefixOffset + PREFIX_STRIDE + end], prefix[prefixOffset + 2u * PREFIX_STRIDE + end]),
      vec2<u32>(prefix[prefixOffset + PREFIX_STRIDE + first], prefix[prefixOffset + 2u * PREFIX_STRIDE + first])
    );
    let squares = subtractWide(
      vec2<u32>(prefix[prefixOffset + 3u * PREFIX_STRIDE + end], prefix[prefixOffset + 4u * PREFIX_STRIDE + end]),
      vec2<u32>(prefix[prefixOffset + 3u * PREFIX_STRIDE + first], prefix[prefixOffset + 4u * PREFIX_STRIDE + first])
    );
    return getClassCost(count, sum, squares);
  }`;
    const prefixRead = {name: 'prefix', view: prefixSums, type: 'u32', access: 'read'} as const;
    const totalsWrite = {name: 'totals', view: totals, type: 'f32', access: 'read_write'} as const;
    const startsWrite = {name: 'starts', view: starts, type: 'u32', access: 'read_write'} as const;

    // Layer 0: one class over bins [0, end].
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${stagePrefix}-layer-0`,
        operation,
        variant: 'natural-breaks-layer',
        bindings: [stateRead, prefixRead, totalsWrite, startsWrite],
        invocationCount: binCount,
        declarations: rangeCostWGSL,
        body: `if (${gate}) {
      return;
    }
    totals[totalsOffset + index] = getRangeCost(0u, index);
    starts[startsOffset + index] = 0u;`
      })
    );

    // Layer c: the best split of bins [0, end] into c + 1 classes by the monotone divide and
    // conquer. The layer index is `state.layerCursor`, advanced by a one-thread kernel that also
    // writes the layer kernel's indirect dispatch: zero workgroups once the per-frame class count
    // (or the method) needs no further layer. Every layer shares the same two pipelines.
    const layerDispatch = createTransientView(
      graph,
      `${stagePrefix}-layer-dispatch`,
      'uint32',
      3,
      Buffer.STORAGE | Buffer.INDIRECT
    );
    const layerCursor = `state[stateOffset + ${CLASS_BREAKS_STATE.layerCursor}u]`;
    const layerActive = `state[stateOffset + ${CLASS_BREAKS_STATE.method}u] == ${CODES['natural-breaks']}u && layer < state[stateOffset + ${CLASS_BREAKS_STATE.classCount}u]`;
    for (let layer = 1; layer < maximumClassCount; layer++) {
      const layerId = `${stagePrefix}-layer-${layer}`;
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${layerId}-advance`,
          operation,
          variant: 'natural-breaks-layer-advance',
          bindings: [
            {name: 'state', view: props.state, type: 'u32', access: 'read_write'},
            {name: 'dispatch', view: layerDispatch, type: 'u32', access: 'read_write'}
          ],
          invocationCount: 1,
          body: `let layer = ${layerCursor} + 1u;
    ${layerCursor} = layer;
    ${getClassBreaksDispatchWGSL(graph, {
      operation,
      slot: 0,
      invocationCount: WORKGROUP_SIZE,
      workgroupSize: WORKGROUP_SIZE,
      activeExpression: `(${layerActive})`
    })}`
        }),
        createWGSLKernelNode<Parameters>(graph, {
          id: layerId,
          operation,
          variant: 'natural-breaks-layer',
          bindings: [stateRead, prefixRead, totalsWrite, startsWrite],
          invocationCount: WORKGROUP_SIZE,
          workgroupSize: WORKGROUP_SIZE,
          guardIndex: false,
          ...getClassBreaksIndirectGate<Parameters>(layerId, layerDispatch, 0),
          declarations: `${rangeCostWGSL}
  const WORKGROUP_SIZE: u32 = ${WORKGROUP_SIZE}u;
  // Best start of the last class for every solved end bin (opt_c). 4 B per bin: 8 KiB at 2048 bins.
  var<workgroup> optimum: array<u32, ${binCount}>;
  var<workgroup> candidateCost: array<f32, ${WORKGROUP_SIZE}>;
  var<workgroup> candidateStart: array<u32, ${WORKGROUP_SIZE}>;
  var<workgroup> sharedLayer: u32;

  // Candidates for the last class of a mid at 1-based position 'position' (end bin 'end'): from the
  // optimum of the solved neighbour on the left to that of the solved neighbour on the right.
  fn getSearchLow(layer: u32, position: u32, stride: u32, end: u32) -> u32 {
    if (position == stride) {
      return layer;
    }
    return max(optimum[end - stride], layer);
  }

  fn getSearchHigh(count: u32, position: u32, stride: u32, end: u32, low: u32) -> u32 {
    var high = BIN_COUNT - 1u;
    if (position + stride <= count) {
      high = optimum[end + stride];
    }
    // Rounding in the costs can break monotonicity by one step; never search an empty range.
    return min(end, max(high, low));
  }

  fn getCandidateCost(layer: u32, start: u32, end: u32) -> f32 {
    return totals[totalsOffset + (layer - 1u) * BIN_COUNT + start - 1u] + getRangeCost(start, end);
  }`,
          body: `let local = localInvocationIndex;
    if (local == 0u) {
      sharedLayer = ${layerCursor};
    }
    let layer = workgroupUniformLoad(&sharedLayer);
    if (layer >= BIN_COUNT) {
      return;
    }
    // Ends layer..BIN_COUNT-1 are positions 1..count of a perfect tree with 2^levels - 1 mids.
    let count = BIN_COUNT - layer;
    var levels = 0u;
    while ((1u << levels) - 1u < count) {
      levels = levels + 1u;
    }
    for (var level = 0u; level < levels; level = level + 1u) {
      let stride = 1u << (levels - 1u - level);
      let nodeCount = 1u << level;
      let teamSize = max(WORKGROUP_SIZE >> level, 1u);
      let teamCount = WORKGROUP_SIZE / teamSize;
      let team = local / teamSize;
      let lane = local % teamSize;
      if (teamSize == 1u) {
        // Deep levels: a thread per node, no cooperation and so no barrier in the loop.
        for (var node = team; node < nodeCount; node = node + teamCount) {
          let position = (2u * node + 1u) * stride;
          if (position <= count) {
            let end = layer + position - 1u;
            let low = getSearchLow(layer, position, stride, end);
            let high = getSearchHigh(count, position, stride, end, low);
            var bestCost = ${LARGE_COST};
            var bestStart = low;
            for (var start = low; start <= high; start = start + 1u) {
              let cost = getCandidateCost(layer, start, end);
              if (cost < bestCost) {
                bestCost = cost;
                bestStart = start;
              }
            }
            optimum[end] = bestStart;
            totals[totalsOffset + layer * BIN_COUNT + end] = bestCost;
            starts[startsOffset + layer * BIN_COUNT + end] = bestStart;
          }
        }
      } else {
        // Shallow levels: a team of lanes per node scans its range together and the lanes reduce
        // to the lowest start of least cost.
        let position = (2u * team + 1u) * stride;
        let valid = position <= count;
        let end = layer + position - 1u;
        var bestCost = ${LARGE_COST};
        var bestStart = 0xffffffffu;
        if (valid) {
          let low = getSearchLow(layer, position, stride, end);
          let high = getSearchHigh(count, position, stride, end, low);
          for (var start = low + lane; start <= high; start = start + teamSize) {
            let cost = getCandidateCost(layer, start, end);
            if (cost < bestCost) {
              bestCost = cost;
              bestStart = start;
            }
          }
        }
        candidateCost[local] = bestCost;
        candidateStart[local] = bestStart;
        for (var reduction = teamSize >> 1u; reduction > 0u; reduction = reduction >> 1u) {
          workgroupBarrier();
          if (lane < reduction) {
            let otherCost = candidateCost[local + reduction];
            let otherStart = candidateStart[local + reduction];
            if (otherCost < candidateCost[local] || (otherCost == candidateCost[local] && otherStart < candidateStart[local])) {
              candidateCost[local] = otherCost;
              candidateStart[local] = otherStart;
            }
          }
        }
        if (valid && lane == 0u) {
          optimum[end] = candidateStart[local];
          totals[totalsOffset + layer * BIN_COUNT + end] = candidateCost[local];
          starts[startsOffset + layer * BIN_COUNT + end] = candidateStart[local];
        }
      }
      // The next level reads the optima of this one.
      workgroupBarrier();
    }`
        })
      );
    }
    return nodes;
  };

  nodes.push(...getCoarseNodes(''));
  nodes.push(
    ...getRefinementNodes(graph, props, {
      prefix,
      u32,
      f32,
      starts,
      getCoarseNodes,
      gate,
      binDeclarations,
      extremesRead,
      maskBinding,
      maskTest
    })
  );
  return nodes;
}

/** Everything the refinement nodes take from the coarse stage. */
type RefinementContext<Parameters> = {
  prefix: string;
  u32: (name: string, length: number) => GraphDataView<'uint32'>;
  f32: (name: string, length: number) => GraphDataView<'float32'>;
  starts: GraphDataView<'uint32'>;
  getCoarseNodes: (stage: string) => GPUCommandNode<Parameters>[];
  gate: string;
  binDeclarations: string;
  extremesRead: string;
  maskBinding: readonly WGSLKernelBinding[];
  maskTest: string;
};

/**
 * Backtrack of the coarse partition into cuts, class extents, and the refinement half-rounds.
 */
function getRefinementNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: NaturalBreaksNodesProps,
  context: RefinementContext<Parameters>
): GPUCommandNode<Parameters>[] {
  const {operation, values, maximumClassCount, binCount, edges, edgeState} = props;
  const {prefix, gate, binDeclarations, extremesRead, maskBinding, maskTest} = context;
  // A class count above the bin count cannot be produced; every kernel clamps it the same way.
  const classCountRead = `min(state[stateOffset + ${CLASS_BREAKS_STATE.classCount}u], BIN_COUNT)`;
  const binCountConstant = `const BIN_COUNT: u32 = ${binCount}u;`;
  const rowCount = values.length;
  const cellCount = NATURAL_BREAKS_CELL_COUNT;
  const elementStride = NATURAL_BREAKS_ELEMENT_STRIDE;
  const elementCount = cellCount + 2;
  const breakCount = Math.max(maximumClassCount - 1, 1);
  const methodRead = `state[stateOffset + ${CLASS_BREAKS_STATE.method}u]`;
  const stateRead = {name: 'state', view: props.state, type: 'u32', access: 'read'} as const;
  const nodes: GPUCommandNode<Parameters>[] = [];

  // Per-class extents (smallest key, largest key, row count), per-break window state
  // [halfWidth, movement, zoomLevel] x (maximumClassCount + 1), per-break plan [anchor, cellWidth,
  // inverseUnit] x (maximumClassCount + 1), and the cells of every break. The second coarse stage
  // keeps its cuts in `bulkEdges` and the state it overwrote in `backup`: [method, classCount,
  // minimumKey, maximumKey, bulk class, active, empty class count, unused].
  const extents = context.u32('extents', 3 * maximumClassCount);
  const bulkEdges = context.f32('bulk-edges', maximumClassCount + 1);
  const backup = context.u32('backup', 8);
  const breakState = context.u32('break-state', 3 * (maximumClassCount + 1));
  const plan = context.f32('plan', 3 * (maximumClassCount + 1));
  const cells = context.u32('cells', breakCount * elementCount * elementStride);
  // Indirect dispatch triples: 0 the extents pass, 5 the extents pass over the partition of the
  // second coarse stage, then per half-round 1 the plan, 2 the cell clear, 3 the row pass, 4 the
  // solve. The backtrack writes slot 0, the assemble kernel slot 5 and the advance kernels 1-4.
  const dispatch = createTransientView(
    graph,
    `${prefix}-refinement-dispatch`,
    'uint32',
    18,
    Buffer.STORAGE | Buffer.INDIRECT
  );
  const cellWordCount = breakCount * elementCount * elementStride;
  const rowWorkgroupSize = getAutomaticWGSLKernelWorkgroupSize(rowCount);
  const breakWorkgroupSize = getAutomaticWGSLKernelWorkgroupSize(breakCount);
  const cellWorkgroupSize = getAutomaticWGSLKernelWorkgroupSize(cellWordCount);
  const maximumClassCountConstant = `const MAXIMUM_CLASS_COUNT: u32 = ${maximumClassCount}u;
${binCountConstant}`;
  const breakStateLayout = `const HALF_WIDTH_BASE: u32 = 0u;
const MOVEMENT_BASE: u32 = ${maximumClassCount + 1}u;
const LEVEL_BASE: u32 = ${2 * (maximumClassCount + 1)}u;`;
  const cellLayout = `const CELL_COUNT: u32 = ${cellCount}u;
const ELEMENT_COUNT: u32 = ${elementCount}u;
const ELEMENT_STRIDE: u32 = ${elementStride}u;
const NO_MINIMUM: u32 = ${NO_MINIMUM_KEY};`;
  const quantization = 2 ** NATURAL_BREAKS_QUANTIZATION_BITS;

  // Backtrack the coarse starts into cut values (bin edges), reset the class extents and arm the
  // extents pass. The dispatch is written before the method gate so a stale arm never survives a
  // frame that uses another method.
  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${prefix}-backtrack`,
      operation,
      variant: 'natural-breaks-backtrack',
      bindings: [
        stateRead,
        {name: 'extremes', view: props.extremes, type: 'u32', access: 'read'},
        {name: 'starts', view: context.starts, type: 'u32', access: 'read'},
        {name: 'edges', view: edges, type: 'f32', access: 'read_write'},
        {name: 'edgeState', view: edgeState, type: 'u32', access: 'read_write'},
        {name: 'extents', view: extents, type: 'u32', access: 'read_write'},
        {name: 'dispatch', view: dispatch, type: 'u32', access: 'read_write'}
      ],
      invocationCount: 1,
      declarations: `${binDeclarations}
const MAXIMUM_CLASS_COUNT: u32 = ${maximumClassCount}u;`,
      body: `let classCount = ${classCountRead};
  let isActive = !(${gate}) && classCount >= 2u && extremes[extremesOffset + ${CLASS_BREAKS_EXTREMES.minimumKey}u] < extremes[extremesOffset + ${CLASS_BREAKS_EXTREMES.maximumKey}u];
  ${getClassBreaksDispatchWGSL(graph, {
    operation,
    slot: 0,
    invocationCount: rowCount,
    workgroupSize: rowWorkgroupSize,
    activeExpression: 'isActive'
  })}
  if (${gate}) {
    return;
  }
  ${extremesRead}
  var end = BIN_COUNT - 1u;
  for (var layer = classCount - 1u; layer > 0u; layer = layer - 1u) {
    let start = starts[startsOffset + layer * BIN_COUNT + end];
    // The class starting at bin 'start' begins at that bin's lower edge.
    edges[edgesOffset + layer] = minimum + f32(start) * width;
    end = start - 1u;
  }
  edgeState[edgeStateOffset] = classCount;
  for (var index = 0u; index < MAXIMUM_CLASS_COUNT; index = index + 1u) {
    extents[extentsOffset + index] = 0xffffffffu;
    extents[extentsOffset + MAXIMUM_CLASS_COUNT + index] = 0u;
    extents[extentsOffset + 2u * MAXIMUM_CLASS_COUNT + index] = 0u;
  }`
    })
  );

  if (maximumClassCount < 2) {
    return nodes;
  }

  // Extents pass: the smallest and largest value and the row count of every class of a partition.
  const createExtentsNode = (id: string, slot: number) =>
    createWGSLKernelNode<Parameters>(graph, {
      id,
      operation,
      variant: 'natural-breaks-extents',
      bindings: [
        {name: 'values', view: values, type: 'f32', access: 'read'},
        ...maskBinding,
        stateRead,
        {name: 'edges', view: edges, type: 'f32', access: 'read'},
        {name: 'extents', view: extents, type: 'atomic<u32>', access: 'read_write'}
      ],
      invocationCount: rowCount,
      workgroupSize: rowWorkgroupSize,
      ...getClassBreaksIndirectGate<Parameters>(id, dispatch, slot),
      declarations: `${COLUMN_ORDERED_KEY_WGSL}
${maximumClassCountConstant}`,
      body: `let value = values[valuesOffset + index];
  if ((bitcast<u32>(value) & 0x7f800000u) == 0x7f800000u || !(${maskTest})) {
    return;
  }
  let classCount = ${classCountRead};
  let key = getOrderedKey(value);
  // The class is the number of cuts e[1..classCount-1] at or below the key.
  var low = 1u;
  var high = classCount;
  while (low < high) {
    let middle = (low + high) >> 1u;
    if (getOrderedKey(edges[edgesOffset + middle]) <= key) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  atomicMin(&extents[extentsOffset + low - 1u], key);
  atomicMax(&extents[extentsOffset + MAXIMUM_CLASS_COUNT + low - 1u], key);
  atomicAdd(&extents[extentsOffset + 2u * MAXIMUM_CLASS_COUNT + low - 1u], 1u);`
    });

  // Every non-empty class starts at its smallest value (cuts become data values), and the window
  // state of every break is reset: no window yet, neighbours assumed to have moved a lot.
  const createSnapNode = (id: string) =>
    createWGSLKernelNode<Parameters>(graph, {
      id,
      operation,
      variant: 'natural-breaks-snap',
      bindings: [
        stateRead,
        {name: 'extents', view: extents, type: 'u32', access: 'read'},
        {name: 'edges', view: edges, type: 'f32', access: 'read_write'},
        {name: 'breakState', view: breakState, type: 'u32', access: 'read_write'}
      ],
      invocationCount: 1,
      declarations: `${COLUMN_ORDERED_KEY_WGSL}
${maximumClassCountConstant}
${breakStateLayout}`,
      body: `if (${gate}) {
    return;
  }
  let classCount = ${classCountRead};
  for (var cut = 1u; cut < classCount; cut = cut + 1u) {
    let smallest = extents[extentsOffset + cut];
    if (smallest != 0xffffffffu) {
      edges[edgesOffset + cut] = decodeOrderedKey(smallest);
    }
  }
  for (var index = 0u; index <= MAXIMUM_CLASS_COUNT; index = index + 1u) {
    breakState[breakStateOffset + HALF_WIDTH_BASE + index] = 0u;
    breakState[breakStateOffset + MOVEMENT_BASE + index] = bitcast<u32>(${LARGE_COST});
    breakState[breakStateOffset + LEVEL_BASE + index] = 0u;
  }
  // No break lies beside the first and last class.
  breakState[breakStateOffset + MOVEMENT_BASE] = 0u;
  breakState[breakStateOffset + MOVEMENT_BASE + classCount] = 0u;`
    });
  nodes.push(createExtentsNode(`${prefix}-extents`, 0));
  nodes.push(createSnapNode(`${prefix}-snap`));

  // Empty classes after the first coarse stage mean the bins collapsed structure inside the most
  // populated class (a single extreme outlier puts the whole bulk in one bin). That class is
  // split by the coarse stage run again over its own range, with the classes the empty ones would
  // have held: the switch kernel points the shared state words and extremes at the class (and
  // switches the stage off otherwise), the second stage runs on the same pipelines, and the
  // assemble kernel restores everything and merges the cuts. The partition changed, so the
  // extents are measured again.
  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${prefix}-bulk-switch`,
      operation,
      variant: 'natural-breaks-bulk-switch',
      bindings: [
        {name: 'state', view: props.state, type: 'u32', access: 'read_write'},
        {name: 'extremes', view: props.extremes, type: 'u32', access: 'read_write'},
        {name: 'extents', view: extents, type: 'u32', access: 'read'},
        {name: 'backup', view: backup, type: 'u32', access: 'read_write'}
      ],
      invocationCount: 1,
      declarations: `${maximumClassCountConstant}
const NO_MINIMUM: u32 = ${NO_MINIMUM_KEY};`,
      body: `let classCount = ${classCountRead};
  let minimumKey = extremes[extremesOffset + ${CLASS_BREAKS_EXTREMES.minimumKey}u];
  let maximumKey = extremes[extremesOffset + ${CLASS_BREAKS_EXTREMES.maximumKey}u];
  backup[backupOffset] = ${methodRead};
  backup[backupOffset + 1u] = state[stateOffset + ${CLASS_BREAKS_STATE.classCount}u];
  backup[backupOffset + 2u] = minimumKey;
  backup[backupOffset + 3u] = maximumKey;
  backup[backupOffset + 4u] = 0u;
  backup[backupOffset + 5u] = 0u;
  backup[backupOffset + 6u] = 0u;
  var bulk = 0u;
  var bulkRows = 0u;
  var nonEmpty = 0u;
  for (var index = 0u; index < classCount; index = index + 1u) {
    if (extents[extentsOffset + index] != NO_MINIMUM) {
      nonEmpty = nonEmpty + 1u;
      let rows = extents[extentsOffset + 2u * MAXIMUM_CLASS_COUNT + index];
      if (rows > bulkRows) {
        bulkRows = rows;
        bulk = index;
      }
    }
  }
  let emptyCount = classCount - nonEmpty;
  let bulkLow = extents[extentsOffset + bulk];
  let bulkHigh = extents[extentsOffset + MAXIMUM_CLASS_COUNT + bulk];
  let isActive = ${methodRead} == ${CODES['natural-breaks']}u && classCount >= 2u && minimumKey < maximumKey
    && emptyCount > 0u && bulkLow < bulkHigh;
  if (isActive) {
    backup[backupOffset + 4u] = bulk;
    backup[backupOffset + 5u] = 1u;
    backup[backupOffset + 6u] = emptyCount;
    extremes[extremesOffset + ${CLASS_BREAKS_EXTREMES.minimumKey}u] = bulkLow;
    extremes[extremesOffset + ${CLASS_BREAKS_EXTREMES.maximumKey}u] = bulkHigh;
    state[stateOffset + ${CLASS_BREAKS_STATE.classCount}u] = min(1u + emptyCount, BIN_COUNT);
    ${`state[stateOffset + ${CLASS_BREAKS_STATE.layerCursor}u]`} = 0u;
  } else {
    // Every node of the second stage returns at once.
    ${methodRead} = 0xffffffffu;
  }`
    })
  );
  nodes.push(...context.getCoarseNodes('-bulk'));
  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${prefix}-bulk-backtrack`,
      operation,
      variant: 'natural-breaks-bulk-backtrack',
      bindings: [
        stateRead,
        {name: 'extremes', view: props.extremes, type: 'u32', access: 'read'},
        {name: 'starts', view: context.starts, type: 'u32', access: 'read'},
        {name: 'bulkEdges', view: bulkEdges, type: 'f32', access: 'read_write'}
      ],
      invocationCount: 1,
      declarations: binDeclarations,
      body: `if (${gate}) {
    return;
  }
  let classCount = ${classCountRead};
  ${extremesRead}
  var end = BIN_COUNT - 1u;
  for (var layer = classCount - 1u; layer > 0u; layer = layer - 1u) {
    let start = starts[startsOffset + layer * BIN_COUNT + end];
    bulkEdges[bulkEdgesOffset + layer] = minimum + f32(start) * width;
    end = start - 1u;
  }`
    })
  );
  // Restore the shared words, merge the cuts of the second stage into the partition (every
  // non-empty class contributes its smallest value, the split class its sub-classes too), clear
  // the extents and arm the extents pass over the merged partition. The dispatch is written first
  // so a stale arm never survives.
  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${prefix}-bulk-assemble`,
      operation,
      variant: 'natural-breaks-bulk-assemble',
      bindings: [
        {name: 'state', view: props.state, type: 'u32', access: 'read_write'},
        {name: 'extremes', view: props.extremes, type: 'u32', access: 'read_write'},
        {name: 'extents', view: extents, type: 'u32', access: 'read_write'},
        {name: 'edges', view: edges, type: 'f32', access: 'read_write'},
        {name: 'bulkEdges', view: bulkEdges, type: 'f32', access: 'read'},
        {name: 'backup', view: backup, type: 'u32', access: 'read'},
        {name: 'dispatch', view: dispatch, type: 'u32', access: 'read_write'}
      ],
      invocationCount: 1,
      declarations: `${COLUMN_ORDERED_KEY_WGSL}
${maximumClassCountConstant}
const NO_MINIMUM: u32 = ${NO_MINIMUM_KEY};`,
      body: `let isActive = backup[backupOffset + 5u] == 1u;
  ${getClassBreaksDispatchWGSL(graph, {
    operation,
    slot: 5,
    invocationCount: rowCount,
    workgroupSize: rowWorkgroupSize,
    activeExpression: 'isActive'
  })}
  ${methodRead} = backup[backupOffset];
  state[stateOffset + ${CLASS_BREAKS_STATE.classCount}u] = backup[backupOffset + 1u];
  extremes[extremesOffset + ${CLASS_BREAKS_EXTREMES.minimumKey}u] = backup[backupOffset + 2u];
  extremes[extremesOffset + ${CLASS_BREAKS_EXTREMES.maximumKey}u] = backup[backupOffset + 3u];
  if (!isActive) {
    return;
  }
  let classCount = min(backup[backupOffset + 1u], BIN_COUNT);
  let bulk = backup[backupOffset + 4u];
  let parts = min(1u + backup[backupOffset + 6u], BIN_COUNT);
  var emitted = 0u;
  var isFirst = true;
  for (var index = 0u; index < classCount; index = index + 1u) {
    if (extents[extentsOffset + index] == NO_MINIMUM) {
      continue;
    }
    // The smallest value of the class starts it; the first start is the column minimum.
    if (isFirst) {
      isFirst = false;
    } else {
      emitted = emitted + 1u;
      edges[edgesOffset + emitted] = decodeOrderedKey(extents[extentsOffset + index]);
    }
    if (index == bulk) {
      for (var part = 1u; part < parts; part = part + 1u) {
        emitted = emitted + 1u;
        edges[edgesOffset + emitted] = bulkEdges[bulkEdgesOffset + part];
      }
    }
  }
  for (var index = 0u; index < MAXIMUM_CLASS_COUNT; index = index + 1u) {
    extents[extentsOffset + index] = NO_MINIMUM;
    extents[extentsOffset + MAXIMUM_CLASS_COUNT + index] = 0u;
    extents[extentsOffset + 2u * MAXIMUM_CLASS_COUNT + index] = 0u;
  }`
    })
  );
  nodes.push(createExtentsNode(`${prefix}-extents-bulk`, 5));
  nodes.push(createSnapNode(`${prefix}-snap-bulk`));

  const refinementCursor = `state[stateOffset + ${CLASS_BREAKS_STATE.refinementCursor}u]`;
  const halfRound = `(${refinementCursor} - 1u)`;
  const wantedParity = `(1u - (${halfRound} & 1u))`;
  const cellMidpoint = cellCount / 2;
  const halfWidthPerCell = 2 / cellCount;

  for (let half = 0; half < MAXIMUM_HALF_ROUND_COUNT; half++) {
    const halfId = `${prefix}-refine-${half}`;

    // Decide whether this half-round runs, and arm or disarm the four gated nodes. At a round
    // boundary, a round that moved no cut with every break zoomed enough ends the refinement.
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${halfId}-advance`,
        operation,
        variant: 'natural-breaks-refine-advance',
        bindings: [
          {name: 'state', view: props.state, type: 'u32', access: 'read_write'},
          {name: 'extremes', view: props.extremes, type: 'u32', access: 'read'},
          {name: 'breakState', view: breakState, type: 'u32', access: 'read'},
          {name: 'dispatch', view: dispatch, type: 'u32', access: 'read_write'}
        ],
        invocationCount: 1,
        declarations: `${binCountConstant}
${breakStateLayout}`,
        body: `let cursor = ${refinementCursor};
  let classCount = ${classCountRead};
  var isActive = ${methodRead} == ${CODES['natural-breaks']}u && classCount >= 2u
    && extremes[extremesOffset + ${CLASS_BREAKS_EXTREMES.minimumKey}u] < extremes[extremesOffset + ${CLASS_BREAKS_EXTREMES.maximumKey}u]
    && state[stateOffset + ${CLASS_BREAKS_STATE.refinementDone}u] == 0u
    && cursor < ${MAXIMUM_HALF_ROUND_COUNT}u;
  if (isActive && cursor > 0u && (cursor & 1u) == 0u) {
    var zoomed = true;
    for (var cut = 1u; cut < classCount; cut = cut + 1u) {
      if (breakState[breakStateOffset + LEVEL_BASE + cut] < ${NATURAL_BREAKS_SETTLED_ZOOM_LEVEL}u) {
        zoomed = false;
      }
    }
    if (state[stateOffset + ${CLASS_BREAKS_STATE.refinementMoved}u] == 0u && zoomed) {
      state[stateOffset + ${CLASS_BREAKS_STATE.refinementDone}u] = 1u;
      isActive = false;
    }
    state[stateOffset + ${CLASS_BREAKS_STATE.refinementMoved}u] = 0u;
  }
  ${refinementCursor} = cursor + 1u;
  ${[
    {slot: 1, invocationCount: breakCount, workgroupSize: breakWorkgroupSize},
    {slot: 2, invocationCount: cellWordCount, workgroupSize: cellWorkgroupSize},
    {slot: 3, invocationCount: rowCount, workgroupSize: rowWorkgroupSize},
    {slot: 4, invocationCount: breakCount, workgroupSize: breakWorkgroupSize}
  ]
    .map(shape =>
      getClassBreaksDispatchWGSL(graph, {operation, ...shape, activeExpression: 'isActive'})
    )
    .join('\n  ')}`
      })
    );

    // Plan: the window of every active break of this half-round.
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${halfId}-plan`,
        operation,
        variant: 'natural-breaks-refine-plan',
        bindings: [
          stateRead,
          {name: 'edges', view: edges, type: 'f32', access: 'read'},
          {name: 'extents', view: extents, type: 'u32', access: 'read'},
          {name: 'breakState', view: breakState, type: 'u32', access: 'read_write'},
          {name: 'plan', view: plan, type: 'f32', access: 'read_write'}
        ],
        invocationCount: breakCount,
        workgroupSize: breakWorkgroupSize,
        ...getClassBreaksIndirectGate<Parameters>(`${halfId}-plan`, dispatch, 1),
        declarations: `${COLUMN_ORDERED_KEY_WGSL}
${maximumClassCountConstant}
${breakStateLayout}`,
        body: `let cut = index + 1u;
  if (cut >= ${classCountRead}) {
    return;
  }
  plan[planOffset + 3u * cut + 1u] = 0.0;
  if ((cut & 1u) != ${wantedParity}) {
    return;
  }
  // The data extent of the two classes beside the cut.
  let lowKey = min(extents[extentsOffset + cut - 1u], extents[extentsOffset + cut]);
  let highKey = max(extents[extentsOffset + MAXIMUM_CLASS_COUNT + cut - 1u], extents[extentsOffset + MAXIMUM_CLASS_COUNT + cut]);
  if (lowKey == 0xffffffffu) {
    breakState[breakStateOffset + LEVEL_BASE + cut] = ${SETTLED_UNZOOMED_LEVEL}u;
    return;
  }
  let lowValue = decodeOrderedKey(lowKey);
  let highValue = decodeOrderedKey(highKey);
  // A cut beside an empty class lies outside the data; centre the window inside the extent.
  let anchor = min(max(edges[edgesOffset + cut], lowValue), highValue);
  let span = max(anchor - lowValue, highValue - anchor);
  if (!(span > 0.0)) {
    breakState[breakStateOffset + LEVEL_BASE + cut] = ${SETTLED_UNZOOMED_LEVEL}u;
    return;
  }
  // The whole extent until the break has been refined; then the zoomed window, widened to follow
  // neighbours that moved since the break's own last update.
  var halfWidth = span;
  let stored = bitcast<f32>(breakState[breakStateOffset + HALF_WIDTH_BASE + cut]);
  if (stored > 0.0) {
    let shift = max(
      bitcast<f32>(breakState[breakStateOffset + MOVEMENT_BASE + cut - 1u]),
      bitcast<f32>(breakState[breakStateOffset + MOVEMENT_BASE + cut + 1u])
    );
    halfWidth = min(max(stored, 2.0 * min(shift, span)), span);
  }
  plan[planOffset + 3u * cut] = anchor;
  plan[planOffset + 3u * cut + 1u] = halfWidth * ${getWGSLFloatLiteral(halfWidthPerCell)};
  plan[planOffset + 3u * cut + 2u] = ${getWGSLFloatLiteral(quantization)} / span;`
      })
    );

    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${halfId}-clear`,
        operation,
        variant: 'natural-breaks-refine-clear',
        bindings: [{name: 'cells', view: cells, type: 'u32', access: 'read_write'}],
        invocationCount: cellWordCount,
        workgroupSize: cellWorkgroupSize,
        ...getClassBreaksIndirectGate<Parameters>(`${halfId}-clear`, dispatch, 2),
        declarations: cellLayout,
        body: `// Words: count, sum, squares (low and high), smallest key, largest key.
  let word = index % ELEMENT_STRIDE;
  cells[cellsOffset + index] = select(0u, NO_MINIMUM, word == 5u);`
      })
    );

    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${halfId}-accumulate`,
        operation,
        variant: 'natural-breaks-refine-accumulate',
        bindings: [
          {name: 'values', view: values, type: 'f32', access: 'read'},
          ...maskBinding,
          stateRead,
          {name: 'edges', view: edges, type: 'f32', access: 'read'},
          {name: 'plan', view: plan, type: 'f32', access: 'read'},
          {name: 'cells', view: cells, type: 'atomic<u32>', access: 'read_write'}
        ],
        invocationCount: rowCount,
        workgroupSize: rowWorkgroupSize,
        ...getClassBreaksIndirectGate<Parameters>(`${halfId}-accumulate`, dispatch, 3),
        declarations: `${COLUMN_ORDERED_KEY_WGSL}
${CLASS_BREAKS_WIDE_WGSL}
${binCountConstant}
${cellLayout}
const QUANTIZATION: f32 = ${getWGSLFloatLiteral(quantization)};`,
        body: `let value = values[valuesOffset + index];
  if ((bitcast<u32>(value) & 0x7f800000u) == 0x7f800000u || !(${maskTest})) {
    return;
  }
  let classCount = ${classCountRead};
  let key = getOrderedKey(value);
  // The class of the row: the number of cuts e[1..classCount-1] at or below its key.
  var low = 1u;
  var high = classCount;
  while (low < high) {
    let middle = (low + high) >> 1u;
    if (getOrderedKey(edges[edgesOffset + middle]) <= key) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  let rowClass = low - 1u;
  // The one break of this half-round beside the class: below it (rowClass) or above it.
  let wanted = ${wantedParity};
  var cut = 0u;
  if (rowClass >= 1u && (rowClass & 1u) == wanted) {
    cut = rowClass;
  } else if (rowClass + 1u < classCount && ((rowClass + 1u) & 1u) == wanted) {
    cut = rowClass + 1u;
  }
  if (cut == 0u) {
    return;
  }
  let cellWidth = plan[planOffset + 3u * cut + 1u];
  if (!(cellWidth > 0.0)) {
    return;
  }
  let offset = value - plan[planOffset + 3u * cut];
  // Cell of the row, with the current boundary at cell ${cellMidpoint}: rows of the class on the
  // right of the cut never fall left of it and the other way round.
  var cell = i32(floor(clamp(offset / cellWidth, -${2 * cellCount}.0, ${2 * cellCount}.0))) + ${cellMidpoint};
  if (rowClass == cut) {
    cell = max(cell, ${cellMidpoint});
  } else {
    cell = min(cell, ${cellMidpoint - 1});
  }
  // Element 0 holds the rows left of the window, 1..CELL_COUNT the cells, the last the rows right.
  var element = u32(cell) + 1u;
  if (cell < 0) {
    element = 0u;
  } else if (cell >= ${cellCount}) {
    element = ELEMENT_COUNT - 1u;
  }
  let quantized = i32(round(clamp(offset * plan[planOffset + 3u * cut + 2u], -QUANTIZATION, QUANTIZATION)));
  let magnitude = u32(abs(quantized));
  let square = multiplyWide(magnitude, magnitude);
  let base = cellsOffset + ((cut - 1u) * ELEMENT_COUNT + element) * ELEMENT_STRIDE;
  atomicAdd(&cells[base], 1u);
  ${getAtomicAddWideWGSL('cells', 'base + 1u', 'base + 2u', 'bitcast<u32>(quantized)', 'select(0u, 0xffffffffu, (quantized < 0))')}
  ${getAtomicAddWideWGSL('cells', 'base + 3u', 'base + 4u', 'square.x', 'square.y')}
  atomicMin(&cells[base + 5u], key);
  atomicMax(&cells[base + 6u], key);`
      })
    );

    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${halfId}-solve`,
        operation,
        variant: 'natural-breaks-refine-solve',
        bindings: [
          {name: 'state', view: props.state, type: 'atomic<u32>', access: 'read_write'},
          {name: 'edges', view: edges, type: 'f32', access: 'read_write'},
          {name: 'extents', view: extents, type: 'u32', access: 'read_write'},
          {name: 'breakState', view: breakState, type: 'u32', access: 'read_write'},
          {name: 'plan', view: plan, type: 'f32', access: 'read'},
          {name: 'cells', view: cells, type: 'u32', access: 'read'}
        ],
        invocationCount: breakCount,
        workgroupSize: breakWorkgroupSize,
        ...getClassBreaksIndirectGate<Parameters>(`${halfId}-solve`, dispatch, 4),
        declarations: `${COLUMN_ORDERED_KEY_WGSL}
${CLASS_BREAKS_WIDE_WGSL}
${maximumClassCountConstant}
${breakStateLayout}
${cellLayout}`,
        body: `let cut = index + 1u;
  if (cut >= min(atomicLoad(&state[stateOffset + ${CLASS_BREAKS_STATE.classCount}u]), BIN_COUNT)) {
    return;
  }
  let cellWidth = plan[planOffset + 3u * cut + 1u];
  if (!(cellWidth > 0.0)) {
    return;
  }
  let base = cellsOffset + (cut - 1u) * ELEMENT_COUNT * ELEMENT_STRIDE;
  var totalCount = 0u;
  var totalSum = vec2<u32>(0u, 0u);
  var totalSquares = vec2<u32>(0u, 0u);
  for (var element = 0u; element < ELEMENT_COUNT; element = element + 1u) {
    let at = base + element * ELEMENT_STRIDE;
    totalCount = totalCount + cells[at];
    totalSum = addWide(totalSum, vec2<u32>(cells[at + 1u], cells[at + 2u]));
    totalSquares = addWide(totalSquares, vec2<u32>(cells[at + 3u], cells[at + 4u]));
  }
  // Cost of every boundary 0..CELL_COUNT: elements 0..boundary are left of it. The current cut is
  // boundary ${cellMidpoint}; another boundary has to be strictly cheaper to replace it.
  var leftCount = cells[base];
  var leftSum = vec2<u32>(cells[base + 1u], cells[base + 2u]);
  var leftSquares = vec2<u32>(cells[base + 3u], cells[base + 4u]);
  var bestCost = ${LARGE_COST};
  var bestBoundary = ${cellMidpoint}u;
  var currentCost = ${LARGE_COST};
  var currentLeftCount = 0u;
  var cost = ${LARGE_COST};
  for (var boundary = 0u; boundary <= CELL_COUNT; boundary = boundary + 1u) {
    // An empty cell leaves the partition, and so the cost, as it was one boundary before.
    var isChanged = boundary == 0u || boundary == ${cellMidpoint}u;
    if (boundary > 0u) {
      let at = base + boundary * ELEMENT_STRIDE;
      isChanged = isChanged || cells[at] != 0u;
      leftCount = leftCount + cells[at];
      leftSum = addWide(leftSum, vec2<u32>(cells[at + 1u], cells[at + 2u]));
      leftSquares = addWide(leftSquares, vec2<u32>(cells[at + 3u], cells[at + 4u]));
    }
    if (isChanged) {
      cost = getClassCost(leftCount, leftSum, leftSquares)
        + getClassCost(totalCount - leftCount, subtractWide(totalSum, leftSum), subtractWide(totalSquares, leftSquares));
    }
    if (boundary == ${cellMidpoint}u) {
      currentCost = cost;
      currentLeftCount = leftCount;
    }
    if (cost < bestCost) {
      bestCost = cost;
      bestBoundary = boundary;
    }
  }
  if (currentCost <= bestCost) {
    bestBoundary = ${cellMidpoint}u;
  }
  // Extents of the two new classes: elements 0..bestBoundary on the left, the rest on the right.
  var leftMinimum = NO_MINIMUM;
  var leftMaximum = 0u;
  var rightMinimum = NO_MINIMUM;
  var rightMaximum = 0u;
  for (var element = 0u; element < ELEMENT_COUNT; element = element + 1u) {
    let at = base + element * ELEMENT_STRIDE;
    if (cells[at] != 0u) {
      if (element <= bestBoundary) {
        leftMinimum = min(leftMinimum, cells[at + 5u]);
        leftMaximum = max(leftMaximum, cells[at + 6u]);
      } else {
        rightMinimum = min(rightMinimum, cells[at + 5u]);
        rightMaximum = max(rightMaximum, cells[at + 6u]);
      }
    }
  }
  extents[extentsOffset + cut - 1u] = leftMinimum;
  extents[extentsOffset + MAXIMUM_CLASS_COUNT + cut - 1u] = leftMaximum;
  extents[extentsOffset + cut] = rightMinimum;
  extents[extentsOffset + MAXIMUM_CLASS_COUNT + cut] = rightMaximum;
  // The cut is the smallest value on its right: an actual data value.
  var movement = 0.0;
  if (rightMinimum != NO_MINIMUM) {
    let newCut = decodeOrderedKey(rightMinimum);
    movement = abs(newCut - edges[edgesOffset + cut]);
    edges[edgesOffset + cut] = newCut;
    if (movement > 0.0) {
      atomicOr(&state[stateOffset + ${CLASS_BREAKS_STATE.refinementMoved}u], 1u);
    }
  }
  breakState[breakStateOffset + MOVEMENT_BASE + cut] = bitcast<u32>(movement);
  // Zoom around the chosen boundary, unless a class beside the break was empty (the window must
  // cover the extent again) or the optimum sits at the window edge (the cut may continue).
  let degenerate = currentLeftCount == 0u || totalCount - currentLeftCount == 0u;
  let atEdge = abs(i32(bestBoundary) - ${cellMidpoint}) >= ${cellMidpoint - NATURAL_BREAKS_ZOOM_CELLS};
  let halfWidth = cellWidth * ${getWGSLFloatLiteral(cellMidpoint)};
  var nextHalfWidth = halfWidth * ${getWGSLFloatLiteral((2 * NATURAL_BREAKS_ZOOM_CELLS) / cellCount)};
  var level = breakState[breakStateOffset + LEVEL_BASE + cut] + 1u;
  if (degenerate) {
    nextHalfWidth = 0.0;
    level = 0u;
  } else if (atEdge) {
    nextHalfWidth = halfWidth;
    level = level - 1u;
  }
  breakState[breakStateOffset + HALF_WIDTH_BASE + cut] = bitcast<u32>(nextHalfWidth);
  breakState[breakStateOffset + LEVEL_BASE + cut] = level;`
      })
    );
  }

  // A cut whose class holds no value is dropped, leaving the produced class count.
  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${prefix}-finalize`,
      operation,
      variant: 'natural-breaks-finalize',
      bindings: [
        stateRead,
        {name: 'extents', view: extents, type: 'u32', access: 'read'},
        {name: 'edges', view: edges, type: 'f32', access: 'read_write'},
        {name: 'edgeState', view: edgeState, type: 'u32', access: 'read_write'}
      ],
      invocationCount: 1,
      declarations: maximumClassCountConstant,
      body: `if (${gate}) {
    return;
  }
  let classCount = edgeState[edgeStateOffset];
  var kept = 0u;
  for (var cut = 1u; cut < classCount; cut = cut + 1u) {
    if (extents[extentsOffset + cut] != 0xffffffffu) {
      kept = kept + 1u;
      edges[edgesOffset + kept] = edges[edgesOffset + cut];
    }
  }
  edgeState[edgeStateOffset] = kept + 1u;`
    })
  );
  return nodes;
}
