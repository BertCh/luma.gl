// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUScan,
  GPUSort,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {
  createFillNode,
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../utils/wgsl-kernel-nodes';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';

const OPERATION = 'GPULineMerge';

/** Value written to per-line outputs for a line that is in no chain (fewer than two vertices). */
export const GPU_LINE_MERGE_NONE = 0xffffffff;

/**
 * Caller-owned outputs of {@link GPULineMerge}.
 *
 * Capacities follow from the input: a merge never creates vertices or chains, so
 * `chainOffsets.length - 1 >= lineCount` and `positions.length >= input vertex count` always suffice.
 */
export type GPULineMergeOutput = {
  /**
   * Chain `c` owns `positions[chainOffsets[c] .. chainOffsets[c + 1])`. Rows past `count` repeat
   * the final offset. Length `lineCount + 1`.
   */
  chainOffsets: GraphDataView<'uint32'>;
  /** Chain vertices in chain order. Rows past the last chain are left untouched. */
  positions: GraphDataView<'float32x2'>;
  /** One-row scalar receiving the number of chains. */
  count: GraphDataView<'uint32'>;
  /** Optional chain index per input line, or {@link GPU_LINE_MERGE_NONE} for lines with under two vertices. */
  lineChains?: GraphDataView<'uint32'>;
  /** Optional position of each input line within its chain (0 for the first line). */
  lineOrders?: GraphDataView<'uint32'>;
  /** Optional 1 when the chain traverses the line against its input direction, else 0. */
  lineReversed?: GraphDataView<'uint32'>;
};

/** Properties for {@link GPULineMerge}. */
export type GPULineMergeProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'line-merge'`. */
  id?: string;
  /** Vertices of every input linestring, concatenated. */
  positions: GraphDataView<'float32x2'>;
  /** Line-to-vertex offsets with `lineCount + 1` rows (GeoArrow layout). */
  lineOffsets: GraphDataView<'uint32'>;
  /**
   * Respect line direction, like `shapely.line_merge(directed=True)`. Defaults to `false`.
   *
   * A join then needs the end of one line to meet the start of the next, so lines never reverse:
   * chains follow input direction, `output.lineReversed` is all zero, and two lines that start or
   * end at the same point stay separate. A vertex where three or more line ends meet still stops
   * every chain. Structural: changing it changes the compiled kernels.
   */
  directed?: boolean;
  /** Caller-owned output. */
  output: GPULineMergeOutput;
};

/**
 * Merges linestrings that share endpoints into maximal chains, like Shapely `linemerge` and turf
 * `lineMerge` (undirected, no direction constraint).
 *
 * Two line ends join only where exactly two line ends meet at the same coordinate, so a vertex shared
 * by three or more line ends (a junction) stops every chain, and a closed line never joins itself.
 * Coordinates match exactly (bitwise `float32`, `-0` equals `0`); lines with fewer than two vertices
 * are ignored.
 *
 * Endpoints are grouped by two stable radix sorts (y then x bits), the join partner of each end is
 * read from its neighbours in the sorted order, and one thread per chain walks the chain. Paths
 * are found from their ends; closed cycles of several lines are found by their lowest line index.
 * Walking is serial per chain, so a chain of `n` lines costs `n` steps in one thread.
 *
 * With `directed: true` only an end that meets a start joins (Shapely `directed=True`); chains then
 * always follow the input direction.
 *
 * Deterministic order and orientation: chains are ordered by their leader line index, where a path's
 * leader is the line holding its canonical end and a cycle's leader is its lowest line. A path starts
 * at the end with the lower endpoint id (`2 * line` for a line start, `2 * line + 1` for a line end)
 * and a cycle starts at the start of its lowest line and follows that line's direction. Cycles are
 * emitted closed (the last vertex repeats the first).
 */
export class GPULineMerge implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPULineMergeProps;
  /** Number of input lines. */
  readonly lineCount: number;

  constructor(props: GPULineMergeProps) {
    this.id = props.id ?? 'line-merge';
    this.props = props;
    const {id} = this;
    const {output} = props;
    validatePackedView(props.positions, ['float32x2'], `${id} positions`);
    validatePackedView(output.positions, ['float32x2'], `${id} output.positions`);
    for (const [name, view] of [
      ['lineOffsets', props.lineOffsets],
      ['output.chainOffsets', output.chainOffsets],
      ['output.count', output.count],
      ['output.lineChains', output.lineChains],
      ['output.lineOrders', output.lineOrders],
      ['output.lineReversed', output.lineReversed]
    ] as const) {
      if (view) {
        validatePackedUint32View(view, `${id} ${name}`);
      }
    }
    if (props.lineOffsets.length < 2) {
      throw new Error(`${id} lineOffsets needs at least one line`);
    }
    this.lineCount = props.lineOffsets.length - 1;
    if (output.chainOffsets.length < this.lineCount + 1) {
      throw new Error(`${id} output.chainOffsets must contain lineCount + 1 rows`);
    }
    if (output.positions.length < props.positions.length) {
      throw new Error(`${id} output.positions must hold at least the input vertex count`);
    }
    if (output.count.length !== 1) {
      throw new Error(`${id} output.count must contain exactly one row`);
    }
    for (const [name, view] of [
      ['lineChains', output.lineChains],
      ['lineOrders', output.lineOrders],
      ['lineReversed', output.lineReversed]
    ] as const) {
      if (view && view.length !== this.lineCount) {
        throw new Error(`${id} output.${name} length must equal the line count`);
      }
    }
    const outputs = getOutputs(props);
    validateGraphOutputsDisjointFromInputs(id, outputs, getInputs(props));
    const outputBuffers = outputs.filter(view => view !== undefined).map(view => view.buffer);
    if (new Set(outputBuffers).size !== outputBuffers.length) {
      throw new Error(`${id} outputs must use separate buffers`);
    }
  }

  /**
   * Returns endpoint keys, two sorts, `partners`, `heads` and `cycles` walks, `flags`, two scans,
   * `offsets`, `positions` and optional `line-info` nodes.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, lineCount} = this;
    const {output} = props;
    validateGraphViewsBelongToGraph(id, graph, [...getInputs(props), ...getOutputs(props)]);
    const endpointCount = lineCount * 2;
    const isDirected = props.directed === true;
    const transient = <Format extends 'uint32' | 'float32'>(
      name: string,
      format: Format,
      length: number
    ) => createTransientView(graph, `${id}-${name}`, format, Math.max(length, 1));
    const read = (name: string, view: GraphDataView, type: 'u32' | 'f32' = 'u32') =>
      ({name, view, type, access: 'read'}) as WGSLKernelBinding;
    const write = (name: string, view: GraphDataView, type: 'u32' | 'f32' = 'u32') =>
      ({name, view, type, access: 'read_write'}) as WGSLKernelBinding;
    const kernel = (
      variant: string,
      bindings: WGSLKernelBinding[],
      invocationCount: number,
      body: string,
      declarations = '',
      nodeSuffix = ''
    ) =>
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-${variant}${nodeSuffix}`,
        operation: OPERATION,
        variant,
        bindings,
        invocationCount,
        declarations: `const LINE_COUNT: u32 = ${lineCount}u;
const NONE: u32 = 0xffffffffu;
const INVALID_KEY: u32 = 0xffffffffu;
const DIRECTED: bool = ${isDirected};
${declarations}`,
        body
      });

    const keyY = transient('key-y', 'uint32', endpointCount);
    const keyX = transient('key-x', 'uint32', endpointCount);
    const endpointIds = transient('endpoint-ids', 'uint32', endpointCount);
    const sortedY = transient('sorted-y', 'uint32', endpointCount);
    const orderY = transient('order-y', 'uint32', endpointCount);
    const secondKeys = transient('second-keys', 'uint32', endpointCount);
    const sortedX = transient('sorted-x', 'uint32', endpointCount);
    const order = transient('order', 'uint32', endpointCount);
    const partner = transient('partner', 'uint32', endpointCount);
    const leaderExit = transient('leader-exit', 'uint32', lineCount);
    const leaderVertices = transient('leader-vertices', 'uint32', lineCount);
    const flags = transient('flags', 'uint32', lineCount);
    const flagScan = transient('flag-scan', 'uint32', lineCount);
    const vertexScan = transient('vertex-scan', 'uint32', lineCount);

    const nodes: GPUCommandNode<Parameters>[] = [];
    // Bit patterns; -0 is folded to +0 so equal coordinates compare equal. Lines with fewer than
    // two vertices or a non-finite endpoint get the NaN key, which never joins.
    nodes.push(
      kernel(
        'keys',
        [
          read('positions', props.positions, 'f32'),
          read('lineOffsets', props.lineOffsets),
          write('keyY', keyY),
          write('keyX', keyX),
          write('endpointIds', endpointIds)
        ],
        endpointCount,
        `let line = index / 2u;
  let first = lineOffsets[lineOffsetsOffset + line];
  let last = lineOffsets[lineOffsetsOffset + line + 1u];
  endpointIds[endpointIdsOffset + index] = index;
  var x = 0.0;
  var y = 0.0;
  var valid = last >= first + 2u;
  if (valid) {
    let vertex = select(first, last - 1u, (index & 1u) == 1u);
    x = positions[positionsOffset + vertex * 2u];
    y = positions[positionsOffset + vertex * 2u + 1u];
    valid = abs(x) < 3.0e38 && abs(y) < 3.0e38;
  }
  if (valid) {
    x = select(x, 0.0, x == 0.0);
    y = select(y, 0.0, y == 0.0);
    keyX[keyXOffset + index] = bitcast<u32>(x);
    keyY[keyYOffset + index] = bitcast<u32>(y);
  } else {
    keyX[keyXOffset + index] = INVALID_KEY;
    keyY[keyYOffset + index] = INVALID_KEY;
  }`
      ),
      ...new GPUSort({
        id: `${id}-sort-y`,
        keys: keyY,
        values: endpointIds,
        outputKeys: sortedY,
        outputValues: orderY
      }).getCommandNodes(graph),
      kernel(
        'gather-x',
        [read('keyX', keyX), read('orderY', orderY), write('secondKeys', secondKeys)],
        endpointCount,
        `secondKeys[secondKeysOffset + index] = keyX[keyXOffset + orderY[orderYOffset + index]];`
      ),
      ...new GPUSort({
        id: `${id}-sort-x`,
        keys: secondKeys,
        values: orderY,
        outputKeys: sortedX,
        outputValues: order
      }).getCommandNodes(graph),
      // In the (x, y)-sorted order equal coordinates are adjacent. An end joins its partner when its
      // run has exactly two members that belong to different lines.
      kernel(
        'partners',
        [
          read('sortedX', sortedX),
          read('order', order),
          read('keyY', keyY),
          write('partner', partner)
        ],
        endpointCount,
        `let end = order[orderOffset + index];
  var result = NONE;
  if (sortedX[sortedXOffset + index] != INVALID_KEY) {
    let isLeftSame = index > 0u && isSame(index - 1u, index);
    let isRightSame = index + 1u < ENDPOINT_COUNT && isSame(index, index + 1u);
    if (isLeftSame && !isRightSame && !(index > 1u && isSame(index - 2u, index - 1u))) {
      result = order[orderOffset + index - 1u];
    } else if (isRightSame && !isLeftSame &&
        !(index + 2u < ENDPOINT_COUNT && isSame(index + 1u, index + 2u))) {
      result = order[orderOffset + index + 1u];
    }
    if (result != NONE && (result >> 1u) == (end >> 1u)) {
      result = NONE;
    }
    // Directed joins pair a line end (odd id) with a line start (even id).
    if (DIRECTED && result != NONE && ((result ^ end) & 1u) == 0u) {
      result = NONE;
    }
  }
  partner[partnerOffset + end] = result;`,
        `const ENDPOINT_COUNT: u32 = ${endpointCount}u;
fn isSame(a: u32, b: u32) -> bool {
  return sortedX[sortedXOffset + a] == sortedX[sortedXOffset + b] &&
    keyY[keyYOffset + order[orderOffset + a]] == keyY[keyYOffset + order[orderOffset + b]];
}`
      )
    );

    // Chains are ranked by list ranking (Wyllie pointer jumping) instead of one thread walking
    // each chain. A traversal is a line in one direction, named by its exit end id `e`
    // (`2 * line + 1` forward, `2 * line` reversed); its predecessor traversal is `partner[e ^ 1]`.
    // Predecessor lists are paths (ending at a head with no predecessor) or closed cycles.
    // Phase 1 jumps `ceil(log2(lineCount))` rounds carrying the minimum traversal key: a node whose
    // predecessor is still defined afterwards lies on a cycle, and the cycle's minimum key says which
    // orientation leads (an even key is the orientation that traverses its lowest line forward).
    // Phase 2 cuts each canonical cycle before its leader, which makes it a path, and jumps again
    // carrying the head, the vertex-count prefix and (when needed) the line count. Rounds are fixed,
    // so there are no readbacks: `O(L log L)` work in `O(log L)` dispatches instead of `O(L)` serial
    // steps per chain, and every output is then written by one thread per traversal.
    const rounds = Math.max(1, Math.ceil(Math.log2(Math.max(lineCount, 2))));
    const needLineCounts = Boolean(output.lineOrders);
    const walkDeclarations = `
fn vertexCount(line: u32) -> u32 {
  return lineOffsets[lineOffsetsOffset + line + 1u] - lineOffsets[lineOffsetsOffset + line];
}`;
    const pair = (name: string) =>
      [
        transient(`${name}-a`, 'uint32', endpointCount),
        transient(`${name}-b`, 'uint32', endpointCount)
      ] as const;
    const minPredecessors = pair('min-predecessor');
    const minKeys = pair('min-key');
    const rankPredecessors = pair('rank-predecessor');
    const rankHeads = pair('rank-head');
    const rankSums = pair('rank-sum');
    const rankCounts = pair('rank-count');
    const leaderBits = transient('leader-bits', 'uint32', endpointCount);
    const lineKey = (variable: string) =>
      `(${variable} >> 1u) * 2u + select(1u, 0u, (${variable} & 1u) == 1u)`;

    nodes.push(
      kernel(
        'min-init',
        [
          read('partner', partner),
          write('predecessor', minPredecessors[0]),
          write('minKey', minKeys[0])
        ],
        endpointCount,
        `predecessor[predecessorOffset + index] = partner[partnerOffset + (index ^ 1u)];
  minKey[minKeyOffset + index] = ${lineKey('index')};`
      )
    );
    for (let round = 0; round < rounds; round++) {
      const from = round % 2;
      const to = 1 - from;
      nodes.push(
        kernel(
          'min-round',
          [
            read('predecessorIn', minPredecessors[from]),
            read('keyIn', minKeys[from]),
            write('predecessorOut', minPredecessors[to]),
            write('keyOut', minKeys[to])
          ],
          endpointCount,
          `let previous = predecessorIn[predecessorInOffset + index];
  var key = keyIn[keyInOffset + index];
  var nextPrevious = previous;
  if (previous != NONE) {
    key = min(key, keyIn[keyInOffset + previous]);
    nextPrevious = predecessorIn[predecessorInOffset + previous];
  }
  predecessorOut[predecessorOutOffset + index] = nextPrevious;
  keyOut[keyOutOffset + index] = key;`,
          '',
          `-${round}`
        )
      );
    }
    const minFinal = rounds % 2;
    nodes.push(
      // leaderBits: 0 inactive, 1 active, 3 active cycle leader. A cycle node is canonical when its
      // cycle's minimum key is even; the traversal holding that key leads and loses its predecessor.
      kernel(
        'rank-init',
        [
          read('partner', partner),
          read('lineOffsets', props.lineOffsets),
          read('cyclePredecessor', minPredecessors[minFinal]),
          read('cycleKey', minKeys[minFinal]),
          write('predecessor', rankPredecessors[0]),
          write('head', rankHeads[0]),
          write('sum', rankSums[0]),
          write('leaderBits', leaderBits)
        ],
        endpointCount,
        `let isCycle = cyclePredecessor[cyclePredecessorOffset + index] != NONE;
  let minimumKey = cycleKey[cycleKeyOffset + index];
  let isCanonicalCycle = isCycle && (minimumKey & 1u) == 0u;
  let vertices = vertexCount(index >> 1u);
  let isActive = vertices >= 2u && (!isCycle || isCanonicalCycle);
  let isLeader = isActive && isCycle && minimumKey == ${lineKey('index')};
  var previous = NONE;
  if (isActive && !isLeader) {
    previous = partner[partnerOffset + (index ^ 1u)];
  }
  predecessor[predecessorOffset + index] = previous;
  head[headOffset + index] = index;
  sum[sumOffset + index] = select(0u, vertices - 1u, isActive);
  leaderBits[leaderBitsOffset + index] = select(0u, select(1u, 3u, isLeader), isActive);`,
        walkDeclarations
      )
    );
    if (needLineCounts) {
      nodes.push(
        createFillNode<Parameters>(graph, {
          id: `${id}-rank-count-init`,
          operation: OPERATION,
          view: rankCounts[0],
          type: 'u32',
          value: '1u'
        })
      );
    }
    for (let round = 0; round < rounds; round++) {
      const from = round % 2;
      const to = 1 - from;
      nodes.push(
        kernel(
          'rank-round',
          [
            read('predecessorIn', rankPredecessors[from]),
            read('headIn', rankHeads[from]),
            read('sumIn', rankSums[from]),
            write('predecessorOut', rankPredecessors[to]),
            write('headOut', rankHeads[to]),
            write('sumOut', rankSums[to])
          ],
          endpointCount,
          `let previous = predecessorIn[predecessorInOffset + index];
  var head = headIn[headInOffset + index];
  var sum = sumIn[sumInOffset + index];
  var nextPrevious = previous;
  if (previous != NONE) {
    head = headIn[headInOffset + previous];
    sum += sumIn[sumInOffset + previous];
    nextPrevious = predecessorIn[predecessorInOffset + previous];
  }
  predecessorOut[predecessorOutOffset + index] = nextPrevious;
  headOut[headOutOffset + index] = head;
  sumOut[sumOutOffset + index] = sum;`,
          '',
          `-${round}`
        )
      );
      if (needLineCounts) {
        nodes.push(
          kernel(
            'rank-count-round',
            [
              read('predecessorIn', rankPredecessors[from]),
              read('countIn', rankCounts[from]),
              write('countOut', rankCounts[to])
            ],
            endpointCount,
            `let previous = predecessorIn[predecessorInOffset + index];
  var count = countIn[countInOffset + index];
  if (previous != NONE) {
    count += countIn[countInOffset + previous];
  }
  countOut[countOutOffset + index] = count;`,
            '',
            `-${round}`
          )
        );
      }
    }
    const rankFinal = rounds % 2;
    const headFinal = rankHeads[rankFinal];
    const sumFinal = rankSums[rankFinal];
    nodes.push(
      createFillNode<Parameters>(graph, {
        id: `${id}-clear-leaders`,
        operation: OPERATION,
        view: leaderVertices,
        type: 'u32',
        value: '0u'
      }),
      // The tail of every canonical chain publishes the chain at its head line. A path is canonical
      // in the orientation whose entry end id is lower than its tail exit id (undirected) or that
      // runs forward (directed); a cycle is canonical at its leader.
      kernel(
        'tails',
        [
          read('partner', partner),
          read('leaderBits', leaderBits),
          read('head', headFinal),
          read('sum', sumFinal),
          write('leaderExit', leaderExit),
          write('leaderVertices', leaderVertices)
        ],
        endpointCount,
        `if (leaderBits[leaderBitsOffset + index] == 0u) {
    return;
  }
  let next = partner[partnerOffset + index];
  if (next != NONE && (leaderBits[leaderBitsOffset + (next ^ 1u)] & 2u) == 0u) {
    return;
  }
  let first = head[headOffset + index];
  var isCanonical = (leaderBits[leaderBitsOffset + first] & 2u) != 0u;
  if (!isCanonical) {
    isCanonical = select((first ^ 1u) < index, (first & 1u) == 1u, DIRECTED);
  }
  if (isCanonical) {
    leaderVertices[leaderVerticesOffset + (first >> 1u)] = sum[sumOffset + index] + 1u;
    leaderExit[leaderExitOffset + (first >> 1u)] = first;
  }`
      ),
      kernel(
        'flags',
        [read('leaderVertices', leaderVertices), write('flags', flags)],
        lineCount,
        `flags[flagsOffset + index] = select(0u, 1u, leaderVertices[leaderVerticesOffset + index] > 0u);`
      ),
      ...new GPUScan({
        id: `${id}-chain-scan`,
        input: flags,
        output: flagScan,
        mode: 'inclusive'
      }).getCommandNodes(graph),
      ...new GPUScan({
        id: `${id}-vertex-scan`,
        input: leaderVertices,
        output: vertexScan,
        mode: 'inclusive'
      }).getCommandNodes(graph),
      kernel(
        'offsets',
        [
          read('leaderVertices', leaderVertices),
          read('flagScan', flagScan),
          read('vertexScan', vertexScan),
          write('chainOffsets', output.chainOffsets),
          write('count', output.count)
        ],
        lineCount + 1,
        `let chainCount = flagScan[flagScanOffset + LINE_COUNT - 1u];
  let totalVertices = vertexScan[vertexScanOffset + LINE_COUNT - 1u];
  if (index == 0u) {
    count[countOffset] = chainCount;
  }
  if (index >= chainCount) {
    chainOffsets[chainOffsetsOffset + index] = totalVertices;
  }
  if (index < LINE_COUNT && leaderVertices[leaderVerticesOffset + index] > 0u) {
    let chain = flagScan[flagScanOffset + index] - 1u;
    chainOffsets[chainOffsetsOffset + chain] =
      vertexScan[vertexScanOffset + index] - leaderVertices[leaderVerticesOffset + index];
  }`
      ),
      // One thread per traversal: a line of a canonical chain copies its vertices to the chain base
      // plus its exclusive vertex prefix (a joined line skips its first, shared, vertex).
      kernel(
        'positions',
        [
          read('positions', props.positions, 'f32'),
          read('lineOffsets', props.lineOffsets),
          read('head', headFinal),
          read('sum', sumFinal),
          read('leaderExit', leaderExit),
          read('leaderVertices', leaderVertices),
          read('vertexScan', vertexScan),
          write('outPositions', output.positions, 'f32')
        ],
        endpointCount,
        `let first = head[headOffset + index];
  let headLine = first >> 1u;
  if (leaderVertices[leaderVerticesOffset + headLine] == 0u ||
      leaderExit[leaderExitOffset + headLine] != first) {
    return;
  }
  let line = index >> 1u;
  let firstVertex = lineOffsets[lineOffsetsOffset + line];
  let count = vertexCount(line);
  let isForward = (index & 1u) == 1u;
  let base = vertexScan[vertexScanOffset + headLine] - leaderVertices[leaderVerticesOffset + headLine];
  let cursor = base + sum[sumOffset + index] - (count - 1u);
  for (var k = select(1u, 0u, index == first); k < count; k++) {
    let vertex = select(firstVertex + count - 1u - k, firstVertex + k, isForward);
    let slot = cursor + k;
    outPositions[outPositionsOffset + slot * 2u] = positions[positionsOffset + vertex * 2u];
    outPositions[outPositionsOffset + slot * 2u + 1u] = positions[positionsOffset + vertex * 2u + 1u];
  }`,
        walkDeclarations
      )
    );
    if (output.lineChains || output.lineOrders || output.lineReversed) {
      const infoBindings: WGSLKernelBinding[] = [
        read('head', headFinal),
        read('leaderExit', leaderExit),
        read('leaderVertices', leaderVertices),
        read('flagScan', flagScan)
      ];
      let infoBody = `let first = head[headOffset + index];
  let headLine = first >> 1u;
  if (leaderVertices[leaderVerticesOffset + headLine] == 0u ||
      leaderExit[leaderExitOffset + headLine] != first) {
    return;
  }
  let line = index >> 1u;`;
      if (output.lineChains) {
        infoBindings.push(write('lineChains', output.lineChains));
        infoBody += `
  lineChains[lineChainsOffset + line] = flagScan[flagScanOffset + headLine] - 1u;`;
        nodes.push(
          createFillNode<Parameters>(graph, {
            id: `${id}-clear-line-chains`,
            operation: OPERATION,
            view: output.lineChains,
            type: 'u32',
            value: '0xffffffffu'
          })
        );
      }
      if (output.lineOrders) {
        infoBindings.push(
          read('lineCountPrefix', rankCounts[rankFinal]),
          write('lineOrders', output.lineOrders)
        );
        infoBody += `
  lineOrders[lineOrdersOffset + line] = lineCountPrefix[lineCountPrefixOffset + index] - 1u;`;
      }
      if (output.lineReversed) {
        infoBindings.push(write('lineReversed', output.lineReversed));
        infoBody += `
  lineReversed[lineReversedOffset + line] = select(1u, 0u, (index & 1u) == 1u);`;
      }
      nodes.push(kernel('line-info', infoBindings, endpointCount, infoBody));
    }
    return nodes;
  }
}

/** Returns every read-only view of a line-merge contributor. */
function getInputs(props: GPULineMergeProps): (GraphDataView | undefined)[] {
  return [props.positions, props.lineOffsets];
}

/** Returns every writable view of a line-merge contributor. */
function getOutputs(props: GPULineMergeProps): (GraphDataView | undefined)[] {
  const {output} = props;
  return [
    output.chainOffsets,
    output.positions,
    output.count,
    output.lineChains,
    output.lineOrders,
    output.lineReversed
  ];
}
