// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUScan,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {
  createFillNode,
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../utils/wgsl-kernel-nodes';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {PERMUTATION_RANDOM_WGSL} from '../../gpu-spatial-analysis/permutation-inference/permutation-random';
import {GPUNetworkSnapping} from '../network-accessibility/index';
import {recommendLaneCount} from '../network-accessibility/network-accessibility-lanes';
import {ACCESSIBILITY_NONE} from '../network-accessibility/network-accessibility-passes';
import {GPUNetworkReachability} from '../network-reachability/index';
import {GPU_NETWORK_K_FUNCTION_PARAMETER_LENGTH} from './network-k-function-parameters';

const OPERATION = 'GPUNetworkKFunction';

/** Philox counter tag of the simulated event streams, keeping them apart from other users of the RNG. @internal */
export const NETWORK_K_FUNCTION_STREAM = 0x4b46554e;

/** Quantization range of the edge-length prefix sum used to draw random edges. @internal */
export const NETWORK_K_FUNCTION_LENGTH_RANGE = 2 ** 29;

/** Largest accepted `bandCount`. */
export const GPU_NETWORK_K_FUNCTION_MAXIMUM_BAND_COUNT = 1024;

const NONE = 0xffffffff;
const DEFAULT_BAND_COUNT = 32;
const DEFAULT_ROWS_PER_BLOCK = 32;
const DEFAULT_LOCAL_ITERATIONS = 16;

/**
 * Properties for {@link GPUNetworkKFunction}.
 *
 * Compile-time: node, edge and event counts, `bandCount`, `simulationCount`, `rowsPerBlock`,
 * `laneCount`, `maxIterations`, `localIterations`, `candidateCapacity` and which optional views
 * exist. Per-frame: event and node positions, the CSR contents, `maxDistance`, `networkLength`,
 * `maxSnapDistance` and `parameters` (seed and active simulation count).
 */
export type GPUNetworkKFunctionProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'network-k-function'`. */
  id?: string;
  /** Planar events (for example points of interest) to snap onto the network. */
  points: GraphDataView<'float32x2'>;
  /** Planar node positions, in the same coordinates as `points`. */
  nodePositions: GraphDataView<'float32x2'>;
  /** CSR row offsets with `nodeCount + 1` rows. The network must list both directions of every road. */
  offsets: GraphDataView<'uint32'>;
  /** CSR destination node per edge. */
  neighbors: GraphDataView<'uint32'>;
  /** Non-negative length (cost) per edge, equal for the two directions of a road. */
  weights: GraphDataView<'float32'>;
  /** Optional one-row maximum snap distance. Events farther from every edge are excluded. */
  maxSnapDistance?: GraphDataView<'float32'>;
  /** Optional compile-time candidate capacity of `GPUNetworkSnapping`. Requires `maxSnapDistance`. */
  candidateCapacity?: number;
  /** Compile-time. Morton-sort edges before the BVH build of the snapping candidate search. */
  spatialSort?: boolean;
  /** One-row largest distance `d_max` of the K function: the distance bands span `[0, d_max]`. */
  maxDistance: GraphDataView<'float32'>;
  /**
   * One-row total length `L` of the network with every road counted once, that is, half of the sum
   * of `weights` of a CSR that lists both directions. It scales K and sets the resolution of the
   * random edge draw.
   */
  networkLength: GraphDataView<'float32'>;
  /**
   * `GPU_NETWORK_K_FUNCTION_PARAMETER_LENGTH` uint32 elements written with
   * `getGPUNetworkKFunctionParameterValues`. Required.
   */
  parameters: GraphDataView<'uint32'>;
  /** Compile-time number of distance bands (thresholds) `B`, 2 to 1024. Defaults to 32. */
  bandCount?: number;
  /** Compile-time number of simulated patterns `S` of the envelope. Defaults to 0. */
  simulationCount?: number;
  /**
   * Compile-time shortest-path searches (rows) run together per block. Scratch memory grows with
   * `rowsPerBlock * nodeCount`. Defaults to `recommendLaneCount` for the whole row set, capped at 32.
   */
  rowsPerBlock?: number;
  /** Compile-time lanes (searches sharing one CSR) of each block's searches, shared by all blocks. Defaults to `rowsPerBlock`. */
  laneCount?: number;
  /** Rounds of every search, as `GPUNetworkReachability.maxIterations`. Defaults to 64. */
  maxIterations?: number;
  /** Hops chained per round, as `GPUNetworkReachability.localIterations`. Defaults to 16. */
  localIterations?: number;
  /**
   * `(1 + simulationCount) * bandCount` float32 values: `K(d_k)` of the observed pattern in row 0
   * (the first `bandCount` values) and of simulation `s` in row `s`, where
   * `d_k = maxDistance * k / (bandCount - 1)`.
   */
  kValues: GraphDataView<'float32'>;
  /**
   * Optional `(1 + simulationCount) * bandCount` uint32 cumulative counts of unordered event pairs
   * with network distance strictly below `d_k`, laid out like `kValues`.
   */
  pairCounts?: GraphDataView<'uint32'>;
  /**
   * Optional `3 * bandCount` float32 simulation envelope: the minimum, the mean and the maximum
   * of the active simulations per band, in three consecutive rows of `bandCount` values.
   * All zero without active simulations.
   */
  envelope?: GraphDataView<'float32'>;
  /** Optional one-row number of events that snapped onto the network (the `n` of K). */
  snappedEventCount?: GraphDataView<'uint32'>;
  /** Optional one-row flag: 1 when the snapping candidate search overflowed. */
  overflow?: GraphDataView<'uint32'>;
  /** Optional one-row flag: 1 when every shortest-path batch reached a fixpoint. */
  converged?: GraphDataView<'uint32'>;
};

/**
 * Network-constrained Ripley K function (Okabe and Yamada; spaghetti `GlobalAutoK`) of events on
 * a road network, with a Monte Carlo envelope of random points on the same network.
 *
 * Events are snapped onto their nearest edge with `GPUNetworkSnapping`. For every event a
 * multi-source `GPUNetworkReachability` search (as `GPUNetworkCostMatrix`, seeded at the two edge
 * endpoints with the snapped offsets and limited to `maxDistance`) gives the network distance to
 * every node, and the distance to another event is the cheaper of its two edge endpoints, or the
 * direct offset difference when both lie on the same road. Pairs are counted per band with integer
 * atomics, so the result is deterministic. `K(d) = 2 * pairs(d) * L / n^2` where `pairs(d)`
 * counts unordered pairs closer than `d` (strictly, as spaghetti does), `n` the snapped events and
 * `L` the network length, over `bandCount` thresholds `d_k = maxDistance * k / (bandCount - 1)`.
 *
 * The envelope draws `simulationCount` patterns of `n` events from a Philox stream keyed by the
 * seed: an edge with probability proportional to its length (an integer prefix sum of quantized
 * lengths) and a uniform position on it. Searches of all patterns share one set of row blocks
 * (`rowsPerBlock` rows each, searched by one `GPUNetworkReachability` per block over one lane-expanded CSR shared by all blocks), so peak scratch
 * memory is set by `rowsPerBlock`, not by the event or simulation count. Unlike spaghetti, whose
 * envelope scales its extremes by the significance threshold, the envelope here is the plain
 * minimum, mean and maximum of the simulated K values.
 *
 * Roads that share both end nodes (parallel edges) count as one road when two events fall on them.
 * Each block adds about `maxIterations + 6` command nodes and the total is
 * `ceil((1 + simulationCount) * events / rowsPerBlock)` blocks.
 */
export class GPUNetworkKFunction implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUNetworkKFunctionProps;
  /** Number of events (points). */
  readonly eventCount: number;
  /** Number of network nodes. */
  readonly nodeCount: number;
  /** Number of CSR edges. */
  readonly edgeCount: number;
  /** Resolved number of distance bands. */
  readonly bandCount: number;
  /** Resolved number of simulated patterns. */
  readonly simulationCount: number;
  /** Total searched rows, `(1 + simulationCount) * eventCount`. */
  readonly rowCount: number;
  /** Resolved rows per block. */
  readonly rowsPerBlock: number;
  /** Number of row blocks. */
  readonly blockCount: number;

  constructor(props: GPUNetworkKFunctionProps) {
    this.id = props.id ?? 'network-k-function';
    this.props = props;
    const {id} = this;
    for (const [name, view] of [
      ['offsets', props.offsets],
      ['neighbors', props.neighbors],
      ['parameters', props.parameters],
      ['pairCounts', props.pairCounts],
      ['snappedEventCount', props.snappedEventCount],
      ['overflow', props.overflow],
      ['converged', props.converged]
    ] as const) {
      if (view) {
        validatePackedUint32View(view, `${id} ${name}`);
      }
    }
    for (const [name, view] of [
      ['weights', props.weights],
      ['maxSnapDistance', props.maxSnapDistance],
      ['maxDistance', props.maxDistance],
      ['networkLength', props.networkLength],
      ['kValues', props.kValues],
      ['envelope', props.envelope]
    ] as const) {
      if (view) {
        validatePackedView(view, ['float32'], `${id} ${name}`);
      }
    }
    for (const [name, view] of [
      ['points', props.points],
      ['nodePositions', props.nodePositions]
    ] as const) {
      validatePackedView(view, ['float32x2'], `${id} ${name}`);
    }
    this.eventCount = props.points.length;
    this.nodeCount = props.nodePositions.length;
    this.edgeCount = props.neighbors.length;
    this.bandCount = props.bandCount ?? DEFAULT_BAND_COUNT;
    this.simulationCount = props.simulationCount ?? 0;
    if (this.eventCount < 2) {
      throw new Error(`${id} requires at least two events`);
    }
    if (this.nodeCount < 1 || this.edgeCount < 1) {
      throw new Error(`${id} requires at least one node and one edge`);
    }
    if (props.offsets.length !== this.nodeCount + 1) {
      throw new Error(`${id} offsets must contain one more row than nodePositions`);
    }
    if (props.weights.length !== this.edgeCount) {
      throw new Error(`${id} weights length must equal neighbors length`);
    }
    if (
      !Number.isSafeInteger(this.bandCount) ||
      this.bandCount < 2 ||
      this.bandCount > GPU_NETWORK_K_FUNCTION_MAXIMUM_BAND_COUNT
    ) {
      throw new Error(
        `${id} bandCount must be an integer in [2, ${GPU_NETWORK_K_FUNCTION_MAXIMUM_BAND_COUNT}]`
      );
    }
    if (!Number.isSafeInteger(this.simulationCount) || this.simulationCount < 0) {
      throw new Error(`${id} simulationCount must be a non-negative integer`);
    }
    if (props.parameters.length < GPU_NETWORK_K_FUNCTION_PARAMETER_LENGTH) {
      throw new Error(
        `${id} parameters must hold ${GPU_NETWORK_K_FUNCTION_PARAMETER_LENGTH} elements`
      );
    }
    for (const [name, view] of [
      ['maxSnapDistance', props.maxSnapDistance],
      ['maxDistance', props.maxDistance],
      ['networkLength', props.networkLength],
      ['snappedEventCount', props.snappedEventCount],
      ['overflow', props.overflow],
      ['converged', props.converged]
    ] as const) {
      if (view && view.length !== 1) {
        throw new Error(`${id} ${name} must contain exactly one row`);
      }
    }
    const patternCount = 1 + this.simulationCount;
    for (const [name, view, length] of [
      ['kValues', props.kValues, patternCount * this.bandCount],
      ['pairCounts', props.pairCounts, patternCount * this.bandCount],
      ['envelope', props.envelope, 3 * this.bandCount]
    ] as const) {
      if (view && view.length !== length) {
        throw new Error(`${id} ${name} length must be ${length}`);
      }
    }
    if (props.overflow && props.candidateCapacity === undefined) {
      throw new Error(`${id} overflow requires candidateCapacity`);
    }
    this.rowCount = patternCount * this.eventCount;
    this.rowsPerBlock =
      props.rowsPerBlock ??
      Math.min(
        this.rowCount,
        Math.max(
          1,
          Math.min(
            DEFAULT_ROWS_PER_BLOCK,
            recommendLaneCount({
              rowCount: this.rowCount,
              nodeCount: this.nodeCount,
              edgeCount: this.edgeCount
            })
          )
        )
      );
    if (
      !Number.isSafeInteger(this.rowsPerBlock) ||
      this.rowsPerBlock < 1 ||
      this.rowsPerBlock > this.rowCount
    ) {
      throw new Error(`${id} rowsPerBlock must be an integer in [1, ${this.rowCount}]`);
    }
    this.blockCount = Math.ceil(this.rowCount / this.rowsPerBlock);
    const outputs = getOutputs(props);
    validateGraphOutputsDisjointFromInputs(id, outputs, getInputs(props));
    const outputBuffers = outputs.filter(view => view !== undefined).map(view => view.buffer);
    if (new Set(outputBuffers).size !== outputBuffers.length) {
      throw new Error(`${id} outputs must use separate buffers`);
    }
  }

  /**
   * Returns the snapping nodes, the event count, the random-edge preparation (`quantize`, a `GPUScan`
   * prefix sum, `simulate`; only with simulations), one `clear`, then per block the cost-matrix nodes and a `count`
   * node, and finally `finalize`, optional `envelope` and `converged` nodes.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, eventCount, nodeCount, edgeCount, bandCount, simulationCount} = this;
    const {rowCount, rowsPerBlock, blockCount} = this;
    validateGraphViewsBelongToGraph(id, graph, [...getInputs(props), ...getOutputs(props)]);
    const nodes: GPUCommandNode<Parameters>[] = [];
    const patternCount = 1 + simulationCount;
    const u32 = (name: string, length: number) =>
      createTransientView(graph, `${id}-${name}`, 'uint32', length);

    // Seed table: row r = pattern * eventCount + event; entries 2r and 2r + 1 are the edge
    // endpoint nodes and the costs from the event to them (NONE and -1 for an inactive event).
    const seedNodes = u32('seed-nodes', 2 * rowCount);
    const seedCosts = createTransientView(graph, `${id}-seed-costs`, 'float32', 2 * rowCount);
    const snappedEdges = u32('snapped-edges', eventCount);
    nodes.push(
      ...new GPUNetworkSnapping({
        id: `${id}-snap`,
        points: props.points,
        nodePositions: props.nodePositions,
        offsets: props.offsets,
        edgeTargets: props.neighbors,
        edgeCosts: props.weights,
        maxSnapDistance: props.maxSnapDistance,
        candidateCapacity: props.candidateCapacity,
        spatialSort: props.spatialSort,
        seedDirection: 'both',
        snappedEdges,
        seedNodes: getSubView(graph, seedNodes, 0, 2 * eventCount),
        seedCosts: getSubView(graph, seedCosts, 0, 2 * eventCount),
        overflow: props.overflow
      }).getCommandNodes(graph)
    );
    const validCount = props.snappedEventCount ?? u32('event-count', 1);
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-event-count`,
        operation: OPERATION,
        variant: 'event-count',
        bindings: [
          {name: 'seedNodes', view: seedNodes, type: 'u32', access: 'read'},
          {name: 'validCount', view: validCount, type: 'u32', access: 'read_write'}
        ],
        invocationCount: 1,
        declarations: `const NODE_COUNT: u32 = ${nodeCount}u;
const EVENT_COUNT: u32 = ${eventCount}u;`,
        body: `var count = 0u;
  for (var eventIndex = 0u; eventIndex < EVENT_COUNT; eventIndex++) {
    if (seedNodes[seedNodesOffset + 2u * eventIndex] < NODE_COUNT) {
      count++;
    }
  }
  validCount[validCountOffset] = count;`
      })
    );

    if (simulationCount > 0) {
      const quantized = u32('quantized-lengths', edgeCount);
      const prefix = u32('length-prefix', edgeCount);
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-quantize`,
          operation: OPERATION,
          variant: 'quantize',
          bindings: [
            {name: 'neighbors', view: props.neighbors, type: 'u32', access: 'read'},
            {name: 'weights', view: props.weights, type: 'f32', access: 'read'},
            {name: 'networkLength', view: props.networkLength, type: 'f32', access: 'read'},
            {name: 'quantized', view: quantized, type: 'u32', access: 'read_write'}
          ],
          invocationCount: edgeCount,
          declarations: `const NODE_COUNT: u32 = ${nodeCount}u;`,
          body: `let weight = weights[weightsOffset + index];
  let networkTotal = networkLength[networkLengthOffset];
  var value = 0u;
  if (networkTotal > 0.0 && weight >= 0.0 && neighbors[neighborsOffset + index] < NODE_COUNT) {
    value = u32(weight * (${NETWORK_K_FUNCTION_LENGTH_RANGE}.0 / networkTotal));
  }
  quantized[quantizedOffset + index] = value;`
        }),
        ...new GPUScan({
          id: `${id}-length-scan`,
          input: quantized,
          output: prefix,
          mode: 'inclusive'
        }).getCommandNodes(graph),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-simulate`,
          operation: OPERATION,
          variant: 'simulate',
          bindings: [
            {name: 'prefix', view: prefix, type: 'u32', access: 'read'},
            {name: 'offsets', view: props.offsets, type: 'u32', access: 'read'},
            {name: 'neighbors', view: props.neighbors, type: 'u32', access: 'read'},
            {name: 'weights', view: props.weights, type: 'f32', access: 'read'},
            {name: 'validCount', view: validCount, type: 'u32', access: 'read'},
            {name: 'parameters', view: props.parameters, type: 'u32', access: 'read'},
            {name: 'outNodes', view: seedNodes, type: 'u32', access: 'read_write'},
            {name: 'outCosts', view: seedCosts, type: 'f32', access: 'read_write'}
          ],
          invocationCount: simulationCount * eventCount,
          declarations: `const NODE_COUNT: u32 = ${nodeCount}u;
const EDGE_COUNT: u32 = ${edgeCount}u;
const EVENT_COUNT: u32 = ${eventCount}u;
const SIMULATION_COUNT: u32 = ${simulationCount}u;
const NONE: u32 = ${NONE}u;
const STREAM_TAG: u32 = ${NETWORK_K_FUNCTION_STREAM}u;
${PERMUTATION_RANDOM_WGSL}`,
          body: `let pattern = 1u + index / EVENT_COUNT;
  let eventIndex = index - (pattern - 1u) * EVENT_COUNT;
  let slot = 2u * (pattern * EVENT_COUNT + eventIndex);
  let activeCount = min(parameters[parametersOffset + 2u], SIMULATION_COUNT);
  let total = prefix[prefixOffset + EDGE_COUNT - 1u];
  var firstNode = NONE;
  var secondNode = NONE;
  var firstCost = -1.0;
  var secondCost = -1.0;
  if (pattern <= activeCount && eventIndex < validCount[validCountOffset] && total > 0u) {
    var stream = createPhiloxStream(
      vec2<u32>(parameters[parametersOffset], parameters[parametersOffset + 1u]),
      eventIndex,
      pattern,
      STREAM_TAG
    );
    let draw = nextPhiloxBelow(&stream, total);
    let fractionBits = nextPhiloxUint32(&stream);
    var low = 0u;
    var high = EDGE_COUNT;
    while (low < high) {
      let middle = (low + high) / 2u;
      if (prefix[prefixOffset + middle] > draw) {
        high = middle;
      } else {
        low = middle + 1u;
      }
    }
    let edge = low;
    var nodeLow = 0u;
    var nodeHigh = NODE_COUNT;
    while (nodeLow < nodeHigh) {
      let middle = (nodeLow + nodeHigh) / 2u;
      if (offsets[offsetsOffset + middle + 1u] > edge) {
        nodeHigh = middle;
      } else {
        nodeLow = middle + 1u;
      }
    }
    let fraction = f32(fractionBits >> 8u) / 16777216.0;
    let weight = weights[weightsOffset + edge];
    firstNode = nodeLow;
    secondNode = neighbors[neighborsOffset + edge];
    firstCost = fraction * weight;
    secondCost = (1.0 - fraction) * weight;
  }
  outNodes[outNodesOffset + slot] = firstNode;
  outNodes[outNodesOffset + slot + 1u] = secondNode;
  outCosts[outCostsOffset + slot] = firstCost;
  outCosts[outCostsOffset + slot + 1u] = secondCost;`
        })
      );
    }

    const histogram = u32('histogram', patternCount * bandCount);
    nodes.push(
      createFillNode<Parameters>(graph, {
        id: `${id}-clear`,
        operation: OPERATION,
        view: histogram,
        type: 'u32',
        value: '0u'
      })
    );

    const costs = createTransientView(graph, `${id}-costs`, 'float32', rowsPerBlock * nodeCount);

    // Lane expansion: `laneCount` lanes that all read the one CSR (no per-lane copy), shared by
    // every block. Only the seeds are expanded, once for all blocks.
    const laneCount = Math.min(props.laneCount ?? rowsPerBlock, rowsPerBlock);
    if (laneCount * nodeCount >= ACCESSIBILITY_NONE) {
      throw new Error(`${id} laneCount * nodeCount must fit in uint32`);
    }
    const expandedSeeds = u32('expanded-seeds', 2 * rowCount);
    nodes.push(
      // Seed slots 2r and 2r + 1 of row r go to lane (r % rowsPerBlock) % laneCount.
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-expand-seeds`,
        operation: OPERATION,
        variant: 'expand-seeds',
        bindings: [
          {name: 'seedNodes', view: seedNodes, type: 'u32', access: 'read'},
          {name: 'expandedSeeds', view: expandedSeeds, type: 'u32', access: 'read_write'}
        ],
        invocationCount: 2 * rowCount,
        declarations: `const NODE_COUNT: u32 = ${nodeCount}u;
const ROWS_PER_BLOCK: u32 = ${rowsPerBlock}u;
const LANE_COUNT: u32 = ${laneCount}u;`,
        body: `let lane = ((index / 2u) % ROWS_PER_BLOCK) % LANE_COUNT;
  let node = seedNodes[seedNodesOffset + index];
  expandedSeeds[expandedSeedsOffset + index] =
    select(${ACCESSIBILITY_NONE}u, lane * NODE_COUNT + node, node < NODE_COUNT);`
      })
    );

    const batchesPerBlock = Math.ceil(rowsPerBlock / laneCount);
    const batchConverged = props.converged
      ? u32('batch-converged', blockCount * batchesPerBlock)
      : undefined;
    let convergedBatchCount = 0;
    for (let block = 0; block < blockCount; block++) {
      const firstRow = block * rowsPerBlock;
      const blockRows = Math.min(rowsPerBlock, rowCount - firstRow);
      const blockCosts = getSubView(graph, costs, 0, blockRows * nodeCount);
      for (let firstLaneRow = 0; firstLaneRow < blockRows; firstLaneRow += laneCount) {
        const batchRows = Math.min(laneCount, blockRows - firstLaneRow);
        const batchId = `${id}-block-${block}-batch-${firstLaneRow / laneCount}`;
        nodes.push(
          ...new GPUNetworkReachability({
            id: batchId,
            offsets: props.offsets,
            neighbors: props.neighbors,
            weights: props.weights,
            laneCount: batchRows,
            sources: getSubView(graph, expandedSeeds, 2 * (firstRow + firstLaneRow), 2 * batchRows),
            sourceCosts: getSubView(graph, seedCosts, 2 * (firstRow + firstLaneRow), 2 * batchRows),
            costLimit: props.maxDistance,
            maxIterations: props.maxIterations,
            localIterations: props.localIterations ?? DEFAULT_LOCAL_ITERATIONS,
            costs: getSubView(graph, blockCosts, firstLaneRow * nodeCount, batchRows * nodeCount),
            converged: batchConverged
              ? getSubView(graph, batchConverged, convergedBatchCount, 1)
              : undefined
          }).getCommandNodes(graph)
        );
        convergedBatchCount++;
      }
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-block-${block}-count`,
          operation: OPERATION,
          variant: 'count',
          bindings: [
            {name: 'costs', view: blockCosts, type: 'f32', access: 'read'},
            {name: 'seedNodes', view: seedNodes, type: 'u32', access: 'read'},
            {name: 'seedCosts', view: seedCosts, type: 'f32', access: 'read'},
            {name: 'maxDistanceIn', view: props.maxDistance, type: 'f32', access: 'read'},
            {name: 'histogram', view: histogram, type: 'atomic<u32>', access: 'read_write'}
          ],
          invocationCount: blockRows * eventCount,
          // Needs workgroup barriers, so out-of-range invocations must not return early.
          guardIndex: false,
          declarations: `const NODE_COUNT: u32 = ${nodeCount}u;
const EVENT_COUNT: u32 = ${eventCount}u;
const BAND_COUNT: u32 = ${bandCount}u;
const FIRST_ROW: u32 = ${firstRow}u;
const WORKGROUP_SIZE: u32 = 64u;
// Per-workgroup histogram of the pattern that owns the workgroup's first invocation. A workgroup
// spans 64 consecutive (row, column) pairs, so nearly all of them share one pattern and one
// global atomic per pair collapses into one local atomic and at most BAND_COUNT global adds.
var<workgroup> localHistogram: array<atomic<u32>, ${bandCount}>;
fn isFiniteCost(cost: f32) -> bool {
  return bitcast<u32>(cost) < 0x7f800000u;
}
fn getThreshold(maxDistance: f32, band: u32) -> f32 {
  return maxDistance * f32(band) / f32(BAND_COUNT - 1u);
}
fn getPattern(index: u32) -> u32 {
  return (FIRST_ROW + index / EVENT_COUNT) / EVENT_COUNT;
}
// Distance band of the unordered pair of row and column events, or BAND_COUNT when the pair is
// not counted.
fn classifyPair(index: u32) -> u32 {
  let localRow = index / EVENT_COUNT;
  let columnEvent = index - localRow * EVENT_COUNT;
  let row = FIRST_ROW + localRow;
  let pattern = row / EVENT_COUNT;
  let eventIndex = row - pattern * EVENT_COUNT;
  let maxDistance = maxDistanceIn[maxDistanceInOffset];
  if (columnEvent <= eventIndex || !(maxDistance > 0.0)) {
    return BAND_COUNT;
  }
  let rowSlot = 2u * row;
  let columnSlot = 2u * (pattern * EVENT_COUNT + columnEvent);
  let rowFirst = seedNodes[seedNodesOffset + rowSlot];
  let rowSecond = seedNodes[seedNodesOffset + rowSlot + 1u];
  let columnFirst = seedNodes[seedNodesOffset + columnSlot];
  let columnSecond = seedNodes[seedNodesOffset + columnSlot + 1u];
  if (rowFirst >= NODE_COUNT || rowSecond >= NODE_COUNT ||
      columnFirst >= NODE_COUNT || columnSecond >= NODE_COUNT) {
    return BAND_COUNT;
  }
  let rowFirstCost = seedCosts[seedCostsOffset + rowSlot];
  let columnFirstCost = seedCosts[seedCostsOffset + columnSlot];
  let columnSecondCost = seedCosts[seedCostsOffset + columnSlot + 1u];
  var best = 0.0;
  var found = false;
  let viaFirst = costs[costsOffset + localRow * NODE_COUNT + columnFirst];
  if (isFiniteCost(viaFirst)) {
    best = viaFirst + columnFirstCost;
    found = true;
  }
  let viaSecond = costs[costsOffset + localRow * NODE_COUNT + columnSecond];
  if (isFiniteCost(viaSecond)) {
    let candidate = viaSecond + columnSecondCost;
    if (!found || candidate < best) {
      best = candidate;
      found = true;
    }
  }
  var sameEdge = -1.0;
  if (rowFirst != rowSecond) {
    if (rowFirst == columnFirst && rowSecond == columnSecond) {
      sameEdge = abs(rowFirstCost - columnFirstCost);
    } else if (rowFirst == columnSecond && rowSecond == columnFirst) {
      sameEdge = abs(rowFirstCost - columnSecondCost);
    }
  }
  if (sameEdge >= 0.0 && (!found || sameEdge < best)) {
    best = sameEdge;
    found = true;
  }
  if (!found || !(best < maxDistance)) {
    return BAND_COUNT;
  }
  var band = u32(max(floor(best / (maxDistance / f32(BAND_COUNT - 1u))), 0.0)) + 1u;
  band = min(band, BAND_COUNT);
  while (band > 0u && best < getThreshold(maxDistance, band - 1u)) {
    band--;
  }
  while (band < BAND_COUNT && !(best < getThreshold(maxDistance, band))) {
    band++;
  }
  return band;
}`,
          workgroupSize: 64,
          body: `let firstIndex = index - localInvocationIndex;
  let ownerPattern = getPattern(min(firstIndex, INVOCATION_COUNT - 1u));
  for (var bin = localInvocationIndex; bin < BAND_COUNT; bin += WORKGROUP_SIZE) {
    atomicStore(&localHistogram[bin], 0u);
  }
  workgroupBarrier();
  if (index < INVOCATION_COUNT) {
    let band = classifyPair(index);
    if (band < BAND_COUNT) {
      let pattern = getPattern(index);
      if (pattern == ownerPattern) {
        atomicAdd(&localHistogram[band], 1u);
      } else {
        atomicAdd(&histogram[histogramOffset + pattern * BAND_COUNT + band], 1u);
      }
    }
  }
  workgroupBarrier();
  for (var bin = localInvocationIndex; bin < BAND_COUNT; bin += WORKGROUP_SIZE) {
    let binCount = atomicLoad(&localHistogram[bin]);
    if (binCount > 0u) {
      atomicAdd(&histogram[histogramOffset + ownerPattern * BAND_COUNT + bin], binCount);
    }
  }`
        })
      );
    }

    const finalizeBindings: WGSLKernelBinding[] = [
      {name: 'histogram', view: histogram, type: 'u32', access: 'read'},
      {name: 'networkLength', view: props.networkLength, type: 'f32', access: 'read'},
      {name: 'validCount', view: validCount, type: 'u32', access: 'read'},
      {name: 'parameters', view: props.parameters, type: 'u32', access: 'read'},
      {name: 'kValues', view: props.kValues, type: 'f32', access: 'read_write'}
    ];
    if (props.pairCounts) {
      finalizeBindings.push({
        name: 'pairCounts',
        view: props.pairCounts,
        type: 'u32',
        access: 'read_write'
      });
    }
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-finalize`,
        operation: OPERATION,
        variant: 'finalize',
        bindings: finalizeBindings,
        invocationCount: patternCount * bandCount,
        declarations: `const BAND_COUNT: u32 = ${bandCount}u;
const SIMULATION_COUNT: u32 = ${simulationCount}u;`,
        body: `let pattern = index / BAND_COUNT;
  let band = index - pattern * BAND_COUNT;
  let activeCount = min(parameters[parametersOffset + 2u], SIMULATION_COUNT);
  var cumulative = 0u;
  if (pattern <= activeCount) {
    for (var bin = 0u; bin <= band; bin++) {
      cumulative += histogram[histogramOffset + pattern * BAND_COUNT + bin];
    }
  }
  let count = f32(validCount[validCountOffset]);
  var k = 0.0;
  if (count >= 2.0) {
    k = 2.0 * f32(cumulative) * networkLength[networkLengthOffset] / (count * count);
  }
  kValues[kValuesOffset + index] = k;
  ${props.pairCounts ? 'pairCounts[pairCountsOffset + index] = cumulative;' : ''}`
      })
    );

    if (props.envelope) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-envelope`,
          operation: OPERATION,
          variant: 'envelope',
          bindings: [
            {name: 'kValues', view: props.kValues, type: 'f32', access: 'read'},
            {name: 'parameters', view: props.parameters, type: 'u32', access: 'read'},
            {name: 'envelope', view: props.envelope, type: 'f32', access: 'read_write'}
          ],
          invocationCount: bandCount,
          declarations: `const BAND_COUNT: u32 = ${bandCount}u;
const SIMULATION_COUNT: u32 = ${simulationCount}u;`,
          body: `let activeCount = min(parameters[parametersOffset + 2u], SIMULATION_COUNT);
  var lowest = 0.0;
  var highest = 0.0;
  var sum = 0.0;
  for (var pattern = 1u; pattern <= activeCount; pattern++) {
    let value = kValues[kValuesOffset + pattern * BAND_COUNT + index];
    lowest = select(min(lowest, value), value, pattern == 1u);
    highest = select(max(highest, value), value, pattern == 1u);
    sum += value;
  }
  envelope[envelopeOffset + index] = lowest;
  envelope[envelopeOffset + BAND_COUNT + index] = select(0.0, sum / f32(activeCount), activeCount > 0u);
  envelope[envelopeOffset + 2u * BAND_COUNT + index] = highest;`
        })
      );
    }

    if (props.converged && batchConverged) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-converged`,
          operation: OPERATION,
          variant: 'converged',
          bindings: [
            {name: 'batchConverged', view: batchConverged, type: 'u32', access: 'read'},
            {name: 'converged', view: props.converged, type: 'u32', access: 'read_write'}
          ],
          invocationCount: 1,
          declarations: `const BATCH_COUNT: u32 = ${convergedBatchCount}u;`,
          body: `var allDone = 1u;
  for (var batch = 0u; batch < BATCH_COUNT; batch++) {
    allDone = min(allDone, batchConverged[batchConvergedOffset + batch]);
  }
  converged[convergedOffset] = allDone;`
        })
      );
    }
    return nodes;
  }
}

/** Returns a packed range of `length` rows starting at row `firstRow` of a packed view. */
function getSubView<Format extends 'uint32' | 'float32', Parameters>(
  graph: GPUCommandGraph<Parameters>,
  view: GraphDataView<Format>,
  firstRow: number,
  length: number
): GraphDataView<Format> {
  return graph.createDataView(view.buffer, {
    format: view.format as Format,
    length,
    byteOffset: view.byteOffset + firstRow * view.byteStride
  });
}

/** Returns every read-only view. */
function getInputs(props: GPUNetworkKFunctionProps): (GraphDataView | undefined)[] {
  return [
    props.points,
    props.nodePositions,
    props.offsets,
    props.neighbors,
    props.weights,
    props.maxSnapDistance,
    props.maxDistance,
    props.networkLength,
    props.parameters
  ];
}

/** Returns every writable view. */
function getOutputs(props: GPUNetworkKFunctionProps): (GraphDataView | undefined)[] {
  return [
    props.kValues,
    props.pairCounts,
    props.envelope,
    props.snappedEventCount,
    props.overflow,
    props.converged
  ];
}
