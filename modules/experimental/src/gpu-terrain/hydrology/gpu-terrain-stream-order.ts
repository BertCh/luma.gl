// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUScan,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {
  createRasterIterationFinalizeNode,
  createRasterGatedLoopNodes,
  createRasterIterationResetNode,
  createRasterIterationState,
  validateRasterIterations
} from '../../gpu-raster/cost-distance/raster-relaxation';
import type {GPURasterMetadata} from '../../gpu-raster/index';
import {
  createWGSLDispatchGate,
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../utils/wgsl-kernel-nodes';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {
  validateRasterBuffersDistinct,
  validateRasterGrid
} from '../../gpu-raster/raster-band-utils';
import {GPU_TERRAIN_FLOW_STREAM_UNRESOLVED} from './gpu-terrain-flow';
import {TERRAIN_FLOW_MAXIMUM_WALK_LENGTH} from './terrain-flow-passes';
import {
  createTerrainFlowEulerTourNodes,
  getTerrainFlowEulerTourWGSL,
  TERRAIN_FLOW_TOUR_RESOLVED_WGSL
} from './terrain-flow-euler-tour';
import {
  getTerrainFlowTracingWGSL,
  validateTerrainTracingCellView,
  validateTerrainTracingStatus
} from './terrain-flow-tracing';

const OPERATION = 'GPUTerrainStreamOrder';

/** One-row status words of {@link GPUTerrainStreamOrder}. */
export type GPUTerrainStreamOrderStatus = {
  /**
   * One row set to 1 when every stream cell resolved. Conservative on the pull fallback: can be 0
   * when the last round finished everything. On the Euler-tour path it is 1 unless a receiver cycle
   * was found.
   */
  converged?: GraphDataView<'uint32'>;
  /** One row: gated rounds executed (list-ranking rounds on the Euler-tour path). */
  iterationCount?: GraphDataView<'uint32'>;
};

/** Caller-owned outputs of {@link GPUTerrainStreamOrder}. */
export type GPUTerrainStreamOrderOutput = {
  /**
   * Strahler order per cell: 0 for non-stream and invalid cells, at least 1 on stream cells.
   * Cells that remain unresolved when the iteration limit is hit read `0xffffffff`, and so do
   * cells whose stream word is `GPU_TERRAIN_FLOW_STREAM_UNRESOLVED` or whose direction is
   * `GPU_TERRAIN_FLOW_UNRESOLVED`, and the stream cells they drain into: their order is unknown.
   * `0xffffffff` is not an order a converged run can produce. `status.converged` only reports the
   * pull rounds.
   */
  streamOrder: GraphDataView<'uint32'>;
  /** One-row status words. */
  status?: GPUTerrainStreamOrderStatus;
};

/**
 * Properties for {@link GPUTerrainStreamOrder}.
 *
 * Compile-time: `grid` size, `maximumIterations`, and the presence of `output.status` views. Per-frame: the
 * contents of `flowDirections` and `streams`. The contributor works on D8 receiver indices only, so it
 * has no cell-size model (`uniform`, `web-mercator` and `geographic` spacing give the same orders).
 *
 * Output aliasing: outputs never share a buffer with an input or with each other.
 */
export type GPUTerrainStreamOrderProps = {
  /** Prefix for node and transient IDs. Defaults to `'terrain-stream-order'`. Compile-time. */
  id?: string;
  /** Raster grid: its width and height size the kernels. */
  grid: GPURasterMetadata;
  /**
   * ESRI D8 codes, one per cell: 1 east, 2 south-east, 4 south, 8 south-west, 16 west, 32
   * north-west, 64 north, 128 north-east; 0 terminal; `0xffffffff` invalid;
   * `GPU_TERRAIN_FLOW_UNRESOLVED` (0xfffffffe) unknown: a stream cell with that direction, and a
   * stream cell with a stream neighbor of that direction (which might donate to it), keep the
   * unresolved order. Any other code, and any code pointing outside the grid, is treated as
   * terminal. Typically `GPUTerrainFlow.flowDirections`. Per-frame.
   */
  flowDirections: GraphDataView<'uint32'>;
  /**
   * Stream mask, one word per cell, non-zero for stream cells. Typically `GPUTerrainFlow.streams`,
   * whose `GPU_TERRAIN_FLOW_STREAM_UNRESOLVED` word (accumulation round limit hit) marks a cell of
   * unknown stream status. Per-frame.
   */
  streams: GraphDataView<'uint32'>;
  /** Caller-owned outputs. */
  output: GPUTerrainStreamOrderOutput;
  /**
   * Maximum gated rounds in `[1, 1024]`. Defaults to 64. Compile-time. It bounds the pull fallback
   * only (grids above 4096 x 4096 cells); the Euler-tour path needs no round limit.
   */
  maximumIterations?: number;
};

/**
 * Strahler stream order (Strahler 1957) on one raster tile.
 *
 * Stream donors of a stream cell are the stream cells whose D8 receiver is that cell. A stream cell
 * without stream donors is order 1; otherwise with `m` the highest donor order, the cell has order
 * `m + 1` when at least two donors have order `m`, else `m`. Orders follow D8 receivers, so with
 * multiple-flow-direction or D-infinity accumulation thresholds the stream mask can be
 * discontinuous along D8 paths; a stream cell that drains into a non-stream cell simply
 * contributes nothing downstream.
 *
 * The evaluation uses levels on the Euler tour of the stream forest (parent of a stream cell: its D8
 * receiver when that is a stream cell). Wyllie list ranking (`ceil(log2(2 n)) + 1` gated rounds)
 * gives every tour edge a position, so the count of any indicator over a subtree is a difference of
 * two entries of one exclusive scan. Level `A(1)` is the stream cells and `A(k+1)` the cells whose
 * subtree holds a cell with two children in `A(k)`; the order of a cell is the number of levels
 * that hold it, exact integers whatever the stream length. Levels are gated on the GPU (at most
 * `floor(log2(n))`), and the kernels are the same for every level. Unresolved cells follow the
 * pull contract through one more scan over seed indicators. Memory is about 13 words per cell.
 * When the `2 n` tour words do not fit one storage binding (grids above 4096 x 4096 cells) the
 * contributor falls back to a deterministic pull over the receiver forest, the pattern of
 * {@link GPUTerrainFlow} weighted accumulation: one `atomic<u32>` word per cell holds both readiness
 * (sentinel `0xffffffff`) and the final order, a cell is written once from finalized donors only,
 * and after finalizing a cell a thread walks downstream (up to 1024 steps). Order values depend on
 * the donor orders alone, so they never depend on scheduling. Rounds are bounded by the height of
 * the stream tree. Receiver cycles (caller-supplied directions only) never resolve.
 */
export class GPUTerrainStreamOrder implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUTerrainStreamOrderProps;

  constructor(props: GPUTerrainStreamOrderProps) {
    this.id = props.id ?? 'terrain-stream-order';
    this.props = props;
    const {id} = this;
    const {output} = props;
    const cellCount = validateRasterGrid(id, props.grid);
    validateRasterIterations(id, 'maximumIterations', props.maximumIterations ?? 64);
    validateTerrainTracingCellView(id, 'flowDirections', props.flowDirections, 'uint32', cellCount);
    validateTerrainTracingCellView(id, 'streams', props.streams, 'uint32', cellCount);
    validateTerrainTracingCellView(
      id,
      'output.streamOrder',
      output.streamOrder,
      'uint32',
      cellCount
    );
    validateTerrainTracingStatus(id, output.status);
    const outputs = [output.streamOrder, output.status?.converged, output.status?.iterationCount];
    validateGraphOutputsDisjointFromInputs(id, outputs, [props.flowDirections, props.streams]);
    validateRasterBuffersDistinct(id, outputs, []);
  }

  /**
   * Returns the Euler-tour nodes (tour link, gated ranking rounds, positions, and the gated Strahler
   * levels), or the pull rounds when the tour arrays (`2 * width * height` words) do not fit one
   * storage binding.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {width, height} = this.props.grid;
    return 8 * width * height <= graph.device.limits.maxStorageBufferBindingSize
      ? this.getTourCommandNodes(graph)
      : this.getPullCommandNodes(graph);
  }

  /** Strahler orders by levels on the Euler tour of the stream forest. */
  private getTourCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {output} = props;
    const {width, height} = props.grid;
    validateGraphViewsBelongToGraph(id, graph, [
      props.flowDirections,
      props.streams,
      output.streamOrder,
      output.status?.converged,
      output.status?.iterationCount
    ]);
    const cellCount = width * height;
    const edgeCount = 2 * cellCount;
    const create = (suffix: string, length: number) =>
      createTransientView(graph, `${id}-order-${suffix}`, 'uint32', length);
    const member = create('member', cellCount);
    const rootLength = create('root-length', cellCount);
    const rootOffset = create('root-offset', cellCount);
    const position = create('position', edgeCount);
    const levelStatus = create('level-status', 4);
    const parentBindings: WGSLKernelBinding[] = [
      {name: 'flowDirections', view: props.flowDirections, type: 'u32', access: 'read'},
      {name: 'streams', view: props.streams, type: 'u32', access: 'read'}
    ];
    const tracing = getTerrainFlowTracingWGSL(width, height);
    // The stream forest: a stream cell's parent is its D8 receiver when that is a stream cell too.
    const parentDeclarations = `fn isStreamCell(cell: u32) -> bool {
  return streams[streamsOffset + cell] != 0u && flowDirections[flowDirectionsOffset + cell] != NO_CELL;
}

fn getTourParent(cell: u32) -> u32 {
  if (!isStreamCell(cell)) { return NO_CELL; }
  let receiver = getFlowReceiver(cell, flowDirections[flowDirectionsOffset + cell]);
  if (receiver == NO_CELL || !isStreamCell(receiver)) { return NO_CELL; }
  return receiver;
}`;
    const tourDeclarations = `${tracing}
${getTerrainFlowEulerTourWGSL(width, height)}
${parentDeclarations}`;
    const resolvedDeclarations = `${tourDeclarations}
${TERRAIN_FLOW_TOUR_RESOLVED_WGSL}`;

    const tour = createTerrainFlowEulerTourNodes<Parameters>(graph, {
      id,
      operation: OPERATION,
      contributorId: id,
      width,
      height,
      declarations: `${tracing}\n${parentDeclarations}`,
      parentBindings
    });
    const tourBindings: WGSLKernelBinding[] = [
      {name: 'tourSucc', view: tour.successor, type: 'u32', access: 'read'},
      {name: 'tourDist', view: tour.distance, type: 'u32', access: 'read'}
    ];
    const [indicator, prefix] = tour.scratch;
    const nodes: GPUCommandNode<Parameters>[] = [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-order-init`,
        operation: OPERATION,
        variant: 'order-tour-init',
        bindings: [
          ...parentBindings,
          {name: 'order', view: output.streamOrder, type: 'u32', access: 'read_write'},
          {name: 'member', view: member, type: 'u32', access: 'read_write'}
        ],
        invocationCount: cellCount,
        declarations: `${tracing}\n${parentDeclarations}`,
        // Every stream cell starts in level 1 with order 1; other cells are final zeros.
        body: `let isStream = isStreamCell(index);
  order[orderOffset + index] = select(0u, 1u, isStream);
  member[memberOffset + index] = select(0u, 1u, isStream);`
      }),
      ...tour.nodes,
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-order-root-length`,
        operation: OPERATION,
        variant: 'order-root-length',
        bindings: [
          ...parentBindings,
          ...tourBindings,
          {name: 'rootLength', view: rootLength, type: 'u32', access: 'read_write'}
        ],
        invocationCount: cellCount,
        declarations: resolvedDeclarations,
        body: `let isRoot = isTourEdgeResolved(2u * index) && getTourParent(index) == TOUR_NONE;
  rootLength[rootLengthOffset + index] = select(0u, tourDist[tourDistOffset + 2u * index] + 1u, isRoot);`
      }),
      ...new GPUScan({
        id: `${id}-order-root-scan`,
        input: rootLength,
        output: rootOffset,
        mode: 'exclusive'
      }).getCommandNodes(graph),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-order-position`,
        operation: OPERATION,
        variant: 'order-position',
        bindings: [
          ...parentBindings,
          ...tourBindings,
          {name: 'rootOffset', view: rootOffset, type: 'u32', access: 'read'},
          {name: 'position', view: position, type: 'u32', access: 'read_write'}
        ],
        invocationCount: edgeCount,
        declarations: resolvedDeclarations,
        // Position in the concatenated tours: the root's offset plus the rank from the tour start.
        body: `if (!isTourEdgeResolved(index)) {
    position[positionOffset + index] = TOUR_NONE;
    return;
  }
  let root = tourSucc[tourSuccOffset + index] >> 1u;
  position[positionOffset + index] = rootOffset[rootOffsetOffset + root] +
    tourDist[tourDistOffset + 2u * root] - tourDist[tourDistOffset + index];`
      })
    ];

    // Levels: A(k+1) = cells whose stream subtree holds a cell with two children in A(k).
    // Order = number of levels holding the cell. Each level is one scatter, one scan, one gather.
    const levelCount = Math.min(32, Math.max(1, 31 - Math.clz32(Math.max(cellCount, 1))));
    const levelGate = createWGSLDispatchGate<Parameters>(graph, {
      id: `${id}-order-level-dispatch`,
      operation: OPERATION,
      shapes: [
        [cellCount, 256],
        [edgeCount, 256]
      ]
    });
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-order-level-reset`,
        operation: OPERATION,
        variant: 'order-level-reset',
        bindings: [
          {name: 'status', view: levelStatus, type: 'u32', access: 'read_write'},
          levelGate.binding
        ],
        invocationCount: 1,
        body: `status[statusOffset + 0u] = 0u;
  status[statusOffset + 1u] = 1u;
  status[statusOffset + 2u] = 0u;
  status[statusOffset + 3u] = 0u;
  ${levelGate.armWGSL}`
      })
    );
    const gateKernel = (nodeId: string, slot: number) => {
      const {condition, extraResources} = levelGate.getGate(slot);
      return {condition: {...condition, id: `${nodeId}-gate`}, extraResources};
    };
    for (let level = 0; level < levelCount; level++) {
      const scatterId = `${id}-order-level-${level}-scatter`;
      const gatherId = `${id}-order-level-${level}-gather`;
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: scatterId,
          operation: OPERATION,
          variant: 'order-level-scatter',
          bindings: [
            ...parentBindings,
            {name: 'position', view: position, type: 'u32', access: 'read'},
            {name: 'member', view: member, type: 'u32', access: 'read'},
            {name: 'indicator', view: indicator, type: 'u32', access: 'read_write'}
          ],
          invocationCount: edgeCount,
          declarations: tourDeclarations,
          // A junction (two children in the level) marks its enter edge; every other edge is 0.
          body: `let place = position[positionOffset + index];
  if (place == TOUR_NONE) { return; }
  var junction = 0u;
  if ((index & 1u) == 0u) {
    let cell = index >> 1u;
    var childCount = 0u;
    for (var direction = 0u; direction < 8u; direction++) {
      let neighbor = getTourNeighbor(cell, direction);
      if (neighbor != TOUR_NONE && getTourParent(neighbor) == cell && member[memberOffset + neighbor] != 0u) {
        childCount = childCount + 1u;
      }
    }
    junction = select(0u, 1u, childCount >= 2u);
  }
  indicator[indicatorOffset + place] = junction;`,
          ...gateKernel(scatterId, 1)
        }),
        ...new GPUScan({
          id: `${id}-order-level-${level}-scan`,
          input: indicator,
          output: prefix,
          mode: 'exclusive'
        }).getCommandNodes(graph),
        createWGSLKernelNode<Parameters>(graph, {
          id: gatherId,
          operation: OPERATION,
          variant: 'order-level-gather',
          bindings: [
            {name: 'position', view: position, type: 'u32', access: 'read'},
            {name: 'prefix', view: prefix, type: 'u32', access: 'read'},
            {name: 'member', view: member, type: 'u32', access: 'read_write'},
            {name: 'order', view: output.streamOrder, type: 'u32', access: 'read_write'},
            {name: 'status', view: levelStatus, type: 'atomic<u32>', access: 'read_write'}
          ],
          invocationCount: cellCount,
          declarations: 'const TOUR_NONE: u32 = 0xffffffffu;',
          // Junctions in the subtree = prefix over [enter, exit]; exit positions hold 0.
          body: `if (member[memberOffset + index] == 0u) { return; }
  let enterPlace = position[positionOffset + 2u * index];
  if (enterPlace == TOUR_NONE) { return; }
  let exitPlace = position[positionOffset + 2u * index + 1u];
  if (prefix[prefixOffset + exitPlace] - prefix[prefixOffset + enterPlace] > 0u) {
    order[orderOffset + index] = order[orderOffset + index] + 1u;
    atomicAdd(&status[statusOffset + 0u], 1u);
  } else {
    member[memberOffset + index] = 0u;
  }`,
          ...gateKernel(gatherId, 0)
        }),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-order-level-${level}-gate`,
          operation: OPERATION,
          variant: 'order-level-gate',
          bindings: [
            {name: 'status', view: levelStatus, type: 'u32', access: 'read_write'},
            levelGate.binding
          ],
          invocationCount: 1,
          // The next level is empty when no cell joined this one.
          body: `if (status[statusOffset + 1u] == 0u) { return; }
  let joined = status[statusOffset + 0u];
  status[statusOffset + 0u] = 0u;
  status[statusOffset + 2u] = status[statusOffset + 2u] + 1u;
  if (joined == 0u) {
    status[statusOffset + 1u] = 0u;
    ${levelGate.disarmWGSL}
  }`
        })
      );
    }

    // Unknown orders: cells whose subtree holds an unresolved stream cell, or that never reached a
    // root (receiver cycles), read 0xffffffff.
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-order-unknown-scatter`,
        operation: OPERATION,
        variant: 'order-unknown-scatter',
        bindings: [
          ...parentBindings,
          {name: 'position', view: position, type: 'u32', access: 'read'},
          {name: 'indicator', view: indicator, type: 'u32', access: 'read_write'}
        ],
        invocationCount: edgeCount,
        declarations: tourDeclarations,
        // A seed is a stream cell of unknown stream status or direction, or a stream cell next to
        // a stream cell of unknown direction (which might donate to it).
        body: `let place = position[positionOffset + index];
  if (place == TOUR_NONE) { return; }
  var seed = 0u;
  let cell = index >> 1u;
  if ((index & 1u) == 0u && isStreamCell(cell)) {
    var isSeed = streams[streamsOffset + cell] == ${GPU_TERRAIN_FLOW_STREAM_UNRESOLVED}u ||
      flowDirections[flowDirectionsOffset + cell] == UNRESOLVED_DIRECTION;
    for (var direction = 0u; direction < 8u; direction++) {
      let neighbor = getTourNeighbor(cell, direction);
      if (neighbor != TOUR_NONE && isStreamCell(neighbor) &&
          flowDirections[flowDirectionsOffset + neighbor] == UNRESOLVED_DIRECTION) {
        isSeed = true;
      }
    }
    seed = select(0u, 1u, isSeed);
  }
  indicator[indicatorOffset + place] = seed;`
      }),
      ...new GPUScan({
        id: `${id}-order-unknown-scan`,
        input: indicator,
        output: prefix,
        mode: 'exclusive'
      }).getCommandNodes(graph),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-order-unknown-gather`,
        operation: OPERATION,
        variant: 'order-unknown-gather',
        bindings: [
          ...parentBindings,
          {name: 'position', view: position, type: 'u32', access: 'read'},
          {name: 'prefix', view: prefix, type: 'u32', access: 'read'},
          {name: 'order', view: output.streamOrder, type: 'u32', access: 'read_write'},
          {name: 'status', view: tour.state.status, type: 'atomic<u32>', access: 'read_write'}
        ],
        invocationCount: cellCount,
        declarations: tourDeclarations,
        body: `if (!isStreamCell(index)) { return; }
  let enterPlace = position[positionOffset + 2u * index];
  if (enterPlace == TOUR_NONE) {
    order[orderOffset + index] = NO_CELL;
    atomicStore(&status[statusOffset + 3u], 0u);
    return;
  }
  let exitPlace = position[positionOffset + 2u * index + 1u];
  if (prefix[prefixOffset + exitPlace] - prefix[prefixOffset + enterPlace] > 0u) {
    order[orderOffset + index] = NO_CELL;
  }`
      })
    );
    if (output.status?.converged || output.status?.iterationCount) {
      nodes.push(
        createRasterIterationFinalizeNode<Parameters>(graph, {
          id: `${id}-order-finalize`,
          operation: OPERATION,
          state: tour.state,
          converged: output.status.converged,
          iterationCount: output.status.iterationCount
        })
      );
    }
    return nodes;
  }

  /** Returns the initialization, loop reset, and gated pull rounds. */
  private getPullCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {output} = props;
    const {width, height} = props.grid;
    validateGraphViewsBelongToGraph(id, graph, [
      props.flowDirections,
      props.streams,
      output.streamOrder,
      output.status?.converged,
      output.status?.iterationCount
    ]);
    const cellCount = width * height;
    const maxIterations = props.maximumIterations ?? 64;
    const state = createRasterIterationState(graph, `${id}-order`, OPERATION, cellCount, id);
    const tracing = getTerrainFlowTracingWGSL(width, height);
    const nodes: GPUCommandNode<Parameters>[] = [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-order-init`,
        operation: OPERATION,
        variant: 'order-init',
        bindings: [
          {name: 'flowDirections', view: props.flowDirections, type: 'u32', access: 'read'},
          {name: 'streams', view: props.streams, type: 'u32', access: 'read'},
          {name: 'order', view: output.streamOrder, type: 'u32', access: 'read_write'}
        ],
        invocationCount: cellCount,
        declarations: tracing,
        // Non-stream and invalid cells are final zeros; stream cells await their donors.
        body: `let isStream = streams[streamsOffset + index] != 0u &&
    flowDirections[flowDirectionsOffset + index] != NO_CELL;
  order[orderOffset + index] = select(0u, NO_CELL, isStream);`
      }),
      createRasterIterationResetNode<Parameters>(graph, {
        id: `${id}-order-reset`,
        operation: OPERATION,
        state
      })
    ];
    const bindings: WGSLKernelBinding[] = [
      {name: 'order', view: output.streamOrder, type: 'atomic<u32>', access: 'read_write'},
      {name: 'flowDirections', view: props.flowDirections, type: 'u32', access: 'read'},
      {name: 'streams', view: props.streams, type: 'u32', access: 'read'},
      {name: 'status', view: state.status, type: 'atomic<u32>', access: 'read_write'}
    ];
    const declarations = `${tracing}
const SENTINEL: u32 = 0xffffffffu;
const MAX_WALK_LENGTH: u32 = ${TERRAIN_FLOW_MAXIMUM_WALK_LENGTH}u;

// Unknown stream status (GPUTerrainFlow ran out of accumulation rounds): never finalized, so the
// cell and every stream cell downstream of it keep the unresolved sentinel.
fn isUnresolvedStream(cell: u32) -> bool {
  return streams[streamsOffset + cell] == ${GPU_TERRAIN_FLOW_STREAM_UNRESOLVED}u ||
    flowDirections[flowDirectionsOffset + cell] == UNRESOLVED_DIRECTION;
}

// Returns true when this thread wrote the final order of 'cell'.
fn tryFinalize(cell: u32) -> bool {
  if (atomicLoad(&order[orderOffset + cell]) != SENTINEL || isUnresolvedStream(cell)) { return false; }
  var maximumOrder = 0u;
  var maximumCount = 0u;
  for (var direction = 0u; direction < 8u; direction++) {
    // The neighbor in 'direction' donates when its own code points back at this cell.
    let donor = getFlowReceiver(cell, 1u << direction);
    if (donor == NO_CELL) { continue; }
    if (streams[streamsOffset + donor] == 0u) { continue; }
    // A stream neighbor with an unknown direction might donate: this order cannot be known.
    if (flowDirections[flowDirectionsOffset + donor] == UNRESOLVED_DIRECTION) { return false; }
    if (flowDirections[flowDirectionsOffset + donor] != (1u << ((direction + 4u) & 7u))) { continue; }
    let donorOrder = atomicLoad(&order[orderOffset + donor]);
    if (donorOrder == SENTINEL) { return false; }
    if (donorOrder > maximumOrder) {
      maximumOrder = donorOrder;
      maximumCount = 1u;
    } else if (donorOrder == maximumOrder) {
      maximumCount = maximumCount + 1u;
    }
  }
  var result = 1u;
  if (maximumCount > 0u) {
    result = maximumOrder + select(0u, 1u, maximumCount >= 2u);
  }
  return atomicCompareExchangeWeak(&order[orderOffset + cell], SENTINEL, result).exchanged;
}`;
    const getBody = (
      markChangedWGSL: string
    ) => `if (atomicLoad(&order[orderOffset + index]) != SENTINEL) { return; }
  // Stays unresolved without marking a change: no round can resolve it.
  if (isUnresolvedStream(index)) { return; }
  if (tryFinalize(index)) {
    var walk = getFlowReceiver(index, flowDirections[flowDirectionsOffset + index]);
    for (var step = 0u; step < MAX_WALK_LENGTH; step++) {
      if (walk == NO_CELL || !tryFinalize(walk)) { break; }
      walk = getFlowReceiver(walk, flowDirections[flowDirectionsOffset + walk]);
    }
  }
  if (atomicLoad(&order[orderOffset + index]) == SENTINEL) {
    ${markChangedWGSL}
  }`;
    nodes.push(
      ...createRasterGatedLoopNodes<Parameters>(graph, {
        id: `${id}-order-round`,
        gateId: `${id}-order-gate`,
        operation: OPERATION,
        state,
        maxIterations,
        createRound: ({nodeId, slotBinding, markChangedWGSL, condition, extraResources}) =>
          createWGSLKernelNode<Parameters>(graph, {
            id: nodeId,
            operation: OPERATION,
            variant: 'order-round',
            bindings: [...bindings, slotBinding],
            invocationCount: cellCount,
            declarations,
            body: getBody(markChangedWGSL),
            condition,
            extraResources
          })
      })
    );
    if (output.status?.converged || output.status?.iterationCount) {
      nodes.push(
        createRasterIterationFinalizeNode<Parameters>(graph, {
          id: `${id}-order-finalize`,
          operation: OPERATION,
          state,
          converged: output.status.converged,
          iterationCount: output.status.iterationCount
        })
      );
    }
    return nodes;
  }
}
