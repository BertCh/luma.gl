// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {
  createRasterIterationFinalizeNode,
  createRasterIterationGateNode,
  createRasterIterationResetNode,
  createRasterIterationState,
  getRasterIterationCondition,
  validateRasterIterations
} from '../cost-distance/raster-relaxation';
import {createMapGraphKernelNode, type MapGraphKernelBinding} from '../map-graph-kernels';
import type {GPUMapGraphRecipe} from '../map-graph-types';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../map-graph-utils';
import {
  validateTerrainBuffersDistinct,
  validateTerrainGrid
} from '../terrain-analysis/terrain-analysis-utils';
import {TERRAIN_FLOW_MAXIMUM_WALK_LENGTH} from './terrain-flow-passes';
import {
  getTerrainFlowTracingWGSL,
  validateTerrainTracingCellView,
  validateTerrainTracingConvergedView
} from './terrain-flow-tracing';

const OPERATION = 'GPUTerrainStreamOrder';

/**
 * Properties for {@link GPUTerrainStreamOrder}.
 *
 * Compile-time: `width`, `height`, `maxIterations`, and the presence of `converged`. Per-frame: the
 * contents of `flowDirections` and `streams`. The recipe works on D8 receiver indices only, so it
 * has no cell-size model (`uniform`, `web-mercator` and `geographic` spacing give the same orders).
 *
 * Output aliasing: outputs never share a buffer with an input or with each other.
 */
export type GPUTerrainStreamOrderProps = {
  /** Prefix for node and transient IDs. Defaults to `'terrain-stream-order'`. Compile-time. */
  id?: string;
  /** Grid width in cells. Compile-time. */
  width: number;
  /** Grid height in cells. Compile-time. */
  height: number;
  /**
   * ESRI D8 codes, one per cell: 1 east, 2 south-east, 4 south, 8 south-west, 16 west, 32
   * north-west, 64 north, 128 north-east; 0 terminal; `0xffffffff` invalid. Any other code, and
   * any code pointing outside the grid, is treated as terminal. Typically
   * `GPUTerrainFlow.flowDirections`. Per-frame.
   */
  flowDirections: GraphDataView<'uint32'>;
  /** Stream mask, one word per cell, non-zero for stream cells. Typically `GPUTerrainFlow.streams`. Per-frame. */
  streams: GraphDataView<'uint32'>;
  /**
   * Strahler order per cell: 0 for non-stream and invalid cells, at least 1 on stream cells.
   * Cells that remain unresolved when the iteration limit is hit read `0xffffffff`.
   */
  streamOrder: GraphDataView<'uint32'>;
  /** One row set to 1 when every stream cell resolved. Conservative: can be 0 when the last round finished everything. */
  converged?: GraphDataView<'uint32'>;
  /** Maximum gated rounds in `[1, 1024]`. Defaults to 64. Compile-time. */
  maxIterations?: number;
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
 * The evaluation is a deterministic pull over the receiver forest, the same pattern as
 * {@link GPUTerrainFlow} accumulation: one `atomic<u32>` word per cell holds both readiness
 * (sentinel `0xffffffff`) and the final order, a cell is written once from finalized donors only,
 * and after finalizing a cell a thread walks downstream (up to 1024 steps). Order values depend on
 * the donor orders alone, so they never depend on scheduling. Rounds are bounded by the height of
 * the stream tree. Receiver cycles (caller-supplied directions only) never resolve.
 */
export class GPUTerrainStreamOrder implements GPUMapGraphRecipe {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Stable recipe name. */
  readonly recipe = 'terrain-stream-order';
  /** Validated properties. */
  readonly props: GPUTerrainStreamOrderProps;

  constructor(props: GPUTerrainStreamOrderProps) {
    this.id = props.id ?? this.recipe;
    this.props = props;
    const {id} = this;
    const cellCount = validateTerrainGrid(id, props.width, props.height);
    validateRasterIterations(id, 'maxIterations', props.maxIterations ?? 64);
    validateTerrainTracingCellView(id, 'flowDirections', props.flowDirections, 'uint32', cellCount);
    validateTerrainTracingCellView(id, 'streams', props.streams, 'uint32', cellCount);
    validateTerrainTracingCellView(id, 'streamOrder', props.streamOrder, 'uint32', cellCount);
    validateTerrainTracingConvergedView(id, props.converged);
    const outputs = [props.streamOrder, props.converged];
    validateGraphOutputsDisjointFromInputs(id, outputs, [props.flowDirections, props.streams]);
    validateTerrainBuffersDistinct(id, outputs, []);
  }

  /** Returns the initialization, loop reset, and gated pull rounds. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {width, height} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      props.flowDirections,
      props.streams,
      props.streamOrder,
      props.converged
    ]);
    const cellCount = width * height;
    const maxIterations = props.maxIterations ?? 64;
    const state = createRasterIterationState(graph, `${id}-order`, OPERATION, cellCount, id);
    const tracing = getTerrainFlowTracingWGSL(width, height);
    const nodes: GPUCommandNode<Parameters>[] = [
      createMapGraphKernelNode<Parameters>(graph, {
        id: `${id}-order-init`,
        operation: OPERATION,
        variant: 'order-init',
        bindings: [
          {name: 'flowDirections', view: props.flowDirections, type: 'u32', access: 'read'},
          {name: 'streams', view: props.streams, type: 'u32', access: 'read'},
          {name: 'order', view: props.streamOrder, type: 'u32', access: 'read_write'}
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
    const bindings: MapGraphKernelBinding[] = [
      {name: 'order', view: props.streamOrder, type: 'atomic<u32>', access: 'read_write'},
      {name: 'flowDirections', view: props.flowDirections, type: 'u32', access: 'read'},
      {name: 'streams', view: props.streams, type: 'u32', access: 'read'},
      {name: 'status', view: state.status, type: 'atomic<u32>', access: 'read_write'}
    ];
    const declarations = `${tracing}
const SENTINEL: u32 = 0xffffffffu;
const MAX_WALK_LENGTH: u32 = ${TERRAIN_FLOW_MAXIMUM_WALK_LENGTH}u;

// Returns true when this thread wrote the final order of 'cell'.
fn tryFinalize(cell: u32) -> bool {
  if (atomicLoad(&order[orderOffset + cell]) != SENTINEL) { return false; }
  var maximumOrder = 0u;
  var maximumCount = 0u;
  for (var direction = 0u; direction < 8u; direction++) {
    // The neighbor in 'direction' donates when its own code points back at this cell.
    let donor = getFlowReceiver(cell, 1u << direction);
    if (donor == NO_CELL) { continue; }
    if (flowDirections[flowDirectionsOffset + donor] != (1u << ((direction + 4u) & 7u))) { continue; }
    if (streams[streamsOffset + donor] == 0u) { continue; }
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
    const body = `if (atomicLoad(&order[orderOffset + index]) != SENTINEL) { return; }
  if (tryFinalize(index)) {
    var walk = getFlowReceiver(index, flowDirections[flowDirectionsOffset + index]);
    for (var step = 0u; step < MAX_WALK_LENGTH; step++) {
      if (walk == NO_CELL || !tryFinalize(walk)) { break; }
      walk = getFlowReceiver(walk, flowDirections[flowDirectionsOffset + walk]);
    }
  }
  if (atomicLoad(&order[orderOffset + index]) == SENTINEL) {
    atomicStore(&status[statusOffset], 1u);
  }`;
    for (let iteration = 0; iteration < maxIterations; iteration++) {
      const nodeId = `${id}-order-round-${iteration}`;
      const {condition, extraResources} = getRasterIterationCondition<Parameters>(state, nodeId);
      nodes.push(
        createMapGraphKernelNode<Parameters>(graph, {
          id: nodeId,
          operation: OPERATION,
          variant: 'order-round',
          bindings,
          invocationCount: cellCount,
          declarations,
          body,
          condition,
          extraResources
        }),
        createRasterIterationGateNode<Parameters>(graph, {
          id: `${id}-order-gate-${iteration}`,
          operation: OPERATION,
          state,
          maxIterations
        })
      );
    }
    if (props.converged) {
      nodes.push(
        createRasterIterationFinalizeNode<Parameters>(graph, {
          id: `${id}-order-finalize`,
          operation: OPERATION,
          state,
          converged: props.converged
        })
      );
    }
    return nodes;
  }
}
