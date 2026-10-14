// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  doGraphDataViewsOverlap,
  getViewBindingRange,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPURasterBand, GPURasterMetadata} from '../../gpu-raster/index';
import {writeRasterGridPrefix} from '../../gpu-raster/cost-distance/raster-grid-utils';
import {
  getGPURasterRelaxationContinuationStateLength,
  validateRasterIterations
} from '../../gpu-raster/cost-distance/raster-relaxation';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validatePackedView
} from '../../utils/gpu-contributor-utils';
import {
  getRasterBandViews,
  getRasterBufferBandNodes,
  validateParameterTarget,
  validateRasterBandBelongsToGraph,
  validateRasterGrid,
  validateRasterGridAxisAligned,
  validateRasterParameters
} from '../../gpu-raster/raster-band-utils';
import {
  validateRasterCellSizeMode,
  type GPURasterCellSizeMode
} from '../../analysis-core/raster-grid';
import {
  createTerrainFlowAccumulationNodes,
  createTerrainFlowDirectionNode,
  createTerrainFlowStreamsNode
} from './terrain-flow-passes';
import {createTerrainFlowSpillFillNodes} from './terrain-flow-spill-fill';
import {createTerrainFlowUnresolvedNodes} from './terrain-flow-unresolved';
import {createTerrainFlowFlatNodes} from './terrain-flow-flats';
import {createTerrainFlowRoutingAccumulationNodes} from './terrain-flow-routing';

/**
 * Caller-owned state for three continued flat-resolution passes. All six views must be distinct
 * imported buffers, never graph transients. Each state is zero-filled initially and has
 * `getGPURasterRelaxationContinuationStateLength(grid)` uint32 words; values has one float32 per
 * cell. Set `GPU_RASTER_RELAXATION_RESTART_WORD` in any state when inputs change. Completed
 * stages stay frozen until the entire computation completes; the next execution then restarts.
 * The maximum pass waits on the GPU for the higher pass. No CPU readback drives the schedule.
 */
export type GPUTerrainFlowFlatContinuation = {
  /** Hop distances toward lower edges. */
  lower: {state: GraphDataView<'uint32'>; values: GraphDataView<'float32'>};
  /** Hop distances away from higher edges. */
  higher: {state: GraphDataView<'uint32'>; values: GraphDataView<'float32'>};
  /** Negated component maximum of the higher distances. */
  maximum: {state: GraphDataView<'uint32'>; values: GraphDataView<'float32'>};
};

/** Number of float32 values read from `GPUTerrainFlowProps.parameters`. */
export const GPU_TERRAIN_FLOW_PARAMETER_LENGTH = 8;

/** Receiver and flow-direction marker for cells without a receiver or without data. */
export const GPU_TERRAIN_FLOW_NONE = 0xffffffff;

/**
 * `flowDirections` value of a valid cell whose direction is unresolved because depression filling
 * (`fillDepressions`) or a flat-resolution relaxation (`resolveFlats`) stopped at its iteration
 * limit. Invalid cells keep `GPU_TERRAIN_FLOW_NONE`. A converged run never writes it. Test
 * `>= GPU_TERRAIN_FLOW_UNRESOLVED` for "no usable direction" (this value or `GPU_TERRAIN_FLOW_NONE`).
 */
export const GPU_TERRAIN_FLOW_UNRESOLVED = 0xfffffffe;

/**
 * `streams` value of a valid cell whose accumulation is unresolved (NaN) because the accumulation
 * round limit was hit: whether it is a stream is unknown. A converged run never writes it.
 * {@link GPUTerrainStreamOrder} and {@link GPUTerrainHeightAboveDrainage} propagate it.
 */
export const GPU_TERRAIN_FLOW_STREAM_UNRESOLVED = 0xffffffff;

/**
 * Per-cell classification written to `GPUTerrainFlowOutput.cellClasses`.
 *
 * `draining` cells have a strictly lower neighbor. `flat`, `pit`, and `outlet` cells are terminal
 * (direction 0): a flat has an equal-height neighbor, a pit is a strict local minimum, and an
 * outlet is a boundary cell (grid border or next to an invalid cell) with no lower neighbor.
 * `invalid` cells have no elevation. `unresolved` marks a valid cell of a run whose depression
 * filling or flat resolution hit its iteration limit: its class is unknown. A converged run never
 * writes it.
 */
export const GPU_TERRAIN_FLOW_CELL_CLASS = {
  draining: 0,
  flat: 1,
  pit: 2,
  outlet: 3,
  invalid: 4,
  unresolved: 5
} as const;

/** What a cell contributes to accumulation: `'cells'` counts, `'area'` uses ground area in m^2. */
export type GPUTerrainFlowAccumulationUnits = 'cells' | 'area';

/**
 * How {@link GPUTerrainFlow} splits a cell's accumulated flow among its downslope neighbors.
 *
 * - `'d8'`: everything to the steepest D8 receiver (O'Callaghan and Mark 1984).
 * - `'d-infinity'`: Tarboton (1997) triangular facets; the steepest facet direction splits the
 *   flow between its two bounding neighbors in proportion to the angle.
 * - `'mfd-freeman'`: Freeman (1991) multiple flow direction, weights `tan(beta)^p` over all lower
 *   neighbors (published exponent `p = 1.1`).
 * - `'mfd-quinn'`: Quinn et al. (1991), weights `(tan(beta) * L)^p` with contour lengths `L` of
 *   half the cell side (cardinal) and a quarter of the cell diagonal (diagonal); published `p = 1`.
 *
 * Cells without a strictly lower valid neighbor that received a D8 receiver from flat resolution
 * pass all of their flow to that receiver under every routing.
 */
export type GPUTerrainFlowRouting = 'd8' | 'd-infinity' | 'mfd-freeman' | 'mfd-quinn';

/**
 * CPU-side description packed by {@link getGPUTerrainFlowParameterValues}. Cell size and latitude
 * band come from the grid passed next to it.
 */
export type GPUTerrainFlowSettings = {
  /**
   * Minimum rise per D8 step on filled surfaces, in elevation units. Negative or non-finite values
   * act as 0. Defaults to 0. Must exceed the float32 spacing at the elevation magnitude to create
   * drainable gradients; 0 leaves flats.
   */
  fillEpsilon?: number;
  /** Accumulation at or above which `streams` is 1. Defaults to `Infinity` (no streams). */
  streamThreshold?: number;
  /**
   * Exponent `p` of the multiple-flow-direction weights. Values that are not finite and positive
   * (including the default 0) select the routing's published exponent: 1.1 for `'mfd-freeman'`,
   * 1 for `'mfd-quinn'`. Ignored by `'d8'` and `'d-infinity'`.
   */
  flowExponent?: number;
};

/**
 * Packs `[cellSizeX, cellSizeY, northEdge, southEdge, fillEpsilon, streamThreshold, flowExponent, 0]`,
 * with the cell size and the row-edge world y taken from `grid`.
 *
 * @throws If `target` holds fewer than 8 values.
 */
export function getGPUTerrainFlowParameterValues(
  settings: GPUTerrainFlowSettings,
  grid: GPURasterMetadata,
  target: Float32Array = new Float32Array(GPU_TERRAIN_FLOW_PARAMETER_LENGTH)
): Float32Array {
  validateParameterTarget(
    'getGPUTerrainFlowParameterValues',
    target,
    GPU_TERRAIN_FLOW_PARAMETER_LENGTH
  );
  writeRasterGridPrefix(target, grid);
  target[4] = settings.fillEpsilon ?? 0;
  target[5] = settings.streamThreshold ?? Infinity;
  target[6] = settings.flowExponent ?? 0;
  target[7] = 0;
  return target;
}

/**
 * Enforces the output aliasing rule of {@link GPUTerrainFlow}.
 *
 * An output never shares a buffer with an input. Outputs share a buffer only over disjoint byte
 * ranges, and outputs that one node binds together must also fall in different 256-byte storage
 * binding windows (WebGPU rejects overlapping writable bindings of one buffer in a dispatch).
 */
function validateTerrainFlowOutputAliasing(id: string, props: GPUTerrainFlowProps): void {
  const {output} = props;
  const outputs = [
    ['filledElevation', output.filledElevation],
    ['flowDirections', output.flowDirections],
    ['cellClasses', output.cellClasses],
    ['accumulation', output.accumulation],
    ['streams', output.streams],
    ['status.fillConverged', output.status?.fillConverged],
    ['status.accumulationConverged', output.status?.accumulationConverged],
    ['status.flatsConverged', output.status?.flatsConverged],
    ['status.fillIterationCount', output.status?.fillIterationCount],
    ['status.flatsIterationCount', output.status?.flatsIterationCount],
    ['status.accumulationIterationCount', output.status?.accumulationIterationCount]
  ] as const;
  validateGraphOutputsDisjointFromInputs(
    id,
    outputs.map(([, view]) => view),
    [...getRasterBandViews(props.elevation), props.parameters, props.runoff]
  );
  // Names of outputs that are bound together by one scheduled node.
  const boundTogether = [
    ['filledElevation', 'flowDirections', 'cellClasses'],
    ['accumulation', 'streams']
  ];
  for (const [index, [firstName, first]] of outputs.entries()) {
    for (const [secondName, second] of outputs.slice(index + 1)) {
      if (!first || !second || first.buffer !== second.buffer) {
        continue;
      }
      if (doGraphDataViewsOverlap(first, second)) {
        throw new Error(
          `${id} outputs must not share buffers with overlapping byte ranges (${firstName} and ${secondName})`
        );
      }
      const firstWindow = getViewBindingRange(first);
      const secondWindow = getViewBindingRange(second);
      if (
        boundTogether.some(group => group.includes(firstName) && group.includes(secondName)) &&
        firstWindow.offset < secondWindow.offset + secondWindow.size &&
        secondWindow.offset < firstWindow.offset + firstWindow.size
      ) {
        throw new Error(
          `${id} ${firstName} and ${secondName} are written by one node and must be at least 256 bytes apart when they share a buffer`
        );
      }
    }
  }
}

/** One-row status words of {@link GPUTerrainFlow}. Each view has one row. */
export type GPUTerrainFlowStatus = {
  /** One row set to 1 when depression filling reached its fixpoint. Requires `fillDepressions`. */
  fillConverged?: GraphDataView<'uint32'>;
  /**
   * One row set to 1 when accumulation resolved every valid cell. On the pull path (D8 with
   * `runoff` or `'area'` units, or MFD and D-infinity routing) the flag is conservative: it can be
   * 0 when the last allowed round happened to finish everything. On the Euler-tour path (D8,
   * `'cells'`, no `runoff`) it is 1 unless a receiver cycle was found. Requires
   * `output.accumulation` or `output.streams`.
   */
  accumulationConverged?: GraphDataView<'uint32'>;
  /**
   * One row set to 1 when every flat-resolution relaxation reached its fixpoint. Requires
   * `resolveFlats`.
   */
  flatsConverged?: GraphDataView<'uint32'>;
  /**
   * One row: Boruvka rounds of the spill-graph fill, or gated relaxation iterations (including the
   * final unchanged one) when `fillEpsilon` > 0 or the grid exceeds the spill path's binding size.
   * Requires `fillDepressions`. Presence is compile-time.
   */
  fillIterationCount?: GraphDataView<'uint32'>;
  /** One row: total iterations executed by the three flat-resolution relaxations. Requires `resolveFlats`. Presence is compile-time. */
  flatsIterationCount?: GraphDataView<'uint32'>;
  /**
   * One row: gated accumulation rounds executed (pull rounds, or list-ranking rounds of the
   * Euler-tour path). Requires `output.accumulation` or `output.streams`. Presence is compile-time.
   */
  accumulationIterationCount?: GraphDataView<'uint32'>;
};

/**
 * Caller-owned outputs of {@link GPUTerrainFlow}; at least one of `filledElevation`,
 * `flowDirections`, `cellClasses`, `accumulation` and `streams` is required. Presence of every
 * view is compile-time.
 */
export type GPUTerrainFlowOutput = {
  /**
   * Depression-filled elevation, NaN for invalid cells and for every valid cell when the fill did
   * not converge (`status.fillConverged` 0). Requires `fillDepressions`.
   */
  filledElevation?: GraphDataView<'float32'>;
  /**
   * ESRI D8 code toward the receiver (1 east, 2 south-east, 4 south, 8 south-west, 16 west, 32
   * north-west, 64 north, 128 north-east); 0 for terminal cells; `GPU_TERRAIN_FLOW_NONE` for
   * invalid cells; `GPU_TERRAIN_FLOW_UNRESOLVED` on every valid cell when depression filling or
   * flat resolution did not converge.
   */
  flowDirections?: GraphDataView<'uint32'>;
  /**
   * `GPU_TERRAIN_FLOW_CELL_CLASS` per cell; `unresolved` on every valid cell when depression
   * filling or flat resolution did not converge.
   */
  cellClasses?: GraphDataView<'uint32'>;
  /**
   * Total upstream weight including the cell itself. NaN for invalid cells, for cells left
   * unresolved when accumulation did not converge, and for every valid cell when depression
   * filling or flat resolution did not converge.
   */
  accumulation?: GraphDataView<'float32'>;
  /**
   * 1 where accumulation is finite and at least the `streamThreshold` setting; 0 below the threshold
   * and on invalid cells; `GPU_TERRAIN_FLOW_STREAM_UNRESOLVED` on valid cells whose accumulation
   * is unresolved (NaN) because `maximumAccumulationIterations` ran out, so an unresolved trunk never
   * reads as "not a stream". Test `== 1u` for "is a stream"; `!= 0u` also counts unresolved
   * cells. Also `GPU_TERRAIN_FLOW_STREAM_UNRESOLVED` on every valid cell when depression filling
   * or flat resolution did not converge. Accumulation is computed into a transient when `accumulation` is not provided.
   */
  streams?: GraphDataView<'uint32'>;
  /** One-row convergence and iteration-count words. */
  status?: GPUTerrainFlowStatus;
};

/**
 * Properties for {@link GPUTerrainFlow}.
 *
 * Compile-time: `grid` size, `cellSizeMode`, `fillDepressions`, `resolveFlats`,
 * `flowRouting`, iteration limits, `accumulationUnits`, the elevation format and calibration, and
 * which optional views are provided. Per-frame: the contents of `elevation`, `parameters`, and `runoff`.
 *
 * Output aliasing: an output view never shares a buffer with an input (`elevation`, its validity,
 * `parameters`, `runoff`). Outputs may share one buffer, for example `status.fillConverged` and
 * `status.accumulationConverged` as two 1-row views of one 8-byte summary buffer, only over disjoint byte
 * ranges. Outputs that a single node binds together (`filledElevation`, `flowDirections` and
 * `cellClasses`; `accumulation` and `streams`) must additionally start in different 256-byte
 * storage binding windows of a shared buffer, because WebGPU rejects overlapping writable bindings
 * of one buffer within a dispatch. The convergence flags are written by their own nodes, so they
 * may sit in the same window.
 */
export type GPUTerrainFlowProps = {
  /** Prefix for node and transient IDs. Defaults to `'terrain-flow'`. Compile-time. */
  id?: string;
  /** Raster grid: its width and height size the kernels. */
  grid: GPURasterMetadata;
  /**
   * Elevation band, buffer or texture. Validity, nodata, scale, and offset are honored; cells
   * without a finite calibrated elevation are invalid. Format is compile-time, contents per-frame.
   */
  elevation: GPURasterBand;
  /** Parameters with at least 8 float32 values, see {@link getGPUTerrainFlowParameterValues}. Per-frame. */
  parameters: GraphDataView<'float32'>;
  /** Cell size interpretation. Defaults to `'uniform'`. Compile-time. */
  cellSizeMode?: GPURasterCellSizeMode;
  /** Fill depressions before routing flow. Defaults to false. Compile-time. */
  fillDepressions?: boolean;
  /**
   * Maximum gated iterations in `[1, 1024]` of the relaxation fill, which runs when `fillEpsilon` > 0
   * (selected per frame on the GPU) or when `8 * (cellCount + 1)` bytes exceed one storage binding.
   * Defaults to 512. Compile-time. After 96 plain tiled iterations the relaxation adds directional
   * row and column sweeps; read `output.status.fillConverged` for basins that are still longer.
   * The spill-graph fill used otherwise has no iteration cap.
   */
  maximumFillIterations?: number;
  /**
   * Route flow across flats (Barnes, Lehman and Mulla 2014): cells of class `flat` that are
   * connected through equal-height cells to a cell that drains get a D8 receiver on a synthetic
   * gradient toward lower terrain and away from higher terrain, without changing elevations.
   * Resolved cells become `draining`; flats with no outlet stay `flat`. Defaults to false.
   * Compile-time.
   */
  resolveFlats?: boolean;
  /** Maximum gated iterations of each flat-resolution relaxation in `[1, 1024]`. Defaults to 512, or 64 with flatContinuation. Compile-time. */
  maximumFlatIterations?: number;
  /** Opt-in persistent GPU flat passes, 64 rounds per pass per execution by default. Requires resolveFlats. Inputs must remain unchanged until convergence or an explicit restart. */
  flatContinuation?: GPUTerrainFlowFlatContinuation;
  /**
   * Flow routing used for `output.accumulation` and `output.streams`. `output.flowDirections` and
   * `output.cellClasses` are always D8. Defaults to `'d8'`. Compile-time.
   */
  flowRouting?: GPUTerrainFlowRouting;
  /**
   * Maximum gated accumulation rounds in `[1, 1024]`. Defaults to 64. Compile-time. It bounds the
   * pull path only (D8 with `runoff` or `'area'` units, MFD and D-infinity routing, and D8 grids
   * whose `2 * width * height` tour words exceed one storage binding). D8 `'cells'` accumulation
   * without `runoff` is exact and unbounded: it needs `ceil(log2(2 n)) + 1` rounds whatever the
   * stem length.
   */
  maximumAccumulationIterations?: number;
  /**
   * Optional per-cell weight, one value per cell. Non-finite or negative values count as 0.
   * Presence is compile-time, contents are per-frame.
   */
  runoff?: GraphDataView<'float32'>;
  /**
   * Whether each cell contributes its weight (`'cells'`, default) or its weight times its ground
   * area in m^2 (`'area'`, evaluated at the row center). Compile-time.
   */
  accumulationUnits?: GPUTerrainFlowAccumulationUnits;
  /** Caller-owned outputs. */
  output: GPUTerrainFlowOutput;
};

/**
 * Terrain hydrology on one raster tile: optional depression filling, D8 flow directions, flow
 * accumulation, and a stream mask.
 *
 * Algorithm choices:
 * - Filling computes the Planchon-Darboux fixpoint
 *   `W(c) = max(Z(c), min over neighbors n of W(n) + epsilon)` (boundary cells keep their
 *   elevation). The fixpoint is unique, so every schedule gives the same bits. With `epsilon` 0 it
 *   runs as a tiled spill graph (Barnes 2016): tile-local fill and outlet labels, a Boruvka
 *   spanning tree of the spill graph and bottleneck paths to the grid boundary, in O(log n) GPU
 *   rounds whatever the basin shape. With `epsilon` > 0 (selected per frame on the GPU, no
 *   readback) it runs as a monotone min-relaxation over the shared tiled raster relaxation.
 *   With `epsilon` 0 filled depressions become exact flats (class `flat`, direction 0, no
 *   routing); with `epsilon` above the float32 spacing every filled cell gets a strict descent
 *   toward a boundary.
 * - Flow directions are steepest D8 descent on ground distance (so diagonal moves and latitude
 *   dependent cell sizes are honored); only strictly positive drops route, ties pick the lowest
 *   direction index. Strict drops make the receiver graph a forest.
 * - D8 `'cells'` accumulation without `runoff` is tree contraction on the Euler tour of the
 *   receiver forest (Tarjan and Vishkin 1985): every cell enumerates its children in fixed
 *   direction order, so the tour successors need no CSR, and Wyllie pointer jumping over the `2 n`
 *   tour edges ranks them in `ceil(log2(2 n)) + 1` data-independent rounds (fewer when the
 *   lists converge earlier). The subtree size of a cell is
 *   `(rank(enter) - rank(exit) + 1) / 2`, an exact integer converted once to float32, so long
 *   stems that exhaust the pull rounds are exact. A receiver cycle (never produced by the flow
 *   direction pass) leaves its cells NaN and clears `status.accumulationConverged`.
 * - Weighted accumulation (`runoff`, `'area'`) and the other routings use a deterministic pull
 *   over the receiver forest. A cell's value is written once,
 *   from finalized donors summed in fixed direction order, so its bits never depend on scheduling:
 *   a duplicate or lost compare-exchange race only changes when a value appears, never what it is.
 *   One atomic word carries both readiness (the NaN sentinel `0xffffffff`) and the value, so no
 *   cross-address memory ordering is needed. After finalizing a cell a thread walks downstream
 *   (up to 1024 steps) finalizing receivers whose donors are all ready, so a drainage network
 *   usually resolves in a few rounds; rounds are bounded by the tree height. Alternatives were
 *   rejected: float `atomicAdd` push accumulates in scheduling order (nondeterministic rounding),
 *   pure Jacobi pull resolves one tree level per iteration, and pointer jumping yields ancestor
 *   path sums rather than subtree sums without an Euler tour.
 *
 * A valid cell has a finite calibrated elevation. A boundary cell is valid and lies on the grid
 * border or next to an invalid cell: water leaves the grid there. Not converging within the
 * iteration limits leaves `output.status.fillConverged` or `output.status.accumulationConverged` at 0, unresolved
 * accumulation cells at NaN, and their `streams` at `GPU_TERRAIN_FLOW_STREAM_UNRESOLVED`
 * (never 0, which would drop the downstream trunk from the stream network).
 *
 * A depression fill or flat resolution that stops at its iteration limit (`status.fillConverged` or
 * `status.flatsConverged` 0) leaves the routing surface wrong everywhere it propagates, so on every valid
 * cell `flowDirections` is `GPU_TERRAIN_FLOW_UNRESOLVED`, `cellClasses` is
 * `GPU_TERRAIN_FLOW_CELL_CLASS.unresolved`, `accumulation` is NaN, `streams` is
 * `GPU_TERRAIN_FLOW_STREAM_UNRESOLVED`, and `filledElevation` is NaN (when the fill itself did not
 * converge). Invalid cells keep their codes. A converged run is bit-identical to before.
 *
 * - Flat resolution (`resolveFlats`) follows Barnes, Lehman and Mulla (2014): hop distances toward
 *   the flat's draining edge and away from its higher edge are GPU relaxations over equal-height
 *   cells, combined as `2 * towardLower + (flatMaximum - awayFromHigher)`, and each resolved cell
 *   drains to the equal-height neighbor with the smallest strictly lower value.
 * - D-infinity and multiple-flow-direction accumulation (`flowRouting`) use the same
 *   deterministic pull as D8, with each donor's fraction recomputed from the surface, so values are
 *   written once from finalized donors in fixed direction order.
 *
 * Downstream products take this contributor's outputs: {@link GPUTerrainHeightAboveDrainage},
 * {@link GPUTerrainWatersheds}, {@link GPUTerrainStreamOrder} and
 * {@link GPUTerrainHydrologicIndices}.
 *
 * Non-goals: least-cost breaching and cross-tile (seamless) watersheds. Sequential Priority-Flood
 * is the test oracle of the spill-graph fill, not an option.
 */
export class GPUTerrainFlow implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUTerrainFlowProps;

  constructor(props: GPUTerrainFlowProps) {
    this.id = props.id ?? 'terrain-flow';
    this.props = props;
    const {id} = this;
    const {output} = props;
    const status = output.status;
    const cellCount = validateRasterGrid(id, props.grid);
    validateRasterParameters(id, props.parameters, GPU_TERRAIN_FLOW_PARAMETER_LENGTH);
    validateRasterCellSizeMode(id, props.cellSizeMode);
    validateRasterGridAxisAligned(id, props.grid, props.cellSizeMode);
    if (!['cells', 'area'].includes(props.accumulationUnits ?? 'cells')) {
      throw new Error(`${id} accumulationUnits must be cells or area`);
    }
    validateRasterIterations(id, 'maximumFillIterations', props.maximumFillIterations ?? 512);
    validateRasterIterations(id, 'maximumFlatIterations', props.maximumFlatIterations ?? 512);
    if (props.flatContinuation) {
      if (!props.resolveFlats) throw new Error(`${id} flatContinuation requires resolveFlats`);
      const views = Object.values(props.flatContinuation).flatMap(stage => [
        stage.state,
        stage.values
      ]);
      for (const stage of Object.values(props.flatContinuation)) {
        validatePackedView(stage.state, ['uint32'], `${id} flatContinuation.state`);
        validatePackedView(stage.values, ['float32'], `${id} flatContinuation.values`);
        if (
          stage.state.length !== getGPURasterRelaxationContinuationStateLength(props.grid) ||
          stage.values.length !== cellCount
        ) {
          throw new Error(`${id} flatContinuation has the wrong length`);
        }
      }
      validateGraphOutputsDisjointFromInputs(id, views, [
        ...getRasterBandViews(props.elevation),
        props.parameters,
        props.runoff,
        output.filledElevation,
        output.flowDirections,
        output.cellClasses,
        output.accumulation,
        output.streams,
        ...Object.values(output.status ?? {})
      ]);
      // Continuation stages must never alias one another, even through binding windows.
      for (let first = 0; first < views.length; first++)
        for (let second = first + 1; second < views.length; second++) {
          if (views[first].buffer === views[second].buffer)
            throw new Error(`${id} flatContinuation buffers must be distinct`);
        }
    }
    if (!['d8', 'd-infinity', 'mfd-freeman', 'mfd-quinn'].includes(props.flowRouting ?? 'd8')) {
      throw new Error(`${id} flowRouting must be d8, d-infinity, mfd-freeman, or mfd-quinn`);
    }
    if ((status?.flatsConverged || status?.flatsIterationCount) && !props.resolveFlats) {
      throw new Error(
        `${id} output.status.flatsConverged and flatsIterationCount require resolveFlats`
      );
    }
    if (status?.fillIterationCount && !props.fillDepressions) {
      throw new Error(`${id} output.status.fillIterationCount requires fillDepressions`);
    }
    if (status?.accumulationIterationCount && !output.accumulation && !output.streams) {
      throw new Error(
        `${id} output.status.accumulationIterationCount requires output.accumulation or output.streams`
      );
    }
    validateRasterIterations(
      id,
      'maximumAccumulationIterations',
      props.maximumAccumulationIterations ?? 64
    );
    if (
      !output.filledElevation &&
      !output.flowDirections &&
      !output.cellClasses &&
      !output.accumulation &&
      !output.streams
    ) {
      throw new Error(`${id} requires at least one output`);
    }
    if ((output.filledElevation || status?.fillConverged) && !props.fillDepressions) {
      throw new Error(
        `${id} output.filledElevation and output.status.fillConverged require fillDepressions`
      );
    }
    if (status?.accumulationConverged && !output.accumulation && !output.streams) {
      throw new Error(
        `${id} output.status.accumulationConverged requires output.accumulation or output.streams`
      );
    }
    for (const [name, view] of [
      ['output.filledElevation', output.filledElevation],
      ['output.accumulation', output.accumulation],
      ['runoff', props.runoff]
    ] as const) {
      if (view) {
        validatePackedView(view, ['float32'], `${id} ${name}`);
        if (view.length !== cellCount) {
          throw new Error(`${id} ${name} must contain one value per cell`);
        }
      }
    }
    for (const [name, view] of [
      ['output.flowDirections', output.flowDirections],
      ['output.cellClasses', output.cellClasses],
      ['output.streams', output.streams]
    ] as const) {
      if (view) {
        validatePackedView(view, ['uint32'], `${id} ${name}`);
        if (view.length !== cellCount) {
          throw new Error(`${id} ${name} must contain one value per cell`);
        }
      }
    }
    for (const [name, view] of [
      ['fillConverged', status?.fillConverged],
      ['accumulationConverged', status?.accumulationConverged],
      ['flatsConverged', status?.flatsConverged],
      ['fillIterationCount', status?.fillIterationCount],
      ['flatsIterationCount', status?.flatsIterationCount],
      ['accumulationIterationCount', status?.accumulationIterationCount]
    ] as const) {
      if (view) {
        validatePackedView(view, ['uint32'], `${id} output.status.${name}`);
        if (view.length !== 1) {
          throw new Error(`${id} output.status.${name} must contain one row`);
        }
      }
    }
    validateTerrainFlowOutputAliasing(id, props);
  }

  /** Returns canonicalization, fill, direction, accumulation, and stream nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {output} = props;
    const {status} = output;
    const {width, height} = props.grid;
    const cellSizeMode = props.cellSizeMode ?? 'uniform';
    validateRasterBandBelongsToGraph(id, graph, props.elevation, []);
    validateGraphViewsBelongToGraph(id, graph, [
      props.parameters,
      props.runoff,
      ...Object.values(props.flatContinuation ?? {}).flatMap(stage => [stage.state, stage.values]),
      output.filledElevation,
      output.flowDirections,
      output.cellClasses,
      output.accumulation,
      output.streams,
      status?.fillConverged,
      status?.accumulationConverged,
      status?.flatsConverged,
      status?.fillIterationCount,
      status?.flatsIterationCount,
      status?.accumulationIterationCount
    ]);
    const cellCount = width * height;
    const grid = {width, height, cellSizeMode};
    const source = getRasterBufferBandNodes(graph, id, props.elevation, props.grid, true);
    const nodes: GPUCommandNode<Parameters>[] = [...source.nodes];
    const elevation = source.band.storage.values as GraphDataView<'float32'>;

    let surface = elevation;
    let fillStatus: GraphDataView<'uint32'> | undefined;
    let flatStatuses: GraphDataView<'uint32'>[] = [];
    // Marks the outputs unresolved when the fill or a flat relaxation hit its iteration limit.
    const getUnresolvedNodes = () =>
      createTerrainFlowUnresolvedNodes(graph, {
        id,
        cellCount,
        elevation,
        fillStatus,
        flatStatuses,
        filledElevation: output.filledElevation,
        directions: output.flowDirections,
        classes: output.cellClasses,
        accumulation: output.accumulation,
        streams: output.streams,
        codes: {
          direction: GPU_TERRAIN_FLOW_UNRESOLVED,
          cellClass: GPU_TERRAIN_FLOW_CELL_CLASS.unresolved,
          stream: GPU_TERRAIN_FLOW_STREAM_UNRESOLVED
        }
      });
    if (props.fillDepressions) {
      const filled =
        output.filledElevation ?? createTransientView(graph, `${id}-filled`, 'float32', cellCount);
      const fill = createTerrainFlowSpillFillNodes(graph, {
        ...grid,
        id,
        maxIterations: props.maximumFillIterations ?? 512,
        elevation,
        filled,
        settings: props.parameters,
        converged: status?.fillConverged,
        iterationCount: status?.fillIterationCount
      });
      nodes.push(...fill.nodes);
      fillStatus = fill.status;
      surface = filled;
    }

    const needsAccumulation = Boolean(output.accumulation || output.streams);
    if (!output.flowDirections && !output.cellClasses && !needsAccumulation) {
      nodes.push(...getUnresolvedNodes());
      return nodes;
    }
    const receivers = needsAccumulation
      ? createTransientView(graph, `${id}-receivers`, 'uint32', cellCount)
      : undefined;
    // Flat resolution reads the D8 classes, so they exist even when the caller did not ask.
    const classes =
      output.cellClasses ??
      (props.resolveFlats
        ? createTransientView(graph, `${id}-classes`, 'uint32', cellCount)
        : undefined);
    nodes.push(
      createTerrainFlowDirectionNode(graph, {
        ...grid,
        id: `${id}-flow-direction`,
        elevation,
        surface,
        settings: props.parameters,
        directions: output.flowDirections,
        classes,
        receivers,
        cellClass: GPU_TERRAIN_FLOW_CELL_CLASS
      })
    );
    if (props.resolveFlats && classes) {
      const flats = createTerrainFlowFlatNodes(graph, {
        ...grid,
        id,
        maxIterations: props.maximumFlatIterations ?? (props.flatContinuation ? 64 : 512),
        continuation: props.flatContinuation,
        elevation,
        surface,
        settings: props.parameters,
        classes,
        directions: output.flowDirections,
        receivers,
        converged: status?.flatsConverged,
        iterationCount: status?.flatsIterationCount,
        cellClass: GPU_TERRAIN_FLOW_CELL_CLASS
      });
      nodes.push(...flats.nodes);
      flatStatuses = flats.statuses;
    }

    if (receivers) {
      const accumulation =
        output.accumulation ??
        createTransientView(graph, `${id}-accumulation`, 'float32', cellCount);
      const flowRouting = props.flowRouting ?? 'd8';
      const accumulationProps = {
        ...grid,
        id,
        maxIterations: props.maximumAccumulationIterations ?? 64,
        elevation,
        receivers,
        accumulation,
        settings: props.parameters,
        runoff: props.runoff,
        area: (props.accumulationUnits ?? 'cells') === 'area',
        converged: status?.accumulationConverged,
        iterationCount: status?.accumulationIterationCount
      };
      nodes.push(
        ...(flowRouting === 'd8'
          ? createTerrainFlowAccumulationNodes(graph, accumulationProps)
          : createTerrainFlowRoutingAccumulationNodes(graph, {
              ...accumulationProps,
              routing: flowRouting,
              surface
            }))
      );
      if (output.streams) {
        nodes.push(
          createTerrainFlowStreamsNode(graph, {
            id: `${id}-streams`,
            cellCount,
            elevation,
            accumulation,
            streams: output.streams,
            settings: props.parameters,
            unresolvedValue: GPU_TERRAIN_FLOW_STREAM_UNRESOLVED
          })
        );
      }
    }
    nodes.push(...getUnresolvedNodes());
    return nodes;
  }
}
