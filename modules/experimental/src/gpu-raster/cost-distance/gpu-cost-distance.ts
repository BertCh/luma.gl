// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUHistogram,
  type GPUCommandGraph,
  type GPUCommandGraphNodeCondition,
  type GPUCommandNode,
  type GraphBufferUse,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPURasterBand, GPURasterMetadata} from '../types';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {
  validateGraphViewsBelongToGraph,
  validatePackedView
} from '../../utils/gpu-contributor-utils';
import {
  validateRasterCellSizeMode,
  type GPURasterCellSizeMode
} from '../../analysis-core/raster-grid';
import {
  getRasterBandViews,
  getRasterBufferBandNodes,
  validateParameterTarget,
  validateRasterBandBelongsToGraph,
  validateRasterBuffersDistinct,
  validateRasterGrid,
  validateRasterGridAxisAligned,
  validateRasterParameters
} from '../raster-band-utils';
import {
  createContributorTransientView,
  getRasterGridWGSL,
  writeRasterGridPrefix
} from './raster-grid-utils';
import {
  createRasterIterationFinalizeNode,
  createRasterIterationState,
  getRasterIterationCondition,
  type RasterIterationState,
  createRasterTiledRelaxation,
  createRasterTiledRelaxationNodes,
  getGPURasterRelaxationContinuationStateLength,
  getRasterRelaxationTileWGSL,
  validateRasterIterations
} from './raster-relaxation';
import {GET_QUIET_NAN_WGSL, IS_FINITE_VALUE_WGSL} from '../../utils/wgsl-common';

const OPERATION = 'GPUCostDistance';
const DEFAULT_MAXIMUM_ITERATIONS = 512;
/**
 * Default iterations per execution with `continuation`: 64 unrolled rounds are 16 gate nodes and
 * about 80 graph nodes instead of 640, and two executions cover the 96 plain tiled rounds that
 * precede the first directional sweep, so even a corridor needs a handful of submissions.
 */
const DEFAULT_CONTINUATION_MAXIMUM_ITERATIONS = 64;
const DEFAULT_SWEEP_AFTER_ITERATION = 96;

/** Number of float32 values read from {@link GPUCostDistanceProps.parameters}. */
export const GPU_COST_DISTANCE_PARAMETER_LENGTH = 8;

/** Sentinel written for "no back-link" (unreached or impassable) and "no band". */
export const GPU_COST_DISTANCE_NONE = 0xffffffff;

/**
 * Back-link, band and band count written when the iteration budget ran out before the value
 * could be certified (costs are then NaN, see {@link GPUCostDistanceOutput.costs}). Back-links: every
 * passable cell when the cost relaxation did not converge (its cost may still drop, and an
 * infinite cost may still become finite), and a zero-cost plateau cell whose tie level did not
 * converge. A converged run never writes it.
 */
export const GPU_COST_DISTANCE_UNRESOLVED = 0xfffffffe;

/**
 * CPU description of the settings packed by {@link getGPUCostDistanceParameterValues}. Cell size
 * and latitude band come from the grid passed next to it.
 */
export type GPUCostDistanceSettings = {
  /** Candidate costs above this value stay unreached. Defaults to `Infinity`. */
  costLimit?: number;
};

/**
 * Packs settings into the 8-float layout read by {@link GPUCostDistance}:
 * `[cellSizeX, cellSizeY, northEdge, southEdge, costLimit, 0, 0, 0]`, with the cell size and the
 * row-edge world y taken from `grid`.
 */
export function getGPUCostDistanceParameterValues(
  settings: GPUCostDistanceSettings,
  grid: GPURasterMetadata,
  target: Float32Array = new Float32Array(GPU_COST_DISTANCE_PARAMETER_LENGTH)
): Float32Array {
  validateParameterTarget(
    'getGPUCostDistanceParameterValues',
    target,
    GPU_COST_DISTANCE_PARAMETER_LENGTH
  );
  target.fill(0, 0, GPU_COST_DISTANCE_PARAMETER_LENGTH);
  writeRasterGridPrefix(target, grid);
  target[4] = settings.costLimit ?? Infinity;
  return target;
}

/** One-row status words of {@link GPUCostDistance}. */
export type GPUCostDistanceStatus = {
  /**
   * One row: 1 when relaxation reached a fixpoint, 0 when `maximumIterations` stopped it. With
   * `output.backLinks`, also 0 when `maximumTieIterations` stopped the tie phase; the affected
   * back-links then read `GPU_COST_DISTANCE_UNRESOLVED`.
   */
  converged?: GraphDataView<'uint32'>;
  /** One row: number of relaxation iterations executed. */
  iterationCount?: GraphDataView<'uint32'>;
};

/**
 * Caller-owned state that lets {@link GPUCostDistance} spread one relaxation over several
 * executions (submissions) instead of unrolling every round into one graph.
 *
 * WebGPU has no GPU-side command loop, so rounds beyond the unrolled count can only run in a later
 * submission. The progress must outlive an execution, and graph transients may be aliased by the
 * lifetime planner, so both views must be imported buffers that nothing else writes.
 */
export type GPUCostDistanceContinuation = {
  /**
   * `uint32` state of {@link getGPURasterRelaxationContinuationStateLength} words (loop status,
   * restart word, tile activity). Create it zero-filled: zero means "no previous execution".
   * To force a restart, write a nonzero value at element
   * {@link GPU_RASTER_RELAXATION_RESTART_WORD} before the execution; the GPU clears it.
   */
  state: GraphDataView<'uint32'>;
  /**
   * `float32` working costs, one per cell, distinct from `output.costs`. They carry the partial
   * result between executions, while `output.costs` is rewritten each execution (NaN where
   * unresolved, see {@link GPUCostDistanceOutput.costs}).
   */
  values: GraphDataView<'float32'>;
};

/** Caller-owned outputs of {@link GPUCostDistance}. */
export type GPUCostDistanceOutput = {
  /**
   * Cost per cell, `+Infinity` when unreached or impassable. Its length must equal
   * `grid.width * grid.height`. When `maximumIterations` stopped the cost relaxation before a
   * fixpoint, every passable cell is NaN (unresolved: its cost might still drop, and an unreached
   * cell might still be reached), so a truncated run never reads as a converged one (library
   * rule 9). A converged run never writes NaN.
   */
  costs: GraphDataView<'float32'>;
  /**
   * Optional D8 code (ESRI power of two) of the move from a cell toward the next cell on a
   * least-cost path back to a source. 0 at cells reached as a source,
   * `GPU_COST_DISTANCE_NONE` when unreached or impassable, and `GPU_COST_DISTANCE_UNRESOLVED`
   * when an iteration budget ran out first (see that constant), so a truncated run never
   * produces a link that reads as a source or a path that looks complete.
   *
   * Cycle-safe across zero-cost plateaus: a cell links to its smallest-direction strictly cheaper
   * tight neighbor when one exists; otherwise (it was reached only across zero-cost edges) to the
   * smallest-direction equal-cost zero-cost neighbor exactly one tie level closer to a strict entry
   * or a source. Cost never increases along links and the level strictly decreases at equal cost, so
   * following links always reaches a source. The tie phase runs only when this view is requested.
   */
  backLinks?: GraphDataView<'uint32'>;
  /**
   * Optional band per cell: cost in `(costBreaks[i - 1], costBreaks[i]]`, `GPU_COST_DISTANCE_NONE`
   * beyond the last break or unreached, and `GPU_COST_DISTANCE_UNRESOLVED` where the cost is
   * unresolved (see `costs`).
   */
  bands?: GraphDataView<'uint32'>;
  /**
   * Optional cell count per band (composes `GPUHistogram`). Every count is
   * `GPU_COST_DISTANCE_UNRESOLVED` when the cost relaxation did not converge, since unresolved
   * cells belong to no band yet.
   */
  bandCounts?: GraphDataView<'uint32'>;
  /** Optional one-row convergence and iteration-count words. */
  status?: GPUCostDistanceStatus;
};

/**
 * Properties for {@link GPUCostDistance}.
 *
 * Compile-time: `grid` size, `cellSizeMode`, `maximumIterations`, the capacity of `sources`, the
 * friction band's format and calibration, and which optional views exist. Per-frame: the contents
 * of `friction`, `parameters` (cell size, latitude edges, `costLimit`), `sources`, `sourceCosts`,
 * `sourceCount`, `sourceMask`, and `costBreaks`.
 */
export type GPUCostDistanceProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'cost-distance'`. */
  id?: string;
  /** Raster grid: its width and height size the kernels. */
  grid: GPURasterMetadata;
  /**
   * Cost per ground meter. Compile-time format and calibration; contents are per-frame. Validity,
   * nodata, scale, and offset are honored. Invalid, NaN, infinite, or negative samples are
   * impassable barriers; zero friction is allowed.
   */
  friction: GPURasterBand;
  /**
   * Optional per-frame friction calibration with at least 2 float32 values
   * `[scale, offset]`, applied after the band's own calibration: `friction = value * scale +
   * offset`. Presence is compile-time; contents are per-frame, so a slider can change the cost
   * surface without recompiling. Absent means the identity. Results that turn negative or
   * non-finite are impassable like any other invalid friction.
   */
  frictionParameters?: GraphDataView<'float32'>;
  /** Per-frame parameters with at least 8 float32 values, see {@link getGPUCostDistanceParameterValues}. */
  parameters: GraphDataView<'float32'>;
  /** Cell size interpretation. Compile-time. Defaults to `'uniform'`. */
  cellSizeMode?: GPURasterCellSizeMode;
  /**
   * Source cell indices (`row * width + column`). Capacity is the length (compile-time); contents
   * are per-frame. Out-of-range or impassable cells are ignored.
   */
  sources?: GraphDataView<'uint32'>;
  /** Optional initial cost per source row, default 0. Negative or non-finite costs are ignored. Per-frame contents. */
  sourceCosts?: GraphDataView<'float32'>;
  /** Optional one-row active source count; rows at or after it are ignored. Per-frame contents. */
  sourceCount?: GraphDataView<'uint32'>;
  /** Optional per-cell mask; a nonzero cell is a source at cost 0. Per-frame contents. */
  sourceMask?: GraphDataView<'uint32'>;
  /**
   * Number of unrolled gated relaxation iterations, 1 to `GPU_RASTER_MAXIMUM_ITERATIONS`.
   * Compile-time. Defaults to 512, or to 64 with `continuation`, where it is the budget of one
   * execution.
   */
  maximumIterations?: number;
  /**
   * Opt-in continuation across executions, see {@link GPUCostDistanceContinuation}. Each
   * execution resumes where the previous one stopped unconverged: it skips initialization and
   * seeding and runs up to `maximumIterations` more rounds, with the sweep schedule following the
   * cumulative iteration count. `output.status` reports `converged` and `iterationCount`
   * cumulatively; `output.costs` is NaN for passable cells until the execution that converges.
   * An execution after a converged one, or after the restart word was set, starts from scratch. A
   * caller that changes any input (friction, parameters, sources) must set the restart word;
   * between restarts, inputs must stay unchanged. The caller polls `converged` (a readback of its
   * choosing) and submits again until it is 1 or its own budget ends. Not supported with
   * `output.backLinks`. Absent, behavior is unchanged and every encoding starts from scratch.
   */
  continuation?: GPUCostDistanceContinuation;
  /**
   * Iteration index at which relaxation switches from plain tiled passes to a cycle of tiled and
   * directional tile sweeps (right, left, down, up). Compile-time. Defaults to 96. Plain tiled
   * passes move a front one 16x16 tile per iteration; a sweep pass carries it along a whole row or
   * column of tiles, so long corridors, serpentines, and spirals converge in far fewer iterations.
   * Open terrain normally converges before the switch and is unaffected. Set it to `maximumIterations`
   * or more to keep plain tiled passes only.
   */
  sweepAfterIteration?: number;
  /**
   * Number of unrolled gated tie-level iterations, 1 to `GPU_RASTER_MAXIMUM_ITERATIONS`. Compile-time.
   * Defaults to `maximumIterations`. Used only when `output.backLinks` is requested. The tie relaxation uses
   * the cost phase's schedule: plain tiled passes (one 16x16 tile of plateau per iteration), then
   * directional tile sweeps from `sweepAfterIteration` on, so long or winding zero-cost plateaus
   * converge. Iterations after convergence are skipped on the GPU. A plateau that the budget
   * cannot cover leaves `output.status.converged` at 0 and its unresolved cells at
   * `GPU_COST_DISTANCE_UNRESOLVED`.
   */
  maximumTieIterations?: number;
  /** Isochrone cost breaks, ascending. Required when `output.bands` or `output.bandCounts` is given. Per-frame contents, compile-time length. */
  costBreaks?: GraphDataView<'float32'>;
  /** Caller-owned outputs. */
  output: GPUCostDistanceOutput;
};

/**
 * Computes least-accumulated-cost distance over an 8-connected raster friction surface, with
 * optional back-links, isochrone bands, and a GPU convergence flag.
 *
 * The edge cost of a D8 move is the ground length of the move times the mean friction of its two
 * cells. Relaxation is tiled min-relaxation: each 16x16 workgroup tile repeats the update in
 * workgroup memory, and a GPU-written indirect dispatch switches the remaining unrolled
 * iterations off once nothing changes, so nothing is read back. Diagonal moves are allowed
 * whenever both endpoints are passable (no corner-cutting restriction). Every encoding recomputes
 * from scratch, unless `continuation` resumes an unconverged run. Zero-friction plateau cells share their cost with the source; their back-links
 * still walk to the source (see `backLinks`).
 *
 * Non-goals: anisotropic or vertical-factor costs (slope-dependent, Tobler hiking), knight's-move
 * (16-neighbor) connectivity, corner-cutting rules, multi-tile or cross-tile seams, and an
 * allocation (nearest-source ID) output, which is an open extension.
 */
export class GPUCostDistance implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUCostDistanceProps;
  /** Resolved compile-time iteration count. */
  readonly maximumIterations: number;
  /** Resolved compile-time tie-phase iteration count. */
  readonly maximumTieIterations: number;

  constructor(props: GPUCostDistanceProps) {
    this.id = props.id ?? 'cost-distance';
    this.props = props;
    this.maximumIterations =
      props.maximumIterations ??
      (props.continuation ? DEFAULT_CONTINUATION_MAXIMUM_ITERATIONS : DEFAULT_MAXIMUM_ITERATIONS);
    const {id} = this;
    const {output} = props;
    const cellCount = validateRasterGrid(id, props.grid);
    validateRasterCellSizeMode(id, props.cellSizeMode);
    validateRasterGridAxisAligned(id, props.grid, props.cellSizeMode);
    validateRasterIterations(id, 'maximumIterations', this.maximumIterations);
    this.maximumTieIterations = props.maximumTieIterations ?? this.maximumIterations;
    validateRasterIterations(id, 'maximumTieIterations', this.maximumTieIterations);
    validateRasterParameters(id, props.parameters, GPU_COST_DISTANCE_PARAMETER_LENGTH);
    for (const [name, view] of [
      ['sources', props.sources],
      ['sourceCount', props.sourceCount],
      ['sourceMask', props.sourceMask],
      ['output.backLinks', output.backLinks],
      ['output.bands', output.bands],
      ['output.bandCounts', output.bandCounts],
      ['output.status.converged', output.status?.converged],
      ['output.status.iterationCount', output.status?.iterationCount]
    ] as const) {
      if (view) {
        validatePackedView(view, ['uint32'], `${id} ${name}`);
      }
    }
    for (const [name, view] of [
      ['sourceCosts', props.sourceCosts],
      ['output.costs', output.costs],
      ['costBreaks', props.costBreaks],
      ['frictionParameters', props.frictionParameters],
      ['continuation.values', props.continuation?.values]
    ] as const) {
      if (view) {
        validatePackedView(view, ['float32'], `${id} ${name}`);
      }
    }
    if (props.continuation) {
      validatePackedView(props.continuation.state, ['uint32'], `${id} continuation.state`);
      if (output.backLinks) {
        throw new Error(`${id} continuation does not support output.backLinks`);
      }
      if (props.continuation.values.length !== cellCount) {
        throw new Error(`${id} continuation.values must contain one value per cell`);
      }
      if (
        props.continuation.state.length !==
        getGPURasterRelaxationContinuationStateLength(props.grid)
      ) {
        throw new Error(`${id} continuation.state has the wrong length`);
      }
    }
    const frictionStorage = props.friction.storage;
    if (frictionStorage.kind === 'buffer' && frictionStorage.values.length !== cellCount) {
      throw new Error(`${id} friction must contain one value per cell`);
    }
    if (props.friction.validity && props.friction.validity.length !== cellCount) {
      throw new Error(`${id} friction validity must contain one value per cell`);
    }
    if (!props.sources && !props.sourceMask) {
      throw new Error(`${id} requires sources or sourceMask`);
    }
    if ((props.sourceCosts || props.sourceCount) && !props.sources) {
      throw new Error(`${id} sourceCosts and sourceCount require sources`);
    }
    if (props.sourceCosts && props.sourceCosts.length !== props.sources?.length) {
      throw new Error(`${id} sourceCosts length must equal sources length`);
    }
    if (props.sourceMask && props.sourceMask.length !== cellCount) {
      throw new Error(`${id} sourceMask must contain one value per cell`);
    }
    for (const [name, view] of [
      ['output.costs', output.costs],
      ['output.backLinks', output.backLinks],
      ['output.bands', output.bands]
    ] as const) {
      if (view && view.length !== cellCount) {
        throw new Error(`${id} ${name} must contain one value per cell`);
      }
    }
    for (const [name, view] of [
      ['sourceCount', props.sourceCount],
      ['output.status.converged', output.status?.converged],
      ['output.status.iterationCount', output.status?.iterationCount]
    ] as const) {
      if (view && view.length !== 1) {
        throw new Error(`${id} ${name} must contain exactly one row`);
      }
    }
    if (props.frictionParameters && props.frictionParameters.length < 2) {
      throw new Error(`${id} frictionParameters must contain at least 2 values`);
    }
    const needsCostBreaks = Boolean(output.bands || output.bandCounts);
    if (needsCostBreaks !== Boolean(props.costBreaks)) {
      throw new Error(
        `${id} costBreaks is required exactly when output.bands or output.bandCounts is given`
      );
    }
    if (props.costBreaks && props.costBreaks.length < 1) {
      throw new Error(`${id} costBreaks must contain at least one break`);
    }
    if (output.bandCounts && output.bandCounts.length !== props.costBreaks?.length) {
      throw new Error(`${id} output.bandCounts length must equal costBreaks length`);
    }
    validateRasterBuffersDistinct(
      id,
      [
        output.costs,
        output.backLinks,
        output.bands,
        output.bandCounts,
        output.status?.converged,
        output.status?.iterationCount,
        props.continuation?.state,
        props.continuation?.values
      ],
      [
        ...getRasterBandViews(props.friction),
        props.parameters,
        props.sources,
        props.sourceCosts,
        props.sourceCount,
        props.sourceMask,
        props.costBreaks,
        props.frictionParameters
      ]
    );
  }

  /** Returns friction preparation, gated tiled relaxation, and optional output nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, maximumIterations} = this;
    const {output} = props;
    const {width, height} = props.grid;
    const cellSizeMode = props.cellSizeMode ?? 'uniform';
    validateRasterBandBelongsToGraph(id, graph, props.friction, []);
    validateGraphViewsBelongToGraph(id, graph, [
      props.parameters,
      props.frictionParameters,
      props.sources,
      props.sourceCosts,
      props.sourceCount,
      props.sourceMask,
      output.costs,
      output.backLinks,
      props.costBreaks,
      output.bands,
      output.bandCounts,
      output.status?.converged,
      output.status?.iterationCount,
      props.continuation?.state,
      props.continuation?.values
    ]);
    const cellCount = width * height;
    const friction = getRasterBufferBandNodes(
      graph,
      `${id}-friction`,
      props.friction,
      props.grid,
      true
    );
    const auxiliary = createContributorTransientView(
      graph,
      OPERATION,
      id,
      `${id}-friction-auxiliary`,
      'float32',
      cellCount
    );
    const nodes: GPUCommandNode<Parameters>[] = [
      ...friction.nodes,
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-friction-prepare`,
        operation: OPERATION,
        variant: 'friction-prepare',
        bindings: [
          {
            name: 'frictionValues',
            view: (friction.band.storage as {values: GraphDataView<'float32'>}).values,
            type: 'f32',
            access: 'read'
          },
          {
            name: 'frictionValidity',
            view: friction.band.validity!,
            type: 'u32',
            access: 'read'
          },
          {
            name: 'auxiliary',
            view: auxiliary,
            type: 'f32',
            access: 'read_write'
          },
          ...(props.frictionParameters
            ? [
                {
                  name: 'frictionParameters',
                  view: props.frictionParameters,
                  type: 'f32' as const,
                  access: 'read' as const
                }
              ]
            : [])
        ],
        invocationCount: cellCount,
        declarations: /* wgsl */ `
${IS_FINITE_VALUE_WGSL}
${GET_QUIET_NAN_WGSL}`,
        body: `${
          props.frictionParameters
            ? 'let value = frictionValues[frictionValuesOffset + index] * frictionParameters[frictionParametersOffset] + frictionParameters[frictionParametersOffset + 1u];'
            : 'let value = frictionValues[frictionValuesOffset + index];'
        }
  let passable = frictionValidity[frictionValidityOffset + index] != 0u && isFiniteValue(value) && value >= 0.0;
  auxiliary[auxiliaryOffset + index] = select(getQuietNaN(), value, passable);`
      })
    ];
    const {continuation} = props;
    // A continued run keeps its progress in the caller-owned working costs; `output.costs` is
    // republished from them each execution, because the unresolved pass overwrites it with NaN.
    const working = continuation?.values ?? output.costs;
    const seedCapacity = Math.max(props.sources?.length ?? 0, props.sourceMask ? cellCount : 0);
    const relaxationProps = {
      id,
      operation: OPERATION,
      width,
      height,
      cellSizeMode,
      maxIterations: maximumIterations,
      continuation: continuation && {
        state: continuation.state,
        // Slot 0 is the engine's tile reset; these are the initialize and seed kernels.
        restartShapes: [[cellCount], [Math.max(seedCapacity, 1)]] as const
      }
    };
    const sweepAfterIteration = props.sweepAfterIteration ?? DEFAULT_SWEEP_AFTER_ITERATION;
    const {relaxation, resetNodes} = createRasterTiledRelaxation<Parameters>(
      graph,
      relaxationProps,
      false
    );
    const initializeGate = relaxation.continuation?.getRestartGate(0, `${id}-initialize`);
    nodes.push(
      ...resetNodes,
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-initialize`,
        operation: OPERATION,
        variant: 'initialize',
        bindings: [
          {
            name: 'costs',
            view: working,
            type: 'u32',
            access: 'read_write'
          }
        ],
        invocationCount: cellCount,
        body: 'costs[costsOffset + index] = 0x7f800000u;',
        condition: initializeGate?.condition,
        extraResources: initializeGate?.extraResources
      })
    );
    let hasSeedNode = false;
    if ((props.sources?.length ?? 0) > 0 || props.sourceMask) {
      nodes.push(
        this.getSeedNode(graph, auxiliary, relaxation.tileStamps, cellSizeMode, 'costs', {
          costs: working,
          gate: relaxation.continuation?.getRestartGate(1, `${id}-seed`)
        })
      );
      hasSeedNode = true;
    }
    nodes.push(
      ...createRasterTiledRelaxationNodes<Parameters>(graph, {
        ...relaxationProps,
        relaxation,
        sweepAfterIteration,
        values: working,
        auxiliary,
        settings: props.parameters,
        additiveEdgeCosts: true,
        declarations: /* wgsl */ `
${EDGE_COST_WGSL}
fn getRelaxationEdgeCost(neighborAuxiliary: f32, centerAuxiliary: f32, centerRow: u32, direction: u32) -> f32 {
  return getEdgeCost(direction, centerRow, neighborAuxiliary, centerAuxiliary);
}
fn getRelaxationLimit() -> f32 { return settings[settingsOffset + 4u]; }`
      })
    );
    let tie: {levels: GraphDataView<'float32'>; masks: GraphDataView<'float32'>} | undefined;
    if (output.backLinks) {
      tie = this.getTieNodes(graph, {
        nodes,
        auxiliary,
        cellSizeMode,
        hasSeedNode,
        sweepAfterIteration,
        costState: relaxation.state
      });
    } else if (output.status?.converged || output.status?.iterationCount) {
      nodes.push(
        createRasterIterationFinalizeNode<Parameters>(graph, {
          id: `${id}-finalize`,
          operation: OPERATION,
          state: relaxation.state,
          converged: output.status?.converged,
          iterationCount: output.status?.iterationCount
        })
      );
    }
    if (output.backLinks && tie) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-back-links`,
          operation: OPERATION,
          variant: 'back-links',
          bindings: [
            {name: 'costs', view: output.costs, type: 'f32', access: 'read'},
            {name: 'auxiliary', view: auxiliary, type: 'f32', access: 'read'},
            {
              name: 'settings',
              view: props.parameters,
              type: 'f32',
              access: 'read'
            },
            {name: 'levels', view: tie.levels, type: 'f32', access: 'read'},
            {name: 'masks', view: tie.masks, type: 'f32', access: 'read'},
            {
              name: 'costStatus',
              view: relaxation.state.status,
              type: 'u32',
              access: 'read'
            },
            {
              name: 'backLinks',
              view: output.backLinks,
              type: 'u32',
              access: 'read_write'
            }
          ],
          invocationCount: cellCount,
          declarations: `${getRasterGridWGSL({width, height, cellSizeMode})}
${EDGE_COST_WGSL}
${STRICT_PREDECESSOR_WGSL}
const UNRESOLVED: u32 = ${GPU_COST_DISTANCE_UNRESOLVED}u;`,
          body: `let centerAuxiliary = auxiliary[auxiliaryOffset + index];
  let centerCost = costs[costsOffset + index];
  var link = GRID_NONE;
  if (!isNaNValue(centerAuxiliary) && costStatus[costStatusOffset + 3u] == 0u) {
    // The cost relaxation stopped at maxIterations: any cost may still drop and an infinite
    // cost may still become finite, so no link of a passable cell is certified.
    link = UNRESOLVED;
  } else if (!isNaNValue(centerAuxiliary) && isFiniteValue(centerCost)) {
    let strictDirection = getStrictDirection(index, centerAuxiliary, centerCost);
    if (strictDirection != GRID_NONE) {
      link = getD8Code(strictDirection);
    } else {
      // Reached only across zero-cost edges (or a root source): link to the smallest direction
      // one tie level lower. Level 0 without a strict link is a root (link 0). A level the tie
      // relaxation never reached, or one with no neighbor one level lower (a budget cut it
      // short), is unresolved: link 0 would claim a source.
      let level = levels[levelsOffset + index];
      if (isFiniteValue(level) && level < 0.5) {
        link = 0u;
      } else {
        link = UNRESOLVED;
        if (isFiniteValue(level)) {
          let mask = u32(masks[masksOffset + index]);
          for (var direction = 0u; direction < 8u; direction++) {
            if (((mask >> direction) & 1u) == 0u) { continue; }
            let neighbor = getD8Neighbor(index, direction);
            if (neighbor != GRID_NONE && levels[levelsOffset + neighbor] == level - 1.0) {
              link = getD8Code(direction);
              break;
            }
          }
        }
      }
    }
  }
  backLinks[backLinksOffset + index] = link;`
        })
      );
    }
    if (continuation) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-publish`,
          operation: OPERATION,
          variant: 'publish',
          bindings: [
            {name: 'working', view: continuation.values, type: 'u32', access: 'read'},
            {name: 'costs', view: output.costs, type: 'u32', access: 'read_write'}
          ],
          invocationCount: cellCount,
          body: 'costs[costsOffset + index] = working[workingOffset + index];'
        })
      );
    }
    nodes.push(...this.getUnresolvedCostNodes(graph, auxiliary, relaxation.state));
    if (props.costBreaks && (output.bands || output.bandCounts)) {
      const bands =
        output.bands ??
        createContributorTransientView(
          graph,
          OPERATION,
          id,
          `${id}-bands-scratch`,
          'uint32',
          cellCount
        );
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-bands`,
          operation: OPERATION,
          variant: 'bands',
          bindings: [
            {name: 'costs', view: output.costs, type: 'f32', access: 'read'},
            {
              name: 'thresholds',
              view: props.costBreaks,
              type: 'f32',
              access: 'read'
            },
            {name: 'bands', view: bands, type: 'u32', access: 'read_write'}
          ],
          invocationCount: cellCount,
          declarations: `const THRESHOLD_COUNT: u32 = ${props.costBreaks.length}u;`,
          body: `let cost = costs[costsOffset + index];
  var band = ${GPU_COST_DISTANCE_NONE}u;
  if ((bitcast<u32>(cost) & 0x7fffffffu) > 0x7f800000u) {
    // NaN: the cost relaxation did not converge, so the cell's band is not known yet.
    band = ${GPU_COST_DISTANCE_UNRESOLVED}u;
  } else if (bitcast<u32>(cost) < 0x7f800000u) {
    var passedCount = 0u;
    for (var thresholdIndex = 0u; thresholdIndex < THRESHOLD_COUNT; thresholdIndex++) {
      if (thresholds[thresholdsOffset + thresholdIndex] < cost) { passedCount++; }
    }
    if (passedCount < THRESHOLD_COUNT) { band = passedCount; }
  }
  bands[bandsOffset + index] = band;`
        })
      );
      if (output.bandCounts) {
        nodes.push(
          ...new GPUHistogram({
            id: `${id}-band-counts`,
            input: bands,
            output: output.bandCounts,
            domain: [0, props.costBreaks.length]
          }).getCommandNodes(graph)
        );
        nodes.push(
          createWGSLKernelNode<Parameters>(graph, {
            id: `${id}-band-counts-unresolved`,
            operation: OPERATION,
            variant: 'band-counts-unresolved',
            bindings: [
              {name: 'costStatus', view: relaxation.state.status, type: 'u32', access: 'read'},
              {name: 'bandCounts', view: output.bandCounts, type: 'u32', access: 'read_write'}
            ],
            invocationCount: output.bandCounts.length,
            body: `if (costStatus[costStatusOffset + 3u] == 0u) {
    bandCounts[bandCountsOffset + index] = ${GPU_COST_DISTANCE_UNRESOLVED}u;
  }`
          })
        );
      }
    }
    return nodes;
  }

  /**
   * Returns the nodes that mark every passable cost NaN when the cost relaxation stopped at
   * `maxIterations` (library rule 9): a one-thread node arms an indirect dispatch only in that
   * case, so a converged run pays one tiny dispatch, not a pass over the grid.
   */
  private getUnresolvedCostNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    auxiliary: GraphDataView<'float32'>,
    costState: RasterIterationState<Parameters>
  ): GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {output} = props;
    const cellCount = props.grid.width * props.grid.height;
    const gate = createRasterIterationState(graph, `${id}-unresolved`, OPERATION, cellCount, id);
    const {x, y, z} = gate.layout;
    const {condition, extraResources} = getRasterIterationCondition<Parameters>(
      gate,
      `${id}-unresolved-costs`
    );
    return [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-unresolved-arm`,
        operation: OPERATION,
        variant: 'unresolved-arm',
        bindings: [
          {name: 'costStatus', view: costState.status, type: 'u32', access: 'read'},
          {name: 'dispatch', view: gate.dispatch, type: 'u32', access: 'read_write'}
        ],
        invocationCount: 1,
        body: `let converged = costStatus[costStatusOffset + 3u] != 0u;
  dispatch[dispatchOffset] = select(${x}u, 0u, converged);
  dispatch[dispatchOffset + 1u] = ${y}u;
  dispatch[dispatchOffset + 2u] = ${z}u;`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-unresolved-costs`,
        operation: OPERATION,
        variant: 'unresolved-costs',
        bindings: [
          {name: 'auxiliary', view: auxiliary, type: 'f32', access: 'read'},
          {name: 'costs', view: output.costs, type: 'u32', access: 'read_write'}
        ],
        invocationCount: gate.invocationCount,
        workgroupSize: 256,
        // Passable cells have a non-NaN auxiliary friction; the quiet NaN is written as bits.
        body: `if ((bitcast<u32>(auxiliary[auxiliaryOffset + index]) & 0x7fffffffu) <= 0x7f800000u) {
    costs[costsOffset + index] = 0x7fc00000u;
  }`,
        condition,
        extraResources
      })
    ];
  }

  /**
   * Appends the tie-level phase, the combined finalize node, and returns the tie views read by the
   * back-link kernel.
   *
   * Phases: `tie-prepare` writes each cell's tight equal-cost neighbor mask (as a float so it can
   * ride the relaxation's auxiliary raster) and level 0 where a strictly cheaper tight neighbor
   * exists; `tie-roots` zeroes the level of seeds whose cost survived; a tiled min-relaxation with
   * the cost phase's sweep schedule then propagates `level + 1` across masked edges. Finally `finalize` reports `converged` only when
   * both the cost and the tie relaxation reached a fixpoint.
   */
  private getTieNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    context: {
      nodes: GPUCommandNode<Parameters>[];
      auxiliary: GraphDataView<'float32'>;
      cellSizeMode: GPURasterCellSizeMode;
      hasSeedNode: boolean;
      sweepAfterIteration: number;
      costState: RasterIterationState<Parameters>;
    }
  ): {levels: GraphDataView<'float32'>; masks: GraphDataView<'float32'>} {
    const {id, props, maximumTieIterations} = this;
    const {output} = props;
    const {width, height} = props.grid;
    const {nodes, auxiliary, cellSizeMode} = context;
    const cellCount = width * height;
    const levels = createContributorTransientView(
      graph,
      OPERATION,
      id,
      `${id}-tie-levels`,
      'float32',
      cellCount
    );
    const masks = createContributorTransientView(
      graph,
      OPERATION,
      id,
      `${id}-tie-masks`,
      'float32',
      cellCount
    );
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-tie-prepare`,
        operation: OPERATION,
        variant: 'tie-prepare',
        bindings: [
          {name: 'costs', view: output.costs, type: 'f32', access: 'read'},
          {name: 'auxiliary', view: auxiliary, type: 'f32', access: 'read'},
          {
            name: 'settings',
            view: props.parameters,
            type: 'f32',
            access: 'read'
          },
          {name: 'levels', view: levels, type: 'u32', access: 'read_write'},
          {name: 'masks', view: masks, type: 'f32', access: 'read_write'}
        ],
        invocationCount: cellCount,
        declarations: `${getRasterGridWGSL({width, height, cellSizeMode})}
${EDGE_COST_WGSL}
${STRICT_PREDECESSOR_WGSL}`,
        body: `let centerAuxiliary = auxiliary[auxiliaryOffset + index];
  let centerCost = costs[costsOffset + index];
  var level = 0x7f800000u;
  var mask = 0u;
  var maskValue = getQuietNaN();
  if (!isNaNValue(centerAuxiliary) && isFiniteValue(centerCost)) {
    if (getStrictDirection(index, centerAuxiliary, centerCost) != GRID_NONE) { level = 0u; }
    let centerRow = index / GRID_WIDTH;
    for (var direction = 0u; direction < 8u; direction++) {
      let neighbor = getD8Neighbor(index, direction);
      if (neighbor == GRID_NONE) { continue; }
      let neighborAuxiliary = auxiliary[auxiliaryOffset + neighbor];
      let neighborCost = costs[costsOffset + neighbor];
      if (isNaNValue(neighborAuxiliary) || !(neighborCost == centerCost)) { continue; }
      // A tight tie edge adds nothing to the cost, so the sum stays exactly the neighbor cost.
      if (neighborCost + getEdgeCost(direction, centerRow, neighborAuxiliary, centerAuxiliary) <= centerCost) {
        mask |= 1u << direction;
      }
    }
    maskValue = f32(mask);
  }
  levels[levelsOffset + index] = level;
  masks[masksOffset + index] = maskValue;`
      })
    );
    if (context.hasSeedNode) {
      nodes.push(this.getSeedNode(graph, auxiliary, levels, cellSizeMode, 'roots'));
    }
    const tieProps = {
      id: `${id}-tie`,
      operation: OPERATION,
      width,
      height,
      cellSizeMode,
      maxIterations: maximumTieIterations
    };
    const {relaxation: tieRelaxation, resetNodes} = createRasterTiledRelaxation<Parameters>(
      graph,
      tieProps,
      true
    );
    nodes.push(
      ...resetNodes,
      ...createRasterTiledRelaxationNodes<Parameters>(graph, {
        ...tieProps,
        relaxation: tieRelaxation,
        sweepAfterIteration: context.sweepAfterIteration,
        values: levels,
        auxiliary: masks,
        settings: props.parameters,
        additiveEdgeCosts: true,
        declarations: /* wgsl */ `
fn getRelaxationEdgeCost(neighborAuxiliary: f32, centerAuxiliary: f32, centerRow: u32, direction: u32) -> f32 {
  if (((u32(centerAuxiliary) >> direction) & 1u) == 0u) { return getInfinity(); }
  return 1.0;
}
fn getRelaxationLimit() -> f32 { return getInfinity(); }`
      })
    );
    const bindings: WGSLKernelBinding[] = [
      {
        name: 'costStatus',
        view: context.costState.status,
        type: 'u32',
        access: 'read'
      },
      {
        name: 'tieStatus',
        view: tieRelaxation.state.status,
        type: 'u32',
        access: 'read'
      }
    ];
    if (output.status?.converged) {
      bindings.push({
        name: 'converged',
        view: output.status?.converged,
        type: 'u32',
        access: 'read_write'
      });
    }
    if (output.status?.iterationCount) {
      bindings.push({
        name: 'iterationCount',
        view: output.status?.iterationCount,
        type: 'u32',
        access: 'read_write'
      });
    }
    if (output.status?.converged || output.status?.iterationCount) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-finalize`,
          operation: OPERATION,
          variant: 'tie-finalize',
          bindings,
          invocationCount: 1,
          body: `${output.status?.converged ? 'converged[convergedOffset] = costStatus[costStatusOffset + 3u] & tieStatus[tieStatusOffset + 3u];' : ''}
  ${output.status?.iterationCount ? 'iterationCount[iterationCountOffset] = costStatus[costStatusOffset + 2u];' : ''}`
        })
      );
    }
    return {levels, masks};
  }

  /**
   * Builds the seed kernel. In `'costs'` mode it writes initial source costs and marks their tiles;
   * in `'roots'` mode (`target` is the tie-level raster) it zeroes the tie level of every accepted
   * seed whose cost survived relaxation, which makes it a root of the back-link forest.
   */
  private getSeedNode<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    auxiliary: GraphDataView<'float32'>,
    target: GraphDataView<'uint32'> | GraphDataView<'float32'>,
    cellSizeMode: GPURasterCellSizeMode,
    mode: 'costs' | 'roots' = 'costs',
    options: {
      /** Cost values the seed kernel writes; defaults to `output.costs`. */
      costs?: GraphDataView<'float32'>;
      /** Restart gate: seed only when a continued execution starts from scratch. */
      gate?: {
        condition: GPUCommandGraphNodeCondition<Parameters>;
        extraResources: GraphBufferUse[];
      };
    } = {}
  ): GPUCommandNode<Parameters> {
    const {id, props} = this;
    const {output} = props;
    const {width, height} = props.grid;
    const sourceCapacity = props.sources?.length ?? 0;
    const cellCount = width * height;
    const bindings: WGSLKernelBinding[] = [
      {name: 'auxiliary', view: auxiliary, type: 'f32', access: 'read'},
      {name: 'settings', view: props.parameters, type: 'f32', access: 'read'}
    ];
    if (props.sources) {
      bindings.push({
        name: 'sources',
        view: props.sources,
        type: 'u32',
        access: 'read'
      });
    }
    if (props.sourceCosts) {
      bindings.push({
        name: 'sourceCosts',
        view: props.sourceCosts,
        type: 'f32',
        access: 'read'
      });
    }
    if (props.sourceCount) {
      bindings.push({
        name: 'sourceCount',
        view: props.sourceCount,
        type: 'u32',
        access: 'read'
      });
    }
    if (props.sourceMask) {
      bindings.push({
        name: 'sourceMask',
        view: props.sourceMask,
        type: 'u32',
        access: 'read'
      });
    }
    bindings.push(
      {
        name: 'costBits',
        view: options.costs ?? output.costs,
        type: 'atomic<u32>',
        access: 'read_write'
      },
      mode === 'costs'
        ? {
            name: 'tileStamps',
            view: target as GraphDataView<'uint32'>,
            type: 'atomic<u32>',
            access: 'read_write'
          }
        : {
            name: 'levelBits',
            view: target as GraphDataView<'float32'>,
            type: 'atomic<u32>',
            access: 'read_write'
          }
    );
    return createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-${mode === 'costs' ? 'seed' : 'tie-roots'}`,
      operation: OPERATION,
      variant: mode === 'costs' ? 'seed' : 'tie-roots',
      bindings,
      invocationCount: Math.max(sourceCapacity, props.sourceMask ? cellCount : 0),
      declarations: `${getRasterGridWGSL({width, height, cellSizeMode})}
${mode === 'costs' ? getRasterRelaxationTileWGSL(width) : ''}
const CELL_COUNT: u32 = ${cellCount}u;
const SOURCE_CAPACITY: u32 = ${sourceCapacity}u;

fn seedCell(cell: u32, cost: f32) {
  if (cell >= CELL_COUNT) { return; }
  if (!(cost >= 0.0) || cost > settings[settingsOffset + 4u]) { return; }
  let friction = auxiliary[auxiliaryOffset + cell];
  // Compare bits: NaN comparisons may be folded away by shader compilers.
  if ((bitcast<u32>(friction) & 0x7fffffffu) > 0x7f800000u) { return; }
  // Fold -0.0 so the u32 bit order of non-negative floats matches numeric order.
  let bits = select(bitcast<u32>(cost), 0u, cost == 0.0);
  if (bits >= 0x7f800000u) { return; }
${
  mode === 'costs'
    ? `atomicMin(&costBits[costBitsOffset + cell], bits);
  atomicMax(&tileStamps[tileStampsOffset + getRelaxationTile(cell)], 1u);`
    : `if (atomicLoad(&costBits[costBitsOffset + cell]) == bits) {
    atomicStore(&levelBits[levelBitsOffset + cell], 0u);
  }`
}
}`,
      body: `${
        props.sources
          ? `if (index < SOURCE_CAPACITY${props.sourceCount ? ' && index < sourceCount[sourceCountOffset]' : ''}) {
    seedCell(sources[sourcesOffset + index], ${props.sourceCosts ? 'sourceCosts[sourceCostsOffset + index]' : '0.0'});
  }`
          : ''
      }
  ${props.sourceMask ? 'if (index < CELL_COUNT && sourceMask[sourceMaskOffset + index] != 0u) { seedCell(index, 0.0); }' : ''}`,
      condition: options.gate?.condition,
      extraResources: options.gate?.extraResources
    });
  }
}

/** Shared edge cost: ground distance of the move times the mean friction of both cells. */
const EDGE_COST_WGSL = /* wgsl */ `
fn getEdgeCost(direction: u32, centerRow: u32, neighborFriction: f32, centerFriction: f32) -> f32 {
  return getD8Distance(direction, centerRow) * (0.5 * (neighborFriction + centerFriction));
}`;

/**
 * Returns the direction of the strictly cheaper tight neighbor with the smallest candidate cost (the
 * smallest direction on exact ties), or GRID_NONE when no strictly cheaper neighbor is tight. Shared
 * by the tie preparation and back-link kernels so that both agree on which cells link strictly.
 *
 * Tight means the recomputed move cost is within 4 ulp of the cell cost. The relaxation stored
 * `fl(neighborCost + fl(edge))`; a compiler that contracts the recomputation into
 * `fma(distance, meanFriction, neighborCost)` differs from it by at most the product rounding
 * (0.5 ulp of the edge, which is at most the cost) plus both sum roundings, 1.5 ulp in all. A wider
 * relative slack (formerly 1e-5) let a cell reached across a zero-cost edge link to a strictly
 * cheaper neighbor whose move costs more, off the least-cost path.
 */
const STRICT_PREDECESSOR_WGSL = /* wgsl */ `
fn getStrictDirection(index: u32, centerAuxiliary: f32, centerCost: f32) -> u32 {
  var bestCandidate = getInfinity();
  var bestDirection = GRID_NONE;
  let centerRow = index / GRID_WIDTH;
  for (var direction = 0u; direction < 8u; direction++) {
    let neighbor = getD8Neighbor(index, direction);
    if (neighbor == GRID_NONE) { continue; }
    let neighborAuxiliary = auxiliary[auxiliaryOffset + neighbor];
    let neighborCost = costs[costsOffset + neighbor];
    if (isNaNValue(neighborAuxiliary) || !(neighborCost < centerCost)) { continue; }
    let candidate = neighborCost + getEdgeCost(direction, centerRow, neighborAuxiliary, centerAuxiliary);
    if (candidate < bestCandidate) {
      bestCandidate = candidate;
      bestDirection = direction;
    }
  }
  // 4 ulp of f32 (2^-21) relative: see the derivation in the TSDoc of STRICT_PREDECESSOR_WGSL.
  if (bestDirection != GRID_NONE && bestCandidate <= centerCost * (1.0 + 4.76837158e-7)) { return bestDirection; }
  return GRID_NONE;
}`;
