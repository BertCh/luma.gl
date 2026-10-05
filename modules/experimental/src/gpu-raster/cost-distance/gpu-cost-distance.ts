// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUHistogram,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPURasterBand} from '../index';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {validateGraphViewsBelongToGraph} from '../../utils/gpu-contributor-utils';
import type {GPUTerrainCellSizeMode} from '../../gpu-terrain/terrain-analysis/index';
import {
  getTerrainBandViews,
  getTerrainElevationNodes,
  validateTerrainBandBelongsToGraph,
  validateTerrainBuffersDistinct,
  validateTerrainGrid,
  validateTerrainSettings
} from '../../gpu-terrain/terrain-analysis/terrain-analysis-utils';
import {
  createRecipeTransientView,
  getRasterGridWGSL,
  validateRasterCellSizeMode,
  writeRasterGridSettings,
  type RasterGridSettings
} from './raster-grid-utils';
import {
  createRasterIterationFinalizeNode,
  type RasterIterationState,
  createRasterTiledRelaxation,
  createRasterTiledRelaxationNodes,
  getRasterRelaxationTileWGSL,
  validateRasterIterations
} from './raster-relaxation';

const OPERATION = 'GPUCostDistance';
const DEFAULT_MAXIMUM_ITERATIONS = 64;
const DEFAULT_MAXIMUM_TIE_ITERATIONS = 8;

/** Number of float32 values read from {@link GPUCostDistanceProps.settings}. */
export const GPU_COST_DISTANCE_PARAMETER_LENGTH = 8;

/** Sentinel written for "no back-link" (unreached or impassable) and "no band". */
export const GPU_COST_DISTANCE_NONE = 0xffffffff;

/** CPU description of the settings packed by {@link getGPUCostDistanceParameterValues}. */
export type GPUCostDistanceSettings = RasterGridSettings & {
  /** Candidate costs above this value stay unreached. Defaults to `Infinity`. */
  costLimit?: number;
};

/**
 * Packs settings into the 8-float layout read by {@link GPUCostDistance}:
 * `[cellSizeX, cellSizeY, northEdge, southEdge, costLimit, 0, 0, 0]`.
 */
export function getGPUCostDistanceParameterValues(
  settings: GPUCostDistanceSettings,
  target: Float32Array = new Float32Array(GPU_COST_DISTANCE_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_COST_DISTANCE_PARAMETER_LENGTH) {
    throw new Error('Cost distance settings target must hold 8 values');
  }
  target.fill(0, 0, GPU_COST_DISTANCE_PARAMETER_LENGTH);
  writeRasterGridSettings(target, settings);
  target[4] = settings.costLimit ?? Infinity;
  return target;
}

/**
 * Properties for {@link GPUCostDistance}.
 *
 * Compile-time: `width`, `height`, `cellSizeMode`, `maxIterations`, the capacity of `sources`, the
 * friction band's format and calibration, and which optional views exist. Per-frame: the contents
 * of `friction`, `settings` (cell size, latitude edges, `costLimit`), `sources`, `sourceCosts`,
 * `sourceCount`, `sourceMask`, and `bandThresholds`.
 */
export type GPUCostDistanceProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'cost-distance'`. */
  id?: string;
  /** Grid width in cells. Compile-time. */
  width: number;
  /** Grid height in cells. Compile-time. */
  height: number;
  /**
   * Cost per ground meter. Compile-time format and calibration; contents are per-frame. Validity,
   * nodata, scale, and offset are honored. Invalid, NaN, infinite, or negative samples are
   * impassable barriers; zero friction is allowed.
   */
  friction: GPURasterBand;
  /** Per-frame settings with at least 8 float32 values, see {@link getGPUCostDistanceParameterValues}. */
  settings: GraphDataView<'float32'>;
  /** Cell size interpretation. Compile-time. Defaults to `'uniform'`. */
  cellSizeMode?: GPUTerrainCellSizeMode;
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
  /** Number of unrolled gated relaxation iterations, 1 to `GPU_RASTER_MAXIMUM_ITERATIONS`. Compile-time. Defaults to 64. */
  maxIterations?: number;
  /**
   * Number of unrolled gated tie-level iterations, 1 to `GPU_RASTER_MAXIMUM_ITERATIONS`. Compile-time.
   * Defaults to 8. Used only when `backLinks` is requested: each iteration propagates the tie level
   * through a whole 16x16 tile, so a zero-cost plateau needs about its tile-crossing length in
   * iterations. A plateau deeper than the budget leaves `converged` at 0.
   */
  maxTieIterations?: number;
  /** Output cost per cell, `+Infinity` when unreached or impassable. Its length must equal `width * height`. */
  costs: GraphDataView<'float32'>;
  /**
   * Optional output D8 code (ESRI power of two) of the move from a cell toward the next cell on a
   * least-cost path back to a source. 0 at cells reached as a source,
   * `GPU_COST_DISTANCE_NONE` when unreached or impassable.
   *
   * Cycle-safe across zero-cost plateaus: a cell links to its smallest-direction strictly cheaper
   * tight neighbor when one exists; otherwise (it was reached only across zero-cost edges) to the
   * smallest-direction equal-cost zero-cost neighbor exactly one tie level closer to a strict entry
   * or a source. Cost never increases along links and the level strictly decreases at equal cost, so
   * following links always reaches a source. The tie phase runs only when this view is requested.
   */
  backLinks?: GraphDataView<'uint32'>;
  /** Isochrone thresholds, ascending. Required when `bands` or `bandCounts` is given. Per-frame contents, compile-time length. */
  bandThresholds?: GraphDataView<'float32'>;
  /** Optional output band per cell: cost in `(t[i - 1], t[i]]`, or `GPU_COST_DISTANCE_NONE` beyond the last threshold or unreached. */
  bands?: GraphDataView<'uint32'>;
  /** Optional output cell count per band (composes `GPUHistogram`). */
  bandCounts?: GraphDataView<'uint32'>;
  /**
   * Optional one-row output: 1 when relaxation reached a fixpoint, 0 when `maxIterations` stopped
   * it. With `backLinks`, also 0 when `maxTieIterations` stopped the tie phase.
   */
  converged?: GraphDataView<'uint32'>;
  /** Optional one-row output: number of relaxation iterations executed. */
  iterationCount?: GraphDataView<'uint32'>;
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
 * from scratch. With zero-friction plateaus, plateau cells share their cost with the source, so
 * their back-link is 0.
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
  readonly maxIterations: number;
  /** Resolved compile-time tie-phase iteration count. */
  readonly maxTieIterations: number;

  constructor(props: GPUCostDistanceProps) {
    this.id = props.id ?? 'cost-distance';
    this.props = props;
    this.maxIterations = props.maxIterations ?? DEFAULT_MAXIMUM_ITERATIONS;
    const {id} = this;
    const cellCount = validateTerrainGrid(id, props.width, props.height);
    validateRasterCellSizeMode(id, props.cellSizeMode ?? 'uniform');
    validateRasterIterations(id, 'maxIterations', this.maxIterations);
    this.maxTieIterations = props.maxTieIterations ?? DEFAULT_MAXIMUM_TIE_ITERATIONS;
    validateRasterIterations(id, 'maxTieIterations', this.maxTieIterations);
    validateTerrainSettings(id, props.settings, GPU_COST_DISTANCE_PARAMETER_LENGTH);
    for (const [name, view] of [
      ['sources', props.sources],
      ['sourceCount', props.sourceCount],
      ['sourceMask', props.sourceMask],
      ['backLinks', props.backLinks],
      ['bands', props.bands],
      ['bandCounts', props.bandCounts],
      ['converged', props.converged],
      ['iterationCount', props.iterationCount]
    ] as const) {
      if (view) {
        validatePackedUint32View(view, `${id} ${name}`);
      }
    }
    for (const [name, view] of [
      ['sourceCosts', props.sourceCosts],
      ['costs', props.costs],
      ['bandThresholds', props.bandThresholds]
    ] as const) {
      if (view) {
        validatePackedView(view, ['float32'], `${id} ${name}`);
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
      ['costs', props.costs],
      ['backLinks', props.backLinks],
      ['bands', props.bands]
    ] as const) {
      if (view && view.length !== cellCount) {
        throw new Error(`${id} ${name} must contain one value per cell`);
      }
    }
    for (const [name, view] of [
      ['sourceCount', props.sourceCount],
      ['converged', props.converged],
      ['iterationCount', props.iterationCount]
    ] as const) {
      if (view && view.length !== 1) {
        throw new Error(`${id} ${name} must contain exactly one row`);
      }
    }
    const needsThresholds = Boolean(props.bands || props.bandCounts);
    if (needsThresholds !== Boolean(props.bandThresholds)) {
      throw new Error(`${id} bandThresholds is required exactly when bands or bandCounts is given`);
    }
    if (props.bandThresholds && props.bandThresholds.length < 1) {
      throw new Error(`${id} bandThresholds must contain at least one threshold`);
    }
    if (props.bandCounts && props.bandCounts.length !== props.bandThresholds?.length) {
      throw new Error(`${id} bandCounts length must equal bandThresholds length`);
    }
    validateTerrainBuffersDistinct(
      id,
      [
        props.costs,
        props.backLinks,
        props.bands,
        props.bandCounts,
        props.converged,
        props.iterationCount
      ],
      [
        ...getTerrainBandViews(props.friction),
        props.settings,
        props.sources,
        props.sourceCosts,
        props.sourceCount,
        props.sourceMask,
        props.bandThresholds
      ]
    );
  }

  /** Returns friction preparation, gated tiled relaxation, and optional output nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, maxIterations} = this;
    const {width, height} = props;
    const cellSizeMode = props.cellSizeMode ?? 'uniform';
    validateTerrainBandBelongsToGraph(id, graph, props.friction, []);
    validateGraphViewsBelongToGraph(id, graph, [
      props.settings,
      props.sources,
      props.sourceCosts,
      props.sourceCount,
      props.sourceMask,
      props.costs,
      props.backLinks,
      props.bandThresholds,
      props.bands,
      props.bandCounts,
      props.converged,
      props.iterationCount
    ]);
    const cellCount = width * height;
    const friction = getTerrainElevationNodes(
      graph,
      `${id}-friction`,
      props.friction,
      width,
      height,
      true
    );
    const auxiliary = createRecipeTransientView(
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
          }
        ],
        invocationCount: cellCount,
        declarations: /* wgsl */ `
fn isFiniteValue(value: f32) -> bool { return (bitcast<u32>(value) & 0x7fffffffu) < 0x7f800000u; }
fn getQuietNaN() -> f32 { var bits = 0x7fc00000u; return bitcast<f32>(bits); }`,
        body: `let value = frictionValues[frictionValuesOffset + index];
  let passable = frictionValidity[frictionValidityOffset + index] != 0u && isFiniteValue(value) && value >= 0.0;
  auxiliary[auxiliaryOffset + index] = select(getQuietNaN(), value, passable);`
      })
    ];
    const relaxationProps = {
      id,
      operation: OPERATION,
      width,
      height,
      cellSizeMode,
      maxIterations
    };
    const {relaxation, resetNodes} = createRasterTiledRelaxation<Parameters>(
      graph,
      relaxationProps,
      false
    );
    nodes.push(
      ...resetNodes,
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-initialize`,
        operation: OPERATION,
        variant: 'initialize',
        bindings: [
          {
            name: 'costs',
            view: props.costs,
            type: 'u32',
            access: 'read_write'
          }
        ],
        invocationCount: cellCount,
        body: 'costs[costsOffset + index] = 0x7f800000u;'
      })
    );
    let hasSeedNode = false;
    if ((props.sources?.length ?? 0) > 0 || props.sourceMask) {
      nodes.push(this.getSeedNode(graph, auxiliary, relaxation.tileStamps, cellSizeMode));
      hasSeedNode = true;
    }
    nodes.push(
      ...createRasterTiledRelaxationNodes<Parameters>(graph, {
        ...relaxationProps,
        relaxation,
        values: props.costs,
        auxiliary,
        settings: props.settings,
        declarations: /* wgsl */ `
${EDGE_COST_WGSL}
fn getRelaxationCandidate(neighborValue: f32, neighborAuxiliary: f32, centerAuxiliary: f32, centerRow: u32, direction: u32) -> f32 {
  let candidate = neighborValue + getEdgeCost(direction, centerRow, neighborAuxiliary, centerAuxiliary);
  if (candidate > settings[settingsOffset + 4u]) { return getInfinity(); }
  return candidate;
}`
      })
    );
    let tie: {levels: GraphDataView<'float32'>; masks: GraphDataView<'float32'>} | undefined;
    if (props.backLinks) {
      tie = this.getTieNodes(graph, {
        nodes,
        auxiliary,
        cellSizeMode,
        hasSeedNode,
        costState: relaxation.state
      });
    } else if (props.converged || props.iterationCount) {
      nodes.push(
        createRasterIterationFinalizeNode<Parameters>(graph, {
          id: `${id}-finalize`,
          operation: OPERATION,
          state: relaxation.state,
          converged: props.converged,
          iterationCount: props.iterationCount
        })
      );
    }
    if (props.backLinks && tie) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-back-links`,
          operation: OPERATION,
          variant: 'back-links',
          bindings: [
            {name: 'costs', view: props.costs, type: 'f32', access: 'read'},
            {name: 'auxiliary', view: auxiliary, type: 'f32', access: 'read'},
            {
              name: 'settings',
              view: props.settings,
              type: 'f32',
              access: 'read'
            },
            {name: 'levels', view: tie.levels, type: 'f32', access: 'read'},
            {name: 'masks', view: tie.masks, type: 'f32', access: 'read'},
            {
              name: 'backLinks',
              view: props.backLinks,
              type: 'u32',
              access: 'read_write'
            }
          ],
          invocationCount: cellCount,
          declarations: `${getRasterGridWGSL({width, height, cellSizeMode})}
${EDGE_COST_WGSL}
${STRICT_PREDECESSOR_WGSL}`,
          body: `let centerAuxiliary = auxiliary[auxiliaryOffset + index];
  let centerCost = costs[costsOffset + index];
  var link = GRID_NONE;
  if (!isNaNValue(centerAuxiliary) && isFiniteValue(centerCost)) {
    link = 0u;
    let strictDirection = getStrictDirection(index, centerAuxiliary, centerCost);
    if (strictDirection != GRID_NONE) {
      link = getD8Code(strictDirection);
    } else {
      // Reached only across zero-cost edges (or a root source): link to the smallest direction
      // one tie level lower. Level 0 without a strict link is a root, which keeps link 0.
      let level = levels[levelsOffset + index];
      if (isFiniteValue(level) && level > 0.5) {
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
  backLinks[backLinksOffset + index] = link;`
        })
      );
    }
    if (props.bandThresholds && (props.bands || props.bandCounts)) {
      const bands =
        props.bands ??
        createRecipeTransientView(graph, OPERATION, id, `${id}-bands-scratch`, 'uint32', cellCount);
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-bands`,
          operation: OPERATION,
          variant: 'bands',
          bindings: [
            {name: 'costs', view: props.costs, type: 'f32', access: 'read'},
            {
              name: 'thresholds',
              view: props.bandThresholds,
              type: 'f32',
              access: 'read'
            },
            {name: 'bands', view: bands, type: 'u32', access: 'read_write'}
          ],
          invocationCount: cellCount,
          declarations: `const THRESHOLD_COUNT: u32 = ${props.bandThresholds.length}u;`,
          body: `let cost = costs[costsOffset + index];
  var band = ${GPU_COST_DISTANCE_NONE}u;
  if (bitcast<u32>(cost) < 0x7f800000u) {
    var passedCount = 0u;
    for (var thresholdIndex = 0u; thresholdIndex < THRESHOLD_COUNT; thresholdIndex++) {
      if (thresholds[thresholdsOffset + thresholdIndex] < cost) { passedCount++; }
    }
    if (passedCount < THRESHOLD_COUNT) { band = passedCount; }
  }
  bands[bandsOffset + index] = band;`
        })
      );
      if (props.bandCounts) {
        nodes.push(
          ...new GPUHistogram({
            id: `${id}-band-counts`,
            input: bands,
            output: props.bandCounts,
            domain: [0, props.bandThresholds.length]
          }).getCommandNodes(graph)
        );
      }
    }
    return nodes;
  }

  /**
   * Appends the tie-level phase, the combined finalize node, and returns the tie views read by the
   * back-link kernel.
   *
   * Phases: `tie-prepare` writes each cell's tight equal-cost neighbor mask (as a float so it can
   * ride the relaxation's auxiliary raster) and level 0 where a strictly cheaper tight neighbor
   * exists; `tie-roots` zeroes the level of seeds whose cost survived; a tiled min-relaxation then
   * propagates `level + 1` across masked edges. Finally `finalize` reports `converged` only when
   * both the cost and the tie relaxation reached a fixpoint.
   */
  private getTieNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    context: {
      nodes: GPUCommandNode<Parameters>[];
      auxiliary: GraphDataView<'float32'>;
      cellSizeMode: GPUTerrainCellSizeMode;
      hasSeedNode: boolean;
      costState: RasterIterationState;
    }
  ): {levels: GraphDataView<'float32'>; masks: GraphDataView<'float32'>} {
    const {id, props, maxTieIterations} = this;
    const {width, height} = props;
    const {nodes, auxiliary, cellSizeMode} = context;
    const cellCount = width * height;
    const levels = createRecipeTransientView(
      graph,
      OPERATION,
      id,
      `${id}-tie-levels`,
      'float32',
      cellCount
    );
    const masks = createRecipeTransientView(
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
          {name: 'costs', view: props.costs, type: 'f32', access: 'read'},
          {name: 'auxiliary', view: auxiliary, type: 'f32', access: 'read'},
          {
            name: 'settings',
            view: props.settings,
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
      maxIterations: maxTieIterations
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
        values: levels,
        auxiliary: masks,
        settings: props.settings,
        declarations: /* wgsl */ `
fn getRelaxationCandidate(neighborValue: f32, neighborAuxiliary: f32, centerAuxiliary: f32, centerRow: u32, direction: u32) -> f32 {
  if (((u32(centerAuxiliary) >> direction) & 1u) == 0u) { return getInfinity(); }
  return neighborValue + 1.0;
}`
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
    if (props.converged) {
      bindings.push({
        name: 'converged',
        view: props.converged,
        type: 'u32',
        access: 'read_write'
      });
    }
    if (props.iterationCount) {
      bindings.push({
        name: 'iterationCount',
        view: props.iterationCount,
        type: 'u32',
        access: 'read_write'
      });
    }
    if (props.converged || props.iterationCount) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-finalize`,
          operation: OPERATION,
          variant: 'tie-finalize',
          bindings,
          invocationCount: 1,
          body: `${props.converged ? 'converged[convergedOffset] = costStatus[costStatusOffset + 3u] & tieStatus[tieStatusOffset + 3u];' : ''}
  ${props.iterationCount ? 'iterationCount[iterationCountOffset] = costStatus[costStatusOffset + 2u];' : ''}`
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
    cellSizeMode: GPUTerrainCellSizeMode,
    mode: 'costs' | 'roots' = 'costs'
  ): GPUCommandNode<Parameters> {
    const {id, props} = this;
    const {width, height} = props;
    const sourceCapacity = props.sources?.length ?? 0;
    const cellCount = width * height;
    const bindings: WGSLKernelBinding[] = [
      {name: 'auxiliary', view: auxiliary, type: 'f32', access: 'read'},
      {name: 'settings', view: props.settings, type: 'f32', access: 'read'}
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
        view: props.costs,
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
  ${props.sourceMask ? 'if (index < CELL_COUNT && sourceMask[sourceMaskOffset + index] != 0u) { seedCell(index, 0.0); }' : ''}`
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
 * smallest direction on exact ties), or GRID_NONE when no strictly cheaper neighbor is tight within
 * a 1e-5 relative tolerance. Shared by the tie preparation and back-link kernels so that both agree
 * on which cells link strictly.
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
  if (bestDirection != GRID_NONE && bestCandidate <= centerCost * 1.00001) { return bestDirection; }
  return GRID_NONE;
}`;
