// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  doGraphDataViewsOverlap,
  getViewBindingRange,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPURasterBand} from '../../gpu-raster/index';
import {
  validateRasterCellSizeMode,
  writeRasterGridSettings,
  type RasterGridSettings
} from '../cost-distance/raster-grid-utils';
import {validateRasterIterations} from '../cost-distance/raster-relaxation';
import type {GPUMapGraphRecipe} from '../map-graph-types';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../map-graph-utils';
import {
  getTerrainBandViews,
  getTerrainElevationNodes,
  validateTerrainBandBelongsToGraph,
  validateTerrainGrid,
  validateTerrainSettings
} from '../terrain-analysis/terrain-analysis-utils';
import type {GPUTerrainCellSizeMode} from '../terrain-analysis/index';
import {
  createTerrainFlowAccumulationNodes,
  createTerrainFlowDirectionNode,
  createTerrainFlowFillNodes,
  createTerrainFlowStreamsNode
} from './terrain-flow-passes';
import {createTerrainFlowFlatNodes} from './terrain-flow-flats';
import {createTerrainFlowRoutingAccumulationNodes} from './terrain-flow-routing';

/** Number of float32 values read from `GPUTerrainFlowProps.settings`. */
export const GPU_TERRAIN_FLOW_PARAMETER_LENGTH = 8;

/** Receiver and flow-direction marker for cells without a receiver or without data. */
export const GPU_TERRAIN_FLOW_NONE = 0xffffffff;

/**
 * Per-cell classification written to `GPUTerrainFlowProps.cellClasses`.
 *
 * `draining` cells have a strictly lower neighbor. `flat`, `pit`, and `outlet` cells are terminal
 * (direction 0): a flat has an equal-height neighbor, a pit is a strict local minimum, and an
 * outlet is a boundary cell (grid border or next to an invalid cell) with no lower neighbor.
 * `invalid` cells have no elevation.
 */
export const GPU_TERRAIN_FLOW_CELL_CLASS = {
  draining: 0,
  flat: 1,
  pit: 2,
  outlet: 3,
  invalid: 4
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

/** CPU-side description packed by {@link getGPUTerrainFlowParameterValues}. */
export type GPUTerrainFlowSettings = RasterGridSettings & {
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
 * Packs `[cellSizeX, cellSizeY, northEdge, southEdge, fillEpsilon, streamThreshold, flowExponent, 0]`.
 *
 * @throws If `target` holds fewer than 8 values.
 */
export function getGPUTerrainFlowParameterValues(
  settings: GPUTerrainFlowSettings,
  target: Float32Array = new Float32Array(GPU_TERRAIN_FLOW_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_TERRAIN_FLOW_PARAMETER_LENGTH) {
    throw new Error('Terrain flow settings target must hold 8 values');
  }
  writeRasterGridSettings(target, settings);
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
  const outputs = [
    ['filledElevation', props.filledElevation],
    ['flowDirections', props.flowDirections],
    ['cellClasses', props.cellClasses],
    ['accumulation', props.accumulation],
    ['streams', props.streams],
    ['fillConverged', props.fillConverged],
    ['accumulationConverged', props.accumulationConverged],
    ['flatsConverged', props.flatsConverged]
  ] as const;
  validateGraphOutputsDisjointFromInputs(
    id,
    outputs.map(([, view]) => view),
    [...getTerrainBandViews(props.elevation), props.settings, props.runoff]
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

/**
 * Properties for {@link GPUTerrainFlow}.
 *
 * Compile-time: `width`, `height`, `cellSizeMode`, `fillDepressions`, `resolveFlats`,
 * `flowRouting`, iteration limits, `accumulationUnits`, the elevation format and calibration, and
 * which optional views are provided. Per-frame: the contents of `elevation`, `settings`, and `runoff`.
 *
 * Output aliasing: an output view never shares a buffer with an input (`elevation`, its validity,
 * `settings`, `runoff`). Outputs may share one buffer, for example `fillConverged` and
 * `accumulationConverged` as two 1-row views of one 8-byte summary buffer, only over disjoint byte
 * ranges. Outputs that a single node binds together (`filledElevation`, `flowDirections` and
 * `cellClasses`; `accumulation` and `streams`) must additionally start in different 256-byte
 * storage binding windows of a shared buffer, because WebGPU rejects overlapping writable bindings
 * of one buffer within a dispatch. The convergence flags are written by their own nodes, so they
 * may sit in the same window.
 */
export type GPUTerrainFlowProps = {
  /** Prefix for node and transient IDs. Defaults to `'terrain-flow'`. Compile-time. */
  id?: string;
  /** Grid width in cells. Compile-time. */
  width: number;
  /** Grid height in cells. Compile-time. */
  height: number;
  /**
   * Elevation band, buffer or texture. Validity, nodata, scale, and offset are honored; cells
   * without a finite calibrated elevation are invalid. Format is compile-time, contents per-frame.
   */
  elevation: GPURasterBand;
  /** Settings with at least 8 float32 values, see {@link getGPUTerrainFlowParameterValues}. Per-frame. */
  settings: GraphDataView<'float32'>;
  /** Cell size interpretation. Defaults to `'uniform'`. Compile-time. */
  cellSizeMode?: GPUTerrainCellSizeMode;
  /** Fill depressions before routing flow. Defaults to false. Compile-time. */
  fillDepressions?: boolean;
  /** Maximum gated fill iterations in `[1, 1024]`. Defaults to 128. Compile-time. */
  maxFillIterations?: number;
  /**
   * Route flow across flats (Barnes, Lehman and Mulla 2014): cells of class `flat` that are
   * connected through equal-height cells to a cell that drains get a D8 receiver on a synthetic
   * gradient toward lower terrain and away from higher terrain, without changing elevations.
   * Resolved cells become `draining`; flats with no outlet stay `flat`. Defaults to false.
   * Compile-time.
   */
  resolveFlats?: boolean;
  /** Maximum gated iterations of each flat-resolution relaxation in `[1, 1024]`. Defaults to 128. Compile-time. */
  maxFlatIterations?: number;
  /**
   * Flow routing used for `accumulation` and `streams`. `flowDirections` and `cellClasses` are
   * always D8. Defaults to `'d8'`. Compile-time.
   */
  flowRouting?: GPUTerrainFlowRouting;
  /** Maximum gated accumulation rounds in `[1, 1024]`. Defaults to 64. Compile-time. */
  maxAccumulationIterations?: number;
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
  /** Depression-filled elevation, NaN for invalid cells. Requires `fillDepressions`. Presence is compile-time. */
  filledElevation?: GraphDataView<'float32'>;
  /**
   * ESRI D8 code toward the receiver (1 east, 2 south-east, 4 south, 8 south-west, 16 west, 32
   * north-west, 64 north, 128 north-east); 0 for terminal cells; `GPU_TERRAIN_FLOW_NONE` for
   * invalid cells. Presence is compile-time.
   */
  flowDirections?: GraphDataView<'uint32'>;
  /** `GPU_TERRAIN_FLOW_CELL_CLASS` per cell. Presence is compile-time. */
  cellClasses?: GraphDataView<'uint32'>;
  /**
   * Total upstream weight including the cell itself. NaN for invalid cells and for cells left
   * unresolved when accumulation did not converge. Presence is compile-time.
   */
  accumulation?: GraphDataView<'float32'>;
  /**
   * 1 where accumulation is finite and at least `settings.streamThreshold`, else 0. Accumulation
   * is computed into a transient when `accumulation` is not provided. Presence is compile-time.
   */
  streams?: GraphDataView<'uint32'>;
  /** One row set to 1 when depression filling reached its fixpoint. Requires `fillDepressions`. */
  fillConverged?: GraphDataView<'uint32'>;
  /**
   * One row set to 1 when accumulation resolved every valid cell. The flag is conservative: it can
   * be 0 when the last allowed round happened to finish everything. Requires `accumulation` or
   * `streams`.
   */
  accumulationConverged?: GraphDataView<'uint32'>;
  /**
   * One row set to 1 when every flat-resolution relaxation reached its fixpoint. Requires
   * `resolveFlats`.
   */
  flatsConverged?: GraphDataView<'uint32'>;
};

/**
 * Terrain hydrology on one raster tile: optional depression filling, D8 flow directions, flow
 * accumulation, and a stream mask.
 *
 * Algorithm choices:
 * - Filling is Planchon-Darboux expressed as a monotone min-relaxation over the shared tiled
 *   raster relaxation: boundary cells start at their elevation, other valid cells at +infinity,
 *   and the fixpoint is `W(c) = max(Z(c), min over neighbors n of W(n) + epsilon)`. The fixpoint is
 *   unique and independent of update order, so the chaotic GPU schedule is deterministic.
 *   With `epsilon` 0 filled depressions become exact flats (class `flat`, direction 0, no
 *   routing); with `epsilon` above the float32 spacing every filled cell gets a strict descent
 *   toward a boundary.
 * - Flow directions are steepest D8 descent on ground distance (so diagonal moves and latitude
 *   dependent cell sizes are honored); only strictly positive drops route, ties pick the lowest
 *   direction index. Strict drops make the receiver graph a forest.
 * - Accumulation is a deterministic pull over the receiver forest. A cell's value is written once,
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
 * iteration limits leaves `fillConverged` or `accumulationConverged` at 0 and unresolved
 * accumulation cells at NaN.
 *
 * - Flat resolution (`resolveFlats`) follows Barnes, Lehman and Mulla (2014): hop distances toward
 *   the flat's draining edge and away from its higher edge are GPU relaxations over equal-height
 *   cells, combined as `2 * towardLower + (flatMaximum - awayFromHigher)`, and each resolved cell
 *   drains to the equal-height neighbor with the smallest strictly lower value.
 * - D-infinity and multiple-flow-direction accumulation (`flowRouting`) use the same
 *   deterministic pull as D8, with each donor's fraction recomputed from the surface, so values are
 *   written once from finalized donors in fixed direction order.
 *
 * Downstream products take this recipe's outputs: {@link GPUTerrainHeightAboveDrainage},
 * {@link GPUTerrainWatersheds}, {@link GPUTerrainStreamOrder} and
 * {@link GPUTerrainHydrologicIndices}.
 *
 * Non-goals: Priority-Flood filling and least-cost breaching (sequential priority-queue
 * algorithms) and cross-tile (seamless) watersheds.
 */
export class GPUTerrainFlow implements GPUMapGraphRecipe {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Stable recipe name. */
  readonly recipe = 'terrain-flow';
  /** Validated properties. */
  readonly props: GPUTerrainFlowProps;

  constructor(props: GPUTerrainFlowProps) {
    this.id = props.id ?? this.recipe;
    this.props = props;
    const {id} = this;
    const cellCount = validateTerrainGrid(id, props.width, props.height);
    validateTerrainSettings(id, props.settings, GPU_TERRAIN_FLOW_PARAMETER_LENGTH);
    validateRasterCellSizeMode(id, props.cellSizeMode ?? 'uniform');
    if (!['cells', 'area'].includes(props.accumulationUnits ?? 'cells')) {
      throw new Error(`${id} accumulationUnits must be cells or area`);
    }
    validateRasterIterations(id, 'maxFillIterations', props.maxFillIterations ?? 128);
    validateRasterIterations(id, 'maxFlatIterations', props.maxFlatIterations ?? 128);
    if (!['d8', 'd-infinity', 'mfd-freeman', 'mfd-quinn'].includes(props.flowRouting ?? 'd8')) {
      throw new Error(`${id} flowRouting must be d8, d-infinity, mfd-freeman, or mfd-quinn`);
    }
    if (props.flatsConverged && !props.resolveFlats) {
      throw new Error(`${id} flatsConverged requires resolveFlats`);
    }
    validateRasterIterations(
      id,
      'maxAccumulationIterations',
      props.maxAccumulationIterations ?? 64
    );
    if (
      !props.filledElevation &&
      !props.flowDirections &&
      !props.cellClasses &&
      !props.accumulation &&
      !props.streams
    ) {
      throw new Error(`${id} requires at least one output`);
    }
    if ((props.filledElevation || props.fillConverged) && !props.fillDepressions) {
      throw new Error(`${id} filledElevation and fillConverged require fillDepressions`);
    }
    if (props.accumulationConverged && !props.accumulation && !props.streams) {
      throw new Error(`${id} accumulationConverged requires accumulation or streams`);
    }
    for (const [name, view] of [
      ['filledElevation', props.filledElevation],
      ['accumulation', props.accumulation],
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
      ['flowDirections', props.flowDirections],
      ['cellClasses', props.cellClasses],
      ['streams', props.streams]
    ] as const) {
      if (view) {
        validatePackedUint32View(view, `${id} ${name}`);
        if (view.length !== cellCount) {
          throw new Error(`${id} ${name} must contain one value per cell`);
        }
      }
    }
    for (const [name, view] of [
      ['fillConverged', props.fillConverged],
      ['accumulationConverged', props.accumulationConverged],
      ['flatsConverged', props.flatsConverged]
    ] as const) {
      if (view) {
        validatePackedUint32View(view, `${id} ${name}`);
        if (view.length !== 1) {
          throw new Error(`${id} ${name} must contain one row`);
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
    const {width, height} = props;
    const cellSizeMode = props.cellSizeMode ?? 'uniform';
    validateTerrainBandBelongsToGraph(id, graph, props.elevation, []);
    validateGraphViewsBelongToGraph(id, graph, [
      props.settings,
      props.runoff,
      props.filledElevation,
      props.flowDirections,
      props.cellClasses,
      props.accumulation,
      props.streams,
      props.fillConverged,
      props.accumulationConverged,
      props.flatsConverged
    ]);
    const cellCount = width * height;
    const grid = {width, height, cellSizeMode};
    const source = getTerrainElevationNodes(graph, id, props.elevation, width, height, true);
    const nodes: GPUCommandNode<Parameters>[] = [...source.nodes];
    const elevation = source.band.storage.values as GraphDataView<'float32'>;

    let surface = elevation;
    if (props.fillDepressions) {
      const filled =
        props.filledElevation ?? createTransientView(graph, `${id}-filled`, 'float32', cellCount);
      nodes.push(
        ...createTerrainFlowFillNodes(graph, {
          ...grid,
          id,
          maxIterations: props.maxFillIterations ?? 128,
          elevation,
          filled,
          settings: props.settings,
          converged: props.fillConverged
        })
      );
      surface = filled;
    }

    const needsAccumulation = Boolean(props.accumulation || props.streams);
    if (!props.flowDirections && !props.cellClasses && !needsAccumulation) {
      return nodes;
    }
    const receivers = needsAccumulation
      ? createTransientView(graph, `${id}-receivers`, 'uint32', cellCount)
      : undefined;
    // Flat resolution reads the D8 classes, so they exist even when the caller did not ask.
    const classes =
      props.cellClasses ??
      (props.resolveFlats
        ? createTransientView(graph, `${id}-classes`, 'uint32', cellCount)
        : undefined);
    nodes.push(
      createTerrainFlowDirectionNode(graph, {
        ...grid,
        id: `${id}-flow-direction`,
        elevation,
        surface,
        settings: props.settings,
        directions: props.flowDirections,
        classes,
        receivers,
        cellClass: GPU_TERRAIN_FLOW_CELL_CLASS
      })
    );
    if (props.resolveFlats && classes) {
      nodes.push(
        ...createTerrainFlowFlatNodes(graph, {
          ...grid,
          id,
          maxIterations: props.maxFlatIterations ?? 128,
          elevation,
          surface,
          settings: props.settings,
          classes,
          directions: props.flowDirections,
          receivers,
          converged: props.flatsConverged,
          cellClass: GPU_TERRAIN_FLOW_CELL_CLASS
        })
      );
    }

    if (receivers) {
      const accumulation =
        props.accumulation ??
        createTransientView(graph, `${id}-accumulation`, 'float32', cellCount);
      const flowRouting = props.flowRouting ?? 'd8';
      const accumulationProps = {
        ...grid,
        id,
        maxIterations: props.maxAccumulationIterations ?? 64,
        elevation,
        receivers,
        accumulation,
        settings: props.settings,
        runoff: props.runoff,
        area: (props.accumulationUnits ?? 'cells') === 'area',
        converged: props.accumulationConverged
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
      if (props.streams) {
        nodes.push(
          createTerrainFlowStreamsNode(graph, {
            id: `${id}-streams`,
            cellCount,
            accumulation,
            streams: props.streams,
            settings: props.settings
          })
        );
      }
    }
    return nodes;
  }
}
