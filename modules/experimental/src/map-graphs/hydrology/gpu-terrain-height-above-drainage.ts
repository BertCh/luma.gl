// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPURasterBand} from '../../gpu-raster/index';
import {validateRasterIterations} from '../cost-distance/raster-relaxation';
import {createMapGraphKernelNode, type MapGraphKernelBinding} from '../map-graph-kernels';
import type {GPUMapGraphRecipe} from '../map-graph-types';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../map-graph-utils';
import {
  getTerrainBandViews,
  getTerrainElevationNodes,
  validateTerrainBandBelongsToGraph,
  validateTerrainBuffersDistinct,
  validateTerrainGrid
} from '../terrain-analysis/terrain-analysis-utils';
import {
  createTerrainFlowPointerNodes,
  getTerrainFlowTracingWGSL,
  validateTerrainTracingCellView,
  validateTerrainTracingConvergedView
} from './terrain-flow-tracing';

const OPERATION = 'GPUTerrainHeightAboveDrainage';

/** Value of `drainageCells` for cells with no stream downstream, and for invalid cells. */
export const GPU_TERRAIN_DRAINAGE_NONE = 0xffffffff;

/**
 * Properties for {@link GPUTerrainHeightAboveDrainage}.
 *
 * Compile-time: `width`, `height`, `maxIterations`, the elevation format and calibration, and which
 * optional outputs are provided. Per-frame: the contents of `elevation`, `flowDirections` and
 * `streams`. The recipe works on D8 receiver indices only, so it has no cell-size model: the result
 * is a vertical difference in elevation units, independent of `uniform`, `web-mercator` or
 * `geographic` cell spacing.
 *
 * Output aliasing: outputs never share a buffer with an input or with each other.
 */
export type GPUTerrainHeightAboveDrainageProps = {
  /** Prefix for node and transient IDs. Defaults to `'terrain-height-above-drainage'`. Compile-time. */
  id?: string;
  /** Grid width in cells. Compile-time. */
  width: number;
  /** Grid height in cells. Compile-time. */
  height: number;
  /**
   * Routing surface whose heights are differenced: normally `GPUTerrainFlow.filledElevation`, or
   * the raw DEM. Validity, nodata, scale and offset are honored; cells without a finite calibrated
   * elevation get NaN. Format is compile-time, contents per-frame.
   */
  elevation: GPURasterBand;
  /**
   * ESRI D8 codes, one per cell: 1 east, 2 south-east, 4 south, 8 south-west, 16 west, 32
   * north-west, 64 north, 128 north-east; 0 terminal; `0xffffffff` invalid. Any other code, and
   * any code pointing outside the grid, is treated as terminal. Typically
   * `GPUTerrainFlow.flowDirections`; grids from TauDEM or Whitebox work too. Per-frame.
   */
  flowDirections: GraphDataView<'uint32'>;
  /** Stream mask, one word per cell, non-zero for stream cells. Typically `GPUTerrainFlow.streams`. Per-frame. */
  streams: GraphDataView<'uint32'>;
  /**
   * Elevation of a cell minus the elevation of the stream cell its D8 path reaches first, in float32
   * elevation units. 0 on stream cells. NaN for invalid cells, cells whose path ends without
   * reaching a stream, and cells whose elevation or drainage elevation is not finite. Presence is compile-time.
   */
  heightAboveDrainage?: GraphDataView<'float32'>;
  /**
   * Cell index of the stream cell each cell drains to (the cell itself for stream cells), or
   * `GPU_TERRAIN_DRAINAGE_NONE`. Presence is compile-time.
   */
  drainageCells?: GraphDataView<'uint32'>;
  /** One row set to 1 when pointer jumping reached its fixpoint. Presence is compile-time. */
  converged?: GraphDataView<'uint32'>;
  /**
   * Maximum gated pointer-jumping rounds in `[1, 1024]`. Defaults to 32: paths of up to 2^32 cells
   * need at most 32 doubling rounds. Compile-time.
   */
  maxIterations?: number;
};

/**
 * Height Above the Nearest Drainage (HAND) on one raster tile, following D8 receivers.
 *
 * HAND is the elevation of a cell above the stream cell its flow path reaches (Renno et al. 2008;
 * Nobre et al. 2011). Here the drainage cell is the first stream cell on the cell's D8 path, found
 * by GPU pointer jumping: every cell points at its receiver (or itself when it is a stream cell or
 * has no receiver) and `pointer[c] = pointer[pointer[c]]` is repeated in place until nothing
 * changes. The fixpoint is the first stream cell downstream, or the terminal cell when there is
 * none; it is unique, so the result does not depend on thread scheduling. Heights are one float32
 * subtraction, so they are bit-exact against a sequential walk.
 *
 * The drainage path is the D8 path. TauDEM's D-infinity `DistDown` variant (vertical drop along the
 * D-infinity path) is not implemented. The output is a vertical difference in elevation units, not
 * a path length. Receiver cycles, possible only in caller-supplied directions, leave `converged` at 0.
 *
 * Compose it with {@link GPUTerrainFlow}: pass its `filledElevation` (or the raw DEM),
 * `flowDirections` and `streams`. It is a separate recipe so grids from other tools can be used.
 */
export class GPUTerrainHeightAboveDrainage implements GPUMapGraphRecipe {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Stable recipe name. */
  readonly recipe = 'terrain-height-above-drainage';
  /** Validated properties. */
  readonly props: GPUTerrainHeightAboveDrainageProps;

  constructor(props: GPUTerrainHeightAboveDrainageProps) {
    this.id = props.id ?? this.recipe;
    this.props = props;
    const {id} = this;
    const cellCount = validateTerrainGrid(id, props.width, props.height);
    validateRasterIterations(id, 'maxIterations', props.maxIterations ?? 32);
    if (!props.heightAboveDrainage && !props.drainageCells) {
      throw new Error(`${id} requires at least one output`);
    }
    validateTerrainTracingCellView(id, 'flowDirections', props.flowDirections, 'uint32', cellCount);
    validateTerrainTracingCellView(id, 'streams', props.streams, 'uint32', cellCount);
    if (props.heightAboveDrainage) {
      validateTerrainTracingCellView(
        id,
        'heightAboveDrainage',
        props.heightAboveDrainage,
        'float32',
        cellCount
      );
    }
    if (props.drainageCells) {
      validateTerrainTracingCellView(id, 'drainageCells', props.drainageCells, 'uint32', cellCount);
    }
    if (
      props.elevation.storage.kind === 'buffer' &&
      props.elevation.storage.values.length !== cellCount
    ) {
      throw new Error(`${id} elevation must contain one value per cell`);
    }
    validateTerrainTracingConvergedView(id, props.converged);
    const outputs = [props.heightAboveDrainage, props.drainageCells, props.converged];
    validateGraphOutputsDisjointFromInputs(id, outputs, [
      ...getTerrainBandViews(props.elevation),
      props.flowDirections,
      props.streams
    ]);
    validateTerrainBuffersDistinct(id, outputs, []);
  }

  /** Returns canonicalization, pointer-jumping and HAND nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {width, height} = props;
    validateTerrainBandBelongsToGraph(id, graph, props.elevation, []);
    validateGraphViewsBelongToGraph(id, graph, [
      props.flowDirections,
      props.streams,
      props.heightAboveDrainage,
      props.drainageCells,
      props.converged
    ]);
    const cellCount = width * height;
    const nodes: GPUCommandNode<Parameters>[] = [];
    let elevation: GraphDataView<'float32'> | undefined;
    if (props.heightAboveDrainage) {
      const source = getTerrainElevationNodes(graph, id, props.elevation, width, height, true);
      nodes.push(...source.nodes);
      elevation = source.band.storage.values as GraphDataView<'float32'>;
    }
    const pointer =
      props.drainageCells ?? createTransientView(graph, `${id}-pointer`, 'uint32', cellCount);
    nodes.push(
      ...createTerrainFlowPointerNodes(graph, {
        id,
        recipeId: id,
        operation: OPERATION,
        width,
        height,
        maxIterations: props.maxIterations ?? 32,
        flowDirections: props.flowDirections,
        pointer,
        stops: {kind: 'streams', streams: props.streams},
        converged: props.converged
      })
    );
    const bindings: MapGraphKernelBinding[] = [
      {name: 'flowDirections', view: props.flowDirections, type: 'u32', access: 'read'},
      {name: 'streams', view: props.streams, type: 'u32', access: 'read'},
      {name: 'pointer', view: pointer, type: 'u32', access: 'read_write'}
    ];
    if (elevation && props.heightAboveDrainage) {
      bindings.push(
        {name: 'elevation', view: elevation, type: 'f32', access: 'read'},
        {name: 'heightOut', view: props.heightAboveDrainage, type: 'f32', access: 'read_write'}
      );
    }
    // Each thread reads and rewrites only its own pointer word, so publishing drainage cells in
    // place over the pointer buffer is safe.
    nodes.push(
      createMapGraphKernelNode<Parameters>(graph, {
        id: `${id}-height`,
        operation: OPERATION,
        variant: 'height-above-drainage',
        bindings,
        invocationCount: cellCount,
        declarations: `${getTerrainFlowTracingWGSL(width, height)}
fn getQuietNaN() -> f32 { var bits = 0x7fc00000u; return bitcast<f32>(bits); }
fn isFiniteValue(value: f32) -> bool { return (bitcast<u32>(value) & 0x7fffffffu) < 0x7f800000u; }`,
        body: `let reached = pointer[pointerOffset + index];
  var drainage = NO_CELL;
  if (flowDirections[flowDirectionsOffset + index] != NO_CELL &&
      flowDirections[flowDirectionsOffset + reached] != NO_CELL &&
      streams[streamsOffset + reached] != 0u) {
    drainage = reached;
  }
  ${
    elevation
      ? `var result = getQuietNaN();
  if (drainage != NO_CELL) {
    let cellElevation = elevation[elevationOffset + index];
    let drainageElevation = elevation[elevationOffset + drainage];
    if (isFiniteValue(cellElevation) && isFiniteValue(drainageElevation)) {
      result = cellElevation - drainageElevation;
    }
  }
  heightOut[heightOutOffset + index] = result;`
      : ''
  }
  pointer[pointerOffset + index] = drainage;`
      })
    );
    return nodes;
  }
}
