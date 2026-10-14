// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPURasterBand, GPURasterMetadata} from '../../gpu-raster/index';
import {validateRasterIterations} from '../../gpu-raster/cost-distance/raster-relaxation';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {
  getRasterBandViews,
  getRasterBufferBandNodes,
  validateRasterBandBelongsToGraph,
  validateRasterBuffersDistinct,
  validateRasterGrid
} from '../../gpu-raster/raster-band-utils';
import {
  createTerrainFlowPointerNodes,
  getTerrainFlowTracingWGSL,
  validateTerrainTracingCellView,
  validateTerrainTracingStatus
} from './terrain-flow-tracing';
import {GET_QUIET_NAN_WGSL, IS_FINITE_VALUE_WGSL} from '../../utils/wgsl-common';
import {GPU_TERRAIN_FLOW_STREAM_UNRESOLVED} from './gpu-terrain-flow';

const OPERATION = 'GPUTerrainHeightAboveDrainage';

/** Value of `drainageCells` for cells with no stream downstream, and for invalid cells. */
export const GPU_TERRAIN_DRAINAGE_NONE = 0xffffffff;

/**
 * Value of `drainageCells` for cells whose D8 path meets a `GPU_TERRAIN_FLOW_STREAM_UNRESOLVED`
 * stream word or a `GPU_TERRAIN_FLOW_UNRESOLVED` direction before a stream cell, or whose pointer
 * jumping stopped at `maximumIterations` before reaching its fixpoint: the drainage cell is unknown, and
 * HAND is NaN. A converged run on resolved inputs never writes it.
 */
export const GPU_TERRAIN_DRAINAGE_UNRESOLVED = 0xfffffffe;

/** One-row status words of {@link GPUTerrainHeightAboveDrainage}. */
export type GPUTerrainHeightAboveDrainageStatus = {
  /** One row set to 1 when pointer jumping reached its fixpoint. Presence is compile-time. */
  converged?: GraphDataView<'uint32'>;
  /** One row: gated pointer-jumping rounds executed. Presence is compile-time. */
  iterationCount?: GraphDataView<'uint32'>;
};

/** Caller-owned outputs of {@link GPUTerrainHeightAboveDrainage}; at least one view is required. */
export type GPUTerrainHeightAboveDrainageOutput = {
  /**
   * Elevation of a cell minus the elevation of the stream cell its D8 path reaches first, in float32
   * elevation units. 0 on stream cells. NaN for invalid cells, cells whose path ends without
   * reaching a stream, cells whose path first meets an unresolved stream word, and cells whose
   * elevation or drainage elevation is not finite, and cells whose drainage cell is
   * `GPU_TERRAIN_DRAINAGE_UNRESOLVED`. Presence is compile-time.
   */
  heightAboveDrainage?: GraphDataView<'float32'>;
  /**
   * Cell index of the stream cell each cell drains to (the cell itself for stream cells),
   * `GPU_TERRAIN_DRAINAGE_UNRESOLVED` when the path first meets an unresolved stream word or
   * direction, or pointer jumping did not converge, or `GPU_TERRAIN_DRAINAGE_NONE`. Presence is
   * compile-time.
   */
  drainageCells?: GraphDataView<'uint32'>;
  /** One-row status words. */
  status?: GPUTerrainHeightAboveDrainageStatus;
};

/**
 * Properties for {@link GPUTerrainHeightAboveDrainage}.
 *
 * Compile-time: `grid` size, `maximumIterations`, the elevation format and calibration, and which
 * optional outputs are provided. Per-frame: the contents of `elevation`, `flowDirections` and
 * `streams`. The contributor works on D8 receiver indices only, so it has no cell-size model: the result
 * is a vertical difference in elevation units, independent of `uniform`, `web-mercator` or
 * `geographic` cell spacing.
 *
 * Output aliasing: outputs never share a buffer with an input or with each other.
 */
export type GPUTerrainHeightAboveDrainageProps = {
  /** Prefix for node and transient IDs. Defaults to `'terrain-height-above-drainage'`. Compile-time. */
  id?: string;
  /** Raster grid: its width and height size the kernels. */
  grid: GPURasterMetadata;
  /**
   * Routing surface whose heights are differenced: normally `GPUTerrainFlow.filledElevation`, or
   * the raw DEM. Validity, nodata, scale and offset are honored; cells without a finite calibrated
   * elevation get NaN. Format is compile-time, contents per-frame.
   */
  elevation: GPURasterBand;
  /**
   * ESRI D8 codes, one per cell: 1 east, 2 south-east, 4 south, 8 south-west, 16 west, 32
   * north-west, 64 north, 128 north-east; 0 terminal; `0xffffffff` invalid;
   * `GPU_TERRAIN_FLOW_UNRESOLVED` (0xfffffffe) unknown, which makes cells whose path meets it
   * before a stream cell unresolved. Any other code, and any code pointing outside the grid, is
   * treated as terminal. Typically `GPUTerrainFlow.flowDirections`; grids from TauDEM or Whitebox
   * work too. Per-frame.
   */
  flowDirections: GraphDataView<'uint32'>;
  /**
   * Stream mask, one word per cell, non-zero for stream cells. Typically `GPUTerrainFlow.streams`,
   * whose `GPU_TERRAIN_FLOW_STREAM_UNRESOLVED` word (accumulation round limit hit) marks a cell of
   * unknown stream status. Per-frame.
   */
  streams: GraphDataView<'uint32'>;
  /** Caller-owned outputs. */
  output: GPUTerrainHeightAboveDrainageOutput;
  /**
   * Maximum gated pointer-jumping rounds in `[1, 1024]`. Defaults to 32: paths of up to 2^32 cells
   * need at most 32 doubling rounds. Compile-time.
   */
  maximumIterations?: number;
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
 * a path length. Receiver cycles, possible only in caller-supplied directions, leave `output.status.converged` at 0
 * and the affected cells at `GPU_TERRAIN_DRAINAGE_UNRESOLVED`.
 *
 * Compose it with {@link GPUTerrainFlow}: pass its `filledElevation` (or the raw DEM),
 * `flowDirections` and `streams`. It is a separate contributor so grids from other tools can be used.
 */
export class GPUTerrainHeightAboveDrainage implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUTerrainHeightAboveDrainageProps;

  constructor(props: GPUTerrainHeightAboveDrainageProps) {
    this.id = props.id ?? 'terrain-height-above-drainage';
    this.props = props;
    const {id} = this;
    const {output} = props;
    const cellCount = validateRasterGrid(id, props.grid);
    validateRasterIterations(id, 'maximumIterations', props.maximumIterations ?? 32);
    if (!output.heightAboveDrainage && !output.drainageCells) {
      throw new Error(`${id} requires at least one output`);
    }
    validateTerrainTracingCellView(id, 'flowDirections', props.flowDirections, 'uint32', cellCount);
    validateTerrainTracingCellView(id, 'streams', props.streams, 'uint32', cellCount);
    if (output.heightAboveDrainage) {
      validateTerrainTracingCellView(
        id,
        'output.heightAboveDrainage',
        output.heightAboveDrainage,
        'float32',
        cellCount
      );
    }
    if (output.drainageCells) {
      validateTerrainTracingCellView(
        id,
        'output.drainageCells',
        output.drainageCells,
        'uint32',
        cellCount
      );
    }
    if (
      props.elevation.storage.kind === 'buffer' &&
      props.elevation.storage.values.length !== cellCount
    ) {
      throw new Error(`${id} elevation must contain one value per cell`);
    }
    validateTerrainTracingStatus(id, output.status);
    const outputs = [
      output.heightAboveDrainage,
      output.drainageCells,
      output.status?.converged,
      output.status?.iterationCount
    ];
    validateGraphOutputsDisjointFromInputs(id, outputs, [
      ...getRasterBandViews(props.elevation),
      props.flowDirections,
      props.streams
    ]);
    validateRasterBuffersDistinct(id, outputs, []);
  }

  /** Returns canonicalization, pointer-jumping and HAND nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {output} = props;
    const {width, height} = props.grid;
    validateRasterBandBelongsToGraph(id, graph, props.elevation, []);
    validateGraphViewsBelongToGraph(id, graph, [
      props.flowDirections,
      props.streams,
      output.heightAboveDrainage,
      output.drainageCells,
      output.status?.converged,
      output.status?.iterationCount
    ]);
    const cellCount = width * height;
    const nodes: GPUCommandNode<Parameters>[] = [];
    let elevation: GraphDataView<'float32'> | undefined;
    if (output.heightAboveDrainage) {
      const source = getRasterBufferBandNodes(graph, id, props.elevation, props.grid, true);
      nodes.push(...source.nodes);
      elevation = source.band.storage.values as GraphDataView<'float32'>;
    }
    // The pointer stays untouched by the final kernel, which reads other cells' pointers to check
    // that each path reached its fixpoint, so drainage cells are written to their own view.
    const pointer = createTransientView(graph, `${id}-pointer`, 'uint32', cellCount);
    nodes.push(
      ...createTerrainFlowPointerNodes(graph, {
        id,
        contributorId: id,
        operation: OPERATION,
        width,
        height,
        maxIterations: props.maximumIterations ?? 32,
        flowDirections: props.flowDirections,
        pointer,
        stops: {kind: 'streams', streams: props.streams},
        converged: output.status?.converged,
        iterationCount: output.status?.iterationCount
      })
    );
    const bindings: WGSLKernelBinding[] = [
      {name: 'flowDirections', view: props.flowDirections, type: 'u32', access: 'read'},
      {name: 'streams', view: props.streams, type: 'u32', access: 'read'},
      {name: 'pointer', view: pointer, type: 'u32', access: 'read'}
    ];
    if (output.drainageCells) {
      bindings.push({
        name: 'drainageOut',
        view: output.drainageCells,
        type: 'u32',
        access: 'read_write'
      });
    }
    if (elevation && output.heightAboveDrainage) {
      bindings.push(
        {name: 'elevation', view: elevation, type: 'f32', access: 'read'},
        {name: 'heightOut', view: output.heightAboveDrainage, type: 'f32', access: 'read_write'}
      );
    }
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-height`,
        operation: OPERATION,
        variant: 'height-above-drainage',
        bindings,
        invocationCount: cellCount,
        declarations: `${getTerrainFlowTracingWGSL(width, height)}
${GET_QUIET_NAN_WGSL}
${IS_FINITE_VALUE_WGSL}`,
        body: `let reached = pointer[pointerOffset + index];
  var drainage = NO_CELL;
  if (flowDirections[flowDirectionsOffset + index] != NO_CELL &&
      flowDirections[flowDirectionsOffset + reached] != NO_CELL) {
    if (pointer[pointerOffset + reached] != reached) {
      // Pointer jumping stopped early: 'reached' is an intermediate cell, not the first stream.
      drainage = ${GPU_TERRAIN_DRAINAGE_UNRESOLVED}u;
    } else if (streams[streamsOffset + reached] != 0u) {
      // An unresolved stream word stops the path too: the first stream may be that cell or lie
      // further down, so the drainage cell is unknown rather than wrong.
      drainage = select(reached, ${GPU_TERRAIN_DRAINAGE_UNRESOLVED}u,
        streams[streamsOffset + reached] == ${GPU_TERRAIN_FLOW_STREAM_UNRESOLVED}u);
    } else if (flowDirections[flowDirectionsOffset + reached] == UNRESOLVED_DIRECTION ||
        flowDirections[flowDirectionsOffset + index] == UNRESOLVED_DIRECTION) {
      // The path ends on a cell whose direction is unknown, not on an outlet or a sink.
      drainage = ${GPU_TERRAIN_DRAINAGE_UNRESOLVED}u;
    }
  }
  ${
    elevation
      ? `var result = getQuietNaN();
  if (drainage < ${GPU_TERRAIN_DRAINAGE_UNRESOLVED}u) {
    let cellElevation = elevation[elevationOffset + index];
    let drainageElevation = elevation[elevationOffset + drainage];
    if (isFiniteValue(cellElevation) && isFiniteValue(drainageElevation)) {
      result = cellElevation - drainageElevation;
    }
  }
  heightOut[heightOutOffset + index] = result;`
      : ''
  }
  ${output.drainageCells ? 'drainageOut[drainageOutOffset + index] = drainage;' : ''}`
      })
    );
    return nodes;
  }
}
