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
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {
  validateCapacity,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validatePackedView
} from '../../utils/gpu-contributor-utils';
import {
  getGPUGreatCircleRadius,
  GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import {
  createIdentitySourceLineIndicesNode,
  createLineOffsetsPublishNode,
  createLinePrefixNode,
  createSegmentizeCountNode,
  createSegmentizeEmitNode
} from './line-segmentize-kernels';
import {GPU_LINE_SEGMENTIZE_PARAMETER_LENGTH} from './line-segmentize-parameters';
import {type GPULineCoordinateSystem, type GPULineGeometryOutput} from './line-segmentize-types';
import {getLineGeometryOutputViews, validateLineGeometryOutput} from './line-geometry-output';

const OPERATION = 'GPULineSegmentize';

// Segment subdivision uses measured segment lengths.
const SPATIAL_CONTEXT_SUPPORT = Object.freeze({
  ...GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
  measure: 'distance'
});

/** Default compile-time cap on output pieces per input segment. */
export const GPU_LINE_SEGMENTIZE_DEFAULT_MAXIMUM_PIECE_COUNT = 1024;

/**
 * Properties for {@link GPULineSegmentize}.
 *
 * Per-frame (no recompile): the contents of `parameters` (maximum segment length) and of every
 * input buffer. Compile-time: view lengths, the line count, `spatialContext`,
 * `maximumPieceCount`, the output capacity, and which optional outputs are present.
 */
export type GPULineSegmentizeProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'line-segmentize'`. */
  id?: string;
  /**
   * Packed positions sorted by line: planar coordinates, or longitude/latitude degrees with a
   * great-circle `spatialContext`.
   */
  positions: GraphDataView<'float32x2'>;
  /** `lineCount + 1` monotonic row offsets; line `p` owns rows `[lineOffsets[p], lineOffsets[p + 1])`. */
  lineOffsets: GraphDataView<'uint32'>;
  /**
   * Planar (default, straight segments) or `{coordinateSpace: 'longitude-latitude', metric:
   * 'great-circle'}` (slerped great-circle arcs, lengths in the context unit on its
   * `sphereRadius`, meters on the mean Earth radius by default). Replaces the former
   * `coordinateSystem` and `radius` props.
   */
  spatialContext?: GPUSpatialContext;
  /**
   * Compile-time cap on output pieces per input segment, so a tiny maximum length cannot explode
   * counts. Default 1024. `positions.length * maximumPieceCount` must fit in 32 bits.
   */
  maximumPieceCount?: number;
  /**
   * Per-frame packed float32 view of at least {@link GPU_LINE_SEGMENTIZE_PARAMETER_LENGTH} elements
   * written with `getGPULineSegmentizeParameterValues`.
   */
  parameters: GraphDataView<'float32'>;
  /** Densified lines, one output line per input line. */
  output: GPULineGeometryOutput;
};

/**
 * Densifies many lines so that no output segment is longer than a per-frame maximum length
 * (turf-style densify, PostGIS `ST_Segmentize`, the CPU tessellation step of arc and line layers).
 *
 * Every input segment of length `L` is split into
 * `clamp(ceil(L / maximumSegmentLength), 1, maximumPieceCount)` equal pieces; input vertices
 * are always kept. A planar context lerps; a great-circle longitude/latitude context treats
 * positions as degrees, measures haversine lengths times the sphere radius (meters by default), and slerps interior points evenly in angle along the
 * great circle. Spherical output longitudes are unwrapped so each line stays continuous across
 * the antimeridian (values may leave `[-180, 180]`), which deck.gl `PathLayer` handles with
 * `wrapLongitude` or as-is in a globe view.
 *
 * Composition: optional per-line prefix (Neumaier-compensated cumulative measures and longitude
 * unwrapping shifts, sequential in row order), per-row count, exclusive `GPUScan`, per-row emit,
 * and a publish kernel that clamps line offsets to the capacity. Deterministic; no atomics.
 *
 * Precision: f32. Interior spherical points come from f32 unit vectors, about 1e-7 rad (under a
 * meter on Earth) from the f64 slerp.
 */
export class GPULineSegmentize implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPULineSegmentizeProps;
  /** Resolved spatial context (planar unless given). */
  readonly spatialContext: GPUSpatialContext;
  /** Kernel model derived from `spatialContext`. */
  readonly coordinateSystem: GPULineCoordinateSystem;
  /** Resolved sphere radius in the context's length unit (1 for planar). */
  readonly radius: number;
  /** Resolved compile-time piece cap. */
  readonly maximumPieceCount: number;

  constructor(props: GPULineSegmentizeProps) {
    this.id = props.id ?? 'line-segmentize';
    this.props = props;
    this.spatialContext = resolveGPUSpatialContext(
      this.id,
      props.spatialContext,
      SPATIAL_CONTEXT_SUPPORT
    );
    this.coordinateSystem =
      this.spatialContext.coordinateSpace === 'longitude-latitude' ? 'spherical' : 'planar';
    this.radius = getGPUGreatCircleRadius(this.spatialContext) ?? 1;
    this.maximumPieceCount =
      props.maximumPieceCount ?? GPU_LINE_SEGMENTIZE_DEFAULT_MAXIMUM_PIECE_COUNT;
    const {id} = this;
    validatePackedView(props.positions, ['float32x2'], `${id} positions`);
    const rowCount = props.positions.length;
    if (rowCount < 1) {
      throw new Error(`${id} needs at least one position`);
    }
    // positions.length * maximumPieceCount must fit in 32 bits.
    validateCapacity(
      id,
      'maximumPieceCount',
      this.maximumPieceCount,
      1,
      Math.floor(0xffffffff / rowCount)
    );
    validatePackedView(props.lineOffsets, ['uint32'], `${id} lineOffsets`);
    if (props.lineOffsets.length < 2) {
      throw new Error(`${id} lineOffsets needs two rows`);
    }
    validatePackedView(props.parameters, ['float32'], `${id} parameters`);
    if (props.parameters.length < GPU_LINE_SEGMENTIZE_PARAMETER_LENGTH) {
      throw new Error(
        `${id} parameters must hold ${GPU_LINE_SEGMENTIZE_PARAMETER_LENGTH} float32 values`
      );
    }
    validateLineGeometryOutput(id, props.output, props.lineOffsets.length - 1);
    validateGraphOutputsDisjointFromInputs(id, getLineGeometryOutputViews(props.output), [
      props.positions,
      props.lineOffsets,
      props.parameters
    ]);
  }

  /**
   * Returns the optional prefix node, the count node, the scan nodes, the emit node, the publish
   * node and an optional source-line-index node.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id, coordinateSystem, radius} = this;
    const {output} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      props.positions,
      props.lineOffsets,
      props.parameters,
      ...getLineGeometryOutputViews(output)
    ]);
    const rowCount = props.positions.length;
    const lineCount = props.lineOffsets.length - 1;
    const counts = createTransientView(graph, `${id}-counts`, 'uint32', rowCount);
    const starts = createTransientView(graph, `${id}-starts`, 'uint32', rowCount);
    const rowMeasures = output.measures
      ? createTransientView(graph, `${id}-row-measures`, 'float32', rowCount)
      : undefined;
    const rowShifts =
      coordinateSystem === 'spherical'
        ? createTransientView(graph, `${id}-row-shifts`, 'float32', rowCount)
        : undefined;
    const nodes: GPUCommandNode<Parameters>[] = [];
    if (rowMeasures || rowShifts) {
      nodes.push(
        createLinePrefixNode<Parameters>(graph, {
          id: `${id}-prefix`,
          operation: OPERATION,
          positions: props.positions,
          lineOffsets: props.lineOffsets,
          coordinateSystem,
          radius,
          rowMeasures,
          rowShifts
        })
      );
    }
    nodes.push(
      createSegmentizeCountNode<Parameters>(graph, {
        id: `${id}-count`,
        operation: OPERATION,
        positions: props.positions,
        lineOffsets: props.lineOffsets,
        parameters: props.parameters,
        coordinateSystem,
        radius,
        maximumPieceCount: this.maximumPieceCount,
        counts
      }),
      ...new GPUScan({
        id: `${id}-scan`,
        input: counts,
        output: starts,
        mode: 'exclusive'
      }).getCommandNodes(graph),
      createSegmentizeEmitNode<Parameters>(graph, {
        id: `${id}-emit`,
        operation: OPERATION,
        positions: props.positions,
        coordinateSystem,
        radius,
        counts,
        starts,
        rowMeasures,
        rowShifts,
        outputPositions: output.positions,
        outputSourceRows: output.sourceRows,
        outputMeasures: output.measures
      }),
      createLineOffsetsPublishNode<Parameters>(graph, {
        id: `${id}-publish`,
        operation: OPERATION,
        lineOffsets: props.lineOffsets,
        lineCount,
        rowCount,
        counts,
        starts,
        outputLineOffsets: output.lineOffsets,
        capacity: output.positions.length,
        count: output.status.count,
        overflow: output.status.overflow,
        requiredCount: output.status.requiredCount,
        lineCountOutput: output.status.lineCount
      })
    );
    if (output.sourceLineIndices) {
      nodes.push(
        createIdentitySourceLineIndicesNode<Parameters>(graph, {
          id: `${id}-source-line-indexs`,
          operation: OPERATION,
          sourceLineIndices: output.sourceLineIndices
        })
      );
    }
    return nodes;
  }
}
