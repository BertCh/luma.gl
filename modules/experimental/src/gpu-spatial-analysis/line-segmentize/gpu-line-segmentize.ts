// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUScan,
  GraphVectorView,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {GPU_GEODESIC_MEAN_EARTH_RADIUS} from '../geometry-measures/geodesic-wgsl';
import {
  createIdentitySourcePathsNode,
  createPathOffsetsPublishNode,
  createPathPrefixNode,
  createSegmentizeCountNode,
  createSegmentizeEmitNode
} from './line-segmentize-kernels';
import {GPU_LINE_SEGMENTIZE_PARAMETER_LENGTH} from './line-segmentize-parameters';
import type {GPULineCoordinateSystem, GPULinePathOutput} from './line-segmentize-types';
import {getLinePathOutputViews, validateLinePathOutput} from './line-path-output';

const OPERATION = 'GPULineSegmentize';

/** Default compile-time cap on output pieces per input segment. */
export const GPU_LINE_SEGMENTIZE_DEFAULT_MAXIMUM_PIECES = 1024;

/**
 * Properties for {@link GPULineSegmentize}.
 *
 * Per-frame (no recompile): the contents of `parameters` (maximum segment length) and of every
 * input buffer. Compile-time: view lengths, the path count, `coordinateSystem`, `radius`,
 * `maximumPiecesPerSegment`, the output capacity, and which optional outputs are present.
 */
export type GPULineSegmentizeProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'line-segmentize'`. */
  id?: string;
  /**
   * Packed positions sorted by path: planar coordinates, or longitude/latitude degrees for
   * `'spherical'`.
   */
  positions: GraphDataView<'float32x2'>;
  /** `pathCount + 1` monotonic row offsets; path `p` owns rows `[pathOffsets[p], pathOffsets[p + 1])`. */
  pathOffsets: GraphDataView<'uint32'>;
  /** `'planar'` (default) lerps straight segments; `'spherical'` slerps great-circle arcs. */
  coordinateSystem?: GPULineCoordinateSystem;
  /** Sphere radius for `'spherical'` lengths. Defaults to {@link GPU_GEODESIC_MEAN_EARTH_RADIUS}. */
  radius?: number;
  /**
   * Compile-time cap on output pieces per input segment, so a tiny maximum length cannot explode
   * counts. Default 1024. `positions.length * maximumPiecesPerSegment` must fit in 32 bits.
   */
  maximumPiecesPerSegment?: number;
  /**
   * Per-frame packed float32 view of at least {@link GPU_LINE_SEGMENTIZE_PARAMETER_LENGTH} elements
   * written with `getGPULineSegmentizeParameterValues`.
   */
  parameters: GraphDataView<'float32'>;
  /** Densified paths, one output path per input path. */
  output: GPULinePathOutput;
};

/**
 * Densifies many paths so that no output segment is longer than a per-frame maximum length
 * (turf-style densify, PostGIS `ST_Segmentize`, the CPU tessellation step of arc and path layers).
 *
 * Every input segment of length `L` is split into
 * `clamp(ceil(L / maximumSegmentLength), 1, maximumPiecesPerSegment)` equal pieces; input vertices
 * are always kept. `'planar'` lerps; `'spherical'` treats positions as longitude/latitude degrees,
 * measures haversine lengths times `radius`, and slerps interior points evenly in angle along the
 * great circle. Spherical output longitudes are unwrapped so each path stays continuous across
 * the antimeridian (values may leave `[-180, 180]`), which deck.gl `PathLayer` handles with
 * `wrapLongitude` or as-is in a globe view.
 *
 * Composition: optional per-path prefix (Neumaier-compensated cumulative measures and longitude
 * unwrapping shifts, sequential in row order), per-row count, exclusive `GPUScan`, per-row emit,
 * and a publish kernel that clamps path offsets to the capacity. Deterministic; no atomics.
 *
 * Precision: f32. Interior spherical points come from f32 unit vectors, about 1e-7 rad (under a
 * meter on Earth) from the f64 slerp.
 */
export class GPULineSegmentize implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPULineSegmentizeProps;
  /** Resolved coordinate system. */
  readonly coordinateSystem: GPULineCoordinateSystem;
  /** Resolved sphere radius. */
  readonly radius: number;
  /** Resolved compile-time piece cap. */
  readonly maximumPiecesPerSegment: number;

  constructor(props: GPULineSegmentizeProps) {
    this.id = props.id ?? 'line-segmentize';
    this.props = props;
    this.coordinateSystem = props.coordinateSystem ?? 'planar';
    this.radius = props.radius ?? GPU_GEODESIC_MEAN_EARTH_RADIUS;
    this.maximumPiecesPerSegment =
      props.maximumPiecesPerSegment ?? GPU_LINE_SEGMENTIZE_DEFAULT_MAXIMUM_PIECES;
    const {id} = this;
    for (const [name, view] of [
      ['positions', props.positions],
      ['pathOffsets', props.pathOffsets],
      ['parameters', props.parameters]
    ] as const) {
      if ((view as unknown) instanceof GraphVectorView) {
        throw new Error(`${id} ${name} must be a single packed view, not a chunked vector`);
      }
    }
    if (this.coordinateSystem !== 'planar' && this.coordinateSystem !== 'spherical') {
      throw new Error(`${id} coordinateSystem must be 'planar' or 'spherical'`);
    }
    if (!Number.isFinite(this.radius) || this.radius <= 0) {
      throw new Error(`${id} radius must be a positive finite number`);
    }
    validatePackedView(props.positions, ['float32x2'], `${id} positions`);
    const rowCount = props.positions.length;
    if (rowCount < 1) {
      throw new Error(`${id} needs at least one position`);
    }
    if (
      !Number.isSafeInteger(this.maximumPiecesPerSegment) ||
      this.maximumPiecesPerSegment < 1 ||
      rowCount * this.maximumPiecesPerSegment > 0xffffffff
    ) {
      throw new Error(
        `${id} maximumPiecesPerSegment must be a positive integer with positions.length * maximumPiecesPerSegment < 2^32`
      );
    }
    validatePackedUint32View(props.pathOffsets, `${id} pathOffsets`);
    if (props.pathOffsets.length < 2) {
      throw new Error(`${id} pathOffsets must contain at least two rows`);
    }
    validatePackedView(props.parameters, ['float32'], `${id} parameters`);
    if (props.parameters.length < GPU_LINE_SEGMENTIZE_PARAMETER_LENGTH) {
      throw new Error(
        `${id} parameters must hold ${GPU_LINE_SEGMENTIZE_PARAMETER_LENGTH} float32 values`
      );
    }
    validateLinePathOutput(id, props.output, props.pathOffsets.length - 1);
    validateGraphOutputsDisjointFromInputs(id, getLinePathOutputViews(props.output), [
      props.positions,
      props.pathOffsets,
      props.parameters
    ]);
  }

  /**
   * Returns the optional prefix node, the count node, the scan nodes, the emit node, the publish
   * node and an optional source-path node.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id, coordinateSystem, radius} = this;
    const {output} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      props.positions,
      props.pathOffsets,
      props.parameters,
      ...getLinePathOutputViews(output)
    ]);
    const rowCount = props.positions.length;
    const pathCount = props.pathOffsets.length - 1;
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
        createPathPrefixNode<Parameters>(graph, {
          id: `${id}-prefix`,
          operation: OPERATION,
          positions: props.positions,
          pathOffsets: props.pathOffsets,
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
        pathOffsets: props.pathOffsets,
        parameters: props.parameters,
        coordinateSystem,
        radius,
        maximumPiecesPerSegment: this.maximumPiecesPerSegment,
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
      createPathOffsetsPublishNode<Parameters>(graph, {
        id: `${id}-publish`,
        operation: OPERATION,
        pathOffsets: props.pathOffsets,
        pathCount,
        rowCount,
        counts,
        starts,
        outputPathOffsets: output.pathOffsets,
        capacity: output.positions.length,
        count: output.count,
        overflow: output.overflow,
        requiredCount: output.requiredCount,
        pathCountOutput: output.pathCount
      })
    );
    if (output.sourcePaths) {
      nodes.push(
        createIdentitySourcePathsNode<Parameters>(graph, {
          id: `${id}-source-paths`,
          operation: OPERATION,
          sourcePaths: output.sourcePaths
        })
      );
    }
    return nodes;
  }
}
