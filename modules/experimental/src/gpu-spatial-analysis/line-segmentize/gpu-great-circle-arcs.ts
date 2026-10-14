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
  getGPUSphereRadius,
  GPU_GREAT_CIRCLE_SPATIAL_CONTEXT,
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import {
  createArcCountNode,
  createArcEmitNode,
  createIdentitySourceLineIndicesNode,
  createLineOffsetsPublishNode
} from './line-segmentize-kernels';
import {GPU_LINE_SEGMENTIZE_PARAMETER_LENGTH} from './line-segmentize-parameters';
import type {GPULineGeometryOutput} from './line-segmentize-types';
import {getLineGeometryOutputViews, validateLineGeometryOutput} from './line-geometry-output';

const OPERATION = 'GPUGreatCircleArcs';

// Arc subdivision uses great-circle distance.
const SPATIAL_CONTEXT_SUPPORT = Object.freeze({
  coordinateSpaces: ['longitude-latitude'] as const,
  metrics: ['great-circle'] as const,
  measure: 'distance'
});

/** Default compile-time cap on segments per arc. */
export const GPU_GREAT_CIRCLE_ARCS_DEFAULT_MAXIMUM_SEGMENT_COUNT = 256;

/**
 * Properties for {@link GPUGreatCircleArcs}.
 *
 * Per-frame (no recompile): the contents of `parameters` (resolution) and of `sources` and
 * `targets`. Compile-time: the pair count, `spatialContext`, `maximumSegmentCount`, the output capacity, and
 * which optional outputs are present.
 */
export type GPUGreatCircleArcsProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'great-circle-arcs'`. */
  id?: string;
  /** Arc sources, longitude/latitude degrees, one row per pair. */
  sources: GraphDataView<'float32x2'>;
  /** Arc targets, longitude/latitude degrees, aligned with `sources`. */
  targets: GraphDataView<'float32x2'>;
  /**
   * Longitude/latitude with a great-circle metric (the default, meters on the mean Earth radius).
   * `sphereRadius` and `units` set the sphere and the length unit. Planar contexts throw.
   * Replaces the former `radius` prop.
   */
  spatialContext?: GPUSpatialContext;
  /**
   * Compile-time cap on segments per arc. Default 256. `pairCount * (maximumSegmentCount + 1)` must
   * fit in 32 bits.
   */
  maximumSegmentCount?: number;
  /**
   * Per-frame packed float32 view of at least {@link GPU_LINE_SEGMENTIZE_PARAMETER_LENGTH} elements
   * written with `getGPUGreatCircleArcsParameterValues`.
   */
  parameters: GraphDataView<'float32'>;
  /** Arc lines, one output line per pair (`lineOffsets` has `pairCount + 1` rows). */
  output: GPULineGeometryOutput;
};

/**
 * Tessellates great-circle arcs between many origin/destination pairs on the GPU (turf
 * `greatCircle`, the CPU step behind flow maps drawn with `PathLayer`).
 *
 * Pair `i` becomes one line of `segments + 1` vertices, where
 * `segments = clamp(max(ceil(distance / maximumSegmentLength), minimumSegments), 1, maximumSegmentCount)`
 * and `distance` is the haversine distance times the sphere radius. Interior vertices are slerped evenly in
 * angle; the first vertex is the source and the last is the target exactly.
 *
 * Antimeridian: longitudes are unwrapped to stay continuous from the source (for example a
 * Tokyo to San Francisco arc runs from 139.7 to 237.6), never split into two lines. Feed the
 * result to `PathLayer` with `wrapLongitude: true`, or use it directly in a globe view.
 *
 * Limits: a pair less than about 1e-3 rad from antipodal has no unique great circle; its arc is
 * numerically unstable in f32 and should be split by the caller. Zero-length pairs emit
 * `minimumSegments + 1` copies of the source.
 */
export class GPUGreatCircleArcs implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUGreatCircleArcsProps;
  /** Resolved spatial context (great-circle longitude/latitude). */
  readonly spatialContext: GPUSpatialContext;
  /** Resolved sphere radius in the context's length unit. */
  readonly radius: number;
  /** Resolved compile-time segment cap. */
  readonly maximumSegmentCount: number;

  constructor(props: GPUGreatCircleArcsProps) {
    this.id = props.id ?? 'great-circle-arcs';
    this.props = props;
    this.spatialContext = resolveGPUSpatialContext(
      this.id,
      props.spatialContext ?? GPU_GREAT_CIRCLE_SPATIAL_CONTEXT,
      SPATIAL_CONTEXT_SUPPORT
    );
    this.radius = getGPUSphereRadius(this.spatialContext);
    this.maximumSegmentCount =
      props.maximumSegmentCount ?? GPU_GREAT_CIRCLE_ARCS_DEFAULT_MAXIMUM_SEGMENT_COUNT;
    const {id} = this;
    validatePackedView(props.sources, ['float32x2'], `${id} sources`);
    validatePackedView(props.targets, ['float32x2'], `${id} targets`);
    const pairCount = props.sources.length;
    if (pairCount < 1) {
      throw new Error(`${id} needs at least one pair`);
    }
    if (props.targets.length !== pairCount) {
      throw new Error(`${id} targets length must equal sources length`);
    }
    // pairCount * (maximumSegmentCount + 1) must fit in 32 bits.
    validateCapacity(
      id,
      'maximumSegmentCount',
      this.maximumSegmentCount,
      1,
      Math.floor(0xffffffff / pairCount) - 1
    );
    validatePackedView(props.parameters, ['float32'], `${id} parameters`);
    if (props.parameters.length < GPU_LINE_SEGMENTIZE_PARAMETER_LENGTH) {
      throw new Error(
        `${id} parameters must hold ${GPU_LINE_SEGMENTIZE_PARAMETER_LENGTH} float32 values`
      );
    }
    validateLineGeometryOutput(id, props.output, pairCount);
    validateGraphOutputsDisjointFromInputs(id, getLineGeometryOutputViews(props.output), [
      props.sources,
      props.targets,
      props.parameters
    ]);
  }

  /** Returns the count node, the scan nodes, the emit node, the publish node and an optional source-line-index node. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id, radius} = this;
    const {output} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      props.sources,
      props.targets,
      props.parameters,
      ...getLineGeometryOutputViews(output)
    ]);
    const pairCount = props.sources.length;
    const counts = createTransientView(graph, `${id}-counts`, 'uint32', pairCount);
    const starts = createTransientView(graph, `${id}-starts`, 'uint32', pairCount);
    const nodes: GPUCommandNode<Parameters>[] = [
      createArcCountNode<Parameters>(graph, {
        id: `${id}-count`,
        operation: OPERATION,
        sources: props.sources,
        targets: props.targets,
        parameters: props.parameters,
        radius,
        maximumSegmentCount: this.maximumSegmentCount,
        counts
      }),
      ...new GPUScan({
        id: `${id}-scan`,
        input: counts,
        output: starts,
        mode: 'exclusive'
      }).getCommandNodes(graph),
      createArcEmitNode<Parameters>(graph, {
        id: `${id}-emit`,
        operation: OPERATION,
        sources: props.sources,
        targets: props.targets,
        radius,
        counts,
        starts,
        outputPositions: output.positions,
        outputSourceRows: output.sourceRows,
        outputMeasures: output.measures
      }),
      createLineOffsetsPublishNode<Parameters>(graph, {
        id: `${id}-publish`,
        operation: OPERATION,
        lineCount: pairCount,
        rowCount: pairCount,
        counts,
        starts,
        outputLineOffsets: output.lineOffsets,
        capacity: output.positions.length,
        count: output.status.count,
        overflow: output.status.overflow,
        requiredCount: output.status.requiredCount,
        lineCountOutput: output.status.lineCount
      })
    ];
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
