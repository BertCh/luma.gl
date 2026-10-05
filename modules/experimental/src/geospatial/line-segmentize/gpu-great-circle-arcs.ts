// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUScan,
  GraphVectorView,
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
  createArcCountNode,
  createArcEmitNode,
  createIdentitySourcePathsNode,
  createPathOffsetsPublishNode
} from './line-segmentize-kernels';
import {GPU_LINE_SEGMENTIZE_PARAMETER_LENGTH} from './line-segmentize-parameters';
import type {GPULinePathOutput} from './line-segmentize-types';
import {getLinePathOutputViews, validateLinePathOutput} from './line-path-output';

const OPERATION = 'GPUGreatCircleArcs';

/** Default compile-time cap on segments per arc. */
export const GPU_GREAT_CIRCLE_ARCS_DEFAULT_MAXIMUM_SEGMENTS = 256;

/**
 * Properties for {@link GPUGreatCircleArcs}.
 *
 * Per-frame (no recompile): the contents of `parameters` (resolution) and of `sources` and
 * `targets`. Compile-time: the pair count, `radius`, `maximumSegments`, the output capacity, and
 * which optional outputs are present.
 */
export type GPUGreatCircleArcsProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'great-circle-arcs'`. */
  id?: string;
  /** Arc sources, longitude/latitude degrees, one row per pair. */
  sources: GraphDataView<'float32x2'>;
  /** Arc targets, longitude/latitude degrees, aligned with `sources`. */
  targets: GraphDataView<'float32x2'>;
  /** Sphere radius. Defaults to {@link GPU_GEODESIC_MEAN_EARTH_RADIUS} (meters). */
  radius?: number;
  /**
   * Compile-time cap on segments per arc. Default 256. `pairCount * (maximumSegments + 1)` must
   * fit in 32 bits.
   */
  maximumSegments?: number;
  /**
   * Per-frame packed float32 view of at least {@link GPU_LINE_SEGMENTIZE_PARAMETER_LENGTH} elements
   * written with `getGPUGreatCircleArcsParameterValues`.
   */
  parameters: GraphDataView<'float32'>;
  /** Arc paths, one output path per pair (`pathOffsets` has `pairCount + 1` rows). */
  output: GPULinePathOutput;
};

/**
 * Tessellates great-circle arcs between many origin/destination pairs on the GPU (turf
 * `greatCircle`, the CPU step behind flow maps drawn with `PathLayer`).
 *
 * Pair `i` becomes one path of `segments + 1` vertices, where
 * `segments = clamp(max(ceil(distance / maximumSegmentLength), minimumSegments), 1, maximumSegments)`
 * and `distance` is the haversine distance times `radius`. Interior vertices are slerped evenly in
 * angle; the first vertex is the source and the last is the target exactly.
 *
 * Antimeridian: longitudes are unwrapped to stay continuous from the source (for example a
 * Tokyo to San Francisco arc runs from 139.7 to 237.6), never split into two paths. Feed the
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
  /** Resolved sphere radius. */
  readonly radius: number;
  /** Resolved compile-time segment cap. */
  readonly maximumSegments: number;

  constructor(props: GPUGreatCircleArcsProps) {
    this.id = props.id ?? 'great-circle-arcs';
    this.props = props;
    this.radius = props.radius ?? GPU_GEODESIC_MEAN_EARTH_RADIUS;
    this.maximumSegments = props.maximumSegments ?? GPU_GREAT_CIRCLE_ARCS_DEFAULT_MAXIMUM_SEGMENTS;
    const {id} = this;
    for (const [name, view] of [
      ['sources', props.sources],
      ['targets', props.targets],
      ['parameters', props.parameters]
    ] as const) {
      if ((view as unknown) instanceof GraphVectorView) {
        throw new Error(`${id} ${name} must be a single packed view, not a chunked vector`);
      }
    }
    validatePackedView(props.sources, ['float32x2'], `${id} sources`);
    validatePackedView(props.targets, ['float32x2'], `${id} targets`);
    const pairCount = props.sources.length;
    if (pairCount < 1) {
      throw new Error(`${id} needs at least one pair`);
    }
    if (props.targets.length !== pairCount) {
      throw new Error(`${id} targets length must equal sources length`);
    }
    if (!Number.isFinite(this.radius) || this.radius <= 0) {
      throw new Error(`${id} radius must be a positive finite number`);
    }
    if (
      !Number.isSafeInteger(this.maximumSegments) ||
      this.maximumSegments < 1 ||
      pairCount * (this.maximumSegments + 1) > 0xffffffff
    ) {
      throw new Error(
        `${id} maximumSegments must be a positive integer with pairCount * (maximumSegments + 1) < 2^32`
      );
    }
    validatePackedView(props.parameters, ['float32'], `${id} parameters`);
    if (props.parameters.length < GPU_LINE_SEGMENTIZE_PARAMETER_LENGTH) {
      throw new Error(
        `${id} parameters must hold ${GPU_LINE_SEGMENTIZE_PARAMETER_LENGTH} float32 values`
      );
    }
    validateLinePathOutput(id, props.output, pairCount);
    validateGraphOutputsDisjointFromInputs(id, getLinePathOutputViews(props.output), [
      props.sources,
      props.targets,
      props.parameters
    ]);
  }

  /** Returns the count node, the scan nodes, the emit node, the publish node and an optional source-path node. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id, radius} = this;
    const {output} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      props.sources,
      props.targets,
      props.parameters,
      ...getLinePathOutputViews(output)
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
        maximumSegments: this.maximumSegments,
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
      createPathOffsetsPublishNode<Parameters>(graph, {
        id: `${id}-publish`,
        operation: OPERATION,
        pathCount: pairCount,
        rowCount: pairCount,
        counts,
        starts,
        outputPathOffsets: output.pathOffsets,
        capacity: output.positions.length,
        count: output.count,
        overflow: output.overflow,
        totalCount: output.totalCount,
        pathCountOutput: output.pathCount
      })
    ];
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
