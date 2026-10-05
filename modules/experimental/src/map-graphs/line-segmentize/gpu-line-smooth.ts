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
import type {GPUMapGraphRecipe} from '../map-graph-types';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../map-graph-utils';
import {createIdentitySourcePathsNode} from './line-segmentize-kernels';
import {
  createSmoothLevelNode,
  createSmoothPrepareNode,
  createSmoothPublishNode
} from './line-smooth-kernels';
import type {GPULinePathOutput} from './line-segmentize-types';
import {getLinePathOutputViews, validateLinePathOutput} from './line-path-output';

const OPERATION = 'GPULineSmooth';

/** Largest accepted number of Chaikin iterations. */
export const GPU_LINE_SMOOTH_MAXIMUM_ITERATIONS = 10;

/** Number of float32 elements in a `GPULineSmooth` parameter buffer. */
export const GPU_LINE_SMOOTH_PARAMETER_LENGTH = 4;

/** CPU description of the per-frame parameters of `GPULineSmooth`. */
export type GPULineSmoothParameters = {
  /**
   * Chaikin cut ratio in `(0, 0.5]`. Each edge `(a, b)` is replaced by `mix(a, b, ratio)` and
   * `mix(a, b, 1 - ratio)`. Default 0.25 (classic Chaikin).
   */
  ratio?: number;
};

/**
 * Packs `GPULineSmooth` parameters into the 4-element float32 layout `[ratio, 0, 0, 0]`.
 *
 * @param parameters Parameters to encode.
 * @param target Optional destination of at least 4 elements.
 * @throws If the ratio is outside `(0, 0.5]` or `target` is too short.
 */
export function getGPULineSmoothParameterValues(
  parameters: GPULineSmoothParameters = {},
  target: Float32Array = new Float32Array(GPU_LINE_SMOOTH_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_LINE_SMOOTH_PARAMETER_LENGTH) {
    throw new Error(`Line smooth target must hold ${GPU_LINE_SMOOTH_PARAMETER_LENGTH} elements`);
  }
  const ratio = parameters.ratio ?? 0.25;
  if (!(ratio > 0 && ratio <= 0.5)) {
    throw new Error('Line smooth ratio must be in (0, 0.5]');
  }
  target.set([ratio, 0, 0, 0]);
  return target;
}

/**
 * Properties for {@link GPULineSmooth}.
 *
 * Per-frame (no recompile): the contents of `parameters` (cut ratio) and of every input buffer.
 * Compile-time: view lengths, the path count, `iterations`, `closed`, the output capacity, and
 * which optional outputs are present.
 */
export type GPULineSmoothProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'line-smooth'`. */
  id?: string;
  /** Packed planar positions sorted by path. */
  positions: GraphDataView<'float32x2'>;
  /** `pathCount + 1` monotonic row offsets. */
  pathOffsets: GraphDataView<'uint32'>;
  /** Compile-time number of Chaikin iterations in `[1, 10]`. Each one doubles the vertex count. */
  iterations: number;
  /**
   * Treat every path as a closed ring (polygon outlines). A repeated closing vertex in the input is
   * ignored, and every smoothed ring repeats its first vertex at the end. Default false.
   */
  closed?: boolean;
  /**
   * Per-frame packed float32 view of at least {@link GPU_LINE_SMOOTH_PARAMETER_LENGTH} elements
   * written with `getGPULineSmoothParameterValues`.
   */
  parameters: GraphDataView<'float32'>;
  /**
   * Smoothed paths, one per input path. `sourceRows` and `measures` are not supported (a smoothed
   * vertex blends several input vertices).
   */
  output: GPULinePathOutput;
};

/**
 * Chaikin corner-cutting smoothing of many paths or rings at once (turf `polygonSmooth`, QGIS
 * Smooth, the usual way to soften generalized boundaries or GPS tracks).
 *
 * One iteration replaces every edge `(a, b)` with `mix(a, b, r)` and `mix(a, b, 1 - r)`. Open paths
 * keep their two endpoints, so `n` vertices become `2n`; closed rings of `n` distinct vertices
 * become `2n` (plus the repeated first vertex in the output). Paths too short to smooth (fewer than
 * two vertices, or rings with fewer than three) are copied unchanged. After `k` iterations a path
 * has `n * 2^k` vertices.
 *
 * Output sizes depend only on the input layout, so two exclusive scans of per-path counts give
 * every level's path offsets in closed form; each level is one kernel over the capacity whose
 * slots find their path by binary search and blend two vertices of the previous level. Levels
 * ping-pong through two capacity-sized graph transients, which is enough because a vertex past the
 * capacity at one level only feeds vertices past the capacity at the next. Deterministic.
 *
 * Planar only: geographic coordinates are smoothed as planar longitude/latitude.
 */
export class GPULineSmooth implements GPUMapGraphRecipe {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Stable recipe name. */
  readonly recipe = 'line-smooth';
  /** Validated properties. */
  readonly props: GPULineSmoothProps;
  /** Whether paths are closed rings. */
  readonly closed: boolean;

  constructor(props: GPULineSmoothProps) {
    this.id = props.id ?? this.recipe;
    this.props = props;
    this.closed = props.closed ?? false;
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
    const {iterations} = props;
    if (
      !Number.isInteger(iterations) ||
      iterations < 1 ||
      iterations > GPU_LINE_SMOOTH_MAXIMUM_ITERATIONS
    ) {
      throw new Error(
        `${id} iterations must be an integer in [1, ${GPU_LINE_SMOOTH_MAXIMUM_ITERATIONS}]`
      );
    }
    validatePackedView(props.positions, ['float32x2'], `${id} positions`);
    const rowCount = props.positions.length;
    if (rowCount < 1) {
      throw new Error(`${id} needs at least one position`);
    }
    validatePackedUint32View(props.pathOffsets, `${id} pathOffsets`);
    if (props.pathOffsets.length < 2) {
      throw new Error(`${id} pathOffsets must contain at least two rows`);
    }
    const pathCount = props.pathOffsets.length - 1;
    if (rowCount * 2 ** iterations + pathCount > 0xffffffff) {
      throw new Error(`${id} positions.length * 2^iterations must fit in 32 bits`);
    }
    validatePackedView(props.parameters, ['float32'], `${id} parameters`);
    if (props.parameters.length < GPU_LINE_SMOOTH_PARAMETER_LENGTH) {
      throw new Error(
        `${id} parameters must hold ${GPU_LINE_SMOOTH_PARAMETER_LENGTH} float32 values`
      );
    }
    validateLinePathOutput(id, props.output, pathCount);
    if (props.output.sourceRows || props.output.measures) {
      throw new Error(`${id} output.sourceRows and output.measures are not supported`);
    }
    validateGraphOutputsDisjointFromInputs(id, getLinePathOutputViews(props.output), [
      props.positions,
      props.pathOffsets,
      props.parameters
    ]);
  }

  /** Returns the prepare node, the scan nodes, one node per iteration, publish and source paths. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id, closed} = this;
    const {output, iterations} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      props.positions,
      props.pathOffsets,
      props.parameters,
      ...getLinePathOutputViews(output)
    ]);
    const pathCount = props.pathOffsets.length - 1;
    const capacity = output.positions.length;
    const column = (name: string) =>
      createTransientView(graph, `${id}-${name}`, 'uint32', pathCount + 1);
    const smoothCounts = column('smooth-counts');
    const plainCounts = column('plain-counts');
    const smoothFlags = column('smooth-flags');
    const smoothStarts = column('smooth-starts');
    const plainStarts = column('plain-starts');
    const smoothFlagStarts = closed ? column('smooth-flag-starts') : undefined;
    const nodes: GPUCommandNode<Parameters>[] = [
      createSmoothPrepareNode<Parameters>(graph, {
        id: `${id}-prepare`,
        operation: OPERATION,
        positions: props.positions,
        pathOffsets: props.pathOffsets,
        closed,
        smoothCounts,
        plainCounts,
        smoothFlags
      }),
      ...new GPUScan({
        id: `${id}-smooth-scan`,
        input: smoothCounts,
        output: smoothStarts
      }).getCommandNodes(graph),
      ...new GPUScan({
        id: `${id}-plain-scan`,
        input: plainCounts,
        output: plainStarts
      }).getCommandNodes(graph)
    ];
    if (smoothFlagStarts) {
      nodes.push(
        ...new GPUScan({
          id: `${id}-flag-scan`,
          input: smoothFlags,
          output: smoothFlagStarts
        }).getCommandNodes(graph)
      );
    }
    // Two ping-pong level buffers, only as many as the intermediate levels need.
    const levelBuffers = [0, 1]
      .slice(0, Math.min(iterations - 1, 2))
      .map(index => createTransientView(graph, `${id}-level-${index}`, 'float32x2', capacity));
    let source: GraphDataView<'float32x2'> = props.positions;
    for (let level = 0; level < iterations; level++) {
      const isFinal = level === iterations - 1;
      const destination = isFinal ? output.positions : levelBuffers[level % 2];
      nodes.push(
        createSmoothLevelNode<Parameters>(graph, {
          id: `${id}-level-${level}`,
          operation: OPERATION,
          level,
          isFinal,
          closed,
          pathCount,
          source,
          pathOffsets: props.pathOffsets,
          smoothStarts,
          plainStarts,
          smoothFlagStarts,
          parameters: props.parameters,
          destination
        })
      );
      source = destination;
    }
    nodes.push(
      createSmoothPublishNode<Parameters>(graph, {
        id: `${id}-publish`,
        operation: OPERATION,
        iterations,
        closed,
        pathCount,
        smoothStarts,
        plainStarts,
        smoothFlagStarts,
        capacity,
        outputPathOffsets: output.pathOffsets,
        count: output.count,
        overflow: output.overflow,
        totalCount: output.totalCount,
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
