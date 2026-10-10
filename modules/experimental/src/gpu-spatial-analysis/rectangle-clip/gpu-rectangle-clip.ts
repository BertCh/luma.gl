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
import {getLinePathOutputViews, validateLinePathOutput} from '../line-segmentize/line-path-output';
import type {GPULinePathOutput} from '../line-segmentize/line-segmentize-types';
import {
  createLineClipCountNode,
  createLineClipEmitNode,
  createLineClipPublishNode,
  createPathTailNode,
  createPolygonPublishNode,
  createPolygonStageCountNode,
  createPolygonStageEmitNode,
  createPolygonStageOffsetsNode,
  GPU_RECTANGLE_CLIP_PARAMETER_LENGTH
} from './rectangle-clip-kernels';

const OPERATION = 'GPURectangleClip';

export {GPU_RECTANGLE_CLIP_PARAMETER_LENGTH};

/** CPU description of the per-frame clip rectangle of {@link GPURectangleClip}. */
export type GPURectangleClipParameters = {
  /** Left edge. */
  minX: number;
  /** Bottom edge. */
  minY: number;
  /** Right edge, at least `minX`. */
  maxX: number;
  /** Top edge, at least `minY`. */
  maxY: number;
};

/**
 * Packs {@link GPURectangleClipParameters} into the 4-element float32 layout
 * `[minX, minY, maxX, maxY]`. Write the result into a `GPUParameterBuffer` between encodings to
 * move the clip rectangle (for example the viewport) without recompiling.
 *
 * @param parameters Parameters to encode.
 * @param target Optional destination of at least 4 elements.
 * @throws If a value is not finite, the rectangle is inverted, or `target` is too short.
 */
export function getGPURectangleClipParameterValues(
  parameters: GPURectangleClipParameters,
  target: Float32Array = new Float32Array(GPU_RECTANGLE_CLIP_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_RECTANGLE_CLIP_PARAMETER_LENGTH) {
    throw new Error(
      `Rectangle clip target must hold ${GPU_RECTANGLE_CLIP_PARAMETER_LENGTH} elements`
    );
  }
  const {minX, minY, maxX, maxY} = parameters;
  if (![minX, minY, maxX, maxY].every(Number.isFinite)) {
    throw new Error('Rectangle clip parameters must be finite');
  }
  if (maxX < minX || maxY < minY) {
    throw new Error('Rectangle clip requires maxX >= minX and maxY >= minY');
  }
  target.set([minX, minY, maxX, maxY]);
  return target;
}

/**
 * Properties for {@link GPURectangleClip}.
 *
 * Per-frame (no recompile): the rectangle in `parameters` and every input buffer. Compile-time:
 * view lengths, `geometryType`, and the output capacities.
 */
export type GPURectangleClipProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'rectangle-clip'`. */
  id?: string;
  /** Packed planar vertex positions. */
  positions: GraphDataView<'float32x2'>;
  /** `'lines'`: open paths clipped with Liang-Barsky. `'polygons'`: closed rings clipped with Sutherland-Hodgman. */
  geometryType: 'lines' | 'polygons';
  /**
   * `pathCount + 1` (or `ringCount + 1`) monotonic vertex offsets; path `p` owns rows
   * `[pathOffsets[p], pathOffsets[p + 1])`. Polygon rings close implicitly.
   */
  pathOffsets: GraphDataView<'uint32'>;
  /** Per-frame packed float32 view written with {@link getGPURectangleClipParameterValues}. */
  parameters: GraphDataView<'float32'>;
  /**
   * Bounded output. `positions.length` is the vertex capacity. For `'polygons'`,
   * `pathOffsets.length` must equal the input `pathOffsets.length` (one output ring per input
   * ring, possibly empty) and `sourcePaths`, `sourceRows`, `measures` are not supported. For
   * `'lines'`, `pathOffsets.length - 1` is the output path capacity; `sourcePaths` gives each
   * output path's input path.
   */
  output: GPULinePathOutput;
};

/**
 * Clips lines and polygon rings to an axis-aligned rectangle (turf `bboxClip`, Shapely
 * `clip_by_rect`, PostGIS `ST_ClipByBox2D`), with the rectangle read per frame, so a viewport clip
 * runs without recompiling.
 *
 * **Lines** use Liang-Barsky per segment. Surviving segments are joined back into paths: a
 * path piece continues across a vertex when both neighboring segments survive unclipped there,
 * and a line that leaves and re-enters the rectangle becomes several output paths. Output
 * vertices and paths come from scans, so order follows the input; single-vertex paths and
 * segments that only touch the rectangle at a point are dropped.
 *
 * **Polygons** use Sutherland-Hodgman against the four edges in turn (left, right, bottom, top),
 * each stage a count, scan and emit over the vertices of the previous stage, so a ring keeps its
 * position in `pathOffsets` (possibly with zero vertices). Concave polygons that leave and
 * re-enter the rectangle keep zero-width bridges along the clip edges, as always with
 * Sutherland-Hodgman: the filled area is exact, the outline may trace along the border. Rings are
 * clipped independently, so a polygon with holes keeps working under any hole rule.
 *
 * Output is bounded by `output.positions.length` (all four stages share that capacity); when it
 * overflows, `overflow` is 1 and later vertices and paths are truncated while offsets stay
 * valid. Deterministic, no atomics. Precision: f32; intersections lie on the rectangle edges
 * exactly in the clipped coordinate and within an f32 rounding of the true crossing in the other.
 */
export class GPURectangleClip implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPURectangleClipProps;

  constructor(props: GPURectangleClipProps) {
    this.id = props.id ?? 'rectangle-clip';
    this.props = props;
    const {id} = this;
    for (const [name, view] of Object.entries({
      positions: props.positions,
      pathOffsets: props.pathOffsets,
      parameters: props.parameters,
      ...Object.fromEntries(
        Object.entries(props.output).map(([key, view]) => [`output.${key}`, view])
      )
    })) {
      if ((view as unknown) instanceof GraphVectorView) {
        throw new Error(`${id} ${name} must be a single packed view, not a chunked vector`);
      }
    }
    if (props.geometryType !== 'lines' && props.geometryType !== 'polygons') {
      throw new Error(`${id} geometryType must be 'lines' or 'polygons'`);
    }
    validatePackedView(props.positions, ['float32x2'], `${id} positions`);
    if (props.positions.length < 1) {
      throw new Error(`${id} needs at least one position`);
    }
    validatePackedUint32View(props.pathOffsets, `${id} pathOffsets`);
    if (props.pathOffsets.length < 2) {
      throw new Error(`${id} pathOffsets must contain at least two rows`);
    }
    validatePackedView(props.parameters, ['float32'], `${id} parameters`);
    if (props.parameters.length < GPU_RECTANGLE_CLIP_PARAMETER_LENGTH) {
      throw new Error(
        `${id} parameters must hold ${GPU_RECTANGLE_CLIP_PARAMETER_LENGTH} float32 values`
      );
    }
    validateLinePathOutput(id, props.output, props.output.pathOffsets.length - 1);
    if (props.geometryType === 'polygons') {
      if (props.output.pathOffsets.length !== props.pathOffsets.length) {
        throw new Error(`${id} output.pathOffsets must hold ${props.pathOffsets.length} rows`);
      }
      for (const name of ['sourcePaths', 'sourceRows', 'measures'] as const) {
        if (props.output[name]) {
          throw new Error(`${id} output.${name} is not supported for polygons`);
        }
      }
    } else {
      for (const name of ['sourceRows', 'measures'] as const) {
        if (props.output[name]) {
          throw new Error(`${id} output.${name} is not supported`);
        }
      }
    }
    validateGraphOutputsDisjointFromInputs(id, getLinePathOutputViews(props.output), [
      props.positions,
      props.pathOffsets,
      props.parameters
    ]);
  }

  /** Returns the line nodes or the four polygon stages followed by the publish node. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    validateGraphViewsBelongToGraph(id, graph, [
      props.positions,
      props.pathOffsets,
      props.parameters,
      ...getLinePathOutputViews(props.output)
    ]);
    return props.geometryType === 'lines'
      ? this._getLineNodes(graph)
      : this._getPolygonNodes(graph);
  }

  private _getLineNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    const {output} = props;
    const rowCount = props.positions.length;
    const capacity = output.positions.length;
    const pathCapacity = output.pathOffsets.length - 1;
    const inputs = {
      operation: OPERATION,
      positions: props.positions,
      pathOffsets: props.pathOffsets,
      parameters: props.parameters
    };
    const counts = createTransientView(graph, `${id}-counts`, 'uint32', rowCount);
    const startFlags = createTransientView(graph, `${id}-start-flags`, 'uint32', rowCount);
    const starts = createTransientView(graph, `${id}-starts`, 'uint32', rowCount);
    const pathIndices = createTransientView(graph, `${id}-path-indices`, 'uint32', rowCount);
    const totals = createTransientView(graph, `${id}-totals`, 'uint32', 2);
    return [
      createLineClipCountNode<Parameters>(graph, {
        ...inputs,
        id: `${id}-count`,
        counts,
        startFlags
      }),
      ...new GPUScan({
        id: `${id}-vertex-scan`,
        input: counts,
        output: starts,
        mode: 'exclusive'
      }).getCommandNodes(graph),
      ...new GPUScan({
        id: `${id}-path-scan`,
        input: startFlags,
        output: pathIndices,
        mode: 'exclusive'
      }).getCommandNodes(graph),
      createLineClipEmitNode<Parameters>(graph, {
        ...inputs,
        id: `${id}-emit`,
        starts,
        pathIndices,
        outputPositions: output.positions,
        outputPathOffsets: output.pathOffsets,
        outputSourcePaths: output.sourcePaths
      }),
      createLineClipPublishNode<Parameters>(graph, {
        id: `${id}-publish`,
        operation: OPERATION,
        capacity,
        pathCapacity,
        starts,
        counts,
        pathIndices,
        startFlags,
        totals,
        count: output.count,
        overflow: output.overflow,
        requiredCount: output.requiredCount
      }),
      createPathTailNode<Parameters>(graph, {
        id: `${id}-path-tail`,
        operation: OPERATION,
        totals,
        pathOffsets: output.pathOffsets,
        sourcePaths: output.sourcePaths,
        pathCount: output.pathCount
      })
    ];
  }

  private _getPolygonNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    const {output} = props;
    const capacity = output.positions.length;
    const ringCount = props.pathOffsets.length - 1;
    const stageFlags = createTransientView(graph, `${id}-stage-totals`, 'uint32', 4);
    const positionBuffers = [
      createTransientView(graph, `${id}-positions-a`, 'float32x2', capacity),
      createTransientView(graph, `${id}-positions-b`, 'float32x2', capacity)
    ];
    const offsetBuffers = [
      createTransientView(graph, `${id}-offsets-a`, 'uint32', ringCount + 1),
      createTransientView(graph, `${id}-offsets-b`, 'uint32', ringCount + 1)
    ];
    const nodes: GPUCommandNode<Parameters>[] = [];
    let inputPositions = props.positions;
    let inputOffsets = props.pathOffsets;
    for (let stage = 0; stage < 4; stage++) {
      const isLast = stage === 3;
      const outputPositions = isLast ? output.positions : positionBuffers[stage % 2];
      const outputOffsets = isLast ? output.pathOffsets : offsetBuffers[stage % 2];
      const rows = inputPositions.length;
      const counts = createTransientView(graph, `${id}-counts-${stage}`, 'uint32', rows);
      const starts = createTransientView(graph, `${id}-starts-${stage}`, 'uint32', rows);
      const stageInputs = {
        operation: OPERATION,
        boundary: stage,
        positions: inputPositions,
        ringOffsets: inputOffsets,
        parameters: props.parameters
      };
      nodes.push(
        createPolygonStageCountNode<Parameters>(graph, {
          ...stageInputs,
          id: `${id}-count-${stage}`,
          counts
        }),
        ...new GPUScan({
          id: `${id}-scan-${stage}`,
          input: counts,
          output: starts,
          mode: 'exclusive'
        }).getCommandNodes(graph),
        createPolygonStageEmitNode<Parameters>(graph, {
          ...stageInputs,
          id: `${id}-emit-${stage}`,
          starts,
          outputPositions
        }),
        createPolygonStageOffsetsNode<Parameters>(graph, {
          id: `${id}-offsets-${stage}`,
          operation: OPERATION,
          capacity,
          inputOffsets,
          starts,
          counts,
          outputOffsets,
          stageFlags,
          stage
        })
      );
      inputPositions = outputPositions;
      inputOffsets = outputOffsets;
    }
    nodes.push(
      createPolygonPublishNode<Parameters>(graph, {
        id: `${id}-publish`,
        operation: OPERATION,
        capacity,
        ringCount,
        stageFlags,
        count: output.count,
        overflow: output.overflow,
        requiredCount: output.requiredCount,
        pathCount: output.pathCount
      })
    );
    return nodes;
  }
}
