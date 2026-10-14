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
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validatePackedView
} from '../../utils/gpu-contributor-utils';
import type {
  GPULineGeometryFields,
  GPUPolygonGeometryFields
} from '../../analysis-core/contracts/index';
import {
  getLineGeometryOutputViews,
  validateLineGeometryOutput
} from '../line-segmentize/line-geometry-output';
import type {GPULineGeometryOutput} from '../line-segmentize/line-segmentize-types';
import {
  createLineClipCountNode,
  createLineClipEmitNode,
  createLineClipPublishNode,
  createLineTailNode,
  createPolygonPublishNode,
  createPolygonStageCountNode,
  createPolygonStageEmitNode,
  createPolygonStageOffsetsNode,
  GPU_RECTANGLE_CLIP_PARAMETER_LENGTH
} from './rectangle-clip-kernels';
import {
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';

const OPERATION = 'GPURectangleClip';

export {GPU_RECTANGLE_CLIP_PARAMETER_LENGTH};

/** CPU description of the per-frame clip rectangle of {@link GPURectangleClip}. */
export type GPURectangleClipSettings = {
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
 * Packs {@link GPURectangleClipSettings} into the 4-element float32 layout
 * `[minX, minY, maxX, maxY]`. Write the result into a `GPUParameterBuffer` between encodings to
 * move the clip rectangle (for example the viewport) without recompiling.
 *
 * @param settings Settings to encode.
 * @param target Optional destination of at least 4 elements.
 * @throws If a value is not finite, the rectangle is inverted, or `target` is too short.
 */
export function getGPURectangleClipParameterValues(
  settings: GPURectangleClipSettings,
  target: Float32Array = new Float32Array(GPU_RECTANGLE_CLIP_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_RECTANGLE_CLIP_PARAMETER_LENGTH) {
    throw new Error(
      `Rectangle clip target must hold ${GPU_RECTANGLE_CLIP_PARAMETER_LENGTH} elements`
    );
  }
  const {minX, minY, maxX, maxY} = settings;
  if (![minX, minY, maxX, maxY].every(Number.isFinite)) {
    throw new Error('Rectangle clip settings must be finite');
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
export type GPURectangleClipProps = (
  | ({
      /** Open lines clipped with Liang-Barsky. Takes `positions` and `lineOffsets`. */
      geometryType: 'lines';
    } & GPULineGeometryFields)
  | ({
      /**
       * Polygon rings clipped with Sutherland-Hodgman. Takes the four-level polygon layout
       * (`positions`, `featureOffsets`, `polygonOffsets`, `ringOffsets`). Rings close implicitly.
       */
      geometryType: 'polygons';
    } & GPUPolygonGeometryFields)
) & {
  /** Prefix for generated node and transient IDs. Defaults to `'rectangle-clip'`. */
  id?: string;
  /**
   * Planar only (default: native planar coordinates). A longitude-latitude context throws at
   * construction: project positions to a planar frame first, for example with
   * `createGPULocalFrame` (`contracts/local-frame`) and `GPULocalFrameProjection`
   * (`geometry-measures`), and pass the frame's context.
   */
  spatialContext?: GPUSpatialContext;
  /** Per-frame packed float32 view written with {@link getGPURectangleClipParameterValues}. */
  parameters: GraphDataView<'float32'>;
  /**
   * Bounded output. `positions.length` is the vertex capacity. For `'polygons'`,
   * `output.lineOffsets` holds the output ring offsets and must have the length of the input
   * `ringOffsets` (one output ring per input ring, possibly empty), so the input `featureOffsets`
   * and `polygonOffsets` describe the output rings unchanged; `sourceLineIndices`, `sourceRows` and
   * `measures` are not supported. For
   * `'lines'`, `lineOffsets.length - 1` is the output line capacity; `sourceLineIndices` gives each
   * output line's input line.
   */
  output: GPULineGeometryOutput;
};

/**
 * Clips lines and polygon rings to an axis-aligned rectangle (turf `bboxClip`, Shapely
 * `clip_by_rect`, PostGIS `ST_ClipByBox2D`), with the rectangle read per frame, so a viewport clip
 * runs without recompiling.
 *
 * **Lines** use Liang-Barsky per segment. Surviving segments are joined back into lines: a
 * line piece continues across a vertex when both neighboring segments survive unclipped there,
 * and a line that leaves and re-enters the rectangle becomes several output lines. Output
 * vertices and lines come from scans, so order follows the input; single-vertex lines and
 * segments that only touch the rectangle at a point are dropped.
 *
 * **Polygons** use Sutherland-Hodgman against the four edges in turn (left, right, bottom, top),
 * each stage a count, scan and emit over the vertices of the previous stage, so a ring keeps its
 * position in `ringOffsets` (possibly with zero vertices). Concave polygons that leave and
 * re-enter the rectangle keep zero-width bridges along the clip edges, as always with
 * Sutherland-Hodgman: the filled area is exact, the outline may trace along the border. Rings are
 * clipped independently, so a polygon with holes keeps working under any hole rule.
 *
 * Output is bounded by `output.positions.length` (all four stages share that capacity); when it
 * overflows, `overflow` is 1 and later vertices and lines are truncated while offsets stay
 * valid. Deterministic, no atomics. Precision: f32; intersections lie on the rectangle edges
 * exactly in the clipped coordinate and within an f32 rounding of the true crossing in the other.
 */
export class GPURectangleClip implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPURectangleClipProps;
  /** Resolved spatial context (planar only). */
  readonly spatialContext: GPUSpatialContext;
  /** Line offsets, or the polygon ring offsets, as read by the kernels. */
  readonly boundaryOffsets: GraphDataView<'uint32'>;

  constructor(props: GPURectangleClipProps) {
    this.id = props.id ?? 'rectangle-clip';
    this.spatialContext = resolveGPUSpatialContext(this.id, props.spatialContext);
    this.props = props;
    const {id} = this;
    this.boundaryOffsets =
      props.geometryType === 'polygons' ? props.ringOffsets : props.lineOffsets;
    if (props.geometryType !== 'lines' && props.geometryType !== 'polygons') {
      throw new Error(`${id} geometryType must be 'lines' or 'polygons'`);
    }
    validatePackedView(props.positions, ['float32x2'], `${id} positions`);
    if (props.positions.length < 1) {
      throw new Error(`${id} needs at least one position`);
    }
    const offsetsName = props.geometryType === 'polygons' ? 'ringOffsets' : 'lineOffsets';
    validatePackedView(this.boundaryOffsets, ['uint32'], `${id} ${offsetsName}`);
    if (this.boundaryOffsets.length < 2) {
      throw new Error(`${id} ${offsetsName} needs two rows`);
    }
    if (props.geometryType === 'polygons') {
      validatePackedView(props.featureOffsets, ['uint32'], `${id} featureOffsets`);
      validatePackedView(props.polygonOffsets, ['uint32'], `${id} polygonOffsets`);
    }
    validatePackedView(props.parameters, ['float32'], `${id} parameters`);
    if (props.parameters.length < GPU_RECTANGLE_CLIP_PARAMETER_LENGTH) {
      throw new Error(
        `${id} parameters must hold ${GPU_RECTANGLE_CLIP_PARAMETER_LENGTH} float32 values`
      );
    }
    validateLineGeometryOutput(id, props.output, props.output.lineOffsets.length - 1);
    if (props.geometryType === 'polygons') {
      if (props.output.lineOffsets.length !== this.boundaryOffsets.length) {
        throw new Error(`${id} output.lineOffsets must hold ${this.boundaryOffsets.length} rows`);
      }
      for (const name of ['sourceLineIndices', 'sourceRows', 'measures'] as const) {
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
    validateGraphOutputsDisjointFromInputs(id, getLineGeometryOutputViews(props.output), [
      props.positions,
      this.boundaryOffsets,
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
      this.boundaryOffsets,
      props.parameters,
      ...getLineGeometryOutputViews(props.output)
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
    const lineCapacity = output.lineOffsets.length - 1;
    const inputs = {
      operation: OPERATION,
      positions: props.positions,
      lineOffsets: this.boundaryOffsets,
      parameters: props.parameters
    };
    const counts = createTransientView(graph, `${id}-counts`, 'uint32', rowCount);
    const startFlags = createTransientView(graph, `${id}-start-flags`, 'uint32', rowCount);
    const starts = createTransientView(graph, `${id}-starts`, 'uint32', rowCount);
    const lineIndices = createTransientView(graph, `${id}-line-indices`, 'uint32', rowCount);
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
        id: `${id}-line-scan`,
        input: startFlags,
        output: lineIndices,
        mode: 'exclusive'
      }).getCommandNodes(graph),
      createLineClipEmitNode<Parameters>(graph, {
        ...inputs,
        id: `${id}-emit`,
        starts,
        lineIndices,
        outputPositions: output.positions,
        outputLineOffsets: output.lineOffsets,
        outputSourceLineIndices: output.sourceLineIndices
      }),
      createLineClipPublishNode<Parameters>(graph, {
        id: `${id}-publish`,
        operation: OPERATION,
        capacity,
        lineCapacity,
        starts,
        counts,
        lineIndices,
        startFlags,
        totals,
        count: output.status.count,
        overflow: output.status.overflow,
        requiredCount: output.status.requiredCount
      }),
      createLineTailNode<Parameters>(graph, {
        id: `${id}-line-tail`,
        operation: OPERATION,
        totals,
        lineOffsets: output.lineOffsets,
        sourceLineIndices: output.sourceLineIndices,
        lineCount: output.status.lineCount
      })
    ];
  }

  private _getPolygonNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    const {output} = props;
    const capacity = output.positions.length;
    const ringCount = this.boundaryOffsets.length - 1;
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
    let inputOffsets = this.boundaryOffsets;
    for (let stage = 0; stage < 4; stage++) {
      const isLast = stage === 3;
      const outputPositions = isLast ? output.positions : positionBuffers[stage % 2];
      const outputOffsets = isLast ? output.lineOffsets : offsetBuffers[stage % 2];
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
        count: output.status.count,
        overflow: output.status.overflow,
        requiredCount: output.status.requiredCount,
        lineCount: output.status.lineCount
      })
    );
    return nodes;
  }
}
