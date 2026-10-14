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
import {
  getGPUGreatCircleRadius,
  GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import {createLinePrefixNode} from './line-segmentize-kernels';
import {
  createPieceCountNode,
  createPieceEmitNode,
  createPiecePublishNode,
  createPieceRangeNode,
  createPieceSourceLineIndicesNode,
  LINE_PIECE_STRIDE,
  type LinePieceMode
} from './line-chunk-kernels';
import {type GPULineCoordinateSystem, type GPULineGeometryOutput} from './line-segmentize-types';
import {getLineGeometryOutputViews, validateLineGeometryOutput} from './line-geometry-output';

const OPERATION = 'GPULineChunk';

/** Number of float32 elements in a `GPULineChunk` parameter buffer. */
export const GPU_LINE_CHUNK_PARAMETER_LENGTH = 4;

/** CPU description of the per-frame parameters of `GPULineChunk`. */
export type GPULineChunkSettings = {
  /** `'chunk'` mode: piece length in position (or sphere-radius) units. */
  chunkLength?: number;
  /** `'substring'` mode: start distance from each line's first vertex, clamped to the line. */
  startMeasure?: number;
  /** `'substring'` mode: end distance, clamped to the line. A start past the end gives an empty line. */
  endMeasure?: number;
};

/**
 * Packs `GPULineChunk` parameters into the 4-element float32 layout
 * `[chunkLength, startMeasure, endMeasure, 0]`.
 *
 * @param settings Settings to encode. Missing values are 0 (`endMeasure` defaults to the
 * largest f32, the whole line).
 * @param target Optional destination of at least 4 elements.
 * @throws If a value is NaN, the chunk length is negative, or `target` is too short.
 */
export function getGPULineChunkParameterValues(
  settings: GPULineChunkSettings,
  target: Float32Array = new Float32Array(GPU_LINE_CHUNK_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_LINE_CHUNK_PARAMETER_LENGTH) {
    throw new Error(`Line chunk target must hold ${GPU_LINE_CHUNK_PARAMETER_LENGTH} elements`);
  }
  const chunkLength = settings.chunkLength ?? 0;
  const startMeasure = settings.startMeasure ?? 0;
  const endMeasure = Math.min(settings.endMeasure ?? 3.4e38, 3.4e38);
  if ([chunkLength, startMeasure, endMeasure].some(Number.isNaN) || chunkLength < 0) {
    throw new Error('Line chunk settings must be numbers and chunkLength non-negative');
  }
  target.set([chunkLength, Math.max(startMeasure, -3.4e38), endMeasure, 0]);
  return target;
}

/**
 * Properties for {@link GPULineChunk}.
 *
 * Per-frame (no recompile): the contents of `parameters` and of every input buffer. Compile-time:
 * view lengths, the line count, `mode`, `spatialContext`, the vertex and line
 * capacities, and which optional outputs are present.
 */
export type GPULineChunkProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'line-chunk'`. */
  id?: string;
  /** Packed positions sorted by line (longitude/latitude degrees for a great-circle `spatialContext`). */
  positions: GraphDataView<'float32x2'>;
  /** `lineCount + 1` monotonic row offsets. */
  lineOffsets: GraphDataView<'uint32'>;
  /**
   * `'chunk'` splits every line into pieces of `chunkLength` (turf `lineChunk`); `'substring'`
   * extracts the `[startMeasure, endMeasure]` part of every line (turf `lineSliceAlong`, PostGIS
   * `ST_LineSubstring` by distance).
   */
  mode: LinePieceMode;
  /**
   * Planar (default), or `{coordinateSpace: 'longitude-latitude', metric: 'great-circle'}` for
   * great-circle measures and interpolation in the context unit (meters by default). Replaces the
   * former `coordinateSystem` and `radius` props.
   */
  spatialContext?: GPUSpatialContext;
  /**
   * Per-frame packed float32 view of at least {@link GPU_LINE_CHUNK_PARAMETER_LENGTH} elements
   * written with `getGPULineChunkParameterValues`.
   */
  parameters: GraphDataView<'float32'>;
  /**
   * Output pieces. In `'chunk'` mode the line capacity is `output.lineOffsets.length - 1` and the
   * piece count is data dependent (`status.lineCount` and `sourceLineIndices` report it); in `'substring'` mode
   * there is exactly one output line per input line.
   */
  output: GPULineGeometryOutput;
};

/**
 * Splits lines into fixed-length chunks, or extracts a measure range from every line, on the GPU.
 *
 * Measures are cumulative line lengths (planar, or great-circle times the sphere radius, meters by default) from a per-line
 * Neumaier-compensated prefix. Chunk `i` of a line of length `L` covers
 * `[i * chunkLength, min((i + 1) * chunkLength, L)]`; a line gets `ceil(L / chunkLength)` chunks,
 * capped at the line capacity plus one so a tiny chunk length overflows instead of wrapping counts.
 * Every piece is an interpolated start vertex, the input vertices strictly inside its range, and an
 * interpolated end vertex, so consecutive chunks share their boundary point. Zero-length lines
 * (including single vertices) are copied as one piece; empty lines produce none. A substring whose
 * clamped start is past its clamped end is an empty line; equal measures give two equal vertices.
 *
 * Composition: piece counts per line, exclusive `GPUScan`, one range kernel per piece slot (binary
 * searches over measures), vertex counts scan, one emit kernel per piece, and a publish kernel that
 * clamps line offsets and reports overflow of either capacity. Deterministic.
 */
export class GPULineChunk implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPULineChunkProps;
  /** Resolved spatial context (planar unless given). */
  readonly spatialContext: GPUSpatialContext;
  /** Kernel model derived from `spatialContext`. */
  readonly coordinateSystem: GPULineCoordinateSystem;
  /** Resolved sphere radius in the context's length unit (1 for planar). */
  readonly radius: number;
  /** Number of output line slots. */
  readonly lineCapacity: number;

  constructor(props: GPULineChunkProps) {
    this.id = props.id ?? 'line-chunk';
    this.props = props;
    this.spatialContext = resolveGPUSpatialContext(
      this.id,
      props.spatialContext,
      GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT
    );
    this.coordinateSystem =
      this.spatialContext.coordinateSpace === 'longitude-latitude' ? 'spherical' : 'planar';
    this.radius = getGPUGreatCircleRadius(this.spatialContext) ?? 1;
    const {id} = this;
    if (props.mode !== 'chunk' && props.mode !== 'substring') {
      throw new Error(`${id} mode must be 'chunk' or 'substring'`);
    }
    validatePackedView(props.positions, ['float32x2'], `${id} positions`);
    const rowCount = props.positions.length;
    if (rowCount < 1) {
      throw new Error(`${id} needs at least one position`);
    }
    validatePackedView(props.lineOffsets, ['uint32'], `${id} lineOffsets`);
    if (props.lineOffsets.length < 2) {
      throw new Error(`${id} lineOffsets needs two rows`);
    }
    const lineCount = props.lineOffsets.length - 1;
    validatePackedView(props.parameters, ['float32'], `${id} parameters`);
    if (props.parameters.length < GPU_LINE_CHUNK_PARAMETER_LENGTH) {
      throw new Error(
        `${id} parameters must hold ${GPU_LINE_CHUNK_PARAMETER_LENGTH} float32 values`
      );
    }
    this.lineCapacity =
      props.mode === 'chunk' ? Math.max(props.output.lineOffsets.length - 1, 1) : lineCount;
    validateLineGeometryOutput(id, props.output, this.lineCapacity);
    if (
      lineCount * (this.lineCapacity + 1) > 0xffffffff ||
      2 * this.lineCapacity + rowCount > 0xffffffff
    ) {
      throw new Error(`${id} line count times line capacity must fit in 32 bits`);
    }
    validateGraphOutputsDisjointFromInputs(id, getLineGeometryOutputViews(props.output), [
      props.positions,
      props.lineOffsets,
      props.parameters
    ]);
  }

  /**
   * Returns the prefix node, piece count and scan nodes, the range node, the vertex scan nodes,
   * the emit node, the publish node and an optional source-line-index node.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id, coordinateSystem, lineCapacity} = this;
    const {output} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      props.positions,
      props.lineOffsets,
      props.parameters,
      ...getLineGeometryOutputViews(output)
    ]);
    const rowCount = props.positions.length;
    const lineCount = props.lineOffsets.length - 1;
    const rowMeasures = createTransientView(graph, `${id}-row-measures`, 'float32', rowCount);
    const rowShifts =
      coordinateSystem === 'spherical'
        ? createTransientView(graph, `${id}-row-shifts`, 'float32', rowCount)
        : undefined;
    const pieceCounts = createTransientView(graph, `${id}-piece-counts`, 'uint32', lineCount);
    const pieceStarts = createTransientView(graph, `${id}-piece-starts`, 'uint32', lineCount);
    const pieces = createTransientView(
      graph,
      `${id}-pieces`,
      'uint32',
      lineCapacity * LINE_PIECE_STRIDE
    );
    const vertexCounts = createTransientView(graph, `${id}-vertex-counts`, 'uint32', lineCapacity);
    const vertexStarts = createTransientView(graph, `${id}-vertex-starts`, 'uint32', lineCapacity);
    const pieceTotal = createTransientView(graph, `${id}-piece-total`, 'uint32', 1);
    const nodes: GPUCommandNode<Parameters>[] = [
      createLinePrefixNode<Parameters>(graph, {
        id: `${id}-prefix`,
        operation: OPERATION,
        positions: props.positions,
        lineOffsets: props.lineOffsets,
        coordinateSystem,
        radius: this.radius,
        rowMeasures,
        rowShifts
      }),
      createPieceCountNode<Parameters>(graph, {
        id: `${id}-piece-count`,
        operation: OPERATION,
        mode: props.mode,
        lineOffsets: props.lineOffsets,
        rowMeasures,
        parameters: props.parameters,
        rowCount,
        maximumPieces: lineCapacity + 1,
        pieceCounts
      }),
      ...new GPUScan({
        id: `${id}-piece-scan`,
        input: pieceCounts,
        output: pieceStarts
      }).getCommandNodes(graph),
      createPieceRangeNode<Parameters>(graph, {
        id: `${id}-piece-range`,
        operation: OPERATION,
        mode: props.mode,
        lineOffsets: props.lineOffsets,
        rowMeasures,
        parameters: props.parameters,
        pieceCounts,
        pieceStarts,
        rowCount,
        pieces,
        vertexCounts,
        pieceTotal
      }),
      ...new GPUScan({
        id: `${id}-vertex-scan`,
        input: vertexCounts,
        output: vertexStarts
      }).getCommandNodes(graph),
      createPieceEmitNode<Parameters>(graph, {
        id: `${id}-emit`,
        operation: OPERATION,
        coordinateSystem,
        positions: props.positions,
        rowMeasures,
        rowShifts,
        pieces,
        vertexStarts,
        outputPositions: output.positions,
        outputMeasures: output.measures,
        outputSourceRows: output.sourceRows
      }),
      createPiecePublishNode<Parameters>(graph, {
        id: `${id}-publish`,
        operation: OPERATION,
        pieceTotal,
        pieces,
        vertexStarts,
        capacity: output.positions.length,
        outputLineOffsets: output.lineOffsets,
        count: output.status.count,
        overflow: output.status.overflow,
        requiredCount: output.status.requiredCount,
        lineCountOutput: output.status.lineCount
      })
    ];
    if (output.sourceLineIndices) {
      nodes.push(
        createPieceSourceLineIndicesNode<Parameters>(graph, {
          id: `${id}-source-line-indexs`,
          operation: OPERATION,
          pieces,
          sourceLineIndices: output.sourceLineIndices
        })
      );
    }
    return nodes;
  }
}
