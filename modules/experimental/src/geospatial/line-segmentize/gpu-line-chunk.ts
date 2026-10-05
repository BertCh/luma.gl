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
import {createPathPrefixNode} from './line-segmentize-kernels';
import {
  createPieceCountNode,
  createPieceEmitNode,
  createPiecePublishNode,
  createPieceRangeNode,
  createPieceSourcePathsNode,
  LINE_PIECE_STRIDE,
  type LinePieceMode
} from './line-chunk-kernels';
import type {GPULineCoordinateSystem, GPULinePathOutput} from './line-segmentize-types';
import {getLinePathOutputViews, validateLinePathOutput} from './line-path-output';

const OPERATION = 'GPULineChunk';

/** Number of float32 elements in a `GPULineChunk` parameter buffer. */
export const GPU_LINE_CHUNK_PARAMETER_LENGTH = 4;

/** CPU description of the per-frame parameters of `GPULineChunk`. */
export type GPULineChunkParameters = {
  /** `'chunk'` mode: piece length in position (or sphere-radius) units. */
  chunkLength?: number;
  /** `'substring'` mode: start distance from each path's first vertex, clamped to the path. */
  startMeasure?: number;
  /** `'substring'` mode: end distance, clamped to the path. A start past the end gives an empty path. */
  endMeasure?: number;
};

/**
 * Packs `GPULineChunk` parameters into the 4-element float32 layout
 * `[chunkLength, startMeasure, endMeasure, 0]`.
 *
 * @param parameters Parameters to encode. Missing values are 0 (`endMeasure` defaults to the
 * largest f32, the whole path).
 * @param target Optional destination of at least 4 elements.
 * @throws If a value is NaN, the chunk length is negative, or `target` is too short.
 */
export function getGPULineChunkParameterValues(
  parameters: GPULineChunkParameters,
  target: Float32Array = new Float32Array(GPU_LINE_CHUNK_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_LINE_CHUNK_PARAMETER_LENGTH) {
    throw new Error(`Line chunk target must hold ${GPU_LINE_CHUNK_PARAMETER_LENGTH} elements`);
  }
  const chunkLength = parameters.chunkLength ?? 0;
  const startMeasure = parameters.startMeasure ?? 0;
  const endMeasure = Math.min(parameters.endMeasure ?? 3.4e38, 3.4e38);
  if ([chunkLength, startMeasure, endMeasure].some(Number.isNaN) || chunkLength < 0) {
    throw new Error('Line chunk parameters must be numbers and chunkLength non-negative');
  }
  target.set([chunkLength, Math.max(startMeasure, -3.4e38), endMeasure, 0]);
  return target;
}

/**
 * Properties for {@link GPULineChunk}.
 *
 * Per-frame (no recompile): the contents of `parameters` and of every input buffer. Compile-time:
 * view lengths, the path count, `mode`, `coordinateSystem`, `radius`, the vertex and path
 * capacities, and which optional outputs are present.
 */
export type GPULineChunkProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'line-chunk'`. */
  id?: string;
  /** Packed positions sorted by path (longitude/latitude degrees for `'spherical'`). */
  positions: GraphDataView<'float32x2'>;
  /** `pathCount + 1` monotonic row offsets. */
  pathOffsets: GraphDataView<'uint32'>;
  /**
   * `'chunk'` splits every path into pieces of `chunkLength` (turf `lineChunk`); `'substring'`
   * extracts the `[startMeasure, endMeasure]` part of every path (turf `lineSliceAlong`, PostGIS
   * `ST_LineSubstring` by distance).
   */
  mode: LinePieceMode;
  /** `'planar'` (default) or `'spherical'` (great-circle measures and interpolation). */
  coordinateSystem?: GPULineCoordinateSystem;
  /** Sphere radius for `'spherical'`. Defaults to {@link GPU_GEODESIC_MEAN_EARTH_RADIUS}. */
  radius?: number;
  /**
   * Per-frame packed float32 view of at least {@link GPU_LINE_CHUNK_PARAMETER_LENGTH} elements
   * written with `getGPULineChunkParameterValues`.
   */
  parameters: GraphDataView<'float32'>;
  /**
   * Output pieces. In `'chunk'` mode the path capacity is `output.pathOffsets.length - 1` and the
   * piece count is data dependent (`pathCount` and `sourcePaths` report it); in `'substring'` mode
   * there is exactly one output path per input path.
   */
  output: GPULinePathOutput;
};

/**
 * Splits paths into fixed-length chunks, or extracts a measure range from every path, on the GPU.
 *
 * Measures are cumulative path lengths (planar, or great-circle times `radius`) from a per-path
 * Neumaier-compensated prefix. Chunk `i` of a path of length `L` covers
 * `[i * chunkLength, min((i + 1) * chunkLength, L)]`; a path gets `ceil(L / chunkLength)` chunks,
 * capped at the path capacity plus one so a tiny chunk length overflows instead of wrapping counts.
 * Every piece is an interpolated start vertex, the input vertices strictly inside its range, and an
 * interpolated end vertex, so consecutive chunks share their boundary point. Zero-length paths
 * (including single vertices) are copied as one piece; empty paths produce none. A substring whose
 * clamped start is past its clamped end is an empty path; equal measures give two equal vertices.
 *
 * Composition: piece counts per path, exclusive `GPUScan`, one range kernel per piece slot (binary
 * searches over measures), vertex counts scan, one emit kernel per piece, and a publish kernel that
 * clamps path offsets and reports overflow of either capacity. Deterministic.
 */
export class GPULineChunk implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPULineChunkProps;
  /** Resolved coordinate system. */
  readonly coordinateSystem: GPULineCoordinateSystem;
  /** Resolved sphere radius. */
  readonly radius: number;
  /** Number of output path slots. */
  readonly pathCapacity: number;

  constructor(props: GPULineChunkProps) {
    this.id = props.id ?? 'line-chunk';
    this.props = props;
    this.coordinateSystem = props.coordinateSystem ?? 'planar';
    this.radius = props.radius ?? GPU_GEODESIC_MEAN_EARTH_RADIUS;
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
    if (props.mode !== 'chunk' && props.mode !== 'substring') {
      throw new Error(`${id} mode must be 'chunk' or 'substring'`);
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
    validatePackedUint32View(props.pathOffsets, `${id} pathOffsets`);
    if (props.pathOffsets.length < 2) {
      throw new Error(`${id} pathOffsets must contain at least two rows`);
    }
    const pathCount = props.pathOffsets.length - 1;
    validatePackedView(props.parameters, ['float32'], `${id} parameters`);
    if (props.parameters.length < GPU_LINE_CHUNK_PARAMETER_LENGTH) {
      throw new Error(
        `${id} parameters must hold ${GPU_LINE_CHUNK_PARAMETER_LENGTH} float32 values`
      );
    }
    this.pathCapacity =
      props.mode === 'chunk' ? Math.max(props.output.pathOffsets.length - 1, 1) : pathCount;
    validateLinePathOutput(id, props.output, this.pathCapacity);
    if (
      pathCount * (this.pathCapacity + 1) > 0xffffffff ||
      2 * this.pathCapacity + rowCount > 0xffffffff
    ) {
      throw new Error(`${id} path count times path capacity must fit in 32 bits`);
    }
    validateGraphOutputsDisjointFromInputs(id, getLinePathOutputViews(props.output), [
      props.positions,
      props.pathOffsets,
      props.parameters
    ]);
  }

  /**
   * Returns the prefix node, piece count and scan nodes, the range node, the vertex scan nodes,
   * the emit node, the publish node and an optional source-path node.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id, coordinateSystem, pathCapacity} = this;
    const {output} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      props.positions,
      props.pathOffsets,
      props.parameters,
      ...getLinePathOutputViews(output)
    ]);
    const rowCount = props.positions.length;
    const pathCount = props.pathOffsets.length - 1;
    const rowMeasures = createTransientView(graph, `${id}-row-measures`, 'float32', rowCount);
    const rowShifts =
      coordinateSystem === 'spherical'
        ? createTransientView(graph, `${id}-row-shifts`, 'float32', rowCount)
        : undefined;
    const pieceCounts = createTransientView(graph, `${id}-piece-counts`, 'uint32', pathCount);
    const pieceStarts = createTransientView(graph, `${id}-piece-starts`, 'uint32', pathCount);
    const pieces = createTransientView(
      graph,
      `${id}-pieces`,
      'uint32',
      pathCapacity * LINE_PIECE_STRIDE
    );
    const vertexCounts = createTransientView(graph, `${id}-vertex-counts`, 'uint32', pathCapacity);
    const vertexStarts = createTransientView(graph, `${id}-vertex-starts`, 'uint32', pathCapacity);
    const pieceTotal = createTransientView(graph, `${id}-piece-total`, 'uint32', 1);
    const nodes: GPUCommandNode<Parameters>[] = [
      createPathPrefixNode<Parameters>(graph, {
        id: `${id}-prefix`,
        operation: OPERATION,
        positions: props.positions,
        pathOffsets: props.pathOffsets,
        coordinateSystem,
        radius: this.radius,
        rowMeasures,
        rowShifts
      }),
      createPieceCountNode<Parameters>(graph, {
        id: `${id}-piece-count`,
        operation: OPERATION,
        mode: props.mode,
        pathOffsets: props.pathOffsets,
        rowMeasures,
        parameters: props.parameters,
        rowCount,
        maximumPieces: pathCapacity + 1,
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
        pathOffsets: props.pathOffsets,
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
        outputPathOffsets: output.pathOffsets,
        count: output.count,
        overflow: output.overflow,
        totalCount: output.totalCount,
        pathCountOutput: output.pathCount
      })
    ];
    if (output.sourcePaths) {
      nodes.push(
        createPieceSourcePathsNode<Parameters>(graph, {
          id: `${id}-source-paths`,
          operation: OPERATION,
          pieces,
          sourcePaths: output.sourcePaths
        })
      );
    }
    return nodes;
  }
}
