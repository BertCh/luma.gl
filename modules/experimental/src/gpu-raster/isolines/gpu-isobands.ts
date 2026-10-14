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
import {
  createWGSLKernelNode,
  createPublishNode,
  type WGSLKernelBinding
} from '../../utils/wgsl-kernel-nodes';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {
  getBreakSearchWGSL,
  getRasterAlgebraValueWGSL,
  validateRasterAlgebraAliasing,
  validateRasterAlgebraCount,
  validateRasterAlgebraGraph,
  validateRasterAlgebraNoData,
  validateRasterAlgebraView
} from '../raster-algebra/raster-algebra-utils';
import {validateRasterGrid} from '../raster-band-utils';
import type {GPURasterMetadata} from '../types';
import {GPU_ISOBANDS_PARAMETER_LENGTH} from './isobands-parameters';
import {getIsobandsCellWGSL} from './isobands-wgsl';

const OPERATION = 'GPUIsobands';

/**
 * Caller-owned outputs of {@link GPUIsobands}. Provide any of `bandClasses`, the geometry group
 * (`triangles`, `triangleBands`, `status.count`, `status.overflow`, optional `status.requiredCount`
 * and `status.vertexCount`) and the boundary edge group (`edges`, `edgeBands`, `status.edgeCount`,
 * `status.edgeOverflow`, optional `status.edgeRequiredCount`).
 */
export type GPUIsobandsOutput = {
  /**
   * Band class per sample (`width * height` rows): the number of active breaks `<=` the value,
   * `0xffffffff` for nodata samples. Always covers every band, regardless of `firstBand` and
   * `lastBand`.
   */
  bandClasses?: GraphDataView<'uint32'>;
  /**
   * Triangle vertices in world coordinates relative to the parameters' `localOrigin`, three
   * consecutive rows per triangle, counter-clockwise. The triangle capacity is
   * `triangleBands.length`, so this view needs `3 * triangleBands.length` rows.
   */
  triangles?: GraphDataView<'float32x2'>;
  /** Band index per triangle. Its length is the triangle capacity. */
  triangleBands?: GraphDataView<'uint32'>;
  /**
   * Band boundary edges `(x0, y0, x1, y1)`, counter-clockwise around each band region (the band
   * lies on the left). Edges shared by two cells cancel, so only true region boundaries remain:
   * level segments, the raster border and borders against nodata cells. Chains of one band close
   * into the region's rings (see `GPUIsobandRings`). The edge capacity is `edgeBands.length`.
   */
  edges?: GraphDataView<'float32x4'>;
  /** Band index per edge. Its length is the edge capacity. */
  edgeBands?: GraphDataView<'uint32'>;
  /**
   * One-row status words. `count`, `overflow`, `requiredCount` and `vertexCount` go with the
   * geometry group; `edgeCount`, `edgeOverflow` and `edgeRequiredCount` go with the edges.
   */
  status?: GPUIsobandsStatus;
};

/** One-row status words of {@link GPUIsobands}. */
export type GPUIsobandsStatus = {
  /** One-row scalar receiving the triangle count, clamped to the capacity. */
  count?: GraphDataView<'uint32'>;
  /** Optional one-row scalar receiving the unclamped triangle count. */
  requiredCount?: GraphDataView<'uint32'>;
  /** Optional one-row scalar receiving `3 * count`, the vertex count of a non-indexed indirect draw. */
  vertexCount?: GraphDataView<'uint32'>;
  /** One-row scalar receiving the edge count, clamped to the capacity. */
  edgeCount?: GraphDataView<'uint32'>;
  /** Optional one-row scalar receiving the unclamped edge count. */
  edgeRequiredCount?: GraphDataView<'uint32'>;
  /** One-row scalar receiving 1 when more triangles were produced than fit, otherwise 0. */
  overflow?: GraphDataView<'uint32'>;
  /** One-row scalar receiving 1 when more edges were produced than fit, otherwise 0. */
  edgeOverflow?: GraphDataView<'uint32'>;
};

/**
 * Properties for {@link GPUIsobands}.
 *
 * Per-frame (no recompile): the contents of `values`, `validity`, `breaks` and `parameters` (active
 * break count, extent, optional band window). Topology: `grid` size, `breaks.length` (the
 * maximum break count), `noDataValue`, the optional views and the output capacity.
 */
export type GPUIsobandsProps = {
  /** Prefix for generated node IDs. Defaults to `'isobands'`. */
  id?: string;
  /** Raster grid: its width and height (each at least 2) size the kernels. */
  grid: GPURasterMetadata;
  /** Packed row-major float32 raster, at least `width * height` rows. NaN and infinite samples are nodata. */
  values: GraphDataView<'float32'>;
  /** Optional row-major `uint32` validity; zero marks a nodata sample. */
  validity?: GraphDataView<'uint32'>;
  /** Optional finite nodata sentinel compared exactly against sample values. */
  noDataValue?: number;
  /**
   * Break values. The view length is the compile-time maximum break count; the first
   * `breakCount` rows must be strictly ascending. Bands whose bounds are not ascending
   * (`b[k - 1] >= b[k]`) are empty and emit no geometry; with unsorted breaks classes and geometry
   * are deterministic but not meaningful.
   */
  breaks: GraphDataView<'float32'>;
  /** Per-frame float32 view written with `getGPUIsobandsParameterValues`. */
  parameters: GraphDataView<'float32'>;
  /** Caller-owned outputs. */
  output: GPUIsobandsOutput;
};

/**
 * Filled contours (isobands) of a float32 raster with per-frame breaks.
 *
 * Raster mode writes the band class of each sample (the number of breaks `<=` the value, found by
 * binary search) for fragment-side shading. Geometry mode writes triangles: for every marching-squares
 * cell and band `[b[k - 1], b[k])`, the region `Above(b[k - 1]) ∩ Below(b[k])` is built
 * combinatorially from the cell's boundary walk, with level crossings joined through each level's
 * segment partners (saddles by the centre-value rule), so at most two convex pieces per cell and
 * band, fan-triangulated. Every vertex is a sample position or a crossing computed exactly like
 * `GPUIsolines`, so band boundaries coincide with isolines of the same breaks. Cells with a nodata
 * corner emit nothing. Infinite sample values give unspecified geometry.
 *
 * Triangles are ordered by cell, band, piece and fan index (per-cell counts, `GPUScan`, scatter), so
 * output is deterministic. Changing breaks, `breakCount`, the extent or the band window never
 * recompiles the graph.
 *
 * Precision: vertices are `minX + (gridX + 0.5) * cellWidth` in float32 relative to the packer's
 * `localOrigin`. Absolute vertices (the default origin `[0, 0]`) are quantized to the float32 ulp,
 * 0.5 m at a UTM northing of 4.6e6 m (2 m at 2e7 m in EPSG:3857). With `localOrigin` set to the
 * raster or tile corner, a vertex is within about `1.2e-7` of its distance from that corner of the
 * float64 crossing (1.2 cm at 100 km); add the origin back on the CPU or as the layer's coordinate
 * origin.
 */
export class GPUIsobands implements GPUCommandNodeProducer {
  /** Prefix for every node ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUIsobandsProps;
  /** Compile-time maximum break count (`breaks.length`). */
  readonly maximumBreakCount: number;
  /** `(width - 1) * (height - 1)` marching-squares cells. */
  readonly cellCount: number;
  /** Triangle capacity, or 0 without geometry output. */
  readonly triangleCapacity: number;
  /** Boundary edge capacity, or 0 without edge output. */
  readonly edgeCapacity: number;

  constructor(props: GPUIsobandsProps) {
    this.id = props.id ?? 'isobands';
    this.props = props;
    const {id} = this;
    const {output} = props;
    const sampleCount = validateRasterGrid(id, props.grid);
    const {width, height} = props.grid;
    validateRasterAlgebraCount(id, 'grid width', width, 2, 0x7fffffff);
    validateRasterAlgebraCount(id, 'grid height', height, 2, 0x7fffffff);
    this.cellCount = (width - 1) * (height - 1);
    validateRasterAlgebraView(id, 'values', props.values, 'float32', sampleCount);
    validateRasterAlgebraNoData(id, props.noDataValue);
    validateRasterAlgebraView(id, 'validity', props.validity, 'uint32', sampleCount);
    validateRasterAlgebraView(id, 'breaks', props.breaks, 'float32', 1);
    this.maximumBreakCount = props.breaks.length;
    validateRasterAlgebraView(
      id,
      'parameters',
      props.parameters,
      'float32',
      GPU_ISOBANDS_PARAMETER_LENGTH
    );
    validateRasterAlgebraView(id, 'output.bandClasses', output.bandClasses, 'uint32', sampleCount);
    const geometryViews = [
      output.triangles,
      output.triangleBands,
      output.status?.count,
      output.status?.overflow
    ];
    const hasGeometry = geometryViews.some(view => view !== undefined);
    if (hasGeometry && geometryViews.some(view => view === undefined)) {
      throw new Error(
        `${id} geometry output needs triangles, triangleBands, status.count and status.overflow together`
      );
    }
    const edgeViews = [
      output.edges,
      output.edgeBands,
      output.status?.edgeCount,
      output.status?.edgeOverflow
    ];
    const hasEdges = edgeViews.some(view => view !== undefined);
    if (hasEdges && edgeViews.some(view => view === undefined)) {
      throw new Error(
        `${id} edge output needs edges, edgeBands, status.edgeCount and status.edgeOverflow together`
      );
    }
    if (!hasEdges && output.status?.edgeRequiredCount) {
      throw new Error(`${id} status.edgeRequiredCount belongs to the edge output`);
    }
    if (!output.bandClasses && !hasGeometry && !hasEdges) {
      throw new Error(`${id} needs output.bandClasses, the geometry output or the edge output`);
    }
    this.edgeCapacity = output.edgeBands?.length ?? 0;
    if (hasEdges) {
      if (output.edges!.format !== 'float32x4') {
        throw new Error(`${id} output.edges must be float32x4`);
      }
      validateRasterAlgebraView(id, 'output.edgeBands', output.edgeBands, 'uint32', 1);
      if (output.edges!.length < this.edgeCapacity) {
        throw new Error(
          `${id} output.edges must hold edgeBands.length (${this.edgeCapacity}) rows`
        );
      }
      for (const [name, view] of [
        ['status.edgeCount', output.status?.edgeCount],
        ['status.edgeOverflow', output.status?.edgeOverflow],
        ['status.edgeRequiredCount', output.status?.edgeRequiredCount]
      ] as const) {
        validateRasterAlgebraView(id, `output.${name}`, view, 'uint32', 1);
      }
      if (output.edges!.byteStride !== 16 || output.edges!.rowByteLength !== 16) {
        throw new Error(`${id} output.edges must be packed float32x4`);
      }
    }
    if (!hasGeometry && (output.status?.requiredCount || output.status?.vertexCount)) {
      throw new Error(
        `${id} status.requiredCount and status.vertexCount belong to the geometry output`
      );
    }
    this.triangleCapacity = output.triangleBands?.length ?? 0;
    if (hasGeometry) {
      if (output.triangles!.format !== 'float32x2') {
        throw new Error(`${id} output.triangles must be float32x2`);
      }
      validateRasterAlgebraView(id, 'output.triangleBands', output.triangleBands, 'uint32', 1);
      if (output.triangles!.length < 3 * this.triangleCapacity) {
        throw new Error(
          `${id} output.triangles must hold 3 * triangleBands.length (${3 * this.triangleCapacity}) rows`
        );
      }
      for (const [name, view] of [
        ['status.count', output.status?.count],
        ['status.overflow', output.status?.overflow],
        ['status.requiredCount', output.status?.requiredCount],
        ['status.vertexCount', output.status?.vertexCount]
      ] as const) {
        validateRasterAlgebraView(id, `output.${name}`, view, 'uint32', 1);
      }
      const trianglesBuffer = output.triangles!;
      if (trianglesBuffer.byteStride !== 8 || trianglesBuffer.rowByteLength !== 8) {
        throw new Error(`${id} output.triangles must be packed float32x2`);
      }
    }
    const writable = [
      output.bandClasses,
      output.triangleBands,
      output.status?.count,
      output.status?.overflow,
      output.status?.requiredCount,
      output.status?.vertexCount,
      output.edgeBands,
      output.status?.edgeCount,
      output.status?.edgeOverflow,
      output.status?.edgeRequiredCount
    ];
    validateRasterAlgebraAliasing(
      id,
      [...writable, output.triangles, output.edges] as (GraphDataView | undefined)[],
      [props.values, props.validity, props.breaks, props.parameters]
    );
  }

  /** Returns the optional class kernel and, for geometry, count, scan, total, scatter and publish nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, maximumBreakCount, cellCount, triangleCapacity, edgeCapacity} = this;
    const {output} = props;
    const {width, height} = props.grid;
    validateRasterAlgebraGraph(id, graph, [
      props.values,
      props.validity,
      props.breaks,
      props.parameters,
      output.bandClasses,
      output.triangles,
      output.triangleBands,
      output.status?.count,
      output.status?.overflow,
      output.status?.requiredCount,
      output.status?.vertexCount,
      output.edges,
      output.edgeBands,
      output.status?.edgeCount,
      output.status?.edgeOverflow,
      output.status?.edgeRequiredCount
    ]);
    const noEdges = `const EMIT_EDGES: bool = false;
fn writeEdge(index: u32, start: vec2<f32>, end: vec2<f32>, band: u32) {}`;
    const noTriangles = `fn writeTriangle(index: u32, a: vec2<f32>, b: vec2<f32>, c: vec2<f32>, band: u32) {}`;
    const common = `const WIDTH: u32 = ${width}u;
const HEIGHT: u32 = ${height}u;
const MAXIMUM_BREAK_COUNT: u32 = ${maximumBreakCount}u;
const NO_DATA_CLASS: u32 = 0xffffffffu;
${getRasterAlgebraValueWGSL(props.noDataValue)}
${getBreakSearchWGSL('countBreaksBelow', 'breaks')}`;
    const inputs: WGSLKernelBinding[] = [
      {name: 'values', view: props.values, type: 'f32', access: 'read'},
      {name: 'breaks', view: props.breaks, type: 'f32', access: 'read'},
      {name: 'params', view: props.parameters, type: 'f32', access: 'read'}
    ];
    if (props.validity) {
      inputs.push({name: 'validity', view: props.validity, type: 'u32', access: 'read'});
    }
    const nodes: GPUCommandNode<Parameters>[] = [];

    if (output.bandClasses) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-classes`,
          operation: OPERATION,
          variant: 'classes',
          bindings: [
            ...inputs,
            {name: 'classesOut', view: output.bandClasses, type: 'u32', access: 'read_write'}
          ],
          invocationCount: width * height,
          declarations: common,
          body: `let value = values[valuesOffset + index];
  var isValid = isFiniteValue(value) && !isNoDataValue(value);
  ${props.validity ? 'isValid = isValid && validity[validityOffset + index] != 0u;' : ''}
  let breakCountValue = params[paramsOffset];
  let breakCount = min(u32(clamp(select(0.0, breakCountValue, isFiniteValue(breakCountValue)), 0.0, 4294967040.0)), MAXIMUM_BREAK_COUNT);
  classesOut[classesOutOffset + index] = select(NO_DATA_CLASS, countBreaksBelow(0u, breakCount, value, false), isValid);`
        })
      );
    }

    if (
      output.triangles &&
      output.triangleBands &&
      output.status?.count &&
      output.status?.overflow
    ) {
      const cellWGSL = getIsobandsCellWGSL(Boolean(props.validity));
      const cellCounts = createTransientView(graph, `${id}-cell-counts`, 'uint32', cellCount);
      const cellOffsets = createTransientView(graph, `${id}-cell-offsets`, 'uint32', cellCount);
      const total = createTransientView(graph, `${id}-total`, 'uint32', 1);
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-count`,
          operation: OPERATION,
          variant: 'count',
          bindings: [
            ...inputs,
            {name: 'cellCountsOut', view: cellCounts, type: 'u32', access: 'read_write'}
          ],
          invocationCount: cellCount,
          declarations: `${common}
${noTriangles}
${noEdges}
${cellWGSL}`,
          // Every cell writes its count, so the scratch needs no clear.
          body: 'cellCountsOut[cellCountsOutOffset + index] = processCell(index, false, 0u);'
        })
      );
      nodes.push(
        ...new GPUScan({
          id: `${id}-scan`,
          input: cellCounts,
          output: cellOffsets
        }).getCommandNodes(graph)
      );
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-total`,
          operation: OPERATION,
          variant: 'total',
          bindings: [
            {name: 'cellCountsIn', view: cellCounts, type: 'u32', access: 'read'},
            {name: 'cellOffsets', view: cellOffsets, type: 'u32', access: 'read'},
            {name: 'totalOut', view: total, type: 'u32', access: 'read_write'}
          ],
          invocationCount: 1,
          body: `totalOut[totalOutOffset] = cellOffsets[cellOffsetsOffset + ${cellCount - 1}u] + cellCountsIn[cellCountsInOffset + ${cellCount - 1}u];`
        })
      );
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-scatter`,
          operation: OPERATION,
          variant: 'scatter',
          bindings: [
            ...inputs,
            {name: 'cellOffsets', view: cellOffsets, type: 'u32', access: 'read'},
            {name: 'trianglesOut', view: output.triangles, type: 'f32', access: 'read_write'},
            {name: 'bandsOut', view: output.triangleBands, type: 'u32', access: 'read_write'}
          ],
          invocationCount: cellCount,
          declarations: `${common}
const CAPACITY: u32 = ${triangleCapacity}u;
${noEdges}
fn writeTriangle(index: u32, a: vec2<f32>, b: vec2<f32>, c: vec2<f32>, band: u32) {
  if (index >= CAPACITY) {
    return;
  }
  let offset = trianglesOutOffset + index * 6u;
  trianglesOut[offset] = a.x;
  trianglesOut[offset + 1u] = a.y;
  trianglesOut[offset + 2u] = b.x;
  trianglesOut[offset + 3u] = b.y;
  trianglesOut[offset + 4u] = c.x;
  trianglesOut[offset + 5u] = c.y;
  bandsOut[bandsOutOffset + index] = band;
}
${cellWGSL}`,
          body: 'processCell(index, true, cellOffsets[cellOffsetsOffset + index]);'
        })
      );
      nodes.push(
        createPublishNode<Parameters>(graph, {
          id: `${id}-publish`,
          operation: OPERATION,
          requiredCount: total,
          output: {
            ids: output.triangleBands,
            status: {
              count: output.status!.count!,
              overflow: output.status!.overflow!,
              requiredCount: output.status!.requiredCount
            }
          }
        })
      );
      if (output.status?.vertexCount) {
        nodes.push(
          createWGSLKernelNode<Parameters>(graph, {
            id: `${id}-vertex-count`,
            operation: OPERATION,
            variant: 'vertex-count',
            bindings: [
              {name: 'countIn', view: output.status?.count, type: 'u32', access: 'read'},
              {
                name: 'vertexCountOut',
                view: output.status?.vertexCount,
                type: 'u32',
                access: 'read_write'
              }
            ],
            invocationCount: 1,
            body: 'vertexCountOut[vertexCountOutOffset] = 3u * countIn[countInOffset];'
          })
        );
      }
    }

    if (
      output.edges &&
      output.edgeBands &&
      output.status?.edgeCount &&
      output.status?.edgeOverflow
    ) {
      const cellWGSL = getIsobandsCellWGSL(Boolean(props.validity));
      const edgeCounts = createTransientView(graph, `${id}-edge-cell-counts`, 'uint32', cellCount);
      const edgeOffsets = createTransientView(
        graph,
        `${id}-edge-cell-offsets`,
        'uint32',
        cellCount
      );
      const edgeTotal = createTransientView(graph, `${id}-edge-total`, 'uint32', 1);
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-edge-count`,
          operation: OPERATION,
          variant: 'edge-count',
          bindings: [
            ...inputs,
            {name: 'edgeCountsOut', view: edgeCounts, type: 'u32', access: 'read_write'}
          ],
          invocationCount: cellCount,
          declarations: `${common}
${noTriangles}
const EMIT_EDGES: bool = true;
fn writeEdge(index: u32, start: vec2<f32>, end: vec2<f32>, band: u32) {}
${cellWGSL}`,
          body: `processCell(index, false, 0u);
  edgeCountsOut[edgeCountsOutOffset + index] = cellEdgeCount;`
        })
      );
      nodes.push(
        ...new GPUScan({
          id: `${id}-edge-scan`,
          input: edgeCounts,
          output: edgeOffsets
        }).getCommandNodes(graph)
      );
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-edge-total`,
          operation: OPERATION,
          variant: 'edge-total',
          bindings: [
            {name: 'cellCountsIn', view: edgeCounts, type: 'u32', access: 'read'},
            {name: 'cellOffsets', view: edgeOffsets, type: 'u32', access: 'read'},
            {name: 'totalOut', view: edgeTotal, type: 'u32', access: 'read_write'}
          ],
          invocationCount: 1,
          body: `totalOut[totalOutOffset] = cellOffsets[cellOffsetsOffset + ${cellCount - 1}u] + cellCountsIn[cellCountsInOffset + ${cellCount - 1}u];`
        })
      );
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-edge-scatter`,
          operation: OPERATION,
          variant: 'edge-scatter',
          bindings: [
            ...inputs,
            {name: 'cellOffsets', view: edgeOffsets, type: 'u32', access: 'read'},
            {name: 'edgesOut', view: output.edges, type: 'f32', access: 'read_write'},
            {name: 'edgeBandsOut', view: output.edgeBands, type: 'u32', access: 'read_write'}
          ],
          invocationCount: cellCount,
          declarations: `${common}
const EDGE_CAPACITY: u32 = ${edgeCapacity}u;
${noTriangles}
const EMIT_EDGES: bool = true;
fn writeEdge(index: u32, start: vec2<f32>, end: vec2<f32>, band: u32) {
  if (index >= EDGE_CAPACITY) {
    return;
  }
  let offset = edgesOutOffset + index * 4u;
  edgesOut[offset] = start.x;
  edgesOut[offset + 1u] = start.y;
  edgesOut[offset + 2u] = end.x;
  edgesOut[offset + 3u] = end.y;
  edgeBandsOut[edgeBandsOutOffset + index] = band;
}
${cellWGSL}`,
          body: `edgeBase = cellOffsets[cellOffsetsOffset + index];
  processCell(index, true, 0u);`
        })
      );
      nodes.push(
        createPublishNode<Parameters>(graph, {
          id: `${id}-edge-publish`,
          operation: OPERATION,
          requiredCount: edgeTotal,
          output: {
            ids: output.edgeBands,
            status: {
              count: output.status!.edgeCount!,
              overflow: output.status!.edgeOverflow!,
              requiredCount: output.status!.edgeRequiredCount
            }
          }
        })
      );
    }
    return nodes;
  }
}
