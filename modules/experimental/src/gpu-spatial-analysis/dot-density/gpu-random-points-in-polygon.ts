// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {
  GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT,
  resolveGPUSpatialContext,
  type GPUPolygonGeometryFields,
  type GPUSpatialContext
} from '../../analysis-core/contracts/index';
import {
  DEFAULT_DOT_MAXIMUM_ATTEMPTS,
  getDotSamplingNodes,
  validateDotSamplingConfig,
  type DotSamplingConfig,
  type GPUDotDensityMask,
  type GPUDotDensityOutput
} from './dot-density-sampling';
import {validatePolygonTriangulationScratch} from '../polygon-triangulation/gpu-polygon-triangulation';

// Uniform polygon sampling weights triangles by area.
const SPATIAL_CONTEXT_SUPPORT = Object.freeze({
  ...GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT,
  measure: 'area'
});

/**
 * Properties for {@link GPURandomPointsInPolygon}.
 *
 * Per-frame (no recompile): the contents of `parameters` (seed, mask extent), of `counts`, of the
 * polygon buffers and of the mask. Topology: as {@link GPUDotDensityProps}.
 */
export type GPURandomPointsInPolygonProps = GPUPolygonGeometryFields & {
  /** Prefix for generated node and transient IDs. Defaults to `'random-points-in-polygon'`. */
  id?: string;
  /**
   * Coordinate and metric contract of the polygon positions. Planar/native only (omitted means
   * planar/native). Random placement is uniform in coordinate space, so longitude/latitude degrees
   * would be uniform in degrees squared, not in area: project to an equal-area planar frame first.
   * Other contexts throw at construction.
   */
  spatialContext?: GPUSpatialContext;
  /** Requested points per feature. Zero-area features get no slots and report failed requests. */
  counts: GraphDataView<'uint32'>;
  /** Per-frame uint32 parameters written with `getGPUDotDensityParameterValues`. */
  parameters: GraphDataView<'uint32'>;
  /**
   * Mask-rejection attempts per point. Compile-time. Defaults to 32. Only used with a `mask`;
   * without one every point is placed by its first (exact, area-weighted) draw.
   */
  maximumAttempts?: number;
  /** Optional weight raster: points are kept with probability equal to the cell weight. */
  mask?: GPUDotDensityMask;
  /** Caller-owned outputs. `categories` is not used. */
  output: Omit<GPUDotDensityOutput, 'categories'>;
};

/**
 * Places `counts[feature]` uniformly random points inside each polygon feature (holes excluded),
 * optionally thinned by a weight raster. Empty polygons and degenerate rings are skipped on the
 * GPU; zero-area features produce zero points and report requests in `output.status.failedCount`.
 *
 * Point `j` of feature `f` is drawn from `philox((f, j, attempt, 0), (seed, 7))` exactly as in
 * {@link GPUDotDensity}: an area-weighted triangle of the GPU triangulation, then a uniform point in
 * it, so raising a count only appends points and existing points never move. Output order,
 * capacity, failure handling and determinism are as in {@link GPUDotDensity}.
 */
export class GPURandomPointsInPolygon implements GPUCommandNodeProducer {
  /** Prefix for graph node and transient IDs. */
  readonly id: string;
  /** Resolved coordinate and metric contract (planar/native). */
  readonly spatialContext: GPUSpatialContext;
  /** Number of polygon features. */
  readonly featureCount: number;
  private readonly config: DotSamplingConfig;

  constructor(props: GPURandomPointsInPolygonProps) {
    const id = props.id ?? 'random-points-in-polygon';
    this.spatialContext = resolveGPUSpatialContext(
      id,
      props.spatialContext,
      SPATIAL_CONTEXT_SUPPORT
    );
    this.config = {
      id,
      operation: 'GPURandomPointsInPolygon',
      positions: props.positions,
      featureOffsets: props.featureOffsets,
      polygonOffsets: props.polygonOffsets,
      ringOffsets: props.ringOffsets,
      source: {kind: 'counts', counts: props.counts},
      categoryCount: 1,
      parameters: props.parameters,
      maximumAttempts: props.maximumAttempts ?? DEFAULT_DOT_MAXIMUM_ATTEMPTS,
      mask: props.mask,
      output: props.output
    };
    this.featureCount = validateDotSamplingConfig(this.config);
    this.id = id;
  }

  /** Returns the feature, count, scan, locate, sample and publish nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    // The internal triangulation scratch must fit one storage binding of the graph's device.
    const {positions, ringOffsets, polygonOffsets} = this.config;
    validatePolygonTriangulationScratch(
      graph.device,
      this.id,
      positions.length,
      ringOffsets.length - 1,
      polygonOffsets.length - 1
    );
    return getDotSamplingNodes(graph, this.config, this.featureCount);
  }
}
