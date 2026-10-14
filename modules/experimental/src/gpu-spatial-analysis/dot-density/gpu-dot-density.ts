// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import type {GPUPolygonGeometryFields} from '../../analysis-core/contracts/index';
import {
  resolveGPUSpatialContext,
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

/**
 * Properties for {@link GPUDotDensity}.
 *
 * Per-frame (no recompile): the contents of `parameters` (seed, dots per unit, mask extent), of
 * `values`, of the polygon buffers and of the mask. Topology (needs a new graph): the feature,
 * category, mask and output sizes, `maximumAttempts`, and which optional outputs are present.
 */
export type GPUDotDensityProps = GPUPolygonGeometryFields & {
  /** Prefix for generated node and transient IDs. Defaults to `'dot-density'`. */
  id?: string;
  /**
   * Coordinate and metric contract of the polygon positions. Planar/native only (omitted means
   * planar/native). Random placement is uniform in coordinate space, so longitude/latitude degrees
   * would be uniform in degrees squared, not in area: project to an equal-area planar frame first.
   * Other contexts throw at construction.
   */
  spatialContext?: GPUSpatialContext;
  /**
   * Values to draw as dots, `featureCount * categoryCount` rows, feature-major
   * (`row = feature * categoryCount + category`), for example population by group.
   * NaN and non-positive values draw no dots.
   */
  values: GraphDataView<'float32'>;
  /** Number of categories per feature. Compile-time. Defaults to 1. */
  categoryCount?: number;
  /** Per-frame uint32 parameters written with `getGPUDotDensityParameterValues`. */
  parameters: GraphDataView<'uint32'>;
  /**
   * Mask-rejection attempts per dot. Compile-time. Defaults to 32. Only used with a `mask`: without
   * one every dot is placed by its first (exact, area-weighted) draw and this value is ignored.
   */
  maximumAttempts?: number;
  /** Optional dasymetric weight raster, for example built-up area or land cover. */
  mask?: GPUDotDensityMask;
  /**
   * Caller-owned outputs. The capacity is `output.positions.length`, at most
   * {@link getGPUDotDensityMaximumCapacity} (8388608 on a default device).
   */
  output: GPUDotDensityOutput;
};

/**
 * Dasymetric dot-density map: places `value * dotsPerUnit` random dots inside each polygon, per
 * category, without flicker when the dot value changes with zoom.
 *
 * A slot is one (feature, category) pair. It draws `n = ceil(value * dotsPerUnit - u)` dots, where
 * `u` is a per-slot uniform from `philox(counter = (slot, 0, 0, 0), key = (seed, 6))`: that is
 * `floor(x)` dots plus one more with probability `frac(x)`, so the expected count is exact, and
 * `n` never decreases when `dotsPerUnit` grows. Dot `j` of a slot is placed exactly: every polygon
 * is triangulated on the GPU (`GPUPolygonTriangulation`; the first ring of a polygon is the shell
 * and later rings are holes), a triangle is chosen with probability proportional to its area by a
 * binary search of the feature's cumulative triangle areas with
 * `philox((slot, j, attempt, 0), (seed, 7)).x`, and the point is uniform in that triangle
 * (`r1 = sqrt(u1)`, `P = (1 - r1) A + r1 (1 - u2) B + r1 u2 C`). The draws depend on neither
 * `dotsPerUnit` nor any other slot, so zooming in only adds dots and every existing dot keeps its
 * exact position: the dots of a coarser dot value are a stable prefix. Placement never fails for a
 * valid polygon, however thin or concave it is relative to its bounding box. Triangle areas are
 * quantized to a 32-bit fixed point per feature (selection bias below one part in 1e9 per triangle).
 *
 * With a mask, a candidate is also kept only when word w of the same draw is below the mask weight
 * of its cell, which concentrates dots where the weight is high (dasymetric mapping). A masked dot
 * is retried up to `maximumAttempts` times (the `attempt` counter), so it fails with probability
 * `(1 - w)^maximumAttempts` for a mean weight `w` over the feature.
 *
 * Dots are ordered by slot (feature, then category) and by rank within a slot, with offsets from an
 * exclusive prefix scan. A dot that cannot be placed keeps its slot with a NaN position and is
 * counted in `output.failedCount`: that happens when one of the feature's polygons failed to
 * triangulate (self-intersecting rings, a shell with fewer than three vertices, or the
 * triangulation work budget exceeded: the whole feature is then reported, not just the bad
 * polygon), when the feature has zero or non-finite area, and when all mask attempts miss. A
 * feature with no requested dots reports nothing.
 *
 * Coordinates: positions are sampled uniformly in coordinate space, so the context must be
 * planar/native (`spatialContext`); longitude/latitude throws because dots would be uniform in
 * degrees squared rather than in area (project to an equal-area frame first).
 *
 * Containment follows GeoArrow ring roles, not an even-odd test: ring 0 of each polygon is the
 * shell and later rings are holes. Points exactly on an edge have probability zero.
 *
 * Determinism: integer counts, an integer scan and counter-based random numbers; the only atomic
 * is the order-independent u32 failure count. Results are bitwise reproducible.
 */
export class GPUDotDensity implements GPUCommandNodeProducer {
  /** Prefix for graph node and transient IDs. */
  readonly id: string;
  /** Resolved coordinate and metric contract (planar/native). */
  readonly spatialContext: GPUSpatialContext;
  /** Number of polygon features. */
  readonly featureCount: number;
  /** Number of categories per feature. */
  readonly categoryCount: number;
  private readonly config: DotSamplingConfig;

  constructor(props: GPUDotDensityProps) {
    const id = props.id ?? 'dot-density';
    this.spatialContext = resolveGPUSpatialContext(id, props.spatialContext);
    this.config = {
      id,
      operation: 'GPUDotDensity',
      positions: props.positions,
      featureOffsets: props.featureOffsets,
      polygonOffsets: props.polygonOffsets,
      ringOffsets: props.ringOffsets,
      source: {kind: 'values', values: props.values},
      categoryCount: props.categoryCount ?? 1,
      parameters: props.parameters,
      maximumAttempts: props.maximumAttempts ?? DEFAULT_DOT_MAXIMUM_ATTEMPTS,
      mask: props.mask,
      output: props.output
    };
    this.featureCount = validateDotSamplingConfig(this.config);
    this.id = id;
    this.categoryCount = this.config.categoryCount;
  }

  /** Returns the feature, count, scan, locate, sample and publish nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    return getDotSamplingNodes(graph, this.config, this.featureCount);
  }
}
