// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {
  DEFAULT_DOT_MAXIMUM_ATTEMPTS,
  getDotSamplingNodes,
  validateDotSamplingConfig,
  type DotSamplingConfig,
  type GPUDotDensityMask,
  type GPUDotDensityOutput,
  type GPUDotDensityPolygons
} from './dot-density-sampling';

/**
 * Properties for {@link GPUDotDensity}.
 *
 * Per-frame (no recompile): the contents of `parameters` (seed, dots per unit, mask extent), of
 * `values`, of the polygon buffers and of the mask. Topology (needs a new graph): the feature,
 * category, mask and output sizes, `maximumAttempts`, and which optional outputs are present.
 */
export type GPUDotDensityProps = GPUDotDensityPolygons & {
  /** Prefix for generated node and transient IDs. Defaults to `'dot-density'`. */
  id?: string;
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
  /** Rejection-sampling attempts per dot. Compile-time. Defaults to 32. */
  maximumAttempts?: number;
  /** Optional dasymetric weight raster, for example built-up area or land cover. */
  mask?: GPUDotDensityMask;
  /** Caller-owned outputs. The capacity is `output.positions.length`. */
  output: GPUDotDensityOutput;
};

/**
 * Dasymetric dot-density map: places `value * dotsPerUnit` random dots inside each polygon, per
 * category, without flicker when the dot value changes with zoom.
 *
 * A slot is one (feature, category) pair. It draws `n = ceil(value * dotsPerUnit - u)` dots, where
 * `u` is a per-slot uniform from `philox(counter = (slot, 0, 0, 0), key = (seed, 1))`: that is
 * `floor(x)` dots plus one more with probability `frac(x)`, so the expected count is exact, and
 * `n` never decreases when `dotsPerUnit` grows. Dot `j` of a slot is placed by rejection sampling
 * in the feature's bounding box with candidates `philox((slot, j, attempt, 0), (seed, 2))`, which
 * depend on neither `dotsPerUnit` nor any other slot. So zooming in only adds dots and every
 * existing dot keeps its exact position: the dots of a coarser dot value are a stable prefix.
 * With a mask, a candidate is also kept only when word z of the same draw is below the mask weight
 * of its cell, which concentrates dots where the weight is high (dasymetric mapping).
 *
 * Dots are ordered by slot (feature, then category) and by rank within a slot, with offsets from an
 * exclusive prefix scan. A dot whose `maximumAttempts` candidates all miss (a thin polygon in a
 * large bounding box, or a mask that is zero over the polygon) keeps its slot with a NaN position
 * and is counted in `failedCount`.
 *
 * Containment is an even-odd crossing test over all rings of the feature, which equals
 * multipolygon containment for valid geometry (holes inside shells, parts not overlapping).
 * Points exactly on an edge have probability zero.
 *
 * Determinism: integer counts, an integer scan and counter-based random numbers; the only atomic
 * is the order-independent u32 failure count. Results are bitwise reproducible.
 */
export class GPUDotDensity implements GPUCommandNodeProducer {
  /** Prefix for graph node and transient IDs. */
  readonly id: string;
  /** Number of polygon features. */
  readonly featureCount: number;
  /** Number of categories per feature. */
  readonly categoryCount: number;
  private readonly config: DotSamplingConfig;

  constructor(props: GPUDotDensityProps) {
    const id = props.id ?? 'dot-density';
    this.config = {
      id,
      operation: 'GPUDotDensity',
      polygonPositions: props.polygonPositions,
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

/**
 * Properties for {@link GPURandomPointsInPolygon}.
 *
 * Per-frame (no recompile): the contents of `parameters` (seed, mask extent), of `counts`, of the
 * polygon buffers and of the mask. Topology: as {@link GPUDotDensityProps}.
 */
export type GPURandomPointsInPolygonProps = GPUDotDensityPolygons & {
  /** Prefix for generated node and transient IDs. Defaults to `'random-points-in-polygon'`. */
  id?: string;
  /** Number of points to place in each feature, one row per feature. */
  counts: GraphDataView<'uint32'>;
  /** Per-frame uint32 parameters written with `getGPUDotDensityParameterValues`. */
  parameters: GraphDataView<'uint32'>;
  /** Rejection-sampling attempts per point. Compile-time. Defaults to 32. */
  maximumAttempts?: number;
  /** Optional weight raster: points are kept with probability equal to the cell weight. */
  mask?: GPUDotDensityMask;
  /** Caller-owned outputs. `categories` is not used. */
  output: Omit<GPUDotDensityOutput, 'categories'>;
};

/**
 * Places `counts[feature]` uniformly random points inside each polygon feature (holes excluded),
 * optionally thinned by a weight raster.
 *
 * Point `j` of feature `f` is the first of `maximumAttempts` candidates
 * `philox((f, j, attempt, 0), (seed, 2))`, mapped to the feature's bounding box, that passes the
 * even-odd containment test, so raising a count only appends points and existing points never
 * move. Output order, capacity, failure handling and determinism are as in {@link GPUDotDensity}.
 */
export class GPURandomPointsInPolygon implements GPUCommandNodeProducer {
  /** Prefix for graph node and transient IDs. */
  readonly id: string;
  /** Number of polygon features. */
  readonly featureCount: number;
  private readonly config: DotSamplingConfig;

  constructor(props: GPURandomPointsInPolygonProps) {
    const id = props.id ?? 'random-points-in-polygon';
    this.config = {
      id,
      operation: 'GPURandomPointsInPolygon',
      polygonPositions: props.polygonPositions,
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
    return getDotSamplingNodes(graph, this.config, this.featureCount);
  }
}
