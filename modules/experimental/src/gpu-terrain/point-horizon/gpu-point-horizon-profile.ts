// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import type {GPURasterBand} from '../../gpu-raster/index';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {validateGraphViewsBelongToGraph} from '../../utils/gpu-contributor-utils';
import {
  getTerrainBandViews,
  validateTerrainBandBelongsToGraph,
  validateTerrainBuffersDistinct,
  validateTerrainSettings
} from '../terrain-analysis/terrain-analysis-utils';
import {
  createPointHorizonSource,
  getPointHorizonMarchWGSL,
  resolvePointHorizonModel,
  validatePointHorizonRows,
  type GPUPointHorizonModelOptions,
  type ResolvedPointHorizonModel
} from './point-horizon-march';
import {GPU_POINT_HORIZON_PARAMETER_LENGTH} from './point-horizon-parameters';

/** Default number of rays per compute node of {@link GPUPointHorizonProfile}. */
export const GPU_POINT_HORIZON_RAYS_PER_DISPATCH = 65536;

/**
 * Properties for {@link GPUPointHorizonProfile}.
 *
 * Topology (baked into WGSL, see {@link GPUPointHorizonModelOptions}): grid size, projection,
 * row direction, height reference, traversal, azimuth division and the distance lattice. Per-frame:
 * `settings` (curvature, cell size or world size, distance cap), `observers` and the terrain
 * contents.
 *
 * Projection and cell-size model: `'planar'` rasters are in projected meters with a per-frame
 * `cellSize = [x, y]`; `'web-mercator'` rasters are Web Mercator pixel windows with a per-frame
 * `worldPixelSize` and origin row and great-circle rays on a sphere of radius 6371008.8 m. See
 * {@link GPUPointHorizonProjection} and the settings types in `point-horizon-parameters`.
 *
 * Curvature convention: terrain drops by `c * d^2` with `c = (1 - k) / (2R)` (see
 * `getGPUTerrainCurvatureCoefficient`; mt-image default `k = 0.13`, GDAL `-cc 0.85714` is `k = 1/7`).
 */
export type GPUPointHorizonProfileProps = GPUPointHorizonModelOptions & {
  /** Prefix for node and transient IDs. Defaults to `'point-horizon-profile'`. */
  id?: string;
  /** Grid width in pixels (at least 2). */
  width: number;
  /** Grid height in pixels (at least 2). */
  height: number;
  /** Terrain band (elevation), buffer or texture. Calibration, no-data and validity are honored. */
  terrain: GPURasterBand;
  /**
   * Observer rows `[column, row, height, 0]` in pixel-center index space (fractional allowed).
   * With `heightReference: 'ground'` (default) the eye is the bilinear ground plus `height`, with
   * `'absolute'` it is `height`. An observer outside the grid, or on an invalid ground sample in
   * ground mode, yields NaN in every output.
   */
  observers: GraphDataView<'float32x4'>;
  /** Per-frame settings, see `getGPUPointHorizonParameterValues`. At least 8 float32. */
  settings: GraphDataView<'float32'>;
  /**
   * Optional `observers.length * azimuthSpan` tangents `tBest = max((h - h_eye) / d - c d)`;
   * `-3.4028234663852886e38` when no valid sample was seen, NaN for an invalid observer.
   */
  tangent?: GraphDataView<'float32'>;
  /** Optional skyline elevation angles in degrees, `atan(tBest)`; -90 when no sample, NaN invalid. */
  elevation?: GraphDataView<'float32'>;
  /** Optional distance of the skyline sample in meters; 0 when no sample, NaN invalid. */
  distance?: GraphDataView<'float32'>;
  /** Rays per compute node. Defaults to 65536. */
  raysPerDispatch?: number;
};

/**
 * Computes the skyline (horizon) profile seen from many observers over one terrain window.
 *
 * Each ray marches an exact power-of-two distance lattice (see
 * {@link GPUPointHorizonDistanceLatticeOptions}) with bilinear samples and keeps the maximum
 * apparent tangent. The `'pyramid'` traversal (default) skips samples a min-max mip pyramid proves
 * cannot raise the maximum, and is bit-identical to the exhaustive `'march'` traversal (Tevs, Ihrke
 * and Seidel 2008 "Maximum mipmaps"; Dick et al. 2009; the exact-skip proof is in
 * `point-horizon-march.ts`).
 *
 * Precision ladder (from mt-image's GPU horizon, f32 on the GPU against f64 on the CPU): the eye is
 * an integer pixel plus an f32 fraction so sample positions keep sub-milli-pixel precision near the
 * eye; terrain heights are interpolated relative to the eye as an f32 hi/lo pair guarded by an
 * opaque zero so Metal fast math cannot re-associate them; Web Mercator breakpoints are offsets
 * from cancellation-free difference terms (`sin(phi2) - sin(phi1)`, `1 - cos(D)`) with small-angle
 * series for `atan` and `atanh`; azimuth sine and cosine use exact integer octant reduction; the
 * final `atan(tBest)` uses a reduced-argument series. No builtin `sin`, `cos` or `atan` is used
 * where accuracy matters (their WGSL accuracy is 2^-11 absolute / about 4096 ULP).
 *
 * Output index: `observer * azimuthSpan + a`, azimuth `(firstAzimuth + a) * 360 / azimuthCount`
 * degrees clockwise from north. One compute node per chunk of at most `raysPerDispatch` rays.
 */
export class GPUPointHorizonProfile implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUPointHorizonProfileProps;
  /** Resolved model: lattice, segments and pyramid layout. @internal */
  readonly model: ResolvedPointHorizonModel;

  constructor(props: GPUPointHorizonProfileProps) {
    this.id = props.id ?? 'point-horizon-profile';
    this.props = props;
    const {id} = this;
    this.model = resolvePointHorizonModel(id, props.width, props.height, props);
    if (!props.tangent && !props.elevation && !props.distance) {
      throw new Error(`${id} requires at least one output`);
    }
    validateTerrainSettings(id, props.settings, GPU_POINT_HORIZON_PARAMETER_LENGTH);
    validatePointHorizonRows(id, 'observers', props.observers, 'float32x4');
    if (props.observers.length < 1) {
      throw new Error(`${id} requires at least one observer`);
    }
    const rayCount = props.observers.length * this.model.azimuthSpan;
    for (const [name, view] of [
      ['tangent', props.tangent],
      ['elevation', props.elevation],
      ['distance', props.distance]
    ] as const) {
      if (view) {
        validatePointHorizonRows(id, name, view, 'float32');
        if (view.length !== rayCount) {
          throw new Error(`${id} ${name} must contain observers * azimuthSpan values`);
        }
      }
    }
    const raysPerDispatch = props.raysPerDispatch ?? GPU_POINT_HORIZON_RAYS_PER_DISPATCH;
    if (!Number.isInteger(raysPerDispatch) || raysPerDispatch < 1) {
      throw new Error(`${id} raysPerDispatch must be a positive integer`);
    }
    validateTerrainBuffersDistinct(
      id,
      [props.tangent, props.elevation, props.distance],
      [...getTerrainBandViews(props.terrain), props.settings, props.observers]
    );
  }

  /** Returns canonical elevation, pyramid level and per-chunk march nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, model} = this;
    validateTerrainBandBelongsToGraph(id, graph, props.terrain, []);
    validateGraphViewsBelongToGraph(id, graph, [
      props.settings,
      props.observers,
      props.tangent,
      props.elevation,
      props.distance
    ]);
    const source = createPointHorizonSource(graph, id, model, props.terrain);
    const nodes: GPUCommandNode<Parameters>[] = [...source.nodes];
    const bindings: WGSLKernelBinding[] = [
      ...source.bindings,
      {name: 'settings', view: props.settings, type: 'f32', access: 'read'},
      {name: 'observers', view: props.observers, type: 'f32', access: 'read'}
    ];
    if (props.tangent) {
      bindings.push({
        name: 'tangentOutput',
        view: props.tangent,
        type: 'f32',
        access: 'read_write'
      });
    }
    if (props.elevation) {
      bindings.push({
        name: 'elevationOutput',
        view: props.elevation,
        type: 'f32',
        access: 'read_write'
      });
    }
    if (props.distance) {
      bindings.push({
        name: 'distanceOutput',
        view: props.distance,
        type: 'f32',
        access: 'read_write'
      });
    }
    if (bindings.length > 8) {
      throw new Error(`${id} needs more than 8 storage bindings`);
    }
    const march = getPointHorizonMarchWGSL(model);
    const rayCount = props.observers.length * model.azimuthSpan;
    const raysPerDispatch = props.raysPerDispatch ?? GPU_POINT_HORIZON_RAYS_PER_DISPATCH;
    const writes = (value: string) =>
      [
        props.tangent
          ? `tangentOutput[tangentOutputOffset + ray] = ${value === 'nan' ? 'nanValue()' : 'select(-BIG, result.t, result.has)'};`
          : '',
        props.elevation
          ? `elevationOutput[elevationOutputOffset + ray] = ${value === 'nan' ? 'nanValue()' : 'select(-90.0, atanAccurate(result.t) * DEGREES, result.has)'};`
          : '',
        props.distance
          ? `distanceOutput[distanceOutputOffset + ray] = ${value === 'nan' ? 'nanValue()' : 'select(0.0, result.d, result.has)'};`
          : ''
      ].join('\n  ');
    for (
      let chunk = 0, rayOffset = 0;
      rayOffset < rayCount;
      chunk++, rayOffset += raysPerDispatch
    ) {
      const count = Math.min(raysPerDispatch, rayCount - rayOffset);
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-march-${chunk}`,
          operation: 'GPUPointHorizonProfile',
          variant: model.traversal,
          bindings,
          invocationCount: count,
          workgroupSize: 64,
          declarations: `const RAY_OFFSET: u32 = ${rayOffset}u;
${march}`,
          body: `let ray = RAY_OFFSET + index;
  let observerIndex = ray / AZIMUTH_SPAN;
  let azimuthOffset = ray - observerIndex * AZIMUTH_SPAN;
  let observerBase = observersOffset + observerIndex * 4u;
  let eye = makeEye(observers[observerBase], observers[observerBase + 1u], observers[observerBase + 2u]);
  if (!eye.valid) {
    ${writes('nan')}
    return;
  }
  let direction = azimuthSinCos(FIRST_AZIMUTH + azimuthOffset, AZIMUTH_COUNT);
  let result = marchRay(eye, direction.x, direction.y, 0.0, false);
  ${writes('result')}`
        })
      );
    }
    return nodes;
  }
}
