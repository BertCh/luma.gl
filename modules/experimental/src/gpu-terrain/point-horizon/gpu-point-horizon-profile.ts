// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import type {GPURasterBand, GPURasterMetadata} from '../../gpu-raster/index';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {validateGraphViewsBelongToGraph} from '../../utils/gpu-contributor-utils';
import {
  getRasterBandViews,
  validateRasterBandBelongsToGraph,
  validateRasterBuffersDistinct,
  validateRasterParameters
} from '../../gpu-raster/raster-band-utils';
import {
  createPointHorizonChunkStarts,
  createPointHorizonSource,
  getPointHorizonMarchWGSL,
  PRECISION_WGSL,
  resolvePointHorizonModel,
  validatePointHorizonRows,
  type GPUPointHorizonModelOptions,
  type ResolvedPointHorizonModel
} from './point-horizon-march';
import {
  GPU_POINT_HORIZON_PROFILE_PARAMETER_LENGTH,
  POINT_HORIZON_PROFILE_PROJECTION_SLOT
} from './point-horizon-parameters';

/** Default number of rays per compute node of {@link GPUPointHorizonProfile}. */
export const GPU_POINT_HORIZON_RAYS_PER_DISPATCH = 65536;

/**
 * Properties for {@link GPUPointHorizonProfile}.
 *
 * Topology (baked into WGSL, see {@link GPUPointHorizonModelOptions}): grid size, projection,
 * row direction, height reference, traversal, azimuth division and the distance lattice. Per-frame:
 * `parameters` (curvature, distance cap, and the grid cell size or world size), `observers` and
 * the terrain contents.
 *
 * Projection and cell-size model: `'planar'` rasters are in projected meters with the cell size
 * of the grid; `'web-mercator'` rasters are windows of EPSG:3857 meters whose world size and top
 * edge the packer derives from the grid, with great-circle rays on a sphere of radius 6371008.8 m.
 * See {@link GPUPointHorizonProjection} and the settings type in `point-horizon-parameters`.
 *
 * Curvature convention: terrain drops by `c * d^2` with `c = (1 - k) / (2R)` (see
 * `getGPUTerrainCurvatureCoefficient`; mt-image default `k = 0.13`, GDAL `-cc 0.85714` is `k = 1/7`).
 */
export type GPUPointHorizonProfileProps = GPUPointHorizonModelOptions & {
  /** Prefix for node and transient IDs. Defaults to `'point-horizon-profile'`. */
  id?: string;
  /** Raster grid: its width and height (at least 2) size the kernels. */
  grid: GPURasterMetadata;
  /** Terrain band (elevation), buffer or texture. Calibration, no-data and validity are honored. */
  elevation: GPURasterBand;
  /**
   * Observer rows `[column, row, height, 0]` in pixel-center index space (fractional allowed).
   * With `heightReference: 'ground'` (default) the eye is the bilinear ground plus `height`, with
   * `'absolute'` it is `height`. An observer outside the grid, or on an invalid ground sample in
   * ground mode, yields NaN in every output.
   */
  observers: GraphDataView<'float32x4'>;
  /** Per-frame parameters, see `getGPUPointHorizonProfileParameterValues`. At least 8 float32. */
  parameters: GraphDataView<'float32'>;
  /** Caller-owned outputs; at least one is required. */
  output: GPUPointHorizonProfileOutput;
  /** Rays per compute node. Defaults to 65536. */
  raysPerDispatch?: number;
};

/** Caller-owned outputs of {@link GPUPointHorizonProfile}, one value per ray. */
export type GPUPointHorizonProfileOutput = {
  /**
   * Optional `observers.length * azimuthSpan` tangents `tBest = max((h - h_eye) / d - c d)`;
   * `-3.4028234663852886e38` when no valid sample was seen, NaN for an invalid observer.
   */
  tangent?: GraphDataView<'float32'>;
  /** Optional skyline elevation angles in degrees, `atan(tBest)`; -90 when no sample, NaN invalid. */
  skylineAngle?: GraphDataView<'float32'>;
  /** Optional distance of the skyline sample in meters; 0 when no sample, NaN invalid. */
  distance?: GraphDataView<'float32'>;
  /**
   * Optional debug counter: the number of lattice samples whose bilinear height was evaluated per
   * ray (`'march'` evaluates every sample up to the first out-of-grid one, `'pyramid'` omits the
   * skipped ones). Spends a ninth binding when all three outputs are present, which is rejected:
   * omit one output to use it.
   */
  samples?: GraphDataView<'uint32'>;
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
    this.model = resolvePointHorizonModel(id, props.grid, props);
    const {output} = props;
    if (!output.tangent && !output.skylineAngle && !output.distance) {
      throw new Error(`${id} requires at least one output`);
    }
    validateRasterParameters(id, props.parameters, GPU_POINT_HORIZON_PROFILE_PARAMETER_LENGTH);
    validatePointHorizonRows(id, 'observers', props.observers, 'float32x4');
    if (props.observers.length < 1) {
      throw new Error(`${id} requires at least one observer`);
    }
    const rayCount = props.observers.length * this.model.azimuthSpan;
    for (const [name, view] of [
      ['tangent', output.tangent],
      ['skylineAngle', output.skylineAngle],
      ['distance', output.distance]
    ] as const) {
      if (view) {
        validatePointHorizonRows(id, `output.${name}`, view, 'float32');
        if (view.length !== rayCount) {
          throw new Error(`${id} output.${name} must contain observers * azimuthSpan values`);
        }
      }
    }
    const raysPerDispatch = props.raysPerDispatch ?? GPU_POINT_HORIZON_RAYS_PER_DISPATCH;
    if (!Number.isInteger(raysPerDispatch) || raysPerDispatch < 1) {
      throw new Error(`${id} raysPerDispatch must be a positive integer`);
    }
    validateRasterBuffersDistinct(
      id,
      [output.tangent, output.skylineAngle, output.distance, output.samples],
      [...getRasterBandViews(props.elevation), props.parameters, props.observers]
    );
  }

  /** Returns canonical elevation, pyramid level and per-chunk march nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, model} = this;
    const {output} = props;
    validateRasterBandBelongsToGraph(id, graph, props.elevation, []);
    validateGraphViewsBelongToGraph(id, graph, [
      props.pyramid?.combined,
      props.parameters,
      props.observers,
      output.tangent,
      output.skylineAngle,
      output.distance,
      output.samples
    ]);
    const source = createPointHorizonSource(graph, id, model, props.elevation);
    const nodes: GPUCommandNode<Parameters>[] = [...source.nodes];
    const bindings: WGSLKernelBinding[] = [
      ...source.bindings,
      {name: 'settings', view: props.parameters, type: 'f32', access: 'read'},
      {name: 'observers', view: props.observers, type: 'f32', access: 'read'}
    ];
    if (output.tangent) {
      bindings.push({
        name: 'tangentOutput',
        view: output.tangent,
        type: 'f32',
        access: 'read_write'
      });
    }
    // With both angle outputs the march writes only the tangent and a per-ray pass derives the
    // skyline angle from it with the same pinned `atanAccurate` (bit-identical), so the march stays
    // within 8 storage buffers with every output on the pyramid traversal (core limit).
    const deriveSkylineAngle = Boolean(output.tangent && output.skylineAngle);
    if (output.skylineAngle && !deriveSkylineAngle) {
      bindings.push({
        name: 'skylineAngleOutput',
        view: output.skylineAngle,
        type: 'f32',
        access: 'read_write'
      });
    }
    if (output.distance) {
      bindings.push({
        name: 'distanceOutput',
        view: output.distance,
        type: 'f32',
        access: 'read_write'
      });
    }
    if (output.samples) {
      bindings.push({
        name: 'samplesOutput',
        view: output.samples,
        type: 'u32',
        access: 'read_write'
      });
    }
    // The chunk start word is the eighth binding at most.
    if (bindings.length > 7) {
      throw new Error(`${id} needs more than 8 storage bindings`);
    }
    const march = getPointHorizonMarchWGSL(model, POINT_HORIZON_PROFILE_PROJECTION_SLOT);
    const rayCount = props.observers.length * model.azimuthSpan;
    const raysPerDispatch = props.raysPerDispatch ?? GPU_POINT_HORIZON_RAYS_PER_DISPATCH;
    const writes = (value: string) =>
      [
        output.tangent
          ? `tangentOutput[tangentOutputOffset + ray] = ${value === 'nan' ? 'nanValue()' : 'select(-BIG, result.t, result.has)'};`
          : '',
        output.skylineAngle && !deriveSkylineAngle
          ? `skylineAngleOutput[skylineAngleOutputOffset + ray] = ${value === 'nan' ? 'nanValue()' : 'select(-90.0, atanAccurate(result.t) * DEGREES, result.has)'};`
          : '',
        output.distance
          ? `distanceOutput[distanceOutputOffset + ray] = ${value === 'nan' ? 'nanValue()' : 'select(0.0, result.d, result.has)'};`
          : '',
        output.samples
          ? `samplesOutput[samplesOutputOffset + ray] = ${value === 'nan' ? '0u' : 'result.evaluated'};`
          : ''
      ].join('\n  ');
    const chunks = createPointHorizonChunkStarts<Parameters>(graph, {
      id,
      operation: 'GPUPointHorizonProfile',
      itemCount: rayCount,
      maximumChunkSize: raysPerDispatch
    });
    nodes.push(chunks.writeNode);
    for (let chunk = 0; chunk < chunks.chunkCount; chunk++) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-march-${chunk}`,
          operation: 'GPUPointHorizonProfile',
          variant: model.traversal,
          bindings: [...bindings, chunks.getChunkStart(chunk)],
          invocationCount: chunks.chunkSize,
          workgroupSize: 64,
          declarations: `const RAY_COUNT: u32 = ${rayCount}u;
${march}`,
          body: `let ray = chunkStart[chunkStartOffset] + index;
  if (ray >= RAY_COUNT) { return; }
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
    if (output.tangent && output.skylineAngle && deriveSkylineAngle) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-skyline-angle`,
          operation: 'GPUPointHorizonProfile',
          variant: 'skyline-angle',
          bindings: [
            {name: 'settings', view: props.parameters, type: 'f32', access: 'read'},
            {name: 'tangent', view: output.tangent, type: 'f32', access: 'read'},
            {
              name: 'skylineAngleOutput',
              view: output.skylineAngle,
              type: 'f32',
              access: 'read_write'
            }
          ],
          invocationCount: rayCount,
          declarations: `const RAY_COUNT: u32 = ${rayCount}u;
${PRECISION_WGSL}`,
          // The march wrote nanValue() (invalid eye, copied through), -BIG (no horizon) or the tangent.
          body: `if (index >= RAY_COUNT) { return; }
  let t = tangent[tangentOffset + index];
  var angle = t;
  if ((bitcast<u32>(t) & 0x7fffffffu) <= 0x7f800000u) {
    angle = select(-90.0, atanAccurate(t) * DEGREES, t != -BIG);
  }
  skylineAngleOutput[skylineAngleOutputOffset + index] = angle;`
        })
      );
    }
    return nodes;
  }
}
