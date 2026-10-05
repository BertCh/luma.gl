// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validatePackedUint32View,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPURasterBand} from '../../gpu-raster/index';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {validateGraphViewsBelongToGraph} from '../../utils/gpu-contributor-utils';
import {GPU_TERRAIN_VISIBILITY} from '../terrain-analysis/gpu-terrain-viewshed';
import {
  getTerrainBandViews,
  validateTerrainBandBelongsToGraph,
  validateTerrainBuffersDistinct,
  validateTerrainSettings
} from '../terrain-analysis/terrain-analysis-utils';
import {GPU_POINT_HORIZON_RAYS_PER_DISPATCH} from './gpu-point-horizon-profile';
import {
  createPointHorizonSource,
  getPointHorizonMarchWGSL,
  resolvePointHorizonModel,
  validatePointHorizonRows,
  type GPUPointHorizonModelOptions,
  type ResolvedPointHorizonModel
} from './point-horizon-march';
import {GPU_POINT_HORIZON_VISIBILITY_PARAMETER_LENGTH} from './point-horizon-parameters';

/**
 * Properties for {@link GPUPointHorizonVisibility}.
 *
 * Model options (projection, cell-size model, curvature convention, lattice) are those of
 * {@link GPUPointHorizonModelOptions}; `azimuthCount`, `firstAzimuth` and `azimuthSpan` are
 * ignored because the ray direction is the exact bearing to each target.
 */
export type GPUPointHorizonVisibilityProps = GPUPointHorizonModelOptions & {
  /** Prefix for node and transient IDs. Defaults to `'point-horizon-visibility'`. */
  id?: string;
  /** Grid width in pixels (at least 2). */
  width: number;
  /** Grid height in pixels (at least 2). */
  height: number;
  /** Terrain band (elevation). */
  terrain: GPURasterBand;
  /** Observer rows `[column, row, height, 0]`, see `GPUPointHorizonProfileProps.observers`. */
  observers: GraphDataView<'float32x4'>;
  /**
   * Target rows `[column, row, height, observerIndex]` in pixel-center index space. The target
   * elevation is the bilinear ground plus `height`; `observerIndex` is the integer-valued index of
   * the observer row it is tested from. An invalid index, an observer or target outside the grid,
   * or an invalid ground sample yields `noData`.
   */
  targets: GraphDataView<'float32x4'>;
  /** Per-frame settings, see `getGPUPointHorizonVisibilityParameterValues`. At least 12 float32. */
  settings: GraphDataView<'float32'>;
  /** One `GPU_TERRAIN_VISIBILITY` code per target (hidden, visible, outOfRange, noData, marginal). */
  visibility: GraphDataView<'uint32'>;
  /**
   * Optional rows `[targetElevationDegrees, occluderElevationDegrees, skylineElevationDegrees,
   * onSkyline ? 1 : 0]`; the occluder and skyline are -90 when no sample was seen and the row is NaN
   * for `noData`, `outOfRange` and coincident points.
   */
  details?: GraphDataView<'float32x4'>;
  /** Targets per compute node. Defaults to 65536. */
  targetsPerDispatch?: number;
};

/**
 * Classifies targets (peaks, towers, points of interest) as hidden, marginal or visible from
 * observers, and flags those on the skyline, following mt-image's `classifyPeak`.
 *
 * Per target the contributor computes the distance and bearing from the observer without cancellation
 * (planar: metres offsets; Web Mercator: a spherical inverse written in haversine form with
 * `Δm = -2πΔrow/worldPixelSize` and `Δλ = 2πΔcolumn/worldPixelSize`, series for small `sinh`,
 * `sin` and `asin`), marches the full ray along that bearing with the same march as
 * `GPUPointHorizonProfile`, and records the running maximum tangent `tQ` before
 * `q = dP - max(targetIgnoreDistance, targetIgnoreFraction * dP)` and the skyline `tSky`. With
 * `αP = atan(tP)`, `αOcc = atan(tQ)`, `αSky = atan(tSky)` (accurate series, degrees) and
 * `tol = toleranceDegrees + (sigmaZ / dP) * 180 / π`:
 * `hidden` if `αP < αOcc - tol`; `marginal` if `|αP - αOcc| < tol`; otherwise `visible`. `onSkyline`
 * is `(visible or marginal) and αP >= αSky - skylineToleranceDegrees`.
 */
export class GPUPointHorizonVisibility implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUPointHorizonVisibilityProps;
  /** Resolved model: lattice, segments and pyramid layout. @internal */
  readonly model: ResolvedPointHorizonModel;

  constructor(props: GPUPointHorizonVisibilityProps) {
    this.id = props.id ?? 'point-horizon-visibility';
    this.props = props;
    const {id} = this;
    this.model = resolvePointHorizonModel(id, props.width, props.height, {
      ...props,
      azimuthCount: 720,
      firstAzimuth: 0,
      azimuthSpan: 720
    });
    validateTerrainSettings(id, props.settings, GPU_POINT_HORIZON_VISIBILITY_PARAMETER_LENGTH);
    validatePointHorizonRows(id, 'observers', props.observers, 'float32x4');
    validatePointHorizonRows(id, 'targets', props.targets, 'float32x4');
    if (props.observers.length < 1) {
      throw new Error(`${id} requires at least one observer`);
    }
    validatePackedUint32View(props.visibility, `${id} visibility`);
    if (props.visibility.length !== props.targets.length) {
      throw new Error(`${id} visibility must contain one value per target`);
    }
    if (props.details) {
      validatePointHorizonRows(id, 'details', props.details, 'float32x4');
      if (props.details.length !== props.targets.length) {
        throw new Error(`${id} details must contain one row per target`);
      }
    }
    const targetsPerDispatch = props.targetsPerDispatch ?? GPU_POINT_HORIZON_RAYS_PER_DISPATCH;
    if (!Number.isInteger(targetsPerDispatch) || targetsPerDispatch < 1) {
      throw new Error(`${id} targetsPerDispatch must be a positive integer`);
    }
    validateTerrainBuffersDistinct(
      id,
      [props.visibility, props.details],
      [...getTerrainBandViews(props.terrain), props.settings, props.observers, props.targets]
    );
  }

  /** Returns canonical elevation, pyramid level and per-chunk classification nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, model} = this;
    validateTerrainBandBelongsToGraph(id, graph, props.terrain, []);
    validateGraphViewsBelongToGraph(id, graph, [
      props.settings,
      props.observers,
      props.targets,
      props.visibility,
      props.details
    ]);
    const source = createPointHorizonSource(graph, id, model, props.terrain);
    const nodes: GPUCommandNode<Parameters>[] = [...source.nodes];
    const bindings: WGSLKernelBinding[] = [
      ...source.bindings,
      {name: 'settings', view: props.settings, type: 'f32', access: 'read'},
      {name: 'observers', view: props.observers, type: 'f32', access: 'read'},
      {name: 'targets', view: props.targets, type: 'f32', access: 'read'},
      {name: 'visibilityOutput', view: props.visibility, type: 'u32', access: 'read_write'}
    ];
    if (props.details) {
      bindings.push({
        name: 'detailsOutput',
        view: props.details,
        type: 'f32',
        access: 'read_write'
      });
    }
    if (bindings.length > 8) {
      throw new Error(`${id} needs more than 8 storage bindings`);
    }
    const mercator = model.projection === 'web-mercator';
    const geometry = mercator
      ? `let worldPixelSize = settings[settingsOffset + 1u];
  let halfLambda = PI * deltaColumn / worldPixelSize;
  let halfM = -PI * deltaRow / worldPixelSize;
  let m2 = eye.m + 2.0 * halfM;
  let meanM = eye.m + halfM;
  let cosP2 = 1.0 / coshS(m2);
  let sinHalfLambda = sinAny(halfLambda);
  let cosHalfLambda = cosAny(halfLambda);
  let sinhHalfM = sinhS(halfM);
  let haversine = eye.cosP * cosP2 * (sinhHalfM * sinhHalfM + sinHalfLambda * sinHalfLambda);
  let distance = 2.0 * EARTH_RADIUS * asinS(sqrt(min(haversine, 1.0)));
  let sinDeltaPhi = 2.0 * coshS(meanM) * sinhHalfM * eye.cosP * cosP2;
  let north = sinDeltaPhi + eye.sinP * cosP2 * 2.0 * sinHalfLambda * sinHalfLambda;
  let east = 2.0 * sinHalfLambda * cosHalfLambda * cosP2;
  let rho = sqrt(north * north + east * east);
  let sinA = east / max(rho, 1.0e-30);
  let cosA = north / max(rho, 1.0e-30);`
      : `let deltaX = deltaColumn * settings[settingsOffset + 1u];
  let deltaY = deltaRow * settings[settingsOffset + 2u];
  let distance = sqrt(deltaX * deltaX + deltaY * deltaY);
  let sinA = deltaX / max(distance, 1.0e-30);
  let cosA = ${model.rowDirection === 'south' ? '-' : ''}deltaY / max(distance, 1.0e-30);`;
    const march = getPointHorizonMarchWGSL(model);
    const {targets} = props;
    const chunkSize = props.targetsPerDispatch ?? GPU_POINT_HORIZON_RAYS_PER_DISPATCH;
    const detailsWrite = props.details
      ? `let detailsBase = detailsOutputOffset + (TARGET_OFFSET + index) * 4u;
  detailsOutput[detailsBase] = result.details.x;
  detailsOutput[detailsBase + 1u] = result.details.y;
  detailsOutput[detailsBase + 2u] = result.details.z;
  detailsOutput[detailsBase + 3u] = result.details.w;`
      : '';
    for (let chunk = 0, offset = 0; offset < targets.length; chunk++, offset += chunkSize) {
      const count = Math.min(chunkSize, targets.length - offset);
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-classify-${chunk}`,
          operation: 'GPUPointHorizonVisibility',
          variant: model.traversal,
          bindings,
          invocationCount: count,
          workgroupSize: 64,
          declarations: `const TARGET_OFFSET: u32 = ${offset}u;
const OBSERVER_COUNT: u32 = ${props.observers.length}u;
const HIDDEN: u32 = ${GPU_TERRAIN_VISIBILITY.hidden}u;
const VISIBLE: u32 = ${GPU_TERRAIN_VISIBILITY.visible}u;
const OUT_OF_RANGE: u32 = ${GPU_TERRAIN_VISIBILITY.outOfRange}u;
const NO_DATA: u32 = ${GPU_TERRAIN_VISIBILITY.noData}u;
const MARGINAL: u32 = ${GPU_TERRAIN_VISIBILITY.marginal}u;
${march}
fn sinAny(x: f32) -> f32 { if (abs(x) > 1.6) { return sin(x); } return sinSeries(x); }
fn cosAny(x: f32) -> f32 { if (abs(x) > 1.6) { return cos(x); } return cosSeries(x); }

struct Classification { code: u32, details: vec4<f32> };

fn classifyTarget(index: u32) -> Classification {
  let notANumber = vec4<f32>(nanValue());
  let base = targetsOffset + (TARGET_OFFSET + index) * 4u;
  let targetColumn = targets[base];
  let targetRow = targets[base + 1u];
  let targetHeight = targets[base + 2u];
  let observerValue = targets[base + 3u];
  if (!(observerValue >= 0.0 && observerValue < f32(OBSERVER_COUNT))) {
    return Classification(NO_DATA, notANumber);
  }
  let observerBase = observersOffset + u32(observerValue) * 4u;
  let observerColumn = observers[observerBase];
  let observerRow = observers[observerBase + 1u];
  let eye = makeEye(observerColumn, observerRow, observers[observerBase + 2u]);
  if (!eye.valid || !isInsideGrid(targetColumn, targetRow) || !isFiniteValue(targetHeight)) {
    return Classification(NO_DATA, notANumber);
  }
  let targetSample = sampleRelative(targetColumn, targetRow, eye.hi, eye.lo);
  if (targetSample.y == 0.0) {
    return Classification(NO_DATA, notANumber);
  }
  let relativeHeight = targetSample.x + targetHeight;
  let deltaColumn = targetColumn - observerColumn;
  let deltaRow = targetRow - observerRow;
  ${geometry}
  let maximumFrame = settings[settingsOffset + 3u];
  let maximumDistance = select(MAXIMUM_DISTANCE, min(maximumFrame, MAXIMUM_DISTANCE), maximumFrame > 0.0);
  if (distance > maximumDistance) {
    return Classification(OUT_OF_RANGE, notANumber);
  }
  if (!(distance > 0.0)) {
    return Classification(VISIBLE, notANumber);
  }
  let curvature = settings[settingsOffset];
  let stop = distance - max(settings[settingsOffset + 8u], settings[settingsOffset + 9u] * distance);
  let ray = marchRay(eye, sinA, cosA, stop, stop > 0.0);
  let targetTangent = relativeHeight / distance - opaque(distance * curvature);
  let targetAngle = atanAccurate(targetTangent) * DEGREES;
  let occluderAngle = select(-90.0, atanAccurate(ray.tQ) * DEGREES, ray.tQ > -3.0e38);
  let skylineAngle = select(-90.0, atanAccurate(ray.t) * DEGREES, ray.has);
  let tolerance = settings[settingsOffset + 5u] + (settings[settingsOffset + 6u] / distance) * DEGREES;
  var code = VISIBLE;
  if (targetAngle < occluderAngle - tolerance) {
    code = HIDDEN;
  } else if (abs(targetAngle - occluderAngle) < tolerance) {
    code = MARGINAL;
  }
  let onSkyline = code != HIDDEN && targetAngle >= skylineAngle - settings[settingsOffset + 7u];
  return Classification(code, vec4<f32>(targetAngle, occluderAngle, skylineAngle, select(0.0, 1.0, onSkyline)));
}`,
          body: `let result = classifyTarget(index);
  visibilityOutput[visibilityOutputOffset + TARGET_OFFSET + index] = result.code;
  ${detailsWrite}`
        })
      );
    }
    return nodes;
  }
}
