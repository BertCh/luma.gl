// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView,
  type GraphTextureView
} from '@luma.gl/gpgpu/gpu-core';
import {
  GPURasterBufferToTexture,
  GPURasterGradient,
  type GPURasterBand,
  type GPURasterBorderMode
} from '../../gpu-raster/index';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {captureGraphCommandNodes, validateGraphViewsBelongToGraph} from '../../utils/gpu-contributor-utils';
import {
  getTerrainBandViews,
  getTerrainElevationNodes,
  TERRAIN_WGSL_HELPERS,
  validateTerrainBandBelongsToGraph,
  validateTerrainBuffersDistinct,
  validateTerrainGrid,
  validateTerrainSettings,
  validateTerrainTexture
} from './terrain-analysis-utils';

/** How the shade kernel converts settings cell sizes into ground meters per pixel for each row. */
export type GPUTerrainCellSizeMode = 'uniform' | 'web-mercator' | 'geographic';

/** Slope output unit. */
export type GPUTerrainSlopeUnits = 'degrees' | 'percent';

/** Number of float32 values read from `GPUTerrainDerivativesProps.settings`. */
export const GPU_TERRAIN_DERIVATIVES_PARAMETER_LENGTH = 8;

/** CPU-side description packed by {@link getGPUTerrainDerivativesParameterValues}. */
export type GPUTerrainDerivativesSettings = {
  /** `[x, y]` cell size: meters (uniform), equatorial Web Mercator meters, or degrees (geographic). */
  cellSize: readonly [number, number];
  /** Elevation multiplier applied to derivatives. Defaults to 1. */
  zFactor?: number;
  /** Direction the light comes from, degrees clockwise from north. Defaults to 315. */
  azimuthDegrees?: number;
  /** Light elevation above the horizon in degrees. Defaults to 45. */
  altitudeDegrees?: number;
  /** Top edge of row 0: normalized Web Mercator y in `[0, 1]` or latitude degrees. Defaults to 0. */
  northEdge?: number;
  /** Bottom edge of the last row, same units as `northEdge`. Defaults to 0. */
  southEdge?: number;
};

/**
 * Packs settings into the 8-float layout read by {@link GPUTerrainDerivatives}:
 * `[cellSizeX, cellSizeY, zFactor, azimuthDegrees, altitudeDegrees, northEdge, southEdge, 0]`.
 */
export function getGPUTerrainDerivativesParameterValues(
  settings: GPUTerrainDerivativesSettings,
  target: Float32Array = new Float32Array(GPU_TERRAIN_DERIVATIVES_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_TERRAIN_DERIVATIVES_PARAMETER_LENGTH) {
    throw new Error('Terrain derivative settings target must hold 8 values');
  }
  target.set([
    settings.cellSize[0],
    settings.cellSize[1],
    settings.zFactor ?? 1,
    settings.azimuthDegrees ?? 315,
    settings.altitudeDegrees ?? 45,
    settings.northEdge ?? 0,
    settings.southEdge ?? 0,
    0
  ]);
  return target;
}

/**
 * Properties for {@link GPUTerrainDerivatives}.
 *
 * Topology: grid size, elevation format and calibration, which outputs exist, `cellSizeMode`,
 * `slopeUnits`, `rowDirection`, and `borderMode`. Per-frame: `settings` and elevation contents.
 */
export type GPUTerrainDerivativesProps = {
  /** Prefix for node and transient IDs. Defaults to `'terrain-derivatives'`. */
  id?: string;
  /** Grid width in pixels. */
  width: number;
  /** Grid height in pixels. */
  height: number;
  /** Elevation band, buffer or texture; scale, offset, nodata, and validity are honored. */
  elevation: GPURasterBand;
  /** Per-frame settings with at least 8 float32 values, see {@link getGPUTerrainDerivativesParameterValues}. */
  settings: GraphDataView<'float32'>;
  /** Optional slope per pixel in degrees `[0, 90)` or percent. */
  slope?: GraphDataView<'float32'>;
  /** Optional downslope direction per pixel, degrees clockwise from north; -1 for flat cells. */
  aspect?: GraphDataView<'float32'>;
  /** Optional hillshade per pixel in `[0, 1]`. */
  hillshade?: GraphDataView<'float32'>;
  /** Optional per-pixel 1 where every output is valid, else 0. */
  validity?: GraphDataView<'uint32'>;
  /** Optional storage texture (`r32float` or `rgba32float`, channel 0) receiving the hillshade. */
  hillshadeTexture?: GraphTextureView<'r32float' | 'rgba32float'>;
  /** Cell size interpretation. Defaults to `'uniform'`. */
  cellSizeMode?: GPUTerrainCellSizeMode;
  /** Slope unit. Defaults to `'degrees'`. */
  slopeUnits?: GPUTerrainSlopeUnits;
  /** Direction in which the row index increases. Defaults to `'south'` (north-up rasters). */
  rowDirection?: 'south' | 'north';
  /** Sobel border treatment forwarded to `GPURasterGradient`. Defaults to `'clamp'`. */
  borderMode?: GPURasterBorderMode;
};

/**
 * Computes slope, aspect, and hillshade for an elevation tile with Horn's 3x3 method.
 *
 * Composes two Sobel `GPURasterGradient` passes and one fused shade kernel. Cell size, z factor,
 * sun position, and latitude band are read from `settings` every encoding. Columns increase east
 * and, with the default `rowDirection`, rows increase south. Invalid pixels (an invalid center or
 * 3x3 neighbor, or invalid settings) receive NaN and validity 0. For seamless tiles, pass a tile
 * with a one-pixel halo; the recipe satisfies the `GPURasterHaloStage` contract.
 */
export class GPUTerrainDerivatives implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUTerrainDerivativesProps;
  /** Receptive field in pixels (`GPURasterHaloStage` contract). */
  readonly requiredHalo = 1;

  constructor(props: GPUTerrainDerivativesProps) {
    this.id = props.id ?? 'terrain-derivatives';
    this.props = props;
    const {id} = this;
    const pixelCount = validateTerrainGrid(id, props.width, props.height);
    if (
      !props.slope &&
      !props.aspect &&
      !props.hillshade &&
      !props.validity &&
      !props.hillshadeTexture
    ) {
      throw new Error(`${id} requires at least one output`);
    }
    for (const [name, view] of [
      ['slope', props.slope],
      ['aspect', props.aspect],
      ['hillshade', props.hillshade]
    ] as const) {
      if (view) {
        validatePackedView(view, ['float32'], `${id} ${name}`);
        if (view.length !== pixelCount) {
          throw new Error(`${id} ${name} must contain one value per pixel`);
        }
      }
    }
    if (props.validity) {
      validatePackedUint32View(props.validity, `${id} validity`);
      if (props.validity.length !== pixelCount) {
        throw new Error(`${id} validity must contain one value per pixel`);
      }
    }
    validateTerrainSettings(id, props.settings, GPU_TERRAIN_DERIVATIVES_PARAMETER_LENGTH);
    validateTerrainTexture(
      id,
      'hillshadeTexture',
      props.hillshadeTexture,
      ['r32float', 'rgba32float'],
      props.width,
      props.height
    );
    if (!['uniform', 'web-mercator', 'geographic'].includes(props.cellSizeMode ?? 'uniform')) {
      throw new Error(`${id} cellSizeMode must be uniform, web-mercator, or geographic`);
    }
    if (!['degrees', 'percent'].includes(props.slopeUnits ?? 'degrees')) {
      throw new Error(`${id} slopeUnits must be degrees or percent`);
    }
    if (!['south', 'north'].includes(props.rowDirection ?? 'south')) {
      throw new Error(`${id} rowDirection must be south or north`);
    }
    validateTerrainBuffersDistinct(
      id,
      [props.slope, props.aspect, props.hillshade, props.validity],
      [...getTerrainBandViews(props.elevation), props.settings]
    );
  }

  /** Returns optional elevation gather, two Sobel passes, shade, and optional texture nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {width, height} = props;
    validateTerrainBandBelongsToGraph(id, graph, props.elevation, [props.hillshadeTexture]);
    validateGraphViewsBelongToGraph(id, graph, [
      props.settings,
      props.slope,
      props.aspect,
      props.hillshade,
      props.validity
    ]);
    const pixelCount = width * height;
    const borderMode = props.borderMode ?? 'clamp';
    const source = getTerrainElevationNodes(graph, id, props.elevation, width, height, false);
    const nodes: GPUCommandNode<Parameters>[] = [...source.nodes];
    const gradients = (['x', 'y'] as const).map(direction => {
      const values = createTransientView(
        graph,
        `${id}-gradient-${direction}-values`,
        'float32',
        pixelCount
      );
      const validity = createTransientView(
        graph,
        `${id}-gradient-${direction}-validity`,
        'uint32',
        pixelCount
      );
      nodes.push(
        ...captureGraphCommandNodes(graph, () =>
          new GPURasterGradient({
            id: `${id}-gradient-${direction}`,
            width,
            height,
            input: source.band,
            output: values,
            outputValidity: validity,
            operator: 'sobel',
            direction,
            borderMode,
            scale: 1
          }).addToGraph(graph)
        )
      );
      return {values, validity};
    });
    const hillshadeTarget =
      props.hillshade ??
      (props.hillshadeTexture
        ? createTransientView(graph, `${id}-hillshade`, 'float32', pixelCount)
        : undefined);
    nodes.push(
      getShadeNode(graph, {
        id: `${id}-shade`,
        width,
        height,
        gradientX: gradients[0].values,
        gradientY: gradients[1].values,
        gradientValidity: gradients[0].validity,
        settings: props.settings,
        slope: props.slope,
        aspect: props.aspect,
        hillshade: hillshadeTarget,
        validity: props.validity,
        cellSizeMode: props.cellSizeMode ?? 'uniform',
        slopeUnits: props.slopeUnits ?? 'degrees',
        rowDirection: props.rowDirection ?? 'south'
      })
    );
    if (props.hillshadeTexture && hillshadeTarget) {
      const texture = props.hillshadeTexture;
      nodes.push(
        ...captureGraphCommandNodes(graph, () =>
          new GPURasterBufferToTexture({
            id: `${id}-hillshade-texture`,
            input: {
              id: `${id}-hillshade-band`,
              format: 'float32',
              storage: {kind: 'buffer', values: hillshadeTarget}
            },
            output: texture,
            channel: 0
          }).addToGraph(graph)
        )
      );
    }
    return nodes;
  }
}

/** Builds the fused per-pixel slope, aspect, hillshade, and validity kernel. */
function getShadeNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    width: number;
    height: number;
    gradientX: GraphDataView<'float32'>;
    gradientY: GraphDataView<'float32'>;
    gradientValidity: GraphDataView<'uint32'>;
    settings: GraphDataView<'float32'>;
    slope?: GraphDataView<'float32'>;
    aspect?: GraphDataView<'float32'>;
    hillshade?: GraphDataView<'float32'>;
    validity?: GraphDataView<'uint32'>;
    cellSizeMode: GPUTerrainCellSizeMode;
    slopeUnits: GPUTerrainSlopeUnits;
    rowDirection: 'south' | 'north';
  }
): GPUCommandNode<Parameters> {
  const bindings: WGSLKernelBinding[] = [
    {name: 'gradientX', view: props.gradientX, type: 'f32', access: 'read'},
    {name: 'gradientY', view: props.gradientY, type: 'f32', access: 'read'},
    {name: 'gradientValidity', view: props.gradientValidity, type: 'u32', access: 'read'},
    {name: 'settings', view: props.settings, type: 'f32', access: 'read'}
  ];
  if (props.slope)
    bindings.push({name: 'slopeValues', view: props.slope, type: 'f32', access: 'read_write'});
  if (props.aspect) {
    bindings.push({name: 'aspectValues', view: props.aspect, type: 'f32', access: 'read_write'});
  }
  if (props.hillshade) {
    bindings.push({
      name: 'hillshadeValues',
      view: props.hillshade,
      type: 'f32',
      access: 'read_write'
    });
  }
  if (props.validity) {
    bindings.push({
      name: 'validityValues',
      view: props.validity,
      type: 'u32',
      access: 'read_write'
    });
  }
  const groundCellSource =
    props.cellSizeMode === 'uniform'
      ? 'return cellSize;'
      : `let rowFraction = (f32(row) + 0.5) / f32(HEIGHT);
  let edge = mix(settings[settingsOffset + 5u], settings[settingsOffset + 6u], rowFraction);
  ${
    props.cellSizeMode === 'web-mercator'
      ? '// cos(latitude) = 1 / cosh(PI * (1 - 2y)) for normalized Web Mercator y.\n  return cellSize / cosh(PI * (1.0 - 2.0 * edge));'
      : 'return cellSize * METERS_PER_DEGREE * vec2<f32>(cos(edge * DEGREES_TO_RADIANS), 1.0);'
  }`;
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: 'GPUTerrainDerivatives',
    variant: `shade-${props.cellSizeMode}`,
    bindings,
    invocationCount: props.width * props.height,
    declarations: `const WIDTH: u32 = ${props.width}u;
const HEIGHT: u32 = ${props.height}u;
const PI: f32 = 3.141592653589793;
const DEGREES_TO_RADIANS: f32 = 0.017453292519943295;
const RADIANS_TO_DEGREES: f32 = 57.29577951308232;
const METERS_PER_DEGREE: f32 = 111319.49079327357;
const ROW_NORTH_SIGN: f32 = ${props.rowDirection === 'south' ? '-1.0' : '1.0'};
${TERRAIN_WGSL_HELPERS}
fn getGroundCellSize(row: u32) -> vec2<f32> {
  let cellSize = vec2<f32>(settings[settingsOffset], settings[settingsOffset + 1u]);
  ${groundCellSource}
}`,
    body: `let row = index / WIDTH;
  let rawX = gradientX[gradientXOffset + index];
  let rawY = gradientY[gradientYOffset + index];
  let groundCell = getGroundCellSize(row);
  let zFactor = settings[settingsOffset + 2u];
  // Sobel responds 8 to a unit-per-pixel ramp, so Horn's derivative is raw / (8 * cell).
  let eastGradient = zFactor * rawX / (8.0 * groundCell.x);
  let northGradient = ROW_NORTH_SIGN * zFactor * rawY / (8.0 * groundCell.y);
  let larger = max(abs(eastGradient), abs(northGradient));
  let smaller = min(abs(eastGradient), abs(northGradient));
  var gradientMagnitude = 0.0;
  if (larger > 0.0) {
    let ratio = smaller / larger;
    gradientMagnitude = larger * sqrt(1.0 + ratio * ratio);
  }
  let isValid = gradientValidity[gradientValidityOffset + index] != 0u &&
    isFiniteValue(rawX) && isFiniteValue(rawY) && groundCell.x > 0.0 && groundCell.y > 0.0 &&
    isFiniteValue(eastGradient) && isFiniteValue(northGradient) && isFiniteValue(gradientMagnitude);
  let invalidValue = bitcast<f32>(0x7fc00000u | (index & 0u));
  ${
    props.slope
      ? `let slope = ${props.slopeUnits === 'degrees' ? 'atan(gradientMagnitude) * RADIANS_TO_DEGREES' : 'gradientMagnitude * 100.0'};
  slopeValues[slopeValuesOffset + index] = select(invalidValue, slope, isValid);`
      : ''
  }
  ${
    props.aspect
      ? `var aspect = -1.0;
  if (gradientMagnitude > 0.0) {
    aspect = atan2(-eastGradient, -northGradient) * RADIANS_TO_DEGREES;
    if (aspect < 0.0) { aspect += 360.0; }
    if (aspect >= 360.0) { aspect -= 360.0; }
  }
  aspectValues[aspectValuesOffset + index] = select(invalidValue, aspect, isValid);`
      : ''
  }
  ${
    props.hillshade
      ? `let azimuth = settings[settingsOffset + 3u] * DEGREES_TO_RADIANS;
  let altitude = settings[settingsOffset + 4u] * DEGREES_TO_RADIANS;
  let light = vec3<f32>(sin(azimuth) * cos(altitude), cos(azimuth) * cos(altitude), sin(altitude));
  let normal = normalize(vec3<f32>(-eastGradient, -northGradient, 1.0));
  let shade = max(dot(normal, light), 0.0);
  hillshadeValues[hillshadeValuesOffset + index] = select(invalidValue, shade, isValid);`
      : ''
  }
  ${props.validity ? 'validityValues[validityValuesOffset + index] = select(0u, 1u, isValid);' : ''}`
  });
}
