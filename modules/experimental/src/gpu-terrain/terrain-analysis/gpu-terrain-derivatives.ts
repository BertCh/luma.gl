// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView,
  type GraphTextureView
} from '@luma.gl/gpgpu/gpu-core';
import {
  GPURasterBufferToTexture,
  GPURasterGradient,
  type GPURasterBand,
  type GPURasterBorderMode,
  type GPURasterMetadata
} from '../../gpu-raster/index';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {
  captureGraphCommandNodes,
  validateGraphViewsBelongToGraph,
  validatePackedView
} from '../../utils/gpu-contributor-utils';
import {
  getRasterBandViews,
  getRasterBufferBandNodes,
  RASTER_WGSL_HELPERS,
  validateParameterTarget,
  validateRasterBandBelongsToGraph,
  validateRasterBuffersDistinct,
  validateRasterGrid,
  validateRasterGridAxisAligned,
  validateRasterParameters,
  validateRasterOutputTexture
} from '../../gpu-raster/raster-band-utils';
import {
  getRasterGroundCellSizeWGSL,
  getRasterRowNorthSignWGSL,
  type GPURasterCellSizeMode,
  type GPURasterRowDirection,
  RASTER_DEGREES_TO_RADIANS_WGSL,
  validateRasterCellSizeMode,
  validateRasterRowDirection,
  writeRasterGridParameterValues
} from '../../analysis-core/raster-grid';

/** Slope output unit. */
export type GPUTerrainSlopeUnits = 'degrees' | 'percent';

/** Number of float32 values read from `GPUTerrainDerivativesProps.parameters`. */
export const GPU_TERRAIN_DERIVATIVES_PARAMETER_LENGTH = 8;

/** Cell size and row-edge slots of the derivative parameter layout. */
const TERRAIN_DERIVATIVES_GRID_SLOTS = {cellSizeIndex: 0, northEdgeIndex: 5} as const;

/**
 * CPU-side description packed by {@link getGPUTerrainDerivativesParameterValues}. Cell size and
 * latitude band come from the grid passed next to it.
 */
export type GPUTerrainDerivativesSettings = {
  /** Elevation multiplier applied to derivatives. Defaults to 1. */
  zFactor?: number;
  /** Direction the light comes from, degrees clockwise from north. Defaults to 315. */
  azimuthDegrees?: number;
  /** Light elevation above the horizon in degrees. Defaults to 45. */
  altitudeDegrees?: number;
};

/**
 * Packs settings into the 8-float layout read by {@link GPUTerrainDerivatives}:
 * `[cellSizeX, cellSizeY, zFactor, azimuthDegrees, altitudeDegrees, northEdge, southEdge, 0]`,
 * with the cell size and the row-edge world y taken from `grid`.
 */
export function getGPUTerrainDerivativesParameterValues(
  settings: GPUTerrainDerivativesSettings,
  grid: GPURasterMetadata,
  target: Float32Array = new Float32Array(GPU_TERRAIN_DERIVATIVES_PARAMETER_LENGTH)
): Float32Array {
  validateParameterTarget(
    'getGPUTerrainDerivativesParameterValues',
    target,
    GPU_TERRAIN_DERIVATIVES_PARAMETER_LENGTH
  );
  target[2] = settings.zFactor ?? 1;
  target[3] = settings.azimuthDegrees ?? 315;
  target[4] = settings.altitudeDegrees ?? 45;
  target[7] = 0;
  writeRasterGridParameterValues(target, grid, TERRAIN_DERIVATIVES_GRID_SLOTS);
  return target;
}

/** Caller-owned outputs of {@link GPUTerrainDerivatives}; at least one is required. */
export type GPUTerrainDerivativesOutput = {
  /** Slope per pixel in degrees `[0, 90)` or percent. */
  slope?: GraphDataView<'float32'>;
  /** Downslope direction per pixel, degrees clockwise from north; -1 for flat cells. */
  aspect?: GraphDataView<'float32'>;
  /** Hillshade per pixel in `[0, 1]`. */
  hillshade?: GraphDataView<'float32'>;
  /** Per-pixel 1 where every output is valid, else 0. */
  outputValidity?: GraphDataView<'uint32'>;
  /** Storage texture (`r32float` or `rgba32float`, channel 0) receiving the hillshade. */
  hillshadeTexture?: GraphTextureView<'r32float' | 'rgba32float'>;
};

/**
 * Properties for {@link GPUTerrainDerivatives}.
 *
 * Topology: grid size, elevation format and calibration, which outputs exist, `cellSizeMode`,
 * `slopeUnits`, `rowDirection`, and `borderMode`. Per-frame: `parameters` (including the grid
 * cell size and latitude band) and elevation contents.
 */
export type GPUTerrainDerivativesProps = {
  /** Prefix for node and transient IDs. Defaults to `'terrain-derivatives'`. */
  id?: string;
  /** Raster grid: its width and height size the kernels. */
  grid: GPURasterMetadata;
  /** Elevation band, buffer or texture; scale, offset, nodata, and validity are honored. */
  elevation: GPURasterBand;
  /** Per-frame parameters with at least 8 float32 values, see {@link getGPUTerrainDerivativesParameterValues}. */
  parameters: GraphDataView<'float32'>;
  /** Caller-owned outputs. */
  output: GPUTerrainDerivativesOutput;
  /** Cell size interpretation. Defaults to `'uniform'`. */
  cellSizeMode?: GPURasterCellSizeMode;
  /** Slope unit. Defaults to `'degrees'`. */
  slopeUnits?: GPUTerrainSlopeUnits;
  /** Direction in which the row index increases. Defaults to `'south'` (north-up rasters). */
  rowDirection?: GPURasterRowDirection;
  /** Sobel border treatment forwarded to `GPURasterGradient`. Defaults to `'clamp'`. */
  borderMode?: GPURasterBorderMode;
};

/**
 * Computes slope, aspect, and hillshade for an elevation tile with Horn's 3x3 method.
 *
 * Composes two Sobel `GPURasterGradient` passes and one fused shade kernel. Cell size, z factor,
 * sun position, and latitude band are read from `parameters` every encoding. Columns increase east
 * and, with the default `rowDirection`, rows increase south. Invalid pixels (an invalid center or
 * 3x3 neighbor, or invalid settings) receive NaN and validity 0. For seamless tiles, pass a tile
 * with a one-pixel halo; the contributor satisfies the `GPURasterHaloStage` contract.
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
    const {output} = props;
    const pixelCount = validateRasterGrid(id, props.grid);
    if (
      !output.slope &&
      !output.aspect &&
      !output.hillshade &&
      !output.outputValidity &&
      !output.hillshadeTexture
    ) {
      throw new Error(`${id} requires at least one output`);
    }
    for (const [name, view, format] of [
      ['slope', output.slope, 'float32'],
      ['aspect', output.aspect, 'float32'],
      ['hillshade', output.hillshade, 'float32'],
      ['outputValidity', output.outputValidity, 'uint32']
    ] as const) {
      if (view) {
        validatePackedView(view, [format], `${id} output.${name}`);
        if (view.length !== pixelCount) {
          throw new Error(`${id} output.${name} must contain one value per pixel`);
        }
      }
    }
    validateRasterParameters(id, props.parameters, GPU_TERRAIN_DERIVATIVES_PARAMETER_LENGTH);
    validateRasterOutputTexture(
      id,
      'output.hillshadeTexture',
      output.hillshadeTexture,
      ['r32float', 'rgba32float'],
      props.grid
    );
    validateRasterCellSizeMode(id, props.cellSizeMode);
    validateRasterGridAxisAligned(id, props.grid, props.cellSizeMode);
    if (!['degrees', 'percent'].includes(props.slopeUnits ?? 'degrees')) {
      throw new Error(`${id} slopeUnits must be degrees or percent`);
    }
    validateRasterRowDirection(id, props.rowDirection);
    validateRasterBuffersDistinct(
      id,
      [output.slope, output.aspect, output.hillshade, output.outputValidity],
      [...getRasterBandViews(props.elevation), props.parameters]
    );
  }

  /** Returns optional elevation gather, two Sobel passes, shade, and optional texture nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {output} = props;
    const {width, height} = props.grid;
    validateRasterBandBelongsToGraph(id, graph, props.elevation, [output.hillshadeTexture]);
    validateGraphViewsBelongToGraph(id, graph, [
      props.parameters,
      output.slope,
      output.aspect,
      output.hillshade,
      output.outputValidity
    ]);
    const pixelCount = width * height;
    const borderMode = props.borderMode ?? 'clamp';
    const source = getRasterBufferBandNodes(graph, id, props.elevation, props.grid, false);
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
      output.hillshade ??
      (output.hillshadeTexture
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
        settings: props.parameters,
        slope: output.slope,
        aspect: output.aspect,
        hillshade: hillshadeTarget,
        validity: output.outputValidity,
        cellSizeMode: props.cellSizeMode ?? 'uniform',
        slopeUnits: props.slopeUnits ?? 'degrees',
        rowDirection: props.rowDirection ?? 'south'
      })
    );
    if (output.hillshadeTexture && hillshadeTarget) {
      const texture = output.hillshadeTexture;
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
    cellSizeMode: GPURasterCellSizeMode;
    slopeUnits: GPUTerrainSlopeUnits;
    rowDirection: GPURasterRowDirection;
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
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: 'GPUTerrainDerivatives',
    variant: `shade-${props.cellSizeMode}`,
    bindings,
    invocationCount: props.width * props.height,
    declarations: `const WIDTH: u32 = ${props.width}u;
const HEIGHT: u32 = ${props.height}u;
const PI: f32 = 3.141592653589793;
${RASTER_DEGREES_TO_RADIANS_WGSL}
const RADIANS_TO_DEGREES: f32 = 57.29577951308232;
${getRasterRowNorthSignWGSL(props.rowDirection)}
${RASTER_WGSL_HELPERS}
${getRasterGroundCellSizeWGSL(props.cellSizeMode, TERRAIN_DERIVATIVES_GRID_SLOTS)}`,
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
