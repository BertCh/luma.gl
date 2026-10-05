// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Binding, BindingDeclaration} from '@luma.gl/core';
import {Computation} from '@luma.gl/engine';
import {
  createGPUComputeCommandNode,
  getViewBinding,
  getViewElementOffset,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView,
  type GraphResourceUse,
  type GraphTextureView
} from '@luma.gl/gpgpu/gpu-core';
import {
  assertRasterStorageBindingFits,
  getRasterDispatchSize,
  RASTER_WORKGROUP_DIMENSION
} from '../../gpu-raster/raster-utils';
import {createWGSLKernelNode, getWGSLFloatLiteral} from '../../utils/wgsl-kernel-nodes';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {validateGraphViewsBelongToGraph} from '../../utils/gpu-contributor-utils';
import {
  validateTerrainBuffersDistinct,
  validateTerrainGrid
} from '../terrain-analysis/terrain-analysis-utils';

/** RGB elevation encodings understood by {@link GPUTerrainRGBDecode}. */
export type GPUTerrainRGBEncoding = 'terrarium' | 'mapbox';

/** Default `validRange` in metres: just below the Challenger Deep (-10935 m) to just above Everest (8849 m). */
export const GPU_TERRAIN_RGB_DEFAULT_VALID_RANGE: readonly [number, number] = [-11000, 9000];

/** Largest height, in metres, that the sea clamp of `clampBathymetry` flattens (exclusive). */
export const GPU_TERRAIN_RGB_SEA_CLAMP_FLOOR = -12000;

/**
 * Input of {@link GPUTerrainRGBDecode}: exactly one of a sampled RGBA8 texture or a packed word buffer.
 *
 * The encoded RGB bytes must be read exactly (see the decode-before-filter rule on the class).
 */
export type GPUTerrainRGBDecodeInput =
  | {
      /**
       * `rgba8unorm` texture, loaded with `textureLoad` (never filtered). Bytes are recovered as
       * `round(clamp(v, 0, 1) * 255)`, which is exact for unorm8 storage.
       */
      texture: GraphTextureView<'rgba8unorm'>;
      buffer?: undefined;
    }
  | {
      texture?: undefined;
      /**
       * One packed RGBA8 word per pixel, little-endian: `R = word & 0xff`, `G = (word >> 8) & 0xff`,
       * `B = (word >> 16) & 0xff`, `A = word >> 24`. This is `getImageData().data` viewed as a
       * `Uint32Array` on a little-endian host.
       */
      buffer: GraphDataView<'uint32'>;
    };

/**
 * Properties for {@link GPUTerrainRGBDecode}.
 *
 * Everything here is topology or compile-time (it becomes WGSL constants), so there are no
 * per-frame settings. Cell-size independent: decoding is a per-pixel operation on heights in metres.
 */
export type GPUTerrainRGBDecodeProps = {
  /** Prefix for node IDs. Defaults to `'terrain-rgb-decode'`. */
  id?: string;
  /** Grid width in pixels. */
  width: number;
  /** Grid height in pixels. */
  height: number;
  /**
   * `'terrarium'`: `h = R * 256 + G + B / 256 - 32768`.
   * `'mapbox'`: Mapbox terrain-RGB, `h = -10000 + 0.1 * (R * 65536 + G * 256 + B)`.
   */
  encoding: GPUTerrainRGBEncoding;
  /** Encoded source, see {@link GPUTerrainRGBDecodeInput}. */
  input: GPUTerrainRGBDecodeInput;
  /** Optional source validity for the buffer input. Zero marks a pixel as nodata. */
  inputValidity?: GraphDataView<'uint32'>;
  /** When true (default), a pixel whose alpha byte is 0 is nodata (transparent PNG pixels). */
  alphaNoData?: boolean;
  /** Optional encoded nodata colour as bytes `[R, G, B]`; matching pixels are nodata. */
  noDataRGB?: readonly [number, number, number];
  /**
   * Closed range of valid decoded heights in metres. Heights outside it are nodata. Defaults to
   * {@link GPU_TERRAIN_RGB_DEFAULT_VALID_RANGE}, which removes blank-canvas pixels: Terrarium
   * (0, 0, 0) decodes to -32768 and (255, 255, 255) to 32767.996. Either side may be
   * `+/-Infinity`, which omits that comparison. Finite bounds are rounded to float32.
   */
  validRange?: readonly [number, number];
  /**
   * Sea clamp ported from mt-image: a decoded height `h` with `-12000 < h < 0` becomes `+0`.
   * Default false. This is an application choice and a trap: it flattens genuine land below sea
   * level (the Dead Sea at about -430 m, Dutch polders, Death Valley) and every shallow bay.
   * Validity is evaluated on the unclamped height.
   */
  clampBathymetry?: boolean;
  /** Caller-owned float32 heights, one per pixel. Nodata pixels hold NaN bits `0x7fc00000`. */
  values: GraphDataView<'float32'>;
  /** Caller-owned uint32 validity, one per pixel: 1 for decoded heights, 0 for nodata. */
  validity: GraphDataView<'uint32'>;
};

const INVALID_VALUE_EXPRESSION = 'bitcast<f32>(0x7fc00000u | (pixelIndex & 0u))';

/**
 * Decodes RGB-encoded elevation (Terrarium or Mapbox terrain-RGB) into float32 heights plus
 * uint32 validity, following the luma nodata convention (invalid pixels hold NaN bits
 * `0x7fc00000` and validity 0).
 *
 * Decode before filter. The encoded texture must be read with `textureLoad` (nearest, exact), never
 * bilinear-filtered or mip-mapped before decoding: filtering mixes the byte channels without
 * carries, so a blend of (R=1, G=255) and (R=2, G=0) is garbage. Decode first, then filter or
 * resample the float heights.
 *
 * Exactness of Terrarium. With `N = R * 65536 + G * 256 + B < 2^24` the kernel computes
 * `f32(N) * 2^-8 - 32768`. `f32(N)` is exact (24 bits), the multiply by a power of two is exact, and
 * the difference is a multiple of 1/256 with magnitude below 2^15, so at most 24 significant bits:
 * exact. The float formula `R * 256 + G + B * (1/256) - 32768` is also exact for the same reason
 * (mt-image's argument): every partial sum in any association order, and any FMA contraction, is a
 * multiple of 1/256 below 2^16 in magnitude. Division by 256 is never used because WGSL f32
 * division is only accurate to 2.5 ULP. So the GPU float32 equals the exact value and the CPU
 * float64 decode stored to float32, bit for bit, for all 2^24 RGB triples.
 *
 * Mapbox. With `M = N - 100000` (exact integer, `|M| < 2^24`) the kernel computes
 * `f32(M) * f32(0.1)`: one correctly rounded multiply, so no fused multiply-add is possible, and
 * bit-identical to `Math.fround(M * Math.fround(0.1))` (the float64 product of two 24-bit
 * significands is exact). 0.1 is not dyadic, so this is not the correctly rounded `M / 10`. The error
 * is bounded by `|h - M/10| <= 0.5 ulp(h) + |M| * |fl(0.1) - 0.1|`, about `0.5 ulp + |M| * 1.49e-9`;
 * for `|h| <= 9000 m` that is below 1 mm (measured maximum 0.59 mm), far below the 0.1 m
 * quantisation. Measured exhaustively in the node spec: 3,355,441 of the 2^24 codes (20%) differ
 * from the correctly rounded float32 of `M / 10`, by exactly 1 ulp at most, and the worst error
 * reaches 0.96 of the stated bound.
 *
 * Nodata never bleeds: alpha 0, `noDataRGB`, `inputValidity` and `validRange` mark a pixel invalid
 * and its decoded value is replaced by NaN bits, never blended into anything.
 */
export class GPUTerrainRGBDecode implements GPUCommandNodeProducer {
  /** Prefix for every node ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUTerrainRGBDecodeProps;
  private readonly validRange: readonly [number, number];

  constructor(props: GPUTerrainRGBDecodeProps) {
    this.id = props.id ?? 'terrain-rgb-decode';
    this.props = props;
    const {id} = this;
    const pixelCount = validateTerrainGrid(id, props.width, props.height);
    if (props.encoding !== 'terrarium' && props.encoding !== 'mapbox') {
      throw new Error(`${id} encoding must be 'terrarium' or 'mapbox'`);
    }
    const {texture, buffer} = props.input;
    if (Boolean(texture) === Boolean(buffer)) {
      throw new Error(`${id} input requires exactly one of texture or buffer`);
    }
    if (texture) {
      if (texture.format !== 'rgba8unorm') {
        throw new Error(`${id} input texture must be rgba8unorm`);
      }
      if (
        texture.dimension !== '2d' ||
        texture.mipLevelCount !== 1 ||
        texture.arrayLayerCount !== 1 ||
        texture.width !== props.width ||
        texture.height !== props.height
      ) {
        throw new Error(`${id} input texture must be one 2D mip with the grid extent`);
      }
      if (props.inputValidity) {
        throw new Error(`${id} inputValidity requires a buffer input`);
      }
    }
    if (buffer) {
      validatePackedUint32View(buffer, `${id} input buffer`);
      if (buffer.length !== pixelCount) {
        throw new Error(`${id} input buffer must contain one packed RGBA8 word per pixel`);
      }
    }
    if (props.inputValidity) {
      validatePackedUint32View(props.inputValidity, `${id} inputValidity`);
      if (props.inputValidity.length !== pixelCount) {
        throw new Error(`${id} inputValidity must contain one value per pixel`);
      }
    }
    validatePackedView(props.values, ['float32'], `${id} values`);
    validatePackedUint32View(props.validity, `${id} validity`);
    if (props.values.length !== pixelCount || props.validity.length !== pixelCount) {
      throw new Error(`${id} outputs must contain one value per pixel`);
    }
    if (props.noDataRGB) {
      if (
        props.noDataRGB.length !== 3 ||
        !props.noDataRGB.every(byte => Number.isInteger(byte) && byte >= 0 && byte <= 255)
      ) {
        throw new Error(`${id} noDataRGB must be three integer bytes`);
      }
    }
    this.validRange = props.validRange ?? GPU_TERRAIN_RGB_DEFAULT_VALID_RANGE;
    const [rangeMinimum, rangeMaximum] = this.validRange;
    if (Number.isNaN(rangeMinimum) || Number.isNaN(rangeMaximum) || rangeMinimum > rangeMaximum) {
      throw new Error(`${id} validRange must be [minimum, maximum] with minimum <= maximum`);
    }
    validateTerrainBuffersDistinct(
      id,
      [props.values, props.validity],
      [buffer, props.inputValidity]
    );
  }

  /** Returns the single decode node. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    validateGraphViewsBelongToGraph(id, graph, [
      props.input.buffer,
      props.inputValidity,
      props.values,
      props.validity
    ]);
    if (props.input.texture) {
      if (props.input.texture.texture.graph !== graph) {
        throw new Error(`${id} views must belong to the target graph`);
      }
      return [this.getTextureNode(graph, props.input.texture)];
    }
    const source = props.input.buffer as GraphDataView<'uint32'>;
    const bindings = [
      {name: 'inputWords', view: source, type: 'u32' as const, access: 'read' as const},
      ...(props.inputValidity
        ? [
            {
              name: 'inputValidity',
              view: props.inputValidity,
              type: 'u32' as const,
              access: 'read' as const
            }
          ]
        : []),
      {
        name: 'outputValues',
        view: props.values,
        type: 'f32' as const,
        access: 'read_write' as const
      },
      {
        name: 'outputValidity',
        view: props.validity,
        type: 'u32' as const,
        access: 'read_write' as const
      }
    ];
    return [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-decode`,
        operation: 'GPUTerrainRGBDecode',
        variant: `${props.encoding}-buffer`,
        bindings,
        invocationCount: props.width * props.height,
        declarations: this.getDecodeDeclarations(),
        body: `let pixelIndex = index;
  let word = inputWords[inputWordsOffset + pixelIndex];
  let red = word & 0xffu;
  let green = (word >> 8u) & 0xffu;
  let blue = (word >> 16u) & 0xffu;
  let alpha = word >> 24u;
  var isValid = isEncodedPixelValid(red, green, blue, alpha);
  ${props.inputValidity ? 'isValid = isValid && inputValidity[inputValidityOffset + pixelIndex] != 0u;' : ''}
  ${this.getDecodeStatements()}`
      })
    ];
  }

  private getDecodeDeclarations(): string {
    const {props, validRange} = this;
    const [rangeMinimum, rangeMaximum] = validRange;
    const noDataRGB = props.noDataRGB;
    const rangeConditions = [
      Number.isFinite(rangeMinimum) ? `height >= ${getWGSLFloatLiteral(rangeMinimum)}` : '',
      Number.isFinite(rangeMaximum) ? `height <= ${getWGSLFloatLiteral(rangeMaximum)}` : ''
    ].filter(Boolean);
    const heightExpression =
      props.encoding === 'terrarium'
        ? 'f32(code) * 0.00390625 - 32768.0'
        : `f32(i32(code) - 100000) * ${getWGSLFloatLiteral(0.1)}`;
    return `const WIDTH: u32 = ${props.width}u;
const HEIGHT: u32 = ${props.height}u;
fn isEncodedPixelValid(red: u32, green: u32, blue: u32, alpha: u32) -> bool {
  ${props.alphaNoData === false ? '' : 'if (alpha == 0u) { return false; }'}
  ${noDataRGB ? `if (red == ${noDataRGB[0]}u && green == ${noDataRGB[1]}u && blue == ${noDataRGB[2]}u) { return false; }` : ''}
  return true;
}
// Integer code N = R * 65536 + G * 256 + B < 2^24, then exactly one float operation sequence.
fn decodeHeight(red: u32, green: u32, blue: u32) -> f32 {
  let code = (red << 16u) | (green << 8u) | blue;
  return ${heightExpression};
}
fn isHeightInRange(height: f32) -> bool {
  return ${rangeConditions.join(' && ') || 'true'};
}
fn applySeaClamp(height: f32) -> f32 {
  ${props.clampBathymetry ? `if (height < 0.0 && height > ${getWGSLFloatLiteral(GPU_TERRAIN_RGB_SEA_CLAMP_FLOOR)}) { return 0.0; }` : ''}
  return height;
}`;
  }

  private getDecodeStatements(): string {
    return `let height = decodeHeight(red, green, blue);
  isValid = isValid && isHeightInRange(height);
  outputValidity[outputValidityOffset + pixelIndex] = select(0u, 1u, isValid);
  outputValues[outputValuesOffset + pixelIndex] =
    select(${INVALID_VALUE_EXPRESSION}, applySeaClamp(height), isValid);`;
  }

  private getTextureNode<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    texture: GraphTextureView<'rgba8unorm'>
  ): GPUCommandNode<Parameters> {
    const {id, props} = this;
    assertRasterStorageBindingFits(graph.device, props.values, `${id} values`);
    assertRasterStorageBindingFits(graph.device, props.validity, `${id} validity`);
    const [horizontalCount, verticalCount] = getRasterDispatchSize(
      graph.device,
      props.width,
      props.height,
      id
    );
    const resources: GraphResourceUse[] = [
      {texture, usage: 'sampled'},
      {buffer: props.values, usage: 'storage-write'},
      {buffer: props.validity, usage: 'storage-write'}
    ];
    const source = /* wgsl */ `
const OUTPUT_VALUES_OFFSET: u32 = ${getViewElementOffset(props.values)}u;
const OUTPUT_VALIDITY_OFFSET: u32 = ${getViewElementOffset(props.validity)}u;
@group(0) @binding(0) var sourceTexture: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> outputValuesBuffer: array<f32>;
@group(0) @binding(2) var<storage, read_write> outputValidityBuffer: array<u32>;
${this.getDecodeDeclarations()}

@compute @workgroup_size(${RASTER_WORKGROUP_DIMENSION}, ${RASTER_WORKGROUP_DIMENSION})
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
  if (globalId.x >= WIDTH || globalId.y >= HEIGHT) { return; }
  let pixelIndex = globalId.y * WIDTH + globalId.x;
  // Exact for unorm8: the load is within half a code of k / 255.
  let texel = clamp(textureLoad(sourceTexture, vec2<i32>(globalId.xy), 0), vec4<f32>(0.0), vec4<f32>(1.0));
  let bytes = vec4<u32>(round(texel * 255.0));
  var isValid = isEncodedPixelValid(bytes.x, bytes.y, bytes.z, bytes.w);
  let height = decodeHeight(bytes.x, bytes.y, bytes.z);
  isValid = isValid && isHeightInRange(height);
  outputValidityBuffer[OUTPUT_VALIDITY_OFFSET + pixelIndex] = select(0u, 1u, isValid);
  outputValuesBuffer[OUTPUT_VALUES_OFFSET + pixelIndex] =
    select(${INVALID_VALUE_EXPRESSION}, applySeaClamp(height), isValid);
}`;
    return createGPUComputeCommandNode<Parameters>({
      id: `${id}-decode`,
      resources,
      compile: ({device}) => {
        const bindingDeclarations: BindingDeclaration[] = [
          {name: 'sourceTexture', type: 'texture', group: 0, location: 0, sampleType: 'float'},
          {name: 'outputValuesBuffer', type: 'storage', group: 0, location: 1},
          {name: 'outputValidityBuffer', type: 'storage', group: 0, location: 2}
        ];
        const computation = new Computation(device, {
          id: `${id}-decode`,
          source,
          shaderLayout: {bindings: bindingDeclarations}
        });
        return {
          encode: ({computePass, getBuffer, getTextureView}) => {
            const resolvedBindings: Record<string, Binding> = {
              sourceTexture: getTextureView(texture),
              outputValuesBuffer: getViewBinding(props.values, getBuffer),
              outputValidityBuffer: getViewBinding(props.validity, getBuffer)
            };
            computation.setBindings(resolvedBindings);
            computation.dispatch(computePass, horizontalCount, verticalCount);
          },
          destroy: () => computation.destroy()
        };
      }
    });
  }
}
