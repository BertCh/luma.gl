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
import {GPURasterBufferToTexture, type GPURasterBand} from '../../gpu-raster/index';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral,
  type WGSLKernelBinding
} from '../../utils/wgsl-kernel-nodes';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {
  captureGraphCommandNodes,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {
  getTerrainBandViews,
  getTerrainElevationNodes,
  validateTerrainBandBelongsToGraph,
  validateTerrainBuffersDistinct,
  validateTerrainGrid,
  validateTerrainSettings,
  validateTerrainTexture
} from '../terrain-analysis/terrain-analysis-utils';
import {
  TERRAIN_ILLUMINATION_WGSL_CONSTANTS,
  validateTerrainIlluminationView
} from './terrain-illumination-utils';

/** Largest supported `GPUTextureShadingProps.levelCount`. */
export const GPU_TEXTURE_SHADING_MAX_LEVEL_COUNT = 8;

/**
 * Minimum incremental cascade sigma, in pixels of the previous level's grid, for a level to move
 * to a grid decimated by 2.
 */
export const DOWNSAMPLE_MIN_SIGMA = 4;

/** Number of float32 values read from `GPUTextureShadingProps.settings`. */
export const GPU_TEXTURE_SHADING_PARAMETER_LENGTH = 12;

/** CPU-side description packed by {@link getGPUTextureShadingParameterValues}. */
export type GPUTextureShadingSettings = {
  /**
   * Fractional Laplacian order `alpha` (Leland Brown's "detail"): band `k` gets weight
   * `2^(-k * detail)`. 0 weighs every band equally; 1 approaches a Laplacian. Defaults to 0.5.
   */
  detail?: number;
  /** Output multiplier. Defaults to 1. */
  gain?: number;
  /** Explicit per-band weights (up to 8), overriding `detail`. */
  levelWeights?: readonly number[];
};

/**
 * Packs settings into the 12-float layout read by {@link GPUTextureShading}:
 * `[gain, w0, w1, ..., w7, 0, 0, 0]`.
 *
 * Band `k` is centered near spatial frequency `1 / sigma_k` with `sigma_k = baseSigma * 2^k`, so
 * the fractional Laplacian response `|f|^alpha` is approximated by `w_k = 2^(-k * alpha)`.
 *
 * @throws If more than 8 weights are given, a value is not finite, or `target` is too short.
 */
export function getGPUTextureShadingParameterValues(
  settings: GPUTextureShadingSettings = {},
  target: Float32Array = new Float32Array(GPU_TEXTURE_SHADING_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_TEXTURE_SHADING_PARAMETER_LENGTH) {
    throw new Error('Texture shading settings target must hold 12 values');
  }
  const detail = settings.detail ?? 0.5;
  const weights =
    settings.levelWeights ??
    Array.from({length: GPU_TEXTURE_SHADING_MAX_LEVEL_COUNT}, (_, level) => 2 ** (-level * detail));
  if (weights.length > GPU_TEXTURE_SHADING_MAX_LEVEL_COUNT) {
    throw new Error('Texture shading supports at most 8 level weights');
  }
  const gain = settings.gain ?? 1;
  if (![detail, gain, ...weights].every(Number.isFinite)) {
    throw new Error('Texture shading settings must be finite');
  }
  target.fill(0, 0, GPU_TEXTURE_SHADING_PARAMETER_LENGTH);
  target[0] = gain;
  target.set(weights, 1);
  return target;
}

/**
 * Returns the discrete Gaussian of one cascade pass: normalized samples of
 * `exp(-x^2 / (2 sigma^2))` over `[-radius, radius]` with `radius = ceil(3 sigma)`.
 *
 * Weights are rounded to float32 so CPU oracles match the baked WGSL constants.
 */
export function getGPUTextureShadingKernel(sigma: number): Float32Array {
  const radius = Math.max(1, Math.ceil(3 * sigma));
  const samples = Array.from({length: 2 * radius + 1}, (_, index) =>
    Math.exp(-((index - radius) ** 2) / (2 * sigma * sigma))
  );
  const sum = samples.reduce((total, value) => total + value, 0);
  return Float32Array.from(samples, value => value / sum);
}

/**
 * Returns the incremental blur sigma of each cascade level:
 * `sqrt(sigma_k^2 - sigma_(k-1)^2)` with `sigma_k = baseSigma * 2^k` and `sigma_(-1) = 0`.
 */
export function getGPUTextureShadingCascadeSigmas(levelCount: number, baseSigma: number): number[] {
  return Array.from({length: levelCount}, (_, level) => {
    const sigma = baseSigma * 2 ** level;
    const previous = level === 0 ? 0 : baseSigma * 2 ** (level - 1);
    return Math.sqrt(sigma * sigma - previous * previous);
  });
}

/** Returns the log2 grid decimation of each level (non-decreasing, steps of at most one). */
function getLevelScaleLog2(sigmas: readonly number[], downsample: boolean): number[] {
  let scaleLog2 = 0;
  return sigmas.map((sigma, level) => {
    if (downsample && level > 0 && sigma / 2 ** scaleLog2 >= DOWNSAMPLE_MIN_SIGMA) {
      scaleLog2++;
    }
    return scaleLog2;
  });
}

/**
 * Properties for {@link GPUTextureShading}.
 *
 * Topology: grid size, elevation format and calibration, `levelCount`, `baseSigma`, and which
 * outputs exist. Per-frame: `settings` (gain and band weights) and elevation contents.
 */
export type GPUTextureShadingProps = {
  /** Prefix for node and transient IDs. Defaults to `'texture-shading'`. */
  id?: string;
  /** Grid width in pixels. */
  width: number;
  /** Grid height in pixels. */
  height: number;
  /** Elevation band, buffer or texture; scale, offset, nodata, and validity are honored. */
  elevation: GPURasterBand;
  /** Per-frame settings with at least 12 float32 values, see {@link getGPUTextureShadingParameterValues}. */
  settings: GraphDataView<'float32'>;
  /** Number of band-pass levels, 1 to 8. Defaults to 6. */
  levelCount?: number;
  /** Finest Gaussian sigma in pixels. Defaults to 1. */
  baseSigma?: number;
  /**
   * Whether the elevation can contain nodata, NaN, or invalid pixels. Defaults to `true`.
   *
   * Pass `false` only when every pixel of the elevation is valid and finite. The cascade then
   * blurs a single channel instead of a value and a validity channel, which is exact (it equals
   * the normalized convolution of an all-valid input) and roughly halves the blur cost. If
   * invalid pixels exist anyway they bleed into their neighbors; invalid centers still output NaN.
   */
  hasNodata?: boolean;
  /**
   * Computes cascade levels whose incremental sigma is at least {@link DOWNSAMPLE_MIN_SIGMA}
   * pixels on a 2x decimated grid per level, then upsamples bilinearly where the band is
   * accumulated. Defaults to `true`. Pass `false` for the exact full-resolution cascade.
   *
   * Tolerance: the maximum absolute difference to the exact cascade stays below 1% of the output
   * range (`max - min` of the exact output) on the test terrains. Results are close to, not
   * bit-identical with, the exact cascade, mostly near borders and nodata hole edges.
   */
  downsampleLevels?: boolean;
  /** Optional texture shade per pixel, in elevation units times `gain`. */
  textureShade?: GraphDataView<'float32'>;
  /** Optional per-pixel 1 where the center elevation is valid, else 0. */
  validity?: GraphDataView<'uint32'>;
  /** Optional storage texture (`r32float` or `rgba32float`, channel 0) receiving the texture shade. */
  textureShadeTexture?: GraphTextureView<'r32float' | 'rgba32float'>;
};

/**
 * Leland Brown's texture shading, a fractional Laplacian of order `alpha` (`|f|^alpha` in the
 * frequency domain), approximated by a multi-scale difference-of-Gaussians band-pass pyramid.
 *
 * `G_-1` is the elevation and `G_k` a Gaussian blur with `sigma_k = baseSigma * 2^k`, built as a
 * cascade of separable passes ({@link getGPUTextureShadingCascadeSigmas},
 * {@link getGPUTextureShadingKernel}). The output is `gain * sum_k w_k (G_(k-1) - G_k)` summed in
 * fixed level order, so changing `detail`, weights, or gain is a settings write and never
 * recompiles. Nodata uses normalized convolution: value-times-validity and validity channels are
 * blurred together and divided, so holes neither bleed nor bias. Borders replicate the edge pixel.
 * Invalid centers receive NaN and validity 0.
 *
 * Output units are elevation units times `gain`; callers usually map it with a symmetric stretch
 * (for example a percentile of `|value|`) to `[-1, 1]` before blending, as in
 * `GPUReliefShading.textureShade`. Exact full-resolution cost is linear in the summed kernel
 * widths, about `10.4 * baseSigma * 2^levelCount` taps per pixel per channel. With
 * `downsampleLevels` (default), levels with incremental sigma of at least
 * {@link DOWNSAMPLE_MIN_SIGMA} pixels run on grids decimated by 2 per level and the cost of those
 * levels becomes nearly constant; the result then differs from the exact cascade by under 1% of
 * the output range. `hasNodata: false` skips the validity channel. `requiredHalo` is the nominal
 * receptive field of the exact cascade; decimated levels add at most a few pixels beyond the
 * 3-sigma kernel cutoff, where the Gaussian weight is negligible.
 */
export class GPUTextureShading implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUTextureShadingProps;
  /** Number of band-pass levels. */
  readonly levelCount: number;
  /** Decimation of each level's grid as a power of two (0 is full resolution). */
  readonly levelScaleLog2: readonly number[];
  /** Nominal full-resolution kernel of each cascade level. */
  readonly kernels: readonly Float32Array[];
  /** Receptive field in pixels: the summed pass radii (`GPURasterHaloStage` contract). */
  readonly requiredHalo: number;

  constructor(props: GPUTextureShadingProps) {
    this.id = props.id ?? 'texture-shading';
    this.props = props;
    const {id} = this;
    const pixelCount = validateTerrainGrid(id, props.width, props.height);
    const levelCount = props.levelCount ?? 6;
    if (
      !Number.isSafeInteger(levelCount) ||
      levelCount < 1 ||
      levelCount > GPU_TEXTURE_SHADING_MAX_LEVEL_COUNT
    ) {
      throw new Error(`${id} levelCount must be an integer in [1, 8]`);
    }
    const baseSigma = props.baseSigma ?? 1;
    if (!Number.isFinite(baseSigma) || baseSigma <= 0) {
      throw new Error(`${id} baseSigma must be finite and positive`);
    }
    this.levelCount = levelCount;
    this.levelScaleLog2 = getLevelScaleLog2(
      getGPUTextureShadingCascadeSigmas(levelCount, baseSigma),
      props.downsampleLevels ?? true
    );
    this.kernels = getGPUTextureShadingCascadeSigmas(levelCount, baseSigma).map(
      getGPUTextureShadingKernel
    );
    this.requiredHalo = this.kernels.reduce((sum, kernel) => sum + (kernel.length - 1) / 2, 0);
    if (!props.textureShade && !props.validity && !props.textureShadeTexture) {
      throw new Error(`${id} requires at least one output`);
    }
    validateTerrainIlluminationView(id, 'textureShade', props.textureShade, 'float32', pixelCount);
    validateTerrainIlluminationView(id, 'validity', props.validity, 'uint32', pixelCount);
    validateTerrainSettings(id, props.settings, GPU_TEXTURE_SHADING_PARAMETER_LENGTH);
    validateTerrainTexture(
      id,
      'textureShadeTexture',
      props.textureShadeTexture,
      ['r32float', 'rgba32float'],
      props.width,
      props.height
    );
    validateTerrainBuffersDistinct(
      id,
      [props.textureShade, props.validity],
      [...getTerrainBandViews(props.elevation), props.settings]
    );
  }

  /** Returns canonicalization, premultiply, two passes plus one accumulate per level, and finalize nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {width, height} = props;
    validateTerrainBandBelongsToGraph(id, graph, props.elevation, [props.textureShadeTexture]);
    validateGraphViewsBelongToGraph(id, graph, [
      props.settings,
      props.textureShade,
      props.validity
    ]);
    const pixelCount = width * height;
    const source = getTerrainElevationNodes(graph, id, props.elevation, width, height, true);
    const nodes: GPUCommandNode<Parameters>[] = [...source.nodes];
    const elevationValues = source.band.storage.values as GraphDataView<'float32'>;
    const elevationValidity = source.band.validity as GraphDataView<'uint32'>;
    const hasNodata = props.hasNodata ?? true;
    const createPair = (name: string, size: number): BlurPair => ({
      numerator: createTransientView(graph, `${id}-${name}-numerator`, 'float32', size),
      denominator: hasNodata
        ? createTransientView(graph, `${id}-${name}-denominator`, 'float32', size)
        : undefined
    });
    const levels = [createPair('level-a', pixelCount), createPair('level-b', pixelCount)];
    const scratch = createPair('scratch', pixelCount);
    const accumulator = createTransientView(graph, `${id}-accumulator`, 'float32', pixelCount);
    // Level 0 reads the canonical elevation directly when no validity channel is needed.
    let previous: GridPair = {
      ...(hasNodata ? levels[0] : {numerator: elevationValues}),
      scaleLog2: 0,
      width,
      height
    };
    if (hasNodata) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-premultiply`,
          operation: 'GPUTextureShading',
          variant: 'premultiply',
          bindings: [
            {name: 'elevationValues', view: elevationValues, type: 'f32', access: 'read'},
            {name: 'elevationValidity', view: elevationValidity, type: 'u32', access: 'read'},
            {name: 'numerator', view: levels[0].numerator, type: 'f32', access: 'read_write'},
            {
              name: 'denominator',
              view: levels[0].denominator!,
              type: 'f32',
              access: 'read_write'
            }
          ],
          invocationCount: pixelCount,
          body: `let isValid = elevationValidity[elevationValidityOffset + index] != 0u;
  numerator[numeratorOffset + index] = select(0.0, elevationValues[elevationValuesOffset + index], isValid);
  denominator[denominatorOffset + index] = select(0.0, 1.0, isValid);`
        })
      );
    }
    const sigmas = getGPUTextureShadingCascadeSigmas(this.levelCount, props.baseSigma ?? 1);
    for (let level = 0; level < this.levelCount; level++) {
      const scaleLog2 = this.levelScaleLog2[level];
      let current: GridPair;
      if (scaleLog2 === 0) {
        const output = levels[(level + 1) % 2];
        current = {...output, scaleLog2, width, height};
        const kernel = this.kernels[level];
        for (const [axis, input, target] of [
          ['horizontal', previous, scratch],
          ['vertical', scratch, output]
        ] as const) {
          nodes.push(
            getBlurNode(graph, {
              id: `${id}-level-${level}-${axis}`,
              width,
              height,
              axis,
              kernel: Array.from(kernel, getWGSLFloatLiteral).join(', '),
              radius: (kernel.length - 1) / 2,
              input,
              output: target
            })
          );
        }
      } else {
        const scale = 2 ** scaleLog2;
        const coarseWidth = Math.ceil((width - 1) / scale) + 1;
        const coarseHeight = Math.ceil((height - 1) / scale) + 1;
        const coarseSize = coarseWidth * coarseHeight;
        const kernel = getGPUTextureShadingKernel(sigmas[level] / scale);
        const grid = createPair(`level-${level}-grid`, coarseSize);
        const coarseScratch = createPair(`level-${level}-scratch`, coarseSize);
        current = {...grid, scaleLog2, width: coarseWidth, height: coarseHeight};
        let blurInput: BlurPair = previous as BlurPair;
        if (scaleLog2 > previous.scaleLog2) {
          nodes.push(
            getDecimateNode(graph, {
              id: `${id}-level-${level}-decimate`,
              input: previous,
              output: grid,
              outputWidth: coarseWidth,
              outputHeight: coarseHeight
            })
          );
          blurInput = grid;
        }
        for (const [axis, input, target] of [
          ['horizontal', blurInput, coarseScratch],
          ['vertical', coarseScratch, grid]
        ] as const) {
          nodes.push(
            getBlurNode(graph, {
              id: `${id}-level-${level}-${axis}`,
              width: coarseWidth,
              height: coarseHeight,
              axis,
              kernel: Array.from(kernel, getWGSLFloatLiteral).join(', '),
              radius: (kernel.length - 1) / 2,
              input,
              output: target
            })
          );
        }
      }
      const bindings: WGSLKernelBinding[] = [];
      for (const [side, grid] of [
        ['previous', previous],
        ['current', current]
      ] as const) {
        bindings.push({
          name: `${side}Numerator`,
          view: grid.numerator,
          type: 'f32',
          access: 'read'
        });
        if (grid.denominator) {
          bindings.push({
            name: `${side}Denominator`,
            view: grid.denominator,
            type: 'f32',
            access: 'read'
          });
        }
      }
      bindings.push(
        {name: 'settings', view: props.settings, type: 'f32', access: 'read'},
        {name: 'accumulator', view: accumulator, type: 'f32', access: 'read_write'}
      );
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-level-${level}-accumulate`,
          operation: 'GPUTextureShading',
          variant: 'accumulate',
          bindings,
          invocationCount: pixelCount,
          declarations: `const LEVEL: u32 = ${level}u;
fn getBlurred(numerator: f32, denominator: f32) -> f32 {
  return select(0.0, numerator / denominator, denominator > 0.0);
}`,
          body: `let column = index % ${width}u;
  let row = index / ${width}u;
  ${getGridSampleCode('previous', previous)}
  ${getGridSampleCode('current', current)}
  let contribution = settings[settingsOffset + 1u + LEVEL] * (previousValue - currentValue);
  accumulator[accumulatorOffset + index] ${level === 0 ? '=' : '+='} contribution;`
        })
      );
      previous = current;
    }
    const target =
      props.textureShade ??
      (props.textureShadeTexture
        ? createTransientView(graph, `${id}-texture-shade`, 'float32', pixelCount)
        : undefined);
    const bindings: WGSLKernelBinding[] = [
      {
        name: 'elevationValidity',
        view: elevationValidity,
        type: 'u32',
        access: 'read'
      },
      {name: 'accumulator', view: accumulator, type: 'f32', access: 'read'},
      {name: 'settings', view: props.settings, type: 'f32', access: 'read'}
    ];
    if (target) {
      bindings.push({
        name: 'textureShade',
        view: target,
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
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-finalize`,
        operation: 'GPUTextureShading',
        variant: 'finalize',
        bindings,
        invocationCount: pixelCount,
        declarations: TERRAIN_ILLUMINATION_WGSL_CONSTANTS,
        body: `let value = settings[settingsOffset] * accumulator[accumulatorOffset + index];
  let isValid = elevationValidity[elevationValidityOffset + index] != 0u && isFiniteValue(value);
  ${target ? 'textureShade[textureShadeOffset + index] = select(getNaN(index), value, isValid);' : ''}
  ${props.validity ? 'validityValues[validityValuesOffset + index] = select(0u, 1u, isValid);' : ''}`
      })
    );
    if (props.textureShadeTexture && target) {
      const texture = props.textureShadeTexture;
      nodes.push(
        ...captureGraphCommandNodes(graph, () =>
          new GPURasterBufferToTexture({
            id: `${id}-texture`,
            input: {
              id: `${id}-texture-band`,
              format: 'float32',
              storage: {kind: 'buffer', values: target}
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

type BlurPair = {
  numerator: GraphDataView<'float32'>;
  /** Absent when the elevation has no nodata. */
  denominator?: GraphDataView<'float32'>;
};

/** A blurred level on a grid decimated by `2^scaleLog2`. */
type GridPair = BlurPair & {scaleLog2: number; width: number; height: number};

/**
 * Returns WGSL declaring `<side>Value`: the normalized blurred value of `grid` at the full
 * resolution pixel `(column, row)`, bilinearly upsampled when `grid` is decimated. Coarse sample
 * `c` sits exactly on full-resolution pixel `c * S` (corner-aligned), so pixel `x` maps to `x / S`
 * and the replicated border of the coarse grid coincides with the replicated border of the
 * full-resolution grid. The coarse grid extends to the first sample at or beyond the last pixel.
 */
function getGridSampleCode(side: string, grid: GridPair): string {
  const numerator = `${side}Numerator`;
  const denominator = `${side}Denominator`;
  const fetch = (indexName: string) =>
    grid.denominator
      ? `getBlurred(${numerator}[${numerator}Offset + ${indexName}], ${denominator}[${denominator}Offset + ${indexName}])`
      : `${numerator}[${numerator}Offset + ${indexName}]`;
  if (grid.scaleLog2 === 0) {
    return `let ${side}Value = ${fetch('index')};`;
  }
  const scale = 2 ** grid.scaleLog2;
  const bilinear = (name: string) =>
    `mix(mix(${name}[${name}Offset + ${side}Index00], ${name}[${name}Offset + ${side}Index10], ${side}FractionX),
      mix(${name}[${name}Offset + ${side}Index01], ${name}[${name}Offset + ${side}Index11], ${side}FractionX), ${side}FractionY)`;
  return `let ${side}CoordinateX = f32(column) / ${scale}.0;
  let ${side}CoordinateY = f32(row) / ${scale}.0;
  let ${side}FloorX = floor(${side}CoordinateX);
  let ${side}FloorY = floor(${side}CoordinateY);
  let ${side}FractionX = ${side}CoordinateX - ${side}FloorX;
  let ${side}FractionY = ${side}CoordinateY - ${side}FloorY;
  let ${side}Column0 = u32(clamp(i32(${side}FloorX), 0, ${grid.width - 1}));
  let ${side}Column1 = u32(clamp(i32(${side}FloorX) + 1, 0, ${grid.width - 1}));
  let ${side}Row0 = u32(clamp(i32(${side}FloorY), 0, ${grid.height - 1})) * ${grid.width}u;
  let ${side}Row1 = u32(clamp(i32(${side}FloorY) + 1, 0, ${grid.height - 1})) * ${grid.width}u;
  let ${side}Index00 = ${side}Row0 + ${side}Column0;
  let ${side}Index10 = ${side}Row0 + ${side}Column1;
  let ${side}Index01 = ${side}Row1 + ${side}Column0;
  let ${side}Index11 = ${side}Row1 + ${side}Column1;
  let ${side}Numerator_ = ${bilinear(numerator)};
  ${
    grid.denominator
      ? `let ${side}Denominator_ = ${bilinear(denominator)};
  let ${side}Value = getBlurred(${side}Numerator_, ${side}Denominator_);`
      : `let ${side}Value = ${side}Numerator_;`
  }`;
}

/**
 * Builds a 2x decimation by injection (every second sample, corner-aligned, replicating the
 * edge). The input is a Gaussian level with sigma of at least about 2 of its own pixels, so its
 * spectrum above the decimated Nyquist frequency is negligible and no prefilter is needed.
 */
function getDecimateNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    input: GridPair;
    output: BlurPair;
    outputWidth: number;
    outputHeight: number;
  }
): GPUCommandNode<Parameters> {
  const {input, output} = props;
  const channels = [
    ['Numerator', input.numerator, output.numerator],
    ...(input.denominator && output.denominator
      ? [['Denominator', input.denominator, output.denominator] as const]
      : [])
  ] as const;
  const bindings: WGSLKernelBinding[] = channels.flatMap(([channel, inputView, outputView]) => [
    {name: `input${channel}`, view: inputView, type: 'f32', access: 'read'},
    {name: `output${channel}`, view: outputView, type: 'f32', access: 'read_write'}
  ]);
  const copy = (channel: string) =>
    `output${channel}[output${channel}Offset + index] = input${channel}[input${channel}Offset + sourceIndex];`;
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: 'GPUTextureShading',
    variant: 'decimate',
    bindings,
    invocationCount: props.outputWidth * props.outputHeight,
    body: `let column = min(2u * (index % ${props.outputWidth}u), ${input.width - 1}u);
  let row = min(2u * (index / ${props.outputWidth}u), ${input.height - 1}u);
  let sourceIndex = row * ${input.width}u + column;
  ${channels.map(([channel]) => copy(channel)).join('\n  ')}`
  });
}

/** Builds one separable Gaussian pass over a premultiplied numerator/denominator pair. */
function getBlurNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    width: number;
    height: number;
    axis: 'horizontal' | 'vertical';
    kernel: string;
    radius: number;
    input: BlurPair;
    output: BlurPair;
  }
): GPUCommandNode<Parameters> {
  const horizontal = props.axis === 'horizontal';
  const {input, output} = props;
  const hasDenominator = Boolean(input.denominator && output.denominator);
  const bindings: WGSLKernelBinding[] = [
    {name: 'inputNumerator', view: input.numerator, type: 'f32', access: 'read'},
    {name: 'outputNumerator', view: output.numerator, type: 'f32', access: 'read_write'}
  ];
  if (hasDenominator) {
    bindings.push(
      {name: 'inputDenominator', view: input.denominator!, type: 'f32', access: 'read'},
      {name: 'outputDenominator', view: output.denominator!, type: 'f32', access: 'read_write'}
    );
  }
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: 'GPUTextureShading',
    variant: `blur-${props.axis}`,
    bindings,
    invocationCount: props.width * props.height,
    declarations: `const WIDTH: u32 = ${props.width}u;
const HEIGHT: u32 = ${props.height}u;
const RADIUS: i32 = ${props.radius};
const KERNEL = array<f32, ${2 * props.radius + 1}>(${props.kernel});`,
    body: `let column = i32(index % WIDTH);
  let row = i32(index / WIDTH);
  var numeratorSum = 0.0;
  ${hasDenominator ? 'var denominatorSum = 0.0;' : ''}
  for (var offset = -RADIUS; offset <= RADIUS; offset++) {
    ${
      horizontal
        ? 'let sampleIndex = u32(row) * WIDTH + u32(clamp(column + offset, 0, i32(WIDTH) - 1));'
        : 'let sampleIndex = u32(clamp(row + offset, 0, i32(HEIGHT) - 1)) * WIDTH + u32(column);'
    }
    let weight = KERNEL[u32(offset + RADIUS)];
    numeratorSum += weight * inputNumerator[inputNumeratorOffset + sampleIndex];
    ${hasDenominator ? 'denominatorSum += weight * inputDenominator[inputDenominatorOffset + sampleIndex];' : ''}
  }
  outputNumerator[outputNumeratorOffset + index] = numeratorSum;
  ${hasDenominator ? 'outputDenominator[outputDenominatorOffset + index] = denominatorSum;' : ''}`
  });
}
