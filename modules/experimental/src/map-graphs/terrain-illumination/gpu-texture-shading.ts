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
  createMapGraphKernelNode,
  getWGSLFloatLiteral,
  type MapGraphKernelBinding
} from '../map-graph-kernels';
import type {GPUMapGraphRecipe} from '../map-graph-types';
import {captureGraphCommandNodes, validateGraphViewsBelongToGraph} from '../map-graph-utils';
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
 * `GPUReliefShading.textureShade`. The cost is linear in the summed kernel widths, about
 * `10.4 * baseSigma * 2^levelCount` taps per pixel per channel; the receptive field is
 * `requiredHalo` pixels.
 */
export class GPUTextureShading implements GPUMapGraphRecipe {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Stable recipe name. */
  readonly recipe = 'texture-shading';
  /** Validated properties. */
  readonly props: GPUTextureShadingProps;
  /** Number of band-pass levels. */
  readonly levelCount: number;
  /** Baked kernel of each cascade level. */
  readonly kernels: readonly Float32Array[];
  /** Receptive field in pixels: the summed pass radii (`GPURasterHaloStage` contract). */
  readonly requiredHalo: number;

  constructor(props: GPUTextureShadingProps) {
    this.id = props.id ?? this.recipe;
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
    const createPair = (name: string) => ({
      numerator: createTransientView(graph, `${id}-${name}-numerator`, 'float32', pixelCount),
      denominator: createTransientView(graph, `${id}-${name}-denominator`, 'float32', pixelCount)
    });
    const levels = [createPair('level-a'), createPair('level-b')];
    const scratch = createPair('scratch');
    const accumulator = createTransientView(graph, `${id}-accumulator`, 'float32', pixelCount);
    nodes.push(
      createMapGraphKernelNode<Parameters>(graph, {
        id: `${id}-premultiply`,
        operation: 'GPUTextureShading',
        variant: 'premultiply',
        bindings: [
          {
            name: 'elevationValues',
            view: elevationValues,
            type: 'f32',
            access: 'read'
          },
          {
            name: 'elevationValidity',
            view: elevationValidity,
            type: 'u32',
            access: 'read'
          },
          {
            name: 'numerator',
            view: levels[0].numerator,
            type: 'f32',
            access: 'read_write'
          },
          {
            name: 'denominator',
            view: levels[0].denominator,
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
    for (let level = 0; level < this.levelCount; level++) {
      const previous = levels[level % 2];
      const current = levels[(level + 1) % 2];
      const kernel = Array.from(this.kernels[level], getWGSLFloatLiteral).join(', ');
      for (const [axis, input, output] of [
        ['horizontal', previous, scratch],
        ['vertical', scratch, current]
      ] as const) {
        nodes.push(
          getBlurNode(graph, {
            id: `${id}-level-${level}-${axis}`,
            width,
            height,
            axis,
            kernel,
            radius: (this.kernels[level].length - 1) / 2,
            input,
            output
          })
        );
      }
      nodes.push(
        createMapGraphKernelNode<Parameters>(graph, {
          id: `${id}-level-${level}-accumulate`,
          operation: 'GPUTextureShading',
          variant: 'accumulate',
          bindings: [
            {
              name: 'previousNumerator',
              view: previous.numerator,
              type: 'f32',
              access: 'read'
            },
            {
              name: 'previousDenominator',
              view: previous.denominator,
              type: 'f32',
              access: 'read'
            },
            {
              name: 'currentNumerator',
              view: current.numerator,
              type: 'f32',
              access: 'read'
            },
            {
              name: 'currentDenominator',
              view: current.denominator,
              type: 'f32',
              access: 'read'
            },
            {
              name: 'settings',
              view: props.settings,
              type: 'f32',
              access: 'read'
            },
            {
              name: 'accumulator',
              view: accumulator,
              type: 'f32',
              access: 'read_write'
            }
          ],
          invocationCount: pixelCount,
          declarations: `const LEVEL: u32 = ${level}u;
fn getBlurred(numerator: f32, denominator: f32) -> f32 {
  return select(0.0, numerator / denominator, denominator > 0.0);
}`,
          body: `let band = getBlurred(previousNumerator[previousNumeratorOffset + index],
      previousDenominator[previousDenominatorOffset + index]) -
    getBlurred(currentNumerator[currentNumeratorOffset + index],
      currentDenominator[currentDenominatorOffset + index]);
  let contribution = settings[settingsOffset + 1u + LEVEL] * band;
  accumulator[accumulatorOffset + index] ${level === 0 ? '=' : '+='} contribution;`
        })
      );
    }
    const target =
      props.textureShade ??
      (props.textureShadeTexture
        ? createTransientView(graph, `${id}-texture-shade`, 'float32', pixelCount)
        : undefined);
    const bindings: MapGraphKernelBinding[] = [
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
      createMapGraphKernelNode<Parameters>(graph, {
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
  denominator: GraphDataView<'float32'>;
};

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
  return createMapGraphKernelNode<Parameters>(graph, {
    id: props.id,
    operation: 'GPUTextureShading',
    variant: `blur-${props.axis}`,
    bindings: [
      {
        name: 'inputNumerator',
        view: props.input.numerator,
        type: 'f32',
        access: 'read'
      },
      {
        name: 'inputDenominator',
        view: props.input.denominator,
        type: 'f32',
        access: 'read'
      },
      {
        name: 'outputNumerator',
        view: props.output.numerator,
        type: 'f32',
        access: 'read_write'
      },
      {
        name: 'outputDenominator',
        view: props.output.denominator,
        type: 'f32',
        access: 'read_write'
      }
    ],
    invocationCount: props.width * props.height,
    declarations: `const WIDTH: u32 = ${props.width}u;
const HEIGHT: u32 = ${props.height}u;
const RADIUS: i32 = ${props.radius};
var<private> KERNEL: array<f32, ${2 * props.radius + 1}> = array<f32, ${2 * props.radius + 1}>(${props.kernel});`,
    body: `let column = i32(index % WIDTH);
  let row = i32(index / WIDTH);
  var numeratorSum = 0.0;
  var denominatorSum = 0.0;
  for (var offset = -RADIUS; offset <= RADIUS; offset++) {
    ${
      horizontal
        ? 'let sampleIndex = u32(row) * WIDTH + u32(clamp(column + offset, 0, i32(WIDTH) - 1));'
        : 'let sampleIndex = u32(clamp(row + offset, 0, i32(HEIGHT) - 1)) * WIDTH + u32(column);'
    }
    let weight = KERNEL[u32(offset + RADIUS)];
    numeratorSum += weight * inputNumerator[inputNumeratorOffset + sampleIndex];
    denominatorSum += weight * inputDenominator[inputDenominatorOffset + sampleIndex];
  }
  outputNumerator[outputNumeratorOffset + index] = numeratorSum;
  outputDenominator[outputDenominatorOffset + index] = denominatorSum;`
  });
}
