// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {validateGraphViewsBelongToGraph} from '../../utils/gpu-contributor-utils';
import {
  validateTerrainBuffersDistinct,
  validateTerrainGrid,
  validateTerrainSettings
} from '../terrain-analysis/terrain-analysis-utils';
import {
  TERRAIN_ILLUMINATION_WGSL_CONSTANTS,
  validateTerrainIlluminationView
} from '../terrain-illumination/terrain-illumination-utils';

/** Largest supported number of blended layers. */
export const GPU_RELIEF_BLEND_MAX_LAYER_COUNT = 5;

/** Number of float32 values per layer in the settings buffer. */
export const GPU_RELIEF_BLEND_LAYER_PARAMETER_LENGTH = 5;

/** Number of float32 values of a settings buffer holding the maximum layer count. */
export const GPU_RELIEF_BLEND_PARAMETER_LENGTH =
  GPU_RELIEF_BLEND_MAX_LAYER_COUNT * GPU_RELIEF_BLEND_LAYER_PARAMETER_LENGTH;

/**
 * Blend mode of one layer over the layers below it. Single-band (greyscale) forms of the
 * Photoshop-style modes used by RVT. `'luminosity'` on greyscale equals `'normal'`.
 */
export type GPUReliefBlendMode =
  | 'normal'
  | 'multiply'
  | 'screen'
  | 'overlay'
  | 'soft-light'
  | 'luminosity';

/** Numeric code of every {@link GPUReliefBlendMode}, as written into the settings buffer. */
export const GPU_RELIEF_BLEND_MODE_CODES: Readonly<Record<GPUReliefBlendMode, number>> = {
  normal: 0,
  multiply: 1,
  screen: 2,
  overlay: 3,
  'soft-light': 4,
  luminosity: 5
};

/** Per-frame description of one layer, packed by {@link getGPUReliefBlendParameterValues}. */
export type GPUReliefBlendLayerSettings = {
  /** Input value mapped to 0. */
  minimum: number;
  /** Input value mapped to 1. Must be greater than `minimum`. */
  maximum: number;
  /**
   * Flips the normalized value to `1 - n`. RVT inverts slope gradient and negative openness so
   * that steep terrain is dark.
   */
  invert?: boolean;
  /** Blend mode over the layers below. Ignored for the bottom layer. Defaults to `'normal'`. */
  blendMode?: GPUReliefBlendMode;
  /**
   * Opacity from 0 to 1. Ignored for the bottom layer. Defaults to 1. RVT states opacity in
   * percent; divide by 100.
   */
  opacity?: number;
};

/**
 * Packs layer settings into `[minimum, maximum, invert, blendModeCode, opacity]` per layer,
 * ordered bottom to top.
 *
 * @throws If there are more than 5 layers, a value is not finite, `maximum <= minimum`, the
 * opacity is outside `[0, 1]`, or `target` is shorter than `5 * layers.length`.
 */
export function getGPUReliefBlendParameterValues(
  layers: readonly GPUReliefBlendLayerSettings[],
  target: Float32Array = new Float32Array(layers.length * GPU_RELIEF_BLEND_LAYER_PARAMETER_LENGTH)
): Float32Array {
  if (layers.length < 1 || layers.length > GPU_RELIEF_BLEND_MAX_LAYER_COUNT) {
    throw new Error('Relief blend supports 1 to 5 layers');
  }
  if (target.length < layers.length * GPU_RELIEF_BLEND_LAYER_PARAMETER_LENGTH) {
    throw new Error('Relief blend settings target must hold 5 values per layer');
  }
  for (const [layer, settings] of layers.entries()) {
    const opacity = settings.opacity ?? 1;
    if (![settings.minimum, settings.maximum, opacity].every(Number.isFinite)) {
      throw new Error('Relief blend settings must be finite');
    }
    if (settings.maximum <= settings.minimum) {
      throw new Error('Relief blend layer maximum must be greater than minimum');
    }
    if (opacity < 0 || opacity > 1) {
      throw new Error('Relief blend opacity must be in [0, 1]');
    }
    const code = GPU_RELIEF_BLEND_MODE_CODES[settings.blendMode ?? 'normal'];
    if (code === undefined) {
      throw new Error('Relief blend blendMode is unknown');
    }
    const base = layer * GPU_RELIEF_BLEND_LAYER_PARAMETER_LENGTH;
    target[base] = settings.minimum;
    target[base + 1] = settings.maximum;
    target[base + 2] = settings.invert ? 1 : 0;
    target[base + 3] = code;
    target[base + 4] = opacity;
  }
  return target;
}

/**
 * RVT's "Archaeological (VAT)" combination (Kokalj and Somrak 2019), general terrain stretches,
 * ordered bottom to top.
 *
 * RVT lists layers top first: SVF `[0.7, 1]` multiply 25 %, positive openness `[68, 93]` overlay
 * 50 %, slope `[0, 50]` luminosity 50 %, hillshade `[0, 1]` normal 100 %. Here the hillshade is
 * index 0. Inputs, in order: hillshade, slope gradient in degrees (inverted), positive openness
 * in degrees, sky-view factor.
 */
export const GPU_RELIEF_BLEND_VAT_ARCHAEOLOGICAL: readonly GPUReliefBlendLayerSettings[] =
  Object.freeze([
    {minimum: 0, maximum: 1, blendMode: 'normal', opacity: 1},
    {minimum: 0, maximum: 50, invert: true, blendMode: 'luminosity', opacity: 0.5},
    {minimum: 68, maximum: 93, blendMode: 'overlay', opacity: 0.5},
    {minimum: 0.7, maximum: 1, blendMode: 'multiply', opacity: 0.25}
  ]);

/**
 * The same combination with RVT's "flat" terrain stretches: slope `[0, 15]`, positive openness
 * `[85, 93]`, sky-view factor `[0.9, 1]`, hillshade `[0, 1]`. Same input order as
 * {@link GPU_RELIEF_BLEND_VAT_ARCHAEOLOGICAL}.
 */
export const GPU_RELIEF_BLEND_VAT_FLAT: readonly GPUReliefBlendLayerSettings[] = Object.freeze([
  {minimum: 0, maximum: 1, blendMode: 'normal', opacity: 1},
  {minimum: 0, maximum: 15, invert: true, blendMode: 'luminosity', opacity: 0.5},
  {minimum: 85, maximum: 93, blendMode: 'overlay', opacity: 0.5},
  {minimum: 0.9, maximum: 1, blendMode: 'multiply', opacity: 0.25}
]);

/**
 * Properties for {@link GPUReliefBlend}.
 *
 * Topology: grid size, the number of layers (`layers.length`) and which outputs exist.
 * Per-frame: `settings` (stretch, inversion, blend mode and opacity per layer) and layer contents.
 *
 * Cell size model: pixel units, cell size not used; layers are already-computed per-pixel values.
 */
export type GPUReliefBlendProps = {
  /** Prefix for node and transient IDs. Defaults to `'relief-blend'`. */
  id?: string;
  /** Grid width in pixels. */
  width: number;
  /** Grid height in pixels. */
  height: number;
  /**
   * One to five float32 layers of `width * height` values, ordered BOTTOM to TOP. RVT lists its
   * layers top first, so reverse an RVT list to use it here.
   */
  layers: readonly GraphDataView<'float32'>[];
  /** Per-frame settings with at least `5 * layers.length` float32 values. */
  settings: GraphDataView<'float32'>;
  /** Optional blended greyscale in `[0, 1]`, NaN where any layer is not finite. */
  blend?: GraphDataView<'float32'>;
  /** Optional packed RGBA8 grey (`pack4x8unorm`, red in the low byte, alpha 255; 0 where invalid). */
  color?: GraphDataView<'uint32'>;
  /** Optional per-pixel 1 where every layer is finite, else 0. */
  validity?: GraphDataView<'uint32'>;
};

/**
 * Blends up to five single-band visualizations into one greyscale image, the Visualization for
 * Archaeological Topography (VAT) workflow of Kokalj and Somrak (2019), following RVT's
 * `BlenderCombination.render_all_images`.
 *
 * Each layer is normalized `clamp((x - min) / (max - min), 0, 1)`, optionally inverted, then
 * rendered from the bottom up: the bottom layer is used as is, and every next layer `a` over the
 * background `b` gives `top = mode(a, b)` and `b' = top * opacity + b * (1 - opacity)`. Modes:
 * multiply `a b`, screen `1 - (1 - a)(1 - b)`, overlay (switches on the BACKGROUND, `b > 0.5`:
 * `1 - (1 - 2 (b - 0.5))(1 - a)`, else `2 b a`), soft light (switches on the ACTIVE layer,
 * `a < 0.5`: `2 b a + b^2 (1 - 2 a)`, else `2 b (1 - a) + sqrt(b)(2 a - 1)`), luminosity and
 * normal `a`.
 *
 * RVT's single-band overlay and soft light mutate the background array in place, so its
 * `render_images` then blends the result with itself and the layer opacity has no effect for
 * those two modes. This recipe applies the documented opacity to every mode instead.
 *
 * Percentile ("percent") stretches are not implemented because they need a histogram; callers
 * pass the numeric `minimum` and `maximum`, for example from `GPURasterStatistics`.
 */
export class GPUReliefBlend implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUReliefBlendProps;
  /** Per-pixel operator: no neighbourhood, so no halo. */
  readonly requiredHalo = 0;

  constructor(props: GPUReliefBlendProps) {
    this.id = props.id ?? 'relief-blend';
    this.props = props;
    const {id} = this;
    const pixelCount = validateTerrainGrid(id, props.width, props.height);
    const layerCount = props.layers.length;
    if (layerCount < 1 || layerCount > GPU_RELIEF_BLEND_MAX_LAYER_COUNT) {
      throw new Error(`${id} layers must contain 1 to ${GPU_RELIEF_BLEND_MAX_LAYER_COUNT} views`);
    }
    for (const [layer, view] of props.layers.entries()) {
      validateTerrainIlluminationView(id, `layer ${layer}`, view, 'float32', pixelCount);
    }
    if (!props.blend && !props.color && !props.validity) {
      throw new Error(`${id} requires at least one output`);
    }
    validateTerrainIlluminationView(id, 'blend', props.blend, 'float32', pixelCount);
    validateTerrainIlluminationView(id, 'color', props.color, 'uint32', pixelCount);
    validateTerrainIlluminationView(id, 'validity', props.validity, 'uint32', pixelCount);
    validateTerrainSettings(
      id,
      props.settings,
      layerCount * GPU_RELIEF_BLEND_LAYER_PARAMETER_LENGTH
    );
    validateTerrainBuffersDistinct(
      id,
      [props.blend, props.color, props.validity],
      [...props.layers, props.settings]
    );
  }

  /** Returns the blend node and, when color or validity is requested, one encode node. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const pixelCount = props.width * props.height;
    validateGraphViewsBelongToGraph(id, graph, [
      ...props.layers,
      props.settings,
      props.blend,
      props.color,
      props.validity
    ]);
    const target =
      props.blend ??
      (props.color || props.validity
        ? createTransientView(graph, `${id}-blend`, 'float32', pixelCount)
        : undefined);
    const bindings: WGSLKernelBinding[] = [
      ...props.layers.map(
        (view, layer): WGSLKernelBinding => ({
          name: `layer${layer}`,
          view,
          type: 'f32',
          access: 'read'
        })
      ),
      {name: 'settings', view: props.settings, type: 'f32', access: 'read'},
      {name: 'blend', view: target!, type: 'f32', access: 'read_write'}
    ];
    const statements = props.layers.map((_, layer) => {
      const base = layer * GPU_RELIEF_BLEND_LAYER_PARAMETER_LENGTH;
      const normalized = `normalizeLayer(layer${layer}[layer${layer}Offset + index], ${base}u)`;
      return layer === 0
        ? `allFinite = allFinite && isFiniteValue(layer0[layer0Offset + index]);
  var background = ${normalized};`
        : `allFinite = allFinite && isFiniteValue(layer${layer}[layer${layer}Offset + index]);
  background = renderLayer(${normalized}, background, ${base}u);`;
    });
    const nodes: GPUCommandNode<Parameters>[] = [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-blend`,
        operation: 'GPUReliefBlend',
        variant: 'blend',
        bindings,
        invocationCount: pixelCount,
        declarations: `${TERRAIN_ILLUMINATION_WGSL_CONSTANTS}
fn normalizeLayer(value: f32, base: u32) -> f32 {
  let minimum = settings[settingsOffset + base];
  let maximum = settings[settingsOffset + base + 1u];
  let normalized = clamp((value - minimum) / (maximum - minimum), 0.0, 1.0);
  return select(normalized, 1.0 - normalized, settings[settingsOffset + base + 2u] > 0.5);
}
fn blendLayer(mode: u32, foreground: f32, background: f32) -> f32 {
  if (mode == 1u) {
    return foreground * background;
  }
  if (mode == 2u) {
    return 1.0 - (1.0 - foreground) * (1.0 - background);
  }
  if (mode == 3u) {
    return select(
      2.0 * background * foreground,
      1.0 - (1.0 - 2.0 * (background - 0.5)) * (1.0 - foreground),
      background > 0.5
    );
  }
  if (mode == 4u) {
    return select(
      2.0 * background * (1.0 - foreground) + sqrt(background) * (2.0 * foreground - 1.0),
      2.0 * background * foreground + background * background * (1.0 - 2.0 * foreground),
      foreground < 0.5
    );
  }
  return foreground;
}
fn renderLayer(foreground: f32, background: f32, base: u32) -> f32 {
  let mode = u32(settings[settingsOffset + base + 3u] + 0.5);
  let opacity = settings[settingsOffset + base + 4u];
  let top = blendLayer(mode, foreground, background);
  return top * opacity + background * (1.0 - opacity);
}`,
        body: `var allFinite = true;
  ${statements.join('\n  ')}
  blend[blendOffset + index] = select(getNaN(index), background, allFinite);`
      })
    ];
    if (props.color || props.validity) {
      const encodeBindings: WGSLKernelBinding[] = [
        {name: 'blend', view: target!, type: 'f32', access: 'read'}
      ];
      if (props.color) {
        encodeBindings.push({name: 'color', view: props.color, type: 'u32', access: 'read_write'});
      }
      if (props.validity) {
        encodeBindings.push({
          name: 'validityValues',
          view: props.validity,
          type: 'u32',
          access: 'read_write'
        });
      }
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-encode`,
          operation: 'GPUReliefBlend',
          variant: 'encode',
          bindings: encodeBindings,
          invocationCount: pixelCount,
          declarations: TERRAIN_ILLUMINATION_WGSL_CONSTANTS,
          body: `let value = blend[blendOffset + index];
  let isValid = isFiniteValue(value);
  let grey = clamp(value, 0.0, 1.0);
  ${props.color ? 'color[colorOffset + index] = select(0u, pack4x8unorm(vec4<f32>(grey, grey, grey, 1.0)), isValid);' : ''}
  ${props.validity ? 'validityValues[validityValuesOffset + index] = select(0u, 1u, isValid);' : ''}`
        })
      );
    }
    return nodes;
  }
}
