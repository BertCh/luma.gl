// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device, type Texture, type TextureFormatColor} from '@luma.gl/core';
import {type ModelProps, ShaderPassRenderer} from '@luma.gl/engine';
import type {Light} from '@luma.gl/shadertools';
import {Matrix4, type NumberArray3} from '@math.gl/core';
import {
  createDeferredLightingCompositeShaderPass,
  type DeferredPointLight,
  MAX_DEFERRED_POINT_LIGHTS,
  makeDeferredPointLightBufferData
} from '../rendering/deferred-lighting';
import {GBuffer} from '../rendering/g-buffer';
import {DEFERRED_SCENE_WGSL_SHADER} from './deferred-scene-shaders';
import {
  getSceneAlphaMode,
  SceneRenderer,
  type SceneRenderOptions,
  type SceneRenderStatistics,
  type SceneSurface
} from './scene-renderer';

const COLOR_ATTACHMENT_FORMATS = [
  'rgba16float',
  'rgba8unorm',
  'rgba8unorm',
  'rgba16float'
] satisfies (TextureFormatColor | null)[];

/** Deferred G-buffer support, including the first unsupported scene feature. */
export type DeferredSceneSupport =
  | {supported: true}
  | {supported: false; feature: string; reason: string};

/** Returns whether a scene can be represented by the shared metallic-roughness G-buffer. */
export function supportsDeferredScene(options: SceneRenderOptions): boolean {
  return getDeferredSceneSupport(options).supported;
}

/** Returns whether a scene fits the deferred G-buffer and, if not, the first failing feature. */
export function getDeferredSceneSupport(options: SceneRenderOptions): DeferredSceneSupport {
  if (options.renderMode && options.renderMode !== 'default') {
    return {
      supported: false,
      feature: 'render-mode',
      reason: 'Deferred rendering requires the default render mode.'
    };
  }
  let directionalLightCount = 0;
  let pointLightCount = 0;
  for (const light of options.lights || []) {
    if (light.type === 'spot') {
      return {
        supported: false,
        feature: 'spot-light',
        reason: 'Spot lights require forward rendering.'
      };
    }
    if (light.type === 'directional' && ++directionalLightCount > 1) {
      return {
        supported: false,
        feature: 'directional-light-count',
        reason: 'Deferred rendering supports one directional light.'
      };
    }
    if (light.type === 'point' && ++pointLightCount > MAX_DEFERRED_POINT_LIGHTS) {
      return {
        supported: false,
        feature: 'point-light-count',
        reason: 'The point light count exceeds the deferred limit.'
      };
    }
  }
  if (
    options.environment?.diffuseTexture ||
    options.environment?.specularTexture ||
    options.environment?.brdfLUTTexture
  ) {
    return {
      supported: false,
      feature: 'image-based-lighting',
      reason: 'Environment illumination requires forward rendering.'
    };
  }

  for (const [index, surface] of options.surfaces.entries()) {
    const uniforms = surface.material.uniforms || {};
    const bindings = surface.material.bindings || {};
    if (getSceneAlphaMode(surface.material) === 'BLEND') {
      return {
        supported: false,
        feature: `surface:${surface.id ?? index}/alpha-blend`,
        reason: 'Alpha blending requires forward rendering.'
      };
    }
    if (uniforms.unlit) {
      return {
        supported: false,
        feature: `surface:${surface.id ?? index}/unlit`,
        reason: 'Unlit shading requires forward rendering.'
      };
    }
    if (uniforms.transmissionFactor && uniforms.transmissionFactor > 0) {
      return {
        supported: false,
        feature: `surface:${surface.id ?? index}/transmission`,
        reason: 'Transmission requires forward rendering.'
      };
    }
    if (uniforms.diffuseTransmissionFactor && uniforms.diffuseTransmissionFactor > 0) {
      return {
        supported: false,
        feature: `surface:${surface.id ?? index}/diffuse-transmission`,
        reason: 'Diffuse transmission requires forward rendering.'
      };
    }
    if ((uniforms.multiscatterColorFactor || []).some(component => component > 0)) {
      return {
        supported: false,
        feature: `surface:${surface.id ?? index}/volume-scattering`,
        reason: 'Volume scattering requires forward rendering.'
      };
    }
    if (uniforms.bumpMapEnabled) {
      return {
        supported: false,
        feature: `surface:${surface.id ?? index}/bump-map`,
        reason: 'Bump mapping requires forward rendering.'
      };
    }
    if (bindings.pbr_bumpSampler) {
      return {
        supported: false,
        feature: `surface:${surface.id ?? index}/bump-map-binding`,
        reason: 'Bump textures requires forward rendering.'
      };
    }
    if (uniforms.thicknessFactor && uniforms.thicknessFactor > 0) {
      return {
        supported: false,
        feature: `surface:${surface.id ?? index}/volume`,
        reason: 'Volume thickness requires forward rendering.'
      };
    }
    if (uniforms.clearcoatFactor && uniforms.clearcoatFactor > 0) {
      return {
        supported: false,
        feature: `surface:${surface.id ?? index}/clearcoat`,
        reason: 'Clearcoat requires forward rendering.'
      };
    }
    if (uniforms.iridescenceFactor && uniforms.iridescenceFactor > 0) {
      return {
        supported: false,
        feature: `surface:${surface.id ?? index}/iridescence`,
        reason: 'Iridescence requires forward rendering.'
      };
    }
    if (uniforms.anisotropyStrength && uniforms.anisotropyStrength > 0) {
      return {
        supported: false,
        feature: `surface:${surface.id ?? index}/anisotropy`,
        reason: 'Anisotropy requires forward rendering.'
      };
    }
    if ((uniforms.sheenColorFactor || []).some(component => component > 0)) {
      return {
        supported: false,
        feature: `surface:${surface.id ?? index}/sheen`,
        reason: 'Sheen requires forward rendering.'
      };
    }
    if (uniforms.ior !== undefined && uniforms.ior !== 1.5) {
      return {
        supported: false,
        feature: `surface:${surface.id ?? index}/ior`,
        reason: 'Non-default index of refraction requires forward rendering.'
      };
    }
    if (uniforms.specularIntensityFactor !== undefined && uniforms.specularIntensityFactor !== 1) {
      return {
        supported: false,
        feature: `surface:${surface.id ?? index}/specular-intensity`,
        reason: 'Non-default specular intensity requires forward rendering.'
      };
    }
    if (!(uniforms.specularColorFactor || [1, 1, 1]).every(component => component === 1)) {
      return {
        supported: false,
        feature: `surface:${surface.id ?? index}/specular-color`,
        reason: 'Non-default specular color requires forward rendering.'
      };
    }
    if (uniforms.specularColorMapEnabled) {
      return {
        supported: false,
        feature: `surface:${surface.id ?? index}/specular-color-map`,
        reason: 'Specular color maps requires forward rendering.'
      };
    }
    if (uniforms.specularIntensityMapEnabled) {
      return {
        supported: false,
        feature: `surface:${surface.id ?? index}/specular-intensity-map`,
        reason: 'Specular intensity maps requires forward rendering.'
      };
    }
    if (bindings.pbr_specularColorSampler) {
      return {
        supported: false,
        feature: `surface:${surface.id ?? index}/specular-color-binding`,
        reason: 'Specular color textures requires forward rendering.'
      };
    }
    if (bindings.pbr_specularIntensitySampler) {
      return {
        supported: false,
        feature: `surface:${surface.id ?? index}/specular-intensity-binding`,
        reason: 'Specular intensity textures requires forward rendering.'
      };
    }
  }
  return {supported: true};
}

/**
 * Captures generic PBR surfaces into a WebGPU G-buffer and resolves scene lighting.
 *
 * Scenes requiring physical extensions unsupported by the G-buffer transparently use the shared
 * forward renderer while retaining the same scene descriptor and statistics contract.
 */
export class DeferredSceneRenderer extends SceneRenderer {
  private readonly buffers = new Map<string, GBuffer>();
  private readonly pointLightBuffer: Buffer;
  private readonly lightingRenderer: ShaderPassRenderer;
  private forwardRenderer: SceneRenderer | null = null;
  private readonly lastDeferredFrameIdentifiers = new Set<string>();

  constructor(device: Device) {
    if (device.type !== 'webgpu') {
      throw new Error('Deferred scene rendering requires a WebGPU device.');
    }
    super(device);
    this.pointLightBuffer = device.createBuffer({
      id: 'deferred-scene-point-lights',
      data: makeDeferredPointLightBufferData([], MAX_DEFERRED_POINT_LIGHTS),
      usage: Buffer.STORAGE | Buffer.COPY_DST
    });
    this.lightingRenderer = new ShaderPassRenderer(device, {
      shaderPasses: [createDeferredLightingCompositeShaderPass()],
      colorFormat: 'rgba16float',
      flipY: true
    });
  }

  /** Draws compatible scenes through deferred lighting and falls back for advanced materials. */
  override render(options: SceneRenderOptions): SceneRenderStatistics {
    if (!supportsDeferredScene(options)) {
      this.lastDeferredFrameIdentifiers.delete(options.id);
      this.forwardRenderer ||= new SceneRenderer(this.device);
      return this.forwardRenderer.render(options);
    }

    const [width, height] = getSceneSize(this.device, options);
    const gBuffer = this.getGBuffer(options.id, width, height);
    this.lastDeferredFrameIdentifiers.add(options.id);
    const scene = this.prepareScene(options);
    const background = options.background || [0, 0, 0, 1];
    const renderPass = this.device.beginRenderPass({
      id: `scene-${options.id}-deferred-gbuffer`,
      framebuffer: gBuffer.framebuffer,
      clearColors: [
        new Float32Array([background[0], background[1], background[2], background[3] ?? 1]),
        new Float32Array([0.5, 0.5, 1, 1]),
        new Float32Array([0, 0, 0, 0]),
        new Float32Array([0, 0, 0, 0])
      ],
      clearDepth: 1
    });
    scene.statistics.drawCount = this.drawPreparedScene(scene, renderPass);
    renderPass.end();

    const lights = getDeferredSceneLights(
      options.lights || [],
      new Matrix4(options.camera.viewMatrix)
    );
    this.pointLightBuffer.write(
      makeDeferredPointLightBufferData(lights.pointLights, MAX_DEFERRED_POINT_LIGHTS)
    );
    this.lightingRenderer.resize([width, height]);
    const lightingOptions = {
      sourceTexture: gBuffer.colorTexture,
      bindings: {
        depthTexture: gBuffer.depthTexture,
        normalTexture: gBuffer.normalRoughnessTexture,
        baseColorMetallicTexture: gBuffer.getExtraColorTexture('baseColorMetallic'),
        emissiveOcclusionTexture: gBuffer.getExtraColorTexture('emissiveOcclusion'),
        pointLights: this.pointLightBuffer
      },
      uniforms: {
        deferredLighting: {
          inverseProjectionMatrix: new Matrix4(options.camera.projectionMatrix).invert(),
          ambientColor: lights.ambientColor,
          exposure: options.exposure ?? 1,
          fogColor: options.fogColor || [0, 0, 0],
          fogDensity: options.fogDensity ?? 0,
          directionalLightDirectionView: lights.directionalLightDirectionView,
          directionalLightColor: lights.directionalLightColor,
          directionalLightIntensity: lights.directionalLightIntensity,
          pointLightCount: lights.pointLights.length
        }
      }
    };

    if (options.framebuffer) {
      const lightingTexture = this.lightingRenderer.renderToTexture(lightingOptions);
      if (lightingTexture) {
        const presentationModel = this.lightingRenderer.textureModel;
        presentationModel.setProps({backgroundTexture: lightingTexture});
        presentationModel.predraw(this.device.commandEncoder);
        const presentationPass = this.device.beginRenderPass({
          id: `scene-${options.id}-deferred-resolve`,
          framebuffer: options.framebuffer,
          clearDepth: false
        });
        presentationModel.draw(presentationPass);
        presentationPass.end();
      }
    } else {
      this.lightingRenderer.renderToScreen(lightingOptions);
    }
    return scene.statistics;
  }

  /**
   * Returns the sampleable depth written by the last compatible deferred render for a frame.
   *
   * A null result means the last render used the forward fallback, so callers should sample
   * their forward framebuffer depth instead.
   *
   * @internal
   */
  getLastDepthTexture(frameIdentifier: string): Texture | null {
    if (!this.lastDeferredFrameIdentifiers.has(frameIdentifier)) {
      return null;
    }
    return this.buffers.get(frameIdentifier)?.depthTexture || null;
  }

  /** Releases cached forward/deferred models and G-buffer attachments for one frame. */
  override destroyFrame(frameIdentifier: string): void {
    super.destroyFrame(frameIdentifier);
    this.forwardRenderer?.destroyFrame(frameIdentifier);
    this.buffers.get(frameIdentifier)?.destroy();
    this.buffers.delete(frameIdentifier);
    this.lastDeferredFrameIdentifiers.delete(frameIdentifier);
  }

  /** Releases all cached G-buffers, lighting resources, and fallback forward models. */
  override destroy(): void {
    super.destroy();
    this.forwardRenderer?.destroy();
    for (const buffer of this.buffers.values()) {
      buffer.destroy();
    }
    this.buffers.clear();
    this.lastDeferredFrameIdentifiers.clear();
    this.lightingRenderer.destroy();
    this.pointLightBuffer.destroy();
  }

  /** Reuses shared material creation and instancing while specializing MRT output. */
  protected override getSurfaceModelOptions(
    _surface: SceneSurface,
    _options: SceneRenderOptions
  ): Partial<ModelProps> {
    return {
      source: DEFERRED_SCENE_WGSL_SHADER,
      colorAttachmentFormats: COLOR_ATTACHMENT_FORMATS,
      depthStencilAttachmentFormat: 'depth24plus'
    };
  }

  private getGBuffer(frameIdentifier: string, width: number, height: number): GBuffer {
    let buffer = this.buffers.get(frameIdentifier);
    if (!buffer) {
      buffer = new GBuffer(this.device, {
        id: `scene-${frameIdentifier}-deferred`,
        width,
        height,
        colorFormat: 'rgba16float',
        normalRoughnessFormat: 'rgba8unorm',
        velocity: false,
        depthStencilFormat: 'depth24plus',
        extraColorAttachments: [
          {name: 'baseColorMetallic', format: 'rgba8unorm'},
          {name: 'emissiveOcclusion', format: 'rgba16float'}
        ]
      });
      this.buffers.set(frameIdentifier, buffer);
    } else {
      buffer.resize({width, height});
    }
    return buffer;
  }
}

function getSceneSize(device: Device, options: SceneRenderOptions): [number, number] {
  if (options.framebuffer) {
    return [options.framebuffer.width, options.framebuffer.height];
  }
  if (options.width && options.height) {
    return [options.width, options.height];
  }
  return device.getDefaultCanvasContext().getDrawingBufferSize();
}

function getDeferredSceneLights(
  lights: readonly Light[],
  viewMatrix: Matrix4
): {
  ambientColor: NumberArray3;
  directionalLightDirectionView: NumberArray3;
  directionalLightColor: NumberArray3;
  directionalLightIntensity: number;
  pointLights: DeferredPointLight[];
} {
  const ambientColor: NumberArray3 = [0, 0, 0];
  const directionalLightDirectionView: NumberArray3 = [0, 0, 1];
  const directionalLightColor: NumberArray3 = [1, 1, 1];
  const pointLights: DeferredPointLight[] = [];
  let directionalLightIntensity = 0;

  for (const light of lights) {
    const color = normalizeSceneLightColor(light.color || [1, 1, 1]);
    const intensity = light.intensity ?? 1;
    switch (light.type) {
      case 'ambient':
        ambientColor[0] += color[0] * intensity;
        ambientColor[1] += color[1] * intensity;
        ambientColor[2] += color[2] * intensity;
        break;
      case 'directional': {
        const direction = viewMatrix.transformAsVector(light.direction);
        const length = Math.hypot(direction[0], direction[1], direction[2]) || 1;
        directionalLightDirectionView[0] = -direction[0] / length;
        directionalLightDirectionView[1] = -direction[1] / length;
        directionalLightDirectionView[2] = -direction[2] / length;
        directionalLightColor[0] = color[0];
        directionalLightColor[1] = color[1];
        directionalLightColor[2] = color[2];
        directionalLightIntensity = intensity;
        break;
      }
      case 'point':
        if (pointLights.length < MAX_DEFERRED_POINT_LIGHTS) {
          const position = viewMatrix.transformAsPoint(light.position);
          pointLights.push({
            position: [position[0], position[1], position[2]],
            range: Math.max(4, Math.sqrt(Math.max(intensity, 0)) * 3),
            color,
            intensity
          });
        }
        break;
    }
  }

  return {
    ambientColor,
    directionalLightDirectionView,
    directionalLightColor,
    directionalLightIntensity,
    pointLights
  };
}

function normalizeSceneLightColor(color: Readonly<NumberArray3>): [number, number, number] {
  const scale = color[0] > 1 || color[1] > 1 || color[2] > 1 ? 1 / 255 : 1;
  return [color[0] * scale, color[1] * scale, color[2] * scale];
}
