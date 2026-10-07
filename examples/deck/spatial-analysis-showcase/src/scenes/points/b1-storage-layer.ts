// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  COORDINATE_SYSTEM,
  Layer,
  project32,
  type LayerContext,
  type LayerProps,
  type UpdateParameters
} from '@deck.gl/core';
import {Buffer, type RenderPass} from '@luma.gl/core';
import {Model} from '@luma.gl/engine';

/** Uniform bytes every layer built on {@link StorageModelLayer} owns (room for a palette and config). */
export const STORAGE_MODEL_STYLE_BYTE_LENGTH = 160;
const STYLE_BYTE_LENGTH = STORAGE_MODEL_STYLE_BYTE_LENGTH;
export const STORAGE_MODEL_BLEND = {
  depthWriteEnabled: false,
  depthCompare: 'always',
  blend: true,
  blendColorOperation: 'add',
  blendAlphaOperation: 'add',
  blendColorSrcFactor: 'src-alpha',
  blendColorDstFactor: 'one-minus-src-alpha',
  blendAlphaSrcFactor: 'one',
  blendAlphaDstFactor: 'one-minus-src-alpha'
} as const;

type StorageLayerState = {model: Model; styleBuffer: Buffer};

/** Shared lifecycle: one instanced model, an owned style uniform, storage bindings from props. */
export abstract class StorageModelLayer<PropsT extends LayerProps> extends Layer<PropsT> {
  static override layerName = 'StorageModelLayer';
  static override defaultProps = {
    coordinateSystem: COORDINATE_SYSTEM.METER_OFFSETS,
    parameters: STORAGE_MODEL_BLEND
  };

  protected abstract getShaderSource(): string;
  protected abstract getVertexCount(): number;
  protected abstract getInstanceCount(): number;
  protected abstract getBindings(styleBuffer: Buffer): Record<string, Buffer>;
  protected abstract writeStyle(styleBuffer: Buffer): void;

  override getAttributeManager() {
    return null;
  }

  override initializeState({device}: LayerContext): void {
    if (device.type !== 'webgpu') throw new Error(`${this.id} requires WebGPU`);
    const styleBuffer = device.createBuffer({
      id: `${this.id}-style`,
      byteLength: STYLE_BYTE_LENGTH,
      usage: Buffer.UNIFORM | Buffer.COPY_DST
    });
    const model = new Model(device, {
      ...this.getShaders({modules: [project32], source: this.getShaderSource()}),
      id: `${this.id}-model`,
      topology: 'triangle-list',
      isInstanced: true,
      vertexCount: this.getVertexCount(),
      instanceCount: 0,
      bufferLayout: [],
      bindings: this.getBindings(styleBuffer),
      parameters: STORAGE_MODEL_BLEND
    });
    this.setState({model, styleBuffer} satisfies StorageLayerState);
  }

  override updateState(parameters: UpdateParameters<this>): void {
    super.updateState(parameters);
    const {model, styleBuffer} = this.state as StorageLayerState;
    model.setBindings(this.getBindings(styleBuffer));
  }

  override getModels(): Model[] {
    return [(this.state as StorageLayerState).model];
  }

  override draw({renderPass}: {renderPass: RenderPass}): void {
    const {model, styleBuffer} = this.state as StorageLayerState;
    this.writeStyle(styleBuffer);
    model.setInstanceCount(this.getInstanceCount());
    model.draw(renderPass);
  }

  override finalizeState(context: LayerContext): void {
    const {model, styleBuffer} = this.state as StorageLayerState;
    model.destroy();
    styleBuffer.destroy();
    super.finalizeState(context);
  }
}
