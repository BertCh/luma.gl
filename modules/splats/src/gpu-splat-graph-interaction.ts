// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  Buffer,
  BufferLayout,
  CommandEncoder,
  CompareFunction,
  Device,
  RenderPass,
  ShaderLayout,
  TextureFormatColor,
  TextureFormatDepthStencil
} from '@luma.gl/core';
import {Model, PickingManager, type PickInfo, type PickingShouldPickOptions} from '@luma.gl/engine';
import type {GPUSplatData} from './splat-data';
import type {SplatPickingInfo, SplatPickingProps} from './splat-picking';
import type {SplatMixedRenderOptions} from './splat-renderer';
import {GPUSplatGraphRenderer} from './gpu-splat-graph-renderer';
import {
  GPU_SPLAT_COMPATIBLE_RENDER_SHADER,
  GPU_SPLAT_COMPATIBLE_RENDER_SHADER_LAYOUT,
  GPU_SPLAT_FRAGMENT_SHARED_SHADER_WGSL,
  GPU_SPLAT_GRAPH_SHARED_WGSL,
  GPU_SPLAT_PROJECTED_RECORD_BYTE_LENGTH,
  GPU_SPLAT_QUAD_EXPANSION_SHADER_WGSL,
  GPU_SPLAT_RENDER_SHADER,
  GPU_SPLAT_RENDER_SHADER_LAYOUT
} from './gpu-splat-graph-shaders';

const EMPTY_GPU_SPLAT_GRAPH_PICKING_INFO: SplatPickingInfo = {
  batchIndex: null,
  rowIndex: null,
  batchRowIndex: null,
  semanticId: null
};

const GPU_SPLAT_GRAPH_PICKING_SHADER_LAYOUT = {
  attributes: [],
  bindings: [
    {name: 'graphUniforms', type: 'uniform', group: 0, location: 0},
    {name: 'projectedRecords', type: 'read-only-storage', group: 0, location: 1},
    {name: 'sortedIds', type: 'read-only-storage', group: 0, location: 2}
  ]
} satisfies ShaderLayout;

const GPU_SPLAT_GRAPH_COMPATIBLE_PICKING_SHADER_LAYOUT = {
  attributes: [
    {name: 'instanceClipCenter', location: 0, type: 'vec4<f32>', stepMode: 'instance'},
    {name: 'instancePackedRecord', location: 1, type: 'vec4<u32>', stepMode: 'instance'},
    {name: 'instanceSortedId', location: 2, type: 'u32', stepMode: 'instance'}
  ],
  bindings: [{name: 'graphUniforms', type: 'uniform', group: 0, location: 0}]
} satisfies ShaderLayout;

/** Interleaved instance stream over one sorted projected-record buffer. @internal */
const SORTED_RECORD_BUFFER_LAYOUT: BufferLayout = {
  name: 'sortedRecords',
  stepMode: 'instance',
  byteStride: GPU_SPLAT_PROJECTED_RECORD_BYTE_LENGTH,
  attributes: [
    {attribute: 'instanceClipCenter', format: 'float32x4', byteOffset: 0},
    {attribute: 'instancePackedRecord', format: 'uint32x4', byteOffset: 16}
  ]
};

/** Source-row identity for the compatibility picking stream. @internal */
const SORTED_ID_BUFFER_LAYOUT: BufferLayout = {
  name: 'sortedIds',
  stepMode: 'instance',
  attributes: [{attribute: 'instanceSortedId', format: 'uint32', byteOffset: 0}]
};

/**
 * GPU-native picking shader consuming already projected, globally sorted Gaussian records.
 *
 * Picking a volumetric primitive by first hit is ambiguous in a way picking a triangle is not: the
 * outer support of a large, nearly transparent Gaussian routinely sits in front of a small opaque
 * one while contributing almost nothing to the pixel. This pass therefore rejects any fragment
 * whose own coverage at the pixel falls below `pickingAlphaThreshold` before the depth test
 * resolves the winner, so what is picked is what is visible.
 *
 * @remarks
 * A pick resolved against accumulated transmittance rather than per-splat coverage would also
 * account for occlusion by the Gaussians in front of the candidate; that needs an ordered resolve
 * the rasterizer cannot express in a single pass and is not implemented here.
 */
export const GPU_SPLAT_GRAPH_PICKING_SHADER = /* wgsl */ `\
${GPU_SPLAT_GRAPH_SHARED_WGSL}
${GPU_SPLAT_QUAD_EXPANSION_SHADER_WGSL}
${GPU_SPLAT_FRAGMENT_SHARED_SHADER_WGSL}

@group(0) @binding(0) var<uniform> graphUniforms: GraphSplatUniforms;
@group(0) @binding(1) var<storage, read> projectedRecords: array<ProjectedSplat>;
@group(0) @binding(2) var<storage, read> sortedIds: array<u32>;

struct GraphSplatPickingFragmentInputs {
  @builtin(position) position: vec4<f32>,
  @location(0) gaussianCoordinate: vec2<f32>,
  @location(1) pixelHalfWidth: vec2<f32>,
  @location(2) alpha: f32,
  @location(3) @interpolate(flat) projectedRowIndex: u32,
};

struct GraphSplatPickingFragmentOutputs {
  @location(0) color: vec4<f32>,
  @location(1) pickingIndices: vec2<i32>,
};

@vertex
fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> GraphSplatPickingFragmentInputs {
  let projectedRowIndex = sortedIds[instanceIndex];
  let projected = projectedRecords[projectedRowIndex];
  let quad = expandSplatQuad(
    graphUniforms,
    vertexIndex,
    projected.clipCenter,
    projected.packedAxis0,
    projected.packedAxis1,
    projected.packedColorRG,
    projected.packedColorBA
  );

  var output: GraphSplatPickingFragmentInputs;
  output.position = quad.position;
  output.gaussianCoordinate = quad.gaussianCoordinate;
  output.pixelHalfWidth = quad.pixelHalfWidth;
  output.alpha = quad.color.a;
  output.projectedRowIndex = projectedRowIndex;
  return output;
}

@fragment
fn fragmentMain(input: GraphSplatPickingFragmentInputs) -> GraphSplatPickingFragmentOutputs {
  let coverage = getSplatFragmentCoverage(
    graphUniforms,
    input.gaussianCoordinate,
    input.pixelHalfWidth
  );
  let alpha = input.alpha * coverage;
  if (alpha <= 0.0 || alpha < max(graphUniforms.alphaCutoff, graphUniforms.pickingAlphaThreshold)) {
    discard;
  }

  var output: GraphSplatPickingFragmentOutputs;
  output.color = vec4<f32>(0.0);
  output.pickingIndices = vec2<i32>(i32(input.projectedRowIndex), 0);
  return output;
}
`;

/**
 * Compatibility picking shader with no storage buffers in the vertex stage.
 *
 * The sorted records and their source-row identities arrive as two instance streams, so the pick
 * still resolves to the original projected row without any vertex-stage indirection.
 */
export const GPU_SPLAT_GRAPH_COMPATIBLE_PICKING_SHADER = /* wgsl */ `\
${GPU_SPLAT_GRAPH_SHARED_WGSL}
${GPU_SPLAT_QUAD_EXPANSION_SHADER_WGSL}
${GPU_SPLAT_FRAGMENT_SHARED_SHADER_WGSL}

@group(0) @binding(0) var<uniform> graphUniforms: GraphSplatUniforms;

struct GraphSplatPickingFragmentInputs {
  @builtin(position) position: vec4<f32>,
  @location(0) gaussianCoordinate: vec2<f32>,
  @location(1) pixelHalfWidth: vec2<f32>,
  @location(2) alpha: f32,
  @location(3) @interpolate(flat) projectedRowIndex: u32,
};

struct GraphSplatPickingFragmentOutputs {
  @location(0) color: vec4<f32>,
  @location(1) pickingIndices: vec2<i32>,
};

@vertex
fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @location(0) instanceClipCenter: vec4<f32>,
  @location(1) instancePackedRecord: vec4<u32>,
  @location(2) instanceSortedId: u32
) -> GraphSplatPickingFragmentInputs {
  let quad = expandSplatQuad(
    graphUniforms,
    vertexIndex,
    instanceClipCenter,
    instancePackedRecord.x,
    instancePackedRecord.y,
    instancePackedRecord.z,
    instancePackedRecord.w
  );

  var output: GraphSplatPickingFragmentInputs;
  output.position = quad.position;
  output.gaussianCoordinate = quad.gaussianCoordinate;
  output.pixelHalfWidth = quad.pixelHalfWidth;
  output.alpha = quad.color.a;
  output.projectedRowIndex = instanceSortedId;
  return output;
}

@fragment
fn fragmentMain(input: GraphSplatPickingFragmentInputs) -> GraphSplatPickingFragmentOutputs {
  let coverage = getSplatFragmentCoverage(
    graphUniforms,
    input.gaussianCoordinate,
    input.pixelHalfWidth
  );
  let alpha = input.alpha * coverage;
  if (alpha <= 0.0 || alpha < max(graphUniforms.alphaCutoff, graphUniforms.pickingAlphaThreshold)) {
    discard;
  }

  var output: GraphSplatPickingFragmentOutputs;
  output.color = vec4<f32>(0.0);
  output.pickingIndices = vec2<i32>(i32(input.projectedRowIndex), 0);
  return output;
}
`;

/** Resolves a packed projected-row pick into its untouched, independently prepared source batch. */
export function resolveGPUSplatGraphPickInfo(
  pickInfo: PickInfo | null | undefined,
  batches: readonly GPUSplatData[]
): SplatPickingInfo {
  if (
    !pickInfo ||
    pickInfo.batchIndex !== 0 ||
    pickInfo.objectIndex === null ||
    !Number.isSafeInteger(pickInfo.objectIndex) ||
    pickInfo.objectIndex < 0
  ) {
    return {...EMPTY_GPU_SPLAT_GRAPH_PICKING_INFO};
  }

  let batchOffset = 0;
  for (const batch of batches) {
    const batchRowIndex = pickInfo.objectIndex - batchOffset;
    if (batchRowIndex < batch.rowCount) {
      if (batch.destroyed) {
        return {...EMPTY_GPU_SPLAT_GRAPH_PICKING_INFO};
      }
      return {
        batchIndex: batch.sourceBatchIndex,
        rowIndex: batch.rowIndexBase + batchRowIndex,
        batchRowIndex,
        semanticId: batch.source.semanticIds?.[batchRowIndex] ?? null
      };
    }
    batchOffset += batch.rowCount;
  }

  return {...EMPTY_GPU_SPLAT_GRAPH_PICKING_INFO};
}

/** Live graph-owned buffers one consumer model borrows, whichever render path is active. */
type GraphSplatModelSources = {
  uniformBuffer: Buffer;
  projectedRecordBuffer?: Buffer;
  sortedIndexBuffer?: Buffer;
  sortedRecordBuffer?: Buffer;
};

/** Reads the renderer's current buffers, or `undefined` before the graph has been compiled. */
function getGraphSplatModelSources(
  renderer: GPUSplatGraphRenderer
): GraphSplatModelSources | undefined {
  const uniformBuffer = renderer.uniformBuffer;
  if (!uniformBuffer) {
    return undefined;
  }
  if (renderer.renderPath === 'compatible') {
    const sortedRecordBuffer = renderer.sortedRecordBuffer;
    const sortedIndexBuffer = renderer.sortedIndexBuffer;
    if (!sortedRecordBuffer || !sortedIndexBuffer) {
      return undefined;
    }
    return {uniformBuffer, sortedRecordBuffer, sortedIndexBuffer};
  }
  const projectedRecordBuffer = renderer.projectedRecordBuffer;
  const sortedIndexBuffer = renderer.sortedIndexBuffer;
  if (!projectedRecordBuffer || !sortedIndexBuffer) {
    return undefined;
  }
  return {uniformBuffer, projectedRecordBuffer, sortedIndexBuffer};
}

/** Whether a cached consumer model still borrows exactly the renderer's live buffers. */
function areGraphSplatModelSourcesEqual(
  left: GraphSplatModelSources | undefined,
  right: GraphSplatModelSources
): boolean {
  return (
    left !== undefined &&
    left.uniformBuffer === right.uniformBuffer &&
    left.projectedRecordBuffer === right.projectedRecordBuffer &&
    left.sortedIndexBuffer === right.sortedIndexBuffer &&
    left.sortedRecordBuffer === right.sortedRecordBuffer
  );
}

/** Binds one consumer model to the renderer's current buffers for the active render path. */
function bindGraphSplatModel(
  model: Model,
  renderPath: 'storage' | 'compatible',
  sources: GraphSplatModelSources,
  renderPass: RenderPass,
  options: {includeSortedIdStream?: boolean} = {}
): void {
  renderPass.setPipeline(model.pipeline);
  if (renderPath === 'compatible') {
    model.setAttributes({
      sortedRecords: sources.sortedRecordBuffer!,
      ...(options.includeSortedIdStream ? {sortedIds: sources.sortedIndexBuffer!} : {})
    });
    renderPass.setVertexArray(model.vertexArray);
    renderPass.setBindings({graphUniforms: sources.uniformBuffer});
    return;
  }
  renderPass.setVertexArray(model.vertexArray);
  renderPass.setBindings({
    graphUniforms: sources.uniformBuffer,
    projectedRecords: sources.projectedRecordBuffer!,
    sortedIds: sources.sortedIndexBuffer!
  });
}

/** Attachment formats and shared mesh-depth policy for projected Gaussian graph composition. */
export type GPUSplatGraphMixedRendererProps = {
  /** Existing render-pass color format; defaults to the WebGPU presentation format. */
  colorAttachmentFormat?: TextureFormatColor;
  /** Existing shared render-pass depth format; defaults to the device depth format. */
  depthStencilAttachmentFormat?: TextureFormatDepthStencil;
  /** Depth comparison against opaque meshes already drawn into the same render pass. */
  depthCompare?: CompareFunction;
  /** Whether transparent Gaussian fragments should update shared scene depth. */
  depthWriteEnabled?: boolean;
};

/**
 * Composites graph-projected Gaussians between opaque and transparent meshes in one shared pass.
 *
 * Call `predraw(commandEncoder)` before opening the external render pass. Graph projection must
 * already exist or will be encoded first. A renderer constructed with `presentation: false`
 * records no draw of its own during that preparation, which is the configuration this compositor
 * is meant for: the mixed pass then reuses the projected records, global sort, and GPU-visible
 * indirect command without CPU projection, source uploads, or a discarded presentation pass.
 */
export class GPUSplatGraphMixedRenderer {
  /** WebGPU device shared by the borrowing graph renderer and mixed scene. */
  readonly device: Device;
  /** Graph renderer retaining caller-owned independently prepared source batches. */
  readonly renderer: GPUSplatGraphRenderer;
  /** Caller-selected render-pass attachment formats and Gaussian depth behavior. */
  readonly props: Required<GPUSplatGraphMixedRendererProps>;
  /** Reusable display model borrowing graph-owned projected rows, sort indices, and uniforms. */
  model?: Model;

  private modelSources?: GraphSplatModelSources;
  private isDestroyed = false;

  /** Borrows one live graph renderer without compiling a graph or allocating a display model. */
  constructor(renderer: GPUSplatGraphRenderer, props: GPUSplatGraphMixedRendererProps = {}) {
    if (renderer.destroyed) {
      throw new Error('GPUSplatGraphMixedRenderer requires a live Gaussian splat graph renderer');
    }

    this.renderer = renderer;
    this.device = renderer.device;
    this.props = {
      colorAttachmentFormat: props.colorAttachmentFormat ?? this.device.preferredColorFormat,
      depthStencilAttachmentFormat:
        props.depthStencilAttachmentFormat ??
        (this.device.preferredDepthFormat === 'depth16'
          ? 'depth24plus'
          : this.device.preferredDepthFormat),
      depthCompare: props.depthCompare ?? 'less-equal',
      depthWriteEnabled: props.depthWriteEnabled ?? false
    };
  }

  /** Whether this compositor has already released its independently owned display model. */
  get destroyed(): boolean {
    return this.isDestroyed;
  }

  /**
   * Refreshes graph projection and prepares the mixed-scene model before the caller opens a pass.
   *
   * Caller-owned command submission is preserved.
   */
  predraw(commandEncoder: CommandEncoder): boolean {
    if (this.isDestroyed || this.renderer.destroyed || this.renderer.batches.length === 0) {
      return false;
    }
    this.renderer.encode(commandEncoder);
    const model = this.getMixedModel();
    if (!model || model.pipeline.isErrored) {
      return false;
    }
    model.predraw(commandEncoder);
    return true;
  }

  /** Draws opaque meshes, one visible-row Gaussian indirect command, then transparent meshes. */
  draw(renderPass: RenderPass, options: SplatMixedRenderOptions = {}): boolean {
    if (this.isDestroyed || this.renderer.destroyed) {
      return false;
    }

    let recordedDraw = false;
    let drawSuccess = true;
    for (const mesh of options.opaqueMeshes ?? []) {
      drawSuccess = mesh.draw(renderPass) !== false && drawSuccess;
      recordedDraw = true;
    }

    const model = this.getMixedModel();
    if (
      model &&
      !model.pipeline.isErrored &&
      this.renderer.batches.length > 0 &&
      this.modelSources
    ) {
      bindGraphSplatModel(model, this.renderer.renderPath, this.modelSources, renderPass);
      this.renderer.drawCommands.draw(renderPass, 0);
      recordedDraw = true;
    }

    for (const mesh of options.transparentMeshes ?? []) {
      drawSuccess = mesh.draw(renderPass) !== false && drawSuccess;
      recordedDraw = true;
    }
    return recordedDraw && drawSuccess;
  }

  /** Releases the mixed-scene model while preserving borrowed graph and caller-owned sources. */
  destroy(): void {
    if (this.isDestroyed) {
      return;
    }
    this.model?.destroy();
    this.model = undefined;
    this.modelSources = undefined;
    this.isDestroyed = true;
  }

  private getMixedModel(): Model | undefined {
    const sources = getGraphSplatModelSources(this.renderer);
    if (!sources) {
      return undefined;
    }
    if (this.model && areGraphSplatModelSourcesEqual(this.modelSources, sources)) {
      return this.model;
    }

    this.model?.destroy();
    this.modelSources = sources;
    const isCompatible = this.renderer.renderPath === 'compatible';
    this.model = new Model(this.device, {
      id: 'gaussian-splat-graph-mixed-renderer',
      source: isCompatible ? GPU_SPLAT_COMPATIBLE_RENDER_SHADER : GPU_SPLAT_RENDER_SHADER,
      shaderLayout: isCompatible
        ? GPU_SPLAT_COMPATIBLE_RENDER_SHADER_LAYOUT
        : GPU_SPLAT_RENDER_SHADER_LAYOUT,
      ...(isCompatible
        ? {
            bufferLayout: [SORTED_RECORD_BUFFER_LAYOUT],
            attributes: {sortedRecords: sources.sortedRecordBuffer!},
            bindings: {graphUniforms: sources.uniformBuffer}
          }
        : {
            bindings: {
              graphUniforms: sources.uniformBuffer,
              projectedRecords: sources.projectedRecordBuffer!,
              sortedIds: sources.sortedIndexBuffer!
            }
          }),
      colorAttachmentFormats: [this.props.colorAttachmentFormat],
      depthStencilAttachmentFormat: this.props.depthStencilAttachmentFormat,
      isInstanced: true,
      instanceCount: this.renderer.capacity.splatCount,
      vertexCount: 4,
      topology: 'triangle-strip',
      parameters: {
        depthWriteEnabled: this.props.depthWriteEnabled,
        depthCompare: this.props.depthCompare,
        blend: true,
        blendColorOperation: 'add',
        blendAlphaOperation: 'add',
        blendColorSrcFactor: 'src-alpha',
        blendColorDstFactor: 'one-minus-src-alpha',
        blendAlphaSrcFactor: 'one',
        blendAlphaDstFactor: 'one-minus-src-alpha'
      }
    });
    return this.model;
  }
}

/**
 * WebGPU picking that reuses one graph's projected records, global sort, and indirect draw.
 *
 * Original source batches remain borrowed and unchanged. Every pick draws only the graph's
 * GPU-counted visible rows, reads one integer pixel asynchronously, and resolves original source
 * row, batch, and semantic identity without CPU projection, source uploads, or graph rebuilding.
 * A Gaussian only claims a pixel where its own coverage reaches the renderer's
 * `pickingAlphaThreshold`, so the faint outer support of a large splat cannot shadow a small
 * opaque one behind it.
 */
export class GPUSplatGraphPicker {
  /** WebGPU device shared with the graph renderer and its caller-owned source batches. */
  readonly device: Device;
  /** Renderer supplying GPU-projected source rows, global ordering, and visible indirect counts. */
  readonly renderer: GPUSplatGraphRenderer;
  /** Integer picking attachments, pointer tracking, and asynchronous single-pixel GPU readback. */
  readonly manager: PickingManager;
  /** Application-owned source-row callback and optional tooltip provider. */
  props: SplatPickingProps;
  /** Dedicated integer-picking model borrowing graph-owned buffers. */
  model?: Model;
  /** Latest resolved original source batch, global row, batch-local row, and semantic identity. */
  pickInfo: SplatPickingInfo = {...EMPTY_GPU_SPLAT_GRAPH_PICKING_INFO};

  private modelSources?: GraphSplatModelSources;
  private pickingBatches: readonly GPUSplatData[] = [];
  private pendingPickingRequest: Promise<void> = Promise.resolve();
  private pickingGeneration = 0;
  private activePickingGeneration = -1;
  private isDestroyed = false;

  /** Creates an integer picking target while leaving graph compilation and GPU drawing lazy. */
  constructor(renderer: GPUSplatGraphRenderer, props: SplatPickingProps = {}) {
    if (renderer.destroyed) {
      throw new Error('GPUSplatGraphPicker requires a live Gaussian splat graph renderer');
    }
    if (props.mode === 'color') {
      throw new Error('GPUSplatGraphPicker requires integer WebGPU picking');
    }

    this.renderer = renderer;
    this.device = renderer.device;
    this.props = props;
    this.manager = new PickingManager(this.device, {
      mode: 'index',
      onObjectPicked: this.handleObjectPicked,
      getTooltip: pickInfo =>
        this.activePickingGeneration === this.pickingGeneration
          ? (this.props.getTooltip?.(resolveGPUSplatGraphPickInfo(pickInfo, this.pickingBatches)) ??
            null)
          : null
    });
  }

  /** Graph-native Gaussian picking always uses exact signed integer WebGPU attachments. */
  get mode(): 'index' {
    return 'index';
  }

  /** Whether this picker has already released its owned framebuffer and model. */
  get destroyed(): boolean {
    return this.isDestroyed;
  }

  /**
   * Draws the current graph-visible rows and reads one stable source-row identity asynchronously.
   *
   * An unchanged cursor reuses its last result unless `force` is supplied for updated source
   * visibility, camera properties, or streamed/resident source batches.
   */
  pick(
    mousePosition: readonly [number, number] | number[] | null | undefined,
    options: PickingShouldPickOptions = {}
  ): Promise<SplatPickingInfo | null> {
    if (this.isDestroyed || this.renderer.destroyed) {
      return Promise.resolve(null);
    }
    if (!mousePosition) {
      this.clear();
      return Promise.resolve(this.pickInfo);
    }

    const requestedPosition: [number, number] = [mousePosition[0], mousePosition[1]];
    const requestedOptions = {...options};
    const requestedGeneration = this.pickingGeneration;
    const requestedPick = this.pendingPickingRequest.then(() =>
      this.pickCurrent(requestedPosition, requestedOptions, requestedGeneration)
    );
    this.pendingPickingRequest = requestedPick.then(
      () => undefined,
      () => undefined
    );
    return requestedPick;
  }

  private async pickCurrent(
    mousePosition: [number, number],
    options: PickingShouldPickOptions,
    requestedGeneration: number
  ): Promise<SplatPickingInfo | null> {
    if (this.isDestroyed || this.renderer.destroyed) {
      return null;
    }
    if (requestedGeneration !== this.pickingGeneration) {
      return this.pickInfo;
    }

    if (!this.manager.shouldPick(mousePosition, options)) {
      return this.pickInfo;
    }
    if (this.renderer.batches.length === 0) {
      this.clear();
      return this.pickInfo;
    }

    this.renderer.encode(this.device.commandEncoder);
    const model = this.getPickingModel();
    if (!model || model.pipeline.isErrored) {
      this.clear();
      return this.pickInfo;
    }

    this.activePickingGeneration = requestedGeneration;
    this.pickingBatches = [...this.renderer.batches];
    model.predraw(this.device.commandEncoder);
    const renderPass = this.manager.beginRenderPass();
    try {
      this.draw(renderPass, model);
    } finally {
      renderPass.end();
    }

    this.device.submit();
    const pickInfo = await this.manager.updatePickInfo(mousePosition);
    if (this.isDestroyed || this.renderer.destroyed) {
      return null;
    }
    if (requestedGeneration !== this.pickingGeneration) {
      this.manager.clearPickState();
      this.manager.pickInfo = {batchIndex: null, objectIndex: null};
      return this.pickInfo;
    }
    if (!pickInfo) {
      return null;
    }
    return this.updatePickingResult(pickInfo);
  }

  /** Clears a previously selected graph-projected source row and notifies its caller once. */
  clear(): void {
    this.pickingGeneration++;
    this.activePickingGeneration = -1;
    const hadSelection = this.pickInfo.rowIndex !== null;
    this.manager.clearPickState();
    this.manager.pickInfo = {batchIndex: null, objectIndex: null};
    this.pickInfo = {...EMPTY_GPU_SPLAT_GRAPH_PICKING_INFO};
    this.pickingBatches = [];
    if (hadSelection) {
      this.props.onPick?.(this.pickInfo);
    }
  }

  /** Destroys owned picking resources without touching the graph or original source allocations. */
  destroy(): void {
    if (this.isDestroyed) {
      return;
    }
    this.pickingGeneration++;
    this.activePickingGeneration = -1;
    this.model?.destroy();
    this.model = undefined;
    this.manager.destroy();
    this.pickingBatches = [];
    this.modelSources = undefined;
    this.isDestroyed = true;
  }

  private readonly handleObjectPicked = (pickInfo: PickInfo): void => {
    if (
      !this.isDestroyed &&
      !this.renderer.destroyed &&
      this.activePickingGeneration === this.pickingGeneration
    ) {
      this.updatePickingResult(pickInfo);
    }
  };

  private updatePickingResult(pickInfo: PickInfo): SplatPickingInfo {
    const resolved = resolveGPUSplatGraphPickInfo(pickInfo, this.pickingBatches);
    if (
      resolved.batchIndex !== this.pickInfo.batchIndex ||
      resolved.rowIndex !== this.pickInfo.rowIndex ||
      resolved.batchRowIndex !== this.pickInfo.batchRowIndex ||
      resolved.semanticId !== this.pickInfo.semanticId
    ) {
      this.pickInfo = resolved;
      this.props.onPick?.(resolved);
    }
    return this.pickInfo;
  }

  private getPickingModel(): Model | undefined {
    const sources = getGraphSplatModelSources(this.renderer);
    if (!sources) {
      return undefined;
    }
    if (this.model && areGraphSplatModelSourcesEqual(this.modelSources, sources)) {
      return this.model;
    }

    this.model?.destroy();
    this.modelSources = sources;
    const isCompatible = this.renderer.renderPath === 'compatible';
    this.model = new Model(this.device, {
      id: 'gaussian-splat-graph-index-picking',
      source: isCompatible
        ? GPU_SPLAT_GRAPH_COMPATIBLE_PICKING_SHADER
        : GPU_SPLAT_GRAPH_PICKING_SHADER,
      shaderLayout: isCompatible
        ? GPU_SPLAT_GRAPH_COMPATIBLE_PICKING_SHADER_LAYOUT
        : GPU_SPLAT_GRAPH_PICKING_SHADER_LAYOUT,
      ...(isCompatible
        ? {
            bufferLayout: [SORTED_RECORD_BUFFER_LAYOUT, SORTED_ID_BUFFER_LAYOUT],
            attributes: {
              sortedRecords: sources.sortedRecordBuffer!,
              sortedIds: sources.sortedIndexBuffer!
            },
            bindings: {graphUniforms: sources.uniformBuffer}
          }
        : {
            bindings: {
              graphUniforms: sources.uniformBuffer,
              projectedRecords: sources.projectedRecordBuffer!,
              sortedIds: sources.sortedIndexBuffer!
            }
          }),
      colorAttachmentFormats: ['rgba8unorm', 'rg32sint'],
      depthStencilAttachmentFormat: 'depth24plus',
      isInstanced: true,
      instanceCount: this.renderer.capacity.splatCount,
      vertexCount: 4,
      topology: 'triangle-strip',
      parameters: {depthWriteEnabled: true, depthCompare: 'less-equal', blend: false}
    });
    return this.model;
  }

  private draw(renderPass: RenderPass, model: Model): void {
    if (!this.modelSources) {
      return;
    }
    bindGraphSplatModel(model, this.renderer.renderPath, this.modelSources, renderPass, {
      includeSortedIdStream: true
    });
    this.renderer.drawCommands.draw(renderPass, 0);
  }
}
