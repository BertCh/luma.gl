// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {
  createGPUComputeCommandNode,
  createGPUCopyCommandNode,
  createGPURenderCommandNode,
  GraphVectorView,
  validatePackedUint32View,
  type CompiledGPUCommandGraph,
  type GPUCommandGraph,
  type GPUCommandGraphEncoding,
  type GPUCommandNode,
  type GPUComputeCommandNode,
  type GPUCopyCommandNode,
  type GPURenderCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {getGPUVectorFormatInfo, type GPUVectorFormat} from '@luma.gl/gpgpu/gpu-data';
import type {GPUCompactOutput} from './gpu-contributor-types';

/** Scalar formats supported by {@link GPUParameterBuffer}. */
export type GPUParameterFormat = 'float32' | 'uint32' | 'sint32';

/** Properties for one application-owned, per-frame parameter buffer. */
export type GPUParameterBufferProps<Format extends GPUParameterFormat> = {
  /** Buffer ID, also used as the graph resource ID by {@link GPUParameterBuffer.importToGraph}. */
  id: string;
  /** Packed scalar format of every element. */
  format: Format;
  /** Element count. Fixed for the lifetime of any compiled graph that imports the buffer. */
  length: number;
  /** Optional initial values. Unwritten elements are zero. */
  values?: Float32Array | Uint32Array | Int32Array;
};

/**
 * A small application-owned storage buffer for values that change between encodings.
 *
 * Recipes read viewports, time windows, thresholds, and similar per-frame state from a packed
 * storage view instead of baking them into WGSL, so the application can call {@link write} and
 * encode the same compiled graph again without recompiling. Like `DrawCommandBuffer`, the buffer is
 * created on a device and imported into one or more graphs, which never destroy it.
 */
export class GPUParameterBuffer<
  Format extends GPUParameterFormat = GPUParameterFormat
> {
  /** Buffer and default graph resource ID. */
  readonly id: string;
  /** Packed scalar format of every element. */
  readonly format: Format;
  /** Element count. */
  readonly length: number;
  /** Owned backing buffer with storage, copy-source, and copy-destination usage. */
  readonly buffer: Buffer;

  constructor(device: Device, props: GPUParameterBufferProps<Format>) {
    this.id = props.id;
    this.format = props.format;
    this.length = props.length;
    this.buffer = device.createBuffer({
      id: props.id,
      byteLength: Math.max(props.length, 1) * Uint32Array.BYTES_PER_ELEMENT,
      usage: Buffer.STORAGE | Buffer.COPY_DST | Buffer.COPY_SRC
    });
    if (props.values) {
      this.buffer.write(props.values);
    }
  }

  /**
   * Imports the buffer into a graph and returns a packed view to pass into recipe props.
   *
   * @param graph Graph that reads the parameters.
   * @param id Graph resource ID. Defaults to the buffer ID.
   */
  importToGraph<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    id: string = this.id
  ): GraphDataView<Format> {
    return importGraphBuffer(graph, id, this.buffer, this.format, this.length);
  }

  /** Overwrites elements starting at `firstElement`; visible to the next submitted encoding. */
  write(values: Float32Array | Uint32Array | Int32Array, firstElement: number = 0): void {
    this.buffer.write(values, firstElement * Uint32Array.BYTES_PER_ELEMENT);
  }

  /** Destroys the owned backing buffer. Destroy compiled graphs that use it first. */
  destroy(): void {
    this.buffer.destroy();
  }
}

/**
 * Imports one caller-owned buffer into a graph and returns a packed typed view over it.
 *
 * @param graph Graph that receives the imported buffer handle.
 * @param id Graph resource ID. Must be unique within the graph.
 * @param buffer Caller-owned buffer. The graph never destroys it.
 * @param format Fixed-width element format.
 * @param length Row count. Defaults to the number of whole rows the buffer can hold.
 */
export function importGraphBuffer<Format extends GPUVectorFormat, Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  buffer: Buffer,
  format: Format,
  length?: number
): GraphDataView<Format> {
  const rowByteLength = getGPUVectorFormatInfo(format).byteLength;
  const handle = graph.importBuffer(
    {id, byteLength: buffer.byteLength, usage: buffer.usage},
    buffer
  );
  return graph.createDataView(handle, {
    format,
    length: length ?? Math.floor(buffer.byteLength / rowByteLength)
  });
}

/** Returns the ordered source chunks of a data or vector view without repacking them. */
export function getGraphViewChunks<Format extends GPUVectorFormat>(
  view: GraphDataView<Format> | GraphVectorView<Format>
): readonly GraphDataView<Format>[] {
  return view instanceof GraphVectorView ? view.data : [view];
}

/** Throws when any view or vector chunk belongs to a different graph than `graph`. */
export function validateGraphViewsBelongToGraph<Parameters>(
  id: string,
  graph: GPUCommandGraph<Parameters>,
  views: readonly (GraphDataView | GraphVectorView | undefined)[]
): void {
  for (const view of views) {
    if (!view) {
      continue;
    }
    for (const chunk of getGraphViewChunks(view)) {
      if (chunk.buffer.graph !== graph) {
        // Recipe inputs and outputs must be created or imported on the graph passed to graph.add().
        throw new Error(`${id} views must belong to the target graph`);
      }
    }
  }
}

/**
 * Throws when any writable view shares a graph buffer with any input view.
 *
 * Output-versus-output aliasing is left to the graph compiler, which rejects overlapping writers.
 */
export function validateGraphOutputsDisjointFromInputs(
  id: string,
  outputs: readonly (GraphDataView | GraphVectorView | undefined)[],
  inputs: readonly (GraphDataView | GraphVectorView | undefined)[]
): void {
  const inputBuffers = new Set(
    inputs.flatMap(view => (view ? getGraphViewChunks(view).map(chunk => chunk.buffer) : []))
  );
  for (const view of outputs) {
    if (view && getGraphViewChunks(view).some(chunk => inputBuffers.has(chunk.buffer))) {
      throw new Error(`${id} outputs must not share buffers with inputs`);
    }
  }
}

/**
 * Throws unless `view` has the same row count and chunk boundaries as `template`.
 *
 * A packed template requires a packed view of equal length; a vector template requires a vector
 * with identical ordered chunk lengths.
 */
export function validateGraphViewTopology(
  id: string,
  name: string,
  template: GraphDataView | GraphVectorView,
  view: GraphDataView | GraphVectorView | undefined
): void {
  if (!view) {
    return;
  }
  const templateChunks = getGraphViewChunks(template);
  const viewChunks = getGraphViewChunks(view);
  if (
    template instanceof GraphVectorView !== view instanceof GraphVectorView ||
    templateChunks.length !== viewChunks.length ||
    templateChunks.some((chunk, chunkIndex) => chunk.length !== viewChunks[chunkIndex].length)
  ) {
    throw new Error(`${id} ${name} must match the source topology`);
  }
}

/** Validates the packed layout and minimum row counts of a {@link GPUCompactOutput}. */
export function validateCompactOutput(id: string, output: GPUCompactOutput): void {
  validatePackedUint32View(output.ids, `${id} output.ids`);
  for (const [name, scalar] of [
    ['count', output.count],
    ['overflow', output.overflow],
    ['totalCount', output.totalCount]
  ] as const) {
    if (!scalar) {
      continue;
    }
    validatePackedUint32View(scalar, `${id} output.${name}`);
    if (scalar.length < 1) {
      throw new Error(`${id} output.${name} must contain one uint32 row`);
    }
  }
}

/**
 * Collects the command nodes that an `addToGraph(graph)`-style contributor schedules.
 *
 * Geospatial, raster, and projection operations in `@luma.gl/experimental` schedule nodes directly
 * through `addToGraph(graph)`, while GPU Core contributors return them from
 * `getCommandNodes(graph)`. A recipe's `getCommandNodes(graph)` wraps such calls with this helper
 * so every recipe composes through `graph.add(recipe)` in declaration order. Transient resources
 * created by `addNodes` are still declared on `graph`.
 *
 * @param graph Graph passed to the contributor.
 * @param addNodes Callback that calls one or more `addToGraph(graph)` methods.
 * @returns The scheduled nodes, in order, without adding them to `graph`.
 */
export function captureGraphCommandNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  addNodes: () => void
): GPUCommandNode<Parameters>[] {
  const nodes: GPUCommandNode<Parameters>[] = [];
  const recorders = {
    addComputePass: (node: Omit<GPUComputeCommandNode<Parameters>, 'type'>) =>
      nodes.push(createGPUComputeCommandNode(node)),
    addRenderPass: (node: Omit<GPURenderCommandNode<Parameters>, 'type'>) =>
      nodes.push(createGPURenderCommandNode(node)),
    addCopyPass: (node: Omit<GPUCopyCommandNode<Parameters>, 'type'>) =>
      nodes.push(createGPUCopyCommandNode(node))
  };
  const previousDescriptors = Object.keys(recorders).map(
    name => [name, Object.getOwnPropertyDescriptor(graph, name)] as const
  );
  for (const [name, recorder] of Object.entries(recorders)) {
    Object.defineProperty(graph, name, {configurable: true, writable: true, value: recorder});
  }
  try {
    addNodes();
  } finally {
    for (const [name, descriptor] of previousDescriptors) {
      if (descriptor) {
        Object.defineProperty(graph, name, descriptor);
      } else {
        Reflect.deleteProperty(graph, name);
      }
    }
  }
  return nodes;
}

/**
 * Application-side convenience that encodes one compiled graph into a new encoder and submits it.
 *
 * Recipes never call this. It exists for standalone use, examples, and tests.
 */
export function submitGraph<Parameters>(
  device: Device,
  compiled: CompiledGPUCommandGraph<Parameters>,
  parameters: Parameters
): GPUCommandGraphEncoding {
  const commandEncoder = device.createCommandEncoder({id: `${compiled.id}-encoder`});
  const encoding = compiled.encode(commandEncoder, {parameters});
  device.submit(commandEncoder.finish());
  return encoding;
}
