// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert, Buffer, type Device} from '@luma.gl/core';
import {
  createGPUComputeCommandNode,
  createGPUCopyCommandNode,
  createGPURenderCommandNode,
  GraphVectorView,
  validatePackedUint32View,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUComputeCommandNode,
  type GPUCopyCommandNode,
  type GPURenderCommandNode,
  type GraphDataView,
  validatePackedView as validatePackedGraphView
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
 * Contributors read viewports, time windows, thresholds, and similar per-frame state from a packed
 * storage view instead of baking them into WGSL, so the application can call {@link write} and
 * encode the same compiled graph again without recompiling. Like `DrawCommandBuffer`, the buffer is
 * created on a device and imported into one or more graphs, which never destroy it.
 */
export class GPUParameterBuffer<Format extends GPUParameterFormat = GPUParameterFormat> {
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
   * Imports the buffer into a graph and returns a packed view to pass into contributor props.
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
    // The written range must lie inside the buffer's `length` elements.
    assert(firstElement >= 0 && firstElement + values.length <= this.length);
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
        // Contributor inputs and outputs must be created or imported on the graph passed to graph.add().
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

/**
 * Throws unless `view` is one packed, uint32-aligned view of one of `formats`. Like the gpgpu
 * `validatePackedView` (same arguments), but a chunked `GraphVectorView` is rejected with a short
 * message instead of being read as one view.
 *
 * @param view View to check.
 * @param formats Accepted formats.
 * @param name Prop name prefixed with the contributor ID, for the message.
 */
export function validatePackedView<Format extends GPUVectorFormat>(
  view: GraphDataView | GraphVectorView,
  formats: readonly Format[],
  name: string
): asserts view is GraphDataView<Format> {
  if (view instanceof GraphVectorView) {
    throw new Error(`${name} must be one packed view`);
  }
  validatePackedGraphView(view, formats, name);
}

/**
 * Throws unless a capacity or count prop is an integer in `[minimum, maximum]`.
 *
 * @param id Contributor ID, for the message.
 * @param name Prop name, for the message.
 * @param value Value to check.
 * @param minimum Smallest accepted value. Defaults to 1.
 * @param maximum Largest accepted value. Defaults to `2^32 - 1`, the largest uint32 row count.
 */
export function validateCapacity(
  id: string,
  name: string,
  value: number,
  minimum: number = 1,
  maximum: number = 0xffffffff
): void {
  if (!Number.isInteger(value) || value < minimum || value > maximum) {
    throw new Error(`${id} ${name} must be an integer in [${minimum}, ${maximum}]`);
  }
}

/**
 * Throws unless a count derived from props (rows of a view, cells of a grid, a product of
 * capacities) is at most `maximum`. Use it for documented maxima, such as the core-limit maxima of
 * {@link getCoreBindingElementLimit}; the message names the prop and the limit.
 *
 * @param id Contributor ID, for the message.
 * @param name What is counted, such as `'positions rows'` or `'indexGridSize cells'`.
 * @param value Count to check.
 * @param maximum Largest accepted count.
 */
export function validateMaximumCount(
  id: string,
  name: string,
  value: number,
  maximum: number
): void {
  if (!(value <= maximum)) {
    throw new Error(`${id} ${name} must be at most ${maximum}`);
  }
}

/**
 * Default WebGPU `maxStorageBufferBindingSize`, 128 MiB. The documented maxima of the analysis
 * contributors are sized so that every binding fits at this limit (library rule 16), so they hold
 * on every device.
 *
 * @internal
 */
export const GPU_CORE_MAX_STORAGE_BUFFER_BINDING_SIZE = 134217728;

/**
 * Largest element count `n` whose binding of `(n + extraElementCount) * bytesPerElement` bytes fits
 * the device's `maxStorageBufferBindingSize`. Contributors derive their documented maxima from it at
 * declaration (`getCommandNodes(graph)`, with `graph.device`), so a device created with larger
 * binding limits accepts larger inputs; a default device (128 MiB) gives the core values.
 *
 * @param device Device whose limit applies.
 * @param bytesPerElement Bytes of one element of the largest binding.
 * @param extraElementCount Elements the binding holds beyond `n`, such as a trailing offset.
 */
export function getStorageBindingElementLimit(
  device: Device,
  bytesPerElement: number,
  extraElementCount: number = 0
): number {
  // Element sizes and extra counts are constants of the calling contributor's layout.
  assert(bytesPerElement > 0 && extraElementCount >= 0);
  return (
    Math.floor(device.limits.maxStorageBufferBindingSize / bytesPerElement) - extraElementCount
  );
}

/**
 * Largest cell count (`gridSize[0] * gridSize[1]`) of a uniform grid index over points on
 * `device`: its `cellCount + 1` uint32 cell offsets must fit one storage binding, so `2^25 - 1` on
 * a default device. Kernel density, inverse distance weighting, kriging, spatial clustering,
 * neighbor search and trajectory encounters validate their index grid against it.
 *
 * @param device Device whose `maxStorageBufferBindingSize` applies.
 */
export function getGPUGridIndexMaximumCellCount(device: Device): number {
  return getStorageBindingElementLimit(device, 4, 1);
}

/**
 * Largest element count `n` whose binding of `(n + extraElementCount) * bytesPerElement` bytes fits
 * {@link GPU_CORE_MAX_STORAGE_BUFFER_BINDING_SIZE}. For example an offsets array of `n + 1` uint32
 * rows gives `getCoreBindingElementLimit(4, 1)`, `2^25 - 1`.
 *
 * @param bytesPerElement Bytes of one element of the largest binding.
 * @param extraElementCount Elements the binding holds beyond `n`, such as a trailing offset.
 * @internal
 */
export function getCoreBindingElementLimit(
  bytesPerElement: number,
  extraElementCount: number = 0
): number {
  // Element sizes and extra counts are constants of the calling contributor's layout.
  assert(bytesPerElement > 0 && extraElementCount >= 0);
  return Math.floor(GPU_CORE_MAX_STORAGE_BUFFER_BINDING_SIZE / bytesPerElement) - extraElementCount;
}

/**
 * Throws unless `extent` is `[minX, minY, maxX, maxY]` with finite entries and `max > min` on both
 * axes. A reversed or empty extent would mirror or collapse the grid.
 *
 * @param id Contributor ID, for the message.
 * @param extent Extent to check.
 * @param name Prop name, for the message. Defaults to `'extent'`.
 */
export function validateExtent(
  id: string,
  extent: readonly number[],
  name: string = 'extent'
): void {
  const [minX, minY, maxX, maxY] = extent;
  if (extent.length !== 4 || !extent.every(Number.isFinite) || !(maxX > minX) || !(maxY > minY)) {
    throw new Error(`${id} ${name} must be finite with max > min`);
  }
}

/**
 * Throws unless `value` is a power of two in `[minimum, maximum]`.
 *
 * @param id Contributor ID, for the message.
 * @param name Prop name, for the message.
 * @param value Value to check.
 * @param minimum Smallest accepted value. Defaults to 1.
 * @param maximum Largest accepted value. Defaults to `2^31`, the largest uint32 power of two.
 */
export function validatePowerOfTwo(
  id: string,
  name: string,
  value: number,
  minimum: number = 1,
  maximum: number = 2 ** 31
): void {
  if (
    !Number.isInteger(value) ||
    value < minimum ||
    value > maximum ||
    getNextPowerOfTwo(value) !== value
  ) {
    throw new Error(`${id} ${name} must be a power of two in [${minimum}, ${maximum}]`);
  }
}

/** Validates the packed layout and minimum row counts of a {@link GPUCompactOutput}. */
export function validateCompactOutput(id: string, output: GPUCompactOutput): void {
  validatePackedUint32View(output.ids, `${id} output.ids`);
  for (const [name, scalar] of [
    ['count', output.status.count],
    ['overflow', output.status.overflow],
    ['requiredCount', output.status.requiredCount]
  ] as const) {
    if (!scalar) {
      continue;
    }
    validatePackedUint32View(scalar, `${id} output.status.${name}`);
    if (scalar.length < 1) {
      throw new Error(`${id} output.status.${name} must contain one uint32 row`);
    }
  }
}

/**
 * Collects the command nodes that an `addToGraph(graph)`-style contributor schedules.
 *
 * Geospatial, raster, and projection operations in `@luma.gl/experimental` schedule nodes directly
 * through `addToGraph(graph)`, while GPU Core contributors return them from
 * `getCommandNodes(graph)`. A contributor's `getCommandNodes(graph)` wraps such calls with this helper
 * so every contributor composes through `graph.add(contributor)` in declaration order. Transient resources
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

/** Smallest power of two at or above `value`, and at least one. */
export function getNextPowerOfTwo(value: number): number {
  let result = 1;
  while (result < value) {
    result *= 2;
  }
  return result;
}

/**
 * Throws when one storage binding of `byteLength` bytes exceeds the device's
 * `maxStorageBufferBindingSize` (128 MiB on default WebGPU devices). Without this check WebGPU
 * rejects the bind group at dispatch with a validation error; call it where row-proportional
 * scratch is sized, with `hint` naming the prop to change.
 *
 * @param device Device whose limit applies.
 * @param id Contributor ID, for the message.
 * @param name Binding or scratch name, for the message.
 * @param byteLength Bytes the binding needs.
 * @param hint What to reduce, for the message.
 */
export function validateGPUStorageBindingByteLength(
  device: Device,
  id: string,
  name: string,
  byteLength: number,
  hint: string
): void {
  const limit = device.limits.maxStorageBufferBindingSize;
  if (byteLength > limit) {
    throw new Error(
      `${id} ${name} needs ${byteLength} bytes in one binding (limit ${limit}); ${hint}`
    );
  }
}
