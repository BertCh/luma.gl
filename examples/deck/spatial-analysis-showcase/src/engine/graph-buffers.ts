// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import type {
  CompiledGPUCommandGraph,
  GPUCommandGraph,
  GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {getGPUVectorFormatInfo, type GPUVectorFormat} from '@luma.gl/gpgpu/gpu-data';

/**
 * Imports an application-owned buffer into a command graph and returns a typed view of it, using
 * only the public `GPUCommandGraph.importBuffer()` and `createDataView()` methods.
 *
 * @param graph Graph that receives the buffer.
 * @param id Debug id of the imported resource.
 * @param buffer Buffer owned by the application; the graph never destroys it.
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

/** Encodes a compiled graph that takes no per-frame parameters and submits it to the device queue. */
export function submitGraph(device: Device, compiled: CompiledGPUCommandGraph<undefined>): void {
  const commandEncoder = device.createCommandEncoder({id: `${compiled.id}-encoder`});
  compiled.encode(commandEncoder, {parameters: undefined});
  device.submit(commandEncoder.finish());
}
