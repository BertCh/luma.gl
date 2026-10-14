// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {
  GraphVectorView,
  type CompiledGPUCommandGraph,
  type GPUCommandGraphEncoding,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {getGPUVectorFormatInfo, type GPUVectorFormat} from '@luma.gl/gpgpu/gpu-data';
import {NullDevice} from '@luma.gl/test-utils';

/**
 * Returns a CPU-only device that GPUCommandGraph accepts, for `.node.spec.ts` wiring tests.
 *
 * It supports graph declaration and `getCommandNodes(graph)` but not compilation or encoding.
 */
export function createNullWebGPUDevice(): Device {
  const device = new NullDevice({});
  Object.defineProperty(device, 'type', {value: 'webgpu'});
  Object.defineProperty(device, 'limits', {
    value: {
      ...device.limits,
      maxComputeWorkgroupsPerDimension: 65535,
      maxStorageBufferBindingSize: 134217728,
      maxStorageBuffersPerShaderStage: 8,
      maxComputeWorkgroupStorageSize: 16384,
      maxComputeInvocationsPerWorkgroup: 256,
      maxComputeWorkgroupSizeX: 256,
      maxComputeWorkgroupSizeY: 256,
      maxComputeWorkgroupSizeZ: 64
    }
  });
  return device;
}

/** WGSL of one compute kernel, as recorded by {@link getGPUCommandNodeKernelSources}. */
export type RecordedGPUKernelSource = {
  /** Kernel ID (the pipeline ID). */
  id: string;
  /** `workload.operation` of the command node that compiled the kernel. */
  operation: string;
  /** The complete WGSL module passed to the pipeline. */
  source: string;
};

/**
 * Compiles command nodes on a CPU-only WebGPU device that records the WGSL of every compute
 * pipeline instead of creating it, for `.node.spec.ts` checks of generated kernels (WGSL parsing,
 * binding counts, private and workgroup memory) at the core limits of {@link createNullWebGPUDevice}.
 *
 * Use it instead of `vi.mock`: the node project runs with `isolate: false`, so a module mock of
 * `@luma.gl/engine` is not applied reliably when another spec file loaded the module first.
 */
export function getGPUCommandNodeKernelSources(
  nodes: readonly unknown[]
): RecordedGPUKernelSource[] {
  const device = createNullWebGPUDevice();
  Object.defineProperty(device, 'props', {
    value: {...device.props, _cacheShaders: false, _cachePipelines: false}
  });
  const sources: RecordedGPUKernelSource[] = [];
  let operation = '';
  Object.defineProperty(device, 'createComputePipeline', {
    value: (props: {id?: string; shader: {source: string}}) => {
      sources.push({id: props.id ?? '', operation, source: props.shader.source});
      return {id: props.id ?? '', hash: '', destroy: () => {}};
    }
  });
  for (const node of nodes as {
    workload?: {operation?: string};
    compile?: (context: {device: Device}) => unknown;
  }[]) {
    operation = node.workload?.operation ?? '';
    node.compile?.({device});
  }
  device.destroy();
  return sources;
}

/** Creates a storage input buffer initialized with `values`. */
export function createInputBuffer(
  device: Device,
  values: Float32Array | Uint32Array | Int32Array
): Buffer {
  return device.createBuffer({data: values, usage: Buffer.STORAGE | Buffer.COPY_DST});
}

/** Creates a zero-filled storage output buffer with `length` 32-bit elements. */
export function createOutputBuffer(device: Device, length: number, usage: number = 0): Buffer {
  return device.createBuffer({
    byteLength: Math.max(length, 1) * Uint32Array.BYTES_PER_ELEMENT,
    usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST | usage
  });
}

/** Reads the first `length` uint32 values of a submitted buffer. */
export async function readUint32(buffer: Buffer, length: number): Promise<number[]> {
  const bytes = await buffer.readAsync();
  return Array.from(new Uint32Array(bytes.buffer, bytes.byteOffset, length));
}

/** Reads the first `length` float32 values of a submitted buffer. */
export async function readFloat32(buffer: Buffer, length: number): Promise<number[]> {
  const bytes = await buffer.readAsync();
  return Array.from(new Float32Array(bytes.buffer, bytes.byteOffset, length));
}

/** Reads a compact result prefix: `count` first, then that many IDs. */
export async function readCompactIds(ids: Buffer, count: Buffer): Promise<number[]> {
  const [visibleCount] = await readUint32(count, 1);
  return readUint32(ids, visibleCount);
}

/** Numeric comparator for stable assertions on unordered GPU output. */
export function sortNumbers(left: number, right: number): number {
  return left - right;
}

/** Wraps ordered packed chunks of one format in a graph vector view. */
export function createVectorView<Format extends GPUVectorFormat>(
  id: string,
  format: Format,
  data: GraphDataView<Format>[]
): GraphVectorView<Format> {
  const formatInfo = getGPUVectorFormatInfo(format);
  const length = data.reduce((sum, chunk) => sum + chunk.length, 0);
  const components = formatInfo.byteLength / 4;
  return new GraphVectorView({
    id,
    name: id,
    format,
    length,
    valueLength: length * components,
    stride: components,
    byteStride: formatInfo.byteLength,
    rowByteLength: formatInfo.byteLength,
    data
  });
}

/** Returns whether a device is a software or fallback adapter with relaxed float behavior. */
export function isSoftwareDevice(device: Device): boolean {
  return (
    device.info.gpu === 'software' || device.info.gpuType === 'cpu' || Boolean(device.info.fallback)
  );
}

/**
 * Encodes one compiled graph into a new command encoder and submits it.
 *
 * Contributors never submit; this is a test convenience for headless specs.
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
