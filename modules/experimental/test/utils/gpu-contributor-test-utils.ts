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
import {WgslReflect} from 'wgsl_reflect';

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
  /** Pipeline-overridable constants passed with the module (a copy; empty when none). */
  constants: Readonly<Record<string, number>>;
  /** Shader layout passed with the module, if any. */
  shaderLayout?: unknown;
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
    value: (props: {
      id?: string;
      shader: {source: string};
      constants?: Record<string, number>;
      shaderLayout?: unknown;
    }) => {
      sources.push({
        id: props.id ?? '',
        operation,
        source: props.shader.source,
        constants: {...props.constants},
        shaderLayout: props.shaderLayout
      });
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

/** Options of {@link getGPUCommandNodePipelineKeys}. */
export type PipelineKeyOptions = {
  /** Only count kernels whose ID matches, for checks on one family (for example every round). */
  idPattern?: RegExp;
  /** Also key on the shader layout, as `PipelineFactory` hashes it (network specs do; default false). */
  includeShaderLayout?: boolean;
};

/**
 * Compiles command nodes (see {@link getGPUCommandNodeKernelSources}) and returns one key per
 * distinct compute pipeline: the WGSL source plus its sorted override constants, the pair
 * `PipelineFactory` hashes. Library rule 12: building a graph at N and N + 1 rounds (batches,
 * periods, powers, classes) must give the same `size`. Use `getGPUCommandNodeKernelSources(nodes).length`
 * for the total number of pipeline requests.
 */
export function getGPUCommandNodePipelineKeys(
  nodes: readonly unknown[],
  options: PipelineKeyOptions = {}
): Set<string> {
  const keys = new Set<string>();
  for (const kernel of getGPUCommandNodeKernelSources(nodes)) {
    if (options.idPattern && !options.idPattern.test(kernel.id)) {
      continue;
    }
    const constants = Object.entries(kernel.constants).sort(([left], [right]) =>
      left < right ? -1 : left > right ? 1 : 0
    );
    const layout = options.includeShaderLayout ? JSON.stringify(kernel.shaderLayout ?? null) : '';
    keys.add(`${kernel.source}\n${JSON.stringify(constants)}\n${layout}`);
  }
  return keys;
}

/**
 * Returns the WGSL of every compute kernel the nodes compile, by kernel ID, after checking that
 * each module parses with `wgsl_reflect` (a parse failure names the kernel).
 */
export function getParsedKernelSources(nodes: readonly unknown[]): Map<string, string> {
  const sources = new Map<string, string>();
  for (const kernel of getGPUCommandNodeKernelSources(nodes)) {
    try {
      new WgslReflect(kernel.source);
    } catch (error) {
      throw new Error(`kernel ${kernel.id} does not parse: ${String(error)}`);
    }
    sources.set(kernel.id, kernel.source);
  }
  return sources;
}

/** Core WebGPU limit on private (function-scope) memory per invocation, in bytes. */
export const CORE_PRIVATE_MEMORY_LIMIT_BYTES = 8192;
/** Core WebGPU `maxComputeWorkgroupStorageSize`, in bytes. */
export const CORE_WORKGROUP_STORAGE_LIMIT_BYTES = 16384;

function getTypeBytes(type: string): number {
  const array = /^array<\s*([^,>]+(?:<[^>]*>)?)\s*,\s*(\d+)u?\s*>$/.exec(type.trim());
  if (array) {
    return getTypeBytes(array[1]) * Number(array[2]);
  }
  const vector = /^vec([234])</.exec(type.trim());
  if (vector) {
    return vector[1] === '2' ? 8 : 16;
  }
  return 4;
}

function sumDeclarations(source: string, pattern: RegExp): number {
  let total = 0;
  for (const match of source.matchAll(pattern)) {
    total += getTypeBytes(match[1]);
  }
  return total;
}

/**
 * Upper bound of the private memory one invocation of a kernel uses: every `var<private>` plus
 * every function-scope `var` of array type (scalar locals are registers). Arrays are counted at
 * their declared length.
 */
export function getPrivateMemoryBytes(source: string): number {
  return (
    sumDeclarations(source, /var<private>\s+\w+\s*:\s*([^;]+);/g) +
    sumDeclarations(source, /^\s+var\s+\w+\s*:\s*(array<[^;]+>);/gm)
  );
}

/** Bytes of `var<workgroup>` declarations of a kernel. */
export function getWorkgroupMemoryBytes(source: string): number {
  return sumDeclarations(source, /var<workgroup>\s+\w+\s*:\s*([^;]+);/g);
}

/** Number of `var<storage, ...>` bindings of a kernel. */
export function getStorageBindingCount(source: string): number {
  return (source.match(/var<storage,/g) ?? []).length;
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

/** One compute pipeline a command-node list would create. */
export type RecordedPipeline = {
  /** Pipeline ID. */
  id: string;
  /** Complete WGSL module. */
  source: string;
  /** Override constants passed to the pipeline (empty when none). */
  constants: Record<string, number>;
  /** `source` plus the serialized override constants: the key the pipeline factory hashes. */
  key: string;
};

/** Compiles `nodes` on a recording device and returns each pipeline's source and constants. */
export function recordComputePipelines(nodes: readonly unknown[]): RecordedPipeline[] {
  const device: Device = createNullWebGPUDevice();
  Object.defineProperty(device, 'props', {
    value: {...device.props, _cacheShaders: false, _cachePipelines: false}
  });
  const pipelines: RecordedPipeline[] = [];
  Object.defineProperty(device, 'createComputePipeline', {
    value: (props: {id?: string; shader: {source: string}; constants?: object}) => {
      pipelines.push({
        id: props.id ?? '',
        source: props.shader.source,
        constants: {...((props.constants ?? {}) as Record<string, number>)},
        key: `${props.shader.source}\n${JSON.stringify(props.constants ?? null)}`
      });
      return {id: props.id ?? '', hash: '', destroy: () => {}};
    }
  });
  for (const node of nodes as {compile?: (context: {device: Device}) => unknown}[]) {
    node.compile?.({device});
  }
  device.destroy();
  return pipelines;
}

/** Number of distinct `(source, constants)` pairs, i.e. the pipelines a real device would build. */
export function countDistinctPipelines(nodes: readonly unknown[]): number {
  return new Set(recordComputePipelines(nodes).map(pipeline => pipeline.key)).size;
}
