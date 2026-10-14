// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, Texture, type Device, type TextureFormat} from '@luma.gl/core';
import {
  createTransientView,
  GPUCommandGraph,
  type GraphBufferHandle,
  type GraphDataView,
  type GraphTextureHandle
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUVectorFormat} from '@luma.gl/gpgpu/gpu-data';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

/** Fixed-width vector formats a builder can request with {@link KernelAuditContext.view}. */
export type KernelAuditViewFormat = Exclude<GPUVectorFormat, `${string}-list` | `${string}list`>;

/** What a kernel-audit builder gets: a fake core device, its graph and transient-view factories. */
export type KernelAuditContext = {
  /** CPU-only WebGPU device with the core limits (8 storage buffers, 128 MiB bindings, ...). */
  device: Device;
  /** Graph the builder adds its contributor to (`graph.add`, `addToGraph` or `add*Recipe`). */
  graph: GPUCommandGraph;
  /** Creates a transient storage view; nothing is allocated. */
  view: <Format extends KernelAuditViewFormat>(
    format: Format,
    length: number,
    name?: string
  ) => GraphDataView<Format>;
  /** Creates a transient buffer handle of `byteLength` bytes. */
  buffer: (byteLength: number, name?: string) => GraphBufferHandle;
  /** Creates a transient 2D texture usable as sampled and storage texture. */
  texture: <Format extends TextureFormat>(
    format: Format,
    width: number,
    height: number,
    name?: string
  ) => GraphTextureHandle<Format>;
  /** CSR spatial weights over `rows` rows with `slotsPerRow` neighbor slots per row. */
  weights: (
    rows: number,
    slotsPerRow?: number
  ) => {
    offsets: GraphDataView<'uint32'>;
    neighbors: GraphDataView<'uint32'>;
    weights: GraphDataView<'float32'>;
  };
  /**
   * Largest row count whose `bytesPerRow`-byte rows fit one storage binding
   * (`maxStorageBufferBindingSize`). Use it for capacities at their binding-limited maximum.
   */
  maximumRows: (bytesPerRow: number) => number;
};

/**
 * One configuration of one public contributor. `build` adds the contributor to `context.graph` at
 * maximal documented options: every optional output wired, every compile-time feature on, and
 * capacities and sizes at their documented maximum. A case one step above a documented maximum
 * has an entry in `KERNEL_AUDIT_REJECTIONS` and must be refused at declaration.
 */
export type KernelAuditCase = {
  /** Barrel export name of the contributor or recipe function. */
  exportName: string;
  /** Short configuration label, unique per export. */
  name: string;
  /** Adds the contributor to `context.graph`. */
  build: (context: KernelAuditContext) => void;
};

/** The fake core device: `createNullWebGPUDevice` plus the core texture and buffer limits. */
function createCoreDevice(): Device {
  const device = createNullWebGPUDevice();
  // NullDevice.destroy() only releases its command encoder. Recipes also allocate caller-owned
  // parameter/draw buffers, and a failed compile can leave shaders or pipelines behind.
  const resources = new Set<{destroy: () => void}>();
  const trackResource = <Resource extends {destroy: () => void}>(resource: Resource): Resource => {
    resources.add(resource);
    const destroyResource = resource.destroy.bind(resource);
    resource.destroy = () => {
      resources.delete(resource);
      destroyResource();
    };
    return resource;
  };
  const createBuffer = device.createBuffer.bind(device);
  const createTexture = device.createTexture.bind(device);
  const createShader = device.createShader.bind(device);
  const createRenderPipeline = device.createRenderPipeline.bind(device);
  const destroyDevice = device.destroy.bind(device);
  device.createBuffer = props => trackResource(createBuffer(props));
  device.createTexture = props => trackResource(createTexture(props));
  device.createShader = props => trackResource(createShader(props));
  device.createRenderPipeline = props => trackResource(createRenderPipeline(props));
  device.destroy = () => {
    try {
      for (const resource of [...resources].reverse()) resource.destroy();
    } finally {
      resources.clear();
      destroyDevice();
    }
  };
  // NullDevice defaults to 2048 texels per dimension; core WebGPU allows 8192.
  Object.defineProperty(device, 'limits', {
    value: {
      ...device.limits,
      maxTextureDimension1D: 8192,
      maxTextureDimension2D: 8192,
      maxTextureDimension3D: 2048,
      maxBufferSize: 268435456
    }
  });
  return device;
}

let serial = 0;

/** Creates a fresh fake core device, graph and view factories for one case. */
export function createKernelAuditContext(id: string): KernelAuditContext {
  const device = createCoreDevice();
  const graph = new GPUCommandGraph(device, {id: `kernel-audit-${id}-${serial++}`});
  const view: KernelAuditContext['view'] = (format, length, name = 'view') =>
    createTransientView(graph, `${name}-${serial++}`, format as never, length) as never;
  return {
    device,
    graph,
    view,
    buffer: (byteLength, name = 'buffer') =>
      graph.createTransientBuffer({
        id: `${name}-${serial++}`,
        byteLength,
        usage: Buffer.STORAGE | Buffer.COPY_DST | Buffer.COPY_SRC
      }),
    texture: (format, width, height, name = 'texture') =>
      graph.createTransientTexture({
        id: `${name}-${serial++}`,
        format,
        width,
        height,
        usage:
          Texture.SAMPLE | Texture.STORAGE | Texture.COPY_DST | Texture.COPY_SRC | Texture.RENDER
      }),
    weights: (rows, slotsPerRow = 8) => ({
      offsets: view('uint32', rows + 1, 'weights-offsets'),
      neighbors: view('uint32', rows * slotsPerRow, 'weights-neighbors'),
      weights: view('float32', rows * slotsPerRow, 'weights-values')
    }),
    maximumRows: bytesPerRow => Math.floor(device.limits.maxStorageBufferBindingSize / bytesPerRow)
  };
}

/** The command nodes a case declared (the graph stores them in declaration order). */
export function getDeclaredNodes(context: KernelAuditContext): unknown[] {
  return (context.graph as unknown as {nodes: unknown[]}).nodes;
}

/** One shader module a node compiled. */
export type RecordedShaderModule = {
  /** Node that compiled it. */
  nodeId: string;
  /** Pipeline or shader ID. */
  id: string;
  /** `'compute'`, `'vertex'` or `'fragment'`. */
  stage: 'compute' | 'vertex' | 'fragment';
  /** Complete WGSL. */
  source: string;
  /** Override constants passed with the pipeline. */
  constants: Readonly<Record<string, number>>;
};

/**
 * Compiles every node and visits every compute/render shader module in declaration order.
 * Consumes the node array: maximum-round network cases have nearly 200,000 nodes, so retaining
 * their source closures alongside all compiled shaders can exhaust a Vitest worker's heap.
 * Throws with the node ID when a node's `compile` throws.
 */
export function recordShaderModules(
  nodes: unknown[],
  visitModule: (module: RecordedShaderModule) => void
): void {
  const device = createCoreDevice();
  Object.defineProperty(device, 'props', {
    value: {...device.props, _cacheShaders: false, _cachePipelines: false}
  });
  let nodeId = '';
  Object.defineProperty(device, 'createComputePipeline', {
    value: (props: {id?: string; shader: {source: string}; constants?: Record<string, number>}) => {
      visitModule({
        nodeId,
        id: props.id ?? '',
        stage: 'compute',
        source: props.shader.source,
        constants: {...props.constants}
      });
      return {id: props.id ?? '', hash: '', destroy: () => {}};
    }
  });
  const createRenderPipeline = device.createRenderPipeline.bind(device);
  Object.defineProperty(device, 'createRenderPipeline', {
    value: (props: {
      id?: string;
      vs?: {source: string};
      fs?: {source: string} | null;
      constants?: Record<string, number>;
    }) => {
      const id = props.id ?? '';
      const constants = {...props.constants};
      if (props.vs) visitModule({nodeId, id, stage: 'vertex', source: props.vs.source, constants});
      if (props.fs && props.fs !== props.vs) {
        visitModule({nodeId, id, stage: 'fragment', source: props.fs.source, constants});
      }
      return createRenderPipeline(props as never);
    }
  });
  try {
    for (let nodeIndex = 0; nodeIndex < nodes.length; nodeIndex++) {
      const node = nodes[nodeIndex] as {
        id: string;
        compile?: (context: {device: Device}) => {destroy?: () => void};
      };
      nodeId = node.id;
      try {
        const compiled = node.compile?.({device});
        compiled?.destroy?.();
      } catch (error) {
        throw new Error(`node ${node.id} compile failed: ${String(error)}`);
      } finally {
        // The audit never executes the graph; release each source closure after compilation.
        nodes[nodeIndex] = undefined;
      }
    }
  } finally {
    nodes.length = 0;
    device.destroy();
  }
}
