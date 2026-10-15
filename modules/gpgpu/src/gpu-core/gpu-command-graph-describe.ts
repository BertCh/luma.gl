// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  GPUComputationEdge,
  GPUComputationGraph,
  GPUComputationNode,
  GPUComputationResource,
  GPUComputationSourceRef
} from '../computation-record/computation-record-types';
import {
  getBufferHandle,
  getTextureHandle,
  getTextureByteLength,
  inferGPUCommandGraphDependencies,
  isBufferReadUsage,
  isBufferWriteUsage,
  isGraphBufferUse,
  isGraphTextureUse,
  isTextureReadUsage,
  isTextureWriteUsage,
  type GPUCommandGraphCompilation
} from './gpu-command-graph-compiler';
import {getGPUComputeDispatchWorkgroups} from './gpu-command-dispatch-metadata';
import type {GPUCommandGraphNodeEncodingStats} from './gpu-command-graph-types';

/** Compiled graph structure and its deduplicated source references. */
export type GPUCommandGraphDescription = {
  /** Scheduled graph, logical resources and dependencies. */
  graph: GPUComputationGraph;
  /** Sources in first scheduled occurrence order. */
  sources: readonly GPUComputationSourceRef[];
};

/** Builds a compiled record without accessing GPU state. @internal */
export function describeGPUCommandGraph<Parameters>(
  props: Pick<
    GPUCommandGraphCompilation<Parameters>,
    | 'id'
    | 'compiledNodes'
    | 'buffers'
    | 'textures'
    | 'externalTextures'
    | 'bufferTransientAllocations'
    | 'textureTransientAllocations'
    | 'retainedResourceIds'
    | 'observationTapIds'
    | 'observerCopySourceIds'
  >,
  options: {graphId?: string; label?: string; domain?: string} = {}
): GPUCommandGraphDescription {
  const graphId = options.graphId ?? `${props.id}-compiled`;
  validateGraphId(graphId);
  const sources = new Map<string, GPUComputationSourceRef>();
  const lifetimes = new Map<string, {firstNode: string; lastNode: string}>();
  const scheduledNodes = props.compiledNodes.map(({node}) => node);
  const nodes = scheduledNodes.map(node => {
    const annotation = node.annotation;
    if (annotation?.source) {
      const previous = sources.get(annotation.source.id);
      if (previous && previous.text !== annotation.source.text) {
        throw new Error(`GPUCommandGraph source id "${annotation.source.id}" has conflicting text`);
      }
      if (!previous) sources.set(annotation.source.id, omitUndefined(annotation.source));
    }
    const reads = new Set<string>();
    const writes = new Set<string>();
    for (const resource of node.resources ?? []) {
      let id: string;
      if (isGraphBufferUse(resource)) {
        id = getBufferHandle(resource.buffer).id;
        if (isBufferReadUsage(resource.usage)) reads.add(id);
        if (isBufferWriteUsage(resource.usage)) writes.add(id);
      } else if (isGraphTextureUse(resource)) {
        id = getTextureHandle(resource.texture).id;
        if (isTextureReadUsage(resource.usage)) reads.add(id);
        if (isTextureWriteUsage(resource.usage)) writes.add(id);
      } else {
        id = resource.externalTexture.id;
        reads.add(id);
      }
      const lifetime = lifetimes.get(id);
      if (lifetime) lifetime.lastNode = node.id;
      else lifetimes.set(id, {firstNode: node.id, lastNode: node.id});
    }
    const workgroups = node.type === 'compute' ? getGPUComputeDispatchWorkgroups(node) : undefined;
    return omitUndefined({
      id: node.id,
      label: annotation?.label ?? annotation?.dedupName ?? node.id,
      namespace: annotation?.namespace ?? '',
      kind: node.type,
      op: annotation?.op ?? node.workload?.operation,
      datum: annotation?.datum,
      loop: annotation?.loop,
      dedupName: annotation?.dedupName,
      derivedFrom: annotation?.derivedFrom,
      formula: annotation?.formula,
      source: annotation?.source?.id,
      workgroupSize: annotation?.workgroupSize,
      bindings: annotation?.bindings,
      dispatch: annotation?.dispatch ?? (workgroups ? {workgroups} : undefined),
      reads: [...reads],
      writes: [...writes],
      workload: node.workload ? omitUndefined(node.workload) : undefined,
      condition: node.condition
        ? {
            id: node.condition.id,
            source: node.condition.source,
            mode: node.condition.source === 'cpu' ? ('skip' as const) : ('indirect' as const),
            ...(node.condition.source === 'gpu' ? {resource: node.condition.buffer.id} : {})
          }
        : undefined,
      publication: node.publication ? {...node.publication} : undefined,
      attrs: {
        ...annotation?.attrs,
        ...(props.observationTapIds?.has(node.id) ? {observationTap: true} : {})
      }
    });
  });
  const edges: GPUComputationEdge[] = [];
  inferGPUCommandGraphDependencies(scheduledNodes, edge => edges.push(edge));
  const resources: GPUComputationResource[] = [];
  for (const buffer of props.buffers.values()) {
    const allocationIndex = props.bufferTransientAllocations.findIndex(allocation =>
      allocation.handles.includes(buffer)
    );
    resources.push({
      id: buffer.id,
      kind: 'buffer',
      bytes: buffer.byteLength,
      ownership: buffer.transient ? 'transient' : 'imported',
      ...(allocationIndex >= 0
        ? {
            allocation: `transient-buffer-${allocationIndex}`,
            allocationBytes: props.bufferTransientAllocations[allocationIndex].byteLength
          }
        : {}),
      ...lifetimes.get(buffer.id),
      ...(props.retainedResourceIds?.has(buffer.id) ? {retained: true} : {}),
      // Observer effect: the allocation gained COPY_SRC only so a capture or tap can read it.
      ...(props.observerCopySourceIds?.has(buffer.id) ? {attrs: {observerUsage: 'copy-src'}} : {})
    });
  }
  for (const texture of props.textures.values()) {
    const allocationIndex = props.textureTransientAllocations.findIndex(allocation =>
      allocation.handles.includes(texture)
    );
    resources.push({
      id: texture.id,
      kind: 'texture',
      bytes: getTextureByteLength(texture),
      ownership: texture.transient ? 'transient' : 'imported',
      ...(allocationIndex >= 0
        ? {
            allocation: `transient-texture-${allocationIndex}`,
            allocationBytes: props.textureTransientAllocations[allocationIndex].byteLength
          }
        : {}),
      ...lifetimes.get(texture.id),
      ...(props.retainedResourceIds?.has(texture.id) ? {retained: true} : {}),
      format: texture.format,
      attrs: {
        width: texture.width,
        height: texture.height,
        depth: texture.depth,
        mipLevels: texture.mipLevels,
        dimension: texture.dimension
      }
    });
  }
  for (const texture of props.externalTextures.values()) {
    resources.push({
      id: texture.id,
      kind: 'external-texture',
      bytes: 0,
      ownership: 'imported',
      ...lifetimes.get(texture.id)
    });
  }
  const taps = props.observationTapIds?.size ?? 0;
  const retained = props.retainedResourceIds?.size ?? 0;
  return {
    graph: {
      id: graphId,
      form: 'compiled',
      domain: options.domain ?? 'gpu-core',
      label: options.label ?? props.id,
      nodes,
      edges,
      resources,
      ...(taps || retained
        ? {attrs: {...(taps ? {taps} : {}), ...(retained ? {retained} : {})}}
        : {})
    },
    sources: [...sources.values()]
  };
}

/** Describes the realized passes of an encoding, including legacy synthetic encodings. @internal */
export function describeGPUCommandGraphEncoding(
  stats: readonly GPUCommandGraphNodeEncodingStats[],
  props: {compiledGraphId: string; graphId?: string}
): GPUComputationGraph {
  const graphId = props.graphId ?? `${props.compiledGraphId}-encoded`;
  validateGraphId(graphId);
  validateGraphId(props.compiledGraphId);
  const nodes: GPUComputationNode[] = [];
  const passes = new Map<number, GPUComputationNode>();
  const skipped: string[] = [];
  let nextPassIndex = 0;
  for (const node of stats) {
    if (node.condition?.outcome === 'skipped') {
      skipped.push(node.id);
      continue;
    }
    const derivedFrom = `${props.compiledGraphId}:${node.id}`;
    if (node.type === 'copy') {
      nodes.push({
        id: `copy-${node.id}`,
        label: node.id,
        namespace: '',
        kind: 'copy',
        derivedFrom: [derivedFrom],
        reads: [],
        writes: [],
        attrs: {}
      });
      continue;
    }
    const passIndex = node.passIndex ?? nextPassIndex;
    nextPassIndex = Math.max(nextPassIndex, passIndex + 1);
    const previous = passes.get(passIndex);
    if (previous) {
      previous.derivedFrom = [...previous.derivedFrom!, derivedFrom];
      previous.attrs = {...previous.attrs, nodeCount: previous.derivedFrom.length};
    } else {
      const pass: GPUComputationNode = {
        id: `pass-${passIndex}`,
        kind: 'pass',
        label: `${node.type} pass ${passIndex}`,
        namespace: '',
        derivedFrom: [derivedFrom],
        reads: [],
        writes: [],
        attrs: {passType: node.type, nodeCount: 1}
      };
      passes.set(passIndex, pass);
      nodes.push(pass);
    }
  }
  return {
    id: graphId,
    form: 'encoded',
    domain: 'gpu-core',
    nodes,
    edges: [],
    ...(skipped.length ? {attrs: {skipped: skipped.join(',')}} : {})
  };
}

function validateGraphId(id: string): void {
  if (id.includes(':')) throw new Error(`GPUCommandGraph graph id "${id}" must not contain ':'`);
}

function omitUndefined<Value extends object>(value: Value): Value {
  if (Array.isArray(value)) {
    return value.map(entry =>
      entry !== null && typeof entry === 'object' ? omitUndefined(entry) : entry
    ) as Value;
  }
  return Object.fromEntries(
    Object.entries(value)
      .filter(([, entry]) => entry !== undefined)
      .map(([key, entry]) => [
        key,
        entry !== null && typeof entry === 'object' ? omitUndefined(entry) : entry
      ])
  ) as Value;
}
