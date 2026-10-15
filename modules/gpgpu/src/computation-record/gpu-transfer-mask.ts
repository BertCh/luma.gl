// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Binding} from '@luma.gl/core';
import {Kernel} from '@luma.gl/engine';
import {
  createGPUComputeCommandNode,
  getBoundedDispatchLayout,
  getBoundedInvocationIndexSource,
  getViewBinding,
  getViewElementOffset,
  validatePackedUint32View,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {hashGPUComputationText} from './hash-computation-text';

const TRANSFER_WORKGROUP_SIZE = 256;

/** GPU-resident mapping from source keys to target keys; `0xffffffff` denotes no target. */
export type GPUTransferMaskRelation =
  | {layout: 'index'; targets: GraphDataView}
  | {layout: 'csr'; offsets: GraphDataView; neighbors: GraphDataView};

/** Inputs for transferring selection across an index or CSR relation without readback. */
export type GPUTransferMaskProps = {
  /** Prefix for the clear and scatter node ids. */
  id: string;
  /** Packed uint32 flags over the source key space; nonzero means selected. */
  source: GraphDataView;
  /** Packed uint32 target indices, optionally with CSR row offsets. */
  relation: GPUTransferMaskRelation;
  /** Packed uint32 output over the target key space; cleared before scattering. */
  target: GraphDataView;
  /** `any` writes one; `count` counts selected incoming relation entries. Defaults to `any`. */
  mode?: 'any' | 'count';
};

/** Clears and scatters a selection mask across a GPU-resident relation. */
export class GPUTransferMask implements GPUCommandNodeProducer {
  private readonly props: GPUTransferMaskProps;

  /** Validates packed uint32 formats, relation lengths and separate output storage. */
  constructor(props: GPUTransferMaskProps) {
    this.props = {...props, relation: {...props.relation}};
    for (const [name, view] of Object.entries(this.getBindings())) {
      validatePackedUint32View(view, `${props.id} ${name}`);
      if (name !== 'targetMask' && view.buffer === props.target.buffer) {
        throw new Error(`${props.id} target must use separate storage from source and relation`);
      }
    }
    if (props.relation.layout === 'index') {
      if (props.relation.targets.length !== props.source.length) {
        throw new Error(`${props.id} targets length must equal source length`);
      }
    } else if (props.relation.offsets.length !== props.source.length + 1) {
      throw new Error(`${props.id} offsets length must equal source length + 1`);
    }
  }

  /** Produces two bounded compute nodes without adding them to the graph. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, source, relation, target} = this.props;
    const mode = this.props.mode ?? 'any';
    const allBindings = this.getBindings();
    for (const view of Object.values(allBindings)) {
      if (view.buffer.graph !== graph) {
        throw new Error(`${id} views must belong to the target graph`);
      }
    }
    return (['clear', 'scatter'] as const).map(phase => {
      const bindings = phase === 'clear' ? {targetMask: target} : allBindings;
      const entries = Object.entries(bindings);
      const elementCount = phase === 'clear' ? target.length : source.length;
      const dispatch = getBoundedDispatchLayout(
        id,
        elementCount,
        TRANSFER_WORKGROUP_SIZE,
        graph.device.limits.maxComputeWorkgroupsPerDimension
      );
      const declarations = entries
        .map(
          ([name, view], location) => `const ${name}Offset: u32 = ${getViewElementOffset(view)}u;
@group(0) @binding(${location}) var<storage, ${name === 'targetMask' ? 'read_write' : 'read'}> ${name}: array<${name === 'targetMask' ? 'atomic<u32>' : 'u32'}>;`
        )
        .join('\n');
      const scatter = `if (targetIndex < ${target.length}u) {
    ${mode === 'count' ? 'atomicAdd' : 'atomicMax'}(&targetMask[targetMaskOffset + targetIndex], 1u);
  }`;
      const body =
        phase === 'clear'
          ? 'atomicStore(&targetMask[targetMaskOffset + index], 0u);'
          : `if (sourceMask[sourceMaskOffset + index] == 0u) { return; }
  ${
    relation.layout === 'index'
      ? `let targetIndex = targets[targetsOffset + index];
  ${scatter}`
      : `let start = offsets[offsetsOffset + index];
  let end = min(offsets[offsetsOffset + index + 1u], ${relation.neighbors.length}u);
  for (var neighborIndex = start; neighborIndex < end; neighborIndex++) {
    let targetIndex = neighbors[neighborsOffset + neighborIndex];
    ${scatter}
  }`
  }`;
      const shaderSource = /* wgsl */ `
${declarations}
@compute @workgroup_size(${TRANSFER_WORKGROUP_SIZE})
fn main(
  @builtin(local_invocation_index) localInvocationIndex: u32,
  @builtin(workgroup_id) workgroupId: vec3<u32>
) {
  ${getBoundedInvocationIndexSource(dispatch, TRANSFER_WORKGROUP_SIZE)}
  if (index >= ${elementCount}u) { return; }
  ${body}
}`;
      const nodeId = `${id}-${phase}`;
      const node = createGPUComputeCommandNode<Parameters>({
        id: nodeId,
        ...(phase === 'scatter' && {dependsOn: [`${id}-clear`]}),
        resources: entries.map(([name, view]) => ({
          buffer: view,
          usage: name === 'targetMask' ? 'storage-read-write' : 'storage-read'
        })),
        workload: {
          operation: 'GPUTransferMask',
          variant: `${relation.layout}-${mode}`,
          commandCount: 1,
          maximumWorkgroupCount: dispatch.x * dispatch.y * dispatch.z,
          maximumInvocationCount: dispatch.x * dispatch.y * dispatch.z * TRANSFER_WORKGROUP_SIZE,
          readByteLength:
            phase === 'clear' ? 0 : entries.reduce((total, [, view]) => total + view.length * 4, 0),
          writeByteLength: target.length * 4
        },
        compile: ({device}) => {
          const kernel = new Kernel(device, {
            id: nodeId,
            source: shaderSource,
            shaderLayout: {
              bindings: entries.map(([name], location) => ({
                name,
                type: 'storage' as const,
                group: 0,
                location
              }))
            }
          });
          return {
            encode: ({computePass, getBuffer}) => {
              const resolvedBindings: Record<string, Binding> = {};
              for (const [name, view] of entries) {
                resolvedBindings[name] = getViewBinding(view, getBuffer);
              }
              kernel.dispatch(computePass, {bindings: resolvedBindings, ...dispatch});
            },
            destroy: () => kernel.destroy()
          };
        }
      });
      if (graph.recording) {
        const hash = hashGPUComputationText(shaderSource);
        node.annotation = {
          op: 'GPUTransferMask',
          attrs: {layout: relation.layout, mode},
          source: {
            id: `wgsl:${hash}`,
            language: 'wgsl',
            moduleId: `GPUTransferMask/${phase}`,
            hash,
            entryPoint: 'main',
            ...(graph.recording.sourceText && {text: shaderSource})
          },
          workgroupSize: [TRANSFER_WORKGROUP_SIZE, 1, 1],
          dispatch: {
            workgroups: [dispatch.x, dispatch.y, dispatch.z],
            invocationCount: elementCount
          },
          bindings: entries.map(([name, view]) => ({
            name,
            resource: view.buffer.id,
            access: name === 'targetMask' ? 'read_write' : 'read',
            elementType: name === 'targetMask' ? 'atomic<u32>' : 'u32',
            elementOffset: getViewElementOffset(view),
            length: view.length
          }))
        };
      }
      return node;
    });
  }

  private getBindings(): Record<string, GraphDataView> {
    const {source, relation, target} = this.props;
    return {
      sourceMask: source,
      ...(relation.layout === 'index'
        ? {targets: relation.targets}
        : {offsets: relation.offsets, neighbors: relation.neighbors}),
      targetMask: target
    };
  }
}
