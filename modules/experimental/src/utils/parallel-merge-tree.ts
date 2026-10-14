// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {
  createWGSLDispatchGate,
  createWGSLKernelNode,
  type WGSLKernelBinding,
  type WGSLKernelProps
} from './wgsl-kernel-nodes';
type WordView = GraphDataView<'uint32'>;

function readBinding(name: string, view: WordView): WGSLKernelBinding {
  return {name, view, type: 'u32', access: 'read'};
}
function writeBinding(name: string, view: WordView, atomic = false): WGSLKernelBinding {
  return {name, view, type: atomic ? 'atomic<u32>' : 'u32', access: 'read_write'};
}

/** Graph builder with GPU-only iterative state, never round-dependent pipeline constants. @internal */
export function createParallelTreeBuilder<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  operation: string
) {
  const nodes: GPUCommandNode<Parameters>[] = [];
  const createWordView = (name: string, length: number) =>
    createTransientView(graph, `${id}-${name}`, 'uint32', Math.max(length, 1));
  const appendKernel = (
    name: string,
    length: number,
    bindings: WGSLKernelBinding[],
    body: string,
    declarations = '',
    gate?: Pick<WGSLKernelProps<Parameters>, 'condition' | 'extraResources'>
  ) => {
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-${name}`,
        operation,
        variant: name.replace(/-\d+/g, ''),
        invocationCount: length,
        bindings,
        declarations: `const NONE: u32 = 0xffffffffu;\n${declarations}`,
        body,
        ...gate
      })
    );
  };
  /**
   * Canonical table -> scratch -> canonical table. The copy shares the jump's indirect gate,
   * so early convergence never leaves a stale ping-pong parity as the advertised result.
   * Every executing lane writes its complete record and sets pending only for an unresolved link.
   */
  const appendJumpNodes = (
    name: string,
    length: number,
    words: number,
    rounds: number,
    table: WordView,
    body: string,
    declarations = '',
    extraBindings: WGSLKernelBinding[] = []
  ) => {
    const scratch = createWordView(`${name}-scratch`, length * words);
    const pending = createWordView(`${name}-pending`, 1);
    const dispatch = createWGSLDispatchGate<Parameters>(graph, {
      id: `${id}-${name}-dispatch`,
      operation,
      shapes: [[length]]
    });
    appendKernel(
      `${name}-seed`,
      1,
      [writeBinding('pending', pending)],
      'pending[pendingOffset] = 1u;'
    );
    for (let round = 0; round < rounds; round++) {
      appendKernel(
        `${name}-gate-${round}`,
        1,
        [writeBinding('pending', pending), dispatch.binding],
        `
  if (pending[pendingOffset] != 0u) { ${dispatch.armWGSL} }
  else { ${dispatch.disarmWGSL} }
  pending[pendingOffset] = 0u;`
      );
      appendKernel(
        `${name}-jump-${round}`,
        length,
        [
          readBinding('table', table),
          writeBinding('scratch', scratch),
          writeBinding('pending', pending, true),
          ...extraBindings
        ],
        body,
        declarations,
        dispatch.getGate(0)
      );
      appendKernel(
        `${name}-copy-${round}`,
        length,
        [readBinding('scratch', scratch), writeBinding('table', table)],
        `
  for (var word = 0u; word < ${words}u; word++) {
    table[tableOffset + index * ${words}u + word] = scratch[scratchOffset + index * ${words}u + word];
  }`,
        '',
        dispatch.getGate(0)
      );
    }
  };
  return {nodes, createWordView, appendKernel, appendJumpNodes};
}

/**
 * Builds a ranked-edge reconstruction forest with one lane per edge. The adapter declares
 * node, firstEndpoint and secondEndpoint (or returns for unused edges). Leaf IDs precede merge
 * IDs, which increase with edge rank. Atomic-min splicing retries only at increasing ranks;
 * it needs neither locks nor cross-workgroup publication. A cycle edge creates a unary node.
 * Children are in ascending ID order; an absent child is NONE and a unary child occupies both
 * slots. The caller can append reductions using the returned GPU-only graph builder.
 * @internal
 */
export function getParallelMergeTreeNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    nodeCount: number;
    edgeCount: number;
    edgeBindings: WGSLKernelBinding[];
    edgeBody: string;
    declarations?: string;
    initializeBindings?: WGSLKernelBinding[];
    initializeBody?: string;
  }
) {
  const builder = createParallelTreeBuilder(graph, props.id, props.operation);
  const {createWordView, appendKernel} = builder;
  const parents = createWordView('parents', props.nodeCount);
  const children = createWordView('children', props.nodeCount * 2);
  appendKernel(
    'init',
    props.nodeCount,
    [
      writeBinding('parents', parents),
      writeBinding('children', children),
      ...(props.initializeBindings ?? [])
    ],
    `
  parents[parentsOffset + index] = NONE;
  children[childrenOffset + index * 2u] = NONE;
  children[childrenOffset + index * 2u + 1u] = 0u;
  ${props.initializeBody ?? ''}`
  );
  appendKernel(
    'splice',
    props.edgeCount,
    [...props.edgeBindings, writeBinding('parents', parents, true)],
    `
  ${props.edgeBody}
  for (var endpoint = 0u; endpoint < 2u; endpoint++) {
    var first = select(firstEndpoint, secondEndpoint, endpoint == 1u);
    var second = node;
    loop {
      if (first == second) { break; }
      let lower = min(first, second);
      let higher = max(first, second);
      let previous = atomicMin(&parents[parentsOffset + lower], higher);
      if (previous == NONE || previous == higher) { break; }
      first = previous;
      second = higher;
    }
  }`,
    props.declarations
  );
  appendKernel(
    'children',
    props.nodeCount,
    [readBinding('parents', parents), writeBinding('children', children, true)],
    `
  let parent = parents[parentsOffset + index];
  if (parent != NONE) {
    atomicMin(&children[childrenOffset + parent * 2u], index);
    atomicMax(&children[childrenOffset + parent * 2u + 1u], index);
  }`
  );
  return {...builder, parents, children};
}
