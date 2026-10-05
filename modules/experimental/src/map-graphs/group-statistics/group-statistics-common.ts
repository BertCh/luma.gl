// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {createMapGraphKernelNode, type MapGraphKernelBinding} from '../map-graph-kernels';

/** Marker for "no group" in per-row and per-position group indices. @internal */
export const GROUP_NONE = 0xffffffff;

/** Read-only storage binding. @internal */
export function readBinding(
  name: string,
  view: GraphDataView,
  type: 'u32' | 'f32' = 'u32'
): MapGraphKernelBinding {
  return {name, view, type, access: 'read'};
}

/** Read-write storage binding. @internal */
export function writeBinding(
  name: string,
  view: GraphDataView,
  type: 'u32' | 'f32' = 'u32'
): MapGraphKernelBinding {
  return {name, view, type, access: 'read_write'};
}

/** Read-write `atomic<u32>` storage binding. @internal */
export function atomicBinding(name: string, view: GraphDataView): MapGraphKernelBinding {
  return {name, view, type: 'atomic<u32>', access: 'read_write'};
}

/**
 * Value keys for order statistics and extremes. Works on the raw f32 bits so NaN handling never
 * depends on float comparisons: every non-finite value (NaN, +-Infinity) maps to `0xffffffff`
 * (sorts last), `-0` is canonicalised to `+0`, and finite values get the order-preserving key.
 *
 * @internal
 */
export const GROUP_VALUE_KEY_WGSL = /* wgsl */ `
fn isFiniteBits(bits: u32) -> bool {
  return (bits & 0x7f800000u) != 0x7f800000u;
}

fn getValueKey(rawBits: u32) -> u32 {
  if (!isFiniteBits(rawBits)) {
    return 0xffffffffu;
  }
  var bits = rawBits;
  if ((bits & 0x7fffffffu) == 0u) {
    bits = 0u;
  }
  return select(bits ^ 0x80000000u, ~bits, (bits & 0x80000000u) != 0u);
}
`;

/** Creates `target[i] = source[indices[i]]` for `u32` rows. @internal */
export function createGatherNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    indices: GraphDataView<'uint32'>;
    source: GraphDataView<'uint32'>;
    target: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  return createMapGraphKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'gather',
    bindings: [
      readBinding('indices', props.indices),
      readBinding('source', props.source),
      writeBinding('gathered', props.target)
    ],
    invocationCount: props.target.length,
    body: `gathered[gatheredOffset + index] = source[sourceOffset + indices[indicesOffset + index]];`
  });
}
