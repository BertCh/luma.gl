// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Binding, Buffer} from '@luma.gl/core';
import {Kernel} from '@luma.gl/engine';
import {createGPUComputeCommandNode, type GPUCommandNode} from './gpu-command-node';
import type {GPUCommandGraph, GraphBufferUse, GraphDataView} from './gpu-command-graph';
import type {GPUBoundedDispatchLayout} from './gpu-dispatch-utils';
import {
  doGraphDataViewsOverlap,
  getViewBinding,
  getViewBindingRange
} from './graph-data-view-utils';

/** WGSL helpers shared by grid kernels that map finite coordinates into ordered bounds. @internal */
export const GPU_GRID_COORDINATE_WGSL = /* wgsl */ `
fn isFiniteGridValue(value: f32) -> bool {
  return value == value && abs(value) <= 3.402823466e+38;
}

// The scaled cross-zero branch avoids overflowing maximum - minimum for valid float32 bounds.
fn getGridCoordinate(value: f32, minimum: f32, maximum: f32, size: u32) -> u32 {
  if (maximum == minimum || value == minimum) { return 0u; }
  if (value == maximum) { return size - 1u; }
  if (minimum < 0.0 && maximum > 0.0) {
    let scale = max(abs(minimum), abs(maximum));
    let scaledValue = value / scale;
    let scaledMinimum = minimum / scale;
    let scaledMaximum = maximum / scale;
    return min(
      u32((scaledValue - scaledMinimum) / (scaledMaximum - scaledMinimum) * f32(size)),
      size - 1u
    );
  }
  return min(u32((value - minimum) / (maximum - minimum) * f32(size)), size - 1u);
}
`;

/** Returns whether a JavaScript number can be represented as a finite WGSL `f32`. @internal */
export function isFiniteFloat32(value: number): boolean {
  return Number.isFinite(Math.fround(value));
}

/** Formats a finite, representable JavaScript number as a WGSL `f32` literal. @internal */
export function getSpatialFloat32Literal(value: number): string {
  const float32Value = Math.fround(value);
  if (!Number.isFinite(float32Value)) {
    throw new Error(
      'GPU spatial numeric properties must be representable as finite float32 values'
    );
  }
  if (Object.is(float32Value, -0)) {
    return '-0.0';
  }
  const literal = String(float32Value);
  return literal.includes('.') || /e/i.test(literal) ? literal : `${literal}.0`;
}

/** Validates writable spatial columns without changing their storage. @internal */
export function validateSpatialWrites(
  id: string,
  inputs: readonly GraphDataView[],
  outputs: readonly GraphDataView[]
): void {
  for (const [index, output] of outputs.entries()) {
    if (
      inputs.some(input => doGraphDataViewsOverlap(input, output)) ||
      outputs.slice(0, index).some(previous => doGraphDataViewsOverlap(previous, output))
    ) {
      throw new Error(`${id} writable views must not overlap inputs or other outputs`);
    }
  }
}

/** Shared bounded spatial dispatch with deferred buffer resolution. @internal */
export function getSpatialCommandNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    source: string;
    resources: GraphBufferUse[];
    bindings: Record<string, GraphDataView>;
    dispatch: GPUBoundedDispatchLayout;
  }
): readonly GPUCommandNode<Parameters>[] {
  for (const view of Object.values(props.bindings)) {
    if (getViewBindingRange(view).size > graph.device.limits.maxStorageBufferBindingSize) {
      throw new Error(`${props.id} active chunks must fit a storage binding`);
    }
  }
  return [
    createGPUComputeCommandNode<Parameters>({
      id: props.id,
      resources: props.resources,
      compile: ({device}) => {
        const kernel = new Kernel(device, {
          id: props.id,
          source: props.source,
          shaderLayout: {
            bindings: Object.keys(props.bindings).map((name, location) => ({
              name,
              type: 'storage' as const,
              group: 0,
              location
            }))
          }
        });
        const bindingEntries = Object.entries(props.bindings);
        let resolvedBuffers: Buffer[] = [];
        let resolvedBindings: Record<string, Binding> = {};
        let bindGroupCacheKey = {};
        return {
          encode: ({computePass, getBuffer}) => {
            const nextBuffers = bindingEntries.map(([, view]) => getBuffer(view));
            const bindingsChanged = nextBuffers.some(
              (buffer, index) => buffer !== resolvedBuffers[index]
            );
            if (bindingsChanged || nextBuffers.length !== resolvedBuffers.length) {
              resolvedBuffers = nextBuffers;
              resolvedBindings = {};
              for (const [index, [name, view]] of bindingEntries.entries()) {
                const buffer = nextBuffers[index]!;
                resolvedBindings[name] = getViewBinding(view, () => buffer);
              }
              bindGroupCacheKey = {};
            }

            kernel.dispatch(computePass, {
              bindings: resolvedBindings,
              _bindGroupCacheKeys: {0: bindGroupCacheKey},
              x: props.dispatch.x,
              y: props.dispatch.y,
              z: props.dispatch.z
            });
          },
          destroy: () => kernel.destroy()
        };
      }
    })
  ];
}
