// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';

/** Threads per reduction workgroup. */
const REDUCTION_WORKGROUP_SIZE = 256;

/** Items one workgroup of the first level reduces before the level is widened. */
const ITEMS_PER_WORKGROUP = 4096;

/** Largest first-level workgroup count: the second level reduces the partials in one workgroup. */
const MAXIMUM_PARTIAL_COUNT = REDUCTION_WORKGROUP_SIZE;

/**
 * Reduces the bounds of the valid items of a scene to `[minX, minY, maxX, maxY]` (all zero when
 * no item is valid) in two levels: a grid-stride pass over up to 256 workgroups writes one
 * partial box each, then one workgroup merges the partials. A single workgroup reading every item
 * would use one compute unit of the GPU; this spreads the read over the whole device, so the
 * reduction runs at memory bandwidth. Min and max are exact and order independent, so the result
 * does not depend on the workgroup count.
 *
 * `declarations` must define `fn readItem(row: u32) -> vec4f` returning `(x, y, valid, 0)` where a
 * `valid` of at least 0.5 includes the point `(x, y)` in the bounds, and may use the `bindings`.
 * The module constant `ITEM_COUNT` is defined here.
 *
 * @internal
 */
export function getBoundsReductionNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    variant: string;
    /** Read-only item bindings used by `readItem`. */
    bindings: readonly WGSLKernelBinding[];
    /** WGSL defining `readItem`. */
    declarations: string;
    itemCount: number;
    /** Four `float32` rows receiving the bounds. */
    output: GraphDataView<'float32'>;
  }
): GPUCommandNode<Parameters>[] {
  const {id, operation, variant, itemCount, output} = props;
  const partialCount = Math.min(
    MAXIMUM_PARTIAL_COUNT,
    Math.max(1, Math.ceil(itemCount / ITEMS_PER_WORKGROUP))
  );
  const commonDeclarations = `const ITEM_COUNT: u32 = ${itemCount}u;
const FLOAT32_MAXIMUM: f32 = 3.402823466e+38;
const PARTIAL_COUNT: u32 = ${partialCount}u;
const WORKGROUP_THREADS: u32 = ${REDUCTION_WORKGROUP_SIZE}u;
${props.declarations}
var<workgroup> sharedMinima: array<vec2f, ${REDUCTION_WORKGROUP_SIZE}>;
var<workgroup> sharedMaxima: array<vec2f, ${REDUCTION_WORKGROUP_SIZE}>;`;
  // Shared tail: tree-reduces the per-thread boxes of the workgroup into `sharedMinima[0]` and
  // `sharedMaxima[0]`.
  const reduceWorkgroup = `sharedMinima[localInvocationIndex] = localMinimum;
  sharedMaxima[localInvocationIndex] = localMaximum;
  workgroupBarrier();
  for (var stride = ${REDUCTION_WORKGROUP_SIZE / 2}u; stride > 0u; stride = stride >> 1u) {
    if (localInvocationIndex < stride) {
      sharedMinima[localInvocationIndex] = min(sharedMinima[localInvocationIndex], sharedMinima[localInvocationIndex + stride]);
      sharedMaxima[localInvocationIndex] = max(sharedMaxima[localInvocationIndex], sharedMaxima[localInvocationIndex + stride]);
    }
    workgroupBarrier();
  }`;
  const writeFinal = (target: string) => `if (localInvocationIndex == 0u) {
    let minimum = sharedMinima[0];
    let maximum = sharedMaxima[0];
    let hasValid = minimum.x <= maximum.x && minimum.y <= maximum.y;
    ${target}[${target}Offset] = select(0.0, minimum.x, hasValid);
    ${target}[${target}Offset + 1u] = select(0.0, minimum.y, hasValid);
    ${target}[${target}Offset + 2u] = select(0.0, maximum.x, hasValid);
    ${target}[${target}Offset + 3u] = select(0.0, maximum.y, hasValid);
  }`;
  const scanItems = `var localMinimum = vec2f(FLOAT32_MAXIMUM);
  var localMaximum = vec2f(-FLOAT32_MAXIMUM);
  // Grid-stride loop: consecutive threads read consecutive items on every trip.
  for (var row = index; row < ITEM_COUNT; row += PARTIAL_COUNT * WORKGROUP_THREADS) {
    let item = readItem(row);
    if (item.z > 0.5) {
      localMinimum = min(localMinimum, item.xy);
      localMaximum = max(localMaximum, item.xy);
    }
  }
  ${reduceWorkgroup}`;

  if (partialCount === 1) {
    return [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-bounds`,
        operation,
        variant,
        bindings: [
          ...props.bindings,
          {name: 'boundsOut', view: output, type: 'f32', access: 'read_write'}
        ],
        invocationCount: REDUCTION_WORKGROUP_SIZE,
        guardIndex: false,
        declarations: commonDeclarations,
        body: `// Exactly one workgroup is dispatched, so index == localInvocationIndex.
  ${scanItems}
  ${writeFinal('boundsOut')}`
      })
    ];
  }

  const partials = createTransientView(graph, `${id}-bounds-partials`, 'float32', partialCount * 4);
  return [
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-bounds-blocks`,
      operation,
      variant: `${variant}-blocks`,
      bindings: [
        ...props.bindings,
        {name: 'partials', view: partials, type: 'f32', access: 'read_write'}
      ],
      invocationCount: partialCount * REDUCTION_WORKGROUP_SIZE,
      guardIndex: false,
      declarations: commonDeclarations,
      body: `${scanItems}
  if (localInvocationIndex == 0u) {
    let block = index / WORKGROUP_THREADS;
    partials[partialsOffset + block * 4u] = sharedMinima[0].x;
    partials[partialsOffset + block * 4u + 1u] = sharedMinima[0].y;
    partials[partialsOffset + block * 4u + 2u] = sharedMaxima[0].x;
    partials[partialsOffset + block * 4u + 3u] = sharedMaxima[0].y;
  }`
    }),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-bounds`,
      operation,
      variant: `${variant}-merge`,
      bindings: [
        {name: 'partials', view: partials, type: 'f32', access: 'read'},
        {name: 'boundsOut', view: output, type: 'f32', access: 'read_write'}
      ],
      invocationCount: REDUCTION_WORKGROUP_SIZE,
      guardIndex: false,
      // The item reader is not bound in this kernel, so only the shared-memory declarations apply.
      declarations: `const PARTIAL_COUNT: u32 = ${partialCount}u;
const FLOAT32_MAXIMUM: f32 = 3.402823466e+38;
var<workgroup> sharedMinima: array<vec2f, ${REDUCTION_WORKGROUP_SIZE}>;
var<workgroup> sharedMaxima: array<vec2f, ${REDUCTION_WORKGROUP_SIZE}>;`,
      body: `// Exactly one workgroup is dispatched, so index == localInvocationIndex.
  var localMinimum = vec2f(FLOAT32_MAXIMUM);
  var localMaximum = vec2f(-FLOAT32_MAXIMUM);
  if (localInvocationIndex < PARTIAL_COUNT) {
    let base = partialsOffset + localInvocationIndex * 4u;
    localMinimum = vec2f(partials[base], partials[base + 1u]);
    localMaximum = vec2f(partials[base + 2u], partials[base + 3u]);
  }
  ${reduceWorkgroup}
  ${writeFinal('boundsOut')}`
    })
  ];
}
