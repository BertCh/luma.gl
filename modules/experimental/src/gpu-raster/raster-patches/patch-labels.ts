// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  validatePackedUint32View,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import {MAXIMUM_RASTER_PIXEL_COUNT} from '../raster-utils';

/**
 * Dense patch labels as produced by `GPURasterConnectedComponents` followed by
 * `GPURasterDenseComponents`: label 0 is background, labels `1..` identify patches.
 *
 * The optional guards mirror the upstream scalars so a failed clump never reaches a measurement.
 */
export type GPURasterPatchLabels = {
  /** Raster width in pixels. */
  width: number;
  /** Raster height in pixels. */
  height: number;
  /** One dense label per pixel, row-major. 0 is background. */
  labels: GraphDataView<'uint32'>;
  /** Optional per-pixel observation validity; pixels with 0 are treated as background. */
  labelValidity?: GraphDataView<'uint32'>;
  /** Optional upstream convergence scalar; 0 turns every pixel into background. */
  converged?: GraphDataView<'uint32'>;
  /** Optional upstream bounded component count; labels above it are background. */
  componentCount?: GraphDataView<'uint32'>;
  /** Optional upstream overflow scalar; a nonzero value turns every pixel into background. */
  overflow?: GraphDataView<'uint32'>;
};

/** Validates a {@link GPURasterPatchLabels} and returns its pixel count. @internal */
export function validatePatchLabels(id: string, input: GPURasterPatchLabels): number {
  const {width, height} = input;
  if (
    !Number.isSafeInteger(width) ||
    width < 1 ||
    !Number.isSafeInteger(height) ||
    height < 1 ||
    width * height > MAXIMUM_RASTER_PIXEL_COUNT
  ) {
    throw new Error(`${id} dimensions must be positive integers with a uint32 pixel count`);
  }
  const pixelCount = width * height;
  validatePackedUint32View(input.labels, `${id} labels`);
  if (input.labels.length !== pixelCount) {
    throw new Error(`${id} labels length must equal width * height`);
  }
  if (input.labelValidity) {
    validatePackedUint32View(input.labelValidity, `${id} labelValidity`);
    if (input.labelValidity.length !== pixelCount) {
      throw new Error(`${id} labelValidity length must equal width * height`);
    }
  }
  for (const [name, view] of [
    ['converged', input.converged],
    ['componentCount', input.componentCount],
    ['overflow', input.overflow]
  ] as const) {
    if (view) {
      validatePackedUint32View(view, `${id} ${name}`);
      if (view.length < 1) {
        throw new Error(`${id} ${name} must hold one uint32`);
      }
    }
  }
  return pixelCount;
}

/** The views of a {@link GPURasterPatchLabels} that a consumer reads. @internal */
export function getPatchLabelViews(input: GPURasterPatchLabels): GraphDataView[] {
  return [
    input.labels,
    input.labelValidity,
    input.converged,
    input.componentCount,
    input.overflow
  ].filter(view => view !== undefined) as GraphDataView[];
}

/**
 * Adds one node that writes `keys`: the accepted patch label of each pixel, or 0 for background,
 * invalid, over-capacity and guarded-out pixels.
 *
 * @internal
 */
export function createPatchKeysNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  operation: string,
  input: GPURasterPatchLabels,
  patchCapacity: number
): {node: GPUCommandNode<Parameters>; keys: GraphDataView<'uint32'>} {
  const keys = createTransientView(graph, `${id}-keys`, 'uint32', input.labels.length);
  const bindings: WGSLKernelBinding[] = [
    {name: 'labels', view: input.labels, type: 'u32', access: 'read'}
  ];
  const conditions: string[] = [];
  if (input.labelValidity) {
    bindings.push({name: 'labelValidity', view: input.labelValidity, type: 'u32', access: 'read'});
    conditions.push('labelValidity[labelValidityOffset + index] != 0u');
  }
  if (input.converged) {
    bindings.push({name: 'converged', view: input.converged, type: 'u32', access: 'read'});
    conditions.push('converged[convergedOffset] != 0u');
  }
  if (input.overflow) {
    bindings.push({name: 'overflow', view: input.overflow, type: 'u32', access: 'read'});
    conditions.push('overflow[overflowOffset] == 0u');
  }
  if (input.componentCount) {
    bindings.push({
      name: 'componentCount',
      view: input.componentCount,
      type: 'u32',
      access: 'read'
    });
    conditions.push('label <= componentCount[componentCountOffset]');
  }
  bindings.push({name: 'keys', view: keys, type: 'u32', access: 'read_write'});
  const node = createWGSLKernelNode<Parameters>(graph, {
    id: `${id}-keys`,
    operation,
    variant: 'keys',
    bindings,
    invocationCount: input.labels.length,
    body: `let label = labels[labelsOffset + index];
  let accepted = label != 0u && label <= ${patchCapacity}u${conditions.map(c => ` && ${c}`).join('')};
  keys[keysOffset + index] = select(0u, label, accepted);`
  });
  return {node, keys};
}
