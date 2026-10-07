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

/** Columns scanned serially by one invocation of a run-aggregated patch kernel. @internal */
export const PATCH_SEGMENT_LENGTH = 16;

/** Invocation count of a run-aggregated patch kernel: one per row segment. @internal */
export function getPatchSegmentCount(width: number, height: number): number {
  return height * Math.ceil(width / PATCH_SEGMENT_LENGTH);
}

/**
 * Returns a WGSL kernel body (with `WIDTH` and a `keys` binding in scope) that scans one
 * {@link PATCH_SEGMENT_LENGTH}-column row segment per invocation and calls `onRun` once per
 * maximal run of equal nonzero keys inside the segment, with `runKey`, `runStart`, `runEnd`
 * (exclusive) and `row` in scope.
 *
 * Per-patch reductions (counts, extents, perimeter faces) otherwise issue one atomic per pixel at
 * the patch's single counter, which serializes a large patch at that address. Folding each run
 * locally first cuts the atomics by up to the segment length on large patches, and the integer
 * results are unchanged. `onPixel` runs for every nonzero-key pixel (`column`, `row`, `key` in
 * scope) and `onRunStart` resets `runState` variables declared by `declareRunState`.
 *
 * @internal
 */
export function getPatchRunSegmentsBody(
  width: number,
  props: {
    onRun: string;
    declareRunState?: string;
    onRunStart?: string;
    onPixel?: string;
  }
): string {
  return `let segmentsPerRow = ${Math.ceil(width / PATCH_SEGMENT_LENGTH)}u;
  let row = index / segmentsPerRow;
  let segmentStart = (index % segmentsPerRow) * ${PATCH_SEGMENT_LENGTH}u;
  let segmentEnd = min(segmentStart + ${PATCH_SEGMENT_LENGTH}u, WIDTH);
  let rowBase = row * WIDTH;
  var runKey = 0u;
  var runStart = segmentStart;
  ${props.declareRunState ?? ''}
  // The extra iteration at segmentEnd sees a sentinel key that flushes the last run.
  for (var column = segmentStart; column <= segmentEnd; column++) {
    var key = 0xffffffffu;
    if (column < segmentEnd) {
      key = keys[keysOffset + rowBase + column];
    }
    if (key != runKey) {
      if (runKey != 0u) {
        let runEnd = column;
        ${props.onRun}
      }
      runKey = key;
      runStart = column;
      ${props.onRunStart ?? ''}
    }
    ${props.onPixel ? `if (column < segmentEnd && key != 0u) {\n      ${props.onPixel}\n    }` : ''}
  }`;
}
