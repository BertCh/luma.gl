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
import {validatePackedView} from '../../utils/gpu-contributor-utils';
import {validateRasterGrid} from '../raster-band-utils';
import type {GPUOverflowStatusPort, GPUStatusScalar} from '../../utils/gpu-contributor-types';
import type {GPUIterationStatusPort} from '../../analysis-core/contracts/status';
import type {GPURasterMetadata} from '../types';
import {
  createRasterIterationState,
  getRasterIterationCondition
} from '../cost-distance/raster-relaxation';

/**
 * Value of every `uint32` output word of {@link GPURasterPatchMetrics} and {@link GPURasterSieve}
 * (and NaN for every `float32` output word) when the upstream labelling did not finish: the
 * `converged` guard reads 0 or the `overflow` guard is nonzero. No finished run can produce it:
 * counts, extents and labels are below 2^32 - 1.
 *
 * Without the guards the contributors cannot tell, so pass them whenever the labels come from a
 * capped loop.
 */
export const GPU_RASTER_PATCH_UNRESOLVED = 0xffffffff;

/**
 * Dense patch labels as produced by `GPURasterConnectedComponents` followed by
 * `GPURasterDenseComponents`: label 0 is background, labels `1..` identify patches.
 *
 * The optional guards mirror the upstream scalars so a failed clump never reaches a measurement.
 */
export type GPURasterPatchLabels = {
  /** Raster grid: its width and height size the kernels; its affine places the pixels. */
  grid: GPURasterMetadata;
  /** One dense label per pixel, row-major. 0 is background. */
  labels: GraphDataView<'uint32'>;
  /** Optional per-pixel observation validity; pixels with 0 are treated as background. */
  validity?: GraphDataView<'uint32'>;
  /** Optional one-row status scalars the labeling passes wrote, read here as guards. */
  status?: GPURasterPatchLabelsStatus;
};

/** Status scalars of the labeling passes that produced {@link GPURasterPatchLabels}. */
export type GPURasterPatchLabelsStatus = Pick<GPUIterationStatusPort, 'converged'> &
  Partial<GPUOverflowStatusPort> & {
    /** Upstream bounded component count; labels above it are background. */
    componentCount?: GPUStatusScalar;
  };

/** Validates a {@link GPURasterPatchLabels} and returns its pixel count. @internal */
export function validatePatchLabels(id: string, input: GPURasterPatchLabels): number {
  const pixelCount = validateRasterGrid(id, input.grid);
  validatePackedView(input.labels, ['uint32'], `${id} labels`);
  if (input.labels.length !== pixelCount) {
    throw new Error(`${id} labels length must equal grid width * height`);
  }
  if (input.validity) {
    validatePackedView(input.validity, ['uint32'], `${id} validity`);
    if (input.validity.length !== pixelCount) {
      throw new Error(`${id} validity length must equal grid width * height`);
    }
  }
  for (const [name, view] of [
    ['status.converged', input.status?.converged],
    ['status.componentCount', input.status?.componentCount],
    ['status.overflow', input.status?.overflow]
  ] as const) {
    if (view) {
      validatePackedView(view, ['uint32'], `${id} ${name}`);
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
    input.validity,
    input.status?.converged,
    input.status?.componentCount,
    input.status?.overflow
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
  if (input.validity) {
    bindings.push({name: 'labelValidity', view: input.validity, type: 'u32', access: 'read'});
    conditions.push('labelValidity[labelValidityOffset + index] != 0u');
  }
  if (input.status?.converged) {
    bindings.push({name: 'converged', view: input.status?.converged, type: 'u32', access: 'read'});
    conditions.push('converged[convergedOffset] != 0u');
  }
  if (input.status?.overflow) {
    bindings.push({name: 'overflow', view: input.status?.overflow, type: 'u32', access: 'read'});
    conditions.push('overflow[overflowOffset] == 0u');
  }
  if (input.status?.componentCount) {
    bindings.push({
      name: 'componentCount',
      view: input.status?.componentCount,
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

/** An output overwritten with the unresolved sentinel by {@link createPatchUnresolvedNodes}. @internal */
export type PatchUnresolvedOutput = {
  view: GraphDataView<'uint32'> | GraphDataView<'float32'>;
  type: 'u32' | 'f32';
};

/**
 * Returns nodes that overwrite `outputs` with {@link GPU_RASTER_PATCH_UNRESOLVED} (NaN for floats)
 * when the labels' `converged` guard reads 0 or their `overflow` guard is nonzero (library rule 9).
 * Returns no nodes without those guards. A one-thread node arms an indirect dispatch only in that
 * case, so a finished run pays one tiny dispatch. Append after the nodes that write `outputs`.
 *
 * @internal
 */
export function createPatchUnresolvedNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  operation: string,
  input: GPURasterPatchLabels,
  outputs: readonly PatchUnresolvedOutput[]
): GPUCommandNode<Parameters>[] {
  if ((!input.status?.converged && !input.status?.overflow) || outputs.length === 0) {
    return [];
  }
  const longest = Math.max(...outputs.map(output => output.view.length));
  const gate = createRasterIterationState(graph, `${id}-unresolved`, operation, longest, id);
  const {x, y, z} = gate.layout;
  const armBindings: WGSLKernelBinding[] = [];
  const unfinished: string[] = [];
  if (input.status?.converged) {
    armBindings.push({
      name: 'converged',
      view: input.status?.converged,
      type: 'u32',
      access: 'read'
    });
    unfinished.push('converged[convergedOffset] == 0u');
  }
  if (input.status?.overflow) {
    armBindings.push({name: 'overflow', view: input.status?.overflow, type: 'u32', access: 'read'});
    unfinished.push('overflow[overflowOffset] != 0u');
  }
  armBindings.push({name: 'dispatch', view: gate.dispatch, type: 'u32', access: 'read_write'});
  const nodes: GPUCommandNode<Parameters>[] = [
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-unresolved-arm`,
      operation,
      variant: 'unresolved-arm',
      bindings: armBindings,
      invocationCount: 1,
      body: `let unfinished = ${unfinished.join(' || ')};
  dispatch[dispatchOffset] = select(0u, ${x}u, unfinished);
  dispatch[dispatchOffset + 1u] = ${y}u;
  dispatch[dispatchOffset + 2u] = ${z}u;`
    })
  ];
  for (const [outputIndex, output] of outputs.entries()) {
    const {condition, extraResources} = getRasterIterationCondition<Parameters>(
      gate,
      `${id}-unresolved-${outputIndex}`
    );
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-unresolved-${outputIndex}`,
        operation,
        variant: output.type === 'f32' ? 'unresolved-float' : 'unresolved-word',
        bindings: [
          {name: 'unresolvedTarget', view: output.view, type: output.type, access: 'read_write'}
        ],
        invocationCount: gate.invocationCount,
        workgroupSize: 256,
        // The quiet NaN is built at run time: a constant bitcast of it does not compile.
        body:
          output.type === 'f32'
            ? `if (index < ${output.view.length}u) {
    var nanBits = 0x7fc00000u;
    unresolvedTarget[unresolvedTargetOffset + index] = bitcast<f32>(nanBits);
  }`
            : `if (index < ${output.view.length}u) {
    unresolvedTarget[unresolvedTargetOffset + index] = ${GPU_RASTER_PATCH_UNRESOLVED}u;
  }`,
        condition,
        extraResources
      })
    );
  }
  return nodes;
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
