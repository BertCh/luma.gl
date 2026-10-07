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
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {createFillNode, createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {
  createPatchKeysNode,
  getPatchLabelViews,
  getPatchRunSegmentsBody,
  getPatchSegmentCount,
  validatePatchLabels,
  type GPURasterPatchLabels
} from './patch-labels';
import {GPU_RASTER_SIEVE_PARAMETER_LENGTH} from './gpu-raster-sieve-parameters';

const OPERATION = 'GPURasterSieve';
const NO_LABEL = '0xffffffffu';

/** How a sieved patch is resolved. */
export type GPURasterSieveMode = 'remove' | 'merge';

/** Pixel adjacency used to find merge neighbors. */
export type GPURasterSieveConnectivity = 4 | 8;

/** Caller-owned outputs of {@link GPURasterSieve}. */
export type GPURasterSieveOutput = {
  /**
   * Sieved label per pixel. Surviving patches keep their label, sieved patches become 0
   * (`'remove'`) or the label of their merge target (`'merge'`).
   */
  labels: GraphDataView<'uint32'>;
  /**
   * Optional per-patch result, `patchCapacity` rows: the final label of patch `r + 1`; itself when
   * kept, the target when merged, 0 when removed or empty. An explain column for the decision.
   */
  patchTargets?: GraphDataView<'uint32'>;
  /** Optional one-row output: number of non-empty patches smaller than `minimumPixels`. */
  sievedCount?: GraphDataView<'uint32'>;
};

/**
 * Properties for {@link GPURasterSieve}.
 *
 * Per-frame (no recompile): the contents of `parameters` and of the label views. Compile-time:
 * dimensions, `patchCapacity`, `mode`, `connectivity` and which optional outputs are present.
 */
export type GPURasterSieveProps = GPURasterPatchLabels & {
  /** Prefix for generated node and transient IDs. Defaults to `'raster-sieve'`. */
  id?: string;
  /** Number of patch rows (largest dense label considered). Labels above it become background. */
  patchCapacity: number;
  /**
   * Per-frame packed uint32 view of at least {@link GPU_RASTER_SIEVE_PARAMETER_LENGTH} elements
   * written with `getGPURasterSieveParameterValues`: the minimum patch size in pixels.
   */
  parameters: GraphDataView<'uint32'>;
  /**
   * `'remove'` (default) turns small patches into background. `'merge'` gives each small patch the
   * label of its largest adjacent patch that is not small (ties go to the lowest label); a small
   * patch with no such neighbor becomes background.
   */
  mode?: GPURasterSieveMode;
  /** Adjacency used by `'merge'` to find neighbors. Default 4. */
  connectivity?: GPURasterSieveConnectivity;
  /** Caller-owned outputs. */
  output: GPURasterSieveOutput;
};

/**
 * Sieve filter for a dense patch label raster (GDAL `gdal_sieve`, GRASS `r.reclass.area`): patches
 * smaller than a threshold are removed or merged into their largest neighbor.
 *
 * Patch sizes are exact integer atomic counts. In `'merge'` mode the target of a patch is one
 * decision per patch (largest adjacent large patch, lowest label on ties), so patches never split.
 * Merging applies to label rasters whose different labels touch, such as segmentations; patches
 * from `GPURasterConnectedComponents` never touch another label in the same connectivity, so
 * `'remove'` is the useful mode for clumps (and for filling small holes, run it on the inverted
 * mask's clumps). Merging is a single pass: a small patch whose neighbors are all small is removed
 * rather than merged into a small neighbor, and no chains form.
 *
 * `labelValidity`, `converged`, `componentCount` and `overflow` turn pixels into background exactly
 * as in `GPURasterPatchMetrics`.
 */
export class GPURasterSieve implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPURasterSieveProps;
  /** Resolved mode. */
  readonly mode: GPURasterSieveMode;
  /** Resolved connectivity. */
  readonly connectivity: GPURasterSieveConnectivity;

  constructor(props: GPURasterSieveProps) {
    const id = props.id ?? 'raster-sieve';
    this.id = id;
    this.props = props;
    this.mode = props.mode ?? 'remove';
    this.connectivity = props.connectivity ?? 4;
    const pixelCount = validatePatchLabels(id, props);
    if (this.mode !== 'remove' && this.mode !== 'merge') {
      throw new Error(`${id} mode must be 'remove' or 'merge'`);
    }
    if (this.connectivity !== 4 && this.connectivity !== 8) {
      throw new Error(`${id} connectivity must be 4 or 8`);
    }
    if (
      !Number.isSafeInteger(props.patchCapacity) ||
      props.patchCapacity < 1 ||
      props.patchCapacity > pixelCount
    ) {
      throw new Error(`${id} patchCapacity must be an integer in [1, pixel count]`);
    }
    validatePackedUint32View(props.parameters, `${id} parameters`);
    if (props.parameters.length < GPU_RASTER_SIEVE_PARAMETER_LENGTH) {
      throw new Error(
        `${id} parameters must hold ${GPU_RASTER_SIEVE_PARAMETER_LENGTH} uint32 values`
      );
    }
    const {output} = props;
    validatePackedUint32View(output.labels, `${id} output.labels`);
    if (output.labels.length !== pixelCount) {
      throw new Error(`${id} output.labels length must equal width * height`);
    }
    if (output.patchTargets) {
      validatePackedUint32View(output.patchTargets, `${id} output.patchTargets`);
      if (output.patchTargets.length !== props.patchCapacity) {
        throw new Error(`${id} output.patchTargets length must equal patchCapacity`);
      }
    }
    if (output.sievedCount) {
      validatePackedUint32View(output.sievedCount, `${id} output.sievedCount`);
      if (output.sievedCount.length < 1) {
        throw new Error(`${id} output.sievedCount must hold one uint32`);
      }
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [output.labels, output.patchTargets, output.sievedCount],
      [...getPatchLabelViews(props), props.parameters]
    );
  }

  /** Returns the key, count, optional merge-target, patch and relabel nodes in order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, mode, connectivity} = this;
    const {output, width, height, patchCapacity, parameters} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      ...getPatchLabelViews(props),
      parameters,
      output.labels,
      output.patchTargets,
      output.sievedCount
    ]);
    const pixelLength = width * height;
    const view = (name: string, length: number) =>
      createTransientView(graph, `${id}-${name}`, 'uint32', length);
    const {node: keysNode, keys} = createPatchKeysNode(graph, id, OPERATION, props, patchCapacity);
    const pixelCounts = view('pixel-counts', patchCapacity);
    const patchTargets = output.patchTargets ?? view('patch-targets', patchCapacity);
    const geometry = `const WIDTH: u32 = ${width}u;
const HEIGHT: u32 = ${height}u;`;
    const nodes: GPUCommandNode<Parameters>[] = [
      keysNode,
      createFillNode<Parameters>(graph, {
        id: `${id}-counts-clear`,
        operation: OPERATION,
        view: pixelCounts,
        type: 'u32',
        value: '0u'
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-counts`,
        operation: OPERATION,
        variant: 'counts',
        bindings: [
          {name: 'keys', view: keys, type: 'u32', access: 'read'},
          {name: 'pixelCounts', view: pixelCounts, type: 'atomic<u32>', access: 'read_write'}
        ],
        invocationCount: getPatchSegmentCount(width, height),
        declarations: geometry,
        body: getPatchRunSegmentsBody(width, {
          onRun: 'atomicAdd(&pixelCounts[pixelCountsOffset + runKey - 1u], runEnd - runStart);'
        })
      })
    ];

    // Merge targets: the largest adjacent patch that is not small, then its lowest label.
    let bestCounts: GraphDataView<'uint32'> | undefined;
    let targets: GraphDataView<'uint32'> | undefined;
    if (mode === 'merge') {
      bestCounts = view('best-counts', patchCapacity);
      targets = view('merge-targets', patchCapacity);
      const neighborLoop = (onNeighbor: string) => `let key = keys[keysOffset + index];
  if (key != 0u && pixelCounts[pixelCountsOffset + key - 1u] < parameters[parametersOffset]) {
    let column = index % WIDTH;
    let row = index / WIDTH;
    for (var dy = -1; dy <= 1; dy++) {
      for (var dx = -1; dx <= 1; dx++) {
        if ((dx == 0 && dy == 0) || (${connectivity} == 4 && dx != 0 && dy != 0)) {
          continue;
        }
        let neighborColumn = i32(column) + dx;
        let neighborRow = i32(row) + dy;
        if (neighborColumn < 0 || neighborRow < 0 || neighborColumn >= i32(WIDTH) || neighborRow >= i32(HEIGHT)) {
          continue;
        }
        let neighbor = keys[keysOffset + u32(neighborRow) * WIDTH + u32(neighborColumn)];
        if (neighbor == 0u || neighbor == key) {
          continue;
        }
        let neighborCount = pixelCounts[pixelCountsOffset + neighbor - 1u];
        if (neighborCount >= parameters[parametersOffset]) {
          ${onNeighbor}
        }
      }
    }
  }`;
      nodes.push(
        createFillNode<Parameters>(graph, {
          id: `${id}-best-counts-clear`,
          operation: OPERATION,
          view: bestCounts,
          type: 'u32',
          value: '0u'
        }),
        createFillNode<Parameters>(graph, {
          id: `${id}-merge-targets-clear`,
          operation: OPERATION,
          view: targets,
          type: 'u32',
          value: NO_LABEL
        }),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-best-counts`,
          operation: OPERATION,
          variant: 'best-counts',
          bindings: [
            {name: 'keys', view: keys, type: 'u32', access: 'read'},
            {name: 'pixelCounts', view: pixelCounts, type: 'u32', access: 'read'},
            {name: 'parameters', view: parameters, type: 'u32', access: 'read'},
            {name: 'bestCounts', view: bestCounts, type: 'atomic<u32>', access: 'read_write'}
          ],
          invocationCount: pixelLength,
          declarations: geometry,
          body: neighborLoop('atomicMax(&bestCounts[bestCountsOffset + key - 1u], neighborCount);')
        }),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-merge-targets`,
          operation: OPERATION,
          variant: 'merge-targets',
          bindings: [
            {name: 'keys', view: keys, type: 'u32', access: 'read'},
            {name: 'pixelCounts', view: pixelCounts, type: 'u32', access: 'read'},
            {name: 'parameters', view: parameters, type: 'u32', access: 'read'},
            {name: 'bestCounts', view: bestCounts, type: 'u32', access: 'read'},
            {name: 'targets', view: targets, type: 'atomic<u32>', access: 'read_write'}
          ],
          invocationCount: pixelLength,
          declarations: geometry,
          body: neighborLoop(`if (neighborCount == bestCounts[bestCountsOffset + key - 1u]) {
            atomicMin(&targets[targetsOffset + key - 1u], neighbor);
          }`)
        })
      );
    }

    const sievedCount = output.sievedCount;
    nodes.push(
      ...(sievedCount
        ? [
            createFillNode<Parameters>(graph, {
              id: `${id}-sieved-clear`,
              operation: OPERATION,
              view: sievedCount,
              type: 'u32',
              value: '0u',
              componentCount: 1
            })
          ]
        : []),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-patch-targets`,
        operation: OPERATION,
        variant: 'patch-targets',
        bindings: [
          {name: 'pixelCounts', view: pixelCounts, type: 'u32', access: 'read'},
          {name: 'parameters', view: parameters, type: 'u32', access: 'read'},
          {name: 'patchTargets', view: patchTargets, type: 'u32', access: 'read_write'},
          ...(bestCounts && targets
            ? [
                {
                  name: 'bestCounts',
                  view: bestCounts,
                  type: 'u32' as const,
                  access: 'read' as const
                },
                {name: 'targets', view: targets, type: 'u32' as const, access: 'read' as const}
              ]
            : []),
          ...(sievedCount
            ? [
                {
                  name: 'sievedCount',
                  view: sievedCount,
                  type: 'atomic<u32>' as const,
                  access: 'read_write' as const
                }
              ]
            : [])
        ],
        invocationCount: patchCapacity,
        body: `let count = pixelCounts[pixelCountsOffset + index];
  var finalLabel = 0u;
  if (count >= parameters[parametersOffset]) {
    finalLabel = index + 1u;
  } else if (count > 0u) {
    ${sievedCount ? 'atomicAdd(&sievedCount[sievedCountOffset], 1u);' : ''}
    ${
      mode === 'merge'
        ? `if (bestCounts[bestCountsOffset + index] != 0u) {
      finalLabel = targets[targetsOffset + index];
    }`
        : ''
    }
  }
  patchTargets[patchTargetsOffset + index] = finalLabel;`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-relabel`,
        operation: OPERATION,
        variant: 'relabel',
        bindings: [
          {name: 'keys', view: keys, type: 'u32', access: 'read'},
          {name: 'patchTargets', view: patchTargets, type: 'u32', access: 'read'},
          {name: 'sievedLabels', view: output.labels, type: 'u32', access: 'read_write'}
        ],
        invocationCount: pixelLength,
        body: `let key = keys[keysOffset + index];
  sievedLabels[sievedLabelsOffset + index] =
    select(0u, patchTargets[patchTargetsOffset + key - 1u], key != 0u);`
      })
    );
    return nodes;
  }
}
