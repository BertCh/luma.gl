// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView,
  type GraphTextureView
} from '@luma.gl/gpgpu/gpu-core';
import {GPURasterBufferToTexture} from '../../gpu-raster/index';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import type {GPUNetworkAdjacency} from '../../analysis-core/network-topology';
import {
  getGPUNetworkTopologyViews,
  validateGPUNetworkTopology
} from '../../analysis-core/network-topology';
import {createFillNode, createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import type {WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import {
  captureGraphCommandNodes,
  validateCapacity,
  validatePackedView,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';

/** Number of words of the per-frame `parameters` view: `[rowStart, rowEnd, colStart, colEnd]`. */
export const GPU_ADJACENCY_MATRIX_PARAMETER_LENGTH = 4;

/** Default fixed-point scale of weight sums: one unit of weight is 1024 integer steps. */
export const GPU_ADJACENCY_MATRIX_DEFAULT_WEIGHT_SCALE = 1024;

/** Per-frame zoom window of one {@link GPUAdjacencyMatrix}, in matrix positions. Ends are exclusive. */
export type GPUAdjacencyMatrixSettings = {
  /** First row position. */
  rowStart: number;
  /** One past the last row position. */
  rowEnd: number;
  /** First column position. */
  colStart: number;
  /** One past the last column position. */
  colEnd: number;
  /** Matrix resolution `R` of the {@link GPUAdjacencyMatrix}, used to bound the window extents. */
  resolution: number;
};

/**
 * Packs the per-frame zoom window into the four `uint32` words of the `parameters` view.
 *
 * Write the result with `GPUParameterBuffer.write`. An empty window (`end <= start`)
 * produces an all-zero matrix. The GPU computes `(position - start) * resolution` in `u32`, so
 * each extent times `settings.resolution` must stay below 2^32.
 *
 * @throws If a bound is not a uint32 integer or an extent times `resolution` reaches 2^32.
 */
export function getGPUAdjacencyMatrixParameterValues(
  settings: GPUAdjacencyMatrixSettings
): Uint32Array {
  const {rowStart, rowEnd, colStart, colEnd, resolution} = settings;
  for (const value of [rowStart, rowEnd, colStart, colEnd]) {
    if (!Number.isSafeInteger(value) || value < 0 || value > 0xffffffff) {
      throw new Error('GPUAdjacencyMatrix window bounds must be uint32 integers');
    }
  }
  for (const extent of [rowEnd - rowStart, colEnd - colStart]) {
    if (extent * resolution > 0xffffffff) {
      throw new Error('GPUAdjacencyMatrix window extent times resolution must be below 2^32');
    }
  }
  return new Uint32Array([rowStart, rowEnd, colStart, colEnd]);
}

/** Fixed-point weight contributed by one edge: `clamp(round(weight * scale), 0, 2^32 - 1)`. */
export function getGPUAdjacencyMatrixFixedWeight(weight: number, scale: number): number {
  const scaled = Math.fround(weight) * scale;
  if (!(scaled > 0)) {
    return 0;
  }
  return Math.min(Math.floor(scaled + 0.5), 0xffffffff);
}

/**
 * Properties for {@link GPUAdjacencyMatrix}.
 *
 * Compile-time: node count, `resolution`, `directed`, `mirrorSlots`, `weightScale`, and
 * which optional views exist. Per-frame: CSR contents, masks, `order` and `parameters`.
 */
export type GPUAdjacencyMatrixProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'adjacency-matrix'`. */
  id?: string;
  /**
   * Forward CSR network. Neighbor indices `>= nodeCount` are dead slots. `weights` are required
   * for weight outputs.
   */
  topology: GPUNetworkAdjacency;
  /** Treat slots as directed edges (row = source, column = target). Default false. */
  directed?: boolean;
  /**
   * Undirected only, default false. An undirected CSR lists both directions (as in
   * `GPUNetworkStatistics` and gpu-graph), so by default each slot is written once and the matrix
   * is symmetric because the CSR is. When true, each slot `(u, v)` is also written transposed at
   * `(v, u)` (a self-loop once), for edge-list-style CSRs that list each edge once.
   */
  mirrorSlots?: boolean;
  /** Optional per-node mask, nonzero = live. */
  nodeMask?: GraphDataView<'uint32'>;
  /** Optional per-slot mask, nonzero = live. A slot is live iff both endpoints are live too. */
  edgeMask?: GraphDataView<'uint32'>;
  /** Optional permutation view (node to matrix position, `nodeCount` rows). Default identity. */
  order?: GraphDataView<'uint32'>;
  /**
   * Per-frame `'uint32'` view of four words, see {@link getGPUAdjacencyMatrixParameterValues}.
   * When omitted the window is `[0, nodeCount)` on both axes.
   */
  parameters?: GraphDataView<'uint32'>;
  /** Matrix resolution `R` in bins per axis, compile-time, 1 to 4096. */
  resolution: number;
  /**
   * Fixed-point scale for weight sums, default 1024. Per-edge weights are rounded to
   * `round(weight * scale)`, negative and NaN weights count 0, and sums wrap modulo 2^32.
   */
  weightScale?: number;
  /** Caller-owned outputs. */
  output: {
    /** `resolution * resolution` row-major `uint32` edge counts (`row * R + column`). */
    counts: GraphDataView<'uint32'>;
    /** Optional same-size fixed-point weight sums. Requires `weights`. */
    weightSums?: GraphDataView<'uint32'>;
    /** Optional one-row scalar receiving the maximum of `counts`. */
    maximumCount?: GraphDataView<'uint32'>;
    /** Optional one-row scalar receiving the maximum of `weightSums`. */
    maximumWeightSum?: GraphDataView<'uint32'>;
    /** Optional `r32float` `R x R` storage texture; row = y, column = x. */
    texture?: GraphTextureView<'r32float'>;
    /** Texture contents: raw counts (default) or `weightSums / weightScale`. */
    textureStatistic?: 'count' | 'weightSum';
  };
};

/**
 * Bins a graph into an `R x R` adjacency-matrix image on the GPU, for a matrix view linked to a
 * node-link view.
 *
 * A live slot `u -> v` lands in cell `(bin(position(u), rows), bin(position(v), columns))`, where
 * `position` is the optional `order` permutation (identity by default) and, for window
 * `[start, end)`, `bin(p) = floor((p - start) * R / (end - start))` for `start <= p < end`;
 * positions outside the window are dropped. Panning or zooming rewrites the four `parameters` words and
 * the next encoding rewrites every bin, with no recompilation.
 *
 * Counts are exact `u32` atomic adds. Weight sums are fixed-point `u32` atomic adds of
 * `round(weight * weightScale)`; integer addition is associative, so results are bit-identical
 * regardless of thread order. The cost is a quantization of at most half a step per edge and
 * wraparound at 2^32 total (4,194,304 units of weight per cell at the default scale). Float
 * atomics are not used. Dead nodes keep their matrix positions but write no cells. Maxima are
 * `atomicMax` over the finished bins. `R * R` zeroing, binning and the maxima are separate nodes.
 *
 * The binning kernels use one invocation per node that walks its row, like
 * `GPUNetworkStatistics`; a hub node serializes its own row. Node positions must be a
 * permutation if the caller wants distinct rows per node; this is not validated on the GPU.
 */
export class GPUAdjacencyMatrix implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUAdjacencyMatrixProps;
  /** Number of nodes, `topology.offsets.length - 1`. */
  readonly nodeCount: number;
  /** Bins per axis. */
  readonly resolution: number;
  /** Fixed-point scale of weight sums. */
  readonly weightScale: number;

  constructor(props: GPUAdjacencyMatrixProps) {
    this.id = props.id ?? 'adjacency-matrix';
    this.props = props;
    const {id} = this;
    this.nodeCount = validateGPUNetworkTopology(id, props.topology).nodeCount;
    this.resolution = props.resolution;
    validateCapacity(id, 'resolution', props.resolution, 1, 4096);
    if (this.nodeCount * props.resolution > 0xffffffff) {
      throw new Error(`${id} node count times resolution must be below 2^32`);
    }
    this.weightScale = props.weightScale ?? GPU_ADJACENCY_MATRIX_DEFAULT_WEIGHT_SCALE;
    if (!Number.isFinite(this.weightScale) || this.weightScale <= 0) {
      throw new Error(`${id} weightScale must be positive and finite`);
    }
    for (const [name, view, length] of [
      ['nodeMask', props.nodeMask, this.nodeCount],
      ['order', props.order, this.nodeCount],
      ['edgeMask', props.edgeMask, props.topology.neighbors.length]
    ] as const) {
      if (!view) {
        continue;
      }
      validatePackedView(view, ['uint32'], `${id} ${name}`);
      if (view.length !== length) {
        throw new Error(`${id} ${name} must contain ${length} rows`);
      }
    }
    if (props.parameters) {
      validatePackedView(props.parameters, ['uint32'], `${id} parameters`);
      if (props.parameters.length < GPU_ADJACENCY_MATRIX_PARAMETER_LENGTH) {
        throw new Error(`${id} parameters must contain at least four uint32 rows`);
      }
    }
    const {output} = props;
    const cellCount = props.resolution * props.resolution;
    for (const [name, view, length] of [
      ['counts', output.counts, cellCount],
      ['weightSums', output.weightSums, cellCount],
      ['maximumCount', output.maximumCount, 1],
      ['maximumWeightSum', output.maximumWeightSum, 1]
    ] as const) {
      if (!view) {
        continue;
      }
      validatePackedView(view, ['uint32'], `${id} output.${name}`);
      if (
        name === 'maximumCount' || name === 'maximumWeightSum'
          ? view.length < 1
          : view.length !== length
      ) {
        throw new Error(
          `${id} output.${name} must contain ${length === 1 ? 'at least one row' : `exactly ${length} rows`}`
        );
      }
    }
    if (
      (output.weightSums || output.maximumWeightSum || output.textureStatistic === 'weightSum') &&
      !props.topology.weights
    ) {
      throw new Error(`${id} weight outputs require weights`);
    }
    if (output.maximumWeightSum && !output.weightSums) {
      throw new Error(`${id} output.maximumWeightSum requires output.weightSums`);
    }
    if (output.textureStatistic === 'weightSum' && !output.weightSums) {
      throw new Error(`${id} textureStatistic 'weightSum' requires output.weightSums`);
    }
    if (
      output.texture &&
      (output.texture.format !== 'r32float' ||
        output.texture.width !== props.resolution ||
        output.texture.height !== props.resolution)
    ) {
      throw new Error(`${id} output.texture must be r32float with resolution x resolution extent`);
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [output.counts, output.weightSums, output.maximumCount, output.maximumWeightSum],
      [
        ...getGPUNetworkTopologyViews(props.topology),
        props.nodeMask,
        props.edgeMask,
        props.order,
        props.parameters
      ]
    );
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, nodeCount, resolution} = this;
    const {output} = props;
    const {offsets, neighbors, weights} = props.topology;
    validateGraphViewsBelongToGraph(id, graph, [
      ...getGPUNetworkTopologyViews(props.topology),
      props.nodeMask,
      props.edgeMask,
      props.order,
      props.parameters,
      output.counts,
      output.weightSums,
      output.maximumCount,
      output.maximumWeightSum
    ]);
    if (output.texture && output.texture.texture.graph !== graph) {
      throw new Error(`${id} views must belong to the target graph`);
    }
    const operation = 'GPUAdjacencyMatrix';
    const directed = Boolean(props.directed);
    const mirror = !directed && Boolean(props.mirrorSlots);
    const nodes: GPUCommandNode<Parameters>[] = [];
    const fill = (step: string, view: GraphDataView<'uint32'> | undefined) => {
      if (view) {
        nodes.push(
          createFillNode<Parameters>(graph, {
            id: `${id}-${step}`,
            operation,
            view,
            type: 'u32',
            value: '0u'
          })
        );
      }
    };
    fill('zero-counts', output.counts);
    fill('zero-weight-sums', output.weightSums);
    fill('zero-max-count', output.maximumCount);
    fill('zero-max-weight-sum', output.maximumWeightSum);

    const windowBinding: WGSLKernelBinding[] = props.parameters
      ? [
          {
            name: 'windowWords',
            view: props.parameters,
            type: 'u32',
            access: 'read'
          }
        ]
      : [];
    const windowSource = props.parameters
      ? `let rowStart = windowWords[windowWordsOffset];
  let rowEnd = windowWords[windowWordsOffset + 1u];
  let colStart = windowWords[windowWordsOffset + 2u];
  let colEnd = windowWords[windowWordsOffset + 3u];`
      : `let rowStart = 0u;
  let rowEnd = NODE_COUNT;
  let colStart = 0u;
  let colEnd = NODE_COUNT;`;
    // One kernel bins every requested statistic, so the CSR is walked once and the bin math is
    // shared. It needs one storage binding per optional input and output, so when that exceeds the
    // device limit each statistic gets its own kernel, as before.
    const addBinKernel = (
      step: string,
      targets: readonly {kind: 'count' | 'weight'; view: GraphDataView<'uint32'>}[]
    ) => {
      const bindings: WGSLKernelBinding[] = [
        {name: 'offsets', view: offsets, type: 'u32', access: 'read'},
        {
          name: 'neighbors',
          view: neighbors,
          type: 'u32',
          access: 'read'
        },
        ...windowBinding
      ];
      if (props.order) {
        bindings.push({
          name: 'order',
          view: props.order,
          type: 'u32',
          access: 'read'
        });
      }
      if (props.nodeMask) {
        bindings.push({
          name: 'vertexMask',
          view: props.nodeMask,
          type: 'u32',
          access: 'read'
        });
      }
      if (props.edgeMask) {
        bindings.push({
          name: 'edgeMask',
          view: props.edgeMask,
          type: 'u32',
          access: 'read'
        });
      }
      if (targets.some(target => target.kind === 'weight')) {
        bindings.push({
          name: 'weights',
          view: weights!,
          type: 'f32',
          access: 'read'
        });
      }
      for (const {kind, view} of targets) {
        bindings.push({
          name: `${kind}Cells`,
          view,
          type: 'atomic<u32>',
          access: 'read_write'
        });
      }
      const amountSource = (kind: 'count' | 'weight') =>
        kind === 'count' ? '1u' : `fixedWeight(weights[weightsOffset + slot])`;
      const addSource = (cell: string) =>
        targets
          .map(
            ({kind}) =>
              `atomicAdd(&${kind}Cells[${kind}CellsOffset + ${cell}], ${amountSource(kind)});`
          )
          .join('\n      ');
      const mirrorSource = (cell: string) =>
        targets
          .map(
            ({kind}) =>
              `atomicAdd(&${kind}Cells[${kind}CellsOffset + ${cell}], ${amountSource(kind)});`
          )
          .join('\n      ');
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-${step}`,
          operation,
          variant: step,
          bindings,
          invocationCount: nodeCount,
          declarations: `const NODE_COUNT: u32 = ${nodeCount}u;
const RESOLUTION: u32 = ${resolution}u;
const WEIGHT_SCALE: f32 = ${Math.fround(this.weightScale)};
fn fixedWeight(weight: f32) -> u32 {
  let scaled = weight * WEIGHT_SCALE + 0.5;
  if (!(scaled > 0.5)) {
    return 0u;
  }
  if (scaled >= 4294967040.0) {
    return 0xffffffffu;
  }
  return u32(floor(scaled));
}
fn getBin(position: u32, start: u32, end: u32) -> u32 {
  return (position - start) * RESOLUTION / (end - start);
}`,
          body: `${windowSource}
  if (${props.nodeMask ? 'vertexMask[vertexMaskOffset + index] == 0u' : 'false'}) {
    return;
  }
  let source = ${props.order ? 'order[orderOffset + index]' : 'index'};
  // A row contributes only from its own row position, or, when mirrored, from its column
  // position too. Skipping every other row keeps a zoomed window O(window rows), not O(slots).
  if (!((source >= rowStart && source < rowEnd)${mirror ? ' || (source >= colStart && source < colEnd)' : ''})) {
    return;
  }
  let rowBegin = offsets[offsetsOffset + index];
  let rowEnd2 = offsets[offsetsOffset + index + 1u];
  for (var slot = rowBegin; slot < rowEnd2; slot++) {
    let neighbor = neighbors[neighborsOffset + slot];
    if (neighbor >= NODE_COUNT) {
      continue;
    }
    ${props.nodeMask ? 'if (vertexMask[vertexMaskOffset + neighbor] == 0u) { continue; }' : ''}
    ${props.edgeMask ? 'if (edgeMask[edgeMaskOffset + slot] == 0u) { continue; }' : ''}
    let targetPosition = ${props.order ? 'order[orderOffset + neighbor]' : 'neighbor'};
    if (source >= rowStart && source < rowEnd && targetPosition >= colStart && targetPosition < colEnd) {
      let cell = getBin(source, rowStart, rowEnd) * RESOLUTION + getBin(targetPosition, colStart, colEnd);
      ${addSource('cell')}
    }
    ${
      mirror
        ? `if (targetPosition != source && targetPosition >= rowStart && targetPosition < rowEnd && source >= colStart && source < colEnd) {
      let mirrorCell = getBin(targetPosition, rowStart, rowEnd) * RESOLUTION + getBin(source, colStart, colEnd);
      ${mirrorSource('mirrorCell')}
    }`
        : ''
    }
  }`
        })
      );
    };
    const binTargets: {kind: 'count' | 'weight'; view: GraphDataView<'uint32'>}[] = [
      {kind: 'count', view: output.counts}
    ];
    if (output.weightSums) {
      binTargets.push({kind: 'weight', view: output.weightSums});
    }
    const fusedBindingCount =
      2 +
      windowBinding.length +
      Number(Boolean(props.order)) +
      Number(Boolean(props.nodeMask)) +
      Number(Boolean(props.edgeMask)) +
      Number(binTargets.length > 1) +
      binTargets.length;
    if (
      binTargets.length === 1 ||
      fusedBindingCount <= graph.device.limits.maxStorageBuffersPerShaderStage
    ) {
      addBinKernel(binTargets.length === 1 ? 'bin-counts' : 'bin', binTargets);
    } else {
      addBinKernel('bin-counts', [binTargets[0]]);
      addBinKernel('bin-weights', [binTargets[1]]);
    }

    if (output.maximumCount || output.maximumWeightSum) {
      // One workgroup-local maximum per workgroup, then one global atomicMax per workgroup instead
      // of one per cell on a single address.
      const bindings: WGSLKernelBinding[] = [];
      let load = '';
      let reduce = '';
      let flush = '';
      let shared = '';
      if (output.maximumCount) {
        bindings.push(
          {name: 'counts', view: output.counts, type: 'u32', access: 'read'},
          {
            name: 'maximumCountOut',
            view: output.maximumCount,
            type: 'atomic<u32>',
            access: 'read_write'
          }
        );
        shared += 'var<workgroup> sharedCounts: array<u32, 256>;\n';
        load +=
          'sharedCounts[localInvocationIndex] = select(0u, counts[countsOffset + index], inRange);\n  ';
        reduce +=
          'sharedCounts[localInvocationIndex] = max(sharedCounts[localInvocationIndex], sharedCounts[localInvocationIndex + stride]);\n      ';
        flush += 'atomicMax(&maximumCountOut[maximumCountOutOffset], sharedCounts[0]);\n    ';
      }
      if (output.maximumWeightSum) {
        bindings.push(
          {
            name: 'sums',
            view: output.weightSums!,
            type: 'u32',
            access: 'read'
          },
          {
            name: 'maxSumOut',
            view: output.maximumWeightSum,
            type: 'atomic<u32>',
            access: 'read_write'
          }
        );
        shared += 'var<workgroup> sharedSums: array<u32, 256>;\n';
        load +=
          'sharedSums[localInvocationIndex] = select(0u, sums[sumsOffset + index], inRange);\n  ';
        reduce +=
          'sharedSums[localInvocationIndex] = max(sharedSums[localInvocationIndex], sharedSums[localInvocationIndex + stride]);\n      ';
        flush += 'atomicMax(&maxSumOut[maxSumOutOffset], sharedSums[0]);\n    ';
      }
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-maxima`,
          operation,
          variant: 'maxima',
          bindings,
          invocationCount: resolution * resolution,
          guardIndex: false,
          declarations: shared,
          // No early return: every invocation of a workgroup must reach the barriers.
          body: `let inRange = index < INVOCATION_COUNT;
  ${load}workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
    if (localInvocationIndex < stride) {
      ${reduce}}
    workgroupBarrier();
  }
  if (localInvocationIndex == 0u) {
    ${flush}}`
        })
      );
    }

    if (output.texture) {
      const source = output.textureStatistic === 'weightSum' ? output.weightSums! : output.counts;
      const asFloat = createTransientView(
        graph,
        `${id}-texture-values`,
        'float32',
        resolution * resolution
      );
      const scale = output.textureStatistic === 'weightSum' ? Math.fround(this.weightScale) : 1;
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-texture-values`,
          operation,
          variant: 'texture-values',
          bindings: [
            {name: 'source', view: source, type: 'u32', access: 'read'},
            {
              name: 'valuesOut',
              view: asFloat,
              type: 'f32',
              access: 'read_write'
            }
          ],
          invocationCount: resolution * resolution,
          body: `valuesOut[valuesOutOffset + index] = f32(source[sourceOffset + index]) / ${scale}${scale === 1 ? '.0' : ''};`
        })
      );
      const texture = output.texture;
      nodes.push(
        ...captureGraphCommandNodes(graph, () =>
          new GPURasterBufferToTexture({
            id: `${id}-texture`,
            input: {
              id: `${id}-texture-band`,
              format: 'float32',
              storage: {kind: 'buffer', values: asFloat}
            },
            output: texture
          }).addToGraph(graph)
        )
      );
    }
    return nodes;
  }
}
