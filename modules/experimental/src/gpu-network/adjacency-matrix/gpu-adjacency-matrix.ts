// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView,
  type GraphTextureView
} from '@luma.gl/gpgpu/gpu-core';
import {GPURasterBufferToTexture} from '../../gpu-raster/index';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {createFillNode, createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import type {WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import {
  captureGraphCommandNodes,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';

/** Number of words of the per-frame window view: `[rowStart, rowEnd, colStart, colEnd]`. */
export const GPU_ADJACENCY_MATRIX_WINDOW_LENGTH = 4;

/** Default fixed-point scale of weight sums: one unit of weight is 1024 integer steps. */
export const GPU_ADJACENCY_MATRIX_DEFAULT_WEIGHT_SCALE = 1024;

/** A zoom window in matrix positions. Ends are exclusive. */
export type GPUAdjacencyMatrixWindow = {
  rowStart: number;
  rowEnd: number;
  colStart: number;
  colEnd: number;
};

/**
 * Encodes the per-frame zoom window into the four `uint32` words of the `window` view.
 *
 * Write the result with `GPUParameterBuffer.write`. An empty window (`end <= start`)
 * produces an all-zero matrix. The GPU computes `(position - start) * resolution` in `u32`, so
 * each extent times `resolution` must stay below 2^32; this function throws otherwise.
 */
export function encodeGPUAdjacencyMatrixWindow(
  window: GPUAdjacencyMatrixWindow,
  resolution: number
): Uint32Array {
  const words = new Uint32Array([window.rowStart, window.rowEnd, window.colStart, window.colEnd]);
  for (const value of [window.rowStart, window.rowEnd, window.colStart, window.colEnd]) {
    if (!Number.isSafeInteger(value) || value < 0 || value > 0xffffffff) {
      throw new Error('GPUAdjacencyMatrix window bounds must be uint32 integers');
    }
  }
  for (const extent of [window.rowEnd - window.rowStart, window.colEnd - window.colStart]) {
    if (extent * resolution > 0xffffffff) {
      throw new Error('GPUAdjacencyMatrix window extent times resolution must be below 2^32');
    }
  }
  return words;
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
 * which optional views exist. Per-frame: CSR contents, masks, `order` and `window`.
 */
export type GPUAdjacencyMatrixProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'adjacency-matrix'`. */
  id?: string;
  /** Forward CSR row offsets with `nodeCount + 1` rows. */
  offsets: GraphDataView<'uint32'>;
  /** Forward CSR destination per slot. Indices `>= nodeCount` are dead slots. */
  neighbors: GraphDataView<'uint32'>;
  /** Optional per-slot `float32` weights aligned with `neighbors`. Required for weight outputs. */
  weights?: GraphDataView<'float32'>;
  /** Treat slots as directed edges (row = source, column = target). Default false. */
  directed?: boolean;
  /**
   * Undirected only, default false. An undirected CSR lists both directions (as in
   * `GPUNetworkStatistics` and gpu-graph), so by default each slot is written once and the matrix
   * is symmetric because the CSR is. When true, each slot `(u, v)` is also written transposed at
   * `(v, u)` (a self-loop once), for edge-list-style CSRs that list each edge once.
   */
  mirrorSlots?: boolean;
  /** Optional per-vertex mask, nonzero = live. */
  vertexMask?: GraphDataView<'uint32'>;
  /** Optional per-slot mask, nonzero = live. A slot is live iff both endpoints are live too. */
  edgeMask?: GraphDataView<'uint32'>;
  /** Optional permutation view (vertex to matrix position, `nodeCount` rows). Default identity. */
  order?: GraphDataView<'uint32'>;
  /**
   * Per-frame `'uint32'` view of four words, see {@link encodeGPUAdjacencyMatrixWindow}. When
   * omitted the window is `[0, nodeCount)` on both axes.
   */
  window?: GraphDataView<'uint32'>;
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
    maxCount?: GraphDataView<'uint32'>;
    /** Optional one-row scalar receiving the maximum of `weightSums`. */
    maxWeightSum?: GraphDataView<'uint32'>;
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
 * positions outside the window are dropped. Panning or zooming rewrites the four window words and
 * the next encoding rewrites every bin, with no recompilation.
 *
 * Counts are exact `u32` atomic adds. Weight sums are fixed-point `u32` atomic adds of
 * `round(weight * weightScale)`; integer addition is associative, so results are bit-identical
 * regardless of thread order. The cost is a quantization of at most half a step per edge and
 * wraparound at 2^32 total (4,194,304 units of weight per cell at the default scale). Float
 * atomics are not used. Dead vertices keep their matrix positions but write no cells. Maxima are
 * `atomicMax` over the finished bins. `R * R` zeroing, binning and the maxima are separate nodes.
 *
 * The binning kernels use one invocation per vertex that walks its row, like
 * `GPUNetworkStatistics`; a hub vertex serializes its own row. Vertex positions must be a
 * permutation if the caller wants distinct rows per vertex; this is not validated on the GPU.
 */
export class GPUAdjacencyMatrix implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUAdjacencyMatrixProps;
  /** Number of vertices, `offsets.length - 1`. */
  readonly nodeCount: number;
  /** Bins per axis. */
  readonly resolution: number;
  /** Fixed-point scale of weight sums. */
  readonly weightScale: number;

  constructor(props: GPUAdjacencyMatrixProps) {
    this.id = props.id ?? 'adjacency-matrix';
    this.props = props;
    const {id} = this;
    validatePackedView(props.offsets, ['uint32'], `${id} offsets`);
    validatePackedView(props.neighbors, ['uint32'], `${id} neighbors`);
    this.nodeCount = props.offsets.length - 1;
    if (this.nodeCount < 1) {
      throw new Error(`${id} offsets must contain at least two rows`);
    }
    this.resolution = props.resolution;
    if (
      !Number.isSafeInteger(props.resolution) ||
      props.resolution < 1 ||
      props.resolution > 4096
    ) {
      throw new Error(`${id} resolution must be an integer between 1 and 4096`);
    }
    if (this.nodeCount * props.resolution > 0xffffffff) {
      throw new Error(`${id} node count times resolution must be below 2^32`);
    }
    this.weightScale = props.weightScale ?? GPU_ADJACENCY_MATRIX_DEFAULT_WEIGHT_SCALE;
    if (!Number.isFinite(this.weightScale) || this.weightScale <= 0) {
      throw new Error(`${id} weightScale must be positive and finite`);
    }
    for (const [name, view, length] of [
      ['vertexMask', props.vertexMask, this.nodeCount],
      ['order', props.order, this.nodeCount],
      ['edgeMask', props.edgeMask, props.neighbors.length]
    ] as const) {
      if (!view) {
        continue;
      }
      validatePackedView(view, ['uint32'], `${id} ${name}`);
      if (view.length !== length) {
        throw new Error(`${id} ${name} must contain ${length} rows`);
      }
    }
    if (props.weights) {
      validatePackedView(props.weights, ['float32'], `${id} weights`);
      if (props.weights.length !== props.neighbors.length) {
        throw new Error(`${id} weights must contain one row per neighbor slot`);
      }
    }
    if (props.window) {
      validatePackedView(props.window, ['uint32'], `${id} window`);
      if (props.window.length < GPU_ADJACENCY_MATRIX_WINDOW_LENGTH) {
        throw new Error(`${id} window must contain at least four uint32 rows`);
      }
    }
    const {output} = props;
    const cellCount = props.resolution * props.resolution;
    for (const [name, view, length] of [
      ['counts', output.counts, cellCount],
      ['weightSums', output.weightSums, cellCount],
      ['maxCount', output.maxCount, 1],
      ['maxWeightSum', output.maxWeightSum, 1]
    ] as const) {
      if (!view) {
        continue;
      }
      validatePackedView(view, ['uint32'], `${id} output.${name}`);
      if (
        name === 'maxCount' || name === 'maxWeightSum' ? view.length < 1 : view.length !== length
      ) {
        throw new Error(
          `${id} output.${name} must contain ${length === 1 ? 'at least one row' : `exactly ${length} rows`}`
        );
      }
    }
    if (
      (output.weightSums || output.maxWeightSum || output.textureStatistic === 'weightSum') &&
      !props.weights
    ) {
      throw new Error(`${id} weight outputs require weights`);
    }
    if (output.maxWeightSum && !output.weightSums) {
      throw new Error(`${id} output.maxWeightSum requires output.weightSums`);
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
      [output.counts, output.weightSums, output.maxCount, output.maxWeightSum],
      [
        props.offsets,
        props.neighbors,
        props.weights,
        props.vertexMask,
        props.edgeMask,
        props.order,
        props.window
      ]
    );
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, nodeCount, resolution} = this;
    const {output} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      props.offsets,
      props.neighbors,
      props.weights,
      props.vertexMask,
      props.edgeMask,
      props.order,
      props.window,
      output.counts,
      output.weightSums,
      output.maxCount,
      output.maxWeightSum
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
    fill('zero-max-count', output.maxCount);
    fill('zero-max-weight-sum', output.maxWeightSum);

    const windowBinding: WGSLKernelBinding[] = props.window
      ? [
          {
            name: 'windowWords',
            view: props.window,
            type: 'u32',
            access: 'read'
          }
        ]
      : [];
    const windowSource = props.window
      ? `let rowStart = windowWords[windowWordsOffset];
  let rowEnd = windowWords[windowWordsOffset + 1u];
  let colStart = windowWords[windowWordsOffset + 2u];
  let colEnd = windowWords[windowWordsOffset + 3u];`
      : `let rowStart = 0u;
  let rowEnd = NODE_COUNT;
  let colStart = 0u;
  let colEnd = NODE_COUNT;`;
    const addBinKernel = (
      step: string,
      target: GraphDataView<'uint32'>,
      kind: 'count' | 'weight'
    ) => {
      const bindings: WGSLKernelBinding[] = [
        {name: 'offsets', view: props.offsets, type: 'u32', access: 'read'},
        {
          name: 'neighbors',
          view: props.neighbors,
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
      if (props.vertexMask) {
        bindings.push({
          name: 'vertexMask',
          view: props.vertexMask,
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
      if (kind === 'weight') {
        bindings.push({
          name: 'weights',
          view: props.weights!,
          type: 'f32',
          access: 'read'
        });
      }
      bindings.push({
        name: 'cells',
        view: target,
        type: 'atomic<u32>',
        access: 'read_write'
      });
      const amount = kind === 'count' ? '1u' : `fixedWeight(weights[weightsOffset + slot])`;
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
  if (${props.vertexMask ? 'vertexMask[vertexMaskOffset + index] == 0u' : 'false'}) {
    return;
  }
  let source = ${props.order ? 'order[orderOffset + index]' : 'index'};
  let rowBegin = offsets[offsetsOffset + index];
  let rowEnd2 = offsets[offsetsOffset + index + 1u];
  for (var slot = rowBegin; slot < rowEnd2; slot++) {
    let neighbor = neighbors[neighborsOffset + slot];
    if (neighbor >= NODE_COUNT) {
      continue;
    }
    ${props.vertexMask ? 'if (vertexMask[vertexMaskOffset + neighbor] == 0u) { continue; }' : ''}
    ${props.edgeMask ? 'if (edgeMask[edgeMaskOffset + slot] == 0u) { continue; }' : ''}
    let targetPosition = ${props.order ? 'order[orderOffset + neighbor]' : 'neighbor'};
    let amount = ${amount};
    if (source >= rowStart && source < rowEnd && targetPosition >= colStart && targetPosition < colEnd) {
      atomicAdd(&cells[cellsOffset + getBin(source, rowStart, rowEnd) * RESOLUTION + getBin(targetPosition, colStart, colEnd)], amount);
    }
    ${
      mirror
        ? `if (targetPosition != source && targetPosition >= rowStart && targetPosition < rowEnd && source >= colStart && source < colEnd) {
      atomicAdd(&cells[cellsOffset + getBin(targetPosition, rowStart, rowEnd) * RESOLUTION + getBin(source, colStart, colEnd)], amount);
    }`
        : ''
    }
  }`
        })
      );
    };
    addBinKernel('bin-counts', output.counts, 'count');
    if (output.weightSums) {
      addBinKernel('bin-weights', output.weightSums, 'weight');
    }

    if (output.maxCount || output.maxWeightSum) {
      const bindings: WGSLKernelBinding[] = [];
      let body = '';
      if (output.maxCount) {
        bindings.push(
          {name: 'counts', view: output.counts, type: 'u32', access: 'read'},
          {
            name: 'maxCountOut',
            view: output.maxCount,
            type: 'atomic<u32>',
            access: 'read_write'
          }
        );
        body += 'atomicMax(&maxCountOut[maxCountOutOffset], counts[countsOffset + index]);\n  ';
      }
      if (output.maxWeightSum) {
        bindings.push(
          {
            name: 'sums',
            view: output.weightSums!,
            type: 'u32',
            access: 'read'
          },
          {
            name: 'maxSumOut',
            view: output.maxWeightSum,
            type: 'atomic<u32>',
            access: 'read_write'
          }
        );
        body += 'atomicMax(&maxSumOut[maxSumOutOffset], sums[sumsOffset + index]);';
      }
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-maxima`,
          operation,
          variant: 'maxima',
          bindings,
          invocationCount: resolution * resolution,
          body
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
