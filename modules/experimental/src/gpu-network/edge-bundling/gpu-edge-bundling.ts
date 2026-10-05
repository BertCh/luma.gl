// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {validateGraphViewsBelongToGraph} from '../../utils/gpu-contributor-utils';
import {
  createEdgeBundlingBoxNode,
  createEdgeBundlingBoxResetNode,
  createEdgeBundlingClearNode,
  createEdgeBundlingFinalizeNode,
  createEdgeBundlingIndicesNode,
  createEdgeBundlingInitializeNode,
  createEdgeBundlingSplatNode,
  createEdgeBundlingUpdateNode,
  type EdgeBundlingConstants,
  type EdgeBundlingParameterView
} from './edge-bundling-passes';

/** Largest accepted compile-time `iterations`. */
export const GPU_EDGE_BUNDLING_MAXIMUM_ITERATIONS = 64;
/** Largest accepted compile-time `pointsPerEdge`. */
export const GPU_EDGE_BUNDLING_MAXIMUM_POINTS_PER_EDGE = 64;
/** Row count of the parameter layout: active iterations, radius, lambda, smoothing, step scale. */
export const GPU_EDGE_BUNDLING_PARAMETER_LENGTH = 5;
/**
 * Padding added on each side of the live-endpoint bounding square, as a fraction of its side.
 * The work box side is `maxSide * (1 + 2 * padding)`.
 */
export const GPU_EDGE_BUNDLING_WORK_BOX_PADDING = 0.05;

/** Defaults for compile-time props and for per-frame parameters omitted by the caller. */
export const GPU_EDGE_BUNDLING_DEFAULTS = {
  pointsPerEdge: 16,
  iterations: 15,
  densityResolution: 256,
  /** Initial kernel radius as a fraction of the square work box side. */
  kernelRadius: 0.03,
  /** Per-iteration radius decay, clamped to `[0.5, 0.9]` on the GPU. */
  lambda: 0.85,
  /** Laplacian smoothing strength per iteration, clamped to `[0, 1]` on the GPU. */
  smoothing: 0.5,
  /** Multiplier on the advection step, which is one kernel radius at scale 1. */
  stepScale: 1
} as const;

const MAXIMUM_FIXED_POINT_EXPONENT = 16;
const MINIMUM_FIXED_POINT_EXPONENT = 8;
const MINIMUM_DENSITY_RESOLUTION = 8;

/**
 * Returns the fixed-point exponent `k` such that density weights are quantized to `2^k` units.
 *
 * It is the largest `k <= 16` with `pointCount * 2^k <= 2^32 - 1`. A single point adds at most
 * `2^k` to a cell, so even if every control point landed in one cell the `atomic<u32>` could not
 * overflow. For 160 000 control points `k` is 14. Returns 0 when even `2^0` would overflow.
 */
export function getGPUEdgeBundlingFixedPointExponent(pointCount: number): number {
  let exponent = MAXIMUM_FIXED_POINT_EXPONENT;
  while (exponent > 0 && pointCount * 2 ** exponent > 0xffffffff) {
    exponent--;
  }
  return exponent;
}

/** Per-frame bundling parameters, all optional. Omitted values take the defaults. */
export type GPUEdgeBundlingParameterValues = {
  /** Iterations to run, clamped to the compiled `iterations`. Default: the compiled maximum. */
  activeIterations?: number;
  /** Initial kernel radius as a fraction of the work box side. Default 0.03. */
  kernelRadius?: number;
  /** Radius decay per iteration, clamped to `[0.5, 0.9]`. Default 0.85. */
  lambda?: number;
  /** Laplacian smoothing strength, clamped to `[0, 1]`. Default 0.5. */
  smoothing?: number;
  /** Advection step multiplier. Default 1. */
  stepScale?: number;
};

/**
 * Packs per-frame parameters into the layout read by {@link GPUEdgeBundling}.
 *
 * For `'uint32'`, word 0 is the integer iteration count and words 1 to 4 are the float bit
 * patterns of radius, lambda, smoothing and step scale. For `'float32'` every word is a float.
 * Write the result with `GPUParameterBuffer.write`.
 */
export function createGPUEdgeBundlingParameterValues(
  values: GPUEdgeBundlingParameterValues,
  format: 'uint32'
): Uint32Array;
export function createGPUEdgeBundlingParameterValues(
  values: GPUEdgeBundlingParameterValues,
  format: 'float32'
): Float32Array;
export function createGPUEdgeBundlingParameterValues(
  values: GPUEdgeBundlingParameterValues,
  format: 'uint32' | 'float32'
): Uint32Array | Float32Array {
  const floats = Float32Array.of(
    values.activeIterations ?? GPU_EDGE_BUNDLING_MAXIMUM_ITERATIONS,
    values.kernelRadius ?? GPU_EDGE_BUNDLING_DEFAULTS.kernelRadius,
    values.lambda ?? GPU_EDGE_BUNDLING_DEFAULTS.lambda,
    values.smoothing ?? GPU_EDGE_BUNDLING_DEFAULTS.smoothing,
    values.stepScale ?? GPU_EDGE_BUNDLING_DEFAULTS.stepScale
  );
  if (format === 'float32') {
    return floats;
  }
  const words = new Uint32Array(floats.buffer.slice(0));
  words[0] = Math.max(
    0,
    Math.floor(values.activeIterations ?? GPU_EDGE_BUNDLING_MAXIMUM_ITERATIONS)
  );
  return words;
}

/**
 * Properties for {@link GPUEdgeBundling}.
 *
 * Compile-time: edge and vertex counts, `pointsPerEdge`, `iterations`, `densityResolution`, and
 * which optional views exist. Per-frame: the contents of `positions`, endpoints, `edgeMask` and
 * `parameters`.
 */
export type GPUEdgeBundlingProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'edge-bundling'`. */
  id?: string;
  /** Vertex positions, one `float32x2` row per vertex, for example a layout's output. */
  positions: GraphDataView<'float32x2'>;
  /** Source vertex per edge. Its length defines the edge count. Out-of-range makes the edge dead. */
  sourceVertices: GraphDataView<'uint32'>;
  /** Target vertex per edge, same length as `sourceVertices`. Out-of-range makes the edge dead. */
  targetVertices: GraphDataView<'uint32'>;
  /**
   * Optional per-edge liveness, nonzero is live. Dead edges (masked, out-of-range, or with
   * non-finite endpoint positions) do not splat or affect the work box, and every point of their
   * output path equals the source vertex position (or `(0, 0)` when the source is out of range),
   * so a fixed draw layout keeps working.
   */
  edgeMask?: GraphDataView<'uint32'>;
  /** Compile-time control points per edge, 2 to 64. Defaults to 16. */
  pointsPerEdge?: number;
  /** Compile-time maximum iteration count, 1 to 64. Defaults to 15. */
  iterations?: number;
  /** Compile-time density grid resolution per axis, at least 8. Defaults to 256. */
  densityResolution?: number;
  /**
   * Optional per-frame parameters with at least 4 rows (5 to include `stepScale`):
   * `[activeIterations, kernelRadius, lambda, smoothing, stepScale]`. A `uint32` view holds the
   * count as an integer and the rest as float bit patterns; a `float32` view holds all as floats.
   * See {@link createGPUEdgeBundlingParameterValues}. Changing them never recompiles.
   */
  parameters?: EdgeBundlingParameterView;
  /**
   * Output polylines, `edgeCount * pointsPerEdge` rows, edge-major (edge `e` point `i` at row
   * `e * pointsPerEdge + i`), in the caller's coordinates. Endpoints equal the vertex positions
   * exactly.
   */
  paths: GraphDataView<'float32x2'>;
  /** Optional `edgeCount + 1` rows holding `e * pointsPerEdge`, for PathLayer binary attributes. */
  startIndices?: GraphDataView<'uint32'>;
  /**
   * Optional 4-word `drawIndirect` record `[pointsPerEdge, edgeCount, 0, 0]` (vertexCount,
   * instanceCount, firstVertex, firstInstance). Draw an instanced line strip where the vertex
   * shader reads `paths[instance_index * pointsPerEdge + vertex_index]`.
   */
  drawRecord?: GraphDataView<'uint32'>;
};

/**
 * Kernel-density edge bundling (KDEEB) as WebGPU compute, producing renderable polylines.
 *
 * Each edge starts as a straight subdivision, then every iteration clears a density grid, splats
 * Epanechnikov weights of every control point as fixed-point `atomic<u32>` into a storage buffer,
 * advects interior points one kernel radius along the bilinearly sampled normalized gradient,
 * resamples to uniform arc length, and applies one Laplacian smoothing pass from a snapshot. The
 * radius anneals as `radius0 * lambda^k`. Endpoints are pinned.
 *
 * All lengths are relative to a square work box computed on the GPU every encoding from live-edge
 * endpoints (bounding square, side padded by 5% per side), so results are scale invariant. The
 * density is a storage buffer, so there is no float-renderable texture requirement or texture
 * size cap. Weights are quantized to `2^k` units, with `k` from
 * {@link getGPUEdgeBundlingFixedPointExponent}, which cannot overflow.
 *
 * Per iteration: clear, splat, update (advect, resample, smooth in one invocation per edge).
 * Extra iteration nodes beyond `activeIterations` return immediately.
 */
export class GPUEdgeBundling implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUEdgeBundlingProps;
  /** Resolved compile-time control points per edge. */
  readonly pointsPerEdge: number;
  /** Resolved compile-time maximum iterations. */
  readonly iterations: number;
  /** Resolved compile-time density resolution. */
  readonly densityResolution: number;
  /** Number of edges. */
  readonly edgeCount: number;
  /** Fixed-point exponent of the density quantization. */
  readonly fixedPointExponent: number;

  constructor(props: GPUEdgeBundlingProps) {
    this.id = props.id ?? 'edge-bundling';
    this.props = props;
    this.pointsPerEdge = props.pointsPerEdge ?? GPU_EDGE_BUNDLING_DEFAULTS.pointsPerEdge;
    this.iterations = props.iterations ?? GPU_EDGE_BUNDLING_DEFAULTS.iterations;
    this.densityResolution =
      props.densityResolution ?? GPU_EDGE_BUNDLING_DEFAULTS.densityResolution;
    this.edgeCount = props.sourceVertices.length;
    const {id, pointsPerEdge, iterations, densityResolution, edgeCount} = this;
    validatePackedView(props.positions, ['float32x2'], `${id} positions`);
    validatePackedView(props.paths, ['float32x2'], `${id} paths`);
    for (const [name, view] of [
      ['sourceVertices', props.sourceVertices],
      ['targetVertices', props.targetVertices],
      ['edgeMask', props.edgeMask],
      ['startIndices', props.startIndices],
      ['drawRecord', props.drawRecord]
    ] as const) {
      if (view) {
        validatePackedUint32View(view, `${id} ${name}`);
      }
    }
    if (props.parameters) {
      validatePackedView(props.parameters, ['uint32', 'float32'], `${id} parameters`);
      if (props.parameters.length < GPU_EDGE_BUNDLING_PARAMETER_LENGTH - 1) {
        throw new Error(`${id} parameters must contain at least 4 rows`);
      }
    }
    if (edgeCount < 1) {
      throw new Error(`${id} must have at least one edge`);
    }
    if (props.positions.length < 1) {
      throw new Error(`${id} positions must contain at least one vertex`);
    }
    if (props.targetVertices.length !== edgeCount) {
      throw new Error(`${id} targetVertices length must equal sourceVertices length`);
    }
    if (props.edgeMask && props.edgeMask.length !== edgeCount) {
      throw new Error(`${id} edgeMask length must equal the edge count`);
    }
    if (
      !Number.isSafeInteger(pointsPerEdge) ||
      pointsPerEdge < 2 ||
      pointsPerEdge > GPU_EDGE_BUNDLING_MAXIMUM_POINTS_PER_EDGE
    ) {
      throw new Error(
        `${id} pointsPerEdge must be an integer in [2, ${GPU_EDGE_BUNDLING_MAXIMUM_POINTS_PER_EDGE}]`
      );
    }
    if (
      !Number.isSafeInteger(iterations) ||
      iterations < 1 ||
      iterations > GPU_EDGE_BUNDLING_MAXIMUM_ITERATIONS
    ) {
      throw new Error(
        `${id} iterations must be an integer in [1, ${GPU_EDGE_BUNDLING_MAXIMUM_ITERATIONS}]`
      );
    }
    if (
      !Number.isSafeInteger(densityResolution) ||
      densityResolution < MINIMUM_DENSITY_RESOLUTION
    ) {
      throw new Error(`${id} densityResolution must be an integer of at least 8`);
    }
    if (props.paths.length !== edgeCount * pointsPerEdge) {
      throw new Error(`${id} paths length must equal edgeCount * pointsPerEdge`);
    }
    if (props.startIndices && props.startIndices.length !== edgeCount + 1) {
      throw new Error(`${id} startIndices length must equal edgeCount + 1`);
    }
    if (props.drawRecord && props.drawRecord.length !== 4) {
      throw new Error(`${id} drawRecord must contain exactly four rows`);
    }
    this.fixedPointExponent = getGPUEdgeBundlingFixedPointExponent(edgeCount * pointsPerEdge);
    if (this.fixedPointExponent < MINIMUM_FIXED_POINT_EXPONENT) {
      throw new Error(`${id} edgeCount * pointsPerEdge is too large for 32-bit density`);
    }
    const outputs = [props.paths, props.startIndices, props.drawRecord]
      .filter(view => view !== undefined)
      .map(view => view.buffer);
    const inputs = [
      props.positions,
      props.sourceVertices,
      props.targetVertices,
      props.edgeMask,
      props.parameters
    ]
      .filter(view => view !== undefined)
      .map(view => view.buffer);
    if (
      new Set(outputs).size !== outputs.length ||
      outputs.some(buffer => inputs.includes(buffer))
    ) {
      throw new Error(`${id} outputs must use separate buffers from each other and from inputs`);
    }
  }

  /**
   * Returns box reset and accumulate, initialize, three nodes per iteration, finalize, and an
   * optional indices node: `7 + 3 * iterations` nodes at most.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, iterations, pointsPerEdge, densityResolution, edgeCount} = this;
    validateGraphViewsBelongToGraph(id, graph, [
      props.positions,
      props.sourceVertices,
      props.targetVertices,
      props.edgeMask,
      props.parameters,
      props.paths,
      props.startIndices,
      props.drawRecord
    ]);
    const densityByteLength = densityResolution * densityResolution * Uint32Array.BYTES_PER_ELEMENT;
    if (densityByteLength > graph.device.limits.maxStorageBufferBindingSize) {
      throw new Error(`${id} density grid exceeds maxStorageBufferBindingSize`);
    }
    const constants: EdgeBundlingConstants = {
      edgeCount,
      vertexCount: props.positions.length,
      pointsPerEdge,
      densityResolution,
      fixedPointExponent: this.fixedPointExponent,
      iterationCount: iterations,
      boxPadding: GPU_EDGE_BUNDLING_WORK_BOX_PADDING
    };
    const inputs = {
      positions: props.positions,
      sources: props.sourceVertices,
      targets: props.targetVertices,
      mask: props.edgeMask
    };
    const boxKeys = createTransientView(graph, `${id}-box`, 'uint32', 4);
    const density = createTransientView(
      graph,
      `${id}-density`,
      'uint32',
      densityResolution * densityResolution
    );
    const work = createTransientView(graph, `${id}-work`, 'float32x2', edgeCount * pointsPerEdge);
    const nodes: GPUCommandNode<Parameters>[] = [
      createEdgeBundlingBoxResetNode<Parameters>(graph, {
        id: `${id}-box-reset`,
        boxKeys
      }),
      createEdgeBundlingBoxNode<Parameters>(graph, {
        id: `${id}-box`,
        constants,
        inputs,
        boxKeys
      }),
      createEdgeBundlingInitializeNode<Parameters>(graph, {
        id: `${id}-initialize`,
        constants,
        inputs,
        boxKeys,
        work
      })
    ];
    for (let iteration = 0; iteration < iterations; iteration++) {
      const iterationProps = {
        constants,
        iteration,
        parameters: props.parameters,
        work,
        density
      };
      nodes.push(
        createEdgeBundlingClearNode<Parameters>(graph, {
          id: `${id}-clear-${iteration}`,
          density
        }),
        createEdgeBundlingSplatNode<Parameters>(graph, {
          id: `${id}-splat-${iteration}`,
          ...iterationProps
        }),
        createEdgeBundlingUpdateNode<Parameters>(graph, {
          id: `${id}-update-${iteration}`,
          ...iterationProps
        })
      );
    }
    nodes.push(
      createEdgeBundlingFinalizeNode<Parameters>(graph, {
        id: `${id}-finalize`,
        constants,
        inputs,
        boxKeys,
        work,
        paths: props.paths
      })
    );
    if (props.startIndices || props.drawRecord) {
      nodes.push(
        createEdgeBundlingIndicesNode<Parameters>(graph, {
          id: `${id}-indices`,
          constants,
          startIndices: props.startIndices,
          drawRecord: props.drawRecord
        })
      );
    }
    return nodes;
  }
}
