// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {createFillNode, createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validatePackedView
} from '../../utils/gpu-contributor-utils';
import {
  createRasterIterationFinalizeNode,
  createRasterIterationGateNode,
  createRasterIterationResetNode,
  createRasterIterationState,
  getRasterIterationCondition,
  validateRasterIterations
} from '../../gpu-raster/cost-distance/raster-relaxation';
import {
  type GPUSpatialWeights,
  validateGPUSpatialWeights
} from '../spatial-weights/spatial-weights';
import type {GPUIterationStatusPort} from '../../analysis-core/contracts/status';

const OPERATION = 'GPUMapColoring';

/** Degree comparison of the largest-degree-first order, placed before the hash comparison. */
const DEGREE_COMPARISON_WGSL = `
  if (otherDegree != rowDegree) {
    return otherDegree > rowDegree;
  }`;

/** Color value of a row that was not colored (only when `converged` is 0). */
export const GPU_MAP_COLORING_UNCOLORED = 0xffffffff;

/** Default compile-time cap on coloring rounds. */
export const GPU_MAP_COLORING_DEFAULT_MAXIMUM_ITERATIONS = 64;

/** Number of `uint32` words in the per-frame `parameters` view of {@link GPUMapColoring}. */
export const GPU_MAP_COLORING_PARAMETER_LENGTH = 1;

/**
 * Order in which rows claim colors.
 *
 * - `'largest-degree-first'`: rows with more neighbors go first (Welsh-Powell order inside
 *   Jones-Plassmann). Ties break by the seeded hash, then by lower row ID.
 * - `'random'`: seeded hash only, then lower row ID.
 */
export type GPUMapColoringPriority = 'largest-degree-first' | 'random';

/** Per-frame settings of {@link GPUMapColoring}, packed by {@link getGPUMapColoringParameterValues}. */
export type GPUMapColoringSettings = {
  /**
   * Seed of the hashed priorities, a uint32. Default 0. Different seeds give different valid
   * colorings (with `'largest-degree-first'` the seed only orders rows of equal degree).
   */
  seed?: number;
};

/**
 * Packs {@link GPUMapColoringSettings} into the {@link GPU_MAP_COLORING_PARAMETER_LENGTH} `uint32`
 * words of the `parameters` view: `[seed]`.
 *
 * @throws If `seed` is not an integer in `[0, 2^32 - 1]`.
 */
export function getGPUMapColoringParameterValues(
  settings: GPUMapColoringSettings = {}
): Uint32Array {
  const seed = settings.seed ?? 0;
  if (!Number.isInteger(seed) || seed < 0 || seed > 0xffffffff) {
    throw new Error('GPUMapColoring seed must be a uint32');
  }
  return Uint32Array.of(seed);
}

/** Caller-owned outputs of {@link GPUMapColoring}. */
export type GPUMapColoringOutput = {
  /**
   * Color index per row, from 0. A row that did not finish within `maximumIterations` holds
   * {@link GPU_MAP_COLORING_UNCOLORED}.
   */
  colors: GraphDataView<'uint32'>;
  /** Optional one-row output: number of colors used (largest color plus one). */
  colorCount?: GraphDataView<'uint32'>;
  /**
   * Optional one-row output: number of neighbor pairs `(i, j)` with `j > i` whose rows share a
   * color. 0 means the coloring is proper (for a symmetric pattern). Uncolored rows never conflict.
   */
  conflictCount?: GraphDataView<'uint32'>;
  /**
   * Optional status words: `converged` is 1 when every row was colored within
   * `maximumIterations`, else 0; `iterationCount` is the number of rounds that ran.
   */
  status?: Pick<GPUIterationStatusPort, 'converged' | 'iterationCount'>;
};

/**
 * Properties for {@link GPUMapColoring}.
 *
 * Compile-time (needs a new graph): view lengths, `maximumIterations` and which optional outputs
 * are present. Per-frame: the contents of `weights` and `parameters` (the seed).
 */
export type GPUMapColoringProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'map-coloring'`. */
  id?: string;
  /**
   * Square self-join adjacency, usually `GPUContiguityWeights` output (rook for maps, queen to also
   * separate polygons that touch at a point). Only the pattern is used, so weights of any value
   * work. The pattern must be symmetric (`j` in row `i` implies `i` in row `j`); an asymmetric
   * pattern can yield a color conflict, which `conflictCount` reports.
   * Weights from `GPUNeighborSearch` carry their own `spatialContext`; topology-only weights
   * may omit it. No context check is needed because coloring uses only adjacency.
   */
  weights: GPUSpatialWeights;
  /**
   * Per-frame settings: {@link GPU_MAP_COLORING_PARAMETER_LENGTH} `uint32` words written with
   * {@link getGPUMapColoringParameterValues}.
   */
  parameters: GraphDataView<'uint32'>;
  /** Caller-owned outputs. */
  output: GPUMapColoringOutput;
  /**
   * Compile-time cap on rounds, in `[1, 1024]`. Default 64. Each round adds two nodes, and
   * rounds after convergence are skipped on the GPU. Hashed priorities need about `O(log n)` rounds
   * on map graphs; the longest decreasing-priority chain bounds the count. Largest-degree-first
   * chains are a short run per distinct degree, so planar maps need a few rounds more than
   * `'random'`; the default is still ample.
   */
  maximumIterations?: number;
  /**
   * Compile-time row order, `'largest-degree-first'` (default) or `'random'`. The result is a
   * proper, seed-deterministic coloring equal to sequential greedy in that order. Largest-degree-first
   * usually needs fewer colors. The degree of a row is its CSR row length, so self loops and
   * duplicate entries count (a contiguity pattern has neither); the order stays a strict total
   * order, so correctness never depends on the degree being exact.
   */
  priority?: GPUMapColoringPriority;
};

/**
 * Greedy parallel graph coloring of polygon contiguity (Jones-Plassmann, largest degree first), so adjacent polygons
 * never share a color: the classic "map coloring" for choropleth-free categorical maps.
 *
 * **Algorithm.** Rows are ordered by priority: by default descending degree (CSR row length), ties
 * broken by a seeded integer hash of the row ID, then by the lowest ID (`priority: 'random'` uses
 * the hash and ID only). In each round every uncolored row whose priority is higher than
 * every uncolored neighbor takes the lowest color not used by its colored neighbors (first free
 * slot of 32-color bit windows). Such rows are never adjacent. Each round is one fused pass that
 * reads and writes colors with atomics (see the kernel comment); the result equals sequential greedy coloring in priority order and is deterministic for a given
 * seed.
 *
 * **Quality.** The color count is not minimal: greedy coloring uses at most `maxDegree + 1`
 * colors, and the four-color theorem's bound is not targeted. Largest-degree-first typically saves
 * a color over random order; expect about 4 to 5 colors on typical planar subdivisions. Try several
 * seeds (graphs per seed) and keep the one with the fewest colors if that matters.
 *
 * **Verification.** `conflictCount` counts neighbor pairs that share a color; it is 0 for a symmetric
 * pattern. `output.status.converged` reports whether every row finished within `maximumIterations`.
 *
 * Consumes a contiguity (adjacency) pattern only, so it takes no `spatialContext`.
 */
export class GPUMapColoring implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUMapColoringProps;
  /** Compile-time round cap. */
  readonly maximumIterations: number;
  /** Row order. */
  readonly priority: GPUMapColoringPriority;

  constructor(props: GPUMapColoringProps) {
    const id = props.id ?? 'map-coloring';
    this.id = id;
    this.props = props;
    this.maximumIterations = props.maximumIterations ?? GPU_MAP_COLORING_DEFAULT_MAXIMUM_ITERATIONS;
    validateRasterIterations(id, 'maximumIterations', this.maximumIterations);
    this.priority = props.priority ?? 'largest-degree-first';
    if (this.priority !== 'largest-degree-first' && this.priority !== 'random') {
      throw new Error(`${id} priority must be 'largest-degree-first' or 'random'`);
    }
    const rows = validateGPUSpatialWeights(id, props.weights);
    const {output} = props;
    validatePackedView(props.parameters, ['uint32'], `${id} parameters`);
    if (props.parameters.length < GPU_MAP_COLORING_PARAMETER_LENGTH) {
      throw new Error(`${id} parameters must hold ${GPU_MAP_COLORING_PARAMETER_LENGTH} uint32`);
    }
    validatePackedView(output.colors, ['uint32'], `${id} output.colors`);
    if (output.colors.length !== rows) {
      throw new Error(`${id} output.colors length must equal the weights row count`);
    }
    for (const [name, view] of [
      ['colorCount', output.colorCount],
      ['conflictCount', output.conflictCount],
      ['status.converged', output.status?.converged],
      ['status.iterationCount', output.status?.iterationCount]
    ] as const) {
      if (view) {
        validatePackedView(view, ['uint32'], `${id} output.${name}`);
        if (view.length < 1) {
          throw new Error(`${id} output.${name} must hold one uint32`);
        }
      }
    }
    validateGraphOutputsDisjointFromInputs(id, getOutputs(output), [
      props.weights.offsets,
      props.weights.neighbors,
      props.weights.weights,
      props.weights.distances,
      props.parameters
    ]);
  }

  /** Returns the clear, gated round, and verification nodes in dependency order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, maximumIterations, priority} = this;
    const {weights, parameters, output} = props;
    const {colors} = output;
    validateGraphViewsBelongToGraph(id, graph, [
      weights.offsets,
      weights.neighbors,
      weights.weights,
      weights.distances,
      parameters,
      ...getOutputs(output)
    ]);
    const rows = colors.length;
    const state = createRasterIterationState(graph, `${id}-rounds`, OPERATION, rows, id);
    const colorCount =
      output.colorCount ?? createTransientView(graph, `${id}-color-count`, 'uint32', 1);
    // The seed is read per frame from `parameters`, so the source is the same for every seed.
    const degreeFirst = priority === 'largest-degree-first';
    const priorityWGSL = `const UNCOLORED: u32 = ${GPU_MAP_COLORING_UNCOLORED}u;

fn priorityHash(value: u32, seed: u32) -> u32 {
  var h = value ^ seed;
  h = h ^ (h >> 16u);
  h = h * 0x7feb352du;
  h = h ^ (h >> 15u);
  h = h * 0x846ca68bu;
  h = h ^ (h >> 16u);
  return h;
}

// True when row 'other' is claimed before row 'index' (a strict total order: no two rows tie).
fn outranks(other: u32, otherDegree: u32, index: u32, rowDegree: u32, seed: u32) -> bool {${degreeFirst ? DEGREE_COMPARISON_WGSL : ''}
  let otherHash = priorityHash(other, seed);
  let rowHash = priorityHash(index, seed);
  return otherHash > rowHash || (otherHash == rowHash && other < index);
}`;

    const nodes: GPUCommandNode<Parameters>[] = [
      createFillNode<Parameters>(graph, {
        id: `${id}-colors-clear`,
        operation: OPERATION,
        view: colors,
        type: 'u32',
        value: `${GPU_MAP_COLORING_UNCOLORED}u`
      }),
      createFillNode<Parameters>(graph, {
        id: `${id}-color-count-clear`,
        operation: OPERATION,
        view: colorCount,
        type: 'u32',
        value: '0u',
        componentCount: 1
      }),
      createRasterIterationResetNode<Parameters>(graph, {
        id: `${id}-rounds-reset`,
        operation: OPERATION,
        state
      })
    ];
    for (let round = 0; round < maximumIterations; round++) {
      const roundId = `${id}-round-${round}`;
      const gate = getRasterIterationCondition<Parameters>(state, roundId);
      nodes.push(
        // One fused pass per round. A row colors itself as soon as every higher-priority neighbor
        // is colored, reading neighbor colors with atomic loads while other rows write theirs. That
        // is race free for the result: an uncolored row only ever sees colored neighbors that
        // outrank it (a lower-priority neighbor waits for this row), and those colors are final, so
        // the outcome is still sequential greedy in priority order for any scheduling. Rows that
        // see a neighbor colored earlier in the same round finish sooner than in a two-phase
        // round, so fewer rounds are needed.
        createWGSLKernelNode<Parameters>(graph, {
          id: `${roundId}-select`,
          operation: OPERATION,
          variant: 'select-apply',
          bindings: [
            {name: 'offsets', view: weights.offsets, type: 'u32', access: 'read'},
            {name: 'neighbors', view: weights.neighbors, type: 'u32', access: 'read'},
            {name: 'parameters', view: parameters, type: 'u32', access: 'read'},
            {name: 'colors', view: colors, type: 'atomic<u32>', access: 'read_write'},
            {name: 'status', view: state.status, type: 'atomic<u32>', access: 'read_write'},
            {name: 'colorCount', view: colorCount, type: 'atomic<u32>', access: 'read_write'}
          ],
          invocationCount: rows,
          declarations: priorityWGSL,
          body: `if (atomicLoad(&colors[colorsOffset + index]) != UNCOLORED) {
    return;
  }
  let begin = offsets[offsetsOffset + index];
  let end = offsets[offsetsOffset + index + 1u];
  let seed = parameters[parametersOffset];
  let rowDegree = ${degreeFirst ? 'end - begin' : '0u'};
  // One neighbor pass finds both a blocking (uncolored, higher-priority) neighbor and the used
  // colors of the first 32-color window.
  var used = 0u;
  var blocked = false;
  for (var slot = begin; slot < end; slot++) {
    let other = neighbors[neighborsOffset + slot];
    if (other == index) {
      continue;
    }
    let color = atomicLoad(&colors[colorsOffset + other]);
    if (color == UNCOLORED) {
      let otherDegree = ${degreeFirst ? 'offsets[offsetsOffset + other + 1u] - offsets[offsetsOffset + other]' : '0u'};
      if (outranks(other, otherDegree, index, rowDegree, seed)) {
        blocked = true;
        break;
      }
    } else if (color < 32u) {
      used = used | (1u << color);
    }
  }
  if (blocked) {
    atomicStore(&status[statusOffset], 1u);
    return;
  }
  var chosen = UNCOLORED;
  if (used != 0xffffffffu) {
    chosen = firstTrailingBit(~used);
  } else {
    // Rare: 32 or more colors used around this row; scan the following windows.
    var base = 32u;
    for (var window = 1u; window <= (end - begin) / 32u; window++) {
      var windowUsed = 0u;
      for (var slot = begin; slot < end; slot++) {
        let other = neighbors[neighborsOffset + slot];
        let color = atomicLoad(&colors[colorsOffset + other]);
        if (other != index && color != UNCOLORED && color >= base && color < base + 32u) {
          windowUsed = windowUsed | (1u << (color - base));
        }
      }
      if (windowUsed != 0xffffffffu) {
        chosen = base + firstTrailingBit(~windowUsed);
        break;
      }
      base += 32u;
    }
  }
  if (chosen != UNCOLORED) {
    atomicStore(&colors[colorsOffset + index], chosen);
    atomicMax(&colorCount[colorCountOffset], chosen + 1u);
  }`,
          condition: gate.condition,
          extraResources: gate.extraResources
        }),
        createRasterIterationGateNode<Parameters>(graph, {
          id: `${roundId}-gate`,
          operation: OPERATION,
          state,
          maxIterations: maximumIterations
        })
      );
    }
    if (output.status?.converged || output.status?.iterationCount) {
      nodes.push(
        createRasterIterationFinalizeNode<Parameters>(graph, {
          id: `${id}-status`,
          operation: OPERATION,
          state,
          converged: output.status?.converged,
          iterationCount: output.status?.iterationCount
        })
      );
    }
    if (output.conflictCount) {
      nodes.push(
        createFillNode<Parameters>(graph, {
          id: `${id}-conflicts-clear`,
          operation: OPERATION,
          view: output.conflictCount,
          type: 'u32',
          value: '0u',
          componentCount: 1
        }),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-conflicts`,
          operation: OPERATION,
          variant: 'conflicts',
          bindings: [
            {name: 'offsets', view: weights.offsets, type: 'u32', access: 'read'},
            {name: 'neighbors', view: weights.neighbors, type: 'u32', access: 'read'},
            {name: 'colors', view: colors, type: 'u32', access: 'read'},
            {
              name: 'conflictCount',
              view: output.conflictCount,
              type: 'atomic<u32>',
              access: 'read_write'
            }
          ],
          invocationCount: rows,
          body: `let color = colors[colorsOffset + index];
  var conflicts = 0u;
  if (color != ${GPU_MAP_COLORING_UNCOLORED}u) {
    for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
      let other = neighbors[neighborsOffset + slot];
      if (other > index && colors[colorsOffset + other] == color) {
        conflicts++;
      }
    }
  }
  if (conflicts > 0u) {
    atomicAdd(&conflictCount[conflictCountOffset], conflicts);
  }`
        })
      );
    }
    return nodes;
  }
}

/** Returns every caller-owned output view of a map-coloring contributor. */
function getOutputs(output: GPUMapColoringOutput): (GraphDataView | undefined)[] {
  return [
    output.colors,
    output.colorCount,
    output.conflictCount,
    output.status?.converged,
    output.status?.iterationCount
  ];
}
