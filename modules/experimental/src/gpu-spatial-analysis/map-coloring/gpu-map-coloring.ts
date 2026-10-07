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

const OPERATION = 'GPUMapColoring';

/** Color value of a row that was not colored (only when `converged` is 0). */
export const GPU_MAP_COLORING_UNCOLORED = 0xffffffff;

/** Default compile-time cap on coloring rounds. */
export const GPU_MAP_COLORING_DEFAULT_MAXIMUM_ROUNDS = 64;

/**
 * Properties for {@link GPUMapColoring}.
 *
 * Compile-time (needs a new graph): view lengths, `seed`, `maximumRounds` and which optional
 * outputs are present. Per-frame: the contents of `weights`.
 */
export type GPUMapColoringProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'map-coloring'`. */
  id?: string;
  /**
   * Square self-join adjacency, usually `GPUContiguityWeights` output (rook for maps, queen to also
   * separate polygons that touch at a point). Only the pattern is used, so weights of any value
   * work. The pattern must be symmetric (`j` in row `i` implies `i` in row `j`); an asymmetric
   * pattern can yield a color conflict, which `conflictCount` reports.
   */
  weights: GPUSpatialWeights;
  /**
   * Caller-owned output: color index per row, from 0. A row that did not finish within
   * `maximumRounds` holds {@link GPU_MAP_COLORING_UNCOLORED}.
   */
  colors: GraphDataView<'uint32'>;
  /** Optional one-row output: number of colors used (largest color plus one). */
  colorCount?: GraphDataView<'uint32'>;
  /**
   * Optional one-row output: number of neighbor pairs `(i, j)` with `j > i` whose rows share a
   * color. 0 means the coloring is proper (for a symmetric pattern). Uncolored rows never conflict.
   */
  conflictCount?: GraphDataView<'uint32'>;
  /** Optional one-row output: 1 when every row was colored within `maximumRounds`, else 0. */
  converged?: GraphDataView<'uint32'>;
  /** Optional one-row output: rounds that ran. */
  roundCount?: GraphDataView<'uint32'>;
  /** Seed of the hashed priorities. Default 0. Different seeds give different valid colorings. */
  seed?: number;
  /**
   * Compile-time cap on rounds, in `[1, 1024]`. Default 64. Each round adds two nodes, and
   * rounds after convergence are skipped on the GPU. Hashed priorities need about `O(log n)` rounds
   * on map graphs; the longest decreasing-priority chain bounds the count.
   */
  maximumRounds?: number;
};

/**
 * Greedy parallel graph coloring of polygon contiguity (Jones-Plassmann), so adjacent polygons
 * never share a color: the classic "map coloring" for choropleth-free categorical maps.
 *
 * **Algorithm.** Every row gets a priority from a seeded integer hash of its ID; ties between equal
 * hashes go to the lowest ID. In each round every uncolored row whose priority is higher than
 * every uncolored neighbor takes the lowest color not used by its colored neighbors (first free
 * slot of 32-color bit windows). Such rows are never adjacent. Each round is one fused pass that
 * reads and writes colors with atomics (see the kernel comment); the result equals sequential greedy coloring in priority order and is deterministic for a given
 * `seed`.
 *
 * **Quality.** The color count is not minimal: greedy coloring uses at most `maxDegree + 1`
 * colors, and the four-color theorem's bound is not targeted. Expect about 4 to 6 colors on typical
 * planar subdivisions. Try several seeds (graphs per seed) and keep the one with the fewest colors
 * if that matters.
 *
 * **Verification.** `conflictCount` counts neighbor pairs that share a color; it is 0 for a symmetric
 * pattern. `converged` reports whether every row finished within `maximumRounds`.
 */
export class GPUMapColoring implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUMapColoringProps;
  /** Compile-time round cap. */
  readonly maximumRounds: number;

  constructor(props: GPUMapColoringProps) {
    const id = props.id ?? 'map-coloring';
    this.id = id;
    this.props = props;
    this.maximumRounds = props.maximumRounds ?? GPU_MAP_COLORING_DEFAULT_MAXIMUM_ROUNDS;
    validateRasterIterations(id, 'maximumRounds', this.maximumRounds);
    if (
      props.seed !== undefined &&
      (!Number.isInteger(props.seed) || props.seed < 0 || props.seed > 0xffffffff)
    ) {
      throw new Error(`${id} seed must be an integer in [0, 2^32 - 1]`);
    }
    const rows = validateGPUSpatialWeights(id, props.weights);
    validatePackedUint32View(props.colors, `${id} colors`);
    if (props.colors.length !== rows) {
      throw new Error(`${id} colors length must equal the weights row count`);
    }
    for (const [name, view] of [
      ['colorCount', props.colorCount],
      ['conflictCount', props.conflictCount],
      ['converged', props.converged],
      ['roundCount', props.roundCount]
    ] as const) {
      if (view) {
        validatePackedUint32View(view, `${id} ${name}`);
        if (view.length < 1) {
          throw new Error(`${id} ${name} must hold one uint32`);
        }
      }
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [props.colors, props.colorCount, props.conflictCount, props.converged, props.roundCount],
      [
        props.weights.offsets,
        props.weights.neighbors,
        props.weights.weights,
        props.weights.distances
      ]
    );
  }

  /** Returns the clear, gated round, and verification nodes in dependency order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, maximumRounds} = this;
    const {weights, colors} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      weights.offsets,
      weights.neighbors,
      weights.weights,
      weights.distances,
      colors,
      props.colorCount,
      props.conflictCount,
      props.converged,
      props.roundCount
    ]);
    const rows = colors.length;
    const seed = props.seed ?? 0;
    const state = createRasterIterationState(graph, `${id}-rounds`, OPERATION, rows, id);
    const colorCount =
      props.colorCount ?? createTransientView(graph, `${id}-color-count`, 'uint32', 1);
    const priorityWGSL = `const SEED: u32 = ${seed}u;
const UNCOLORED: u32 = ${GPU_MAP_COLORING_UNCOLORED}u;

fn priorityHash(value: u32) -> u32 {
  var h = value ^ SEED;
  h = h ^ (h >> 16u);
  h = h * 0x7feb352du;
  h = h ^ (h >> 15u);
  h = h * 0x846ca68bu;
  h = h ^ (h >> 16u);
  return h;
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
    for (let round = 0; round < maximumRounds; round++) {
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
  let rowHash = priorityHash(index);
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
      let otherHash = priorityHash(other);
      if (otherHash > rowHash || (otherHash == rowHash && other < index)) {
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
          maxIterations: maximumRounds
        })
      );
    }
    if (props.converged || props.roundCount) {
      nodes.push(
        createRasterIterationFinalizeNode<Parameters>(graph, {
          id: `${id}-status`,
          operation: OPERATION,
          state,
          converged: props.converged,
          iterationCount: props.roundCount
        })
      );
    }
    if (props.conflictCount) {
      nodes.push(
        createFillNode<Parameters>(graph, {
          id: `${id}-conflicts-clear`,
          operation: OPERATION,
          view: props.conflictCount,
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
              view: props.conflictCount,
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
