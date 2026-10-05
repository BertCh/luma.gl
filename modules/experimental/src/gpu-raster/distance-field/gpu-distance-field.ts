// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView,
  type GraphTextureView
} from '@luma.gl/gpgpu/gpu-core';
import {GPURasterBufferToTexture} from '../index';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {
  captureGraphCommandNodes,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {
  createDistanceFieldColumnNode,
  createDistanceFieldFinalizeNode,
  createDistanceFieldJumpFloodNodes,
  createDistanceFieldRowNode,
  createDistanceFieldSeedNodes
} from './distance-field-kernels';
import {
  GPU_DISTANCE_FIELD_MAXIMUM_DIMENSION,
  GPU_DISTANCE_FIELD_PARAMETER_LENGTH
} from './distance-field-parameters';

/**
 * Distance-field algorithm.
 *
 * - `'exact'`: separable Felzenszwalb-Huttenlocher lower envelopes, exact nearest seed per cell.
 * - `'jump-flood'`: jump flooding, an approximate preview in `ceil(log2(max(width, height)))`
 *   full-grid passes plus refinement passes.
 */
export type GPUDistanceFieldMode = 'exact' | 'jump-flood';

/** Caller-owned results, one row per cell in row-major order. Every view uses its own buffer. */
export type GPUDistanceFieldOutput = {
  /**
   * Ground distance from each cell center to the center of its nearest seed cell, `+Infinity`
   * when there is no seed or the nearest seed is beyond `maxDistance`.
   */
  distances: GraphDataView<'float32'>;
  /** Optional ID of the nearest seed (Euclidean allocation / Voronoi zones); ties take the smallest ID. */
  allocation?: GraphDataView<'uint32'>;
  /** Optional row-major cell index of the nearest seed, or `GPU_DISTANCE_FIELD_NONE`. */
  nearestCells?: GraphDataView<'uint32'>;
  /** Optional mask: `1` when a seed lies within `maxDistance`, otherwise `0`. */
  withinDistance?: GraphDataView<'uint32'>;
  /** Optional `r32float` texture copy of `distances` with `width x height` texels. */
  texture?: GraphTextureView<'r32float'>;
};

/**
 * Properties for {@link GPUDistanceField}.
 *
 * Compile-time: `width`, `height`, `mode`, `jumpFloodRefinementPasses`, the seed capacity
 * (`seedPositions.length`), and which optional views exist. Per-frame: the contents of
 * `settings` (origin, cell size, `maxDistance`), `seedPositions`, `seedIds`, `seedCount`, and
 * `seedMask`.
 */
export type GPUDistanceFieldProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'distance-field'`. */
  id?: string;
  /** Grid width in cells, 1 to 32768. Compile-time. */
  width: number;
  /** Grid height in cells, 1 to 32768. Compile-time. Row 0 is the minimum-y edge. */
  height: number;
  /** Per-frame settings with at least 8 float32 values, see `getGPUDistanceFieldParameterValues`. */
  settings: GraphDataView<'float32'>;
  /**
   * Optional seed points in ground coordinates. Each point snaps to the cell that contains it;
   * points outside the grid or with non-finite coordinates are ignored. The length is the
   * compile-time seed capacity; contents are per-frame.
   */
  seedPositions?: GraphDataView<'float32x2'>;
  /** Optional seed ID per point row; defaults to the row index. `0xffffffff` disables a row. */
  seedIds?: GraphDataView<'uint32'>;
  /** Optional one-row active point count; rows at or after it are ignored. Per-frame contents. */
  seedCount?: GraphDataView<'uint32'>;
  /**
   * Optional per-cell seed raster. A nonzero value `v` marks a seed with ID `v - 1`: write `1`
   * for a plain mask (every mask seed has ID 0) or `label + 1` for per-zone allocation.
   */
  seedMask?: GraphDataView<'uint32'>;
  /** Algorithm. Compile-time. Defaults to `'exact'`. */
  mode?: GPUDistanceFieldMode;
  /** Extra jump-flood passes after the step-1 pass: `0`, `1` (JFA+1, default), or `2` (JFA+2). */
  jumpFloodRefinementPasses?: 0 | 1 | 2;
  /** Caller-owned results. */
  output: GPUDistanceFieldOutput;
};

/**
 * Computes a Euclidean distance transform and nearest-seed allocation (Voronoi zones) on a
 * raster grid, from seed points snapped to cells and/or a seed raster.
 *
 * The exact mode is the separable Felzenszwalb-Huttenlocher transform: one invocation per column
 * finds the nearest seed row, then one invocation per row builds the discrete lower envelope of
 * those candidates. Candidates are ordered by squared ground distance (exact `u32` integer keys
 * with equal cell sizes, f32 keys otherwise), then by the smaller seed ID, so allocation is
 * deterministic and ties resolve to the smallest ID. The jump-flood mode is a cheaper
 * approximate preview with the same outputs. Every encoding recomputes from scratch, and
 * rewriting seeds or settings never recompiles the graph.
 *
 * Distances are between cell centers; a seed point's position inside its cell is not used.
 * Non-goals: cost-weighted (non-Euclidean) allocation, which `GPUCostDistance` covers, geodesic
 * distances on longitude/latitude grids, and polygon or line seeds (rasterize them into
 * `seedMask`).
 */
export class GPUDistanceField implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUDistanceFieldProps;
  /** Resolved algorithm. */
  readonly mode: GPUDistanceFieldMode;
  /** Resolved jump-flood refinement pass count. */
  readonly jumpFloodRefinementPasses: 0 | 1 | 2;
  /** `width * height`. */
  readonly cellCount: number;

  constructor(props: GPUDistanceFieldProps) {
    this.id = props.id ?? 'distance-field';
    this.props = props;
    this.mode = props.mode ?? 'exact';
    this.jumpFloodRefinementPasses = props.jumpFloodRefinementPasses ?? 1;
    const {id} = this;
    const {width, height, output} = props;
    for (const [name, size] of [
      ['width', width],
      ['height', height]
    ] as const) {
      if (!Number.isSafeInteger(size) || size < 1 || size > GPU_DISTANCE_FIELD_MAXIMUM_DIMENSION) {
        throw new Error(
          `${id} ${name} must be an integer from 1 to ${GPU_DISTANCE_FIELD_MAXIMUM_DIMENSION}`
        );
      }
    }
    this.cellCount = width * height;
    if (!['exact', 'jump-flood'].includes(this.mode)) {
      throw new Error(`${id} mode must be exact or jump-flood`);
    }
    if (![0, 1, 2].includes(this.jumpFloodRefinementPasses)) {
      throw new Error(`${id} jumpFloodRefinementPasses must be 0, 1, or 2`);
    }
    validatePackedView(props.settings, ['float32'], `${id} settings`);
    if (props.settings.length < GPU_DISTANCE_FIELD_PARAMETER_LENGTH) {
      throw new Error(
        `${id} settings must contain ${GPU_DISTANCE_FIELD_PARAMETER_LENGTH} float32 values`
      );
    }
    if (!props.seedPositions && !props.seedMask) {
      throw new Error(`${id} requires seedPositions or seedMask`);
    }
    if (props.seedPositions) {
      validatePackedView(props.seedPositions, ['float32x2'], `${id} seedPositions`);
    }
    if ((props.seedIds || props.seedCount) && !props.seedPositions) {
      throw new Error(`${id} seedIds and seedCount require seedPositions`);
    }
    for (const [name, view] of [
      ['seedIds', props.seedIds],
      ['seedCount', props.seedCount],
      ['seedMask', props.seedMask],
      ['output.allocation', output.allocation],
      ['output.nearestCells', output.nearestCells],
      ['output.withinDistance', output.withinDistance]
    ] as const) {
      if (view) {
        validatePackedUint32View(view, `${id} ${name}`);
      }
    }
    validatePackedView(output.distances, ['float32'], `${id} output.distances`);
    if (props.seedIds && props.seedIds.length !== props.seedPositions?.length) {
      throw new Error(`${id} seedIds length must equal seedPositions length`);
    }
    if (props.seedCount && props.seedCount.length !== 1) {
      throw new Error(`${id} seedCount must contain exactly one row`);
    }
    for (const [name, view] of [
      ['seedMask', props.seedMask],
      ['output.distances', output.distances],
      ['output.allocation', output.allocation],
      ['output.nearestCells', output.nearestCells],
      ['output.withinDistance', output.withinDistance]
    ] as const) {
      if (view && view.length !== this.cellCount) {
        throw new Error(`${id} ${name} must contain one row per cell`);
      }
    }
    if (
      output.texture &&
      (output.texture.format !== 'r32float' ||
        output.texture.width !== width ||
        output.texture.height !== height)
    ) {
      throw new Error(`${id} output.texture must be r32float with a width x height extent`);
    }
    const outputs = [
      output.distances,
      output.allocation,
      output.nearestCells,
      output.withinDistance
    ];
    validateGraphOutputsDisjointFromInputs(id, outputs, [
      props.settings,
      props.seedPositions,
      props.seedIds,
      props.seedCount,
      props.seedMask
    ]);
    const outputBuffers = outputs.filter(view => view !== undefined).map(view => view.buffer);
    if (new Set(outputBuffers).size !== outputBuffers.length) {
      throw new Error(`${id} outputs must use separate buffers`);
    }
  }

  /** Returns seed rasterization, exact or jump-flood nearest-seed passes, and finalize nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, cellCount} = this;
    const {output, settings} = props;
    const grid = {width: props.width, height: props.height};
    validateGraphViewsBelongToGraph(id, graph, [
      settings,
      props.seedPositions,
      props.seedIds,
      props.seedCount,
      props.seedMask,
      output.distances,
      output.allocation,
      output.nearestCells,
      output.withinDistance
    ]);
    if (output.texture && output.texture.texture.graph !== graph) {
      throw new Error(`${id} views must belong to the target graph`);
    }
    const createScratch = (name: string) =>
      createTransientView(graph, `${id}-${name}`, 'uint32', cellCount);
    const cellSeeds = createScratch('cell-seeds');
    const nodes: GPUCommandNode<Parameters>[] = createDistanceFieldSeedNodes<Parameters>(graph, {
      id,
      grid,
      settings,
      cellSeeds,
      seedMask: props.seedMask,
      seedPositions: props.seedPositions,
      seedIds: props.seedIds,
      seedCount: props.seedCount
    });
    let nearest: {
      cells: GraphDataView<'uint32'>;
      ids: GraphDataView<'uint32'>;
    };
    if (this.mode === 'exact') {
      const columnRows = createScratch('column-rows');
      const columnIds = createScratch('column-ids');
      nearest = {
        cells: createScratch('nearest-cells'),
        ids: createScratch('nearest-ids')
      };
      nodes.push(
        createDistanceFieldColumnNode<Parameters>(graph, {
          id,
          grid,
          cellSeeds,
          columnRows,
          columnIds
        }),
        createDistanceFieldRowNode<Parameters>(graph, {
          id,
          grid,
          settings,
          columnRows,
          columnIds,
          stackColumns: createScratch('stack-columns'),
          stackStarts: createScratch('stack-starts'),
          nearestCells: nearest.cells,
          nearestIds: nearest.ids
        })
      );
    } else {
      const jumpFlood = createDistanceFieldJumpFloodNodes<Parameters>(graph, {
        id,
        grid,
        settings,
        cellSeeds,
        buffers: [
          {
            cells: createScratch('jump-flood-cells-0'),
            ids: createScratch('jump-flood-ids-0')
          },
          {
            cells: createScratch('jump-flood-cells-1'),
            ids: createScratch('jump-flood-ids-1')
          }
        ],
        refinementPasses: this.jumpFloodRefinementPasses
      });
      nodes.push(...jumpFlood.nodes);
      nearest = jumpFlood.result;
    }
    nodes.push(
      createDistanceFieldFinalizeNode<Parameters>(graph, {
        id,
        grid,
        settings,
        nearestCells: nearest.cells,
        nearestIds: nearest.ids,
        distances: output.distances,
        allocation: output.allocation,
        nearestCellsOutput: output.nearestCells,
        withinDistance: output.withinDistance
      })
    );
    if (output.texture) {
      const texture = output.texture;
      nodes.push(
        ...captureGraphCommandNodes(graph, () =>
          new GPURasterBufferToTexture({
            id: `${id}-texture`,
            input: {
              id: `${id}-texture-band`,
              format: 'float32',
              storage: {kind: 'buffer', values: output.distances}
            },
            output: texture
          }).addToGraph(graph)
        )
      );
    }
    return nodes;
  }
}
