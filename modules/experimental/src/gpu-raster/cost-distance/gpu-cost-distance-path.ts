// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode, createPublishNode} from '../../utils/wgsl-kernel-nodes';
import type {GPUCompactOutput} from '../../utils/gpu-contributor-types';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {
  validateGraphViewsBelongToGraph,
  validatePackedView
} from '../../utils/gpu-contributor-utils';
import {validateRasterBuffersDistinct, validateRasterGrid} from '../raster-band-utils';
import type {GPURasterMetadata} from '../types';
import {createContributorTransientView} from './raster-grid-utils';
import {GPU_COST_DISTANCE_NONE, GPU_COST_DISTANCE_UNRESOLVED} from './gpu-cost-distance';

const OPERATION = 'GPUCostDistancePath';

/** One-row status words of {@link GPUCostDistancePath}. */
export type GPUCostDistancePathStatus = {
  /**
   * One row: 1 when the walk met a `GPU_COST_DISTANCE_UNRESOLVED` back-link, that is,
   * {@link GPUCostDistance} ran out of `maximumIterations` or `maximumTieIterations` before this
   * path was certified; the path is then empty. Otherwise 0.
   */
  iterationLimitReached?: GraphDataView<'uint32'>;
};

/** Caller-owned outputs of {@link GPUCostDistancePath}. */
export type GPUCostDistancePathOutput = {
  /**
   * Bounded compact output. `ids` holds cell indices from the target (first) to the source
   * cell (last); capacity is `ids.length` (compile-time). `count` is clamped, `overflow` is 1 when
   * the path is longer than the capacity, `requiredCount` is the exact path length. The path is empty
   * (count 0) when the target is out of range or unreached, when the walk meets a
   * `GPU_COST_DISTANCE_UNRESOLVED` link (see `status.iterationLimitReached`), and when the links are
   * not a valid {@link GPUCostDistance} forest (an invalid code, a link off the grid or into an
   * unreached cell, or a cycle). A path is only ever reported complete, from the target to a
   * source.
   */
  path: GPUCompactOutput;
  /** Optional one-row status words. */
  status?: GPUCostDistancePathStatus;
};

/**
 * Properties for {@link GPUCostDistancePath}.
 *
 * Compile-time: `grid` size, the capacity of `output.path.ids`, and the presence of
 * `output.status.iterationLimitReached`. Per-frame: the contents of `backLinks` and `target`.
 */
export type GPUCostDistancePathProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'cost-distance-path'`. */
  id?: string;
  /** Raster grid: its width and height size the kernels. */
  grid: GPURasterMetadata;
  /** Back-link raster from {@link GPUCostDistance}, one value per cell. Per-frame contents. */
  backLinks: GraphDataView<'uint32'>;
  /** One-row target cell index (`row * width + column`). Per-frame contents. */
  target: GraphDataView<'uint32'>;
  /** Caller-owned outputs. */
  output: GPUCostDistancePathOutput;
};

/**
 * Extracts the least-cost path from a target cell back to its source by following the back-links
 * written by {@link GPUCostDistance}.
 *
 * One GPU thread walks the path, bounded by the cell count, so the exact length is always
 * reported. A walk that cannot reach a source (an unresolved link from a truncated
 * `GPUCostDistance` run, or links that are not a valid forest) reports an empty path instead of
 * a truncated one. Non-goals: multiple targets per encoding, path simplification, and parallel pointer
 * jumping; a very long path on a large grid is walked serially.
 */
export class GPUCostDistancePath implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUCostDistancePathProps;

  constructor(props: GPUCostDistancePathProps) {
    this.id = props.id ?? 'cost-distance-path';
    this.props = props;
    const {id} = this;
    const cellCount = validateRasterGrid(id, props.grid);
    const {output} = props;
    const {path} = output;
    const iterationLimitReached = output.status?.iterationLimitReached;
    for (const [name, view] of [
      ['backLinks', props.backLinks],
      ['target', props.target],
      ['output.path.ids', path.ids],
      ['output.path.count', path.status.count],
      ['output.path.overflow', path.status.overflow],
      ['output.path.requiredCount', path.status.requiredCount],
      ['output.status.iterationLimitReached', iterationLimitReached]
    ] as const) {
      if (view) {
        validatePackedView(view, ['uint32'], `${id} ${name}`);
      }
    }
    if (props.backLinks.length !== cellCount) {
      throw new Error(`${id} backLinks must contain one value per cell`);
    }
    for (const [name, view] of [
      ['target', props.target],
      ['output.path.count', path.status.count],
      ['output.path.overflow', path.status.overflow],
      ['output.path.requiredCount', path.status.requiredCount],
      ['output.status.iterationLimitReached', iterationLimitReached]
    ] as const) {
      if (view && view.length !== 1) {
        throw new Error(`${id} ${name} must contain exactly one row`);
      }
    }
    if (path.ids.length < 1) {
      throw new Error(`${id} output.path.ids must have a capacity of at least one`);
    }
    validateRasterBuffersDistinct(
      id,
      [
        path.ids,
        path.status.count,
        path.status.overflow,
        path.status.requiredCount,
        iterationLimitReached
      ],
      [props.backLinks, props.target]
    );
  }

  /** Returns the path-walk node and the publish node. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {output} = props;
    const {path} = output;
    const {width, height} = props.grid;
    const iterationLimitReached = output.status?.iterationLimitReached;
    validateGraphViewsBelongToGraph(id, graph, [
      props.backLinks,
      props.target,
      path.ids,
      path.status.count,
      path.status.overflow,
      path.status.requiredCount,
      iterationLimitReached
    ]);
    const cellCount = width * height;
    const total = createContributorTransientView(graph, OPERATION, id, `${id}-total`, 'uint32', 1);
    return [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-walk`,
        operation: OPERATION,
        variant: 'walk',
        bindings: [
          {name: 'backLinks', view: props.backLinks, type: 'u32', access: 'read'},
          {name: 'targetCells', view: props.target, type: 'u32', access: 'read'},
          {name: 'ids', view: path.ids, type: 'u32', access: 'read_write'},
          {name: 'total', view: total, type: 'u32', access: 'read_write'},
          ...(iterationLimitReached
            ? [
                {
                  name: 'iterationLimitReached',
                  view: iterationLimitReached,
                  type: 'u32' as const,
                  access: 'read_write' as const
                }
              ]
            : [])
        ],
        invocationCount: 1,
        declarations: `const GRID_WIDTH: u32 = ${width}u;
const GRID_HEIGHT: u32 = ${height}u;
const CELL_COUNT: u32 = ${cellCount}u;
const CAPACITY: u32 = ${path.ids.length}u;
const NONE: u32 = ${GPU_COST_DISTANCE_NONE}u;
const UNRESOLVED: u32 = ${GPU_COST_DISTANCE_UNRESOLVED}u;

fn getColumnOffset(direction: u32) -> i32 {
  var offsets = array<i32, 8>(1, 1, 0, -1, -1, -1, 0, 1);
  return offsets[direction];
}
fn getRowOffset(direction: u32) -> i32 {
  var offsets = array<i32, 8>(0, 1, 1, 1, 0, -1, -1, -1);
  return offsets[direction];
}
fn getNeighbor(cell: u32, direction: u32) -> u32 {
  let column = i32(cell % GRID_WIDTH) + getColumnOffset(direction);
  let row = i32(cell / GRID_WIDTH) + getRowOffset(direction);
  if (column < 0 || row < 0 || column >= i32(GRID_WIDTH) || row >= i32(GRID_HEIGHT)) { return NONE; }
  return u32(row) * GRID_WIDTH + u32(column);
}`,
        body: `let targetCell = targetCells[targetCellsOffset];
  var pathLength = 0u;
  var unresolved = false;
  if (targetCell < CELL_COUNT && backLinks[backLinksOffset + targetCell] != NONE) {
    var cell = targetCell;
    // Only a walk that ends at a source (code 0) is a path; every other exit reports none.
    var complete = false;
    for (var walkIndex = 0u; walkIndex < CELL_COUNT; walkIndex++) {
      let code = backLinks[backLinksOffset + cell];
      if (code == UNRESOLVED) { unresolved = true; break; }
      if (pathLength < CAPACITY) { ids[idsOffset + pathLength] = cell; }
      pathLength++;
      if (code == 0u) { complete = true; break; }
      if (code > 128u || countOneBits(code) != 1u) { break; }
      let next = getNeighbor(cell, firstTrailingBit(code));
      if (next == NONE || backLinks[backLinksOffset + next] == NONE) { break; }
      cell = next;
    }
    if (!complete) { pathLength = 0u; }
  }
  total[totalOffset] = pathLength;
  ${iterationLimitReached ? 'iterationLimitReached[iterationLimitReachedOffset] = select(0u, 1u, unresolved);' : ''}`
      }),
      createPublishNode<Parameters>(graph, {
        id: `${id}-publish`,
        operation: OPERATION,
        requiredCount: total,
        output: path
      })
    ];
  }
}
