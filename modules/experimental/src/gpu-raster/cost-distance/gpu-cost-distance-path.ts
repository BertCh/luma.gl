// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validatePackedUint32View,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode, createPublishNode} from '../../utils/wgsl-kernel-nodes';
import type {GPUCompactOutput} from '../../utils/gpu-contributor-types';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {validateGraphViewsBelongToGraph} from '../../utils/gpu-contributor-utils';
import {
  validateTerrainBuffersDistinct,
  validateTerrainGrid
} from '../../gpu-terrain/terrain-analysis/terrain-analysis-utils';
import {createContributorTransientView} from './raster-grid-utils';
import {GPU_COST_DISTANCE_NONE} from './gpu-cost-distance';

const OPERATION = 'GPUCostDistancePath';

/**
 * Properties for {@link GPUCostDistancePath}.
 *
 * Compile-time: `width`, `height`, and the capacity of `output.ids`. Per-frame: the contents of
 * `backLinks` and `target`.
 */
export type GPUCostDistancePathProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'cost-distance-path'`. */
  id?: string;
  /** Grid width in cells. Compile-time. */
  width: number;
  /** Grid height in cells. Compile-time. */
  height: number;
  /** Back-link raster from {@link GPUCostDistance}, one value per cell. Per-frame contents. */
  backLinks: GraphDataView<'uint32'>;
  /** One-row target cell index (`row * width + column`). Per-frame contents. */
  target: GraphDataView<'uint32'>;
  /**
   * Caller-owned bounded output. `ids` holds cell indices from the target (first) to the source
   * cell (last); capacity is `ids.length` (compile-time). `count` is clamped, `overflow` is 1 when
   * the path is longer than the capacity, `requiredCount` is the exact path length. The path is empty
   * (count 0) when the target is out of range or unreached.
   */
  output: GPUCompactOutput;
};

/**
 * Extracts the least-cost path from a target cell back to its source by following the back-links
 * written by {@link GPUCostDistance}.
 *
 * One GPU thread walks the path, bounded by the cell count, so the exact length is always
 * reported. Non-goals: multiple targets per encoding, path simplification, and parallel pointer
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
    const cellCount = validateTerrainGrid(id, props.width, props.height);
    const {output} = props;
    for (const [name, view] of [
      ['backLinks', props.backLinks],
      ['target', props.target],
      ['output ids', output.ids],
      ['output count', output.count],
      ['output overflow', output.overflow],
      ['output requiredCount', output.requiredCount]
    ] as const) {
      if (view) {
        validatePackedUint32View(view, `${id} ${name}`);
      }
    }
    if (props.backLinks.length !== cellCount) {
      throw new Error(`${id} backLinks must contain one value per cell`);
    }
    for (const [name, view] of [
      ['target', props.target],
      ['output count', output.count],
      ['output overflow', output.overflow],
      ['output requiredCount', output.requiredCount]
    ] as const) {
      if (view && view.length !== 1) {
        throw new Error(`${id} ${name} must contain exactly one row`);
      }
    }
    if (output.ids.length < 1) {
      throw new Error(`${id} output ids must have a capacity of at least one`);
    }
    validateTerrainBuffersDistinct(
      id,
      [output.ids, output.count, output.overflow, output.requiredCount],
      [props.backLinks, props.target]
    );
  }

  /** Returns the path-walk node and the publish node. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {output} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      props.backLinks,
      props.target,
      output.ids,
      output.count,
      output.overflow,
      output.requiredCount
    ]);
    const cellCount = props.width * props.height;
    const total = createContributorTransientView(graph, OPERATION, id, `${id}-total`, 'uint32', 1);
    return [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-walk`,
        operation: OPERATION,
        variant: 'walk',
        bindings: [
          {name: 'backLinks', view: props.backLinks, type: 'u32', access: 'read'},
          {name: 'targetCells', view: props.target, type: 'u32', access: 'read'},
          {name: 'ids', view: output.ids, type: 'u32', access: 'read_write'},
          {name: 'total', view: total, type: 'u32', access: 'read_write'}
        ],
        invocationCount: 1,
        declarations: `const GRID_WIDTH: u32 = ${props.width}u;
const GRID_HEIGHT: u32 = ${props.height}u;
const CELL_COUNT: u32 = ${cellCount}u;
const CAPACITY: u32 = ${output.ids.length}u;
const NONE: u32 = ${GPU_COST_DISTANCE_NONE}u;

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
  if (targetCell < CELL_COUNT && backLinks[backLinksOffset + targetCell] != NONE) {
    var cell = targetCell;
    for (var walkIndex = 0u; walkIndex < CELL_COUNT; walkIndex++) {
      if (pathLength < CAPACITY) { ids[idsOffset + pathLength] = cell; }
      pathLength++;
      let code = backLinks[backLinksOffset + cell];
      if (code == 0u || code > 128u || countOneBits(code) != 1u) { break; }
      let next = getNeighbor(cell, firstTrailingBit(code));
      if (next == NONE || backLinks[backLinksOffset + next] == NONE) { break; }
      cell = next;
    }
  }
  total[totalOffset] = pathLength;`
      }),
      createPublishNode<Parameters>(graph, {
        id: `${id}-publish`,
        operation: OPERATION,
        requiredCount: total,
        output
      })
    ];
  }
}
