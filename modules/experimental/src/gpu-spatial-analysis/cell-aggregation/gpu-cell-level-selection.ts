// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validatePackedUint32View,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCardinalityPort} from '../../analysis-core/contracts/index';
import {createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import {validateGraphViewsBelongToGraph} from '../../utils/gpu-contributor-utils';

/** Caller-owned outputs of {@link GPUCellLevelSelection}. */
export type GPUCellLevelSelectionOutput = {
  /** One-row status scalars; `count` receives the active level's occupied row count. */
  status: GPUCardinalityPort;
  /** Optional one-row scalar receiving the active level's first row (`levelFirstRows[level]`). */
  firstRow?: GraphDataView<'uint32'>;
  /**
   * Optional indirect draw arguments `[vertexCount, instanceCount, firstVertex, firstInstance]`.
   * Only `instanceCount` and `firstInstance` are written.
   */
  drawArguments?: GraphDataView<'uint32'>;
};

/** Properties for {@link GPUCellLevelSelection}. */
export type GPUCellLevelSelectionProps = {
  /** Prefix for generated node IDs. Defaults to `'cell-level-selection'`. */
  id?: string;
  /** Per-level clamped counts, for example `GPUCellPyramidProps.levelCounts`. */
  levelCounts: GraphDataView<'uint32'>;
  /**
   * Per-frame active level index (one `uint32`), typically chosen from the zoom. Values past the
   * last level select the last (coarsest) level.
   */
  activeLevel: GraphDataView<'uint32'>;
  /** Row offset of each level's cells in a shared buffer. Defaults to zeros. */
  levelFirstRows?: readonly number[];
  output: GPUCellLevelSelectionOutput;
};

/**
 * Publishes the active pyramid level's count, first row and indirect draw arguments from a
 * per-frame level index, so a layer switches levels by rewriting one `uint32` without rebuilding
 * or re-running the pyramid. Put it in a small per-frame graph next to the draw.
 */
export class GPUCellLevelSelection implements GPUCommandNodeProducer {
  /** Prefix for every node ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUCellLevelSelectionProps;

  constructor(props: GPUCellLevelSelectionProps) {
    this.id = props.id ?? 'cell-level-selection';
    this.props = props;
    const id = this.id;
    validatePackedUint32View(props.levelCounts, `${id} levelCounts`);
    validatePackedUint32View(props.activeLevel, `${id} activeLevel`);
    validatePackedUint32View(props.output.status.count, `${id} output.status.count`);
    if (props.levelCounts.length < 1 || props.activeLevel.length < 1) {
      throw new Error(`${id} levelCounts and activeLevel must hold at least one row`);
    }
    const levelFirstRows = props.levelFirstRows ?? [];
    if (
      props.levelFirstRows &&
      (levelFirstRows.length !== props.levelCounts.length ||
        levelFirstRows.some(row => !Number.isInteger(row) || row < 0 || row > 0xffffffff))
    ) {
      throw new Error(`${id} levelFirstRows must hold one uint32 per level`);
    }
    if (props.output.firstRow) {
      validatePackedUint32View(props.output.firstRow, `${id} output.firstRow`);
    }
    if (props.output.drawArguments) {
      validatePackedUint32View(props.output.drawArguments, `${id} output.drawArguments`);
      if (props.output.drawArguments.length < 4) {
        throw new Error(`${id} output.drawArguments must hold 4 uint32 values`);
      }
    }
  }

  /** Returns one single-invocation node. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {output} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      props.levelCounts,
      props.activeLevel,
      output.status.count,
      output.firstRow,
      output.drawArguments
    ]);
    const levelCount = props.levelCounts.length;
    const firstRows = props.levelFirstRows ?? new Array<number>(levelCount).fill(0);
    return [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-select`,
        operation: 'GPUCellLevelSelection',
        bindings: [
          {
            name: 'levelCounts',
            view: props.levelCounts,
            type: 'u32',
            access: 'read'
          },
          {
            name: 'activeLevel',
            view: props.activeLevel,
            type: 'u32',
            access: 'read'
          },
          {
            name: 'countOut',
            view: output.status.count,
            type: 'u32',
            access: 'read_write'
          },
          ...(output.firstRow
            ? [
                {
                  name: 'firstRowOut',
                  view: output.firstRow,
                  type: 'u32',
                  access: 'read_write'
                } as const
              ]
            : []),
          ...(output.drawArguments
            ? [
                {
                  name: 'drawArguments',
                  view: output.drawArguments,
                  type: 'u32',
                  access: 'read_write'
                } as const
              ]
            : [])
        ],
        invocationCount: 1,
        declarations: `const LEVEL_FIRST_ROWS = array<u32, ${levelCount}>(${firstRows.map(row => `${row}u`).join(', ')});`,
        body: `let level = min(activeLevel[activeLevelOffset], ${levelCount - 1}u);
  let count = levelCounts[levelCountsOffset + level];
  var firstRows = LEVEL_FIRST_ROWS;
  let firstRow = firstRows[level];
  countOut[countOutOffset] = count;
  ${output.firstRow ? 'firstRowOut[firstRowOutOffset] = firstRow;' : ''}
  ${
    output.drawArguments
      ? `drawArguments[drawArgumentsOffset + 1u] = count;
  drawArguments[drawArgumentsOffset + 3u] = firstRow;`
      : ''
  }`
      })
    ];
  }
}
