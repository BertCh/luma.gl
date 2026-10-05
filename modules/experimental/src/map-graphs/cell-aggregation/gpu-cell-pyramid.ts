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
import type {GPUMapGraphRecipe} from '../map-graph-types';
import {createMapGraphKernelNode} from '../map-graph-kernels';
import {validateGraphViewsBelongToGraph} from '../map-graph-utils';
import {getCellKeyLayout} from './cell-keys';
import {getCellTableFirstRow, validateCellTable, type GPUCellTable} from './cell-table';
import {GPUCellAggregation, type GPUCellAggregationProps} from './gpu-cell-aggregation';
import {GPUCellRollup} from './gpu-cell-rollup';

/** One level of a {@link GPUCellPyramid}. */
export type GPUCellPyramidLevel = {
  /** Resolution of this level. Levels are listed finest first with strictly decreasing resolutions. */
  resolution: number;
  /** Caller-owned table of this level. */
  output: GPUCellTable;
};

/**
 * Properties for {@link GPUCellPyramid}: the row inputs of {@link GPUCellAggregation} plus the
 * levels to build.
 */
export type GPUCellPyramidProps = Omit<
  GPUCellAggregationProps,
  'id' | 'resolution' | 'output' | 'extraCounts'
> & {
  /** Prefix for generated node and transient IDs. Defaults to `'cell-pyramid'`. */
  id?: string;
  /** Levels, finest first. Level 0 aggregates the rows; level `i` rolls up level `i - 1`. */
  levels: readonly GPUCellPyramidLevel[];
  /**
   * Optional packed `uint32` view of `levels.length` rows receiving each level's clamped count,
   * the input of {@link GPUCellLevelSelection}.
   */
  levelCounts?: GraphDataView<'uint32'>;
};

/**
 * Builds a zoom pyramid of cell tables: aggregates rows once at the finest level, then rolls each
 * level up into the next coarser one without re-reading the rows.
 *
 * Coarser levels that request sums or extremes need them on every finer level; missing
 * intermediate columns become graph transients. Every level is exact (see
 * {@link GPUCellRollup}), and a level's overflow flag includes every finer level's overflow.
 */
export class GPUCellPyramid implements GPUMapGraphRecipe {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Stable recipe name. */
  readonly recipe = 'cell-pyramid';
  /** Validated properties. */
  readonly props: GPUCellPyramidProps;

  constructor(props: GPUCellPyramidProps) {
    this.id = props.id ?? this.recipe;
    this.props = props;
    const id = this.id;
    if (props.levels.length < 1) {
      throw new Error(`${id} needs at least one level`);
    }
    for (const [levelIndex, level] of props.levels.entries()) {
      getCellKeyLayout(props.family, level.resolution);
      validateCellTable(id, `levels[${levelIndex}].output`, level.output);
      if (levelIndex > 0 && level.resolution >= props.levels[levelIndex - 1].resolution) {
        throw new Error(`${id} level resolutions must strictly decrease`);
      }
    }
    if (props.levelCounts) {
      validatePackedUint32View(props.levelCounts, `${id} levelCounts`);
      if (props.levelCounts.length !== props.levels.length) {
        throw new Error(`${id} levelCounts must hold one row per level`);
      }
    }
    // Validate the finest level eagerly; roll-ups validate when nodes are built.
    this.createAggregation(props.levels[0].output, []);
  }

  /** Row offset of each level's `cells` within its buffer, for {@link GPUCellLevelSelection}. */
  get levelFirstRows(): number[] {
    return this.props.levels.map(level => getCellTableFirstRow(level.output));
  }

  /** Returns the aggregation nodes of level 0 followed by one roll-up per coarser level. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    validateGraphViewsBelongToGraph(id, graph, [props.levelCounts]);
    const levelCountViews = props.levels.map((_, levelIndex) =>
      props.levelCounts
        ? [
            graph.createDataView(props.levelCounts.buffer, {
              format: 'uint32',
              length: 1,
              byteOffset: props.levelCounts.byteOffset + levelIndex * Uint32Array.BYTES_PER_ELEMENT
            })
          ]
        : []
    );
    const tables = this.getLevelTables(graph);
    const nodes: GPUCommandNode<Parameters>[] = [
      ...this.createAggregation(tables[0], levelCountViews[0]).getCommandNodes(graph)
    ];
    for (let levelIndex = 1; levelIndex < tables.length; levelIndex++) {
      nodes.push(
        ...new GPUCellRollup({
          id: `${id}-level-${levelIndex}`,
          family: props.family,
          sourceResolution: props.levels[levelIndex - 1].resolution,
          resolution: props.levels[levelIndex].resolution,
          source: tables[levelIndex - 1],
          sumScale: props.sumScale,
          output: tables[levelIndex],
          extraCounts: levelCountViews[levelIndex]
        }).getCommandNodes(graph)
      );
    }
    return nodes;
  }

  private createAggregation(
    output: GPUCellTable,
    extraCounts: readonly GraphDataView<'uint32'>[]
  ): GPUCellAggregation {
    const {levels, levelCounts: _levelCounts, ...rowProps} = this.props;
    return new GPUCellAggregation({
      ...rowProps,
      id: `${this.id}-level-0`,
      resolution: levels[0].resolution,
      output,
      extraCounts
    });
  }

  /** Level tables with graph-transient sums and extremes where a coarser level needs them. */
  private getLevelTables<Parameters>(graph: GPUCommandGraph<Parameters>): GPUCellTable[] {
    const {levels} = this.props;
    const tables = levels.map(level => ({...level.output}));
    let needsSums = false;
    let needsMinimums = false;
    let needsMaximums = false;
    for (let levelIndex = tables.length - 1; levelIndex >= 0; levelIndex--) {
      const table = tables[levelIndex];
      const capacity = table.cells.length;
      const prefix = `${this.id}-level-${levelIndex}`;
      if (needsSums && !table.sums) {
        table.sums = createTransientView(graph, `${prefix}-sums`, 'uint32x2', capacity);
      }
      if (needsMinimums && !table.minimums) {
        table.minimums = createTransientView(graph, `${prefix}-minimums`, 'float32', capacity);
      }
      if (needsMaximums && !table.maximums) {
        table.maximums = createTransientView(graph, `${prefix}-maximums`, 'float32', capacity);
      }
      needsSums ||= Boolean(table.sums || table.sumValues);
      needsMinimums ||= Boolean(table.minimums);
      needsMaximums ||= Boolean(table.maximums);
    }
    return tables;
  }
}

/** Caller-owned outputs of {@link GPUCellLevelSelection}. */
export type GPUCellLevelSelectionOutput = {
  /** One-row scalar receiving the active level's occupied row count. */
  count: GraphDataView<'uint32'>;
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
export class GPUCellLevelSelection implements GPUMapGraphRecipe {
  /** Prefix for every node ID. */
  readonly id: string;
  /** Stable recipe name. */
  readonly recipe = 'cell-level-selection';
  /** Validated properties. */
  readonly props: GPUCellLevelSelectionProps;

  constructor(props: GPUCellLevelSelectionProps) {
    this.id = props.id ?? this.recipe;
    this.props = props;
    const id = this.id;
    validatePackedUint32View(props.levelCounts, `${id} levelCounts`);
    validatePackedUint32View(props.activeLevel, `${id} activeLevel`);
    validatePackedUint32View(props.output.count, `${id} output.count`);
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
      output.count,
      output.firstRow,
      output.drawArguments
    ]);
    const levelCount = props.levelCounts.length;
    const firstRows = props.levelFirstRows ?? new Array<number>(levelCount).fill(0);
    return [
      createMapGraphKernelNode<Parameters>(graph, {
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
            view: output.count,
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
