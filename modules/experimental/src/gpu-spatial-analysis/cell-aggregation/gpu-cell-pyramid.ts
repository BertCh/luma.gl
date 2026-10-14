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
import {
  validateGraphViewsBelongToGraph,
  validatePackedView
} from '../../utils/gpu-contributor-utils';
import {createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import {getCellKeyLayout} from './cell-keys';
import {getCellTableFirstRow, validateCellTable, type GPUCellTable} from './cell-table';
import {GPUCellAggregation, type GPUCellAggregationProps} from './gpu-cell-aggregation';
import {GPUCellRollup} from './gpu-cell-rollup';
import type {GPUSpatialContext} from '../../analysis-core/contracts/spatial-context';

/** Words per level-count slot: 64 words = 256 bytes, the storage binding offset alignment. */
const LEVEL_COUNT_SLOT_STRIDE = 64;

/** Caller-owned outputs of {@link GPUCellPyramid}. */
export type GPUCellPyramidOutput = {
  /** One table per level, finest first, matching `resolutions`. */
  levels: readonly GPUCellTable[];
  /**
   * Optional packed `uint32` view of `levels.length` rows receiving each level's clamped count,
   * the input of {@link GPUCellLevelSelection}.
   */
  levelCounts?: GraphDataView<'uint32'>;
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
  /**
   * Resolution of each level, finest first and strictly decreasing. Level 0 aggregates the rows;
   * level `i` rolls up level `i - 1`.
   */
  resolutions: readonly number[];
  /** Caller-owned level tables and counts. */
  output: GPUCellPyramidOutput;
};

/**
 * Builds a zoom pyramid of cell tables: aggregates rows once at the finest level, then rolls each
 * level up into the next coarser one without re-reading the rows.
 *
 * Coarser levels that request sums or extremes need them on every finer level; missing
 * intermediate columns become graph transients. Every level is exact (see
 * {@link GPUCellRollup}), and a level's overflow flag includes every finer level's overflow.
 */
export class GPUCellPyramid implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUCellPyramidProps;
  /** Resolved spatial context of the input rows, see {@link GPUCellAggregation}. */
  readonly spatialContext: GPUSpatialContext;

  constructor(props: GPUCellPyramidProps) {
    this.id = props.id ?? 'cell-pyramid';
    this.props = props;
    const id = this.id;
    const {resolutions, output} = props;
    if (resolutions.length < 1) {
      throw new Error(`${id} needs at least one level`);
    }
    if (output.levels.length !== resolutions.length) {
      throw new Error(`${id} output.levels must hold one table per resolution`);
    }
    for (const [levelIndex, resolution] of resolutions.entries()) {
      getCellKeyLayout(props.family, resolution);
      validateCellTable(id, `output.levels[${levelIndex}]`, output.levels[levelIndex]);
      if (levelIndex > 0 && resolution >= resolutions[levelIndex - 1]) {
        throw new Error(`${id} resolutions must strictly decrease`);
      }
    }
    if (output.levelCounts) {
      validatePackedView(output.levelCounts, ['uint32'], `${id} output.levelCounts`);
      if (output.levelCounts.length !== resolutions.length) {
        throw new Error(`${id} output.levelCounts must hold one row per level`);
      }
    }
    // Validate the finest level eagerly; roll-ups validate when nodes are built.
    this.spatialContext = this.createAggregation(output.levels[0], []).spatialContext;
  }

  /** Row offset of each level's `cells` within its buffer, for {@link GPUCellLevelSelection}. */
  get levelFirstRows(): number[] {
    return this.props.output.levels.map(getCellTableFirstRow);
  }

  /**
   * Returns the aggregation nodes of level 0, one roll-up per coarser level and, with
   * `output.levelCounts`, the node that packs the level counts.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {levelCounts} = props.output;
    validateGraphViewsBelongToGraph(id, graph, [levelCounts]);
    // Rule 12: each level publishes its count into its own 256-byte slot of a scratch view, so the
    // slot binds at element offset 0 and every level shares one publish module (a slice of the
    // packed `levelCounts` would bake a different element offset per level). One copy node then
    // packs the slots into `levelCounts`.
    const levelCountSlots = levelCounts
      ? createTransientView(
          graph,
          `${id}-level-count-slots`,
          'uint32',
          props.resolutions.length * LEVEL_COUNT_SLOT_STRIDE
        )
      : undefined;
    const levelCountViews = props.resolutions.map((_, levelIndex) =>
      levelCountSlots
        ? [
            graph.createDataView(levelCountSlots.buffer, {
              format: 'uint32',
              length: 1,
              byteOffset:
                levelCountSlots.byteOffset +
                levelIndex * LEVEL_COUNT_SLOT_STRIDE * Uint32Array.BYTES_PER_ELEMENT
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
          sourceResolution: props.resolutions[levelIndex - 1],
          resolution: props.resolutions[levelIndex],
          source: tables[levelIndex - 1],
          sumScale: props.sumScale,
          output: tables[levelIndex],
          extraCounts: levelCountViews[levelIndex]
        }).getCommandNodes(graph)
      );
    }
    if (levelCounts && levelCountSlots) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-level-counts`,
          operation: 'GPUCellPyramid',
          variant: 'level-counts',
          bindings: [
            {name: 'slots', view: levelCountSlots, type: 'u32', access: 'read'},
            {name: 'levelCounts', view: levelCounts, type: 'u32', access: 'read_write'}
          ],
          invocationCount: props.resolutions.length,
          body: `levelCounts[levelCountsOffset + index] = slots[slotsOffset + index * ${LEVEL_COUNT_SLOT_STRIDE}u];`
        })
      );
    }
    return nodes;
  }

  private createAggregation(
    output: GPUCellTable,
    extraCounts: readonly GraphDataView<'uint32'>[]
  ): GPUCellAggregation {
    const {resolutions, output: _output, ...rowProps} = this.props;
    return new GPUCellAggregation({
      ...rowProps,
      id: `${this.id}-level-0`,
      resolution: resolutions[0],
      output,
      extraCounts
    });
  }

  /** Level tables with graph-transient sums and extremes where a coarser level needs them. */
  private getLevelTables<Parameters>(graph: GPUCommandGraph<Parameters>): GPUCellTable[] {
    const tables = this.props.output.levels.map(table => ({...table}));
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
