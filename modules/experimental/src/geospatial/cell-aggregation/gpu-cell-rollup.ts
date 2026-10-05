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
import {createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {CELL_KEY_WGSL, getCellKeyLayout, type CellKeyLayout, type GPUCellFamily} from './cell-keys';
import {
  getCellTableNodes,
  getCellTableViews,
  GPU_CELL_DEFAULT_SUM_SCALE,
  hasCellKeyHighWord,
  validateCellTable,
  validateSumScale,
  type GPUCellTable
} from './cell-table';

const OPERATION = 'GPUCellRollup';

/**
 * Properties for {@link GPUCellRollup}.
 *
 * Per-frame (no recompile): the contents of the source table. Topology: family, resolutions,
 * `sumScale`, which columns are present, and capacities.
 */
export type GPUCellRollupProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'cell-rollup'`. */
  id?: string;
  /** Grid family of both tables. */
  family: GPUCellFamily;
  /** Resolution of every key in `source`. */
  sourceResolution: number;
  /** Coarser output resolution, at most `sourceResolution`. */
  resolution: number;
  /**
   * A table written by {@link GPUCellAggregation} or another {@link GPUCellRollup} at
   * `sourceResolution`: ascending keys in rows `[0, count)`. Output sums need `source.sums`, and
   * output extremes need the matching source extremes.
   */
  source: GPUCellTable;
  /** Fixed-point scale used by the source sums. Defaults to 65536. */
  sumScale?: number;
  /** Caller-owned output table at `resolution`. */
  output: GPUCellTable;
  /** Optional extra one-row destinations of the clamped output count. @internal */
  extraCounts?: readonly GraphDataView<'uint32'>[];
};

/**
 * Rolls a sorted cell table up to a coarser resolution without re-reading the source rows.
 *
 * The parent key truncates the cell path (`cellToParent` for Quadbin and H3), which is monotone
 * in the child key, so the source order is already the parent order and no sort is needed: one
 * key pass, a head scan, and per-parent reductions of the child cells. Counts and fixed-point sums
 * are integer sums, so the result equals a direct aggregation at `resolution` bit for bit. The
 * output overflow flag includes the source overflow.
 */
export class GPUCellRollup implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUCellRollupProps;
  /** Key layout of the output resolution. */
  readonly layout: CellKeyLayout;
  /** Fixed-point scale of the sums. */
  readonly sumScale: number;

  constructor(props: GPUCellRollupProps) {
    this.id = props.id ?? 'cell-rollup';
    this.props = props;
    const id = this.id;
    this.layout = getCellKeyLayout(props.family, props.resolution);
    getCellKeyLayout(props.family, props.sourceResolution);
    if (props.resolution > props.sourceResolution) {
      throw new Error(`${id} resolution must not exceed sourceResolution`);
    }
    this.sumScale = props.sumScale ?? GPU_CELL_DEFAULT_SUM_SCALE;
    validateSumScale(id, this.sumScale);
    validateCellTable(id, 'source', props.source);
    validateCellTable(id, 'output', props.output);
    const {source, output} = props;
    if ((output.sums || output.sumValues) && !source.sums) {
      throw new Error(`${id} source.sums is required to roll up sums`);
    }
    if ((output.minimums && !source.minimums) || (output.maximums && !source.maximums)) {
      throw new Error(`${id} source extremes are required to roll up extremes`);
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [...getCellTableViews(output), ...(props.extraCounts ?? [])],
      getCellTableViews(source)
    );
  }

  /** Returns parent-key, boundary, table, reduction, and publish nodes in order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, layout} = this;
    validateGraphViewsBelongToGraph(id, graph, [
      ...getCellTableViews(props.source),
      ...getCellTableViews(props.output),
      ...(props.extraCounts ?? [])
    ]);
    const rows = props.source.cells.length;
    const twoWords = hasCellKeyHighWord(layout);
    const keyLow = createTransientView(graph, `${id}-key-low`, 'uint32', rows);
    const keyHigh = twoWords
      ? createTransientView(graph, `${id}-key-high`, 'uint32', rows)
      : undefined;
    const keyNode = createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-keys`,
      operation: OPERATION,
      variant: 'parent-keys',
      bindings: [
        {
          name: 'childCells',
          view: props.source.cells,
          type: 'u32',
          access: 'read'
        },
        {
          name: 'childCount',
          view: props.source.count,
          type: 'u32',
          access: 'read'
        },
        {name: 'keyLow', view: keyLow, type: 'u32', access: 'read_write'},
        ...(keyHigh
          ? [
              {
                name: 'keyHigh',
                view: keyHigh,
                type: 'u32',
                access: 'read_write'
              } as const
            ]
          : [])
      ],
      invocationCount: rows,
      declarations: `const SOURCE_RESOLUTION: u32 = ${props.sourceResolution}u;
const RESOLUTION_MASK: u32 = ${props.family === 'quadbin' ? '0x1fu' : '0xfu'};
${CELL_KEY_WGSL}`,
      body: `var compact = cellShiftLeft(vec2u(0u, 1u), ${layout.width}u);
  let key = vec2u(childCells[childCellsOffset + 2u * index + 1u], childCells[childCellsOffset + 2u * index]);
  if (index < childCount[childCountOffset] && ((key.x >> 20u) & RESOLUTION_MASK) == SOURCE_RESOLUTION) {
    compact = cellGetCompactKey(key, ${layout.lowBit}u, ${layout.width}u);
  }
  keyLow[keyLowOffset + index] = compact.y;
  ${keyHigh ? 'keyHigh[keyHighOffset + index] = compact.x;' : ''}`
    });
    return [
      keyNode,
      ...getCellTableNodes<Parameters>(graph, {
        id,
        operation: OPERATION,
        layout,
        keyLow,
        keyHigh,
        sorted: true,
        source: {kind: 'cells', table: props.source},
        sumScale: this.sumScale,
        output: props.output,
        overflowSources: [props.source.overflow],
        extraCounts: props.extraCounts
      })
    ];
  }
}
