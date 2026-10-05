// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUVisibilityWorkflow,
  GraphVectorView,
  validatePackedUint32View,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUVisibilityPredicate,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCompactOutput} from '../../utils/gpu-contributor-types';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode, createPublishNode} from '../../utils/wgsl-kernel-nodes';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateCompactOutput
} from '../../utils/gpu-contributor-utils';

/**
 * Properties for {@link GPUResidentRowSelection}.
 *
 * Per-frame (no recompile): the contents of every input buffer, including `liveMask`,
 * `tileVisibility.tileMask`, and the additional predicate masks. Topology (needs a new graph):
 * the row count, `output.ids.length`, `tileVisibility.tileMask.length`, which optional props are
 * present, and the number of `additionalPredicates`.
 *
 * Every row-aligned prop must be a single packed `GraphDataView`. Arena columns are one buffer of
 * fixed row capacity by design, so `GraphVectorView` inputs are rejected.
 */
export type GPUResidentRowSelectionProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'resident-row-selection'`. */
  id?: string;
  /**
   * Packed uint32 live mask: nonzero for live rows. Its length is the row count, normally the
   * arena `rowCapacity`.
   */
  liveMask: GraphDataView<'uint32'>;
  /**
   * Optional per-frame, per-tile gate, for example written by the app or a tile LOD selection
   * contributor. A row passes when `slot < tileMask.length && tileMask[slot] != 0`, so dead rows
   * (`GPU_RESIDENCY_ARENA_DEAD_SLOT`) and slots beyond the mask are rejected.
   */
  tileVisibility?: {
    /** Packed uint32 tile slot of each row, row-aligned with `liveMask`. */
    rowTileSlots: GraphDataView<'uint32'>;
    /** Packed uint32 per-tile mask, indexed by tile slot. */
    tileMask: GraphDataView<'uint32'>;
  };
  /** Optional packed, row-aligned predicate masks (for example a time mask) ANDed with the rest. */
  additionalPredicates?: readonly GPUVisibilityPredicate[];
  /** Optional packed row-aligned stable IDs. Arena row indices are emitted when omitted. */
  sourceIds?: GraphDataView<'uint32'>;
  /** Caller-owned bounded result. `output.ids.length` is the capacity. */
  output: GPUCompactOutput;
  /** Optional caller-owned canonical 0/1 mask of selected rows (all predicates ANDed). */
  outputMask?: GraphDataView<'uint32'>;
  /**
   * Optional packed one-row destination that receives the clamped count, typically an indirect
   * draw record's `instanceCount` imported with `graph.importGPUData(id, drawCommands.getInstanceCountData(0))`.
   */
  drawInstanceCount?: GraphDataView<'uint32'>;
};

/**
 * Selects resident rows of a fixed-topology residency arena: live mask, optional per-tile gate,
 * and optional extra predicates reduce to bounded stable IDs, a clamped count with overflow, and
 * an optional indirect-draw instance count.
 *
 * Composition: an optional tile-expansion kernel, `GPUVisibilityWorkflow` (predicates in order:
 * live as `'selection'`, tile gate as `'lod'`, then `additionalPredicates`), and one publish
 * kernel. The node count depends only on which optional props are set, never on how many tiles
 * are resident or where they sit, so tiles can stream in and out without recompiling the graph.
 */
export class GPUResidentRowSelection implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUResidentRowSelectionProps;

  constructor(props: GPUResidentRowSelectionProps) {
    this.id = props.id ?? 'resident-row-selection';
    this.props = props;
    const id = this.id;
    const rows = props.liveMask.length;
    const additionalPredicates = props.additionalPredicates ?? [];

    const rowAligned: (readonly [string, unknown])[] = [
      ['liveMask', props.liveMask],
      ['tileVisibility.rowTileSlots', props.tileVisibility?.rowTileSlots],
      ['tileVisibility.tileMask', props.tileVisibility?.tileMask],
      ['sourceIds', props.sourceIds],
      ['outputMask', props.outputMask],
      ...additionalPredicates.map(
        (predicate, index) => [`additionalPredicates[${index}]`, predicate.mask] as const
      )
    ];
    for (const [name, view] of rowAligned) {
      if (!view) {
        continue;
      }
      if (view instanceof GraphVectorView) {
        throw new Error(
          `${id} ${name} must be a packed GraphDataView; arena columns are single views`
        );
      }
      validatePackedUint32View(view as GraphDataView<'uint32'>, `${id} ${name}`);
      if (name !== 'tileVisibility.tileMask' && (view as GraphDataView).length !== rows) {
        throw new Error(`${id} ${name} length must equal liveMask length`);
      }
    }
    validateCompactOutput(id, props.output);
    if (props.drawInstanceCount) {
      validatePackedUint32View(props.drawInstanceCount, `${id} drawInstanceCount`);
      if (props.drawInstanceCount.length < 1) {
        throw new Error(`${id} drawInstanceCount must contain one uint32 row`);
      }
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [
        props.output.ids,
        props.output.count,
        props.output.overflow,
        props.output.totalCount,
        props.outputMask,
        props.drawInstanceCount
      ],
      [
        props.liveMask,
        props.tileVisibility?.rowTileSlots,
        props.tileVisibility?.tileMask,
        props.sourceIds,
        ...additionalPredicates.map(predicate => predicate.mask)
      ]
    );
  }

  /** Returns the optional tile-expansion, visibility, and publish nodes in dependency order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    const {output, tileVisibility, liveMask} = props;
    const additionalPredicates = props.additionalPredicates ?? [];
    validateGraphViewsBelongToGraph(id, graph, [
      liveMask,
      tileVisibility?.rowTileSlots,
      tileVisibility?.tileMask,
      props.sourceIds,
      props.outputMask,
      props.drawInstanceCount,
      output.ids,
      output.count,
      output.overflow,
      output.totalCount,
      ...additionalPredicates.map(predicate => predicate.mask)
    ]);
    const rows = liveMask.length;
    const nodes: GPUCommandNode<Parameters>[] = [];
    const predicates: GPUVisibilityPredicate[] = [{kind: 'selection', mask: liveMask}];

    if (tileVisibility) {
      const {rowTileSlots, tileMask} = tileVisibility;
      const rowTileMask = createTransientView(graph, `${id}-tile-row-mask`, 'uint32', rows);
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-tile-expand`,
          operation: 'GPUResidentRowSelection',
          variant: 'tile-expand',
          bindings: [
            {
              name: 'rowTileSlots',
              view: rowTileSlots,
              type: 'u32',
              access: 'read'
            },
            {name: 'tileMask', view: tileMask, type: 'u32', access: 'read'},
            {
              name: 'rowMaskOut',
              view: rowTileMask,
              type: 'u32',
              access: 'read_write'
            }
          ],
          invocationCount: rows,
          declarations: `const TILE_COUNT: u32 = ${tileMask.length}u;`,
          body: `let slot = rowTileSlots[rowTileSlotsOffset + index];
  var tileVisible = 0u;
  if (slot < TILE_COUNT && tileMask[tileMaskOffset + slot] != 0u) {
    tileVisible = 1u;
  }
  rowMaskOut[rowMaskOutOffset + index] = tileVisible;`
        })
      );
      predicates.push({kind: 'lod', mask: rowTileMask});
    }
    predicates.push(...additionalPredicates);

    // Compact straight into the caller's IDs when they can hold every row; otherwise compact into
    // full-size scratch and let the publish kernel copy the bounded prefix.
    const direct = rows > 0 && output.ids.length >= rows;
    const total = createTransientView(graph, `${id}-total`, 'uint32', 1);
    const compactIds = direct
      ? output.ids
      : createTransientView(graph, `${id}-compact-ids`, 'uint32', rows);
    nodes.push(
      ...new GPUVisibilityWorkflow({
        id: `${id}-visibility`,
        predicates,
        output: compactIds,
        count: total,
        outputMask: props.outputMask,
        sourceIds: props.sourceIds
      }).getCommandNodes(graph)
    );
    nodes.push(
      createPublishNode<Parameters>(graph, {
        id: `${id}-publish`,
        operation: 'GPUResidentRowSelection',
        totalCount: total,
        compactIds: direct ? undefined : compactIds,
        output,
        extraCounts: props.drawInstanceCount ? [props.drawInstanceCount] : []
      })
    );
    return nodes;
  }
}
