// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUScan,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {getCellKeyLayout, type GPUCellFamily} from '../cell-aggregation/cell-keys';
import type {GPUMapGraphRecipe} from '../map-graph-types';
import {createMapGraphPublishNode} from '../map-graph-kernels';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../map-graph-utils';
import {
  CELL_COVER_RANGE_STRIDE,
  createCoverCountNode,
  createCoverFinalizeNode,
  createCoverTestNode,
  createCoverWriteNode,
  type CellCoverKernelContext
} from './cell-cover-kernels';

/** Containment rule of {@link GPUCellCover}. */
export type GPUCellCoverContainment = 'center' | 'full' | 'intersects';

/**
 * Caller-owned, capacity-bounded result of {@link GPUCellCover}.
 *
 * `featureIds` and `cells` are parallel columns of the same capacity. `count` is clamped to that
 * capacity and `overflow` is 1 when the accepted cells or the candidate capacity overflowed.
 */
export type GPUCellCoverOutput = {
  /** Source feature ID (or feature row when no `featureIds` input) per output cell. */
  featureIds: GraphDataView<'uint32'>;
  /** Cell keys as two little-endian `uint32` words (`low`, `high`), the Arrow `Uint64` layout. */
  cells: GraphDataView<'uint32x2'>;
  /** One-row scalar receiving `min(totalCount, capacity)`. */
  count: GraphDataView<'uint32'>;
  /** One-row scalar receiving 1 when any capacity overflowed. */
  overflow: GraphDataView<'uint32'>;
  /** Optional one-row scalar receiving the unclamped accepted cell count. */
  totalCount?: GraphDataView<'uint32'>;
};

/**
 * Properties for {@link GPUCellCover}.
 *
 * Per-frame (no recompile): the contents of every input buffer. Compile-time: `family`,
 * `resolution`, `containment`, view lengths, `candidateCapacity` and which optional views exist.
 */
export type GPUCellCoverProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'cell-cover'`. */
  id?: string;
  /** Cell family. `'quadbin'` supports every containment; `'h3'` supports only `'center'`. */
  family: GPUCellFamily;
  /** Cell resolution (topology): Quadbin 0 to 26, H3 0 to 15. */
  resolution: number;
  /**
   * `'center'` (default): the cell center is inside the polygon. `'full'`: the whole cell is
   * inside (no polygon edge enters the cell interior and the cell is inside). `'intersects'`: the
   * cell interior and polygon interior overlap. Cells that merely touch the boundary are excluded
   * from `'full'` and `'intersects'` unless their interior overlaps.
   */
  containment?: GPUCellCoverContainment;
  /** Flattened polygon vertices, planar longitude/latitude degrees (as `GPUPointInPolygonJoin`). */
  polygonPositions: GraphDataView<'float32x2'>;
  /** Feature-to-polygon offsets with `featureCount + 1` entries, first 0. */
  featureOffsets: GraphDataView<'uint32'>;
  /** Polygon-to-ring offsets with a terminal entry. Every ring (shell or hole) is even-odd. */
  polygonOffsets: GraphDataView<'uint32'>;
  /** Ring-to-vertex offsets with a terminal entry. Rings close implicitly. */
  ringOffsets: GraphDataView<'uint32'>;
  /** Optional stable feature IDs written instead of feature rows. */
  featureIds?: GraphDataView<'uint32'>;
  /**
   * Maximum candidates (Quadbin tiles of the feature bounding boxes, or H3 lattice points) tested
   * per encoding, summed over features. Exceeding it sets `output.overflow`; features are then
   * covered in order until the candidates run out.
   */
  candidateCapacity: number;
  /** Capacity-bounded output. */
  output: GPUCellCoverOutput;
};

const DEFAULT_ID = 'cell-cover';

/**
 * Polyfills polygon features with Quadbin or H3 cells.
 *
 * Geometry is planar in longitude/latitude degrees (like `h3.polygonToCells` and turf); polygons
 * must not cross the antimeridian and rings are even-odd (holes by nesting). Pipeline: per-feature
 * candidate count, exclusive scan, per-candidate test, inclusive scan of accept flags, write. No
 * atomic append is used, so output is deterministic.
 *
 * Output order: by feature row, then candidate order within the feature. Quadbin candidates are the
 * tiles of the feature bounding box, rows from north to south and columns from west to east. H3
 * candidates are lattice points (south to north rows, west to east), and the emitted cell is the
 * one whose center is nearest that lattice point, so each cell appears exactly once.
 *
 * Quadbin tile edges are exact in longitude and computed with f32 `atan`/`exp` in latitude, so
 * results are exact except for polygon edges within about 1e-4 degrees of a tile edge or center,
 * and resolutions above about 16 are below f32 position resolution. H3 `'center'` compares the f32
 * cell center to the polygon, so it can differ from `h3.polygonToCells` for centers within about
 * 1e-5 degrees of an edge, and for the f32 forward index mismatches described in `H3_INDEX_WGSL`.
 * H3 lattices are clamped to latitudes within 89 degrees, and cost grows as longitude spacing
 * divided by cos(latitude).
 */
export class GPUCellCover implements GPUMapGraphRecipe {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Stable recipe name. */
  readonly recipe = 'cell-cover';
  /** Validated properties. */
  readonly props: GPUCellCoverProps;
  /** Number of features, `featureOffsets.length - 1`. */
  readonly featureCount: number;
  /** Resolved containment. */
  readonly containment: GPUCellCoverContainment;

  constructor(props: GPUCellCoverProps) {
    this.id = props.id ?? DEFAULT_ID;
    this.props = props;
    const {id} = this;
    if (props.family !== 'quadbin' && props.family !== 'h3') {
      throw new Error(`${id} family must be 'quadbin' or 'h3'`);
    }
    getCellKeyLayout(props.family, props.resolution);
    this.containment = props.containment ?? 'center';
    if (!['center', 'full', 'intersects'].includes(this.containment)) {
      throw new Error(`${id} containment must be 'center', 'full' or 'intersects'`);
    }
    if (props.family === 'h3' && this.containment !== 'center') {
      throw new Error(`${id} h3 supports only containment 'center'`);
    }
    validatePackedView(props.polygonPositions, ['float32x2'], `${id} polygonPositions`);
    for (const [name, view] of [
      ['featureOffsets', props.featureOffsets],
      ['polygonOffsets', props.polygonOffsets],
      ['ringOffsets', props.ringOffsets]
    ] as const) {
      validatePackedUint32View(view, `${id} ${name}`);
      if (view.length < 1) {
        throw new Error(`${id} ${name} requires a terminal entry`);
      }
    }
    this.featureCount = props.featureOffsets.length - 1;
    if (props.featureIds) {
      validatePackedUint32View(props.featureIds, `${id} featureIds`);
      if (props.featureIds.length !== this.featureCount) {
        throw new Error(`${id} featureIds length must equal the feature count`);
      }
    }
    if (
      !Number.isSafeInteger(props.candidateCapacity) ||
      props.candidateCapacity < 1 ||
      (this.featureCount + 1) * (props.candidateCapacity + 1) >= 0xffffffff
    ) {
      throw new Error(
        `${id} candidateCapacity must be a positive integer with (featureCount + 1) * (candidateCapacity + 1) below 2^32`
      );
    }
    const {output} = props;
    validatePackedUint32View(output.featureIds, `${id} output.featureIds`);
    validatePackedView(output.cells, ['uint32x2'], `${id} output.cells`);
    if (output.cells.length !== output.featureIds.length) {
      throw new Error(`${id} output.cells and output.featureIds must have equal lengths`);
    }
    if (output.featureIds.length < 1) {
      throw new Error(`${id} output capacity must be at least 1`);
    }
    for (const [name, view] of [
      ['count', output.count],
      ['overflow', output.overflow],
      ['totalCount', output.totalCount]
    ] as const) {
      if (view) {
        validatePackedUint32View(view, `${id} output.${name}`);
        if (view.length < 1) {
          throw new Error(`${id} output.${name} must contain one uint32 row`);
        }
      }
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [output.featureIds, output.cells, output.count, output.overflow, output.totalCount],
      [
        props.polygonPositions,
        props.featureOffsets,
        props.polygonOffsets,
        props.ringOffsets,
        props.featureIds
      ]
    );
  }

  /** Returns count, candidate scan, test, accept scan, write, finalize and publish nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, featureCount} = this;
    const {output, candidateCapacity} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      props.polygonPositions,
      props.featureOffsets,
      props.polygonOffsets,
      props.ringOffsets,
      props.featureIds,
      output.featureIds,
      output.cells,
      output.count,
      output.overflow,
      output.totalCount
    ]);
    const context: CellCoverKernelContext = {
      id,
      family: props.family,
      resolution: props.resolution,
      containment: this.containment,
      featureCount,
      candidateCapacity,
      polygonPositions: props.polygonPositions,
      featureOffsets: props.featureOffsets,
      polygonOffsets: props.polygonOffsets,
      ringOffsets: props.ringOffsets
    };
    const counts = createTransientView(graph, `${id}-counts`, 'uint32', featureCount + 1);
    const starts = createTransientView(graph, `${id}-starts`, 'uint32', featureCount + 1);
    const ranges = createTransientView(
      graph,
      `${id}-ranges`,
      'uint32',
      Math.max(featureCount, 1) * CELL_COVER_RANGE_STRIDE
    );
    const flags = createTransientView(graph, `${id}-flags`, 'uint32', candidateCapacity);
    const accepted = createTransientView(graph, `${id}-accepted`, 'uint32', candidateCapacity);
    const candidateCells = createTransientView(
      graph,
      `${id}-candidate-cells`,
      'uint32x2',
      candidateCapacity
    );
    const total = createTransientView(graph, `${id}-total`, 'uint32', 1);
    const candidateOverflow = createTransientView(graph, `${id}-candidate-overflow`, 'uint32', 1);
    return [
      createCoverCountNode<Parameters>(graph, context, counts, ranges),
      ...new GPUScan({
        id: `${id}-candidate-scan`,
        input: counts,
        output: starts,
        mode: 'exclusive'
      }).getCommandNodes(graph),
      createCoverTestNode<Parameters>(graph, context, {ranges, starts, flags, candidateCells}),
      ...new GPUScan({
        id: `${id}-accept-scan`,
        input: flags,
        output: accepted,
        mode: 'inclusive'
      }).getCommandNodes(graph),
      createCoverWriteNode<Parameters>(graph, context, {
        starts,
        accepted,
        candidateCells,
        featureIds: props.featureIds,
        outputFeatureIds: output.featureIds,
        outputCells: output.cells
      }),
      createCoverFinalizeNode<Parameters>(graph, context, {
        starts,
        accepted,
        total,
        candidateOverflow
      }),
      createMapGraphPublishNode<Parameters>(graph, {
        id: `${id}-publish`,
        operation: 'GPUCellCover',
        totalCount: total,
        output: {
          ids: output.featureIds,
          count: output.count,
          overflow: output.overflow,
          totalCount: output.totalCount
        },
        overflowSources: [candidateOverflow]
      })
    ];
  }
}
