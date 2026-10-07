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
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {createFillNode, createPublishNode} from '../../utils/wgsl-kernel-nodes';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {
  CELL_COVER_RANGE_STRIDE,
  createCoverCountNode,
  createCoverRingFeatureNode,
  createCoverSlabBasesNode,
  createCoverSlabEdgesNode,
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
  /**
   * Optional core flag per output cell, same length as `cells`: 1 when the cell is a core cell
   * (Mosaic "chip" terminology), provably inside the feature, 0 for border cells. A join may skip
   * the exact point-in-polygon test for points that fall in core cells. Conservative: a cell is
   * core only when its center is inside and no polygon edge enters the cell (its bounding box for
   * H3) grown by a small safety margin (about 2e-4 degrees plus 0.1% of the cell size), so cells
   * touching the boundary or within f32 error of it are reported as border. Cells that are not
   * core may still lie entirely inside. Rows past `count` are not written.
   */
  core?: GraphDataView<'uint32'>;
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
  /**
   * Compile-time. Builds a per-feature edge-slab index (edges bucketed by y) so each candidate
   * tests only the edges near its latitude instead of every edge of its feature: the test cost per
   * candidate drops from the feature's vertex count to roughly the edges crossing one slab. Results
   * are identical, and features that do not fit the index (too few candidates, non-finite
   * vertices, index capacity) test every edge as before. Needs GeoArrow offsets that are
   * monotone. Defaults to `true` when `candidateCapacity * vertexCount` is at least 4 million
   * (about where the extra passes pay for themselves), else `false`.
   */
  edgeSlabs?: boolean;
  /**
   * Compile-time. Capacity of the edge-slab index in (edge, slab) entries, over all features.
   * Defaults to four times the vertex count, which holds polygons whose edges span a few slabs
   * each; features that exceed it test every edge instead. Ignored without `edgeSlabs`.
   */
  edgeSlabEntryCapacity?: number;
  /** Capacity-bounded output. */
  output: GPUCellCoverOutput;
};

const DEFAULT_ID = 'cell-cover';
/** Candidate capacity times vertex count from which the edge-slab index is built by default. */
const EDGE_SLAB_DEFAULT_WORK = 4_000_000;

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
 * With `output.core`, cells provably inside the polygon are flagged so joins can skip the exact
 * test for them (see {@link GPUCellCoverOutput.core}). H3 lattices are clamped to latitudes within 89 degrees, and cost grows as longitude spacing
 * divided by cos(latitude).
 */
export class GPUCellCover implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUCellCoverProps;
  /** Number of features, `featureOffsets.length - 1`. */
  readonly featureCount: number;
  /** Resolved containment. */
  readonly containment: GPUCellCoverContainment;
  /** Whether the edge-slab index is built. */
  readonly edgeSlabs: boolean;

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
    this.edgeSlabs =
      (props.edgeSlabs ??
        props.candidateCapacity * props.polygonPositions.length >= EDGE_SLAB_DEFAULT_WORK) &&
      props.ringOffsets.length >= 2 &&
      props.polygonOffsets.length >= 2;
    if (
      props.edgeSlabEntryCapacity !== undefined &&
      (!Number.isSafeInteger(props.edgeSlabEntryCapacity) || props.edgeSlabEntryCapacity < 1)
    ) {
      throw new Error(`${id} edgeSlabEntryCapacity must be a positive integer`);
    }
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
    if (output.core) {
      validatePackedUint32View(output.core, `${id} output.core`);
      if (output.core.length !== output.featureIds.length) {
        throw new Error(`${id} output.core must have the same length as output.featureIds`);
      }
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
      [
        output.featureIds,
        output.cells,
        output.core,
        output.count,
        output.overflow,
        output.totalCount
      ],
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
      output.core,
      output.count,
      output.overflow,
      output.totalCount
    ]);
    const vertexCount = props.polygonPositions.length;
    const slabCapacities = this.edgeSlabs
      ? {
          slabCapacity: Math.floor(vertexCount / 4) + 1,
          entryCapacity: props.edgeSlabEntryCapacity ?? Math.max(4 * vertexCount, 1024)
        }
      : undefined;
    const context: CellCoverKernelContext = {
      id,
      family: props.family,
      resolution: props.resolution,
      containment: this.containment,
      computeCore: Boolean(output.core),
      featureCount,
      candidateCapacity,
      edgeSlabs: slabCapacities,
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
    const slabNodes: GPUCommandNode<Parameters>[] = [];
    let slabViews:
      | {
          slabOffsets: GraphDataView<'uint32'>;
          slabEntries: GraphDataView<'uint32x2'>;
          nextVertex: GraphDataView<'uint32'>;
          slabNumbers: GraphDataView<'uint32'>;
        }
      | undefined;
    if (slabCapacities) {
      // Edge-slab index: per-feature slab counts and bases, then a counting sort of the edges into
      // slabs (count, scan, fill), all vertex-parallel. Entries are (first vertex, second vertex).
      const {slabCapacity, entryCapacity} = slabCapacities;
      const slabNumbers = createTransientView(
        graph,
        `${id}-slab-numbers`,
        'uint32',
        featureCount + 1
      );
      const slabStarts = createTransientView(
        graph,
        `${id}-slab-starts`,
        'uint32',
        featureCount + 1
      );
      const ringFeature = createTransientView(
        graph,
        `${id}-ring-feature`,
        'uint32',
        props.ringOffsets.length - 1
      );
      const nextVertex = createTransientView(graph, `${id}-next-vertex`, 'uint32', vertexCount);
      const vertexFeature = createTransientView(
        graph,
        `${id}-vertex-feature`,
        'uint32',
        vertexCount
      );
      const slabCounters = createTransientView(
        graph,
        `${id}-slab-counters`,
        'uint32',
        slabCapacity + 1
      );
      const slabOffsets = createTransientView(
        graph,
        `${id}-slab-offsets`,
        'uint32',
        slabCapacity + 1
      );
      const slabEntries = createTransientView(
        graph,
        `${id}-slab-entries`,
        'uint32x2',
        entryCapacity
      );
      const clearCounters = (name: string) =>
        createFillNode<Parameters>(graph, {
          id: `${id}-slab-clear-${name}`,
          operation: 'GPUCellCover',
          view: slabCounters,
          type: 'u32',
          value: '0u'
        });
      slabNodes.push(
        ...new GPUScan({
          id: `${id}-slab-number-scan`,
          input: slabNumbers,
          output: slabStarts,
          mode: 'exclusive'
        }).getCommandNodes(graph),
        createCoverSlabBasesNode<Parameters>(graph, context, {ranges, slabStarts}),
        createCoverRingFeatureNode<Parameters>(graph, context, ringFeature),
        clearCounters('count'),
        createCoverSlabEdgesNode<Parameters>(graph, context, 'count', {
          ranges,
          nextVertex,
          vertexFeature,
          ringFeature,
          slabCounters
        }),
        ...new GPUScan({
          id: `${id}-slab-offset-scan`,
          input: slabCounters,
          output: slabOffsets,
          mode: 'exclusive'
        }).getCommandNodes(graph),
        clearCounters('cursor'),
        createCoverSlabEdgesNode<Parameters>(graph, context, 'fill', {
          ranges,
          nextVertex,
          vertexFeature,
          slabCounters,
          slabOffsets,
          slabEntries
        })
      );
      slabViews = {slabOffsets, slabEntries, nextVertex, slabNumbers};
    }
    return [
      createCoverCountNode<Parameters>(graph, context, counts, ranges, slabViews?.slabNumbers),
      ...new GPUScan({
        id: `${id}-candidate-scan`,
        input: counts,
        output: starts,
        mode: 'exclusive'
      }).getCommandNodes(graph),
      ...slabNodes,
      createCoverTestNode<Parameters>(graph, context, {
        ranges,
        starts,
        flags,
        candidateCells,
        slabOffsets: slabViews?.slabOffsets,
        slabEntries: slabViews?.slabEntries,
        nextVertex: slabViews?.nextVertex
      }),
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
        outputCells: output.cells,
        outputCore: output.core
      }),
      createCoverFinalizeNode<Parameters>(graph, context, {
        starts,
        accepted,
        total,
        candidateOverflow
      }),
      createPublishNode<Parameters>(graph, {
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
