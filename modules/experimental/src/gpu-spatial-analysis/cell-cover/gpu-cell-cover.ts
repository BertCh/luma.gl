// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  createTransientView,
  GPUScan,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {getCellKeyLayout, type GPUCellFamily} from '../cell-aggregation/cell-keys';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import type {GPUPolygonGeometryFields} from '../../analysis-core/contracts/index';
import {
  GPU_LONGITUDE_LATITUDE_SPATIAL_CONTEXT,
  GPU_LONGITUDE_LATITUDE_SPATIAL_CONTEXT_SUPPORT,
  GPU_LINEAR_EDGE_MODELS,
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import {createFillNode, createPublishNode} from '../../utils/wgsl-kernel-nodes';
import {getPolygonEdgeSlabNodes} from '../spatial-join/polygon-edge-slabs';
import {
  getStorageBindingElementLimit,
  validateCapacity,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateMaximumCount,
  validatePackedView
} from '../../utils/gpu-contributor-utils';
import {
  CELL_COVER_SCANLINE_AUTO_MINIMUM_CANDIDATE_CAPACITY,
  CELL_COVER_SCANLINE_MAXIMUM_SLOT_CAPACITY,
  getCellCoverRangesLength,
  getCellCoverScanlineNodes,
  type CellCoverScanlineSettings
} from './cell-cover-scanline';
import {
  CELL_COVER_MINIMUM_SLAB_CANDIDATES,
  CELL_COVER_RANGE_STRIDE,
  createCoverCountNode,
  createCoverFinalizeNode,
  createCoverTestNode,
  createCoverWriteNode,
  type CellCoverKernelContext
} from './cell-cover-kernels';

/**
 * Largest `candidateCapacity` of {@link GPUCellCover} on `device`: the candidate cell table holds one
 * `uint32x2` (8 bytes) per candidate in one storage binding, so 2^24 on a default device (128 MiB
 * bindings).
 */
export function getGPUCellCoverMaximumCandidateCapacity(device: Device): number {
  return getStorageBindingElementLimit(device, 8);
}

/**
 * Largest `positions.length + edgeSlabEntryCapacity` of {@link GPUCellCover} on `device` with the
 * edge-slab index: the slab entry table holds one `uint32x2` (8 bytes) per vertex plus one per extra
 * entry in one storage binding, so 2^24 on a default device. The default `edgeSlabEntryCapacity` of
 * `max(4 * positions.length, 1024)` is reduced to fit.
 */
export function getGPUCellCoverMaximumEdgeSlabEntryCount(device: Device): number {
  return getStorageBindingElementLimit(device, 8);
}

/** Containment rule of {@link GPUCellCover}. */
export type GPUCellCoverContainment = 'center' | 'full' | 'intersects';

/**
 * Caller-owned, capacity-bounded result of {@link GPUCellCover}.
 *
 * `featureIds` and `cells` are parallel columns of the same capacity. `count` is clamped to that
 * capacity. `status.overflow` reports only final output truncation; `status.candidateOverflow`
 * independently reports an incomplete candidate stage.
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
  /** Count, completeness and diagnostic scalars. */
  status: {
    /** One-row scalar receiving `min(requiredCount, capacity)`. */
    count: GraphDataView<'uint32'>;
    /** Optional unclamped accepted count; exact only when `candidateOverflow` is zero. */
    requiredCount?: GraphDataView<'uint32'>;
    /** One-row scalar receiving 1 when accepted cells exceeded the final output capacity. */
    overflow: GraphDataView<'uint32'>;
    /** One-row scalar receiving 1 when `candidateCapacity` truncated candidate enumeration. */
    candidateOverflow: GraphDataView<'uint32'>;
    /**
     * Optional one-row scalar receiving the unclamped number of enumerated candidates, summed over
     * features (the bounding-box cells, or the cells of the scanline runs of
     * {@link GPUCellCoverProps.candidateEnumeration}). It is the work the exact test performs, so it
     * measures what `'scanline'` saves; compare it with `candidateCapacity`.
     */
    candidateCount?: GraphDataView<'uint32'>;
    /**
     * Optional one-row scalar receiving the number of features with a non-pole edge whose raw
     * longitude difference exceeds 180 degrees (crossing the antimeridian, not split).
     * Rejected features produce no cells; pre-split multipolygons are accepted.
     */
    invalidCount?: GraphDataView<'uint32'>;
    /**
     * Optional one-row scalar receiving the number of covered features whose bounding box reaches
     * beyond the cover's latitude limit (89 degrees for H3, the Web Mercator limit of 85.051129
     * for Quadbin), so cells there are missing or clipped.
     */
    uncertainCount?: GraphDataView<'uint32'>;
  };
};

/**
 * Properties for {@link GPUCellCover}.
 *
 * Per-frame (no recompile): the contents of every input buffer. Compile-time: `family`,
 * `resolution`, `containment`, view lengths, `candidateCapacity` and which optional views exist.
 */
export type GPUCellCoverProps = GPUPolygonGeometryFields & {
  /** Prefix for generated node and transient IDs. Defaults to `'cell-cover'`. */
  id?: string;
  /** Cell family. `'quadbin'` supports every containment; `'h3'` supports only `'center'`. */
  family: GPUCellFamily;
  /**
   * Coordinate contract of `positions`. Defaults to, and must be, longitude/latitude without
   * a metric; the cover is planar in degrees, like `h3.polygonToCells` and turf.
   */
  spatialContext?: GPUSpatialContext;
  /** Cell resolution (topology): Quadbin 0 to 26, H3 0 to 15. */
  resolution: number;
  /**
   * `'center'` (default): the cell center is inside the polygon. `'full'`: the whole cell is
   * inside (no polygon edge enters the cell interior and the cell is inside). `'intersects'`: the
   * cell interior and polygon interior overlap. Cells that merely touch the boundary are excluded
   * from `'full'` and `'intersects'` unless their interior overlaps.
   */
  containment?: GPUCellCoverContainment;
  /** Optional stable feature IDs written instead of feature rows. */
  featureIds?: GraphDataView<'uint32'>;
  /**
   * Maximum candidates (Quadbin tiles or H3 lattice points) enumerated and tested per encoding,
   * summed over features. A candidate is a cell of the feature's bounding box, or, with the scanline
   * enumeration, a cell inside the row intervals where the polygon can be (far fewer for thin
   * diagonal or sparse multipart polygons, so the same capacity then covers much larger bounding
   * boxes). Exceeding it sets `output.status.candidateOverflow`; features are then covered in order
   * until the candidates run out. At most {@link getGPUCellCoverMaximumCandidateCapacity} (2^24 on
   * a default device).
   */
  candidateCapacity: number;
  /**
   * Compile-time. How candidates are enumerated per feature; the cells are bitwise identical for all
   * values, only the work differs.
   *
   * - `'bounds'`: every cell of the feature's bounding box is a candidate.
   * - `'scanline'`: boundary trace plus scanline fill. For each lattice row, the x intervals where
   *   a polygon edge meets the row or the row's test line is inside the polygon are computed
   *   (edge-parallel spans, edge crossings sorted by row and x, even-odd pairing), grown by an f32
   *   slack and merged; only cells inside them are candidates. Features that do not fit the stage
   *   (see `scanlineSlotCapacity`), have a non-finite vertex, or would enumerate 2^32 or more
   *   candidates keep the bounding box.
   * - `'auto'` (default): the stage is built when `candidateCapacity` is at least 65536, and each
   *   feature picks the scanline enumeration on the GPU when its bounding box has more than four
   *   times as many cells as the scanline work (twice the rows plus the edge-row slots).
   *
   * The built stage runs a fixed set of passes over `scanlineSlotCapacity` slots (event sorts and
   * scans included) even when no feature uses it.
   */
  candidateEnumeration?: 'bounds' | 'scanline' | 'auto';
  /**
   * Compile-time. Capacity of the scanline stage in (edge, row) slots over all features that use
   * it; events and runs are three times as many. A feature uses the scanline enumeration only if
   * its slots plus its rows fit after the earlier features, else it keeps the bounding box.
   * Defaults to `min(candidateCapacity, 2^20, max(16 * vertexCount, 16384))`. At most
   * `candidateCapacity` and 2^20. Ignored with `candidateEnumeration: 'bounds'`.
   */
  scanlineSlotCapacity?: number;
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
   * each (reduced so that the entry table fits, see
   * {@link getGPUCellCoverMaximumEdgeSlabEntryCount}); features that exceed it test every edge
   * instead. `positions.length + edgeSlabEntryCapacity` is at most that value (2^24 on a default device). Ignored without `edgeSlabs`.
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
 * Geometry is planar in longitude/latitude degrees (like `h3.polygonToCells` and turf) and rings
 * are even-odd (holes by nesting). Only linear edges are accepted. A feature with a non-pole
 * edge whose raw longitude difference exceeds 180 degrees gets no cells and increments
 * `output.status.invalidCount`; prepare it with `GPULonLatPolygonPreparation` first. Pre-split
 * multipolygons and same-pole edges are accepted. Full-width features enumerate full-width
 * lattices: with `candidateEnumeration: 'bounds'`, provision `candidateCapacity` accordingly
 * and inspect `candidateCount` and `candidateOverflow`. Empty rings contribute no bounds.
 * Features reaching beyond the latitude limit (H3 lattices are clamped to
 * 89 degrees, Quadbin to the Web Mercator limit) lose their polar cells and are counted in
 * `output.status.uncertainCount`. Pipeline: per-feature
 * candidate count, optional scanline enumeration (see `candidateEnumeration`), exclusive scan,
 * per-candidate test, inclusive scan of accept flags, write. No atomic append is used, so output is
 * deterministic.
 *
 * Candidates are the cells of the feature bounding box, or, for thin or sparse polygons whose box
 * is mostly empty, only the cells in the row intervals where the polygon can be (boundary trace
 * plus scanline fill, chosen per feature on the GPU). The exact test and the emitted cells and
 * their order are the same either way.
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
 * test for them (see {@link GPUCellCoverOutput.core}). H3 lattices are clamped to latitudes within 89 degrees. The
 * lattice's longitude spacing is taken at the feature's most equatorward latitude, so a tall
 * feature reaching high latitudes is oversampled near its poleward edge: cost grows as
 * 1 / cos(most equatorward latitude) times the feature's longitude span.
 */
export class GPUCellCover implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUCellCoverProps;
  /** Resolved spatial context (longitude/latitude, no metric). */
  readonly spatialContext: GPUSpatialContext;
  /** Number of features, `featureOffsets.length - 1`. */
  readonly featureCount: number;
  /** Resolved containment. */
  readonly containment: GPUCellCoverContainment;
  /** Whether the edge-slab index is built. */
  readonly edgeSlabs: boolean;
  /** Resolved `candidateEnumeration` (default `'auto'`). */
  readonly candidateEnumeration: 'bounds' | 'scanline' | 'auto';
  /** Whether the scanline enumeration stage is built. */
  readonly scanline: boolean;
  /** Resolved capacity of the scanline stage in (edge, row) slots (unused unless `scanline`). */
  readonly scanlineSlotCapacity: number;

  constructor(props: GPUCellCoverProps) {
    this.id = props.id ?? DEFAULT_ID;
    this.props = props;
    const {id} = this;
    this.spatialContext = resolveGPUSpatialContext(
      id,
      props.spatialContext ?? GPU_LONGITUDE_LATITUDE_SPATIAL_CONTEXT,
      {...GPU_LONGITUDE_LATITUDE_SPATIAL_CONTEXT_SUPPORT, edgeModels: GPU_LINEAR_EDGE_MODELS}
    );
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
    validatePackedView(props.positions, ['float32x2'], `${id} positions`);
    for (const [name, view] of [
      ['featureOffsets', props.featureOffsets],
      ['polygonOffsets', props.polygonOffsets],
      ['ringOffsets', props.ringOffsets]
    ] as const) {
      validatePackedView(view, ['uint32'], `${id} ${name}`);
      if (view.length < 1) {
        throw new Error(`${id} ${name} requires a terminal entry`);
      }
    }
    this.featureCount = props.featureOffsets.length - 1;
    this.edgeSlabs =
      (props.edgeSlabs ??
        props.candidateCapacity * props.positions.length >= EDGE_SLAB_DEFAULT_WORK) &&
      props.ringOffsets.length >= 2 &&
      props.polygonOffsets.length >= 2;
    if (props.edgeSlabEntryCapacity !== undefined) {
      validateCapacity(id, 'edgeSlabEntryCapacity', props.edgeSlabEntryCapacity);
    }
    if (props.featureIds) {
      validatePackedView(props.featureIds, ['uint32'], `${id} featureIds`);
      if (props.featureIds.length !== this.featureCount) {
        throw new Error(`${id} featureIds length must equal the feature count`);
      }
    }
    validateCapacity(id, 'candidateCapacity', props.candidateCapacity);
    if ((this.featureCount + 1) * (props.candidateCapacity + 1) >= 0xffffffff) {
      throw new Error(`${id} (featureCount + 1) * (candidateCapacity + 1) must be below 2^32`);
    }
    this.candidateEnumeration = props.candidateEnumeration ?? 'auto';
    if (!['bounds', 'scanline', 'auto'].includes(this.candidateEnumeration)) {
      throw new Error(`${id} candidateEnumeration must be 'bounds', 'scanline' or 'auto'`);
    }
    // Features without rings or vertices have no edges to trace and no candidates.
    this.scanline =
      (this.candidateEnumeration === 'scanline' ||
        (this.candidateEnumeration === 'auto' &&
          props.candidateCapacity >= CELL_COVER_SCANLINE_AUTO_MINIMUM_CANDIDATE_CAPACITY)) &&
      this.featureCount >= 1 &&
      props.positions.length >= 1 &&
      props.ringOffsets.length >= 2 &&
      props.polygonOffsets.length >= 2;
    this.scanlineSlotCapacity =
      props.scanlineSlotCapacity ??
      Math.min(
        props.candidateCapacity,
        CELL_COVER_SCANLINE_MAXIMUM_SLOT_CAPACITY,
        Math.max(16 * props.positions.length, 16384)
      );
    if (props.scanlineSlotCapacity !== undefined || this.scanline) {
      validateCapacity(
        id,
        'scanlineSlotCapacity',
        this.scanlineSlotCapacity,
        1,
        Math.min(props.candidateCapacity, CELL_COVER_SCANLINE_MAXIMUM_SLOT_CAPACITY)
      );
    }
    const {output} = props;
    const {status} = output;
    validatePackedView(output.featureIds, ['uint32'], `${id} output.featureIds`);
    validatePackedView(output.cells, ['uint32x2'], `${id} output.cells`);
    if (output.cells.length !== output.featureIds.length) {
      throw new Error(`${id} output.cells and output.featureIds must have equal lengths`);
    }
    if (output.featureIds.length < 1) {
      throw new Error(`${id} output capacity must be at least 1`);
    }
    if (output.core) {
      validatePackedView(output.core, ['uint32'], `${id} output.core`);
      if (output.core.length !== output.featureIds.length) {
        throw new Error(`${id} output.core must have the same length as output.featureIds`);
      }
    }
    for (const [name, view] of [
      ['status.count', status.count],
      ['status.requiredCount', status.requiredCount],
      ['status.overflow', status.overflow],
      ['status.candidateOverflow', status.candidateOverflow],
      ['status.candidateCount', status.candidateCount],
      ['status.invalidCount', status.invalidCount],
      ['status.uncertainCount', status.uncertainCount]
    ] as const) {
      if (view) {
        validatePackedView(view, ['uint32'], `${id} output.${name}`);
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
        status.count,
        status.requiredCount,
        status.overflow,
        status.candidateOverflow,
        status.candidateCount,
        status.invalidCount,
        status.uncertainCount
      ],
      [
        props.positions,
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
      props.positions,
      props.featureOffsets,
      props.polygonOffsets,
      props.ringOffsets,
      props.featureIds,
      output.featureIds,
      output.cells,
      output.core,
      output.status.count,
      output.status.requiredCount,
      output.status.overflow,
      output.status.candidateOverflow,
      output.status.candidateCount,
      output.status.invalidCount,
      output.status.uncertainCount
    ]);
    validateCapacity(
      id,
      'candidateCapacity',
      candidateCapacity,
      1,
      getGPUCellCoverMaximumCandidateCapacity(graph.device)
    );
    const slabEntryMaximum = getGPUCellCoverMaximumEdgeSlabEntryCount(graph.device);
    if (this.edgeSlabs) {
      validateMaximumCount(
        id,
        'positions rows + edgeSlabEntryCapacity',
        props.positions.length + (props.edgeSlabEntryCapacity ?? 1),
        slabEntryMaximum
      );
    }
    const counts = createTransientView(graph, `${id}-counts`, 'uint32', featureCount + 1);
    const starts = createTransientView(graph, `${id}-starts`, 'uint32', featureCount + 1);
    const scanlineSettings: CellCoverScanlineSettings | undefined = this.scanline
      ? {
          slotCapacity: this.scanlineSlotCapacity,
          force: this.candidateEnumeration === 'scanline'
        }
      : undefined;
    const rangesLength = getCellCoverRangesLength(
      featureCount,
      scanlineSettings,
      CELL_COVER_RANGE_STRIDE
    );
    validateMaximumCount(
      id,
      'range table words',
      rangesLength,
      getStorageBindingElementLimit(graph.device, 4)
    );
    const ranges = createTransientView(graph, `${id}-ranges`, 'uint32', rangesLength);
    const flags = createTransientView(graph, `${id}-flags`, 'uint32', candidateCapacity);
    const accepted = createTransientView(graph, `${id}-accepted`, 'uint32', candidateCapacity);
    const candidateCells = createTransientView(
      graph,
      `${id}-candidate-cells`,
      'uint32x2',
      candidateCapacity
    );
    const total = createTransientView(graph, `${id}-total`, 'uint32', 1);
    // Edge-slab index (owner = feature): per-feature slab counts for features with enough
    // candidates, then a counting sort of the edges into slabs, all vertex-parallel.
    const edgeSlabIndex = this.edgeSlabs
      ? getPolygonEdgeSlabNodes<Parameters>(graph, {
          id,
          operation: 'GPUCellCover',
          owner: 'feature',
          positions: props.positions,
          featureOffsets: props.featureOffsets,
          polygonOffsets: props.polygonOffsets,
          ringOffsets: props.ringOffsets,
          entryCapacity:
            props.edgeSlabEntryCapacity ??
            Math.max(
              1,
              Math.min(
                Math.max(4 * props.positions.length, 1024),
                slabEntryMaximum - props.positions.length
              )
            ),
          gate: {counts, minimum: CELL_COVER_MINIMUM_SLAB_CANDIDATES}
        })
      : undefined;
    const context: CellCoverKernelContext = {
      id,
      family: props.family,
      resolution: props.resolution,
      containment: this.containment,
      computeCore: Boolean(output.core),
      featureCount,
      candidateCapacity,
      edgeSlabs: edgeSlabIndex?.slabs,
      scanline: scanlineSettings,
      positions: props.positions,
      featureOffsets: props.featureOffsets,
      polygonOffsets: props.polygonOffsets,
      ringOffsets: props.ringOffsets
    };
    const statusNodes = (['invalidCount', 'uncertainCount'] as const).flatMap(name => {
      const view = output.status[name];
      return view
        ? [
            createFillNode<Parameters>(graph, {
              id: `${id}-clear-${name}`,
              operation: 'GPUCellCover',
              view,
              type: 'u32',
              value: '0u'
            })
          ]
        : [];
    });
    const status = {
      invalidCount: output.status.invalidCount,
      uncertainCount: output.status.uncertainCount
    };
    return [
      ...statusNodes,
      createCoverCountNode<Parameters>(graph, context, counts, ranges, status),
      // The scanline stage replaces the bounding-box counts of the features it takes over, so it
      // runs before the candidate scan and before the slab gate that reads the counts.
      ...(scanlineSettings
        ? getCellCoverScanlineNodes<Parameters>(
            graph,
            {...context, scanline: scanlineSettings},
            {counts, ranges}
          )
        : []),
      ...new GPUScan({
        id: `${id}-candidate-scan`,
        input: counts,
        output: starts,
        mode: 'exclusive'
      }).getCommandNodes(graph),
      ...(edgeSlabIndex?.nodes ?? []),
      createCoverTestNode<Parameters>(graph, context, {
        ranges,
        starts,
        flags,
        candidateCells
      }),
      // Why not `GPUFlagOffsets`: it yields exclusive offsets, while the write kernels index the
      // inclusive scan as `accepted[index - 1]`, and `finalize` fuses the overflow flag with the total.
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
        candidateOverflow: output.status.candidateOverflow,
        candidateCount: output.status.candidateCount
      }),
      createPublishNode<Parameters>(graph, {
        id: `${id}-publish`,
        operation: 'GPUCellCover',
        requiredCount: total,
        output: {
          ids: output.featureIds,
          status: {
            count: output.status.count,
            overflow: output.status.overflow,
            requiredCount: output.status.requiredCount
          }
        },
        overflowSources: []
      })
    ];
  }
}
