// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {createFillNode, createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {createSegmentSumNode, getSortedSegmentSumNodes} from '../../utils/sorted-segment-sums';
import {GET_NAN_WGSL} from '../../utils/wgsl-common';
import {
  resolveGPUSpatialContext,
  type GPUSpatialContext,
  type GPUSpatialContextSupport
} from '../contracts/spatial-context';
import {
  createSphericalCellWeightsNode,
  getRasterSphereRadius,
  validateGeographicRaster
} from './spherical-cell-area';
import {getZoneAnchorNodes} from './zone-anchor-nodes';
import {getKeyGroupNodes, KEY_PAIR_INVALID} from '../spatial-weights/key-pair-grouping';
import {getBoundedKeyPairSortNodes} from './bounded-key-pair-sort';
import {
  type GPUSpatialWeights,
  validateGPUSpatialWeights
} from '../spatial-weights/spatial-weights';

const OPERATION = 'GPUArealInterpolation';
/** Planar coordinates, or longitude/latitude rasters weighted by spherical cell area. */
const SUPPORT: GPUSpatialContextSupport = Object.freeze({
  coordinateSpaces: Object.freeze(['planar', 'longitude-latitude'] as const),
  metrics: Object.freeze(['none', 'native', 'great-circle'] as const)
});
const INVALID = `${KEY_PAIR_INVALID}u`;
/** Consecutive cells folded into runs by one zone-count thread. */
const ZONE_COUNT_CHUNK = 8;

/** Normalization of the area-share weights. */
export type GPUArealInterpolationKind = 'extensive' | 'intensive';

/** Which area a share is divided by. */
export type GPUArealInterpolationDenominator = 'zone' | 'overlap';

/** Denominator per variable kind. Omitted kinds use their default. */
export type GPUArealInterpolationDenominators = {
  /** Default `'zone'` (tobler): mass that falls outside the target system is lost and reported. */
  extensive?: GPUArealInterpolationDenominator;
  /** Default `'overlap'`: a partly covered target is the mean of what covers it. */
  intensive?: GPUArealInterpolationDenominator;
};

/** Per-target category shares of a categorical source variable. */
export type GPUArealCategories = {
  /** Category per source, in `[0, categoryCount)`; other values contribute to no category. */
  sourceCategories: GraphDataView<'uint32'>;
  /** Number of categories `K`. */
  categoryCount: number;
  /**
   * Caller-owned `targetCount * K` shares, row-major (`output[t * K + k]`). `share(t, k)` is the
   * overlap area of target `t` with sources of category `k`, divided by the area of `t` covered by
   * any source, so the shares of a covered target sum to 1. Uncovered targets output 0.
   */
  output: GraphDataView<'float32'>;
};

/**
 * Properties for {@link GPUArealInterpolation}.
 *
 * Compile-time: `sourceCount`, `targetCount`, `denominator`, `mode`, `spatialContext`, `width`, view lengths
 * and the pair capacity (`weights.neighbors.length`). Per-frame: the contents of the zone rasters,
 * `cellWeights`, source values and categories.
 */
export type GPUArealInterpolationProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'areal-interpolation'`. */
  id?: string;
  /**
   * Source-zone raster: one cell per entry, the source feature row of each cell or any value
   * `>= sourceCount` (such as `GPU_POLYGON_RASTERIZATION_NO_ZONE`) for no source. Typically the
   * `zones` output of a `GPUPolygonRasterization` over the source polygons.
   */
  sourceZones: GraphDataView<'uint32'>;
  /**
   * Target-zone raster on the same grid as `sourceZones` (same extent, width, height; same length),
   * from a `GPUPolygonRasterization` of the target polygons, or any zone raster such as H3 or
   * hexagon cells rasterized to the grid.
   */
  targetZones: GraphDataView<'uint32'>;
  /** Number of source zones `S`. */
  sourceCount: number;
  /** Number of target zones `T`; also the row count of the output weights. */
  targetCount: number;
  /**
   * Optional dasymetric raster with one non-negative weight per cell (for example built-up
   * fraction or population density). Every area becomes a sum of these weights instead of a cell
   * count. Cells with a non-finite or non-positive weight carry no area and never create a pair.
   */
  cellWeights?: GraphDataView<'float32'>;
  /**
   * Which area a share is divided by, as one value for both kinds or per kind. Default
   * `{extensive: 'zone', intensive: 'overlap'}`.
   *
   * `'zone'` divides by the total area of the source (extensive) or target (intensive), including
   * parts that do not overlap the other system. `'overlap'` divides by the part of the zone that
   * overlaps the other system, so extensive shares of a source sum to 1 over its targets and
   * intensive shares of a target sum to 1 over its sources.
   *
   * Why the kinds differ. An extensive variable (counts) keeps tobler's `'zone'`: the share of a
   * source that falls outside every target is not invented onto the targets, and the lost fraction
   * is reported in `sourceTransferredFractions`. An intensive variable (rate, mean) must not be
   * diluted when a target is only partly covered by sources, so it defaults to `'overlap'` and is
   * the mean of what covers the target. Earlier versions applied `'zone'` to both.
   */
  denominator?: GPUArealInterpolationDenominator | GPUArealInterpolationDenominators;
  /**
   * Coordinate space of the zone rasters. Omitted means native planar: every area is a cell count
   * (or a sum of `cellWeights`). A `longitude-latitude` context (metric `'none'` or
   * `'great-circle'`) means the rasters are longitude/latitude grids described by `extent` and
   * `width`; each cell is then weighted by its spherical area in square meters,
   * `R^2 dLon (sin(latTop) - sin(latBottom))` with `R` = `context.sphereRadius` or the WGS84
   * authalic radius, times `cellWeights` when given, so a latitude band is not over-counted.
   * Cell membership of the zones is by cell center in degrees. Other spaces throw.
   */
  spatialContext?: GPUSpatialContext;
  /**
   * `longitude-latitude` only: the per-frame raster placement `[originLongitude, originLatitude,
   * cellWidth, cellHeight]` in degrees (the same view as `GPUPolygonRasterization.extent`, row 0
   * at the smallest latitude). Required for, and only valid with, a longitude/latitude context.
   */
  extent?: GraphDataView<'float32'>;
  /** `longitude-latitude` only: raster columns. Compile-time; must divide the cell count. */
  width?: number;
  /**
   * Optional per-source fallback cell (`sourceCount` rows of cell indices; out of range means
   * none). A source zone that rasterized to no cell claims its anchor cell, for example the cell
   * containing the zone's label point, so its mass is not lost. The claim happens in a transient
   * copy of `sourceZones` and succeeds only if the cell belongs to no source or to a source that
   * keeps another cell; shared anchors go to the lowest source row. Sources that still have no
   * cell are counted in `invalidCount`.
   */
  sourceAnchorCells?: GraphDataView<'uint32'>;
  /**
   * Optional output, `sourceCount` rows: the whole raster area `A_s` of each source (cell count,
   * or the sum of the cell weights; square meters on a longitude/latitude raster, after anchoring).
   * 0 when the source rasterized to no cell.
   */
  sourceAreas?: GraphDataView<'float32'>;
  /**
   * Optional output, `sourceCount` rows: the conservation fraction of each source,
   * `sum_t a_st / A_s` with the extensive denominator, where `a_st` sums over targets (before
   * `weights` slot clamping). With `'zone'`, 1 means the source lies entirely inside the target
   * system and its extensive mass is fully transferred, and `1 - f` is the lost fraction. With
   * `'overlap'` it is 1 wherever the source overlaps any target and 0 where it does not. NaN when
   * the source has no raster area (see `sourceAreas`).
   */
  sourceTransferredFractions?: GraphDataView<'float32'>;
  /**
   * Optional one-row output: the number of sources with no raster area after anchoring. Their mass
   * is lost; this is the `invalidCount` of the status vocabulary.
   */
  invalidCount?: GraphDataView<'uint32'>;
  /**
   * Which normalization is written to `weights.weights`. Defaults to `'extensive'`. The other one
   * is written to `alternateWeights` when given.
   */
  mode?: GPUArealInterpolationKind;
  /**
   * Caller-owned output CSR with `targetCount + 1` offsets: row `t` (target) lists its overlapping
   * sources `s` as `neighbors`, ascending, with the area-share weight of `mode`. The slot capacity
   * is `neighbors.length` and must hold the number of distinct (source, target) pairs that share
   * at least one cell. `distances` is not written. This is a cross (target to source) weights
   * matrix: its neighbors index sources, not rows.
   */
  weights: GPUSpatialWeights;
  /** Optional caller-owned weights (one per slot) holding the normalization that is not `mode`. */
  alternateWeights?: GraphDataView<'float32'>;
  /**
   * Optional caller-owned overlap area per slot (cell count, or the sum of `cellWeights`; square
   * meters on a longitude/latitude raster).
   */
  areas?: GraphDataView<'float32'>;
  /** Caller-owned one-row flag: 1 when the pairs did not fit the slot capacity, else 0. */
  overflow: GraphDataView<'uint32'>;
  /** Optional caller-owned one-row unclamped distinct pair count, for sizing the capacity. */
  totalPairs?: GraphDataView<'uint32'>;
  /** Optional categorical variable to turn into per-target category shares. */
  categories?: GPUArealCategories;
  /**
   * Compile-time. Without `cellWeights` every cell weighs 1, so pair areas are the lengths of the
   * sorted pair runs and zone areas are the zone cell counts; the fast path computes exactly those
   * and skips the mask gather, the per-pair segmented sum and the per-zone sort, scan, gather and
   * segmented sums. Defaults to `true`; results are identical. Set `false` to force the generic
   * weighted path (for A/B timing or tests). Ignored when `cellWeights` is given.
   */
  unweightedFastPath?: boolean;
};

/**
 * Area-weighted transfer between two zone systems (the tobler `area_interpolate` and CARTO
 * `ENRICH_POLYGONS` workload) on a common fine raster.
 *
 * Definition. Both zone systems are rasterized to the same grid (use `GPUPolygonRasterization` for
 * polygons; H3 or hexagon zones can be rasterized the same way). A cell with source `s`, target
 * `t` and weight `m` (1, or `cellWeights`) adds `m` to the overlap area `a_st`. With source totals
 * `A_s` and target totals `B_t`:
 * - extensive (counts, totals; tobler `extensive`): `w_ts = a_st / A_s`, so the transferred value
 *   of target `t` is `sum_s w_ts x_s` and the mass of source `s` is divided between its targets.
 * - intensive (rates, densities, means; tobler `intensive`): `w_ts = a_st / B_t`, so the value of
 *   target `t` is the area-weighted mean `sum_s w_ts x_s` of its sources.
 * - categorical: `share(t, k) = sum_{s in k} a_st / sum_s a_st` for each target, in `categories`.
 * `A_s` and `B_t` are the whole zone areas (`denominator: 'zone'`, tobler) or only their overlap
 * with the other system (`'overlap'`, which conserves mass over the shared extent). The default is
 * `'zone'` for extensive and `'overlap'` for intensive variables (see `denominator`).
 *
 * Conservation diagnostics. `sourceAreas` and `sourceTransferredFractions` report per source how
 * much of its raster area reached the targets, and `invalidCount` counts sources that rasterized
 * to no cell (a source smaller than a cell, or a polygon the cell centers all miss): their mass is
 * lost unless `sourceAnchorCells` gives them a fallback cell. Dropped mass is therefore reported,
 * never silent.
 *
 * Coordinates. With a `longitude-latitude` `spatialContext` the areas are spherical cell areas in
 * square meters, so the weights are ground-area shares on a geographic raster (a cell at 60 degrees
 * latitude weighs half of one at the equator). Without it every cell weighs 1 (or `cellWeights`),
 * which is only an equal-area share on a planar raster.
 *
 * Accuracy. The result is exact only for the rasterized zones, never for the polygon
 * intersection: each cell is assigned by its center point, so every area is a count of cell
 * centers and the error shrinks with the cell size (about one cell width along every zone
 * boundary). Sliver overlaps smaller than a cell can be missed entirely. Areas are float32 sums, so
 * counts are exact up to 2^24 cells per zone. Choose the raster so cells are small against the
 * smallest zone of either system, and use `cellWeights` for dasymetric refinement.
 *
 * Algorithm (deterministic): per cell `(t, s)` pair keys, two stable radix sorts of the cells
 * (`getBoundedKeyPairSortNodes`, radix passes limited to the zone-id bit widths), run detection, a fixed-order segmented sum of the cell weights per
 * pair run (one workgroup per pair), a lower-bound search for the CSR offsets, and the same sorted
 * segmented reduction per zone for `A_s` and `B_t`. Cost is a few radix sorts over the cell count;
 * rows are ordered by target and slots by source, so the CSR invariants hold exactly. Tied cells
 * keep ascending cell order.
 *
 * Outputs: the CSR `weights`, optionally the raw `areas` per slot (explain column),
 * `alternateWeights` (the other normalization) and categorical shares. Transfer variables with
 * `GPUSpatialLag` (`sourceCount` = `sourceCount`, optional `columnCount` for several variables at
 * once) over `weights`, using `alternateWeights` as its `weights.weights` for the other kind. When the
 * pairs exceed the slot capacity, `overflow` is 1, the offsets are clamped to the capacity and the
 * dropped pairs are the ones of the highest targets.
 *
 * Unweighted fast path (no `cellWeights`, `unweightedFastPath` not `false`): every cell has weight
 * 1, so the pair areas are read off the run lengths and the zone totals off the zone counts. This
 * removes two sorts, two scans and five reduction kernels of the generic path with identical
 * results (counts are exact integers in float32 up to 2^24).
 */
export class GPUArealInterpolation implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUArealInterpolationProps;
  /** Resolved coordinate context: native planar unless a longitude/latitude raster was declared. */
  readonly spatialContext: GPUSpatialContext;
  /** Resolved denominator of extensive variables. */
  readonly extensiveDenominator: GPUArealInterpolationDenominator;
  /** Resolved denominator of intensive variables. */
  readonly intensiveDenominator: GPUArealInterpolationDenominator;

  constructor(props: GPUArealInterpolationProps) {
    const id = props.id ?? 'areal-interpolation';
    this.id = id;
    this.props = props;
    this.spatialContext = resolveGPUSpatialContext(id, props.spatialContext, SUPPORT);
    const geographic = this.spatialContext.coordinateSpace === 'longitude-latitude';
    if (geographic) {
      validateGeographicRaster(id, props.extent, props.width, props.sourceZones.length);
    } else if (props.extent || props.width !== undefined) {
      throw new Error(`${id} extent and width need a longitude-latitude spatialContext`);
    }
    const {sourceCount, targetCount} = props;
    for (const [name, value] of [
      ['sourceCount', sourceCount],
      ['targetCount', targetCount]
    ] as const) {
      if (!Number.isSafeInteger(value) || value < 1 || value >= 0xffffffff) {
        throw new Error(`${id} ${name} must be a positive integer`);
      }
    }
    validatePackedUint32View(props.sourceZones, `${id} sourceZones`);
    validatePackedUint32View(props.targetZones, `${id} targetZones`);
    if (props.sourceZones.length < 1 || props.targetZones.length !== props.sourceZones.length) {
      throw new Error(`${id} sourceZones and targetZones must have the same nonzero length`);
    }
    if (props.cellWeights) {
      validatePackedView(props.cellWeights, ['float32'], `${id} cellWeights`);
      if (props.cellWeights.length !== props.sourceZones.length) {
        throw new Error(`${id} cellWeights length must equal the raster cell count`);
      }
    }
    const denominators = typeof props.denominator === 'object' ? props.denominator : undefined;
    const single = typeof props.denominator === 'string' ? props.denominator : undefined;
    this.extensiveDenominator = denominators?.extensive ?? single ?? 'zone';
    this.intensiveDenominator = denominators?.intensive ?? single ?? 'overlap';
    for (const denominator of [this.extensiveDenominator, this.intensiveDenominator]) {
      if (denominator !== 'zone' && denominator !== 'overlap') {
        throw new Error(`${id} denominator must be 'zone' or 'overlap'`);
      }
    }
    if (props.mode !== undefined && props.mode !== 'extensive' && props.mode !== 'intensive') {
      throw new Error(`${id} mode must be 'extensive' or 'intensive'`);
    }
    if (props.sourceAnchorCells) {
      validatePackedUint32View(props.sourceAnchorCells, `${id} sourceAnchorCells`);
      if (props.sourceAnchorCells.length !== sourceCount) {
        throw new Error(`${id} sourceAnchorCells must have sourceCount rows`);
      }
    }
    for (const [name, view] of [
      ['sourceAreas', props.sourceAreas],
      ['sourceTransferredFractions', props.sourceTransferredFractions]
    ] as const) {
      if (view) {
        validatePackedView(view, ['float32'], `${id} ${name}`);
        if (view.length !== sourceCount) {
          throw new Error(`${id} ${name} must have sourceCount rows`);
        }
      }
    }
    const rows = validateGPUSpatialWeights(id, props.weights);
    if (rows !== targetCount) {
      throw new Error(`${id} weights.offsets length must equal targetCount + 1`);
    }
    const capacity = props.weights.neighbors.length;
    for (const [name, view] of [
      ['alternateWeights', props.alternateWeights],
      ['areas', props.areas]
    ] as const) {
      if (view) {
        validatePackedView(view, ['float32'], `${id} ${name}`);
        if (view.length !== capacity) {
          throw new Error(`${id} ${name} length must equal weights.neighbors length`);
        }
      }
    }
    for (const [name, view] of [
      ['overflow', props.overflow],
      ['totalPairs', props.totalPairs],
      ['invalidCount', props.invalidCount]
    ] as const) {
      if (view) {
        validatePackedUint32View(view, `${id} ${name}`);
        if (view.length < 1) {
          throw new Error(`${id} ${name} must hold one uint32`);
        }
      }
    }
    if (props.categories) {
      const {sourceCategories, categoryCount, output} = props.categories;
      validatePackedUint32View(sourceCategories, `${id} categories.sourceCategories`);
      validatePackedView(output, ['float32'], `${id} categories.output`);
      if (!Number.isSafeInteger(categoryCount) || categoryCount < 1) {
        throw new Error(`${id} categories.categoryCount must be a positive integer`);
      }
      if (sourceCategories.length !== sourceCount) {
        throw new Error(`${id} categories.sourceCategories must have sourceCount rows`);
      }
      if (output.length !== targetCount * categoryCount) {
        throw new Error(`${id} categories.output must have targetCount * categoryCount rows`);
      }
    }
    validateGraphOutputsDisjointFromInputs(id, this.getOutputViews(), this.getInputViews());
  }

  private getInputViews(): (GraphDataView | undefined)[] {
    const {props} = this;
    return [
      props.sourceZones,
      props.targetZones,
      props.cellWeights,
      props.extent,
      props.sourceAnchorCells,
      props.categories?.sourceCategories
    ];
  }

  private getOutputViews(): (GraphDataView | undefined)[] {
    const {props} = this;
    return [
      props.weights.offsets,
      props.weights.neighbors,
      props.weights.weights,
      props.weights.distances,
      props.alternateWeights,
      props.areas,
      props.overflow,
      props.totalPairs,
      props.invalidCount,
      props.sourceAreas,
      props.sourceTransferredFractions,
      props.categories?.output
    ];
  }

  /** Returns the pair, total, weight and category nodes in dependency order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    validateGraphViewsBelongToGraph(id, graph, [...this.getInputViews(), ...this.getOutputViews()]);
    const {targetZones, sourceCount, targetCount, weights} = props;
    const cellCount = props.sourceZones.length;
    const capacity = weights.neighbors.length;
    const slotCount = Math.min(capacity, cellCount);
    const mode = props.mode ?? 'extensive';
    const geographic = this.spatialContext.coordinateSpace === 'longitude-latitude';
    const extensiveOverlap = this.extensiveDenominator === 'overlap';
    const intensiveOverlap = this.intensiveDenominator === 'overlap';
    const needSourceKey = extensiveOverlap || Boolean(props.sourceTransferredFractions);
    const needTargetKey = intensiveOverlap;
    const needOverlapKeys = needSourceKey || needTargetKey;

    const nodes: GPUCommandNode<Parameters>[] = [];

    // Source zones that rasterized to no cell claim their anchor cell in a transient copy.
    let sourceZones = props.sourceZones;
    if (props.sourceAnchorCells) {
      const anchored = getZoneAnchorNodes(graph, {
        id,
        operation: OPERATION,
        zones: props.sourceZones,
        zoneCount: sourceCount,
        anchorCells: props.sourceAnchorCells
      });
      sourceZones = anchored.zones;
      nodes.push(...anchored.nodes);
    }

    // Longitude/latitude rasters: the effective cell weight includes the spherical cell area.
    let cellWeights = props.cellWeights;
    if (geographic) {
      const areaWeights = createTransientView(graph, `${id}-area-weights`, 'float32', cellCount);
      nodes.push(
        createSphericalCellWeightsNode(graph, {
          id: `${id}-area-weights`,
          operation: OPERATION,
          cellCount,
          width: props.width!,
          radius: getRasterSphereRadius(this.spatialContext),
          extent: props.extent!,
          cellWeights,
          output: areaWeights
        })
      );
      cellWeights = areaWeights;
    }

    const keyHigh = createTransientView(graph, `${id}-key-high`, 'uint32', cellCount);
    const keyLow = createTransientView(graph, `${id}-key-low`, 'uint32', cellCount);
    const fastPath = !cellWeights && props.unweightedFastPath !== false;
    const cellMask = fastPath
      ? undefined
      : createTransientView(graph, `${id}-cell-mask`, 'float32', cellCount);
    const overlapSourceKey = needSourceKey
      ? createTransientView(graph, `${id}-source-key`, 'uint32', cellCount)
      : undefined;
    const overlapTargetKey = needTargetKey
      ? createTransientView(graph, `${id}-target-key`, 'uint32', cellCount)
      : undefined;

    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-cell-keys`,
        operation: OPERATION,
        variant: `cell-keys-${needSourceKey ? 's' : ''}${needTargetKey ? 't' : ''}${needOverlapKeys ? 'overlap' : 'zone'}${cellWeights ? '-masked' : ''}${fastPath ? '-fast' : ''}`,
        bindings: [
          {name: 'sourceZones', view: sourceZones, type: 'u32', access: 'read'},
          {name: 'targetZones', view: targetZones, type: 'u32', access: 'read'},
          ...(cellWeights
            ? [
                {
                  name: 'cellWeights',
                  view: cellWeights,
                  type: 'f32' as const,
                  access: 'read' as const
                }
              ]
            : []),
          {name: 'keyHigh', view: keyHigh, type: 'u32', access: 'read_write'},
          {name: 'keyLow', view: keyLow, type: 'u32', access: 'read_write'},
          ...(cellMask
            ? [
                {
                  name: 'cellMask',
                  view: cellMask,
                  type: 'f32' as const,
                  access: 'read_write' as const
                }
              ]
            : []),
          ...(overlapSourceKey
            ? [
                {
                  name: 'sourceKey',
                  view: overlapSourceKey,
                  type: 'u32' as const,
                  access: 'read_write' as const
                }
              ]
            : []),
          ...(overlapTargetKey
            ? [
                {
                  name: 'targetKey',
                  view: overlapTargetKey,
                  type: 'u32' as const,
                  access: 'read_write' as const
                }
              ]
            : [])
        ],
        invocationCount: cellCount,
        declarations: `const SOURCE_COUNT: u32 = ${sourceCount}u;
const TARGET_COUNT: u32 = ${targetCount}u;`,
        body: `let source = sourceZones[sourceZonesOffset + index];
  let targetRow = targetZones[targetZonesOffset + index];
  ${
    cellWeights
      ? `var weight = cellWeights[cellWeightsOffset + index];
  // Non-finite (NaN or infinity) and non-positive weights carry no area.
  if ((bitcast<u32>(weight) & 0x7f800000u) == 0x7f800000u || !(weight > 0.0)) {
    weight = 0.0;
  }`
      : 'let weight = 1.0;'
  }
  let valid = source < SOURCE_COUNT && targetRow < TARGET_COUNT && weight > 0.0;
  keyHigh[keyHighOffset + index] = select(${INVALID}, targetRow, valid);
  keyLow[keyLowOffset + index] = select(${INVALID}, source, valid);
  ${cellMask ? 'cellMask[cellMaskOffset + index] = weight;' : ''}
  ${overlapSourceKey ? `sourceKey[sourceKeyOffset + index] = select(${INVALID}, source, valid);` : ''}
  ${overlapTargetKey ? `targetKey[targetKeyOffset + index] = select(${INVALID}, targetRow, valid);` : ''}`
      })
    );

    // Pair runs, one per distinct (target, source), in ascending (target, source) order.
    // Zone ids need only log2(zoneCount) radix bits per half; invalid cells hold all ones.
    const sort = getBoundedKeyPairSortNodes(
      graph,
      `${id}-cell`,
      OPERATION,
      cellCount,
      keyHigh,
      keyLow,
      {lowKeyLimit: sourceCount, highKeyLimit: targetCount}
    );
    const groups = getKeyGroupNodes(
      graph,
      `${id}-cell`,
      OPERATION,
      cellCount,
      keyHigh,
      keyLow,
      sort.sortedItems
    );
    nodes.push(...sort.nodes, ...groups.nodes);

    const areas = props.areas ?? createTransientView(graph, `${id}-areas`, 'float32', capacity);
    if (cellMask) {
      const sortedMask = createTransientView(graph, `${id}-sorted-mask`, 'float32', cellCount);
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-gather-mask`,
          operation: OPERATION,
          variant: 'gather-mask',
          bindings: [
            {name: 'sortedItems', view: sort.sortedItems, type: 'u32', access: 'read'},
            {name: 'cellMask', view: cellMask, type: 'f32', access: 'read'},
            {name: 'sortedMask', view: sortedMask, type: 'f32', access: 'read_write'}
          ],
          invocationCount: cellCount,
          body: 'sortedMask[sortedMaskOffset + index] = cellMask[cellMaskOffset + sortedItems[sortedItemsOffset + index]];'
        }),
        createSegmentSumNode<Parameters>(graph, {
          id: `${id}-pair-areas`,
          operation: OPERATION,
          segmentCount: slotCount,
          input: sortedMask,
          segmentOffsets: groups.groupStarts,
          output: areas
          // Not within-input provable: `groupStarts` slots past the last run are never written, and
          // `areas` has `capacity` rows, which can exceed `slotCount`.
        })
      );
    }

    // Whole-zone (or overlap-only) areas of every source and target. Weighted: the sorted
    // reduction. Unweighted fast path: the zone cell counts are the areas.
    const zoneTotals = (
      name: string,
      keys: GraphDataView<'uint32'>,
      count: number
    ): GraphDataView<'float32'> => {
      const counts = createTransientView(graph, `${id}-${name}-counts`, 'uint32', count);
      const totals = createTransientView(graph, `${id}-${name}-totals`, 'float32', count);
      nodes.push(
        createFillNode<Parameters>(graph, {
          id: `${id}-${name}-counts-clear`,
          operation: OPERATION,
          view: counts,
          type: 'u32',
          value: '0u'
        }),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-${name}-counts`,
          operation: OPERATION,
          variant: 'zone-counts',
          bindings: [
            {name: 'keys', view: keys, type: 'u32', access: 'read'},
            {name: 'counts', view: counts, type: 'atomic<u32>', access: 'read_write'}
          ],
          // Zone rasters are spatially coherent, so a thread folds a chunk of consecutive cells
          // into runs of equal keys and issues one atomic per run, not per cell. Integer sums, so
          // the totals stay exact and independent of thread order.
          invocationCount: Math.ceil(cellCount / ZONE_COUNT_CHUNK),
          declarations: `const ZONE_COUNT: u32 = ${count}u;
const CELL_COUNT: u32 = ${cellCount}u;
const CHUNK: u32 = ${ZONE_COUNT_CHUNK}u;`,
          body: `let end = min((index + 1u) * CHUNK, CELL_COUNT);
  var runKey = 0xffffffffu;
  var runLength = 0u;
  for (var cell = index * CHUNK; cell < end; cell++) {
    let key = keys[keysOffset + cell];
    if (key != runKey) {
      if (runKey < ZONE_COUNT) {
        atomicAdd(&counts[countsOffset + runKey], runLength);
      }
      runKey = key;
      runLength = 0u;
    }
    runLength++;
  }
  if (runKey < ZONE_COUNT) {
    atomicAdd(&counts[countsOffset + runKey], runLength);
  }`
        })
      );
      if (cellMask) {
        nodes.push(
          ...getSortedSegmentSumNodes(graph, {
            id: `${id}-${name}`,
            operation: OPERATION,
            segmentCount: count,
            segmentKeys: keys,
            sumContributions: cellMask,
            sums: totals
          })
        );
      } else {
        nodes.push(
          createWGSLKernelNode<Parameters>(graph, {
            id: `${id}-${name}-totals`,
            operation: OPERATION,
            variant: 'zone-totals-from-counts',
            bindings: [
              {name: 'counts', view: counts, type: 'u32', access: 'read'},
              {name: 'totals', view: totals, type: 'f32', access: 'read_write'}
            ],
            invocationCount: count,
            body: 'totals[totalsOffset + index] = f32(counts[countsOffset + index]);'
          })
        );
      }
      return totals;
    };
    // Each kind divides by its own denominator: extensive by source totals, intensive by target.
    const sourceTotals = zoneTotals(
      'source',
      extensiveOverlap ? overlapSourceKey! : sourceZones,
      sourceCount
    );
    const targetTotals = zoneTotals(
      'target',
      intensiveOverlap ? overlapTargetKey! : targetZones,
      targetCount
    );

    // Conservation diagnostics per source.
    const {sourceAreas, sourceTransferredFractions, invalidCount} = props;
    if (sourceAreas || sourceTransferredFractions || invalidCount) {
      const zoneAreas = extensiveOverlap
        ? zoneTotals('source-zone', sourceZones, sourceCount)
        : sourceTotals;
      const overlapAreas = sourceTransferredFractions
        ? extensiveOverlap
          ? sourceTotals
          : zoneTotals('source-overlap', overlapSourceKey!, sourceCount)
        : undefined;
      if (invalidCount) {
        nodes.push(
          createFillNode<Parameters>(graph, {
            id: `${id}-invalid-count-clear`,
            operation: OPERATION,
            view: invalidCount,
            type: 'u32',
            value: '0u',
            componentCount: 1
          })
        );
      }
      // The same total can serve several roles (whole-zone area, denominator, overlap area); bind
      // each distinct view once and refer to it by the first role's name.
      const alias = {
        zoneAreas: 'zoneAreas',
        denominatorAreas: 'denominatorAreas',
        overlapAreas: 'overlapAreas'
      };
      const roles: [keyof typeof alias, GraphDataView<'float32'> | undefined][] = [
        ['zoneAreas', zoneAreas],
        ['denominatorAreas', sourceTotals],
        ['overlapAreas', overlapAreas]
      ];
      const bound = new Map<GraphDataView<'float32'>, string>();
      for (const [role, view] of roles) {
        if (view) {
          alias[role] = bound.get(view) ?? role;
          bound.set(view, alias[role]);
        }
      }
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-source-diagnostics`,
          operation: OPERATION,
          variant: 'source-diagnostics',
          bindings: [
            ...[...bound].map(([view, name]) => ({
              name,
              view,
              type: 'f32' as const,
              access: 'read' as const
            })),
            ...(sourceAreas
              ? [
                  {
                    name: 'sourceAreas',
                    view: sourceAreas,
                    type: 'f32' as const,
                    access: 'read_write' as const
                  }
                ]
              : []),
            ...(sourceTransferredFractions
              ? [
                  {
                    name: 'fractions',
                    view: sourceTransferredFractions,
                    type: 'u32' as const,
                    access: 'read_write' as const
                  }
                ]
              : []),
            ...(invalidCount
              ? [
                  {
                    name: 'invalidCount',
                    view: invalidCount,
                    type: 'atomic<u32>' as const,
                    access: 'read_write' as const
                  }
                ]
              : [])
          ],
          invocationCount: sourceCount,
          declarations: GET_NAN_WGSL,
          body: `let area = ${alias.zoneAreas}[${alias.zoneAreas}Offset + index];
  let hasArea = area > 0.0;
  ${sourceAreas ? 'sourceAreas[sourceAreasOffset + index] = select(0.0, area, hasArea);' : ''}
  ${
    sourceTransferredFractions
      ? `let denominator = ${alias.denominatorAreas}[${alias.denominatorAreas}Offset + index];
  var fraction = bitcast<u32>(getNaN());
  if (hasArea) {
    fraction = bitcast<u32>(select(0.0, ${alias.overlapAreas}[${alias.overlapAreas}Offset + index] / denominator, denominator > 0.0));
  }
  fractions[fractionsOffset + index] = fraction;`
      : ''
  }
  ${invalidCount ? 'if (!hasArea) { atomicAdd(&invalidCount[invalidCountOffset], 1u); }' : ''}`
        })
      );
    }

    const pairTarget = createTransientView(graph, `${id}-pair-target`, 'uint32', capacity);
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-pair-keys`,
        operation: OPERATION,
        variant: cellMask ? 'pair-keys' : 'pair-keys-run-lengths',
        bindings: [
          {name: 'sortedItems', view: sort.sortedItems, type: 'u32', access: 'read'},
          {name: 'groupStarts', view: groups.groupStarts, type: 'u32', access: 'read'},
          {name: 'groupIndex', view: groups.groupIndex, type: 'u32', access: 'read'},
          {name: 'keyHigh', view: keyHigh, type: 'u32', access: 'read'},
          {name: 'keyLow', view: keyLow, type: 'u32', access: 'read'},
          {name: 'neighbors', view: weights.neighbors, type: 'u32', access: 'read_write'},
          {name: 'pairTarget', view: pairTarget, type: 'u32', access: 'read_write'},
          ...(cellMask
            ? []
            : [{name: 'areas', view: areas, type: 'f32' as const, access: 'read_write' as const}])
        ],
        invocationCount: slotCount,
        declarations: `const CELL_COUNT: u32 = ${cellCount}u;`,
        body: `let groupCount = groupIndex[groupIndexOffset + CELL_COUNT - 1u];
  var neighbor = ${INVALID};
  var targetRow = ${INVALID};
  if (index < groupCount) {
    let item = sortedItems[sortedItemsOffset + groupStarts[groupStartsOffset + index]];
    neighbor = keyLow[keyLowOffset + item];
    targetRow = keyHigh[keyHighOffset + item];
  }
  neighbors[neighborsOffset + index] = neighbor;
  pairTarget[pairTargetOffset + index] = targetRow;${
    cellMask
      ? ''
      : `
  // Unit cell weights: the pair area is the run length.
  var area = 0.0;
  if (index < groupCount) {
    area = f32(groupStarts[groupStartsOffset + index + 1u] - groupStarts[groupStartsOffset + index]);
  }
  areas[areasOffset + index] = area;`
  }`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-offsets`,
        operation: OPERATION,
        variant: 'offsets',
        bindings: [
          {name: 'sortedItems', view: sort.sortedItems, type: 'u32', access: 'read'},
          {name: 'groupStarts', view: groups.groupStarts, type: 'u32', access: 'read'},
          {name: 'groupIndex', view: groups.groupIndex, type: 'u32', access: 'read'},
          {name: 'keyHigh', view: keyHigh, type: 'u32', access: 'read'},
          {name: 'offsets', view: weights.offsets, type: 'u32', access: 'read_write'},
          {name: 'overflow', view: props.overflow, type: 'u32', access: 'read_write'},
          ...(props.totalPairs
            ? [
                {
                  name: 'totalPairs',
                  view: props.totalPairs,
                  type: 'u32' as const,
                  access: 'read_write' as const
                }
              ]
            : [])
        ],
        invocationCount: targetCount + 1,
        declarations: `const CELL_COUNT: u32 = ${cellCount}u;
const ROWS: u32 = ${targetCount}u;
const CAPACITY: u32 = ${capacity}u;`,
        body: `// Lower bound: the first pair run whose targetRow is >= index (invalid runs hold 0xffffffff).
  let groupCount = groupIndex[groupIndexOffset + CELL_COUNT - 1u];
  var low = 0u;
  var high = groupCount;
  while (low < high) {
    let middle = (low + high) / 2u;
    let item = sortedItems[sortedItemsOffset + groupStarts[groupStartsOffset + middle]];
    if (keyHigh[keyHighOffset + item] < index) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  offsets[offsetsOffset + index] = min(low, CAPACITY);
  if (index == ROWS) {
    overflow[overflowOffset] = select(0u, 1u, low > CAPACITY);
    ${props.totalPairs ? 'totalPairs[totalPairsOffset] = low;' : ''}
  }`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-weights`,
        operation: OPERATION,
        variant: `weights-${mode}${props.alternateWeights ? '-both' : ''}`,
        bindings: [
          {name: 'areas', view: areas, type: 'f32', access: 'read'},
          {name: 'neighbors', view: weights.neighbors, type: 'u32', access: 'read'},
          {name: 'pairTarget', view: pairTarget, type: 'u32', access: 'read'},
          {name: 'sourceTotals', view: sourceTotals, type: 'f32', access: 'read'},
          {name: 'targetTotals', view: targetTotals, type: 'f32', access: 'read'},
          {name: 'weights', view: weights.weights, type: 'f32', access: 'read_write'},
          ...(props.alternateWeights
            ? [
                {
                  name: 'alternateWeights',
                  view: props.alternateWeights,
                  type: 'f32' as const,
                  access: 'read_write' as const
                }
              ]
            : [])
        ],
        invocationCount: slotCount,
        declarations: `const SOURCE_COUNT: u32 = ${sourceCount}u;
const TARGET_COUNT: u32 = ${targetCount}u;`,
        body: `let source = neighbors[neighborsOffset + index];
  let targetRow = pairTarget[pairTargetOffset + index];
  var extensive = 0.0;
  var intensive = 0.0;
  if (source < SOURCE_COUNT && targetRow < TARGET_COUNT) {
    let area = areas[areasOffset + index];
    let sourceTotal = sourceTotals[sourceTotalsOffset + source];
    let targetTotal = targetTotals[targetTotalsOffset + targetRow];
    extensive = select(0.0, area / sourceTotal, sourceTotal > 0.0);
    intensive = select(0.0, area / targetTotal, targetTotal > 0.0);
  }
  weights[weightsOffset + index] = ${mode === 'extensive' ? 'extensive' : 'intensive'};
  ${
    props.alternateWeights
      ? `alternateWeights[alternateWeightsOffset + index] = ${mode === 'extensive' ? 'intensive' : 'extensive'};`
      : ''
  }`
      })
    );

    if (props.categories) {
      const {sourceCategories, categoryCount, output} = props.categories;
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-category-shares`,
          operation: OPERATION,
          variant: 'category-shares',
          bindings: [
            {name: 'offsets', view: weights.offsets, type: 'u32', access: 'read'},
            {name: 'neighbors', view: weights.neighbors, type: 'u32', access: 'read'},
            {name: 'areas', view: areas, type: 'f32', access: 'read'},
            {name: 'sourceCategories', view: sourceCategories, type: 'u32', access: 'read'},
            {name: 'output', view: output, type: 'f32', access: 'read_write'}
          ],
          invocationCount: targetCount * categoryCount,
          declarations: `const SOURCE_COUNT: u32 = ${sourceCount}u;
const CATEGORY_COUNT: u32 = ${categoryCount}u;`,
          body: `let targetRow = index / CATEGORY_COUNT;
  let category = index % CATEGORY_COUNT;
  var covered = 0.0;
  var inCategory = 0.0;
  for (var slot = offsets[offsetsOffset + targetRow]; slot < offsets[offsetsOffset + targetRow + 1u]; slot++) {
    let source = neighbors[neighborsOffset + slot];
    if (source < SOURCE_COUNT) {
      let area = areas[areasOffset + slot];
      covered += area;
      if (sourceCategories[sourceCategoriesOffset + source] == category) {
        inCategory += area;
      }
    }
  }
  output[outputOffset + index] = select(0.0, inCategory / covered, covered > 0.0);`
        })
      );
    }
    return nodes;
  }
}
