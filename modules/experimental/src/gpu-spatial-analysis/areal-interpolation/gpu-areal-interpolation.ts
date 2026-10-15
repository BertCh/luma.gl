// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {createFillNode, createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import {
  getStorageBindingElementLimit,
  validateCapacity,
  validateMaximumCount,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validatePackedView
} from '../../utils/gpu-contributor-utils';
import {createSegmentSumNode, getSortedSegmentSumNodes} from '../../utils/sorted-segment-sums';
import {GET_NAN_WGSL} from '../../utils/wgsl-common';
import {
  getGPUAuthalicSphereRadius,
  GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
  resolveGPUSpatialContext,
  type GPUSpatialContext,
  type GPUSpatialContextSupport
} from '../../analysis-core/contracts/spatial-context';
import {
  createSphericalCellWeightsNode,
  validateGeographicRaster
} from '../../analysis-core/spherical-cell-area';
import {validateRasterGrid} from '../../gpu-raster/raster-band-utils';
import type {GPURasterMetadata} from '../../gpu-raster/types';
import type {GPUOverflowStatusPort, GPUStatusScalar} from '../../analysis-core/contracts/status';
import {getZoneAnchorNodes} from './zone-anchor-nodes';
import {getKeyGroupNodes, KEY_PAIR_INVALID} from '../spatial-weights/key-pair-grouping';
import {getBoundedKeyPairSortNodes} from './bounded-key-pair-sort';
import {
  type GPUSpatialWeights,
  validateGPUSpatialWeights
} from '../spatial-weights/spatial-weights';

const OPERATION = 'GPUArealInterpolation';

/**
 * Largest grid cell count of {@link GPUArealInterpolation} on `device`: the sorted cell groups keep
 * `cells + 1` uint32 group starts in one storage binding, so `2^25 - 1` on a default device
 * (128 MiB bindings).
 *
 * @param device Device whose `maxStorageBufferBindingSize` applies.
 */
export function getGPUArealInterpolationMaximumCellCount(device: Device): number {
  return getStorageBindingElementLimit(device, 4, 1);
}

/**
 * Largest pair slot count, `min(output.weights.neighbors rows, cells)`, of
 * {@link GPUArealInterpolation} with `cellWeights`, `unweightedFastPath: false` or a
 * longitude/latitude raster: the pair areas
 * are sorted segment sums with one 256-thread workgroup per pair, and `slots * 256` invocations must
 * fit one uint32 dispatch, so `2^24 - 1`.
 */
export const GPU_AREAL_INTERPOLATION_MAXIMUM_WEIGHTED_PAIR_COUNT = 2 ** 24 - 1;
/** Planar coordinates, or longitude/latitude rasters weighted by spherical cell area. */
const SUPPORT: GPUSpatialContextSupport = Object.freeze({
  ...GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
  measure: 'area'
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

/** Categorical source variable turned into per-target category shares (`output.categoryShares`). */
export type GPUArealCategories = {
  /** Category per source, in `[0, categoryCount)`; other values contribute to no category. */
  sourceCategories: GraphDataView<'uint32'>;
  /** Number of categories `K`. */
  categoryCount: number;
};

/** Caller-owned outputs of {@link GPUArealInterpolation}. */
export type GPUArealInterpolationOutput = {
  /**
   * Output CSR with `targetZoneCount + 1` offsets: row `t` (target) lists its overlapping sources
   * `s` as `neighbors`, ascending, with the area-share weight of `mode`. The slot capacity is
   * `neighbors.length` and must hold the number of distinct (source, target) pairs that share at
   * least one cell. `distances` is not written. This is a cross (target to source) weights matrix:
   * its neighbors index sources, not rows.
   */
  weights: GPUSpatialWeights;
  /** Status words: pair overflow, the optional required pair count and the optional invalid count. */
  status: GPUOverflowStatusPort & {
    /** Optional one-row unclamped distinct pair count, for sizing the slot capacity. */
    requiredPairCount?: GPUStatusScalar;
    /**
     * Optional one row: the number of sources with no raster area after anchoring. Their mass is
     * lost.
     */
    invalidCount?: GPUStatusScalar;
  };
  /** Optional weights (one per slot) holding the normalization that is not `mode`. */
  alternateWeights?: GraphDataView<'float32'>;
  /**
   * Optional overlap area per slot (cell count, or the sum of `cellWeights`; square units of the
   * context, square meters by default, on a longitude/latitude raster).
   */
  areas?: GraphDataView<'float32'>;
  /**
   * Optional `sourceZoneCount` rows: the whole raster area `A_s` of each source (cell count, or the
   * sum of the cell weights; square units of the context, square meters by default, on a
   * longitude/latitude raster, after anchoring). 0 when the source rasterized to no cell.
   */
  sourceAreas?: GraphDataView<'float32'>;
  /**
   * Optional `sourceZoneCount` rows: the conservation fraction of each source, `sum_t a_st / A_s`
   * with the extensive denominator, where `a_st` sums over targets (before `weights` slot
   * clamping). With `'zone'`, 1 means the source lies entirely inside the target system and its
   * extensive mass is fully transferred, and `1 - f` is the lost fraction. With `'overlap'` it is 1
   * wherever the source overlaps any target and 0 where it does not. NaN when the source has no
   * raster area (see `sourceAreas`).
   */
  sourceTransferredFractions?: GraphDataView<'float32'>;
  /**
   * `targetZoneCount * K` category shares, row-major (`categoryShares[t * K + k]`); required with
   * `categories`. `share(t, k)` is the overlap area of target `t` with sources of category `k`,
   * divided by the area of `t` covered by any source, so the shares of a covered target sum to 1.
   * Uncovered targets output 0.
   */
  categoryShares?: GraphDataView<'float32'>;
};

/**
 * Properties for {@link GPUArealInterpolation}.
 *
 * Compile-time: `sourceZoneCount`, `targetZoneCount`, `denominator`, `mode`, `spatialContext`, the `grid` size,
 * view lengths and the pair capacity (`output.weights.neighbors.length`). Per-frame: the contents of the zone rasters,
 * `cellWeights`, source values and categories.
 */
export type GPUArealInterpolationProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'areal-interpolation'`. */
  id?: string;
  /**
   * Raster grid shared by the zone rasters (and `cellWeights`): its width and height size the
   * kernels, and every raster holds exactly `grid.width * grid.height` rows.
   */
  grid: GPURasterMetadata;
  /**
   * Source-zone raster: one cell per entry, the source feature row of each cell or any value
   * `>= sourceZoneCount` (such as `GPU_POLYGON_RASTERIZATION_NO_ZONE`) for no source. Typically the
   * `zones` output of a `GPUPolygonRasterization` over the source polygons. At most
   * {@link getGPUArealInterpolationMaximumCellCount} cells (2^25 - 1 on a default device).
   */
  sourceZones: GraphDataView<'uint32'>;
  /**
   * Target-zone raster on the same grid as `sourceZones` (same grid, same length),
   * from a `GPUPolygonRasterization` of the target polygons, or any zone raster such as H3 or
   * hexagon cells rasterized to the grid.
   */
  targetZones: GraphDataView<'uint32'>;
  /** Number of source zones `S`. */
  sourceZoneCount: number;
  /** Number of target zones `T`; also the row count of the output weights. */
  targetZoneCount: number;
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
   * is reported in `output.sourceTransferredFractions`. An intensive variable (rate, mean) must not be
   * diluted when a target is only partly covered by sources, so it defaults to `'overlap'` and is
   * the mean of what covers the target. Earlier versions applied `'zone'` to both.
   */
  denominator?: GPUArealInterpolationDenominator | GPUArealInterpolationDenominators;
  /**
   * Coordinate space of the zone rasters. Omitted means native planar: every area is a cell count
   * (or a sum of `cellWeights`). A `longitude-latitude` context with metric `'great-circle'` means
   * the rasters are longitude/latitude grids placed by `parameters` and `grid`; each cell is then
   * weighted by its spherical area in square units of the context (square meters by default),
   * `R^2 dLon (sin(latTop) - sin(latBottom))` with `R` = `context.sphereRadius` or the WGS84
   * authalic radius in the context's length unit, times `cellWeights` when given, so a latitude band
   * is not over-counted.
   * Cell membership of the zones is by cell center in degrees. Other spaces throw.
   */
  spatialContext?: GPUSpatialContext;
  /**
   * `longitude-latitude` only: the per-frame raster placement, the `getGPUPolygonRasterizationParameterValues(grid)`
   * view, the same view as `GPUPolygonRasterization.parameters` (float32, at least 4 values). A
   * negative cell height (north-up rows) is handled: cell areas follow the mirrored rows. Required
   * for, and only valid with, a longitude/latitude context.
   */
  parameters?: GraphDataView<'float32'>;
  /**
   * Optional per-source fallback cell (`sourceZoneCount` rows of cell indices; out of range means
   * none). A source zone that rasterized to no cell claims its anchor cell, for example the cell
   * containing the zone's label point, so its mass is not lost. The claim happens in a transient
   * copy of `sourceZones` and succeeds only if the cell belongs to no source or to a source that
   * keeps another cell; shared anchors go to the lowest source row. Sources that still have no
   * cell are counted in `output.status.invalidCount`.
   */
  sourceAnchorCells?: GraphDataView<'uint32'>;
  /**
   * Which normalization is written to `output.weights.weights`. Defaults to `'extensive'`. The
   * other one is written to `output.alternateWeights` when given.
   */
  mode?: GPUArealInterpolationKind;
  /** Optional categorical variable to turn into `output.categoryShares`. */
  categories?: GPUArealCategories;
  /** Caller-owned outputs. */
  output: GPUArealInterpolationOutput;
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
 * Conservation diagnostics. `output.sourceAreas` and `output.sourceTransferredFractions` report
 * per source how much of its raster area reached the targets, and `output.status.invalidCount` counts sources that rasterized
 * to no cell (a source smaller than a cell, or a polygon the cell centers all miss): their mass is
 * lost unless `sourceAnchorCells` gives them a fallback cell. Dropped mass is therefore reported,
 * never silent.
 *
 * Coordinates. With a `longitude-latitude` `spatialContext` the areas are spherical cell areas in
 * square units of the context (square meters by default), so the weights are ground-area shares on
 * a geographic raster (a cell at 60 degrees latitude weighs half of one at the equator). Without it every cell weighs 1 (or `cellWeights`),
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
 * Outputs: the CSR `output.weights`, optionally the raw `areas` per slot (explain column),
 * `alternateWeights` (the other normalization) and `categoryShares`. Transfer variables with
 * `GPUSpatialLag` (`values` with `sourceZoneCount` rows, optional `columnCount` for several
 * variables at once) over `output.weights`, using `alternateWeights` as its `weights.weights` for the
 * other kind. When the pairs exceed the slot capacity, `overflow` is 1, the offsets are clamped to the capacity and the
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
    const gridCellCount = validateRasterGrid(id, props.grid);
    if (geographic) {
      validateGeographicRaster(id, props.parameters);
    } else if (props.parameters) {
      throw new Error(`${id} parameters need a longitude-latitude spatialContext`);
    }
    const {sourceZoneCount: sourceCount, targetZoneCount: targetCount, output} = props;
    // The zone IDs and the no-zone sentinel 0xffffffff share uint32, so counts stop one below it.
    validateCapacity(id, 'sourceZoneCount', sourceCount, 1, 0xfffffffe);
    validateCapacity(id, 'targetZoneCount', targetCount, 1, 0xfffffffe);
    validatePackedView(props.sourceZones, ['uint32'], `${id} sourceZones`);
    validatePackedView(props.targetZones, ['uint32'], `${id} targetZones`);
    if (props.sourceZones.length !== gridCellCount || props.targetZones.length !== gridCellCount) {
      throw new Error(`${id} sourceZones and targetZones must contain one row per grid cell`);
    }
    if (props.cellWeights) {
      validatePackedView(props.cellWeights, ['float32'], `${id} cellWeights`);
      if (props.cellWeights.length !== gridCellCount) {
        throw new Error(`${id} cellWeights length must equal the grid cell count`);
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
      validatePackedView(props.sourceAnchorCells, ['uint32'], `${id} sourceAnchorCells`);
      if (props.sourceAnchorCells.length !== sourceCount) {
        throw new Error(`${id} sourceAnchorCells must have sourceZoneCount rows`);
      }
    }
    for (const [name, view] of [
      ['output.sourceAreas', output.sourceAreas],
      ['output.sourceTransferredFractions', output.sourceTransferredFractions]
    ] as const) {
      if (view) {
        validatePackedView(view, ['float32'], `${id} ${name}`);
        if (view.length !== sourceCount) {
          throw new Error(`${id} ${name} must have sourceZoneCount rows`);
        }
      }
    }
    const rows = validateGPUSpatialWeights(id, output.weights, 'output.weights');
    if (rows !== targetCount) {
      throw new Error(`${id} output.weights.offsets length must equal targetZoneCount + 1`);
    }
    const capacity = output.weights.neighbors.length;
    if (
      props.cellWeights ||
      props.unweightedFastPath === false ||
      this.spatialContext.coordinateSpace === 'longitude-latitude'
    ) {
      validateMaximumCount(
        id,
        'pair slots (output.weights.neighbors rows, cells)',
        Math.min(capacity, props.sourceZones.length),
        GPU_AREAL_INTERPOLATION_MAXIMUM_WEIGHTED_PAIR_COUNT
      );
    }
    for (const [name, view] of [
      ['output.alternateWeights', output.alternateWeights],
      ['output.areas', output.areas]
    ] as const) {
      if (view) {
        validatePackedView(view, ['float32'], `${id} ${name}`);
        if (view.length !== capacity) {
          throw new Error(`${id} ${name} length must equal output.weights.neighbors length`);
        }
      }
    }
    for (const [name, view] of [
      ['output.status.overflow', output.status.overflow],
      ['output.status.requiredPairCount', output.status.requiredPairCount],
      ['output.status.invalidCount', output.status.invalidCount]
    ] as const) {
      if (view) {
        validatePackedView(view, ['uint32'], `${id} ${name}`);
        if (view.length < 1) {
          throw new Error(`${id} ${name} must hold one uint32`);
        }
      }
    }
    if (props.categories) {
      const {sourceCategories, categoryCount} = props.categories;
      const {categoryShares} = output;
      validatePackedView(sourceCategories, ['uint32'], `${id} categories.sourceCategories`);
      validateCapacity(id, 'categories.categoryCount', categoryCount);
      if (sourceCategories.length !== sourceCount) {
        throw new Error(`${id} categories.sourceCategories must have sourceZoneCount rows`);
      }
      if (!categoryShares) {
        throw new Error(`${id} categories needs output.categoryShares`);
      }
      validatePackedView(categoryShares, ['float32'], `${id} output.categoryShares`);
      if (categoryShares.length !== targetCount * categoryCount) {
        throw new Error(
          `${id} output.categoryShares must have targetZoneCount * categoryCount rows`
        );
      }
    } else if (output.categoryShares) {
      throw new Error(`${id} output.categoryShares needs categories`);
    }
    validateGraphOutputsDisjointFromInputs(id, this.getOutputViews(), this.getInputViews());
  }

  private getInputViews(): (GraphDataView | undefined)[] {
    const {props} = this;
    return [
      props.sourceZones,
      props.targetZones,
      props.cellWeights,
      props.parameters,
      props.sourceAnchorCells,
      props.categories?.sourceCategories
    ];
  }

  private getOutputViews(): (GraphDataView | undefined)[] {
    const {output} = this.props;
    return [
      output.weights.offsets,
      output.weights.neighbors,
      output.weights.weights,
      output.weights.distances,
      output.alternateWeights,
      output.areas,
      output.status.overflow,
      output.status.requiredPairCount,
      output.status.invalidCount,
      output.sourceAreas,
      output.sourceTransferredFractions,
      output.categoryShares
    ];
  }

  /** Returns the pair, total, weight and category nodes in dependency order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    validateGraphViewsBelongToGraph(id, graph, [...this.getInputViews(), ...this.getOutputViews()]);
    validateMaximumCount(
      id,
      'sourceZones rows',
      props.sourceZones.length,
      getGPUArealInterpolationMaximumCellCount(graph.device)
    );
    const {targetZones, sourceZoneCount: sourceCount, targetZoneCount: targetCount, output} = props;
    const {weights} = output;
    const cellCount = props.sourceZones.length;
    const capacity = weights.neighbors.length;
    const slotCount = Math.min(capacity, cellCount);
    const mode = props.mode ?? 'extensive';
    const geographic = this.spatialContext.coordinateSpace === 'longitude-latitude';
    const extensiveOverlap = this.extensiveDenominator === 'overlap';
    const intensiveOverlap = this.intensiveDenominator === 'overlap';
    const needSourceKey = extensiveOverlap || Boolean(output.sourceTransferredFractions);
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
          width: props.grid.width,
          height: props.grid.height,
          radius: getGPUAuthalicSphereRadius(this.spatialContext),
          extent: props.parameters!,
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

    const areas = output.areas ?? createTransientView(graph, `${id}-areas`, 'float32', capacity);
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
    const {sourceAreas, sourceTransferredFractions} = output;
    const {invalidCount} = output.status;
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
          {name: 'overflow', view: output.status.overflow, type: 'u32', access: 'read_write'},
          ...(output.status.requiredPairCount
            ? [
                {
                  name: 'totalPairs',
                  view: output.status.requiredPairCount,
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
    ${output.status.requiredPairCount ? 'totalPairs[totalPairsOffset] = low;' : ''}
  }`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-weights`,
        operation: OPERATION,
        variant: `weights-${mode}${output.alternateWeights ? '-both' : ''}`,
        record:
          mode === 'extensive' && this.extensiveDenominator === 'zone'
            ? {formula: 'gpu-spatial-analysis.areal-interpolation'}
            : undefined,
        bindings: [
          {name: 'areas', view: areas, type: 'f32', access: 'read'},
          {name: 'neighbors', view: weights.neighbors, type: 'u32', access: 'read'},
          {name: 'pairTarget', view: pairTarget, type: 'u32', access: 'read'},
          {name: 'sourceTotals', view: sourceTotals, type: 'f32', access: 'read'},
          {name: 'targetTotals', view: targetTotals, type: 'f32', access: 'read'},
          {name: 'weights', view: weights.weights, type: 'f32', access: 'read_write'},
          ...(output.alternateWeights
            ? [
                {
                  name: 'alternateWeights',
                  view: output.alternateWeights,
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
    output.alternateWeights
      ? `alternateWeights[alternateWeightsOffset + index] = ${mode === 'extensive' ? 'intensive' : 'extensive'};`
      : ''
  }`
      })
    );

    if (props.categories && output.categoryShares) {
      const {sourceCategories, categoryCount} = props.categories;
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
            {name: 'output', view: output.categoryShares, type: 'f32', access: 'read_write'}
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
