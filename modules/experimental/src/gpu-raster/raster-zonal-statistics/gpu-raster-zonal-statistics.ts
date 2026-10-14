// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  createTransientView,
  GPUGroupAggregation,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {getSortedSegmentSumNodes} from '../../utils/sorted-segment-sums';
import type {GPURasterBand, GPURasterMetadata} from '../types';
import {
  createFillNode,
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../utils/wgsl-kernel-nodes';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {
  getStorageBindingElementLimit,
  validateCapacity,
  validateGraphViewsBelongToGraph,
  validatePackedView
} from '../../utils/gpu-contributor-utils';
import {
  getRasterBandViews,
  getRasterBufferBandNodes,
  validateRasterBandBelongsToGraph,
  validateParameterTarget,
  validateRasterBuffersDistinct,
  validateRasterGrid,
  validateRasterParameters
} from '../raster-band-utils';
import {IS_FINITE_VALUE_WGSL} from '../../utils/wgsl-common';
import {GPU_EARTH_AUTHALIC_RADIUS_METERS} from '../../analysis-core/contracts/earth';
import {
  getSphericalCellAreaWGSL,
  getSphericalRasterCellArea
} from '../../analysis-core/spherical-cell-area';
import {
  RASTER_GRID_PARAMETER_LENGTH,
  writeRasterGridPrefix
} from '../cost-distance/raster-grid-utils';

/**
 * Cell weighting of {@link GPURasterZonalStatistics}, following the `cellSizeMode` vocabulary of the
 * other raster and terrain contributors.
 * - `'uniform'` (default): every cell has weight one (planar or projected rasters).
 * - `'geographic'`: cells are longitude/latitude degrees and are weighted by their area in square
 *   meters on the WGS84 authalic sphere, `R^2 dLon (sin(top) - sin(bottom))`.
 */
export type GPURasterZonalStatisticsCellSizeMode = 'uniform' | 'geographic';

/** Number of float32 values read from `GPURasterZonalStatisticsProps.parameters`. */
export const GPU_RASTER_ZONAL_STATISTICS_PARAMETER_LENGTH = RASTER_GRID_PARAMETER_LENGTH;

/**
 * Packs the grid words read by {@link GPURasterZonalStatistics} with `cellSizeMode:
 * 'geographic'`: `[cellSizeX, cellSizeY, northEdge, southEdge]` in degrees, with the cell size and
 * the row-edge latitudes taken from `grid`. There are no other settings.
 *
 * @param grid The frame's grid; its width and height must match the contributor's.
 * @param target Optional destination of at least 4 elements.
 */
export function getGPURasterZonalStatisticsParameterValues(
  grid: GPURasterMetadata,
  target: Float32Array = new Float32Array(GPU_RASTER_ZONAL_STATISTICS_PARAMETER_LENGTH)
): Float32Array {
  validateParameterTarget(
    'getGPURasterZonalStatisticsParameterValues',
    target,
    GPU_RASTER_ZONAL_STATISTICS_PARAMETER_LENGTH
  );
  writeRasterGridPrefix(target, grid);
  return target;
}

/**
 * Largest `zoneCapacity` of {@link GPURasterZonalStatistics} on `device` when no sorted sums run:
 * each output column holds 4 bytes per zone in one storage binding, so 33554432 on a default
 * device (128 MiB bindings).
 */
export function getGPURasterZonalStatisticsMaximumZoneCapacity(device: Device): number {
  return getStorageBindingElementLimit(device, 4);
}

/**
 * Largest `zoneCapacity` of {@link GPURasterZonalStatistics} on `device` when `sumOrder` is
 * `'sorted'` and a sum, mean, variance, standard deviation or weight sum is requested (or
 * `cellSizeMode` is `'geographic'`): the sorted reduction sums each zone with one 256-invocation
 * workgroup, and the total invocation count must fit a uint32 (`floor((2^32 - 1) / 256)`); the
 * `zoneCapacity + 1` uint32 segment offsets must also fit one storage binding. 16777215 on a
 * default device (128 MiB bindings).
 */
export function getGPURasterZonalStatisticsMaximumSortedZoneCapacity(device: Device): number {
  return Math.min(Math.floor(0xffffffff / 256), getStorageBindingElementLimit(device, 4, 1));
}

/** Zone ID that marks a cell as belonging to no zone. Default value of `ignoredZone`. */
export const GPU_RASTER_ZONAL_STATISTICS_NO_ZONE = 0xffffffff;

/**
 * Caller-owned per-zone result columns. At least one column is required and every column must hold
 * exactly `zoneCapacity` rows. Zones with no contributing cell yield `0` for counts and sums and
 * NaN for `means`, `minimums`, and `maximums`.
 */
export type GPURasterZonalStatisticsOutput = {
  /** Per zone: cells whose zone ID is in range and not ignored, regardless of value validity. */
  cellCounts?: GraphDataView<'uint32'>;
  /** Per zone: cells with a valid, finite calibrated value. */
  valueCounts?: GraphDataView<'uint32'>;
  /** Per zone: sum of valid calibrated values (0 for empty zones). */
  sums?: GraphDataView<'float32'>;
  /** Per zone: mean of valid calibrated values (NaN for zones without valid values). */
  means?: GraphDataView<'float32'>;
  /** Per zone: minimum valid calibrated value (NaN for zones without valid values). */
  minimums?: GraphDataView<'float32'>;
  /** Per zone: maximum valid calibrated value (NaN for zones without valid values). */
  maximums?: GraphDataView<'float32'>;
  /** Optional one-row status words, rewritten on every encoding. */
  status?: GPURasterZonalStatisticsStatus;
};

/** One-row status words of {@link GPURasterZonalStatistics}. */
export type GPURasterZonalStatisticsStatus = {
  /**
   * One-row flag: 1 when any cell has a zone ID at or above `zoneCapacity` that is not
   * `ignoredZone`, otherwise 0. Presence is compile-time.
   */
  overflow?: GraphDataView<'uint32'>;
};

/**
 * Optional weight-sum and dispersion columns of {@link GPURasterZonalStatistics}, with the same
 * length rule as {@link GPURasterZonalStatisticsOutput}. Kept separate so the base type keeps
 * naming exactly the six basic statistics.
 */
export type GPURasterZonalStatisticsMomentOutput = {
  /**
   * Per zone: sum of the cell weights of valid values. Square meters with
   * `cellSizeMode: 'geographic'`, otherwise the valid value count as a float. 0 for empty zones.
   */
  weightSums?: GraphDataView<'float32'>;
  /**
   * Per zone: unbiased sample variance of the valid values. Uniform cells:
   * `sum (v - mean)^2 / (n - 1)`. Geographic cells, whose areas act as reliability weights:
   * `sum w (v - m)^2 / (V1 - V2 / V1)` with `m` the area-weighted mean, `V1 = sum w` and
   * `V2 = sum w^2` (`numpy.cov(v, aweights=w, ddof=1)`), the convention of `GPUZonalStatistics`.
   * Two passes (mean first), so there is no catastrophic cancellation. NaN below two valid values
   * or for a non-positive denominator.
   */
  variances?: GraphDataView<'float32'>;
  /** Per zone: `sqrt(variances)`; NaN where the variance is NaN. */
  standardDeviations?: GraphDataView<'float32'>;
};

/**
 * How per-zone sums (and means) are accumulated.
 * - `'sorted'` (default): cells are stably sorted by zone and reduced per zone in a fixed tree, so
 *   `sums` and `means` are bitwise reproducible across encodings on one device. Cost is roughly
 *   flat (about 10 to 25 ms at 2048x2048 on an Apple M3 Pro) whatever the zone layout.
 * - `'atomic'`: float compare-exchange atomics. The last bits of `sums` and `means` depend on
 *   accumulation order and can differ between encodings. Only worthwhile for many (1000 or more)
 *   spatially scattered zones, where it is up to about 1.5x faster. It is 10 to 60x slower when
 *   many cells in a wave share a zone, which is the normal case for contiguous administrative
 *   polygons and for few zones, because every lane retries a compare-exchange on the same address.
 */
export type GPURasterZonalStatisticsSumOrder = 'atomic' | 'sorted';

/**
 * Properties for {@link GPURasterZonalStatistics}.
 *
 * Compile-time: `grid` size, `zoneCapacity`, `ignoredZone`, the band format, calibration and
 * storage kind, which output columns and `overflow` exist, and every view length. Per-frame: the
 * contents of `zones` and `values` (and its validity), which callers rewrite between encodings.
 */
export type GPURasterZonalStatisticsProps = {
  /** Prefix for node and transient IDs. Defaults to `'raster-zonal-statistics'`. Compile-time. */
  id?: string;
  /** Raster grid: its width and height size the kernels. */
  grid: GPURasterMetadata;
  /**
   * Packed row-major zone ID per cell, for example rasterized administrative areas. Length is
   * `grid.width * grid.height`. Length is compile-time, contents are per-frame.
   */
  zones: GraphDataView<'uint32'>;
  /**
   * Value band, buffer or texture backed. Validity, `noDataValue`, `scale`, and `offset` are
   * honored. Band configuration is compile-time, contents are per-frame.
   */
  values: GPURasterBand;
  /**
   * Number of dense zones. Zone IDs in `[0, zoneCapacity)` aggregate. Every output column must
   * have exactly this length. Compile-time. An integer in
   * [1, {@link getGPURasterZonalStatisticsMaximumZoneCapacity}] (33554432 on a default device), or
   * at most {@link getGPURasterZonalStatisticsMaximumSortedZoneCapacity} (16777215) when sorted
   * sums run.
   */
  zoneCapacity: number;
  /**
   * Zone ID silently skipped, for example background `0`. Defaults to
   * {@link GPU_RASTER_ZONAL_STATISTICS_NO_ZONE}. Compile-time.
   */
  ignoredZone?: number;
  /**
   * Cell weighting. Defaults to `'uniform'`. With `'geographic'`, `sums`, `means`, `weightSums` and
   * `variances` weight each cell by its spherical area; counts, minimums and maximums are never
   * weighted. Compile-time.
   */
  cellSizeMode?: GPURasterZonalStatisticsCellSizeMode;
  /**
   * Grid words for `cellSizeMode: 'geographic'` (required then, rejected otherwise), see
   * {@link getGPURasterZonalStatisticsParameterValues}:
   * `[cellSizeX, cellSizeY, northEdge, southEdge]` in degrees, the shared raster grid prefix of
   * `GPUCostDistance`. `cellSizeX` is the cell width in longitude; `northEdge` and `southEdge` are the
   * outer latitudes of the first and last rows, so row `r` spans `(southEdge - northEdge) / height`
   * degrees of latitude from `northEdge + r * (southEdge - northEdge) / height`. `cellSizeY` is not
   * read. Contents are per-frame.
   */
  parameters?: GraphDataView<'float32'>;
  /**
   * Accumulation order for `sums` and `means`. Defaults to `'sorted'`. Compile-time.
   * `'sorted'` makes them bitwise reproducible and avoids same-address atomic contention, at the
   * cost of a radix sort of all cells, a scan, and a gather per encoding. `'atomic'` skips the sort
   * and only wins for many scattered zones. Counts, minimums, and maximums are exact either way.
   */
  sumOrder?: GPURasterZonalStatisticsSumOrder;
  /** Per-zone result columns, rewritten on every encoding. Which columns exist is compile-time. */
  output: GPURasterZonalStatisticsOutput & GPURasterZonalStatisticsMomentOutput;
};

const COLUMN_OPERATIONS = [
  ['cellCounts', 'count'],
  ['valueCounts', 'count'],
  ['sums', 'sum'],
  ['means', 'mean'],
  ['minimums', 'min'],
  ['maximums', 'max']
] as const;

/** Float columns computed by the weighted moment path rather than `GPUGroupAggregation`. */
const MOMENT_COLUMNS = ['weightSums', 'variances', 'standardDeviations'] as const;

/**
 * Area in square meters of one cell of row `row` of a geographic raster on the authalic sphere,
 * the CPU twin of the WGSL cell weight of {@link GPURasterZonalStatistics}.
 *
 * @param settings `[cellSizeX, cellSizeY, northEdge, southEdge]` in degrees.
 * @internal
 */
export function getRasterZonalCellArea(
  settings: ArrayLike<number>,
  row: number,
  height: number
): number {
  const rowHeight = (settings[3] - settings[2]) / height;
  return Math.abs(getSphericalRasterCellArea(settings[2], settings[0], rowHeight, row));
}

/**
 * WGSL `fn getCellWeight(cell: u32) -> f32`: the spherical area of a geographic cell from the
 * `settings` binding (the `parameters` view), through `getCellArea` of the shared spherical cell-area pattern (which reads
 * `[originLongitude, originLatitude, cellWidth, cellHeight]` from `extent`). Non-finite or
 * non-positive areas weigh 0. Needs `isFiniteValue`.
 */
function getCellWeightWGSL(width: number, height: number): string {
  return `const ZONAL_GRID_WIDTH: u32 = ${width}u;
const ZONAL_GRID_HEIGHT: f32 = ${height}.0;
var<private> extent: array<f32, 4>;
const extentOffset: u32 = 0u;
${getSphericalCellAreaWGSL(GPU_EARTH_AUTHALIC_RADIUS_METERS)}
fn getCellWeight(cell: u32) -> f32 {
  let northEdge = settings[settingsOffset + 2u];
  // Signed row height: negative when rows advance south, which makes getCellArea negative.
  let rowHeight = (settings[settingsOffset + 3u] - northEdge) / ZONAL_GRID_HEIGHT;
  extent = array<f32, 4>(0.0, northEdge, settings[settingsOffset], rowHeight);
  let area = abs(getCellArea(cell / ZONAL_GRID_WIDTH));
  return select(0.0, area, isFiniteValue(area) && area > 0.0);
}`;
}

/**
 * Whether the encoding runs the sorted per-zone sums: `sumOrder` is `'sorted'` and a column needs a
 * sum (moments, geographic weights, sums or means).
 */
function hasSortedSums(
  sumOrder: GPURasterZonalStatisticsSumOrder,
  props: GPURasterZonalStatisticsProps
): boolean {
  const {output} = props;
  const momentPath =
    props.cellSizeMode === 'geographic' ||
    Boolean(output?.variances || output?.standardDeviations || output?.weightSums);
  return sumOrder === 'sorted' && (momentPath || Boolean(output?.sums || output?.means));
}

/**
 * Computes per-zone count, sum, mean, minimum, and maximum of a raster band, where zones are a
 * pre-rasterized grid of dense zone IDs.
 *
 * Cell weights: with the default `cellSizeMode: 'uniform'` every cell weighs one, the right model
 * for planar or projected rasters. Longitude/latitude rasters, whose cells shrink with
 * `cos(latitude)`, set `cellSizeMode: 'geographic'` and `parameters`: `sums` become `sum(a * v)`,
 * `means` `sum(a * v) / sum(a)` and `variances` reliability-weighted, with `a` the cell area in
 * square meters on the authalic sphere. Counts, minimums and maximums are never weighted.
 *
 * Variances and standard deviations take a second pass over the cells after the means (and, with
 * `sumOrder: 'sorted'`, a second sort), so they are only computed when requested.
 *
 * The value band is canonicalized (calibration, nodata, validity, non-finite samples). Counts,
 * minimums, and maximums are reduced together (with the optional overflow flag) in one pass over
 * the cells using workgroup-private per-zone tables and one global merge per workgroup, when
 * `zoneCapacity` words per requested column fit in workgroup storage. Otherwise, and for atomic
 * sums and means, a kernel derives zone and value masks and one `GPUGroupAggregation` runs per
 * requested column. Only kernels and masks that a requested column needs are scheduled. Empty zones produce counts of 0, sums of 0, and NaN means, minimums, and
 * maximums. Zones with cells but no valid values count in `cellCounts` only.
 *
 * Determinism: counts, minimums, and maximums are exact. With the default `sumOrder: 'sorted'`
 * the cells are stably sorted by zone ID, scanned into zone offsets, and each zone is reduced with
 * a fixed tree, so `sums` and `means` (sum divided by `valueCounts`) are bitwise reproducible
 * across encodings on one device. This adds a sort over `width * height` cells, a scan, a gather,
 * and one segmented reduction per encoding, but it avoids the same-address contention that makes
 * float atomics 10 to 60x slower on contiguous zones. With `sumOrder: 'atomic'`, sums and means
 * accumulate with float atomics, so their last bits depend on accumulation order and can differ
 * between runs. Results may still differ from a sequential CPU sum in the last bits, and between
 * devices.
 *
 * Non-goals:
 * - No median or percentiles, and no fractional cell coverage of zones.
 * - No sparse or hashed zone IDs: IDs must be dense, so remap upstream (for example with
 *   `GPUBatchHashIndex`).
 * - No polygon rasterization: rasterize zones upstream.
 * - Single tile only: no cross-tile merge, although counts, sums, minimums, and maximums are
 *   mergeable by the caller.
 */
export class GPURasterZonalStatistics implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPURasterZonalStatisticsProps;
  /** Validated sum order. */
  readonly sumOrder: GPURasterZonalStatisticsSumOrder;

  constructor(props: GPURasterZonalStatisticsProps) {
    this.id = props.id ?? 'raster-zonal-statistics';
    this.props = props;
    const {id} = this;
    this.sumOrder = props.sumOrder ?? 'sorted';
    if (this.sumOrder !== 'atomic' && this.sumOrder !== 'sorted') {
      throw new Error(`${id} sumOrder must be atomic or sorted`);
    }
    const cellCount = validateRasterGrid(id, props.grid);
    validatePackedView(props.zones, ['uint32'], `${id} zones`);
    if (props.zones.length !== cellCount) {
      throw new Error(`${id} zones must contain one value per cell`);
    }
    if (
      !Number.isSafeInteger(props.zoneCapacity) ||
      props.zoneCapacity < 1 ||
      props.zoneCapacity > 0xffffffff
    ) {
      throw new Error(`${id} zoneCapacity must be a positive integer that fits in uint32`);
    }
    const ignoredZone = props.ignoredZone ?? GPU_RASTER_ZONAL_STATISTICS_NO_ZONE;
    if (!Number.isInteger(ignoredZone) || ignoredZone < 0 || ignoredZone > 0xffffffff) {
      throw new Error(`${id} ignoredZone must be a uint32 integer`);
    }
    const cellSizeMode = props.cellSizeMode ?? 'uniform';
    if (cellSizeMode !== 'uniform' && cellSizeMode !== 'geographic') {
      throw new Error(`${id} cellSizeMode must be uniform or geographic`);
    }
    if (cellSizeMode === 'geographic') {
      if (!props.parameters) {
        throw new Error(`${id} geographic cells need parameters`);
      }
      validateRasterParameters(id, props.parameters, GPU_RASTER_ZONAL_STATISTICS_PARAMETER_LENGTH);
    } else if (props.parameters) {
      throw new Error(`${id} parameters need cellSizeMode geographic`);
    }
    const {output} = props;
    const columns = [
      ...COLUMN_OPERATIONS.filter(([name]) => props.output[name]),
      ...MOMENT_COLUMNS.filter(name => props.output[name])
    ];
    if (columns.length === 0) {
      throw new Error(`${id} requires at least one output`);
    }
    for (const [name, operation] of COLUMN_OPERATIONS) {
      const view = props.output[name];
      if (!view) {
        continue;
      }
      if (operation === 'count') {
        validatePackedView(view, ['uint32'], `${id} output.${name}`);
      } else {
        validatePackedView(view, ['float32'], `${id} output.${name}`);
      }
      if (view.length !== props.zoneCapacity) {
        throw new Error(`${id} output.${name} must contain zoneCapacity rows`);
      }
    }
    for (const name of MOMENT_COLUMNS) {
      const view = props.output[name];
      if (view) {
        validatePackedView(view, ['float32'], `${id} output.${name}`);
        if (view.length !== props.zoneCapacity) {
          throw new Error(`${id} output.${name} must contain zoneCapacity rows`);
        }
      }
    }
    const overflow = output.status?.overflow;
    if (overflow) {
      validatePackedView(overflow, ['uint32'], `${id} output.status.overflow`);
      if (overflow.length !== 1) {
        throw new Error(`${id} output.status.overflow must contain one uint32 row`);
      }
    }
    validateRasterBuffersDistinct(
      id,
      [
        ...COLUMN_OPERATIONS.map(([name]) => props.output[name]),
        ...MOMENT_COLUMNS.map(name => props.output[name]),
        overflow
      ],
      [props.zones, props.parameters, ...getRasterBandViews(props.values)]
    );
  }

  /** Returns band canonicalization, overflow reset, mask, and per-column aggregation nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {zones, zoneCapacity, output} = props;
    const overflow = output.status?.overflow;
    const {width, height} = props.grid;
    const ignoredZone = props.ignoredZone ?? GPU_RASTER_ZONAL_STATISTICS_NO_ZONE;
    validateRasterBandBelongsToGraph(id, graph, props.values, []);
    validateGraphViewsBelongToGraph(id, graph, [
      zones,
      overflow,
      props.parameters,
      ...COLUMN_OPERATIONS.map(([name]) => output[name]),
      ...MOMENT_COLUMNS.map(name => output[name])
    ]);
    // The sorted sums add one 256-invocation workgroup per zone, which lowers the maximum.
    validateCapacity(
      id,
      'zoneCapacity',
      zoneCapacity,
      1,
      hasSortedSums(this.sumOrder, props)
        ? getGPURasterZonalStatisticsMaximumSortedZoneCapacity(graph.device)
        : getGPURasterZonalStatisticsMaximumZoneCapacity(graph.device)
    );
    const cellCount = width * height;
    const weighted = props.cellSizeMode === 'geographic';
    const needsVariance = Boolean(output.variances || output.standardDeviations);
    // The moment path computes sums, weight sums, means and variances from per-cell contributions
    // (weighted by cell area when geographic) instead of `GPUGroupAggregation` sum and mean.
    const momentPath = weighted || needsVariance || Boolean(output.weightSums);
    const needsMeans = Boolean(output.means) || needsVariance;
    const needsWeightSums = Boolean(output.weightSums) || (weighted && needsMeans);
    const sortedSums = hasSortedSums(this.sumOrder, props);
    const needsValueCounts =
      Boolean(output.valueCounts) ||
      (sortedSums && Boolean(output.means)) ||
      (momentPath && needsMeans);
    const needsValueColumn =
      momentPath ||
      Boolean(
        output.valueCounts || output.sums || output.means || output.minimums || output.maximums
      );
    // Counts, minimums and maximums are reduced together in ONE pass with workgroup-private
    // tables (see `getFusedStatisticsNodes`) whenever the tables fit in workgroup memory. Without
    // that pass every column re-reads zones and a mask through its own `GPUGroupAggregation`.
    const fusedColumnCount = [
      output.cellCounts,
      needsValueCounts,
      output.minimums,
      output.maximums
    ].filter(Boolean).length;
    const fused =
      fusedColumnCount > 0 &&
      fusedColumnCount * 4 * zoneCapacity <= graph.device.limits.maxComputeWorkgroupStorageSize;
    // Masks only feed the per-column aggregations that the fused pass replaces.
    const needsZoneMask = !fused && Boolean(output.cellCounts);
    const needsValueMask = fused
      ? !sortedSums && (momentPath || Boolean(output.sums || output.means))
      : momentPath || COLUMN_OPERATIONS.some(([name]) => name !== 'cellCounts' && output[name]);
    const nodes: GPUCommandNode<Parameters>[] = [];

    let canonicalValues: GraphDataView<'float32'> | undefined;
    let canonicalValidity: GraphDataView<'uint32'> | undefined;
    if (needsValueColumn) {
      const source = getRasterBufferBandNodes(
        graph,
        `${id}-values`,
        props.values,
        props.grid,
        'reuse-canonical'
      );
      nodes.push(...source.nodes);
      canonicalValues = (source.band.storage as {values: GraphDataView<'float32'>}).values;
      canonicalValidity = source.band.validity;
    }

    if (overflow) {
      nodes.push(
        createFillNode<Parameters>(graph, {
          id: `${id}-overflow-reset`,
          operation: 'GPURasterZonalStatistics',
          view: overflow,
          type: 'u32',
          value: '0u'
        })
      );
    }

    const zoneMask = needsZoneMask
      ? createTransientView(graph, `${id}-zone-mask`, 'uint32', cellCount)
      : undefined;
    const valueMask = needsValueMask
      ? createTransientView(graph, `${id}-value-mask`, 'uint32', cellCount)
      : undefined;
    const sortRows = sortedSums
      ? createTransientView(graph, `${id}-sort-rows`, 'uint32', cellCount)
      : undefined;
    const contributions =
      sortedSums || momentPath
        ? createTransientView(graph, `${id}-contributions`, 'float32', cellCount)
        : undefined;
    const weightContributions =
      momentPath && needsWeightSums
        ? createTransientView(graph, `${id}-weight-contributions`, 'float32', cellCount)
        : undefined;
    // The fused pass also raises the overflow flag, so the mask kernel only does when it is absent.
    const maskOverflow = fused ? undefined : overflow;
    // The mask pass binds zones plus every per-cell output it writes. All of them together can
    // reach 10 storage buffers; above the core limit of 8 the zone-only outputs (zone mask, sort
    // rows, overflow flag) get their own pass, which reads only the zones.
    type MaskPart = {
      zoneMask?: GraphDataView<'uint32'>;
      valueMask?: GraphDataView<'uint32'>;
      sortRows?: GraphDataView<'uint32'>;
      contributions?: GraphDataView<'float32'>;
      weightContributions?: GraphDataView<'float32'>;
      maskOverflow?: GraphDataView<'uint32'>;
      /** Whether the pass binds and reads the canonical values and validity. */
      needsValues: boolean;
    };
    const addMaskNode = (nodeId: string, part: MaskPart) => {
      const {needsValues} = part;
      const bindings: WGSLKernelBinding[] = [
        {name: 'zones', view: zones, type: 'u32', access: 'read'}
      ];
      if (needsValues) {
        bindings.push(
          {
            name: 'values',
            view: canonicalValues!,
            type: 'f32',
            access: 'read'
          },
          {
            name: 'validity',
            view: canonicalValidity!,
            type: 'u32',
            access: 'read'
          }
        );
      }
      if (part.valueMask) {
        bindings.push({
          name: 'valueMask',
          view: part.valueMask,
          type: 'u32',
          access: 'read_write'
        });
      }
      if (part.zoneMask) {
        bindings.push({
          name: 'zoneMask',
          view: part.zoneMask,
          type: 'u32',
          access: 'read_write'
        });
      }
      if (part.sortRows) {
        bindings.push({
          name: 'sortRows',
          view: part.sortRows,
          type: 'u32',
          access: 'read_write'
        });
      }
      if (part.contributions) {
        bindings.push({
          name: 'contributions',
          view: part.contributions,
          type: 'f32',
          access: 'read_write'
        });
      }
      if (part.weightContributions) {
        bindings.push({
          name: 'weightContributions',
          view: part.weightContributions,
          type: 'f32',
          access: 'read_write'
        });
      }
      if (weighted && part.contributions) {
        bindings.push({name: 'settings', view: props.parameters!, type: 'f32', access: 'read'});
      }
      if (part.maskOverflow) {
        bindings.push({
          name: 'overflowFlag',
          view: part.maskOverflow,
          type: 'atomic<u32>',
          access: 'read_write'
        });
      }
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: nodeId,
          operation: 'GPURasterZonalStatistics',
          variant: 'mask',
          bindings,
          invocationCount: cellCount,
          declarations: `const CAPACITY: u32 = ${zoneCapacity}u;
const IGNORED: u32 = ${ignoredZone}u;
${IS_FINITE_VALUE_WGSL}
${weighted && part.contributions ? getCellWeightWGSL(width, height) : ''}`,
          body: `let zone = zones[zonesOffset + index];
  let inZone = zone < CAPACITY && zone != IGNORED;
  ${part.zoneMask ? 'zoneMask[zoneMaskOffset + index] = select(0u, 1u, inZone);' : ''}
  ${
    needsValues
      ? `let value = values[valuesOffset + index];
  let isValid = inZone && validity[validityOffset + index] != 0u && isFiniteValue(value);`
      : ''
  }
  ${part.valueMask ? 'valueMask[valueMaskOffset + index] = select(0u, 1u, isValid);' : ''}
  ${part.sortRows ? 'sortRows[sortRowsOffset + index] = select(CAPACITY, zone, inZone);' : ''}
  ${
    part.contributions
      ? weighted
        ? `let weight = getCellWeight(index);
  contributions[contributionsOffset + index] = select(0.0, weight * value, isValid);`
        : 'contributions[contributionsOffset + index] = select(0.0, value, isValid);'
      : ''
  }
  ${
    part.weightContributions
      ? `weightContributions[weightContributionsOffset + index] = select(0.0, ${weighted ? 'weight' : '1.0'}, isValid);`
      : ''
  }
  ${part.maskOverflow ? 'if (zone >= CAPACITY && zone != IGNORED) {\n    atomicStore(&overflowFlag[overflowFlagOffset], 1u);\n  }' : ''}`
        })
      );
    };
    const maskPart: MaskPart = {
      zoneMask,
      valueMask,
      sortRows,
      contributions,
      weightContributions,
      maskOverflow,
      needsValues: Boolean(valueMask || sortRows || contributions)
    };
    const maskBindingCount =
      1 +
      (maskPart.needsValues ? 2 : 0) +
      [zoneMask, valueMask, sortRows, contributions, weightContributions, maskOverflow].filter(
        Boolean
      ).length +
      (weighted && contributions ? 1 : 0);
    if (maskBindingCount > 8) {
      addMaskNode(`${id}-mask-zones`, {zoneMask, sortRows, maskOverflow, needsValues: false});
      addMaskNode(`${id}-mask`, {valueMask, contributions, weightContributions, needsValues: true});
    } else if (zoneMask || valueMask || sortRows || contributions || maskOverflow) {
      addMaskNode(`${id}-mask`, maskPart);
    }

    // Sorted sums find segment sizes by binary search, so cell counts are only built on request;
    // means still need value counts.
    const cellCountsView = output.cellCounts;
    const valueCountsView =
      output.valueCounts ??
      (needsValueCounts
        ? createTransientView(graph, `${id}-value-counts`, 'uint32', zoneCapacity)
        : undefined);
    const countViews: Record<string, GraphDataView<'uint32'> | undefined> = {
      cellCounts: cellCountsView,
      valueCounts: valueCountsView
    };
    if (fused) {
      nodes.push(
        ...getFusedStatisticsNodes<Parameters>(graph, {
          id,
          zones,
          values: canonicalValues,
          validity: canonicalValidity,
          zoneCapacity,
          ignoredZone,
          cellCount,
          cellCounts: cellCountsView,
          valueCounts: valueCountsView,
          minimums: output.minimums,
          maximums: output.maximums,
          overflow
        })
      );
    } else if (sortedSums || momentPath) {
      for (const name of ['cellCounts', 'valueCounts'] as const) {
        if (!output[name] && countViews[name]) {
          nodes.push(
            ...new GPUGroupAggregation({
              id: `${id}-${name}`,
              keys: zones,
              mask: name === 'cellCounts' ? zoneMask! : valueMask!,
              output: countViews[name]!,
              operation: 'count'
            }).getCommandNodes(graph)
          );
        }
      }
    }

    for (const [name, operation] of COLUMN_OPERATIONS) {
      const view = output[name];
      if (!view) {
        continue;
      }
      if ((sortedSums || momentPath) && (name === 'sums' || name === 'means')) {
        continue;
      }
      if (fused && name !== 'sums' && name !== 'means') {
        continue;
      }
      const common = {
        id: `${id}-${name}`,
        keys: zones,
        mask: name === 'cellCounts' ? zoneMask! : valueMask!
      };
      const aggregation =
        operation === 'count'
          ? new GPUGroupAggregation({
              ...common,
              output: view as GraphDataView<'uint32'>,
              operation
            })
          : new GPUGroupAggregation({
              ...common,
              values: canonicalValues!,
              output: view as GraphDataView<'float32'>,
              operation
            });
      nodes.push(...aggregation.getCommandNodes(graph));
    }
    if (sortedSums || momentPath) {
      nodes.push(
        ...this.getMomentNodes(graph, {
          weighted,
          needsMeans,
          needsVariance,
          needsWeightSums,
          sortedSums,
          canonicalValues,
          canonicalValidity,
          valueMask,
          sortRows,
          contributions: contributions!,
          weightContributions,
          valueCounts: valueCountsView
        })
      );
    }
    return nodes;
  }

  /**
   * Sums, weight sums, means and variances from the per-cell contributions of the mask kernel:
   * one sorted segmented reduction (or `GPUGroupAggregation` sums with `sumOrder: 'atomic'`), a
   * mean kernel, then, for variances, a deviation pass, a second reduction and a finalize kernel.
   */
  private getMomentNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    plan: {
      weighted: boolean;
      needsMeans: boolean;
      needsVariance: boolean;
      needsWeightSums: boolean;
      sortedSums: boolean;
      canonicalValues?: GraphDataView<'float32'>;
      canonicalValidity?: GraphDataView<'uint32'>;
      valueMask?: GraphDataView<'uint32'>;
      sortRows?: GraphDataView<'uint32'>;
      contributions: GraphDataView<'float32'>;
      weightContributions?: GraphDataView<'float32'>;
      valueCounts?: GraphDataView<'uint32'>;
    }
  ): GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {zones, zoneCapacity, output} = props;
    const {width, height} = props.grid;
    const ignoredZone = props.ignoredZone ?? GPU_RASTER_ZONAL_STATISTICS_NO_ZONE;
    const {weighted, sortedSums} = plan;
    const nodes: GPUCommandNode<Parameters>[] = [];
    const sumsView =
      output.sums ??
      (plan.needsMeans
        ? createTransientView(graph, `${id}-sorted-sums`, 'float32', zoneCapacity)
        : undefined);
    const weightSumsView =
      output.weightSums ??
      (plan.needsWeightSums
        ? createTransientView(graph, `${id}-weight-sums`, 'float32', zoneCapacity)
        : undefined);
    const reductions = [
      ...(sumsView ? [{name: 'sums', contributions: plan.contributions, output: sumsView}] : []),
      ...(weightSumsView
        ? [{name: 'weights', contributions: plan.weightContributions!, output: weightSumsView}]
        : [])
    ];
    nodes.push(...this.getSumNodes(graph, `${id}-sorted`, `${id}-`, sortedSums, plan, reductions));

    const meansView =
      output.means ??
      (plan.needsVariance
        ? createTransientView(graph, `${id}-mean-values`, 'float32', zoneCapacity)
        : undefined);
    if (meansView) {
      const bindings: WGSLKernelBinding[] = [
        {name: 'sums', view: sumsView!, type: 'f32', access: 'read'},
        {name: 'valueCounts', view: plan.valueCounts!, type: 'u32', access: 'read'}
      ];
      if (weighted) {
        bindings.push({name: 'weightSums', view: weightSumsView!, type: 'f32', access: 'read'});
      }
      bindings.push({name: 'means', view: meansView, type: 'u32', access: 'read_write'});
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-means`,
          operation: 'GPURasterZonalStatistics',
          variant: weighted ? 'weighted-mean' : 'sorted-mean',
          bindings,
          invocationCount: zoneCapacity,
          body: weighted
            ? `let count = valueCounts[valueCountsOffset + index];
  let weightSum = weightSums[weightSumsOffset + index];
  var mean = 0x7fc00000u;
  // A zone whose valid cells have no area has no weighted mean.
  if (count > 0u && weightSum > 0.0) {
    mean = bitcast<u32>(sums[sumsOffset + index] / weightSum);
  }
  means[meansOffset + index] = mean;`
            : `let count = valueCounts[valueCountsOffset + index];
  var mean = 0x7fc00000u;
  if (count > 0u) {
    mean = bitcast<u32>(sums[sumsOffset + index] / f32(count));
  }
  means[meansOffset + index] = mean;`
        })
      );
    }
    if (!plan.needsVariance) {
      return nodes;
    }

    // Second pass: w (v - mean)^2 and w^2 per valid cell, reduced per zone.
    const cellCount = width * height;
    const deviations = createTransientView(graph, `${id}-deviations`, 'float32', cellCount);
    const weightSquares = weighted
      ? createTransientView(graph, `${id}-weight-squares`, 'float32', cellCount)
      : undefined;
    const deviationBindings: WGSLKernelBinding[] = [
      {name: 'zones', view: zones, type: 'u32', access: 'read'},
      {name: 'values', view: plan.canonicalValues!, type: 'f32', access: 'read'},
      {name: 'validity', view: plan.canonicalValidity!, type: 'u32', access: 'read'},
      {name: 'means', view: meansView!, type: 'f32', access: 'read'},
      {name: 'deviations', view: deviations, type: 'f32', access: 'read_write'}
    ];
    if (weightSquares) {
      deviationBindings.push(
        {name: 'weightSquares', view: weightSquares, type: 'f32', access: 'read_write'},
        {name: 'settings', view: props.parameters!, type: 'f32', access: 'read'}
      );
    }
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-deviations`,
        operation: 'GPURasterZonalStatistics',
        variant: weighted ? 'weighted-deviations' : 'deviations',
        bindings: deviationBindings,
        invocationCount: cellCount,
        declarations: `const CAPACITY: u32 = ${zoneCapacity}u;
const IGNORED: u32 = ${ignoredZone}u;
${IS_FINITE_VALUE_WGSL}
${weighted ? getCellWeightWGSL(width, height) : ''}`,
        body: `let zone = zones[zonesOffset + index];
  let value = values[valuesOffset + index];
  let isValid = zone < CAPACITY && zone != IGNORED && validity[validityOffset + index] != 0u &&
    isFiniteValue(value);
  let weight = ${weighted ? 'getCellWeight(index)' : '1.0'};
  var deviation = 0.0;
  var weightSquare = 0.0;
  if (isValid && weight > 0.0) {
    let difference = value - means[meansOffset + zone];
    deviation = weight * difference * difference;
    weightSquare = weight * weight;
  }
  deviations[deviationsOffset + index] = deviation;
  ${weightSquares ? 'weightSquares[weightSquaresOffset + index] = weightSquare;' : ''}`
      })
    );
    const squaredDeviationSums = createTransientView(
      graph,
      `${id}-squared-deviation-sums`,
      'float32',
      zoneCapacity
    );
    const weightSquareSums = weightSquares
      ? createTransientView(graph, `${id}-weight-square-sums`, 'float32', zoneCapacity)
      : undefined;
    nodes.push(
      ...this.getSumNodes(graph, `${id}-variance-sorted`, `${id}-variance-`, sortedSums, plan, [
        {name: 'deviations', contributions: deviations, output: squaredDeviationSums},
        ...(weightSquares && weightSquareSums
          ? [{name: 'weight-squares', contributions: weightSquares, output: weightSquareSums}]
          : [])
      ])
    );

    const finalizeBindings: WGSLKernelBinding[] = [
      {name: 'valueCounts', view: plan.valueCounts!, type: 'u32', access: 'read'},
      {name: 'squaredDeviationSums', view: squaredDeviationSums, type: 'f32', access: 'read'}
    ];
    if (weightSquareSums) {
      finalizeBindings.push(
        {name: 'weightSums', view: weightSumsView!, type: 'f32', access: 'read'},
        {name: 'weightSquareSums', view: weightSquareSums, type: 'f32', access: 'read'}
      );
    }
    for (const name of ['variances', 'standardDeviations'] as const) {
      if (output[name]) {
        finalizeBindings.push({name, view: output[name], type: 'u32', access: 'read_write'});
      }
    }
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-variances`,
        operation: 'GPURasterZonalStatistics',
        variant: weighted ? 'weighted-variances' : 'variances',
        bindings: finalizeBindings,
        invocationCount: zoneCapacity,
        // Reliability weights, as GPUZonalStatistics: sum w (v - m)^2 / (V1 - V2 / V1), which is
        // sum (v - m)^2 / (n - 1) for unit weights.
        body: `let valueCount = valueCounts[valueCountsOffset + index];
  var variance = 0x7fc00000u;
  var deviation = 0x7fc00000u;
  var denominator = 0.0;
  ${
    weightSquareSums
      ? `let weightSum = weightSums[weightSumsOffset + index];
  if (weightSum != 0.0) {
    denominator = weightSum - weightSquareSums[weightSquareSumsOffset + index] / weightSum;
  }`
      : 'denominator = f32(valueCount) - 1.0;'
  }
  if (valueCount >= 2u && denominator > 0.0) {
    let result = squaredDeviationSums[squaredDeviationSumsOffset + index] / denominator;
    variance = bitcast<u32>(result);
    deviation = bitcast<u32>(sqrt(result));
  }
  ${output.variances ? 'variances[variancesOffset + index] = variance;' : ''}
  ${output.standardDeviations ? 'standardDeviations[standardDeviationsOffset + index] = deviation;' : ''}`
      })
    );
    return nodes;
  }

  /** Per-zone sums of contributions: sorted segmented sums, or one atomic aggregation each. */
  private getSumNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    sortedId: string,
    atomicPrefix: string,
    sortedSums: boolean,
    plan: {sortRows?: GraphDataView<'uint32'>; valueMask?: GraphDataView<'uint32'>},
    reductions: {
      name: string;
      contributions: GraphDataView<'float32'>;
      output: GraphDataView<'float32'>;
    }[]
  ): GPUCommandNode<Parameters>[] {
    if (reductions.length === 0) {
      return [];
    }
    if (sortedSums) {
      return getSortedSegmentSumNodes<Parameters>(graph, {
        id: sortedId,
        operation: 'GPURasterZonalStatistics',
        segmentCount: this.props.zoneCapacity,
        segmentKeys: plan.sortRows!,
        reductions
      });
    }
    return reductions.flatMap(reduction =>
      new GPUGroupAggregation({
        id: `${atomicPrefix}${reduction.name}`,
        keys: this.props.zones,
        mask: plan.valueMask!,
        values: reduction.contributions,
        output: reduction.output,
        operation: 'sum'
      }).getCommandNodes(graph)
    );
  }
}

const FUSED_WORKGROUP_SIZE = 256;

/**
 * One-pass zone reduction of counts, minimums and maximums with workgroup-private tables.
 *
 * Each workgroup walks a contiguous tile of `cellsPerThread * 256` cells, accumulates into
 * `var<workgroup>` atomic tables (shared-memory atomics, no global traffic), and merges the non-empty
 * entries into the global result once per workgroup. Compared with one `GPUGroupAggregation` per
 * column, zones and values are read once instead of once per column, the zone and value masks are
 * never materialised, and global atomic traffic is at most `zoneCapacity` per tile instead of one
 * per cell. Minimums and maximums use order-preserving u32 keys (the same encoding as the core
 * aggregation), so counts, minimums and maximums stay exact.
 *
 * The table size is `zoneCapacity` words per requested column, so callers must check that it
 * fits in workgroup storage first.
 */
function getFusedStatisticsNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    zones: GraphDataView<'uint32'>;
    values?: GraphDataView<'float32'>;
    validity?: GraphDataView<'uint32'>;
    zoneCapacity: number;
    ignoredZone: number;
    cellCount: number;
    cellCounts?: GraphDataView<'uint32'>;
    valueCounts?: GraphDataView<'uint32'>;
    minimums?: GraphDataView<'float32'>;
    maximums?: GraphDataView<'float32'>;
    overflow?: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters>[] {
  const {id, zoneCapacity, cellCount} = props;
  const columns = [
    {name: 'cellCounts', view: props.cellCounts, initial: '0u', merge: 'atomicAdd'},
    {name: 'valueCounts', view: props.valueCounts, initial: '0u', merge: 'atomicAdd'},
    {name: 'minimums', view: props.minimums, initial: '0xffffffffu', merge: 'atomicMin'},
    {name: 'maximums', view: props.maximums, initial: '0u', merge: 'atomicMax'}
  ].filter(column => column.view) as {
    name: string;
    view: GraphDataView;
    initial: string;
    merge: string;
  }[];
  const needsValues = Boolean(props.valueCounts || props.minimums || props.maximums);
  const cellsPerThread = Math.min(64, Math.max(8, Math.ceil(zoneCapacity / 32)));
  const nodes: GPUCommandNode<Parameters>[] = [];

  // Global results start at the merge identity; the tables merge into them atomically.
  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-zone-reduce-init`,
      operation: 'GPURasterZonalStatistics',
      variant: 'zone-reduce-init',
      bindings: columns.map(column => ({
        name: column.name,
        view: column.view,
        type: 'u32' as const,
        access: 'read_write' as const
      })),
      invocationCount: zoneCapacity,
      body: columns
        .map(column => `${column.name}[${column.name}Offset + index] = ${column.initial};`)
        .join('\n  ')
    })
  );

  const bindings: WGSLKernelBinding[] = [
    {name: 'zones', view: props.zones, type: 'u32', access: 'read'}
  ];
  if (needsValues) {
    bindings.push(
      {name: 'values', view: props.values!, type: 'f32', access: 'read'},
      {name: 'validity', view: props.validity!, type: 'u32', access: 'read'}
    );
  }
  for (const column of columns) {
    bindings.push({
      name: column.name,
      view: column.view,
      type: 'atomic<u32>',
      access: 'read_write'
    });
  }
  if (props.overflow) {
    bindings.push({
      name: 'overflowFlag',
      view: props.overflow,
      type: 'atomic<u32>',
      access: 'read_write'
    });
  }
  const tables = columns
    .map(column => `var<workgroup> local_${column.name}: array<atomic<u32>, ${zoneCapacity}>;`)
    .join('\n');
  const hasMinimum = Boolean(props.minimums);
  const hasMaximum = Boolean(props.maximums);
  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-zone-reduce`,
      operation: 'GPURasterZonalStatistics',
      variant: 'zone-reduce',
      bindings,
      invocationCount: Math.ceil(cellCount / cellsPerThread),
      guardIndex: false,
      workgroupSize: FUSED_WORKGROUP_SIZE,
      declarations: `const CAPACITY: u32 = ${zoneCapacity}u;
const IGNORED: u32 = ${props.ignoredZone}u;
const CELL_COUNT: u32 = ${cellCount}u;
const CELLS_PER_THREAD: u32 = ${cellsPerThread}u;
${tables}
${IS_FINITE_VALUE_WGSL}
fn encodeOrderedFloat(value: f32) -> u32 {
  let bits = bitcast<u32>(value);
  return select(bits ^ 0x80000000u, ~bits, (bits & 0x80000000u) != 0u);
}`,
      body: `for (var slot = localInvocationIndex; slot < CAPACITY; slot += ${FUSED_WORKGROUP_SIZE}u) {
    ${columns.map(column => `atomicStore(&local_${column.name}[slot], ${column.initial});`).join('\n    ')}
  }
  workgroupBarrier();
  let tileStart = (index - localInvocationIndex) * CELLS_PER_THREAD;
  for (var step = 0u; step < CELLS_PER_THREAD; step++) {
    let cell = tileStart + step * ${FUSED_WORKGROUP_SIZE}u + localInvocationIndex;
    if (cell >= CELL_COUNT) {
      break;
    }
    let zone = zones[zonesOffset + cell];
    if (zone >= CAPACITY || zone == IGNORED) {
      ${props.overflow ? 'if (zone != IGNORED) {\n        atomicStore(&overflowFlag[overflowFlagOffset], 1u);\n      }' : ''}
      continue;
    }
    ${props.cellCounts ? 'atomicAdd(&local_cellCounts[zone], 1u);' : ''}
    ${
      needsValues
        ? `let value = values[valuesOffset + cell];
    if (validity[validityOffset + cell] != 0u && isFiniteValue(value)) {
      ${props.valueCounts ? 'atomicAdd(&local_valueCounts[zone], 1u);' : ''}
      ${hasMinimum || hasMaximum ? 'let key = encodeOrderedFloat(value);' : ''}
      ${hasMinimum ? 'atomicMin(&local_minimums[zone], key);' : ''}
      ${hasMaximum ? 'atomicMax(&local_maximums[zone], key);' : ''}
    }`
        : ''
    }
  }
  workgroupBarrier();
  for (var slot = localInvocationIndex; slot < CAPACITY; slot += ${FUSED_WORKGROUP_SIZE}u) {
    ${columns
      .map(
        column => `let partial_${column.name} = atomicLoad(&local_${column.name}[slot]);
    if (partial_${column.name} != ${column.initial}) {
      ${column.merge}(&${column.name}[${column.name}Offset + slot], partial_${column.name});
    }`
      )
      .join('\n    ')}
  }`
    })
  );

  // Decode the order-preserving keys in place; empty zones become NaN.
  const decoded = columns.filter(
    column => column.name === 'minimums' || column.name === 'maximums'
  );
  if (decoded.length > 0) {
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-zone-reduce-decode`,
        operation: 'GPURasterZonalStatistics',
        variant: 'zone-reduce-decode',
        bindings: decoded.map(column => ({
          name: column.name,
          view: column.view,
          type: 'u32' as const,
          access: 'read_write' as const
        })),
        invocationCount: zoneCapacity,
        declarations: `fn decodeOrderedKey(key: u32) -> u32 {
  return select(~key, key ^ 0x80000000u, (key & 0x80000000u) != 0u);
}`,
        body: decoded
          .map(
            column => `let key_${column.name} = ${column.name}[${column.name}Offset + index];
  ${column.name}[${column.name}Offset + index] = select(decodeOrderedKey(key_${column.name}), 0x7fc00000u, key_${column.name} == ${column.initial});`
          )
          .join('\n  ')
      })
    );
  }
  return nodes;
}
