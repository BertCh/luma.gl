// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUGroupAggregation,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {getMapGraphSortedSumNodes} from '../map-graph-sorted-sums';
import type {GPURasterBand} from '../../gpu-raster/index';
import {
  createMapGraphFillNode,
  createMapGraphKernelNode,
  type MapGraphKernelBinding
} from '../map-graph-kernels';
import type {GPUMapGraphRecipe} from '../map-graph-types';
import {validateGraphViewsBelongToGraph} from '../map-graph-utils';
import {
  getTerrainBandViews,
  getTerrainElevationNodes,
  validateTerrainBandBelongsToGraph,
  validateTerrainBuffersDistinct,
  validateTerrainGrid
} from '../terrain-analysis/terrain-analysis-utils';

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
};

/**
 * How per-zone sums (and means) are accumulated.
 * - `'atomic'`: float compare-exchange atomics. Fast, but the last bits of `sums` and `means`
 *   depend on accumulation order and can differ between encodings.
 * - `'sorted'`: cells are stably sorted by zone and reduced per zone in a fixed tree, so `sums`
 *   and `means` are bitwise reproducible across encodings on one device.
 */
export type GPURasterZonalStatisticsSumOrder = 'atomic' | 'sorted';

/**
 * Properties for {@link GPURasterZonalStatistics}.
 *
 * Compile-time: `width`, `height`, `zoneCapacity`, `ignoredZone`, the band format, calibration and
 * storage kind, which output columns and `overflow` exist, and every view length. Per-frame: the
 * contents of `zones` and `values` (and its validity), which callers rewrite between encodings.
 */
export type GPURasterZonalStatisticsProps = {
  /** Prefix for node and transient IDs. Defaults to `'raster-zonal-statistics'`. Compile-time. */
  id?: string;
  /** Grid width in cells. Compile-time. */
  width: number;
  /** Grid height in cells. Compile-time. */
  height: number;
  /**
   * Packed row-major zone ID per cell, for example rasterized administrative areas. Length is
   * `width * height`. Length is compile-time, contents are per-frame.
   */
  zones: GraphDataView<'uint32'>;
  /**
   * Value band, buffer or texture backed. Validity, `noDataValue`, `scale`, and `offset` are
   * honored. Band configuration is compile-time, contents are per-frame.
   */
  values: GPURasterBand;
  /**
   * Number of dense zones. Zone IDs in `[0, zoneCapacity)` aggregate. Every output column must
   * have exactly this length. Compile-time.
   */
  zoneCapacity: number;
  /**
   * Zone ID silently skipped, for example background `0`. Defaults to
   * {@link GPU_RASTER_ZONAL_STATISTICS_NO_ZONE}. Compile-time.
   */
  ignoredZone?: number;
  /**
   * Accumulation order for `sums` and `means`. Defaults to `'atomic'`. Compile-time.
   * `'sorted'` makes them bitwise reproducible at the cost of a radix sort of all cells, a scan,
   * and a gather per encoding. Counts, minimums, and maximums are exact either way.
   */
  sumOrder?: GPURasterZonalStatisticsSumOrder;
  /** Per-zone result columns, rewritten on every encoding. Which columns exist is compile-time. */
  output: GPURasterZonalStatisticsOutput;
  /**
   * Optional one-row flag rewritten on every encoding: 1 when any cell has a zone ID at or above
   * `zoneCapacity` that is not `ignoredZone`, otherwise 0. Presence is compile-time.
   */
  overflow?: GraphDataView<'uint32'>;
};

const COLUMN_OPERATIONS = [
  ['cellCounts', 'count'],
  ['valueCounts', 'count'],
  ['sums', 'sum'],
  ['means', 'mean'],
  ['minimums', 'min'],
  ['maximums', 'max']
] as const;

/**
 * Computes per-zone count, sum, mean, minimum, and maximum of a raster band, where zones are a
 * pre-rasterized grid of dense zone IDs.
 *
 * The value band is canonicalized (calibration, nodata, validity, non-finite samples), one kernel
 * derives a zone mask and a value mask (and the optional overflow flag), and one
 * `GPUGroupAggregation` runs per requested column. Only kernels and masks that a requested column
 * needs are scheduled. Empty zones produce counts of 0, sums of 0, and NaN means, minimums, and
 * maximums. Zones with cells but no valid values count in `cellCounts` only.
 *
 * Determinism: counts, minimums, and maximums are exact. With the default `sumOrder: 'atomic'`,
 * sums and means accumulate with float atomics, so their last bits depend on accumulation order
 * and can differ between runs. With `sumOrder: 'sorted'` the cells are stably sorted by zone ID,
 * scanned into zone offsets, and each zone is reduced with a fixed tree, so `sums` and `means`
 * (sum divided by `valueCounts`) are bitwise reproducible across encodings on one device. This
 * adds a sort over `width * height` cells, a scan, a gather, and one segmented reduction per
 * encoding, which is markedly slower than atomics on large grids. Results may still differ from a
 * sequential CPU sum in the last bits, and between devices.
 *
 * Non-goals:
 * - No median, percentiles, or standard deviation.
 * - No sparse or hashed zone IDs: IDs must be dense, so remap upstream (for example with
 *   `GPUBatchHashIndex`).
 * - No polygon rasterization: rasterize zones upstream.
 * - Single tile only: no cross-tile merge, although counts, sums, minimums, and maximums are
 *   mergeable by the caller.
 */
export class GPURasterZonalStatistics implements GPUMapGraphRecipe {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Stable recipe name. */
  readonly recipe = 'raster-zonal-statistics';
  /** Validated properties. */
  readonly props: GPURasterZonalStatisticsProps;
  /** Validated sum order. */
  readonly sumOrder: GPURasterZonalStatisticsSumOrder;

  constructor(props: GPURasterZonalStatisticsProps) {
    this.id = props.id ?? this.recipe;
    this.props = props;
    const {id} = this;
    this.sumOrder = props.sumOrder ?? 'atomic';
    if (this.sumOrder !== 'atomic' && this.sumOrder !== 'sorted') {
      throw new Error(`${id} sumOrder must be atomic or sorted`);
    }
    const cellCount = validateTerrainGrid(id, props.width, props.height);
    validatePackedUint32View(props.zones, `${id} zones`);
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
    const columns = COLUMN_OPERATIONS.filter(([name]) => props.output[name]);
    if (columns.length === 0) {
      throw new Error(`${id} requires at least one output`);
    }
    for (const [name, operation] of COLUMN_OPERATIONS) {
      const view = props.output[name];
      if (!view) {
        continue;
      }
      if (operation === 'count') {
        validatePackedUint32View(view, `${id} ${name}`);
      } else {
        validatePackedView(view, ['float32'], `${id} ${name}`);
      }
      if (view.length !== props.zoneCapacity) {
        throw new Error(`${id} ${name} must contain zoneCapacity rows`);
      }
    }
    if (props.overflow) {
      validatePackedUint32View(props.overflow, `${id} overflow`);
      if (props.overflow.length !== 1) {
        throw new Error(`${id} overflow must contain one uint32 row`);
      }
    }
    validateTerrainBuffersDistinct(
      id,
      [...COLUMN_OPERATIONS.map(([name]) => props.output[name]), props.overflow],
      [props.zones, ...getTerrainBandViews(props.values)]
    );
  }

  /** Returns band canonicalization, overflow reset, mask, and per-column aggregation nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {width, height, zones, zoneCapacity, output, overflow} = props;
    const ignoredZone = props.ignoredZone ?? GPU_RASTER_ZONAL_STATISTICS_NO_ZONE;
    validateTerrainBandBelongsToGraph(id, graph, props.values, []);
    validateGraphViewsBelongToGraph(id, graph, [
      zones,
      overflow,
      ...COLUMN_OPERATIONS.map(([name]) => output[name])
    ]);
    const cellCount = width * height;
    const sortedSums = this.sumOrder === 'sorted' && Boolean(output.sums || output.means);
    const needsZoneMask = Boolean(output.cellCounts) || sortedSums;
    const needsValueMask = COLUMN_OPERATIONS.some(
      ([name]) => name !== 'cellCounts' && output[name]
    );
    const nodes: GPUCommandNode<Parameters>[] = [];

    let canonicalValues: GraphDataView<'float32'> | undefined;
    let canonicalValidity: GraphDataView<'uint32'> | undefined;
    if (needsValueMask) {
      const source = getTerrainElevationNodes(
        graph,
        `${id}-values`,
        props.values,
        width,
        height,
        true
      );
      nodes.push(...source.nodes);
      canonicalValues = (source.band.storage as {values: GraphDataView<'float32'>}).values;
      canonicalValidity = source.band.validity;
    }

    if (overflow) {
      nodes.push(
        createMapGraphFillNode<Parameters>(graph, {
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
    const contributions = sortedSums
      ? createTransientView(graph, `${id}-contributions`, 'float32', cellCount)
      : undefined;
    if (zoneMask || valueMask || overflow) {
      const bindings: MapGraphKernelBinding[] = [
        {name: 'zones', view: zones, type: 'u32', access: 'read'}
      ];
      if (valueMask) {
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
          },
          {
            name: 'valueMask',
            view: valueMask,
            type: 'u32',
            access: 'read_write'
          }
        );
      }
      if (zoneMask) {
        bindings.push({
          name: 'zoneMask',
          view: zoneMask,
          type: 'u32',
          access: 'read_write'
        });
      }
      if (sortRows) {
        bindings.push(
          {
            name: 'sortRows',
            view: sortRows,
            type: 'u32',
            access: 'read_write'
          },
          {
            name: 'contributions',
            view: contributions!,
            type: 'f32',
            access: 'read_write'
          }
        );
      }
      if (overflow) {
        bindings.push({
          name: 'overflowFlag',
          view: overflow,
          type: 'atomic<u32>',
          access: 'read_write'
        });
      }
      nodes.push(
        createMapGraphKernelNode<Parameters>(graph, {
          id: `${id}-mask`,
          operation: 'GPURasterZonalStatistics',
          variant: 'mask',
          bindings,
          invocationCount: cellCount,
          declarations: `const CAPACITY: u32 = ${zoneCapacity}u;
const IGNORED: u32 = ${ignoredZone}u;
fn isFiniteValue(value: f32) -> bool { return (bitcast<u32>(value) & 0x7fffffffu) < 0x7f800000u; }`,
          body: `let zone = zones[zonesOffset + index];
  let inZone = zone < CAPACITY && zone != IGNORED;
  ${zoneMask ? 'zoneMask[zoneMaskOffset + index] = select(0u, 1u, inZone);' : ''}
  ${
    valueMask
      ? `let value = values[valuesOffset + index];
  valueMask[valueMaskOffset + index] = select(0u, 1u, inZone && validity[validityOffset + index] != 0u && isFiniteValue(value));`
      : ''
  }
  ${
    sortRows
      ? `sortRows[sortRowsOffset + index] = select(CAPACITY, zone, inZone);
  contributions[contributionsOffset + index] = select(0.0, values[valuesOffset + index], valueMask[valueMaskOffset + index] != 0u);`
      : ''
  }
  ${overflow ? 'if (zone >= CAPACITY && zone != IGNORED) {\n    atomicStore(&overflowFlag[overflowFlagOffset], 1u);\n  }' : ''}`
        })
      );
    }

    // Sorted sums need cell counts per zone (segment sizes) and, for means, value counts.
    const cellCountsView =
      output.cellCounts ??
      (sortedSums
        ? createTransientView(graph, `${id}-cell-counts`, 'uint32', zoneCapacity)
        : undefined);
    const valueCountsView =
      output.valueCounts ??
      (sortedSums && output.means
        ? createTransientView(graph, `${id}-value-counts`, 'uint32', zoneCapacity)
        : undefined);
    const countViews: Record<string, GraphDataView<'uint32'> | undefined> = {
      cellCounts: cellCountsView,
      valueCounts: valueCountsView
    };
    if (sortedSums) {
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
      if (sortedSums && (name === 'sums' || name === 'means')) {
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
    if (sortedSums) {
      const sumsView =
        output.sums ?? createTransientView(graph, `${id}-sorted-sums`, 'float32', zoneCapacity);
      nodes.push(
        ...getMapGraphSortedSumNodes<Parameters>(graph, {
          id: `${id}-sorted`,
          operation: 'GPURasterZonalStatistics',
          segmentCount: zoneCapacity,
          segmentKeys: sortRows!,
          segmentCounts: cellCountsView!,
          reductions: [{name: 'sums', contributions: contributions!, output: sumsView}]
        })
      );
      if (output.means) {
        nodes.push(
          createMapGraphKernelNode<Parameters>(graph, {
            id: `${id}-means`,
            operation: 'GPURasterZonalStatistics',
            variant: 'sorted-mean',
            bindings: [
              {name: 'sums', view: sumsView, type: 'f32', access: 'read'},
              {
                name: 'valueCounts',
                view: valueCountsView!,
                type: 'u32',
                access: 'read'
              },
              {
                name: 'means',
                view: output.means,
                type: 'u32',
                access: 'read_write'
              }
            ],
            invocationCount: zoneCapacity,
            body: `let count = valueCounts[valueCountsOffset + index];
  var mean = 0x7fc00000u;
  if (count > 0u) {
    mean = bitcast<u32>(sums[sumsOffset + index] / f32(count));
  }
  means[meansOffset + index] = mean;`
          })
        );
      }
    }
    return nodes;
  }
}
