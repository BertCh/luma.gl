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
import {getSortedSegmentSumNodes} from '../../utils/sorted-segment-sums';
import type {GPURasterBand} from '../index';
import {
  createFillNode,
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../utils/wgsl-kernel-nodes';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {validateGraphViewsBelongToGraph} from '../../utils/gpu-contributor-utils';
import {
  getTerrainBandViews,
  getTerrainElevationNodes,
  validateTerrainBandBelongsToGraph,
  validateTerrainBuffersDistinct,
  validateTerrainGrid
} from '../../gpu-terrain/terrain-analysis/terrain-analysis-utils';

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
   * Accumulation order for `sums` and `means`. Defaults to `'sorted'`. Compile-time.
   * `'sorted'` makes them bitwise reproducible and avoids same-address atomic contention, at the
   * cost of a radix sort of all cells, a scan, and a gather per encoding. `'atomic'` skips the sort
   * and only wins for many scattered zones. Counts, minimums, and maximums are exact either way.
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
 * - No median, percentiles, or standard deviation.
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
    const needsValueColumn = Boolean(
      output.valueCounts || output.sums || output.means || output.minimums || output.maximums
    );
    // Counts, minimums and maximums are reduced together in ONE pass with workgroup-private
    // tables (see `getFusedStatisticsNodes`) whenever the tables fit in workgroup memory. Without
    // that pass every column re-reads zones and a mask through its own `GPUGroupAggregation`.
    const fusedColumnCount = [
      output.cellCounts,
      output.valueCounts || (sortedSums && output.means),
      output.minimums,
      output.maximums
    ].filter(Boolean).length;
    const fused =
      fusedColumnCount > 0 &&
      fusedColumnCount * 4 * zoneCapacity <= graph.device.limits.maxComputeWorkgroupStorageSize;
    // Masks only feed the per-column aggregations that the fused pass replaces.
    const needsZoneMask = !fused && Boolean(output.cellCounts);
    const needsValueMask = fused
      ? !sortedSums && Boolean(output.sums || output.means)
      : COLUMN_OPERATIONS.some(([name]) => name !== 'cellCounts' && output[name]);
    const nodes: GPUCommandNode<Parameters>[] = [];

    let canonicalValues: GraphDataView<'float32'> | undefined;
    let canonicalValidity: GraphDataView<'uint32'> | undefined;
    if (needsValueColumn) {
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
    const contributions = sortedSums
      ? createTransientView(graph, `${id}-contributions`, 'float32', cellCount)
      : undefined;
    // The fused pass also raises the overflow flag, so the mask kernel only does when it is absent.
    const maskOverflow = fused ? undefined : overflow;
    if (zoneMask || valueMask || sortRows || maskOverflow) {
      const needsValues = Boolean(valueMask || sortRows);
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
      if (valueMask) {
        bindings.push({
          name: 'valueMask',
          view: valueMask,
          type: 'u32',
          access: 'read_write'
        });
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
      if (maskOverflow) {
        bindings.push({
          name: 'overflowFlag',
          view: maskOverflow,
          type: 'atomic<u32>',
          access: 'read_write'
        });
      }
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
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
    needsValues
      ? `let value = values[valuesOffset + index];
  let isValid = inZone && validity[validityOffset + index] != 0u && isFiniteValue(value);`
      : ''
  }
  ${valueMask ? 'valueMask[valueMaskOffset + index] = select(0u, 1u, isValid);' : ''}
  ${
    sortRows
      ? `sortRows[sortRowsOffset + index] = select(CAPACITY, zone, inZone);
  contributions[contributionsOffset + index] = select(0.0, value, isValid);`
      : ''
  }
  ${maskOverflow ? 'if (zone >= CAPACITY && zone != IGNORED) {\n    atomicStore(&overflowFlag[overflowFlagOffset], 1u);\n  }' : ''}`
        })
      );
    }

    // Sorted sums find segment sizes by binary search, so cell counts are only built on request;
    // means still need value counts.
    const cellCountsView = output.cellCounts;
    const valueCountsView =
      output.valueCounts ??
      (sortedSums && output.means
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
    } else if (sortedSums) {
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
    if (sortedSums) {
      const sumsView =
        output.sums ?? createTransientView(graph, `${id}-sorted-sums`, 'float32', zoneCapacity);
      nodes.push(
        ...getSortedSegmentSumNodes<Parameters>(graph, {
          id: `${id}-sorted`,
          operation: 'GPURasterZonalStatistics',
          segmentCount: zoneCapacity,
          segmentKeys: sortRows!,
          reductions: [{name: 'sums', contributions: contributions!, output: sumsView}]
        })
      );
      if (output.means) {
        nodes.push(
          createWGSLKernelNode<Parameters>(graph, {
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
      declarations: `const CAPACITY: u32 = ${zoneCapacity}u;
const IGNORED: u32 = ${props.ignoredZone}u;
const CELL_COUNT: u32 = ${cellCount}u;
const CELLS_PER_THREAD: u32 = ${cellsPerThread}u;
${tables}
fn isFiniteValue(value: f32) -> bool { return (bitcast<u32>(value) & 0x7fffffffu) < 0x7f800000u; }
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
