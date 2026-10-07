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
import {getKeyGroupNodes, KEY_PAIR_INVALID} from '../spatial-weights/key-pair-grouping';
import {getBoundedKeyPairSortNodes} from './bounded-key-pair-sort';
import {
  type GPUSpatialWeights,
  validateGPUSpatialWeights
} from '../spatial-weights/spatial-weights';

const OPERATION = 'GPUArealInterpolation';
const INVALID = `${KEY_PAIR_INVALID}u`;
/** Consecutive cells folded into runs by one zone-count thread. */
const ZONE_COUNT_CHUNK = 8;

/** Normalization of the area-share weights. */
export type GPUArealInterpolationKind = 'extensive' | 'intensive';

/** Which area a share is divided by. */
export type GPUArealInterpolationDenominator = 'zone' | 'overlap';

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
 * Compile-time: `sourceCount`, `targetCount`, `denominator`, `mode`, view lengths
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
   * Which area the share is divided by. `'zone'` (default, tobler semantics) divides by the total
   * area of the source (extensive) or target (intensive), including parts that do not overlap the
   * other system, so mass that falls outside the other system is lost. `'overlap'` divides by the
   * part of the zone that overlaps the other system, so extensive shares of a source sum to 1 over
   * its targets and intensive shares of a target sum to 1 over its sources.
   */
  denominator?: GPUArealInterpolationDenominator;
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
  /** Optional caller-owned overlap area per slot (cell count, or the sum of `cellWeights`). */
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
 * with the other system (`'overlap'`, which conserves mass over the shared extent).
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

  constructor(props: GPUArealInterpolationProps) {
    const id = props.id ?? 'areal-interpolation';
    this.id = id;
    this.props = props;
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
    if (
      props.denominator !== undefined &&
      props.denominator !== 'zone' &&
      props.denominator !== 'overlap'
    ) {
      throw new Error(`${id} denominator must be 'zone' or 'overlap'`);
    }
    if (props.mode !== undefined && props.mode !== 'extensive' && props.mode !== 'intensive') {
      throw new Error(`${id} mode must be 'extensive' or 'intensive'`);
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
      ['totalPairs', props.totalPairs]
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
      props.categories?.output
    ];
  }

  /** Returns the pair, total, weight and category nodes in dependency order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    validateGraphViewsBelongToGraph(id, graph, [...this.getInputViews(), ...this.getOutputViews()]);
    const {sourceZones, targetZones, sourceCount, targetCount, weights} = props;
    const cellCount = sourceZones.length;
    const capacity = weights.neighbors.length;
    const slotCount = Math.min(capacity, cellCount);
    const denominator = props.denominator ?? 'zone';
    const mode = props.mode ?? 'extensive';
    const overlapOnly = denominator === 'overlap';

    const keyHigh = createTransientView(graph, `${id}-key-high`, 'uint32', cellCount);
    const keyLow = createTransientView(graph, `${id}-key-low`, 'uint32', cellCount);
    const fastPath = !props.cellWeights && props.unweightedFastPath !== false;
    const cellMask = fastPath
      ? undefined
      : createTransientView(graph, `${id}-cell-mask`, 'float32', cellCount);
    const pairSourceKey = overlapOnly
      ? createTransientView(graph, `${id}-source-key`, 'uint32', cellCount)
      : sourceZones;
    const pairTargetKey = overlapOnly
      ? createTransientView(graph, `${id}-target-key`, 'uint32', cellCount)
      : targetZones;
    const {cellWeights} = props;

    const nodes: GPUCommandNode<Parameters>[] = [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-cell-keys`,
        operation: OPERATION,
        variant: `cell-keys-${overlapOnly ? 'overlap' : 'zone'}${cellWeights ? '-masked' : ''}${fastPath ? '-fast' : ''}`,
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
          ...(overlapOnly
            ? [
                {
                  name: 'sourceKey',
                  view: pairSourceKey,
                  type: 'u32' as const,
                  access: 'read_write' as const
                },
                {
                  name: 'targetKey',
                  view: pairTargetKey,
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
  ${
    overlapOnly
      ? `sourceKey[sourceKeyOffset + index] = select(${INVALID}, source, valid);
  targetKey[targetKeyOffset + index] = select(${INVALID}, targetRow, valid);`
      : ''
  }`
      })
    ];

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
    const sourceTotals = zoneTotals('source', pairSourceKey, sourceCount);
    const targetTotals = zoneTotals('target', pairTargetKey, targetCount);

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
