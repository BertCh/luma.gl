// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUHistogram,
  GPUPointSpatialFilter,
  GPUReduction,
  GPUVisibilityWorkflow,
  GraphVectorView,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {
  createMapGraphFillNode,
  createMapGraphKernelNode,
  createMapGraphPublishNode,
  type MapGraphKernelBinding
} from '../map-graph-kernels';
import type {
  GPUMapGraphCompactOutput,
  GPUMapGraphPositions2D,
  GPUMapGraphRecipe,
  GPUMapGraphUint32Rows
} from '../map-graph-types';
import {
  getGraphViewChunks,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateMapGraphCompactOutput
} from '../map-graph-utils';
import {GPUPickRegionMask} from './gpu-pick-region-mask';
import {GPURegionMask, REGION_STATISTICS_WGSL_HELPERS} from './gpu-region-mask';
import {
  getGridCandidateNodes,
  getGridMaskScatterNode,
  validateGridIndexProps,
  type GridCandidates
} from './region-statistics-grid';
import {
  GPU_REGION_STATISTICS_HEADER_LENGTH,
  type GPURegionHistogramProps,
  type GPURegionPolygon,
  type GPURegionRadius,
  type GPURegionRectangle,
  type GPURegionSelection,
  type GPURegionStatisticsGridIndex
} from './region-statistics-types';

const OPERATION = 'GPURegionStatistics';
const MAXIMUM_BIN_COUNT = 65536;

/**
 * Properties for {@link GPURegionStatistics}.
 *
 * Per-frame: region bounds, circle, lasso vertices and vertex count, screen transform, a view
 * histogram domain, and every input buffer's contents. Compile-time: the row count, the lasso
 * vertex capacity, `histogram.binCount`, a literal domain, `batchIndex`, the selection kind, which
 * optional views exist, `output.ids.length`, and `spatialIndex`.
 */
export type GPURegionStatisticsProps = {
  /** ID prefix for every node and transient. Defaults to `'region-statistics'`. */
  id?: string;
  /** Region input. */
  selection: GPURegionSelection;
  /** Packed 2D positions. Required for `rectangle`, `polygon`, and `radius`. */
  positions?: GPUMapGraphPositions2D;
  /** Optional `float32` value column aligned with the source rows. */
  values?: GraphDataView<'float32'> | GraphVectorView<'float32'>;
  /** Optional stable IDs used in `output.ids`. Row indices are used when omitted. */
  sourceIds?: GPUMapGraphUint32Rows;
  /** Optional histogram. Requires `values`. */
  histogram?: GPURegionHistogramProps;
  /** Optional caller-owned 0/1 selection mask for highlighting. Not allowed with `kind: 'mask'`. */
  outputMask?: GraphDataView<'uint32'>;
  /** Optional caller-owned bounded stable-ID result. `output.ids.length` must be at least 1. */
  output?: GPUMapGraphCompactOutput;
  /**
   * Optional packed one-row view that receives the clamped selected count, the same value as
   * `output.count` (`min(selected, output.ids.length)`), rewritten every encoding. Typically the
   * `instanceCount` word of an indirect draw record, imported as a 1-row view at an element offset
   * so an instanced draw of the selected rows needs no readback. Requires `output`: the draw
   * indexes rows through `output.ids`, so a recipe with no id list has nothing to draw, and
   * statistics-only recipes read the selected count from `summary` instead. Must not alias any
   * other output or input.
   */
  drawInstanceCount?: GraphDataView<'uint32'>;
  /** Caller-owned packed summary of `8 + binCount` uint32 words. */
  summary: GraphDataView<'uint32'>;
  /**
   * Compile-time. Gathers candidates from a uniform grid index instead of testing every row. Only
   * for world-space `rectangle`, `radius`, and `polygon` selections with packed `positions`,
   * `values`, and `sourceIds`. See {@link GPURegionStatisticsGridIndex} for the identity guarantee
   * and the new `candidatesTruncated` summary flag.
   */
  spatialIndex?: GPURegionStatisticsGridIndex;
};

/**
 * Selects points inside a per-frame rectangle, circle, lasso, or pick region and reduces them
 * into counts, value statistics, a histogram, and stable selected IDs.
 *
 * Every statistic lands in one small `summary` buffer so the application reads back a single
 * bounded copy, for example through `GPURegionStatisticsReadback`.
 */
export class GPURegionStatistics implements GPUMapGraphRecipe {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Stable recipe name. */
  readonly recipe = 'region-statistics';
  /** Validated properties. */
  readonly props: GPURegionStatisticsProps;
  /** Resolved source row count. */
  readonly rowCount: number;

  constructor(props: GPURegionStatisticsProps) {
    this.id = props.id ?? this.recipe;
    this.props = props;
    const {id} = this;
    const {selection, positions, values, sourceIds, histogram, outputMask, output, summary} = props;
    const rowCount =
      selection.kind === 'mask'
        ? selection.mask.length
        : (positions?.length ?? values?.length ?? sourceIds?.length ?? outputMask?.length);
    if (rowCount === undefined) {
      throw new Error(`${id} cannot infer the source row count`);
    }
    if (rowCount === 0) {
      throw new Error(`${id} requires at least one source row`);
    }
    this.rowCount = rowCount;
    if (
      (selection.kind === 'rectangle' ||
        selection.kind === 'polygon' ||
        selection.kind === 'radius') &&
      !positions
    ) {
      throw new Error(`${id} positions are required for ${selection.kind} selections`);
    }
    for (const [name, view] of [
      ['positions', positions],
      ['values', values],
      ['sourceIds', sourceIds],
      ['outputMask', outputMask],
      ['selection.mask', selection.kind === 'mask' ? selection.mask : undefined]
    ] as const) {
      if (view && view.length !== rowCount) {
        throw new Error(`${id} ${name} length must equal the source row count`);
      }
    }
    for (const chunk of values ? getGraphViewChunks(values) : []) {
      validatePackedView(chunk, ['float32'], `${id} values`);
    }
    if (histogram) {
      if (!values) {
        throw new Error(`${id} histogram requires values`);
      }
      if (
        !Number.isSafeInteger(histogram.binCount) ||
        histogram.binCount < 1 ||
        histogram.binCount > MAXIMUM_BIN_COUNT
      ) {
        throw new Error(`${id} histogram.binCount must be an integer in [1, ${MAXIMUM_BIN_COUNT}]`);
      }
      const {domain} = histogram;
      if (Array.isArray(domain)) {
        const [minimum, maximum] = domain as readonly number[];
        if (!Number.isFinite(minimum) || !Number.isFinite(maximum) || minimum > maximum) {
          throw new Error(`${id} histogram.domain must be finite with min <= max`);
        }
      } else if (domain && domain !== 'selection') {
        validatePackedView(domain as GraphDataView, ['float32'], `${id} histogram.domain`);
        if ((domain as GraphDataView).length !== 2) {
          throw new Error(`${id} histogram.domain view must hold two float32 values`);
        }
      }
    }
    if (selection.kind === 'radius') {
      validatePackedView(selection.circle, ['float32'], `${id} circle`);
      if (selection.circle.length !== 3) {
        throw new Error(`${id} circle must contain three float32 values`);
      }
    }
    if (selection.kind === 'mask' && outputMask) {
      throw new Error(`${id} outputMask cannot be combined with a mask selection`);
    }
    if (outputMask) {
      validatePackedUint32View(outputMask, `${id} outputMask`);
    }
    validatePackedUint32View(summary, `${id} summary`);
    const binCount = histogram?.binCount ?? 0;
    if (summary.length !== GPU_REGION_STATISTICS_HEADER_LENGTH + binCount) {
      throw new Error(
        `${id} summary must hold ${GPU_REGION_STATISTICS_HEADER_LENGTH + binCount} uint32 words`
      );
    }
    if (output) {
      validateMapGraphCompactOutput(id, output);
      if (output.ids.length < 1) {
        throw new Error(`${id} output.ids must hold at least one row`);
      }
    }
    if (props.drawInstanceCount) {
      validatePackedUint32View(props.drawInstanceCount, `${id} drawInstanceCount`);
      if (props.drawInstanceCount.length < 1) {
        throw new Error(`${id} drawInstanceCount must contain one uint32 row`);
      }
      if (!output) {
        throw new Error(`${id} drawInstanceCount requires output`);
      }
    }
    if (props.spatialIndex) {
      validateGridIndexProps(id, {...props, spatialIndex: props.spatialIndex});
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [
        summary,
        outputMask,
        output?.ids,
        output?.count,
        output?.overflow,
        output?.totalCount,
        props.drawInstanceCount
      ],
      getStatisticsInputs(props)
    );
  }

  /** Returns selection, count, reduction, histogram, compaction, and summary nodes in order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, rowCount} = this;
    const {selection, positions, values, sourceIds, histogram, outputMask, output, summary} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      summary,
      outputMask,
      output?.ids,
      output?.count,
      output?.overflow,
      output?.totalCount,
      props.drawInstanceCount,
      ...getStatisticsInputs(props)
    ]);
    const nodes: GPUCommandNode<Parameters>[] = [];
    const regionOverflow = createTransientView(graph, `${id}-region-overflow`, 'uint32', 1);
    const {spatialIndex} = props;
    const needsSourceMask = !spatialIndex || Boolean(outputMask || output);
    const mask =
      selection.kind === 'mask'
        ? selection.mask
        : (outputMask ??
          (needsSourceMask
            ? createTransientView(graph, `${id}-mask`, 'uint32', rowCount)
            : undefined));
    let candidates: GridCandidates<Parameters> | undefined;

    // Stage 1: selection mask.
    if (spatialIndex) {
      candidates = getGridCandidateNodes(graph, {
        id,
        selection: selection as GPURegionRectangle | GPURegionRadius | GPURegionPolygon,
        positions: positions as GPUMapGraphPositions2D,
        values: values as GraphDataView<'float32'> | undefined,
        spatialIndex,
        rowCount,
        regionOverflow
      });
      nodes.push(...candidates.nodes);
      if (mask) {
        // O(rowCount): clear the source-aligned mask, then scatter the candidate mask into it.
        nodes.push(
          createMapGraphFillNode<Parameters>(graph, {
            id: `${id}-mask-clear`,
            operation: OPERATION,
            view: mask,
            type: 'u32',
            value: '0u'
          }),
          getGridMaskScatterNode(graph, {id: `${id}-mask-scatter`, candidates, mask, rowCount})
        );
      }
    } else
      switch (selection.kind) {
        case 'rectangle':
        case 'radius':
          if (selection.kind === 'radius' || !selection.screenTransform) {
            nodes.push(
              ...new GPUPointSpatialFilter({
                id: `${id}-filter`,
                positions: positions as GPUMapGraphPositions2D,
                kind: selection.kind === 'radius' ? 'radius' : 'bounds',
                query: selection.kind === 'radius' ? selection.circle : selection.bounds,
                outputMask: mask as GraphDataView<'uint32'>,
                overflow: regionOverflow
              }).getCommandNodes(graph)
            );
            break;
          }
          nodes.push(
            ...new GPURegionMask({
              id: `${id}-region`,
              positions: positions as GPUMapGraphPositions2D,
              region: selection,
              outputMask: mask as GraphDataView<'uint32'>,
              overflow: regionOverflow
            }).getCommandNodes(graph)
          );
          break;
        case 'polygon':
          nodes.push(
            ...new GPURegionMask({
              id: `${id}-region`,
              positions: positions as GPUMapGraphPositions2D,
              region: selection,
              outputMask: mask as GraphDataView<'uint32'>,
              overflow: regionOverflow
            }).getCommandNodes(graph)
          );
          break;
        case 'pick-region':
          nodes.push(
            ...new GPUPickRegionMask({
              id: `${id}-pick`,
              result: selection.result,
              batchIndex: selection.batchIndex,
              outputMask: mask as GraphDataView<'uint32'>,
              overflow: regionOverflow
            }).getCommandNodes(graph)
          );
          break;
        case 'mask':
          nodes.push(
            createMapGraphFillNode<Parameters>(graph, {
              id: `${id}-region-overflow-clear`,
              operation: OPERATION,
              view: regionOverflow,
              type: 'u32',
              value: '0u'
            })
          );
          break;
      }

    // Stage 2: selected and finite-value counts, plus the selected-and-finite value mask.
    // With a spatial index the stages below run over the candidate slots instead of source rows.
    const statisticsRowCount = candidates ? candidates.capacity : rowCount;
    const statisticsMask = candidates ? candidates.mask : (mask as GraphDataView<'uint32'>);
    const statisticsValues = candidates ? candidates.values : values;
    const counts = createTransientView(graph, `${id}-counts`, 'uint32', candidates ? 3 : 2);
    const valueMask = statisticsValues
      ? createTransientView(graph, `${id}-value-mask`, 'uint32', statisticsRowCount)
      : undefined;
    nodes.push(
      createMapGraphFillNode<Parameters>(graph, {
        id: `${id}-counts-clear`,
        operation: OPERATION,
        view: counts,
        type: 'u32',
        value: '0u'
      })
    );
    if (candidates) {
      nodes.push(candidates.getStatusNode(counts));
    }
    nodes.push(
      ...getCountNodes(
        graph,
        id,
        statisticsMask,
        counts,
        statisticsRowCount,
        statisticsValues,
        valueMask
      )
    );

    // Stage 3: value statistics.
    let extentValue: GraphDataView<'float32'> | undefined;
    let sumValue: GraphDataView<'float32'> | undefined;
    if (statisticsValues && valueMask) {
      sumValue = createTransientView(graph, `${id}-sum-value`, 'float32', 1);
      extentValue = createTransientView(graph, `${id}-extent-value`, 'float32', 2);
      nodes.push(
        ...new GPUReduction({
          id: `${id}-sum`,
          input: statisticsValues,
          mask: valueMask,
          output: sumValue,
          operation: 'sum'
        }).getCommandNodes(graph)
      );
      nodes.push(
        ...new GPUReduction({
          id: `${id}-extent`,
          input: statisticsValues,
          mask: valueMask,
          output: extentValue,
          operation: 'extent'
        }).getCommandNodes(graph)
      );
    }

    // Stage 4: histogram.
    let bins: GraphDataView<'uint32'> | undefined;
    let binsTotal: GraphDataView<'uint32'> | undefined;
    if (histogram && statisticsValues && valueMask && extentValue) {
      bins = createTransientView(graph, `${id}-bins`, 'uint32', histogram.binCount);
      binsTotal = createTransientView(graph, `${id}-bins-total-value`, 'uint32', 1);
      const domain =
        histogram.domain === undefined || histogram.domain === 'selection'
          ? extentValue
          : histogram.domain;
      nodes.push(
        ...new GPUHistogram<'float32'>({
          id: `${id}-histogram`,
          input: statisticsValues,
          mask: valueMask,
          output: bins,
          domain
        }).getCommandNodes(graph)
      );
      nodes.push(
        ...new GPUReduction({
          id: `${id}-bins-total`,
          input: bins,
          output: binsTotal,
          operation: 'sum'
        }).getCommandNodes(graph)
      );
    }

    // Stage 5: stable selected IDs, clamped to the caller's capacity.
    let selectionTotal: GraphDataView<'uint32'> | undefined;
    if (output) {
      const selectionIds = createTransientView(graph, `${id}-selection-ids`, 'uint32', rowCount);
      selectionTotal = createTransientView(graph, `${id}-selection-total`, 'uint32', 1);
      nodes.push(
        ...new GPUVisibilityWorkflow({
          id: `${id}-visibility`,
          predicates: [{kind: 'selection', mask: mask as GraphDataView<'uint32'>}],
          output: selectionIds,
          count: selectionTotal,
          sourceIds
        }).getCommandNodes(graph)
      );
      nodes.push(
        createMapGraphPublishNode<Parameters>(graph, {
          id: `${id}-selection-finalize`,
          operation: OPERATION,
          totalCount: selectionTotal,
          compactIds: selectionIds,
          output,
          overflowSources: candidates
            ? [regionOverflow, candidates.candidateFlag]
            : [regionOverflow],
          extraCounts: props.drawInstanceCount ? [props.drawInstanceCount] : []
        })
      );
    }

    // Stage 6: pack the summary.
    nodes.push(
      getSummaryNode(graph, {
        id: `${id}-summary`,
        counts,
        hasCandidateFlag: Boolean(candidates),
        regionOverflow,
        sumValue,
        extentValue,
        bins,
        binsTotal,
        selectionTotal,
        selectionCapacity: output?.ids.length ?? 0,
        summary
      })
    );
    return nodes;
  }
}

/** Returns every read-only view used by a region statistics recipe. */
function getStatisticsInputs(
  props: GPURegionStatisticsProps
): (GraphDataView | GraphVectorView | undefined)[] {
  const {selection, histogram} = props;
  const selectionViews: (GraphDataView | undefined)[] = [];
  switch (selection.kind) {
    case 'rectangle':
      selectionViews.push(selection.bounds, selection.screenTransform);
      break;
    case 'polygon':
      selectionViews.push(selection.vertices, selection.vertexCount, selection.screenTransform);
      break;
    case 'radius':
      selectionViews.push(selection.circle);
      break;
    case 'pick-region':
      selectionViews.push(selection.result);
      break;
    case 'mask':
      selectionViews.push(selection.mask);
      break;
  }
  const domain =
    histogram?.domain && typeof histogram.domain === 'object' && !Array.isArray(histogram.domain)
      ? (histogram.domain as GraphDataView)
      : undefined;
  const {index} = props.spatialIndex ?? {};
  return [
    props.positions,
    props.values,
    props.sourceIds,
    domain,
    ...selectionViews,
    ...(index ? [index.cellOffsets, index.objectIds, index.count, index.overflow] : [])
  ];
}

/** Builds the fused count pass, one node per nonempty values chunk. */
function getCountNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  mask: GraphDataView<'uint32'>,
  counts: GraphDataView<'uint32'>,
  rowCount: number,
  values: GraphDataView<'float32'> | GraphVectorView<'float32'> | undefined,
  valueMask: GraphDataView<'uint32'> | undefined
): GPUCommandNode<Parameters>[] {
  const chunks = values ? getGraphViewChunks(values) : [undefined];
  const isVector = values instanceof GraphVectorView;
  const nodes: GPUCommandNode<Parameters>[] = [];
  let rowStart = 0;
  for (const [chunkIndex, chunk] of chunks.entries()) {
    const chunkLength = chunk ? chunk.length : rowCount;
    if (chunkLength > 0) {
      const bindings: MapGraphKernelBinding[] = [
        {name: 'mask', view: mask, type: 'u32', access: 'read'}
      ];
      if (chunk && valueMask) {
        bindings.push({name: 'values', view: chunk, type: 'f32', access: 'read'});
        bindings.push({name: 'valueMask', view: valueMask, type: 'u32', access: 'read_write'});
      }
      bindings.push({name: 'counts', view: counts, type: 'atomic<u32>', access: 'read_write'});
      nodes.push(
        createMapGraphKernelNode<Parameters>(graph, {
          id: isVector ? `${id}-count-chunk-${chunkIndex}` : `${id}-count`,
          operation: OPERATION,
          variant: 'count',
          bindings,
          invocationCount: chunkLength,
          guardIndex: false,
          declarations: `const ROW_START: u32 = ${rowStart}u;
var<workgroup> localSelectedCount: atomic<u32>;
var<workgroup> localValueCount: atomic<u32>;
${REGION_STATISTICS_WGSL_HELPERS}`,
          body: `if (localInvocationIndex == 0u) {
    atomicStore(&localSelectedCount, 0u);
    atomicStore(&localValueCount, 0u);
  }
  workgroupBarrier();
  if (index < INVOCATION_COUNT) {
    let row = ROW_START + index;
    let selected = mask[maskOffset + row] != 0u;
    ${
      chunk
        ? `let value = values[valuesOffset + index];
    let valued = selected && isFiniteValue(value);
    valueMask[valueMaskOffset + row] = select(0u, 1u, valued);
    if (valued) { atomicAdd(&localValueCount, 1u); }`
        : ''
    }
    if (selected) { atomicAdd(&localSelectedCount, 1u); }
  }
  workgroupBarrier();
  if (localInvocationIndex == 0u) {
    let selectedTotal = atomicLoad(&localSelectedCount);
    let valueTotal = atomicLoad(&localValueCount);
    if (selectedTotal > 0u) { atomicAdd(&counts[countsOffset], selectedTotal); }
    if (valueTotal > 0u) { atomicAdd(&counts[countsOffset + 1u], valueTotal); }
  }`
        })
      );
    }
    rowStart += chunkLength;
  }
  return nodes;
}

/** Builds the node that packs counts, value statistics, flags, and bins into the summary. */
function getSummaryNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    counts: GraphDataView<'uint32'>;
    hasCandidateFlag: boolean;
    regionOverflow: GraphDataView<'uint32'>;
    sumValue?: GraphDataView<'float32'>;
    extentValue?: GraphDataView<'float32'>;
    bins?: GraphDataView<'uint32'>;
    binsTotal?: GraphDataView<'uint32'>;
    selectionTotal?: GraphDataView<'uint32'>;
    selectionCapacity: number;
    summary: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  const {sumValue, extentValue, bins, binsTotal, selectionTotal} = props;
  const bindings: MapGraphKernelBinding[] = [
    {name: 'counts', view: props.counts, type: 'u32', access: 'read'},
    {name: 'regionOverflow', view: props.regionOverflow, type: 'u32', access: 'read'}
  ];
  if (sumValue) bindings.push({name: 'sumValue', view: sumValue, type: 'f32', access: 'read'});
  if (extentValue) {
    bindings.push({name: 'extentValue', view: extentValue, type: 'f32', access: 'read'});
  }
  if (bins) bindings.push({name: 'bins', view: bins, type: 'u32', access: 'read'});
  if (binsTotal) bindings.push({name: 'binsTotal', view: binsTotal, type: 'u32', access: 'read'});
  if (selectionTotal) {
    bindings.push({name: 'selectionTotal', view: selectionTotal, type: 'u32', access: 'read'});
  }
  bindings.push({name: 'summary', view: props.summary, type: 'u32', access: 'read_write'});
  const binCount = bins?.length ?? 0;
  return createMapGraphKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'summary',
    bindings,
    invocationCount: Math.max(binCount, 1),
    declarations: `const BIN_COUNT: u32 = ${binCount}u;
const SELECTION_CAPACITY: u32 = ${props.selectionCapacity}u;`,
    body: `${bins ? 'if (index < BIN_COUNT) { summary[summaryOffset + 8u + index] = bins[binsOffset + index]; }' : ''}
  if (index == 0u) {
    let selectedCount = counts[countsOffset];
    let valueCount = counts[countsOffset + 1u];
    let hasValues = valueCount > 0u;
    ${
      sumValue && extentValue
        ? `let sum = select(0.0, sumValue[sumValueOffset], hasValues);
    let mean = select(0.0, sum / f32(max(valueCount, 1u)), hasValues);
    let minimum = select(0.0, extentValue[extentValueOffset], hasValues);
    let maximum = select(0.0, extentValue[extentValueOffset + 1u], hasValues);`
        : `let sum = 0.0;
    let mean = 0.0;
    let minimum = 0.0;
    let maximum = 0.0;`
    }
    ${
      binsTotal
        ? `let binnedCount = binsTotal[binsTotalOffset];
    let outsideCount = select(0u, valueCount - binnedCount, valueCount > binnedCount);`
        : 'let outsideCount = 0u;'
    }
    let selectionTruncated = ${selectionTotal ? 'selectionTotal[selectionTotalOffset] > SELECTION_CAPACITY' : 'false'};
    let regionTruncated = regionOverflow[regionOverflowOffset] != 0u;
    let candidatesTruncated = ${props.hasCandidateFlag ? 'counts[countsOffset + 2u] != 0u' : 'false'};
    summary[summaryOffset + 0u] = selectedCount;
    summary[summaryOffset + 1u] = valueCount;
    summary[summaryOffset + 2u] = bitcast<u32>(sum);
    summary[summaryOffset + 3u] = bitcast<u32>(mean);
    summary[summaryOffset + 4u] = bitcast<u32>(minimum);
    summary[summaryOffset + 5u] = bitcast<u32>(maximum);
    summary[summaryOffset + 6u] = outsideCount;
    summary[summaryOffset + 7u] = select(0u, 1u, selectionTruncated) | select(0u, 2u, regionTruncated) | select(0u, 4u, candidatesTruncated);
  }`
  });
}
