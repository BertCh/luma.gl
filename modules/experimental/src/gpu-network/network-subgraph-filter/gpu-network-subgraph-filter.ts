// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUScan,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCompactOutput} from '../../utils/gpu-contributor-types';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {
  createFillNode,
  createWGSLKernelNode,
  createPublishNode,
  type WGSLKernelBinding
} from '../../utils/wgsl-kernel-nodes';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateCompactOutput
} from '../../utils/gpu-contributor-utils';
import {TIME_WORDS_WGSL} from '../../gpu-dataframe/time-window-filter/time-words';
import {
  getGPUNetworkSubgraphFilterParameterLength,
  GPU_NETWORK_SUBGRAPH_FILTER_COUNT_LENGTH,
  GPU_NETWORK_SUBGRAPH_FILTER_COUNT_WORD,
  GPU_NETWORK_SUBGRAPH_FILTER_PARAMETER_STRIDE
} from './network-subgraph-filter-parameters';

/** Number of words of the `timeWordParameters` view, see `getGPUTimeWindowWordParameterValues`. */
const TIME_WORD_PARAMETER_LENGTH = 8;
/** Columns per kernel: the 8-storage-binding limit minus the mask and the parameter view. */
const COLUMNS_PER_KERNEL = 6;
const OPERATION = 'GPUNetworkSubgraphFilter';

/**
 * Plain caller-owned induced CSR over the original vertex ids.
 *
 * Row `v` of `offsets`/`neighbors` holds the live slots of vertex `v` in their original order;
 * dead vertices and dead slots have no entries. `offsets` is exact (`nodeCount + 1` rows) even when
 * `neighbors` overflows. Bind the views to another recipe as its forward CSR.
 */
export type GPUNetworkSubgraphFilterInducedCSR = {
  /** `nodeCount + 1` rows. Row `nodeCount` is the live slot total. */
  offsets: GraphDataView<'uint32'>;
  /** Capacity for the live slots; at most `neighbors.length` are written. */
  neighbors: GraphDataView<'uint32'>;
  /** Optional original CSR slot id of every induced slot, aligned with `neighbors`. */
  sourceSlots?: GraphDataView<'uint32'>;
  /** One row receiving 1 when the live slot total exceeds `neighbors.length`, else 0. */
  overflow: GraphDataView<'uint32'>;
};

/** Caller-owned outputs of {@link GPUNetworkSubgraphFilter}. */
export type GPUNetworkSubgraphFilterOutput = {
  /** One `uint32` per vertex: 1 live, 0 dead. Feeds `GPUNetworkStatistics.vertexMask` and `filterMask`. */
  vertexMask: GraphDataView<'uint32'>;
  /** One `uint32` per CSR slot: 1 live, 0 dead. Feeds `GPUNetworkStatistics.edgeMask` and `filterMask`. */
  edgeMask: GraphDataView<'uint32'>;
  /** Optional four-word counts, see {@link GPU_NETWORK_SUBGRAPH_FILTER_COUNT_WORD}. */
  counts?: GraphDataView<'uint32'>;
  /** Optional compact live vertex ids in ascending order, for indirect draws. */
  liveVertices?: GPUCompactOutput;
  /** Optional compact live CSR slot ids in ascending order, for indirect draws. */
  liveEdgeSlots?: GPUCompactOutput;
  /** Optional compacted induced CSR. */
  inducedCSR?: GPUNetworkSubgraphFilterInducedCSR;
};

/**
 * Properties for {@link GPUNetworkSubgraphFilter}.
 *
 * Compile-time: node count, `directed`, `dropIsolated`, the number of columns, which optional views
 * exist and the output capacities. Per-frame: every range in `parameters`, the time-word window, the
 * column, CSR and mask contents.
 */
export type GPUNetworkSubgraphFilterProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'network-subgraph-filter'`. */
  id?: string;
  /** Forward CSR row offsets with `nodeCount + 1` rows. */
  offsets: GraphDataView<'uint32'>;
  /**
   * Forward CSR destination per slot. Indices `>= nodeCount` are dead slots. Undirected graphs list
   * each edge in both directions (a self-loop once), like `GPUNetworkStatistics`.
   */
  neighbors: GraphDataView<'uint32'>;
  /**
   * Treat slots as directed edges. Undirected (default) graphs list each edge in both slots; this
   * affects `counts.liveEdgeCount` and `pairUndirectedSlots`.
   */
  directed?: boolean;
  /**
   * Undirected only. When true (default) an edge is live only if both of its slots pass their own
   * predicates (caller mask, columns and time gates), so both slots always agree even when the
   * caller's per-slot values differ. The j-th slot `u -> v` pairs with the j-th slot `v -> u`
   * (parallel edges pair in row order); self-loops and slots without a reverse slot use their own
   * result. The pairing scans two rows per slot, O(degree) per slot, so for hub-heavy graphs whose
   * columns are already symmetric pass false to skip it. Ignored when `directed`.
   */
  pairUndirectedSlots?: boolean;
  /** Optional caller vertex mask, nonzero = live, ANDed with the vertex predicates. */
  vertexMask?: GraphDataView<'uint32'>;
  /** Optional caller per-slot mask, nonzero = live, ANDed with the edge predicates. */
  edgeMask?: GraphDataView<'uint32'>;
  /** Float32 vertex columns, one row per vertex. At most 64 together with the edge columns. */
  vertexColumns?: readonly GraphDataView<'float32'>[];
  /** Float32 edge columns, one row per CSR slot. */
  edgeColumns?: readonly GraphDataView<'float32'>[];
  /** Optional float32 time per CSR slot, gated by the closed `edgeTimeWindow` record. */
  edgeTimes?: GraphDataView<'float32'>;
  /** Optional exact Int64 time per CSR slot as `uint32x2` `(low, high)` words. */
  edgeTimeWords?: GraphDataView<'uint32x2'>;
  /**
   * Per-frame `float32` ranges, required with any vertex column, edge column or `edgeTimes`; see
   * {@link getGPUNetworkSubgraphFilterParameterValues} for the layout and range rules.
   */
  parameters?: GraphDataView<'float32'>;
  /**
   * Per-frame closed window over `edgeTimeWords`, eight `uint32` words packed by
   * `getGPUTimeWindowWordParameterValues`. Required with `edgeTimeWords`.
   */
  timeWordParameters?: GraphDataView<'uint32'>;
  /**
   * Also kill vertices left with no live incident slot (in or out, a self-loop counts), judged
   * after the edge predicates and endpoint rule. Vertices that were already dead stay dead. Default
   * false.
   */
  dropIsolated?: boolean;
  /** Caller-owned outputs. */
  output: GPUNetworkSubgraphFilterOutput;
};

/**
 * Turns attribute and time predicates on vertex and edge rows into a consistent induced subgraph.
 *
 * Vertex `v` is live when its caller mask (if any) is nonzero and every enabled vertex range
 * accepts its column value. Slot `u -> v` is live when its caller mask (if any) is nonzero, every
 * enabled edge range and time gate accepts it, `v < nodeCount`, and both `u` and `v` are live. With
 * `dropIsolated`, vertices with no live incident slot are then killed; this never kills a live
 * slot, because a live slot's endpoints are incident to it. Masks use 1 and 0, matching the
 * nonzero-is-live rule of `GPUNetworkStatistics` and the deck-arrow-layers `filterMask`.
 *
 * Undirected graphs list each edge in both slots and, by default, an edge is live only if both slots
 * pass (see `pairUndirectedSlots`), so `edgeMask` is always symmetric. With pairing off, slots agree
 * only if the caller's columns and masks do. The vertex rule is symmetric by construction.
 *
 * Ranges live in a float32 `parameters` view and never recompile (see
 * {@link getGPUNetworkSubgraphFilterParameterValues}); all arithmetic is integer or comparison, so
 * results are exact and deterministic. Optional outputs: counts, compact live vertex and slot ids
 * (stable ascending, via `GPUScan`), and an induced CSR from a prefix sum over live degree.
 *
 * Add the recipe to one graph at a time. It owns no GPU resources.
 */
export class GPUNetworkSubgraphFilter implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUNetworkSubgraphFilterProps;
  /** Number of nodes, `offsets.length - 1`. */
  readonly nodeCount: number;
  /** Number of CSR slots, `neighbors.length`. */
  readonly slotCount: number;

  constructor(props: GPUNetworkSubgraphFilterProps) {
    this.id = props.id ?? 'network-subgraph-filter';
    this.props = props;
    const {id} = this;
    validatePackedView(props.offsets, ['uint32'], `${id} offsets`);
    validatePackedView(props.neighbors, ['uint32'], `${id} neighbors`);
    this.nodeCount = props.offsets.length - 1;
    this.slotCount = props.neighbors.length;
    const {nodeCount, slotCount} = this;
    if (nodeCount < 1) {
      throw new Error(`${id} offsets must contain at least two rows`);
    }
    if (slotCount < 1) {
      throw new Error(`${id} neighbors must contain at least one row`);
    }
    const vertexColumns = props.vertexColumns ?? [];
    const edgeColumns = props.edgeColumns ?? [];
    if (vertexColumns.length + edgeColumns.length > 64) {
      throw new Error(`${id} supports at most 64 columns`);
    }
    const lengthChecks: [string, GraphDataView | undefined, number, string][] = [
      ['vertexMask', props.vertexMask, nodeCount, 'node'],
      ['edgeMask', props.edgeMask, slotCount, 'neighbor slot'],
      ['edgeTimes', props.edgeTimes, slotCount, 'neighbor slot'],
      ['edgeTimeWords', props.edgeTimeWords, slotCount, 'neighbor slot'],
      ['output.vertexMask', props.output.vertexMask, nodeCount, 'node'],
      ['output.edgeMask', props.output.edgeMask, slotCount, 'neighbor slot']
    ];
    for (const [index, column] of vertexColumns.entries()) {
      validatePackedView(column, ['float32'], `${id} vertexColumns[${index}]`);
      lengthChecks.push([`vertexColumns[${index}]`, column, nodeCount, 'node']);
    }
    for (const [index, column] of edgeColumns.entries()) {
      validatePackedView(column, ['float32'], `${id} edgeColumns[${index}]`);
      lengthChecks.push([`edgeColumns[${index}]`, column, slotCount, 'neighbor slot']);
    }
    for (const [name, view, length, unit] of lengthChecks) {
      if (!view) continue;
      if (name === 'edgeTimeWords') {
        validatePackedView(view, ['uint32x2'], `${id} ${name}`);
      } else if (name === 'edgeTimes') {
        validatePackedView(view, ['float32'], `${id} ${name}`);
      } else if (!name.startsWith('vertexColumns') && !name.startsWith('edgeColumns')) {
        validatePackedView(view, ['uint32'], `${id} ${name}`);
      }
      if (view.length !== length) {
        throw new Error(`${id} ${name} must contain one row per ${unit}`);
      }
    }
    const parameterLength = getGPUNetworkSubgraphFilterParameterLength({
      vertexColumnCount: vertexColumns.length,
      edgeColumnCount: edgeColumns.length,
      hasEdgeTimes: Boolean(props.edgeTimes)
    });
    if (parameterLength > 0) {
      if (!props.parameters) {
        throw new Error(`${id} parameters are required with columns or edgeTimes`);
      }
      validatePackedView(props.parameters, ['float32'], `${id} parameters`);
      if (props.parameters.length < parameterLength) {
        throw new Error(`${id} parameters must contain at least ${parameterLength} float32 rows`);
      }
    }
    if (props.edgeTimeWords) {
      if (!props.timeWordParameters) {
        throw new Error(`${id} timeWordParameters are required with edgeTimeWords`);
      }
      validatePackedView(props.timeWordParameters, ['uint32'], `${id} timeWordParameters`);
      if (props.timeWordParameters.length < TIME_WORD_PARAMETER_LENGTH) {
        throw new Error(`${id} timeWordParameters must contain at least 8 uint32 rows`);
      }
    }
    const {output} = props;
    if (output.counts) {
      validatePackedView(output.counts, ['uint32'], `${id} output.counts`);
      if (output.counts.length < GPU_NETWORK_SUBGRAPH_FILTER_COUNT_LENGTH) {
        throw new Error(`${id} output.counts must contain at least 4 uint32 rows`);
      }
    }
    for (const [name, compact] of [
      ['liveVertices', output.liveVertices],
      ['liveEdgeSlots', output.liveEdgeSlots]
    ] as const) {
      if (compact) {
        validateCompactOutput(`${id} output.${name}`, compact);
      }
    }
    const csr = output.inducedCSR;
    if (csr) {
      validatePackedView(csr.offsets, ['uint32'], `${id} output.inducedCSR.offsets`);
      validatePackedView(csr.neighbors, ['uint32'], `${id} output.inducedCSR.neighbors`);
      validatePackedView(csr.overflow, ['uint32'], `${id} output.inducedCSR.overflow`);
      if (csr.offsets.length !== nodeCount + 1) {
        throw new Error(`${id} output.inducedCSR.offsets must contain nodeCount + 1 rows`);
      }
      if (csr.overflow.length < 1) {
        throw new Error(`${id} output.inducedCSR.overflow must contain one uint32 row`);
      }
      if (csr.sourceSlots) {
        validatePackedView(csr.sourceSlots, ['uint32'], `${id} output.inducedCSR.sourceSlots`);
        if (csr.sourceSlots.length !== csr.neighbors.length) {
          throw new Error(`${id} output.inducedCSR.sourceSlots must match neighbors in length`);
        }
      }
    }
    validateGraphOutputsDisjointFromInputs(id, this.getOutputViews(), this.getInputViews());
  }

  private getInputViews(): (GraphDataView | undefined)[] {
    const {props} = this;
    return [
      props.offsets,
      props.neighbors,
      props.vertexMask,
      props.edgeMask,
      ...(props.vertexColumns ?? []),
      ...(props.edgeColumns ?? []),
      props.edgeTimes,
      props.edgeTimeWords,
      props.parameters,
      props.timeWordParameters
    ];
  }

  private getOutputViews(): (GraphDataView | undefined)[] {
    const {output} = this.props;
    return [
      output.vertexMask,
      output.edgeMask,
      output.counts,
      output.liveVertices?.ids,
      output.liveVertices?.count,
      output.liveVertices?.overflow,
      output.liveVertices?.totalCount,
      output.liveEdgeSlots?.ids,
      output.liveEdgeSlots?.count,
      output.liveEdgeSlots?.overflow,
      output.liveEdgeSlots?.totalCount,
      output.inducedCSR?.offsets,
      output.inducedCSR?.neighbors,
      output.inducedCSR?.sourceSlots,
      output.inducedCSR?.overflow
    ];
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, nodeCount, slotCount} = this;
    validateGraphViewsBelongToGraph(id, graph, [...this.getInputViews(), ...this.getOutputViews()]);
    const {output} = props;
    const vertexColumns = props.vertexColumns ?? [];
    const edgeColumns = props.edgeColumns ?? [];
    const nodes: GPUCommandNode<Parameters>[] = [];
    const word = GPU_NETWORK_SUBGRAPH_FILTER_COUNT_WORD;
    const stride = GPU_NETWORK_SUBGRAPH_FILTER_PARAMETER_STRIDE;
    const kernel = (
      step: string,
      variant: string,
      bindings: WGSLKernelBinding[],
      invocationCount: number,
      body: string,
      declarations?: string
    ) =>
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-${step}`,
          operation: OPERATION,
          variant,
          bindings,
          invocationCount,
          body,
          declarations
        })
      );

    // Range predicates: AND chunks of columns into a mask. One invocation owns one row.
    const addRangeNodes = (
      step: string,
      mask: GraphDataView<'uint32'>,
      columns: readonly GraphDataView<'float32'>[],
      firstRecord: number,
      rowCount: number
    ) => {
      for (let first = 0; first < columns.length; first += COLUMNS_PER_KERNEL) {
        const chunk = columns.slice(first, first + COLUMNS_PER_KERNEL);
        const bindings: WGSLKernelBinding[] = [
          {name: 'maskOut', view: mask, type: 'u32', access: 'read_write'},
          {
            name: 'parameters',
            view: props.parameters!,
            type: 'f32',
            access: 'read'
          },
          ...chunk.map((view, i): WGSLKernelBinding => ({
            name: `column${i}`,
            view,
            type: 'f32',
            access: 'read'
          }))
        ];
        const tests = chunk
          .map(
            (_, i) => `{
    let base = parametersOffset + ${(firstRecord + first + i) * stride}u;
    if (parameters[base + 2u] != 0.0) {
      let value = column${i}[column${i}Offset + index];
      live = live && value >= parameters[base] && value < parameters[base + 1u];
    }
  }`
          )
          .join('\n  ');
        kernel(
          `${step}-columns-${first / COLUMNS_PER_KERNEL}`,
          step,
          bindings,
          rowCount,
          `var live = maskOut[maskOutOffset + index] != 0u;
  ${tests}
  maskOut[maskOutOffset + index] = select(0u, 1u, live);`
        );
      }
    };

    // Vertex predicates.
    kernel(
      'vertex-init',
      'vertex-init',
      [
        {
          name: 'maskOut',
          view: output.vertexMask,
          type: 'u32',
          access: 'read_write'
        },
        ...(props.vertexMask
          ? [
              {
                name: 'maskIn',
                view: props.vertexMask,
                type: 'u32' as const,
                access: 'read' as const
              }
            ]
          : [])
      ],
      nodeCount,
      props.vertexMask
        ? 'maskOut[maskOutOffset + index] = select(0u, 1u, maskIn[maskInOffset + index] != 0u);'
        : 'maskOut[maskOutOffset + index] = 1u;'
    );
    addRangeNodes('vertex', output.vertexMask, vertexColumns, 0, nodeCount);

    // Edge predicates run on `ownMask`: the output itself, or a scratch copy when undirected slots
    // are paired (a pair reads the other slot's own result, so nothing may be overwritten early).
    const pairSlots = !props.directed && props.pairUndirectedSlots !== false;
    const ownMask = pairSlots
      ? createTransientView(graph, `${id}-own-edge-mask`, 'uint32', slotCount)
      : output.edgeMask;
    // Edge predicates.
    kernel(
      'edge-init',
      'edge-init',
      [
        {
          name: 'maskOut',
          view: ownMask,
          type: 'u32',
          access: 'read_write'
        },
        ...(props.edgeMask
          ? [
              {
                name: 'maskIn',
                view: props.edgeMask,
                type: 'u32' as const,
                access: 'read' as const
              }
            ]
          : [])
      ],
      slotCount,
      props.edgeMask
        ? 'maskOut[maskOutOffset + index] = select(0u, 1u, maskIn[maskInOffset + index] != 0u);'
        : 'maskOut[maskOutOffset + index] = 1u;'
    );
    addRangeNodes('edge', ownMask, edgeColumns, vertexColumns.length, slotCount);
    if (props.edgeTimes) {
      const record = (vertexColumns.length + edgeColumns.length) * stride;
      kernel(
        'edge-times',
        'edge-times',
        [
          {
            name: 'maskOut',
            view: ownMask,
            type: 'u32',
            access: 'read_write'
          },
          {
            name: 'parameters',
            view: props.parameters!,
            type: 'f32',
            access: 'read'
          },
          {name: 'times', view: props.edgeTimes, type: 'f32', access: 'read'}
        ],
        slotCount,
        `let base = parametersOffset + ${record}u;
  if (parameters[base + 2u] != 0.0) {
    let time = times[timesOffset + index];
    let live = time >= parameters[base] && time <= parameters[base + 1u];
    if (!live) {
      maskOut[maskOutOffset + index] = 0u;
    }
  }`
      );
    }
    if (props.edgeTimeWords) {
      kernel(
        'edge-time-words',
        'edge-time-words',
        [
          {
            name: 'maskOut',
            view: ownMask,
            type: 'u32',
            access: 'read_write'
          },
          {
            name: 'timeWindow',
            view: props.timeWordParameters!,
            type: 'u32',
            access: 'read'
          },
          {
            name: 'timeWords',
            view: props.edgeTimeWords,
            type: 'u32',
            access: 'read'
          }
        ],
        slotCount,
        `let windowStart = vec2<u32>(timeWindow[timeWindowOffset], timeWindow[timeWindowOffset + 1u]);
  let windowEnd = vec2<u32>(timeWindow[timeWindowOffset + 2u], timeWindow[timeWindowOffset + 3u]);
  let startFraction = bitcast<f32>(timeWindow[timeWindowOffset + 4u]);
  let endFraction = bitcast<f32>(timeWindow[timeWindowOffset + 5u]);
  let time = vec2<u32>(
    timeWords[timeWordsOffset + 2u * index],
    timeWords[timeWordsOffset + 2u * index + 1u]
  );
  let live = isTimeWordsAtLeast(time, 0.0, windowStart, startFraction) &&
    isTimeWordsAtLeast(windowEnd, endFraction, time, 0.0);
  if (!live) {
    maskOut[maskOutOffset + index] = 0u;
  }`,
        TIME_WORDS_WGSL
      );
    }

    // Induced edges: a slot also needs both endpoints and, when paired, its reverse slot. A row's
    // slots belong to one invocation, so writes never race.
    const pairing = pairSlots
      ? `var live = sourceLive && ownMask[ownMaskOffset + slot] != 0u && neighbor < NODE_COUNT;
    if (live && neighbor != index) {
      // The j-th slot index -> neighbor pairs with the j-th slot neighbor -> index. A missing
      // reverse slot leaves the slot's own result in force.
      var ordinal = 0u;
      for (var earlier = rowBegin; earlier < slot; earlier++) {
        ordinal += select(0u, 1u, neighbors[neighborsOffset + earlier] == neighbor);
      }
      let reverseEnd = min(offsets[offsetsOffset + neighbor + 1u], SLOT_COUNT);
      var seen = 0u;
      for (var reverse = offsets[offsetsOffset + neighbor]; reverse < reverseEnd; reverse++) {
        if (neighbors[neighborsOffset + reverse] == index) {
          if (seen == ordinal) {
            live = ownMask[ownMaskOffset + reverse] != 0u;
            break;
          }
          seen++;
        }
      }
    }`
      : `var live = sourceLive && edgeMask[edgeMaskOffset + slot] != 0u && neighbor < NODE_COUNT;`;
    kernel(
      'edge-endpoints',
      'edge-endpoints',
      [
        {name: 'offsets', view: props.offsets, type: 'u32', access: 'read'},
        {
          name: 'neighbors',
          view: props.neighbors,
          type: 'u32',
          access: 'read'
        },
        {
          name: 'vertexMask',
          view: output.vertexMask,
          type: 'u32',
          access: 'read'
        },
        ...(pairSlots
          ? [
              {
                name: 'ownMask',
                view: ownMask,
                type: 'u32' as const,
                access: 'read' as const
              }
            ]
          : []),
        {
          name: 'edgeMask',
          view: output.edgeMask,
          type: 'u32',
          access: 'read_write'
        }
      ],
      nodeCount,
      `let sourceLive = vertexMask[vertexMaskOffset + index] != 0u;
  let rowBegin = offsets[offsetsOffset + index];
  let rowEnd = min(offsets[offsetsOffset + index + 1u], SLOT_COUNT);
  for (var slot = rowBegin; slot < rowEnd; slot++) {
    let neighbor = neighbors[neighborsOffset + slot];
    ${pairing}
    if (live) {
      live = vertexMask[vertexMaskOffset + neighbor] != 0u;
    }
    edgeMask[edgeMaskOffset + slot] = select(0u, 1u, live);
  }`,
      `const NODE_COUNT: u32 = ${nodeCount}u;
const SLOT_COUNT: u32 = ${slotCount}u;`
    );

    if (props.dropIsolated) {
      const incident = createTransientView(graph, `${id}-incident`, 'uint32', nodeCount);
      nodes.push(
        createFillNode<Parameters>(graph, {
          id: `${id}-zero-incident`,
          operation: OPERATION,
          view: incident,
          type: 'u32',
          value: '0u'
        })
      );
      kernel(
        'mark-incident',
        'mark-incident',
        [
          {name: 'offsets', view: props.offsets, type: 'u32', access: 'read'},
          {
            name: 'neighbors',
            view: props.neighbors,
            type: 'u32',
            access: 'read'
          },
          {
            name: 'edgeMask',
            view: output.edgeMask,
            type: 'u32',
            access: 'read'
          },
          {
            name: 'incident',
            view: incident,
            type: 'atomic<u32>',
            access: 'read_write'
          }
        ],
        nodeCount,
        `let rowBegin = offsets[offsetsOffset + index];
  let rowEnd = min(offsets[offsetsOffset + index + 1u], SLOT_COUNT);
  for (var slot = rowBegin; slot < rowEnd; slot++) {
    if (edgeMask[edgeMaskOffset + slot] != 0u) {
      atomicStore(&incident[incidentOffset + index], 1u);
      atomicStore(&incident[incidentOffset + neighbors[neighborsOffset + slot]], 1u);
    }
  }`,
        `const SLOT_COUNT: u32 = ${slotCount}u;`
      );
      kernel(
        'drop-isolated',
        'drop-isolated',
        [
          {
            name: 'vertexMask',
            view: output.vertexMask,
            type: 'u32',
            access: 'read_write'
          },
          {name: 'incident', view: incident, type: 'u32', access: 'read'}
        ],
        nodeCount,
        `if (incident[incidentOffset + index] == 0u) {
    vertexMask[vertexMaskOffset + index] = 0u;
  }`
      );
    }

    // Counts.
    if (output.counts) {
      nodes.push(
        createFillNode<Parameters>(graph, {
          id: `${id}-zero-counts`,
          operation: OPERATION,
          view: output.counts,
          type: 'u32',
          value: '0u',
          componentCount: GPU_NETWORK_SUBGRAPH_FILTER_COUNT_LENGTH
        })
      );
      kernel(
        'counts',
        'counts',
        [
          {name: 'offsets', view: props.offsets, type: 'u32', access: 'read'},
          {
            name: 'neighbors',
            view: props.neighbors,
            type: 'u32',
            access: 'read'
          },
          {
            name: 'vertexMask',
            view: output.vertexMask,
            type: 'u32',
            access: 'read'
          },
          {
            name: 'edgeMask',
            view: output.edgeMask,
            type: 'u32',
            access: 'read'
          },
          {
            name: 'counts',
            view: output.counts,
            type: 'atomic<u32>',
            access: 'read_write'
          }
        ],
        nodeCount,
        `if (vertexMask[vertexMaskOffset + index] != 0u) {
    atomicAdd(&counts[countsOffset + ${word.liveVertexCount}u], 1u);
  }
  let rowBegin = offsets[offsetsOffset + index];
  let rowEnd = min(offsets[offsetsOffset + index + 1u], SLOT_COUNT);
  var liveCount = 0u;
  var selfCount = 0u;
  for (var slot = rowBegin; slot < rowEnd; slot++) {
    if (edgeMask[edgeMaskOffset + slot] != 0u) {
      liveCount++;
      if (neighbors[neighborsOffset + slot] == index) {
        selfCount++;
      }
    }
  }
  if (liveCount > 0u) {
    atomicAdd(&counts[countsOffset + ${word.liveSlotCount}u], liveCount);
  }
  if (selfCount > 0u) {
    atomicAdd(&counts[countsOffset + ${word.selfLoopSlotCount}u], selfCount);
  }`,
        `const SLOT_COUNT: u32 = ${slotCount}u;`
      );
      kernel(
        'counts-finish',
        'counts-finish',
        [
          {
            name: 'counts',
            view: output.counts,
            type: 'u32',
            access: 'read_write'
          }
        ],
        1,
        `let slots = counts[countsOffset + ${word.liveSlotCount}u];
  let selfLoops = counts[countsOffset + ${word.selfLoopSlotCount}u];
  counts[countsOffset + ${word.liveEdgeCount}u] =
    ${props.directed ? 'slots' : '(slots - selfLoops) / 2u + selfLoops'};`
      );
    }

    // Stable compact ids: exclusive scan of the 0/1 mask, scatter, publish.
    const addCompaction = (
      step: string,
      flags: GraphDataView<'uint32'>,
      compact: GPUCompactOutput
    ) => {
      const rowCount = flags.length;
      const offsets = createTransientView(graph, `${id}-${step}-offsets`, 'uint32', rowCount);
      const total = createTransientView(graph, `${id}-${step}-total`, 'uint32', 1);
      nodes.push(
        ...new GPUScan({
          id: `${id}-${step}-scan`,
          input: flags,
          output: offsets
        }).getCommandNodes(graph)
      );
      kernel(
        `${step}-scatter`,
        'scatter',
        [
          {name: 'flags', view: flags, type: 'u32', access: 'read'},
          {name: 'prefix', view: offsets, type: 'u32', access: 'read'},
          {
            name: 'idsOut',
            view: compact.ids,
            type: 'u32',
            access: 'read_write'
          },
          {name: 'totalOut', view: total, type: 'u32', access: 'read_write'}
        ],
        rowCount,
        `let live = flags[flagsOffset + index] != 0u;
  let position = prefix[prefixOffset + index];
  if (live && position < CAPACITY) {
    idsOut[idsOutOffset + position] = index;
  }
  if (index == ROW_COUNT - 1u) {
    totalOut[totalOutOffset] = position + select(0u, 1u, live);
  }`,
        `const CAPACITY: u32 = ${compact.ids.length}u;
const ROW_COUNT: u32 = ${rowCount}u;`
      );
      nodes.push(
        createPublishNode<Parameters>(graph, {
          id: `${id}-${step}-publish`,
          operation: OPERATION,
          totalCount: total,
          output: compact
        })
      );
    };
    if (output.liveVertices) addCompaction('live-vertices', output.vertexMask, output.liveVertices);
    if (output.liveEdgeSlots) addCompaction('live-slots', output.edgeMask, output.liveEdgeSlots);

    // Induced CSR: live degree per vertex, exclusive scan to offsets, scatter live slots.
    const csr = output.inducedCSR;
    if (csr) {
      const degree = createTransientView(graph, `${id}-live-degree`, 'uint32', nodeCount + 1);
      kernel(
        'live-degree',
        'live-degree',
        [
          {name: 'offsets', view: props.offsets, type: 'u32', access: 'read'},
          {
            name: 'edgeMask',
            view: output.edgeMask,
            type: 'u32',
            access: 'read'
          },
          {
            name: 'degreeOut',
            view: degree,
            type: 'u32',
            access: 'read_write'
          }
        ],
        nodeCount + 1,
        `var liveCount = 0u;
  if (index < NODE_COUNT) {
    let rowBegin = offsets[offsetsOffset + index];
    let rowEnd = min(offsets[offsetsOffset + index + 1u], SLOT_COUNT);
    for (var slot = rowBegin; slot < rowEnd; slot++) {
      liveCount += select(0u, 1u, edgeMask[edgeMaskOffset + slot] != 0u);
    }
  }
  degreeOut[degreeOutOffset + index] = liveCount;`,
        `const NODE_COUNT: u32 = ${nodeCount}u;
const SLOT_COUNT: u32 = ${slotCount}u;`
      );
      nodes.push(
        ...new GPUScan({
          id: `${id}-induced-scan`,
          input: degree,
          output: csr.offsets
        }).getCommandNodes(graph)
      );
      const scatterBindings: WGSLKernelBinding[] = [
        {name: 'offsets', view: props.offsets, type: 'u32', access: 'read'},
        {
          name: 'neighbors',
          view: props.neighbors,
          type: 'u32',
          access: 'read'
        },
        {
          name: 'edgeMask',
          view: output.edgeMask,
          type: 'u32',
          access: 'read'
        },
        {
          name: 'inducedOffsets',
          view: csr.offsets,
          type: 'u32',
          access: 'read'
        },
        {
          name: 'inducedNeighbors',
          view: csr.neighbors,
          type: 'u32',
          access: 'read_write'
        },
        {
          name: 'overflowOut',
          view: csr.overflow,
          type: 'u32',
          access: 'read_write'
        }
      ];
      if (csr.sourceSlots) {
        scatterBindings.push({
          name: 'inducedSlots',
          view: csr.sourceSlots,
          type: 'u32',
          access: 'read_write'
        });
      }
      kernel(
        'induced-scatter',
        'induced-scatter',
        scatterBindings,
        nodeCount,
        `if (index == 0u) {
    overflowOut[overflowOutOffset] =
      select(0u, 1u, inducedOffsets[inducedOffsetsOffset + NODE_COUNT] > CAPACITY);
  }
  var position = inducedOffsets[inducedOffsetsOffset + index];
  let rowBegin = offsets[offsetsOffset + index];
  let rowEnd = min(offsets[offsetsOffset + index + 1u], SLOT_COUNT);
  for (var slot = rowBegin; slot < rowEnd; slot++) {
    if (edgeMask[edgeMaskOffset + slot] != 0u) {
      if (position < CAPACITY) {
        inducedNeighbors[inducedNeighborsOffset + position] = neighbors[neighborsOffset + slot];
        ${csr.sourceSlots ? 'inducedSlots[inducedSlotsOffset + position] = slot;' : ''}
      }
      position++;
    }
  }`,
        `const NODE_COUNT: u32 = ${nodeCount}u;
const SLOT_COUNT: u32 = ${slotCount}u;
const CAPACITY: u32 = ${csr.neighbors.length}u;`
      );
    }
    return nodes;
  }
}
