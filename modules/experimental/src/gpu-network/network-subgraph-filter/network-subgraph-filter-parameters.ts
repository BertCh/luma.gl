// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Floats per predicate record in the {@link GPUNetworkSubgraphFilter} `parameters` view. */
export const GPU_NETWORK_SUBGRAPH_FILTER_PARAMETER_STRIDE = 4;

/** Word indices of the optional `counts` output. */
export const GPU_NETWORK_SUBGRAPH_FILTER_COUNT_WORD = {
  /** Vertices with a nonzero `vertexMask`. */
  liveVertexCount: 0,
  /** CSR slots with a nonzero `edgeMask`. */
  liveSlotCount: 1,
  /** Live slots u -> u. */
  selfLoopSlotCount: 2,
  /**
   * Directed: `liveSlotCount`. Undirected: live non-self-loop slots / 2 plus live self-loop slots
   * (the same rule as `GPUNetworkStatistics`).
   */
  liveEdgeCount: 3
} as const;

/** Number of words of the `counts` output. */
export const GPU_NETWORK_SUBGRAPH_FILTER_COUNT_LENGTH = 4;

/**
 * Per-frame description of the f32 predicates of one {@link GPUNetworkSubgraphFilter}.
 *
 * Every entry is one `[min, max]` pair, or `null` / `undefined` to disable that predicate. Attribute
 * ranges are half-open: a row passes when `min <= value < max` (the `GPUAttributeCrossfilter` brush
 * rule; use `Infinity` as the maximum to include everything above the minimum). A NaN value never
 * passes an enabled predicate. The optional edge time window is closed on both ends, like the
 * `GPUTimeWindowFilter` windows: `start <= time <= end`.
 */
export type GPUNetworkSubgraphFilterState = {
  /** One range per `vertexColumns` entry, in order. */
  vertexRanges?: readonly (readonly [number, number] | null | undefined)[];
  /** One range per `edgeColumns` entry, in order. */
  edgeRanges?: readonly (readonly [number, number] | null | undefined)[];
  /** Closed window over the f32 `edgeTimes` column. */
  edgeTimeWindow?: readonly [number, number] | null;
};

/** Shape needed to size and pack the parameter view. */
export type GPUNetworkSubgraphFilterParameterLayout = {
  /** Number of vertex columns. */
  vertexColumnCount: number;
  /** Number of edge columns. */
  edgeColumnCount: number;
  /** Whether an f32 edge time column exists. */
  hasEdgeTimes: boolean;
};

/** Returns the minimum `parameters` view length (float32 rows) for a layout. */
export function getGPUNetworkSubgraphFilterParameterLength(
  layout: GPUNetworkSubgraphFilterParameterLayout
): number {
  return (
    (layout.vertexColumnCount + layout.edgeColumnCount + (layout.hasEdgeTimes ? 1 : 0)) *
    GPU_NETWORK_SUBGRAPH_FILTER_PARAMETER_STRIDE
  );
}

/**
 * Packs ranges into the float32 layout the contributor reads.
 *
 * Records of {@link GPU_NETWORK_SUBGRAPH_FILTER_PARAMETER_STRIDE} floats `[min, max, enabled, 0]`:
 * vertex columns first, then edge columns, then the f32 edge time window. Write the result into the
 * caller-owned parameter buffer between encodings; nothing recompiles.
 *
 * Missing, `null` and `undefined` entries are disabled predicates.
 *
 * @throws If `target` is too short.
 */
export function getGPUNetworkSubgraphFilterParameterValues(
  layout: GPUNetworkSubgraphFilterParameterLayout,
  state: GPUNetworkSubgraphFilterState = {},
  target: Float32Array = new Float32Array(getGPUNetworkSubgraphFilterParameterLength(layout))
): Float32Array {
  if (target.length < getGPUNetworkSubgraphFilterParameterLength(layout)) {
    throw new Error('GPUNetworkSubgraphFilter parameter target is too short');
  }
  const stride = GPU_NETWORK_SUBGRAPH_FILTER_PARAMETER_STRIDE;
  target.fill(0, 0, getGPUNetworkSubgraphFilterParameterLength(layout));
  const records: (readonly [number, number] | null | undefined)[] = [];
  for (let i = 0; i < layout.vertexColumnCount; i++) records.push(state.vertexRanges?.[i]);
  for (let i = 0; i < layout.edgeColumnCount; i++) records.push(state.edgeRanges?.[i]);
  if (layout.hasEdgeTimes) records.push(state.edgeTimeWindow);
  for (const [recordIndex, range] of records.entries()) {
    if (!range) continue;
    target[recordIndex * stride] = range[0];
    target[recordIndex * stride + 1] = range[1];
    target[recordIndex * stride + 2] = 1;
  }
  return target;
}

/** Decoded `counts` output. */
export type GPUNetworkSubgraphFilterCounts = {
  liveVertexCount: number;
  liveSlotCount: number;
  selfLoopSlotCount: number;
  liveEdgeCount: number;
};

/** Decodes the words read back from the `counts` output. */
export function decodeGPUNetworkSubgraphFilterCounts(
  words: ArrayLike<number>
): GPUNetworkSubgraphFilterCounts {
  const word = GPU_NETWORK_SUBGRAPH_FILTER_COUNT_WORD;
  return {
    liveVertexCount: words[word.liveVertexCount],
    liveSlotCount: words[word.liveSlotCount],
    selfLoopSlotCount: words[word.selfLoopSlotCount],
    liveEdgeCount: words[word.liveEdgeCount]
  };
}
