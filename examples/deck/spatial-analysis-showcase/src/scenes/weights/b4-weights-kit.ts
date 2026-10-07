// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Analysis kit of the weights scenes, adapted from the explorer's weights mode. It works on any
 * weights source (polygon contiguity, kNN or distance band on centroids, a regular lattice): the
 * summary (`GPUSpatialWeightsSummary`), the transpose and its symmetrised union
 * (`GPUSpatialWeightsTranspose` + `GPUSpatialWeightsAlgebra`), the spatial lag (`GPUSpatialLag`),
 * and the neighbour links of one focus row or of the whole matrix.
 *
 * Everything is compiled once per source. The normalisation of the lag or the focus row is a
 * variant pick or a buffer write, never a recompile.
 */

import type {Buffer, CommandEncoder, Device} from '@luma.gl/core';
import {
  GPUCommandGraph,
  GPUReadbackRing,
  GPUReduction,
  type CompiledGPUCommandGraph
} from '@luma.gl/gpgpu/gpu-core';
import {
  GPU_SPATIAL_WEIGHTS_SUMMARY_LAYOUT,
  GPUSpatialLag,
  GPUSpatialWeightsAlgebra,
  GPUSpatialWeightsSummary,
  GPUSpatialWeightsTranspose
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {importGraphBuffer} from '../../engine/graph-buffers';
import type {SpatialAnalysisResources} from '../../engine/resources';
import {addKernelPass} from '../../engine/mode-kernels';

/** Which weights matrix the links and the focus readout show. */
export type WeightsMatrixChoice = 'weights' | 'transpose' | 'union';

/** Most neighbor links drawn for the focus row. */
export const FOCUS_SLOTS = 128;

/** A weights CSR held in application buffers. */
export type WeightsCsrBuffers = {offsets: Buffer; neighbors: Buffer; weights: Buffer};

/** Properties of {@link createWeightsKit}. */
export type WeightsKitProps = {
  device: Device;
  resources: SpatialAnalysisResources;
  /** Prefix for debug IDs. */
  id: string;
  rowCount: number;
  /** Slot capacity of `csr`. */
  slots: number;
  /** Slot capacity of the symmetrised union. */
  unionSlots: number;
  /** `float32x2` row positions in meters. */
  positions: Buffer;
  /** Float32 value per row, lagged by the spatial lag. */
  values: Buffer;
  /** Uint32 per row: zero rows are skipped by the lag (missing values, empty lattice cells). */
  mask: Buffer;
  /** The weights the producer writes and every analysis reads. */
  csr: WeightsCsrBuffers;
  /** Producer overflow flag and unclamped neighbor total, copied into the readback. */
  producer: {overflow: Buffer; total: Buffer};
  /** Existing summary buffers of `csr` (12 and 20 bytes) when another graph already fills them. */
  summary?: {statistics: Buffer; counts: Buffer};
  /** Also build full-matrix link segments for the transpose and the union. */
  drawAllLinks: boolean;
};

/** Values the readback delivers for one source. */
export type WeightsKitSummary = {
  s0: number;
  s1: number;
  s2: number;
  slots: number;
  asymmetricSlots: number;
  isolates: number;
  minimumCardinality: number;
  maximumCardinality: number;
  /** Positions of the union of W and its transpose whose weights differ (two per one-way pair). */
  transposeAsymmetricSlots: number;
  unionSlots: number;
  unionOverflow: boolean;
  lagMinimum: number;
  lagMaximum: number;
  /** Per matrix: neighbor count and weight sum of the focus row. */
  focus: Record<WeightsMatrixChoice, {degree: number; weightSum: number}>;
  producerOverflow: boolean;
  producerTotal: number;
};

// statistics 3, counts 5, asymmetric 1, lag extent 2, union overflow and total 2, producer
// overflow and total 2, focus info 3 x 2.
const SUMMARY_WORDS = 3 + 5 + 1 + 2 + 2 + 2 + 6;

/** Weights analysis kit for one source; see the module comment. */
export type WeightsKit = {
  /** Every compiled graph, for the shell's rebuild accounting. */
  compiledGraphs: () => readonly CompiledGPUCommandGraph<never>[];
  /** Per-row neighbor count (uint32) of `csr`; absent when the summary is external. */
  cardinality: Buffer | null;
  lag: Buffer;
  lagExtent: Buffer;
  focusRow: Buffer;
  /** Segments `x0, y0, x1, y1` from the focus row to its neighbors in the chosen matrix. */
  focusSegments: Record<WeightsMatrixChoice, Buffer>;
  /** Neighbor row IDs of the focus row (the focus row itself where unused) for the chosen matrix. */
  focusNeighbors: Record<WeightsMatrixChoice, Buffer>;
  /** Full-matrix segments of the transpose and the union (and the weights segments for sources that draw them). */
  allSegments: Record<WeightsMatrixChoice, {segments: Buffer; fade: Buffer; slots: number} | null>;
  /** Encodes summary, transpose, union, lag and focus links of the current weights. */
  encodeAnalysis: (
    commandEncoder: CommandEncoder,
    options: {normalizeLag: boolean; matrix: WeightsMatrixChoice}
  ) => void;
  /** Re-encodes only the focus links, after the focus row changed. */
  encodeFocus: (commandEncoder: CommandEncoder) => void;
  /** Copies the summaries into a ring ticket and resolves them, or `null` when the ring is busy. */
  readSummary: (commandEncoder: CommandEncoder) => Promise<WeightsKitSummary | null>;
  /** Moves the focus row (a buffer write; call `encodeFocus` afterwards). */
  setFocusRow: (row: number) => void;
};

const SEGMENT_BODY = /* wgsl */ `
  let total = offsets[offsetsOffset + ROW_COUNT];
  if (index >= total) {
    segments[segmentsOffset + index * 4u] = 0.0;
    segments[segmentsOffset + index * 4u + 1u] = 0.0;
    segments[segmentsOffset + index * 4u + 2u] = 0.0;
    segments[segmentsOffset + index * 4u + 3u] = 0.0;
    fade[fadeOffset + index] = 0.0;
    return;
  }
  var low = 0u;
  var high = ROW_COUNT;
  while (low < high) {
    let middle = (low + high + 1u) / 2u;
    if (offsets[offsetsOffset + middle] <= index) {
      low = middle;
    } else {
      high = middle - 1u;
    }
  }
  let neighbor = neighbors[neighborsOffset + index];
  segments[segmentsOffset + index * 4u] = positions[positionsOffset + low * 2u];
  segments[segmentsOffset + index * 4u + 1u] = positions[positionsOffset + low * 2u + 1u];
  segments[segmentsOffset + index * 4u + 2u] = positions[positionsOffset + neighbor * 2u];
  segments[segmentsOffset + index * 4u + 3u] = positions[positionsOffset + neighbor * 2u + 1u];
  fade[fadeOffset + index] = 1.0;`;

const FOCUS_BODY = /* wgsl */ `
  let focus = min(focusRow[focusRowOffset], ROW_COUNT - 1u);
  let start = offsets[offsetsOffset + focus];
  let end = offsets[offsetsOffset + focus + 1u];
  if (index == 0u) {
    var weightSum = 0.0;
    for (var slot = start; slot < end; slot++) {
      weightSum += weights[weightsOffset + slot];
    }
    info[infoOffset] = f32(end - start);
    info[infoOffset + 1u] = weightSum;
  }
  let slot = start + index;
  var other = focus;
  if (slot < end) {
    other = neighbors[neighborsOffset + slot];
  }
  // Unused slots become NaN segments, which the segment layer skips.
  var nanBits = 0x7fc00000u;
  let hidden = bitcast<f32>(nanBits);
  segments[segmentsOffset + index * 4u] = select(hidden, positions[positionsOffset + focus * 2u], slot < end);
  segments[segmentsOffset + index * 4u + 1u] = positions[positionsOffset + focus * 2u + 1u];
  segments[segmentsOffset + index * 4u + 2u] = positions[positionsOffset + other * 2u];
  segments[segmentsOffset + index * 4u + 3u] = positions[positionsOffset + other * 2u + 1u];
  neighborIds[neighborIdsOffset + index] = other;`;

/** Builds the transform, transpose, union, summary, lag and focus graphs of one weights source. */
export function createWeightsKit(props: WeightsKitProps): WeightsKit {
  const {device, resources, id, rowCount, slots, unionSlots, positions, values, csr} = props;
  const create = (name: string, data: number | Uint32Array) =>
    resources.createBuffer(`${id}-${name}`, data);

  const transposed: WeightsCsrBuffers = {
    offsets: create('transpose-offsets', (rowCount + 1) * 4),
    neighbors: create('transpose-neighbors', slots * 4),
    weights: create('transpose-weights', slots * 4)
  };
  const union: WeightsCsrBuffers = {
    offsets: create('union-offsets', (rowCount + 1) * 4),
    neighbors: create('union-neighbors', unionSlots * 4),
    weights: create('union-weights', unionSlots * 4)
  };
  const matrices: Record<WeightsMatrixChoice, {csr: WeightsCsrBuffers; slots: number}> = {
    weights: {csr, slots},
    transpose: {csr: transposed, slots},
    union: {csr: union, slots: unionSlots}
  };
  const asymmetricBuffer = create('transpose-asymmetric', 4);
  const unionOverflowBuffer = create('union-overflow', 4);
  const unionTotalBuffer = create('union-total', 4);
  const lagBuffer = create('lag', rowCount * 4);
  const lagExtentBuffer = create('lag-extent', 2 * 4);
  const focusRowBuffer = create('focus-row', new Uint32Array([0]));
  const summaryStatistics = props.summary?.statistics ?? create('summary-statistics', 3 * 4);
  const summaryCounts = props.summary?.counts ?? create('summary-counts', 5 * 4);
  const cardinalityBuffer = props.summary ? null : create('cardinality', rowCount * 4);
  const infoBuffers = {
    weights: create('focus-info-weights', 2 * 4),
    transpose: create('focus-info-transpose', 2 * 4),
    union: create('focus-info-union', 2 * 4)
  };
  const focusSegments = {
    weights: create('focus-segments-weights', FOCUS_SLOTS * 16),
    transpose: create('focus-segments-transpose', FOCUS_SLOTS * 16),
    union: create('focus-segments-union', FOCUS_SLOTS * 16)
  };
  const focusNeighbors = {
    weights: create('focus-neighbors-weights', FOCUS_SLOTS * 4),
    transpose: create('focus-neighbors-transpose', FOCUS_SLOTS * 4),
    union: create('focus-neighbors-union', FOCUS_SLOTS * 4)
  };
  const readbackRing = resources.track(
    new GPUReadbackRing(device, {id: `${id}-kit-summary`, byteLength: SUMMARY_WORDS * 4})
  );

  const importCsr = (
    graph: GPUCommandGraph<void>,
    name: string,
    matrix: WeightsCsrBuffers,
    count: number
  ) => ({
    offsets: importGraphBuffer(graph, `${name}-offsets`, matrix.offsets, 'uint32', rowCount + 1),
    neighbors: importGraphBuffer(graph, `${name}-neighbors`, matrix.neighbors, 'uint32', count),
    weights: importGraphBuffer(graph, `${name}-weights`, matrix.weights, 'float32', count)
  });
  const importPositions = (graph: GPUCommandGraph<void>) =>
    importGraphBuffer(graph, 'positions', positions, 'float32x2', rowCount);

  // Summary of the weights (own graph unless another graph already fills the buffers).
  let summaryCompiled: CompiledGPUCommandGraph<void> | null = null;
  if (!props.summary) {
    const graph = new GPUCommandGraph<void>(device, {id: `${id}-summary`});
    graph.add(
      new GPUSpatialWeightsSummary({
        id: 'weights-summary',
        weights: importCsr(graph, 'a', csr, slots),
        statistics: importGraphBuffer(graph, 'statistics', summaryStatistics, 'float32', 3),
        counts: importGraphBuffer(graph, 'counts', summaryCounts, 'uint32', 5),
        cardinality: importGraphBuffer(graph, 'cardinality', cardinalityBuffer!, 'uint32', rowCount)
      })
    );
    summaryCompiled = resources.track(graph.compile());
  }

  // W transpose and the symmetrised union W OR W'.
  const transposeGraph = new GPUCommandGraph<void>(device, {id: `${id}-transpose-union`});
  {
    const source = importCsr(transposeGraph, 'a', csr, slots);
    const transposeView = importCsr(transposeGraph, 't', transposed, slots);
    transposeGraph.add(
      new GPUSpatialWeightsTranspose({
        id: 'transpose',
        weights: source,
        output: transposeView,
        asymmetricSlots: importGraphBuffer(
          transposeGraph,
          'asymmetric',
          asymmetricBuffer,
          'uint32',
          1
        )
      })
    );
    transposeGraph.add(
      new GPUSpatialWeightsAlgebra({
        id: 'symmetrize-union',
        operation: 'union',
        left: source,
        right: transposeView,
        output: importCsr(transposeGraph, 's', union, unionSlots),
        overflow: importGraphBuffer(
          transposeGraph,
          'union-overflow',
          unionOverflowBuffer,
          'uint32',
          1
        ),
        totalNeighbors: importGraphBuffer(
          transposeGraph,
          'union-total',
          unionTotalBuffer,
          'uint32',
          1
        )
      })
    );
  }
  const transposeCompiled = resources.track(transposeGraph.compile());

  // Spatial lag of the value column, plain or normalized, and its extent for the color range.
  const lagVariants = new Map<boolean, CompiledGPUCommandGraph<void>>();
  for (const normalize of [false, true]) {
    const graph = new GPUCommandGraph<void>(device, {id: `${id}-lag-${normalize}`});
    const lag = importGraphBuffer(graph, 'lag', lagBuffer, 'float32', rowCount);
    graph.add(
      new GPUSpatialLag({
        id: 'spatial-lag',
        values: importGraphBuffer(graph, 'values', values, 'float32', rowCount),
        mask: importGraphBuffer(graph, 'mask', props.mask, 'uint32', rowCount),
        weights: importCsr(graph, 'a', csr, slots),
        normalize,
        output: lag
      })
    );
    graph.add(
      new GPUReduction({
        id: 'lag-extent',
        input: lag,
        output: importGraphBuffer(graph, 'lag-extent', lagExtentBuffer, 'float32', 2),
        operation: 'extent'
      })
    );
    lagVariants.set(normalize, resources.track(graph.compile()));
  }

  // Neighbor links of the focus row, one graph per matrix (all rows share the focus buffer).
  const focusGraphs = {} as Record<WeightsMatrixChoice, CompiledGPUCommandGraph<void>>;
  for (const name of ['weights', 'transpose', 'union'] as const) {
    const graph = new GPUCommandGraph<void>(device, {id: `${id}-focus-${name}`});
    const view = importCsr(graph, 'm', matrices[name].csr, matrices[name].slots);
    addKernelPass(graph, {
      id: `focus-${name}`,
      invocationCount: FOCUS_SLOTS,
      bindings: [
        {
          name: 'focusRow',
          view: importGraphBuffer(graph, 'focus', focusRowBuffer, 'uint32', 1),
          type: 'u32',
          access: 'read'
        },
        {name: 'offsets', view: view.offsets, type: 'u32', access: 'read'},
        {name: 'neighbors', view: view.neighbors, type: 'u32', access: 'read'},
        {name: 'weights', view: view.weights, type: 'f32', access: 'read'},
        {name: 'positions', view: importPositions(graph), type: 'f32', access: 'read'},
        {
          name: 'segments',
          view: importGraphBuffer(
            graph,
            'segments',
            focusSegments[name],
            'float32',
            FOCUS_SLOTS * 4
          ),
          type: 'f32',
          access: 'read_write'
        },
        {
          name: 'neighborIds',
          view: importGraphBuffer(graph, 'ids', focusNeighbors[name], 'uint32', FOCUS_SLOTS),
          type: 'u32',
          access: 'read_write'
        },
        {
          name: 'info',
          view: importGraphBuffer(graph, 'info', infoBuffers[name], 'float32', 2),
          type: 'f32',
          access: 'read_write'
        }
      ],
      declarations: `const ROW_COUNT: u32 = ${rowCount}u;`,
      body: FOCUS_BODY
    });
    focusGraphs[name] = resources.track(graph.compile());
  }

  // Optional full-matrix segments of the transpose and the union.
  const allSegments: WeightsKit['allSegments'] = {weights: null, transpose: null, union: null};
  const allGraphs: CompiledGPUCommandGraph<void>[] = [];
  const allGraphByMatrix = {} as Record<WeightsMatrixChoice, CompiledGPUCommandGraph<void>>;
  if (props.drawAllLinks) {
    for (const name of ['weights', 'transpose', 'union'] as const) {
      const count = matrices[name].slots;
      const segments = create(`all-segments-${name}`, count * 16);
      const fade = create(`all-fade-${name}`, count * 4);
      const graph = new GPUCommandGraph<void>(device, {id: `${id}-links-${name}`});
      const view = importCsr(graph, 'm', matrices[name].csr, count);
      addKernelPass(graph, {
        id: `links-${name}`,
        invocationCount: count,
        bindings: [
          {name: 'offsets', view: view.offsets, type: 'u32', access: 'read'},
          {name: 'neighbors', view: view.neighbors, type: 'u32', access: 'read'},
          {name: 'positions', view: importPositions(graph), type: 'f32', access: 'read'},
          {
            name: 'segments',
            view: importGraphBuffer(graph, 'segments', segments, 'float32', count * 4),
            type: 'f32',
            access: 'read_write'
          },
          {
            name: 'fade',
            view: importGraphBuffer(graph, 'fade', fade, 'float32', count),
            type: 'f32',
            access: 'read_write'
          }
        ],
        declarations: `const ROW_COUNT: u32 = ${rowCount}u;`,
        body: SEGMENT_BODY
      });
      const compiled = resources.track(graph.compile());
      allGraphs.push(compiled);
      allGraphByMatrix[name] = compiled;
      allSegments[name] = {segments, fade, slots: count};
    }
  }

  const compiledGraphs = () =>
    [
      ...(summaryCompiled ? [summaryCompiled] : []),
      transposeCompiled,
      ...lagVariants.values(),
      ...Object.values(focusGraphs),
      ...allGraphs
    ] as readonly CompiledGPUCommandGraph<never>[];

  const encodeFocus = (commandEncoder: CommandEncoder) => {
    for (const name of ['weights', 'transpose', 'union'] as const) {
      focusGraphs[name].encode(commandEncoder, {parameters: undefined});
    }
  };

  return {
    compiledGraphs,
    cardinality: cardinalityBuffer,
    lag: lagBuffer,
    lagExtent: lagExtentBuffer,
    focusRow: focusRowBuffer,
    focusSegments,
    focusNeighbors,
    allSegments,
    encodeAnalysis(commandEncoder, options) {
      summaryCompiled?.encode(commandEncoder, {parameters: undefined});
      transposeCompiled.encode(commandEncoder, {parameters: undefined});
      lagVariants.get(options.normalizeLag)!.encode(commandEncoder, {parameters: undefined});
      encodeFocus(commandEncoder);
      allGraphByMatrix[options.matrix]?.encode(commandEncoder, {parameters: undefined});
    },
    encodeFocus,
    setFocusRow(row) {
      focusRowBuffer.write(new Uint32Array([Math.max(0, Math.min(rowCount - 1, row))]));
    },
    async readSummary(commandEncoder) {
      const ticket = readbackRing.tryAcquire();
      if (!ticket) return null;
      let word = 0;
      const copy = (source: Buffer, words: number) => {
        commandEncoder.copyBufferToBuffer({
          sourceBuffer: source,
          destinationBuffer: ticket.buffer,
          destinationOffset: word * 4,
          size: words * 4
        });
        word += words;
      };
      copy(summaryStatistics, 3);
      copy(summaryCounts, 5);
      copy(asymmetricBuffer, 1);
      copy(lagExtentBuffer, 2);
      copy(unionOverflowBuffer, 1);
      copy(unionTotalBuffer, 1);
      copy(props.producer.overflow, 1);
      copy(props.producer.total, 1);
      copy(infoBuffers.weights, 2);
      copy(infoBuffers.transpose, 2);
      copy(infoBuffers.union, 2);
      ticket.markEncoded({byteOffset: 0, byteLength: SUMMARY_WORDS * 4});
      const bytes = await ticket.read();
      const floats = new Float32Array(bytes.buffer, bytes.byteOffset, SUMMARY_WORDS);
      const words = new Uint32Array(bytes.buffer, bytes.byteOffset, SUMMARY_WORDS);
      const layout = GPU_SPATIAL_WEIGHTS_SUMMARY_LAYOUT;
      const counts = (field: number) => words[3 + field];
      const info = (start: number) => ({degree: floats[start], weightSum: floats[start + 1]});
      return {
        s0: floats[layout.statistics.s0],
        s1: floats[layout.statistics.s1],
        s2: floats[layout.statistics.s2],
        slots: counts(layout.counts.slots),
        asymmetricSlots: counts(layout.counts.asymmetricSlots),
        isolates: counts(layout.counts.isolates),
        minimumCardinality: counts(layout.counts.minimumCardinality),
        maximumCardinality: counts(layout.counts.maximumCardinality),
        transposeAsymmetricSlots: words[8],
        lagMinimum: floats[9],
        lagMaximum: floats[10],
        unionOverflow: words[11] !== 0,
        unionSlots: words[12],
        producerOverflow: words[13] !== 0,
        producerTotal: words[14],
        focus: {weights: info(15), transpose: info(17), union: info(19)}
      };
    }
  };
}
