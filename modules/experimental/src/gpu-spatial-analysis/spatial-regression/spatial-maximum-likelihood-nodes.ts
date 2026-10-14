// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUVectorFormat} from '@luma.gl/gpgpu/gpu-data';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import type {GPUSpatialWeights} from '../spatial-weights/index';
import {
  getSpatialWeightsTracesProbeWGSL,
  GPU_SPATIAL_WEIGHTS_TRACES_SCRATCH_ELEMENT_LIMIT,
  type GPUSpatialWeightsTracesMode
} from './spatial-weights-traces-parameters';

/** Workgroup size of the tile and probe passes (one invocation per tile). */
export const SPATIAL_MAXIMUM_LIKELIHOOD_TILE_WORKGROUP_SIZE = 64;

/** Builds an `f32` storage binding. */
export const getF32Binding = (
  name: string,
  view: GraphDataView,
  access: 'read' | 'read_write'
): WGSLKernelBinding => ({name, view, type: 'f32', access});

/** Builds a `u32` storage binding. */
export const getU32Binding = (
  name: string,
  view: GraphDataView,
  access: 'read' | 'read_write'
): WGSLKernelBinding => ({name, view, type: 'u32', access});

/** Kernel node factory bound to one graph, operation and ID prefix. */
export function createKernelFactory<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  operation: string
) {
  return (
    name: string,
    bindings: WGSLKernelBinding[],
    invocationCount: number,
    declarations: string,
    body: string,
    options: {workgroupSize?: number; constants?: Record<string, number>} = {}
  ): GPUCommandNode<Parameters> =>
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-${name}`,
      operation,
      variant: name.replace(/-\d+(-\d+)?$/, ''),
      bindings,
      invocationCount,
      declarations,
      body,
      ...(options.constants ? {constants: options.constants} : {}),
      ...(options.workgroupSize ? {workgroupSize: options.workgroupSize} : {})
    });
}

/**
 * Elements per aligned slot: 64 `float32` or `uint32` values are 256 bytes, the storage-buffer
 * offset alignment. A sub-view that starts at a multiple of this count has the same baked
 * `${name}Offset` WGSL constant as the whole view, so kernels that differ only in which slot
 * they bind share one pipeline (rule 12: a loop index never reaches WGSL text or constants).
 */
export const ALIGNED_ELEMENT_COUNT = 64;

/** Rounds `count` up to a multiple of {@link ALIGNED_ELEMENT_COUNT}. */
export function padToAlignedElementCount(count: number): number {
  return Math.ceil(count / ALIGNED_ELEMENT_COUNT) * ALIGNED_ELEMENT_COUNT;
}

/**
 * Returns the sub-view of `length` elements that starts `elementOffset` elements into `view`,
 * sharing its buffer. `elementOffset` must be a multiple of {@link ALIGNED_ELEMENT_COUNT}.
 */
export function getAlignedSubView<Parameters, Format extends GPUVectorFormat>(
  graph: GPUCommandGraph<Parameters>,
  view: GraphDataView<Format>,
  elementOffset: number,
  length: number
): GraphDataView<Format> {
  // Callers pass multiples of ALIGNED_ELEMENT_COUNT so the kernel offset residue stays constant
  // (rule 12); any other offset is a library bug.
  assert(elementOffset % ALIGNED_ELEMENT_COUNT === 0);
  return graph.createDataView(view.buffer, {
    format: view.format,
    length,
    byteOffset: view.byteOffset + elementOffset * view.rowByteLength
  });
}

/** Probe batches whose start and size live on the GPU. */
export type ProbeBatches<Parameters> = {
  /** Writes the batch table. Run it before every node that binds a batch slot. */
  node: GPUCommandNode<Parameters>;
  /** Probes of a full batch; every batch dispatches this many lanes. */
  batchProbeCount: number;
  /** Number of batches. */
  batchCount: number;
  /** Elements between the blocks of consecutive batches in a per-probe table. */
  blockStride: number;
  /**
   * Three uint32 words of batch `batchIndex`: word 0 is the first global probe, word 1 the number
   * of probes (smaller than `batchProbeCount` only for the last batch) and word 2 the per-frame
   * probe seed. Kernels guard `probe < count` and read the seed instead of baking the batch or
   * the seed into WGSL.
   */
  getBatchInformation: (batchIndex: number) => GraphDataView<'uint32'>;
  /** The aligned block of batch `batchIndex` inside a per-probe table of `blockStride` blocks. */
  getBlock: <Format extends GPUVectorFormat>(
    table: GraphDataView<Format>,
    batchIndex: number
  ) => GraphDataView<Format>;
  /**
   * WGSL statements declaring `name` (and `${name}Compensation`): the Kahan sum, in global probe
   * order, of a per-probe table of `batchCount` blocks that starts at the WGSL u32 expression
   * `base` (default `0u`). Reads `${table}[${table}Offset + base + ...]`.
   */
  getSumWGSL: (table: string, name?: string, base?: string) => string;
};

/**
 * Splits `probeTotal` probes into batches of `batchProbeCount` and writes, with one kernel whose
 * WGSL derives everything from the slot index, a uint32 table with one 64-word slot per batch
 * (word 0 the first probe, word 1 the probe count, word 2 the probe seed).
 *
 * @param seedParameters Optional per-frame `GPU_SPATIAL_WEIGHTS_TRACES_PARAMETER_LENGTH` uint32
 * words (`getGPUSpatialWeightsTracesParameterValues`); word 0 is the seed. Omitted means seed 0.
 */
export function createProbeBatches<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  operation: string,
  probeTotal: number,
  batchProbeCount: number,
  seedParameters?: GraphDataView<'uint32'>
): ProbeBatches<Parameters> {
  const batchCount = Math.ceil(probeTotal / batchProbeCount);
  const blockStride = padToAlignedElementCount(batchProbeCount);
  const table = createTransientView(
    graph,
    `${id}-batch-information`,
    'uint32',
    batchCount * ALIGNED_ELEMENT_COUNT
  );
  const node = createKernelFactory(graph, id, operation)(
    'batch-information',
    [
      getU32Binding('batchInformation', table, 'read_write'),
      ...(seedParameters ? [getU32Binding('seedParameters', seedParameters, 'read')] : [])
    ],
    batchCount,
    `const BATCH_PROBES: u32 = ${batchProbeCount}u;
const PROBE_TOTAL: u32 = ${probeTotal}u;
const SLOT_WORDS: u32 = ${ALIGNED_ELEMENT_COUNT}u;`,
    `let start = index * BATCH_PROBES;
  batchInformation[batchInformationOffset + index * SLOT_WORDS] = start;
  batchInformation[batchInformationOffset + index * SLOT_WORDS + 1u] = min(BATCH_PROBES, PROBE_TOTAL - start);
  batchInformation[batchInformationOffset + index * SLOT_WORDS + 2u] = ${
    seedParameters ? 'seedParameters[seedParametersOffset]' : '0u'
  };`
  );
  return {
    node,
    batchProbeCount,
    batchCount,
    blockStride,
    getBatchInformation: batchIndex =>
      getAlignedSubView(graph, table, batchIndex * ALIGNED_ELEMENT_COUNT, 3),
    getBlock: (view, batchIndex) =>
      getAlignedSubView(graph, view, batchIndex * blockStride, batchProbeCount),
    getSumWGSL: (source, name = 'sum', base = '0u') => `var ${name} = 0.0;
  var ${name}Compensation = 0.0;
  for (var sumBatch = 0u; sumBatch < ${batchCount}u; sumBatch++) {
    let sumCount = min(${batchProbeCount}u, ${probeTotal}u - sumBatch * ${batchProbeCount}u);
    for (var sumItem = 0u; sumItem < sumCount; sumItem++) {
      let term = ${source}[${source}Offset + ${base} + sumBatch * ${blockStride}u + sumItem] - ${name}Compensation;
      let next = ${name} + term;
      ${name}Compensation = (next - ${name}) - term;
      ${name} = next;
    }
  }`
  };
}

/** Options of {@link createSeriesProbeNodes}. */
export type SeriesProbeProps = {
  /** Prefix of node and transient IDs. */
  id: string;
  /** Operation name for the workload estimate. */
  operation: string;
  weights: GPUSpatialWeights;
  rowCount: number;
  /** Series length `Q`, at least 2. */
  maximumPower: number;
  /** `'exact'` unit probes or `'stochastic'` Rademacher probes. */
  mode: GPUSpatialWeightsTracesMode;
  /** Probes of the stochastic mode. */
  probeCount: number;
  /**
   * Optional per-frame seed words of the stochastic probes (`getGPUSpatialWeightsTracesParameterValues`).
   * Omitted means seed 0.
   */
  seedParameters?: GraphDataView<'uint32'>;
  tileRowCount: number;
  tileCount: number;
  /** Float32 view whose element `parameterSlot` holds the parameter `rho` the series is evaluated at. */
  workspace: GraphDataView;
  parameterSlot: number;
};

/** Result of {@link createSeriesProbeNodes}. */
export type SeriesProbeNodes<Parameters> = {
  nodes: GPUCommandNode<Parameters>[];
  /**
   * `||W (I - rho W)^-1 z||^2` of each probe `z`, one aligned block per batch (not a packed
   * `probeTotal` vector). Sum it with `dotsSumWGSL`.
   */
  dots: GraphDataView<'float32'>;
  /** WGSL statements declaring `sum`: the Kahan sum of `dots` in global probe order (binding `dots`). */
  dotsSumWGSL: string;
  /** Number of probes (rows in exact mode, `probeCount` in stochastic mode). */
  probeTotal: number;
};

/**
 * The probe pass of `tr((W A)' W A)` with `A = (I - rho W)^-1`, `rho` read from the workspace:
 * for each probe `z`, Horner's rule `u <- z + rho W u` (`Q - 1` times, from `u = z`) gives the
 * truncated `A z = sum_{m<Q} rho^m W^m z`, then one more product gives `W A z`, whose squared norm
 * is accumulated per tile and merged per probe with Kahan compensation. The mean over probes of
 * `||W A z||^2` is `tr((W A)' W A)` for Rademacher probes and the sum over unit probes is exact.
 *
 * Probes run in batches of at most `GPU_SPATIAL_WEIGHTS_TRACES_SCRATCH_ELEMENT_LIMIT / rows`
 * (two ping-pong scratch vectors of at most 8M floats each). The Rademacher stream is the one of
 * `GPUSpatialWeightsTraces` (Philox counter `(probe, row / 4, 0, 'PROB')`, key `(seed, 'TRAC')`),
 * generated by the shared `getSpatialWeightsTracesProbeWGSL`, so both passes draw identical probes.
 */
export function createSeriesProbeNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: SeriesProbeProps
): SeriesProbeNodes<Parameters> {
  const {id, rowCount, maximumPower, weights, tileRowCount, tileCount, workspace} = props;
  const stochastic = props.mode === 'stochastic';
  const probeTotal = stochastic ? props.probeCount : rowCount;
  const batchSize = Math.min(
    probeTotal,
    Math.max(1, Math.floor(GPU_SPATIAL_WEIGHTS_TRACES_SCRATCH_ELEMENT_LIMIT / rowCount))
  );
  const kernel = createKernelFactory(graph, id, props.operation);
  const transient = (name: string, length: number) =>
    createTransientView(graph, `${id}-${name}`, 'float32', length);
  const scratch = [
    transient('probe-a', batchSize * rowCount),
    transient('probe-b', batchSize * rowCount)
  ];
  const partial = transient('probe-partial', batchSize * tileCount);
  const batches = createProbeBatches(
    graph,
    id,
    props.operation,
    probeTotal,
    batchSize,
    props.seedParameters
  );
  const dots = transient('probe-dots', batches.batchCount * batches.blockStride);
  const constants = `const ROWS: u32 = ${rowCount}u;
const TILE_ROWS: u32 = ${tileRowCount}u;
const TILE_COUNT: u32 = ${tileCount}u;
const PARAMETER_SLOT: u32 = ${props.parameterSlot}u;`;
  const weightBindings: WGSLKernelBinding[] = [
    getU32Binding('offsets', weights.offsets, 'read'),
    getU32Binding('neighbors', weights.neighbors, 'read'),
    getF32Binding('weights', weights.weights, 'read')
  ];
  const nodes: GPUCommandNode<Parameters>[] = [batches.node];
  // Every batch runs the full lane count and reads its start and size from the batch table, so all
  // batches share their pipelines whatever the number of batches.
  for (let batchIndex = 0; batchIndex < batches.batchCount; batchIndex++) {
    const batchInformation = getU32Binding(
      'batchInformation',
      batches.getBatchInformation(batchIndex),
      'read'
    );
    for (let step = 1; step <= maximumPower; step++) {
      const last = step === maximumPower;
      const first = step === 1;
      const source = scratch[step % 2];
      const target = scratch[(step - 1) % 2];
      const bindings = [
        ...weightBindings,
        ...(first ? [] : [getF32Binding('vectorIn', source, 'read')]),
        last
          ? getF32Binding('partial', partial, 'read_write')
          : getF32Binding('vectorOut', target, 'read_write'),
        getF32Binding('workspace', workspace, 'read'),
        batchInformation
      ];
      const value = first
        ? 'getProbeValue(probeSeed, globalProbe, neighbor)'
        : 'vectorIn[vectorInOffset + base + neighbor]';
      nodes.push(
        kernel(
          `probe-${last ? 'final' : first ? 'first' : 'step'}-${batchIndex}-${step}`,
          bindings,
          batchSize * tileCount,
          `${constants}
${getSpatialWeightsTracesProbeWGSL(stochastic)}`,
          `let probe = index / TILE_COUNT;
  if (probe >= batchInformation[batchInformationOffset + 1u]) {
    return;
  }
  let rho = workspace[workspaceOffset + PARAMETER_SLOT];
  let tile = index % TILE_COUNT;
  let globalProbe = batchInformation[batchInformationOffset] + probe;
  let probeSeed = batchInformation[batchInformationOffset + 2u];
  let base = probe * ROWS;
  let firstRow = tile * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROWS);
  var squares = 0.0;
  for (var row = firstRow; row < lastRow; row++) {
    var sum = 0.0;
    for (var slot = offsets[offsetsOffset + row]; slot < offsets[offsetsOffset + row + 1u]; slot++) {
      let neighbor = neighbors[neighborsOffset + slot];
      if (neighbor >= ROWS) {
        continue;
      }
      sum += weights[weightsOffset + slot] * ${value};
    }
${
  last
    ? '    squares += sum * sum;'
    : '    vectorOut[vectorOutOffset + base + row] = getProbeValue(probeSeed, globalProbe, row) + rho * sum;'
}
  }
${last ? '  partial[partialOffset + probe * TILE_COUNT + tile] = squares;' : '  _ = squares;'}`,
          {workgroupSize: SPATIAL_MAXIMUM_LIKELIHOOD_TILE_WORKGROUP_SIZE}
        )
      );
    }
    nodes.push(
      kernel(
        `probe-merge-${batchIndex}`,
        [
          getF32Binding('partial', partial, 'read'),
          getF32Binding('dots', batches.getBlock(dots, batchIndex), 'read_write'),
          batchInformation
        ],
        batchSize,
        constants,
        `if (index >= batchInformation[batchInformationOffset + 1u]) {
    return;
  }
  var sum = 0.0;
  var compensation = 0.0;
  for (var tile = 0u; tile < TILE_COUNT; tile++) {
    let term = partial[partialOffset + index * TILE_COUNT + tile] - compensation;
    let next = sum + term;
    compensation = (next - sum) - term;
    sum = next;
  }
  dots[dotsOffset + index] = sum;`
      )
    );
  }
  return {nodes, dots, dotsSumWGSL: batches.getSumWGSL('dots'), probeTotal};
}
