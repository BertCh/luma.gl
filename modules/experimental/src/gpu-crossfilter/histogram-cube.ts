// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
// SPDX-FileComment: Independently implemented for WebGPU; inspired by NVIDIA RAPIDS cuXfilter.
// SPDX-FileComment: Cube construction follows Moritz, Howe, and Heer, "Falcon: Balancing Interactive Latency and Resolution Sensitivity for Scalable Linked Visualizations", CHI 2019.

import {
  type GPUCommandGraph,
  type GPUCommandGraphCPUCondition,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {IS_FINITE_BITS_WGSL} from '../utils/wgsl-common';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral,
  type WGSLKernelBinding
} from '../utils/wgsl-kernel-nodes';
import type {GPUCrossfilterScalarFormat} from './gpu-selection';

/** Cubes up to this many cells accumulate in workgroup memory (8 KiB) before merging. */
const MAXIMUM_LOCAL_CELL_COUNT = 2048;
const CUBE_WORKGROUP_SIZE = 256;
const SENTINEL_BIN = '0xffffffffu';

/** How one histogram view maps a scalar to a bin; mirrors the `GPUHistogram` binning. @internal */
export type HistogramCubeBinning = {
  /** Scalar format of the binned values. */
  format: GPUCrossfilterScalarFormat;
  /** Number of bins, which is the view's `output.length`. */
  binCount: number;
  /** Equal-width inclusive domain, as literal bounds or a two-row GPU view. */
  domain?: readonly [number, number] | GraphDataView;
  /** Irregular `binCount + 1` boundaries, literal or GPU-resident. */
  edges?: readonly number[] | GraphDataView;
};

/** One side of a cube: the binned source chunks and how they are binned. @internal */
export type HistogramCubeSide = {
  /** Source chunks in order; every side and the mask share one chunk topology. */
  chunks: readonly GraphDataView[];
  /** Binning of the side's histogram view. */
  binning: HistogramCubeBinning;
};

/** Shared inputs of the build and readout nodes of one (active view, linked view) pair. @internal */
export type HistogramCubePairProps = {
  /** Node ID prefix. */
  id: string;
  /** Histogram view whose dimension is brushed. */
  active: HistogramCubeSide;
  /** Histogram view whose counts the cube answers. */
  linked: HistogramCubeSide;
  /** Persistent `uint32` cube storage with at least `active.bins * linked.bins` rows. */
  cube: GraphDataView<'uint32'>;
};

/** Number of cube rows (cells) for a pair of bin counts. @internal */
export function getHistogramCubeCellCount(activeBinCount: number, linkedBinCount: number): number {
  return activeBinCount * linkedBinCount;
}

/**
 * Builds the prefix-summed cube `C[a][b]` = rows that pass `masks` with active-view bin `<= a` and
 * linked-view bin `b`, as nodes: clear, one accumulation per nonempty chunk, then an in-place prefix
 * sum over `a`. All counts are exact integers (`atomicAdd` on `u32`).
 *
 * `createCondition` is called once per node because conditions may keep per-node state.
 *
 * @internal
 */
export function createHistogramCubeBuildNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: HistogramCubePairProps & {
    /** Per-chunk row selection shared by both views, or `undefined` to count every row. */
    masks?: readonly GraphDataView<'uint32'>[];
    createCondition: (nodeId: string) => GPUCommandGraphCPUCondition<Parameters> | undefined;
  }
): GPUCommandNode<Parameters>[] {
  const activeBinCount = props.active.binning.binCount;
  const linkedBinCount = props.linked.binning.binCount;
  const cellCount = getHistogramCubeCellCount(activeBinCount, linkedBinCount);
  const nodes: GPUCommandNode<Parameters>[] = [];
  const cubeBinding = (
    type: 'u32' | 'atomic<u32>',
    access: 'read' | 'read_write' = 'read_write'
  ): WGSLKernelBinding => ({name: 'cube', view: props.cube, type, access});

  nodes.push(
    createWGSLKernelNode(graph, {
      id: `${props.id}/clear`,
      operation: 'GPUCrossfilterHistogramCube',
      variant: 'clear',
      bindings: [cubeBinding('u32')],
      invocationCount: cellCount,
      condition: props.createCondition(`${props.id}/clear`),
      body: 'cube[cubeOffset + index] = 0u;'
    })
  );

  const activeBinningSource = createBinningSource('active', props.active.binning);
  const linkedBinningSource = createBinningSource('linked', props.linked.binning);
  const useLocalCells = cellCount <= MAXIMUM_LOCAL_CELL_COUNT;
  for (const [chunkIndex, activeChunk] of props.active.chunks.entries()) {
    if (activeChunk.length === 0) continue;
    const linkedChunk = props.linked.chunks[chunkIndex];
    const maskChunk = props.masks?.[chunkIndex];
    const nodeId = `${props.id}/accumulate-${chunkIndex}`;
    // Uniform trip counts keep the barriers in uniform control flow.
    const roundCount = Math.ceil(cellCount / CUBE_WORKGROUP_SIZE);
    const accumulation = useLocalCells
      ? `
  let lane = localInvocationIndex;
  for (var round = 0u; round < ${roundCount}u; round = round + 1u) {
    let cell = round * ${CUBE_WORKGROUP_SIZE}u + lane;
    if (cell < CELL_COUNT) { atomicStore(&localCells[cell], 0u); }
  }
  workgroupBarrier();
  if (index < INVOCATION_COUNT) {
    ${getRowBodySource(maskChunk !== undefined, 'atomicAdd(&localCells[cellIndex], 1u);')}
  }
  workgroupBarrier();
  for (var round = 0u; round < ${roundCount}u; round = round + 1u) {
    let cell = round * ${CUBE_WORKGROUP_SIZE}u + lane;
    if (cell < CELL_COUNT) {
      let count = atomicLoad(&localCells[cell]);
      if (count != 0u) { atomicAdd(&cube[cubeOffset + cell], count); }
    }
  }`
      : `
  if (index < INVOCATION_COUNT) {
    ${getRowBodySource(maskChunk !== undefined, 'atomicAdd(&cube[cubeOffset + cellIndex], 1u);')}
  }`;
    nodes.push(
      createWGSLKernelNode(graph, {
        id: nodeId,
        operation: 'GPUCrossfilterHistogramCube',
        variant: useLocalCells ? 'accumulate-local' : 'accumulate-global',
        bindings: [
          {
            name: 'activeValues',
            view: activeChunk,
            type: getScalarType(props.active.binning.format),
            access: 'read'
          },
          {
            name: 'linkedValues',
            view: linkedChunk,
            type: getScalarType(props.linked.binning.format),
            access: 'read'
          },
          ...(maskChunk
            ? [
                {
                  name: 'rowMask',
                  view: maskChunk,
                  type: 'u32',
                  access: 'read'
                } satisfies WGSLKernelBinding
              ]
            : []),
          cubeBinding('atomic<u32>'),
          ...activeBinningSource.bindings,
          ...linkedBinningSource.bindings
        ],
        invocationCount: activeChunk.length,
        workgroupSize: CUBE_WORKGROUP_SIZE,
        guardIndex: false,
        condition: props.createCondition(nodeId),
        declarations: `
${IS_FINITE_BITS_WGSL}
const CELL_COUNT: u32 = ${cellCount}u;
const LINKED_BIN_COUNT: u32 = ${linkedBinCount}u;
const ACTIVE_STRIDE: u32 = ${getStride(activeChunk)}u;
const LINKED_STRIDE: u32 = ${getStride(linkedChunk)}u;
${maskChunk ? `const MASK_STRIDE: u32 = ${getStride(maskChunk)}u;` : ''}
${activeBinningSource.declarations}
${linkedBinningSource.declarations}
${useLocalCells ? `var<workgroup> localCells: array<atomic<u32>, ${cellCount}>;` : ''}`,
        body: accumulation
      })
    );
  }

  const prefixId = `${props.id}/prefix`;
  nodes.push(
    createWGSLKernelNode(graph, {
      id: prefixId,
      operation: 'GPUCrossfilterHistogramCube',
      variant: 'prefix',
      bindings: [cubeBinding('u32')],
      invocationCount: linkedBinCount,
      condition: props.createCondition(prefixId),
      declarations: `
const ACTIVE_BIN_COUNT: u32 = ${activeBinCount}u;
const LINKED_BIN_COUNT: u32 = ${linkedBinCount}u;`,
      body: `
  var running = 0u;
  for (var activeBin = 0u; activeBin < ACTIVE_BIN_COUNT; activeBin = activeBin + 1u) {
    let cell = cubeOffset + activeBin * LINKED_BIN_COUNT + index;
    running = running + cube[cell];
    cube[cell] = running;
  }`
    })
  );
  return nodes;
}

/**
 * Reads one linked histogram out of a built cube: the active brush is snapped to the active
 * view's bins, `[first, last]`, and the output is `C[last][b] - C[first - 1][b]` for every bin.
 * An empty or out-of-domain brush writes zeros. Cost is `O(linked bins)`, independent of rows.
 *
 * @internal
 */
export function createHistogramCubeReadoutNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: HistogramCubePairProps & {
    /** Active dimension's five-word selection control block; words 1 and 2 hold the brush. */
    brushState: GraphDataView<'uint32'>;
    /** The linked view's caller-owned bin counts. */
    output: GraphDataView<'uint32'>;
    condition?: GPUCommandGraphCPUCondition<Parameters>;
  }
): GPUCommandNode<Parameters> {
  const activeBinCount = props.active.binning.binCount;
  const linkedBinCount = props.linked.binning.binCount;
  const format = props.active.binning.format;
  const shaderType = getScalarType(format);
  const activeBinningSource = createBinningSource('active', props.active.binning);
  const decode = (word: number): string =>
    format === 'float32'
      ? `bitcast<f32>(brushState[brushStateOffset + ${word}u])`
      : format === 'sint32'
        ? `bitcast<i32>(brushState[brushStateOffset + ${word}u])`
        : `brushState[brushStateOffset + ${word}u]`;
  return createWGSLKernelNode(graph, {
    id: `${props.id}/readout`,
    operation: 'GPUCrossfilterHistogramCube',
    variant: 'readout',
    bindings: [
      {name: 'cube', view: props.cube, type: 'u32', access: 'read'},
      {name: 'brushState', view: props.brushState, type: 'u32', access: 'read'},
      {name: 'outputCounts', view: props.output, type: 'u32', access: 'read_write'},
      ...activeBinningSource.bindings
    ],
    // One workgroup; each lane strides over the linked bins.
    invocationCount: CUBE_WORKGROUP_SIZE,
    workgroupSize: CUBE_WORKGROUP_SIZE,
    condition: props.condition,
    declarations: `
${IS_FINITE_BITS_WGSL}
const ACTIVE_BIN_COUNT: u32 = ${activeBinCount}u;
const LINKED_BIN_COUNT: u32 = ${linkedBinCount}u;
${activeBinningSource.declarations}`,
    body: `
  let brushMinimum: ${shaderType} = ${decode(1)};
  let brushMaximum: ${shaderType} = ${decode(2)};
  let domainMinimum = activeMinimum();
  let domainMaximum = activeMaximum();
  var first = 0u;
  var last = ACTIVE_BIN_COUNT - 1u;
  var empty = !(domainMaximum >= domainMinimum) || brushMaximum < domainMinimum ||
    brushMinimum > domainMaximum;
  if (!empty) {
    if (brushMinimum > domainMinimum) { first = activeBin(brushMinimum); }
    if (brushMaximum < domainMaximum) { last = activeBin(brushMaximum); }
    empty = first == ${SENTINEL_BIN} || last == ${SENTINEL_BIN} || first > last;
  }
  for (var bin = index; bin < LINKED_BIN_COUNT; bin = bin + ${CUBE_WORKGROUP_SIZE}u) {
    var total = 0u;
    if (!empty) {
      total = cube[cubeOffset + last * LINKED_BIN_COUNT + bin];
      if (first > 0u) {
        total = total - cube[cubeOffset + (first - 1u) * LINKED_BIN_COUNT + bin];
      }
    }
    outputCounts[outputCountsOffset + bin] = total;
  }`
  });
}

/** One row of the accumulation: bin both values and add one to the cell they share. */
function getRowBodySource(hasMask: boolean, accumulate: string): string {
  return `${hasMask ? 'if (rowMask[rowMaskOffset + index * MASK_STRIDE] != 0u) {' : '{'}
      let activeBinIndex = activeBin(activeValues[activeValuesOffset + index * ACTIVE_STRIDE]);
      let linkedBinIndex = linkedBin(linkedValues[linkedValuesOffset + index * LINKED_STRIDE]);
      if (activeBinIndex != ${SENTINEL_BIN} && linkedBinIndex != ${SENTINEL_BIN}) {
        let cellIndex = activeBinIndex * LINKED_BIN_COUNT + linkedBinIndex;
        ${accumulate}
      }
    }`;
}

/**
 * WGSL for one view's binning plus the storage bindings it needs. It reproduces `GPUHistogram`'s
 * bin assignment, including inclusive domains, the last-bin maximum, degenerate domains, and
 * binary search over edges, but returns `0xffffffff` for rejected values instead of skipping.
 * Declares `${prefix}Minimum()`, `${prefix}Maximum()` and `${prefix}Bin(value)`.
 */
function createBinningSource(
  prefix: string,
  binning: HistogramCubeBinning
): {declarations: string; bindings: WGSLKernelBinding[]} {
  const {format, binCount} = binning;
  const type = getShaderType(format);
  const bindings: WGSLKernelBinding[] = [];
  const finiteCheck =
    format === 'float32' ? 'if (!isFiniteBits(value)) { return 0xffffffffu; }' : '';
  const binCountName = `${prefix.toUpperCase()}_BINS`;

  if (binning.edges !== undefined) {
    const edges = binning.edges;
    let getEdge: string;
    if (Array.isArray(edges)) {
      const values = (edges as readonly number[]).map(value => getLiteral(value, format));
      getEdge = `const ${prefix}Edges = array<${type}, ${values.length}>(${values.join(', ')});
fn ${prefix}Edge(index: u32) -> ${type} { return ${prefix}Edges[index]; }`;
    } else {
      bindings.push({
        name: `${prefix}EdgeValues`,
        view: edges as GraphDataView,
        type: getScalarType(format),
        access: 'read'
      });
      getEdge = `fn ${prefix}Edge(index: u32) -> ${type} { return ${prefix}EdgeValues[${prefix}EdgeValuesOffset + index]; }`;
    }
    return {
      bindings,
      declarations: `
const ${binCountName}: u32 = ${binCount}u;
${getEdge}
fn ${prefix}Minimum() -> ${type} { return ${prefix}Edge(0u); }
fn ${prefix}Maximum() -> ${type} { return ${prefix}Edge(${binCountName}); }
fn ${prefix}Bin(value: ${type}) -> u32 {
  ${finiteCheck}
  if (!(value >= ${prefix}Minimum() && value <= ${prefix}Maximum())) { return 0xffffffffu; }
  var lower = 0u;
  var upper = ${binCountName};
  loop {
    if (lower >= upper) { break; }
    let middle = lower + (upper - lower) / 2u;
    if (value < ${prefix}Edge(middle + 1u)) {
      upper = middle;
    } else {
      lower = middle + 1u;
    }
  }
  return min(lower, ${binCountName} - 1u);
}`
    };
  }

  const domain = binning.domain;
  let bounds: string;
  if (Array.isArray(domain)) {
    bounds = `fn ${prefix}Minimum() -> ${type} { return ${getLiteral(domain[0], format)}; }
fn ${prefix}Maximum() -> ${type} { return ${getLiteral(domain[1], format)}; }`;
  } else {
    bindings.push({
      name: `${prefix}DomainValues`,
      view: domain as GraphDataView,
      type: getScalarType(format),
      access: 'read'
    });
    bounds = `fn ${prefix}Minimum() -> ${type} { return ${prefix}DomainValues[${prefix}DomainValuesOffset]; }
fn ${prefix}Maximum() -> ${type} { return ${prefix}DomainValues[${prefix}DomainValuesOffset + 1u]; }`;
  }
  const multiplierBitCount = 32 - Math.clz32(binCount);
  const integerHelper =
    format === 'float32'
      ? ''
      : `fn ${prefix}MultiplyDivideFloor(numerator: u32, multiplier: u32, denominator: u32) -> u32 {
  var quotient = 0u;
  var remainder = 0u;
  var bitIndex = ${multiplierBitCount}u;
  loop {
    bitIndex = bitIndex - 1u;
    let doubledThreshold = denominator - remainder;
    if (remainder >= doubledThreshold) {
      remainder = remainder - doubledThreshold;
      quotient = quotient * 2u + 1u;
    } else {
      remainder = remainder * 2u;
      quotient = quotient * 2u;
    }
    if (((multiplier >> bitIndex) & 1u) != 0u) {
      let additionThreshold = denominator - numerator;
      if (remainder >= additionThreshold) {
        remainder = remainder - additionThreshold;
        quotient = quotient + 1u;
      } else {
        remainder = remainder + numerator;
      }
    }
    if (bitIndex == 0u) { break; }
  }
  return quotient;
}`;
  const binCalculation =
    format === 'float32'
      ? `let ratio = (value - minimum) / (maximum - minimum);
  return min(u32(ratio * f32(${binCountName})), ${binCountName} - 1u);`
      : format === 'uint32'
        ? `return ${prefix}MultiplyDivideFloor(value - minimum, ${binCountName}, maximum - minimum);`
        : `let orderedValue = bitcast<u32>(value) ^ 0x80000000u;
  let orderedMinimum = bitcast<u32>(minimum) ^ 0x80000000u;
  let orderedMaximum = bitcast<u32>(maximum) ^ 0x80000000u;
  return ${prefix}MultiplyDivideFloor(
    orderedValue - orderedMinimum, ${binCountName}, orderedMaximum - orderedMinimum);`;
  return {
    bindings,
    declarations: `
const ${binCountName}: u32 = ${binCount}u;
${bounds}
${integerHelper}
fn ${prefix}Bin(value: ${type}) -> u32 {
  let minimum = ${prefix}Minimum();
  let maximum = ${prefix}Maximum();
  if (!(maximum >= minimum)) { return 0xffffffffu; }
  ${finiteCheck}
  if (value < minimum || value > maximum) { return 0xffffffffu; }
  if (maximum == minimum) { return 0u; }
  if (value == maximum) { return ${binCountName} - 1u; }
  ${binCalculation}
}`
  };
}

function getStride(view: GraphDataView): number {
  return view.byteStride / Uint32Array.BYTES_PER_ELEMENT;
}

function getShaderType(format: GPUCrossfilterScalarFormat): 'f32' | 'i32' | 'u32' {
  return format === 'float32' ? 'f32' : format === 'sint32' ? 'i32' : 'u32';
}

/** Kernel binding element type for a scalar format. */
function getScalarType(format: GPUCrossfilterScalarFormat): 'f32' | 'i32' | 'u32' {
  return getShaderType(format);
}

function getLiteral(value: number, format: GPUCrossfilterScalarFormat): string {
  if (format === 'uint32') return `${value}u`;
  if (format === 'sint32') return `${value}`;
  return getWGSLFloatLiteral(value);
}
