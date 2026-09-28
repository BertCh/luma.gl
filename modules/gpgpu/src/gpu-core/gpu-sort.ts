// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {type GPUCommandNode, createGPUComputeCommandNode} from './gpu-command-node';
import {type Binding} from '@luma.gl/core';
import {Computation} from '@luma.gl/engine';
import {
  GPUCommandGraph,
  type GraphBufferUse,
  type GraphDataView,
  GraphVectorView
} from './gpu-command-graph';
import {getGraphVectorData} from './graph-vector-view-utils';
import {getChunkedSortNodes} from './gpu-sort-chunks';
import {doGraphDataViewsOverlap} from './graph-data-view-utils';
import {
  getBoundedDispatchLayout,
  getBoundedInvocationIndexSource,
  type GPUBoundedDispatchLayout
} from './gpu-dispatch-utils';
import {getGPUScanCommandNodesWithDispatchLimit, GPUScan} from './gpu-scan';
import {getGPUShaderSubgroupStrategy} from './gpu-subgroup-utils';
import {
  createTransientView,
  getViewBinding,
  getViewElementOffset,
  validatePackedUint32View
} from './graph-data-view-utils';

const BITONIC_WORKGROUP_SIZE = 256;
const RADIX_WORKGROUP_SIZE = 256;
const RADIX_MASK_WORD_COUNT = RADIX_WORKGROUP_SIZE / 32;
const INVALID_INDEX = 0xffffffff;
const MAXIMUM_LOGICAL_LENGTH = 0x80000000;
const AUTO_BITONIC_MAXIMUM_LENGTH = BITONIC_WORKGROUP_SIZE;

/** Radix digit widths whose per-workgroup ballot masks fit the guaranteed workgroup storage. */
const SUPPORTED_RADIX_DIGIT_BITS = [4, 8] as const;
/** Default digit width. Benchmarks across Apple, NVIDIA and Mali parts keep four-bit digits. */
const DEFAULT_RADIX_DIGIT_BITS = 4;
/**
 * Keys handled by one thread per radix tile.
 *
 * One element per thread makes every workgroup cover only {@link RADIX_WORKGROUP_SIZE} keys, which
 * inflates both the per-pass histogram and the number of hierarchical scan levels above it. Eight
 * keys per thread is the value shipping WebGPU sorters use: the same shader structure with an
 * eighth of the dispatch and scan overhead.
 */
const DEFAULT_RADIX_ELEMENTS_PER_THREAD = 8;
/** Guaranteed WebGPU workgroup storage, used to reject digit widths that cannot fit. */
const MINIMUM_WORKGROUP_STORAGE_BYTES = 16384;

/** Sort implementation requested by {@link GPUSort}. */
export type GPUSortAlgorithm = 'auto' | 'bitonic' | 'radix';

/** Final key ordering requested by {@link GPUSort}. */
export type GPUSortDirection = 'ascending' | 'descending';

/** Properties for one graph-native stable uint32 key/value sort. */
export type GPUSortProps = {
  /** Prefix for generated graph node and transient resource IDs. */
  id?: string;
  /** Packed unsigned sort keys. */
  keys: GraphDataView<'uint32'> | GraphVectorView<'uint32'>;
  /** Packed payload values paired row-for-row with `keys`. */
  values: GraphDataView<'uint32'> | GraphVectorView<'uint32'>;
  /** Caller-owned sorted key destination. */
  outputKeys: GraphDataView<'uint32'> | GraphVectorView<'uint32'>;
  /** Caller-owned payload destination permuted with the keys. */
  outputValues: GraphDataView<'uint32'> | GraphVectorView<'uint32'>;
  /** Requested implementation. Defaults to `'auto'`. */
  algorithm?: GPUSortAlgorithm;
  /** Requested final order. Defaults to `'ascending'`. */
  direction?: GPUSortDirection;
  /** Number of significant least-significant key bits processed by radix sort. Defaults to `32`. */
  keyBits?: number;
  /**
   * Bits consumed by one radix pass. Defaults to `4`.
   *
   * Eight-bit digits halve the pass count for wide keys at the cost of a 256-bucket ballot mask in
   * workgroup storage. Four-bit digits remain the portable default.
   */
  digitBits?: 4 | 8;
  /**
   * Keys processed by one radix thread. Defaults to `8`.
   *
   * The tile a workgroup covers is `256 * elementsPerThread` keys, so raising this shrinks the
   * per-pass histogram, the hierarchical scan above it, and the dispatch count in equal measure.
   */
  elementsPerThread?: number;
};

/** Resolved radix tiling, histogram shape, and pass count for one {@link GPUSort}. */
export type GPUSortRadixPlan = {
  /** Bits consumed per radix pass. */
  digitBits: number;
  /** Buckets scanned per pass, `2 ** digitBits`. */
  bucketCount: number;
  /** Keys processed by one thread within a tile. */
  elementsPerThread: number;
  /** Keys covered by one workgroup, `256 * elementsPerThread`. */
  tileSize: number;
  /** Workgroups dispatched per radix pass. */
  workgroupCount: number;
  /** Entries in the digit-major histogram scanned between each histogram and scatter pass. */
  histogramLength: number;
  /** Radix passes required to cover `keyBits`. */
  passCount: number;
  /** Workgroup storage bytes the scatter pass reserves for ballot masks and bucket cursors. */
  workgroupStorageBytes: number;
};

type BitonicStage = {
  blockWidth: number;
  compareStride: number;
};

/**
 * Stable graph-native sort for paired packed uint32 keys and values.
 *
 * @remarks
 * The operation is out-of-place. Inputs and outputs are caller-owned graph views, while all
 * implementation scratch is graph-owned. `getCommandNodes()` only records work; the caller retains
 * control of graph compilation, command encoding, submission, and optional readback.
 */
export class GPUSort {
  /** Prefix for generated graph node and transient resource IDs. */
  readonly id: string;
  /** Packed unsigned sort keys. */
  readonly keys: GraphDataView<'uint32'> | GraphVectorView<'uint32'>;
  /** Packed payload values paired with the keys. */
  readonly values: GraphDataView<'uint32'> | GraphVectorView<'uint32'>;
  /** Caller-owned sorted key destination. */
  readonly outputKeys: GraphDataView<'uint32'> | GraphVectorView<'uint32'>;
  /** Caller-owned sorted payload destination. */
  readonly outputValues: GraphDataView<'uint32'> | GraphVectorView<'uint32'>;
  /** Algorithm requested by the caller. */
  readonly algorithm: GPUSortAlgorithm;
  /** Final key ordering. */
  readonly direction: GPUSortDirection;
  /** Significant least-significant key bits processed by the radix implementation. */
  readonly keyBits: number;
  /** Bits consumed by one radix pass. */
  readonly digitBits: number;
  /** Keys processed by one radix thread within its workgroup's tile. */
  readonly elementsPerThread: number;
  /** Concrete implementation selected after resolving `'auto'`. */
  readonly resolvedAlgorithm: Exclude<GPUSortAlgorithm, 'auto'>;

  /**
   * Creates and validates an out-of-place stable sort description.
   *
   * @throws If views are not packed `uint32` data, lengths differ, writable buffers alias, or an
   * option or row count is unsupported.
   */
  constructor(props: GPUSortProps) {
    this.id = props.id ?? 'gpu-sort';
    this.keys = props.keys;
    this.values = props.values;
    this.outputKeys = props.outputKeys;
    this.outputValues = props.outputValues;
    this.algorithm = props.algorithm ?? 'auto';
    this.direction = props.direction ?? 'ascending';
    this.keyBits = props.keyBits ?? 32;
    this.digitBits = props.digitBits ?? DEFAULT_RADIX_DIGIT_BITS;
    this.elementsPerThread = props.elementsPerThread ?? DEFAULT_RADIX_ELEMENTS_PER_THREAD;

    for (const [name, view] of [
      ['keys', this.keys],
      ['values', this.values],
      ['outputKeys', this.outputKeys],
      ['outputValues', this.outputValues]
    ] as const) {
      for (const chunk of getGraphVectorData(view))
        validatePackedUint32View(chunk, `${this.id} ${name}`);
    }
    if (!['auto', 'bitonic', 'radix'].includes(this.algorithm)) {
      throw new Error(`${this.id} algorithm must be auto, bitonic, or radix`);
    }
    if (!['ascending', 'descending'].includes(this.direction)) {
      throw new Error(`${this.id} direction must be ascending or descending`);
    }
    if (!Number.isInteger(this.keyBits) || this.keyBits < 1 || this.keyBits > 32) {
      throw new Error(`${this.id} keyBits must be an integer from 1 to 32`);
    }
    if (!SUPPORTED_RADIX_DIGIT_BITS.includes(this.digitBits as 4 | 8)) {
      throw new Error(`${this.id} digitBits must be 4 or 8`);
    }
    if (
      !Number.isInteger(this.elementsPerThread) ||
      this.elementsPerThread < 1 ||
      this.elementsPerThread > 32
    ) {
      throw new Error(`${this.id} elementsPerThread must be an integer from 1 to 32`);
    }
    const radixStorageBytes = getRadixScatterStorageBytes(this.digitBits);
    if (radixStorageBytes > MINIMUM_WORKGROUP_STORAGE_BYTES) {
      throw new Error(`${this.id} digitBits exceeds the guaranteed workgroup storage size`);
    }
    if (
      this.values.length !== this.keys.length ||
      this.outputKeys.length !== this.keys.length ||
      this.outputValues.length !== this.keys.length
    ) {
      throw new Error(`${this.id} key, value, and output lengths must match`);
    }
    if (this.keys.length > MAXIMUM_LOGICAL_LENGTH) {
      throw new Error(`${this.id} supports at most ${MAXIMUM_LOGICAL_LENGTH} rows`);
    }
    validateSeparateWritableBuffers(this);

    this.resolvedAlgorithm =
      this.algorithm === 'auto'
        ? this.keys.length <= AUTO_BITONIC_MAXIMUM_LENGTH
          ? 'bitonic'
          : 'radix'
        : this.algorithm;
  }

  /**
   * Resolved radix tiling and pass structure, for diagnostics and autotuning.
   *
   * @remarks
   * Reported for every sort, including one resolved to the bitonic implementation, because the
   * plan depends only on the row count and the caller's radix options.
   */
  get radixPlan(): GPUSortRadixPlan {
    return getGPUSortRadixPlan(this);
  }

  /**
   * Adds the selected sort implementation and graph-owned scratch to a command graph.
   *
   * Empty inputs add no nodes; one-row inputs add one copy pass. This method does not compile,
   * encode, submit, or read back commands.
   *
   * @remarks
   * Every pass is wait-free. Histogram, hierarchical scan and scatter synchronize only through
   * `workgroupBarrier()` and separate dispatches; no pass spins on another workgroup's result, so
   * the implementation does not depend on forward-progress guarantees WebGPU does not make.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const nodes: GPUCommandNode<Parameters>[] = [];
    nodes.push(
      ...getGPUSortCommandNodesWithDispatchLimit(
        this,
        graph,
        graph.device.limits.maxComputeWorkgroupsPerDimension
      )
    );

    return nodes;
  }
}

/** Adds one stable sort while propagating an explicit bounded dispatch limit. @internal */
export function getGPUSortCommandNodesWithDispatchLimit<Parameters>(
  sort: GPUSort,
  graph: GPUCommandGraph<Parameters>,
  maxComputeWorkgroupsPerDimension: number
): readonly GPUCommandNode<Parameters>[] {
  for (const view of [sort.keys, sort.values, sort.outputKeys, sort.outputValues]) {
    for (const chunk of getGraphVectorData(view)) {
      if (chunk.buffer.graph !== graph)
        throw new Error(`${sort.id} views must belong to the target graph`);
    }
  }
  if (
    [sort.keys, sort.values, sort.outputKeys, sort.outputValues].some(
      view => view instanceof GraphVectorView
    )
  ) {
    return getChunkedSortNodes(graph, sort, maxComputeWorkgroupsPerDimension);
  }
  return getAtomicSortNodes(sort as AtomicSort, graph, maxComputeWorkgroupsPerDimension);
}

type AtomicSort = Omit<GPUSort, 'keys' | 'values' | 'outputKeys' | 'outputValues'> & {
  keys: GraphDataView<'uint32'>;
  values: GraphDataView<'uint32'>;
  outputKeys: GraphDataView<'uint32'>;
  outputValues: GraphDataView<'uint32'>;
};

function getAtomicSortNodes<Parameters>(
  sort: AtomicSort,
  graph: GPUCommandGraph<Parameters>,
  maxComputeWorkgroupsPerDimension: number
): readonly GPUCommandNode<Parameters>[] {
  const nodes: GPUCommandNode<Parameters>[] = [];

  if (sort.keys.length === 0) {
    return nodes;
  }
  if (sort.keys.length === 1) {
    nodes.push(...addCopyPairPass(graph, sort));
    return nodes;
  }

  const dispatchLayout = getBoundedDispatchLayout(
    'GPUSort',
    sort.keys.length,
    RADIX_WORKGROUP_SIZE,
    maxComputeWorkgroupsPerDimension
  );

  if (sort.resolvedAlgorithm === 'bitonic') {
    nodes.push(...addBitonicSort(graph, sort, dispatchLayout, maxComputeWorkgroupsPerDimension));
  } else {
    nodes.push(...addRadixSort(graph, sort, maxComputeWorkgroupsPerDimension));
  }

  return nodes;
}

/** Enforces out-of-place writes and distinct writable destinations. */
function validateSeparateWritableBuffers(sort: GPUSort): void {
  const keys = getGraphVectorData(sort.outputKeys);
  const values = getGraphVectorData(sort.outputValues);
  const inputs = [...getGraphVectorData(sort.keys), ...getGraphVectorData(sort.values)];
  if (
    [...keys, ...values].some(output => inputs.some(input => input.buffer === output.buffer)) ||
    keys.some(key => values.some(value => key.buffer === value.buffer))
  )
    throw new Error(`${sort.id} outputs must use separate buffers from inputs and each other`);
  for (const chunks of [keys, values])
    for (const [index, chunk] of chunks.entries()) {
      if (chunks.slice(0, index).some(previous => doGraphDataViewsOverlap(previous, chunk)))
        throw new Error(`${sort.id} output chunks must not overlap`);
    }
}

/** Copies key/value pairs without allocating additional sort scratch. */
function addCopyPairPass<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  sort: AtomicSort,
  inputKeys: GraphDataView<'uint32'> = sort.keys,
  inputValues: GraphDataView<'uint32'> = sort.values,
  identifier = 'copy-pair',
  dispatchLayout: GPUBoundedDispatchLayout = {x: 1, y: 1, z: 1}
): readonly GPUCommandNode<Parameters>[] {
  const nodes: GPUCommandNode<Parameters>[] = [];
  const source = /* wgsl */ `
const ELEMENT_COUNT: u32 = ${sort.keys.length}u;
const KEYS_OFFSET: u32 = ${getViewElementOffset(inputKeys)}u;
const VALUES_OFFSET: u32 = ${getViewElementOffset(inputValues)}u;
const OUTPUT_KEYS_OFFSET: u32 = ${getViewElementOffset(sort.outputKeys)}u;
const OUTPUT_VALUES_OFFSET: u32 = ${getViewElementOffset(sort.outputValues)}u;
@group(0) @binding(0) var<storage, read> keys: array<u32>;
@group(0) @binding(1) var<storage, read> values: array<u32>;
@group(0) @binding(2) var<storage, read_write> outputKeys: array<u32>;
@group(0) @binding(3) var<storage, read_write> outputValues: array<u32>;

@compute @workgroup_size(${RADIX_WORKGROUP_SIZE}) fn main(
  @builtin(local_invocation_index) localInvocationIndex: u32,
  @builtin(workgroup_id) workgroupId: vec3<u32>
) {
  ${getBoundedInvocationIndexSource(dispatchLayout, RADIX_WORKGROUP_SIZE)}
  if (index >= ELEMENT_COUNT) { return; }
  outputKeys[OUTPUT_KEYS_OFFSET + index] = keys[KEYS_OFFSET + index];
  outputValues[OUTPUT_VALUES_OFFSET + index] = values[VALUES_OFFSET + index];
}`;
  nodes.push(
    ...addComputationPass(graph, {
      id: `${sort.id}-${identifier}`,
      source,
      resources: [
        {buffer: inputKeys, usage: 'storage-read'},
        {buffer: inputValues, usage: 'storage-read'},
        {buffer: sort.outputKeys, usage: 'storage-write'},
        {buffer: sort.outputValues, usage: 'storage-write'}
      ],
      bindings: {
        keys: inputKeys,
        values: inputValues,
        outputKeys: sort.outputKeys,
        outputValues: sort.outputValues
      },
      dispatchLayout
    })
  );

  return nodes;
}

/** Adds padded-index initialization, every bitonic stage, and the final stable gather. */
function addBitonicSort<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  sort: AtomicSort,
  dispatchLayout: GPUBoundedDispatchLayout,
  maxComputeWorkgroupsPerDimension: number
): readonly GPUCommandNode<Parameters>[] {
  const nodes: GPUCommandNode<Parameters>[] = [];
  const paddedLength = getNextPowerOfTwo(sort.keys.length);
  if (paddedLength <= BITONIC_WORKGROUP_SIZE) {
    nodes.push(...addLocalBitonicSortPass(graph, sort, paddedLength));
    return nodes;
  }
  const paddedDispatchLayout = getBoundedDispatchLayout(
    'GPUSort bitonic',
    paddedLength,
    BITONIC_WORKGROUP_SIZE,
    maxComputeWorkgroupsPerDimension
  );
  const indicesA = createTransientView(
    graph,
    `${sort.id}-bitonic-indices-a`,
    'uint32',
    paddedLength
  );
  const indicesB = createTransientView(
    graph,
    `${sort.id}-bitonic-indices-b`,
    'uint32',
    paddedLength
  );
  nodes.push(
    ...addBitonicInitializePass(graph, sort, indicesA, paddedLength, paddedDispatchLayout)
  );

  let currentIndices = indicesA;
  let nextIndices = indicesB;
  for (const stage of getBitonicStages(paddedLength)) {
    nodes.push(
      ...addBitonicStagePass(
        graph,
        sort,
        currentIndices,
        nextIndices,
        paddedLength,
        stage,
        paddedDispatchLayout
      )
    );
    [currentIndices, nextIndices] = [nextIndices, currentIndices];
  }
  nodes.push(...addBitonicGatherPass(graph, sort, currentIndices, dispatchLayout));

  return nodes;
}

/** Sorts one complete stable bitonic network in workgroup memory with one graph dispatch. */
function addLocalBitonicSortPass<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  sort: AtomicSort,
  paddedLength: number
): readonly GPUCommandNode<Parameters>[] {
  const nodes: GPUCommandNode<Parameters>[] = [];
  const descending = sort.direction === 'descending';
  const useSubgroups =
    getGPUShaderSubgroupStrategy(graph.device, {requiresSubgroupId: true}) === 'subgroups';
  const source = /* wgsl */ `
${useSubgroups ? 'enable subgroups;\nrequires subgroup_id;' : ''}
const INVALID_INDEX: u32 = ${INVALID_INDEX}u;
const LOGICAL_LENGTH: u32 = ${sort.keys.length}u;
const PADDED_LENGTH: u32 = ${paddedLength}u;
const KEYS_OFFSET: u32 = ${getViewElementOffset(sort.keys)}u;
const VALUES_OFFSET: u32 = ${getViewElementOffset(sort.values)}u;
const OUTPUT_KEYS_OFFSET: u32 = ${getViewElementOffset(sort.outputKeys)}u;
const OUTPUT_VALUES_OFFSET: u32 = ${getViewElementOffset(sort.outputValues)}u;
@group(0) @binding(0) var<storage, read> keys: array<u32>;
@group(0) @binding(1) var<storage, read> values: array<u32>;
@group(0) @binding(2) var<storage, read_write> outputKeys: array<u32>;
@group(0) @binding(3) var<storage, read_write> outputValues: array<u32>;
var<workgroup> indices: array<u32, ${paddedLength}>;
var<workgroup> cachedKeys: array<u32, ${paddedLength}>;

fn comes_before(leftIndex: u32, rightIndex: u32) -> bool {
  let leftValid = leftIndex != INVALID_INDEX && leftIndex < LOGICAL_LENGTH;
  let rightValid = rightIndex != INVALID_INDEX && rightIndex < LOGICAL_LENGTH;
  if (leftValid != rightValid) { return leftValid; }
  if (!leftValid) { return false; }
  let leftKey = cachedKeys[leftIndex];
  let rightKey = cachedKeys[rightIndex];
  if (leftKey == rightKey) { return leftIndex < rightIndex; }
  return ${descending ? 'leftKey > rightKey' : 'leftKey < rightKey'};
}

@compute @workgroup_size(${paddedLength}) fn main(
  ${useSubgroups ? '@builtin(subgroup_invocation_id) subgroupInvocationId: u32,\n  @builtin(subgroup_size) subgroupSize: u32,\n  @builtin(subgroup_id) subgroupId: u32' : '@builtin(local_invocation_index) localInvocationIndex: u32'}
) {
${useSubgroups ? getSubgroupLocalBitonicShader() : getPortableLocalBitonicShader()}
}`;
  nodes.push(
    ...addComputationPass(graph, {
      id: `${sort.id}-bitonic-local`,
      source,
      resources: [
        {buffer: sort.keys, usage: 'storage-read'},
        {buffer: sort.values, usage: 'storage-read'},
        {buffer: sort.outputKeys, usage: 'storage-write'},
        {buffer: sort.outputValues, usage: 'storage-write'}
      ],
      bindings: {
        keys: sort.keys,
        values: sort.values,
        outputKeys: sort.outputKeys,
        outputValues: sort.outputValues
      },
      dispatchLayout: {x: 1, y: 1, z: 1}
    })
  );

  return nodes;
}

/** Emits the original shared-memory sorting network for CORE WebGPU devices. */
function getPortableLocalBitonicShader(): string {
  return /* wgsl */ `
  indices[localInvocationIndex] = select(
    INVALID_INDEX,
    localInvocationIndex,
    localInvocationIndex < LOGICAL_LENGTH
  );
  if (localInvocationIndex < LOGICAL_LENGTH) {
    cachedKeys[localInvocationIndex] = keys[KEYS_OFFSET + localInvocationIndex];
  } else {
    cachedKeys[localInvocationIndex] = 0u;
  }
  workgroupBarrier();

  for (var blockWidth = 2u; blockWidth <= PADDED_LENGTH; blockWidth <<= 1u) {
    for (var compareStride = blockWidth >> 1u; compareStride > 0u; compareStride >>= 1u) {
      let partnerIndex = localInvocationIndex ^ compareStride;
      if (partnerIndex > localInvocationIndex) {
        let leftIndex = indices[localInvocationIndex];
        let rightIndex = indices[partnerIndex];
        let ascending = (localInvocationIndex & blockWidth) == 0u;
        let shouldSwap = select(
          comes_before(leftIndex, rightIndex),
          comes_before(rightIndex, leftIndex),
          ascending
        );
        indices[localInvocationIndex] = select(leftIndex, rightIndex, shouldSwap);
        indices[partnerIndex] = select(rightIndex, leftIndex, shouldSwap);
      }
      workgroupBarrier();
    }
  }

  if (localInvocationIndex < LOGICAL_LENGTH) {
    let sourceIndex = indices[localInvocationIndex];
    outputKeys[OUTPUT_KEYS_OFFSET + localInvocationIndex] = cachedKeys[sourceIndex];
    outputValues[OUTPUT_VALUES_OFFSET + localInvocationIndex] = values[VALUES_OFFSET + sourceIndex];
  }`;
}

/** Uses register shuffles for every compare/exchange contained by one subgroup. */
function getSubgroupLocalBitonicShader(): string {
  return /* wgsl */ `
  let lane = subgroupId * subgroupSize + subgroupInvocationId;
  var currentIndex = select(INVALID_INDEX, lane, lane < LOGICAL_LENGTH);
  if (lane < LOGICAL_LENGTH) {
    cachedKeys[lane] = keys[KEYS_OFFSET + lane];
  } else {
    cachedKeys[lane] = 0u;
  }
  workgroupBarrier();

  for (var blockWidth = 2u; blockWidth <= PADDED_LENGTH; blockWidth <<= 1u) {
    for (var compareStride = blockWidth >> 1u; compareStride > 0u; compareStride >>= 1u) {
      var partnerIndex = INVALID_INDEX;
      if (compareStride < subgroupSize) {
        partnerIndex = subgroupShuffleXor(currentIndex, compareStride);
      } else {
        indices[lane] = currentIndex;
        workgroupBarrier();
        partnerIndex = indices[lane ^ compareStride];
      }

      let lowerLane = (lane & compareStride) == 0u;
      let leftIndex = select(partnerIndex, currentIndex, lowerLane);
      let rightIndex = select(currentIndex, partnerIndex, lowerLane);
      let ascending = (lane & blockWidth) == 0u;
      let shouldSwap = select(
        comes_before(leftIndex, rightIndex),
        comes_before(rightIndex, leftIndex),
        ascending
      );
      let sortedLeft = select(leftIndex, rightIndex, shouldSwap);
      let sortedRight = select(rightIndex, leftIndex, shouldSwap);
      currentIndex = select(sortedRight, sortedLeft, lowerLane);

      if (compareStride >= subgroupSize) {
        workgroupBarrier();
      }
    }
  }

  if (lane < LOGICAL_LENGTH) {
    outputKeys[OUTPUT_KEYS_OFFSET + lane] = cachedKeys[currentIndex];
    outputValues[OUTPUT_VALUES_OFFSET + lane] = values[VALUES_OFFSET + currentIndex];
  }`;
}

/** Initializes logical indices and invalid padding for a power-of-two bitonic network. */
function addBitonicInitializePass<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  sort: AtomicSort,
  indices: GraphDataView<'uint32'>,
  paddedLength: number,
  dispatchLayout: GPUBoundedDispatchLayout
): readonly GPUCommandNode<Parameters>[] {
  const nodes: GPUCommandNode<Parameters>[] = [];
  const source = /* wgsl */ `
const INVALID_INDEX: u32 = ${INVALID_INDEX}u;
const LOGICAL_LENGTH: u32 = ${sort.keys.length}u;
const PADDED_LENGTH: u32 = ${paddedLength}u;
const INDICES_OFFSET: u32 = ${getViewElementOffset(indices)}u;
@group(0) @binding(0) var<storage, read_write> indices: array<u32>;

@compute @workgroup_size(${BITONIC_WORKGROUP_SIZE}) fn main(
  @builtin(local_invocation_index) localInvocationIndex: u32,
  @builtin(workgroup_id) workgroupId: vec3<u32>
) {
  ${getBoundedInvocationIndexSource(dispatchLayout, BITONIC_WORKGROUP_SIZE)}
  if (index < PADDED_LENGTH) {
    indices[INDICES_OFFSET + index] = select(INVALID_INDEX, index, index < LOGICAL_LENGTH);
  }
}`;
  nodes.push(
    ...addComputationPass(graph, {
      id: `${sort.id}-bitonic-initialize`,
      source,
      resources: [{buffer: indices, usage: 'storage-write'}],
      bindings: {indices},
      dispatchLayout
    })
  );

  return nodes;
}

/** Adds one compare/exchange stage of the stable bitonic sorting network. */
function addBitonicStagePass<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  sort: AtomicSort,
  indicesIn: GraphDataView<'uint32'>,
  indicesOut: GraphDataView<'uint32'>,
  paddedLength: number,
  stage: BitonicStage,
  dispatchLayout: GPUBoundedDispatchLayout
): readonly GPUCommandNode<Parameters>[] {
  const nodes: GPUCommandNode<Parameters>[] = [];
  const descending = sort.direction === 'descending';
  const source = /* wgsl */ `
const INVALID_INDEX: u32 = ${INVALID_INDEX}u;
const LOGICAL_LENGTH: u32 = ${sort.keys.length}u;
const PADDED_LENGTH: u32 = ${paddedLength}u;
const BLOCK_WIDTH: u32 = ${stage.blockWidth}u;
const COMPARE_STRIDE: u32 = ${stage.compareStride}u;
const KEYS_OFFSET: u32 = ${getViewElementOffset(sort.keys)}u;
const INDICES_IN_OFFSET: u32 = ${getViewElementOffset(indicesIn)}u;
const INDICES_OUT_OFFSET: u32 = ${getViewElementOffset(indicesOut)}u;
@group(0) @binding(0) var<storage, read> keys: array<u32>;
@group(0) @binding(1) var<storage, read> indicesIn: array<u32>;
@group(0) @binding(2) var<storage, read_write> indicesOut: array<u32>;

fn is_valid(index: u32) -> bool {
  return index != INVALID_INDEX && index < LOGICAL_LENGTH;
}

fn comes_before(leftIndex: u32, rightIndex: u32) -> bool {
  let leftValid = is_valid(leftIndex);
  let rightValid = is_valid(rightIndex);
  if (leftValid != rightValid) { return leftValid; }
  if (!leftValid) { return false; }
  let leftKey = keys[KEYS_OFFSET + leftIndex];
  let rightKey = keys[KEYS_OFFSET + rightIndex];
  if (leftKey == rightKey) { return leftIndex < rightIndex; }
  return ${descending ? 'leftKey > rightKey' : 'leftKey < rightKey'};
}

@compute @workgroup_size(${BITONIC_WORKGROUP_SIZE}) fn main(
  @builtin(local_invocation_index) localInvocationIndex: u32,
  @builtin(workgroup_id) workgroupId: vec3<u32>
) {
  ${getBoundedInvocationIndexSource(dispatchLayout, BITONIC_WORKGROUP_SIZE)}
  if (index >= PADDED_LENGTH) { return; }
  let partnerIndex = index ^ COMPARE_STRIDE;
  if (partnerIndex <= index) { return; }
  let leftIndex = indicesIn[INDICES_IN_OFFSET + index];
  let rightIndex = indicesIn[INDICES_IN_OFFSET + partnerIndex];
  let ascending = (index & BLOCK_WIDTH) == 0u;
  let shouldSwap = select(
    comes_before(leftIndex, rightIndex),
    comes_before(rightIndex, leftIndex),
    ascending
  );
  indicesOut[INDICES_OUT_OFFSET + index] = select(leftIndex, rightIndex, shouldSwap);
  indicesOut[INDICES_OUT_OFFSET + partnerIndex] = select(rightIndex, leftIndex, shouldSwap);
}`;
  nodes.push(
    ...addComputationPass(graph, {
      id: `${sort.id}-bitonic-${stage.blockWidth}-${stage.compareStride}`,
      source,
      resources: [
        {buffer: sort.keys, usage: 'storage-read'},
        {buffer: indicesIn, usage: 'storage-read'},
        {buffer: indicesOut, usage: 'storage-write'}
      ],
      bindings: {keys: sort.keys, indicesIn, indicesOut},
      dispatchLayout
    })
  );

  return nodes;
}

/** Gathers keys and payloads through the final sorted logical-index permutation. */
function addBitonicGatherPass<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  sort: AtomicSort,
  indices: GraphDataView<'uint32'>,
  dispatchLayout: GPUBoundedDispatchLayout
): readonly GPUCommandNode<Parameters>[] {
  const nodes: GPUCommandNode<Parameters>[] = [];
  const source = /* wgsl */ `
const LOGICAL_LENGTH: u32 = ${sort.keys.length}u;
const KEYS_OFFSET: u32 = ${getViewElementOffset(sort.keys)}u;
const VALUES_OFFSET: u32 = ${getViewElementOffset(sort.values)}u;
const INDICES_OFFSET: u32 = ${getViewElementOffset(indices)}u;
const OUTPUT_KEYS_OFFSET: u32 = ${getViewElementOffset(sort.outputKeys)}u;
const OUTPUT_VALUES_OFFSET: u32 = ${getViewElementOffset(sort.outputValues)}u;
@group(0) @binding(0) var<storage, read> keys: array<u32>;
@group(0) @binding(1) var<storage, read> values: array<u32>;
@group(0) @binding(2) var<storage, read> indices: array<u32>;
@group(0) @binding(3) var<storage, read_write> outputKeys: array<u32>;
@group(0) @binding(4) var<storage, read_write> outputValues: array<u32>;

@compute @workgroup_size(${BITONIC_WORKGROUP_SIZE}) fn main(
  @builtin(local_invocation_index) localInvocationIndex: u32,
  @builtin(workgroup_id) workgroupId: vec3<u32>
) {
  ${getBoundedInvocationIndexSource(dispatchLayout, BITONIC_WORKGROUP_SIZE)}
  if (index >= LOGICAL_LENGTH) { return; }
  let sourceIndex = indices[INDICES_OFFSET + index];
  outputKeys[OUTPUT_KEYS_OFFSET + index] = keys[KEYS_OFFSET + sourceIndex];
  outputValues[OUTPUT_VALUES_OFFSET + index] = values[VALUES_OFFSET + sourceIndex];
}`;
  nodes.push(
    ...addComputationPass(graph, {
      id: `${sort.id}-bitonic-gather`,
      source,
      resources: [
        {buffer: sort.keys, usage: 'storage-read'},
        {buffer: sort.values, usage: 'storage-read'},
        {buffer: indices, usage: 'storage-read'},
        {buffer: sort.outputKeys, usage: 'storage-write'},
        {buffer: sort.outputValues, usage: 'storage-write'}
      ],
      bindings: {
        keys: sort.keys,
        values: sort.values,
        indices,
        outputKeys: sort.outputKeys,
        outputValues: sort.outputValues
      },
      dispatchLayout
    })
  );

  return nodes;
}

/** Resolves the tiling, histogram shape, and pass count one radix sort would use. @internal */
export function getGPUSortRadixPlan(sort: {
  keys: {length: number};
  keyBits: number;
  digitBits: number;
  elementsPerThread: number;
}): GPUSortRadixPlan {
  const digitBits = sort.digitBits;
  const bucketCount = 2 ** digitBits;
  const elementsPerThread = sort.elementsPerThread;
  const tileSize = RADIX_WORKGROUP_SIZE * elementsPerThread;
  const workgroupCount = Math.max(1, Math.ceil(sort.keys.length / tileSize));
  const passCount = Math.ceil(sort.keyBits / digitBits);
  return {
    digitBits,
    bucketCount,
    elementsPerThread,
    tileSize,
    workgroupCount,
    histogramLength: bucketCount * workgroupCount,
    passCount,
    workgroupStorageBytes: getRadixScatterStorageBytes(digitBits)
  };
}

/** Workgroup storage the scatter pass reserves for ballot masks and per-bucket cursors. */
function getRadixScatterStorageBytes(digitBits: number): number {
  const bucketCount = 2 ** digitBits;
  return (bucketCount * RADIX_MASK_WORD_COUNT + bucketCount) * Uint32Array.BYTES_PER_ELEMENT;
}

/**
 * Adds stable least-significant-digit histogram, scan, and scatter partitions.
 *
 * Each workgroup owns a tile of `256 * elementsPerThread` keys addressed in a striped layout, so
 * the slot-major order the scatter pass ranks in is exactly ascending key index and the sort stays
 * stable without a second ordering pass.
 */
function addRadixSort<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  sort: AtomicSort,
  maxComputeWorkgroupsPerDimension: number
): readonly GPUCommandNode<Parameters>[] {
  const nodes: GPUCommandNode<Parameters>[] = [];
  const plan = getGPUSortRadixPlan(sort);
  const dispatchLayout = getBoundedDispatchLayout(
    'GPUSort radix',
    sort.keys.length,
    plan.tileSize,
    maxComputeWorkgroupsPerDimension
  );
  const scratchKeys =
    plan.passCount > 1
      ? createTransientView(graph, `${sort.id}-radix-scratch-keys`, 'uint32', sort.keys.length)
      : undefined;
  const scratchValues =
    plan.passCount > 1
      ? createTransientView(graph, `${sort.id}-radix-scratch-values`, 'uint32', sort.keys.length)
      : undefined;
  let currentKeys = sort.keys;
  let currentValues = sort.values;

  for (let digitIndex = 0; digitIndex < plan.passCount; digitIndex++) {
    const bitOffset = digitIndex * plan.digitBits;
    const digitBits = Math.min(plan.digitBits, sort.keyBits - bitOffset);
    const bucketCount = 2 ** digitBits;
    const histogram = createTransientView(
      graph,
      `${sort.id}-radix-digit-${bitOffset}-histogram`,
      'uint32',
      bucketCount * plan.workgroupCount
    );
    const offsets = createTransientView(
      graph,
      `${sort.id}-radix-digit-${bitOffset}-offsets`,
      'uint32',
      bucketCount * plan.workgroupCount
    );
    const writesFinalOutput = (plan.passCount - digitIndex) % 2 === 1;
    const nextKeys = writesFinalOutput ? sort.outputKeys : scratchKeys;
    const nextValues = writesFinalOutput ? sort.outputValues : scratchValues;
    if (!nextKeys || !nextValues) {
      throw new Error(`${sort.id} radix scratch is missing`);
    }
    nodes.push(
      ...addRadixHistogramPass(graph, sort, currentKeys, histogram, {
        bitOffset,
        digitBits,
        plan,
        dispatchLayout
      })
    );
    const scan = new GPUScan({
      id: `${sort.id}-radix-digit-${bitOffset}-scan`,
      input: histogram,
      output: offsets
    });
    nodes.push(
      ...getGPUScanCommandNodesWithDispatchLimit(scan, graph, maxComputeWorkgroupsPerDimension)
    );
    nodes.push(
      ...addRadixScatterPass(
        graph,
        sort,
        currentKeys,
        currentValues,
        offsets,
        nextKeys,
        nextValues,
        {
          bitOffset,
          digitBits,
          plan,
          dispatchLayout
        }
      )
    );
    currentKeys = nextKeys;
    currentValues = nextValues;
  }

  return nodes;
}

type RadixPassOptions = {
  bitOffset: number;
  digitBits: number;
  plan: GPUSortRadixPlan;
  dispatchLayout: GPUBoundedDispatchLayout;
};

/** Counts one radix digit per tile into a digit-major histogram suitable for global scan. */
function addRadixHistogramPass<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  sort: AtomicSort,
  keys: GraphDataView<'uint32'>,
  histogram: GraphDataView<'uint32'>,
  options: RadixPassOptions
): readonly GPUCommandNode<Parameters>[] {
  const nodes: GPUCommandNode<Parameters>[] = [];
  const {bitOffset, digitBits, plan, dispatchLayout} = options;
  const bucketCount = 2 ** digitBits;
  const descending = sort.direction === 'descending';
  const source = /* wgsl */ `
const ELEMENT_COUNT: u32 = ${sort.keys.length}u;
const BIT_OFFSET: u32 = ${bitOffset}u;
const BUCKET_COUNT: u32 = ${bucketCount}u;
const DIGIT_MASK: u32 = ${bucketCount - 1}u;
const WORKGROUP_COUNT: u32 = ${plan.workgroupCount}u;
const ELEMENTS_PER_THREAD: u32 = ${plan.elementsPerThread}u;
const TILE_SIZE: u32 = ${plan.tileSize}u;
const KEYS_OFFSET: u32 = ${getViewElementOffset(keys)}u;
const HISTOGRAM_OFFSET: u32 = ${getViewElementOffset(histogram)}u;
@group(0) @binding(0) var<storage, read> keys: array<u32>;
@group(0) @binding(1) var<storage, read_write> histogram: array<u32>;
var<workgroup> digitCounts: array<atomic<u32>, ${bucketCount}>;

@compute @workgroup_size(${RADIX_WORKGROUP_SIZE}) fn main(
  @builtin(local_invocation_index) localInvocationIndex: u32,
  @builtin(workgroup_id) workgroupId: vec3<u32>
) {
  let workgroupIndex =
    (workgroupId.z * ${dispatchLayout.y}u + workgroupId.y) * ${dispatchLayout.x}u + workgroupId.x;
  if (workgroupIndex >= WORKGROUP_COUNT) { return; }
  for (var bucket = localInvocationIndex; bucket < BUCKET_COUNT; bucket += ${RADIX_WORKGROUP_SIZE}u) {
    atomicStore(&digitCounts[bucket], 0u);
  }
  workgroupBarrier();

  let tileBase = workgroupIndex * TILE_SIZE;
  for (var slot = 0u; slot < ELEMENTS_PER_THREAD; slot++) {
    let index = tileBase + slot * ${RADIX_WORKGROUP_SIZE}u + localInvocationIndex;
    if (index < ELEMENT_COUNT) {
      let key = keys[KEYS_OFFSET + index];
      let digit = (key >> BIT_OFFSET) & DIGIT_MASK;
      let bucket = ${descending ? 'DIGIT_MASK - digit' : 'digit'};
      atomicAdd(&digitCounts[bucket], 1u);
    }
  }
  workgroupBarrier();

  for (var bucket = localInvocationIndex; bucket < BUCKET_COUNT; bucket += ${RADIX_WORKGROUP_SIZE}u) {
    histogram[HISTOGRAM_OFFSET + bucket * WORKGROUP_COUNT + workgroupIndex] =
      atomicLoad(&digitCounts[bucket]);
  }
}`;
  nodes.push(
    ...addComputationPass(graph, {
      id: `${sort.id}-radix-digit-${bitOffset}-histogram`,
      source,
      resources: [
        {buffer: keys, usage: 'storage-read'},
        {buffer: histogram, usage: 'storage-write'}
      ],
      bindings: {keys, histogram},
      dispatchLayout
    })
  );

  return nodes;
}

/**
 * Stably scatters one digit using workgroup ballot masks and digit-major global offsets.
 *
 * The tile's keys are visited one striped slot at a time. Within a slot the 256 threads rank
 * themselves through a per-bucket ballot mask; across slots a per-bucket cursor carries the count
 * already emitted. Striped addressing makes that slot-major visit order identical to ascending key
 * index, which is what keeps equal keys in their original order.
 */
function addRadixScatterPass<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  sort: AtomicSort,
  keys: GraphDataView<'uint32'>,
  values: GraphDataView<'uint32'>,
  offsets: GraphDataView<'uint32'>,
  outputKeys: GraphDataView<'uint32'>,
  outputValues: GraphDataView<'uint32'>,
  options: RadixPassOptions
): readonly GPUCommandNode<Parameters>[] {
  const nodes: GPUCommandNode<Parameters>[] = [];
  const {bitOffset, digitBits, plan, dispatchLayout} = options;
  const bucketCount = 2 ** digitBits;
  const descending = sort.direction === 'descending';
  const source = /* wgsl */ `
const ELEMENT_COUNT: u32 = ${sort.keys.length}u;
const BIT_OFFSET: u32 = ${bitOffset}u;
const BUCKET_COUNT: u32 = ${bucketCount}u;
const DIGIT_MASK: u32 = ${bucketCount - 1}u;
const WORKGROUP_COUNT: u32 = ${plan.workgroupCount}u;
const ELEMENTS_PER_THREAD: u32 = ${plan.elementsPerThread}u;
const TILE_SIZE: u32 = ${plan.tileSize}u;
const MASK_WORD_COUNT: u32 = ${RADIX_MASK_WORD_COUNT}u;
const MASK_COUNT: u32 = ${bucketCount * RADIX_MASK_WORD_COUNT}u;
const KEYS_OFFSET: u32 = ${getViewElementOffset(keys)}u;
const VALUES_OFFSET: u32 = ${getViewElementOffset(values)}u;
const OFFSETS_OFFSET: u32 = ${getViewElementOffset(offsets)}u;
const OUTPUT_KEYS_OFFSET: u32 = ${getViewElementOffset(outputKeys)}u;
const OUTPUT_VALUES_OFFSET: u32 = ${getViewElementOffset(outputValues)}u;
@group(0) @binding(0) var<storage, read> keys: array<u32>;
@group(0) @binding(1) var<storage, read> values: array<u32>;
@group(0) @binding(2) var<storage, read> offsets: array<u32>;
@group(0) @binding(3) var<storage, read_write> outputKeys: array<u32>;
@group(0) @binding(4) var<storage, read_write> outputValues: array<u32>;
var<workgroup> digitMasks: array<atomic<u32>, ${bucketCount * RADIX_MASK_WORD_COUNT}>;
var<workgroup> bucketCursors: array<u32, ${bucketCount}>;

@compute @workgroup_size(${RADIX_WORKGROUP_SIZE}) fn main(
  @builtin(local_invocation_index) localInvocationIndex: u32,
  @builtin(workgroup_id) workgroupId: vec3<u32>
) {
  let workgroupIndex =
    (workgroupId.z * ${dispatchLayout.y}u + workgroupId.y) * ${dispatchLayout.x}u + workgroupId.x;
  if (workgroupIndex >= WORKGROUP_COUNT) { return; }
  for (var bucket = localInvocationIndex; bucket < BUCKET_COUNT; bucket += ${RADIX_WORKGROUP_SIZE}u) {
    bucketCursors[bucket] = 0u;
  }

  let tileBase = workgroupIndex * TILE_SIZE;
  let maskWord = localInvocationIndex >> 5u;
  let precedingBits = (1u << (localInvocationIndex & 31u)) - 1u;
  for (var slot = 0u; slot < ELEMENTS_PER_THREAD; slot++) {
    for (var mask = localInvocationIndex; mask < MASK_COUNT; mask += ${RADIX_WORKGROUP_SIZE}u) {
      atomicStore(&digitMasks[mask], 0u);
    }
    workgroupBarrier();

    let index = tileBase + slot * ${RADIX_WORKGROUP_SIZE}u + localInvocationIndex;
    let valid = index < ELEMENT_COUNT;
    var key = 0u;
    var bucket = 0u;
    if (valid) {
      key = keys[KEYS_OFFSET + index];
      let digit = (key >> BIT_OFFSET) & DIGIT_MASK;
      bucket = ${descending ? 'DIGIT_MASK - digit' : 'digit'};
      atomicOr(&digitMasks[bucket * MASK_WORD_COUNT + maskWord], 1u << (localInvocationIndex & 31u));
    }
    workgroupBarrier();

    if (valid) {
      let maskBase = bucket * MASK_WORD_COUNT;
      var localRank = 0u;
      for (var word = 0u; word < maskWord; word++) {
        localRank += countOneBits(atomicLoad(&digitMasks[maskBase + word]));
      }
      localRank += countOneBits(atomicLoad(&digitMasks[maskBase + maskWord]) & precedingBits);
      let bucketOffset = offsets[OFFSETS_OFFSET + bucket * WORKGROUP_COUNT + workgroupIndex];
      let outputIndex = bucketOffset + bucketCursors[bucket] + localRank;
      outputKeys[OUTPUT_KEYS_OFFSET + outputIndex] = key;
      outputValues[OUTPUT_VALUES_OFFSET + outputIndex] = values[VALUES_OFFSET + index];
    }
    workgroupBarrier();

    for (var bucket = localInvocationIndex; bucket < BUCKET_COUNT; bucket += ${RADIX_WORKGROUP_SIZE}u) {
      var slotCount = 0u;
      for (var word = 0u; word < MASK_WORD_COUNT; word++) {
        slotCount += countOneBits(atomicLoad(&digitMasks[bucket * MASK_WORD_COUNT + word]));
      }
      bucketCursors[bucket] += slotCount;
    }
    workgroupBarrier();
  }
}`;
  nodes.push(
    ...addComputationPass(graph, {
      id: `${sort.id}-radix-digit-${bitOffset}-scatter`,
      source,
      resources: [
        {buffer: keys, usage: 'storage-read'},
        {buffer: values, usage: 'storage-read'},
        {buffer: offsets, usage: 'storage-read'},
        {buffer: outputKeys, usage: 'storage-write'},
        {buffer: outputValues, usage: 'storage-write'}
      ],
      bindings: {keys, values, offsets, outputKeys, outputValues},
      dispatchLayout
    })
  );

  return nodes;
}

/** Returns the smallest power of two greater than or equal to `length`. */
function getNextPowerOfTwo(length: number): number {
  let paddedLength = 1;
  while (paddedLength < length) {
    paddedLength *= 2;
  }
  return paddedLength;
}

/** Enumerates compare/exchange stages for a complete bitonic network. */
function getBitonicStages(paddedLength: number): BitonicStage[] {
  const stages: BitonicStage[] = [];
  for (let blockWidth = 2; blockWidth <= paddedLength; blockWidth *= 2) {
    for (let compareStride = blockWidth / 2; compareStride >= 1; compareStride /= 2) {
      stages.push({blockWidth, compareStride});
    }
  }
  return stages;
}

/** Wraps generated WGSL in a graph compute node with deferred physical buffer resolution. */
function addComputationPass<GraphParameters>(
  graph: GPUCommandGraph<GraphParameters>,
  props: {
    id: string;
    source: string;
    resources: GraphBufferUse[];
    bindings: Record<string, GraphDataView>;
    dispatchLayout: GPUBoundedDispatchLayout;
  }
): readonly GPUCommandNode<GraphParameters>[] {
  const nodes: GPUCommandNode<GraphParameters>[] = [];
  nodes.push(
    createGPUComputeCommandNode<GraphParameters>({
      id: props.id,
      resources: props.resources,
      compile: ({device}) => {
        const computation = new Computation(device, {
          id: props.id,
          source: props.source,
          shaderLayout: {
            bindings: Object.keys(props.bindings).map((name, location) => ({
              name,
              type: 'storage' as const,
              group: 0,
              location
            }))
          }
        });
        return {
          encode: ({computePass, getBuffer}) => {
            const bindings: Record<string, Binding> = {};
            for (const [name, view] of Object.entries(props.bindings)) {
              bindings[name] = getViewBinding(view, getBuffer);
            }
            computation.setBindings(bindings);
            computation.dispatch(
              computePass,
              props.dispatchLayout.x,
              props.dispatchLayout.y,
              props.dispatchLayout.z
            );
          },
          destroy: () => computation.destroy()
        };
      }
    })
  );

  return nodes;
}
