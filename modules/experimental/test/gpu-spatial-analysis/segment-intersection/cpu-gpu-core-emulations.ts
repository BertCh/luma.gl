// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// CPU stand-ins for upstream cooperative gpu-core primitives, for the CPU graph executor
// (wgsl-graph-executor.ts). The wgsl_reflect interpreter runs invocations one after another, so
// kernels that synchronize a workgroup with barriers (radix and bitonic sorts, scans) cannot run
// there. Specs install these methods with useCPUExecutorTestHooks(), which restores the real
// primitives after every case and works even when contributors were loaded by earlier specs.
// Each produces one compute node whose encode step does the work directly on the executor memory.
// The upstream primitives are trusted (repo rule); these mirror their documented contracts only.

import type {GraphDataView} from '@luma.gl/gpgpu/gpu-core';

type CPUBuffer = {backing: ArrayBuffer};
type EncodeContext = {getBuffer: (view: GraphDataView) => CPUBuffer};

function getWords(context: EncodeContext, view: GraphDataView, length?: number): Uint32Array {
  return new Uint32Array(
    context.getBuffer(view).backing,
    view.byteOffset,
    length ?? (view.length * view.rowByteLength) / 4
  );
}

function createCPUNode(id: string, views: GraphDataView[], run: (context: EncodeContext) => void) {
  return {
    id,
    type: 'compute' as const,
    resources: views.map(view => ({buffer: view, usage: 'storage-read-write' as const})),
    compile: () => ({encode: (context: EncodeContext) => run(context)})
  };
}

/** Stable ascending sort of `(keys, values)` by the low `keyBits` bits of each key (radix semantics). */
export class CPUGPUSort {
  readonly props: {
    id?: string;
    keys: GraphDataView;
    values: GraphDataView;
    outputKeys: GraphDataView;
    outputValues: GraphDataView;
    keyBits?: number;
    direction?: 'ascending' | 'descending';
  };

  constructor(props: CPUGPUSort['props']) {
    this.props = props;
  }

  getCommandNodes(): unknown[] {
    const {keys, values, outputKeys, outputValues} = this.props;
    const keyBits = this.props.keyBits ?? 32;
    const mask = keyBits >= 32 ? 0xffffffff : 2 ** keyBits - 1;
    const descending = this.props.direction === 'descending';
    return [
      createCPUNode(
        `${this.props.id ?? 'sort'}-cpu`,
        [keys, values, outputKeys, outputValues],
        context => {
          const keyWords = getWords(context, keys);
          const valueWords = getWords(context, values);
          const order = Array.from(keyWords.keys());
          const masked = (row: number) => (keyWords[row] & mask) >>> 0;
          order.sort((left, right) => {
            const difference = masked(left) - masked(right);
            return (descending ? -difference : difference) || left - right;
          });
          const sortedKeys = order.map(row => keyWords[row]);
          const sortedValues = order.map(row => valueWords[row]);
          getWords(context, outputKeys).set(sortedKeys);
          getWords(context, outputValues).set(sortedValues);
        }
      )
    ];
  }
}

/** Inclusive or exclusive u32 prefix sum (wrapping), `output[i]` over `input[0..i]`. */
export class CPUGPUScan {
  readonly props: {
    id?: string;
    input: GraphDataView;
    output: GraphDataView;
    mode?: 'inclusive' | 'exclusive';
  };

  constructor(props: CPUGPUScan['props']) {
    this.props = props;
  }

  getCommandNodes(): unknown[] {
    const {input, output} = this.props;
    const exclusive = this.props.mode === 'exclusive';
    return [
      createCPUNode(`${this.props.id ?? 'scan'}-cpu`, [input, output], context => {
        const source = getWords(context, input);
        const destination = getWords(context, output);
        let sum = 0;
        for (let row = 0; row < source.length; row++) {
          const value = source[row];
          if (exclusive) {
            destination[row] = sum;
            sum = (sum + value) >>> 0;
          } else {
            sum = (sum + value) >>> 0;
            destination[row] = sum;
          }
        }
      })
    ];
  }
}

/** Order-preserving compaction of `input` rows whose `flags` are non-zero; writes `count`. */
export class CPUGPUCompaction {
  readonly props: {
    id?: string;
    input: GraphDataView;
    flags: GraphDataView;
    output: GraphDataView;
    count: GraphDataView;
  };

  constructor(props: CPUGPUCompaction['props']) {
    this.props = props;
  }

  getCommandNodes(): unknown[] {
    const {input, flags, output, count} = this.props;
    return [
      createCPUNode(
        `${this.props.id ?? 'compaction'}-cpu`,
        [input, flags, output, count],
        context => {
          const source = getWords(context, input);
          const flagWords = getWords(context, flags);
          const destination = getWords(context, output);
          let written = 0;
          for (let row = 0; row < source.length; row++) {
            if (flagWords[row] !== 0) {
              if (written < destination.length) {
                destination[written] = source[row];
              }
              written++;
            }
          }
          getWords(context, count, 1)[0] = written;
        }
      )
    ];
  }
}

/**
 * Complete binary BVH exactly as `@luma.gl/gpgpu` `GPUBVH` lays it out: node `i` has children
 * `2i + 1` and `2i + 2` for `i < leafCapacity - 1`; leaves are nodes `leafCapacity - 1 + slot`
 * with children `0xffffffff`. Invalid or non-finite source bounds (or min > max) and unused slots
 * keep inverted bounds (`+MAX`, `-MAX`) and parents take component-wise min and max. `leafIds`
 * hold the source row (or `sourceIds[row]`) for the first `min(rows, leafCapacity)` slots and
 * `0xffffffff` elsewhere; `count` is the full source row count and `overflow` is 1 when it exceeds
 * `leafCapacity`.
 */
export class CPUGPUBVH {
  readonly id: string;
  readonly props: CPUGPUBVHProps;
  readonly minima: GraphDataView;
  readonly maxima: GraphDataView;
  readonly sourceIds?: GraphDataView;
  readonly leafCapacity: number;
  readonly nodeMinima: GraphDataView;
  readonly nodeMaxima: GraphDataView;
  readonly nodeChildren: GraphDataView;
  readonly leafIds: GraphDataView;
  readonly count: GraphDataView;
  readonly overflow: GraphDataView;
  readonly dimension: 2 | 3;
  readonly nodeCount: number;
  readonly internalNodeCount: number;
  readonly levelCount: number;
  readonly rootNode = 0;
  readonly topology = 'complete-binary' as const;
  readonly updatePolicy = 'refit' as const;

  constructor(props: CPUGPUBVHProps) {
    this.props = props;
    this.id = props.id ?? 'gpu-bvh';
    this.minima = props.minima;
    this.maxima = props.maxima;
    this.sourceIds = props.sourceIds;
    this.leafCapacity = props.leafCapacity;
    this.nodeMinima = props.nodeMinima;
    this.nodeMaxima = props.nodeMaxima;
    this.nodeChildren = props.nodeChildren;
    this.leafIds = props.leafIds;
    this.count = props.count;
    this.overflow = props.overflow;
    this.dimension = props.minima.format === 'float32x2' ? 2 : 3;
    if (!Number.isInteger(Math.log2(props.leafCapacity))) {
      throw new Error(`${this.id} leafCapacity must be a positive power of two`);
    }
    this.nodeCount = props.leafCapacity * 2 - 1;
    this.internalNodeCount = props.leafCapacity - 1;
    this.levelCount = Math.log2(props.leafCapacity) + 1;
  }

  getCommandNodes(): unknown[] {
    const {
      minima,
      maxima,
      sourceIds,
      nodeMinima,
      nodeMaxima,
      nodeChildren,
      leafIds,
      count,
      overflow
    } = this;
    const {leafCapacity, internalNodeCount, nodeCount, dimension} = this;
    const views = [
      minima,
      maxima,
      ...(sourceIds ? [sourceIds] : []),
      nodeMinima,
      nodeMaxima,
      nodeChildren,
      leafIds,
      count,
      overflow
    ];
    return [
      createCPUNode(`${this.id}-cpu`, views, context => {
        const getFloats = (view: GraphDataView) =>
          new Float32Array(
            context.getBuffer(view).backing,
            view.byteOffset,
            (view.length * view.rowByteLength) / 4
          );
        const sourceMinima = getFloats(minima);
        const sourceMaxima = getFloats(maxima);
        const outMinima = getFloats(nodeMinima);
        const outMaxima = getFloats(nodeMaxima);
        const children = getWords(context, nodeChildren);
        const ids = getWords(context, leafIds);
        const sourceRows = minima.length;
        const storedCount = Math.min(sourceRows, leafCapacity);
        const maximum = Math.fround(3.402823466e38);
        for (let node = 0; node < nodeCount; node++) {
          for (let axis = 0; axis < dimension; axis++) {
            outMinima[node * dimension + axis] = maximum;
            outMaxima[node * dimension + axis] = -maximum;
          }
          const isInternal = node < internalNodeCount;
          children[node * 2] = isInternal ? node * 2 + 1 : 0xffffffff;
          children[node * 2 + 1] = isInternal ? node * 2 + 2 : 0xffffffff;
        }
        for (let slot = 0; slot < leafCapacity; slot++) {
          ids[slot] = 0xffffffff;
          if (slot >= storedCount) {
            continue;
          }
          ids[slot] = slot;
          let valid = true;
          for (let axis = 0; axis < dimension; axis++) {
            const low = sourceMinima[slot * dimension + axis];
            const high = sourceMaxima[slot * dimension + axis];
            valid = valid && Math.abs(low) <= maximum && Math.abs(high) <= maximum && low <= high;
          }
          if (valid) {
            const leaf = internalNodeCount + slot;
            for (let axis = 0; axis < dimension; axis++) {
              outMinima[leaf * dimension + axis] = sourceMinima[slot * dimension + axis];
              outMaxima[leaf * dimension + axis] = sourceMaxima[slot * dimension + axis];
            }
          }
        }
        for (let node = internalNodeCount - 1; node >= 0; node--) {
          for (let axis = 0; axis < dimension; axis++) {
            const left = (node * 2 + 1) * dimension + axis;
            const right = (node * 2 + 2) * dimension + axis;
            outMinima[node * dimension + axis] = Math.min(outMinima[left], outMinima[right]);
            outMaxima[node * dimension + axis] = Math.max(outMaxima[left], outMaxima[right]);
          }
        }
        if (sourceIds) {
          const remap = getWords(context, sourceIds);
          for (let slot = 0; slot < storedCount; slot++) {
            ids[slot] = remap[ids[slot]];
          }
        }
        getWords(context, count, 1)[0] = sourceRows;
        getWords(context, overflow, 1)[0] = sourceRows > leafCapacity ? 1 : 0;
      })
    ];
  }
}

/** Constructor props of {@link CPUGPUBVH}; the same shape as `GPUBVHProps` (`strategy` is ignored). */
export type CPUGPUBVHProps = {
  id?: string;
  strategy?: string;
  minima: GraphDataView;
  maxima: GraphDataView;
  sourceIds?: GraphDataView;
  leafCapacity: number;
  nodeMinima: GraphDataView;
  nodeMaxima: GraphDataView;
  nodeChildren: GraphDataView;
  leafIds: GraphDataView;
  count: GraphDataView;
  overflow: GraphDataView;
};

/** The stand-ins, keyed by the gpu-core export they replace. */
export const CPU_GPU_CORE_EMULATIONS = {
  GPUSort: CPUGPUSort,
  GPUScan: CPUGPUScan,
  GPUCompaction: CPUGPUCompaction,
  GPUBVH: CPUGPUBVH
};
