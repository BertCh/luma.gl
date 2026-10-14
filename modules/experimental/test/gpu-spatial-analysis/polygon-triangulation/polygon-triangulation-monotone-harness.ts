// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Runs the real WGSL kernels of the monotone path of `GPUPolygonTriangulation` in the
 * `wgsl_reflect` interpreter (test only, no GPU). The scans and the two stable sorts, which are
 * luma core operations, are replaced by CPU equivalents; every other node is the exact kernel
 * source the graph compiles. Gated kernels run as many workgroups as the GPU-written count asks
 * for, like an indirect dispatch.
 *
 * The exact orientation helper is replaced by an f32 determinant (the interpreter is slow on the
 * 256-bit accumulators), so use coordinates whose determinants are exact in f32 (integers below
 * about 2^11 in magnitude).
 */

import {createTransientView, GPUCommandGraph, type GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {WgslExec, WgslParser} from 'wgsl_reflect';
import {EXACT_ORIENTATION_WGSL} from '../../../src/utils/exact-orientation-wgsl';
import {createWGSLKernelNode, type WGSLKernelProps} from '../../../src/utils/wgsl-kernel-nodes';
import {getSizeRouteWGSL} from '../../../src/gpu-spatial-analysis/polygon-triangulation/polygon-triangulation-monotone-wgsl';
import {
  getMonotoneTriangulationNodes,
  type MonotoneStep,
  type MonotoneTriangulationProps
} from '../../../src/gpu-spatial-analysis/polygon-triangulation/polygon-triangulation-monotone';
import {getPolygonTriangulationIndexCount} from '../../../src/gpu-spatial-analysis/polygon-triangulation/gpu-polygon-triangulation';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import type {ModelPoint} from './polygon-triangulation-monotone-model';

const NONE = 0xffffffff;

const SIMPLE_ORIENT_WGSL = /* wgsl */ `
fn orientSign(a: vec2f, b: vec2f, c: vec2f) -> i32 {
  let determinant = (a.x - c.x) * (b.y - c.y) - (a.y - c.y) * (b.x - c.x);
  if (determinant > 0.0) { return 1; }
  if (determinant < 0.0) { return -1; }
  return 0;
}
`;

export type HarnessOptions = {
  /** Polygons with at least this many vertices are routed (default 1: all of them). */
  minimumVertices?: number;
  /** Polygons to mark as Earcut failures (route 2) after the size routing. */
  retryPolygons?: number[];
  /** Capacity of the monotone path (default: all vertices). */
  capacity?: number;
  /** Pairs of pointer-jumping rounds to unroll (default: derived from the capacity). */
  jumpPairCount?: number;
  /** `'index'` (default) fills the edge-slab index like the GPU builder, `'walk'` leaves it empty. */
  slabs?: 'index' | 'walk';
};

export type HarnessResult = {
  valid: number[];
  triangleCount: number[];
  indices: number[];
  /** First index slot of each polygon with a terminal entry. */
  sliceStarts: number[];
  /** Flattened input rows. */
  positions: ModelPoint[];
  /** First row of each polygon. */
  polygonRows: number[];
  routing: number[];
  /** Failure bits per polygon. */
  failures: number[];
  /** The eight statistics words of each polygon (failure, clean, holes, triangles, written). */
  stats: number[][];
  /** Polygons the harness's edge-slab index slabbed (a query walks the others). */
  slabbedPolygonCount: number;
  /** Names of the kernels that ran at least one workgroup. */
  executedKernels: string[];
};

type ViewData = {buffer: ArrayBuffer; view: GraphDataView};

/** Runs the kernels for `polygons` (each a list of rings, shell first). */
export function runMonotoneKernels(
  polygons: readonly (readonly (readonly ModelPoint[])[])[],
  options: HarnessOptions = {}
): HarnessResult {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'monotone-harness'});
  const positions: ModelPoint[] = [];
  const ringOffsets: number[] = [0];
  const polygonOffsets: number[] = [0];
  const polygonRows: number[] = [];
  for (const rings of polygons) {
    polygonRows.push(positions.length);
    for (const ring of rings) {
      positions.push(...ring);
      ringOffsets.push(positions.length);
    }
    polygonOffsets.push(ringOffsets.length - 1);
  }
  polygonRows.push(positions.length);
  const vertexCount = positions.length;
  const ringCount = ringOffsets.length - 1;
  const polygonCount = polygons.length;
  const indexCount = getPolygonTriangulationIndexCount(vertexCount, ringCount, polygonCount);

  const data = new Map<GraphDataView, ViewData>();
  const uint32View = (name: string, length: number) =>
    createTransientView(graph, name, 'uint32', length);
  const bufferOf = (graphView: GraphDataView): ArrayBuffer => {
    let entry = data.get(graphView);
    if (!entry) {
      entry = {
        buffer: new ArrayBuffer(Math.max(graphView.length, 1) * graphView.rowByteLength),
        view: graphView
      };
      data.set(graphView, entry);
    }
    return entry.buffer;
  };
  const words = (graphView: GraphDataView) => new Uint32Array(bufferOf(graphView));

  const positionsView = createTransientView(graph, 'positions', 'float32x2', vertexCount);
  const polygonOffsetsView = uint32View('polygon-offsets', polygonOffsets.length);
  const ringOffsetsView = uint32View('ring-offsets', ringOffsets.length);
  const routing = uint32View('routing', polygonCount);
  const indices = uint32View('indices', indexCount);
  const valid = uint32View('valid', polygonCount);
  const triangleCount = uint32View('triangle-count', polygonCount);
  const positionWords = new Float32Array(bufferOf(positionsView));
  positions.forEach(([x, y], row) => {
    positionWords[2 * row] = x;
    positionWords[2 * row + 1] = y;
  });
  words(polygonOffsetsView).set(polygonOffsets);
  words(ringOffsetsView).set(ringOffsets);
  words(indices).fill(0xdeadbeef);

  const steps: MonotoneStep[] = [];
  const props: MonotoneTriangulationProps = {
    id: 'm',
    operation: 'GPUPolygonTriangulation',
    positions: positionsView,
    polygonOffsets: polygonOffsetsView,
    ringOffsets: ringOffsetsView,
    routing,
    output: {indices, valid, triangleCount},
    capacity: options.capacity ?? Math.max(vertexCount, 1),
    minimumVertices: options.minimumVertices ?? 1,
    jumpPairCount: options.jumpPairCount,
    recorder: step => steps.push(step)
  };
  getMonotoneTriangulationNodes(graph, props);

  const executed = new Set<string>();
  let slabbedPolygonCount = 0;
  const run = (kernelProps: WGSLKernelProps<unknown>, requestedGroups: number) => {
    let groups = requestedGroups;
    if (groups <= 0) return;
    const node = createWGSLKernelNode(graph as unknown as GPUCommandGraph<unknown>, kernelProps);
    const [kernel] = getGPUCommandNodeKernelSources([node]);
    let source = kernel.source.replace(EXACT_ORIENTATION_WGSL, SIMPLE_ORIENT_WGSL);
    // The interpreter does not know the phony assignments that keep unused bindings alive.
    source = source
      .split('\n')
      .filter(line => !/^\s*_ = /.test(line))
      .join('\n');
    // The interpreter drops most threads of a large workgroup, so every thread becomes a
    // workgroup of one: same indices, same padded last group.
    const size = Number(/@workgroup_size\((\d+)\)/.exec(source)![1]);
    source = source
      .replace(`@workgroup_size(${size})`, '@workgroup_size(1)')
      .replace(`workgroupIndex * ${size}u`, 'workgroupIndex * 1u');
    groups *= size;
    const exec = new WgslExec(new WgslParser().parse(source));
    const bindGroup: Record<number, Uint32Array | Float32Array> = {};
    kernelProps.bindings.forEach((binding, location) => {
      const buffer = bufferOf(binding.view);
      bindGroup[location] =
        binding.type === 'f32' ? new Float32Array(buffer) : new Uint32Array(buffer);
    });
    try {
      exec.dispatchWorkgroups(
        'main',
        [groups, 1, 1],
        {0: bindGroup},
        {constants: kernel.constants}
      );
    } catch (error) {
      throw new Error(`kernel ${kernelProps.id} failed in the interpreter: ${String(error)}`);
    }
    executed.add(kernelProps.id);
  };
  const workgroupSizeOf = (kernelProps: WGSLKernelProps<unknown>) =>
    kernelProps.workgroupSize ?? 64;

  // Size routing: the Earcut kernel runs this snippet first; here it runs on its own.
  run(
    {
      id: 'm-size-route',
      operation: 'GPUPolygonTriangulation',
      bindings: [
        {name: 'polygonOffsets', view: polygonOffsetsView, type: 'u32', access: 'read'},
        {name: 'ringOffsets', view: ringOffsetsView, type: 'u32', access: 'read'},
        {name: 'routing', view: routing, type: 'u32', access: 'read_write'}
      ],
      invocationCount: polygonCount,
      workgroupSize: 64,
      declarations: `const VERTEX_COUNT: u32 = ${vertexCount}u;${getSizeRouteWGSL(props.minimumVertices)}`,
      body: 'routing[routingOffset + index] = sizeRouteOf(index);'
    },
    Math.ceil(polygonCount / 64)
  );
  // The optional forced retries (the Earcut kernel is not run here).
  for (const polygon of options.retryPolygons ?? []) {
    words(routing)[polygon] = 2;
  }

  /**
   * Fills the shared edge-slab index the way the GPU builder documents it (`polygon-edge-slabs.ts`):
   * per polygon about one slab per four vertices over the y range, every ring edge listed in each
   * slab between the slabs of its two ends, entries in a seeded random order (the GPU fill is an
   * atomic cursor, so its order is unspecified). `'walk'` leaves every polygon unslabbed.
   */
  function buildEdgeSlabs(step: Extract<MonotoneStep, {kind: 'slabs'}>, mode: 'index' | 'walk') {
    const params = words(step.params);
    const offsets = words(step.offsets);
    const entries = words(step.entries);
    const routedCountsView = steps
      .filter((entry): entry is Extract<MonotoneStep, {kind: 'kernel'}> => entry.kind === 'kernel')
      .flatMap(entry => entry.props.bindings)
      .find(binding => binding.name === 'routedCounts')!.view;
    const gateCounts = words(routedCountsView);
    const slabCapacity = Math.floor(vertexCount / 4) + 1;
    const entryCapacity = Math.max(4 * vertexCount, 1024);
    const float = new Float32Array(1);
    const bits = new Uint32Array(float.buffer);
    const slabOf = (y: number, count: number, origin: number, scale: number) => {
      if (count <= 1) return 0;
      float[0] = (Math.fround(y) - origin) * scale;
      return Math.min(Math.max(Math.floor(float[0]), 0), count - 1);
    };
    offsets.fill(0);
    entries.fill(NONE);
    for (let ring = 0; ring < ringCount; ring++) {
      const begin = ringOffsets[ring];
      const end = ringOffsets[ring + 1];
      for (let row = begin; row < end; row++) {
        entries[2 * row] = row;
        entries[2 * row + 1] = row + 1 === end ? begin : row + 1;
      }
    }
    const findRing = (row: number) => {
      let found = 0;
      for (let ring = 0; ring < ringCount; ring++) {
        if (ringOffsets[ring] <= row) found = ring;
      }
      return found;
    };
    let nextBase = 0;
    const slabLists: number[][] = Array.from({length: slabCapacity}, () => []);
    for (let polygon = 0; polygon < polygonCount; polygon++) {
      const first = ringOffsets[polygonOffsets[polygon]];
      const end = ringOffsets[polygonOffsets[polygon + 1]];
      let minimum = Infinity;
      let maximum = -Infinity;
      let finite = true;
      for (let row = first; row < end; row++) {
        const y = positions[row][1];
        if (!Number.isFinite(y) || !Number.isFinite(positions[row][0])) finite = false;
        minimum = Math.min(minimum, y);
        maximum = Math.max(maximum, y);
      }
      const count = end - first;
      let slabCount = 0;
      let scale = 0;
      if (mode === 'index' && gateCounts[polygon] >= 1 && finite && count >= 8) {
        const wanted = Math.min(Math.floor(count / 4), 4096);
        float[0] = wanted / (maximum - minimum);
        scale = float[0];
        if (scale > 0 && scale < 1e30 && nextBase + wanted <= slabCapacity) slabCount = wanted;
      }
      params[6 * polygon] = slabCount;
      params[6 * polygon + 1] = nextBase;
      float[0] = minimum;
      params[6 * polygon + 2] = slabCount ? bits[0] : 0;
      float[0] = scale;
      params[6 * polygon + 3] = slabCount ? bits[0] : 0;
      params[6 * polygon + 4] = first;
      params[6 * polygon + 5] = end;
      if (slabCount) {
        slabbedPolygonCount += 1;
        float[0] = minimum;
        const origin = float[0];
        float[0] = scale;
        const slabScale = float[0];
        for (let row = first; row < end; row++) {
          const ringIndex = findRing(row);
          const ringEnd = ringOffsets[ringIndex + 1];
          const nextRow = row + 1 === ringEnd ? ringOffsets[ringIndex] : row + 1;
          const low = slabOf(
            Math.min(positions[row][1], positions[nextRow][1]),
            slabCount,
            origin,
            slabScale
          );
          const high = slabOf(
            Math.max(positions[row][1], positions[nextRow][1]),
            slabCount,
            origin,
            slabScale
          );
          for (let slab = low; slab <= high; slab++) slabLists[nextBase + slab].push(row);
        }
        nextBase += slabCount;
      }
    }
    // Entries in a seeded shuffled order per slab, packed after the unslabbed region.
    let state = 12345;
    const random = () => {
      state = (Math.imul(state, 1103515245) + 12345) >>> 0;
      return state / 4294967296;
    };
    let cursor = 0;
    for (let slab = 0; slab < slabCapacity; slab++) {
      offsets[slab] = cursor;
      const list = slabLists[slab].slice();
      for (let i = list.length - 1; i > 0; i--) {
        const j = Math.floor(random() * (i + 1));
        [list[i], list[j]] = [list[j], list[i]];
      }
      for (const row of list) {
        // Entries beyond the capacity are dropped; the owners they belong to are walked instead.
        if (cursor < entryCapacity) {
          entries[2 * (vertexCount + cursor)] = row;
          entries[2 * (vertexCount + cursor) + 1] = entries[2 * row + 1];
        }
        cursor += 1;
      }
    }
    offsets[slabCapacity] = cursor;
  }

  const counts = (stepView: GraphDataView) => words(stepView);
  let countsView: GraphDataView | undefined;
  for (const step of steps) {
    if (step.kind === 'kernel') {
      const kernelProps = {...step.props, workgroupSize: step.props.workgroupSize ?? 64};
      const countsBinding = kernelProps.bindings.find(binding => binding.name === 'counts');
      if (countsBinding) countsView = countsBinding.view;
      let groups: number;
      if (step.gate) {
        const countWords = counts(countsView!);
        const items = countWords[step.gate.countRow] * step.gate.invocationsPerItem;
        groups = Math.ceil(items / step.gate.workgroupSize);
      } else {
        groups = Math.ceil(kernelProps.invocationCount / workgroupSizeOf(kernelProps));
      }
      run(kernelProps, groups);
    } else if (step.kind === 'scan') {
      const input = words(step.input);
      const output = words(step.output);
      let sum = 0;
      for (let i = 0; i < input.length; i++) {
        output[i] = sum;
        sum = (sum + input[i]) >>> 0;
      }
    } else if (step.kind === 'sort') {
      const keys = step.keys.map(words);
      const identity = words(step.indices);
      const sorted = words(step.sortedIndices);
      const order = Array.from(identity);
      order.sort((a, b) => {
        for (const key of keys) {
          if (key[a] !== key[b]) return key[a] < key[b] ? -1 : 1;
        }
        return 0;
      });
      sorted.set(order);
    } else {
      buildEdgeSlabs(step, options.slabs ?? 'index');
    }
  }
  const polygonStatsView = steps
    .filter((step): step is Extract<MonotoneStep, {kind: 'kernel'}> => step.kind === 'kernel')
    .flatMap(step => step.props.bindings)
    .find(binding => binding.name === 'polygonStats')!.view;
  const stats = words(polygonStatsView);
  const sliceStarts = [0];
  for (let polygon = 0; polygon < polygonCount; polygon++) {
    const rings = polygonOffsets[polygon + 1] - polygonOffsets[polygon];
    const rows = polygonRows[polygon + 1] - polygonRows[polygon];
    sliceStarts.push(sliceStarts[polygon] + 3 * (rows + 2 * rings - 4));
  }
  return {
    valid: Array.from(words(valid)),
    triangleCount: Array.from(words(triangleCount)),
    indices: Array.from(words(indices)),
    sliceStarts,
    positions,
    polygonRows,
    routing: Array.from(words(routing)),
    failures: Array.from({length: polygonCount}, (_, polygon) => stats[8 * polygon]),
    stats: Array.from({length: polygonCount}, (_, polygon) =>
      Array.from(stats.slice(8 * polygon, 8 * polygon + 8))
    ),
    slabbedPolygonCount,
    executedKernels: [...executed]
  };
}
