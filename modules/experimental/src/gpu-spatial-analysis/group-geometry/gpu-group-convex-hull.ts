// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUReduction,
  GPUScan,
  GPUSort,
  GraphVectorView,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {
  createFillNode,
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../utils/wgsl-kernel-nodes';
import {getSortKeyBits} from '../../utils/sorted-segment-sums';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';

const OPERATION = 'GPUGroupConvexHull';
const MAXIMUM_ROWS = 2 ** 24 - 1;
/** Quantized coordinates lie in `[0, 2^29]`, so every difference fits an `i32` and products fit 60 bits. */
const LATTICE_BITS = 29;
const LATTICE_KEY_BITS = 30;
/** Inputs with at least this many rows run the parallel chunk prefilter before the chain. */
const PREFILTER_MINIMUM_ROWS = 8192;
/** Sorted rows handled by one prefilter invocation. */
const PREFILTER_CHUNK_ROWS = 256;

/** Bit of `overflow` set when a hull has more vertices than `maximumVerticesPerGroup`. */
export const GPU_GROUP_CONVEX_HULL_GROUP_OVERFLOW = 1;
/** Bit of `overflow` set when the emitted hulls do not fit `totalCapacity`. */
export const GPU_GROUP_CONVEX_HULL_TOTAL_OVERFLOW = 2;

/**
 * Caller-owned outputs of {@link GPUGroupConvexHull}. Every output is rewritten on every encoding.
 *
 * Hulls are stored CSR-style. Group `g` owns `vertexIndices[offsets[g] .. offsets[g] + counts[g])`
 * in counter-clockwise order starting at the lexicographically smallest vertex (smallest x, then
 * smallest y on the lattice). A group whose hull exceeded a cap is dropped: its `counts` is 0 and
 * `sizes` still holds the true hull size.
 */
export type GPUGroupConvexHullOutput = {
  /** Original row index of each hull vertex; slots past the emitted total hold `0xffffffff`. */
  vertexIndices: GraphDataView<'uint32'>;
  /** Hull positions `[x, y]`, aligned with `vertexIndices` (NaN past the emitted total). */
  vertexPositions?: GraphDataView<'float32x2'>;
  /** Start of each group in `vertexIndices`, `groupCount + 1` rows (the last is the total). */
  offsets: GraphDataView<'uint32'>;
  /** Emitted vertex count per group: 0 for an empty, dropped or unassigned group. */
  counts: GraphDataView<'uint32'>;
  /** True hull vertex count per group, before any cap (0 for an empty group). */
  sizes?: GraphDataView<'uint32'>;
  /**
   * One row, a bit set: {@link GPU_GROUP_CONVEX_HULL_GROUP_OVERFLOW} and
   * {@link GPU_GROUP_CONVEX_HULL_TOTAL_OVERFLOW}. `0` when every hull was emitted.
   */
  overflow: GraphDataView<'uint32'>;
};

/**
 * Properties for {@link GPUGroupConvexHull}.
 *
 * Per-frame (no recompile): the contents of every input buffer. Topology (needs a new graph):
 * view lengths, `groupCount`, `noiseLabel`, `maximumVerticesPerGroup` and `totalCapacity`.
 */
export type GPUGroupConvexHullProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'group-convex-hull'`. */
  id?: string;
  /** Planar points (fewer than `2^24` rows). Rows with a non-finite coordinate are excluded. */
  positions: GraphDataView<'float32x2'>;
  /** Label per row; labels `>= groupCount` or equal to `noiseLabel` are excluded. */
  labels: GraphDataView<'uint32'>;
  /** Number of groups. */
  groupCount: number;
  /** Optional label that marks noise (only relevant below `groupCount`). */
  noiseLabel?: number;
  /** Largest hull emitted per group. Larger hulls are dropped and flagged. At least 3. */
  maximumVerticesPerGroup: number;
  /** Capacity of `vertexIndices`: hulls that would overflow it are dropped and flagged. */
  totalCapacity: number;
  /** Caller-owned outputs. */
  output: GPUGroupConvexHullOutput;
};

/**
 * Convex hull of every label group by Andrew's monotone chain, with an exact orientation
 * predicate. It outlines clusters, hot regions and trajectory footprints.
 *
 * ## Exactness
 *
 * Float32 cross products are not exact, so coordinates are first snapped to an integer lattice
 * with one global power-of-two scale chosen so the data extent spans `2^29` lattice steps
 * (about `2e-9` of the extent, finer than float32's `6e-8`). The orientation of three lattice
 * points is evaluated exactly with 64-bit integer arithmetic emulated in WGSL, so the hull of the
 * lattice points is exact and matches a CPU monotone chain bit for bit. Points that snap to the
 * same lattice point are one point (the lowest row index represents it). The vertex positions
 * returned are the original float32 coordinates.
 *
 * ## Semantics
 *
 * - Rows are stably sorted by `(label, x, y)` with three `GPUSort` passes, so equal lattice points
 *   keep ascending row order and the lowest row index is the hull vertex.
 * - Collinear points are not hull vertices (monotone chain pops on `cross <= 0`). A group of one
 *   distinct point has a 1-vertex hull, and a group whose points are all collinear has the 2
 *   extreme points.
 * - Vertices are counter-clockwise, ring not closed, starting at the smallest `(x, y)` vertex.
 * - Output is bounded: see {@link GPUGroupConvexHullOutput}. A hull over
 *   `maximumVerticesPerGroup` is dropped whole. Remaining hulls are admitted in group order until
 *   `totalCapacity` runs out: the first group that does not fit and every later group are dropped
 *   (never truncated into a wrong polygon), and `sizes` still reports their true sizes.
 *
 * ## Cost
 *
 * Sorting and orientation are parallel. Inputs of at least 8192 rows first run a chunk prefilter:
 * every 256 consecutive sorted rows get their own monotone chain in parallel, and only chunk hull
 * vertices (the hull of a union is the hull of the chunk hulls) are compacted for the per-group
 * chain, so one group of millions of points no longer runs a serial walk over every row. The
 * result is identical with and without the prefilter.
 *
 * The contributor never compiles, encodes, submits or reads back.
 */
export class GPUGroupConvexHull implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUGroupConvexHullProps;
  /** Number of groups. */
  readonly groupCount: number;

  constructor(props: GPUGroupConvexHullProps) {
    this.id = props.id ?? 'group-convex-hull';
    this.props = props;
    const {id} = this;
    const {output} = props;
    for (const [name, view] of [
      ['positions', props.positions],
      ['labels', props.labels]
    ] as const) {
      if ((view as unknown) instanceof GraphVectorView) {
        throw new Error(`${id} ${name} must be a single packed view, not a chunked vector`);
      }
    }
    validatePackedView(props.positions, ['float32x2'], `${id} positions`);
    const rows = props.positions.length;
    if (rows < 1) {
      throw new Error(`${id} needs at least one row`);
    }
    if (rows > MAXIMUM_ROWS) {
      throw new Error(`${id} supports fewer than 2^24 rows`);
    }
    validatePackedUint32View(props.labels, `${id} labels`);
    if (props.labels.length !== rows) {
      throw new Error(`${id} labels length must equal positions length`);
    }
    this.groupCount = props.groupCount;
    if (
      !Number.isInteger(this.groupCount) ||
      this.groupCount < 1 ||
      this.groupCount > MAXIMUM_ROWS
    ) {
      throw new Error(`${id} groupCount must be a positive integer below 2^24`);
    }
    if (
      props.noiseLabel !== undefined &&
      (!Number.isInteger(props.noiseLabel) || props.noiseLabel < 0 || props.noiseLabel > 0xffffffff)
    ) {
      throw new Error(`${id} noiseLabel must be a uint32`);
    }
    if (!Number.isInteger(props.maximumVerticesPerGroup) || props.maximumVerticesPerGroup < 3) {
      throw new Error(`${id} maximumVerticesPerGroup must be an integer of at least 3`);
    }
    if (!Number.isInteger(props.totalCapacity) || props.totalCapacity < 1) {
      throw new Error(`${id} totalCapacity must be a positive integer`);
    }
    const groups = this.groupCount;
    const checks = [
      ['vertexIndices', output.vertexIndices, ['uint32'], props.totalCapacity],
      ['vertexPositions', output.vertexPositions, ['float32x2'], props.totalCapacity],
      ['offsets', output.offsets, ['uint32'], groups + 1],
      ['counts', output.counts, ['uint32'], groups],
      ['sizes', output.sizes, ['uint32'], groups],
      ['overflow', output.overflow, ['uint32'], 1]
    ] as const;
    for (const [name, view, formats, length] of checks) {
      if (!view) {
        continue;
      }
      validatePackedView(view, formats, `${id} output.${name}`);
      if (view.length < length) {
        throw new Error(`${id} output.${name} must hold at least ${length} rows`);
      }
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      checks.map(([, view]) => view),
      [props.positions, props.labels]
    );
  }

  /** Returns key, lattice, three sort passes, offsets, chain, bound and emit nodes in order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id, groupCount} = this;
    const {output, positions, labels, maximumVerticesPerGroup, totalCapacity} = props;
    validateGraphViewsBelongToGraph(id, graph, [positions, labels, ...Object.values(output)]);
    const rows = positions.length;
    const f32 = (name: string, length: number) =>
      createTransientView(graph, `${id}-${name}`, 'float32', length);
    const u32 = (name: string, length: number) =>
      createTransientView(graph, `${id}-${name}`, 'uint32', length);
    const read = (name: string, view: GraphDataView, type: 'u32' | 'f32'): WGSLKernelBinding => ({
      name,
      view,
      type,
      access: 'read'
    });
    const write = (
      name: string,
      view: GraphDataView,
      type: 'u32' | 'f32' | 'atomic<u32>' = 'f32'
    ): WGSLKernelBinding => ({name, view, type, access: 'read_write'});
    const kernel = (
      step: string,
      invocationCount: number,
      bindings: WGSLKernelBinding[],
      body: string,
      declarations = ''
    ) =>
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-${step}`,
        operation: OPERATION,
        variant: step,
        bindings,
        invocationCount,
        declarations: `const GROUP_COUNT: u32 = ${groupCount}u;\n${declarations}`,
        body
      });
    const nodes: GPUCommandNode<Parameters>[] = [];

    // 1. Group keys, valid flags, planar coordinates and per-group counts.
    const groupKeys = u32('group-keys', rows);
    const validFlags = u32('valid-flags', rows);
    const xs = f32('xs', rows);
    const ys = f32('ys', rows);
    const rowIndices = u32('row-indices', rows);
    const groupCounts = u32('group-counts', groupCount);
    const noiseCheck =
      props.noiseLabel !== undefined && props.noiseLabel < groupCount
        ? `valid = valid && label != ${props.noiseLabel}u;`
        : '';
    nodes.push(
      kernel(
        'keys',
        rows,
        [
          read('positions', positions, 'f32'),
          read('labels', labels, 'u32'),
          write('groupKeys', groupKeys, 'u32'),
          write('validFlags', validFlags, 'u32'),
          write('xs', xs),
          write('ys', ys),
          write('rowIndices', rowIndices, 'u32')
        ],
        `let x = positions[positionsOffset + index * 2u];
  let y = positions[positionsOffset + index * 2u + 1u];
  let label = labels[labelsOffset + index];
  var valid = (bitcast<u32>(x) & 0x7f800000u) != 0x7f800000u && (bitcast<u32>(y) & 0x7f800000u) != 0x7f800000u;
  valid = valid && label < GROUP_COUNT;
  ${noiseCheck}
  groupKeys[groupKeysOffset + index] = select(GROUP_COUNT, label, valid);
  validFlags[validFlagsOffset + index] = select(0u, 1u, valid);
  xs[xsOffset + index] = select(0.0, x, valid);
  ys[ysOffset + index] = select(0.0, y, valid);
  rowIndices[rowIndicesOffset + index] = index;`
      ),
      createFillNode<Parameters>(graph, {
        id: `${id}-zero-counts`,
        operation: OPERATION,
        view: groupCounts,
        type: 'u32',
        value: '0u'
      }),
      kernel(
        'count',
        rows,
        [read('groupKeys', groupKeys, 'u32'), write('groupCounts', groupCounts, 'atomic<u32>')],
        `let group = groupKeys[groupKeysOffset + index];
  if (group < GROUP_COUNT) {
    atomicAdd(&groupCounts[groupCountsOffset + group], 1u);
  }`
      )
    );

    // 2. Global extent, then the lattice: scale is a power of two, so d * scale is exact.
    const extentX = f32('extent-x', 2);
    const extentY = f32('extent-y', 2);
    const lattice = f32('lattice', 3);
    nodes.push(
      ...new GPUReduction({
        id: `${id}-extent-x`,
        input: xs,
        mask: validFlags,
        output: extentX,
        operation: 'extent'
      }).getCommandNodes(graph),
      ...new GPUReduction({
        id: `${id}-extent-y`,
        input: ys,
        mask: validFlags,
        output: extentY,
        operation: 'extent'
      }).getCommandNodes(graph),
      kernel(
        'lattice',
        1,
        [
          read('extentX', extentX, 'f32'),
          read('extentY', extentY, 'f32'),
          write('lattice', lattice)
        ],
        `let width = extentX[extentXOffset + 1u] - extentX[extentXOffset];
  let height = extentY[extentYOffset + 1u] - extentY[extentYOffset];
  let extent = max(width, height);
  var exponent = i32((bitcast<u32>(extent) >> 23u) & 255u) - 127;
  exponent = max(exponent, -99);
  let scale = bitcast<f32>(u32(127 + ${LATTICE_BITS - 1} - exponent) << 23u);
  lattice[latticeOffset] = extentX[extentXOffset];
  lattice[latticeOffset + 1u] = extentY[extentYOffset];
  lattice[latticeOffset + 2u] = select(scale, 1.0, !(extent > 0.0));`
      )
    );
    const latticeX = u32('lattice-x', rows);
    const latticeY = u32('lattice-y', rows);
    nodes.push(
      kernel(
        'quantize',
        rows,
        [
          read('xs', xs, 'f32'),
          read('ys', ys, 'f32'),
          read('validFlags', validFlags, 'u32'),
          read('lattice', lattice, 'f32'),
          write('latticeX', latticeX, 'u32'),
          write('latticeY', latticeY, 'u32')
        ],
        `let isValid = validFlags[validFlagsOffset + index] != 0u;
  let scale = lattice[latticeOffset + 2u];
  let dx = xs[xsOffset + index] - lattice[latticeOffset];
  let dy = ys[ysOffset + index] - lattice[latticeOffset + 1u];
  latticeX[latticeXOffset + index] = select(0u, u32(floor(dx * scale + 0.5)), isValid);
  latticeY[latticeYOffset + index] = select(0u, u32(floor(dy * scale + 0.5)), isValid);`
      )
    );

    // 3. Stable LSD sort: y, then x, then group.
    const sortedYKeys = u32('sorted-y-keys', rows);
    const permutationY = u32('permutation-y', rows);
    const keysX = u32('keys-x', rows);
    const sortedXKeys = u32('sorted-x-keys', rows);
    const permutationX = u32('permutation-x', rows);
    const keysGroup = u32('keys-group', rows);
    const sortedGroupKeys = u32('sorted-group-keys', rows);
    const permutation = u32('permutation', rows);
    nodes.push(
      ...new GPUSort({
        id: `${id}-sort-y`,
        keys: latticeY,
        values: rowIndices,
        outputKeys: sortedYKeys,
        outputValues: permutationY,
        keyBits: LATTICE_KEY_BITS
      }).getCommandNodes(graph),
      kernel(
        'gather-x',
        rows,
        [
          read('permutation', permutationY, 'u32'),
          read('latticeX', latticeX, 'u32'),
          write('keys', keysX, 'u32')
        ],
        'keys[keysOffset + index] = latticeX[latticeXOffset + permutation[permutationOffset + index]];'
      ),
      ...new GPUSort({
        id: `${id}-sort-x`,
        keys: keysX,
        values: permutationY,
        outputKeys: sortedXKeys,
        outputValues: permutationX,
        keyBits: LATTICE_KEY_BITS
      }).getCommandNodes(graph),
      kernel(
        'gather-group',
        rows,
        [
          read('permutation', permutationX, 'u32'),
          read('groupKeys', groupKeys, 'u32'),
          write('keys', keysGroup, 'u32')
        ],
        'keys[keysOffset + index] = groupKeys[groupKeysOffset + permutation[permutationOffset + index]];'
      ),
      ...new GPUSort({
        id: `${id}-sort-group`,
        keys: keysGroup,
        values: permutationX,
        outputKeys: sortedGroupKeys,
        outputValues: permutation,
        keyBits: getSortKeyBits(groupCount)
      }).getCommandNodes(graph)
    );
    const segmentOffsets = u32('segment-offsets', groupCount + 1);
    nodes.push(
      ...new GPUScan({
        id: `${id}-segment-scan`,
        input: groupCounts,
        output: segmentOffsets,
        mode: 'exclusive'
      }).getCommandNodes(graph),
      kernel(
        'segment-total',
        1,
        [read('counts', groupCounts, 'u32'), write('segmentOffsets', segmentOffsets, 'u32')],
        `segmentOffsets[segmentOffsetsOffset + ${groupCount}u] =
    segmentOffsets[segmentOffsetsOffset + ${groupCount - 1}u] + counts[countsOffset + ${groupCount - 1}u];`
      )
    );
    const sortedLatticeX = u32('sorted-lattice-x', rows);
    const sortedLatticeY = u32('sorted-lattice-y', rows);
    nodes.push(
      kernel(
        'gather-lattice',
        rows,
        [
          read('permutation', permutation, 'u32'),
          read('latticeX', latticeX, 'u32'),
          read('latticeY', latticeY, 'u32'),
          write('sortedLatticeX', sortedLatticeX, 'u32'),
          write('sortedLatticeY', sortedLatticeY, 'u32')
        ],
        `let row = permutation[permutationOffset + index];
  sortedLatticeX[sortedLatticeXOffset + index] = latticeX[latticeXOffset + row];
  sortedLatticeY[sortedLatticeYOffset + index] = latticeY[latticeYOffset + row];`
      )
    );

    // 3b. Parallel chunk prefilter: keep only per-chunk hull vertices, compacted in sorted order.
    let chainSegmentOffsets = segmentOffsets;
    let chainLatticeX = sortedLatticeX;
    let chainLatticeY = sortedLatticeY;
    let chainPermutation = permutation;
    if (rows >= PREFILTER_MINIMUM_ROWS) {
      const chunkCount = Math.ceil(rows / PREFILTER_CHUNK_ROWS);
      const survivorFlags = u32('survivor-flags', rows);
      const survivorOffsets = u32('survivor-offsets', rows);
      const prefilterScratch = u32('prefilter-scratch', 2 * rows + 2);
      const compactOffsets = u32('compact-offsets', groupCount + 1);
      nodes.push(
        kernel(
          'prefilter',
          chunkCount,
          [
            read('groupKeys', sortedGroupKeys, 'u32'),
            read('latticeX', sortedLatticeX, 'u32'),
            read('latticeY', sortedLatticeY, 'u32'),
            write('flags', survivorFlags, 'u32'),
            write('scratch', prefilterScratch, 'u32')
          ],
          `let chunkStart = index * ${PREFILTER_CHUNK_ROWS}u;
  let chunkEnd = min(chunkStart + ${PREFILTER_CHUNK_ROWS}u, ${rows}u);
  for (var row = chunkStart; row < chunkEnd; row++) {
    flags[flagsOffset + row] = 0u;
  }
  var runBegin = chunkStart;
  while (runBegin < chunkEnd) {
    let key = groupKeys[groupKeysOffset + runBegin];
    var runEnd = runBegin + 1u;
    while (runEnd < chunkEnd && groupKeys[groupKeysOffset + runEnd] == key) {
      runEnd++;
    }
    if (key < GROUP_COUNT) {
      let begin = runBegin;
      let base = 2u * begin;
      var lastRep = runEnd - 1u;
      while (!isRepresentative(lastRep, begin)) {
        lastRep = lastRep - 1u;
      }
      if (lastRep == begin) {
        flags[flagsOffset + begin] = 1u;
      } else {
        var count = 0u;
        for (var row = begin; row <= lastRep; row++) {
          if (!isRepresentative(row, begin)) {
            continue;
          }
          while (count >= 2u && orientation(scratch[scratchOffset + base + count - 2u], scratch[scratchOffset + base + count - 1u], row) <= 0) {
            count = count - 1u;
          }
          scratch[scratchOffset + base + count] = row;
          count = count + 1u;
        }
        let lowerCount = count + 1u;
        var back = lastRep;
        while (back > begin) {
          back = back - 1u;
          if (!isRepresentative(back, begin)) {
            continue;
          }
          while (count >= lowerCount && orientation(scratch[scratchOffset + base + count - 2u], scratch[scratchOffset + base + count - 1u], back) <= 0) {
            count = count - 1u;
          }
          scratch[scratchOffset + base + count] = back;
          count = count + 1u;
        }
        for (var vertex = 0u; vertex + 1u < count; vertex++) {
          flags[flagsOffset + scratch[scratchOffset + base + vertex]] = 1u;
        }
      }
    }
    runBegin = runEnd;
  }`,
          HULL_WGSL
        ),
        ...new GPUScan({
          id: `${id}-survivor-scan`,
          input: survivorFlags,
          output: survivorOffsets,
          mode: 'exclusive'
        }).getCommandNodes(graph),
        kernel(
          'compact-offsets',
          groupCount + 1,
          [
            read('segmentOffsets', segmentOffsets, 'u32'),
            read('flags', survivorFlags, 'u32'),
            read('survivorOffsets', survivorOffsets, 'u32'),
            write('compactOffsets', compactOffsets, 'u32')
          ],
          `let begin = segmentOffsets[segmentOffsetsOffset + index];
  let total = survivorOffsets[survivorOffsetsOffset + ${rows - 1}u] + flags[flagsOffset + ${rows - 1}u];
  compactOffsets[compactOffsetsOffset + index] = select(survivorOffsets[survivorOffsetsOffset + min(begin, ${rows - 1}u)], total, begin >= ${rows}u);`
        )
      );
      chainLatticeX = u32('compact-lattice-x', rows);
      chainLatticeY = u32('compact-lattice-y', rows);
      chainPermutation = u32('compact-permutation', rows);
      nodes.push(
        kernel(
          'compact',
          rows,
          [
            read('flags', survivorFlags, 'u32'),
            read('survivorOffsets', survivorOffsets, 'u32'),
            read('permutation', permutation, 'u32'),
            read('latticeX', sortedLatticeX, 'u32'),
            read('latticeY', sortedLatticeY, 'u32'),
            write('compactLatticeX', chainLatticeX, 'u32'),
            write('compactLatticeY', chainLatticeY, 'u32'),
            write('compactPermutation', chainPermutation, 'u32')
          ],
          `if (flags[flagsOffset + index] != 0u) {
    let slot = survivorOffsets[survivorOffsetsOffset + index];
    compactLatticeX[compactLatticeXOffset + slot] = latticeX[latticeXOffset + index];
    compactLatticeY[compactLatticeYOffset + slot] = latticeY[latticeYOffset + index];
    compactPermutation[compactPermutationOffset + slot] = permutation[permutationOffset + index];
  }`
        )
      );
      chainSegmentOffsets = compactOffsets;
    }

    // 4. One monotone chain per group over its sorted segment.
    const scratch = u32('chain-scratch', rows + groupCount);
    const trueSizes = output.sizes ?? u32('true-sizes', groupCount);
    nodes.push(
      kernel(
        'chain',
        groupCount,
        [
          read('segmentOffsets', chainSegmentOffsets, 'u32'),
          read('latticeX', chainLatticeX, 'u32'),
          read('latticeY', chainLatticeY, 'u32'),
          write('scratch', scratch, 'u32'),
          write('sizes', trueSizes, 'u32')
        ],
        `let begin = segmentOffsets[segmentOffsetsOffset + index];
  let end = segmentOffsets[segmentOffsetsOffset + index + 1u];
  let base = begin + index;
  if (end == begin) {
    sizes[sizesOffset + index] = 0u;
    return;
  }
  var lastRep = end - 1u;
  while (!isRepresentative(lastRep, begin)) {
    lastRep = lastRep - 1u;
  }
  if (lastRep == begin) {
    scratch[scratchOffset + base] = begin;
    sizes[sizesOffset + index] = 1u;
    return;
  }
  var count = 0u;
  for (var row = begin; row <= lastRep; row++) {
    if (!isRepresentative(row, begin)) {
      continue;
    }
    while (count >= 2u && orientation(scratch[scratchOffset + base + count - 2u], scratch[scratchOffset + base + count - 1u], row) <= 0) {
      count = count - 1u;
    }
    scratch[scratchOffset + base + count] = row;
    count = count + 1u;
  }
  let lowerCount = count + 1u;
  var row = lastRep;
  while (row > begin) {
    row = row - 1u;
    if (!isRepresentative(row, begin)) {
      continue;
    }
    while (count >= lowerCount && orientation(scratch[scratchOffset + base + count - 2u], scratch[scratchOffset + base + count - 1u], row) <= 0) {
      count = count - 1u;
    }
    scratch[scratchOffset + base + count] = row;
    count = count + 1u;
  }
  sizes[sizesOffset + index] = count - 1u;`,
        HULL_WGSL
      )
    );

    // 5. Bounded CSR emission.
    const eligibleSizes = u32('eligible-sizes', groupCount);
    nodes.push(
      createFillNode<Parameters>(graph, {
        id: `${id}-zero-overflow`,
        operation: OPERATION,
        view: output.overflow,
        type: 'u32',
        value: '0u',
        componentCount: 1
      }),
      kernel(
        'eligible',
        groupCount,
        [
          read('sizes', trueSizes, 'u32'),
          write('eligible', eligibleSizes, 'u32'),
          write('overflow', output.overflow, 'atomic<u32>')
        ],
        `let size = sizes[sizesOffset + index];
  let isEligible = size <= ${maximumVerticesPerGroup}u;
  eligible[eligibleOffset + index] = select(0u, size, isEligible);
  if (!isEligible) {
    atomicOr(&overflow[overflowOffset], ${GPU_GROUP_CONVEX_HULL_GROUP_OVERFLOW}u);
  }`
      ),
      ...new GPUScan({
        id: `${id}-offset-scan`,
        input: eligibleSizes,
        output: output.offsets,
        mode: 'exclusive'
      }).getCommandNodes(graph),
      kernel(
        'offset-total',
        1,
        [read('eligible', eligibleSizes, 'u32'), write('offsets', output.offsets, 'u32')],
        `offsets[offsetsOffset + ${groupCount}u] =
    offsets[offsetsOffset + ${groupCount - 1}u] + eligible[eligibleOffset + ${groupCount - 1}u];`
      ),
      createFillNode<Parameters>(graph, {
        id: `${id}-fill-indices`,
        operation: OPERATION,
        view: output.vertexIndices,
        type: 'u32',
        value: '0xffffffffu',
        componentCount: totalCapacity
      }),
      kernel(
        'emit',
        groupCount,
        [
          read('segmentOffsets', chainSegmentOffsets, 'u32'),
          read('offsets', output.offsets, 'u32'),
          read('eligible', eligibleSizes, 'u32'),
          read('scratch', scratch, 'u32'),
          read('permutation', chainPermutation, 'u32'),
          write('vertexIndices', output.vertexIndices, 'u32'),
          write('counts', output.counts, 'u32'),
          write('overflow', output.overflow, 'atomic<u32>')
        ],
        `let size = eligible[eligibleOffset + index];
  let offset = offsets[offsetsOffset + index];
  let base = segmentOffsets[segmentOffsetsOffset + index] + index;
  var emitted = 0u;
  if (size > 0u) {
    if (offset + size > ${totalCapacity}u) {
      atomicOr(&overflow[overflowOffset], ${GPU_GROUP_CONVEX_HULL_TOTAL_OVERFLOW}u);
    } else {
      emitted = size;
      for (var vertex = 0u; vertex < size; vertex++) {
        vertexIndices[vertexIndicesOffset + offset + vertex] = permutation[permutationOffset + scratch[scratchOffset + base + vertex]];
      }
    }
  }
  counts[countsOffset + index] = emitted;`
      )
    );
    if (output.vertexPositions) {
      nodes.push(
        kernel(
          'vertex-positions',
          totalCapacity,
          [
            read('vertexIndices', output.vertexIndices, 'u32'),
            read('positions', positions, 'f32'),
            write('vertexPositions', output.vertexPositions)
          ],
          `let row = vertexIndices[vertexIndicesOffset + index];
  let isEmpty = row == 0xffffffffu;
  let nan = getQuietNaN(index);
  vertexPositions[vertexPositionsOffset + index * 2u] = select(positions[positionsOffset + row * 2u], nan, isEmpty);
  vertexPositions[vertexPositionsOffset + index * 2u + 1u] = select(positions[positionsOffset + row * 2u + 1u], nan, isEmpty);`,
          'fn getQuietNaN(seed: u32) -> f32 { return bitcast<f32>(0x7fc00000u | (seed & 0u)); }'
        )
      );
    }
    return nodes;
  }
}

/** Exact lattice orientation and the representative test, shared by the chain kernel. @internal */
const HULL_WGSL = /* wgsl */ `
fn isRepresentative(row: u32, begin: u32) -> bool {
  return row == begin || latticeX[latticeXOffset + row] != latticeX[latticeXOffset + row - 1u] ||
    latticeY[latticeYOffset + row] != latticeY[latticeYOffset + row - 1u];
}

// Exact 32 x 32 -> 64 bit unsigned product as (high, low); operands are below 2^30.
fn multiplyUnsigned(a: u32, b: u32) -> vec2<u32> {
  let a0 = a & 0xffffu;
  let a1 = a >> 16u;
  let b0 = b & 0xffffu;
  let b1 = b >> 16u;
  let middle = a0 * b1 + a1 * b0;
  let low = a0 * b0 + (middle << 16u);
  let carry = select(0u, 1u, low < a0 * b0);
  return vec2<u32>(a1 * b1 + (middle >> 16u) + carry, low);
}

fn getSign(value: i32) -> i32 { return select(select(0, 1, value > 0), -1, value < 0); }

// Sign of the exact value left - right where each side is (sign, 64-bit magnitude).
fn compareProducts(leftSign: i32, left: vec2<u32>, rightSign: i32, right: vec2<u32>) -> i32 {
  if (leftSign != rightSign) {
    return select(-1, 1, leftSign > rightSign);
  }
  if (leftSign == 0) {
    return 0;
  }
  var magnitude = 0;
  if (left.x != right.x) {
    magnitude = select(-1, 1, left.x > right.x);
  } else if (left.y != right.y) {
    magnitude = select(-1, 1, left.y > right.y);
  }
  return magnitude * leftSign;
}

// Exact orientation of sorted rows a, b, c: 1 counter-clockwise, -1 clockwise, 0 collinear.
fn orientation(a: u32, b: u32, c: u32) -> i32 {
  let ax = i32(latticeX[latticeXOffset + a]);
  let ay = i32(latticeY[latticeYOffset + a]);
  let abx = i32(latticeX[latticeXOffset + b]) - ax;
  let aby = i32(latticeY[latticeYOffset + b]) - ay;
  let acx = i32(latticeX[latticeXOffset + c]) - ax;
  let acy = i32(latticeY[latticeYOffset + c]) - ay;
  let leftSign = getSign(abx) * getSign(acy);
  let rightSign = getSign(aby) * getSign(acx);
  let left = multiplyUnsigned(u32(abs(abx)), u32(abs(acy)));
  let right = multiplyUnsigned(u32(abs(aby)), u32(abs(acx)));
  return compareProducts(leftSign, left, rightSign, right);
}
`;
