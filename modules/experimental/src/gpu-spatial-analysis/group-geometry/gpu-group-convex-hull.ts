// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import {
  createTransientView,
  GPUReduction,
  GPUScan,
  GPUSort,
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
import {getSortedSegmentOffsetNodes} from '../../utils/sorted-segment-offsets';
import {
  validateCapacity,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validatePackedView
} from '../../utils/gpu-contributor-utils';
import {GET_QUIET_NAN_SEEDED_WGSL} from '../../utils/wgsl-common';

const OPERATION = 'GPUGroupConvexHull';
const MAXIMUM_ROWS = 2 ** 24 - 1;
/** Quantized coordinates lie in `[0, 2^29]`, so every difference fits an `i32` and products fit 60 bits. */
const LATTICE_BITS = 29;
const LATTICE_KEY_BITS = 30;
/** Inputs with at least this many rows run the parallel chunk prefilter before the chain. */
const PREFILTER_MINIMUM_ROWS = 8192;
/** Sorted rows handled by one prefilter invocation. */
const PREFILTER_CHUNK_ROWS = 256;
/** Extreme directions per group: min and max of x, y, x + y and x - y. */
const EXTREME_DIRECTION_COUNT = 8;
/** Largest `groupCount` with the extreme-point filter on by default (tables are 16 words per group). */
const EXTREME_FILTER_DEFAULT_MAXIMUM_GROUPS = 65536;
/** Groups up to this count accumulate extremes in workgroup memory first (8 words per group). */
const EXTREME_PRIVATE_GROUP_LIMIT = 256;
/** Rows folded by one invocation of a workgroup-private extreme kernel. */
const EXTREME_ROWS_PER_THREAD = 16;
const EXTREME_WORKGROUP_SIZE = 256;
/** Bias that keeps `x - y` of lattice points non-negative. */
const LATTICE_BIAS = 2 ** LATTICE_BITS;

/** Largest `prefilterLevels` of {@link GPUGroupConvexHull}. */
export const GPU_GROUP_CONVEX_HULL_MAXIMUM_PREFILTER_LEVELS = 4;
/** Default prefilter level count for inputs of at least 8192 rows. */
const DEFAULT_PREFILTER_LEVELS = 2;

/** Bit of `overflow` set when a hull has more vertices than `maximumVerticesPerGroup`. */
export const GPU_GROUP_CONVEX_HULL_GROUP_OVERFLOW = 1;
/** Bit of `overflow` set when the emitted hulls do not fit `vertexCapacity`. */
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
  /** Completeness scalars. */
  status: {
    /**
     * One row, a bit set: {@link GPU_GROUP_CONVEX_HULL_GROUP_OVERFLOW} and
     * {@link GPU_GROUP_CONVEX_HULL_TOTAL_OVERFLOW}. `0` when every hull was emitted.
     */
    overflow: GraphDataView<'uint32'>;
  };
};

/**
 * Properties for {@link GPUGroupConvexHull}.
 *
 * Per-frame (no recompile): the contents of every input buffer. Topology (needs a new graph):
 * view lengths, `groupCount`, `noiseLabel`, `maximumVerticesPerGroup` and `vertexCapacity`.
 */
export type GPUGroupConvexHullProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'group-convex-hull'`. */
  id?: string;
  /**
   * Planar only. Spatial context of `positions`; omitted means native planar coordinates.
   * Longitude/latitude (or any other non-planar) contexts throw at construction: project
   * positions to a planar frame first, for example with `createGPULocalFrame` and
   * `GPULocalFrameProjection`, and pass the frame's context so its `frame` identity is kept.
   */
  spatialContext?: GPUSpatialContext;
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
  vertexCapacity: number;
  /**
   * Compile-time number of parallel prefilter levels before the per-group chain, 0 to
   * {@link GPU_GROUP_CONVEX_HULL_MAXIMUM_PREFILTER_LEVELS}. Every level keeps only the hull
   * vertices of each run of 256 consecutive candidates of a group, so the serial chain walks fewer
   * rows. Defaults to 0 below 8192 rows and 2 above. The hull is identical for every value.
   */
  prefilterLevels?: number;
  /**
   * Compile-time Akl-Toussaint filter before the sorts. Per group it finds the eight extreme
   * lattice points (min and max of x, y, x + y and x - y) and marks every row strictly inside the
   * convex octagon they span as discarded, with exact integer orientation tests. Discarded rows
   * take the excluded-row group key, so they sort last and the prefilter and chain never walk
   * them. Boundary and collinear points are always kept, so the hull is identical for `true` and
   * `false`. Defaults to `true` up to 65536 groups (the extreme tables hold 16 words per group),
   * `false` above, where it throws if forced on.
   */
  extremePointFilter?: boolean;
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
 *   `vertexCapacity` runs out: the first group that does not fit and every later group are dropped
 *   (never truncated into a wrong polygon), and `sizes` still reports their true sizes.
 *
 * ## Cost
 *
 * Before the sorts, `extremePointFilter` (default on) runs the Akl-Toussaint heuristic: per group
 * it finds the eight extreme lattice points, and every row strictly inside their octagon, tested
 * exactly, is given the excluded-row key. Interior rows cannot be hull vertices, and boundary or
 * collinear rows are kept, so the hull is bit for bit the same. `GPUSort` has a fixed length and
 * its nodes cannot be skipped from the GPU, so the three sorts still cover every row; the saving
 * is in everything after them. Discarded rows form the last sorted run, which the chunk prefilter
 * skips whole, so on a blob the prefilter and the chain walk only the few rows outside the octagon.
 * The chunk prefilter levels stay useful after the filter, for groups whose points are mostly on or
 * near the hull (annuli, circles, uniform disks), where the octagon removes little.
 *
 * Sorting and orientation are parallel. Inputs of at least 8192 rows first run chunk prefilter
 * levels (`prefilterLevels`, default 2): every 256 consecutive sorted rows of a group get their own
 * monotone chain in parallel, and only chunk hull vertices (the hull of a union is the hull of the
 * chunk hulls) are compacted, then the next level repeats on the survivors. The per-group chain
 * (one thread per group) then walks only the final survivors, so one group of millions of points
 * no longer runs a serial walk over every row. The result is identical for every level count.
 *
 * The contributor never compiles, encodes, submits or reads back.
 *
 * Planar only: `spatialContext` defaults to native planar coordinates and a longitude/latitude
 * context throws at construction. Project first, for example with `createGPULocalFrame` and
 * `GPULocalFrameProjection`, and pass the frame's context so its `frame` identity is kept.
 */
export class GPUGroupConvexHull implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUGroupConvexHullProps;
  /** Resolved spatial context (planar). */
  readonly spatialContext: GPUSpatialContext;
  /** Number of groups. */
  readonly groupCount: number;
  /** Resolved number of parallel prefilter levels. */
  readonly prefilterLevels: number;
  /** Resolved {@link GPUGroupConvexHullProps.extremePointFilter}. */
  readonly extremePointFilter: boolean;

  constructor(props: GPUGroupConvexHullProps) {
    this.id = props.id ?? 'group-convex-hull';
    this.props = props;
    this.spatialContext = resolveGPUSpatialContext(this.id, props.spatialContext);
    const {id} = this;
    const {output} = props;
    validatePackedView(props.positions, ['float32x2'], `${id} positions`);
    const rows = props.positions.length;
    if (rows < 1) {
      throw new Error(`${id} needs at least one row`);
    }
    if (rows > MAXIMUM_ROWS) {
      throw new Error(`${id} supports fewer than 2^24 rows`);
    }
    validatePackedView(props.labels, ['uint32'], `${id} labels`);
    if (props.labels.length !== rows) {
      throw new Error(`${id} labels length must equal positions length`);
    }
    this.groupCount = props.groupCount;
    validateCapacity(id, 'groupCount', this.groupCount, 1, MAXIMUM_ROWS);
    if (props.noiseLabel !== undefined) {
      validateCapacity(id, 'noiseLabel', props.noiseLabel, 0);
    }
    validateCapacity(id, 'maximumVerticesPerGroup', props.maximumVerticesPerGroup, 3);
    validateCapacity(id, 'vertexCapacity', props.vertexCapacity);
    this.prefilterLevels =
      props.prefilterLevels ?? (rows >= PREFILTER_MINIMUM_ROWS ? DEFAULT_PREFILTER_LEVELS : 0);
    validateCapacity(
      id,
      'prefilterLevels',
      this.prefilterLevels,
      0,
      GPU_GROUP_CONVEX_HULL_MAXIMUM_PREFILTER_LEVELS
    );
    this.extremePointFilter =
      props.extremePointFilter ?? this.groupCount <= EXTREME_FILTER_DEFAULT_MAXIMUM_GROUPS;
    if (this.extremePointFilter && this.groupCount > EXTREME_FILTER_DEFAULT_MAXIMUM_GROUPS) {
      throw new Error(
        `${id} extremePointFilter supports at most ${EXTREME_FILTER_DEFAULT_MAXIMUM_GROUPS} groups`
      );
    }
    const groups = this.groupCount;
    const checks = [
      ['vertexIndices', output.vertexIndices, ['uint32'], props.vertexCapacity],
      ['vertexPositions', output.vertexPositions, ['float32x2'], props.vertexCapacity],
      ['offsets', output.offsets, ['uint32'], groups + 1],
      ['counts', output.counts, ['uint32'], groups],
      ['sizes', output.sizes, ['uint32'], groups],
      ['overflow', output.status.overflow, ['uint32'], 1]
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

  /**
   * Returns key, lattice, extreme-point filter, three sort passes, offsets, prefilter, chain, bound
   * and emit nodes in order.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id, groupCount} = this;
    const {output, positions, labels, maximumVerticesPerGroup, vertexCapacity} = props;
    const {status, ...columns} = output;
    validateGraphViewsBelongToGraph(id, graph, [
      positions,
      labels,
      ...Object.values(columns),
      status.overflow
    ]);
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

    // 2b. Akl-Toussaint filter: rows strictly inside their group's extreme octagon get the
    // excluded-row key, so they sort last and no later stage walks them.
    let sortGroupKeys = groupKeys;
    if (this.extremePointFilter) {
      sortGroupKeys = u32('filtered-group-keys', rows);
      nodes.push(
        ...this.getExtremePointFilterNodes(graph, {
          rows,
          groupKeys,
          latticeX,
          latticeY,
          filteredGroupKeys: sortGroupKeys
        })
      );
    }

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
          read('groupKeys', sortGroupKeys, 'u32'),
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
    // Group offsets are lower bounds in the sorted keys: no counting atomics, zero fill or scan.
    nodes.push(
      ...getSortedSegmentOffsetNodes(graph, {
        id,
        operation: OPERATION,
        groupCount,
        sortedGroupKeys,
        segmentOffsets
      })
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

    // 3b. Parallel chunk prefilter levels: keep only per-chunk hull vertices, compacted in sorted
    // order. Every level shrinks each group's candidates, so the serial chain walks few rows.
    let chainSegmentOffsets = segmentOffsets;
    let chainLatticeX = sortedLatticeX;
    let chainLatticeY = sortedLatticeY;
    let chainPermutation = permutation;
    for (let level = 0; level < this.prefilterLevels; level++) {
      const levelId = `prefilter${level}`;
      const chunkCount = Math.ceil(rows / PREFILTER_CHUNK_ROWS);
      const survivorFlags = u32(`${levelId}-survivor-flags`, rows);
      const survivorOffsets = u32(`${levelId}-survivor-offsets`, rows);
      const prefilterScratch = u32(`${levelId}-scratch`, 2 * rows + 2);
      const compactOffsets = u32(`${levelId}-compact-offsets`, groupCount + 1);
      // Level 0 reads the sorted group keys; later levels find each run from the group offsets.
      const runBindings =
        level === 0
          ? [read('groupKeys', sortedGroupKeys, 'u32')]
          : [read('groupOffsets', chainSegmentOffsets, 'u32')];
      const runLimit = level === 0 ? `${rows}u` : `groupOffsets[groupOffsetsOffset + GROUP_COUNT]`;
      const runStart =
        level === 0
          ? `let key = groupKeys[groupKeysOffset + runBegin];
    // The excluded-row run is last in sorted order: skip it whole instead of walking it.
    var runEnd = chunkEnd;
    if (key < GROUP_COUNT) {
      runEnd = runBegin + 1u;
      while (runEnd < chunkEnd && groupKeys[groupKeysOffset + runEnd] == key) {
        runEnd++;
      }
    }`
          : `let key = findGroup(runBegin);
    let runEnd = min(chunkEnd, groupOffsets[groupOffsetsOffset + key + 1u]);`;
      nodes.push(
        kernel(
          levelId,
          chunkCount,
          [
            ...runBindings,
            read('latticeX', chainLatticeX, 'u32'),
            read('latticeY', chainLatticeY, 'u32'),
            write('flags', survivorFlags, 'u32'),
            write('scratch', prefilterScratch, 'u32')
          ],
          `let chunkStart = index * ${PREFILTER_CHUNK_ROWS}u;
  let slotEnd = min(chunkStart + ${PREFILTER_CHUNK_ROWS}u, ${rows}u);
  for (var row = chunkStart; row < slotEnd; row++) {
    flags[flagsOffset + row] = 0u;
  }
  let chunkEnd = min(slotEnd, ${runLimit});
  var runBegin = chunkStart;
  while (runBegin < chunkEnd) {
    ${runStart}
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
          level === 0
            ? HULL_WGSL
            : `${HULL_WGSL}
// Largest group whose offset is at or before the row (skips empty groups).
fn findGroup(row: u32) -> u32 {
  var low = 0u;
  var high = GROUP_COUNT;
  while (low < high) {
    let middle = (low + high + 1u) >> 1u;
    if (groupOffsets[groupOffsetsOffset + middle] <= row) {
      low = middle;
    } else {
      high = middle - 1u;
    }
  }
  return low;
}`
        ),
        ...new GPUScan({
          id: `${id}-${levelId}-scan`,
          input: survivorFlags,
          output: survivorOffsets,
          mode: 'exclusive'
        }).getCommandNodes(graph),
        kernel(
          `${levelId}-offsets`,
          groupCount + 1,
          [
            read('segmentOffsets', chainSegmentOffsets, 'u32'),
            read('flags', survivorFlags, 'u32'),
            read('survivorOffsets', survivorOffsets, 'u32'),
            write('compactOffsets', compactOffsets, 'u32')
          ],
          `let begin = segmentOffsets[segmentOffsetsOffset + index];
  let total = survivorOffsets[survivorOffsetsOffset + ${rows - 1}u] + flags[flagsOffset + ${rows - 1}u];
  compactOffsets[compactOffsetsOffset + index] = select(survivorOffsets[survivorOffsetsOffset + min(begin, ${rows - 1}u)], total, begin >= ${rows}u);`
        )
      );
      const compactLatticeX = u32(`${levelId}-lattice-x`, rows);
      const compactLatticeY = u32(`${levelId}-lattice-y`, rows);
      const compactPermutation = u32(`${levelId}-permutation`, rows);
      nodes.push(
        kernel(
          `${levelId}-compact`,
          rows,
          [
            read('flags', survivorFlags, 'u32'),
            read('survivorOffsets', survivorOffsets, 'u32'),
            read('permutation', chainPermutation, 'u32'),
            read('latticeX', chainLatticeX, 'u32'),
            read('latticeY', chainLatticeY, 'u32'),
            write('compactLatticeX', compactLatticeX, 'u32'),
            write('compactLatticeY', compactLatticeY, 'u32'),
            write('compactPermutation', compactPermutation, 'u32')
          ],
          `if (flags[flagsOffset + index] != 0u) {
    let slot = survivorOffsets[survivorOffsetsOffset + index];
    compactLatticeX[compactLatticeXOffset + slot] = latticeX[latticeXOffset + index];
    compactLatticeY[compactLatticeYOffset + slot] = latticeY[latticeYOffset + index];
    compactPermutation[compactPermutationOffset + slot] = permutation[permutationOffset + index];
  }`
        )
      );
      chainLatticeX = compactLatticeX;
      chainLatticeY = compactLatticeY;
      chainPermutation = compactPermutation;
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
        view: output.status.overflow,
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
          write('overflow', output.status.overflow, 'atomic<u32>')
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
        componentCount: vertexCapacity
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
          write('overflow', output.status.overflow, 'atomic<u32>')
        ],
        `let size = eligible[eligibleOffset + index];
  let offset = offsets[offsetsOffset + index];
  let base = segmentOffsets[segmentOffsetsOffset + index] + index;
  var emitted = 0u;
  if (size > 0u) {
    if (offset + size > ${vertexCapacity}u) {
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
          vertexCapacity,
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
          GET_QUIET_NAN_SEEDED_WGSL
        )
      );
    }
    return nodes;
  }

  /**
   * Akl-Toussaint filter nodes. Two accumulation passes find the eight extreme lattice points of
   * every group (values first, then a deterministic tie-break on the other coordinate), one
   * kernel orders them into a counter-clockwise octagon, and the last kernel writes the group key
   * of every row, or the excluded-row key when the row is strictly inside its octagon.
   *
   * Up to `EXTREME_PRIVATE_GROUP_LIMIT` groups, each invocation folds 16 rows into
   * workgroup-private tables that are flushed once per workgroup (two stages, library rule 14).
   * Above it the per-row atomics spread over at least 2048 table words.
   */
  private getExtremePointFilterNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    views: {
      rows: number;
      groupKeys: GraphDataView<'uint32'>;
      latticeX: GraphDataView<'uint32'>;
      latticeY: GraphDataView<'uint32'>;
      filteredGroupKeys: GraphDataView<'uint32'>;
    }
  ): GPUCommandNode<Parameters>[] {
    const {id, groupCount} = this;
    const {rows, groupKeys, latticeX, latticeY, filteredGroupKeys} = views;
    const tableLength = groupCount * EXTREME_DIRECTION_COUNT;
    const values = createTransientView(graph, `${id}-extreme-values`, 'uint32', tableLength);
    const ties = createTransientView(graph, `${id}-extreme-ties`, 'uint32', tableLength);
    const octagon = createTransientView(graph, `${id}-octagon`, 'uint32', 2 * tableLength);
    const binding = (
      name: string,
      view: GraphDataView,
      type: 'u32' | 'atomic<u32>' = 'u32',
      access: 'read' | 'read_write' = 'read'
    ): WGSLKernelBinding => ({name, view, type, access});
    const isPrivate = groupCount <= EXTREME_PRIVATE_GROUP_LIMIT;
    const declarations = `const GROUP_COUNT: u32 = ${groupCount}u;
const LATTICE_BIAS: u32 = ${LATTICE_BIAS}u;`;
    // Every table word is a minimum. Maximum directions store the bitwise complement, so a fill of
    // 0xffffffff is the identity for all eight and an empty group decodes harmlessly.
    const createAccumulateNode = (
      step: 'extreme-values' | 'extreme-ties',
      table: GraphDataView<'uint32'>,
      extraBindings: WGSLKernelBinding[],
      getStatements: (update: (slot: string, value: string) => string) => string
    ): GPUCommandNode<Parameters> => {
      const update = (slot: string, value: string) =>
        isPrivate
          ? `atomicMin(&localTable[${slot}], ${value});`
          : `atomicMin(&table[tableOffset + ${slot}], ${value});`;
      const rowSource = `let group = groupKeys[groupKeysOffset + row];
    if (group < GROUP_COUNT) {
      let x = latticeX[latticeXOffset + row];
      let y = latticeY[latticeYOffset + row];
      let sum = x + y;
      let difference = x + LATTICE_BIAS - y;
      let base = group * ${EXTREME_DIRECTION_COUNT}u;
      ${getStatements(update)}
    }`;
      const bindings = [
        binding('groupKeys', groupKeys),
        binding('latticeX', latticeX),
        binding('latticeY', latticeY),
        ...extraBindings,
        binding('table', table, 'atomic<u32>', 'read_write')
      ];
      if (!isPrivate) {
        return createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-${step}`,
          operation: OPERATION,
          variant: step,
          bindings,
          invocationCount: rows,
          declarations,
          body: `let row = index;
  ${rowSource}`
        });
      }
      const threadCount = Math.ceil(rows / EXTREME_ROWS_PER_THREAD);
      const stride = Math.ceil(threadCount / EXTREME_WORKGROUP_SIZE) * EXTREME_WORKGROUP_SIZE;
      return createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-${step}`,
        operation: OPERATION,
        variant: step,
        bindings,
        invocationCount: threadCount,
        workgroupSize: EXTREME_WORKGROUP_SIZE,
        guardIndex: false,
        declarations: `${declarations}
const TABLE_LENGTH: u32 = ${tableLength}u;
var<workgroup> localTable: array<atomic<u32>, ${tableLength}>;`,
        // No early return: every invocation of a workgroup must reach the barriers.
        body: `for (var slot = localInvocationIndex; slot < TABLE_LENGTH; slot += ${EXTREME_WORKGROUP_SIZE}u) {
    atomicStore(&localTable[slot], 0xffffffffu);
  }
  workgroupBarrier();
  for (var rowStep = 0u; rowStep < ${EXTREME_ROWS_PER_THREAD}u; rowStep++) {
    let row = index + rowStep * ${stride}u;
    if (row < ${rows}u) {
    ${rowSource}
    }
  }
  workgroupBarrier();
  for (var slot = localInvocationIndex; slot < TABLE_LENGTH; slot += ${EXTREME_WORKGROUP_SIZE}u) {
    let value = atomicLoad(&localTable[slot]);
    if (value != 0xffffffffu) {
      atomicMin(&table[tableOffset + slot], value);
    }
  }`
      });
    };
    const createFill = (name: string, view: GraphDataView<'uint32'>) =>
      createFillNode<Parameters>(graph, {
        id: `${id}-fill-${name}`,
        operation: OPERATION,
        view,
        type: 'u32',
        value: '0xffffffffu',
        componentCount: tableLength
      });
    const extremeReads = `let minX = values[valuesOffset + base];
      let maxX = ~values[valuesOffset + base + 1u];
      let minY = values[valuesOffset + base + 2u];
      let maxY = ~values[valuesOffset + base + 3u];
      let minSum = values[valuesOffset + base + 4u];
      let maxSum = ~values[valuesOffset + base + 5u];
      let minDifference = values[valuesOffset + base + 6u];
      let maxDifference = ~values[valuesOffset + base + 7u];`;
    return [
      createFill('extreme-values', values),
      createAccumulateNode(
        'extreme-values',
        values,
        [],
        update => `${update('base', 'x')}
      ${update('base + 1u', '~x')}
      ${update('base + 2u', 'y')}
      ${update('base + 3u', '~y')}
      ${update('base + 4u', 'sum')}
      ${update('base + 5u', '~sum')}
      ${update('base + 6u', 'difference')}
      ${update('base + 7u', '~difference')}`
      ),
      createFill('extreme-ties', ties),
      // Ties on an extreme value resolve to the smallest other coordinate (y for the x extremes,
      // x for the others), so each extreme is one lattice point that no scheduling order changes.
      createAccumulateNode(
        'extreme-ties',
        ties,
        [binding('values', values)],
        update => `${extremeReads}
      if (x == minX) { ${update('base', 'y')} }
      if (x == maxX) { ${update('base + 1u', 'y')} }
      if (y == minY) { ${update('base + 2u', 'x')} }
      if (y == maxY) { ${update('base + 3u', 'x')} }
      if (sum == minSum) { ${update('base + 4u', 'x')} }
      if (sum == maxSum) { ${update('base + 5u', 'x')} }
      if (difference == minDifference) { ${update('base + 6u', 'x')} }
      if (difference == maxDifference) { ${update('base + 7u', 'x')} }`
      ),
      // The octagon in counter-clockwise order of support direction: S, SE, E, NE, N, NW, W, SW.
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-extreme-points`,
        operation: OPERATION,
        variant: 'extreme-points',
        bindings: [
          binding('values', values),
          binding('ties', ties),
          binding('octagon', octagon, 'u32', 'read_write')
        ],
        invocationCount: groupCount,
        declarations: `const LATTICE_BIAS: u32 = ${LATTICE_BIAS}u;`,
        body: `let base = index * ${EXTREME_DIRECTION_COUNT}u;
  let outputBase = index * 16u;
  ${extremeReads.replace(/^ {6}/gm, '  ')}
  // Empty groups decode to arbitrary words; no row ever reads them, and u32 arithmetic wraps.
  var points = array<vec2<u32>, 8>(
    vec2<u32>(ties[tiesOffset + base + 2u], minY),
    vec2<u32>(ties[tiesOffset + base + 7u], ties[tiesOffset + base + 7u] + LATTICE_BIAS - maxDifference),
    vec2<u32>(maxX, ties[tiesOffset + base + 1u]),
    vec2<u32>(ties[tiesOffset + base + 5u], maxSum - ties[tiesOffset + base + 5u]),
    vec2<u32>(ties[tiesOffset + base + 3u], maxY),
    vec2<u32>(ties[tiesOffset + base + 6u], ties[tiesOffset + base + 6u] + LATTICE_BIAS - minDifference),
    vec2<u32>(minX, ties[tiesOffset + base]),
    vec2<u32>(ties[tiesOffset + base + 4u], minSum - ties[tiesOffset + base + 4u])
  );
  for (var vertex = 0u; vertex < 8u; vertex++) {
    octagon[octagonOffset + outputBase + 2u * vertex] = points[vertex].x;
    octagon[octagonOffset + outputBase + 2u * vertex + 1u] = points[vertex].y;
  }`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-filter-rows`,
        operation: OPERATION,
        variant: 'filter-rows',
        bindings: [
          binding('groupKeys', groupKeys),
          binding('latticeX', latticeX),
          binding('latticeY', latticeY),
          binding('octagon', octagon),
          binding('filteredKeys', filteredGroupKeys, 'u32', 'read_write')
        ],
        invocationCount: rows,
        declarations: `${declarations}
${ORIENTATION_WGSL}`,
        // A row is dropped only when it is strictly left of every non-degenerate octagon edge, so
        // points on the boundary, on a collinear edge or equal to an extreme always stay.
        body: `let group = groupKeys[groupKeysOffset + index];
  var key = group;
  if (group < GROUP_COUNT) {
    let px = i32(latticeX[latticeXOffset + index]);
    let py = i32(latticeY[latticeYOffset + index]);
    var edgeCount = 0u;
    var isInterior = true;
    for (var edge = 0u; edge < 8u; edge++) {
      let startIndex = octagonOffset + group * 16u + edge * 2u;
      let endIndex = octagonOffset + group * 16u + ((edge + 1u) & 7u) * 2u;
      let ax = i32(octagon[startIndex]);
      let ay = i32(octagon[startIndex + 1u]);
      let bx = i32(octagon[endIndex]);
      let by = i32(octagon[endIndex + 1u]);
      if (ax == bx && ay == by) {
        continue;
      }
      edgeCount++;
      if (orientationOfPoints(ax, ay, bx, by, px, py) <= 0) {
        isInterior = false;
        break;
      }
    }
    if (isInterior && edgeCount > 0u) {
      key = GROUP_COUNT;
    }
  }
  filteredKeys[filteredKeysOffset + index] = key;`
      })
    ];
  }
}

/** Exact lattice orientation of three points; shared by the filter, prefilter and chain. @internal */
const ORIENTATION_WGSL = /* wgsl */ `
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

// Exact orientation of points a, b, c: 1 counter-clockwise, -1 clockwise, 0 collinear.
fn orientationOfPoints(ax: i32, ay: i32, bx: i32, by: i32, cx: i32, cy: i32) -> i32 {
  let abx = bx - ax;
  let aby = by - ay;
  let acx = cx - ax;
  let acy = cy - ay;
  let leftSign = getSign(abx) * getSign(acy);
  let rightSign = getSign(aby) * getSign(acx);
  let left = multiplyUnsigned(u32(abs(abx)), u32(abs(acy)));
  let right = multiplyUnsigned(u32(abs(aby)), u32(abs(acx)));
  return compareProducts(leftSign, left, rightSign, right);
}
`;

/** Representative test and row orientation, shared by the prefilter and chain kernels. @internal */
const HULL_WGSL = /* wgsl */ `${ORIENTATION_WGSL}
fn isRepresentative(row: u32, begin: u32) -> bool {
  return row == begin || latticeX[latticeXOffset + row] != latticeX[latticeXOffset + row - 1u] ||
    latticeY[latticeYOffset + row] != latticeY[latticeYOffset + row - 1u];
}

// Exact orientation of sorted rows a, b, c: 1 counter-clockwise, -1 clockwise, 0 collinear.
fn orientation(a: u32, b: u32, c: u32) -> i32 {
  return orientationOfPoints(
    i32(latticeX[latticeXOffset + a]), i32(latticeY[latticeYOffset + a]),
    i32(latticeX[latticeXOffset + b]), i32(latticeY[latticeYOffset + b]),
    i32(latticeX[latticeXOffset + c]), i32(latticeY[latticeYOffset + c])
  );
}
`;
