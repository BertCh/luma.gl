// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {
  createFillNode,
  createWGSLKernelNode,
  getWGSLFloatLiteral,
  type WGSLKernelBinding
} from '../../utils/wgsl-kernel-nodes';
import {COLUMN_ORDERED_KEY_WGSL} from '../column-classification/column-classification-shared';
import {
  GPU_COLUMN_PROFILE_NULL_CATEGORY,
  GPU_COLUMN_PROFILE_PARAMETERS_PER_COLUMN,
  GPU_COLUMN_PROFILE_STATISTIC,
  GPU_COLUMN_PROFILE_STATISTIC_COUNT
} from './column-profile-parameters';

/** Workgroup size of the reduction kernels. */
export const COLUMN_PROFILE_WORKGROUP_SIZE = 256;
/** Rows each invocation folds sequentially before the workgroup tree. */
export const COLUMN_PROFILE_ROWS_PER_INVOCATION = 8;
/** Rows reduced by one workgroup tile. */
export const COLUMN_PROFILE_TILE_SIZE =
  COLUMN_PROFILE_WORKGROUP_SIZE * COLUMN_PROFILE_ROWS_PER_INVOCATION;
/** Floats per moment record: `[count, nullCount, finiteCount, mean, m2, sum]`. */
const RECORD_STRIDE = 6;
/** Keys per column: `[minimumKey, maximumKey, finiteMinimumKey, finiteMaximumKey]`. */
const KEY_STRIDE = 4;
/** Floats per resolved domain: `[lo, hi, width, mode]`; mode 0 invalid, 1 regular, 2 `lo == hi`. */
const DOMAIN_STRIDE = 4;

/** One normalized profile column. @internal */
export type ColumnProfileColumn = {
  kind: 'numeric' | 'category';
  values: GraphDataView<'float32'> | GraphDataView<'uint32'>;
  categoryCount: number;
  /** Offset of this column's counts in the shared category count array. */
  categoryBase: number;
};

/** Inputs of {@link getColumnProfileNodes}. @internal */
export type ColumnProfileNodeProps = {
  id: string;
  operation: string;
  columns: readonly ColumnProfileColumn[];
  rowCount: number;
  mask?: GraphDataView<'uint32'>;
  parameters?: GraphDataView<'float32'>;
  histogramBinCount: number;
  hyperLogLogPrecision: number;
  topCategoryCount: number;
  output: {
    statistics: GraphDataView<'float32'>;
    counts?: GraphDataView<'uint32'>;
    histograms?: GraphDataView<'uint32'>;
    hyperLogLogRegisters?: GraphDataView<'uint32'>;
    topCategories?: GraphDataView<'uint32'>;
    topCategoryCounts?: GraphDataView<'uint32'>;
  };
};

/**
 * Moment record merged by a fixed-order tree: `count` and `nullCount` are integers held in f32
 * (exact below 2^24), `(finiteCount, mean, m2)` follow the Chan, Golub and LeVeque pairwise
 * update, and `sum` is a plain fixed-order sum. Category columns carry their overflow count in
 * `finiteCount` with `mean = m2 = sum = 0`.
 */
const MOMENTS_WGSL = /* wgsl */ `
struct Moments {
  count: f32,
  nullCount: f32,
  finiteCount: f32,
  mean: f32,
  m2: f32,
  sum: f32,
}

fn emptyMoments() -> Moments {
  return Moments(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
}

fn mergeMoments(left: Moments, right: Moments) -> Moments {
  var result = emptyMoments();
  result.count = left.count + right.count;
  result.nullCount = left.nullCount + right.nullCount;
  result.sum = left.sum + right.sum;
  if (right.finiteCount == 0.0) {
    result.finiteCount = left.finiteCount;
    result.mean = left.mean;
    result.m2 = left.m2;
  } else if (left.finiteCount == 0.0) {
    result.finiteCount = right.finiteCount;
    result.mean = right.mean;
    result.m2 = right.m2;
  } else {
    let count = left.finiteCount + right.finiteCount;
    let delta = right.mean - left.mean;
    result.finiteCount = count;
    result.mean = left.mean + delta * (right.finiteCount / count);
    result.m2 = left.m2 + right.m2 + delta * delta * (left.finiteCount * right.finiteCount / count);
  }
  return result;
}

var<workgroup> sharedMoments: array<Moments, ${COLUMN_PROFILE_WORKGROUP_SIZE}>;

fn reduceSharedMoments(local: u32) {
  for (var stride = ${COLUMN_PROFILE_WORKGROUP_SIZE / 2}u; stride > 0u; stride = stride >> 1u) {
    workgroupBarrier();
    if (local < stride) {
      sharedMoments[local] = mergeMoments(sharedMoments[local], sharedMoments[local + stride]);
    }
  }
}
`;

/**
 * Equal-width bin of `value - minimum`: `floor(difference / width)` corrected one step down and up
 * against correctly rounded products, so a CPU oracle using `Math.fround` agrees exactly.
 */
const BIN_WGSL = /* wgsl */ `
fn getBinIndex(value: f32, minimum: f32, width: f32) -> u32 {
  let difference = value - minimum;
  var bin = floor(difference / width);
  bin = clamp(bin, 0.0, f32(BIN_COUNT - 1u));
  if (difference < bin * width) {
    bin = bin - 1.0;
  } else if (difference >= (bin + 1.0) * width) {
    bin = bin + 1.0;
  }
  return u32(clamp(bin, 0.0, f32(BIN_COUNT - 1u)));
}
`;

/** Murmur3 `fmix32` avalanche hash, applied to the value bits XOR a per-column seed. */
const HASH_WGSL = /* wgsl */ `
fn hashBits(bits: u32) -> u32 {
  var h = bits;
  h = h ^ (h >> 16u);
  h = h * 0x85ebca6bu;
  h = h ^ (h >> 13u);
  h = h * 0xc2b2ae35u;
  h = h ^ (h >> 16u);
  return h;
}
`;

/** Largest slot count counted in workgroup memory (4 KiB of `atomic<u32>`). */
const PRIVATIZED_SLOT_LIMIT = 1024;

/**
 * Builds the `declarations` and `body` of a one-row-one-slot counting kernel. `slotBody` is the
 * body of `fn getSlot(index: u32) -> u32` that returns the slot or `NO_SLOT`. Small tables are
 * counted in a workgroup-private atomic table and flushed with one global atomic per non-empty
 * slot per workgroup, so global atomic traffic drops from one per row to at most one per slot
 * per workgroup (skewed columns no longer serialise on a few hot counters). Large tables keep one
 * global atomic per row.
 */
function getPrivatizedCountingKernel(options: {
  slotCount: number;
  targetName: string;
  /** WGSL expression of the first slot's index in the target. */
  targetBase: string;
  declarations: string;
  slotBody: string;
}): {declarations: string; body: string; guardIndex?: boolean; workgroupSize: number} {
  const {slotCount, targetName, targetBase, slotBody} = options;
  const slotFunction = `const NO_SLOT: u32 = 0xffffffffu;
fn getSlot(index: u32) -> u32 {
  ${slotBody}
}`;
  const target = `${targetName}[${targetName}Offset + ${targetBase} + `;
  if (slotCount > PRIVATIZED_SLOT_LIMIT || slotCount < 1) {
    return {
      workgroupSize: COLUMN_PROFILE_WORKGROUP_SIZE,
      declarations: `${options.declarations}\n${slotFunction}`,
      body: `let slot = getSlot(index);
  if (slot != NO_SLOT) {
    atomicAdd(&${target}slot], 1u);
  }`
    };
  }
  return {
    declarations: `${options.declarations}
${slotFunction}
var<workgroup> localCounts: array<atomic<u32>, ${slotCount}>;`,
    // No early return: every invocation of a workgroup must reach the barriers.
    body: `for (var slot = localInvocationIndex; slot < ${slotCount}u; slot += ${COLUMN_PROFILE_WORKGROUP_SIZE}u) {
    atomicStore(&localCounts[slot], 0u);
  }
  workgroupBarrier();
  if (index < INVOCATION_COUNT) {
    let slot = getSlot(index);
    if (slot != NO_SLOT) {
      atomicAdd(&localCounts[slot], 1u);
    }
  }
  workgroupBarrier();
  for (var slot = localInvocationIndex; slot < ${slotCount}u; slot += ${COLUMN_PROFILE_WORKGROUP_SIZE}u) {
    let count = atomicLoad(&localCounts[slot]);
    if (count != 0u) {
      atomicAdd(&${target}slot], count);
    }
  }`,
    guardIndex: false,
    workgroupSize: COLUMN_PROFILE_WORKGROUP_SIZE
  };
}

/** Per-column HyperLogLog hash seed, `(column + 1) * 0x9e3779b9` modulo 2^32. @internal */
export function getColumnProfileHashSeed(column: number): number {
  return Math.imul(column + 1, 0x9e3779b9) >>> 0;
}

/** HyperLogLog bias constant alpha_m. @internal */
export function getHyperLogLogAlpha(registerCount: number): number {
  if (registerCount === 16) {
    return 0.673;
  }
  if (registerCount === 32) {
    return 0.697;
  }
  if (registerCount === 64) {
    return 0.709;
  }
  return 0.7213 / (1 + 1.079 / registerCount);
}

/** Number of u32 count slots a profile needs for its category columns. @internal */
export function getColumnProfileCategoryTotal(columns: readonly ColumnProfileColumn[]): number {
  return columns.reduce(
    (total, column) => total + (column.kind === 'category' ? column.categoryCount : 0),
    0
  );
}

const read = (name: string, view: GraphDataView, type: 'u32' | 'f32'): WGSLKernelBinding => ({
  name,
  view,
  type,
  access: 'read'
});
const write = (
  name: string,
  view: GraphDataView,
  type: 'u32' | 'f32' | 'atomic<u32>' = 'u32'
): WGSLKernelBinding => ({name, view, type, access: 'read_write'});

/**
 * Returns every command node of a column profile, in execution order.
 *
 * @internal
 */
export function getColumnProfileNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: ColumnProfileNodeProps
): GPUCommandNode<Parameters>[] {
  const {id, operation, columns, rowCount, mask, parameters, output} = props;
  const {histogramBinCount, hyperLogLogPrecision, topCategoryCount} = props;
  const columnCount = columns.length;
  const registerCount = 2 ** hyperLogLogPrecision;
  const tileCount = Math.ceil(rowCount / COLUMN_PROFILE_TILE_SIZE);
  const categoryTotal = getColumnProfileCategoryTotal(columns);
  const hasHistograms = Boolean(output.histograms);
  const hasTopCategories = Boolean(output.topCategories && output.topCategoryCounts);
  const hasCategoryColumn = columns.some(column => column.kind === 'category');
  const needsCategoryCounts = hasCategoryColumn && (hasHistograms || hasTopCategories);
  const maskBinding = mask ? [read('rowMask', mask, 'u32')] : [];
  const valuesType = (column: ColumnProfileColumn) => (column.kind === 'numeric' ? 'f32' : 'u32');

  const u32 = (name: string, length: number) =>
    createTransientView(graph, `${id}-${name}`, 'uint32', length);
  const f32 = (name: string, length: number) =>
    createTransientView(graph, `${id}-${name}`, 'float32', length);
  const partials = f32('partials', columnCount * tileCount * RECORD_STRIDE);
  const records = f32('records', columnCount * RECORD_STRIDE);
  const keys = u32('keys', columnCount * KEY_STRIDE);
  const domains = f32('domains', columnCount * DOMAIN_STRIDE);
  const registers = output.hyperLogLogRegisters ?? u32('registers', columnCount * registerCount);
  const categoryCounts = needsCategoryCounts ? u32('category-counts', categoryTotal) : undefined;
  const nodes: GPUCommandNode<Parameters>[] = [];

  // Init: every accumulator is rewritten on every encoding.
  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-init`,
      operation,
      variant: 'init',
      bindings: [write('keys', keys)],
      invocationCount: columnCount * KEY_STRIDE,
      body: `keys[keysOffset + index] = select(0xffffffffu, 0u, (index & 1u) == 1u);`
    })
  );
  const fill = (name: string, view: GraphDataView, length: number, value: string) =>
    nodes.push(
      createFillNode<Parameters>(graph, {
        id: `${id}-${name}`,
        operation,
        view,
        type: 'u32',
        value,
        componentCount: length
      })
    );
  fill('clear-registers', registers, columnCount * registerCount, '0u');
  if (output.histograms) {
    fill('clear-histograms', output.histograms, columnCount * histogramBinCount, '0u');
  }
  if (categoryCounts) {
    fill('clear-category-counts', categoryCounts, categoryTotal, '0u');
  }
  if (hasTopCategories) {
    fill(
      'clear-top-categories',
      output.topCategories!,
      columnCount * topCategoryCount,
      '0xffffffffu'
    );
    fill(
      'clear-top-category-counts',
      output.topCategoryCounts!,
      columnCount * topCategoryCount,
      '0u'
    );
  }

  // Moments tiles, one node per column.
  for (const [columnIndex, column] of columns.entries()) {
    nodes.push(
      getMomentTileNode(graph, {
        id: `${id}-column-${columnIndex}-moments-tile`,
        operation,
        column,
        columnIndex,
        rowCount,
        tileCount,
        maskBinding,
        partials,
        keys
      })
    );
  }
  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-moments-merge`,
      operation,
      variant: 'moments-merge',
      bindings: [read('partials', partials, 'f32'), write('records', records, 'f32')],
      invocationCount: columnCount * COLUMN_PROFILE_WORKGROUP_SIZE,
      workgroupSize: COLUMN_PROFILE_WORKGROUP_SIZE,
      guardIndex: false,
      declarations: `const TILE_COUNT: u32 = ${tileCount}u;
const STEP_COUNT: u32 = ${Math.ceil(tileCount / COLUMN_PROFILE_WORKGROUP_SIZE)}u;
${MOMENTS_WGSL}`,
      body: `let local = localInvocationIndex;
  let column = workgroupIndex;
  var accumulated = emptyMoments();
  for (var step = 0u; step < STEP_COUNT; step = step + 1u) {
    let tile = step * ${COLUMN_PROFILE_WORKGROUP_SIZE}u + local;
    if (tile < TILE_COUNT) {
      let base = partialsOffset + (column * TILE_COUNT + tile) * ${RECORD_STRIDE}u;
      accumulated = mergeMoments(
        accumulated,
        Moments(partials[base], partials[base + 1u], partials[base + 2u], partials[base + 3u], partials[base + 4u], partials[base + 5u])
      );
    }
  }
  sharedMoments[local] = accumulated;
  reduceSharedMoments(local);
  if (local == 0u) {
    let record = sharedMoments[0];
    let base = recordsOffset + column * ${RECORD_STRIDE}u;
    records[base] = record.count;
    records[base + 1u] = record.nullCount;
    records[base + 2u] = record.finiteCount;
    records[base + 3u] = record.mean;
    records[base + 4u] = record.m2;
    records[base + 5u] = record.sum;
  }`
    })
  );

  if (hasHistograms) {
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-domain`,
        operation,
        variant: 'domain',
        bindings: [
          read('keys', keys, 'u32'),
          ...(parameters ? [read('params', parameters, 'f32')] : []),
          write('domains', domains, 'f32')
        ],
        invocationCount: columnCount,
        declarations: `const INVERSE_BIN_COUNT: f32 = ${getWGSLFloatLiteral(1 / histogramBinCount)};
${COLUMN_ORDERED_KEY_WGSL}
fn isFiniteBits(x: f32) -> bool {
  return (bitcast<u32>(x) & 0x7f800000u) != 0x7f800000u;
}`,
        body: `let column = index;
  let keyBase = keysOffset + column * ${KEY_STRIDE}u;
  var lo = getNaN();
  var hi = getNaN();
  ${
    parameters
      ? `lo = params[paramsOffset + column * ${GPU_COLUMN_PROFILE_PARAMETERS_PER_COLUMN}u];
  hi = params[paramsOffset + column * ${GPU_COLUMN_PROFILE_PARAMETERS_PER_COLUMN}u + 1u];`
      : ''
  }
  let finiteMinimumKey = keys[keyBase + 2u];
  let finiteMaximumKey = keys[keyBase + 3u];
  if (isNanBits(lo)) {
    lo = select(getNaN(), decodeOrderedKey(finiteMinimumKey), finiteMinimumKey != 0xffffffffu);
  }
  if (isNanBits(hi)) {
    hi = select(getNaN(), decodeOrderedKey(finiteMaximumKey), finiteMaximumKey != 0u);
  }
  var mode = 0.0;
  var width = 0.0;
  if (isFiniteBits(lo) && isFiniteBits(hi) && lo <= hi) {
    if (lo == hi) {
      mode = 2.0;
    } else {
      let range = hi - lo;
      width = range * INVERSE_BIN_COUNT;
      if (isFiniteBits(range) && isFiniteBits(width) && width > 0.0) {
        mode = 1.0;
      }
    }
  }
  let base = domainsOffset + column * ${DOMAIN_STRIDE}u;
  domains[base] = lo;
  domains[base + 1u] = hi;
  domains[base + 2u] = width;
  domains[base + 3u] = mode;`
      })
    );
  }

  for (const [columnIndex, column] of columns.entries()) {
    const columnId = `${id}-column-${columnIndex}`;
    const isNumeric = column.kind === 'numeric';
    const type = valuesType(column);
    if (hasHistograms && isNumeric) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${columnId}-histogram`,
          operation,
          variant: 'histogram',
          bindings: [
            read('values', column.values, 'f32'),
            ...maskBinding,
            read('domains', domains, 'f32'),
            write('histogram', output.histograms!, 'atomic<u32>')
          ],
          invocationCount: rowCount,
          ...getPrivatizedCountingKernel({
            slotCount: histogramBinCount,
            targetName: 'histogram',
            targetBase: `COLUMN * BIN_COUNT`,
            declarations: `const BIN_COUNT: u32 = ${histogramBinCount}u;
const COLUMN: u32 = ${columnIndex}u;
${BIN_WGSL}`,
            slotBody: `${mask ? 'if (rowMask[rowMaskOffset + index] == 0u) {\n    return NO_SLOT;\n  }' : ''}
  let value = values[valuesOffset + index];
  if ((bitcast<u32>(value) & 0x7f800000u) == 0x7f800000u) {
    return NO_SLOT;
  }
  let base = domainsOffset + COLUMN * ${DOMAIN_STRIDE}u;
  let mode = domains[base + 3u];
  let lo = domains[base];
  if (mode == 0.0 || value < lo || value > domains[base + 1u]) {
    return NO_SLOT;
  }
  var bin = 0u;
  if (mode == 1.0) {
    bin = getBinIndex(value, lo, domains[base + 2u]);
  }
  return bin;`
          })
        })
      );
    }
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${columnId}-hyperloglog`,
        operation,
        variant: 'hyperloglog',
        bindings: [
          read('values', column.values, type),
          ...maskBinding,
          write('registers', registers, 'atomic<u32>')
        ],
        invocationCount: rowCount,
        declarations: `const PRECISION: u32 = ${hyperLogLogPrecision}u;
const REGISTER_COUNT: u32 = ${registerCount}u;
const COLUMN: u32 = ${columnIndex}u;
const SEED: u32 = ${getColumnProfileHashSeed(columnIndex)}u;
${HASH_WGSL}`,
        body: `${mask ? 'if (rowMask[rowMaskOffset + index] == 0u) {\n    return;\n  }' : ''}
  ${
    isNumeric
      ? `var bits = bitcast<u32>(values[valuesOffset + index]);
  if ((bits & 0x7fffffffu) > 0x7f800000u) {
    return;
  }
  if ((bits & 0x7fffffffu) == 0u) {
    bits = 0u;
  }`
      : `let bits = values[valuesOffset + index];
  if (bits == ${GPU_COLUMN_PROFILE_NULL_CATEGORY}u) {
    return;
  }`
  }
  let hash = hashBits(bits ^ SEED);
  let registerIndex = hash >> (32u - PRECISION);
  let rank = min(countLeadingZeros(hash << PRECISION), 32u - PRECISION) + 1u;
  atomicMax(&registers[registersOffset + COLUMN * REGISTER_COUNT + registerIndex], rank);`
      })
    );
    if (categoryCounts && !isNumeric) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${columnId}-category-counts`,
          operation,
          variant: 'category-counts',
          bindings: [
            read('values', column.values, 'u32'),
            ...maskBinding,
            write('categoryCounts', categoryCounts, 'atomic<u32>')
          ],
          invocationCount: rowCount,
          ...getPrivatizedCountingKernel({
            slotCount: column.categoryCount,
            targetName: 'categoryCounts',
            targetBase: 'BASE',
            declarations: `const CATEGORY_COUNT: u32 = ${column.categoryCount}u;
const BASE: u32 = ${column.categoryBase}u;`,
            slotBody: `${mask ? 'if (rowMask[rowMaskOffset + index] == 0u) {\n    return NO_SLOT;\n  }' : ''}
  let code = values[valuesOffset + index];
  if (code < CATEGORY_COUNT) {
    return code;
  }
  return NO_SLOT;`
          })
        })
      );
    }
  }

  if (hasTopCategories && categoryCounts) {
    for (const [columnIndex, column] of columns.entries()) {
      if (column.kind === 'category') {
        nodes.push(
          getTopCategoriesNode(graph, {
            id: `${id}-column-${columnIndex}-top-categories`,
            operation,
            column,
            columnIndex,
            topCategoryCount,
            categoryCounts,
            topCategories: output.topCategories!,
            topCategoryCounts: output.topCategoryCounts!
          })
        );
      }
    }
  }

  for (const [columnIndex, column] of columns.entries()) {
    nodes.push(
      getFinishNode(graph, {
        id: `${id}-column-${columnIndex}-finish`,
        operation,
        column,
        columnIndex,
        registerCount,
        hyperLogLogPrecision,
        histogramBinCount,
        records,
        keys,
        registers,
        categoryCounts,
        output
      })
    );
  }
  return nodes;
}

function getMomentTileNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    column: ColumnProfileColumn;
    columnIndex: number;
    rowCount: number;
    tileCount: number;
    maskBinding: WGSLKernelBinding[];
    partials: GraphDataView<'float32'>;
    keys: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  const {column, columnIndex, rowCount, tileCount, maskBinding} = props;
  const isNumeric = column.kind === 'numeric';
  const hasMask = maskBinding.length > 0;
  const rowBody = isNumeric
    ? `let bits = bitcast<u32>(value);
      if ((bits & 0x7fffffffu) > 0x7f800000u) {
        accumulated.nullCount = accumulated.nullCount + 1.0;
      } else {
        accumulated.count = accumulated.count + 1.0;
        let key = getOrderedKey(value);
        minimumKey = min(minimumKey, key);
        maximumKey = max(maximumKey, key);
        if ((bits & 0x7fffffffu) != 0x7f800000u) {
          finiteMinimumKey = min(finiteMinimumKey, key);
          finiteMaximumKey = max(finiteMaximumKey, key);
          let count = accumulated.finiteCount + 1.0;
          let delta = value - accumulated.mean;
          accumulated.finiteCount = count;
          accumulated.mean = accumulated.mean + delta / count;
          accumulated.m2 = accumulated.m2 + delta * (value - accumulated.mean);
          accumulated.sum = accumulated.sum + value;
        }
      }`
    : `if (value == ${GPU_COLUMN_PROFILE_NULL_CATEGORY}u) {
        accumulated.nullCount = accumulated.nullCount + 1.0;
      } else {
        accumulated.count = accumulated.count + 1.0;
        if (value >= CATEGORY_COUNT) {
          accumulated.finiteCount = accumulated.finiteCount + 1.0;
        }
        minimumKey = min(minimumKey, value);
        maximumKey = max(maximumKey, value);
      }`;
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'moments-tile',
    bindings: [
      read('values', column.values, isNumeric ? 'f32' : 'u32'),
      ...maskBinding,
      write('partials', props.partials, 'f32'),
      write('keys', props.keys, 'atomic<u32>')
    ],
    invocationCount: tileCount * COLUMN_PROFILE_WORKGROUP_SIZE,
    workgroupSize: COLUMN_PROFILE_WORKGROUP_SIZE,
    guardIndex: false,
    declarations: `const ROW_COUNT: u32 = ${rowCount}u;
const TILE_COUNT: u32 = ${tileCount}u;
const TILE_SIZE: u32 = ${COLUMN_PROFILE_TILE_SIZE}u;
const ROWS_PER_INVOCATION: u32 = ${COLUMN_PROFILE_ROWS_PER_INVOCATION}u;
const COLUMN: u32 = ${columnIndex}u;
const CATEGORY_COUNT: u32 = ${column.categoryCount}u;
${COLUMN_ORDERED_KEY_WGSL}
${MOMENTS_WGSL}
var<workgroup> sharedKeys: array<atomic<u32>, ${KEY_STRIDE}>;`,
    body: `let local = localInvocationIndex;
  let tile = workgroupIndex;
  if (local < ${KEY_STRIDE}u) {
    atomicStore(&sharedKeys[local], select(0xffffffffu, 0u, (local & 1u) == 1u));
  }
  workgroupBarrier();
  var accumulated = emptyMoments();
  var minimumKey = 0xffffffffu;
  var maximumKey = 0u;
  var finiteMinimumKey = 0xffffffffu;
  var finiteMaximumKey = 0u;
  for (var step = 0u; step < ROWS_PER_INVOCATION; step = step + 1u) {
    // Uniform trip count: rows past the end are skipped by a predicate, never by a break.
    let row = tile * TILE_SIZE + step * ${COLUMN_PROFILE_WORKGROUP_SIZE}u + local;
    if (row < ROW_COUNT${hasMask ? ' && rowMask[rowMaskOffset + row] != 0u' : ''}) {
      let value = values[valuesOffset + row];
      ${rowBody}
    }
  }
  sharedMoments[local] = accumulated;
  atomicMin(&sharedKeys[0], minimumKey);
  atomicMax(&sharedKeys[1], maximumKey);
  atomicMin(&sharedKeys[2], finiteMinimumKey);
  atomicMax(&sharedKeys[3], finiteMaximumKey);
  reduceSharedMoments(local);
  workgroupBarrier();
  if (local == 0u) {
    let record = sharedMoments[0];
    let base = partialsOffset + (COLUMN * TILE_COUNT + tile) * ${RECORD_STRIDE}u;
    partials[base] = record.count;
    partials[base + 1u] = record.nullCount;
    partials[base + 2u] = record.finiteCount;
    partials[base + 3u] = record.mean;
    partials[base + 4u] = record.m2;
    partials[base + 5u] = record.sum;
    let keyBase = keysOffset + COLUMN * ${KEY_STRIDE}u;
    atomicMin(&keys[keyBase], atomicLoad(&sharedKeys[0]));
    atomicMax(&keys[keyBase + 1u], atomicLoad(&sharedKeys[1]));
    atomicMin(&keys[keyBase + 2u], atomicLoad(&sharedKeys[2]));
    atomicMax(&keys[keyBase + 3u], atomicLoad(&sharedKeys[3]));
  }`
  });
}

function getTopCategoriesNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    column: ColumnProfileColumn;
    columnIndex: number;
    topCategoryCount: number;
    categoryCounts: GraphDataView<'uint32'>;
    topCategories: GraphDataView<'uint32'>;
    topCategoryCounts: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  const {column, columnIndex, topCategoryCount} = props;
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'top-categories',
    bindings: [
      read('categoryCounts', props.categoryCounts, 'u32'),
      write('topCategories', props.topCategories),
      write('topCategoryCounts', props.topCategoryCounts)
    ],
    invocationCount: COLUMN_PROFILE_WORKGROUP_SIZE,
    workgroupSize: COLUMN_PROFILE_WORKGROUP_SIZE,
    guardIndex: false,
    declarations: `const CATEGORY_COUNT: u32 = ${column.categoryCount}u;
const BASE: u32 = ${column.categoryBase}u;
const COLUMN: u32 = ${columnIndex}u;
const TOP_COUNT: u32 = ${topCategoryCount}u;
const STEP_COUNT: u32 = ${Math.ceil(column.categoryCount / COLUMN_PROFILE_WORKGROUP_SIZE)}u;
var<workgroup> bestCounts: array<u32, ${COLUMN_PROFILE_WORKGROUP_SIZE}>;
var<workgroup> bestCodes: array<u32, ${COLUMN_PROFILE_WORKGROUP_SIZE}>;`,
    body: `let local = localInvocationIndex;
  var previousCount = 0u;
  var previousCode = 0u;
  var hasPrevious = false;
  for (var rank = 0u; rank < TOP_COUNT; rank = rank + 1u) {
    var bestCount = 0u;
    var bestCode = 0xffffffffu;
    for (var step = 0u; step < STEP_COUNT; step = step + 1u) {
      let code = step * ${COLUMN_PROFILE_WORKGROUP_SIZE}u + local;
      if (code < CATEGORY_COUNT) {
        let count = categoryCounts[categoryCountsOffset + BASE + code];
        // Strictly after the previous pick in (count descending, code ascending) order.
        let isCandidate = count > 0u && (!hasPrevious || count < previousCount || (count == previousCount && code > previousCode));
        if (isCandidate && count > bestCount) {
          bestCount = count;
          bestCode = code;
        }
      }
    }
    bestCounts[local] = bestCount;
    bestCodes[local] = bestCode;
    for (var stride = ${COLUMN_PROFILE_WORKGROUP_SIZE / 2}u; stride > 0u; stride = stride >> 1u) {
      workgroupBarrier();
      if (local < stride) {
        let otherCount = bestCounts[local + stride];
        let otherCode = bestCodes[local + stride];
        if (otherCount > bestCounts[local] || (otherCount == bestCounts[local] && otherCode < bestCodes[local])) {
          bestCounts[local] = otherCount;
          bestCodes[local] = otherCode;
        }
      }
    }
    workgroupBarrier();
    let pickedCount = bestCounts[0];
    let pickedCode = bestCodes[0];
    if (local == 0u) {
      topCategories[topCategoriesOffset + COLUMN * TOP_COUNT + rank] = select(0xffffffffu, pickedCode, pickedCount > 0u);
      topCategoryCounts[topCategoryCountsOffset + COLUMN * TOP_COUNT + rank] = pickedCount;
    }
    previousCount = pickedCount;
    previousCode = pickedCode;
    hasPrevious = true;
    workgroupBarrier();
  }`
  });
}

function getFinishNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    column: ColumnProfileColumn;
    columnIndex: number;
    registerCount: number;
    hyperLogLogPrecision: number;
    histogramBinCount: number;
    records: GraphDataView<'float32'>;
    keys: GraphDataView<'uint32'>;
    registers: GraphDataView<'uint32'>;
    categoryCounts?: GraphDataView<'uint32'>;
    output: ColumnProfileNodeProps['output'];
  }
): GPUCommandNode<Parameters> {
  const {column, columnIndex, registerCount, histogramBinCount, output} = props;
  const isNumeric = column.kind === 'numeric';
  const alphaTimesSquare = getHyperLogLogAlpha(registerCount) * registerCount * registerCount;
  const copyHistogram = !isNumeric && output.histograms && props.categoryCounts;
  const stat = GPU_COLUMN_PROFILE_STATISTIC;
  const bindings: WGSLKernelBinding[] = [
    read('records', props.records, 'f32'),
    read('keys', props.keys, 'u32'),
    read('registers', props.registers, 'u32'),
    write('statistics', output.statistics, 'f32')
  ];
  if (output.counts) {
    bindings.push(write('countsOut', output.counts));
  }
  if (copyHistogram) {
    bindings.push(
      read('categoryCounts', props.categoryCounts!, 'u32'),
      write('histogram', output.histograms!)
    );
  }
  const statisticsBody = isNumeric
    ? `let hasValues = count > 0.0;
    let hasFinite = finiteCount > 0.0;
    let variance = select(nan, m2 / finiteCount, hasFinite);
    statistics[statisticsBase + ${stat.minimum}u] = select(nan, decodeOrderedKey(minimumKey), hasValues);
    statistics[statisticsBase + ${stat.maximum}u] = select(nan, decodeOrderedKey(maximumKey), hasValues);
    statistics[statisticsBase + ${stat.sum}u] = select(nan, sum, hasFinite);
    statistics[statisticsBase + ${stat.mean}u] = select(nan, mean, hasFinite);
    statistics[statisticsBase + ${stat.variance}u] = variance;
    statistics[statisticsBase + ${stat.sampleVariance}u] = select(nan, m2 / (finiteCount - 1.0), finiteCount >= 2.0);
    statistics[statisticsBase + ${stat.standardDeviation}u] = select(nan, sqrt(variance), hasFinite);
    statistics[statisticsBase + ${stat.overflowCount}u] = 0.0;`
    : `let hasValues = count > 0.0;
    statistics[statisticsBase + ${stat.minimum}u] = select(nan, f32(minimumKey), hasValues);
    statistics[statisticsBase + ${stat.maximum}u] = select(nan, f32(maximumKey), hasValues);
    statistics[statisticsBase + ${stat.sum}u] = nan;
    statistics[statisticsBase + ${stat.mean}u] = nan;
    statistics[statisticsBase + ${stat.variance}u] = nan;
    statistics[statisticsBase + ${stat.sampleVariance}u] = nan;
    statistics[statisticsBase + ${stat.standardDeviation}u] = nan;
    statistics[statisticsBase + ${stat.overflowCount}u] = finiteCount;`;
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'finish',
    bindings,
    invocationCount: COLUMN_PROFILE_WORKGROUP_SIZE,
    workgroupSize: COLUMN_PROFILE_WORKGROUP_SIZE,
    guardIndex: false,
    declarations: `const COLUMN: u32 = ${columnIndex}u;
const REGISTER_COUNT: u32 = ${registerCount}u;
const REGISTER_STEP_COUNT: u32 = ${Math.ceil(registerCount / COLUMN_PROFILE_WORKGROUP_SIZE)}u;
const ALPHA_TIMES_SQUARE: f32 = ${getWGSLFloatLiteral(alphaTimesSquare)};
const BIN_COUNT: u32 = ${histogramBinCount}u;
const CATEGORY_COUNT: u32 = ${column.categoryCount}u;
const CATEGORY_BASE: u32 = ${column.categoryBase}u;
${COLUMN_ORDERED_KEY_WGSL}
var<workgroup> sharedInverseSums: array<f32, ${COLUMN_PROFILE_WORKGROUP_SIZE}>;
var<workgroup> sharedZeroCounts: array<u32, ${COLUMN_PROFILE_WORKGROUP_SIZE}>;`,
    body: `let local = localInvocationIndex;
  var inverseSum = 0.0;
  var zeroCount = 0u;
  for (var step = 0u; step < REGISTER_STEP_COUNT; step = step + 1u) {
    let registerIndex = step * ${COLUMN_PROFILE_WORKGROUP_SIZE}u + local;
    if (registerIndex < REGISTER_COUNT) {
      let rank = registers[registersOffset + COLUMN * REGISTER_COUNT + registerIndex];
      // 2^-rank is exact: rank is at most ${33 - props.hyperLogLogPrecision}.
      inverseSum = inverseSum + bitcast<f32>((127u - rank) << 23u);
      zeroCount = zeroCount + select(0u, 1u, rank == 0u);
    }
  }
  sharedInverseSums[local] = inverseSum;
  sharedZeroCounts[local] = zeroCount;
  for (var stride = ${COLUMN_PROFILE_WORKGROUP_SIZE / 2}u; stride > 0u; stride = stride >> 1u) {
    workgroupBarrier();
    if (local < stride) {
      sharedInverseSums[local] = sharedInverseSums[local] + sharedInverseSums[local + stride];
      sharedZeroCounts[local] = sharedZeroCounts[local] + sharedZeroCounts[local + stride];
    }
  }
  ${
    copyHistogram
      ? `for (var bin = local; bin < BIN_COUNT; bin = bin + ${COLUMN_PROFILE_WORKGROUP_SIZE}u) {
    histogram[histogramOffset + COLUMN * BIN_COUNT + bin] = select(0u, categoryCounts[categoryCountsOffset + CATEGORY_BASE + bin], bin < CATEGORY_COUNT);
  }`
      : ''
  }
  if (local == 0u) {
    let recordBase = recordsOffset + COLUMN * ${RECORD_STRIDE}u;
    let count = records[recordBase];
    let nullCount = records[recordBase + 1u];
    let finiteCount = records[recordBase + 2u];
    let mean = records[recordBase + 3u];
    let m2 = records[recordBase + 4u];
    let sum = records[recordBase + 5u];
    let keyBase = keysOffset + COLUMN * ${KEY_STRIDE}u;
    let minimumKey = keys[keyBase];
    let maximumKey = keys[keyBase + 1u];
    let nan = getNaN();
    let statisticsBase = statisticsOffset + COLUMN * ${GPU_COLUMN_PROFILE_STATISTIC_COUNT}u;
    statistics[statisticsBase + ${stat.count}u] = count;
    statistics[statisticsBase + ${stat.nullCount}u] = nullCount;
    ${statisticsBody}
    let registerCountFloat = f32(REGISTER_COUNT);
    var estimate = ALPHA_TIMES_SQUARE / sharedInverseSums[0];
    let zeros = sharedZeroCounts[0];
    if (zeros == REGISTER_COUNT) {
      estimate = 0.0;
    } else if (estimate <= 2.5 * registerCountFloat && zeros > 0u) {
      // Linear counting m * ln(m / zeros). For few occupied registers the ratio is close to 1 and
      // log() cancels, so use the series of ln(1 + u) with the exact integer u = (m - zeros) / zeros.
      let occupied = f32(REGISTER_COUNT - zeros);
      let ratio = occupied / f32(zeros);
      if (ratio < 0.0625) {
        let r2 = ratio * ratio;
        estimate = registerCountFloat * (ratio - r2 * (0.5 - ratio * (0.33333334 - ratio * (0.25 - ratio * 0.2))));
      } else {
        estimate = registerCountFloat * log(registerCountFloat / f32(zeros));
      }
    }
    statistics[statisticsBase + ${stat.distinctEstimate}u] = estimate;
    ${
      output.counts
        ? `countsOut[countsOutOffset + COLUMN * 2u] = u32(count);
    countsOut[countsOutOffset + COLUMN * 2u + 1u] = u32(nullCount);`
        : ''
    }
  }`
  });
}
