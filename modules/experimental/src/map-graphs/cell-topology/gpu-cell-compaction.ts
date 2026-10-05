// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {dggs} from '@luma.gl/shadertools';
import {
  createTransientView,
  GPUScan,
  GPUSort,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUMapGraphRecipe} from '../map-graph-types';
import {createMapGraphFillNode, createMapGraphKernelNode} from '../map-graph-kernels';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../map-graph-utils';
import {
  CELL_KEY_WGSL,
  getCellKeyLayout,
  type CellKeyLayout,
  type GPUCellFamily
} from '../cell-aggregation/cell-keys';
import {getCellTableNodes, hasCellKeyHighWord} from '../cell-aggregation/cell-table';

const OPERATION = 'GPUCellCompaction';

/** Default deepest ancestor-to-descendant expansion of one `uncompact` input cell. */
export const GPU_CELL_UNCOMPACT_DEFAULT_MAXIMUM_DEPTH = 8;

/** Word order of the input cell keys. */
export type GPUCellCompactionWordOrder = 'little-endian' | 'high-low';

/**
 * Merges complete sets of sibling cells into their parent, recursively (H3 `compactCells`, CARTO
 * Quadbin compact).
 */
export type GPUCellCompactOperation = {
  type: 'compact';
  /** Resolution of every input cell; rows of another resolution are dropped. */
  resolution: number;
  /** Coarsest resolution a merged cell may reach. Defaults to 0. */
  minimumResolution?: number;
  /**
   * True when the valid input rows are already unique-or-duplicate cells in ascending key order
   * with every dropped (invalid, masked, other-resolution) row absent or trailing. Skips the sort.
   */
  sorted?: boolean;
};

/** Expands every input cell to all of its descendants at `resolution` (H3 `uncompactCells`). */
export type GPUCellUncompactOperation = {
  type: 'uncompact';
  /** Target resolution of the output; input cells must be at or above it. */
  resolution: number;
  /**
   * Largest resolution gap `resolution - cellResolution` expanded per input cell; deeper cells are
   * dropped and counted. Defaults to 8 (4^8 Quadbin or 7^8 H3 descendants), at most 15 for
   * Quadbin and 11 for H3 so one cell's descendant count fits a `u32`.
   */
  maximumDepth?: number;
};

/** Caller-owned, capacity-bounded output of {@link GPUCellCompaction}. */
export type GPUCellCompactionOutput = {
  /** Output cell keys as little-endian `(low, high)` words; rows past `count` are zero. */
  cells: GraphDataView<'uint32x2'>;
  /** One-row scalar receiving `min(totalCount, cells.length)`. */
  count: GraphDataView<'uint32'>;
  /** One-row scalar receiving 1 when more cells exist than `cells.length`, otherwise 0. */
  overflow: GraphDataView<'uint32'>;
  /** Optional one-row scalar receiving the unclamped number of output cells. */
  totalCount?: GraphDataView<'uint32'>;
  /**
   * Optional one-row scalar receiving the number of input rows (within `count`) that were
   * dropped: invalid keys, masked rows, other resolutions (`compact`), cells below the target
   * resolution or deeper than `maximumDepth` (`uncompact`).
   */
  droppedCount?: GraphDataView<'uint32'>;
};

/** Properties for {@link GPUCellCompaction}. */
export type GPUCellCompactionProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'cell-compaction'`. */
  id?: string;
  /** Grid family of the input and output cells. */
  family: GPUCellFamily;
  /** Compact or uncompact. */
  operation: GPUCellCompactOperation | GPUCellUncompactOperation;
  /** Input cell keys, two `uint32` words per row. */
  cells: GraphDataView<'uint32x2'>;
  /** Word order of `cells`. Defaults to `'little-endian'` (`(low, high)`, Arrow layout). */
  wordOrder?: GPUCellCompactionWordOrder;
  /** Optional per-row mask; zero drops the row. */
  mask?: GraphDataView<'uint32'>;
  /** Optional one-row scalar: only the first `count` rows of `cells` are read. */
  count?: GraphDataView<'uint32'>;
  /** Caller-owned bounded output. */
  output: GPUCellCompactionOutput;
};

/** Saturated `7^depth` and pentagon subtree sizes `1 + 5 (7^depth - 1) / 6`, depth 0..11. */
const H3_SUBTREE_DEPTH_LIMIT = 11;

function getH3SubtreeSizes(): {hexagon: number[]; pentagon: number[]} {
  const hexagon = [1];
  const pentagon = [1];
  for (let depth = 1; depth <= H3_SUBTREE_DEPTH_LIMIT; depth++) {
    hexagon.push(hexagon[depth - 1] * 7);
    pentagon.push(pentagon[depth - 1] * 7 - 1);
  }
  return {hexagon, pentagon};
}

/** WGSL helpers shared by both operations. */
function getFamilyWGSL(family: GPUCellFamily, wordOrder: GPUCellCompactionWordOrder): string {
  const sizes = getH3SubtreeSizes();
  const toArray = (name: string, values: number[]) =>
    `const ${name} = array<u32, ${values.length}>(${values.map(value => `${value}u`).join(', ')});`;
  return /* wgsl */ `
${family === 'h3' ? dggs.source : ''}
${CELL_KEY_WGSL}
const BITS_PER_LEVEL: u32 = ${family === 'quadbin' ? 2 : 3}u;
const HEADER_HIGH: u32 = ${getCellKeyLayout(family, 0).headerHigh}u;
${family === 'h3' ? toArray('H3_HEXAGON_SIZES', sizes.hexagon) : ''}
${family === 'h3' ? toArray('H3_PENTAGON_SIZES', sizes.pentagon) : ''}

fn getLowBit(resolution: u32) -> u32 {
  return ${family === 'quadbin' ? '52u - 2u * resolution' : '45u - 3u * resolution'};
}

fn getInputKey(words: vec2u) -> vec2u {
  return ${wordOrder === 'high-low' ? 'words' : 'words.yx'};
}

fn getKeyResolution(key: vec2u) -> u32 {
  return ${family === 'quadbin' ? '(key.x >> 20u) & 0x1fu' : 'dggs_h3_get_resolution(key)'};
}

fn isValidKey(key: vec2u) -> bool {
  return ${family === 'quadbin' ? 'cellIsValidQuadbin(key)' : 'dggs_h3_is_valid_cell_id(key)'};
}

/** Number of descendants of a cell with this compact key, \`depth\` levels down; saturates. */
fn getDescendantCount(compact: vec2u, resolution: u32, depth: u32) -> u32 {
  ${
    family === 'quadbin'
      ? 'return select(0xffffffffu, 1u << (2u * depth), depth < 16u);'
      : `if (depth > ${H3_SUBTREE_DEPTH_LIMIT}u) {
    return 0xffffffffu;
  }
  let pathBits = 3u * resolution;
  let baseCell = cellShiftRight(compact, pathBits).y;
  let isPentagon = dggs_h3_is_base_cell_pentagon(baseCell) &&
    all((compact & cellMaskLow(pathBits)) == vec2u(0u));
  return select(H3_HEXAGON_SIZES[depth], H3_PENTAGON_SIZES[depth], isPentagon);`
  }
}
`;
}

/**
 * Compacts a cell set into its minimal mixed-resolution cover, or expands a mixed-resolution set
 * to one resolution (Quadbin and H3).
 *
 * `compact`: input cells at one resolution are sorted (unless `operation.sorted`), deduplicated
 * (duplicates collapse to one cell), and every cell is replaced by its coarsest ancestor whose
 * complete descendant set (4 children per Quadbin cell, 7 per H3 hexagon, 6 per H3 pentagon whose
 * deleted K subtree has no cells) is present, down to `minimumResolution`. Completeness is
 * decided by counting each unique group of equal ancestors with a binary search over the sorted
 * keys, so every level is checked in one pass; the result equals iterated sibling merging. The
 * output is in ascending canonical key order, which is ascending resolution first (the
 * resolution sits above the cell path in the key), then ascending path. Rows of another
 * resolution, invalid keys and masked rows are dropped and counted in `output.droppedCount`.
 *
 * `uncompact`: every valid input cell at or above `operation.resolution` (and within
 * `maximumDepth`) produces all of its descendants there (4^d Quadbin, 7^d H3 hexagon,
 * `1 + 5 (7^d - 1) / 6` H3 pentagon descendants, `d` the resolution gap) through a count, a
 * scan, and a per-output-slot binary search of the offsets. Descendants of one input cell come in
 * ascending key order and cells in input row order, so the output is globally ascending when the
 * input rows are in ascending path order (for example all at one resolution and sorted).
 * Duplicate or overlapping input cells produce duplicate output cells. The total count is a `u32`
 * and wraps beyond 2^32 cells.
 *
 * Both outputs are capacity bounded: `count` is clamped, the first `count` cells in the order above
 * are kept, `overflow` is written on the GPU, and rows past `count` are zero. Integer arithmetic
 * only, so results are identical on every device.
 */
export class GPUCellCompaction implements GPUMapGraphRecipe {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Stable recipe name. */
  readonly recipe = 'cell-compaction';
  /** Validated properties. */
  readonly props: GPUCellCompactionProps;
  /** Key layout of the input (`compact`) or output (`uncompact`) resolution. */
  readonly layout: CellKeyLayout;

  constructor(props: GPUCellCompactionProps) {
    this.id = props.id ?? this.recipe;
    this.props = props;
    const {id} = this;
    const {operation, output} = props;
    this.layout = getCellKeyLayout(props.family, operation.resolution);
    if (operation.type === 'compact') {
      const minimum = operation.minimumResolution ?? 0;
      if (!Number.isInteger(minimum) || minimum < 0 || minimum > operation.resolution) {
        throw new Error(`${id} minimumResolution must be an integer in [0, resolution]`);
      }
    } else if (operation.type === 'uncompact') {
      const limit = props.family === 'quadbin' ? 15 : H3_SUBTREE_DEPTH_LIMIT;
      const depth = operation.maximumDepth ?? GPU_CELL_UNCOMPACT_DEFAULT_MAXIMUM_DEPTH;
      if (!Number.isInteger(depth) || depth < 0 || depth > limit) {
        throw new Error(`${id} maximumDepth must be an integer in [0, ${limit}]`);
      }
    } else {
      throw new Error(`${id} operation.type must be 'compact' or 'uncompact'`);
    }
    if (props.family !== 'quadbin' && props.family !== 'h3') {
      throw new Error(`${id} family must be 'quadbin' or 'h3'`);
    }
    if (props.wordOrder && props.wordOrder !== 'little-endian' && props.wordOrder !== 'high-low') {
      throw new Error(`${id} wordOrder must be 'little-endian' or 'high-low'`);
    }
    validatePackedView(props.cells, ['uint32x2'], `${id} cells`);
    if (props.cells.length < 1) {
      throw new Error(`${id} cells must hold at least one row`);
    }
    if (props.mask) {
      validatePackedUint32View(props.mask, `${id} mask`);
      if (props.mask.length !== props.cells.length) {
        throw new Error(`${id} mask must have the same length as cells`);
      }
    }
    for (const [name, view] of [
      ['count', props.count],
      ['output.count', output.count],
      ['output.overflow', output.overflow],
      ['output.totalCount', output.totalCount],
      ['output.droppedCount', output.droppedCount]
    ] as const) {
      if (!view) {
        continue;
      }
      validatePackedUint32View(view, `${id} ${name}`);
      if (view.length < 1) {
        throw new Error(`${id} ${name} must contain one uint32 row`);
      }
    }
    validatePackedView(output.cells, ['uint32x2'], `${id} output.cells`);
    if (output.cells.length < 1) {
      throw new Error(`${id} output.cells must hold at least one row`);
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [output.cells, output.count, output.overflow, output.totalCount, output.droppedCount],
      [props.cells, props.mask, props.count]
    );
    const outputs = [
      output.cells,
      output.count,
      output.overflow,
      output.totalCount,
      output.droppedCount
    ].filter(Boolean) as GraphDataView[];
    if (new Set(outputs).size !== outputs.length) {
      throw new Error(`${id} output views must be distinct`);
    }
  }

  /** Returns the key, table, analysis, ordering, and publish nodes of the selected operation. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props} = this;
    validateGraphViewsBelongToGraph(this.id, graph, [
      props.cells,
      props.mask,
      props.count,
      props.output.cells,
      props.output.count,
      props.output.overflow,
      props.output.totalCount,
      props.output.droppedCount
    ]);
    return props.operation.type === 'compact'
      ? this.getCompactNodes(graph, props.operation)
      : this.getUncompactNodes(graph, props.operation);
  }

  private getInputBindings(): {
    bindings: {name: string; view: GraphDataView; type: 'u32'; access: 'read'}[];
    rowCountWGSL: string;
    maskWGSL: string;
  } {
    const {props} = this;
    const bindings: {name: string; view: GraphDataView; type: 'u32'; access: 'read'}[] = [
      {name: 'cells', view: props.cells, type: 'u32', access: 'read'}
    ];
    if (props.mask) {
      bindings.push({name: 'mask', view: props.mask, type: 'u32', access: 'read'});
    }
    if (props.count) {
      bindings.push({name: 'rowCount', view: props.count, type: 'u32', access: 'read'});
    }
    return {
      bindings,
      rowCountWGSL: props.count ? 'rowCount[rowCountOffset]' : `${props.cells.length}u`,
      maskWGSL: props.mask ? 'mask[maskOffset + index] != 0u' : 'true'
    };
  }

  private getCompactNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    operation: GPUCellCompactOperation
  ): GPUCommandNode<Parameters>[] {
    const {id, props, layout} = this;
    const {output} = props;
    const rows = props.cells.length;
    const minimumResolution = operation.minimumResolution ?? 0;
    const maximumDepth = operation.resolution - minimumResolution;
    const sorted = operation.sorted ?? false;
    const twoWords = hasCellKeyHighWord(layout);
    const u32 = (name: string, length = rows) =>
      createTransientView(graph, `${id}-${name}`, 'uint32', length);
    const nodes: GPUCommandNode<Parameters>[] = [];
    const input = this.getInputBindings();
    const familyWGSL = getFamilyWGSL(props.family, props.wordOrder ?? 'little-endian');

    if (output.droppedCount) {
      nodes.push(
        createMapGraphFillNode<Parameters>(graph, {
          id: `${id}-clear-dropped`,
          operation: OPERATION,
          view: output.droppedCount,
          type: 'u32',
          value: '0u',
          componentCount: 1
        })
      );
    }

    // 1. Compact keys of valid rows at the input resolution; every other row gets an invalid key.
    const keyLow = u32('key-low');
    const keyHigh = twoWords ? u32('key-high') : undefined;
    const rowIds = sorted ? undefined : u32('row-ids');
    nodes.push(
      createMapGraphKernelNode<Parameters>(graph, {
        id: `${id}-keys`,
        operation: OPERATION,
        variant: 'compact-keys',
        bindings: [
          ...input.bindings,
          {name: 'keyLow', view: keyLow, type: 'u32', access: 'read_write'},
          ...(keyHigh
            ? [{name: 'keyHigh', view: keyHigh, type: 'u32', access: 'read_write'} as const]
            : []),
          ...(rowIds
            ? [{name: 'rowIds', view: rowIds, type: 'u32', access: 'read_write'} as const]
            : []),
          ...(output.droppedCount
            ? [
                {
                  name: 'droppedCount',
                  view: output.droppedCount,
                  type: 'atomic<u32>',
                  access: 'read_write'
                } as const
              ]
            : [])
        ],
        invocationCount: rows,
        declarations: `const RESOLUTION: u32 = ${operation.resolution}u;
${familyWGSL}`,
        body: `var compact = cellShiftLeft(vec2u(0u, 1u), ${layout.width}u);
  if (index < ${input.rowCountWGSL}) {
    let key = getInputKey(vec2u(cells[cellsOffset + 2u * index], cells[cellsOffset + 2u * index + 1u]));
    if (${input.maskWGSL} && isValidKey(key) && getKeyResolution(key) == RESOLUTION) {
      compact = cellGetCompactKey(key, ${layout.lowBit}u, ${layout.width}u);
    } else {
      ${output.droppedCount ? 'atomicAdd(&droppedCount[droppedCountOffset], 1u);' : ''}
    }
  }
  keyLow[keyLowOffset + index] = compact.y;
  ${keyHigh ? 'keyHigh[keyHighOffset + index] = compact.x;' : ''}
  ${rowIds ? 'rowIds[rowIdsOffset + index] = index;' : ''}`
      })
    );

    // 2. Sort and deduplicate into a transient unique cell table of full capacity.
    const uniqueView = createTransientView(graph, `${id}-unique-table`, 'uint32x2', rows);
    const uniqueCount = u32('unique-count', 1);
    nodes.push(
      ...getCellTableNodes<Parameters>(graph, {
        id: `${id}-table`,
        operation: OPERATION,
        layout,
        keyLow,
        keyHigh,
        rowIds,
        sorted,
        source: {kind: 'rows'},
        sumScale: 65536,
        output: {
          cells: uniqueView,
          counts: u32('unique-counts'),
          count: uniqueCount,
          overflow: u32('unique-overflow', 1)
        }
      })
    );

    // 3. Per row: coarsest complete ancestor depth, and whether this row emits its group.
    const flags = u32('flags');
    const depths = u32('depths');
    nodes.push(
      createMapGraphKernelNode<Parameters>(graph, {
        id: `${id}-analyze`,
        operation: OPERATION,
        variant: 'analyze',
        bindings: [
          {name: 'cells', view: uniqueView, type: 'u32', access: 'read'},
          {name: 'uniqueCount', view: uniqueCount, type: 'u32', access: 'read'},
          {name: 'flags', view: flags, type: 'u32', access: 'read_write'},
          {name: 'depths', view: depths, type: 'u32', access: 'read_write'}
        ],
        invocationCount: rows,
        declarations: `const RESOLUTION: u32 = ${operation.resolution}u;
const MAXIMUM_DEPTH: u32 = ${maximumDepth}u;
${familyWGSL}

fn getCompactAt(row: u32) -> vec2u {
  return cellGetCompactKey(
    vec2u(cells[cellsOffset + 2u * row + 1u], cells[cellsOffset + 2u * row]), ${layout.lowBit}u, ${layout.width}u);
}

fn getAncestor(compact: vec2u, depth: u32) -> vec2u {
  return cellShiftRight(compact, BITS_PER_LEVEL * depth);
}

fn isLess(a: vec2u, b: vec2u) -> bool {
  return a.x < b.x || (a.x == b.x && a.y < b.y);
}

/** First row whose ancestor is >= (or, for upper bounds, > ) \`ancestor\`. */
fn findBound(ancestor: vec2u, depth: u32, rowCount: u32, isUpper: bool) -> u32 {
  var low = 0u;
  var high = rowCount;
  loop {
    if (low >= high) {
      break;
    }
    let middle = (low + high) >> 1u;
    let value = getAncestor(getCompactAt(middle), depth);
    let goRight = select(isLess(value, ancestor), !isLess(ancestor, value), isUpper);
    if (goRight) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  return low;
}`,
        body: `let rowCount = uniqueCount[uniqueCountOffset];
  if (index >= rowCount) {
    flags[flagsOffset + index] = 0u;
    depths[depthsOffset + index] = 0u;
    return;
  }
  let compact = getCompactAt(index);
  var bestDepth = 0u;
  var bestStart = index;
  // Completeness is monotone: a complete ancestor implies complete descendants.
  for (var depth = 1u; depth <= MAXIMUM_DEPTH; depth++) {
    let ancestor = getAncestor(compact, depth);
    let lower = findBound(ancestor, depth, rowCount, false);
    let upper = findBound(ancestor, depth, rowCount, true);
    if (upper - lower != getDescendantCount(ancestor, RESOLUTION - depth, depth)) {
      break;
    }
    bestDepth = depth;
    bestStart = lower;
  }
  flags[flagsOffset + index] = select(0u, 1u, bestStart == index);
  depths[depthsOffset + index] = bestDepth;`
      })
    );

    // 4. Output positions of emitting rows.
    const positions = u32('positions');
    nodes.push(
      ...new GPUScan({
        id: `${id}-scan`,
        input: flags,
        output: positions,
        mode: 'exclusive'
      }).getCommandNodes(graph)
    );

    // 5. Scatter emitted cells (with their resolution as the sort key) and write the total.
    const emitted = createTransientView(graph, `${id}-emitted`, 'uint32x2', rows);
    const sortKeys = u32('sort-keys');
    const sortIds = u32('sort-ids');
    const total = u32('total', 1);
    nodes.push(
      createMapGraphKernelNode<Parameters>(graph, {
        id: `${id}-scatter`,
        operation: OPERATION,
        variant: 'scatter',
        bindings: [
          {name: 'cells', view: uniqueView, type: 'u32', access: 'read'},
          {name: 'flags', view: flags, type: 'u32', access: 'read'},
          {name: 'depths', view: depths, type: 'u32', access: 'read'},
          {name: 'positions', view: positions, type: 'u32', access: 'read'},
          {name: 'emitted', view: emitted, type: 'u32', access: 'read_write'},
          {name: 'sortKeys', view: sortKeys, type: 'u32', access: 'read_write'},
          {name: 'sortIds', view: sortIds, type: 'u32', access: 'read_write'},
          {name: 'total', view: total, type: 'u32', access: 'read_write'}
        ],
        invocationCount: rows,
        declarations: `const RESOLUTION: u32 = ${operation.resolution}u;
const ROW_COUNT: u32 = ${rows}u;
${familyWGSL}`,
        body: `let emitCount = positions[positionsOffset + ROW_COUNT - 1u] + flags[flagsOffset + ROW_COUNT - 1u];
  if (index == 0u) {
    total[totalOffset] = emitCount;
  }
  if (index >= emitCount) {
    // Padding rows sort last (resolutions are below 31).
    sortKeys[sortKeysOffset + index] = 31u;
    sortIds[sortIdsOffset + index] = index;
  }
  if (flags[flagsOffset + index] != 0u) {
    let depth = depths[depthsOffset + index];
    let compact = cellGetCompactKey(
      vec2u(cells[cellsOffset + 2u * index + 1u], cells[cellsOffset + 2u * index]), ${layout.lowBit}u, ${layout.width}u);
    let resolution = RESOLUTION - depth;
    let key = cellGetKey(cellShiftRight(compact, BITS_PER_LEVEL * depth), HEADER_HIGH, resolution, getLowBit(resolution));
    let slot = positions[positionsOffset + index];
    emitted[emittedOffset + 2u * slot] = key.y;
    emitted[emittedOffset + 2u * slot + 1u] = key.x;
    sortKeys[sortKeysOffset + slot] = resolution;
    sortIds[sortIdsOffset + slot] = slot;
  }`
      })
    );

    // 6. Stable sort by resolution: within a resolution the emitted cells are already ascending.
    const orderedKeys = u32('ordered-keys');
    const orderedIds = u32('ordered-ids');
    nodes.push(
      ...new GPUSort({
        id: `${id}-sort-resolution`,
        keys: sortKeys,
        values: sortIds,
        outputKeys: orderedKeys,
        outputValues: orderedIds,
        keyBits: 5
      }).getCommandNodes(graph)
    );

    // 7. Gather into the bounded output and publish count, overflow, and totals.
    nodes.push(
      createMapGraphKernelNode<Parameters>(graph, {
        id: `${id}-emit`,
        operation: OPERATION,
        variant: 'emit',
        bindings: [
          {name: 'total', view: total, type: 'u32', access: 'read'},
          {name: 'orderedIds', view: orderedIds, type: 'u32', access: 'read'},
          {name: 'emitted', view: emitted, type: 'u32', access: 'read'},
          {name: 'cellsOut', view: output.cells, type: 'u32', access: 'read_write'},
          {name: 'countOut', view: output.count, type: 'u32', access: 'read_write'},
          {name: 'overflowOut', view: output.overflow, type: 'u32', access: 'read_write'},
          ...(output.totalCount
            ? [
                {
                  name: 'totalOut',
                  view: output.totalCount,
                  type: 'u32',
                  access: 'read_write'
                } as const
              ]
            : [])
        ],
        invocationCount: output.cells.length,
        declarations: `const CAPACITY: u32 = ${output.cells.length}u;`,
        body: `let totalCells = total[totalOffset];
  let count = min(totalCells, CAPACITY);
  if (index == 0u) {
    countOut[countOutOffset] = count;
    overflowOut[overflowOutOffset] = select(0u, 1u, totalCells > CAPACITY);
    ${output.totalCount ? 'totalOut[totalOutOffset] = totalCells;' : ''}
  }
  var low = 0u;
  var high = 0u;
  if (index < count) {
    let slot = orderedIds[orderedIdsOffset + index];
    low = emitted[emittedOffset + 2u * slot];
    high = emitted[emittedOffset + 2u * slot + 1u];
  }
  cellsOut[cellsOutOffset + 2u * index] = low;
  cellsOut[cellsOutOffset + 2u * index + 1u] = high;`
      })
    );
    return nodes;
  }

  private getUncompactNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    operation: GPUCellUncompactOperation
  ): GPUCommandNode<Parameters>[] {
    const {id, props, layout} = this;
    const {output} = props;
    const rows = props.cells.length;
    const maximumDepth = operation.maximumDepth ?? GPU_CELL_UNCOMPACT_DEFAULT_MAXIMUM_DEPTH;
    const u32 = (name: string, length = rows) =>
      createTransientView(graph, `${id}-${name}`, 'uint32', length);
    const nodes: GPUCommandNode<Parameters>[] = [];
    const input = this.getInputBindings();
    const familyWGSL = getFamilyWGSL(props.family, props.wordOrder ?? 'little-endian');
    const declarations = `const RESOLUTION: u32 = ${operation.resolution}u;
const MAXIMUM_DEPTH: u32 = ${maximumDepth}u;
${familyWGSL}

fn getCompact(key: vec2u, resolution: u32) -> vec2u {
  let lowBit = getLowBit(resolution);
  return cellGetCompactKey(key, lowBit, 52u - lowBit);
}`;

    if (output.droppedCount) {
      nodes.push(
        createMapGraphFillNode<Parameters>(graph, {
          id: `${id}-clear-dropped`,
          operation: OPERATION,
          view: output.droppedCount,
          type: 'u32',
          value: '0u',
          componentCount: 1
        })
      );
    }

    // 1. Descendant count of every input row (zero for dropped rows).
    const counts = u32('counts');
    nodes.push(
      createMapGraphKernelNode<Parameters>(graph, {
        id: `${id}-count`,
        operation: OPERATION,
        variant: 'uncompact-count',
        bindings: [
          ...input.bindings,
          {name: 'counts', view: counts, type: 'u32', access: 'read_write'},
          ...(output.droppedCount
            ? [
                {
                  name: 'droppedCount',
                  view: output.droppedCount,
                  type: 'atomic<u32>',
                  access: 'read_write'
                } as const
              ]
            : [])
        ],
        invocationCount: rows,
        declarations,
        body: `var descendants = 0u;
  if (index < ${input.rowCountWGSL}) {
    let key = getInputKey(vec2u(cells[cellsOffset + 2u * index], cells[cellsOffset + 2u * index + 1u]));
    var keep = false;
    if (${input.maskWGSL} && isValidKey(key)) {
      let resolution = getKeyResolution(key);
      if (resolution <= RESOLUTION && RESOLUTION - resolution <= MAXIMUM_DEPTH) {
        keep = true;
        descendants = getDescendantCount(getCompact(key, resolution), resolution, RESOLUTION - resolution);
      }
    }
    if (!keep) {
      ${output.droppedCount ? 'atomicAdd(&droppedCount[droppedCountOffset], 1u);' : ''}
    }
  }
  counts[countsOffset + index] = descendants;`
      })
    );

    // 2. Output offsets.
    const offsets = u32('offsets');
    nodes.push(
      ...new GPUScan({
        id: `${id}-scan`,
        input: counts,
        output: offsets,
        mode: 'exclusive'
      }).getCommandNodes(graph)
    );

    // 3. One invocation per output slot finds its source row and unranks the descendant.
    nodes.push(
      createMapGraphKernelNode<Parameters>(graph, {
        id: `${id}-expand`,
        operation: OPERATION,
        variant: 'uncompact-expand',
        bindings: [
          {name: 'cells', view: props.cells, type: 'u32', access: 'read'},
          {name: 'counts', view: counts, type: 'u32', access: 'read'},
          {name: 'offsets', view: offsets, type: 'u32', access: 'read'},
          {name: 'cellsOut', view: output.cells, type: 'u32', access: 'read_write'},
          {name: 'countOut', view: output.count, type: 'u32', access: 'read_write'},
          {name: 'overflowOut', view: output.overflow, type: 'u32', access: 'read_write'},
          ...(output.totalCount
            ? [
                {
                  name: 'totalOut',
                  view: output.totalCount,
                  type: 'u32',
                  access: 'read_write'
                } as const
              ]
            : [])
        ],
        invocationCount: output.cells.length,
        declarations: `const CAPACITY: u32 = ${output.cells.length}u;
const ROW_COUNT: u32 = ${rows}u;
${declarations}

/** The \`rank\`-th (ascending) descendant path \`depth\` levels below \`compact\`. */
fn getDescendant(compact: vec2u, resolution: u32, depth: u32, rankIn: u32) -> vec2u {
  ${
    props.family === 'quadbin'
      ? 'return cellShiftLeft(compact, 2u * depth) | vec2u(0u, rankIn);'
      : `var rank = rankIn;
  var digits = vec2u(0u);
  var isChain = dggs_h3_is_base_cell_pentagon(cellShiftRight(compact, 3u * resolution).y) &&
    all((compact & cellMaskLow(3u * resolution)) == vec2u(0u));
  for (var step = 0u; step < depth; step++) {
    let remaining = depth - 1u - step;
    var digit = 0u;
    if (isChain) {
      // Pentagon: digit 0 keeps the pentagon, digit 1 is deleted, digits 2..6 are hexagons.
      if (rank >= H3_PENTAGON_SIZES[remaining]) {
        rank = rank - H3_PENTAGON_SIZES[remaining];
        digit = 2u + rank / H3_HEXAGON_SIZES[remaining];
        rank = rank % H3_HEXAGON_SIZES[remaining];
        isChain = false;
      }
    } else {
      digit = rank / H3_HEXAGON_SIZES[remaining];
      rank = rank % H3_HEXAGON_SIZES[remaining];
    }
    digits = cellShiftLeft(digits, 3u) | vec2u(0u, digit);
  }
  return cellShiftLeft(compact, 3u * depth) | digits;`
  }
}`,
        body: `let totalCells = offsets[offsetsOffset + ROW_COUNT - 1u] + counts[countsOffset + ROW_COUNT - 1u];
  let count = min(totalCells, CAPACITY);
  if (index == 0u) {
    countOut[countOutOffset] = count;
    overflowOut[overflowOutOffset] = select(0u, 1u, totalCells > CAPACITY);
    ${output.totalCount ? 'totalOut[totalOutOffset] = totalCells;' : ''}
  }
  var result = vec2u(0u);
  if (index < count) {
    // Last row whose offset is <= index; zero-count rows share the next row's offset.
    var low = 0u;
    var high = ROW_COUNT;
    loop {
      if (low >= high) {
        break;
      }
      let middle = (low + high) >> 1u;
      if (offsets[offsetsOffset + middle] <= index) {
        low = middle + 1u;
      } else {
        high = middle;
      }
    }
    let row = low - 1u;
    let key = getInputKey(vec2u(cells[cellsOffset + 2u * row], cells[cellsOffset + 2u * row + 1u]));
    let resolution = getKeyResolution(key);
    let depth = RESOLUTION - resolution;
    let compact = getDescendant(getCompact(key, resolution), resolution, depth, index - offsets[offsetsOffset + row]);
    result = cellGetKey(compact, HEADER_HIGH, RESOLUTION, ${layout.lowBit}u);
  }
  cellsOut[cellsOutOffset + 2u * index] = result.y;
  cellsOut[cellsOutOffset + 2u * index + 1u] = result.x;`
      })
    );
    return nodes;
  }
}
