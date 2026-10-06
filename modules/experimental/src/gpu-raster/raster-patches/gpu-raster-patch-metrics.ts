// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {
  createFillNode,
  createWGSLKernelNode,
  getWGSLFloatLiteral,
  type WGSLKernelBinding
} from '../../utils/wgsl-kernel-nodes';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {
  createPatchKeysNode,
  getPatchLabelViews,
  validatePatchLabels,
  type GPURasterPatchLabels
} from './patch-labels';

const OPERATION = 'GPURasterPatchMetrics';
const NO_ROW = '0xffffffffu';

/**
 * Caller-owned per-patch columns of {@link GPURasterPatchMetrics}. Row `r` describes dense label
 * `r + 1`. Every provided column must have the same length (the patch capacity). Provide only the
 * columns you need; at least one is required.
 */
export type GPURasterPatchMetricsOutput = {
  /** Pixel count of each patch (area in pixels). 0 marks an unused row. */
  pixelCounts?: GraphDataView<'uint32'>;
  /** Pixel count times the absolute affine determinant, in CRS units squared. */
  areas?: GraphDataView<'float32'>;
  /**
   * Number of exposed pixel faces: faces whose 4-neighbor holds another label, background, an
   * invalid pixel, or (with `countRasterBorder`) lies outside the raster. Exact integer.
   */
  perimeterFaceCounts?: GraphDataView<'uint32'>;
  /**
   * Perimeter length in CRS units: faces between vertically adjacent pixels have the length of one
   * column step of the affine, faces between horizontally adjacent pixels the length of one row
   * step. Holes contribute their inner boundary.
   */
  perimeters?: GraphDataView<'float32'>;
  /** Inclusive minimum pixel column of each patch (0 for an empty row). */
  minColumns?: GraphDataView<'uint32'>;
  /** Inclusive minimum pixel row of each patch (0 for an empty row). */
  minRows?: GraphDataView<'uint32'>;
  /** Inclusive maximum pixel column of each patch (0 for an empty row). */
  maxColumns?: GraphDataView<'uint32'>;
  /** Inclusive maximum pixel row of each patch (0 for an empty row). */
  maxRows?: GraphDataView<'uint32'>;
};

/**
 * Properties for {@link GPURasterPatchMetrics}.
 *
 * Per-frame: the contents of the label views. Compile-time: dimensions, `affine`,
 * `countRasterBorder`, the output capacity and which columns are present.
 */
export type GPURasterPatchMetricsProps = GPURasterPatchLabels & {
  /** Prefix for generated node and transient IDs. Defaults to `'raster-patch-metrics'`. */
  id?: string;
  /**
   * Pixel-to-world matrix `[a, b, c, d, e, f]` with `x = a * column + b * row + c` and
   * `y = d * column + e * row + f`, as `GPURasterMetadata.affine`. Defaults to the identity, which
   * reports pixel units.
   */
  affine?: readonly [number, number, number, number, number, number];
  /** Whether faces on the raster edge count as perimeter. Default `true`. */
  countRasterBorder?: boolean;
  /** Caller-owned per-patch output columns. */
  output: GPURasterPatchMetricsOutput;
};

/**
 * Per-patch metrics of a dense label raster: area, perimeter and bounding box
 * (the `r.clump` / FRAGSTATS companions to `GPURasterConnectedComponents`).
 *
 * Intensity statistics and centroids stay in `GPURasterRegionMeasurements`; this contributor adds
 * what it lacks, perimeter and extent, and repeats the pixel count and area so one node set is
 * enough for shape ratios. Labels are the output of `GPURasterDenseComponents`; labels above the
 * output capacity are ignored.
 *
 * Counts, extents and perimeter faces are integer atomics, so results are exact and
 * deterministic; `perimeters` and `areas` are single f32 products of exact integers.
 */
export class GPURasterPatchMetrics implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPURasterPatchMetricsProps;
  /** Number of patch rows. */
  readonly patchCapacity: number;

  constructor(props: GPURasterPatchMetricsProps) {
    const id = props.id ?? 'raster-patch-metrics';
    this.id = id;
    this.props = props;
    validatePatchLabels(id, props);
    const columns = Object.entries(props.output).filter(([, view]) => view !== undefined) as [
      string,
      GraphDataView<'uint32'> | GraphDataView<'float32'>
    ][];
    if (columns.length === 0) {
      throw new Error(`${id} output must provide at least one column`);
    }
    this.patchCapacity = columns[0][1].length;
    for (const [name, view] of columns) {
      if (name === 'areas' || name === 'perimeters') {
        validatePackedView(view, ['float32'], `${id} output.${name}`);
      } else {
        validatePackedUint32View(view as GraphDataView<'uint32'>, `${id} output.${name}`);
      }
      if (view.length !== this.patchCapacity) {
        throw new Error(`${id} output columns must have identical lengths`);
      }
    }
    if (this.patchCapacity < 1) {
      throw new Error(`${id} output columns must hold at least one patch row`);
    }
    if (props.affine) {
      if (props.affine.length !== 6 || props.affine.some(value => !Number.isFinite(value))) {
        throw new Error(`${id} affine must hold six finite numbers`);
      }
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      columns.map(([, view]) => view),
      getPatchLabelViews(props)
    );
  }

  /** Returns the key, accumulation and finalize nodes in dependency order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, patchCapacity} = this;
    const {output, width, height} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      ...getPatchLabelViews(props),
      ...(Object.values(output).filter(view => view !== undefined) as GraphDataView[])
    ]);
    const [a, b, , d, e] = props.affine ?? [1, 0, 0, 0, 1, 0];
    const columnStep = Math.hypot(a, d);
    const rowStep = Math.hypot(b, e);
    const pixelArea = Math.abs(a * e - b * d);
    const countBorder = props.countRasterBorder ?? true;
    const view = (name: string, length: number) =>
      createTransientView(graph, `${id}-${name}`, 'uint32', length);

    const {node: keysNode, keys} = createPatchKeysNode(graph, id, OPERATION, props, patchCapacity);
    const nodes: GPUCommandNode<Parameters>[] = [keysNode];
    const pixelCounts = output.pixelCounts ?? view('pixel-counts', patchCapacity);
    const pixelLength = width * height;
    const geometry = `const WIDTH: u32 = ${width}u;
const HEIGHT: u32 = ${height}u;`;

    nodes.push(
      createFillNode<Parameters>(graph, {
        id: `${id}-counts-clear`,
        operation: OPERATION,
        view: pixelCounts,
        type: 'u32',
        value: '0u'
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-counts`,
        operation: OPERATION,
        variant: 'counts',
        bindings: [
          {name: 'keys', view: keys, type: 'u32', access: 'read'},
          {name: 'pixelCounts', view: pixelCounts, type: 'atomic<u32>', access: 'read_write'}
        ],
        invocationCount: pixelLength,
        body: `let key = keys[keysOffset + index];
  if (key != 0u) {
    atomicAdd(&pixelCounts[pixelCountsOffset + key - 1u], 1u);
  }`
      })
    );

    const hasExtent = output.minColumns || output.minRows || output.maxColumns || output.maxRows;
    if (hasExtent) {
      const extents = {
        minColumns: output.minColumns ?? view('min-columns', patchCapacity),
        minRows: output.minRows ?? view('min-rows', patchCapacity),
        maxColumns: output.maxColumns ?? view('max-columns', patchCapacity),
        maxRows: output.maxRows ?? view('max-rows', patchCapacity)
      };
      for (const name of ['minColumns', 'minRows'] as const) {
        nodes.push(
          createFillNode<Parameters>(graph, {
            id: `${id}-${name}-clear`,
            operation: OPERATION,
            view: extents[name],
            type: 'u32',
            value: NO_ROW
          })
        );
      }
      for (const name of ['maxColumns', 'maxRows'] as const) {
        nodes.push(
          createFillNode<Parameters>(graph, {
            id: `${id}-${name}-clear`,
            operation: OPERATION,
            view: extents[name],
            type: 'u32',
            value: '0u'
          })
        );
      }
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-extents`,
          operation: OPERATION,
          variant: 'extents',
          bindings: [
            {name: 'keys', view: keys, type: 'u32', access: 'read'},
            {
              name: 'minColumns',
              view: extents.minColumns,
              type: 'atomic<u32>',
              access: 'read_write'
            },
            {name: 'minRows', view: extents.minRows, type: 'atomic<u32>', access: 'read_write'},
            {
              name: 'maxColumns',
              view: extents.maxColumns,
              type: 'atomic<u32>',
              access: 'read_write'
            },
            {name: 'maxRows', view: extents.maxRows, type: 'atomic<u32>', access: 'read_write'}
          ],
          invocationCount: pixelLength,
          declarations: geometry,
          body: `let key = keys[keysOffset + index];
  if (key != 0u) {
    let column = index % WIDTH;
    let row = index / WIDTH;
    atomicMin(&minColumns[minColumnsOffset + key - 1u], column);
    atomicMin(&minRows[minRowsOffset + key - 1u], row);
    atomicMax(&maxColumns[maxColumnsOffset + key - 1u], column);
    atomicMax(&maxRows[maxRowsOffset + key - 1u], row);
  }`
        }),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-extents-empty`,
          operation: OPERATION,
          variant: 'extents-empty',
          bindings: [
            {name: 'pixelCounts', view: pixelCounts, type: 'u32', access: 'read'},
            {name: 'minColumns', view: extents.minColumns, type: 'u32', access: 'read_write'},
            {name: 'minRows', view: extents.minRows, type: 'u32', access: 'read_write'}
          ],
          invocationCount: patchCapacity,
          body: `if (pixelCounts[pixelCountsOffset + index] == 0u) {
    minColumns[minColumnsOffset + index] = 0u;
    minRows[minRowsOffset + index] = 0u;
  }`
        })
      );
    }

    if (output.perimeterFaceCounts || output.perimeters) {
      const acrossRows = view('faces-across-rows', patchCapacity);
      const acrossColumns = view('faces-across-columns', patchCapacity);
      nodes.push(
        createFillNode<Parameters>(graph, {
          id: `${id}-faces-rows-clear`,
          operation: OPERATION,
          view: acrossRows,
          type: 'u32',
          value: '0u'
        }),
        createFillNode<Parameters>(graph, {
          id: `${id}-faces-columns-clear`,
          operation: OPERATION,
          view: acrossColumns,
          type: 'u32',
          value: '0u'
        }),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-faces`,
          operation: OPERATION,
          variant: 'faces',
          bindings: [
            {name: 'keys', view: keys, type: 'u32', access: 'read'},
            {name: 'acrossRows', view: acrossRows, type: 'atomic<u32>', access: 'read_write'},
            {name: 'acrossColumns', view: acrossColumns, type: 'atomic<u32>', access: 'read_write'}
          ],
          invocationCount: pixelLength,
          declarations: `${geometry}
const COUNT_BORDER: bool = ${countBorder};

// Whether the neighbor at (column + dx, row + dy) differs from \`key\`.
fn isExposed(column: u32, row: u32, dx: i32, dy: i32, key: u32) -> bool {
  let neighborColumn = i32(column) + dx;
  let neighborRow = i32(row) + dy;
  if (neighborColumn < 0 || neighborRow < 0 || neighborColumn >= i32(WIDTH) || neighborRow >= i32(HEIGHT)) {
    return COUNT_BORDER;
  }
  return keys[keysOffset + u32(neighborRow) * WIDTH + u32(neighborColumn)] != key;
}`,
          body: `let key = keys[keysOffset + index];
  if (key != 0u) {
    let column = index % WIDTH;
    let row = index / WIDTH;
    var rowFaces = 0u;
    var columnFaces = 0u;
    if (isExposed(column, row, 0, -1, key)) { rowFaces++; }
    if (isExposed(column, row, 0, 1, key)) { rowFaces++; }
    if (isExposed(column, row, -1, 0, key)) { columnFaces++; }
    if (isExposed(column, row, 1, 0, key)) { columnFaces++; }
    if (rowFaces > 0u) { atomicAdd(&acrossRows[acrossRowsOffset + key - 1u], rowFaces); }
    if (columnFaces > 0u) { atomicAdd(&acrossColumns[acrossColumnsOffset + key - 1u], columnFaces); }
  }`
        })
      );
      const finalizeBindings: WGSLKernelBinding[] = [
        {name: 'acrossRows', view: acrossRows, type: 'u32', access: 'read'},
        {name: 'acrossColumns', view: acrossColumns, type: 'u32', access: 'read'}
      ];
      if (output.perimeterFaceCounts) {
        finalizeBindings.push({
          name: 'faceCounts',
          view: output.perimeterFaceCounts,
          type: 'u32',
          access: 'read_write'
        });
      }
      if (output.perimeters) {
        finalizeBindings.push({
          name: 'perimeters',
          view: output.perimeters,
          type: 'f32',
          access: 'read_write'
        });
      }
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-perimeters`,
          operation: OPERATION,
          variant: 'perimeters',
          bindings: finalizeBindings,
          invocationCount: patchCapacity,
          body: `let rowFaces = acrossRows[acrossRowsOffset + index];
  let columnFaces = acrossColumns[acrossColumnsOffset + index];
  ${output.perimeterFaceCounts ? 'faceCounts[faceCountsOffset + index] = rowFaces + columnFaces;' : ''}
  ${
    output.perimeters
      ? `perimeters[perimetersOffset + index] = f32(rowFaces) * ${getWGSLFloatLiteral(columnStep)} + f32(columnFaces) * ${getWGSLFloatLiteral(rowStep)};`
      : ''
  }`
        })
      );
    }

    if (output.areas) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-areas`,
          operation: OPERATION,
          variant: 'areas',
          bindings: [
            {name: 'pixelCounts', view: pixelCounts, type: 'u32', access: 'read'},
            {name: 'areas', view: output.areas, type: 'f32', access: 'read_write'}
          ],
          invocationCount: patchCapacity,
          body: `areas[areasOffset + index] = f32(pixelCounts[pixelCountsOffset + index]) * ${getWGSLFloatLiteral(pixelArea)};`
        })
      );
    }
    return nodes;
  }
}
