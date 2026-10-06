// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GraphVectorView,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';

const OPERATION = 'GPUGridGenerator';

/** Number of float32 elements in a {@link GPUGridGenerator} parameter buffer. */
export const GPU_GRID_GENERATOR_PARAMETER_LENGTH = 4;

/**
 * Kind of grid produced by {@link GPUGridGenerator}.
 *
 * - `'square'`: `columns x rows` rectangles, 4 vertices each.
 * - `'hex'`: `columns x rows` pointy-top hexagons, 6 vertices each.
 * - `'triangle'`: `columns x rows` strips of alternating up and down triangles, `2 * columns` per
 *   row, 3 vertices each.
 * - `'point'`: `columns x rows` points at the centers of the square cells, no polygon vertices.
 */
export type GPUGridType = 'square' | 'hex' | 'triangle' | 'point';

/** CPU description of the per-frame parameters of {@link GPUGridGenerator}. */
export type GPUGridGeneratorParameters = {
  /** X of the lower-left corner of the grid. */
  minX: number;
  /** Y of the lower-left corner of the grid. */
  minY: number;
  /**
   * Cell width. For `'hex'` it is the flat-to-flat width of a hexagon; for `'triangle'` the base
   * of one triangle.
   */
  cellWidth: number;
  /**
   * Cell height. Ignored by `'hex'` (a regular hexagon's pitch follows from `cellWidth`); for
   * `'triangle'` it is the height of one strip (use `cellWidth * sqrt(3) / 2` for equilateral).
   */
  cellHeight: number;
};

/**
 * Packs {@link GPUGridGeneratorParameters} into the 4-element float32 layout
 * `[minX, minY, cellWidth, cellHeight]`. Write the result into a `GPUParameterBuffer` between
 * encodings to pan or zoom the grid without recompiling.
 *
 * @param parameters Parameters to encode.
 * @param target Optional destination of at least 4 elements.
 * @throws If a value is not finite, a cell size is not positive, or `target` is too short.
 */
export function getGPUGridGeneratorParameterValues(
  parameters: GPUGridGeneratorParameters,
  target: Float32Array = new Float32Array(GPU_GRID_GENERATOR_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_GRID_GENERATOR_PARAMETER_LENGTH) {
    throw new Error(
      `Grid generator target must hold ${GPU_GRID_GENERATOR_PARAMETER_LENGTH} elements`
    );
  }
  const {minX, minY, cellWidth, cellHeight} = parameters;
  for (const value of [minX, minY, cellWidth, cellHeight]) {
    if (!Number.isFinite(value)) {
      throw new Error('Grid generator parameters must be finite');
    }
  }
  if (cellWidth <= 0 || cellHeight <= 0) {
    throw new Error('Grid generator cellWidth and cellHeight must be positive');
  }
  target.set([minX, minY, cellWidth, cellHeight]);
  return target;
}

/** Number of polygon vertices (or points) per cell of a grid type. */
export function getGPUGridVerticesPerCell(gridType: GPUGridType): number {
  switch (gridType) {
    case 'square':
      return 4;
    case 'hex':
      return 6;
    case 'triangle':
      return 3;
    case 'point':
      return 0;
    default:
      throw new Error(`Unknown grid type ${String(gridType)}`);
  }
}

/** Number of cells (polygons or points) a grid of `columns x rows` holds. */
export function getGPUGridCellCount(gridType: GPUGridType, columns: number, rows: number): number {
  return columns * rows * (gridType === 'triangle' ? 2 : 1);
}

/** Caller-owned outputs of {@link GPUGridGenerator}. */
export type GPUGridGeneratorOutput = {
  /**
   * Polygon vertices, `cellCount * verticesPerCell` rows, counter-clockwise, cell `c` owning rows
   * `[c * verticesPerCell, (c + 1) * verticesPerCell)`. Required except for `'point'` grids.
   */
  positions?: GraphDataView<'float32x2'>;
  /** Cell centers (vertex mean for triangles), `cellCount` rows. Required for `'point'` grids. */
  centers?: GraphDataView<'float32x2'>;
};

/**
 * Properties for {@link GPUGridGenerator}.
 *
 * Per-frame (no recompile): the contents of `parameters` (origin and cell size). Compile-time:
 * `gridType`, `columns`, `rows` and which outputs are present.
 */
export type GPUGridGeneratorProps = {
  /** Prefix for generated node IDs. Defaults to `'grid-generator'`. */
  id?: string;
  /** Kind of grid. */
  gridType: GPUGridType;
  /** Cells per row (compile time). */
  columns: number;
  /** Number of rows (compile time). */
  rows: number;
  /** Per-frame packed float32 view written with {@link getGPUGridGeneratorParameterValues}. */
  parameters: GraphDataView<'float32'>;
  /** Output geometry. */
  output: GPUGridGeneratorOutput;
};

/**
 * Generates square, hexagon, triangle and point grids from an origin and a cell size (turf
 * `squareGrid`, `hexGrid`, `triangleGrid`, `pointGrid`; QGIS "Create grid"), the "fishnet then
 * join" half of areal aggregation.
 *
 * The cell count is fixed at compile time, so output sizes are known up front and there is no
 * overflow. The grid origin and cell size are read per frame, so the same compiled graph can pan
 * and zoom a view-filling grid; choose `columns` and `rows` for the largest view you will cover.
 * One invocation per cell; no atomics; deterministic and bitwise reproducible. The generator does
 * not clip to an extent: hexagon and triangle grids have ragged right edges (odd rows are shifted
 * by half a cell), and the grid covers `columns * cellWidth` by roughly `rows * rowPitch`.
 *
 * Layouts (`w` = cellWidth, `h` = cellHeight, row `r`, column `c`):
 * - square: cell `(c, r)` spans `[minX + c * w, minX + (c + 1) * w] x [minY + r * h, minY + (r + 1) * h]`.
 * - hex: pointy-top, circumradius `w / sqrt(3)`, row pitch `1.5 * circumradius`, odd rows shifted
 *   by `w / 2`.
 * - triangle: strip `r` covers `[minY + r * h, minY + (r + 1) * h]`; triangle `2c` points up and
 *   `2c + 1` points down; odd rows are shifted by `w / 2` so the lattice has no T-junctions.
 * - point: cell centers of the square grid.
 *
 * Precision: f32 from the f32 origin and size; coordinates far from the origin lose absolute
 * precision like any f32 positions (use tile-local coordinates for fine grids).
 */
export class GPUGridGenerator implements GPUCommandNodeProducer {
  /** Prefix for every node ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUGridGeneratorProps;
  /** Number of output cells. */
  readonly cellCount: number;
  /** Vertices per polygon cell (0 for points). */
  readonly verticesPerCell: number;

  constructor(props: GPUGridGeneratorProps) {
    this.id = props.id ?? 'grid-generator';
    this.props = props;
    const {id} = this;
    this.verticesPerCell = getGPUGridVerticesPerCell(props.gridType);
    for (const [name, view] of Object.entries({
      parameters: props.parameters,
      ...props.output
    })) {
      if ((view as unknown) instanceof GraphVectorView) {
        throw new Error(`${id} ${name} must be a single packed view, not a chunked vector`);
      }
    }
    for (const [name, value] of [
      ['columns', props.columns],
      ['rows', props.rows]
    ] as const) {
      if (!Number.isSafeInteger(value) || value < 1) {
        throw new Error(`${id} ${name} must be a positive integer`);
      }
    }
    this.cellCount = getGPUGridCellCount(props.gridType, props.columns, props.rows);
    if (this.cellCount * Math.max(this.verticesPerCell, 1) > 0x7fffffff) {
      throw new Error(`${id} grid is too large`);
    }
    validatePackedView(props.parameters, ['float32'], `${id} parameters`);
    if (props.parameters.length < GPU_GRID_GENERATOR_PARAMETER_LENGTH) {
      throw new Error(
        `${id} parameters must hold ${GPU_GRID_GENERATOR_PARAMETER_LENGTH} float32 values`
      );
    }
    const {positions, centers} = props.output;
    if (props.gridType === 'point') {
      if (!centers) {
        throw new Error(`${id} point grids need output.centers`);
      }
      if (positions) {
        throw new Error(`${id} point grids have no output.positions`);
      }
    } else if (!positions) {
      throw new Error(`${id} ${props.gridType} grids need output.positions`);
    }
    if (positions) {
      validatePackedView(positions, ['float32x2'], `${id} output.positions`);
      if (positions.length !== this.cellCount * this.verticesPerCell) {
        throw new Error(
          `${id} output.positions must hold ${this.cellCount * this.verticesPerCell} rows`
        );
      }
    }
    if (centers) {
      validatePackedView(centers, ['float32x2'], `${id} output.centers`);
      if (centers.length !== this.cellCount) {
        throw new Error(`${id} output.centers must hold ${this.cellCount} rows`);
      }
    }
    validateGraphOutputsDisjointFromInputs(id, [positions, centers], [props.parameters]);
  }

  /** Returns the single generation node. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    const {positions, centers} = props.output;
    validateGraphViewsBelongToGraph(id, graph, [props.parameters, positions, centers]);
    const bindings: WGSLKernelBinding[] = [
      {name: 'parameters', view: props.parameters, type: 'f32', access: 'read'}
    ];
    if (positions) {
      bindings.push({name: 'positionsOut', view: positions, type: 'f32', access: 'read_write'});
    }
    if (centers) {
      bindings.push({name: 'centersOut', view: centers, type: 'f32', access: 'read_write'});
    }
    return [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-generate`,
        operation: OPERATION,
        variant: props.gridType,
        bindings,
        invocationCount: this.cellCount,
        declarations: `const COLUMNS: u32 = ${props.columns}u;
const SQRT3: f32 = 1.7320508;
const CELLS_PER_ROW: u32 = ${props.gridType === 'triangle' ? 2 * props.columns : props.columns}u;

${positions ? 'fn writeVertex(cell: u32, slot: u32, p: vec2<f32>) {\n  let base = positionsOutOffset + 2u * (cell * ' + this.verticesPerCell + 'u + slot);\n  positionsOut[base] = p.x;\n  positionsOut[base + 1u] = p.y;\n}' : ''}`,
        body: `let origin = vec2<f32>(parameters[parametersOffset], parameters[parametersOffset + 1u]);
  let w = parameters[parametersOffset + 2u];
  let h = parameters[parametersOffset + 3u];
  let row = index / CELLS_PER_ROW;
  let column = index % CELLS_PER_ROW;
  let shift = select(0.0, 0.5 * w, (row & 1u) == 1u);
  var center = vec2<f32>(0.0);
${getGridBody(props.gridType, Boolean(positions))}
  ${centers ? 'centersOut[centersOutOffset + 2u * index] = center.x;\n  centersOut[centersOutOffset + 2u * index + 1u] = center.y;' : ''}`
      })
    ];
  }
}

/** WGSL statements that compute `center` and write vertices for one grid type. @internal */
function getGridBody(gridType: GPUGridType, writeVertices: boolean): string {
  const write = (source: string) => (writeVertices ? source : '');
  switch (gridType) {
    case 'square':
    case 'point': {
      const body = `let lower = origin + vec2<f32>(f32(column) * w, f32(row) * h);
  center = lower + 0.5 * vec2<f32>(w, h);`;
      return gridType === 'point'
        ? body
        : `${body}
  ${write(`writeVertex(index, 0u, lower);
  writeVertex(index, 1u, lower + vec2<f32>(w, 0.0));
  writeVertex(index, 2u, lower + vec2<f32>(w, h));
  writeVertex(index, 3u, lower + vec2<f32>(0.0, h));`)}`;
    }
    case 'hex':
      return `let circumradius = w / SQRT3;
  center = origin + vec2<f32>(0.5 * w + f32(column) * w + shift, circumradius + f32(row) * 1.5 * circumradius);
  ${write(`for (var corner = 0u; corner < 6u; corner++) {
    // Pointy top: first corner at 30 degrees, counter-clockwise.
    let angle = 0.5235988 + 1.0471976 * f32(corner);
    writeVertex(index, corner, center + circumradius * vec2<f32>(cos(angle), sin(angle)));
  }`)}`;
    case 'triangle':
      return `let pairIndex = column / 2u;
  let x0 = origin.x + f32(pairIndex) * w + shift;
  let y0 = origin.y + f32(row) * h;
  let y1 = y0 + h;
  var a: vec2<f32>;
  var b: vec2<f32>;
  var c: vec2<f32>;
  if ((column & 1u) == 0u) {
    a = vec2<f32>(x0, y0);
    b = vec2<f32>(x0 + w, y0);
    c = vec2<f32>(x0 + 0.5 * w, y1);
  } else {
    a = vec2<f32>(x0 + w, y0);
    b = vec2<f32>(x0 + 1.5 * w, y1);
    c = vec2<f32>(x0 + 0.5 * w, y1);
  }
  center = (a + b + c) / 3.0;
  ${write(`writeVertex(index, 0u, a);
  writeVertex(index, 1u, b);
  writeVertex(index, 2u, c);`)}`;
    default:
      throw new Error(`Unknown grid type ${String(gridType)}`);
  }
}
