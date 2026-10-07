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
import {EXACT_ORIENTATION_WGSL} from '../segment-intersection/exact-orientation-wgsl';

const OPERATION = 'GPUGridGenerator';

/** Number of float32 elements in a {@link GPUGridGenerator} parameter buffer. */
export const GPU_GRID_GENERATOR_PARAMETER_LENGTH = 4;

/**
 * Kind of grid produced by {@link GPUGridGenerator}.
 *
 * - `'square'`: `columns x rows` rectangles, 4 vertices each.
 * - `'hex'`: `columns x rows` hexagons, 6 vertices each; pointy-top unless
 *   {@link GPUGridGeneratorProps.hexOrientation} is `'flat'`.
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
   * Cell width. For `'hex'` it is the distance between opposite flat sides of a hexagon (the
   * horizontal width when pointy-top, the vertical height when flat-topped; GeoPandas
   * `make_grid(cell_size)`); for `'triangle'` the base of one triangle.
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
  /**
   * Optional `uint32` per cell, `cellCount` rows: 1 when the cell intersects
   * {@link GPUGridGeneratorProps.extent} (boundary contact counts), else 0. Point grids test the
   * center point. Requires `extent`.
   */
  intersects?: GraphDataView<'uint32'>;
};

/**
 * Polygon geometry that {@link GPUGridGeneratorOutput.intersects} tests cells against, the
 * `intersect=True` filter of GeoPandas `make_grid`.
 *
 * All rings are combined with the even-odd rule, so holes and multi-polygons work as long as the
 * rings do not cross. A ring may repeat its first vertex or not; it is closed implicitly.
 */
export type GPUGridGeneratorExtent = {
  /** Ring vertices, concatenated. */
  positions: GraphDataView<'float32x2'>;
  /** Ring-to-vertex offsets with `ringCount + 1` rows. */
  ringOffsets: GraphDataView<'uint32'>;
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
  /**
   * Hexagon orientation for `gridType: 'hex'`. Defaults to `'pointy'` (a corner points up, odd rows
   * shift right by half a cell). `'flat'` (GeoPandas `flat_topped=True`) puts a flat side on top,
   * stacks cells vertically at `cellWidth` pitch, lays columns out at `1.5 * circumradius`, and
   * shifts odd columns up by half a cell. Compile time.
   */
  hexOrientation?: 'pointy' | 'flat';
  /** Polygon the optional `output.intersects` mask is computed against. Compile time. */
  extent?: GPUGridGeneratorExtent;
  /** Per-frame packed float32 view written with {@link getGPUGridGeneratorParameterValues}. */
  parameters: GraphDataView<'float32'>;
  /** Output geometry. */
  output: GPUGridGeneratorOutput;
};

/**
 * Generates square, hexagon, triangle and point grids from an origin and a cell size (turf
 * `squareGrid`, `hexGrid`, `triangleGrid`, `pointGrid`; QGIS "Create grid"; GeoPandas `make_grid`),
 * the "fishnet then join" half of areal aggregation.
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
 *   by `w / 2`. With `hexOrientation: 'flat'`: column pitch `1.5 * circumradius`, row pitch `w`,
 *   odd columns shifted up by `w / 2`, first corner at angle 0.
 * - triangle: strip `r` covers `[minY + r * h, minY + (r + 1) * h]`; triangle `2c` points up and
 *   `2c + 1` points down; odd rows are shifted by `w / 2` so the lattice has no T-junctions.
 * - point: cell centers of the square grid.
 *
 * Intersect mask: with `extent` and `output.intersects`, a second kernel tests every cell against
 * the extent polygon with the exact orientation predicate (segment contact, cell corner inside the
 * extent, or extent inside the cell), one thread per cell scanning all extent edges, so the cost is
 * `cellCount * extentEdges`. It is a filter flag, not a compaction: select cells with the mask (for
 * example `GPUCompaction`) to get the `make_grid(intersect=True)` subset in grid order. Cell
 * corners are f32 (hexagon corners use `sin` and `cos`), so cells that only touch the extent within
 * f32 rounding can differ from an f64 reference.
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
    const {intersects} = props.output;
    if (
      props.hexOrientation !== undefined &&
      props.hexOrientation !== 'pointy' &&
      props.hexOrientation !== 'flat'
    ) {
      throw new Error(`${id} hexOrientation must be 'pointy' or 'flat'`);
    }
    if (props.hexOrientation === 'flat' && props.gridType !== 'hex') {
      throw new Error(`${id} hexOrientation 'flat' needs gridType 'hex'`);
    }
    const {extent} = props;
    if (Boolean(intersects) !== Boolean(extent)) {
      throw new Error(`${id} output.intersects and extent must be given together`);
    }
    if (intersects && extent) {
      validatePackedView(intersects, ['uint32'], `${id} output.intersects`);
      validatePackedView(extent.positions, ['float32x2'], `${id} extent.positions`);
      validatePackedView(extent.ringOffsets, ['uint32'], `${id} extent.ringOffsets`);
      if (intersects.length !== this.cellCount) {
        throw new Error(`${id} output.intersects must hold ${this.cellCount} rows`);
      }
      if (extent.ringOffsets.length < 2) {
        throw new Error(`${id} extent.ringOffsets needs at least one ring`);
      }
      if (props.gridType === 'point' ? !centers : !positions) {
        throw new Error(`${id} output.intersects needs the cell geometry output`);
      }
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [positions, centers, intersects],
      [props.parameters, extent?.positions, extent?.ringOffsets]
    );
  }

  /** Returns the single generation node. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    const {positions, centers} = props.output;
    const {intersects} = props.output;
    const {extent} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      props.parameters,
      positions,
      centers,
      intersects,
      extent?.positions,
      extent?.ringOffsets
    ]);
    const isFlatHex = props.gridType === 'hex' && props.hexOrientation === 'flat';
    const bindings: WGSLKernelBinding[] = [
      {name: 'parameters', view: props.parameters, type: 'f32', access: 'read'}
    ];
    if (positions) {
      bindings.push({name: 'positionsOut', view: positions, type: 'f32', access: 'read_write'});
    }
    if (centers) {
      bindings.push({name: 'centersOut', view: centers, type: 'f32', access: 'read_write'});
    }
    const nodes = [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-generate`,
        operation: OPERATION,
        variant: isFlatHex ? 'hex-flat' : props.gridType,
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
${getGridBody(props.gridType, Boolean(positions), isFlatHex)}
  ${centers ? 'centersOut[centersOutOffset + 2u * index] = center.x;\n  centersOut[centersOutOffset + 2u * index + 1u] = center.y;' : ''}`
      })
    ];
    if (intersects && extent) {
      const isPoint = props.gridType === 'point';
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-intersects`,
          operation: OPERATION,
          variant: 'intersects',
          bindings: [
            {
              name: 'cellVertices',
              view: (isPoint ? centers : positions) as GraphDataView,
              type: 'f32',
              access: 'read'
            },
            {name: 'extentPositions', view: extent.positions, type: 'f32', access: 'read'},
            {name: 'extentRingOffsets', view: extent.ringOffsets, type: 'u32', access: 'read'},
            {name: 'intersectsOut', view: intersects, type: 'u32', access: 'read_write'}
          ],
          invocationCount: this.cellCount,
          declarations: `const VERTICES_PER_CELL: u32 = ${Math.max(this.verticesPerCell, 1)}u;
const IS_POINT: bool = ${isPoint};
const RING_COUNT: u32 = ${extent.ringOffsets.length - 1}u;
const EXTENT_VERTEX_COUNT: u32 = ${extent.positions.length}u;
${EXACT_ORIENTATION_WGSL}
${INTERSECT_WGSL}`,
          body: `intersectsOut[intersectsOutOffset + index] = select(0u, 1u, cellIntersectsExtent(index));`
        })
      );
    }
    return nodes;
  }
}

/** WGSL that tests one convex counter-clockwise cell against even-odd extent rings. @internal */
const INTERSECT_WGSL = /* wgsl */ `
fn cellVertex(cell: u32, slot: u32) -> vec2f {
  let base = cellVerticesOffset + 2u * (cell * VERTICES_PER_CELL + slot);
  return vec2f(cellVertices[base], cellVertices[base + 1u]);
}
fn extentVertex(vertex: u32) -> vec2f {
  let base = extentPositionsOffset + 2u * vertex;
  return vec2f(extentPositions[base], extentPositions[base + 1u]);
}
fn ringStart(ring: u32) -> u32 { return extentRingOffsets[extentRingOffsetsOffset + ring]; }
fn ringEnd(ring: u32) -> u32 {
  return min(extentRingOffsets[extentRingOffsetsOffset + ring + 1u], EXTENT_VERTEX_COUNT);
}
// Second vertex of the edge leaving 'vertex' in its ring (the ring closes implicitly).
fn nextVertex(vertex: u32, first: u32, last: u32) -> u32 {
  return select(vertex + 1u, first, vertex + 1u >= last);
}
fn insideBox(a: vec2f, b: vec2f, c: vec2f) -> bool {
  return c.x >= min(a.x, b.x) && c.x <= max(a.x, b.x) && c.y >= min(a.y, b.y) && c.y <= max(a.y, b.y);
}
// True when the closed segments p1p2 and q1q2 share a point.
fn segmentsTouch(p1: vec2f, p2: vec2f, q1: vec2f, q2: vec2f) -> bool {
  let o1 = orientSign(p1, p2, q1);
  let o2 = orientSign(p1, p2, q2);
  let o3 = orientSign(q1, q2, p1);
  let o4 = orientSign(q1, q2, p2);
  if (o1 == 2 || o2 == 2 || o3 == 2 || o4 == 2) { return false; }
  if (o1 != o2 && o3 != o4) { return true; }
  if (o1 == 0 && insideBox(p1, p2, q1)) { return true; }
  if (o2 == 0 && insideBox(p1, p2, q2)) { return true; }
  if (o3 == 0 && insideBox(q1, q2, p1)) { return true; }
  if (o4 == 0 && insideBox(q1, q2, p2)) { return true; }
  return false;
}
// Even-odd containment of 'point' in the extent rings; boundary counts as inside.
fn extentContains(point: vec2f) -> bool {
  var isInside = false;
  for (var ring = 0u; ring < RING_COUNT; ring++) {
    let first = ringStart(ring);
    let last = ringEnd(ring);
    for (var vertex = first; vertex < last; vertex++) {
      let a = extentVertex(vertex);
      let b = extentVertex(nextVertex(vertex, first, last));
      let side = orientSign(a, b, point);
      if (side == 2) { continue; }
      if (side == 0 && insideBox(a, b, point)) { return true; }
      if (a.y <= point.y && b.y > point.y && side > 0) { isInside = !isInside; }
      if (b.y <= point.y && a.y > point.y && side < 0) { isInside = !isInside; }
    }
  }
  return isInside;
}
// Convex counter-clockwise cell contains 'point' (boundary included).
fn cellContains(cell: u32, point: vec2f) -> bool {
  for (var slot = 0u; slot < VERTICES_PER_CELL; slot++) {
    let side = orientSign(cellVertex(cell, slot), cellVertex(cell, (slot + 1u) % VERTICES_PER_CELL), point);
    if (side < 0 || side == 2) { return false; }
  }
  return true;
}
fn cellIntersectsExtent(cell: u32) -> bool {
  if (IS_POINT) { return extentContains(cellVertex(cell, 0u)); }
  var low = cellVertex(cell, 0u);
  var high = low;
  for (var slot = 1u; slot < VERTICES_PER_CELL; slot++) {
    let corner = cellVertex(cell, slot);
    low = min(low, corner);
    high = max(high, corner);
  }
  // Contact test: an extent edge whose box misses the cell box cannot touch the cell, so only the
  // few edges near the cell pay for exact predicates.
  for (var ring = 0u; ring < RING_COUNT; ring++) {
    let first = ringStart(ring);
    let last = ringEnd(ring);
    for (var vertex = first; vertex < last; vertex++) {
      let q1 = extentVertex(vertex);
      let q2 = extentVertex(nextVertex(vertex, first, last));
      if (max(q1.x, q2.x) < low.x || min(q1.x, q2.x) > high.x ||
          max(q1.y, q2.y) < low.y || min(q1.y, q2.y) > high.y) {
        continue;
      }
      for (var slot = 0u; slot < VERTICES_PER_CELL; slot++) {
        if (segmentsTouch(cellVertex(cell, slot), cellVertex(cell, (slot + 1u) % VERTICES_PER_CELL), q1, q2)) {
          return true;
        }
      }
    }
  }
  // No contact, so the cell outline lies wholly inside or wholly outside the extent (it never
  // crosses the boundary): one corner decides, instead of one containment scan per corner.
  if (extentContains(cellVertex(cell, 0u))) { return true; }
  // No contact: a ring is either wholly inside the cell or wholly outside it.
  for (var ring = 0u; ring < RING_COUNT; ring++) {
    if (ringEnd(ring) > ringStart(ring) && cellContains(cell, extentVertex(ringStart(ring)))) {
      return true;
    }
  }
  return false;
}`;

/** WGSL statements that compute `center` and write vertices for one grid type. @internal */
function getGridBody(gridType: GPUGridType, writeVertices: boolean, isFlatHex: boolean): string {
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
      if (isFlatHex) {
        return `let circumradius = w / SQRT3;
  center = origin + vec2<f32>(circumradius + f32(column) * 1.5 * circumradius,
    0.5 * w + f32(row) * w + select(0.0, 0.5 * w, (column & 1u) == 1u));
  ${write(`for (var corner = 0u; corner < 6u; corner++) {
    // Flat top: first corner at angle 0, counter-clockwise.
    let angle = 1.0471976 * f32(corner);
    writeVertex(index, corner, center + circumradius * vec2<f32>(cos(angle), sin(angle)));
  }`)}`;
      }
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
