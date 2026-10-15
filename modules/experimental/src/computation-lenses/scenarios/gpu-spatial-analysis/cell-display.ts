// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {
  createTransientView,
  GPUScan,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {
  GPUCellAggregation,
  GPUCellGeometry,
  GPUCellPyramid,
  GPUCellTopology,
  GPUPointToCell,
  type GPUCellGeometryFamily,
  type GPUCellTable
} from '@luma.gl/experimental/gpu-spatial-analysis';

import type {S3Uploads} from './s3-lens-run';

/** Boundary vertices drawn per H3 hexagon; the display stride for H3 cells. */
export const CELL_DISPLAY_H3_VERTEX_COUNT = 6;
/** Boundary vertices drawn per Quadbin tile. */
export const CELL_DISPLAY_QUADBIN_VERTEX_COUNT = 4;
/** Spare table rows beyond the occupied cells, so a one-cell CPU/GPU keying difference cannot overflow. */
export const CELL_DISPLAY_TABLE_HEADROOM = 8;

/**
 * Adds a `GPUCellGeometry` node that decodes `cells` to boundary polygons purely for the card
 * figure (cell ids carry no coordinates). The stride is the family's vertex count (6 for H3
 * hexagons, 4 for Quadbin), so every row is a complete ring and no zero-filled slot is drawn; H3
 * cells that cross an icosahedron edge or are pentagons would be truncated, and the Chicago window
 * has none. Transient ids are `name` (boundaries) and `${name}Centers`; list both in `retain`.
 * Rows past the occupied cells decode the empty key to zeros, so draw only the occupied prefix
 * with display offsets.
 */
export function addCellBoundaries(
  graph: GPUCommandGraph<undefined>,
  cells: GraphDataView<'uint32x2'>,
  options: {family: GPUCellGeometryFamily; vertexCount: number; name?: string}
) {
  const {family, vertexCount, name = 'boundaries'} = options;
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const boundaries = createTransientView(
    graph,
    name,
    'float32x2',
    cells.length * vertexCount,
    usage
  );
  const centers = createTransientView(graph, `${name}Centers`, 'float32x2', cells.length, usage);
  graph.add(
    new GPUCellGeometry({
      id: `${name}-geometry`,
      family,
      cells,
      maximumVertexCount: vertexCount,
      output: {centers, boundaries}
    })
  );
  return {boundaries, centers};
}

/**
 * Like {@link addCellBoundaries}, but also derives the display offsets on the GPU from the
 * boundary vertex counts that `GPUCellGeometry` writes (an exclusive scan), so a variable-length
 * cell column draws exactly its valid rows: empty or zero keys decode to zero vertices and become
 * empty paths, and no CPU count decides what is drawn. The vertex counts live in an uploaded,
 * zero-filled buffer with one spare row so the scan also yields the final offset. Every valid cell
 * must have `vertexCount` vertices (6 for non-pentagon H3 hexagons, 4 for Quadbin), as in
 * {@link addCellBoundaries}. Transient ids are `name` (boundaries), `${name}Centers` and
 * `${name}Offsets`; list all three in `retain`.
 */
export function addCellBoundariesWithOffsets(
  graph: GPUCommandGraph<undefined>,
  uploads: S3Uploads,
  cells: GraphDataView<'uint32x2'>,
  options: {family: GPUCellGeometryFamily; vertexCount: number; name?: string}
) {
  const {family, vertexCount, name = 'boundaries'} = options;
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const rows = cells.length;
  const countSlots = uploads.view(`${name}VertexCounts`, new Uint32Array(rows + 1), 'uint32');
  const vertexCounts = graph.createDataView(countSlots.buffer, {format: 'uint32', length: rows});
  const boundaries = createTransientView(graph, name, 'float32x2', rows * vertexCount, usage);
  const centers = createTransientView(graph, `${name}Centers`, 'float32x2', rows, usage);
  graph.add(
    new GPUCellGeometry({
      id: `${name}-geometry`,
      family,
      cells,
      maximumVertexCount: vertexCount,
      output: {centers, boundaries, vertexCounts}
    })
  );
  const offsets = createTransientView(graph, `${name}Offsets`, 'uint32', rows + 1, usage);
  graph.add(new GPUScan({id: `${name}-offsets`, input: countSlots, output: offsets}));
  return {boundaries, centers, offsets};
}

/** Keeps every `step`-th lon/lat pair, so cell figures stay at a few hundred drawn cells. */
export function selectEveryNthPoint(positions: Float32Array, step: number): Float32Array {
  const count = Math.floor((positions.length / 2 + step - 1) / step);
  const selected = new Float32Array(count * 2);
  for (let index = 0; index < count; index++) {
    selected[2 * index] = positions[2 * index * step] as number;
    selected[2 * index + 1] = positions[2 * index * step + 1] as number;
  }
  return selected;
}

/**
 * Number of distinct Web Mercator tiles (Quadbin cells) at `zoom` holding at least one lon/lat
 * pair: the occupied-cell count, used to size tables and to draw only occupied rows. The GPU keys
 * with fixed-point arithmetic; the two agree except for points within about 1e-6 of a tile edge.
 */
export function countQuadbinTiles(positions: Float32Array, zoom: number): number {
  const tiles = 2 ** zoom;
  const keys = new Set<number>();
  for (let index = 0; index < positions.length; index += 2) {
    const longitude = positions[index] as number;
    const latitude = Math.min(85.051129, Math.max(-85.051129, positions[index + 1] as number));
    const column = Math.floor(((longitude + 180) / 360) * tiles);
    const mercator = Math.log(Math.tan(Math.PI / 4 + (latitude * Math.PI) / 360));
    const row = Math.floor((0.5 - mercator / (2 * Math.PI)) * tiles);
    keys.add(row * tiles + column);
  }
  return keys.size;
}

/** Creates a cell table of transient views with ids `${name}Cells`, `${name}Counts`, `${name}Count`, `${name}Overflow`. */
export function createCellTable(
  graph: GPUCommandGraph<undefined>,
  name: string,
  capacity: number
): GPUCellTable {
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  return {
    cells: createTransientView(graph, `${name}Cells`, 'uint32x2', capacity, usage),
    counts: createTransientView(graph, `${name}Counts`, 'uint32', capacity, usage),
    status: {
      count: createTransientView(graph, `${name}Count`, 'uint32', 1, usage),
      overflow: createTransientView(graph, `${name}Overflow`, 'uint32', 1, usage)
    }
  };
}

/** Adds a `GPUCellAggregation` of lon/lat places into a Quadbin table at `zoom` (setup for roll-up figures). */
export function addQuadbinTable(
  graph: GPUCommandGraph<undefined>,
  places: GraphDataView<'float32x2'>,
  zoom: number,
  capacity: number,
  name = 'source'
): GPUCellTable {
  const table = createCellTable(graph, name, capacity);
  graph.add(
    new GPUCellAggregation({
      id: `${name}-aggregation`,
      family: 'quadbin',
      resolution: zoom,
      positions: places,
      output: table
    })
  );
  return table;
}

/** Adds a `GPUCellPyramid` of lon/lat places (setup for level-selection figures); returns its tables and counts. */
export function addQuadbinPyramid(
  graph: GPUCommandGraph<undefined>,
  places: GraphDataView<'float32x2'>,
  zooms: readonly number[],
  capacities: readonly number[]
) {
  const levels = zooms.map((_, level) =>
    createCellTable(graph, `level${level}`, capacities[level] as number)
  );
  const levelCounts = createTransientView(
    graph,
    'levelCounts',
    'uint32',
    zooms.length,
    Buffer.STORAGE | Buffer.COPY_SRC
  );
  graph.add(
    new GPUCellPyramid({
      id: 'setup-pyramid',
      family: 'quadbin',
      positions: places,
      resolutions: zooms,
      output: {levels, levelCounts}
    })
  );
  return {levels, levelCounts};
}

/** Row of the lon/lat pair nearest the mean position: a deterministic "central place" of a dataset. */
export function findCentralPlaceIndex(positions: Float32Array): number {
  const count = positions.length / 2;
  let meanLongitude = 0;
  let meanLatitude = 0;
  for (let index = 0; index < count; index++) {
    meanLongitude += (positions[2 * index] as number) / count;
    meanLatitude += (positions[2 * index + 1] as number) / count;
  }
  let nearest = 0;
  let nearestDistance = Number.POSITIVE_INFINITY;
  for (let index = 0; index < count; index++) {
    const distance =
      ((positions[2 * index] as number) - meanLongitude) ** 2 +
      ((positions[2 * index + 1] as number) - meanLatitude) ** 2;
    if (distance < nearestDistance) {
      nearest = index;
      nearestDistance = distance;
    }
  }
  return nearest;
}

/** Adds a `GPUPointToCell` node keying `positions` to H3 cells (setup for cell-input figures); transient id `name`. */
export function addH3Cells(
  graph: GPUCommandGraph<undefined>,
  positions: GraphDataView<'float32x2'>,
  resolution: number,
  name: string
): GraphDataView<'uint32x2'> {
  const cells = createTransientView(
    graph,
    name,
    'uint32x2',
    positions.length,
    Buffer.STORAGE | Buffer.COPY_SRC
  );
  graph.add(
    new GPUPointToCell({
      id: `${name}-keys`,
      family: 'h3',
      resolution,
      positions,
      output: {cells}
    })
  );
  return cells;
}

/**
 * Adds a `GPUCellTopology` `children` node expanding each H3 cell at `fromResolution` to its
 * descendants at `toResolution` (setup for compaction figures). Row `i` owns
 * `[i * 7^depth, (i + 1) * 7^depth)` of the returned view, in ascending key order.
 */
export function addH3Children(
  graph: GPUCommandGraph<undefined>,
  cells: GraphDataView<'uint32x2'>,
  fromResolution: number,
  toResolution: number,
  name: string
): GraphDataView<'uint32x2'> {
  const stride = 7 ** (toResolution - fromResolution);
  const children = createTransientView(
    graph,
    name,
    'uint32x2',
    cells.length * stride,
    Buffer.STORAGE | Buffer.COPY_SRC
  );
  graph.add(
    new GPUCellTopology({
      id: `${name}-children`,
      family: 'h3',
      operation: {type: 'children', resolution: toResolution, inputResolution: fromResolution},
      cells,
      output: {cells: children}
    })
  );
  return children;
}
