// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GraphVectorView,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {dggs} from '@luma.gl/shadertools';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {CELL_KEY_WGSL} from '../cell-aggregation/cell-keys';
import {H3_BOUNDARY_WGSL} from './h3-boundary-wgsl';
import type {GPUCellWordOrder} from '../cell-aggregation';
import type {GPUCellIndexFamily} from './cell-index-families';

const OPERATION = 'GPUCellGeometry';

/** Families {@link GPUCellGeometry} decodes: the indexing families plus A5. */
export type GPUCellGeometryFamily = GPUCellIndexFamily | 'a5';

/**
 * Boundary vertices each family produces: 4 for the rectangular families (Quadbin, quadkey,
 * geohash, S2), 6 for H3 hexagons and 5 for A5 pentagons. H3 cells that cross an icosahedron edge
 * and pentagons have up to 10 vertices: use `maximumVertexCount: 10` for H3 to keep them all
 * (see {@link GPU_CELL_GEOMETRY_H3_MAXIMUM_VERTEX_COUNT}); the 6 here is the hexagon minimum.
 */
export const GPU_CELL_GEOMETRY_VERTEX_COUNTS: Readonly<Record<GPUCellGeometryFamily, number>> = {
  quadbin: 4,
  h3: 6,
  quadkey: 4,
  geohash: 4,
  s2: 4,
  a5: 5
};

/** Most boundary vertices an H3 cell has (distortion vertices included): use as `maximumVertexCount`. */
export const GPU_CELL_GEOMETRY_H3_MAXIMUM_VERTEX_COUNT = 10;

/** Largest accepted `maximumVertexCount`. */
const MAXIMUM_VERTEX_COUNT_LIMIT = 16;

/** Caller-owned outputs of {@link GPUCellGeometry}, all row-aligned with the input cells. */
export type GPUCellGeometryOutput = {
  /**
   * One longitude/latitude pair per cell. Invalid, zero and unsupported cells get `NaN` in both
   * components.
   */
  centers?: GraphDataView<'float32x2'>;
  /**
   * `cells.length * maximumVertexCount` longitude/latitude pairs; cell `row` owns pairs
   * `[row * maximumVertexCount, (row + 1) * maximumVertexCount)`. Pairs at and beyond the cell's
   * vertex count are zero.
   */
  boundaries?: GraphDataView<'float32x2'>;
  /**
   * One vertex count per cell: `GPU_CELL_GEOMETRY_VERTEX_COUNTS[family]`, or `0` for invalid
   * cells. H3 counts follow h3-js `cellToBoundary` (6, 5 for Class II pentagons, up to 10 with
   * distortion vertices) and are clamped to `maximumVertexCount`. Requires `boundaries`.
   */
  vertexCounts?: GraphDataView<'uint32'>;
};

/**
 * Properties for {@link GPUCellGeometry}.
 *
 * Per-frame (no recompile): the contents of `cells`. Topology (needs a new graph): `family`,
 * `wordOrder`, `maximumVertexCount`, the row count and which outputs are present.
 */
export type GPUCellGeometryProps = {
  /** Prefix for generated node IDs. Defaults to `'cell-geometry'`. */
  id?: string;
  /** Family of the input keys. */
  family: GPUCellGeometryFamily;
  /** One 64-bit key per row as two `uint32` words. */
  cells: GraphDataView<'uint32x2'>;
  /** Word order of `cells`. Defaults to `'little-endian'` (`(low, high)`, Arrow layout). */
  wordOrder?: GPUCellWordOrder;
  /**
   * Fixed boundary stride per cell, at least `GPU_CELL_GEOMETRY_VERTEX_COUNTS[family]` and at most
   * 16. Defaults to the family's vertex count. Required to be consistent with `output.boundaries`.
   * H3 cells have up to 10 vertices; a smaller stride truncates them (the count is clamped).
   */
  maximumVertexCount?: number;
  /** Caller-owned outputs; at least one of `centers` and `boundaries`. */
  output: GPUCellGeometryOutput;
};

/**
 * Decodes 64-bit cell keys to centers and boundary polygons in longitude/latitude degrees.
 *
 * One compute kernel per encoding reads each key (little-endian `(low, high)` words unless
 * `wordOrder` says otherwise) and writes row-aligned `centers`, fixed-stride `boundaries` and
 * `vertexCounts`. It reuses the `@luma.gl/shadertools` `dggs` decoders for H3
 * (`dggs_h3_get_center_lnglat`, boundary vertices), S2, geohash, quadkey and A5, plus a small
 * Quadbin bit-layout decoder. Nothing here compiles, submits or reads back.
 *
 * Conventions:
 * - Rectangular families list the vertices clockwise from the north-west corner (Quadbin, quadkey
 *   and geohash: NW, NE, SE, SW) or as `(i, j), (i, j+1), (i+1, j+1), (i+1, j)` face corners (S2).
 *   Quadbin and quadkey centers are the Web Mercator tile center (the middle of the tile in
 *   projected space, not the mean latitude); geohash centers are the middle of the bounds; S2
 *   centers are the face `(i + 0.5, j + 0.5)` point. Edges are straight in lng/lat for the
 *   rectangular families; S2 and H3 and A5 vertices are the exact great-circle corner positions.
 * - H3 boundaries use a WGSL port of H3's `_faceIjkToCellBoundary` and
 *   `_faceIjkPentToCellBoundary` ({@link H3_BOUNDARY_WGSL}): every valid cell gets a boundary
 *   (hexagons, pentagons, cells across icosahedron edges, with the extra edge-crossing vertices)
 *   in h3-js `cellToBoundary` order, up to 10 vertices. Set `maximumVertexCount: 10` to keep them.
 * - A5 boundaries have five vertices (three at resolution 1), in the order of the dggs decoder.
 * - Invalid keys (including the zero key): center `NaN`, vertex count 0, boundary rows zero.
 * - Precision is f32: Quadbin and quadkey coordinates lose precision above zoom 24 (the tile
 *   column itself exceeds the f32 mantissa), H3, S2 and A5 inherit the accuracy of WGSL `sin`,
 *   `cos` and `atan2`, which the WGSL spec leaves implementation defined.
 *
 * Inputs must be single packed views. Outputs must not share buffers with `cells`.
 */
export class GPUCellGeometry implements GPUCommandNodeProducer {
  /** Prefix for every node ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUCellGeometryProps;
  /** Boundary stride per cell. */
  readonly maximumVertexCount: number;

  constructor(props: GPUCellGeometryProps) {
    this.id = props.id ?? 'cell-geometry';
    this.props = props;
    const id = this.id;
    if (!(props.family in GPU_CELL_GEOMETRY_VERTEX_COUNTS)) {
      throw new Error(
        `${id} family must be one of ${Object.keys(GPU_CELL_GEOMETRY_VERTEX_COUNTS).join(', ')}`
      );
    }
    if (props.wordOrder && props.wordOrder !== 'little-endian' && props.wordOrder !== 'high-low') {
      throw new Error(`${id} wordOrder must be 'little-endian' or 'high-low'`);
    }
    const required = GPU_CELL_GEOMETRY_VERTEX_COUNTS[props.family];
    this.maximumVertexCount = props.maximumVertexCount ?? required;
    if (
      !Number.isInteger(this.maximumVertexCount) ||
      this.maximumVertexCount < required ||
      this.maximumVertexCount > MAXIMUM_VERTEX_COUNT_LIMIT
    ) {
      throw new Error(
        `${id} maximumVertexCount must be an integer in [${required}, ${MAXIMUM_VERTEX_COUNT_LIMIT}] for ${props.family}`
      );
    }
    const {output} = props;
    for (const [name, view] of [
      ['cells', props.cells],
      ['output.centers', output.centers],
      ['output.boundaries', output.boundaries],
      ['output.vertexCounts', output.vertexCounts]
    ] as const) {
      if ((view as unknown) instanceof GraphVectorView) {
        throw new Error(`${id} ${name} must be a single packed view, not a chunked vector`);
      }
    }
    validatePackedView(props.cells, ['uint32x2'], `${id} cells`);
    const rows = props.cells.length;
    if (rows < 1) {
      throw new Error(`${id} needs at least one row`);
    }
    if (output.vertexCounts && !output.boundaries) {
      throw new Error(`${id} output.vertexCounts needs output.boundaries`);
    }
    if (!output.centers && !output.boundaries) {
      throw new Error(`${id} needs output.centers or output.boundaries`);
    }
    if (output.centers) {
      validatePackedView(output.centers, ['float32x2'], `${id} output.centers`);
      if (output.centers.length !== rows) {
        throw new Error(`${id} output.centers length must equal the row count`);
      }
    }
    if (output.boundaries) {
      validatePackedView(output.boundaries, ['float32x2'], `${id} output.boundaries`);
      if (output.boundaries.length !== rows * this.maximumVertexCount) {
        throw new Error(
          `${id} output.boundaries length must equal rows * maximumVertexCount (${rows * this.maximumVertexCount})`
        );
      }
    }
    if (output.vertexCounts) {
      validatePackedUint32View(output.vertexCounts, `${id} output.vertexCounts`);
      if (output.vertexCounts.length !== rows) {
        throw new Error(`${id} output.vertexCounts length must equal the row count`);
      }
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [output.centers, output.boundaries, output.vertexCounts],
      [props.cells]
    );
  }

  /** Returns the single decode kernel node. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, maximumVertexCount} = this;
    const {family, output} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      props.cells,
      output.centers,
      output.boundaries,
      output.vertexCounts
    ]);
    const bindings: WGSLKernelBinding[] = [
      {name: 'cells', view: props.cells, type: 'u32', access: 'read'}
    ];
    if (output.centers) {
      bindings.push({name: 'centersOut', view: output.centers, type: 'f32', access: 'read_write'});
    }
    if (output.boundaries) {
      bindings.push({
        name: 'boundariesOut',
        view: output.boundaries,
        type: 'f32',
        access: 'read_write'
      });
    }
    if (output.vertexCounts) {
      bindings.push({
        name: 'vertexCountsOut',
        view: output.vertexCounts,
        type: 'u32',
        access: 'read_write'
      });
    }
    const centerBody = output.centers
      ? `let center = select(vec2f(getNotANumber(index)), geometryGetCenter(key), isValid);
  centersOut[centersOutOffset + 2u * index] = center.x;
  centersOut[centersOutOffset + 2u * index + 1u] = center.y;`
      : '';
    const boundaryBody = output.boundaries
      ? `var points: array<vec2f, 16>;
  var vertexCount = 0u;
  if (isValid) {
    vertexCount = geometryFillBoundary(key, &points);
  }
  vertexCount = min(vertexCount, MAXIMUM_VERTEX_COUNT);
  for (var vertex = 0u; vertex < MAXIMUM_VERTEX_COUNT; vertex++) {
    let point = points[vertex];
    let slot = (index * MAXIMUM_VERTEX_COUNT + vertex) * 2u;
    boundariesOut[boundariesOutOffset + slot] = point.x;
    boundariesOut[boundariesOutOffset + slot + 1u] = point.y;
  }
  ${output.vertexCounts ? 'vertexCountsOut[vertexCountsOutOffset + index] = vertexCount;' : ''}`
      : '';
    return [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-geometry`,
        operation: OPERATION,
        variant: family,
        bindings,
        invocationCount: props.cells.length,
        declarations: `const MAXIMUM_VERTEX_COUNT: u32 = ${maximumVertexCount}u;
// WGSL rejects constant-expression NaN, so the payload depends on a runtime value.
fn getNotANumber(runtimeZero: u32) -> f32 {
  return bitcast<f32>(0x7fc00000u | (runtimeZero & 0u));
}
${dggs.source}
${CELL_KEY_WGSL}
${CELL_GEOMETRY_COMMON_WGSL}
${FAMILY_GEOMETRY_WGSL[family]}
${family === 'h3' ? '' : GENERIC_FILL_BOUNDARY_WGSL}`,
        body: `let words = vec2u(cells[cellsOffset + 2u * index], cells[cellsOffset + 2u * index + 1u]);
  let key = ${props.wordOrder === 'high-low' ? 'words' : 'words.yx'};
  let isValid = geometryIsValid(key);
  ${centerBody}
  ${boundaryBody}`
      })
    ];
  }
}

/** Web Mercator tile helpers shared by the Quadbin and quadkey decoders. */
const CELL_GEOMETRY_COMMON_WGSL = /* wgsl */ `
fn geometryGetTileBounds(tile: vec3u) -> vec4f {
  let tileScale = f32(1u << tile.z);
  let west = f32(tile.x) / tileScale * 360.0 - 180.0;
  let east = f32(tile.x + 1u) / tileScale * 360.0 - 180.0;
  let north = dggs_web_mercator_tile_y_to_latitude(f32(tile.y), tileScale);
  let south = dggs_web_mercator_tile_y_to_latitude(f32(tile.y + 1u), tileScale);
  return vec4f(west, south, east, north);
}

fn geometryGetTileCenter(tile: vec3u) -> vec2f {
  let tileScale = f32(1u << tile.z);
  return vec2f(
    (f32(tile.x) + 0.5) / tileScale * 360.0 - 180.0,
    dggs_web_mercator_tile_y_to_latitude(f32(tile.y) + 0.5, tileScale)
  );
}

fn geometryGetRectangleVertex(bounds: vec4f, vertex: u32) -> vec2f {
  return dggs_bounds_get_boundary_point(bounds, vertex);
}

/** Even bits of a 32-bit value packed into 16 bits (inverse of the Morton spread). */
fn geometryCompactEvenBits(value: u32) -> u32 {
  var x = value & 0x55555555u;
  x = (x | (x >> 1u)) & 0x33333333u;
  x = (x | (x >> 2u)) & 0x0f0f0f0fu;
  x = (x | (x >> 4u)) & 0x00ff00ffu;
  x = (x | (x >> 8u)) & 0x0000ffffu;
  return x;
}
`;

const QUADBIN_GEOMETRY_WGSL = /* wgsl */ `
fn geometryGetQuadbinTile(key: vec2u) -> vec3u {
  let resolution = (key.x >> 20u) & 0x1fu;
  let morton = cellGetCompactKey(key, 52u - 2u * resolution, 2u * resolution);
  let x = geometryCompactEvenBits(morton.y) | (geometryCompactEvenBits(morton.x) << 16u);
  let y = geometryCompactEvenBits(morton.y >> 1u) | (geometryCompactEvenBits(morton.x >> 1u) << 16u);
  return vec3u(x, y, resolution);
}
fn geometryIsValid(key: vec2u) -> bool { return cellIsValidQuadbin(key); }
fn geometryGetCenter(key: vec2u) -> vec2f { return geometryGetTileCenter(geometryGetQuadbinTile(key)); }
fn geometryGetVertexCount(key: vec2u) -> u32 { return 4u; }
fn geometryGetVertex(key: vec2u, vertex: u32) -> vec2f {
  return geometryGetRectangleVertex(geometryGetTileBounds(geometryGetQuadbinTile(key)), vertex);
}
`;

const QUADKEY_GEOMETRY_WGSL = /* wgsl */ `
fn geometryIsValid(key: vec2u) -> bool {
  let length = key.x >> 26u;
  // Digits fill the low 2 * length bits; every bit between them and the length field is zero.
  let unused = cellMaskLow(58u) & ~cellMaskLow(2u * length);
  return length >= 1u && length <= DGGS_QUADKEY_MAX_LENGTH && all((key & unused) == vec2u(0u));
}
fn geometryGetCenter(key: vec2u) -> vec2f { return geometryGetTileCenter(dggs_quadkey_get_tile(key)); }
fn geometryGetVertexCount(key: vec2u) -> u32 { return 4u; }
fn geometryGetVertex(key: vec2u, vertex: u32) -> vec2f {
  return geometryGetRectangleVertex(dggs_quadkey_get_bounds(key), vertex);
}
`;

const GEOHASH_GEOMETRY_WGSL = /* wgsl */ `
fn geometryIsValid(key: vec2u) -> bool {
  let length = key.x >> 28u;
  let unused = cellMaskLow(60u) & ~cellMaskLow(5u * length);
  return length >= 1u && length <= DGGS_GEOHASH_MAX_LENGTH && all((key & unused) == vec2u(0u));
}
fn geometryGetCenter(key: vec2u) -> vec2f {
  let bounds = dggs_geohash_get_bounds(key);
  return vec2f(0.5 * (bounds.x + bounds.z), 0.5 * (bounds.y + bounds.w));
}
fn geometryGetVertexCount(key: vec2u) -> u32 { return 4u; }
fn geometryGetVertex(key: vec2u, vertex: u32) -> vec2f {
  return dggs_geohash_get_boundary_point(key, vertex);
}
`;

const S2_GEOMETRY_WGSL = /* wgsl */ `
// dggs_s2_xyz_to_lnglat calls atan2 with a zero argument at the S2 face centers: atan2(0, 0) at the
// poles (faces 2 and 5) is undefined in WGSL, and on Metal atan2(1, 0) returned -pi/2 for the face
// 1 center (x = -u = -0). Exact zeros are therefore handled without atan2.
fn geometryS2XyzToLngLat(xyz: vec3f) -> vec2f {
  let horizontal = length(xyz.xy);
  var longitude = 0.0;
  if (xyz.x == 0.0) {
    longitude = select(0.0, select(-90.0, 90.0, xyz.y > 0.0), xyz.y != 0.0);
  } else if (xyz.y == 0.0) {
    longitude = select(0.0, 180.0, xyz.x < 0.0);
  } else {
    longitude = atan2(xyz.y, xyz.x) * DGGS_RADIANS_TO_DEGREES;
  }
  return vec2f(longitude, atan2(xyz.z, horizontal) * DGGS_RADIANS_TO_DEGREES);
}
fn geometryS2GetPoint(key: vec2u, offset: vec2f) -> vec2f {
  let st = dggs_s2_ij_to_st(dggs_s2_get_ij(key), dggs_s2_get_level(key), offset);
  return geometryS2XyzToLngLat(dggs_s2_face_uv_to_xyz(dggs_s2_get_face(key), dggs_s2_st_to_uv(st)));
}
fn geometryIsValid(key: vec2u) -> bool { return dggs_s2_is_valid_cell_id(key); }
fn geometryGetCenter(key: vec2u) -> vec2f { return geometryS2GetPoint(key, vec2f(0.5)); }
fn geometryGetVertexCount(key: vec2u) -> u32 { return 4u; }
fn geometryGetVertex(key: vec2u, vertex: u32) -> vec2f {
  return geometryS2GetPoint(key, dggs_s2_get_boundary_offset(vertex));
}
`;

const H3_GEOMETRY_WGSL = /* wgsl */ `
${H3_BOUNDARY_WGSL}
fn geometryIsValid(key: vec2u) -> bool { return dggs_h3_is_valid_cell_id(key); }
fn geometryGetCenter(key: vec2u) -> vec2f { return dggs_h3_get_center_lnglat(key); }
fn geometryFillBoundary(key: vec2u, points: ptr<function, array<vec2f, 16>>) -> u32 {
  let boundary = cellIndexH3GetBoundary(key);
  for (var vertex = 0u; vertex < boundary.count; vertex++) {
    (*points)[vertex] = boundary.points[vertex];
  }
  return boundary.count;
}
`;

/** Fills boundary points one vertex at a time for the families with a closed-form vertex function. */
const GENERIC_FILL_BOUNDARY_WGSL = /* wgsl */ `
fn geometryFillBoundary(key: vec2u, points: ptr<function, array<vec2f, 16>>) -> u32 {
  let count = min(geometryGetVertexCount(key), 16u);
  for (var vertex = 0u; vertex < count; vertex++) {
    (*points)[vertex] = geometryGetVertex(key, vertex);
  }
  return count;
}
`;

const A5_GEOMETRY_WGSL = /* wgsl */ `
fn geometryIsValid(key: vec2u) -> bool {
  return !dggs_u64_is_zero(key) && dggs_a5_deserialize(key).valid != 0u;
}
fn geometryGetCenter(key: vec2u) -> vec2f { return dggs_a5_get_center_lnglat(key); }
fn geometryGetVertexCount(key: vec2u) -> u32 {
  return dggs_a5_get_shape_vertex_count(dggs_a5_deserialize(key));
}
fn geometryGetVertex(key: vec2u, vertex: u32) -> vec2f {
  return dggs_a5_get_boundary_point(key, vertex);
}
`;

const FAMILY_GEOMETRY_WGSL: Record<GPUCellGeometryFamily, string> = {
  quadbin: QUADBIN_GEOMETRY_WGSL,
  quadkey: QUADKEY_GEOMETRY_WGSL,
  geohash: GEOHASH_GEOMETRY_WGSL,
  s2: S2_GEOMETRY_WGSL,
  h3: H3_GEOMETRY_WGSL,
  a5: A5_GEOMETRY_WGSL
};
