// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUScan,
  GraphVectorView,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {dggs} from '@luma.gl/shadertools';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {
  createPublishNode,
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../utils/wgsl-kernel-nodes';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {CELL_KEY_WGSL} from '../cell-aggregation/cell-keys';
import {H3_BOUNDARY_WGSL} from '../cell-indexing/h3-boundary-wgsl';
import {H3_NEIGHBOR_WGSL} from '../cell-topology/h3-neighbor-wgsl';
import {QUADBIN_TOPOLOGY_WGSL} from '../cell-topology/cell-topology-wgsl';
import {GPUSegmentRingAssembly} from '../ring-assembly/index';
import type {
  GPUSegmentRingAssemblyInteriorSide,
  GPUSegmentRingAssemblyOutput
} from '../ring-assembly/index';

const OPERATION = 'GPUCellSetOutline';

/** Cell families {@link GPUCellSetOutline} supports. */
export type GPUCellSetOutlineFamily = 'h3' | 'quadbin';

/**
 * Most boundary edges one cell contributes: 10 for H3 (a hexagon has 6, a cell that crosses an
 * icosahedron edge up to 10 with distortion vertices) and 4 for Quadbin.
 */
export const GPU_CELL_SET_OUTLINE_MAXIMUM_EDGES: Readonly<Record<GPUCellSetOutlineFamily, number>> =
  {h3: 10, quadbin: 4};

/**
 * Caller-owned, capacity-bounded boundary segments of {@link GPUCellSetOutline}.
 *
 * All columns share one capacity (`rows.length`). Segments are ordered by input row, then by edge
 * index, so the order is deterministic and follows the cell order of the input.
 */
export type GPUCellSetOutlineOutput = {
  /** Input row of the cell that owns each segment. Doubles as the compact ID column. */
  rows: GraphDataView<'uint32'>;
  /** Owning cell key per segment as little-endian `(low, high)` words. */
  cells: GraphDataView<'uint32x2'>;
  /**
   * Edge index within the owning cell's boundary: the segment runs from boundary vertex `e` to
   * vertex `e + 1`. Quadbin: 0 north, 1 east, 2 south, 3 west. H3: the `e`-th edge of
   * `cellToBoundary`, so distortion vertices split an edge in two segments.
   */
  edgeIndices: GraphDataView<'uint32'>;
  /** Segment endpoints `(lng0, lat0, lng1, lat1)` in degrees. */
  endpoints: GraphDataView<'float32x4'>;
  /** Optional group label of the owning cell (when `groups` is an input). */
  groups?: GraphDataView<'uint32'>;
  /** One-row scalar receiving `min(requiredCount, capacity)`. */
  count: GraphDataView<'uint32'>;
  /** One-row scalar receiving 1 when more segments existed than the capacity. */
  overflow: GraphDataView<'uint32'>;
  /** Optional one-row scalar receiving the unclamped segment count. */
  requiredCount?: GraphDataView<'uint32'>;
};

/**
 * Properties for {@link GPUCellSetOutline}.
 *
 * Per-frame (no recompile): the contents of `cells`, `count` and `groups`. Compile-time: `family`,
 * view lengths, and which optional views exist.
 */
export type GPUCellSetOutlineProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'cell-set-outline'`. */
  id?: string;
  /** Cell family. */
  family: GPUCellSetOutlineFamily;
  /**
   * Cell keys as little-endian `(low, high)` words, sorted ascending by the 64-bit key with
   * distinct entries, for example the `cells` column of a `GPUCellAggregation` table. Membership
   * is a binary search, so unsorted input gives wrong results. Invalid keys never emit and never
   * match as neighbors.
   */
  cells: GraphDataView<'uint32x2'>;
  /** Optional one-row count of valid rows in `cells`. Defaults to every row. */
  count?: GraphDataView<'uint32'>;
  /**
   * Optional group label per row. An edge is then emitted when the neighbor is absent or has a
   * different label, which yields the region borders between groups (an edge between two groups
   * is emitted once from each side, tagged with that side's group in `output.groups`).
   */
  groups?: GraphDataView<'uint32'>;
  /** Caller-owned output. */
  output: GPUCellSetOutlineOutput;
  /**
   * Optional ring assembly of the emitted segments (`GPUSegmentRingAssembly` composed after the
   * outline): closed rings in GeoArrow-style offsets plus positions, with shell and hole
   * classification. `output.count` and `output.endpoints` (and `output.groups` when present) are the
   * assembly's inputs. The interior side is set from the family; `normalizeWinding` makes shells
   * counter-clockwise for both families.
   */
  rings?: GPUCellSetOutlineRings;
};

/** Ring assembly options of {@link GPUCellSetOutlineProps.rings}. */
export type GPUCellSetOutlineRings = {
  /** Vertex matching distance in degrees. Defaults to the assembly default (5e-5). */
  vertexTolerance?: number;
  /** Orient rings of either family: shells counter-clockwise, holes clockwise. Default false. */
  normalizeWinding?: boolean;
  /** Caller-owned ring output. */
  output: GPUSegmentRingAssemblyOutput;
};

/**
 * Boundary edges of a set of H3 or Quadbin cells, the `cellsToMultiPolygon` outline without ring
 * assembly.
 *
 * One thread per cell decodes the cell boundary, finds the neighbor across each edge, looks it up
 * in the sorted cell set with a binary search and keeps the edges whose neighbor is absent (or
 * belongs to another group). Each cell records its kept edges as a bitmask, an inclusive scan over
 * the per-cell kept counts gives every cell its output offset, and the cell's thread writes its own
 * edges, so the output has no atomics and is ordered by (row, edge index). The scan covers one entry
 * per cell, not one per candidate edge slot.
 *
 * The neighbor across an H3 edge is the neighbor (H3 neighbor stepping, exact across faces and
 * pentagons) whose center is nearest the edge midpoint (longitude differences wrapped and scaled
 * by `cos(latitude)`). Quadbin edges map to tiles directly: columns wrap across the antimeridian,
 * rows beyond the poles are absent, so the map edge is an outline edge. Boundary vertices come from
 * the same f32 decoders as `GPUCellGeometry`, so segment endpoints are f32 accurate and shared
 * endpoints of neighboring cells agree to within f32 round-off. H3 cells that straddle the
 * antimeridian keep `cellToBoundary` longitudes, so segments may span more than 180 degrees of
 * longitude there. Nothing is submitted or read back.
 *
 * Draw `output.endpoints` as line segments (`count` rows), or set `rings` to chain them into closed
 * rings (`GPUSegmentRingAssembly`).
 */
export class GPUCellSetOutline implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUCellSetOutlineProps;
  /** Candidate edge slots per cell: {@link GPU_CELL_SET_OUTLINE_MAXIMUM_EDGES} for the family. */
  readonly edgeSlots: number;
  /**
   * Side of each emitted segment that is inside the cell set: H3 boundaries run counter-clockwise
   * (`'left'`), Quadbin edges run clockwise (`'right'`).
   */
  readonly interiorSide: GPUSegmentRingAssemblyInteriorSide;

  constructor(props: GPUCellSetOutlineProps) {
    this.id = props.id ?? 'cell-set-outline';
    this.props = props;
    const id = this.id;
    if (props.family !== 'h3' && props.family !== 'quadbin') {
      throw new Error(`${id} family must be 'h3' or 'quadbin'`);
    }
    this.edgeSlots = GPU_CELL_SET_OUTLINE_MAXIMUM_EDGES[props.family];
    this.interiorSide = props.family === 'h3' ? 'left' : 'right';
    const {output} = props;
    for (const [name, view] of Object.entries({...props, ...output})) {
      if (name !== 'output' && (view as unknown) instanceof GraphVectorView) {
        throw new Error(`${id} ${name} must be a single packed view, not a chunked vector`);
      }
    }
    validatePackedView(props.cells, ['uint32x2'], `${id} cells`);
    const rows = props.cells.length;
    if (rows < 1) {
      throw new Error(`${id} needs at least one cell row`);
    }
    if (rows * this.edgeSlots >= 0xffffffff) {
      throw new Error(`${id} cells length times ${this.edgeSlots} must stay below 2^32`);
    }
    for (const [name, view] of [
      ['count', props.count],
      ['output.count', output.count],
      ['output.overflow', output.overflow],
      ['output.requiredCount', output.requiredCount]
    ] as const) {
      if (view) {
        validatePackedUint32View(view, `${id} ${name}`);
        if (view.length < 1) {
          throw new Error(`${id} ${name} must contain one uint32 row`);
        }
      }
    }
    if (props.groups) {
      validatePackedUint32View(props.groups, `${id} groups`);
      if (props.groups.length !== rows) {
        throw new Error(`${id} groups length must equal the cell row count`);
      }
    }
    validatePackedUint32View(output.rows, `${id} output.rows`);
    const capacity = output.rows.length;
    if (capacity < 1) {
      throw new Error(`${id} output capacity must be at least 1`);
    }
    validatePackedView(output.cells, ['uint32x2'], `${id} output.cells`);
    validatePackedUint32View(output.edgeIndices, `${id} output.edgeIndices`);
    validatePackedView(output.endpoints, ['float32x4'], `${id} output.endpoints`);
    for (const [name, length] of [
      ['cells', output.cells.length],
      ['edgeIndices', output.edgeIndices.length],
      ['endpoints', output.endpoints.length]
    ] as const) {
      if (length !== capacity) {
        throw new Error(`${id} output.${name} must have the same length as output.rows`);
      }
    }
    if (output.groups) {
      if (!props.groups) {
        throw new Error(`${id} output.groups needs the groups input`);
      }
      validatePackedUint32View(output.groups, `${id} output.groups`);
      if (output.groups.length !== capacity) {
        throw new Error(`${id} output.groups must have the same length as output.rows`);
      }
    }
    validateGraphOutputsDisjointFromInputs(id, getOutputViews(output), [
      props.cells,
      props.count,
      props.groups
    ]);
  }

  /** Returns the edge classification, accept scan, write, total and publish nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, edgeSlots} = this;
    const {output, family} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      props.cells,
      props.count,
      props.groups,
      ...getOutputViews(output)
    ]);
    const rows = props.cells.length;
    const slotCount = rows * edgeSlots;
    const capacity = output.rows.length;
    // One bitmask (bit `e` = edge `e` kept) and one kept-edge count per row, so the compaction scan
    // runs over `rows` entries instead of `rows * edgeSlots` (10x fewer for H3, 4x for Quadbin).
    const masks = createTransientView(graph, `${id}-masks`, 'uint32', rows);
    const flags = createTransientView(graph, `${id}-flags`, 'uint32', rows);
    const accepted = createTransientView(graph, `${id}-accepted`, 'uint32', rows);
    const candidateEndpoints = createTransientView(
      graph,
      `${id}-candidate-endpoints`,
      'float32x4',
      slotCount
    );
    const total = createTransientView(graph, `${id}-total`, 'uint32', 1);

    const classifyBindings: WGSLKernelBinding[] = [
      {name: 'cells', view: props.cells, type: 'u32', access: 'read'}
    ];
    if (props.count) {
      classifyBindings.push({name: 'rowCount', view: props.count, type: 'u32', access: 'read'});
    }
    if (props.groups) {
      classifyBindings.push({name: 'groups', view: props.groups, type: 'u32', access: 'read'});
    }
    classifyBindings.push(
      {name: 'masks', view: masks, type: 'u32', access: 'read_write'},
      {name: 'flags', view: flags, type: 'u32', access: 'read_write'},
      {name: 'candidateEndpoints', view: candidateEndpoints, type: 'f32', access: 'read_write'}
    );
    const isH3 = family === 'h3';
    const classify = createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-classify`,
      operation: OPERATION,
      variant: family,
      bindings: classifyBindings,
      invocationCount: rows,
      declarations: `const ROWS: u32 = ${rows}u;
const EDGE_SLOTS: u32 = ${edgeSlots}u;
const CELL_TOPOLOGY_K: u32 = 0u;
const CELL_TOPOLOGY_CAPACITY: u32 = 1u;
${dggs.source}
${CELL_KEY_WGSL}
${isH3 ? H3_NEIGHBOR_WGSL : ''}
${isH3 ? H3_BOUNDARY_WGSL : ''}
${QUADBIN_TOPOLOGY_WGSL}
${CLASSIFY_COMMON_WGSL.replace(
  'GROUP_TEST',
  props.groups
    ? 'return groups[groupsOffset + neighborRow] == groups[groupsOffset + row];'
    : 'return true;'
)}
${isH3 ? H3_CLASSIFY_WGSL : QUADBIN_CLASSIFY_WGSL}`,
      body: `let validRows = ${props.count ? 'min(rowCount[rowCountOffset], ROWS)' : 'ROWS'};
  masks[masksOffset + index] = 0u;
  if (index < validRows) {
    outlineClassifyRow(index, validRows);
  }
  flags[flagsOffset + index] = countOneBits(masks[masksOffset + index]);`
    });

    const writeBindings: WGSLKernelBinding[] = [
      {name: 'accepted', view: accepted, type: 'u32', access: 'read'},
      {name: 'masks', view: masks, type: 'u32', access: 'read'},
      {name: 'candidateEndpoints', view: candidateEndpoints, type: 'f32', access: 'read'},
      {name: 'cells', view: props.cells, type: 'u32', access: 'read'},
      {name: 'outRows', view: output.rows, type: 'u32', access: 'read_write'},
      {name: 'outCells', view: output.cells, type: 'u32', access: 'read_write'},
      {name: 'outEdges', view: output.edgeIndices, type: 'u32', access: 'read_write'},
      {name: 'outEndpoints', view: output.endpoints, type: 'f32', access: 'read_write'}
    ];
    const write = createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-write`,
      operation: OPERATION,
      variant: 'write',
      bindings: writeBindings,
      invocationCount: rows,
      declarations: `const EDGE_SLOTS: u32 = ${edgeSlots}u;
const OUTPUT_CAPACITY: u32 = ${capacity}u;`,
      // One thread per row writes its kept edges at consecutive slots, in edge order.
      body: `let mask = masks[masksOffset + index];
  if (mask == 0u) { return; }
  var outputSlot = 0u;
  if (index > 0u) { outputSlot = accepted[acceptedOffset + index - 1u]; }
  for (var edge = 0u; edge < EDGE_SLOTS; edge++) {
    if ((mask & (1u << edge)) == 0u) { continue; }
    if (outputSlot < OUTPUT_CAPACITY) {
      let candidate = index * EDGE_SLOTS + edge;
      outRows[outRowsOffset + outputSlot] = index;
      outEdges[outEdgesOffset + outputSlot] = edge;
      outCells[outCellsOffset + 2u * outputSlot] = cells[cellsOffset + 2u * index];
      outCells[outCellsOffset + 2u * outputSlot + 1u] = cells[cellsOffset + 2u * index + 1u];
      for (var component = 0u; component < 4u; component++) {
        outEndpoints[outEndpointsOffset + 4u * outputSlot + component] =
          candidateEndpoints[candidateEndpointsOffset + 4u * candidate + component];
      }
    }
    outputSlot++;
  }`
    });
    // The group column is written by its own pass: with it the write kernel would bind nine
    // storage buffers, above the default WebGPU limit of eight.
    const writeGroups =
      output.groups && props.groups
        ? createWGSLKernelNode<Parameters>(graph, {
            id: `${id}-write-groups`,
            operation: OPERATION,
            variant: 'write-groups',
            bindings: [
              {name: 'accepted', view: accepted, type: 'u32', access: 'read'},
              {name: 'masks', view: masks, type: 'u32', access: 'read'},
              {name: 'groups', view: props.groups, type: 'u32', access: 'read'},
              {name: 'outGroups', view: output.groups, type: 'u32', access: 'read_write'}
            ],
            invocationCount: rows,
            declarations: `const OUTPUT_CAPACITY: u32 = ${capacity}u;`,
            body: `let keptCount = countOneBits(masks[masksOffset + index]);
  if (keptCount == 0u) { return; }
  var outputSlot = 0u;
  if (index > 0u) { outputSlot = accepted[acceptedOffset + index - 1u]; }
  let group = groups[groupsOffset + index];
  for (var kept = 0u; kept < keptCount; kept++) {
    if (outputSlot + kept < OUTPUT_CAPACITY) {
      outGroups[outGroupsOffset + outputSlot + kept] = group;
    }
  }`
          })
        : null;

    const totalNode = createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-total`,
      operation: OPERATION,
      variant: 'total',
      bindings: [
        {name: 'accepted', view: accepted, type: 'u32', access: 'read'},
        {name: 'total', view: total, type: 'u32', access: 'read_write'}
      ],
      invocationCount: 1,
      declarations: `const ROW_COUNT: u32 = ${rows}u;`,
      body: 'total[totalOffset] = accepted[acceptedOffset + ROW_COUNT - 1u];'
    });

    const ringNodes = props.rings
      ? new GPUSegmentRingAssembly({
          id: `${id}-rings`,
          endpoints: output.endpoints,
          count: output.count,
          groups: output.groups,
          vertexTolerance: props.rings.vertexTolerance,
          normalizeWinding: props.rings.normalizeWinding,
          interiorSide: this.interiorSide,
          output: props.rings.output
        }).getCommandNodes(graph)
      : [];
    return [
      classify,
      ...new GPUScan({
        id: `${id}-scan`,
        input: flags,
        output: accepted,
        mode: 'inclusive'
      }).getCommandNodes(graph),
      write,
      ...(writeGroups ? [writeGroups] : []),
      totalNode,
      createPublishNode<Parameters>(graph, {
        id: `${id}-publish`,
        operation: OPERATION,
        requiredCount: total,
        output: {
          ids: output.rows,
          count: output.count,
          overflow: output.overflow,
          requiredCount: output.requiredCount
        }
      }),
      ...ringNodes
    ];
  }
}

function getOutputViews(output: GPUCellSetOutlineOutput) {
  return [
    output.rows,
    output.cells,
    output.edgeIndices,
    output.endpoints,
    output.groups,
    output.count,
    output.overflow,
    output.requiredCount
  ];
}

/** Shared helpers: sorted-set lookup and edge emission. Needs `cells`, `masks`, `candidateEndpoints`. */
const CLASSIFY_COMMON_WGSL = /* wgsl */ `
const OUTLINE_NO_ROW: u32 = 0xffffffffu;

fn outlineGetKey(row: u32) -> vec2u {
  return vec2u(cells[cellsOffset + 2u * row + 1u], cells[cellsOffset + 2u * row]);
}

fn outlineIsKeyLess(left: vec2u, right: vec2u) -> bool {
  return left.x < right.x || (left.x == right.x && left.y < right.y);
}

/** Row of \`key\` in the sorted set, or OUTLINE_NO_ROW. */
fn outlineFindRow(key: vec2u, validRows: u32) -> u32 {
  if (key.x == 0u && key.y == 0u) {
    return OUTLINE_NO_ROW;
  }
  var low = 0u;
  var high = validRows;
  while (low < high) {
    let middle = low + (high - low) / 2u;
    if (outlineIsKeyLess(outlineGetKey(middle), key)) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  if (low < validRows && all(outlineGetKey(low) == key)) {
    return low;
  }
  return OUTLINE_NO_ROW;
}

/** Whether the neighbor key is in the set with the same group as \`row\`. */
fn outlineHasSameSideNeighbor(row: u32, neighbor: vec2u, validRows: u32) -> bool {
  let neighborRow = outlineFindRow(neighbor, validRows);
  if (neighborRow == OUTLINE_NO_ROW) {
    return false;
  }
  GROUP_TEST
}

fn outlineEmit(row: u32, edge: u32, a: vec2f, b: vec2f) {
  let slot = row * EDGE_SLOTS + edge;
  masks[masksOffset + row] = masks[masksOffset + row] | (1u << edge);
  candidateEndpoints[candidateEndpointsOffset + 4u * slot] = a.x;
  candidateEndpoints[candidateEndpointsOffset + 4u * slot + 1u] = a.y;
  candidateEndpoints[candidateEndpointsOffset + 4u * slot + 2u] = b.x;
  candidateEndpoints[candidateEndpointsOffset + 4u * slot + 3u] = b.y;
}
`;

const H3_CLASSIFY_WGSL = /* wgsl */ `
fn outlineClassifyRow(row: u32, validRows: u32) {
  let key = outlineGetKey(row);
  if (!dggs_h3_is_valid_cell_id(key)) {
    return;
  }
  let boundary = cellIndexH3GetBoundary(key);
  var neighbors: array<vec2u, 6>;
  var centers: array<vec2f, 6>;
  for (var direction = 1u; direction <= 6u; direction++) {
    let neighbor = cellTopologyH3Neighbor(key, direction);
    neighbors[direction - 1u] = neighbor;
    if (neighbor.x != 0u || neighbor.y != 0u) {
      centers[direction - 1u] = dggs_h3_get_center_lnglat(neighbor);
    }
  }
  for (var edge = 0u; edge < boundary.count; edge++) {
    let a = boundary.points[edge];
    let b = boundary.points[select(edge + 1u, 0u, edge + 1u == boundary.count)];
    // Wrap the second longitude next to the first so the midpoint is on the short arc.
    let wrappedB = vec2f(a.x + (b.x - a.x) - 360.0 * round((b.x - a.x) / 360.0), b.y);
    let midpoint = 0.5 * (a + wrappedB);
    let cosine = max(cos(radians(midpoint.y)), 0.01);
    var best = 1e30;
    var bestNeighbor = vec2u(0u);
    for (var slot = 0u; slot < 6u; slot++) {
      if (neighbors[slot].x == 0u && neighbors[slot].y == 0u) {
        continue;
      }
      let delta = centers[slot] - midpoint;
      let dx = (delta.x - 360.0 * round(delta.x / 360.0)) * cosine;
      let distance = dx * dx + delta.y * delta.y;
      if (distance < best) {
        best = distance;
        bestNeighbor = neighbors[slot];
      }
    }
    if (!outlineHasSameSideNeighbor(row, bestNeighbor, validRows)) {
      outlineEmit(row, edge, a, b);
    }
  }
}
`;

const QUADBIN_CLASSIFY_WGSL = /* wgsl */ `
fn outlineTileLatitude(row: u32, tileScale: f32) -> f32 {
  return dggs_web_mercator_tile_y_to_latitude(f32(row), tileScale);
}

fn outlineClassifyRow(row: u32, validRows: u32) {
  let key = outlineGetKey(row);
  if (!cellIsValidQuadbin(key)) {
    return;
  }
  let z = (key.x >> 20u) & 0x1fu;
  let tile = cellTopologyQuadbinGetTile(key);
  let tileCount = i32(1u << z);
  let tileScale = f32(1u << z);
  let west = f32(tile.x) / tileScale * 360.0 - 180.0;
  let east = f32(tile.x + 1u) / tileScale * 360.0 - 180.0;
  let north = outlineTileLatitude(tile.y, tileScale);
  let south = outlineTileLatitude(tile.y + 1u, tileScale);
  // Vertices NW, NE, SE, SW; edge e runs vertex e to vertex e + 1. Neighbors: N, E, S, W.
  var corners = array<vec2f, 4>(
    vec2f(west, north), vec2f(east, north), vec2f(east, south), vec2f(west, south)
  );
  var offsets = array<vec2i, 4>(vec2i(0, -1), vec2i(1, 0), vec2i(0, 1), vec2i(-1, 0));
  for (var edge = 0u; edge < 4u; edge++) {
    let neighborTile = vec2i(tile) + offsets[edge];
    var neighbor = vec2u(0u);
    if (neighborTile.y >= 0 && neighborTile.y < tileCount) {
      let x = u32(((neighborTile.x % tileCount) + tileCount) % tileCount);
      neighbor = cellTopologyQuadbinGetKey(x, u32(neighborTile.y), z);
    }
    if (!outlineHasSameSideNeighbor(row, neighbor, validRows)) {
      outlineEmit(row, edge, corners[edge], corners[(edge + 1u) % 4u]);
    }
  }
}
`;
