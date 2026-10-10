// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
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
  createWGSLKernelNode,
  getWGSLFloatLiteral,
  type WGSLKernelBinding
} from '../../utils/wgsl-kernel-nodes';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import type {GPUPolygonGeometryPort} from '../contracts/index';

const OPERATION = 'GPUSegmentRingAssembly';
/** Lanes per workgroup; a workgroup owns this many rings and sums its larger rings cooperatively. */
const RING_LANES = 64;

/** Sentinel for "no ring", "no segment" and "no shell" in uint32 outputs (`0xffffffff`). */
export const GPU_SEGMENT_RING_ASSEMBLY_NONE = 0xffffffff;
/** `segmentFlags` bit: more than one segment starts at this segment's end vertex. */
export const GPU_SEGMENT_RING_ASSEMBLY_FLAG_TOUCHING = 1;
/** `segmentFlags` bit: no segment starts at this segment's end vertex (open chain). */
export const GPU_SEGMENT_RING_ASSEMBLY_FLAG_DANGLING = 2;
/** `segmentFlags` bit: the chosen continuation was already taken by a lower-index segment. */
export const GPU_SEGMENT_RING_ASSEMBLY_FLAG_CONFLICT = 4;

/**
 * `segmentFlags` bit: the segment and an exactly opposite segment (same group, end and start
 * vertices matching within the tolerance) were cancelled by `cancelOpposingSegments`; neither is on
 * a ring and neither counts as open.
 */
export const GPU_SEGMENT_RING_ASSEMBLY_FLAG_CANCELLED = 8;

/** Which side of a directed segment the filled region lies on. */
export type GPUSegmentRingAssemblyInteriorSide = 'left' | 'right';

/**
 * Polygon layout of {@link GPUSegmentRingAssemblyOutput}: rings regrouped so every polygon feature
 * is its shell followed by its holes, in GeoArrow layout with implicitly closed rings (the repeated
 * closing vertex is dropped). Offsets are GPU-written and padded flat past the data, so views of
 * fixed length stay valid for compile-time topologies: unused polygons and rings are empty. Holes
 * with no enclosing shell are left out. The ordering is deterministic: polygons by shell ring
 * index, holes by ring index.
 */
export type GPUSegmentRingPolygonOutput = Omit<GPUPolygonGeometryPort, 'sourceIds'> & {
  /** Ring vertices in polygon order, without closing duplicates. Length at most `positions.length`. */
  positions: GraphDataView<'float32x2'>;
  /** Ring-to-vertex offsets, length `ringCapacity + 1`. */
  ringOffsets: GraphDataView<'uint32'>;
  /** Polygon-to-ring offsets, length `ringCapacity + 1` (the first ring of a polygon is its shell). */
  polygonOffsets: GraphDataView<'uint32'>;
  /** Identity feature-to-polygon offsets, padded with `polygonCount`; one feature per polygon. */
  featureOffsets: GraphDataView<'uint32'>;
  /**
   * Optional stable source ID of each polygon feature (its shell's group). Unused rows hold
   * `0xffffffff` ({@link GPU_SEGMENT_RING_ASSEMBLY_NONE}). Needs the `groups` input.
   */
  sourceIds?: GraphDataView<'uint32'>;
};

/**
 * Caller-owned, capacity-bounded outputs of {@link GPUSegmentRingAssembly}.
 *
 * Ring capacity is `ringOffsets.length - 1`; vertex capacity is `positions.length`. Rings are
 * written whole or not at all: the written rings are a prefix of the full ring list, `count` is the
 * number written and `overflow` is 1 when more rings existed.
 */
export type GPUSegmentRingAssemblyOutput = {
  /**
   * Ring `r` owns `positions[ringOffsets[r] .. ringOffsets[r + 1])`, closed (the last vertex
   * repeats the first, GeoJSON style). Entries past `count` repeat the final offset. Length
   * `ringCapacity + 1`.
   */
  ringOffsets: GraphDataView<'uint32'>;
  /** Ring vertices in ring order. */
  positions: GraphDataView<'float32x2'>;
  /** Optional signed area per ring (shoelace, input units squared; counter-clockwise positive). */
  ringAreas?: GraphDataView<'float32'>;
  /** Optional 1 for rings that are holes (interior on the outside of the ring), else 0. */
  ringIsHole?: GraphDataView<'uint32'>;
  /**
   * Optional owning shell per ring: a shell's own ring index; a hole's innermost enclosing shell
   * (smallest area, lowest index on ties, same group when `groups` is set), or
   * {@link GPU_SEGMENT_RING_ASSEMBLY_NONE} when no shell contains it. Each hole tests every shell's
   * bounding box (four comparisons) and walks the vertices only of shells whose box contains its
   * probe point.
   */
  ringShells?: GraphDataView<'uint32'>;
  /** Optional group label per ring (needs the `groups` input). */
  ringGroups?: GraphDataView<'uint32'>;
  /** Optional ring index per input segment, or NONE when the segment is in no written ring. */
  segmentRings?: GraphDataView<'uint32'>;
  /** Optional index into `positions` of each segment's start vertex, or NONE. */
  segmentVertices?: GraphDataView<'uint32'>;
  /** Optional `GPU_SEGMENT_RING_ASSEMBLY_FLAG_*` bits per input segment. */
  segmentFlags?: GraphDataView<'uint32'>;
  /**
   * Optional polygon layout of the written rings for consumers that need GeoArrow polygon
   * topology, such as `GPUPointInPolygonJoin`.
   */
  polygons?: GPUSegmentRingPolygonOutput;
  /** One-row scalar receiving the number of rings written. */
  count: GraphDataView<'uint32'>;
  /** One-row scalar receiving 1 when ring or vertex capacity dropped rings. */
  overflow: GraphDataView<'uint32'>;
  /** Optional one-row scalar receiving the unclamped ring count. */
  requiredCount?: GraphDataView<'uint32'>;
  /** Optional one-row scalar receiving the number of valid segments that lie on no closed ring. */
  openSegmentCount?: GraphDataView<'uint32'>;
  /** Optional one-row scalar receiving the number of segments whose end vertex is shared. */
  touchingSegmentCount?: GraphDataView<'uint32'>;
};

/** Properties for {@link GPUSegmentRingAssembly}. */
export type GPUSegmentRingAssemblyProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'segment-ring-assembly'`. */
  id?: string;
  /** Directed segments `(x0, y0, x1, y1)`, for example `GPUCellSetOutline` `output.endpoints`. */
  endpoints: GraphDataView<'float32x4'>;
  /** Optional one-row count of valid segment rows. Defaults to every row. */
  count?: GraphDataView<'uint32'>;
  /** Optional group label per segment; a ring never mixes groups. */
  groups?: GraphDataView<'uint32'>;
  /**
   * Treat consecutive even/odd segment rows as opposite directed half-edges and never immediately
   * return along the twin. Used by general polygonization. Defaults to false.
   */
  pairedOpposites?: boolean;
  /**
   * Compile-time vertex matching distance (Chebyshev, input units). Two vertices match when
   * both coordinates differ by at most this value. Choose at least four times the f32 spacing of
   * the coordinates (7.6e-6 degrees near longitude 100) and well below the shortest segment.
   * Defaults to 5e-5.
   */
  vertexTolerance?: number;
  /**
   * Which side of each segment is filled. Decides which ring is a shell and the rule at shared
   * vertices. `GPUCellSetOutline` of H3 is `'left'` (counter-clockwise shells), of Quadbin
   * `'right'`. Defaults to `'left'`.
   */
  interiorSide?: GPUSegmentRingAssemblyInteriorSide;
  /**
   * Reverse rings of `interiorSide: 'right'` input so every shell is counter-clockwise and every
   * hole clockwise (RFC 7946). Defaults to false (rings follow the segment direction).
   */
  normalizeWinding?: boolean;
  /**
   * Scale x differences by `cos(y)` when comparing turn angles at shared vertices (longitude and
   * latitude in degrees). Defaults to true; set false for planar coordinates.
   */
  geographic?: boolean;
  /**
   * Remove pairs of exactly opposite segments (`a -> b` and `b -> a` in one group, within the
   * vertex tolerance) before chaining. Such pairs are zero-width slits or ridges where a region is
   * cut by a line of zero thickness, for example band boundaries of `GPUIsobandRings` on samples
   * that equal a break; left in, they pinch rings into spikes and strand chains. Pairs match one
   * to one by lowest segment index; both segments get
   * {@link GPU_SEGMENT_RING_ASSEMBLY_FLAG_CANCELLED}. Defaults to false.
   */
  cancelOpposingSegments?: boolean;
  /**
   * Split rings that touch themselves at a vertex into separate rings (two holes meeting at a
   * corner, a hole pinched to its shell) at the cost of a second tracing pass. Regions touching at
   * a vertex are always separate rings. Without it the rule keeps interior wedges tight, so
   * touching holes join into one self-touching ring. Defaults to true.
   */
  splitTouchingRings?: boolean;
  /** Caller-owned output. */
  output: GPUSegmentRingAssemblyOutput;
};

/**
 * Chains directed boundary segments into closed rings on the GPU.
 *
 * Every segment's end vertex is matched to the segment starting there (coordinates within
 * `vertexTolerance`, found through a sort of quantized vertex hashes and a 3 x 3 neighborhood
 * lookup). At a vertex shared by several rings (cells touching at a corner) the continuation is
 * the outgoing segment that turns tightest toward the interior side (the face-traversal rule, a
 * bijection on planar boundaries), which keeps regions that touch only at a point as separate
 * rings. Where a choice is still taken twice the lowest segment index wins, the loser is marked
 * {@link GPU_SEGMENT_RING_ASSEMBLY_FLAG_CONFLICT} and left open. That rule makes holes that touch at a
 * corner (or a hole pinched to its shell) one self-touching ring, so a second tracing pass
 * (`splitTouchingRings`, default) swaps the continuations of two segments that end at the same
 * vertex whenever both lie on one ring, which splits it into simple rings.
 *
 * The successor graph is a set of disjoint paths and cycles. Pointer jumping
 * (`ceil(log2(rows))` rounds per phase) finds each cycle's lowest segment index, which becomes its
 * leader and gives the ring ID (rank among leaders, so rings are ordered by lowest segment index),
 * then ranks every segment from the leader. Scans of ring and vertex counts place each ring; a
 * ring starts at its leader segment and follows the segment direction (or is reversed by
 * `normalizeWinding`). Chains that never close (dangling ends, truncated input) are reported
 * through `openSegmentCount` and `segmentFlags` and emit no ring.
 *
 * Rings are written as GeoArrow-style `ringOffsets` plus `positions`. Signed area and orientation
 * identify shells and holes; `ringShells` assigns each hole to its innermost enclosing shell by
 * ray casting a point of the hole's first edge against every shell whose bounding box contains it
 * (`holes * shells` box tests plus the vertices of the shells that pass; ring statistics give each
 * lane its own ring and sum rings above 64 vertices cooperatively, so a single huge ring does not serialize them). `polygons` regroups the rings into polygon topology for
 * `GPUPointInPolygonJoin`. Output order and results are deterministic (integer atomics only).
 *
 * Limits: matching is geometric, so shared vertices must agree within the tolerance (exact for
 * Quadbin, within f32 round-off for H3); rings that span the antimeridian keep the longitudes of
 * the input; area uses f32 arithmetic relative to each ring's first vertex.
 */
export class GPUSegmentRingAssembly implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUSegmentRingAssemblyProps;

  constructor(props: GPUSegmentRingAssemblyProps) {
    this.id = props.id ?? 'segment-ring-assembly';
    this.props = props;
    const id = this.id;
    const {output} = props;
    for (const [name, view] of Object.entries({...props, ...output})) {
      if (name !== 'output' && (view as unknown) instanceof GraphVectorView) {
        throw new Error(`${id} ${name} must be a single packed view, not a chunked vector`);
      }
    }
    validatePackedView(props.endpoints, ['float32x4'], `${id} endpoints`);
    const segments = props.endpoints.length;
    if (segments < 1) {
      throw new Error(`${id} needs at least one segment row`);
    }
    if (props.vertexTolerance !== undefined && !(props.vertexTolerance > 0)) {
      throw new Error(`${id} vertexTolerance must be positive`);
    }
    if (props.interiorSide !== undefined && !['left', 'right'].includes(props.interiorSide)) {
      throw new Error(`${id} interiorSide must be 'left' or 'right'`);
    }
    for (const [name, view] of [
      ['count', props.count],
      ['output.count', output.count],
      ['output.overflow', output.overflow],
      ['output.requiredCount', output.requiredCount],
      ['output.openSegmentCount', output.openSegmentCount],
      ['output.touchingSegmentCount', output.touchingSegmentCount]
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
      if (props.groups.length !== segments) {
        throw new Error(`${id} groups length must equal the segment row count`);
      }
    }
    if (output.ringGroups && !props.groups) {
      throw new Error(`${id} output.ringGroups needs the groups input`);
    }
    validatePackedUint32View(output.ringOffsets, `${id} output.ringOffsets`);
    if (output.ringOffsets.length < 2) {
      throw new Error(`${id} output.ringOffsets needs at least two rows (one ring)`);
    }
    validatePackedView(output.positions, ['float32x2'], `${id} output.positions`);
    if (output.positions.length < 1) {
      throw new Error(`${id} output.positions must not be empty`);
    }
    const ringCapacity = output.ringOffsets.length - 1;
    for (const [name, view] of [
      ['ringIsHole', output.ringIsHole],
      ['ringShells', output.ringShells],
      ['ringGroups', output.ringGroups]
    ] as const) {
      if (view) {
        validatePackedUint32View(view, `${id} output.${name}`);
        if (view.length !== ringCapacity) {
          throw new Error(`${id} output.${name} length must equal the ring capacity`);
        }
      }
    }
    if (output.ringAreas) {
      validatePackedView(output.ringAreas, ['float32'], `${id} output.ringAreas`);
      if (output.ringAreas.length !== ringCapacity) {
        throw new Error(`${id} output.ringAreas length must equal the ring capacity`);
      }
    }
    if (output.polygons) {
      const {polygons} = output;
      validatePackedView(polygons.positions, ['float32x2'], `${id} output.polygons.positions`);
      if (polygons.positions.length !== output.positions.length) {
        throw new Error(`${id} output.polygons.positions must match output.positions length`);
      }
      if (polygons.sourceIds) {
        if (!props.groups) {
          throw new Error(`${id} output.polygons.sourceIds needs the groups input`);
        }
        validatePackedUint32View(polygons.sourceIds, `${id} output.polygons.sourceIds`);
        if (polygons.sourceIds.length !== ringCapacity) {
          throw new Error(
            `${id} output.polygons.sourceIds must have ${ringCapacity} rows (the ring capacity)`
          );
        }
      }
      for (const [name, view, length] of [
        ['ringOffsets', polygons.ringOffsets, ringCapacity + 1],
        ['polygonOffsets', polygons.polygonOffsets, ringCapacity + 1],
        ['featureOffsets', polygons.featureOffsets, ringCapacity + 1]
      ] as const) {
        validatePackedUint32View(view, `${id} output.polygons.${name}`);
        if (view.length !== length) {
          throw new Error(`${id} output.polygons.${name} must have ${length} rows`);
        }
      }
    }
    for (const [name, view] of [
      ['segmentRings', output.segmentRings],
      ['segmentVertices', output.segmentVertices],
      ['segmentFlags', output.segmentFlags]
    ] as const) {
      if (view) {
        validatePackedUint32View(view, `${id} output.${name}`);
        if (view.length !== segments) {
          throw new Error(`${id} output.${name} length must equal the segment row count`);
        }
      }
    }
    validateGraphOutputsDisjointFromInputs(id, getOutputViews(output), [
      props.endpoints,
      props.count,
      props.groups
    ]);
  }

  /** Returns the hash, sort, match, pointer jumping, ranking, placement and ring-stat nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {output} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      props.endpoints,
      props.count,
      props.groups,
      ...getOutputViews(output)
    ]);
    const segments = props.endpoints.length;
    const ringCapacity = output.ringOffsets.length - 1;
    const vertexCapacity = output.positions.length;
    const tolerance = props.vertexTolerance ?? 5e-5;
    const interiorSide = props.interiorSide ?? 'left';
    const flip = Boolean(props.normalizeWinding) && interiorSide === 'right';
    // Orientation sign of a shell in the written rings: +1 counter-clockwise.
    const shellSign = props.normalizeWinding || interiorSide === 'left' ? 1 : -1;
    const geographic = props.geographic ?? true;
    const hasGroups = Boolean(props.groups);
    const none = `${GPU_SEGMENT_RING_ASSEMBLY_NONE}u`;
    const transient = <Format extends 'uint32' | 'float32' | 'float32x2'>(
      name: string,
      format: Format,
      length: number
    ) => createTransientView(graph, `${id}-${name}`, format, length);
    const u32 = (name: string, length: number = segments) => transient(name, 'uint32', length);

    const keys = u32('keys');
    const ids = u32('ids');
    const sortedKeys = u32('sorted-keys');
    const sortedIds = u32('sorted-ids');
    const claim = u32('claim');
    const cancelOpposing = Boolean(props.cancelOpposingSegments);
    const counters = u32('counters', 3);
    const next = u32('next');
    const alternatives = u32('alternatives');
    const swapped = u32('swapped');
    const segmentFlags = output.segmentFlags ?? u32('flags');
    const leader = u32('leader');
    const leaderFlags = u32('leader-flags');
    const leaderVertices = u32('leader-vertices');
    const ringScan = u32('ring-scan');
    const vertexScan = u32('vertex-scan');
    const ringStarts = u32('ring-starts');
    const ringLeaders = u32('ring-leaders');
    const ringCount = u32('ring-count', 1);
    const segmentRings = output.segmentRings ?? u32('segment-rings');
    const segmentVertices = output.segmentVertices ?? u32('segment-vertices');
    const closeVertices = u32('close-vertices');
    const pointerSets = [u32('pointer-a'), u32('pointer-b')];
    const valueSets = [u32('value-a'), u32('value-b')];

    const hashFunctions = `const INVERSE_CELL: f32 = ${getWGSLFloatLiteral(1 / tolerance)};
fn ringQuantize(value: f32) -> i32 { return i32(floor(value * INVERSE_CELL)); }
fn ringHash(qx: i32, qy: i32, group: u32) -> u32 {
  var h = bitcast<u32>(qx) * 0x9E3779B1u;
  h = h ^ (h >> 15u);
  h = (h + bitcast<u32>(qy) * 0x85EBCA77u) * 0xC2B2AE3Du;
  h = h ^ (h >> 13u);
  h = (h + group * 0x27D4EB2Fu) * 0x165667B1u;
  h = h ^ (h >> 16u);
  return select(h, 0xfffffffeu, h == 0xffffffffu);
}`;
    const validRows = (name: string) => `min(${name}[${name}Offset], ${segments}u)`;
    const countBinding: WGSLKernelBinding[] = props.count
      ? [{name: 'rowCount', view: props.count, type: 'u32', access: 'read'}]
      : [];
    const groupBinding: WGSLKernelBinding[] = props.groups
      ? [{name: 'groups', view: props.groups, type: 'u32', access: 'read'}]
      : [];
    const validRowsExpression = props.count ? validRows('rowCount') : `${segments}u`;
    const groupOf = (row: string) => (hasGroups ? `groups[groupsOffset + ${row}]` : '0u');
    const kernel = (
      name: string,
      bindings: WGSLKernelBinding[],
      invocationCount: number,
      body: string,
      declarations: string = ''
    ) =>
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-${name}`,
        operation: OPERATION,
        variant: name,
        bindings,
        invocationCount,
        declarations,
        body
      });
    const read = (name: string, view: GraphDataView, type: 'u32' | 'f32' = 'u32') =>
      ({name, view, type, access: 'read'}) as WGSLKernelBinding;
    const write = (name: string, view: GraphDataView, type: 'u32' | 'f32' = 'u32') =>
      ({name, view, type, access: 'read_write'}) as WGSLKernelBinding;

    const nodes: GPUCommandNode<Parameters>[] = [];

    // 1. Hash start vertices (invalid rows get the maximum key so they sort last).
    nodes.push(
      kernel(
        'hash',
        [
          read('endpoints', props.endpoints, 'f32'),
          ...countBinding,
          ...groupBinding,
          write('keys', keys),
          write('ids', ids),
          write('claim', claim),
          write('counters', counters)
        ],
        segments,
        `ids[idsOffset + index] = index;
  claim[claimOffset + index] = 0xffffffffu;
  if (index == 0u) { counters[countersOffset] = 0u; counters[countersOffset + 1u] = 0u; counters[countersOffset + 2u] = 0u; }
  var key = 0xffffffffu;
  if (index < ${validRowsExpression}) {
    let start = vec2f(endpoints[endpointsOffset + 4u * index], endpoints[endpointsOffset + 4u * index + 1u]);
    key = ringHash(ringQuantize(start.x), ringQuantize(start.y), ${groupOf('index')});
  }
  keys[keysOffset + index] = key;`,
        hashFunctions
      )
    );
    nodes.push(
      ...new GPUSort({
        id: `${id}-sort`,
        keys,
        values: ids,
        outputKeys: sortedKeys,
        outputValues: sortedIds
      }).getCommandNodes(graph)
    );

    // 1b. Cancel exactly opposite segment pairs.
    if (cancelOpposing) {
      nodes.push(
        kernel(
          'cancel',
          [
            read('endpoints', props.endpoints, 'f32'),
            ...countBinding,
            ...groupBinding,
            read('sortedKeys', sortedKeys),
            read('sortedIds', sortedIds),
            write('flags', segmentFlags),
            {name: 'counters', view: counters, type: 'atomic<u32>', access: 'read_write'}
          ],
          segments,
          `flags[flagsOffset + index] = 0u;
  if (index >= ${validRowsExpression}) { return; }
  let partner = findReverse(index);
  if (partner != ${none} && findReverse(partner) == index) {
    flags[flagsOffset + index] = ${GPU_SEGMENT_RING_ASSEMBLY_FLAG_CANCELLED}u;
    atomicAdd(&counters[countersOffset + 2u], 1u);
  }`,
          `${hashFunctions}
const TOLERANCE: f32 = ${getWGSLFloatLiteral(tolerance)};
fn ringStart(row: u32) -> vec2f {
  return vec2f(endpoints[endpointsOffset + 4u * row], endpoints[endpointsOffset + 4u * row + 1u]);
}
fn ringEnd(row: u32) -> vec2f {
  return vec2f(endpoints[endpointsOffset + 4u * row + 2u], endpoints[endpointsOffset + 4u * row + 3u]);
}
// Lowest segment that starts where 'index' ends and ends where it starts.
fn findReverse(index: u32) -> u32 {
  let valid = ${validRowsExpression};
  let origin = ringStart(index);
  let vertex = ringEnd(index);
  let group = ${groupOf('index')};
  let qx = ringQuantize(vertex.x);
  let qy = ringQuantize(vertex.y);
  var seen = array<u32, 9>();
  var seenCount = 0u;
  var best = ${none};
  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      let hash = ringHash(qx + dx, qy + dy, group);
      var duplicate = false;
      for (var j = 0u; j < seenCount; j++) { duplicate = duplicate || seen[j] == hash; }
      if (duplicate) { continue; }
      seen[seenCount] = hash;
      seenCount++;
      var low = 0u;
      var high = ${segments}u;
      while (low < high) {
        let middle = low + (high - low) / 2u;
        if (sortedKeys[sortedKeysOffset + middle] < hash) { low = middle + 1u; } else { high = middle; }
      }
      var position = low;
      while (position < ${segments}u && sortedKeys[sortedKeysOffset + position] == hash) {
        let candidate = sortedIds[sortedIdsOffset + position];
        position++;
        if (candidate >= valid || candidate == index) { continue; }
        let candidateStart = ringStart(candidate);
        let candidateEnd = ringEnd(candidate);
        if (abs(candidateStart.x - vertex.x) > TOLERANCE || abs(candidateStart.y - vertex.y) > TOLERANCE) { continue; }
        if (abs(candidateEnd.x - origin.x) > TOLERANCE || abs(candidateEnd.y - origin.y) > TOLERANCE) { continue; }
        if (${hasGroups ? `groups[groupsOffset + candidate] != group` : 'false'}) { continue; }
        best = min(best, candidate);
      }
    }
  }
  return best;
}`
        )
      );
    }

    // 2. Choose each segment's continuation.
    const sideSign = interiorSide === 'left' ? '1.0' : '-1.0';
    nodes.push(
      kernel(
        'match',
        [
          read('endpoints', props.endpoints, 'f32'),
          ...countBinding,
          ...groupBinding,
          read('sortedKeys', sortedKeys),
          read('sortedIds', sortedIds),
          write('next', next),
          write('flags', segmentFlags),
          write('alt', alternatives)
        ],
        segments,
        `next[nextOffset + index] = ${none};
  alt[altOffset + index] = ${none};
  ${
    cancelOpposing
      ? `let carried = flags[flagsOffset + index] & ${GPU_SEGMENT_RING_ASSEMBLY_FLAG_CANCELLED}u;
  flags[flagsOffset + index] = carried;
  if (carried != 0u) { return; }`
      : 'flags[flagsOffset + index] = 0u;'
  }
  let valid = ${validRowsExpression};
  if (index >= valid) { return; }
  let origin = ringStart(index);
  let vertex = ringEnd(index);
  let group = ${groupOf('index')};
  let incoming = ringScaled(vertex - origin, vertex.y);
  let qx = ringQuantize(vertex.x);
  let qy = ringQuantize(vertex.y);
  var seen = array<u32, 9>();
  var seenCount = 0u;
  var best = ${none};
  var bestTurn = 0.0;
  var matches = 0u;
  var first = ${none};
  var second = ${none};
  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      let hash = ringHash(qx + dx, qy + dy, group);
      var duplicate = false;
      for (var j = 0u; j < seenCount; j++) { duplicate = duplicate || seen[j] == hash; }
      if (duplicate) { continue; }
      seen[seenCount] = hash;
      seenCount++;
      var low = 0u;
      var high = ${segments}u;
      while (low < high) {
        let middle = low + (high - low) / 2u;
        if (sortedKeys[sortedKeysOffset + middle] < hash) { low = middle + 1u; } else { high = middle; }
      }
      var position = low;
      while (position < ${segments}u && sortedKeys[sortedKeysOffset + position] == hash) {
        let candidate = sortedIds[sortedIdsOffset + position];
        position++;
        if (candidate >= valid || candidate == index) { continue; }
        ${props.pairedOpposites ? 'if (candidate == (index ^ 1u)) { continue; }' : ''}
        ${cancelOpposing ? `if ((flags[flagsOffset + candidate] & ${GPU_SEGMENT_RING_ASSEMBLY_FLAG_CANCELLED}u) != 0u) { continue; }` : ''}
        let candidateStart = ringStart(candidate);
        if (abs(candidateStart.x - vertex.x) > TOLERANCE || abs(candidateStart.y - vertex.y) > TOLERANCE) { continue; }
        if (${hasGroups ? `groups[groupsOffset + candidate] != group` : 'false'}) { continue; }
        matches++;
        if (first == ${none}) { first = candidate; } else if (second == ${none}) { second = candidate; }
        let outgoing = ringScaled(ringEnd(candidate) - candidateStart, vertex.y);
        let turn = SIDE * ringTurnKey(incoming.x * outgoing.y - incoming.y * outgoing.x, dot(incoming, outgoing));
        if (best == ${none} || turn > bestTurn || (turn == bestTurn && candidate < best)) {
          best = candidate;
          bestTurn = turn;
        }
      }
    }
  }
  next[nextOffset + index] = best;
  if (matches == 2u) { alt[altOffset + index] = select(first, second, best == first); }
  var flagBits = 0u;
  if (matches > 1u) { flagBits = ${GPU_SEGMENT_RING_ASSEMBLY_FLAG_TOUCHING}u; }
  if (matches == 0u) { flagBits = ${GPU_SEGMENT_RING_ASSEMBLY_FLAG_DANGLING}u; }
  flags[flagsOffset + index] = flagBits;`,
        `${hashFunctions}
const TOLERANCE: f32 = ${getWGSLFloatLiteral(tolerance)};
const SIDE: f32 = ${sideSign};
// Monotone in the turn angle from the incoming to the outgoing direction (counter-clockwise
// positive, u-turn at +2). atan2 is avoided: it misbehaves on signed zeros under fast math.
fn ringTurnKey(cross: f32, dot: f32) -> f32 {
  let denominator = abs(cross) + abs(dot);
  if (denominator == 0.0) { return 0.0; }
  let ratio = cross / denominator;
  if (dot >= 0.0) { return ratio; }
  return select(-2.0, 2.0, cross >= 0.0) - ratio;
}
fn ringStart(row: u32) -> vec2f {
  return vec2f(endpoints[endpointsOffset + 4u * row], endpoints[endpointsOffset + 4u * row + 1u]);
}
fn ringEnd(row: u32) -> vec2f {
  return vec2f(endpoints[endpointsOffset + 4u * row + 2u], endpoints[endpointsOffset + 4u * row + 3u]);
}
fn ringScaled(delta: vec2f, latitude: f32) -> vec2f {
  ${geographic ? 'return vec2f(delta.x * max(cos(radians(latitude)), 0.01), delta.y);' : 'return delta;'}
}`
      )
    );

    // 3. Make the successor map injective: the lowest-index claimant keeps a continuation.
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-claim`,
        operation: OPERATION,
        variant: 'claim',
        bindings: [
          read('next', next),
          {name: 'claim', view: claim, type: 'atomic<u32>', access: 'read_write'}
        ],
        invocationCount: segments,
        body: `let claimed = next[nextOffset + index];
  if (claimed != ${none}) { atomicMin(&claim[claimOffset + claimed], index); }`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-resolve`,
        operation: OPERATION,
        variant: 'resolve',
        bindings: [
          write('next', next),
          read('claim', claim),
          write('flags', segmentFlags),
          {name: 'counters', view: counters, type: 'atomic<u32>', access: 'read_write'}
        ],
        invocationCount: segments,
        body: `let claimed = next[nextOffset + index];
  if (claimed != ${none} && claim[claimOffset + claimed] != index) {
    next[nextOffset + index] = ${none};
    flags[flagsOffset + index] = flags[flagsOffset + index] | ${GPU_SEGMENT_RING_ASSEMBLY_FLAG_CONFLICT}u;
  }
  if ((flags[flagsOffset + index] & ${GPU_SEGMENT_RING_ASSEMBLY_FLAG_TOUCHING}u) != 0u) {
    atomicAdd(&counters[countersOffset + 1u], 1u);
  }`
      })
    );

    // 4. Trace: pointer jumping finds each cycle's lowest segment index (leader) and the hops of
    // every segment to the segment before its leader. Closed segments sit on cycles.
    const rounds = Math.max(1, Math.ceil(Math.log2(segments)));
    const trace = (pass: number, successor: GraphDataView<'uint32'>, countOpen: boolean) => {
      nodes.push(
        kernel(
          `jump-init-${pass}`,
          [read('next', successor), write('pointer', pointerSets[0]), write('value', valueSets[0])],
          segments,
          `pointer[pointerOffset + index] = next[nextOffset + index];
  value[valueOffset + index] = index;`
        )
      );
      let current = 0;
      for (let round = 0; round < rounds; round++) {
        nodes.push(createJumpNode('min', `${pass}-${round}`, current));
        current = 1 - current;
      }
      const cutTarget = 1 - current;
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-cut-${pass}`,
          operation: OPERATION,
          variant: 'cut',
          bindings: [
            read('next', successor),
            read('pointerIn', pointerSets[current]),
            read('valueIn', valueSets[current]),
            ...countBinding,
            write('leader', leader),
            write('pointerOut', pointerSets[cutTarget]),
            write('valueOut', valueSets[cutTarget]),
            {name: 'counters', view: counters, type: 'atomic<u32>', access: 'read_write'}
          ],
          invocationCount: segments,
          body: `let valid = ${validRowsExpression};
  var leaderIndex = ${none};
  var tail = ${none};
  var hopCount = 0u;
  if (index < valid) {
    if (pointerIn[pointerInOffset + index] != ${none}) {
      leaderIndex = valueIn[valueInOffset + index];
      tail = next[nextOffset + index];
      if (tail == leaderIndex) { tail = ${none}; }
      if (tail != ${none}) { hopCount = 1u; }
    } else {
      ${countOpen ? 'atomicAdd(&counters[countersOffset], 1u);' : ''}
    }
  }
  leader[leaderOffset + index] = leaderIndex;
  pointerOut[pointerOutOffset + index] = tail;
  valueOut[valueOutOffset + index] = hopCount;`
        })
      );
      current = cutTarget;
      for (let round = 0; round < rounds; round++) {
        nodes.push(createJumpNode('sum', `${pass}-${round}`, current));
        current = 1 - current;
      }
      return valueSets[current];
    };
    let hops = trace(1, next, props.splitTouchingRings === false);
    if (props.splitTouchingRings !== false) {
      // Two segments ending at one vertex continue into the same two outgoing segments. When both
      // lie on one ring, swapping their continuations splits the ring in two (a figure eight of
      // holes, or a shell pinched to its hole); when they lie on different rings the swap would
      // merge them, so those stay. Touch points of planar boundaries do not interleave, so all
      // swaps can be applied at once.
      nodes.push(
        kernel(
          'split-touching',
          [
            read('next', next),
            read('alt', alternatives),
            read('claim', claim),
            read('leader', leader),
            write('swapped', swapped)
          ],
          segments,
          `let chosen = next[nextOffset + index];
  var result = chosen;
  let other = alt[altOffset + index];
  if (chosen != ${none} && other != ${none}) {
    let partner = claim[claimOffset + other];
    if (partner != ${none} && partner != index &&
        alt[altOffset + partner] == chosen && next[nextOffset + partner] == other &&
        leader[leaderOffset + index] != ${none} &&
        leader[leaderOffset + index] == leader[leaderOffset + partner]) {
      result = other;
    }
  }
  swapped[swappedOffset + index] = result;`
        )
      );
      hops = trace(2, swapped, true);
    }

    // 6. Ring and vertex counts per leader, scanned.
    nodes.push(
      kernel(
        'leaders',
        [
          read('leader', leader),
          read('hops', hops),
          write('leaderFlags', leaderFlags),
          write('leaderVertices', leaderVertices)
        ],
        segments,
        `let isLeader = leader[leaderOffset + index] == index;
  leaderFlags[leaderFlagsOffset + index] = select(0u, 1u, isLeader);
  leaderVertices[leaderVerticesOffset + index] = select(0u, hops[hopsOffset + index] + 2u, isLeader);`
      ),
      ...new GPUScan({
        id: `${id}-ring-scan`,
        input: leaderFlags,
        output: ringScan,
        mode: 'inclusive'
      }).getCommandNodes(graph),
      ...new GPUScan({
        id: `${id}-vertex-scan`,
        input: leaderVertices,
        output: vertexScan,
        mode: 'inclusive'
      }).getCommandNodes(graph),
      kernel(
        'ring-starts',
        [
          read('leader', leader),
          read('hops', hops),
          read('ringScan', ringScan),
          read('vertexScan', vertexScan),
          write('ringStarts', ringStarts),
          write('ringLeaders', ringLeaders)
        ],
        segments,
        `if (leader[leaderOffset + index] != index) { return; }
  let ring = ringScan[ringScanOffset + index] - 1u;
  ringStarts[ringStartsOffset + ring] = vertexScan[vertexScanOffset + index] - (hops[hopsOffset + index] + 2u);
  ringLeaders[ringLeadersOffset + ring] = index;`
      ),
      kernel(
        'ring-count',
        [
          read('ringScan', ringScan),
          read('vertexScan', vertexScan),
          read('ringStarts', ringStarts),
          write('ringCount', ringCount)
        ],
        1,
        `let total = ringScan[ringScanOffset + ${segments - 1}u];
  let vertices = vertexScan[vertexScanOffset + ${segments - 1}u];
  var low = 0u;
  var high = min(total, ${ringCapacity}u);
  while (low < high) {
    let middle = low + (high - low + 1u) / 2u;
    var end = vertices;
    if (middle < total) { end = ringStarts[ringStartsOffset + middle]; }
    if (end <= ${vertexCapacity}u) { low = middle; } else { high = middle - 1u; }
  }
  ringCount[ringCountOffset] = low;`
      )
    );

    // 7. Place and write vertices.
    nodes.push(
      kernel(
        'place',
        [
          read('leader', leader),
          read('hops', hops),
          read('ringScan', ringScan),
          read('vertexScan', vertexScan),
          read('ringCount', ringCount),
          write('segmentRings', segmentRings),
          write('segmentVertices', segmentVertices),
          write('closeVertices', closeVertices)
        ],
        segments,
        `segmentRings[segmentRingsOffset + index] = ${none};
  segmentVertices[segmentVerticesOffset + index] = ${none};
  closeVertices[closeVerticesOffset + index] = ${none};
  let head = leader[leaderOffset + index];
  if (head == ${none}) { return; }
  let ring = ringScan[ringScanOffset + head] - 1u;
  if (ring >= ringCount[ringCountOffset]) { return; }
  let ringLength = hops[hopsOffset + head] + 1u;
  let start = vertexScan[vertexScanOffset + head] - (ringLength + 1u);
  let position = hops[hopsOffset + head] - hops[hopsOffset + index];
  ${flip ? 'let slot = (ringLength - position) % ringLength;' : 'let slot = position;'}
  segmentRings[segmentRingsOffset + index] = ring;
  segmentVertices[segmentVerticesOffset + index] = start + slot;
  if (position == 0u) { closeVertices[closeVerticesOffset + index] = start + ringLength; }`
      ),
      kernel(
        'write',
        [
          read('endpoints', props.endpoints, 'f32'),
          read('segmentVertices', segmentVertices),
          read('closeVertices', closeVertices),
          write('positions', output.positions, 'f32')
        ],
        segments,
        `let slot = segmentVertices[segmentVerticesOffset + index];
  if (slot == ${none}) { return; }
  let x = endpoints[endpointsOffset + 4u * index];
  let y = endpoints[endpointsOffset + 4u * index + 1u];
  positions[positionsOffset + 2u * slot] = x;
  positions[positionsOffset + 2u * slot + 1u] = y;
  let close = closeVertices[closeVerticesOffset + index];
  if (close != ${none}) {
    positions[positionsOffset + 2u * close] = x;
    positions[positionsOffset + 2u * close + 1u] = y;
  }`
      ),
      kernel(
        'offsets',
        [
          read('ringStarts', ringStarts),
          read('ringScan', ringScan),
          read('vertexScan', vertexScan),
          read('ringCount', ringCount),
          write('ringOffsets', output.ringOffsets)
        ],
        ringCapacity + 1,
        `let total = ringScan[ringScanOffset + ${segments - 1}u];
  let written = min(index, ringCount[ringCountOffset]);
  var offset = vertexScan[vertexScanOffset + ${segments - 1}u];
  if (written < total) { offset = ringStarts[ringStartsOffset + written]; }
  ringOffsets[ringOffsetsOffset + index] = offset;`
      )
    );

    // 8. Ring statistics: area, hole flag, bounds, group, shell assignment.
    const ringAreas = output.ringAreas ?? transient('ring-areas', 'float32', ringCapacity);
    const ringIsHole = output.ringIsHole ?? u32('ring-is-hole', ringCapacity);
    const needsShells = Boolean(output.ringShells ?? output.polygons);
    // Ring bounds let the shell search reject a shell with four comparisons instead of walking it.
    const ringBounds = needsShells
      ? createTransientView(graph, `${id}-ring-bounds`, 'float32', ringCapacity * 4)
      : undefined;
    nodes.push(
      // Block hybrid: workgroup w owns rings [RING_LANES * w, RING_LANES * w + RING_LANES). Phase 1:
      // every lane sums its own ring serially when it has at most RING_LANES vertices (bit-identical
      // to the former one-thread-per-ring kernel). Phase 2: the workgroup visits the larger rings of
      // its block in ascending order and splits each across the lanes with a fixed-order tree. Thread
      // count stays the ring count rounded up to RING_LANES, and one huge ring no longer serializes
      // the stage.
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-ring-stats`,
        operation: OPERATION,
        variant: 'ring-stats',
        bindings: [
          read('ringOffsets', output.ringOffsets),
          read('positions', output.positions, 'f32'),
          read('ringCount', ringCount),
          write('areas', ringAreas, 'f32'),
          write('isHole', ringIsHole),
          ...(ringBounds ? [write('bounds', ringBounds, 'f32')] : [])
        ],
        invocationCount: Math.ceil(ringCapacity / RING_LANES) * RING_LANES,
        workgroupSize: RING_LANES,
        guardIndex: false,
        declarations: `const RING_CAPACITY: u32 = ${ringCapacity}u;
const RING_LANES: u32 = ${RING_LANES}u;
const BOUNDS_LARGE: f32 = 3.0e38;
var<workgroup> largeBits: array<atomic<u32>, 2>;
var<workgroup> largeBitsCopy: array<u32, 2>;
var<workgroup> largeFirst: array<u32, ${RING_LANES}>;
var<workgroup> largeLast: array<u32, ${RING_LANES}>;
var<workgroup> partialSum: array<f32, ${RING_LANES}>;
var<workgroup> partialMinimum: array<vec2f, ${RING_LANES}>;
var<workgroup> partialMaximum: array<vec2f, ${RING_LANES}>;
fn getVertex(vertex: u32) -> vec2f {
  return vec2f(positions[positionsOffset + 2u * vertex], positions[positionsOffset + 2u * vertex + 1u]);
}
fn publishRing(ring: u32, sum: f32, minimum: vec2f, maximum: vec2f) {
  let area = 0.5 * sum;
  areas[areasOffset + ring] = area;
  isHole[isHoleOffset + ring] = select(0u, 1u, area * ${getWGSLFloatLiteral(shellSign)} < 0.0);
  ${ringBounds ? 'bounds[boundsOffset + 4u * ring] = minimum.x; bounds[boundsOffset + 4u * ring + 1u] = minimum.y; bounds[boundsOffset + 4u * ring + 2u] = maximum.x; bounds[boundsOffset + 4u * ring + 3u] = maximum.y;' : ''}
}`,
        body: `let lane = localInvocationIndex;
  let ringBase = workgroupIndex * RING_LANES;
  let ownRing = ringBase + lane;
  // Phase 1: this lane's own ring, serially when small; larger rings are queued for phase 2.
  if (ownRing < RING_CAPACITY) {
    if (ownRing >= ringCount[ringCountOffset]) {
      publishRing(ownRing, 0.0, vec2f(0.0), vec2f(0.0));
    } else {
      let first = ringOffsets[ringOffsetsOffset + ownRing];
      let last = ringOffsets[ringOffsetsOffset + ownRing + 1u];
      if (last - first <= RING_LANES) {
        let origin = getVertex(first);
        var sum = 0.0;
        var minimum = origin;
        var maximum = origin;
        for (var vertex = first; vertex + 1u < last; vertex++) {
          let p = getVertex(vertex);
          let q = getVertex(vertex + 1u);
          minimum = min(minimum, q);
          maximum = max(maximum, q);
          let a = p - origin;
          let b = q - origin;
          sum += a.x * b.y - b.x * a.y;
        }
        publishRing(ownRing, sum, minimum, maximum);
      } else {
        largeFirst[lane] = first;
        largeLast[lane] = last;
        atomicOr(&largeBits[lane / 32u], 1u << (lane % 32u));
      }
    }
  }
  workgroupBarrier();
  if (lane == 0u) {
    largeBitsCopy[0] = atomicLoad(&largeBits[0]);
    largeBitsCopy[1] = atomicLoad(&largeBits[1]);
  }
  // Phase 2: every barrier below is in workgroup-uniform control flow (the queue is read through
  // workgroupUniformLoad), and the queued rings are visited in ascending order.
  for (var word = 0u; word < 2u; word++) {
    var pending = workgroupUniformLoad(&largeBitsCopy[word]);
    while (pending != 0u) {
      let slot = word * 32u + firstTrailingBit(pending);
      pending = pending & (pending - 1u);
      let first = workgroupUniformLoad(&largeFirst[slot]);
      let last = workgroupUniformLoad(&largeLast[slot]);
      let origin = getVertex(first);
      var sum = 0.0;
      var minimum = vec2f(BOUNDS_LARGE);
      var maximum = vec2f(-BOUNDS_LARGE);
      for (var vertex = first + lane; vertex + 1u < last; vertex += RING_LANES) {
        let p = getVertex(vertex);
        let q = getVertex(vertex + 1u);
        minimum = min(minimum, q);
        maximum = max(maximum, q);
        let a = p - origin;
        let b = q - origin;
        sum += a.x * b.y - b.x * a.y;
      }
      partialSum[lane] = sum;
      partialMinimum[lane] = minimum;
      partialMaximum[lane] = maximum;
      workgroupBarrier();
      for (var stride = RING_LANES / 2u; stride > 0u; stride = stride / 2u) {
        if (lane < stride) {
          partialSum[lane] += partialSum[lane + stride];
          partialMinimum[lane] = min(partialMinimum[lane], partialMinimum[lane + stride]);
          partialMaximum[lane] = max(partialMaximum[lane], partialMaximum[lane + stride]);
        }
        workgroupBarrier();
      }
      if (lane == 0u) {
        publishRing(ringBase + slot, partialSum[0], min(partialMinimum[0], origin), max(partialMaximum[0], origin));
      }
    }
  }`
      })
    );
    const ringGroups =
      output.ringGroups ?? (hasGroups ? u32('ring-groups', ringCapacity) : undefined);
    if (ringGroups && props.groups) {
      nodes.push(
        kernel(
          'ring-groups',
          [
            read('ringLeaders', ringLeaders),
            read('ringCount', ringCount),
            ...groupBinding,
            write('ringGroups', ringGroups)
          ],
          ringCapacity,
          `ringGroups[ringGroupsOffset + index] = 0u;
  if (index < ringCount[ringCountOffset]) {
    ringGroups[ringGroupsOffset + index] = groups[groupsOffset + ringLeaders[ringLeadersOffset + index]];
  }`
        )
      );
    }
    const ringShells =
      output.ringShells ?? (output.polygons ? u32('ring-shells', ringCapacity) : undefined);
    if (ringShells) {
      nodes.push(
        kernel(
          'ring-shells',
          [
            read('ringOffsets', output.ringOffsets),
            read('positions', output.positions, 'f32'),
            read('ringCount', ringCount),
            read('areas', ringAreas, 'f32'),
            read('isHole', ringIsHole),
            read('ringBounds', ringBounds as GraphDataView<'float32'>, 'f32'),
            write('shells', ringShells),
            ...(ringGroups ? [read('ringGroups', ringGroups)] : [])
          ],
          ringCapacity,
          `shells[shellsOffset + index] = ${none};
  let rings = ringCount[ringCountOffset];
  if (index >= rings) { return; }
  if (isHole[isHoleOffset + index] == 0u) {
    shells[shellsOffset + index] = index;
    return;
  }
  let first = ringOffsets[ringOffsetsOffset + index];
  let a = vec2f(positions[positionsOffset + 2u * first], positions[positionsOffset + 2u * first + 1u]);
  let b = vec2f(positions[positionsOffset + 2u * first + 2u], positions[positionsOffset + 2u * first + 3u]);
  let probe = 0.5 * (a + b);
  var best = ${none};
  var bestArea = 0.0;
  let holeArea = abs(areas[areasOffset + index]);
  for (var shell = 0u; shell < rings; shell++) {
    if (isHole[isHoleOffset + shell] != 0u) { continue; }
    ${ringGroups ? 'if (ringGroups[ringGroupsOffset + shell] != ringGroups[ringGroupsOffset + index]) { continue; }' : ''}
    let area = abs(areas[areasOffset + shell]);
    // A bidirected boundary produces an equal-area reverse walk for the unbounded side. It cannot
    // own itself as a hole; a containing shell must have strictly larger area.
    if (area <= holeArea) { continue; }
    if (best != ${none} && area >= bestArea) { continue; }
    // A probe outside the shell's bounding box (or left of it, where a closed ring crosses the ray
    // an even number of times) cannot be inside: skip the vertex walk.
    let shellBounds = vec4f(
      ringBounds[ringBoundsOffset + 4u * shell], ringBounds[ringBoundsOffset + 4u * shell + 1u],
      ringBounds[ringBoundsOffset + 4u * shell + 2u], ringBounds[ringBoundsOffset + 4u * shell + 3u]);
    if (probe.y < shellBounds.y || probe.y > shellBounds.w || probe.x < shellBounds.x || probe.x > shellBounds.z) { continue; }
    var inside = false;
    let shellFirst = ringOffsets[ringOffsetsOffset + shell];
    let shellLast = ringOffsets[ringOffsetsOffset + shell + 1u];
    for (var vertex = shellFirst; vertex + 1u < shellLast; vertex++) {
      let p = vec2f(positions[positionsOffset + 2u * vertex], positions[positionsOffset + 2u * vertex + 1u]);
      let q = vec2f(positions[positionsOffset + 2u * vertex + 2u], positions[positionsOffset + 2u * vertex + 3u]);
      if ((p.y > probe.y) != (q.y > probe.y)) {
        let crossing = p.x + (probe.y - p.y) * (q.x - p.x) / (q.y - p.y);
        if (crossing > probe.x) { inside = !inside; }
      }
    }
    if (inside) {
      best = shell;
      bestArea = area;
    }
  }
  shells[shellsOffset + index] = best;`
        )
      );
    }

    if (output.polygons && ringShells) {
      const {polygons} = output;
      const polygonKeys = u32('polygon-keys', ringCapacity);
      const polygonIds = u32('polygon-ids', ringCapacity);
      const sortedPolygonKeys = u32('sorted-polygon-keys', ringCapacity);
      const sortedPolygonIds = u32('sorted-polygon-ids', ringCapacity);
      const lengths = u32('polygon-lengths', ringCapacity);
      const shellFlags = u32('polygon-shell-flags', ringCapacity);
      const lengthScan = u32('polygon-length-scan', ringCapacity);
      const shellScan = u32('polygon-shell-scan', ringCapacity);
      const polygonStarts = u32('polygon-starts', ringCapacity);
      const validRings = u32('polygon-valid-rings', 1);
      nodes.push(
        kernel(
          'polygon-keys',
          [
            read('ringCount', ringCount),
            read('areas', ringAreas, 'f32'),
            read('isHole', ringIsHole),
            read('shells', ringShells),
            write('keys', polygonKeys),
            write('ids', polygonIds)
          ],
          ringCapacity,
          `ids[idsOffset + index] = index;
  var key = ${none};
  if (index < ringCount[ringCountOffset] && abs(areas[areasOffset + index]) > 0.0 && shells[shellsOffset + index] != ${none}) {
    key = shells[shellsOffset + index] * 2u + isHole[isHoleOffset + index];
  }
  keys[keysOffset + index] = key;`
        ),
        ...new GPUSort({
          id: `${id}-polygon-sort`,
          keys: polygonKeys,
          values: polygonIds,
          outputKeys: sortedPolygonKeys,
          outputValues: sortedPolygonIds
        }).getCommandNodes(graph),
        kernel(
          'polygon-lengths',
          [
            read('ringOffsets', output.ringOffsets),
            read('sortedKeys', sortedPolygonKeys),
            read('sortedIds', sortedPolygonIds),
            write('lengths', lengths),
            write('shellFlags', shellFlags)
          ],
          ringCapacity,
          `let key = sortedKeys[sortedKeysOffset + index];
  let ring = sortedIds[sortedIdsOffset + index];
  let isValid = key != ${none};
  lengths[lengthsOffset + index] = select(0u, ringOffsets[ringOffsetsOffset + ring + 1u] - ringOffsets[ringOffsetsOffset + ring] - 1u, isValid);
  shellFlags[shellFlagsOffset + index] = select(0u, 1u, isValid && (key & 1u) == 0u);`
        ),
        ...new GPUScan({
          id: `${id}-polygon-length-scan`,
          input: lengths,
          output: lengthScan,
          mode: 'inclusive'
        }).getCommandNodes(graph),
        ...new GPUScan({
          id: `${id}-polygon-shell-scan`,
          input: shellFlags,
          output: shellScan,
          mode: 'inclusive'
        }).getCommandNodes(graph),
        kernel(
          'polygon-starts',
          [
            read('shellFlags', shellFlags),
            read('shellScan', shellScan),
            write('polygonStarts', polygonStarts)
          ],
          ringCapacity,
          `if (shellFlags[shellFlagsOffset + index] != 0u) {
    polygonStarts[polygonStartsOffset + shellScan[shellScanOffset + index] - 1u] = index;
  }`
        ),
        ...(polygons.sourceIds && ringGroups
          ? [
              kernel(
                'polygon-groups',
                [
                  read('shellScan', shellScan),
                  read('polygonStarts', polygonStarts),
                  read('sortedIds', sortedPolygonIds),
                  read('ringGroups', ringGroups),
                  write('sourceIds', polygons.sourceIds)
                ],
                ringCapacity,
                `var group = ${none};
  if (index < shellScan[shellScanOffset + ${ringCapacity - 1}u]) {
    group = ringGroups[ringGroupsOffset + sortedIds[sortedIdsOffset + polygonStarts[polygonStartsOffset + index]]];
  }
  sourceIds[sourceIdsOffset + index] = group;`
              )
            ]
          : []),
        kernel(
          'polygon-counts',
          [
            read('sortedKeys', sortedPolygonKeys),
            read('shellScan', shellScan),
            write('validRings', validRings),
            write('featureOffsets', polygons.featureOffsets)
          ],
          1,
          `var low = 0u;
  var high = ${ringCapacity}u;
  while (low < high) {
    let middle = low + (high - low) / 2u;
    if (sortedKeys[sortedKeysOffset + middle] == ${none}) { high = middle; } else { low = middle + 1u; }
  }
  validRings[validRingsOffset] = low;
  let polygonCount = shellScan[shellScanOffset + ${ringCapacity - 1}u];
  for (var feature = 0u; feature <= ${ringCapacity}u; feature++) {
    featureOffsets[featureOffsetsOffset + feature] = min(feature, polygonCount);
  }`
        ),
        kernel(
          'polygon-offsets',
          [
            read('lengthScan', lengthScan),
            read('shellScan', shellScan),
            read('polygonStarts', polygonStarts),
            read('validRings', validRings),
            write('ringOffsetsOut', polygons.ringOffsets),
            write('polygonOffsetsOut', polygons.polygonOffsets)
          ],
          ringCapacity + 1,
          `let polygonCount = shellScan[shellScanOffset + ${ringCapacity - 1}u];
  var ringOffset = 0u;
  if (index > 0u) { ringOffset = lengthScan[lengthScanOffset + index - 1u]; }
  ringOffsetsOut[ringOffsetsOutOffset + index] = ringOffset;
  var polygonOffset = validRings[validRingsOffset];
  if (index < polygonCount) { polygonOffset = polygonStarts[polygonStartsOffset + index]; }
  polygonOffsetsOut[polygonOffsetsOutOffset + index] = polygonOffset;`
        ),
        kernel(
          'polygon-gather',
          [
            read('sortedIds', sortedPolygonIds),
            read('ringOffsets', output.ringOffsets),
            read('positions', output.positions, 'f32'),
            read('ringOffsetsOut', polygons.ringOffsets),
            write('positionsOut', polygons.positions, 'f32')
          ],
          vertexCapacity,
          `if (index >= ringOffsetsOut[ringOffsetsOutOffset + ${ringCapacity}u]) { return; }
  var low = 0u;
  var high = ${ringCapacity}u;
  while (low + 1u < high) {
    let middle = low + (high - low) / 2u;
    if (ringOffsetsOut[ringOffsetsOutOffset + middle] <= index) { low = middle; } else { high = middle; }
  }
  let ring = sortedIds[sortedIdsOffset + low];
  let source = ringOffsets[ringOffsetsOffset + ring] + (index - ringOffsetsOut[ringOffsetsOutOffset + low]);
  positionsOut[positionsOutOffset + 2u * index] = positions[positionsOffset + 2u * source];
  positionsOut[positionsOutOffset + 2u * index + 1u] = positions[positionsOffset + 2u * source + 1u];`
        )
      );
    }

    // 9. Publish scalars.
    nodes.push(
      kernel(
        'publish',
        [
          read('ringScan', ringScan),
          read('ringCount', ringCount),
          write('outCount', output.count),
          write('outOverflow', output.overflow),
          ...(output.requiredCount ? [write('outTotal', output.requiredCount)] : [])
        ],
        1,
        `let total = ringScan[ringScanOffset + ${segments - 1}u];
  let written = ringCount[ringCountOffset];
  outCount[outCountOffset] = written;
  outOverflow[outOverflowOffset] = select(0u, 1u, written < total);
  ${output.requiredCount ? 'outTotal[outTotalOffset] = total;' : ''}`
      )
    );
    if (output.openSegmentCount || output.touchingSegmentCount) {
      nodes.push(
        kernel(
          'publish-counters',
          [
            read('counters', counters),
            ...(output.openSegmentCount ? [write('outOpen', output.openSegmentCount)] : []),
            ...(output.touchingSegmentCount
              ? [write('outTouching', output.touchingSegmentCount)]
              : [])
          ],
          1,
          `${output.openSegmentCount ? 'outOpen[outOpenOffset] = counters[countersOffset] - counters[countersOffset + 2u];' : ''}
  ${output.touchingSegmentCount ? 'outTouching[outTouchingOffset] = counters[countersOffset + 1u];' : ''}`
        )
      );
    }
    return nodes;

    function createJumpNode(mode: 'min' | 'sum', round: string, source: number) {
      const target = 1 - source;
      return createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-${mode}-jump-${round}`,
        operation: OPERATION,
        variant: `${mode}-jump`,
        bindings: [
          read('pointerIn', pointerSets[source]),
          read('valueIn', valueSets[source]),
          write('pointerOut', pointerSets[target]),
          write('valueOut', valueSets[target])
        ],
        invocationCount: segments,
        body: `let successor = pointerIn[pointerInOffset + index];
  var value = valueIn[valueInOffset + index];
  var pointer = successor;
  if (successor != ${none}) {
    value = ${
      mode === 'min'
        ? 'min(value, valueIn[valueInOffset + successor])'
        : 'value + valueIn[valueInOffset + successor]'
    };
    pointer = pointerIn[pointerInOffset + successor];
  }
  pointerOut[pointerOutOffset + index] = pointer;
  valueOut[valueOutOffset + index] = value;`
      });
    }
  }
}

function getOutputViews(output: GPUSegmentRingAssemblyOutput) {
  const {kind: _kind, ...polygonViews} = output.polygons ?? {};
  return [
    output.ringOffsets,
    output.positions,
    output.ringAreas,
    output.ringIsHole,
    output.ringShells,
    output.ringGroups,
    ...Object.values(polygonViews),
    output.segmentRings,
    output.segmentVertices,
    output.segmentFlags,
    output.count,
    output.overflow,
    output.requiredCount,
    output.openSegmentCount,
    output.touchingSegmentCount
  ];
}
