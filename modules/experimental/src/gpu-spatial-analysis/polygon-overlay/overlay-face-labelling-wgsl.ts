// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import {EXACT_ORIENTATION_WGSL} from '../../utils/exact-orientation-wgsl';
import {EXACT_GEOMETRY_WGSL} from '../../utils/exact-geometry-wgsl';

/**
 * Face labelling for the polygon overlay: WGSL for the exact ordering of ray crossings, the ray
 * shoot over a segment BVH and the pointer-jumping rounds.
 *
 * The arrangement edges are canonical lattice segments `lo -> hi` (`lo` smaller in (y, x)), so
 * every edge crossing a horizontal line goes up. A group of coincident edges is probed on the
 * right-hand side of its midpoint (`ExactProbe`, side -1); the ray toward +x hits the nearest
 * group to its right, whose left-hand face is the probe's face. The "visibility" relation of
 * non-crossing segments is acyclic, so those parents form a forest and a sum along parent chains
 * (pointer jumping) labels every group.
 *
 * Lattice values are integers below 2^23, so all coordinates, half-integers and differences are
 * exact in f32 and in i32.
 *
 * @internal
 */

/** Sentinel group index: no parent. */
export const FACE_LABELLING_NONE = 0xffffffff;

/**
 * Exact ordering of two crossed edges along the probe ray.
 *
 * `compareCrossings(first, second, doubledHeight, offsetSign)` returns the sign of `x1 - x2`, the
 * difference of the x coordinates where the two edges (each `vec4f(lo.x, lo.y, hi.x, hi.y)`, with
 * `hi.y > lo.y`) meet the ray at height `y* = Y + offsetSign * e` (`doubledHeight` = 2 Y, an
 * integer; `e` infinitesimal). `x_t(Y) = N_t / (2 dy_t)` with `N_t = 2 lo.x dy + (2 Y - 2 lo.y) dx`
 * below 2^47, so `x1 < x2` is the sign of `N1 dy2 - N2 dy1` (about 2^70): evaluated in 96-bit two's
 * complement arithmetic on 16-bit limbs. Equal `x(Y)` (a shared endpoint at height `Y`) is decided
 * by the slopes: `x1 - x2 = offsetSign * e * (dx1/dy1 - dx2/dy2)`. Equal slopes return 0.
 *
 * @internal
 */
export const FACE_CROSSING_ORDER_WGSL = /* wgsl */ `
fn wideSetInteger(destination: ptr<function, array<u32, 6>>, value: i32) {
  let bits = bitcast<u32>(value);
  let extension = select(0u, 0xffffu, value < 0);
  (*destination)[0] = bits & 0xffffu;
  (*destination)[1] = (bits >> 16u) & 0xffffu;
  for (var limb = 2u; limb < 6u; limb++) {
    (*destination)[limb] = extension;
  }
}

// product = left * right modulo 2^96 (two's complement, limbs in [0, 65535]).
fn wideMultiply(
  left: ptr<function, array<u32, 6>>,
  right: ptr<function, array<u32, 6>>,
  product: ptr<function, array<u32, 6>>
) {
  var columns: array<u32, 7>;
  for (var i = 0u; i < 6u; i++) {
    for (var j = 0u; i + j < 6u; j++) {
      let partial = (*left)[i] * (*right)[j];
      columns[i + j] = columns[i + j] + (partial & 0xffffu);
      columns[i + j + 1u] = columns[i + j + 1u] + ((partial >> 16u) & 0xffffu);
    }
  }
  var carry = 0u;
  for (var limb = 0u; limb < 6u; limb++) {
    let total = columns[limb] + carry;
    (*product)[limb] = total & 0xffffu;
    carry = (total >> 16u) & 0xffffu;
  }
}

fn wideAdd(
  left: ptr<function, array<u32, 6>>,
  right: ptr<function, array<u32, 6>>,
  sum: ptr<function, array<u32, 6>>
) {
  var carry = 0u;
  for (var limb = 0u; limb < 6u; limb++) {
    let total = (*left)[limb] + (*right)[limb] + carry;
    (*sum)[limb] = total & 0xffffu;
    carry = (total >> 16u) & 0xffffu;
  }
}

fn wideSubtract(
  left: ptr<function, array<u32, 6>>,
  right: ptr<function, array<u32, 6>>,
  difference: ptr<function, array<u32, 6>>
) {
  var carry = 1u;
  for (var limb = 0u; limb < 6u; limb++) {
    let total = (*left)[limb] + ((*right)[limb] ^ 0xffffu) + carry;
    (*difference)[limb] = total & 0xffffu;
    carry = (total >> 16u) & 0xffffu;
  }
}

fn wideSign(value: ptr<function, array<u32, 6>>) -> i32 {
  if ((*value)[5] >= 0x8000u) { return -1; }
  for (var limb = 0u; limb < 6u; limb++) {
    if ((*value)[limb] != 0u) { return 1; }
  }
  return 0;
}

// N = 2 lo.x dy + (2 Y - 2 lo.y) dx, so that x(Y) = N / (2 dy).
fn crossingNumerator(edge: vec4f, doubledHeight: i32, numerator: ptr<function, array<u32, 6>>) {
  let lowX = i32(edge.x);
  let lowY = i32(edge.y);
  let deltaX = i32(edge.z) - lowX;
  let deltaY = i32(edge.w) - lowY;
  var factorA: array<u32, 6>;
  var factorB: array<u32, 6>;
  var factorC: array<u32, 6>;
  var factorD: array<u32, 6>;
  var termOne: array<u32, 6>;
  var termTwo: array<u32, 6>;
  wideSetInteger(&factorA, 2 * lowX);
  wideSetInteger(&factorB, deltaY);
  wideSetInteger(&factorC, doubledHeight - 2 * lowY);
  wideSetInteger(&factorD, deltaX);
  wideMultiply(&factorA, &factorB, &termOne);
  wideMultiply(&factorC, &factorD, &termTwo);
  wideAdd(&termOne, &termTwo, numerator);
}

fn compareCrossings(first: vec4f, second: vec4f, doubledHeight: i32, offsetSign: i32) -> i32 {
  var numeratorOne: array<u32, 6>;
  var numeratorTwo: array<u32, 6>;
  var heightOne: array<u32, 6>;
  var heightTwo: array<u32, 6>;
  var productOne: array<u32, 6>;
  var productTwo: array<u32, 6>;
  var difference: array<u32, 6>;
  crossingNumerator(first, doubledHeight, &numeratorOne);
  crossingNumerator(second, doubledHeight, &numeratorTwo);
  wideSetInteger(&heightOne, i32(first.w) - i32(first.y));
  wideSetInteger(&heightTwo, i32(second.w) - i32(second.y));
  wideMultiply(&numeratorOne, &heightTwo, &productOne);
  wideMultiply(&numeratorTwo, &heightOne, &productTwo);
  wideSubtract(&productOne, &productTwo, &difference);
  let order = wideSign(&difference);
  if (order != 0) { return order; }
  // Same x at height Y: the order at Y +- e follows the slopes dx/dy.
  var slopeOne: array<u32, 6>;
  var slopeTwo: array<u32, 6>;
  wideSetInteger(&slopeOne, i32(first.z) - i32(first.x));
  wideSetInteger(&slopeTwo, i32(second.z) - i32(second.x));
  wideMultiply(&slopeOne, &heightTwo, &productOne);
  wideMultiply(&slopeTwo, &heightOne, &productTwo);
  wideSubtract(&productOne, &productTwo, &difference);
  return offsetSign * wideSign(&difference);
}
`;

/** Names of the bindings the ray-shoot kernel needs, for callers that build its node. @internal */
export type FaceRayShootOptions = {
  /** Internal node count of the group BVH. */
  internalNodeCount: number;
  /** Row of the `counts` binding that holds the number of groups of this level. */
  countRow: number;
  /** Restrict hits to groups of the probe's polygon (level 1). Needs the `groupPolygons` binding. */
  samePolygon: boolean;
};

/**
 * Module-scope WGSL of the ray-shoot kernel: the exact helpers, the group readers and the BVH walk.
 * Bindings: `counts` (u32), `groupLattice` (f32, `vec4f(lo, hi)` per group), `groupPolygons` (u32,
 * only with `samePolygon`), `nodeMinima`, `nodeMaxima` (f32), `leafIds` (u32), `parents` (u32,
 * read_write). `exactProbeCrossing` and `compareCrossings` each have exactly one call site (inside
 * the traversal loop), and `orientSign` is reached only through `exactOrientSigns`.
 *
 * @internal
 */
export function getFaceRayShootDeclarationsWGSL(options: FaceRayShootOptions): string {
  return /* wgsl */ `
const INTERNAL_NODE_COUNT: u32 = ${options.internalNodeCount}u;
const COUNT_ROW: u32 = ${options.countRow}u;
const NO_GROUP: u32 = 0xffffffffu;
const FAR_X: f32 = 3.0e38;
${EXACT_ORIENTATION_WGSL}
${EXACT_GEOMETRY_WGSL}
${FACE_CROSSING_ORDER_WGSL}
fn readGroup(group: u32) -> vec4f {
  let base = groupLatticeOffset + group * 4u;
  return vec4f(groupLattice[base], groupLattice[base + 1u], groupLattice[base + 2u], groupLattice[base + 3u]);
}
// Whether the box of 'node' can hold an edge that the ray from (rayX, rayY) toward +x meets before
// an edge whose largest x is 'limitX'. All values are exact lattice numbers.
fn nodeMeetsRay(node: u32, rayX: f32, rayY: f32, limitX: f32) -> bool {
  let component = node * 2u;
  let minimum = vec2f(nodeMinima[nodeMinimaOffset + component], nodeMinima[nodeMinimaOffset + component + 1u]);
  let maximum = vec2f(nodeMaxima[nodeMaximaOffset + component], nodeMaxima[nodeMaximaOffset + component + 1u]);
  return minimum.y <= rayY && rayY <= maximum.y && rayX <= maximum.x && minimum.x <= limitX;
}`;
}

/**
 * Body of the ray-shoot kernel (one invocation per group slot). Writes `parents[index]`: the
 * nearest group crossed by the ray toward +x from the right-hand side of the group's midpoint, or
 * `NO_GROUP`.
 *
 * @internal
 */
export function getFaceRayShootBodyWGSL(options: {samePolygon: boolean}): string {
  return /* wgsl */ `var parent = NO_GROUP;
  let groupCount = counts[countsOffset + COUNT_ROW];
  if (index < groupCount) {
    let edge = readGroup(index);
    let probe = ExactProbe(0.5 * (edge.xy + edge.zw), edge.xy, edge.zw, -1);
    let rayHeight = probe.point.y;
    let doubledHeight = i32(2.0 * rayHeight);
    // The probe sits at Y + o with o infinitesimal: negative exactly when a vertex at height Y
    // counts as above the probe.
    let offsetSign = select(1, -1, exactProbeAbove(probe, probe.point));
    ${options.samePolygon ? 'let polygon = groupPolygons[groupPolygonsOffset + index];' : ''}
    var bestEdge = vec4f(0.0);
    var node = 0u;
    loop {
      let limitX = select(max(bestEdge.x, bestEdge.z), FAR_X, parent == NO_GROUP);
      if (nodeMeetsRay(node, probe.point.x, rayHeight, limitX)) {
        if (node < INTERNAL_NODE_COUNT) {
          node = node * 2u + 1u;
          continue;
        }
        let other = leafIds[leafIdsOffset + node - INTERNAL_NODE_COUNT];
        if (other < groupCount${options.samePolygon ? ' && groupPolygons[groupPolygonsOffset + other] == polygon' : ''}) {
          let candidate = readGroup(other);
          if (exactProbeCrossing(probe, candidate.xy, candidate.zw).x == 1) {
            // Both edges are crossed: keep the one meeting the ray at the smaller x.
            if (parent == NO_GROUP || compareCrossings(candidate, bestEdge, doubledHeight, offsetSign) < 0) {
              parent = other;
              bestEdge = candidate;
            }
          }
        }
      }
      // Climb while the node is a right child, then step to the right sibling.
      loop {
        if (node == 0u || (node & 1u) == 1u) { break; }
        node = (node - 1u) / 2u;
      }
      if (node == 0u) { break; }
      node = node + 1u;
    }
  }
  parents[parentsOffset + index] = parent;`;
}

/** WGSL `fn insideWinding(winding: i32) -> bool` for a fill rule. @internal */
export function getInsideWindingWGSL(
  fillRule: 'even-odd' | 'positive' | 'negative' | 'nonzero'
): string {
  const test =
    fillRule === 'even-odd'
      ? '(winding & 1) != 0'
      : fillRule === 'positive'
        ? 'winding > 0'
        : fillRule === 'negative'
          ? 'winding < 0'
          : 'winding != 0';
  return `fn insideWinding(winding: i32) -> bool { return ${test}; }`;
}

/**
 * Pointer-jumping rounds: given `parents` (a forest, `FACE_LABELLING_NONE` for roots) and
 * `values` (`componentCount` i32 per group, initially the contribution of the parent), after
 * `rounds` rounds `values` holds the sum over all ancestors whenever `2^rounds` exceeds the
 * deepest chain. Ping-pong between two buffer pairs; every round has the same WGSL and pipeline
 * constants (library rule 12), only the bound buffers alternate. Returns the nodes and the views
 * holding the final parents and values.
 *
 * @internal
 */
export function createPointerJumpNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    parents: GraphDataView<'uint32'>;
    values: GraphDataView<'uint32'>;
    componentCount: number;
    rounds: number;
  }
): {
  nodes: readonly GPUCommandNode<Parameters>[];
  values: GraphDataView<'uint32'>;
} {
  const {id, operation, componentCount} = props;
  const groupCapacity = props.parents.length;
  const alternate = {
    parents: createTransientView(graph, `${id}-parents-alternate`, 'uint32', groupCapacity),
    values: createTransientView(
      graph,
      `${id}-values-alternate`,
      'uint32',
      groupCapacity * componentCount
    )
  };
  let source = {parents: props.parents, values: props.values};
  let destination = alternate;
  const nodes: GPUCommandNode<Parameters>[] = [];
  for (let round = 0; round < props.rounds; round++) {
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-round-${round}`,
        operation,
        variant: 'pointer-jump',
        bindings: [
          {name: 'parentsIn', view: source.parents, type: 'u32', access: 'read'},
          {name: 'valuesIn', view: source.values, type: 'i32', access: 'read'},
          {name: 'parentsOut', view: destination.parents, type: 'u32', access: 'read_write'},
          {name: 'valuesOut', view: destination.values, type: 'i32', access: 'read_write'}
        ],
        invocationCount: groupCapacity,
        declarations: `const COMPONENT_COUNT: u32 = ${componentCount}u;
const NO_GROUP: u32 = 0xffffffffu;`,
        body: `let parent = parentsIn[parentsInOffset + index];
  var grandparent = NO_GROUP;
  if (parent != NO_GROUP) { grandparent = parentsIn[parentsInOffset + parent]; }
  for (var component = 0u; component < COMPONENT_COUNT; component++) {
    var value = valuesIn[valuesInOffset + index * COMPONENT_COUNT + component];
    if (parent != NO_GROUP) {
      value = value + valuesIn[valuesInOffset + parent * COMPONENT_COUNT + component];
    }
    valuesOut[valuesOutOffset + index * COMPONENT_COUNT + component] = value;
  }
  parentsOut[parentsOutOffset + index] = grandparent;`
      })
    );
    [source, destination] = [destination, source];
  }
  // After the last swap 'source' holds the newest buffers.
  return {nodes, values: source.values};
}

/** Number of pointer-jumping rounds that cover chains of up to `groupCapacity` groups. @internal */
export function getPointerJumpRoundCount(groupCapacity: number): number {
  return Math.ceil(Math.log2(Math.max(groupCapacity, 2))) + 1;
}
