// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../../../src/analysis-core/contracts/spatial-context';
import {
  createTransientView,
  GPUCompaction,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {
  validateGraphViewsBelongToGraph,
  validatePackedView
} from '../../../../src/utils/gpu-contributor-utils';
import {createStableLexicographicIndexSortNodes} from '../../../../src/utils/stable-lexicographic-index-sort';
import {createWGSLKernelNode} from '../../../../src/utils/wgsl-kernel-nodes';
import {
  validateGPUStatusPort,
  type GPUBoundedResultStatusPort
} from '../../../../src/analysis-core/contracts/status';
import {EXACT_ORIENTATION_WGSL} from '../../../../src/utils/exact-orientation-wgsl';
import {IS_FINITE_BITS_WGSL} from '../../../../src/utils/wgsl-common';

const OPERATION = 'GPUDelaunayTessellation';
const REBUILD_WORKGROUP_SIZE = 256;
/** Sort key for non-finite sites: a NaN bit pattern, which no finite coordinate key can equal. */
const SITE_KEY_WGSL = /* wgsl */ `const INVALID_SITE_KEY: u32 = 0xffffffffu;`;
const REBUILD_SCAN_WGSL = /* wgsl */ `
const REBUILD_WORKGROUP_SIZE: u32 = ${REBUILD_WORKGROUP_SIZE}u;
var<workgroup> scanValues: array<u32, ${REBUILD_WORKGROUP_SIZE}>;
// Hillis-Steele inclusive scan; the total is left in scanValues[REBUILD_WORKGROUP_SIZE - 1].
fn inclusiveScan(lane: u32, value: u32) -> u32 {
  workgroupBarrier();
  scanValues[lane] = value;
  workgroupBarrier();
  for (var stride = 1u; stride < REBUILD_WORKGROUP_SIZE; stride = stride << 1u) {
    var addend = 0u;
    if (lane >= stride) { addend = scanValues[lane - stride]; }
    workgroupBarrier();
    scanValues[lane] = scanValues[lane] + addend;
    workgroupBarrier();
  }
  return scanValues[lane];
}`;

/**
 * WGSL `fn isRebuildItem(item, oldCount, insertionRow) -> bool` of the stable rebuild compaction.
 * Items `[0, SCRATCH_TRIANGLE_CAPACITY)` are the triangles (kept when not in the cavity); items
 * from `SCRATCH_TRIANGLE_CAPACITY` on are the `3 * capacity` triangle edges (a new triangle per
 * cavity boundary edge; none when the site is a duplicate). Compacting the flags in item order
 * gives survivors first, then new triangles, both in ascending source order: the serial order.
 * Needs `badFlags`, `boundaryFlags` and `duplicateFlags` bindings.
 */
const REBUILD_ITEM_WGSL = /* wgsl */ `
fn isRebuildItem(item: u32, oldCount: u32, insertionRow: u32) -> bool {
  if (item < SCRATCH_TRIANGLE_CAPACITY) {
    return item < oldCount && badFlags[badFlagsOffset + item] == 0u;
  }
  let candidate = item - SCRATCH_TRIANGLE_CAPACITY;
  let candidateCount = select(3u * oldCount, 0u, duplicateFlags[duplicateFlagsOffset + insertionRow] != 0u);
  return candidate < candidateCount && boundaryFlags[boundaryFlagsOffset + candidate] != 0u;
}`;
/**
 * Adaptive-exact incircle predicate. `inCircumcircle(a, b, c, point)` is true when `point` lies
 * inside or exactly on the circumcircle of the counter-clockwise triangle `abc`.
 *
 * A float filter decides clear cases. Its bound, 1e-6 (about 16.8 f32 epsilons) times the
 * permanent, covers the roughly 11 epsilons of Shewchuk's incircle analysis with margin. The
 * analysis assumes each operation is rounded at most once. A fused multiply-add rounds less often,
 * and the three-term sum may be reassociated, so neither can invalidate the bound. Inputs that
 * could underflow fall through to the exact path: nonzero coordinates below 2^-30, or a permanent
 * below 2^-90. Overflow makes the bound infinite or the determinant NaN, and also falls through.
 *
 * The exact path aligns the eight coordinates to the smallest exponent, which makes them integers
 * of at most 79 bits. It then evaluates the determinant of exact coordinate differences in 16-bit
 * limb integer arithmetic, with sign-magnitude terms and separate positive and negative
 * accumulators. Integer arithmetic is used for the same reason as `exact-orientation-wgsl.ts`:
 * shader compilers may contract or reassociate float error-free transforms. When the coordinate
 * exponents span more than 55 bits, or a coordinate is not finite, the exact path returns 2 and
 * the caller uses the float sign.
 */
const ADAPTIVE_INCIRCLE_WGSL = /* wgsl */ `
const INCIRCLE_FILTER_BOUND: f32 = 1.0e-6;
// 2^-30: nonzero coordinates below this could underflow the float filter.
const INCIRCLE_TINY_COORDINATE: f32 = 9.313225746154785e-10;
// 2^-90: below this permanent, underflow error could exceed the filter slack.
const INCIRCLE_MINIMUM_PERMANENT: f32 = 8.077935669463161e-28;
const INCIRCLE_MAXIMUM_SPAN: i32 = 55;
// Workspace layout in 16-bit limbs: six 5-limb differences, two 11-limb scratch values, an
// 11-limb lift, an 11-limb minor, a 22-limb product, and 22-limb positive and negative sums.
const INCIRCLE_DIFFERENCES: u32 = 0u;
const INCIRCLE_SCRATCH_A: u32 = 30u;
const INCIRCLE_SCRATCH_B: u32 = 41u;
const INCIRCLE_LIFT: u32 = 52u;
const INCIRCLE_MINOR: u32 = 63u;
const INCIRCLE_PRODUCT: u32 = 74u;
const INCIRCLE_POSITIVE: u32 = 96u;
const INCIRCLE_NEGATIVE: u32 = 118u;
const INCIRCLE_WORKSPACE_LIMBS: u32 = 140u;

fn incircleCopy(work: ptr<function, array<u32, INCIRCLE_WORKSPACE_LIMBS>>, destination: u32, source: u32, limbs: u32) {
  for (var limb = 0u; limb < limbs; limb++) { (*work)[destination + limb] = (*work)[source + limb]; }
}

// destination += source over limbs; callers guarantee the sum fits.
fn incircleAdd(work: ptr<function, array<u32, INCIRCLE_WORKSPACE_LIMBS>>, destination: u32, source: u32, limbs: u32) {
  var carry = 0u;
  for (var limb = 0u; limb < limbs; limb++) {
    let sum = (*work)[destination + limb] + (*work)[source + limb] + carry;
    (*work)[destination + limb] = sum & 0xffffu;
    carry = sum >> 16u;
  }
}

// destination -= source over limbs; callers guarantee destination >= source.
fn incircleSubtract(work: ptr<function, array<u32, INCIRCLE_WORKSPACE_LIMBS>>, destination: u32, source: u32, limbs: u32) {
  var borrow = 0u;
  for (var limb = 0u; limb < limbs; limb++) {
    let difference = (*work)[destination + limb] + 0x10000u - (*work)[source + limb] - borrow;
    (*work)[destination + limb] = difference & 0xffffu;
    borrow = 1u - (difference >> 16u);
  }
}

fn incircleCompare(work: ptr<function, array<u32, INCIRCLE_WORKSPACE_LIMBS>>, firstOffset: u32, secondOffset: u32, limbs: u32) -> i32 {
  for (var limb = limbs; limb > 0u; limb--) {
    let left = (*work)[firstOffset + limb - 1u];
    let right = (*work)[secondOffset + limb - 1u];
    if (left != right) { return select(-1, 1, left > right); }
  }
  return 0;
}

// destination[0, leftLimbs + rightLimbs) = left * right; destination must not overlap either
// factor. Limbs stay below 2^16 after each row and the running carry is at most 2^16, so no
// u32 operation overflows.
fn incircleMultiply(
  work: ptr<function, array<u32, INCIRCLE_WORKSPACE_LIMBS>>,
  destination: u32, leftOffset: u32, leftLimbs: u32, rightOffset: u32, rightLimbs: u32
) {
  for (var limb = 0u; limb < leftLimbs + rightLimbs; limb++) { (*work)[destination + limb] = 0u; }
  for (var i = 0u; i < leftLimbs; i++) {
    let leftLimb = (*work)[leftOffset + i];
    if (leftLimb != 0u) {
      var carry = 0u;
      for (var j = 0u; j < rightLimbs; j++) {
        let product = leftLimb * (*work)[rightOffset + j];
        let sum = (*work)[destination + i + j] + (product & 0xffffu) + carry;
        (*work)[destination + i + j] = sum & 0xffffu;
        carry = (sum >> 16u) + (product >> 16u);
      }
      (*work)[destination + i + rightLimbs] = carry;
    }
  }
}

// Writes |value| * 2^-minimumExponent as a 5-limb integer.
fn incircleLoadMagnitude(
  work: ptr<function, array<u32, INCIRCLE_WORKSPACE_LIMBS>>, destination: u32, valueBits: u32, alignmentExponent: i32
) {
  let absoluteBits = valueBits & 0x7fffffffu;
  let biasedExponent = absoluteBits >> 23u;
  let significand = select(0x800000u | (absoluteBits & 0x7fffffu), absoluteBits, biasedExponent == 0u);
  let shift = select(i32(biasedExponent) - 150, -149, biasedExponent == 0u) - alignmentExponent;
  for (var limb = 0u; limb < 5u; limb++) {
    let relative = 16 * i32(limb) - shift;
    var value = 0u;
    if (relative >= 0 && relative < 24) {
      value = significand >> u32(relative);
    } else if (relative < 0 && relative > -16) {
      value = significand << u32(-relative);
    }
    (*work)[destination + limb] = value & 0xffffu;
  }
}

// Writes |first - second| into difference slot and returns whether it is negative.
fn incircleDifference(
  work: ptr<function, array<u32, INCIRCLE_WORKSPACE_LIMBS>>, slot: u32, first: u32, second: u32, alignmentExponent: i32
) -> bool {
  let differenceOffset = INCIRCLE_DIFFERENCES + 5u * slot;
  incircleLoadMagnitude(work, INCIRCLE_SCRATCH_A, first, alignmentExponent);
  incircleLoadMagnitude(work, INCIRCLE_SCRATCH_B, second, alignmentExponent);
  let firstNegative = (first >> 31u) == 1u;
  let secondNegative = (second >> 31u) == 1u;
  if (firstNegative != secondNegative) {
    incircleCopy(work, differenceOffset, INCIRCLE_SCRATCH_A, 5u);
    incircleAdd(work, differenceOffset, INCIRCLE_SCRATCH_B, 5u);
    return firstNegative;
  }
  if (incircleCompare(work, INCIRCLE_SCRATCH_A, INCIRCLE_SCRATCH_B, 5u) >= 0) {
    incircleCopy(work, differenceOffset, INCIRCLE_SCRATCH_A, 5u);
    incircleSubtract(work, differenceOffset, INCIRCLE_SCRATCH_B, 5u);
    return firstNegative;
  }
  incircleCopy(work, differenceOffset, INCIRCLE_SCRATCH_B, 5u);
  incircleSubtract(work, differenceOffset, INCIRCLE_SCRATCH_A, 5u);
  return !firstNegative;
}

// Adds sign * lift(x, y) * (p * q - r * t) for difference slots to the exact sums.
fn incircleAccumulateRow(
  work: ptr<function, array<u32, INCIRCLE_WORKSPACE_LIMBS>>,
  negatives: ptr<function, array<u32, 6>>,
  x: u32, y: u32, p: u32, q: u32, r: u32, t: u32, rowNegative: bool
) {
  let differences = INCIRCLE_DIFFERENCES;
  incircleMultiply(work, INCIRCLE_SCRATCH_A, differences + 5u * x, 5u, differences + 5u * x, 5u);
  incircleMultiply(work, INCIRCLE_SCRATCH_B, differences + 5u * y, 5u, differences + 5u * y, 5u);
  (*work)[INCIRCLE_SCRATCH_A + 10u] = 0u;
  (*work)[INCIRCLE_SCRATCH_B + 10u] = 0u;
  incircleCopy(work, INCIRCLE_LIFT, INCIRCLE_SCRATCH_A, 11u);
  incircleAdd(work, INCIRCLE_LIFT, INCIRCLE_SCRATCH_B, 11u);

  incircleMultiply(work, INCIRCLE_SCRATCH_A, differences + 5u * p, 5u, differences + 5u * q, 5u);
  incircleMultiply(work, INCIRCLE_SCRATCH_B, differences + 5u * r, 5u, differences + 5u * t, 5u);
  (*work)[INCIRCLE_SCRATCH_A + 10u] = 0u;
  (*work)[INCIRCLE_SCRATCH_B + 10u] = 0u;
  let firstNegative = ((*negatives)[p] ^ (*negatives)[q]) == 1u;
  let secondNegative = ((*negatives)[r] ^ (*negatives)[t]) == 1u;
  var minorNegative = firstNegative;
  if (firstNegative != secondNegative) {
    incircleCopy(work, INCIRCLE_MINOR, INCIRCLE_SCRATCH_A, 11u);
    incircleAdd(work, INCIRCLE_MINOR, INCIRCLE_SCRATCH_B, 11u);
  } else if (incircleCompare(work, INCIRCLE_SCRATCH_A, INCIRCLE_SCRATCH_B, 11u) >= 0) {
    incircleCopy(work, INCIRCLE_MINOR, INCIRCLE_SCRATCH_A, 11u);
    incircleSubtract(work, INCIRCLE_MINOR, INCIRCLE_SCRATCH_B, 11u);
  } else {
    incircleCopy(work, INCIRCLE_MINOR, INCIRCLE_SCRATCH_B, 11u);
    incircleSubtract(work, INCIRCLE_MINOR, INCIRCLE_SCRATCH_A, 11u);
    minorNegative = !firstNegative;
  }

  incircleMultiply(work, INCIRCLE_PRODUCT, INCIRCLE_LIFT, 11u, INCIRCLE_MINOR, 11u);
  if (minorNegative != rowNegative) {
    incircleAdd(work, INCIRCLE_NEGATIVE, INCIRCLE_PRODUCT, 22u);
  } else {
    incircleAdd(work, INCIRCLE_POSITIVE, INCIRCLE_PRODUCT, 22u);
  }
}

// Exact sign of the incircle determinant: -1, 0 or 1, or 2 when it cannot be evaluated.
fn incircleExact(a: vec2<f32>, b: vec2<f32>, c: vec2<f32>, point: vec2<f32>) -> i32 {
  var bits = array<u32, 8>(
    bitcast<u32>(a.x), bitcast<u32>(a.y), bitcast<u32>(b.x), bitcast<u32>(b.y),
    bitcast<u32>(c.x), bitcast<u32>(c.y), bitcast<u32>(point.x), bitcast<u32>(point.y));
  var minimumExponent: i32 = 100000;
  var maximumExponent: i32 = -100000;
  for (var coordinate = 0u; coordinate < 8u; coordinate++) {
    let magnitude = bits[coordinate] & 0x7fffffffu;
    if (magnitude >= 0x7f800000u) { return 2; }
    if (magnitude != 0u) {
      let exponentBits = magnitude >> 23u;
      let exponent = select(i32(exponentBits) - 150, -149, exponentBits == 0u);
      minimumExponent = min(minimumExponent, exponent);
      maximumExponent = max(maximumExponent, exponent);
    }
  }
  if (minimumExponent > maximumExponent) { return 0; }
  // Aligned coordinates have at most 24 + 55 = 79 bits, so differences fit 5 limbs, lifts and
  // minors 11 limbs, and the three 322-bit terms 22 limbs.
  if (maximumExponent - minimumExponent > INCIRCLE_MAXIMUM_SPAN) { return 2; }
  var work: array<u32, INCIRCLE_WORKSPACE_LIMBS>;
  // Sign bits of the six differences, 1 when negative.
  var negatives: array<u32, 6>;
  for (var slot = 0u; slot < 6u; slot++) {
    let negative = incircleDifference(&work, slot, bits[slot], bits[6u + (slot & 1u)], minimumExponent);
    negatives[slot] = select(0u, 1u, negative);
  }
  // Slots: 0 adx, 1 ady, 2 bdx, 3 bdy, 4 cdx, 5 cdy, all relative to point.
  // alift * (bdx cdy - cdx bdy) - blift * (adx cdy - cdx ady) + clift * (adx bdy - bdx ady)
  incircleAccumulateRow(&work, &negatives, 0u, 1u, 2u, 5u, 4u, 3u, false);
  incircleAccumulateRow(&work, &negatives, 2u, 3u, 0u, 5u, 4u, 1u, true);
  incircleAccumulateRow(&work, &negatives, 4u, 5u, 0u, 3u, 2u, 1u, false);
  return incircleCompare(&work, INCIRCLE_POSITIVE, INCIRCLE_NEGATIVE, 22u);
}

fn inCircumcircle(a: vec2<f32>, b: vec2<f32>, c: vec2<f32>, point: vec2<f32>) -> bool {
  let av = a - point;
  let bv = b - point;
  let cv = c - point;
  let bcLeft = bv.x * cv.y;
  let bcRight = cv.x * bv.y;
  let caLeft = cv.x * av.y;
  let caRight = av.x * cv.y;
  let abLeft = av.x * bv.y;
  let abRight = bv.x * av.y;
  let aLift = av.x * av.x + av.y * av.y;
  let bLift = bv.x * bv.x + bv.y * bv.y;
  let cLift = cv.x * cv.x + cv.y * cv.y;
  let determinant = aLift * (bcLeft - bcRight) + bLift * (caLeft - caRight) + cLift * (abLeft - abRight);
  let permanent = (abs(bcLeft) + abs(bcRight)) * aLift + (abs(caLeft) + abs(caRight)) * bLift +
    (abs(abLeft) + abs(abRight)) * cLift;
  let first = vec4<f32>(a, b);
  let second = vec4<f32>(c, point);
  let tiny = any((abs(first) < vec4<f32>(INCIRCLE_TINY_COORDINATE)) & (first != vec4<f32>(0.0))) ||
    any((abs(second) < vec4<f32>(INCIRCLE_TINY_COORDINATE)) & (second != vec4<f32>(0.0)));
  if (!tiny && permanent >= INCIRCLE_MINIMUM_PERMANENT) {
    let bound = INCIRCLE_FILTER_BOUND * permanent;
    if (determinant > bound) { return true; }
    if (-determinant > bound) { return false; }
  }
  let exact = incircleExact(a, b, c, point);
  // Unrepresentable inputs (non-finite or exponent span over 55 bits) keep the float sign.
  if (exact == 2) { return determinant >= 0.0; }
  return exact >= 0;
}`;

/**
 * Symbolic super vertices. Bowyer-Watson needs a triangle that contains every site, and the old
 * finite supertriangle (corners at the centre plus or minus 32 extents) dropped real hull
 * triangles whose circumcircle contained a corner. The three super vertices are now points at
 * infinity that are never materialized; every predicate that touches one is decided from exact
 * comparisons of real coordinates.
 *
 * Geometry. Super vertex k sits at `c + R_k d_k`, where `c` is any finite centre (it never
 * affects a sign), and the directions are slightly tilted axis directions with `e` infinitesimal:
 * `d_0 = (1, -e)` (towards +x), `d_1 = (e, 1)` (towards +y), `d_2 = (-1, -e)` (towards -x). They
 * positively span the plane, so the super triangle `(S0, S1, S2)` is counter-clockwise and
 * contains every finite site. The radii grow faster than any polynomial in the earlier ones:
 * `R_0 >> 1/e` and `R_(k+1) >> R_k^2 / e`. This is a real, finite configuration for small enough
 * `e` and large enough `R_k`, so the Delaunay triangulation of the sites and the super vertices
 * is a valid triangulation, and every sign below is the limit of the exact sign in it. Because
 * `e` is a tie-breaker, the tilt is applied lexicographically: first the axis comparison, then
 * the tilt term.
 *
 * Orientation `orient(x, y, p)` for a real site `p`:
 * - Two real vertices: exact `orientSign`.
 * - Real `r` and super `k`: `R_k cross(d_k, p - r)` dominates. For k = 0, 1, 2 the sign is
 *   `sign(p.y - r.y)` then `sign(p.x - r.x)`; `-sign(p.x - r.x)` then `sign(p.y - r.y)`; and
 *   `-sign(p.y - r.y)` then `sign(p.x - r.x)`. `p != r`, so it is never zero. Reversing the pair
 *   negates it.
 * - Two supers `i`, `j`: `R_i R_j cross(d_i, d_j)`, which is positive exactly when `j = i + 1
 *   (mod 3)`.
 *
 * In-circle `inside(x, y, z, p)` for a counter-clockwise triangle whose farthest super vertex
 * (highest index) is `z`, with `x`, `y` the other two in counter-clockwise order. A circle through
 * a point at infinity `z` is the half-plane left of the line `x -> y`, so the test is
 * `orient(x, y, p) > 0`: the leading term of the in-circle determinant is `|z|^2 orient(x, y, p)`.
 * - Three supers: `x -> y` is `S0 -> S1`, always positive, so every site is inside.
 * - Two supers: `orient` has one real vertex and one super, and is never zero.
 * - One super: `x`, `y` are real. When `p` is exactly on the line `xy` the chord is inside the
 *   circle for every radius, so `p` is inside iff it lies in the open segment `xy`, decided by
 *   exact coordinate comparisons.
 * Hull edges of the real sites therefore get a ghost triangle `(x, y, S)` that a site beyond the
 * edge destroys, and the real hull triangles are never lost to a super vertex.
 */
const SYMBOLIC_SUPER_WGSL = /* wgsl */ `
fn compareCoordinates(a: f32, b: f32) -> i32 {
  return select(select(0, -1, a < b), 1, a > b);
}
// Sign of orient(r, S_k, p): the tilted direction d_k decides, axis comparison first.
fn orientRealSuper(r: vec2<f32>, superIndex: u32, p: vec2<f32>) -> i32 {
  let dx = compareCoordinates(p.x, r.x);
  let dy = compareCoordinates(p.y, r.y);
  if (superIndex == 0u) { return select(dx, dy, dy != 0); }
  if (superIndex == 1u) { return select(dy, -dx, dx != 0); }
  return select(dx, -dy, dy != 0);
}
// Sign of orient(point(a), point(b), point(p)) for a real site p; a and b may be super vertices.
fn orientSymbolic(a: u32, b: u32, p: u32) -> i32 {
  let point = pointAt(p);
  if (a < POINT_COUNT && b < POINT_COUNT) { return orientSign(pointAt(a), pointAt(b), point); }
  if (a < POINT_COUNT) { return orientRealSuper(pointAt(a), b - POINT_COUNT, point); }
  if (b < POINT_COUNT) { return -orientRealSuper(pointAt(b), a - POINT_COUNT, point); }
  return select(-1, 1, b - POINT_COUNT == (a - POINT_COUNT + 1u) % 3u);
}`;

const SYMBOLIC_INCIRCLE_WGSL = /* wgsl */ `
// p is collinear with a and b: strictly inside the chord, exact comparisons only.
fn isInsideOpenSegment(a: vec2<f32>, b: vec2<f32>, p: vec2<f32>) -> bool {
  if (a.x != b.x) { return p.x > min(a.x, b.x) && p.x < max(a.x, b.x); }
  return p.y > min(a.y, b.y) && p.y < max(a.y, b.y);
}
// True when site pRow is inside or on the circumcircle of the counter-clockwise triangle, with
// super vertices handled symbolically (see SYMBOLIC_SUPER_WGSL).
fn isInsideCircumcircle(v0: u32, v1: u32, v2: u32, pRow: u32) -> bool {
  var rows = array<u32, 3>(v0, v1, v2);
  var farthestSlot = 3u;
  for (var slot = 0u; slot < 3u; slot++) {
    if (rows[slot] >= POINT_COUNT && (farthestSlot == 3u || rows[slot] > rows[farthestSlot])) {
      farthestSlot = slot;
    }
  }
  if (farthestSlot == 3u) {
    return inCircumcircle(pointAt(v0), pointAt(v1), pointAt(v2), pointAt(pRow));
  }
  let x = rows[(farthestSlot + 1u) % 3u];
  let y = rows[(farthestSlot + 2u) % 3u];
  let side = orientSymbolic(x, y, pRow);
  if (side != 0) { return side > 0; }
  return isInsideOpenSegment(pointAt(x), pointAt(y), pointAt(pRow));
}`;

/** Bounded Delaunay triangles and their exact completeness status. */
export type GPUDelaunayTessellationOutput = {
  /** Counter-clockwise triples of input point indices. */
  triangles: GraphDataView<'uint32x3'>;
  /** Canonical bounded-result status. `requiredCount` is exact. */
  status: GPUBoundedResultStatusPort & {
    requiredCount: GraphDataView<'uint32'>;
    /** Optional one-row number of input rows skipped because an identical earlier coordinate owns the site. */
    duplicateCount?: GraphDataView<'uint32'>;
  };
};

/** Properties for {@link GPUDelaunayTessellation}. */
export type GPUDelaunayTessellationProps = {
  /** Prefix for generated nodes and transients. Defaults to `'delaunay-tessellation'`. */
  id?: string;
  /**
   * Planar only. Spatial context of `positions`; omitted means native planar coordinates.
   * Longitude/latitude (or any other non-planar) contexts throw at construction: project
   * positions to a planar frame first, for example with `createGPULocalFrame` and
   * `GPULocalFrameProjection`, and pass the frame's context so its `frame` identity is kept.
   */
  spatialContext?: GPUSpatialContext;
  /** Packed finite planar point coordinates. */
  positions: GraphDataView<'float32x2'>;
  /** Caller-owned bounded output. */
  output: GPUDelaunayTessellationOutput;
};

/** Maximum triangle count for `pointCount` distinct planar sites in general position. */
export function getDelaunayMaximumTriangleCount(pointCount: number): number {
  if (!Number.isSafeInteger(pointCount) || pointCount < 0) {
    throw new Error('pointCount must be a non-negative safe integer');
  }
  return Math.max(0, 2 * pointCount - 5);
}

/**
 * Builds a planar Delaunay triangulation on the GPU with deterministic input-order tie breaking.
 *
 * One bounded Bowyer-Watson invocation performs the topology update while applications obtain
 * throughput by placing independent point sets in separate graphs. Exact duplicate coordinates use
 * first-row ownership. Collinear inputs produce zero triangles. Cocircular sites are accepted into
 * the current cavity, making the selected diagonal stable for a fixed input order. Orientation and
 * incircle tests are adaptive exact, so near-degenerate input cannot break the topology. The output is
 * capacity bounded, but `status.requiredCount` always reports the exact final triangle count.
 *
 * The triangulation starts from a super triangle whose three vertices are symbolic points at
 * infinity (in the +x, +y and -x directions), not finite coordinates. Every orientation or
 * incircle test that touches one reduces to exact comparisons of real coordinates, so the hull
 * triangles whose circumcircle would contain a finite super vertex are never lost: the output is
 * the complete Delaunay triangulation of the sites, `2n - 2 - h` triangles for `n` distinct sites
 * with `h` points on the convex hull boundary. Triangles that touch a super vertex are discarded
 * and only all-real triangles are emitted.
 *
 * This contributor deliberately accepts one packed point set: partitioned inputs must select seam
 * ownership before tessellation, because independently triangulated halos do not define a global
 * Delaunay complex.
 *
 * Planar only: `spatialContext` defaults to native planar coordinates and a longitude/latitude
 * context throws at construction. Project first, for example with `createGPULocalFrame` and
 * `GPULocalFrameProjection`, and pass the frame's context so its `frame` identity is kept.
 */
export class LegacyGPUDelaunayTessellation implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUDelaunayTessellationProps;
  /** Resolved spatial context (planar). */
  readonly spatialContext: GPUSpatialContext;

  constructor(props: GPUDelaunayTessellationProps) {
    this.id = props.id ?? 'delaunay-tessellation';
    this.props = props;
    this.spatialContext = resolveGPUSpatialContext(this.id, props.spatialContext);
    const {id} = this;
    const {output} = props;
    validatePackedView(props.positions, ['float32x2'], `${id} positions`);
    validatePackedView(output.triangles, ['uint32x3'], `${id} output.triangles`);
    validateGPUStatusPort(`${id} output.status`, output.status);
    if (!output.status.requiredCount) {
      throw new Error(`${id} output.status.requiredCount is required`);
    }
    if (output.status.duplicateCount) {
      validatePackedView(
        output.status.duplicateCount,
        ['uint32'],
        `${id} output.status.duplicateCount`
      );
      if (output.status.duplicateCount.length < 1) {
        throw new Error(`${id} output.status.duplicateCount must contain one row`);
      }
    }
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {output} = props;
    const {status} = output;
    validateGraphViewsBelongToGraph(id, graph, [
      props.positions,
      output.triangles,
      status.count,
      status.requiredCount,
      status.overflow,
      status.invalidCount,
      status.duplicateCount
    ]);
    const pointCount = props.positions.length;
    // Active planar triangulations need at most 2n + 1 triangles including the supertriangle;
    // the extra guard rows turn an unexpected numeric/topology failure into explicit status.
    const scratchTriangleCapacity = Math.max(2 * pointCount + 8, 8);
    const trianglesA = createTransientView(
      graph,
      `${id}-triangles-a`,
      'uint32',
      4 * scratchTriangleCapacity
    );
    const trianglesB = createTransientView(
      graph,
      `${id}-triangles-b`,
      'uint32',
      4 * scratchTriangleCapacity
    );
    // [triangleCount, failed, cavityTriangleCount, nextInsertionRow, rebuildOldCount,
    // rebuildInsertionRow]. The insertion row lives in GPU state so every site's kernels share one
    // WGSL source and compile once instead of n times; words 4 and 5 hand the site being rebuilt
    // to the rebuild scatter after the state has advanced.
    const state = createTransientView(graph, `${id}-state`, 'uint32', 6);
    // Stable rebuild compaction: one item per triangle slot and per triangle edge.
    const rebuildItemCount = 4 * scratchTriangleCapacity;
    const rebuildBlockCount = Math.ceil(rebuildItemCount / REBUILD_WORKGROUP_SIZE);
    const rebuildBlockOffsets = createTransientView(
      graph,
      `${id}-rebuild-block-offsets`,
      'uint32',
      rebuildBlockCount
    );
    const cavityTriangleIds = createTransientView(
      graph,
      `${id}-cavity-triangle-ids`,
      'uint32',
      scratchTriangleCapacity
    );
    const duplicateFlags = createTransientView(
      graph,
      `${id}-duplicate-flags`,
      'uint32',
      Math.max(pointCount, 1)
    );
    const badFlags = createTransientView(
      graph,
      `${id}-bad-flags`,
      'uint32',
      scratchTriangleCapacity
    );
    const boundaryFlags = createTransientView(
      graph,
      `${id}-boundary-flags`,
      'uint32',
      3 * scratchTriangleCapacity
    );
    const triangleIds = createTransientView(
      graph,
      `${id}-triangle-ids`,
      'uint32',
      scratchTriangleCapacity
    );
    const resultFlags = createTransientView(
      graph,
      `${id}-result-flags`,
      'uint32',
      scratchTriangleCapacity
    );
    const compactTriangleIds = createTransientView(
      graph,
      `${id}-compact-triangle-ids`,
      'uint32',
      scratchTriangleCapacity
    );
    const invalidCount =
      status.invalidCount ?? createTransientView(graph, `${id}-invalid-count`, 'uint32', 1);
    const duplicateCount =
      status.duplicateCount ?? createTransientView(graph, `${id}-duplicate-count`, 'uint32', 1);
    const sharedDeclarations = getDelaunayCoreWGSL({pointCount, scratchTriangleCapacity});
    const cavityDeclarations = `${sharedDeclarations}
${EXACT_ORIENTATION_WGSL}
${ADAPTIVE_INCIRCLE_WGSL}
${SYMBOLIC_INCIRCLE_WGSL}`;
    const orientationDeclarations = `${sharedDeclarations}
${EXACT_ORIENTATION_WGSL}`;
    const rebuildDeclarations = `${REBUILD_SCAN_WGSL}
const REBUILD_ITEM_COUNT: u32 = ${rebuildItemCount}u;
const REBUILD_BLOCK_COUNT: u32 = ${rebuildBlockCount}u;`;
    const rebuildCapacityDeclaration = `const SCRATCH_TRIANGLE_CAPACITY: u32 = ${scratchTriangleCapacity}u;`;
    const nodes: GPUCommandNode<Parameters>[] = [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-initialize-status`,
        operation: OPERATION,
        variant: 'initialize-status',
        bindings: [
          {name: 'state', view: state, type: 'u32', access: 'read_write'},
          {name: 'invalidCount', view: invalidCount, type: 'u32', access: 'read_write'},
          {name: 'duplicateCount', view: duplicateCount, type: 'u32', access: 'read_write'}
        ],
        invocationCount: 1,
        body: `state[stateOffset] = 0u;
  state[stateOffset + 1u] = 0u;
  state[stateOffset + 2u] = 0u;
  state[stateOffset + 3u] = 0u;
  state[stateOffset + 4u] = 0u;
  state[stateOffset + 5u] = 0u;
  invalidCount[invalidCountOffset] = 0u;
  duplicateCount[duplicateCountOffset] = 0u;`
      })
    ];
    if (pointCount > 0) {
      // Exact duplicate detection by sorting coordinate bit patterns: equal sites become adjacent,
      // and the stable sort over identity row ids puts the owning (lowest) row first in each run.
      const siteRows = createTransientView(graph, `${id}-site-rows`, 'uint32', pointCount);
      const siteXKeys = createTransientView(graph, `${id}-site-x-keys`, 'uint32', pointCount);
      const siteYKeys = createTransientView(graph, `${id}-site-y-keys`, 'uint32', pointCount);
      const siteSort = createStableLexicographicIndexSortNodes<Parameters>(graph, {
        id: `${id}-site-sort`,
        operation: OPERATION,
        indices: siteRows,
        keys: [{view: siteXKeys}, {view: siteYKeys}]
      });
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-classify-sites`,
          operation: OPERATION,
          variant: 'parallel-site-validation',
          bindings: [
            {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
            {name: 'siteRows', view: siteRows, type: 'u32', access: 'read_write'},
            {name: 'siteXKeys', view: siteXKeys, type: 'u32', access: 'read_write'},
            {name: 'siteYKeys', view: siteYKeys, type: 'u32', access: 'read_write'},
            {name: 'invalidCount', view: invalidCount, type: 'atomic<u32>', access: 'read_write'}
          ],
          invocationCount: pointCount,
          declarations: `${SITE_KEY_WGSL}
${IS_FINITE_BITS_WGSL}
// Float equality, not bit equality, decides duplicates: -0 and +0 share one key.
fn coordinateKey(value: f32) -> u32 {
  return select(bitcast<u32>(value), 0u, value == 0.0);
}`,
          body: `let base = positionsOffset + 2u * index;
  let point = vec2<f32>(positions[base], positions[base + 1u]);
  let valid = isFiniteBits(point.x) && isFiniteBits(point.y);
  if (!valid) { atomicAdd(&invalidCount[invalidCountOffset], 1u); }
  siteRows[siteRowsOffset + index] = index;
  // Non-finite rows take a NaN bit pattern that no finite key equals, so they never pair.
  siteXKeys[siteXKeysOffset + index] = select(INVALID_SITE_KEY, coordinateKey(point.x), valid);
  siteYKeys[siteYKeysOffset + index] = select(INVALID_SITE_KEY, coordinateKey(point.y), valid);`
        }),
        ...siteSort.nodes,
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-flag-duplicate-sites`,
          operation: OPERATION,
          variant: 'sorted-duplicate-sites',
          bindings: [
            {name: 'sortedRows', view: siteSort.sortedIndices, type: 'u32', access: 'read'},
            {name: 'sortedXKeys', view: siteSort.sortedPrimaryKeys, type: 'u32', access: 'read'},
            {name: 'siteYKeys', view: siteYKeys, type: 'u32', access: 'read'},
            {name: 'duplicateFlags', view: duplicateFlags, type: 'u32', access: 'read_write'},
            {
              name: 'duplicateCount',
              view: duplicateCount,
              type: 'atomic<u32>',
              access: 'read_write'
            }
          ],
          invocationCount: pointCount,
          declarations: SITE_KEY_WGSL,
          body: `let row = sortedRows[sortedRowsOffset + index];
  let xKey = sortedXKeys[sortedXKeysOffset + index];
  var duplicate = false;
  if (index > 0u && xKey != INVALID_SITE_KEY) {
    let previousRow = sortedRows[sortedRowsOffset + index - 1u];
    duplicate = sortedXKeys[sortedXKeysOffset + index - 1u] == xKey &&
      siteYKeys[siteYKeysOffset + previousRow] == siteYKeys[siteYKeysOffset + row];
  }
  duplicateFlags[duplicateFlagsOffset + row] = select(0u, 1u, duplicate);
  if (duplicate) { atomicAdd(&duplicateCount[duplicateCountOffset], 1u); }`
        })
      );
    }
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-initialize-topology`,
        operation: OPERATION,
        variant: 'initialize-supertriangle',
        bindings: [
          {name: 'triangles', view: trianglesA, type: 'u32', access: 'read_write'},
          {name: 'state', view: state, type: 'u32', access: 'read_write'},
          {name: 'invalidCount', view: invalidCount, type: 'u32', access: 'read'}
        ],
        invocationCount: 1,
        declarations: `const POINT_COUNT: u32 = ${pointCount}u;`,
        // The super triangle (S0, S1, S2) is symbolic: its vertices are rows POINT_COUNT..+2 with no
        // coordinates, and every predicate on them is exact (see SYMBOLIC_SUPER_WGSL).
        body: `if (POINT_COUNT >= 3u && invalidCount[invalidCountOffset] == 0u) {
    triangles[trianglesOffset] = POINT_COUNT;
    triangles[trianglesOffset + 1u] = POINT_COUNT + 1u;
    triangles[trianglesOffset + 2u] = POINT_COUNT + 2u;
    triangles[trianglesOffset + 3u] = 0u;
    state[stateOffset] = 1u;
  }`
      })
    );

    let sourceTriangles = trianglesA;
    let destinationTriangles = trianglesB;
    for (let pointRow = 0; pointRow < pointCount; pointRow++) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-site-${pointRow}-cavity`,
          operation: OPERATION,
          variant: 'parallel-cavity-classification',
          bindings: [
            {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
            {name: 'triangles', view: sourceTriangles, type: 'u32', access: 'read'},
            {name: 'state', view: state, type: 'atomic<u32>', access: 'read_write'},
            {name: 'duplicateFlags', view: duplicateFlags, type: 'u32', access: 'read'},
            {name: 'badFlags', view: badFlags, type: 'u32', access: 'read_write'},
            {name: 'cavityTriangleIds', view: cavityTriangleIds, type: 'u32', access: 'read_write'}
          ],
          invocationCount: scratchTriangleCapacity,
          declarations: cavityDeclarations,
          body: `var bad = false;
  let insertionRow = atomicLoad(&state[stateOffset + 3u]);
  if (index < atomicLoad(&state[stateOffset]) && atomicLoad(&state[stateOffset + 1u]) == 0u && duplicateFlags[duplicateFlagsOffset + insertionRow] == 0u) {
    bad = isInsideCircumcircle(triangleVertex(index, 0u), triangleVertex(index, 1u), triangleVertex(index, 2u), insertionRow);
  }
  badFlags[badFlagsOffset + index] = select(0u, 1u, bad);
  if (bad) {
    // Compact list of cavity triangles so boundary detection scans O(cavity), not O(triangles).
    let slot = atomicAdd(&state[stateOffset + 2u], 1u);
    cavityTriangleIds[cavityTriangleIdsOffset + slot] = index;
  }`
        }),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-site-${pointRow}-boundary`,
          operation: OPERATION,
          variant: 'parallel-cavity-boundary',
          bindings: [
            {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
            {name: 'triangles', view: sourceTriangles, type: 'u32', access: 'read'},
            {name: 'state', view: state, type: 'u32', access: 'read'},
            {name: 'badFlags', view: badFlags, type: 'u32', access: 'read'},
            {name: 'boundaryFlags', view: boundaryFlags, type: 'u32', access: 'read_write'},
            {name: 'cavityTriangleIds', view: cavityTriangleIds, type: 'u32', access: 'read'}
          ],
          invocationCount: 3 * scratchTriangleCapacity,
          declarations: orientationDeclarations,
          body: `let triangle = index / 3u;
  let edge = index % 3u;
  var boundary = triangle < state[stateOffset] && badFlags[badFlagsOffset + triangle] != 0u;
  if (boundary) {
    let insertionRow = state[stateOffset + 3u];
    let a = triangleVertex(triangle, edge);
    let b = triangleVertex(triangle, (edge + 1u) % 3u);
    let cavityCount = state[stateOffset + 2u];
    for (var slot = 0u; slot < cavityCount && boundary; slot++) {
      let other = cavityTriangleIds[cavityTriangleIdsOffset + slot];
      if (other == triangle) { continue; }
      for (var otherEdge = 0u; otherEdge < 3u; otherEdge++) {
        let otherA = triangleVertex(other, otherEdge);
        let otherB = triangleVertex(other, (otherEdge + 1u) % 3u);
        if ((a == otherA && b == otherB) || (a == otherB && b == otherA)) { boundary = false; }
      }
    }
    // Exact: an edge collinear with the site would create a zero-area triangle.
    boundary = boundary && orientSymbolic(a, b, insertionRow) != 0;
  }
  boundaryFlags[boundaryFlagsOffset + index] = select(0u, 1u, boundary);`
        }),
        // Library rule 14: the stable rebuild is a two-stage compaction over all workgroups
        // (block counts, one scan of the block counts, block-offset scatter), not one workgroup
        // looping over every triangle. Slots are exclusive prefix sums of the item flags, so the
        // order is the serial one whatever the workgroup count.
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-site-${pointRow}-rebuild-counts`,
          operation: OPERATION,
          variant: 'stable-topology-rebuild-counts',
          bindings: [
            {name: 'state', view: state, type: 'u32', access: 'read'},
            {name: 'badFlags', view: badFlags, type: 'u32', access: 'read'},
            {name: 'boundaryFlags', view: boundaryFlags, type: 'u32', access: 'read'},
            {name: 'duplicateFlags', view: duplicateFlags, type: 'u32', access: 'read'},
            {name: 'blockOffsets', view: rebuildBlockOffsets, type: 'u32', access: 'read_write'}
          ],
          invocationCount: rebuildBlockCount * REBUILD_WORKGROUP_SIZE,
          workgroupSize: REBUILD_WORKGROUP_SIZE,
          guardIndex: false,
          declarations: `${rebuildCapacityDeclaration}
${rebuildDeclarations}
${REBUILD_ITEM_WGSL}`,
          // No early return: every invocation of a workgroup must reach the barriers.
          body: `let item = index;
  let isItem = item < REBUILD_ITEM_COUNT &&
    isRebuildItem(item, state[stateOffset], state[stateOffset + 3u]);
  _ = inclusiveScan(localInvocationIndex, select(0u, 1u, isItem));
  if (localInvocationIndex == 0u && item < REBUILD_ITEM_COUNT) {
    blockOffsets[blockOffsetsOffset + item / REBUILD_WORKGROUP_SIZE] = scanValues[REBUILD_WORKGROUP_SIZE - 1u];
  }`
        }),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-site-${pointRow}-rebuild-offsets`,
          operation: OPERATION,
          variant: 'stable-topology-rebuild-offsets',
          bindings: [
            {name: 'state', view: state, type: 'u32', access: 'read_write'},
            {name: 'blockOffsets', view: rebuildBlockOffsets, type: 'u32', access: 'read_write'}
          ],
          // Second stage: one workgroup scans the rebuildBlockCount block counts in place.
          invocationCount: REBUILD_WORKGROUP_SIZE,
          workgroupSize: REBUILD_WORKGROUP_SIZE,
          guardIndex: false,
          declarations: `${rebuildCapacityDeclaration}
${rebuildDeclarations}`,
          body: `let lane = localInvocationIndex;
  var total = 0u;
  for (var chunk = 0u; chunk < REBUILD_BLOCK_COUNT; chunk += REBUILD_WORKGROUP_SIZE) {
    let blockRow = chunk + lane;
    var count = 0u;
    if (blockRow < REBUILD_BLOCK_COUNT) { count = blockOffsets[blockOffsetsOffset + blockRow]; }
    let inclusive = inclusiveScan(lane, count);
    if (blockRow < REBUILD_BLOCK_COUNT) {
      blockOffsets[blockOffsetsOffset + blockRow] = total + inclusive - count;
    }
    total += scanValues[REBUILD_WORKGROUP_SIZE - 1u];
  }
  // The scatter reads this site's triangle count and insertion row from words 4 and 5, so the
  // state can advance to the next site here.
  if (lane == 0u && state[stateOffset + 1u] == 0u) {
    let insertionRow = state[stateOffset + 3u];
    state[stateOffset + 4u] = state[stateOffset];
    state[stateOffset + 5u] = insertionRow;
    state[stateOffset + 2u] = 0u;
    state[stateOffset + 3u] = insertionRow + 1u;
    if (total > SCRATCH_TRIANGLE_CAPACITY) {
      state[stateOffset + 1u] = 1u;
    } else {
      state[stateOffset] = total;
    }
  }`
        }),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-site-${pointRow}-rebuild`,
          operation: OPERATION,
          variant: 'stable-topology-rebuild-scatter',
          bindings: [
            {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
            {name: 'triangles', view: sourceTriangles, type: 'u32', access: 'read'},
            {name: 'trianglesOut', view: destinationTriangles, type: 'u32', access: 'read_write'},
            {name: 'state', view: state, type: 'u32', access: 'read'},
            {name: 'badFlags', view: badFlags, type: 'u32', access: 'read'},
            {name: 'boundaryFlags', view: boundaryFlags, type: 'u32', access: 'read'},
            {name: 'duplicateFlags', view: duplicateFlags, type: 'u32', access: 'read'},
            {name: 'blockOffsets', view: rebuildBlockOffsets, type: 'u32', access: 'read'}
          ],
          invocationCount: rebuildBlockCount * REBUILD_WORKGROUP_SIZE,
          workgroupSize: REBUILD_WORKGROUP_SIZE,
          guardIndex: false,
          declarations: `${orientationDeclarations}
${rebuildDeclarations}
${REBUILD_ITEM_WGSL}`,
          // No early return: every invocation of a workgroup must reach the barriers. A failed
          // triangulation (earlier, or this site's overflow) writes nothing.
          body: `let item = index;
  let oldCount = state[stateOffset + 4u];
  let insertionRow = state[stateOffset + 5u];
  let isItem = state[stateOffset + 1u] == 0u && item < REBUILD_ITEM_COUNT &&
    isRebuildItem(item, oldCount, insertionRow);
  let inclusive = inclusiveScan(localInvocationIndex, select(0u, 1u, isItem));
  if (isItem) {
    let slot = blockOffsets[blockOffsetsOffset + item / REBUILD_WORKGROUP_SIZE] + inclusive - 1u;
    let destinationBase = trianglesOutOffset + 4u * slot;
    if (item < SCRATCH_TRIANGLE_CAPACITY) {
      let sourceBase = trianglesOffset + 4u * item;
      trianglesOut[destinationBase] = triangles[sourceBase];
      trianglesOut[destinationBase + 1u] = triangles[sourceBase + 1u];
      trianglesOut[destinationBase + 2u] = triangles[sourceBase + 2u];
      trianglesOut[destinationBase + 3u] = 0u;
    } else if (slot < SCRATCH_TRIANGLE_CAPACITY) {
      let candidate = item - SCRATCH_TRIANGLE_CAPACITY;
      let triangle = candidate / 3u;
      let edge = candidate % 3u;
      var a = triangleVertex(triangle, edge);
      var b = triangleVertex(triangle, (edge + 1u) % 3u);
      if (orientSymbolic(a, b, insertionRow) < 0) {
        let swap = a; a = b; b = swap;
      }
      trianglesOut[destinationBase] = a;
      trianglesOut[destinationBase + 1u] = b;
      trianglesOut[destinationBase + 2u] = insertionRow;
      trianglesOut[destinationBase + 3u] = 0u;
    }
  }`
        })
      );
      [sourceTriangles, destinationTriangles] = [destinationTriangles, sourceTriangles];
    }

    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-publish-internal-failure`,
        operation: OPERATION,
        variant: 'publish-internal-failure',
        bindings: [
          {name: 'state', view: state, type: 'u32', access: 'read'},
          {name: 'invalidCount', view: invalidCount, type: 'u32', access: 'read_write'}
        ],
        invocationCount: 1,
        body: 'invalidCount[invalidCountOffset] += state[stateOffset + 1u];'
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-classify-output`,
        operation: OPERATION,
        variant: 'parallel-output-classification',
        bindings: [
          {name: 'triangles', view: sourceTriangles, type: 'u32', access: 'read'},
          {name: 'state', view: state, type: 'u32', access: 'read'},
          {name: 'triangleIds', view: triangleIds, type: 'u32', access: 'read_write'},
          {name: 'resultFlags', view: resultFlags, type: 'u32', access: 'read_write'}
        ],
        invocationCount: scratchTriangleCapacity,
        declarations: `const POINT_COUNT: u32 = ${pointCount}u;`,
        body: `triangleIds[triangleIdsOffset + index] = index;
  var accepted = index < state[stateOffset] && state[stateOffset + 1u] == 0u;
  if (accepted) {
    let base = trianglesOffset + 4u * index;
    accepted = triangles[base] < POINT_COUNT && triangles[base + 1u] < POINT_COUNT && triangles[base + 2u] < POINT_COUNT;
  }
  resultFlags[resultFlagsOffset + index] = select(0u, 1u, accepted);`
      }),
      ...new GPUCompaction({
        id: `${id}-compact-output`,
        input: triangleIds,
        flags: resultFlags,
        output: compactTriangleIds,
        count: status.requiredCount
      }).getCommandNodes(graph)
    );
    if (output.triangles.length > 0) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-materialize`,
          operation: OPERATION,
          variant: 'parallel-materialization',
          bindings: [
            {name: 'triangles', view: sourceTriangles, type: 'u32', access: 'read'},
            {name: 'compactTriangleIds', view: compactTriangleIds, type: 'u32', access: 'read'},
            {name: 'requiredCount', view: status.requiredCount, type: 'u32', access: 'read'},
            {name: 'trianglesOut', view: output.triangles, type: 'u32', access: 'read_write'}
          ],
          invocationCount: output.triangles.length,
          body: `if (index >= requiredCount[requiredCountOffset]) { return; }
  let triangle = compactTriangleIds[compactTriangleIdsOffset + index];
  let sourceBase = trianglesOffset + 4u * triangle;
  let destinationBase = trianglesOutOffset + 3u * index;
  trianglesOut[destinationBase] = triangles[sourceBase];
  trianglesOut[destinationBase + 1u] = triangles[sourceBase + 1u];
  trianglesOut[destinationBase + 2u] = triangles[sourceBase + 2u];`
        })
      );
    }
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-publish-status`,
        operation: OPERATION,
        variant: 'publish-status',
        bindings: [
          {name: 'requiredCountIn', view: status.requiredCount, type: 'u32', access: 'read'},
          {name: 'countOut', view: status.count, type: 'u32', access: 'read_write'},
          {name: 'overflowOut', view: status.overflow, type: 'u32', access: 'read_write'}
        ],
        invocationCount: 1,
        declarations: `const OUTPUT_CAPACITY: u32 = ${output.triangles.length}u;`,
        body: `let requiredCount = requiredCountIn[requiredCountInOffset];
  countOut[countOutOffset] = min(requiredCount, OUTPUT_CAPACITY);
  overflowOut[overflowOutOffset] = select(0u, 1u, requiredCount > OUTPUT_CAPACITY);`
      })
    );
    return nodes;
  }
}

function getDelaunayCoreWGSL(options: {
  pointCount: number;
  scratchTriangleCapacity: number;
}): string {
  return /* wgsl */ `
const POINT_COUNT: u32 = ${options.pointCount}u;
const SCRATCH_TRIANGLE_CAPACITY: u32 = ${options.scratchTriangleCapacity}u;
// Real sites only: rows at or above POINT_COUNT are symbolic super vertices with no coordinates.
fn pointAt(row: u32) -> vec2<f32> {
  return vec2<f32>(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]);
}
fn triangleVertex(triangle: u32, vertex: u32) -> u32 {
  return triangles[trianglesOffset + 4u * triangle + vertex];
}
${SYMBOLIC_SUPER_WGSL}`;
}
