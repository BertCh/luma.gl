// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {
  validateGraphViewsBelongToGraph,
  validatePackedView
} from '../../utils/gpu-contributor-utils';
import {createStableLexicographicIndexSortNodes} from '../../utils/stable-lexicographic-index-sort';
import {
  createWGSLDispatchGate,
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../utils/wgsl-kernel-nodes';
import {
  validateGPUStatusPort,
  type GPUBoundedResultStatusPort
} from '../../analysis-core/contracts/status';
import {EXACT_ORIENTATION_WGSL} from '../../utils/exact-orientation-wgsl';
import {IS_FINITE_BITS_WGSL} from '../../utils/wgsl-common';

const OPERATION = 'GPUDelaunayTessellation';
/** Sort key for non-finite sites: a NaN bit pattern, which no finite coordinate key can equal. */
const SITE_KEY_WGSL = /* wgsl */ `const INVALID_SITE_KEY: u32 = 0xffffffffu;`;

/**
 * Adaptive-exact incircle predicate. `incircleSign(a, b, c, point)` is 1 when `point` lies strictly
 * inside the circumcircle of the counter-clockwise triangle `abc`, -1 strictly outside and 0 on it.
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

// Sign of the incircle determinant for the counter-clockwise triangle abc: 1 when point is strictly
// inside the circumcircle, -1 strictly outside, 0 on it. Unrepresentable inputs (non-finite, or an
// exponent span over 55 bits) keep the float sign.
fn incircleSign(a: vec2<f32>, b: vec2<f32>, c: vec2<f32>, point: vec2<f32>) -> i32 {
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
    if (determinant > bound) { return 1; }
    if (-determinant > bound) { return -1; }
  }
  let exact = incircleExact(a, b, c, point);
  if (exact == 2) { return i32(sign(determinant)); }
  return exact;
}`;

/**
 * Symbolic super vertices. Insertion needs a triangle that contains every site, and the old
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

/**
 * Strict, exact flip predicate with symbolic super vertices and Simulation of Simplicity.
 *
 * `D(p0, p1, p2, p3)` is the sign of the determinant of the rows `(x, y, x^2 + y^2, 1)`: positive
 * when `p3` is strictly inside the circle of the counter-clockwise triangle `p0 p1 p2`. It is
 * alternating in its arguments, so it can always be evaluated with a real site last.
 *
 * Ties (`D = 0`, four real cocircular sites) are broken by raising the lift of site `r` by
 * `eps^(r + 1)`: the perturbed sign is that of the lift cofactor of the dominant point `j` (the
 * smallest row of the four), `(-1)^j orient(the other three, in order)`. Three of four distinct
 * cocircular sites are never collinear, so one level always decides. The perturbation is a real
 * (weighted) configuration, so Lawson flips still converge, and the result is the unique
 * perturbed Delaunay triangulation whatever the insertion and flip order.
 *
 * With a super vertex in the triangle the existing rule decides (see `SYMBOLIC_SUPER_WGSL`): the
 * circle through the farthest super vertex is the half-plane left of the other two, and a site on
 * that line is inside exactly when it lies in the open chord. That case is never a tie.
 *
 * Each kernel reaches `incircleExact` and `orientSign` through one call site each.
 */
const FLIP_PREDICATE_WGSL = /* wgsl */ `
// p is collinear with a and b: strictly inside the chord, exact comparisons only.
fn isInsideOpenSegment(a: vec2<f32>, b: vec2<f32>, p: vec2<f32>) -> bool {
  if (a.x != b.x) { return p.x > min(a.x, b.x) && p.x < max(a.x, b.x); }
  return p.y > min(a.y, b.y) && p.y < max(a.y, b.y);
}

// Sign of orient(x, y, z) when at least two of the three are super vertices, from their indices
// only: rotations of (S0, S1, S2) are counter-clockwise, and with one real vertex the two super
// vertices that follow it in cyclic order decide, as in orientSymbolic.
fn orientMostlySuper(x: u32, y: u32, z: u32) -> i32 {
  var rows = array<u32, 3>(x, y, z);
  var realSlot = 3u;
  for (var slot = 0u; slot < 3u; slot++) {
    if (rows[slot] < POINT_COUNT) { realSlot = slot; }
  }
  if (realSlot == 3u) {
    return select(-1, 1, rows[1] - POINT_COUNT == (rows[0] - POINT_COUNT + 1u) % 3u);
  }
  let first = rows[(realSlot + 1u) % 3u] - POINT_COUNT;
  let second = rows[(realSlot + 2u) % 3u] - POINT_COUNT;
  return select(-1, 1, second == (first + 1u) % 3u);
}

// D(x, y, z, p) for a counter-clockwise triangle x y z and a real site p: 1 inside, -1 outside.
// Never 0.
fn getInsideSign(x: u32, y: u32, z: u32, p: u32) -> i32 {
  var rows = array<u32, 4>(x, y, z, p);
  var farthestSlot = 3u;
  for (var slot = 0u; slot < 3u; slot++) {
    if (rows[slot] >= POINT_COUNT && (farthestSlot == 3u || rows[slot] > rows[farthestSlot])) {
      farthestSlot = slot;
    }
  }
  if (farthestSlot == 3u) {
    let incircle = incircleSign(pointAt(x), pointAt(y), pointAt(z), pointAt(p));
    if (incircle != 0) { return incircle; }
  }
  // One orientation decides the rest: the chord of a circle through a super vertex, or the
  // cofactor of the dominant site of a cocircular quadruple.
  var first = 0u;
  var second = 0u;
  var third = p;
  var cofactorSign = 1;
  if (farthestSlot != 3u) {
    first = rows[(farthestSlot + 1u) % 3u];
    second = rows[(farthestSlot + 2u) % 3u];
  } else {
    var dominant = 0u;
    for (var slot = 1u; slot < 4u; slot++) {
      if (rows[slot] < rows[dominant]) { dominant = slot; }
    }
    var others = array<u32, 3>(0u, 0u, 0u);
    var otherCount = 0u;
    for (var slot = 0u; slot < 4u; slot++) {
      if (slot != dominant) {
        others[otherCount] = rows[slot];
        otherCount++;
      }
    }
    first = others[0];
    second = others[1];
    third = others[2];
    cofactorSign = select(1, -1, (dominant & 1u) == 1u);
  }
  let side = orientSymbolic(first, second, third);
  if (farthestSlot != 3u) {
    if (side != 0) { return side; }
    return select(-1, 1, isInsideOpenSegment(pointAt(first), pointAt(second), pointAt(p)));
  }
  return cofactorSign * side;
}

// Whether edge a -> b of T1 = (a, b, c), shared with T2 = (b, a, d), must flip: D(a, b, c, d) > 0.
// The arguments are permuted so the query is a real site (an interior edge has one): D(a, b, c, d)
// = D(b, a, d, c) (two swaps), = -D(b, c, d, a) (a 4-cycle), = D(a, c, d, b) (a 3-cycle).
fn isFlipRequired(a: u32, b: u32, c: u32, d: u32) -> bool {
  var x = a;
  var y = b;
  var z = c;
  var p = d;
  var parity = 1;
  var knownCounterClockwise = true;
  if (d >= POINT_COUNT) {
    if (c < POINT_COUNT) {
      x = b; y = a; z = d; p = c;
    } else if (a < POINT_COUNT) {
      x = b; y = c; z = d; p = a; parity = -1; knownCounterClockwise = false;
    } else {
      x = a; y = c; z = d; p = b; knownCounterClockwise = false;
    }
  }
  // Both c and d are super vertices here, so x y z has at most one real vertex.
  if (!knownCounterClockwise && orientMostlySuper(x, y, z) < 0) {
    let swap = x;
    x = y;
    y = swap;
    parity = -parity;
  }
  return parity * getInsideSign(x, y, z, p) > 0;
}`;

/** murmur3 `fmix32`: a bijection on u32, so insertion priorities are distinct and pseudo-random. */
const INSERTION_PRIORITY_WGSL = /* wgsl */ `
fn getInsertionPriority(row: u32) -> u32 {
  var hash = row;
  hash = hash ^ (hash >> 16u);
  hash = hash * 0x85ebca6bu;
  hash = hash ^ (hash >> 13u);
  hash = hash * 0xc2b2ae35u;
  hash = hash ^ (hash >> 16u);
  return hash;
}`;

/**
 * Pseudo-random flip priority of edge `3 * triangle + edge`: a bijection on [0, 2^31) (xorshifts and
 * odd multiplications modulo 2^31), so it never equals the empty vote 0xffffffff and identifies the
 * edge. Slot-ordered keys would let a fan of illegal edges (all triangles around one site) flip only at
 * its local minima, about two per pass; random keys flip a constant fraction (a Luby-style
 * independent set), though geometric dependencies can still require linear chains.
 */
const FLIP_PRIORITY_WGSL = /* wgsl */ `
fn getFlipPriority(edgeId: u32) -> u32 {
  var hash = edgeId & 0x7fffffffu;
  hash = hash ^ (hash >> 15u);
  hash = (hash * 0x2c1b3c6du) & 0x7fffffffu;
  hash = hash ^ (hash >> 12u);
  hash = (hash * 0x297a2d39u) & 0x7fffffffu;
  hash = hash ^ (hash >> 15u);
  return hash;
}`;

/** Bounded Delaunay triangles and their exact completeness status. */
export type GPUDelaunayTessellationOutput = {
  /**
   * Counter-clockwise triples of input point indices, each rotated so its smallest row comes
   * first, in ascending `(first, second)` order.
   */
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
 * Maximum GPU insertion/flip steps. The initial logarithmic/square-root budget is extended to
 * `2n + 32`, capped at 4096 steps. Small inputs finish in two bounded GPU continuation kernels;
 * larger inputs use indirect parallel rounds throughout. No iteration index reaches shader text.
 * The cap is not a guarantee for arbitrary Lawson chains: status reports exhaustion explicitly.
 * @internal
 */
export function getDelaunayStepCount(pointCount: number): number {
  if (pointCount < 1) {
    return 0;
  }
  const initialBudget =
    4 * Math.ceil(Math.log2(pointCount + 2)) + 32 + Math.ceil(0.5 * Math.sqrt(pointCount));
  return Math.min(4096, Math.max(initialBudget, 2 * pointCount + 32));
}

/** Number of parallel steps before the small-input GPU continuation. @internal */
export function getDelaunayParallelStepCount(pointCount: number): number {
  const limit = getDelaunayStepCount(pointCount);
  return pointCount <= 2048
    ? Math.min(
        limit,
        4 * Math.ceil(Math.log2(pointCount + 2)) + 32 + Math.ceil(0.5 * Math.sqrt(pointCount))
      )
    : limit;
}

/**
 * Builds a planar Delaunay triangulation on the GPU by parallel insertion and flipping.
 *
 * The first three steps insert a distinct seed edge and a noncollinear site when one exists.
 * This initial simplex prevents a late lone apex from walking a dense collinear hull.
 * Every subsequent step inserts, in parallel, one site into each triangle that still contains one (the site
 * of smallest pseudo-random priority; a site on an edge splits both triangles), then runs one
 * parallel Lawson flip pass: every edge whose quad fails the exact in-circle test is a candidate,
 * and an edge flips when its pseudo-random priority is the smallest vote of both of its
 * triangles. Insertion is valid in any triangulation and Lawson flipping with exact predicates
 * converges from any triangulation in the plane, so no repair stage (gDel2D's star splaying) is
 * needed. Steps run until every site is
 * inserted and no edge is illegal; unused steps are gated off on the GPU (an indirect dispatch of
 * nothing). Parallel steps do O(n) work in nine graph nodes. Up to 2048 sites, only
 * {@link getDelaunayParallelStepCount} steps are encoded this way; two GPU kernels finish any
 * remaining insertion and then exact flipping with one active invocation each. The total limit
 * remains {@link getDelaunayStepCount} (4096). Arbitrary flip chains need not converge logarithmically.
 *
 * Exact duplicate coordinates use first-row ownership. Collinear inputs produce zero triangles.
 * Orientation and in-circle tests are adaptive exact. Cocircular sites are resolved by Simulation
 * of Simplicity on row order (a lifted perturbation), so the result is the unique perturbed
 * Delaunay triangulation, independent of the parallel schedule. Triangles are written
 * counter-clockwise, rotated so the smallest row comes first, and sorted by their first two rows.
 * The output is capacity bounded, but `status.requiredCount` always reports the exact final
 * triangle count.
 *
 * The triangulation starts from a super triangle whose three vertices are symbolic points at
 * infinity (in the +x, +y and -x directions), not finite coordinates. Every orientation or
 * incircle test that touches one reduces to exact comparisons of real coordinates, so the hull
 * triangles whose circumcircle would contain a finite super vertex are never lost: the output is
 * the complete Delaunay triangulation of the sites, `2n - 2 - h` triangles for `n` distinct sites
 * with `h` points on the convex hull boundary. Triangles that touch a super vertex are discarded
 * and only all-real triangles are emitted.
 *
 * A non-finite site cancels the triangulation (zero triangles, counted in `invalidCount`). An
 * internal failure (scratch capacity, or steps exhausted before convergence) also yields zero
 * triangles and adds one to `invalidCount`. Optional `status.converged`, `status.iterationCount`,
 * and `status.iterationLimitReached` distinguish convergence, executed steps, and exhaustion.
 *
 * This contributor deliberately accepts one packed point set: partitioned inputs must select seam
 * ownership before tessellation, because independently triangulated halos do not define a global
 * Delaunay complex.
 *
 * Planar only: `spatialContext` defaults to native planar coordinates and a longitude/latitude
 * context throws at construction. Project first, for example with `createGPULocalFrame` and
 * `GPULocalFrameProjection`, and pass the frame's context so its `frame` identity is kept.
 */
export class GPUDelaunayTessellation implements GPUCommandNodeProducer {
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
      status.duplicateCount,
      status.converged,
      status.iterationCount,
      status.iterationLimitReached
    ]);
    const pointCount = props.positions.length;
    // Sites plus the super triangle need 2n + 1 triangles; the guard rows turn an unexpected
    // failure into explicit status.
    const scratchTriangleCapacity = Math.max(2 * pointCount + 8, 8);
    const view = (name: string, length: number) =>
      createTransientView(graph, `${id}-${name}`, 'uint32', Math.max(length, 1));
    const triangleVertices = view('triangle-vertices', 4 * scratchTriangleCapacity);
    const triangleAdjacency = view('triangle-adjacency', 4 * scratchTriangleCapacity);
    const replacement = view('replacement', 3 * scratchTriangleCapacity);
    const children = view('children', 4 * scratchTriangleCapacity);
    const votes = view('votes', scratchTriangleCapacity);
    const pointLocations = view('point-locations', pointCount);
    const duplicateFlags = view('duplicate-flags', pointCount);
    // First and last distinct sorted sites, first noncollinear site, last distinct sort rank.
    const seeds = view('seeds', 4);
    // [triangleCount, failed, remainingCount, phase, illegalCount, insertedCount, stepCount, unconverged].
    // The phase id (the stamp of the current insertion phase; its flip phase is phase + 1) lives in
    // GPU state, so every step shares one set of pipelines (rule 12).
    const state = view('state', 8);
    const invalidCount = status.invalidCount ?? view('invalid-count', 1);
    const duplicateCount = status.duplicateCount ?? view('duplicate-count', 1);

    const sharedDeclarations = getDelaunayCoreWGSL({pointCount, scratchTriangleCapacity});
    const orientationDeclarations = `${sharedDeclarations}
${POINT_READ_WGSL}
${EXACT_ORIENTATION_WGSL}
${SYMBOLIC_SUPER_WGSL}`;
    const flipDeclarations = `${orientationDeclarations}
${ADAPTIVE_INCIRCLE_WGSL}
${FLIP_PREDICATE_WGSL}
${FLIP_PRIORITY_WGSL}`;
    const positionsBinding: WGSLKernelBinding = {
      name: 'positions',
      view: props.positions,
      type: 'f32',
      access: 'read'
    };
    const binding = (
      name: string,
      bindingView: GraphDataView<'uint32'>,
      access: 'read' | 'read_write' = 'read',
      type: 'u32' | 'atomic<u32>' = 'u32'
    ): WGSLKernelBinding => ({name, view: bindingView, type, access});

    const nodes: GPUCommandNode<Parameters>[] = [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-initialize-status`,
        operation: OPERATION,
        variant: 'initialize-status',
        bindings: [
          binding('state', state, 'read_write'),
          binding('seeds', seeds, 'read_write'),
          binding('invalidCount', invalidCount, 'read_write'),
          binding('duplicateCount', duplicateCount, 'read_write'),
          binding('requiredCount', status.requiredCount, 'read_write')
        ],
        invocationCount: 1,
        body: `for (var word = 0u; word < 8u; word++) { state[stateOffset + word] = 0u; }
  seeds[seedsOffset] = 0xffffffffu;
  seeds[seedsOffset + 1u] = 0xffffffffu;
  seeds[seedsOffset + 2u] = 0xffffffffu;
  seeds[seedsOffset + 3u] = 0u;
  invalidCount[invalidCountOffset] = 0u;
  duplicateCount[duplicateCountOffset] = 0u;
  requiredCount[requiredCountOffset] = 0u;`
      })
    ];
    if (pointCount > 0) {
      nodes.push(
        ...this._getSiteClassificationNodes(graph, {
          invalidCount,
          duplicateCount,
          duplicateFlags,
          seeds
        })
      );
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-initialize-triangles`,
          operation: OPERATION,
          variant: 'initialize-triangles',
          bindings: [
            binding('triangleVertices', triangleVertices, 'read_write'),
            binding('triangleAdjacency', triangleAdjacency, 'read_write'),
            binding('votes', votes, 'read_write'),
            binding('state', state, 'read_write'),
            binding('invalidCount', invalidCount)
          ],
          invocationCount: scratchTriangleCapacity,
          declarations: `const POINT_COUNT: u32 = ${pointCount}u;
const NONE: u32 = 0xffffffffu;`,
          // The super triangle (S0, S1, S2) is symbolic: its vertices are rows POINT_COUNT..+2 with
          // no coordinates, and every predicate on them is exact (see SYMBOLIC_SUPER_WGSL).
          body: `let base = 4u * index;
  let triangulate = index == 0u && invalidCount[invalidCountOffset] == 0u;
  triangleVertices[triangleVerticesOffset + base] = select(NONE, POINT_COUNT, triangulate);
  triangleVertices[triangleVerticesOffset + base + 1u] = select(NONE, POINT_COUNT + 1u, triangulate);
  triangleVertices[triangleVerticesOffset + base + 2u] = select(NONE, POINT_COUNT + 2u, triangulate);
  triangleVertices[triangleVerticesOffset + base + 3u] = 0u;
  for (var word = 0u; word < 3u; word++) { triangleAdjacency[triangleAdjacencyOffset + base + word] = NONE; }
  triangleAdjacency[triangleAdjacencyOffset + base + 3u] = 0u;
  votes[votesOffset + index] = NONE;
  if (triangulate) {
    state[stateOffset] = 1u;
    // Force one flip evaluation even when every site is a duplicate.
    state[stateOffset + 4u] = 1u;
  }`
        }),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-initialize-sites`,
          operation: OPERATION,
          variant: 'initialize-sites',
          bindings: [
            binding('duplicateFlags', duplicateFlags),
            binding('invalidCount', invalidCount),
            binding('pointLocations', pointLocations, 'read_write'),
            binding('state', state, 'read_write', 'atomic<u32>')
          ],
          invocationCount: pointCount,
          declarations: 'const NONE: u32 = 0xffffffffu;',
          // Every site starts strictly inside the super triangle (slot 0, code 3).
          body: `let located = invalidCount[invalidCountOffset] == 0u && duplicateFlags[duplicateFlagsOffset + index] == 0u;
  pointLocations[pointLocationsOffset + index] = select(NONE, 3u, located);
  if (located) { atomicAdd(&state[stateOffset + 2u], 1u); }`
        })
      );
      nodes.push(
        ...this._getStepNodes(graph, {
          pointCount,
          scratchTriangleCapacity,
          positionsBinding,
          binding,
          triangleVertices,
          triangleAdjacency,
          replacement,
          children,
          votes,
          pointLocations,
          seeds,
          state,
          orientationDeclarations,
          flipDeclarations,
          sharedDeclarations
        })
      );
    }
    nodes.push(
      ...this._getOutputNodes(graph, {
        pointCount,
        scratchTriangleCapacity,
        triangleVertices,
        state,
        invalidCount
      })
    );
    return nodes;
  }

  /** Site validation, the stable coordinate sort and first-row duplicate flags. */
  private _getSiteClassificationNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    views: {
      invalidCount: GraphDataView<'uint32'>;
      duplicateCount: GraphDataView<'uint32'>;
      duplicateFlags: GraphDataView<'uint32'>;
      seeds: GraphDataView<'uint32'>;
    }
  ): GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const pointCount = props.positions.length;
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
    return [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-classify-sites`,
        operation: OPERATION,
        variant: 'parallel-site-validation',
        bindings: [
          {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
          {name: 'siteRows', view: siteRows, type: 'u32', access: 'read_write'},
          {name: 'siteXKeys', view: siteXKeys, type: 'u32', access: 'read_write'},
          {name: 'siteYKeys', view: siteYKeys, type: 'u32', access: 'read_write'},
          {
            name: 'invalidCount',
            view: views.invalidCount,
            type: 'atomic<u32>',
            access: 'read_write'
          }
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
          {name: 'duplicateFlags', view: views.duplicateFlags, type: 'u32', access: 'read_write'},
          {name: 'seeds', view: views.seeds, type: 'atomic<u32>', access: 'read_write'},
          {
            name: 'duplicateCount',
            view: views.duplicateCount,
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
  if (duplicate) { atomicAdd(&duplicateCount[duplicateCountOffset], 1u); }
  if (index == 0u) { atomicStore(&seeds[seedsOffset], row); }
  if (!duplicate && xKey != INVALID_SITE_KEY) { atomicMax(&seeds[seedsOffset + 3u], index); }`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-select-seed-edge`,
        operation: OPERATION,
        variant: 'select-seed-edge',
        bindings: [
          {name: 'sortedRows', view: siteSort.sortedIndices, type: 'u32', access: 'read'},
          {name: 'seeds', view: views.seeds, type: 'u32', access: 'read_write'}
        ],
        invocationCount: 1,
        body: `seeds[seedsOffset + 1u] = sortedRows[sortedRowsOffset + seeds[seedsOffset + 3u]];`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-select-seed-triangle`,
        operation: OPERATION,
        variant: 'select-seed-triangle',
        bindings: [
          {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
          {name: 'seeds', view: views.seeds, type: 'atomic<u32>', access: 'read_write'},
          {name: 'invalidCount', view: views.invalidCount, type: 'u32', access: 'read'}
        ],
        invocationCount: pointCount,
        declarations: `${POINT_READ_WGSL}
${EXACT_ORIENTATION_WGSL}`,
        // Bootstrap a noncollinear simplex before bulk insertion. In particular, a lone apex
        // must precede a dense collinear run, whose symbolic hull otherwise needs a linear flip chain.
        body: `if (invalidCount[invalidCountOffset] != 0u) { return; }
  let first = atomicLoad(&seeds[seedsOffset]);
  let second = atomicLoad(&seeds[seedsOffset + 1u]);
  if (orientSign(pointAt(first), pointAt(second), pointAt(index)) != 0) {
    atomicMin(&seeds[seedsOffset + 2u], index);
  }`
      })
    ];
  }

  /**
   * The unrolled insertion-and-flip steps. Each step: an ungated gate node, four insertion kernels
   * (vote, split, adjacency repair, relocation) and four flip kernels (evaluate, apply, adjacency
   * repair, relocation). Nothing step-dependent reaches WGSL: the phase id is a state word.
   */
  private _getStepNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    context: {
      pointCount: number;
      scratchTriangleCapacity: number;
      positionsBinding: WGSLKernelBinding;
      binding: (
        name: string,
        bindingView: GraphDataView<'uint32'>,
        access?: 'read' | 'read_write',
        type?: 'u32' | 'atomic<u32>'
      ) => WGSLKernelBinding;
      triangleVertices: GraphDataView<'uint32'>;
      triangleAdjacency: GraphDataView<'uint32'>;
      replacement: GraphDataView<'uint32'>;
      children: GraphDataView<'uint32'>;
      votes: GraphDataView<'uint32'>;
      pointLocations: GraphDataView<'uint32'>;
      seeds: GraphDataView<'uint32'>;
      state: GraphDataView<'uint32'>;
      sharedDeclarations: string;
      orientationDeclarations: string;
      flipDeclarations: string;
    }
  ): GPUCommandNode<Parameters>[] {
    const {id} = this;
    const {
      pointCount,
      scratchTriangleCapacity,
      positionsBinding,
      binding,
      triangleVertices,
      triangleAdjacency,
      replacement,
      children,
      votes,
      pointLocations,
      seeds,
      state
    } = context;
    // Slots: 0 per site, 1 per triangle slot (insertion); 0 per triangle slot, 1 per site (flip).
    const insertionGate = createWGSLDispatchGate(graph, {
      id: `${id}-insertion-dispatch`,
      operation: OPERATION,
      shapes: [[pointCount], [scratchTriangleCapacity]]
    });
    const flipGate = createWGSLDispatchGate(graph, {
      id: `${id}-flip-dispatch`,
      operation: OPERATION,
      shapes: [[scratchTriangleCapacity], [pointCount]]
    });
    const repairAdjacency = (
      stepId: string,
      phaseOffset: number,
      gate: ReturnType<typeof insertionGate.getGate>
    ) =>
      createWGSLKernelNode<Parameters>(graph, {
        id: `${stepId}-repair-adjacency`,
        operation: OPERATION,
        variant: phaseOffset === 0 ? 'insertion-adjacency-repair' : 'flip-adjacency-repair',
        ...gate,
        bindings: [
          binding('triangleVertices', triangleVertices),
          binding('triangleAdjacency', triangleAdjacency, 'read_write'),
          binding('replacement', replacement),
          binding('votes', votes, 'read_write'),
          binding('state', state)
        ],
        invocationCount: scratchTriangleCapacity,
        declarations: `${context.sharedDeclarations}
const PHASE_OFFSET: u32 = ${phaseOffset}u;`,
        // An operation copied every external edge's old reference unchanged and flagged its
        // internal references FRESH. An old reference into a triangle consumed in this phase is
        // redirected to the edge's new owner; everything else is already final.
        body: REPAIR_ADJACENCY_BODY_WGSL
      });
    const relocate = (
      stepId: string,
      phaseOffset: number,
      gate: ReturnType<typeof insertionGate.getGate>
    ) =>
      createWGSLKernelNode<Parameters>(graph, {
        id: `${stepId}-relocate`,
        operation: OPERATION,
        variant: phaseOffset === 0 ? 'insertion-relocation' : 'flip-relocation',
        ...gate,
        bindings: [
          positionsBinding,
          binding('pointLocations', pointLocations, 'read_write'),
          binding('triangleVertices', triangleVertices),
          binding('children', children),
          binding('state', state, 'read_write', 'atomic<u32>')
        ],
        invocationCount: pointCount,
        declarations: `${context.orientationDeclarations}
const PHASE_OFFSET: u32 = ${phaseOffset}u;`,
        // A site whose triangle was consumed in this phase lies in the closure of one of the
        // triangles now covering it. One orientSymbolic call site, in the nested loop.
        body: RELOCATE_BODY_WGSL
      });

    const nodes: GPUCommandNode<Parameters>[] = [];
    const stepCount = getDelaunayParallelStepCount(pointCount);
    for (let step = 0; step < stepCount; step++) {
      const stepId = `${id}-step-${step}`;
      const insertionPerSite = insertionGate.getGate(0);
      const insertionPerTriangle = insertionGate.getGate(1);
      const flipPerTriangle = flipGate.getGate(0);
      const flipPerSite = flipGate.getGate(1);
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${stepId}-gate`,
          operation: OPERATION,
          variant: 'step-gate',
          bindings: [
            binding('state', state, 'read_write'),
            {...insertionGate.binding, name: 'insertionDispatch'},
            {...flipGate.binding, name: 'flipDispatch'}
          ],
          invocationCount: 1,
          body: `state[stateOffset + 7u] = select(0u, 1u,
    state[stateOffset + 2u] > 0u || state[stateOffset + 4u] > 0u);
  let isActive = state[stateOffset + 1u] == 0u && state[stateOffset + 7u] != 0u;
  let isInserting = isActive && state[stateOffset + 2u] > 0u;
  state[stateOffset + 3u] = state[stateOffset + 3u] + 2u;
  if (isActive) {
    state[stateOffset + 4u] = 0u;
    state[stateOffset + 5u] = 0u;
    state[stateOffset + 6u] = state[stateOffset + 6u] + 1u;
  }
  if (isInserting) {
    ${renameDispatch(insertionGate.armWGSL, 'insertionDispatch')}
  } else {
    ${renameDispatch(insertionGate.disarmWGSL, 'insertionDispatch')}
  }
  if (isActive) {
    ${renameDispatch(flipGate.armWGSL, 'flipDispatch')}
  } else {
    ${renameDispatch(flipGate.disarmWGSL, 'flipDispatch')}
  }`
        }),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${stepId}-insert-vote`,
          operation: OPERATION,
          variant: 'insertion-vote',
          ...insertionPerSite,
          bindings: [
            binding('pointLocations', pointLocations),
            binding('seeds', seeds),
            binding('state', state),
            binding('triangleAdjacency', triangleAdjacency),
            binding('votes', votes, 'read_write', 'atomic<u32>')
          ],
          invocationCount: pointCount,
          declarations: `const NONE: u32 = 0xffffffffu;
${INSERTION_PRIORITY_WGSL}`,
          // The first three GPU-counted steps insert the seed simplex, then all sites compete.
          // A site on an edge votes for both triangles of the edge (it splits both).
          body: `let seedStep = state[stateOffset + 6u];
  if (seedStep <= 3u) {
    if (index != seeds[seedsOffset + seedStep - 1u]) { return; }
  }
  ${INSERTION_VOTE_BODY_WGSL}`
        }),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${stepId}-insert-split`,
          operation: OPERATION,
          variant: 'insertion-split',
          ...insertionPerSite,
          bindings: [
            binding('pointLocations', pointLocations, 'read_write'),
            binding('triangleVertices', triangleVertices, 'read_write'),
            binding('triangleAdjacency', triangleAdjacency, 'read_write'),
            binding('replacement', replacement, 'read_write'),
            binding('children', children, 'read_write'),
            binding('votes', votes),
            binding('state', state, 'read_write', 'atomic<u32>')
          ],
          invocationCount: pointCount,
          declarations: `${context.sharedDeclarations}
${TOPOLOGY_WRITE_WGSL}
${INSERTION_PRIORITY_WGSL}`,
          body: INSERTION_SPLIT_BODY_WGSL
        }),
        repairAdjacency(`${stepId}-insertion`, 0, insertionPerTriangle),
        relocate(`${stepId}-insertion`, 0, insertionPerSite),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${stepId}-flip-evaluate`,
          operation: OPERATION,
          variant: 'flip-evaluation',
          ...flipPerTriangle,
          bindings: [
            positionsBinding,
            binding('triangleVertices', triangleVertices),
            binding('triangleAdjacency', triangleAdjacency),
            binding('votes', votes, 'read_write', 'atomic<u32>'),
            binding('state', state, 'read_write', 'atomic<u32>')
          ],
          invocationCount: scratchTriangleCapacity,
          declarations: context.flipDeclarations,
          // The lower slot owns each interior edge. One isFlipRequired call site, in the edge loop.
          body: FLIP_EVALUATE_BODY_WGSL
        }),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${stepId}-flip-apply`,
          operation: OPERATION,
          variant: 'flip-application',
          ...flipPerTriangle,
          bindings: [
            binding('triangleVertices', triangleVertices, 'read_write'),
            binding('triangleAdjacency', triangleAdjacency, 'read_write'),
            binding('replacement', replacement, 'read_write'),
            binding('children', children, 'read_write'),
            binding('votes', votes),
            binding('state', state)
          ],
          invocationCount: scratchTriangleCapacity,
          declarations: `${context.sharedDeclarations}
${TOPOLOGY_WRITE_WGSL}
${FLIP_PRIORITY_WGSL}`,
          body: FLIP_APPLY_BODY_WGSL
        }),
        repairAdjacency(`${stepId}-flip`, 1, flipPerTriangle),
        relocate(`${stepId}-flip`, 1, flipPerSite)
      );
    }
    if (stepCount < getDelaunayStepCount(pointCount)) {
      // Finish rare long insertion chains first. With no uninserted sites, the flip loop needs
      // no relocation; keeping these kernels separate also keeps one exact predicate call site.
      // Both kernels run on the GPU with one invocation, using the existing storage topology.
      const iterationLimit = `const ITERATION_LIMIT: u32 = ${getDelaunayStepCount(pointCount)}u;`;
      const readAtomicState = (body: string) =>
        body.replace(/state\[stateOffset(?: \+ \d+u)?\]/g, value => `atomicLoad(&${value})`);
      const serialVotes = (body: string) =>
        body.replace(/atomicMin\(&votes\[([^\]]+)\], ([^)]+)\)/g, 'votes[$1] = min(votes[$1], $2)');
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-continue-insertion`,
          operation: OPERATION,
          variant: 'serial-insertion-continuation',
          bindings: [
            positionsBinding,
            binding('triangleVertices', triangleVertices, 'read_write'),
            binding('triangleAdjacency', triangleAdjacency, 'read_write'),
            binding('replacement', replacement, 'read_write'),
            binding('children', children, 'read_write'),
            binding('votes', votes, 'read_write'),
            binding('pointLocations', pointLocations, 'read_write'),
            binding('state', state, 'read_write', 'atomic<u32>')
          ],
          invocationCount: 1,
          workgroupSize: 32,
          declarations: `${context.orientationDeclarations}
${iterationLimit}
const PHASE_OFFSET: u32 = 0u;
${TOPOLOGY_WRITE_WGSL}
${INSERTION_PRIORITY_WGSL}
fn voteForInsertion(index: u32) { ${serialVotes(INSERTION_VOTE_BODY_WGSL)} }
fn splitInsertion(index: u32) { ${INSERTION_SPLIT_BODY_WGSL} }
fn repairInsertion(index: u32) { ${readAtomicState(REPAIR_ADJACENCY_BODY_WGSL)} }
fn relocateInsertion(index: u32) { ${RELOCATE_BODY_WGSL} }`,
          body: `loop {
    if (atomicLoad(&state[stateOffset + 1u]) != 0u || atomicLoad(&state[stateOffset + 2u]) == 0u ||
        atomicLoad(&state[stateOffset + 6u]) >= ITERATION_LIMIT) { break; }
    atomicAdd(&state[stateOffset + 3u], 2u);
    atomicAdd(&state[stateOffset + 6u], 1u);
    // Force legalization even if the last parallel round found no illegal edge.
    atomicStore(&state[stateOffset + 4u], 1u);
    atomicStore(&state[stateOffset + 5u], 0u);
    for (var site = 0u; site < POINT_COUNT; site++) { voteForInsertion(site); }
    for (var site = 0u; site < POINT_COUNT; site++) { splitInsertion(site); }
    for (var triangle = 0u; triangle < atomicLoad(&state[stateOffset]); triangle++) { repairInsertion(triangle); }
    for (var site = 0u; site < POINT_COUNT; site++) { relocateInsertion(site); }
  }`
        }),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-continue-flips`,
          operation: OPERATION,
          variant: 'serial-flip-continuation',
          bindings: [
            positionsBinding,
            binding('triangleVertices', triangleVertices, 'read_write'),
            binding('triangleAdjacency', triangleAdjacency, 'read_write'),
            binding('replacement', replacement, 'read_write'),
            binding('children', children, 'read_write'),
            binding('votes', votes, 'read_write'),
            binding('state', state, 'read_write')
          ],
          invocationCount: 1,
          workgroupSize: 32,
          declarations: `${context.flipDeclarations}
${iterationLimit}
const PHASE_OFFSET: u32 = 1u;
${TOPOLOGY_WRITE_WGSL}
fn evaluateFlip(index: u32) {
${serialVotes(FLIP_EVALUATE_BODY_WGSL)
  .replace(/atomicLoad\(&([^)]*)\)/g, '$1')
  .replace(
    'atomicAdd(&state[stateOffset + 4u], 1u)',
    'state[stateOffset + 4u] = state[stateOffset + 4u] + 1u'
  )}
}
fn applyFlip(index: u32) { ${FLIP_APPLY_BODY_WGSL} }
fn repairFlip(index: u32) { ${REPAIR_ADJACENCY_BODY_WGSL} }`,
          body: `loop {
    if (state[stateOffset + 1u] != 0u || state[stateOffset + 2u] != 0u ||
        state[stateOffset + 4u] == 0u || state[stateOffset + 6u] >= ITERATION_LIMIT) { break; }
    state[stateOffset + 3u] = state[stateOffset + 3u] + 2u;
    state[stateOffset + 6u] = state[stateOffset + 6u] + 1u;
    state[stateOffset + 4u] = 0u;
    for (var triangle = 0u; triangle < state[stateOffset]; triangle++) { evaluateFlip(triangle); }
    for (var triangle = 0u; triangle < state[stateOffset]; triangle++) { applyFlip(triangle); }
    for (var triangle = 0u; triangle < state[stateOffset]; triangle++) { repairFlip(triangle); }
  }
  state[stateOffset + 7u] = select(0u, 1u, state[stateOffset + 2u] > 0u || state[stateOffset + 4u] > 0u);`
        })
      );
    }
    return nodes;
  }

  /** Convergence check, canonical output order, materialization and status. */
  private _getOutputNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    context: {
      pointCount: number;
      scratchTriangleCapacity: number;
      triangleVertices: GraphDataView<'uint32'>;
      state: GraphDataView<'uint32'>;
      invalidCount: GraphDataView<'uint32'>;
    }
  ): GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {output} = props;
    const {status} = output;
    const {pointCount, scratchTriangleCapacity, triangleVertices, state, invalidCount} = context;
    const sortRows = createTransientView(
      graph,
      `${id}-output-rows`,
      'uint32',
      scratchTriangleCapacity
    );
    const firstKeys = createTransientView(
      graph,
      `${id}-output-first-keys`,
      'uint32',
      scratchTriangleCapacity
    );
    const secondKeys = createTransientView(
      graph,
      `${id}-output-second-keys`,
      'uint32',
      scratchTriangleCapacity
    );
    const outputSort = createStableLexicographicIndexSortNodes<Parameters>(graph, {
      id: `${id}-output-sort`,
      operation: OPERATION,
      indices: sortRows,
      keys: [{view: firstKeys}, {view: secondKeys}]
    });
    const declarations = `${getDelaunayCoreWGSL({pointCount, scratchTriangleCapacity})}
${CANONICAL_TRIANGLE_WGSL}`;
    const nodes: GPUCommandNode<Parameters>[] = [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-finalize`,
        operation: OPERATION,
        variant: 'finalize',
        bindings: [
          {name: 'state', view: state, type: 'u32', access: 'read_write'},
          {name: 'invalidCount', view: invalidCount, type: 'u32', access: 'read_write'},
          ...(['converged', 'iterationCount', 'iterationLimitReached'] as const).flatMap(name =>
            status[name]
              ? [{name, view: status[name], type: 'u32' as const, access: 'read_write' as const}]
              : []
          )
        ],
        invocationCount: 1,
        // Sites left over or an illegal edge after the last step: the steps ran out. That is an
        // internal failure, reported, never a partial triangulation.
        body: `let exhausted = state[stateOffset + 1u] == 0u &&
    (state[stateOffset + 2u] > 0u || state[stateOffset + 4u] > 0u);
  if (exhausted) { state[stateOffset + 1u] = 1u; }
  invalidCount[invalidCountOffset] += state[stateOffset + 1u];
  ${status.converged ? 'converged[convergedOffset] = select(0u, 1u, invalidCount[invalidCountOffset] == 0u);' : ''}
  ${status.iterationCount ? 'iterationCount[iterationCountOffset] = state[stateOffset + 6u];' : ''}
  ${status.iterationLimitReached ? 'iterationLimitReached[iterationLimitReachedOffset] = select(0u, 1u, exhausted);' : ''}`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-classify-output`,
        operation: OPERATION,
        variant: 'output-classification',
        bindings: [
          {name: 'triangleVertices', view: triangleVertices, type: 'u32', access: 'read'},
          {name: 'state', view: state, type: 'u32', access: 'read'},
          {name: 'sortRows', view: sortRows, type: 'u32', access: 'read_write'},
          {name: 'firstKeys', view: firstKeys, type: 'u32', access: 'read_write'},
          {name: 'secondKeys', view: secondKeys, type: 'u32', access: 'read_write'},
          {
            name: 'requiredCount',
            view: status.requiredCount,
            type: 'atomic<u32>',
            access: 'read_write'
          }
        ],
        invocationCount: scratchTriangleCapacity,
        declarations,
        // All-real triangles of a successful triangulation, keyed by their canonical rotation;
        // the rest sort last.
        body: `sortRows[sortRowsOffset + index] = index;
  var accepted = state[stateOffset + 1u] == 0u &&
    index < min(state[stateOffset], SCRATCH_TRIANGLE_CAPACITY);
  var canonical = vec3<u32>(NONE, NONE, NONE);
  if (accepted) {
    canonical = getCanonicalTriangle(index);
    accepted = canonical.x < POINT_COUNT && canonical.y < POINT_COUNT && canonical.z < POINT_COUNT;
  }
  firstKeys[firstKeysOffset + index] = select(NONE, canonical.x, accepted);
  secondKeys[secondKeysOffset + index] = select(NONE, canonical.y, accepted);
  if (accepted) { atomicAdd(&requiredCount[requiredCountOffset], 1u); }`
      }),
      ...outputSort.nodes
    ];
    if (output.triangles.length > 0) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-materialize`,
          operation: OPERATION,
          variant: 'parallel-materialization',
          bindings: [
            {name: 'triangleVertices', view: triangleVertices, type: 'u32', access: 'read'},
            {name: 'sortedRows', view: outputSort.sortedIndices, type: 'u32', access: 'read'},
            {name: 'requiredCount', view: status.requiredCount, type: 'u32', access: 'read'},
            {name: 'trianglesOut', view: output.triangles, type: 'u32', access: 'read_write'}
          ],
          invocationCount: output.triangles.length,
          declarations,
          body: `if (index >= requiredCount[requiredCountOffset]) { return; }
  let canonical = getCanonicalTriangle(sortedRows[sortedRowsOffset + index]);
  let destinationBase = trianglesOutOffset + 3u * index;
  trianglesOut[destinationBase] = canonical.x;
  trianglesOut[destinationBase + 1u] = canonical.y;
  trianglesOut[destinationBase + 2u] = canonical.z;`
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

/** Renames the `dispatch` binding used by a gate's arm/disarm statements. */
function renameDispatch(statements: string, name: string): string {
  return statements.split('dispatch[dispatchOffset').join(`${name}[${name}Offset`);
}

/**
 * Read helpers of the topology: `triangleVertices` holds (v0, v1, v2, stamp) per slot, the stamp
 * being the phase id of the slot's last modification. Needs a `triangleVertices` binding.
 */
function getDelaunayCoreWGSL(options: {
  pointCount: number;
  scratchTriangleCapacity: number;
}): string {
  return /* wgsl */ `
const POINT_COUNT: u32 = ${options.pointCount}u;
const SCRATCH_TRIANGLE_CAPACITY: u32 = ${options.scratchTriangleCapacity}u;
const NONE: u32 = 0xffffffffu;
// Adjacency references are 3 * triangle + edge; bit 31 marks a reference written fresh by the
// operation of the current phase.
const FRESH: u32 = 0x80000000u;
fn getVertex(triangle: u32, vertex: u32) -> u32 {
  return triangleVertices[triangleVerticesOffset + 4u * triangle + vertex];
}
fn getStamp(triangle: u32) -> u32 {
  return triangleVertices[triangleVerticesOffset + 4u * triangle + 3u];
}`;
}

/** Site coordinates. Needs a `positions` binding. */
const POINT_READ_WGSL = /* wgsl */ `
// Real sites only: rows at or above POINT_COUNT are symbolic super vertices with no coordinates.
fn pointAt(row: u32) -> vec2<f32> {
  return vec2<f32>(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]);
}`;

/**
 * Write helpers of the insertion and flip operations. Needs read-write `triangleVertices`,
 * `triangleAdjacency`, `replacement` and `children` bindings.
 */
const TOPOLOGY_WRITE_WGSL = /* wgsl */ `
fn writeTriangle(triangle: u32, a: u32, b: u32, c: u32, phase: u32) {
  let base = triangleVerticesOffset + 4u * triangle;
  triangleVertices[base] = a;
  triangleVertices[base + 1u] = b;
  triangleVertices[base + 2u] = c;
  triangleVertices[base + 3u] = phase;
}
fn writeAdjacency(triangle: u32, edge0: u32, edge1: u32, edge2: u32) {
  let base = triangleAdjacencyOffset + 4u * triangle;
  triangleAdjacency[base] = edge0;
  triangleAdjacency[base + 1u] = edge1;
  triangleAdjacency[base + 2u] = edge2;
  triangleAdjacency[base + 3u] = 0u;
}
fn getAdjacency(triangle: u32, edge: u32) -> u32 {
  return triangleAdjacency[triangleAdjacencyOffset + 4u * triangle + edge % 3u];
}
// The new owner of old edge 'edge' of a consumed triangle, read by the adjacency repair.
fn writeReplacement(triangle: u32, edge: u32, reference: u32) {
  replacement[replacementOffset + 3u * triangle + edge % 3u] = reference;
}
// The triangles that now cover a consumed triangle, read by the relocation.
fn writeChildren(triangle: u32, first: u32, second: u32, third: u32) {
  let base = childrenOffset + 4u * triangle;
  children[base] = first;
  children[base + 1u] = second;
  children[base + 2u] = third;
}
fn getReference(triangle: u32, edge: u32) -> u32 {
  return 3u * triangle + edge;
}`;

/**
 * Insertion of the winning site: 1 -> 3 inside a triangle, 2 -> 4 on an edge. A site wins when
 * every triangle it voted for holds its priority, so it alone touches those triangles. External
 * edges keep their old reference (the adjacency repair redirects it), internal ones are FRESH.
 */
const INSERTION_SPLIT_BODY_WGSL = /* wgsl */ `let location = pointLocations[pointLocationsOffset + index];
  if (location == NONE) { return; }
  let triangle = location >> 2u;
  let code = location & 3u;
  let priority = getInsertionPriority(index);
  if (votes[votesOffset + triangle] != priority) { return; }
  var neighbor = NONE;
  var neighborEdge = 0u;
  if (code < 3u) {
    let reference = getAdjacency(triangle, code);
    neighbor = reference / 3u;
    neighborEdge = reference % 3u;
    if (votes[votesOffset + neighbor] != priority) { return; }
  }
  let first = atomicAdd(&state[stateOffset], 2u);
  if (first + 2u > SCRATCH_TRIANGLE_CAPACITY) {
    atomicStore(&state[stateOffset + 1u], 1u);
    return;
  }
  let second = first + 1u;
  let phase = atomicLoad(&state[stateOffset + 3u]);
  let site = index;
  if (code == 3u) {
    // (a, b, c) -> (a, b, q), (b, c, q), (c, a, q); edge 0 of each is old edge 0, 1, 2.
    let a = getVertex(triangle, 0u);
    let b = getVertex(triangle, 1u);
    let c = getVertex(triangle, 2u);
    let old0 = getAdjacency(triangle, 0u);
    let old1 = getAdjacency(triangle, 1u);
    let old2 = getAdjacency(triangle, 2u);
    writeTriangle(triangle, a, b, site, phase);
    writeTriangle(first, b, c, site, phase);
    writeTriangle(second, c, a, site, phase);
    writeAdjacency(triangle, old0, FRESH | getReference(first, 2u), FRESH | getReference(second, 1u));
    writeAdjacency(first, old1, FRESH | getReference(second, 2u), FRESH | getReference(triangle, 1u));
    writeAdjacency(second, old2, FRESH | getReference(triangle, 2u), FRESH | getReference(first, 1u));
    writeReplacement(triangle, 0u, getReference(triangle, 0u));
    writeReplacement(triangle, 1u, getReference(first, 0u));
    writeReplacement(triangle, 2u, getReference(second, 0u));
    writeChildren(triangle, triangle, first, second);
  } else {
    // Edge a -> b of (a, b, c), shared with (b, a, d): (a, q, c), (q, b, c), (b, q, d), (q, a, d).
    let a = getVertex(triangle, code);
    let b = getVertex(triangle, (code + 1u) % 3u);
    let c = getVertex(triangle, (code + 2u) % 3u);
    let d = getVertex(neighbor, (neighborEdge + 2u) % 3u);
    let nextOld = getAdjacency(triangle, code + 1u);
    let previousOld = getAdjacency(triangle, code + 2u);
    let neighborNextOld = getAdjacency(neighbor, neighborEdge + 1u);
    let neighborPreviousOld = getAdjacency(neighbor, neighborEdge + 2u);
    writeTriangle(triangle, a, site, c, phase);
    writeTriangle(first, site, b, c, phase);
    writeTriangle(neighbor, b, site, d, phase);
    writeTriangle(second, site, a, d, phase);
    writeAdjacency(triangle, FRESH | getReference(second, 0u), FRESH | getReference(first, 2u), previousOld);
    writeAdjacency(first, FRESH | getReference(neighbor, 0u), nextOld, FRESH | getReference(triangle, 1u));
    writeAdjacency(neighbor, FRESH | getReference(first, 0u), FRESH | getReference(second, 2u), neighborPreviousOld);
    writeAdjacency(second, FRESH | getReference(triangle, 0u), neighborNextOld, FRESH | getReference(neighbor, 1u));
    writeReplacement(triangle, code + 1u, getReference(first, 1u));
    writeReplacement(triangle, code + 2u, getReference(triangle, 2u));
    writeReplacement(neighbor, neighborEdge + 1u, getReference(second, 1u));
    writeReplacement(neighbor, neighborEdge + 2u, getReference(neighbor, 2u));
    writeChildren(triangle, triangle, first, NONE);
    writeChildren(neighbor, neighbor, second, NONE);
  }
  pointLocations[pointLocationsOffset + index] = NONE;
  atomicSub(&state[stateOffset + 2u], 1u);
  atomicAdd(&state[stateOffset + 5u], 1u);`;

/**
 * Flip of the edge a -> b of T1 = (a, b, c), shared with T2 = (b, a, d): T1 <- (c, a, d),
 * T2 <- (d, b, c). The lower slot T1 owns the edge and flips it when its key is the smallest vote
 * of both triangles; the threads of T2 and of every other loser return before reading.
 */
const FLIP_APPLY_BODY_WGSL = /* wgsl */ `if (index >= min(state[stateOffset], SCRATCH_TRIANGLE_CAPACITY)) { return; }
  // The vote is the priority of one owned edge of this triangle, or of a neighbor's edge (then no
  // owned edge matches and this thread returns before reading anything else).
  let key = votes[votesOffset + index];
  var edge = 3u;
  for (var candidate = 0u; candidate < 3u; candidate++) {
    if (getFlipPriority(3u * index + candidate) == key) { edge = candidate; }
  }
  if (edge == 3u) { return; }
  let reference = getAdjacency(index, edge);
  let neighbor = reference / 3u;
  let neighborEdge = reference % 3u;
  if (votes[votesOffset + neighbor] != key) { return; }
  let phase = state[stateOffset + 3u] + 1u;
  let a = getVertex(index, edge);
  let b = getVertex(index, (edge + 1u) % 3u);
  let c = getVertex(index, (edge + 2u) % 3u);
  let d = getVertex(neighbor, (neighborEdge + 2u) % 3u);
  let nextOld = getAdjacency(index, edge + 1u);
  let previousOld = getAdjacency(index, edge + 2u);
  let neighborNextOld = getAdjacency(neighbor, neighborEdge + 1u);
  let neighborPreviousOld = getAdjacency(neighbor, neighborEdge + 2u);
  writeTriangle(index, c, a, d, phase);
  writeTriangle(neighbor, d, b, c, phase);
  writeAdjacency(index, previousOld, neighborNextOld, FRESH | getReference(neighbor, 2u));
  writeAdjacency(neighbor, neighborPreviousOld, nextOld, FRESH | getReference(index, 2u));
  writeReplacement(index, edge + 2u, getReference(index, 0u));
  writeReplacement(neighbor, neighborEdge + 1u, getReference(index, 1u));
  writeReplacement(neighbor, neighborEdge + 2u, getReference(neighbor, 0u));
  writeReplacement(index, edge + 1u, getReference(neighbor, 1u));
  writeChildren(index, index, neighbor, NONE);
  writeChildren(neighbor, index, neighbor, NONE);`;

/** Rotation of a counter-clockwise triangle that puts its smallest row first. Needs the core. */
const CANONICAL_TRIANGLE_WGSL = /* wgsl */ `
fn getCanonicalTriangle(triangle: u32) -> vec3<u32> {
  let a = getVertex(triangle, 0u);
  let b = getVertex(triangle, 1u);
  let c = getVertex(triangle, 2u);
  if (b < a && b < c) { return vec3<u32>(b, c, a); }
  if (c < a && c < b) { return vec3<u32>(c, a, b); }
  return vec3<u32>(a, b, c);
}`;

/** Shared phase body for parallel dispatch and small-input continuation. */
const REPAIR_ADJACENCY_BODY_WGSL = /* wgsl */ `if (index >= min(state[stateOffset], SCRATCH_TRIANGLE_CAPACITY)) { return; }
  let phase = state[stateOffset + 3u] + PHASE_OFFSET;
  votes[votesOffset + index] = NONE;
  for (var edge = 0u; edge < 3u; edge++) {
    let reference = triangleAdjacency[triangleAdjacencyOffset + 4u * index + edge];
    if (reference == NONE) { continue; }
    var resolved = reference;
    if ((reference & FRESH) != 0u) {
      resolved = reference & ~FRESH;
    } else if (getStamp(reference / 3u) == phase) {
      resolved = replacement[replacementOffset + reference];
    }
    triangleAdjacency[triangleAdjacencyOffset + 4u * index + edge] = resolved;
  }`;

/** Shared phase body for parallel dispatch and small-input continuation. */
const RELOCATE_BODY_WGSL = /* wgsl */ `let location = pointLocations[pointLocationsOffset + index];
  if (location == NONE) { return; }
  let triangle = location >> 2u;
  let phase = atomicLoad(&state[stateOffset + 3u]) + PHASE_OFFSET;
  if (getStamp(triangle) != phase) { return; }
  var found = NONE;
  var foundCode = 3u;
  for (var slot = 0u; slot < 3u; slot++) {
    let candidate = children[childrenOffset + 4u * triangle + slot];
    if (candidate == NONE || found != NONE) { continue; }
    var outside = false;
    var zeroEdge = 3u;
    for (var edge = 0u; edge < 3u; edge++) {
      let side = orientSymbolic(getVertex(candidate, edge), getVertex(candidate, (edge + 1u) % 3u), index);
      if (side < 0) { outside = true; }
      if (side == 0) { zeroEdge = edge; }
    }
    if (!outside) {
      found = candidate;
      foundCode = zeroEdge;
    }
  }
  if (found == NONE) {
    atomicStore(&state[stateOffset + 1u], 1u);
    return;
  }
  pointLocations[pointLocationsOffset + index] = (found << 2u) | foundCode;`;

/** Shared phase body for parallel dispatch and small-input continuation. */
const FLIP_EVALUATE_BODY_WGSL = /* wgsl */ `if (index >= min(atomicLoad(&state[stateOffset]), SCRATCH_TRIANGLE_CAPACITY)) { return; }
  for (var edge = 0u; edge < 3u; edge++) {
    let reference = triangleAdjacency[triangleAdjacencyOffset + 4u * index + edge];
    if (reference == NONE || reference / 3u <= index) { continue; }
    let neighbor = reference / 3u;
    let a = getVertex(index, edge);
    let b = getVertex(index, (edge + 1u) % 3u);
    let c = getVertex(index, (edge + 2u) % 3u);
    let d = getVertex(neighbor, (reference % 3u + 2u) % 3u);
    if (isFlipRequired(a, b, c, d)) {
      let key = getFlipPriority(3u * index + edge);
      atomicMin(&votes[votesOffset + index], key);
      atomicMin(&votes[votesOffset + neighbor], key);
      atomicAdd(&state[stateOffset + 4u], 1u);
    }
  }`;

/** Vote after the three bootstrap steps. */
const INSERTION_VOTE_BODY_WGSL = /* wgsl */ `let location = pointLocations[pointLocationsOffset + index];
  if (location == NONE) { return; }
  let triangle = location >> 2u;
  let code = location & 3u;
  let priority = getInsertionPriority(index);
  atomicMin(&votes[votesOffset + triangle], priority);
  if (code < 3u) {
    let reference = triangleAdjacency[triangleAdjacencyOffset + 4u * triangle + code];
    atomicMin(&votes[votesOffset + reference / 3u], priority);
  }`;
