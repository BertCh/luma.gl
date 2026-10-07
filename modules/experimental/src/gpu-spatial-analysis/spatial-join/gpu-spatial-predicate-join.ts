// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUScan,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {
  GPU_POINT_IN_POLYGON_CLASSIFICATION,
  GPUPairwisePointInPolygon
} from '../../geospatial/gpu-pairwise-point-in-polygon';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import type {GPUCompactOutput} from '../../utils/gpu-contributor-types';
import {
  captureGraphCommandNodes,
  validateCompactOutput,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {validateGPUSpatialWeights, type GPUSpatialWeights} from '../spatial-weights/index';
import {
  createBoundsNode,
  createFeatureRingsNode,
  getSpatialJoinFeatureCount,
  getSpatialJoinGeometryViews,
  isSameSpatialJoinGeometry,
  validateSpatialJoinGeometry
} from './spatial-join-geometry';
import {
  getSpatialJoinCandidateNodes,
  type SpatialJoinCandidateTree
} from './spatial-join-candidates';
import type {GPUSpatialJoinPrepared} from './spatial-join-prepared';
import {
  getFeatureBVHNodes,
  getNextPowerOfTwo,
  isPowerOfTwo,
  SPATIAL_JOIN_WGSL_HELPERS,
  validateDisjointOutputs
} from './spatial-join-passes';
import {
  getSpatialPredicateWGSL,
  type SpatialDistanceParameter,
  type SpatialLegacyPredicateName,
  type SpatialPredicateName,
  type SpatialPredicateSide
} from './spatial-predicate-wgsl';
import {
  doesRelatePatternAdmitDisjoint,
  getPredicateRelatePatterns,
  getRelateMatchesParametersWGSL,
  getRelateMatchesWGSL,
  getRelatePatterns,
  GPU_SPATIAL_RELATE_PATTERN_WORDS,
  getSpatialKindDimension,
  GPU_SPATIAL_RELATE_UNCERTAIN_BIT,
  packGPUSpatialRelate,
  type GPUSpatialRelatePattern
} from './spatial-relate-types';
import {
  getSpatialDwithinWorkgroupWGSL,
  SPATIAL_DWITHIN_WORKGROUP_SIZE
} from './spatial-dwithin-wgsl';
import {
  getSpatialRelateWGSL,
  getSpatialRelateWorkgroupWGSL,
  SPATIAL_RELATE_WORKGROUP_SIZE
} from './spatial-relate-wgsl';
import type {
  GPUSpatialJoinGeometry,
  GPUSpatialJoinOnAttribute,
  GPUSpatialJoinPairs
} from './spatial-join-types';

const OPERATION = 'GPUSpatialPredicateJoin';

/**
 * Predicates supported by {@link GPUSpatialPredicateJoin}: `intersects`, `contains`, `within`,
 * `dwithin`, `covers`, `coveredBy`, `touches`, `crosses`, `overlaps`, `equals`,
 * `containsProperly`, and `relate` (a user DE-9IM `pattern`).
 */
export type GPUSpatialPredicate = SpatialPredicateName;

/** `how` of {@link GPUSpatialPredicateJoinProps}: matched pairs, or left rows without a match. */
export type GPUSpatialJoinHow = 'inner' | 'anti';

const LEGACY_PREDICATES: readonly SpatialLegacyPredicateName[] = [
  'intersects',
  'contains',
  'within',
  'dwithin'
];
const RELATE_PREDICATES: readonly SpatialPredicateName[] = [
  'covers',
  'coveredBy',
  'touches',
  'crosses',
  'overlaps',
  'equals',
  'containsProperly',
  'relate'
];

/**
 * Product of the average vertex counts per feature from which `engine: 'auto'` evaluates
 * `intersects`, `contains` and `within` with the relate engine. Measured crossover: about 32 by 32
 * vertices (see the spatial relate benchmark).
 */
export const SPATIAL_JOIN_RELATE_ENGINE_MINIMUM_VERTEX_PRODUCT = 1024;

/** Whether a `distance` or `pattern` prop is a graph view (per-frame) rather than a plain value. */
function isGraphView<View extends GraphDataView>(value: unknown): value is View {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/** Average vertices per feature of a non-point geometry. */
function getAverageVertexCount(geometry: GPUSpatialJoinGeometry): number {
  return geometry.positions.length / Math.max(getSpatialJoinFeatureCount(geometry), 1);
}

/** Whether `engine: 'auto'` picks the relate engine for a legacy predicate on these sides. */
function prefersRelateEngine(left: GPUSpatialJoinGeometry, right: GPUSpatialJoinGeometry): boolean {
  if (left.kind === 'points' || right.kind === 'points') {
    return false;
  }
  return (
    getAverageVertexCount(left) * getAverageVertexCount(right) >=
    SPATIAL_JOIN_RELATE_ENGINE_MINIMUM_VERTEX_PRODUCT
  );
}

/**
 * Properties for {@link GPUSpatialPredicateJoin}.
 *
 * Per-frame: the contents of every input buffer, plus `distance` and `pattern` when they are given
 * as views. Compile-time: view lengths, `predicate` (it selects the kernel, and for the named
 * predicates the DE-9IM masks and the geometry-kind dimension logic folded into them), `distance`
 * and `pattern` when they are plain values, `how`, `candidateCapacity`, `leafCapacity`,
 * `excludeSameRow`, `prepared`, and which optional views exist.
 */
export type GPUSpatialPredicateJoinProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'spatial-predicate-join'`. */
  id?: string;
  /** Left features. Output rows (and CSR rows of `weights`) are left features. */
  left: GPUSpatialJoinGeometry;
  /** Right features, indexed by a BVH. Prefer the side with more features or larger extent. */
  right: GPUSpatialJoinGeometry;
  /**
   * Predicate evaluated as `predicate(left, right)`, following OGC Simple Features and the
   * DE-9IM matrix (Shapely and GEOS semantics). `within(a, b)` is exactly `contains(b, a)`. See
   * {@link GPUSpatialPredicateJoin} for the boundary rules and the list of predicates.
   */
  predicate: GPUSpatialPredicate;
  /**
   * Compile-time. Which kernel evaluates `intersects`, `contains`, `within` and `dwithin` (every
   * other predicate always uses the relate engine). `'fast'` is the short-circuiting kernel, one
   * invocation per candidate, which is quickest for small features. `'relate'` is the DE-9IM
   * engine for `intersects`, `contains` and `within`, and a workgroup kernel for `dwithin`
   * (which has no matrix): one workgroup per candidate with bounding-box pruning of the edge work.
   * Both are 3 to 15 times faster from about 32 vertices per feature (measured on polygons with
   * 200 to 1000 vertices). `'auto'` (default) picks `'relate'`
   * when neither side is points and the product of the average vertex counts per feature of the
   * two sides is at least {@link SPATIAL_JOIN_RELATE_ENGINE_MINIMUM_VERTEX_PRODUCT}. Results are
   * identical except for configurations the fast kernel's f32 orientation tests misjudge.
   * Requesting the `relate` output always uses the relate engine.
   */
  engine?: 'auto' | 'fast' | 'relate';
  /**
   * Planar distance in coordinate units for `'dwithin'` (inclusive). Required for it, ignored
   * otherwise.
   *
   * A number is compile-time. A one-row float32 view makes the distance per-frame: it is read on
   * every encoding by the candidate probe (box expansion), the exact kernel and the workgroup
   * kernel, so moving a slider is a buffer write with no recompile and the results are identical to
   * the same number given at compile time. A negative, NaN or infinite value selects nothing.
   * `candidateCapacity` must cover the largest distance written, since the candidate count grows
   * with it.
   */
  distance?: number | GraphDataView<'float32'>;
  /**
   * DE-9IM pattern (or any-of list of patterns) for `predicate: 'relate'`, in the OGC text form:
   * nine characters from `T`, `F`, `*`, `0`, `1`, `2`. Required for `'relate'` and rejected for any
   * other predicate. Only bounding-box candidates are enumerated, so a pattern that admits
   * disjoint geometries (all of II, IB, BI and BB allow `F`) is rejected; use `how: 'anti'` for those.
   *
   * A string or list is compile-time. A `uint32` view makes the pattern per-frame: fill it with
   * {@link packGPUSpatialRelatePattern} (two words per pattern slot, so its length fixes the most
   * patterns an any-of list may hold and must be a positive multiple of
   * {@link GPU_SPATIAL_RELATE_PATTERN_WORDS}). The mask kernel reads it on every encoding, so a
   * change is a buffer write with no recompile; unused slots are zero and match nothing. The
   * packing helper does the pattern validation (including the disjoint check) on the CPU.
   *
   * `predicate` itself stays compile-time: it picks the kernel family (the short-circuiting
   * `intersects`, `contains`, `within` and `dwithin` kernels, the workgroup `dwithin`, or the
   * relate engine) and, for the named predicates, patterns that depend on the geometry kinds, so a
   * per-frame predicate would force the slowest path on every join. For a per-frame choice among
   * named predicates use `predicate: 'relate'` with their patterns.
   */
  pattern?: GPUSpatialRelatePattern | GraphDataView<'uint32'>;
  /**
   * `'inner'` (default) emits matched pairs. `'anti'` emits the left features that match no right
   * feature (the complement of `intersects`, and of any other predicate, for example `disjoint` as
   * an anti-join of `intersects`) into `unmatched`; `pairs`, `weights` and `relate` are then
   * not available. If a capacity overflows, the anti output is incomplete in the unsafe direction
   * (rows whose candidates were dropped look unmatched), so check `unmatched.overflow`.
   */
  how?: GPUSpatialJoinHow;
  /** Compact left rows without a match, ascending. Required for, and only for, `how: 'anti'`. */
  unmatched?: GPUCompactOutput;
  /**
   * Skip pairs with equal left and right rows. Set it for a self-join (`left` and `right` are the
   * same features) so each feature is not matched with itself, as `GPUSpatialWeights` requires.
   * Defaults to `false`.
   */
  excludeSameRow?: boolean;
  /**
   * Attribute-equality condition (GeoPandas `sjoin(on_attribute=...)`): a candidate pair survives
   * only when `onAttribute.left[leftRow] === onAttribute.right[rightRow]`, then goes through the
   * predicate as usual. Both views are `uint32` with one key per feature of their side; their
   * contents are per-frame, their presence is compile-time. Candidates failing the key test are
   * dropped before the exact kernels, so `candidateCount` still reports the pre-filter bounding-box
   * count and `candidateCapacity` must cover it. With `how: 'anti'` a left row is unmatched when no
   * key-equal right row satisfies the predicate.
   */
  onAttribute?: GPUSpatialJoinOnAttribute;
  /** Maximum `(left, right)` bounding-box candidates evaluated per encoding. */
  candidateCapacity: number;
  /** Power-of-two BVH leaf slots over `right`. Defaults to the next power of two of its feature count. */
  leafCapacity?: number;
  /**
   * A prepared (static) right-hand side over the same views as `right`: its BVH is reused across
   * encodings until the handle is invalidated, instead of being rebuilt every encoding. The handle
   * must be added to the graph before the join and must not use `spatialSort`. Its `leafCapacity`
   * wins over this join's.
   */
  prepared?: GPUSpatialJoinPrepared;
  /** Matched pairs sorted by `(left, right)`. With `how: 'inner'`, at least one of `pairs` and `weights` is required. */
  pairs?: GPUSpatialJoinPairs;
  /**
   * Optional packed DE-9IM matrix of every matched pair, aligned with `pairs` (slot `k` belongs to
   * `(pairs.leftIds[k], pairs.rightIds[k])`). Requires `pairs` and the same length as
   * `pairs.leftIds`. Layout: see {@link packGPUSpatialRelate}; decode with
   * {@link formatGPUSpatialRelate}. Not available for `'dwithin'`. Setting it makes every
   * predicate evaluate through the relate engine.
   */
  relate?: GraphDataView<'uint32'>;
  /**
   * Matches as cross spatial weights: one CSR row per left feature, neighbors are right rows
   * ascending, every weight is `1`. `weights.neighbors.length` is the pair capacity and must equal
   * `pairs.leftIds.length` when both are given. `weights.distances` is not supported.
   */
  weights?: GPUSpatialWeights;
  /** One-row flag: 1 when the BVH, candidate, or pair capacity overflowed. Required for `'inner'` without `pairs`. */
  overflow?: GraphDataView<'uint32'>;
  /** Optional one-row unclamped candidate count, for sizing `candidateCapacity`. */
  candidateCount?: GraphDataView<'uint32'>;
  /**
   * Optional one-row count of candidates whose classification could not be certified. Those pairs
   * are decided by an f32 test instead. A candidate counts here when
   * - the robust point/polygon classifier returned `uncertain` (point/polygon pairs), or
   * - for the relate engine on any other kind pair, an orientation sign does not exist: a
   *   non-finite coordinate, or product exponents spanning more than 200 bits. Every other
   *   orientation is decided exactly (f32 filter, then exact integer arithmetic).
   *
   * Zero means every decision is certified. The legacy `intersects`, `contains`, `within` and
   * `dwithin` kernels on non-point/polygon pairs do not track uncertainty and report 0.
   */
  uncertainCount?: GraphDataView<'uint32'>;
};

/**
 * Joins two sets of planar features by a spatial predicate: the GPU equivalent of a GeoPandas
 * `sjoin` or a PostGIS `JOIN ... ON ST_Intersects(l, r)`.
 *
 * Pipeline: per-feature bounds, a `GPUBVH` over the right bounds (or a reused `prepared` one), a
 * counted and scanned bounding box probe from every left feature (candidates come out ordered by
 * left row and then right row), an exact predicate test per candidate, and a scan-based stable
 * compaction into the caller's capacity-bounded output. Nothing is read back; `overflow` reports
 * when a capacity was exceeded, in which case the output holds a prefix-consistent subset
 * (candidates past `candidateCapacity` and matches past the pair capacity are dropped).
 *
 * Geometry kinds on either side: points, linestrings and polygons or multipolygons with holes (see
 * {@link GPUSpatialJoinGeometry}); all nine combinations are supported.
 *
 * **Semantics (OGC / DE-9IM).** A point's interior is the point and its boundary is empty. A
 * linestring's boundary is its two endpoints, unless it is closed, and the rest is interior. A
 * polygon's boundary is its rings and its interior is the area between them (holes are exterior).
 * - `intersects(a, b)`: the closed geometries share at least one point. A point on a polygon
 *   boundary intersects the polygon; polygons that only touch along an edge or at a corner intersect.
 * - `contains(a, b)`: every point of `b` lies in the closure of `a` and the interiors intersect.
 *   A point on the boundary of a polygon is not contained by it; a linestring lying entirely on a
 *   polygon boundary is not contained; a polygon contains itself; a linestring does not contain its
 *   own endpoints (unless closed); a polygon contains a polygon that touches its boundary from
 *   inside; a point contains only an equal point; a lower-dimension geometry never contains a
 *   higher-dimension one.
 * - `within(a, b)`: `contains(b, a)`.
 * - `dwithin(a, b)`: the minimum planar distance between the closed geometries is at most
 *   `distance`. Zero distance counts, so it includes everything that intersects.
 * - `covers` / `coveredBy`: like `contains` / `within`, but boundary contact is allowed (a polygon
 *   covers a point on its boundary and a line lying on its boundary).
 * - `touches`: the geometries share boundary points but no interior points (never true for two
 *   points).
 * - `crosses`: the interiors intersect and the intersection has a lower dimension than the larger
 *   geometry (line/line crossing at points, line through a polygon); never true for point/point,
 *   point/point or polygon/polygon.
 * - `overlaps`: same dimension, interiors intersect, and neither contains the other.
 * - `equals`: the same point set, however the vertices are ordered or repeated.
 * - `containsProperly`: `b` lies in the interior of `a` (no boundary contact).
 * - `relate` with `pattern`: any DE-9IM pattern.
 *
 * **One engine.** All of the above are masks over one per-pair DE-9IM classification (the
 * "relate" matrix), as in GEOS RelateNG and SedonaDB. `intersects`, `contains`, `within` and
 * `dwithin` keep their short-circuiting kernels (an `intersects` test stops at the first shared
 * point, where the matrix needs every part located) and are routed through the matrix only when
 * `relate` output is requested; the test suite checks both paths agree. `covers`, `coveredBy`,
 * `touches`, `crosses`, `overlaps`, `equals`, `containsProperly` and `relate` always use the matrix.
 *
 * Point/polygon pairs use the robust double-single classifier `GPUPairwisePointInPolygon` for the
 * point location in every predicate. Every other combination uses f32 orientation tests in the
 * legacy kernels and exact orientation signs in the relate engine (`orientSign`, an f32 filter
 * with an exact integer fallback); the relate engine counts pairs with no sign (non-finite input,
 * huge exponent span) in `uncertainCount` and decides them by the f32 result. Coordinates must be finite; empty or invalid features never match (their matrix is all `F`).
 * Polygon-in-polygon containment assumes valid polygons; a polygon that coincides with a hole is
 * handled by probing both sides of shared edges.
 *
 * **Relate output.** `relate` writes the packed 9-cell matrix of every matched pair next to
 * `pairs`. Only bounding-box candidates are enumerated, so pairs that do not intersect never appear.
 *
 * **Anti joins.** `how: 'anti'` emits the left rows with no match into `unmatched`.
 *
 * **Static right-hand side.** Pass a {@link GPUSpatialJoinPrepared} handle as `prepared` to reuse the
 * right-hand BVH across encodings.
 *
 * Weights output: a join is a cross spatial-weights matrix (rows are left features, neighbors are
 * right rows, ascending), so `weights` writes the CSR directly from the sorted pairs. For a
 * self-join pass `excludeSameRow` to satisfy the no-self-neighbor invariant.
 */
export class GPUSpatialPredicateJoin implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUSpatialPredicateJoinProps;
  /** Number of left features. */
  readonly leftCount: number;
  /** Number of right features. */
  readonly rightCount: number;
  /** Candidate pair capacity. */
  readonly candidateCapacity: number;
  /** Matched pair capacity. Zero for `how: 'anti'`. */
  readonly pairCapacity: number;
  /** Resolved BVH leaf capacity. */
  readonly leafCapacity: number;
  /** Join mode. */
  readonly how: GPUSpatialJoinHow;
  /** Whether candidates are classified by the DE-9IM relate engine. */
  readonly usesRelateEngine: boolean;
  /** Whether `dwithin` is evaluated by the workgroup kernel (one workgroup per candidate). */
  readonly usesWorkgroupDistance: boolean;

  /** DE-9IM patterns that a candidate's matrix must match; empty when the engine is not used or the pattern is per-frame. */
  private readonly relatePatterns: readonly string[];
  /** Per-frame pattern words, or undefined when the pattern is compile-time. */
  private readonly patternView: GraphDataView<'uint32'> | undefined;
  /** Per-frame `dwithin` distance row, or undefined when the distance is compile-time. */
  private readonly distanceView: GraphDataView<'float32'> | undefined;

  constructor(props: GPUSpatialPredicateJoinProps) {
    this.id = props.id ?? 'spatial-predicate-join';
    this.props = props;
    const {id} = this;
    validateSpatialJoinGeometry(id, 'left', props.left);
    validateSpatialJoinGeometry(id, 'right', props.right);
    this.leftCount = getSpatialJoinFeatureCount(props.left);
    this.rightCount = getSpatialJoinFeatureCount(props.right);
    const {predicate} = props;
    const isLegacy = (LEGACY_PREDICATES as readonly string[]).includes(predicate);
    this.distanceView =
      predicate === 'dwithin' && isGraphView(props.distance) ? props.distance : undefined;
    this.patternView = isGraphView(props.pattern) ? props.pattern : undefined;
    if (predicate === 'dwithin') {
      if (this.distanceView) {
        validatePackedView(this.distanceView, ['float32'], `${id} distance`);
        if (this.distanceView.length < 1) {
          throw new Error(`${id} distance must contain one float32 row`);
        }
      } else if (
        typeof props.distance !== 'number' ||
        !Number.isFinite(props.distance) ||
        props.distance < 0
      ) {
        throw new Error(`${id} dwithin requires a finite, non-negative distance`);
      }
    } else if (!isLegacy && !(RELATE_PREDICATES as readonly string[]).includes(predicate)) {
      throw new Error(`${id} unknown predicate ${String(predicate)}`);
    }
    this.how = props.how ?? 'inner';
    if (this.how !== 'inner' && this.how !== 'anti') {
      throw new Error(`${id} unknown how ${String(this.how)}`);
    }
    if (predicate === 'relate') {
      if (props.pattern === undefined) {
        throw new Error(`${id} predicate 'relate' requires a pattern`);
      }
    } else if (props.pattern !== undefined) {
      throw new Error(`${id} pattern requires predicate 'relate'`);
    }
    if (props.relate && predicate === 'dwithin') {
      throw new Error(`${id} relate output is not available for dwithin`);
    }
    const engine = props.engine ?? 'auto';
    if (engine !== 'auto' && engine !== 'fast' && engine !== 'relate') {
      throw new Error(`${id} unknown engine ${String(engine)}`);
    }
    if (engine === 'fast' && !isLegacy) {
      throw new Error(
        `${id} engine 'fast' is only available for intersects, contains, within and dwithin`
      );
    }
    const prefersWorkgroups =
      engine === 'relate' || (engine === 'auto' && prefersRelateEngine(props.left, props.right));
    this.usesRelateEngine =
      !isLegacy || props.relate !== undefined || (predicate !== 'dwithin' && prefersWorkgroups);
    this.usesWorkgroupDistance = predicate === 'dwithin' && prefersWorkgroups;
    if (this.patternView) {
      const words = this.patternView;
      validatePackedUint32View(words, `${id} pattern`);
      if (words.length < 2 || words.length % GPU_SPATIAL_RELATE_PATTERN_WORDS !== 0) {
        throw new Error(
          `${id} pattern view must hold a positive multiple of ` +
            `${GPU_SPATIAL_RELATE_PATTERN_WORDS} words`
        );
      }
      this.relatePatterns = [];
    } else if (this.usesRelateEngine) {
      const patterns =
        predicate === 'relate'
          ? [...getRelatePatterns(props.pattern as GPUSpatialRelatePattern)]
          : [
              ...getPredicateRelatePatterns(
                predicate as Parameters<typeof getPredicateRelatePatterns>[0],
                getSpatialKindDimension(props.left.kind),
                getSpatialKindDimension(props.right.kind)
              )
            ];
      if (predicate === 'relate' && patterns.length === 0) {
        throw new Error(`${id} pattern must not be empty`);
      }
      if (doesRelatePatternAdmitDisjoint(patterns)) {
        throw new Error(
          `${id} pattern admits disjoint geometries, which are not bounding-box candidates; ` +
            "use how: 'anti' with predicate 'intersects' for disjoint"
        );
      }
      this.relatePatterns = patterns;
    } else {
      this.relatePatterns = [];
    }
    this.candidateCapacity = props.candidateCapacity;
    if (
      !Number.isSafeInteger(this.candidateCapacity) ||
      this.candidateCapacity < 1 ||
      2 * this.candidateCapacity + 2 > 0xffffffff
    ) {
      throw new Error(`${id} candidateCapacity must be a positive integer`);
    }
    this.leafCapacity =
      props.prepared?.leafCapacity ??
      props.leafCapacity ??
      getNextPowerOfTwo(Math.max(this.rightCount, 1));
    if (!isPowerOfTwo(this.leafCapacity)) {
      throw new Error(`${id} leafCapacity must be a positive power of two`);
    }
    if (props.prepared) {
      if (!isSameSpatialJoinGeometry(props.prepared.geometry, props.right)) {
        throw new Error(`${id} prepared must index the same geometry views as right`);
      }
      if (props.prepared.spatialSort) {
        throw new Error(`${id} prepared.spatialSort leaves candidates unsorted; disable it`);
      }
    }
    if (this.how === 'anti') {
      if (!props.unmatched) {
        throw new Error(`${id} how 'anti' requires unmatched`);
      }
      if (props.pairs || props.weights || props.relate) {
        throw new Error(`${id} how 'anti' does not produce pairs, weights or relate`);
      }
      validateCompactOutput(id, props.unmatched);
      this.pairCapacity = 0;
    } else {
      if (props.unmatched) {
        throw new Error(`${id} unmatched requires how 'anti'`);
      }
      if (!props.pairs && !props.weights) {
        throw new Error(`${id} requires pairs, weights, or both`);
      }
      if (!props.pairs && !props.overflow) {
        throw new Error(`${id} requires overflow when pairs is not given`);
      }
      if (props.pairs) {
        const {pairs} = props;
        validatePackedUint32View(pairs.leftIds, `${id} pairs.leftIds`);
        validatePackedUint32View(pairs.rightIds, `${id} pairs.rightIds`);
        if (pairs.leftIds.length < 1 || pairs.rightIds.length !== pairs.leftIds.length) {
          throw new Error(`${id} pairs.leftIds and pairs.rightIds must have equal nonzero length`);
        }
        for (const [name, view] of [
          ['count', pairs.count],
          ['overflow', pairs.overflow],
          ['totalCount', pairs.totalCount]
        ] as const) {
          if (view) {
            validatePackedUint32View(view, `${id} pairs.${name}`);
            if (view.length < 1) {
              throw new Error(`${id} pairs.${name} must contain one uint32 row`);
            }
          }
        }
      }
      if (props.weights) {
        const rows = validateGPUSpatialWeights(id, props.weights);
        if (rows !== this.leftCount) {
          throw new Error(`${id} weights must have one row per left feature`);
        }
        if (props.weights.distances) {
          throw new Error(`${id} weights.distances is not supported`);
        }
        if (props.pairs && props.pairs.leftIds.length !== props.weights.neighbors.length) {
          throw new Error(`${id} weights.neighbors length must equal the pair capacity`);
        }
      }
      this.pairCapacity = props.pairs
        ? props.pairs.leftIds.length
        : (props.weights as GPUSpatialWeights).neighbors.length;
      if (props.relate) {
        if (!props.pairs) {
          throw new Error(`${id} relate output requires pairs`);
        }
        validatePackedUint32View(props.relate, `${id} relate`);
        if (props.relate.length !== this.pairCapacity) {
          throw new Error(`${id} relate length must equal the pair capacity`);
        }
      }
    }
    for (const [name, view] of [
      ['overflow', props.overflow],
      ['candidateCount', props.candidateCount],
      ['uncertainCount', props.uncertainCount]
    ] as const) {
      if (view) {
        validatePackedUint32View(view, `${id} ${name}`);
        if (view.length < 1) {
          throw new Error(`${id} ${name} must contain one uint32 row`);
        }
      }
    }
    if (props.onAttribute) {
      const {left, right} = props.onAttribute;
      validatePackedUint32View(left, `${id} onAttribute.left`);
      validatePackedUint32View(right, `${id} onAttribute.right`);
      if (left.length !== this.leftCount || right.length !== this.rightCount) {
        throw new Error(`${id} onAttribute keys must have one row per left and right feature`);
      }
    }
    validateDisjointOutputs(id, this.getInputViews(), this.getOutputViews());
  }

  private getInputViews(): GraphDataView[] {
    return [
      ...getSpatialJoinGeometryViews(this.props.left),
      ...getSpatialJoinGeometryViews(this.props.right),
      ...(this.distanceView ? [this.distanceView] : []),
      ...(this.patternView ? [this.patternView] : []),
      ...(this.props.onAttribute ? [this.props.onAttribute.left, this.props.onAttribute.right] : [])
    ];
  }

  private getOutputViews(): (GraphDataView | undefined)[] {
    const {pairs, weights, overflow, candidateCount, uncertainCount, unmatched, relate} =
      this.props;
    return [
      pairs?.leftIds,
      pairs?.rightIds,
      pairs?.count,
      pairs?.overflow,
      pairs?.totalCount,
      relate,
      weights?.offsets,
      weights?.neighbors,
      weights?.weights,
      unmatched?.ids,
      unmatched?.count,
      unmatched?.overflow,
      unmatched?.totalCount,
      overflow,
      candidateCount,
      uncertainCount
    ];
  }

  /** Returns bounds, BVH, candidate, exact-test, compaction, and optional CSR nodes in order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, leftCount, rightCount, candidateCapacity, pairCapacity, how} = this;
    const {left, right, predicate, pairs, weights, prepared, unmatched, relate} = props;
    validateGraphViewsBelongToGraph(id, graph, [...this.getInputViews(), ...this.getOutputViews()]);
    if (prepared && !prepared.isDeclaredIn(graph)) {
      throw new Error(`${id} requires prepared to be added to the graph first`);
    }
    const nodes: GPUCommandNode<Parameters>[] = [];
    const isAnti = how === 'anti';

    // Per-side feature-to-ring tables (polygons) and bounds. A prepared right side brings its own.
    const sides = [
      {name: 'left' as const, geometry: left, count: leftCount, isPrepared: false},
      {name: 'right' as const, geometry: right, count: rightCount, isPrepared: Boolean(prepared)}
    ].map(side => {
      let featureRings: GraphDataView<'uint32x2'> | undefined;
      let minima: GraphDataView<'float32x2'> | undefined;
      let maxima: GraphDataView<'float32x2'> | undefined;
      if (side.isPrepared) {
        featureRings = prepared?.storage.featureRings;
        return {...side, minima, maxima, featureRings};
      }
      minima = createTransientView(graph, `${id}-${side.name}-minima`, 'float32x2', side.count);
      maxima = createTransientView(graph, `${id}-${side.name}-maxima`, 'float32x2', side.count);
      if (side.geometry.kind === 'polygons') {
        featureRings = createTransientView(
          graph,
          `${id}-${side.name}-feature-rings`,
          'uint32x2',
          side.count
        );
        nodes.push(
          createFeatureRingsNode<Parameters>(graph, `${id}-${side.name}-rings`, {
            featureCount: side.count,
            geometry: side.geometry,
            featureRings
          })
        );
      }
      nodes.push(
        createBoundsNode<Parameters>(graph, `${id}-${side.name}-bounds`, {
          featureCount: side.count,
          geometry: side.geometry,
          featureRings,
          minima,
          maxima
        })
      );
      return {...side, minima, maxima, featureRings};
    });
    const [leftSide, rightSide] = sides;

    let tree: SpatialJoinCandidateTree;
    let bvhOverflow: GraphDataView<'uint32'>;
    if (prepared) {
      tree = {
        nodeMinima: prepared.storage.nodeMinima,
        nodeMaxima: prepared.storage.nodeMaxima,
        leafIds: prepared.storage.leafIds,
        internalNodeCount: prepared.bvhInternalNodeCount
      };
      bvhOverflow = prepared.storage.overflow;
    } else {
      const {bvh, nodes: bvhNodes} = getFeatureBVHNodes(
        graph,
        id,
        rightSide.minima as GraphDataView<'float32x2'>,
        rightSide.maxima as GraphDataView<'float32x2'>,
        this.leafCapacity
      );
      nodes.push(...bvhNodes);
      tree = {
        nodeMinima: bvh.nodeMinima as GraphDataView<'float32x2'>,
        nodeMaxima: bvh.nodeMaxima as GraphDataView<'float32x2'>,
        leafIds: bvh.leafIds,
        internalNodeCount: bvh.internalNodeCount
      };
      bvhOverflow = bvh.overflow;
    }

    // state: [candidate total, uncertain count, match total, unmatched total]
    const state = createTransientView(graph, `${id}-state`, 'uint32', 4);
    // A per-frame dwithin distance rides in one extra trailing row of the candidate table (its
    // float32 bits in the first word, written by the clear kernel), because a polygon-polygon exact
    // kernel already binds eight storage buffers, the default device limit.
    const candidatePairs = createTransientView(
      graph,
      `${id}-candidate-pairs`,
      'uint32x2',
      candidateCapacity + (this.distanceView ? 1 : 0)
    );
    const flags = createTransientView(graph, `${id}-flags`, 'uint32', candidateCapacity);
    const flagOffsets = isAnti
      ? undefined
      : createTransientView(graph, `${id}-flag-offsets`, 'uint32', candidateCapacity);
    const leftIds = isAnti
      ? undefined
      : (pairs?.leftIds ?? createTransientView(graph, `${id}-left-ids`, 'uint32', pairCapacity));
    const rightIds = isAnti
      ? undefined
      : (pairs?.rightIds ?? createTransientView(graph, `${id}-right-ids`, 'uint32', pairCapacity));
    const leftMatchCounts =
      weights || isAnti
        ? createTransientView(graph, `${id}-left-match-counts`, 'uint32', leftCount + 1)
        : undefined;
    const matrices = this.usesRelateEngine
      ? createTransientView(graph, `${id}-matrices`, 'uint32', candidateCapacity)
      : undefined;

    const usesRobustPointPolygon =
      predicate !== 'dwithin' &&
      ((left.kind === 'points' && right.kind === 'polygons') ||
        (left.kind === 'polygons' && right.kind === 'points'));
    const pairRowCount = 2 * candidateCapacity + 1;
    const polygonSide = left.kind === 'polygons' ? left : right.kind === 'polygons' ? right : null;
    const robust = usesRobustPointPolygon
      ? {
          pairPoints: createTransientView(graph, `${id}-pair-points`, 'float32x2', pairRowCount),
          pairGeometryOffsets: createTransientView(
            graph,
            `${id}-pair-geometry-offsets`,
            'uint32',
            pairRowCount + 1
          ),
          pairClassifications: createTransientView(
            graph,
            `${id}-pair-classifications`,
            'uint32',
            pairRowCount
          )
        }
      : undefined;

    // Clear state, candidate sentinels, match counts, and the robust pair layout.
    {
      const bindings: WGSLKernelBinding[] = [
        {name: 'state', view: state, type: 'u32', access: 'read_write'},
        {name: 'candidatePairs', view: candidatePairs, type: 'u32', access: 'read_write'}
      ];
      if (this.distanceView) {
        bindings.push({name: 'distanceRow', view: this.distanceView, type: 'f32', access: 'read'});
      }
      if (leftMatchCounts) {
        bindings.push({
          name: 'leftMatchCounts',
          view: leftMatchCounts,
          type: 'u32',
          access: 'read_write'
        });
      }
      if (robust) {
        bindings.push(
          {name: 'pairPoints', view: robust.pairPoints, type: 'f32', access: 'read_write'},
          {
            name: 'pairGeometryOffsets',
            view: robust.pairGeometryOffsets,
            type: 'u32',
            access: 'read_write'
          }
        );
      }
      const lengths = [
        4,
        candidateCapacity,
        leftMatchCounts?.length ?? 0,
        robust?.pairGeometryOffsets.length ?? 0
      ];
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-clear`,
          operation: OPERATION,
          variant: 'clear',
          bindings,
          invocationCount: Math.max(...lengths),
          declarations: SPATIAL_JOIN_WGSL_HELPERS,
          body: `if (index < 4u) { state[stateOffset + index] = 0u; }
  if (index < ${candidateCapacity}u) {
    candidatePairs[candidatePairsOffset + index * 2u] = NO_FEATURE;
    candidatePairs[candidatePairsOffset + index * 2u + 1u] = NO_FEATURE;
  }
  ${this.distanceView ? `if (index == 0u) { candidatePairs[candidatePairsOffset + ${candidateCapacity * 2}u] = bitcast<u32>(distanceRow[distanceRowOffset]); }` : ''}
  ${leftMatchCounts ? `if (index <= ${leftCount}u) { leftMatchCounts[leftMatchCountsOffset + index] = 0u; }` : ''}
  ${
    robust
      ? `if (index < ${robust.pairPoints.length}u) {
    // A runtime expression keeps the quiet-NaN bit pattern out of const evaluation.
    let quietNaN = bitcast<f32>(0x7fc00000u | (index & 0u));
    pairPoints[pairPointsOffset + index * 2u] = quietNaN;
    pairPoints[pairPointsOffset + index * 2u + 1u] = quietNaN;
  }
  if (index < ${robust.pairGeometryOffsets.length}u) {
    pairGeometryOffsets[pairGeometryOffsetsOffset + index] =
      select(0u, ${(polygonSide as {polygonOffsets: GraphDataView}).polygonOffsets.length - 1}u, index == ${robust.pairGeometryOffsets.length - 1}u);
  }`
      : ''
  }`
        })
      );
    }

    // Candidate generation: count per left feature, scan, then write in (left, right) order.
    nodes.push(
      ...getSpatialJoinCandidateNodes<Parameters>(graph, {
        id,
        leftCount,
        rightCount,
        leftMinima: leftSide.minima as GraphDataView<'float32x2'>,
        leftMaxima: leftSide.maxima as GraphDataView<'float32x2'>,
        tree,
        candidateCapacity,
        margin: predicate === 'dwithin' ? (this.distanceView ?? (props.distance as number)) : 0,
        state,
        candidatePairs
      })
    );

    // Attribute equality: candidates with unequal keys become empty slots, which every later
    // kernel already skips.
    if (props.onAttribute) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-on-attribute`,
          operation: OPERATION,
          variant: 'on-attribute',
          bindings: [
            {name: 'candidatePairs', view: candidatePairs, type: 'u32', access: 'read_write'},
            {name: 'leftKeys', view: props.onAttribute.left, type: 'u32', access: 'read'},
            {name: 'rightKeys', view: props.onAttribute.right, type: 'u32', access: 'read'}
          ],
          invocationCount: candidateCapacity,
          declarations: SPATIAL_JOIN_WGSL_HELPERS,
          body: `let left = candidatePairs[candidatePairsOffset + index * 2u];
  if (left == NO_FEATURE) { return; }
  let right = candidatePairs[candidatePairsOffset + index * 2u + 1u];
  if (leftKeys[leftKeysOffset + left] != rightKeys[rightKeysOffset + right]) {
    candidatePairs[candidatePairsOffset + index * 2u] = NO_FEATURE;
    candidatePairs[candidatePairsOffset + index * 2u + 1u] = NO_FEATURE;
  }`
        })
      );
    }

    // Exact predicate per candidate slot.
    const sameRowTest = props.excludeSameRow ? ' && left != right' : '';
    const sideSpecs: SpatialPredicateSide[] = sides.map(side => ({
      prefix: side.name,
      kind: side.geometry.kind,
      vertexCount: side.geometry.positions.length
    }));
    const sideBindings: WGSLKernelBinding[] = [];
    const distance: number | SpatialDistanceParameter = this.distanceView
      ? {
          expression: `bitcast<f32>(candidatePairs[candidatePairsOffset + ${candidateCapacity * 2}u])`
        }
      : ((props.distance as number | undefined) ?? 0);
    for (const side of sides) {
      const prefix = side.name;
      sideBindings.push({
        name: `${prefix}Positions`,
        view: side.geometry.positions,
        type: 'f32',
        access: 'read'
      });
      if (side.geometry.kind === 'lines') {
        sideBindings.push({
          name: `${prefix}RingOffsets`,
          view: side.geometry.lineOffsets,
          type: 'u32',
          access: 'read'
        });
      } else if (side.geometry.kind === 'polygons') {
        sideBindings.push(
          {
            name: `${prefix}FeatureRings`,
            view: side.featureRings as GraphDataView,
            type: 'u32',
            access: 'read'
          },
          {
            name: `${prefix}RingOffsets`,
            view: side.geometry.ringOffsets,
            type: 'u32',
            access: 'read'
          }
        );
      }
    }
    if (robust) {
      const pointsOnLeft = left.kind === 'points';
      const pointGeometry = pointsOnLeft ? left : right;
      const polygonGeometry = (pointsOnLeft ? right : left) as Extract<
        GPUSpatialJoinGeometry,
        {kind: 'polygons'}
      >;
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-expand`,
          operation: OPERATION,
          variant: 'expand',
          bindings: [
            {name: 'candidatePairs', view: candidatePairs, type: 'u32', access: 'read'},
            {name: 'pointPositions', view: pointGeometry.positions, type: 'f32', access: 'read'},
            {
              name: 'featureOffsets',
              view: polygonGeometry.featureOffsets,
              type: 'u32',
              access: 'read'
            },
            {name: 'pairPoints', view: robust.pairPoints, type: 'f32', access: 'read_write'},
            {
              name: 'pairGeometryOffsets',
              view: robust.pairGeometryOffsets,
              type: 'u32',
              access: 'read_write'
            }
          ],
          invocationCount: candidateCapacity,
          declarations: SPATIAL_JOIN_WGSL_HELPERS,
          body: `let left = candidatePairs[candidatePairsOffset + index * 2u];
  if (left == NO_FEATURE) { return; }
  let right = candidatePairs[candidatePairsOffset + index * 2u + 1u];
  let pointRow = ${pointsOnLeft ? 'left' : 'right'};
  let polygonRow = ${pointsOnLeft ? 'right' : 'left'};
  let row = index * 2u + 1u;
  pairPoints[pairPointsOffset + row * 2u] = pointPositions[pointPositionsOffset + pointRow * 2u];
  pairPoints[pairPointsOffset + row * 2u + 1u] = pointPositions[pointPositionsOffset + pointRow * 2u + 1u];
  pairGeometryOffsets[pairGeometryOffsetsOffset + row] = featureOffsets[featureOffsetsOffset + polygonRow];
  pairGeometryOffsets[pairGeometryOffsetsOffset + row + 1u] = featureOffsets[featureOffsetsOffset + polygonRow + 1u];`
        })
      );
      nodes.push(
        ...captureGraphCommandNodes(graph, () =>
          new GPUPairwisePointInPolygon({
            id: `${id}-classify`,
            points: robust.pairPoints,
            polygonPositions: polygonGeometry.positions,
            geometryOffsets: robust.pairGeometryOffsets,
            polygonOffsets: polygonGeometry.polygonOffsets,
            ringOffsets: polygonGeometry.ringOffsets,
            output: robust.pairClassifications
          }).addToGraph(graph)
        )
      );
    }

    if (matrices) {
      this.addRelateNodes(graph, nodes, {
        sides,
        sideSpecs,
        sideBindings,
        candidatePairs,
        matrices,
        flags,
        state,
        robust,
        sameRowTest
      });
    } else if (robust) {
      // A point never contains a polygon and a polygon is never within a point; otherwise
      // `contains`/`within` need the point strictly inside and `intersects` accepts the boundary.
      const pointsOnLeft = left.kind === 'points';
      const impossible =
        (predicate === 'contains' && pointsOnLeft) || (predicate === 'within' && !pointsOnLeft);
      const accepted =
        predicate === 'intersects'
          ? 'classification == INSIDE || classification == BOUNDARY'
          : 'classification == INSIDE';
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-resolve`,
          operation: OPERATION,
          variant: 'resolve',
          bindings: [
            {name: 'candidatePairs', view: candidatePairs, type: 'u32', access: 'read'},
            {
              name: 'pairClassifications',
              view: robust.pairClassifications,
              type: 'u32',
              access: 'read'
            },
            {name: 'flags', view: flags, type: 'u32', access: 'read_write'},
            {name: 'state', view: state, type: 'atomic<u32>', access: 'read_write'}
          ],
          invocationCount: candidateCapacity,
          declarations: `${SPATIAL_JOIN_WGSL_HELPERS}
const INSIDE: u32 = ${GPU_POINT_IN_POLYGON_CLASSIFICATION.inside}u;
const BOUNDARY: u32 = ${GPU_POINT_IN_POLYGON_CLASSIFICATION.boundary}u;
const UNCERTAIN: u32 = ${GPU_POINT_IN_POLYGON_CLASSIFICATION.uncertain}u;`,
          body: `let left = candidatePairs[candidatePairsOffset + index * 2u];
  var matched = false;
  if (left != NO_FEATURE) {
    let right = candidatePairs[candidatePairsOffset + index * 2u + 1u];
    let classification = pairClassifications[pairClassificationsOffset + index * 2u + 1u];
    if (classification == UNCERTAIN) {
      atomicAdd(&state[stateOffset + 1u], 1u);
    } else {
      matched = ${impossible ? 'false' : `(${accepted})`}${sameRowTest};
    }
  }
  flags[flagsOffset + index] = select(0u, 1u, matched);`
        })
      );
      // Candidates the robust classifier could not certify are decided by the f32 test.
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-fallback`,
          operation: OPERATION,
          variant: `fallback-${left.kind}-${right.kind}-${predicate}`,
          bindings: [
            ...sideBindings,
            {name: 'candidatePairs', view: candidatePairs, type: 'u32', access: 'read'},
            {
              name: 'pairClassifications',
              view: robust.pairClassifications,
              type: 'u32',
              access: 'read'
            },
            {name: 'flags', view: flags, type: 'u32', access: 'read_write'}
          ],
          invocationCount: candidateCapacity,
          declarations: `${getSpatialPredicateWGSL(sideSpecs[0], sideSpecs[1], predicate as SpatialLegacyPredicateName, 0)}
const UNCERTAIN: u32 = ${GPU_POINT_IN_POLYGON_CLASSIFICATION.uncertain}u;`,
          body: `let left = candidatePairs[candidatePairsOffset + index * 2u];
  if (left == NO_FEATURE) { return; }
  if (pairClassifications[pairClassificationsOffset + index * 2u + 1u] != UNCERTAIN) { return; }
  let right = candidatePairs[candidatePairsOffset + index * 2u + 1u];
  flags[flagsOffset + index] = select(0u, 1u, pairMatches(left, right)${sameRowTest});`
        })
      );
    } else {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-exact`,
          operation: OPERATION,
          variant: `exact-${left.kind}-${right.kind}-${predicate}`,
          bindings: [
            ...sideBindings,
            {name: 'candidatePairs', view: candidatePairs, type: 'u32', access: 'read'},
            {name: 'flags', view: flags, type: 'u32', access: 'read_write'}
          ],
          ...(this.usesWorkgroupDistance
            ? {
                // One workgroup per candidate slot: its lanes split the edges of the left feature.
                invocationCount: candidateCapacity * SPATIAL_DWITHIN_WORKGROUP_SIZE,
                workgroupSize: SPATIAL_DWITHIN_WORKGROUP_SIZE,
                guardIndex: false,
                declarations: getSpatialDwithinWorkgroupWGSL(sideSpecs[0], sideSpecs[1], distance),
                body: `let slot = index / ${SPATIAL_DWITHIN_WORKGROUP_SIZE}u;
  let left = candidatePairs[candidatePairsOffset + slot * 2u];
  let right = candidatePairs[candidatePairsOffset + slot * 2u + 1u];
  let matched = dwithinWorkgroupPair(left, right, left != NO_FEATURE, localInvocationIndex);
  if (localInvocationIndex == 0u) {
    flags[flagsOffset + slot] = select(0u, 1u, matched${sameRowTest});
  }`
              }
            : {
                invocationCount: candidateCapacity,
                declarations: getSpatialPredicateWGSL(
                  sideSpecs[0],
                  sideSpecs[1],
                  predicate as SpatialLegacyPredicateName,
                  distance
                ),
                body: `let left = candidatePairs[candidatePairsOffset + index * 2u];
  var matched = false;
  if (left != NO_FEATURE) {
    let right = candidatePairs[candidatePairsOffset + index * 2u + 1u];
    matched = pairMatches(left, right)${sameRowTest};
  }
  flags[flagsOffset + index] = select(0u, 1u, matched);`
              })
        })
      );
    }

    if (isAnti && unmatched && leftMatchCounts) {
      this.addAntiNodes(graph, nodes, {candidatePairs, flags, leftMatchCounts, unmatched, state});
    } else if (leftIds && rightIds && flagOffsets) {
      // Stable compaction of flagged candidates into the (left, right) output.
      nodes.push(
        ...new GPUScan({
          id: `${id}-scan-flags`,
          input: flags,
          output: flagOffsets,
          mode: 'exclusive'
        }).getCommandNodes(graph)
      );
      const scatterBindings: WGSLKernelBinding[] = [
        {name: 'candidatePairs', view: candidatePairs, type: 'u32', access: 'read'},
        {name: 'flags', view: flags, type: 'u32', access: 'read'},
        {name: 'flagOffsets', view: flagOffsets, type: 'u32', access: 'read'},
        {name: 'leftIds', view: leftIds, type: 'u32', access: 'read_write'},
        {name: 'rightIds', view: rightIds, type: 'u32', access: 'read_write'},
        {name: 'state', view: state, type: 'u32', access: 'read_write'}
      ];
      if (relate && matrices) {
        scatterBindings.push(
          {name: 'matrices', view: matrices, type: 'u32', access: 'read'},
          {name: 'relateOut', view: relate, type: 'u32', access: 'read_write'}
        );
      }
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-scatter`,
          operation: OPERATION,
          variant: relate ? 'scatter-relate' : 'scatter',
          bindings: scatterBindings,
          invocationCount: candidateCapacity,
          declarations: `const PAIR_CAPACITY: u32 = ${pairCapacity}u;`,
          body: `if (flags[flagsOffset + index] != 0u) {
    let slot = flagOffsets[flagOffsetsOffset + index];
    if (slot < PAIR_CAPACITY) {
      leftIds[leftIdsOffset + slot] = candidatePairs[candidatePairsOffset + index * 2u];
      rightIds[rightIdsOffset + slot] = candidatePairs[candidatePairsOffset + index * 2u + 1u];
      ${relate && matrices ? 'relateOut[relateOutOffset + slot] = matrices[matricesOffset + index] & 0x3ffffu;' : ''}
    }
  }
  if (index == ${candidateCapacity - 1}u) {
    state[stateOffset + 2u] = flagOffsets[flagOffsetsOffset + index] + flags[flagsOffset + index];
  }`
        })
      );
    }

    // Scalars.
    {
      const bindings: WGSLKernelBinding[] = [
        {name: 'state', view: state, type: 'u32', access: 'read'},
        {name: 'bvhOverflow', view: bvhOverflow, type: 'u32', access: 'read'}
      ];
      const scalars: [string, GraphDataView<'uint32'> | undefined][] = [
        ['overflow', props.overflow],
        ['pairsOverflow', pairs?.overflow],
        ['pairsCount', pairs?.count],
        ['pairsTotal', pairs?.totalCount],
        ['unmatchedOverflow', unmatched?.overflow],
        ['unmatchedCount', unmatched?.count],
        ['unmatchedTotal', unmatched?.totalCount],
        ['candidateCount', props.candidateCount],
        ['uncertainCount', props.uncertainCount]
      ];
      for (const [name, view] of scalars) {
        if (view) {
          bindings.push({name, view, type: 'u32', access: 'read_write'});
        }
      }
      const has = (name: string) => scalars.some(([key, view]) => key === name && view);
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-finalize`,
          operation: OPERATION,
          variant: 'finalize',
          bindings,
          invocationCount: 1,
          declarations: `const CANDIDATE_CAPACITY: u32 = ${candidateCapacity}u;
const PAIR_CAPACITY: u32 = ${pairCapacity}u;
const UNMATCHED_CAPACITY: u32 = ${unmatched?.ids.length ?? 0}u;`,
          body: `let candidateTotal = state[stateOffset];
  let matchTotal = ${isAnti ? '0u' : 'state[stateOffset + 2u]'};
  let unmatchedTotalValue = ${isAnti ? 'state[stateOffset + 3u]' : '0u'};
  let overflowed = bvhOverflow[bvhOverflowOffset] != 0u || candidateTotal > CANDIDATE_CAPACITY ||
    matchTotal > PAIR_CAPACITY || unmatchedTotalValue > UNMATCHED_CAPACITY;
  let overflowValue = select(0u, 1u, overflowed);
  ${has('overflow') ? 'overflow[overflowOffset] = overflowValue;' : ''}
  ${has('pairsOverflow') ? 'pairsOverflow[pairsOverflowOffset] = overflowValue;' : ''}
  ${has('pairsCount') ? 'pairsCount[pairsCountOffset] = min(matchTotal, PAIR_CAPACITY);' : ''}
  ${has('pairsTotal') ? 'pairsTotal[pairsTotalOffset] = matchTotal;' : ''}
  ${has('unmatchedOverflow') ? 'unmatchedOverflow[unmatchedOverflowOffset] = overflowValue;' : ''}
  ${has('unmatchedCount') ? 'unmatchedCount[unmatchedCountOffset] = min(unmatchedTotalValue, UNMATCHED_CAPACITY);' : ''}
  ${has('unmatchedTotal') ? 'unmatchedTotal[unmatchedTotalOffset] = unmatchedTotalValue;' : ''}
  ${has('candidateCount') ? 'candidateCount[candidateCountOffset] = candidateTotal;' : ''}
  ${has('uncertainCount') ? 'uncertainCount[uncertainCountOffset] = state[stateOffset + 1u];' : ''}`
        })
      );
    }

    // Optional CSR: rows are left features and pairs are already sorted by left row.
    if (weights && leftMatchCounts && leftIds && rightIds) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-count-matches`,
          operation: OPERATION,
          variant: 'count-matches',
          bindings: [
            {name: 'leftIds', view: leftIds, type: 'u32', access: 'read'},
            {name: 'state', view: state, type: 'u32', access: 'read'},
            {
              name: 'leftMatchCounts',
              view: leftMatchCounts,
              type: 'atomic<u32>',
              access: 'read_write'
            }
          ],
          invocationCount: pairCapacity,
          declarations: `const PAIR_CAPACITY: u32 = ${pairCapacity}u;`,
          body: `if (index >= min(state[stateOffset + 2u], PAIR_CAPACITY)) { return; }
  atomicAdd(&leftMatchCounts[leftMatchCountsOffset + leftIds[leftIdsOffset + index]], 1u);`
        })
      );
      nodes.push(
        ...new GPUScan({
          id: `${id}-scan-matches`,
          input: leftMatchCounts,
          output: weights.offsets,
          mode: 'exclusive'
        }).getCommandNodes(graph)
      );
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-fill-weights`,
          operation: OPERATION,
          variant: 'fill-weights',
          bindings: [
            {name: 'rightIds', view: rightIds, type: 'u32', access: 'read'},
            {name: 'state', view: state, type: 'u32', access: 'read'},
            {name: 'neighbors', view: weights.neighbors, type: 'u32', access: 'read_write'},
            {name: 'weightValues', view: weights.weights, type: 'f32', access: 'read_write'}
          ],
          invocationCount: pairCapacity,
          declarations: `const PAIR_CAPACITY: u32 = ${pairCapacity}u;`,
          body: `if (index >= min(state[stateOffset + 2u], PAIR_CAPACITY)) { return; }
  neighbors[neighborsOffset + index] = rightIds[rightIdsOffset + index];
  weightValues[weightValuesOffset + index] = 1.0;`
        })
      );
    }
    return nodes;
  }

  /**
   * Adds the relate-engine classification (matrix per candidate slot) and the mask that turns the
   * matrices into match flags and the uncertain count.
   */
  private addRelateNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    nodes: GPUCommandNode<Parameters>[],
    context: {
      sides: {geometry: GPUSpatialJoinGeometry}[];
      sideSpecs: SpatialPredicateSide[];
      sideBindings: WGSLKernelBinding[];
      candidatePairs: GraphDataView<'uint32x2'>;
      matrices: GraphDataView<'uint32'>;
      flags: GraphDataView<'uint32'>;
      state: GraphDataView<'uint32'>;
      robust:
        | {
            pairClassifications: GraphDataView<'uint32'>;
          }
        | undefined;
      sameRowTest: string;
    }
  ): void {
    const {id, props, candidateCapacity} = this;
    const {left, right} = props;
    const {sideSpecs, sideBindings, candidatePairs, matrices, flags, state, robust} = context;
    const uncertainBit = `${GPU_SPATIAL_RELATE_UNCERTAIN_BIT}u`;
    const classification = GPU_POINT_IN_POLYGON_CLASSIFICATION;
    if (robust) {
      // Point/polygon pairs: the matrix follows from the robust point location alone.
      const pointsOnLeft = left.kind === 'points';
      const [inside, boundary, outside] = pointsOnLeft
        ? ['0FFFFF212', 'F0FFFF212', 'FF0FFF212']
        : ['0F2FF1FF2', 'FF20F1FF2', 'FF2FF10F2'];
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-resolve-relate`,
          operation: OPERATION,
          variant: 'resolve-relate',
          bindings: [
            {name: 'candidatePairs', view: candidatePairs, type: 'u32', access: 'read'},
            {
              name: 'pairClassifications',
              view: robust.pairClassifications,
              type: 'u32',
              access: 'read'
            },
            {name: 'matrices', view: matrices, type: 'u32', access: 'read_write'}
          ],
          invocationCount: candidateCapacity,
          declarations: `${SPATIAL_JOIN_WGSL_HELPERS}
const INSIDE: u32 = ${classification.inside}u;
const BOUNDARY: u32 = ${classification.boundary}u;
const UNCERTAIN: u32 = ${classification.uncertain}u;`,
          body: `var matrix = 0u;
  if (candidatePairs[candidatePairsOffset + index * 2u] != NO_FEATURE) {
    let classification = pairClassifications[pairClassificationsOffset + index * 2u + 1u];
    matrix = select(select(select(${packGPUSpatialRelate(outside)}u, ${packGPUSpatialRelate(boundary)}u, classification == BOUNDARY), ${packGPUSpatialRelate(inside)}u, classification == INSIDE), ${uncertainBit}, classification == UNCERTAIN);
  }
  matrices[matricesOffset + index] = matrix;`
        })
      );
      // Candidates the robust classifier could not certify are classified by the f32 relate engine.
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-fallback-relate`,
          operation: OPERATION,
          variant: `fallback-relate-${left.kind}-${right.kind}`,
          bindings: [
            ...sideBindings,
            {name: 'candidatePairs', view: candidatePairs, type: 'u32', access: 'read'},
            {
              name: 'pairClassifications',
              view: robust.pairClassifications,
              type: 'u32',
              access: 'read'
            },
            {name: 'matrices', view: matrices, type: 'u32', access: 'read_write'}
          ],
          invocationCount: candidateCapacity,
          declarations: `${getSpatialRelateWGSL(sideSpecs[0], sideSpecs[1])}
const UNCERTAIN: u32 = ${classification.uncertain}u;`,
          body: `let left = candidatePairs[candidatePairsOffset + index * 2u];
  if (left == NO_FEATURE) { return; }
  if (pairClassifications[pairClassificationsOffset + index * 2u + 1u] != UNCERTAIN) { return; }
  let right = candidatePairs[candidatePairsOffset + index * 2u + 1u];
  matrices[matricesOffset + index] = pairRelate(left, right) | ${uncertainBit};`
        })
      );
    } else {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-relate`,
          operation: OPERATION,
          variant: `relate-${left.kind}-${right.kind}`,
          bindings: [
            ...sideBindings,
            {name: 'candidatePairs', view: candidatePairs, type: 'u32', access: 'read'},
            {name: 'matrices', view: matrices, type: 'u32', access: 'read_write'}
          ],
          // One workgroup per candidate slot: its lanes split the edge loops of the pair.
          invocationCount: candidateCapacity * SPATIAL_RELATE_WORKGROUP_SIZE,
          workgroupSize: SPATIAL_RELATE_WORKGROUP_SIZE,
          guardIndex: false,
          declarations: `${getSpatialRelateWGSL(sideSpecs[0], sideSpecs[1], true)}
${getSpatialRelateWorkgroupWGSL(sideSpecs)}`,
          body: `let slot = index / ${SPATIAL_RELATE_WORKGROUP_SIZE}u;
  let left = candidatePairs[candidatePairsOffset + slot * 2u];
  let right = candidatePairs[candidatePairsOffset + slot * 2u + 1u];
  let matrix = relateWorkgroupPair(left, right, left != NO_FEATURE, localInvocationIndex);
  if (localInvocationIndex == 0u) {
    matrices[matricesOffset + slot] = matrix;
  }`
        })
      );
    }
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-mask`,
        operation: OPERATION,
        variant: 'mask',
        bindings: [
          {name: 'candidatePairs', view: candidatePairs, type: 'u32', access: 'read'},
          {name: 'matrices', view: matrices, type: 'u32', access: 'read'},
          {name: 'flags', view: flags, type: 'u32', access: 'read_write'},
          {name: 'state', view: state, type: 'atomic<u32>', access: 'read_write'},
          ...(this.patternView
            ? [{name: 'patternWords', view: this.patternView, type: 'u32', access: 'read'} as const]
            : [])
        ],
        invocationCount: candidateCapacity,
        declarations: `${SPATIAL_JOIN_WGSL_HELPERS}
${
  this.patternView
    ? getRelateMatchesParametersWGSL(this.patternView.length / GPU_SPATIAL_RELATE_PATTERN_WORDS)
    : getRelateMatchesWGSL(this.relatePatterns)
}`,
        body: `let left = candidatePairs[candidatePairsOffset + index * 2u];
  var matched = false;
  if (left != NO_FEATURE) {
    let right = candidatePairs[candidatePairsOffset + index * 2u + 1u];
    let raw = matrices[matricesOffset + index];
    if ((raw & ${uncertainBit}) != 0u) { atomicAdd(&state[stateOffset + 1u], 1u); }
    let matrix = raw & 0x3ffffu;
    // An all-F matrix marks an empty or invalid feature.
    matched = matrix != 0u && relateMatches(matrix)${context.sameRowTest};
  }
  flags[flagsOffset + index] = select(0u, 1u, matched);`
      })
    );
  }

  /** Adds the anti-join nodes: per-left match counts, flags, scan and compaction of unmatched rows. */
  private addAntiNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    nodes: GPUCommandNode<Parameters>[],
    context: {
      candidatePairs: GraphDataView<'uint32x2'>;
      flags: GraphDataView<'uint32'>;
      leftMatchCounts: GraphDataView<'uint32'>;
      unmatched: GPUCompactOutput;
      state: GraphDataView<'uint32'>;
    }
  ): void {
    const {id, leftCount, candidateCapacity} = this;
    const {candidatePairs, flags, leftMatchCounts, unmatched, state} = context;
    const unmatchedFlags = createTransientView(graph, `${id}-unmatched-flags`, 'uint32', leftCount);
    const unmatchedOffsets = createTransientView(
      graph,
      `${id}-unmatched-offsets`,
      'uint32',
      leftCount
    );
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-count-left-matches`,
        operation: OPERATION,
        variant: 'count-left-matches',
        bindings: [
          {name: 'candidatePairs', view: candidatePairs, type: 'u32', access: 'read'},
          {name: 'flags', view: flags, type: 'u32', access: 'read'},
          {
            name: 'leftMatchCounts',
            view: leftMatchCounts,
            type: 'atomic<u32>',
            access: 'read_write'
          }
        ],
        invocationCount: candidateCapacity,
        body: `if (flags[flagsOffset + index] != 0u) {
    atomicAdd(&leftMatchCounts[leftMatchCountsOffset + candidatePairs[candidatePairsOffset + index * 2u]], 1u);
  }`
      })
    );
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-unmatched-flags`,
        operation: OPERATION,
        variant: 'unmatched-flags',
        bindings: [
          {name: 'leftMatchCounts', view: leftMatchCounts, type: 'u32', access: 'read'},
          {name: 'unmatchedFlags', view: unmatchedFlags, type: 'u32', access: 'read_write'}
        ],
        invocationCount: leftCount,
        body: `unmatchedFlags[unmatchedFlagsOffset + index] = select(0u, 1u, leftMatchCounts[leftMatchCountsOffset + index] == 0u);`
      })
    );
    nodes.push(
      ...new GPUScan({
        id: `${id}-scan-unmatched`,
        input: unmatchedFlags,
        output: unmatchedOffsets,
        mode: 'exclusive'
      }).getCommandNodes(graph)
    );
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-scatter-unmatched`,
        operation: OPERATION,
        variant: 'scatter-unmatched',
        bindings: [
          {name: 'unmatchedFlags', view: unmatchedFlags, type: 'u32', access: 'read'},
          {name: 'unmatchedOffsets', view: unmatchedOffsets, type: 'u32', access: 'read'},
          {name: 'ids', view: unmatched.ids, type: 'u32', access: 'read_write'},
          {name: 'state', view: state, type: 'u32', access: 'read_write'}
        ],
        invocationCount: leftCount,
        declarations: `const UNMATCHED_CAPACITY: u32 = ${unmatched.ids.length}u;`,
        body: `if (unmatchedFlags[unmatchedFlagsOffset + index] != 0u) {
    let slot = unmatchedOffsets[unmatchedOffsetsOffset + index];
    if (slot < UNMATCHED_CAPACITY) { ids[idsOffset + slot] = index; }
  }
  if (index == ${leftCount - 1}u) {
    state[stateOffset + 3u] = unmatchedOffsets[unmatchedOffsetsOffset + index] + unmatchedFlags[unmatchedFlagsOffset + index];
  }`
      })
    );
  }
}
