// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getSplatLogisticCdf} from './splat-antialiasing';

/**
 * Non-destructive clipping of Gaussians to a half-space, slab, or convex prism.
 *
 * A Gaussian is a volume, not a point, so testing only its center against a clip plane cuts the
 * scene along a visibly ragged boundary: a large splat straddling the plane either disappears
 * whole or stays whole. Measuring the signed distance in units of the splat's *own* extent along
 * the plane normal and attenuating opacity by the resulting partial coverage gives a boundary that
 * follows the geometry, for one extra term per plane and no ray tracing.
 *
 * Nothing is removed from the source data: clipping is a per-frame opacity factor evaluated in the
 * existing projection pass, so a clip region can animate freely.
 */

/** One clip plane in the same space as the source Gaussian centers. */
export type SplatClipPlane = {
  /** Plane normal. The kept half-space is the side the normal points toward. Need not be unit. */
  normal: readonly [number, number, number];
  /** Plane constant, so the plane is the set where `dot(normal, position) + distance == 0`. */
  distance: number;
};

/** How several clip planes combine into one region. */
export type SplatClipCombineMode =
  /** Keep Gaussians inside every plane: a convex prism, slab, or half-space. */
  | 'intersection'
  /** Keep Gaussians inside any plane: the union of half-spaces. */
  | 'union';

/** A region that attenuates Gaussian opacity by partial coverage rather than removing rows. */
export type SplatClipRegion = {
  /** Up to {@link MAXIMUM_SPLAT_CLIP_PLANES} planes bounding the region. */
  planes: readonly SplatClipPlane[];
  /** How the planes combine. Defaults to `'intersection'`. */
  combine?: SplatClipCombineMode;
  /**
   * Width of the soft boundary as a multiple of each Gaussian's own extent along the normal.
   *
   * `1` fades a Gaussian over roughly its own standard deviation, which is the width at which the
   * boundary looks like a cut through a volume rather than a cut through a point set. Values below
   * about `0.05` approach a hard, center-based cut. Defaults to `1`.
   */
  softness?: number;
  /** Keep the complement of the region instead of the region. Defaults to `false`. */
  invert?: boolean;
};

/** Planes one clip region may carry; eight bounds a frustum, a slab, or a rectangular prism. */
export const MAXIMUM_SPLAT_CLIP_PLANES = 8;

/** Padded byte size of the clip-plane uniform block bound by the projection pass. */
export const SPLAT_CLIP_UNIFORM_BYTE_LENGTH = MAXIMUM_SPLAT_CLIP_PLANES * 16 + 16;

/** Smallest extent treated as resolvable when measuring a Gaussian against a plane. */
const MINIMUM_CLIP_EXTENT = 1e-6;

/**
 * Packs one clip region into the uniform block layout the projection shader expects.
 *
 * Plane normals are normalized here so the shader can treat the signed distance as a world-space
 * length without a per-splat square root.
 *
 * @throws If the region declares more than {@link MAXIMUM_SPLAT_CLIP_PLANES} planes.
 */
export function packSplatClipUniforms(region: SplatClipRegion | undefined): ArrayBuffer {
  const uniformData = new ArrayBuffer(SPLAT_CLIP_UNIFORM_BYTE_LENGTH);
  const floatValues = new Float32Array(uniformData);
  const integerValues = new Uint32Array(uniformData);
  const planes = region?.planes ?? [];
  if (planes.length > MAXIMUM_SPLAT_CLIP_PLANES) {
    throw new RangeError(
      `Gaussian splat clip regions support at most ${MAXIMUM_SPLAT_CLIP_PLANES} planes`
    );
  }

  let planeCount = 0;
  for (const plane of planes) {
    const [normalX, normalY, normalZ] = plane.normal;
    const length = Math.hypot(normalX, normalY, normalZ);
    if (!(length > MINIMUM_CLIP_EXTENT) || !Number.isFinite(plane.distance)) {
      continue;
    }
    const offset = planeCount * 4;
    floatValues[offset] = normalX / length;
    floatValues[offset + 1] = normalY / length;
    floatValues[offset + 2] = normalZ / length;
    floatValues[offset + 3] = plane.distance / length;
    planeCount++;
  }

  const scalarOffset = MAXIMUM_SPLAT_CLIP_PLANES * 4;
  integerValues[scalarOffset] = planeCount;
  integerValues[scalarOffset + 1] = region?.combine === 'union' ? 1 : 0;
  floatValues[scalarOffset + 2] = Math.max(region?.softness ?? 1, 0);
  integerValues[scalarOffset + 3] = region?.invert ? 1 : 0;
  return uniformData;
}

/** Whether a region would attenuate anything, so the shader branch can be skipped entirely. */
export function isSplatClipRegionActive(region: SplatClipRegion | undefined): boolean {
  return Boolean(region?.planes.some(plane => Math.hypot(...plane.normal) > MINIMUM_CLIP_EXTENT));
}

/**
 * Returns the CPU-side partial coverage one clip region leaves for a single Gaussian.
 *
 * `scaledAxes` are the three world-space one-sigma axes of the Gaussian, as produced by
 * `getQuaternionScaledAxes`. The extent along a plane normal is `sqrt(n^T Sigma n)`, which for
 * axes `a0, a1, a2` is `sqrt(dot(a0, n)^2 + dot(a1, n)^2 + dot(a2, n)^2)`.
 *
 * @returns A factor in `[0, 1]` to multiply into opacity.
 */
export function getSplatClipCoverage(
  region: SplatClipRegion | undefined,
  position: readonly [number, number, number],
  scaledAxes: readonly (readonly [number, number, number])[]
): number {
  if (!region || !isSplatClipRegionActive(region)) {
    return 1;
  }

  const isUnion = region.combine === 'union';
  const softness = Math.max(region.softness ?? 1, 0);
  let coverage = isUnion ? 0 : 1;
  for (const plane of region.planes) {
    const [normalX, normalY, normalZ] = plane.normal;
    const length = Math.hypot(normalX, normalY, normalZ);
    if (!(length > MINIMUM_CLIP_EXTENT)) {
      continue;
    }
    const unitX = normalX / length;
    const unitY = normalY / length;
    const unitZ = normalZ / length;
    const signedDistance =
      unitX * position[0] + unitY * position[1] + unitZ * position[2] + plane.distance / length;

    let varianceAlongNormal = 0;
    for (const axis of scaledAxes) {
      const projection = axis[0] * unitX + axis[1] * unitY + axis[2] * unitZ;
      varianceAlongNormal += projection * projection;
    }
    const extent = Math.sqrt(varianceAlongNormal) * softness;
    const planeCoverage =
      extent > MINIMUM_CLIP_EXTENT
        ? getSplatLogisticCdf(signedDistance / extent)
        : signedDistance >= 0
          ? 1
          : 0;
    coverage = isUnion
      ? coverage + planeCoverage - coverage * planeCoverage
      : coverage * planeCoverage;
  }

  return region.invert ? 1 - coverage : coverage;
}

/**
 * Shared WGSL clip evaluation matching {@link getSplatClipCoverage}.
 *
 * Expects a `clipUniforms` binding of the layout {@link packSplatClipUniforms} produces and the
 * logistic CDF helper from `SPLAT_ANTIALIASING_WGSL`.
 *
 * @internal
 */
export const SPLAT_CLIPPING_WGSL = /* wgsl */ `\
const SPLAT_MINIMUM_CLIP_EXTENT: f32 = 1e-6;

struct GraphSplatClipUniforms {
  planes: array<vec4<f32>, ${MAXIMUM_SPLAT_CLIP_PLANES}>,
  planeCount: u32,
  combineUnion: u32,
  softness: f32,
  invert: u32,
};

/**
 * Partial coverage a clip region leaves for one Gaussian.
 *
 * \`axis0\`, \`axis1\` and \`axis2\` are the rotated one-sigma world-space axes, so the extent along
 * a unit plane normal is the length of their projections onto it.
 */
fn getSplatClipCoverage(
  clip: GraphSplatClipUniforms,
  position: vec3<f32>,
  axis0: vec3<f32>,
  axis1: vec3<f32>,
  axis2: vec3<f32>
) -> f32 {
  if (clip.planeCount == 0u) {
    return 1.0;
  }
  let isUnion = clip.combineUnion != 0u;
  var coverage = select(1.0, 0.0, isUnion);
  for (var planeIndex = 0u; planeIndex < clip.planeCount; planeIndex++) {
    let plane = clip.planes[planeIndex];
    let normal = plane.xyz;
    let signedDistance = dot(normal, position) + plane.w;
    let projections = vec3<f32>(dot(axis0, normal), dot(axis1, normal), dot(axis2, normal));
    let extent = sqrt(dot(projections, projections)) * clip.softness;
    var planeCoverage = select(0.0, 1.0, signedDistance >= 0.0);
    if (extent > SPLAT_MINIMUM_CLIP_EXTENT) {
      planeCoverage = getSplatLogisticCdf(signedDistance / extent);
    }
    if (isUnion) {
      coverage = coverage + planeCoverage - coverage * planeCoverage;
    } else {
      coverage = coverage * planeCoverage;
    }
  }
  return select(coverage, 1.0 - coverage, clip.invert != 0u);
}
`;
