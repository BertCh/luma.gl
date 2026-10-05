// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Number of float32 values in a packed tile LOD view. See {@link GPU_TILE_LOD_VIEW_OFFSETS}. */
export const GPU_TILE_LOD_VIEW_LENGTH = 56;

/** Float offsets of each field in a packed tile LOD view. Unlisted slots are reserved zeros. */
export const GPU_TILE_LOD_VIEW_OFFSETS = Object.freeze({
  /** 24 floats: six inward normalized planes `(nx, ny, nz, d)`; a sphere is visible when `dot(n, c) + d >= -r`. */
  frustumPlanes: 0,
  /** 16 floats, column-major view-projection matrix, used only by foveation. */
  viewProjectionMatrix: 24,
  /** 3 floats, camera position in hierarchy space. */
  cameraPosition: 40,
  /** Pixels per world unit at unit distance. */
  pixelProjectionScale: 43,
  /** Pixels; refine while the weighted error exceeds this. */
  maximumScreenSpaceError: 44,
  /** 2 floats, viewport width and height in the same pixel unit. */
  viewportSize: 45,
  /** 2 floats, viewport-normalized gaze position, x right and y down. */
  foveationCenter: 48,
  /** Viewport-normalized full-detail radius. */
  foveationRadius: 50,
  /** Error relaxation outside the radius; 0 disables foveation. */
  foveationStrength: 51,
  /** World units; `<= 0` disables distance falloff. */
  focusDistance: 52,
  /** Exponent; 0 disables distance falloff. */
  distanceFalloff: 53
});

/** Gaze-dependent error relaxation. */
export type GPUTileLODFoveation = {
  /** Viewport-normalized gaze position (x right, y down). Defaults to `[0.5, 0.5]`. */
  center?: readonly [number, number];
  /** Viewport-normalized radius that keeps full detail. Defaults to `0.15`. */
  radius?: number;
  /** Error relaxation outside the radius. `0` (default) disables foveation. */
  strength?: number;
};

/** CPU view description packed by {@link getGPUTileLODViewParameterValues}. */
export type GPUTileLODViewProps = {
  /** Column-major view-projection matrix (16 numbers) mapping hierarchy space to clip space. */
  viewProjectionMatrix: readonly number[];
  /** Camera position in hierarchy space. */
  cameraPosition: readonly [number, number, number];
  /** Viewport size in pixels, in the same unit as `maximumScreenSpaceError`. */
  viewportSize: readonly [number, number];
  /** Maximum screen-space error in pixels. */
  maximumScreenSpaceError: number;
  /** Vertical field of view in radians, used when `pixelProjectionScale` is omitted. Defaults to 60 degrees. */
  verticalFieldOfView?: number;
  /** Explicit pixels per world unit at unit distance. Overrides `verticalFieldOfView`. */
  pixelProjectionScale?: number;
  /** Explicit 24-float inward planes. Defaults to planes extracted from `viewProjectionMatrix`. */
  frustumPlanes?: readonly number[];
  /** Optional gaze-dependent relaxation. */
  foveation?: GPUTileLODFoveation;
  /** Distance where distance falloff starts, in hierarchy units. */
  focusDistance?: number;
  /** Exponent for error relaxation past `focusDistance`. `0` (default) disables it. */
  distanceFalloff?: number;
};

/** Packs view state into a {@link GPU_TILE_LOD_VIEW_LENGTH}-float array, writing into `target` when given. */
export function getGPUTileLODViewParameterValues(
  props: GPUTileLODViewProps,
  target: Float32Array = new Float32Array(GPU_TILE_LOD_VIEW_LENGTH)
): Float32Array {
  if (props.viewProjectionMatrix.length !== 16) {
    throw new Error(
      'getGPUTileLODViewParameterValues viewProjectionMatrix must contain 16 numbers'
    );
  }
  if (props.frustumPlanes && props.frustumPlanes.length !== 24) {
    throw new Error('getGPUTileLODViewParameterValues frustumPlanes must contain 24 numbers');
  }
  if (target.length < GPU_TILE_LOD_VIEW_LENGTH) {
    throw new Error(
      `getGPUTileLODViewParameterValues target must hold ${GPU_TILE_LOD_VIEW_LENGTH} values`
    );
  }
  const offsets = GPU_TILE_LOD_VIEW_OFFSETS;
  const fieldOfView = Math.min(
    Math.max(props.verticalFieldOfView ?? Math.PI / 3, 1e-6),
    Math.PI - 1e-6
  );
  target.fill(0, 0, GPU_TILE_LOD_VIEW_LENGTH);
  target.set(
    props.frustumPlanes ?? getGPUTileLODFrustumPlanes(props.viewProjectionMatrix),
    offsets.frustumPlanes
  );
  target.set(props.viewProjectionMatrix, offsets.viewProjectionMatrix);
  target.set(props.cameraPosition, offsets.cameraPosition);
  target[offsets.pixelProjectionScale] =
    props.pixelProjectionScale ?? props.viewportSize[1] / (2 * Math.tan(fieldOfView / 2));
  target[offsets.maximumScreenSpaceError] = props.maximumScreenSpaceError;
  target.set(props.viewportSize, offsets.viewportSize);
  target.set(props.foveation?.center ?? [0.5, 0.5], offsets.foveationCenter);
  target[offsets.foveationRadius] = props.foveation?.radius ?? 0.15;
  target[offsets.foveationStrength] = props.foveation?.strength ?? 0;
  target[offsets.focusDistance] = props.focusDistance ?? 0;
  target[offsets.distanceFalloff] = props.distanceFalloff ?? 0;
  return target;
}

/**
 * Extracts six normalized inward frustum planes (left, right, bottom, top, near, far) from a
 * column-major matrix with the Gribb-Hartmann method and the OpenGL clip convention. The GL near
 * plane is conservative for WebGPU `[0, 1]` depth matrices too.
 */
export function getGPUTileLODFrustumPlanes(viewProjectionMatrix: readonly number[]): Float32Array {
  const m = viewProjectionMatrix;
  const row = (k: number) => [m[k], m[4 + k], m[8 + k], m[12 + k]];
  const [row0, row1, row2, row3] = [row(0), row(1), row(2), row(3)];
  const combine = (sign: number, other: number[]) =>
    row3.map((value, i) => value + sign * other[i]);
  const planes = [
    combine(1, row0),
    combine(-1, row0),
    combine(1, row1),
    combine(-1, row1),
    combine(1, row2),
    combine(-1, row2)
  ];
  const result = new Float32Array(24);
  for (const [planeIndex, plane] of planes.entries()) {
    const length = Math.hypot(plane[0], plane[1], plane[2]);
    if (length > 0) {
      result.set(
        plane.map(value => value / length),
        planeIndex * 4
      );
    }
  }
  return result;
}
