// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Shared, aspect-preserving projection from native coordinates to SVG pixels (y down). */
export type GeometryProjection = {
  model: 'geographic' | 'projected' | 'unknown';
  /** Projection and its limitations, printed beside the figure. */
  statement: string;
  project(x: number, y: number): [number, number];
  /** Native data extent shared by every panel. */
  extent: [[number, number], [number, number]];
};

/** Fits native geometry uniformly into a centred viewport without stretching either axis. */
export function createGeometryProjection(props: {
  coordinateModel: 'geographic' | 'projected' | 'unknown';
  points: Iterable<readonly [number, number]>;
  width: number;
  height: number;
  padding?: number;
  unit?: string;
}): GeometryProjection {
  let minimumX = Infinity;
  let minimumY = Infinity;
  let maximumX = -Infinity;
  let maximumY = -Infinity;
  for (const [horizontal, vertical] of props.points) {
    if (!Number.isFinite(horizontal) || !Number.isFinite(vertical)) continue;
    minimumX = Math.min(minimumX, horizontal);
    minimumY = Math.min(minimumY, vertical);
    maximumX = Math.max(maximumX, horizontal);
    maximumY = Math.max(maximumY, vertical);
  }
  if (minimumX === Infinity) minimumX = maximumX = minimumY = maximumY = 0;
  const middleX = (minimumX + maximumX) / 2;
  const middleY = (minimumY + maximumY) / 2;
  if (minimumX === maximumX) {
    minimumX -= 1;
    maximumX += 1;
  }
  if (minimumY === maximumY) {
    minimumY -= 1;
    maximumY += 1;
  }
  const longitudeScale =
    props.coordinateModel === 'geographic' ? Math.abs(Math.cos((middleY * Math.PI) / 180)) : 1;
  const padding = Math.max(0, props.padding ?? 8);
  const scale = Math.min(
    Math.max(0, props.width - 2 * padding) / ((maximumX - minimumX) * longitudeScale),
    Math.max(0, props.height - 2 * padding) / (maximumY - minimumY)
  );
  return {
    model: props.coordinateModel,
    statement:
      props.coordinateModel === 'geographic'
        ? `Simple equirectangular projection (x scaled by cos ${Number(middleY.toFixed(2))}°); not equal-area.`
        : props.coordinateModel === 'projected'
          ? `Planar coordinates${props.unit ? ` (${props.unit})` : ' (unit unknown)'}; uniform scale.`
          : 'Coordinate model unknown: drawn as planar x, y.',
    extent: [
      [minimumX, minimumY],
      [maximumX, maximumY]
    ],
    project: (horizontal, vertical) => [
      props.width / 2 + (horizontal - middleX) * longitudeScale * scale,
      props.height / 2 - (vertical - middleY) * scale
    ]
  };
}
