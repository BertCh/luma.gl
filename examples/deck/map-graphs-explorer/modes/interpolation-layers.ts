// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {MapGraphsSegmentLayer} from '../map-graphs-layers';

/**
 * Returns a segment layer class for contours of a raster whose extent changes every frame.
 *
 * The stock segment layer maps grid-unit vertices to meters with a literal `positionScale` and
 * `positionOffset`, which is one frame stale when the extent follows the camera. The returned
 * subclass instead reads `[minX, minY, maxX, maxY]` from the first four floats of the `extent`
 * buffer (the same per-frame parameter buffer the interpolation recipe reads) in the vertex
 * shader, so contours and raster always agree. Row 0 of the raster is the south edge. The raster
 * dimensions are compiled into the shader, hence a class per grid size.
 *
 * @param gridSize `[columns, rows]` of the raster the segment vertices were extracted from.
 */
export function createExtentFollowingSegmentLayer(
  gridSize: readonly [number, number]
): typeof MapGraphsSegmentLayer {
  const [columns, rows] = gridSize;
  return class ExtentFollowingSegmentLayer extends MapGraphsSegmentLayer {
    static override layerName = 'ExtentFollowingSegmentLayer';

    protected override getShaderSource(): string {
      const source = super.getShaderSource();
      const replacement = /* wgsl */ `fn getMapGraphsPosition(position: vec2<f32>) -> vec2<f32> {
  let minimum = vec2<f32>(styleExtent[0], styleExtent[1]);
  let maximum = vec2<f32>(styleExtent[2], styleExtent[3]);
  return minimum + position * (maximum - minimum) / vec2<f32>(${columns}.0, ${rows}.0);
}`;
      const patched = source.replace(
        /fn getMapGraphsPosition\(position: vec2<f32>\) -> vec2<f32> \{[^}]*\}/,
        replacement
      );
      if (patched === source) {
        throw new Error('ExtentFollowingSegmentLayer could not patch getMapGraphsPosition');
      }
      return patched;
    }
  };
}
