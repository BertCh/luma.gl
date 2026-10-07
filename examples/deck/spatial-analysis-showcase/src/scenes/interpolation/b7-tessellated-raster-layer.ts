// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {SpatialAnalysisRasterLayer} from '../../engine/layers';

/** Subdivisions per side of the quad. */
const SUBDIVISIONS = 48;

/**
 * A raster layer for extents of hundreds of kilometers.
 *
 * The stock raster layer draws one quad whose four corners are projected exactly; inside the quad
 * the world position is interpolated linearly in screen space, which is wrong by the curvature of
 * the Web Mercator latitude scale. Over a 1,800 km extent that is a shift of about 40 km in the
 * middle of the raster, which would misregister the surface against the stations drawn on top of
 * it. This subclass draws `SUBDIVISIONS x SUBDIVISIONS` quads and projects each of their corners
 * exactly, which brings the error under a kilometer. Square-grid binning only.
 */
export class B7TessellatedRasterLayer extends SpatialAnalysisRasterLayer {
  static override layerName = 'B7TessellatedRasterLayer';

  protected override getVertexCount(): number {
    return 6 * SUBDIVISIONS * SUBDIVISIONS;
  }

  protected override getShaderSource(): string {
    const source = super.getShaderSource();
    const start = source.indexOf('@vertex fn vertexMain');
    const end = source.indexOf('@fragment fn fragmentMain');
    if (start < 0 || end < start) {
      throw new Error('B7TessellatedRasterLayer could not find the raster vertex shader');
    }
    const vertex = /* wgsl */ `@vertex fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> RasterVertexOutput {
  let corners = array<vec2<f32>, 6>(
    vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0),
    vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0)
  );
  let quad = vertexIndex / 6u;
  let cell = corners[vertexIndex % 6u] + vec2<f32>(f32(quad % ${SUBDIVISIONS}u), f32(quad / ${SUBDIVISIONS}u));
  let minimum = vec2<f32>(rasterBounds[0], rasterBounds[1]);
  let maximum = vec2<f32>(rasterBounds[2], rasterBounds[3]);
  let worldPosition = minimum + cell / ${SUBDIVISIONS}.0 * (maximum - minimum);
  var output: RasterVertexOutput;
  output.position = projectSpatialAnalysisPosition(worldPosition);
  output.worldPosition = worldPosition;
  return output;
}

`;
    return source.slice(0, start) + vertex + source.slice(end);
  }
}
