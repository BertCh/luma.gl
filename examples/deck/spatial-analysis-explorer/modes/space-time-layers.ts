// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {SpatialAnalysisRasterLayer} from '../spatial-analysis-layers';

/**
 * WGSL helpers appended to the shared style shader. The shared category palette holds 8 colors,
 * so the 17 emerging hot spot categories need their own table, and the shared scalar colormaps
 * have no diverging ramp for signed change.
 */
const SPACE_TIME_WGSL = /* wgsl */ `
// Emerging hot spot categories: 0 no pattern, 1..8 hot, 9..16 cold (same order, new to historical).
fn spaceTimeCategoryColor(category: u32) -> vec4<f32> {
  let colors = array<vec3<f32>, 17>(
    vec3<f32>(0.0, 0.0, 0.0),
    vec3<f32>(255.0, 224.0, 102.0), vec3<f32>(255.0, 160.0, 60.0), vec3<f32>(165.0, 0.0, 38.0),
    vec3<f32>(215.0, 48.0, 39.0), vec3<f32>(244.0, 160.0, 150.0), vec3<f32>(253.0, 208.0, 162.0),
    vec3<f32>(200.0, 100.0, 200.0), vec3<f32>(170.0, 140.0, 140.0),
    vec3<f32>(160.0, 230.0, 255.0), vec3<f32>(90.0, 170.0, 255.0), vec3<f32>(8.0, 48.0, 107.0),
    vec3<f32>(33.0, 102.0, 172.0), vec3<f32>(150.0, 190.0, 230.0), vec3<f32>(190.0, 230.0, 215.0),
    vec3<f32>(120.0, 100.0, 220.0), vec3<f32>(130.0, 150.0, 170.0)
  );
  if (category == 0u || category > 16u) {
    return vec4<f32>(0.0);
  }
  return vec4<f32>(colors[category] / 255.0, 0.9);
}

// Blue (decrease) through near-white (no change) to red (increase); t is in [0, 1].
fn spaceTimeDiverging(t: f32) -> vec3<f32> {
  let stops = array<vec3<f32>, 5>(
    vec3<f32>(33.0, 102.0, 172.0), vec3<f32>(146.0, 197.0, 222.0), vec3<f32>(247.0, 247.0, 247.0),
    vec3<f32>(244.0, 165.0, 130.0), vec3<f32>(178.0, 24.0, 43.0)
  );
  let scaled = clamp(t, 0.0, 1.0) * 4.0;
  let index = min(u32(floor(scaled)), 3u);
  return mix(stops[index], stops[index + 1u], scaled - f32(index)) / 255.0;
}
`;

const CATEGORY_LINE = 'return spatialAnalysisStyle.palette[raw % size];';
const SCALAR_LINE = 'var rgb = vec3<f32>(t);';

/**
 * A raster layer with two extra colormaps, implemented by extending the shared shader source:
 * the `category` colormap uses the 17 emerging hot spot colors (value 0 is transparent), and the
 * `grayscale` colormap becomes a diverging blue-white-red ramp centered on the middle of
 * `valueRange`.
 */
export class SpaceTimeRasterLayer extends SpatialAnalysisRasterLayer {
  static override layerName = 'SpaceTimeRasterLayer';

  protected override getShaderSource(): string {
    const source = super.getShaderSource();
    if (!source.includes(CATEGORY_LINE) || !source.includes(SCALAR_LINE)) {
      throw new Error('SpaceTimeRasterLayer: the shared style shader changed; update the patch');
    }
    return `${source.replace(CATEGORY_LINE, 'return spaceTimeCategoryColor(raw);').replace(
      SCALAR_LINE,
      `var rgb = vec3<f32>(t);
  if (colormap == 3u) {
    rgb = spaceTimeDiverging(t);
  }`
    )}\n${SPACE_TIME_WGSL}`;
  }
}
