// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {SpatialAnalysisRasterLayer} from '../../engine/layers';
import {EMERGING_CATEGORY_COLORS} from './b13-categories';

const wgslColor = (color: readonly number[]) =>
  `vec3<f32>(${color[0].toFixed(1)}, ${color[1].toFixed(1)}, ${color[2].toFixed(1)})`;

/** WGSL helper that maps the 17 emerging hot spot category codes to colors. */
const CATEGORY_WGSL = /* wgsl */ `
fn emergingCategoryColor(category: u32) -> vec4<f32> {
  let colors = array<vec3<f32>, 17>(
    ${EMERGING_CATEGORY_COLORS.map(wgslColor).join(', ')}
  );
  if (category == 0u || category > 16u) {
    return vec4<f32>(0.0);
  }
  return vec4<f32>(colors[category] / 255.0, 0.9);
}
`;

const CATEGORY_LINE = 'return spatialAnalysisStyle.palette[raw % size];';

/**
 * A raster layer whose `category` colormap knows the 17 emerging hot spot categories. The shared
 * layer has an 8-color palette, so this extends its shader source with a 17-entry table (value 0
 * is transparent).
 */
export class EmergingCategoryRasterLayer extends SpatialAnalysisRasterLayer {
  static override layerName = 'EmergingCategoryRasterLayer';

  protected override getShaderSource(): string {
    const source = super.getShaderSource();
    if (!source.includes(CATEGORY_LINE)) {
      throw new Error('EmergingCategoryRasterLayer: the shared style shader changed; update it');
    }
    return `${source.replace(CATEGORY_LINE, 'return emergingCategoryColor(raw);')}\n${CATEGORY_WGSL}`;
  }
}
