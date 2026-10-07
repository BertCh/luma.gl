// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {SpatialAnalysisRasterLayer} from '../../engine/layers';

/** Emerging hot spot category colors, index = category code (0 is transparent). */
export const EMERGING_CATEGORY_COLORS: readonly (readonly [number, number, number])[] = [
  [0, 0, 0],
  [255, 224, 102],
  [255, 160, 60],
  [165, 0, 38],
  [215, 48, 39],
  [244, 160, 150],
  [253, 208, 162],
  [200, 100, 200],
  [170, 140, 140],
  [160, 230, 255],
  [90, 170, 255],
  [8, 48, 107],
  [33, 102, 172],
  [150, 190, 230],
  [190, 230, 215],
  [120, 100, 220],
  [130, 150, 170]
];

/** Category names in code order. */
export const EMERGING_CATEGORY_NAMES = [
  'No pattern',
  'New hot spot',
  'Consecutive hot spot',
  'Intensifying hot spot',
  'Persistent hot spot',
  'Diminishing hot spot',
  'Sporadic hot spot',
  'Oscillating hot spot',
  'Historical hot spot',
  'New cold spot',
  'Consecutive cold spot',
  'Intensifying cold spot',
  'Persistent cold spot',
  'Diminishing cold spot',
  'Sporadic cold spot',
  'Oscillating cold spot',
  'Historical cold spot'
] as const;

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
