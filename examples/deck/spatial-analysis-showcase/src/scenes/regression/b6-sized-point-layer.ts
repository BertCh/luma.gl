// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {SpatialAnalysisPointLayer} from '../../engine/layers';

const SIZE_LINE = 'project_pixel_size_to_clipspace(corner * spatialAnalysisStyle.sizePixels)';

/** WGSL appended to the shared point shader: the disc radius grows with the row's value. */
const SIZE_WGSL = /* wgsl */ `
// Disc radius in pixels for a row whose float32 value is mapped over valueRange (sqrt scale, so
// the disc area is proportional to the value).
fn getSizedRadiusPixels(row: u32) -> f32 {
  let value = bitcast<f32>(styleValues[getSpatialAnalysisValueRow(row)]);
  let t = clamp(value / max(spatialAnalysisStyle.valueRange.y, 1e-20), 0.0, 1.0);
  return spatialAnalysisStyle.sizePixels * (0.3 + 1.7 * sqrt(t));
}
`;

/**
 * A point layer whose disc radius is proportional to the square root of the row's float32 value
 * (relative to the top of `valueRange`). `radiusPixels` is the size of a disc at the top of the
 * range; rows at or below `discardAtOrBelow` are hidden.
 */
export class SizedPointLayer extends SpatialAnalysisPointLayer {
  static override layerName = 'SizedPointLayer';

  protected override getShaderSource(): string {
    const source = super.getShaderSource();
    if (!source.includes(SIZE_LINE)) {
      throw new Error('SizedPointLayer: the shared point shader changed; update the patch');
    }
    return `${source.replace(SIZE_LINE, 'project_pixel_size_to_clipspace(corner * getSizedRadiusPixels(row))')}\n${SIZE_WGSL}`;
  }
}
