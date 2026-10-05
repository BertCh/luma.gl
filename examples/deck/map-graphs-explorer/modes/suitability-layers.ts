// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {MapGraphsRasterLayer} from '../map-graphs-layers';

/**
 * Returns a raster layer class that reads cell values starting at `cellOffset` in its `values`
 * buffer. The Suitability mode keeps its three rating rasters in one band-sequential stack (the
 * layout `GPUWeightedOverlay` reads), so each band is drawn by a layer class with that band's
 * offset compiled into the shader; the stock raster layer has no value offset.
 *
 * @param cellOffset Number of cells to skip at the start of the `values` buffer.
 */
export function createOffsetRasterLayerClass(cellOffset: number): typeof MapGraphsRasterLayer {
  return class OffsetRasterLayer extends MapGraphsRasterLayer {
    static override layerName = 'OffsetRasterLayer';

    protected override getShaderSource(): string {
      const source = super.getShaderSource();
      const patched = source.replace(
        'getMapGraphsValueRow(u32(row * columns + column))',
        `getMapGraphsValueRow(u32(row * columns + column) + ${cellOffset}u)`
      );
      if (patched === source) {
        throw new Error('OffsetRasterLayer could not patch the raster fragment shader');
      }
      return patched;
    }
  };
}
