// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {SpatialAnalysisRasterLayer} from '../../engine/layers';

/** Web Mercator sphere radius in meters, as used by `wildfire-data.ts`. */
const MERCATOR_RADIUS = 6378137;

/**
 * A raster layer for grids that are uniform in Web Mercator meters (the Terrarium DEM): `bounds`
 * are absolute EPSG:3857 meters, and each quad corner is converted back to longitude and latitude
 * before projection (use `coordinateSystem: COORDINATE_SYSTEM.LNGLAT`). The cell lookup still runs
 * in Mercator meters, so cells land exactly under the map at any extent, with one quad and no
 * tessellation. The stock layer's local-meter projection drifts by kilometers over 500 km.
 */
export class WildfireMercatorRasterLayer extends SpatialAnalysisRasterLayer {
  static override layerName = 'WildfireMercatorRasterLayer';

  protected override getShaderSource(): string {
    const source = super.getShaderSource();
    const target = 'output.position = projectSpatialAnalysisPosition(worldPosition);';
    if (!source.includes(target)) {
      throw new Error('WildfireMercatorRasterLayer: raster shader changed, update the replacement');
    }
    return `
fn wildfireMercatorToLngLat(position: vec2<f32>) -> vec2<f32> {
  let radius = ${MERCATOR_RADIUS.toFixed(1)};
  return vec2<f32>(
    degrees(position.x / radius),
    degrees(2.0 * atan(exp(position.y / radius)) - 1.5707963267948966)
  );
}
${source.replace(target, 'output.position = projectSpatialAnalysisPosition(wildfireMercatorToLngLat(worldPosition));')}`;
  }
}
