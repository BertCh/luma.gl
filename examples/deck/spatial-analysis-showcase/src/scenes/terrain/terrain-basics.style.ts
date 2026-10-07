// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Pure (luma-free) pieces of the terrain-basics scene shared by the scene file (loaded by the
 * gallery) and its compute module: the option state, value ranges and paint of every product.
 */

import type {RampName} from '../../engine/ramps';
import type {PaintSpec} from './b14a-colorize';

type GPUTerrainRGBEncoding = 'terrarium' | 'mapbox';
type GPUTerrainRuggednessAlgorithm = 'riley' | 'wilson';
type GPUTerrainRuggednessEdgeMode = 'nodata' | 'extrapolate';
type GPUTerrainSlopeUnits = 'degrees' | 'percent';

/** Products the scene can display. */
export type BasicsProduct =
  | 'elevation'
  | 'slope'
  | 'aspect'
  | 'hillshade'
  | 'tpi'
  | 'tri'
  | 'roughness'
  | 'vrm';

/** Option state of the terrain-basics scene. */
export type BasicsOptions = {
  product: BasicsProduct;
  encoding: GPUTerrainRGBEncoding;
  validRange: 'default' | 'tight' | 'off';
  alphaNoData: boolean;
  clampBathymetry: boolean;
  missingTile: boolean;
  spikeDensity: number;
  repairSpikes: boolean;
  slopeUnits: GPUTerrainSlopeUnits;
  zFactor: number;
  sunAzimuth: number;
  sunAltitude: number;
  borderMode: 'clamp' | 'nodata';
  triAlgorithm: GPUTerrainRuggednessAlgorithm;
  edgeMode: GPUTerrainRuggednessEdgeMode;
  vrmRadius: number;
  ramp: RampName;
  rangeScale: number;
  opacity: number;
  underlay: boolean;
};

/** Valid height ranges of the decode, meters. */
export const VALID_RANGES: Record<BasicsOptions['validRange'], readonly [number, number]> = {
  default: [-11000, 9000],
  tight: [1800, 4200],
  off: [-Infinity, Infinity]
};

/** Nominal color range of a product before the range scale is applied. */
export function getNominalRange(
  product: BasicsProduct,
  slopeUnits: GPUTerrainSlopeUnits
): {low: number; high: number} {
  switch (product) {
    case 'elevation':
      return {low: 1500, high: 4500};
    case 'slope':
      return {low: 0, high: slopeUnits === 'degrees' ? 70 : 270};
    case 'hillshade':
      return {low: 0, high: 1};
    case 'tpi':
      return {low: -15, high: 15};
    case 'tri':
      return {low: 0, high: 60};
    case 'roughness':
      return {low: 0, high: 120};
    case 'vrm':
      return {low: 0, high: 0.25};
    default:
      return {low: 0, high: 1};
  }
}

/** Paint of a product for the given option state. */
export function getBasicsPaint(state: BasicsOptions): Partial<PaintSpec> {
  const {low, high} = getNominalRange(state.product, state.slopeUnits);
  const scale = state.rangeScale;
  if (state.product === 'aspect') return {mode: 'aspect', low: 0, high: 360, alpha: 1};
  if (state.product === 'hillshade') {
    return {mode: 'ramp', ramp: 'grayscale', low: 0.05, high: 0.95, alpha: 1, fadeMiddle: false};
  }
  if (state.product === 'tpi') {
    return {
      mode: 'ramp',
      ramp: 'diverging',
      low: low * scale,
      high: high * scale,
      alpha: 1,
      fadeMiddle: true
    };
  }
  return {
    mode: 'ramp',
    ramp: state.ramp,
    low: low * scale,
    high: high * scale,
    alpha: 1,
    fadeMiddle: false
  };
}
