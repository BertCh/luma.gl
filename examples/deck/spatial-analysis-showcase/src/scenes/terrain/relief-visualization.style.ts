// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Pure (luma-free) pieces of the relief-visualization scene: option state, product table and the
 * Alpine elevation tint, shared by the scene file and its compute module.
 */

import type {RampName} from '../../engine/ramps';

/** Every product the scene can display. */
export type ReliefProduct =
  | 'hillshade'
  | 'mdow'
  | 'imhof'
  | 'texture'
  | 'sky-view'
  | 'anisotropic'
  | 'positive-openness'
  | 'negative-openness'
  | 'slrm'
  | 'msrm'
  | 'local-dominance'
  | 'vat';

/** Blend mode choice of a VAT layer; `preset` keeps the preset's mode. */
export type VatBlendChoice =
  | 'preset'
  | 'normal'
  | 'multiply'
  | 'screen'
  | 'overlay'
  | 'soft-light'
  | 'luminosity';

/** Option state of the relief-visualization scene. */
export type ReliefOptions = {
  product: ReliefProduct;
  // Light
  lightAzimuth: number;
  lightAltitude: number;
  zFactor: number;
  swissLight: 'imhof-swing' | 'mdow' | 'fixed';
  imhofSwing: number;
  curvatureStrength: number;
  contrastStrength: number;
  contrastHighElevation: number;
  hillshadeStrength: number;
  skyViewStrength: number;
  textureStrength: number;
  exposure: number;
  elevationTint: boolean;
  tintStrength: number;
  // Texture shading
  textureLevels: number;
  textureBaseSigma: number;
  textureHasNodata: boolean;
  textureDownsample: boolean;
  textureDetail: number;
  textureGain: number;
  // Horizon, sky-view, openness
  horizonDirections: '8' | '16' | '32';
  horizonRadius: '32' | '64' | '96' | '192' | '384';
  horizonAlgorithm: 'march' | 'sweep';
  horizonGrowth: number;
  anisotropyAzimuth: number;
  anisotropyLevel: number;
  anisotropyMinimumWeight: number;
  // RVT
  rvtExaggeration: number;
  slrmRadius: number;
  msrmMinimumFeature: number;
  msrmMaximumFeature: number;
  msrmScaling: number;
  dominanceMinimumRadius: number;
  dominanceMaximumRadius: number;
  dominanceIncrement: number;
  dominanceAngle: '10' | '15' | '20' | '30';
  dominanceObserverHeight: number;
  // VAT blend
  vatPreset: 'archaeological' | 'flat' | 'hillshade-only';
  vatSlopeOpacity: number;
  vatOpennessOpacity: number;
  vatSkyViewOpacity: number;
  vatSlopeBlend: VatBlendChoice;
  vatOpennessBlend: VatBlendChoice;
  vatSkyViewBlend: VatBlendChoice;
  // Display
  ramp: RampName;
  autoStretch: boolean;
  clipPercent: number;
  rangeScale: number;
  opacity: number;
};

/** Hypsometric tint of the Swiss relief, elevation meters to RGB (0-1, as the contributor takes). */
export const ALPINE_ELEVATION_STOPS: readonly {
  elevation: number;
  color: [number, number, number];
}[] = [
  {elevation: 1500, color: [0.42, 0.56, 0.34]},
  {elevation: 2100, color: [0.66, 0.7, 0.42]},
  {elevation: 2700, color: [0.78, 0.7, 0.54]},
  {elevation: 3300, color: [0.84, 0.8, 0.76]},
  {elevation: 4000, color: [0.95, 0.95, 0.97]}
];

/** Products drawn as scalar rasters, with their fixed (non-auto) color range and unit. */
export const SCALAR_PRODUCTS: Partial<
  Record<ReliefProduct, {low: number; high: number; unit: string; label: string}>
> = {
  texture: {low: -0.6, high: 0.6, unit: '', label: 'Texture shade'},
  'sky-view': {low: 0.35, high: 1, unit: '', label: 'Sky-view factor'},
  anisotropic: {low: 0.35, high: 1, unit: '', label: 'Anisotropic sky-view factor'},
  'positive-openness': {low: 60, high: 98, unit: '°', label: 'Positive openness'},
  'negative-openness': {low: 60, high: 98, unit: '°', label: 'Negative openness'},
  slrm: {low: -15, high: 15, unit: 'm', label: 'Simple local relief'},
  msrm: {low: -20, high: 20, unit: 'm', label: 'Multi-scale relief'},
  'local-dominance': {low: 0.3, high: 1.8, unit: '°', label: 'Local dominance'}
};
