// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWGSLFloatLiteral} from '../../utils/wgsl-kernel-nodes';
import {ERFC_VALUE_WGSL} from '../../utils/wgsl-common';
import {
  GPU_KERNEL_DENSITY_GAUSSIAN_TRUNCATION,
  type GPUKernelDensityKernel
} from './kernel-density-math';

/**
 * WGSL `getKernelValue(distanceSquared, bandwidth)`: the normalised `K_h(d)` of
 * {@link evaluateGPUKernelDensityKernel}, zero beyond its support.
 * @internal
 */
export function getKernelValueWGSL(kernel: GPUKernelDensityKernel): string {
  if (kernel === 'gaussian') {
    const scale = getWGSLFloatLiteral(1 / (2 * Math.PI));
    const truncation = getWGSLFloatLiteral(GPU_KERNEL_DENSITY_GAUSSIAN_TRUNCATION ** 2);
    return /* wgsl */ `
fn getKernelValue(distanceSquared: f32, bandwidth: f32) -> f32 {
  let bandwidthSquared = bandwidth * bandwidth;
  let t = distanceSquared / bandwidthSquared;
  if (!(t < ${truncation})) { return 0.0; }
  return ${scale} * exp(-0.5 * t) / bandwidthSquared;
}`;
  }
  const power = kernel === 'epanechnikov' ? 1 : 2;
  const scale = getWGSLFloatLiteral((power + 1) / Math.PI);
  return /* wgsl */ `
fn getKernelValue(distanceSquared: f32, bandwidth: f32) -> f32 {
  let bandwidthSquared = bandwidth * bandwidth;
  let t = distanceSquared / bandwidthSquared;
  if (!(t < 1.0)) { return 0.0; }
  let complement = 1.0 - t;
  return ${scale} * ${power === 1 ? 'complement' : 'complement * complement'} / bandwidthSquared;
}`;
}

/**
 * WGSL `getWindowMass(point, windowMin, windowMax, bandwidth)`: the float32 mirror of
 * {@link getGPUKernelDensityWindowMass}. All positions are relative to a common local origin.
 * @internal
 */
export function getWindowMassWGSL(kernel: GPUKernelDensityKernel): string {
  if (kernel === 'gaussian') {
    return /* wgsl */ `
${ERFC_VALUE_WGSL}
fn getUpperTail(z: f32) -> f32 { return 0.5 * erfcValue(z * 0.70710678); }
fn getNormalMass(low: f32, high: f32) -> f32 {
  if (low > 0.0) { return getUpperTail(low) - getUpperTail(high); }
  if (high < 0.0) { return getUpperTail(-high) - getUpperTail(-low); }
  return 1.0 - getUpperTail(-low) - getUpperTail(high);
}
fn getWindowMass(point: vec2f, windowMin: vec2f, windowMax: vec2f, bandwidth: f32) -> f32 {
  let low = (windowMin - point) / bandwidth;
  let high = (windowMax - point) / bandwidth;
  return getNormalMass(low.x, high.x) * getNormalMass(low.y, high.y);
}`;
  }
  const power = kernel === 'epanechnikov' ? 1 : 2;
  const scale = getWGSLFloatLiteral((power + 1) / Math.PI);
  // Closed-form chord integral of (1 - t^2)^(n / 2) over [0, x], n = 3 (power 1) or 5 (power 2).
  const bulk =
    power === 1
      ? 'height * (bulkEnd - bulkEnd3 / 3.0) - height3 * bulkEnd / 3.0'
      : 'height * (bulkEnd - 2.0 * bulkEnd3 / 3.0 + bulkEnd5 / 5.0) - (2.0 * height3 / 3.0) * (bulkEnd - bulkEnd3 / 3.0) + height5 * bulkEnd / 5.0';
  const chordScale = power === 1 ? getWGSLFloatLiteral(2 / 3) : getWGSLFloatLiteral(8 / 15);
  return /* wgsl */ `
fn getChordIntegral(x: f32) -> f32 {
  let clamped = clamp(x, 0.0, 1.0);
  let s = sqrt(max(1.0 - clamped * clamped, 0.0));
  let first = 0.5 * (clamped * s + asin(clamped));
  let third = clamped * s * s * s / 4.0 + 0.75 * first;
  ${power === 1 ? 'return third;' : 'return clamped * s * s * s * s * s / 6.0 + (5.0 / 6.0) * third;'}
}
fn getQuadrantMass(a: f32, b: f32) -> f32 {
  let width = min(a, 1.0);
  let height = min(b, 1.0);
  if (!(width > 0.0) || !(height > 0.0)) { return 0.0; }
  let kink = sqrt(max(1.0 - height * height, 0.0));
  let bulkEnd = min(width, kink);
  let bulkEnd3 = bulkEnd * bulkEnd * bulkEnd;
  let bulkEnd5 = bulkEnd3 * bulkEnd * bulkEnd;
  let height3 = height * height * height;
  let height5 = height3 * height * height;
  var total = ${bulk};
  if (width > kink) {
    total += ${chordScale} * (getChordIntegral(width) - getChordIntegral(kink));
  }
  return ${scale} * total;
}
fn getSignedQuadrantMass(a: f32, b: f32) -> f32 {
  return sign(a) * sign(b) * getQuadrantMass(abs(a), abs(b));
}
fn getWindowMass(point: vec2f, windowMin: vec2f, windowMax: vec2f, bandwidth: f32) -> f32 {
  let low = (windowMin - point) / bandwidth;
  let high = (windowMax - point) / bandwidth;
  return getSignedQuadrantMass(high.x, high.y) - getSignedQuadrantMass(low.x, high.y) -
    getSignedQuadrantMass(high.x, low.y) + getSignedQuadrantMass(low.x, low.y);
}`;
}
