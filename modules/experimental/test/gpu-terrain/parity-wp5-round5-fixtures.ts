// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';
import {decodeOracleFixture} from '../oracles/oracle-fixture';
import reliefFirst from '../oracles/fixtures/parity-wp5-round5-slrm-alps-10-0.json';
import reliefSecond from '../oracles/fixtures/parity-wp5-round5-slrm-alps-10-1.json';
import reliefThird from '../oracles/fixtures/parity-wp5-round5-slrm-alps-20-0.json';
import reliefFourth from '../oracles/fixtures/parity-wp5-round5-slrm-alps-20-1.json';
import reliefSynthetic from '../oracles/fixtures/parity-wp5-round5-slrm-synthetic.json';
import blendFirst from '../oracles/fixtures/parity-wp5-round5-blend-alps-0.json';
import blendSecond from '../oracles/fixtures/parity-wp5-round5-blend-alps-1.json';
import blendSynthetic from '../oracles/fixtures/parity-wp5-round5-blend-synthetic.json';
import peaksFirst from '../oracles/fixtures/parity-wp5-round5-peaks-alps-32.json';
import peaksSecond from '../oracles/fixtures/parity-wp5-round5-peaks-alps-128.json';
import peaksSynthetic from '../oracles/fixtures/parity-wp5-round5-peaks-synthetic.json';
import type {GPUReliefBlendLayerSettings} from '../../src/gpu-terrain/relief-visualization/gpu-relief-blend';

type Tolerance = {profile: string; absolute: number; relative: number};
type Source = {
  kind: string;
  path?: string;
  sha256?: string;
  manifestSha256?: string;
  fullShape?: number[];
  fullFilterVerified?: boolean;
  fullTransects?: boolean;
  coreOrigin?: number[];
  coreSize?: number;
  halo?: number;
};
type Fixture<Case> = {
  versions: Record<string, string>;
  generatorSha256: string;
  conventions: string;
  cases: Case[];
};
type ReliefCase = {
  name: string;
  source: Source;
  width: number;
  height: number;
  elevation: number[];
  validity: number[];
  maximumRadius: number;
  radius: number;
  defaultRadius: boolean;
  verticalExaggeration: number;
  expected: number[];
  expectedValidity: number[];
  tolerance: Tolerance;
  rvtCrossCheck: {performed: boolean; maxAbs: number; derivedAbsoluteBound: number};
};
type BlendCase = {
  name: string;
  source: Source;
  width: number;
  height: number;
  layers: number[][];
  settings: GPUReliefBlendLayerSettings[];
  expected: number[];
  expectedColor: number[];
  expectedValidity: number[];
  tolerance: Tolerance;
};
type PeakCase = {
  name: string;
  source: Source;
  values: number[];
  validity: number[];
  offsets: number[];
  window: number;
  suppressionRadius: number;
  minimumProminence: number;
  wrap: boolean;
  expected: {prominence: number[]; refinedIndex: number[]; refinedValue: number[]};
  expectedMask: number[];
  expectedIds: number[];
  tolerances: {prominence: Tolerance; refinedIndex: Tolerance; refinedValue: Tolerance};
  candidateCount: number;
  suppressionBoundaryMargin: number;
};

export const RELIEF_FIXTURES = [
  reliefFirst,
  reliefSecond,
  reliefThird,
  reliefFourth,
  reliefSynthetic
].map(fixture => decodeOracleFixture<Fixture<ReliefCase>>(fixture));
export const BLEND_FIXTURES = [blendFirst, blendSecond, blendSynthetic].map(fixture =>
  decodeOracleFixture<Fixture<BlendCase>>(fixture)
);
export const PEAK_FIXTURES = [peaksFirst, peaksSecond, peaksSynthetic].map(fixture =>
  decodeOracleFixture<Fixture<PeakCase>>(fixture)
);
export const RELIEF_CASES = RELIEF_FIXTURES.flatMap(fixture => fixture.cases);
export const BLEND_CASES = BLEND_FIXTURES.flatMap(fixture => fixture.cases);
export const PEAK_CASES = PEAK_FIXTURES.flatMap(fixture => fixture.cases);
export const EXACT_TOLERANCE = {profile: 'E', absolute: 0, relative: 0};
export const RELIEF_REFERENCE =
  'Composite: scipy.ndimage.uniform_filter / Hesse 2010; cross-checked rvt.vis.slrm';
export const BLEND_REFERENCE =
  'rvt.blend_func.normalize_lin + blend_images + render_images (preserved background)';
export const PEAK_REFERENCE =
  'scipy.signal.find_peaks/peak_prominences; composite quadratic polyfit + cKDTree greedy suppression';

export function compareRound5Values(
  className: string,
  caseName: string,
  reference: string,
  metric: string,
  actual: readonly number[],
  expected: readonly number[],
  tolerance: Tolerance
): void {
  let maxAbs = 0;
  let maxRel = 0;
  let mismatches = Math.abs(actual.length - expected.length);
  for (let index = 0; index < expected.length; index++) {
    const value = actual[index];
    const target = expected[index];
    if (!Number.isFinite(value) || !Number.isFinite(target)) {
      if (!(Number.isNaN(value) && Number.isNaN(target)) && value !== target) mismatches++;
      continue;
    }
    const difference = Math.abs(value - target);
    maxAbs = Math.max(maxAbs, difference);
    maxRel = Math.max(maxRel, difference / Math.max(Math.abs(target), Number.EPSILON));
    if (difference > tolerance.absolute + tolerance.relative * Math.abs(target)) mismatches++;
  }
  console.log(
    'PARITY',
    JSON.stringify({
      class: className,
      case: caseName,
      reference,
      n: expected.length,
      metric,
      maxAbs,
      maxRel,
      mismatches,
      tolerance
    })
  );
  expect.soft(mismatches, `${className}/${caseName}/${metric}`).toBe(0);
}
