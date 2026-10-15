// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUWeightedOverlaySettings} from '../../src/gpu-raster/raster-algebra';
import sieveFour from './fixtures/parity-wp4-sieve-real-4.json';
import sieveEight from './fixtures/parity-wp4-sieve-real-8.json';
import sieveSynthetic from './fixtures/parity-wp4-sieve-synthetic.json';
import statisticsReal from './fixtures/parity-wp4-statistics-real.json';
import statisticsSynthetic from './fixtures/parity-wp4-statistics-synthetic.json';
import overlay0 from './fixtures/parity-wp4-overlay-real-0.json';
import overlay1 from './fixtures/parity-wp4-overlay-real-1.json';
import overlay2 from './fixtures/parity-wp4-overlay-real-2.json';
import overlay3 from './fixtures/parity-wp4-overlay-real-3.json';
import overlay4 from './fixtures/parity-wp4-overlay-real-4.json';
import overlay5 from './fixtures/parity-wp4-overlay-real-5.json';
import overlay6 from './fixtures/parity-wp4-overlay-real-6.json';
import overlay7 from './fixtures/parity-wp4-overlay-real-7.json';
import overlaySynthetic from './fixtures/parity-wp4-overlay-synthetic.json';
import {decodeOracleFixture} from './oracle-fixture';

export const sieveMetrics = ['labels', 'patchTargets', 'sievedCount'] as const;
export const statisticsMetrics = ['count', 'sum', 'mean', 'extent'] as const;
export const overlayMetrics = ['score', 'outputValidity', 'scoreRange'] as const;
export type SieveCase = {
  name: string;
  reference: string;
  width: number;
  height: number;
  labels: string;
  capacity: number;
  mode: 'remove' | 'merge';
  connectivity: 4 | 8;
  frames: {minimumPixels: number; expected: Record<(typeof sieveMetrics)[number], string>}[];
};
export type StatisticsCase = {
  name: string;
  reference: string;
  width: number;
  height: number;
  values: string;
  validity: string;
  scale: number;
  offset: number;
  frames: {masked: boolean; expected: Record<(typeof statisticsMetrics)[number], number[]>}[];
};
export type OverlayCase = {
  name: string;
  reference: string;
  cellCount: number;
  layerCount: number;
  stack: string;
  remapBreaks: string;
  remapValues: string;
  frames: {
    settings: GPUWeightedOverlaySettings;
    expected: Record<(typeof overlayMetrics)[number], string>;
  }[];
};
const sieveFixtures = [sieveFour, sieveEight, sieveSynthetic];
const statisticsFixtures = [statisticsReal, statisticsSynthetic];
const overlayFixtures = [
  overlay0,
  overlay1,
  overlay2,
  overlay3,
  overlay4,
  overlay5,
  overlay6,
  overlay7,
  overlaySynthetic
];
export const sieveCases = sieveFixtures.flatMap(
  fixture => decodeOracleFixture<{cases: SieveCase[]}>(fixture).cases
);
export const statisticsCases = statisticsFixtures.flatMap(
  fixture => decodeOracleFixture<{cases: StatisticsCase[]}>(fixture).cases
);
export const overlayCases = overlayFixtures.flatMap(
  fixture => decodeOracleFixture<{cases: OverlayCase[]}>(fixture).cases
);
export const roundSevenFixtures = [...sieveFixtures, ...statisticsFixtures, ...overlayFixtures];
