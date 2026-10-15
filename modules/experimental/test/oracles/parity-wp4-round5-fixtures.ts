// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPURasterCellStatisticsSettings} from '../../src/gpu-raster/raster-algebra';
import {decodeOracleFixture} from './oracle-fixture';
import contours from './fixtures/parity-wp4-contours.json';
import real0 from './fixtures/parity-wp4-cell-statistics-real-0.json';
import real1 from './fixtures/parity-wp4-cell-statistics-real-1.json';
import real2 from './fixtures/parity-wp4-cell-statistics-real-2.json';
import real3 from './fixtures/parity-wp4-cell-statistics-real-3.json';
import real4 from './fixtures/parity-wp4-cell-statistics-real-4.json';
import real5 from './fixtures/parity-wp4-cell-statistics-real-5.json';
import real6 from './fixtures/parity-wp4-cell-statistics-real-6.json';
import real7 from './fixtures/parity-wp4-cell-statistics-real-7.json';
import synthetic from './fixtures/parity-wp4-cell-statistics-synthetic.json';

export const floatStatistics = [
  'minimum',
  'maximum',
  'range',
  'sum',
  'mean',
  'standardDeviation',
  'majority',
  'minority'
] as const;
export const integerStatistics = ['variety', 'counts'] as const;
export const allStatistics = [...floatStatistics, ...integerStatistics] as const;
export type Statistic = (typeof allStatistics)[number];
export type CellStatisticsCase = {
  name: string;
  reference: string;
  cellCount: number;
  layerCount: number;
  stack: string;
  frames: {settings: GPURasterCellStatisticsSettings; expected: Record<Statistic, string>}[];
};
export const statisticsFixtures = [
  real0,
  real1,
  real2,
  real3,
  real4,
  real5,
  real6,
  real7,
  synthetic
];
export const statisticsCases: CellStatisticsCase[] = statisticsFixtures.flatMap(
  fixture => decodeOracleFixture<{cases: CellStatisticsCase[]}>(fixture).cases
);
export const ringCases = contours.ringCases;
export const lineCases = contours.lineCases;
export type ReferenceRing = (typeof ringCases)[number]['rings'][number];
export const roundFiveFixtures = [contours, ...statisticsFixtures];
