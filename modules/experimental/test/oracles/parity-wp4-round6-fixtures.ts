// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPURasterMetadata} from '../../src/gpu-raster';
import type {GPURasterSamplingSettings} from '../../src/gpu-raster/raster-sampling';
import {decodeOracleFixture} from './oracle-fixture';
import joinReal from './fixtures/parity-wp4-join-real.json';
import joinSynthetic from './fixtures/parity-wp4-join-synthetic.json';
import patchesFour from './fixtures/parity-wp4-patches-real-4.json';
import patchesEight from './fixtures/parity-wp4-patches-real-8.json';
import patchesSynthetic from './fixtures/parity-wp4-patches-synthetic.json';
import samplingReal from './fixtures/parity-wp4-sampling-real.json';
import samplingSynthetic from './fixtures/parity-wp4-sampling-synthetic.json';

export const joinMetrics = [
  'pointZones',
  'pointBoundaryMask',
  'counts',
  'sums',
  'boundaryCounts',
  'unassignedBoundaryCount',
  'outsideCount'
] as const;
export const integerPatchMetrics = [
  'pixelCounts',
  'perimeterFaceCounts',
  'minColumns',
  'minRows',
  'maxColumns',
  'maxRows'
] as const;
export const patchMetrics = [...integerPatchMetrics, 'areas', 'perimeters'] as const;
export type JoinMetric = (typeof joinMetrics)[number];
export type PatchMetric = (typeof patchMetrics)[number];
export type JoinCase = {
  name: string;
  reference: string;
  grid: GPURasterMetadata;
  zones: string;
  boundary: string;
  points: string;
  values: string;
  pointCount: number;
  zoneCount: number;
  frames: {grid: GPURasterMetadata; expected: Record<JoinMetric, string>}[];
};
export type PatchCase = {
  name: string;
  reference: string;
  grid: GPURasterMetadata;
  labels: string;
  validity: string;
  capacity: number;
  countRasterBorder: boolean;
  frames: {
    componentCount: number;
    converged: number;
    overflow: number;
    expected: Record<PatchMetric, string>;
  }[];
};
export type SamplingCase = {
  name: string;
  reference: string;
  grid: GPURasterMetadata;
  localOrigin: [number, number];
  values: string;
  validity: string;
  points: string;
  pointCount: number;
  frames: (Required<Pick<GPURasterSamplingSettings, 'method' | 'noDataPolicy'>> & {
    activeCount: number;
    expected: {values: string; outputValidity: string};
  })[];
};
export const joinCases = [joinReal, joinSynthetic].flatMap(
  fixture => decodeOracleFixture<{cases: JoinCase[]}>(fixture).cases
);
export const patchCases = [patchesFour, patchesEight, patchesSynthetic].flatMap(
  fixture => decodeOracleFixture<{cases: PatchCase[]}>(fixture).cases
);
export const samplingCases = [samplingReal, samplingSynthetic].flatMap(
  fixture => decodeOracleFixture<{cases: SamplingCase[]}>(fixture).cases
);
export const roundSixFixtures = [
  joinReal,
  joinSynthetic,
  patchesFour,
  patchesEight,
  patchesSynthetic,
  samplingReal,
  samplingSynthetic
];
