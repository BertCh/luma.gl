// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {decodeOracleFixture} from './oracle-fixture';
import type {GPUFocalStatisticsSettings} from '../../src/gpu-spatial-analysis/spatial-interpolation';

export type PackedFixture = {
  generator: string;
  versions: Record<string, string>;
  encoding: string;
  sha256: string;
  data: string;
};

/** Lossless fixture transport only: no reference arithmetic or GPU output modification. */
export async function decodeRound4Fixture<Value>(fixture: PackedFixture): Promise<Value> {
  const bytes = Uint8Array.from(atob(fixture.data), character => character.charCodeAt(0));
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('deflate'));
  const text = await new Response(stream).text();
  return decodeOracleFixture<Value>(JSON.parse(text));
}

export type Round4Case = {
  name: string;
  reference: string;
  n: number;
  inputFloat32Sha256: string;
  provenance: Record<string, unknown>;
};

export const ZONAL_FLOAT_OUTPUTS = [
  'sums',
  'means',
  'minima',
  'maxima',
  'variances',
  'standardDeviations',
  'featureAreas',
  'densities',
  'extent'
] as const;
export type ZonalCase = Round4Case & {
  features: number[][][][][];
  positions: number[];
  values: number[];
  weights: number[];
  boundaryRule: 'open' | 'closed';
  featureCount: number;
  expected: Record<
    | (typeof ZONAL_FLOAT_OUTPUTS)[number]
    | 'weightSums'
    | 'weightedDensities'
    | 'counts'
    | 'valueCounts'
    | 'pointFeatureRows'
    | 'overflow'
    | 'uncertainCount',
    number[]
  >;
};

export const FOCAL_FLOAT_OUTPUTS = [
  'mean',
  'sum',
  'min',
  'max',
  'range',
  'standardDeviation'
] as const;
export type FocalCase = Round4Case & {
  width: number;
  height: number;
  maximumRadius: number;
  noDataValue: number;
  settings: GPUFocalStatisticsSettings;
  values: number[];
  validity: number[];
  rangeScale: number;
  expected: Record<(typeof FOCAL_FLOAT_OUTPUTS)[number] | 'counts', number[]>;
  coreSha256?: Record<string, string>;
};

export type GeographicCase = Round4Case & {
  positions: number[];
  weights: number[];
  groups: number[];
  mask: number[];
  groupCount: number;
  expected: {
    counts: number[];
    weightSums: number[];
    meanCenters: number[];
    standardDistances: number[];
    ellipses: number[];
    angleDefined: boolean[];
  };
};
