// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPURasterArithmeticSettings} from '../../src/gpu-raster/raster-algebra';
import type {GPURasterExtremaPyramidFootprint} from '../../src/gpu-raster/raster-pyramid';
import {decodeOracleFixture} from './oracle-fixture';
import isobands from './fixtures/parity-wp4-isobands.json';
import arithmetic0 from './fixtures/parity-wp4-arithmetic-real-0.json';
import arithmetic1 from './fixtures/parity-wp4-arithmetic-real-1.json';
import arithmetic2 from './fixtures/parity-wp4-arithmetic-real-2.json';
import arithmetic3 from './fixtures/parity-wp4-arithmetic-real-3.json';
import arithmeticSynthetic from './fixtures/parity-wp4-arithmetic-synthetic.json';
import pyramidCell from './fixtures/parity-wp4-pyramid-real-cell.json';
import pyramidBilinear from './fixtures/parity-wp4-pyramid-real-bilinear.json';
import pyramidSynthetic from './fixtures/parity-wp4-pyramid-synthetic.json';

export type ArithmeticCase = {
  name: string;
  reference: string;
  first: string;
  second: string | null;
  settings: GPURasterArithmeticSettings;
  expected: string;
  length: number;
};
export type PyramidCase = {
  name: string;
  reference: string;
  width: number;
  height: number;
  values: string;
  validity: string;
  firstBlockSize: number;
  footprint: GPURasterExtremaPyramidFootprint;
  levels: {blockSize: number; width: number; height: number; offset: number}[];
  maximum: string;
  minimum: string;
  length: number;
  scale: number;
  offset: number;
};
export const arithmeticFixtures = [
  arithmetic0,
  arithmetic1,
  arithmetic2,
  arithmetic3,
  arithmeticSynthetic
];
export const pyramidFixtures = [pyramidCell, pyramidBilinear, pyramidSynthetic];
export const arithmeticCases = arithmeticFixtures.flatMap(
  fixture => decodeOracleFixture<{cases: ArithmeticCase[]}>(fixture).cases
);
export const pyramidCases = pyramidFixtures.flatMap(
  fixture => decodeOracleFixture<{cases: PyramidCase[]}>(fixture).cases
);
export const isobandCases = isobands.cases;
export const roundFourFixtures = [isobands, ...arithmeticFixtures, ...pyramidFixtures];
