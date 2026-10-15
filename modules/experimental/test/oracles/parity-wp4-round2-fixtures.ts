// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPULineIntegralConvolutionSettings} from '../../src/gpu-raster/flow-texture';
import licJSON from './fixtures/parity-wp4-lic.json';
import dixieFourJSON from './fixtures/parity-wp4-components-dixie-4.json';
import dixieEightJSON from './fixtures/parity-wp4-components-dixie-8.json';
import syntheticJSON from './fixtures/parity-wp4-components-synthetic.json';
import {decodeOracleFixture} from './oracle-fixture';

export type LICCase = {
  name: string;
  reference: string;
  width: number;
  height: number;
  fieldWidth: number;
  fieldHeight: number;
  field: number[];
  stepCount: number;
  settings: GPULineIntegralConvolutionSettings;
  values: number[];
  speeds: number[];
  noise: number[];
};

export const licCases = decodeOracleFixture<{cases: LICCase[]}>(licJSON).cases;
export const componentFixtures = [dixieFourJSON, dixieEightJSON, syntheticJSON];
export const componentCases = componentFixtures.flatMap(fixture => fixture.cases);
export type ComponentCase = (typeof componentCases)[number];

/** Lossless transport only: return every original cell, never a summary or sampled check. */
export function decodeComponentRuns(runs: readonly number[], length: number): Uint32Array {
  if (runs.length % 2 !== 0) throw new Error('Incomplete count/value pair');
  const result = new Uint32Array(length);
  let offset = 0;
  for (let index = 0; index < runs.length; index += 2) {
    const count = runs[index];
    const value = runs[index + 1];
    if (!Number.isInteger(count) || count <= 0 || offset + count > length) {
      throw new Error('Invalid lossless run length');
    }
    if (!Number.isInteger(value) || value < 0 || value > 0xffffffff) {
      throw new Error('Invalid uint32 run value');
    }
    result.fill(value, offset, offset + count);
    offset += count;
  }
  if (offset !== length) throw new Error('Incomplete raster');
  return result;
}
