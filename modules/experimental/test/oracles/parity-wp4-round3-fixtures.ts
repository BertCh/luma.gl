// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUStreamlinesSettings} from '../../src/gpu-raster/flow-texture';
import type {StreamlinesCPUConfig} from '../../src/gpu-raster/flow-texture/streamlines-cpu';
import streamsJSON from './fixtures/parity-wp4-streamlines.json';
import isotropicJSON from './fixtures/parity-wp4-distance-isotropic.json';
import anisotropicJSON from './fixtures/parity-wp4-distance-anisotropic.json';
import distanceSyntheticJSON from './fixtures/parity-wp4-distance-synthetic.json';
import polygonJSON from './fixtures/parity-wp4-polygons.json';
import zonalJSON from './fixtures/parity-wp4-zonal.json';
import {decodeOracleFixture} from './oracle-fixture';

export type StreamlineCase = {
  name: string;
  reference: string;
  fieldWidth: number;
  fieldHeight: number;
  field: number[];
  config: StreamlinesCPUConfig;
  settings: GPUStreamlinesSettings;
  candidates: number[];
  spans: number[];
  kept: number[];
  ids: number[];
  offsets: number[];
  points: number[];
  keys: number[];
  minimumCellMargin: number;
};

type EncodedStreamlineCase = Omit<StreamlineCase, 'field' | 'candidates' | 'points'> & {
  field: string;
  candidates: string;
  points: string;
};
export const streamlineCases = decodeOracleFixture<{cases: EncodedStreamlineCase[]}>(
  streamsJSON
).cases;

export async function decodeStreamlineCase(scene: EncodedStreamlineCase): Promise<StreamlineCase> {
  const [field, candidates, points] = await Promise.all([
    decodePackedNumbers(scene.field),
    decodePackedNumbers(scene.candidates),
    decodePackedNumbers(scene.points)
  ]);
  return {...scene, field, candidates, points};
}
export const distanceCases = [isotropicJSON, anisotropicJSON, distanceSyntheticJSON].flatMap(
  fixture => fixture.cases
);
export const polygonCases = polygonJSON.cases;
type ZonalCase = Omit<(typeof zonalJSON.cases)[number], 'expected'> & {
  expected: Record<keyof (typeof zonalJSON.cases)[number]['expected'], number[]>;
};
export const zonalCases = decodeOracleFixture<{cases: ZonalCase[]}>(zonalJSON).cases;
export const roundThreeFixtures = [
  streamsJSON,
  isotropicJSON,
  anisotropicJSON,
  distanceSyntheticJSON,
  polygonJSON,
  zonalJSON
];

/** Standard gzip is lossless transport; decode every entry on both Node and browser. */
export async function decodePackedNumbers(encoded: string): Promise<number[]> {
  const compressed = Uint8Array.from(atob(encoded), character => character.charCodeAt(0));
  const stream = new Blob([compressed]).stream().pipeThrough(new DecompressionStream('gzip'));
  return decodeOracleFixture<number[]>(JSON.parse(await new Response(stream).text()));
}

export async function decodePackedRuns(encoded: string, length: number): Promise<Uint32Array> {
  const runs = await decodePackedNumbers(encoded);
  const values = new Uint32Array(length);
  let offset = 0;
  if (runs.length % 2 !== 0) throw new Error('Incomplete count/value pair');
  for (let index = 0; index < runs.length; index += 2) {
    const count = runs[index];
    const value = runs[index + 1];
    if (
      !Number.isInteger(count) ||
      count <= 0 ||
      offset + count > length ||
      !Number.isInteger(value) ||
      value < 0 ||
      value > 0xffffffff
    ) {
      throw new Error('Invalid lossless uint32 run');
    }
    values.fill(value, offset, offset + count);
    offset += count;
  }
  if (offset !== length) throw new Error('Incomplete raster');
  return values;
}
