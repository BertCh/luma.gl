// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';
import {decodeOracleFixture} from '../../oracles/oracle-fixture';
import evansWest from '../../oracles/fixtures/parity-wp5-terrain-curvature-alps-evans-young-0.json';
import evansEast from '../../oracles/fixtures/parity-wp5-terrain-curvature-alps-evans-young-1.json';
import zevenbergenWest from '../../oracles/fixtures/parity-wp5-terrain-curvature-alps-zevenbergen-thorne-0.json';
import zevenbergenEast from '../../oracles/fixtures/parity-wp5-terrain-curvature-alps-zevenbergen-thorne-1.json';
import florinskyWest from '../../oracles/fixtures/parity-wp5-terrain-curvature-alps-florinsky-0.json';
import florinskyEast from '../../oracles/fixtures/parity-wp5-terrain-curvature-alps-florinsky-1.json';
import evansSynthetic from '../../oracles/fixtures/parity-wp5-terrain-curvature-synthetic-evans-young.json';
import zevenbergenSynthetic from '../../oracles/fixtures/parity-wp5-terrain-curvature-synthetic-zevenbergen-thorne.json';
import florinskySynthetic from '../../oracles/fixtures/parity-wp5-terrain-curvature-synthetic-florinsky.json';
import type {GPUTerrainCurvatureMethod} from '../../../src/gpu-terrain/terrain-curvature';

export const CURVATURE_KINDS = ['profile', 'plan', 'tangential', 'laplacian'] as const;
export type CurvatureKind = (typeof CURVATURE_KINDS)[number];
export type ParityTolerance = {profile: 'R' | 'E'; absolute: number; relative: number};
export type CurvatureCase = {
  name: string;
  width: number;
  height: number;
  method: GPUTerrainCurvatureMethod;
  cellSize: [number, number];
  borderMode: 'clamp' | 'nodata';
  rowDirection: 'south' | 'north';
  zFactor: number;
  references: Record<CurvatureKind, string>;
  source: {
    kind?: string;
    path?: string;
    sha256?: string;
    coreOrigin?: [number, number];
    coreSize?: [number, number];
    halo?: number;
  };
  elevation: number[];
  inputValidity: number[];
  expectedValidity: number[];
  expected: Record<CurvatureKind, number[]>;
  tolerances: Record<CurvatureKind, ParityTolerance>;
  partials?: Record<'east' | 'north' | 'eastEast' | 'eastNorth' | 'northNorth', number[]>;
};
type CurvatureFixture = {
  versions: Record<string, string>;
  generator: string;
  generatorSha256: string;
  cases: CurvatureCase[];
};

export const CURVATURE_FIXTURES = [
  evansWest,
  evansEast,
  zevenbergenWest,
  zevenbergenEast,
  florinskyWest,
  florinskyEast,
  evansSynthetic,
  zevenbergenSynthetic,
  florinskySynthetic
].map(fixture => decodeOracleFixture<CurvatureFixture>(fixture));

export const CURVATURE_CASES = CURVATURE_FIXTURES.flatMap(fixture => fixture.cases);

/** Reports all cells before asserting, including exact NaN/Infinity masks and lengths. */
export function compareCurvatureValues(
  caseName: string,
  reference: string,
  metric: string,
  actual: readonly number[],
  expected: readonly number[],
  tolerance: ParityTolerance
): void {
  let maxAbs = 0;
  let maxRel = 0;
  let mismatches = Math.abs(actual.length - expected.length);
  for (let index = 0; index < expected.length; index++) {
    const value = actual[index];
    const target = expected[index];
    if (!Number.isFinite(value) || !Number.isFinite(target)) {
      if (!(Number.isNaN(value) && Number.isNaN(target)) && value !== target) {
        mismatches++;
      }
      continue;
    }
    const difference = Math.abs(value - target);
    maxAbs = Math.max(maxAbs, difference);
    maxRel = Math.max(maxRel, difference / Math.max(Math.abs(target), Number.EPSILON));
    if (difference > tolerance.absolute + tolerance.relative * Math.abs(target)) {
      mismatches++;
    }
  }
  console.log(
    'PARITY',
    JSON.stringify({
      class: 'GPUTerrainCurvature',
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
  expect(mismatches, `${caseName}: ${metric}`).toBe(0);
}
