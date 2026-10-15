// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';
import {decodeOracleFixture} from '../oracles/oracle-fixture';
import criticalFirst from '../oracles/fixtures/parity-wp5-round6-critical-alps-6-0.json';
import criticalSecond from '../oracles/fixtures/parity-wp5-round6-critical-alps-6-1.json';
import criticalThird from '../oracles/fixtures/parity-wp5-round6-critical-alps-8-0.json';
import criticalFourth from '../oracles/fixtures/parity-wp5-round6-critical-alps-8-1.json';
import criticalSynthetic from '../oracles/fixtures/parity-wp5-round6-critical-synthetic.json';
import flowFirst from '../oracles/fixtures/parity-wp5-round6-flow-alps-0.json';
import flowSecond from '../oracles/fixtures/parity-wp5-round6-flow-alps-1.json';
import flowSynthetic from '../oracles/fixtures/parity-wp5-round6-flow-synthetic.json';
import contoursFirst from '../oracles/fixtures/parity-wp5-round6-contours-alps-0.json';
import contoursSecond from '../oracles/fixtures/parity-wp5-round6-contours-alps-1.json';
import contoursSynthetic from '../oracles/fixtures/parity-wp5-round6-contours-synthetic.json';

export const EXACT_TOLERANCE = {profile: 'E', absolute: 0, relative: 0};
export type Tolerance = {profile: string; absolute: number; relative: number};
type RasterCase = {
  name: string;
  width: number;
  height: number;
  elevation: number[];
  elevationSha256: string;
  validity: number[];
  scale: number;
  offset: number;
  source: {
    kind: string;
    path?: string;
    sha256?: string;
    manifestSha256?: string;
    fullShape?: number[];
    coreSize?: number;
    halo?: number;
    fullReferenceVerified?: boolean;
    unitedReferenceVerified?: boolean;
  };
};
type CriticalCase = RasterCase & {
  connectivity: 6 | 8;
  classes: number[];
  signChanges: number[];
  counts: number[];
};
type FlowCase = RasterCase & {
  cellSize: [number, number];
  winds: [number, number][];
  exaggeration: number;
  expected: number[][];
  tolerances: Tolerance[][];
};
export type ContourLevel = {level: number; segments: number[][]; rings: number[][][]};
export type ContourCase = RasterCase & {
  point: boolean;
  expected: ContourLevel[];
  tolerance: Tolerance;
};
type Fixture<Case> = {
  versions: Record<string, string>;
  conventions: string;
  generatorSha256: string;
  cases: Case[];
};
export const CRITICAL_FIXTURES = [
  criticalFirst,
  criticalSecond,
  criticalThird,
  criticalFourth,
  criticalSynthetic
].map(fixture => decodeOracleFixture<Fixture<CriticalCase>>(fixture));
export const FLOW_FIXTURES = [flowFirst, flowSecond, flowSynthetic].map(fixture =>
  decodeOracleFixture<Fixture<FlowCase>>(fixture)
);
export const CONTOUR_FIXTURES = [contoursFirst, contoursSecond, contoursSynthetic].map(fixture =>
  decodeOracleFixture<Fixture<ContourCase>>(fixture)
);
export const CRITICAL_CASES = CRITICAL_FIXTURES.flatMap(fixture => fixture.cases);
export const FLOW_CASES = FLOW_FIXTURES.flatMap(fixture => fixture.cases);
export const CONTOUR_CASES = CONTOUR_FIXTURES.flatMap(fixture => fixture.cases);
export const CRITICAL_REFERENCE =
  'Composite PL lower-link topology: scipy.ndimage.label + maximum_filter/minimum_filter';
export const FLOW_REFERENCE =
  'Composite numpy.gradient + 3D orthogonal projection (numpy.linalg.norm/einsum)';
export const CONTOUR_REFERENCE =
  'skimage.measure.find_contours; composite bilinear saddle connectivity; shapely.ops.polygonize';

export function compareValues(
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
