// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {decodeOracleFixture} from '../oracles/oracle-fixture';
import type {
  GPUTerrainCurvatureKind,
  GPUTerrainCurvatureMethod
} from '../../src/gpu-terrain/terrain-curvature';
import type {CompositeFixture} from './parity-wp5-round8-composite-fixtures';
import fixture0 from '../oracles/fixtures/parity-wp5-round8-curvature-alps-dem-evans-young-clamp.json';
import fixture1 from '../oracles/fixtures/parity-wp5-round8-curvature-alps-dem-evans-young-nodata.json';
import fixture2 from '../oracles/fixtures/parity-wp5-round8-curvature-alps-dem-florinsky-clamp.json';
import fixture3 from '../oracles/fixtures/parity-wp5-round8-curvature-alps-dem-florinsky-nodata.json';
import fixture4 from '../oracles/fixtures/parity-wp5-round8-curvature-alps-dem-zevenbergen-thorne-clamp.json';
import fixture5 from '../oracles/fixtures/parity-wp5-round8-curvature-alps-dem-zevenbergen-thorne-nodata.json';
import fixture6 from '../oracles/fixtures/parity-wp5-round8-curvature-grand-canyon-dem-evans-young-clamp.json';
import fixture7 from '../oracles/fixtures/parity-wp5-round8-curvature-grand-canyon-dem-evans-young-nodata.json';
import fixture8 from '../oracles/fixtures/parity-wp5-round8-curvature-grand-canyon-dem-florinsky-clamp.json';
import fixture9 from '../oracles/fixtures/parity-wp5-round8-curvature-grand-canyon-dem-florinsky-nodata.json';
import fixture10 from '../oracles/fixtures/parity-wp5-round8-curvature-grand-canyon-dem-zevenbergen-thorne-clamp.json';
import fixture11 from '../oracles/fixtures/parity-wp5-round8-curvature-grand-canyon-dem-zevenbergen-thorne-nodata.json';
export type CurvatureFixture = Omit<CompositeFixture, 'cases'> & {
  cases: {
    name: string;
    method: GPUTerrainCurvatureMethod;
    borderMode: 'clamp' | 'nodata';
    expected: Record<GPUTerrainCurvatureKind, number[]>;
    validity: number[];
  }[];
};
export const CURVATURE_FIXTURES = [
  fixture0,
  fixture1,
  fixture2,
  fixture3,
  fixture4,
  fixture5,
  fixture6,
  fixture7,
  fixture8,
  fixture9,
  fixture10,
  fixture11
].map(value => decodeOracleFixture<CurvatureFixture>(value));
