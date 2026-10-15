// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import alps from '../oracles/fixtures/parity-wp5-deep-alps-dem-visibility.json';
import canyon from '../oracles/fixtures/parity-wp5-deep-grand-canyon-dem-visibility.json';
import ridge from '../oracles/fixtures/parity-wp5-deep-ridge-visibility.json';
import saddle from '../oracles/fixtures/parity-wp5-deep-saddle-visibility.json';
import type {RasterInput, VisibilityCase} from './parity-wp5-round8-fixtures';
import {decodeOracleFixture} from '../oracles/oracle-fixture';

type MatchedVisibilityCase = Omit<VisibilityCase, 'referenceDisagreements'> & {
  coarse: number[][];
  point: number[][];
  coarseCounts: number[];
  margins: Record<'coarse' | 'point', (number | null)[][]>;
};

type MatchedVisibilityFixture = {
  input: RasterInput;
  cases: MatchedVisibilityCase[];
};

/** Native 2D outputs are diagnostics; the radial GDAL inputs preserve every sampled slope. */
export const MATCHED_VISIBILITY_FIXTURES = [alps, canyon, ridge, saddle].map(value =>
  decodeOracleFixture<MatchedVisibilityFixture>(value)
);
