// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {decodeOracleFixture} from '../oracles/oracle-fixture';
import fixture0 from '../oracles/fixtures/parity-wp5-round8-alps-dem-texture.json';
import fixture1 from '../oracles/fixtures/parity-wp5-round8-alps-dem-tpi-broad-0.json';
import fixture2 from '../oracles/fixtures/parity-wp5-round8-alps-dem-tpi-broad-1.json';
import fixture3 from '../oracles/fixtures/parity-wp5-round8-alps-dem-tpi-broad-2.json';
import fixture4 from '../oracles/fixtures/parity-wp5-round8-alps-dem-tpi-broad-3.json';
import fixture5 from '../oracles/fixtures/parity-wp5-round8-alps-dem-tpi-broad-4.json';
import fixture6 from '../oracles/fixtures/parity-wp5-round8-alps-dem-tpi-broad-5.json';
import fixture7 from '../oracles/fixtures/parity-wp5-round8-alps-dem-tpi-fine-0.json';
import fixture8 from '../oracles/fixtures/parity-wp5-round8-alps-dem-tpi-fine-1.json';
import fixture9 from '../oracles/fixtures/parity-wp5-round8-alps-dem-tpi-fine-2.json';
import fixture10 from '../oracles/fixtures/parity-wp5-round8-alps-dem-tpi-fine-3.json';
import fixture11 from '../oracles/fixtures/parity-wp5-round8-alps-dem-tpi-fine-4.json';
import fixture12 from '../oracles/fixtures/parity-wp5-round8-alps-dem-tpi-fine-5.json';
import fixture13 from '../oracles/fixtures/parity-wp5-round8-alps-dem-tpi-landscape-0.json';
import fixture14 from '../oracles/fixtures/parity-wp5-round8-alps-dem-tpi-landscape-1.json';
import fixture15 from '../oracles/fixtures/parity-wp5-round8-alps-dem-tpi-landscape-2.json';
import fixture16 from '../oracles/fixtures/parity-wp5-round8-alps-dem-tpi-landscape-3.json';
import fixture17 from '../oracles/fixtures/parity-wp5-round8-alps-dem-tpi-landscape-4.json';
import fixture18 from '../oracles/fixtures/parity-wp5-round8-alps-dem-tpi-landscape-5.json';
import fixture19 from '../oracles/fixtures/parity-wp5-round8-alps-dem-vrm.json';
import fixture20 from '../oracles/fixtures/parity-wp5-round8-alps-dem-weiss.json';
import fixture21 from '../oracles/fixtures/parity-wp5-round8-grand-canyon-dem-texture.json';
import fixture22 from '../oracles/fixtures/parity-wp5-round8-grand-canyon-dem-tpi-broad-0.json';
import fixture23 from '../oracles/fixtures/parity-wp5-round8-grand-canyon-dem-tpi-broad-1.json';
import fixture24 from '../oracles/fixtures/parity-wp5-round8-grand-canyon-dem-tpi-broad-2.json';
import fixture25 from '../oracles/fixtures/parity-wp5-round8-grand-canyon-dem-tpi-broad-3.json';
import fixture26 from '../oracles/fixtures/parity-wp5-round8-grand-canyon-dem-tpi-broad-4.json';
import fixture27 from '../oracles/fixtures/parity-wp5-round8-grand-canyon-dem-tpi-broad-5.json';
import fixture28 from '../oracles/fixtures/parity-wp5-round8-grand-canyon-dem-tpi-fine-0.json';
import fixture29 from '../oracles/fixtures/parity-wp5-round8-grand-canyon-dem-tpi-fine-1.json';
import fixture30 from '../oracles/fixtures/parity-wp5-round8-grand-canyon-dem-tpi-fine-2.json';
import fixture31 from '../oracles/fixtures/parity-wp5-round8-grand-canyon-dem-tpi-fine-3.json';
import fixture32 from '../oracles/fixtures/parity-wp5-round8-grand-canyon-dem-tpi-fine-4.json';
import fixture33 from '../oracles/fixtures/parity-wp5-round8-grand-canyon-dem-tpi-fine-5.json';
import fixture34 from '../oracles/fixtures/parity-wp5-round8-grand-canyon-dem-tpi-landscape-0.json';
import fixture35 from '../oracles/fixtures/parity-wp5-round8-grand-canyon-dem-tpi-landscape-1.json';
import fixture36 from '../oracles/fixtures/parity-wp5-round8-grand-canyon-dem-tpi-landscape-2.json';
import fixture37 from '../oracles/fixtures/parity-wp5-round8-grand-canyon-dem-tpi-landscape-3.json';
import fixture38 from '../oracles/fixtures/parity-wp5-round8-grand-canyon-dem-tpi-landscape-4.json';
import fixture39 from '../oracles/fixtures/parity-wp5-round8-grand-canyon-dem-tpi-landscape-5.json';
import fixture40 from '../oracles/fixtures/parity-wp5-round8-grand-canyon-dem-vrm.json';
import fixture41 from '../oracles/fixtures/parity-wp5-round8-grand-canyon-dem-weiss.json';
import fixture42 from '../oracles/fixtures/parity-wp5-round8-saddle-texture.json';
import fixture43 from '../oracles/fixtures/parity-wp5-round8-saddle-tpi-broad-0.json';
import fixture44 from '../oracles/fixtures/parity-wp5-round8-saddle-tpi-broad-1.json';
import fixture45 from '../oracles/fixtures/parity-wp5-round8-saddle-tpi-broad-2.json';
import fixture46 from '../oracles/fixtures/parity-wp5-round8-saddle-tpi-broad-3.json';
import fixture47 from '../oracles/fixtures/parity-wp5-round8-saddle-tpi-broad-4.json';
import fixture48 from '../oracles/fixtures/parity-wp5-round8-saddle-tpi-broad-5.json';
import fixture49 from '../oracles/fixtures/parity-wp5-round8-saddle-tpi-fine-0.json';
import fixture50 from '../oracles/fixtures/parity-wp5-round8-saddle-tpi-fine-1.json';
import fixture51 from '../oracles/fixtures/parity-wp5-round8-saddle-tpi-fine-2.json';
import fixture52 from '../oracles/fixtures/parity-wp5-round8-saddle-tpi-fine-3.json';
import fixture53 from '../oracles/fixtures/parity-wp5-round8-saddle-tpi-fine-4.json';
import fixture54 from '../oracles/fixtures/parity-wp5-round8-saddle-tpi-fine-5.json';
import fixture55 from '../oracles/fixtures/parity-wp5-round8-saddle-tpi-landscape-0.json';
import fixture56 from '../oracles/fixtures/parity-wp5-round8-saddle-tpi-landscape-1.json';
import fixture57 from '../oracles/fixtures/parity-wp5-round8-saddle-tpi-landscape-2.json';
import fixture58 from '../oracles/fixtures/parity-wp5-round8-saddle-tpi-landscape-3.json';
import fixture59 from '../oracles/fixtures/parity-wp5-round8-saddle-tpi-landscape-4.json';
import fixture60 from '../oracles/fixtures/parity-wp5-round8-saddle-tpi-landscape-5.json';
import fixture61 from '../oracles/fixtures/parity-wp5-round8-saddle-vrm.json';
import fixture62 from '../oracles/fixtures/parity-wp5-round8-saddle-weiss.json';
export type CompositeCase = {
  name: string;
  quantum?: number;
  scales?: {radius: number; innerRadius?: number}[];
  cellSizeMode?: 'uniform' | 'web-mercator';
  borderMode?: 'clamp' | 'nodata';
  radius?: number;
  standardization?: 'global' | 'local';
  levelCount?: number;
  baseSigma?: number;
  detail?: number;
  gain?: number;
  smallScale?: {radius: number};
  largeScale?: {radius: number};
  expected: Record<string, number[]>;
};
export type CompositeFixture = {
  input: {
    name: string;
    width: number;
    height: number;
    cellSize: number;
    north: number;
    south: number;
    elevation: number[];
    validity: number[];
  };
  cases: CompositeCase[];
  versions: Record<string, string>;
  generatorSha256: string;
  conventions: string;
};
export const TPI_FIXTURES = [
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
  fixture11,
  fixture12,
  fixture13,
  fixture14,
  fixture15,
  fixture16,
  fixture17,
  fixture18,
  fixture22,
  fixture23,
  fixture24,
  fixture25,
  fixture26,
  fixture27,
  fixture28,
  fixture29,
  fixture30,
  fixture31,
  fixture32,
  fixture33,
  fixture34,
  fixture35,
  fixture36,
  fixture37,
  fixture38,
  fixture39,
  fixture43,
  fixture44,
  fixture45,
  fixture46,
  fixture47,
  fixture48,
  fixture49,
  fixture50,
  fixture51,
  fixture52,
  fixture53,
  fixture54,
  fixture55,
  fixture56,
  fixture57,
  fixture58,
  fixture59,
  fixture60
].map(value => decodeOracleFixture<CompositeFixture>(value));
export const VRM_FIXTURES = [fixture19, fixture40, fixture61].map(value =>
  decodeOracleFixture<CompositeFixture>(value)
);
export const WEISS_FIXTURES = [fixture20, fixture41, fixture62].map(value =>
  decodeOracleFixture<CompositeFixture>(value)
);
export const TEXTURE_FIXTURES = [fixture0, fixture21, fixture42].map(value =>
  decodeOracleFixture<CompositeFixture>(value)
);
