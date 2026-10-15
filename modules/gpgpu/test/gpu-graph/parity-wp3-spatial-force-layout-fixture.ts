// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/// <reference types="vite/client" />

import type {LayoutFixture} from './parity-wp3-force-layout-fixture';
import equilibrium from '../../../experimental/test/oracles/fixtures/parity-wp3-spatial-force-layout-equilibrium.json?raw';
import synthetic from '../../../experimental/test/oracles/fixtures/parity-wp3-spatial-force-layout-synthetic.json?raw';
import flights3 from '../../../experimental/test/oracles/fixtures/parity-wp3-spatial-force-layout-openflights-seed-3.json?raw';
import flights17 from '../../../experimental/test/oracles/fixtures/parity-wp3-spatial-force-layout-openflights-seed-17.json?raw';
import flights41 from '../../../experimental/test/oracles/fixtures/parity-wp3-spatial-force-layout-openflights-seed-41.json?raw';

export type SpatialFixture = Omit<LayoutFixture, 'runs'> & {
  pinned: number[];
  gridSize: [number, number];
  bounds: [number, number, number, number];
  runs: (LayoutFixture['runs'][number] & {
    gridReference: {offsets: number[]; centers: number[]; vertexIds: number[]};
  })[];
};
export const spatialFixtures: SpatialFixture[] = [
  equilibrium,
  synthetic,
  flights3,
  flights17,
  flights41
].map(value => JSON.parse(value));

/** GPUGridIndex preserves identity but explicitly leaves within-cell order unspecified. */
export function canonicalizeGridIds(vertexIds: number[], offsets: number[]): number[] {
  return offsets
    .slice(0, -1)
    .flatMap((start, cell) =>
      vertexIds.slice(start, offsets[cell + 1]).sort((first, second) => first - second)
    );
}
