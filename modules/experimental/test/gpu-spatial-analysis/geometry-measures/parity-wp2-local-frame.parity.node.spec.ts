// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  createGPULocalFrame,
  type GPULocalFrameKind
} from '../../../src/analysis-core/contracts/local-frame';
import {decodeOracleFixture} from '../../oracles/oracle-fixture';
import fixtureJson from '../../oracles/fixtures/parity-wp2-local-frame.json';

type FrameCase = {
  name: string;
  kind: GPULocalFrameKind;
  origin: [number, number];
  positions: number[];
  expected: number[];
  validity: number[];
  inverseOfRounded: number[];
};
const fixture = decodeOracleFixture<{cases: FrameCase[]}>(fixtureJson);
for (const entry of fixture.cases) {
  it(`local-frame binary64 patch source and inverse agree with PROJ: ${entry.name}`, () => {
    const frame = createGPULocalFrame({origin: entry.origin, kind: entry.kind});
    for (let row = 0; row < entry.validity.length; row++) {
      if (!entry.validity[row]) continue;
      const expected = entry.expected.slice(2 * row, 2 * row + 2);
      const projected = frame.project(entry.positions[2 * row], entry.positions[2 * row + 1]);
      projected.forEach((value, axis) =>
        expect(Math.abs(value - expected[axis]), `row ${row} axis ${axis}`).toBeLessThan(1e-5)
      );
      const inverse = frame.unproject(Math.fround(expected[0]), Math.fround(expected[1]));
      inverse.forEach((value, axis) => {
        const wanted = entry.inverseOfRounded[2 * row + axis];
        const difference = axis === 0 ? ((value - wanted + 540) % 360) - 180 : value - wanted;
        expect(Math.abs(difference), `inverse row ${row} axis ${axis}`).toBeLessThan(1e-7);
      });
    }
  });
}
