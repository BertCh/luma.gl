// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {it} from 'vitest';
import {simplificationFixtures} from '../../oracles/parity-wp2-line-simplification-fixtures';
import {compareParity} from '../../oracles/parity-wp2-assert';
import {simplifyDouglasPeucker} from './line-simplification-oracle';

// Convention bridge only: GPU specs consume GEOS rows directly, never this local oracle.
for (const fixture of simplificationFixtures)
  for (const entry of fixture.cases) {
    it(`planar f32 convention agrees with GEOS for ${entry.name}`, () => {
      for (const frame of entry.frames) {
        const actual = simplifyDouglasPeucker(
          {
            positions: Float32Array.from(entry.positions),
            lineOffsets: Uint32Array.from(entry.lineOffsets)
          },
          frame.tolerance
        );
        compareParity(
          'GPULineSimplification CPU convention bridge',
          `${entry.name}/${frame.tolerance}`,
          fixture.reference,
          'every kept row',
          actual,
          frame.ids
        );
      }
    });
  }
