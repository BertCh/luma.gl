// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {
  getGPUSegregationLayout,
  GPUSegregation
} from '../../../src/gpu-spatial-analysis/segregation/index';
import {expectClose, GraphRig} from '../areal-interpolation/areal-interpolation-harness';
import {requireTestDevice} from '../../utils/require-test-device';
import {SEGREGATION_REFERENCE, SEGREGATION_TOLERANCES} from './segregation-reference';

// Reference values from PySAL `segregation` 2.6.0 (MultiInfoTheory, MultiDissim, Dissim, Isolation,
// Atkinson with b = 0.5) on a 6x5 grid fixture, from test/oracles/statistics-segregation.py. The
// spatial values are PySAL's spatially implicit indices with libpysal Queen weights (binary, self
// weight 1), which equal `spatialForm: 'smoothed-population'`.
const {
  columns: COLUMNS,
  rows: ROWS,
  groups: GROUPS,
  counts: COUNTS,
  aspatial,
  queen: queenReference
} = SEGREGATION_REFERENCE;
const PYSAL = {aspatial, queen: queenReference};

function createQueenCSR() {
  const offsets = [0];
  const neighbors: number[] = [];
  for (let unit = 0; unit < COLUMNS * ROWS; unit++) {
    const column = unit % COLUMNS;
    const row = Math.floor(unit / COLUMNS);
    for (let other = 0; other < COLUMNS * ROWS; other++) {
      const dx = Math.abs((other % COLUMNS) - column);
      const dy = Math.abs(Math.floor(other / COLUMNS) - row);
      if (other !== unit && dx <= 1 && dy <= 1) neighbors.push(other);
    }
    offsets.push(neighbors.length);
  }
  return {offsets, neighbors};
}

it('GPUSegregation matches PySAL segregation (aspatial and queen-smoothed)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const layout = getGPUSegregationLayout(GROUPS);
  const queen = createQueenCSR();
  const rig = new GraphRig(device);
  const indices = rig.output('float32', 2 * layout.stride);
  rig.run(
    new GPUSegregation({
      groupCount: GROUPS,
      groupCounts: rig.input(Float32Array.from(COUNTS), 'float32'),
      scales: [
        null,
        {
          offsets: rig.input(Uint32Array.from(queen.offsets), 'uint32'),
          neighbors: rig.input(Uint32Array.from(queen.neighbors), 'uint32'),
          weights: rig.input(new Float32Array(queen.neighbors.length).fill(1), 'float32')
        }
      ],
      spatialForm: 'smoothed-population',
      output: {indices: indices.view}
    })
  );
  const result = await indices.readFloat32();
  rig.destroy();
  for (const [scale, reference] of [
    [0, PYSAL.aspatial],
    [1, PYSAL.queen]
  ] as const) {
    const row = result.slice(scale * layout.stride, (scale + 1) * layout.stride);
    const label = scale === 0 ? 'aspatial' : 'queen';
    expect(row[layout.entropy], `${label} entropy`).toBeGreaterThan(0.01);
    const check = (
      actual: ArrayLike<number>,
      expected: number | number[],
      name: keyof typeof SEGREGATION_TOLERANCES
    ) => {
      const {relative, absolute} = SEGREGATION_TOLERANCES[name];
      expectClose(actual, [expected].flat(), `${label} ${name}`, relative, absolute);
    };
    check([row[layout.entropy]], reference.entropy, 'entropy');
    check([row[layout.multiGroupDissimilarity]], reference.multiD, 'multiD');
    check(row.slice(layout.dissimilarity, layout.dissimilarity + 3), reference.D, 'D');
    check(row.slice(layout.isolation, layout.isolation + 3), reference.isolation, 'isolation');
    check(row.slice(layout.atkinson, layout.atkinson + 3), reference.atkinson, 'atkinson');
  }
});
