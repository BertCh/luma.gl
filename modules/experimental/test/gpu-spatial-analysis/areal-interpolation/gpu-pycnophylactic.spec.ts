// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUPycnophylactic} from '../../../src/gpu-spatial-analysis/areal-interpolation/index';
import {computePycnophylacticOracle, createBlockZones} from './areal-interpolation-oracle';
import {expectClose, GraphRig} from './areal-interpolation-harness';

const WIDTH = 20;
const HEIGHT = 15;

async function runPycnophylactic(
  device: NonNullable<Awaited<ReturnType<typeof getWebGPUTestDevice>>>,
  iterations: number,
  kernel: 'rook' | 'box'
) {
  const {zones, zoneCount} = createBlockZones(WIDTH, HEIGHT, 5, 5, {holeSeed: 3, holeRate: 0.06});
  // A skewed field of totals: a peak next to empty zones forces the non-negativity clamp.
  const totals = Float32Array.from({length: zoneCount}, (_, index) =>
    index === 5 ? 900 : index % 3 === 0 ? 0 : 40 + index * 10
  );
  const rig = new GraphRig(device);
  const output = rig.output('float32', WIDTH * HEIGHT);
  rig.run(
    new GPUPycnophylactic({
      width: WIDTH,
      height: HEIGHT,
      zones: rig.input(zones, 'uint32'),
      zoneCount,
      totals: rig.input(totals, 'float32'),
      iterations,
      kernel,
      output: output.view
    })
  );
  const actual = await output.readFloat32();
  rig.destroy();
  const expected = computePycnophylacticOracle({
    width: WIDTH,
    height: HEIGHT,
    zones,
    zoneCount,
    totals,
    iterations,
    kernel
  });
  return {actual, expected, zones, zoneCount, totals};
}

for (const kernel of ['rook', 'box'] as const) {
  for (const iterations of [0, 1, 6, 7]) {
    it(`GPUPycnophylactic matches the oracle (${kernel}, ${iterations} iterations)`, async () => {
      const device = await getWebGPUTestDevice();
      if (!device) return;
      const {actual, expected, zones, zoneCount, totals} = await runPycnophylactic(
        device,
        iterations,
        kernel
      );
      expectClose(actual, expected, 'pycnophylactic surface', 2e-3, 1e-3);
      expect(actual.some(value => value > 0)).toBe(true);
      // Mass preservation per zone and non-negativity.
      const sums = new Array<number>(zoneCount).fill(0);
      for (let cell = 0; cell < actual.length; cell++) {
        expect(actual[cell]).toBeGreaterThanOrEqual(0);
        if (zones[cell] < zoneCount) sums[zones[cell]] += actual[cell];
        else expect(actual[cell]).toBe(0);
      }
      const present = new Set<number>(Array.from(zones).filter(zone => zone < zoneCount));
      for (const zone of present) {
        expect(sums[zone]).toBeCloseTo(totals[zone], 2);
      }
    });
  }
}

it('GPUPycnophylactic (box) smooths the surface across zone borders', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const flat = await runPycnophylactic(device, 0, 'rook');
  const smooth = await runPycnophylactic(device, 6, 'box');
  const roughness = (values: number[]) => {
    let total = 0;
    for (let row = 0; row < HEIGHT; row++) {
      for (let column = 0; column + 1 < WIDTH; column++) {
        total += (values[row * WIDTH + column] - values[row * WIDTH + column + 1]) ** 2;
      }
    }
    return total;
  };
  expect(roughness(smooth.actual)).toBeLessThan(roughness(flat.actual));
});
