// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUPycnophylactic} from '../../../src/gpu-spatial-analysis/areal-interpolation/index';
import {computePycnophylacticOracle, createBlockZones} from './areal-interpolation-oracle';
import {expectClose, GraphRig} from './areal-interpolation-harness';
import {PYCNOPHYLACTIC_NUMPY} from './change-of-support-numpy-fixtures';

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

// Oracle: <scratchpad>/fix/work/D/change-of-support/pycnophylactic.py (numpy 2.5.4, float64),
// pinned in change-of-support-numpy-fixtures.ts and reproduced by the TS oracle in the node spec.
async function runNumpyFixture(
  device: NonNullable<Awaited<ReturnType<typeof getWebGPUTestDevice>>>,
  options: {
    iterations: number;
    anchors: boolean;
    tolerance?: number;
    geographic?: boolean;
    density?: boolean;
  }
) {
  const {fixture} = PYCNOPHYLACTIC_NUMPY;
  const cellCount = fixture.width * fixture.height;
  const rig = new GraphRig(device);
  const output = rig.output('float32', cellCount);
  const density = options.density ? rig.output('float32', cellCount) : undefined;
  const converged = rig.output('uint32', 1);
  const limit = rig.output('uint32', 1);
  const invalid = rig.output('uint32', 1);
  const geographic = PYCNOPHYLACTIC_NUMPY.geographic;
  rig.run(
    new GPUPycnophylactic({
      width: fixture.width,
      height: fixture.height,
      zones: rig.input(Uint32Array.from(fixture.zones), 'uint32'),
      zoneCount: fixture.zoneCount,
      totals: rig.input(Float32Array.from(fixture.totals), 'float32'),
      iterations: options.iterations,
      output: output.view,
      density: density?.view,
      zoneAnchorCells: options.anchors
        ? rig.input(Uint32Array.from(fixture.anchors), 'uint32')
        : undefined,
      tolerance:
        options.tolerance === undefined
          ? undefined
          : rig.input(Float32Array.of(options.tolerance), 'float32'),
      converged: options.tolerance === undefined ? undefined : converged.view,
      iterationLimitReached: options.tolerance === undefined ? undefined : limit.view,
      invalidCount: invalid.view,
      ...(options.geographic
        ? {
            spatialContext: {
              coordinateSpace: 'longitude-latitude',
              metric: 'great-circle'
            } as const,
            extent: rig.input(
              Float32Array.of(
                10,
                geographic.originLatitude,
                geographic.cellWidth,
                geographic.cellHeight
              ),
              'float32'
            )
          }
        : {})
    })
  );
  const result = {
    mass: await output.readFloat32(),
    density: density ? await density.readFloat32() : undefined,
    converged: (await converged.readUint32())[0],
    limit: (await limit.readUint32())[0],
    invalid: (await invalid.readUint32())[0]
  };
  rig.destroy();
  return result;
}

it('GPUPycnophylactic anchors keep the mass of zones without cells (numpy)', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const without = await runNumpyFixture(device, {iterations: 6, anchors: false});
  expectClose(without.mass, PYCNOPHYLACTIC_NUMPY.planar6.mass, 'unanchored', 2e-3, 1e-3);
  expect(without.invalid).toBe(PYCNOPHYLACTIC_NUMPY.planar6.invalidCount);
  const anchored = await runNumpyFixture(device, {iterations: 6, anchors: true});
  expectClose(anchored.mass, PYCNOPHYLACTIC_NUMPY.planarAnchored6.mass, 'anchored', 2e-3, 1e-3);
  expect(anchored.invalid).toBe(0);
  expect(anchored.mass.reduce((sum, value) => sum + value, 0)).toBeCloseTo(507, 1);
});

it('GPUPycnophylactic stops at the tolerance and reports status (numpy)', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const stopped = await runNumpyFixture(device, {
    iterations: 12,
    anchors: true,
    tolerance: PYCNOPHYLACTIC_NUMPY.tolerance
  });
  expectClose(stopped.mass, PYCNOPHYLACTIC_NUMPY.planarTolerance.mass, 'tolerance', 2e-3, 1e-3);
  expect([stopped.converged, stopped.limit]).toEqual([1, 0]);
  const limited = await runNumpyFixture(device, {iterations: 3, anchors: true, tolerance: 1e-12});
  expectClose(limited.mass, PYCNOPHYLACTIC_NUMPY.planarLimit.mass, 'limit', 2e-3, 1e-3);
  expect([limited.converged, limited.limit]).toEqual([0, 1]);
});

it('GPUPycnophylactic smooths density per spherical cell area on a longitude/latitude raster (numpy)', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const result = await runNumpyFixture(device, {
    iterations: 6,
    anchors: true,
    geographic: true,
    density: true
  });
  expectClose(result.mass, PYCNOPHYLACTIC_NUMPY.geographic6.mass, 'geographic mass', 2e-3, 1e-3);
  expectClose(
    result.density!,
    PYCNOPHYLACTIC_NUMPY.geographic6.density,
    'geographic density',
    2e-3,
    1e-12
  );
  expect(result.mass.reduce((sum, value) => sum + value, 0)).toBeCloseTo(507, 1);
});
