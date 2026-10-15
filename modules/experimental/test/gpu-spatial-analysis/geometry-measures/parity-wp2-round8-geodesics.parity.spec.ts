// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import type {GPUSpatialContext} from '../../../src/analysis-core/contracts/spatial-context';
import {GPUGeodesicPairs} from '../../../src/gpu-spatial-analysis/geometry-measures/gpu-geodesic-pairs';
import {GPUGeodesicDestination} from '../../../src/gpu-spatial-analysis/geometry-measures/gpu-geodesic-destination';
import {createParityGraph} from '../../oracles/parity-wp2-geodesic-harness';
import {compareBoundedParity} from '../../oracles/parity-wp2-round4-compare';
import {compareParity} from '../../oracles/parity-wp2-assert';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {decodeOracleFixture} from '../../oracles/oracle-fixture';
import fixtureJson from '../../oracles/fixtures/parity-wp2-round8-geodesics.json';
const fixture = decodeOracleFixture<{
  reference: string;
  cases: (Omit<(typeof fixtureJson.cases)[number], 'spatialContext'> & {
    spatialContext: GPUSpatialContext;
  })[];
}>(fixtureJson);
for (const entry of fixture.cases) {
  it(`geodesic documented modes ${entry.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = createParityGraph(device);
    const rows = entry.distances.length;
    const origins = rig.input('origins', entry.origins, 'float32x2').view;
    const distances = rig.output('distances', rows, 'float32');
    const initial = rig.output('initial', rows, 'float32');
    const final = rig.output('final', rows, 'float32');
    const midpoints = rig.output('midpoints', rows, 'float32x2');
    const destinations = rig.output('destinations', rows, 'float32x2');
    const arrival = rig.output('arrival', rows, 'float32');
    const inverseConverged = rig.output('inverse-converged', rows, 'uint32');
    const directConverged = rig.output('direct-converged', rows, 'uint32');
    const ellipsoidal = entry.spatialContext.metric === 'ellipsoidal';
    rig.graph.add(
      new GPUGeodesicPairs({
        spatialContext: entry.spatialContext,
        origins,
        targets: rig.input('targets', entry.targets, 'float32x2').view,
        output: {
          distances: distances.view,
          initialBearings: initial.view,
          finalBearings: final.view,
          midpoints: midpoints.view,
          ...(ellipsoidal ? {status: {converged: inverseConverged.view}} : {})
        }
      })
    );
    rig.graph.add(
      new GPUGeodesicDestination({
        spatialContext: entry.spatialContext,
        origins,
        bearings: rig.input('bearings', entry.bearings, 'float32').view,
        distances: rig.input('input-distances', entry.inputDistances, 'float32').view,
        output: {
          destinations: destinations.view,
          finalBearings: arrival.view,
          ...(ellipsoidal ? {status: {converged: directConverged.view}} : {})
        }
      })
    );
    const compiled = rig.graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      const compare = (
        className: string,
        metric: string,
        actual: number[],
        expected: number[],
        absolute: number,
        relative = 0
      ) =>
        compareParity(
          className,
          entry.name,
          fixture.reference,
          metric,
          actual,
          expected,
          absolute,
          relative
        );
      compare(
        'GPUGeodesicPairs',
        'distance',
        await readFloat32(distances.buffer, rows),
        entry.distances,
        entry.distanceAbsolute,
        2e-6
      );
      for (const [metric, buffer, expected] of [
        ['initial bearing', initial.buffer, entry.initialBearings],
        ['arrival bearing', final.buffer, entry.finalBearings]
      ] as const) {
        const actual = await readFloat32(buffer, rows);
        compare(
          'GPUGeodesicPairs',
          metric,
          actual.map((value, index) => ((value - expected[index] + 540) % 360) - 180),
          Array(rows).fill(0),
          1e-4
        );
      }
      compare(
        'GPUGeodesicPairs',
        'midpoints',
        await readFloat32(midpoints.buffer, rows * 2),
        entry.midpoints,
        entry.positionAbsolute
      );
      compare(
        'GPUGeodesicDestination',
        'destinations',
        await readFloat32(destinations.buffer, rows * 2),
        entry.destinations,
        entry.positionAbsolute
      );
      const actualArrival = await readFloat32(arrival.buffer, rows);
      compareBoundedParity(
        'GPUGeodesicDestination',
        entry.name,
        fixture.reference,
        'arrival bearing',
        actualArrival.map(
          (value, index) => ((value - entry.destinationBearings[index] + 540) % 360) - 180
        ),
        Array(rows).fill(0),
        entry.destinationBearingBounds
      );
      if (ellipsoidal) {
        compare(
          'GPUGeodesicPairs',
          'convergence',
          await readUint32(inverseConverged.buffer, rows),
          Array(rows).fill(1),
          0
        );
        compare(
          'GPUGeodesicDestination',
          'convergence',
          await readUint32(directConverged.buffer, rows),
          Array(rows).fill(1),
          0
        );
      }
    } finally {
      compiled.destroy();
      rig.destroy();
      device.destroy?.();
    }
  }, 120000);
}
