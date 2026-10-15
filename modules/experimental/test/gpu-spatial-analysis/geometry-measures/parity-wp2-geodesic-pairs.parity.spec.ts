// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {GPUGeodesicPairs} from '../../../src/gpu-spatial-analysis/geometry-measures/gpu-geodesic-pairs';
import {requireTestDevice} from '../../utils/require-test-device';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {compareParity} from '../../oracles/parity-wp2-assert';
import {createParityGraph, compareGeodesicAngles} from '../../oracles/parity-wp2-geodesic-harness';
import fixture0 from '../../oracles/fixtures/parity-wp2-geodesic-pairs-ecco-00.json';
import fixture1 from '../../oracles/fixtures/parity-wp2-geodesic-pairs-ecco-01.json';
import fixture2 from '../../oracles/fixtures/parity-wp2-geodesic-pairs-ecco-02.json';
import fixture3 from '../../oracles/fixtures/parity-wp2-geodesic-pairs-ecco-03.json';
import fixture4 from '../../oracles/fixtures/parity-wp2-geodesic-pairs-ecco-04.json';
import fixture5 from '../../oracles/fixtures/parity-wp2-geodesic-pairs-ecco-05.json';
import fixture6 from '../../oracles/fixtures/parity-wp2-geodesic-pairs-ecco-06.json';
import fixture7 from '../../oracles/fixtures/parity-wp2-geodesic-pairs-ecco-07.json';
import fixture8 from '../../oracles/fixtures/parity-wp2-geodesic-pairs-ecco-08.json';
import fixture9 from '../../oracles/fixtures/parity-wp2-geodesic-pairs-ecco-09.json';
import fixture10 from '../../oracles/fixtures/parity-wp2-geodesic-pairs-ecco-10.json';
import fixture11 from '../../oracles/fixtures/parity-wp2-geodesic-pairs-ecco-11.json';
import fixture12 from '../../oracles/fixtures/parity-wp2-geodesic-pairs-ecco-12.json';
import fixture13 from '../../oracles/fixtures/parity-wp2-geodesic-pairs-ecco-13.json';
import fixture14 from '../../oracles/fixtures/parity-wp2-geodesic-pairs-ecco-14.json';
import fixture15 from '../../oracles/fixtures/parity-wp2-geodesic-pairs-edges.json';
type PairExpected = {
  distances: number[];
  initialBearings: number[];
  finalBearings: number[];
  midpoints: number[];
  converged: number[];
};
type PairCase = {
  name: string;
  origins: number[];
  targets: number[];
  expected: Record<'sphere' | 'wgs84', PairExpected>;
  singular: number[];
  bearingBounds: number[];
  positionBounds: number[];
};
const fixtures: {reference: string; cases: PairCase[]}[] = [
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
  fixture11,
  fixture12,
  fixture13,
  fixture14,
  fixture15
];
for (const fixture of fixtures)
  for (const entry of fixture.cases)
    for (const model of ['sphere', 'wgs84'] as const) {
      it(`GPUGeodesicPairs PROJ parity: ${entry.name}/${model}`, async () => {
        const device = await getWebGPUTestDevice();
        requireTestDevice(device);
        const fixtureGraph = createParityGraph(device);
        const {graph, input, output} = fixtureGraph;
        const length = entry.origins.length / 2;
        const distances = output('distances', length, 'float32');
        const initialBearings = output('initial', length, 'float32');
        const finalBearings = output('final', length, 'float32');
        const midpoints = output('midpoints', length, 'float32x2');
        const converged = output('converged', length, 'uint32');
        graph.add(
          new GPUGeodesicPairs({
            spatialContext: {
              coordinateSpace: 'longitude-latitude',
              metric: model === 'sphere' ? 'great-circle' : 'ellipsoidal',
              units: 'meters'
            },
            origins: input('origins', entry.origins, 'float32x2').view,
            targets: input('targets', entry.targets, 'float32x2').view,
            output: {
              distances: distances.view,
              initialBearings: initialBearings.view,
              finalBearings: finalBearings.view,
              midpoints: midpoints.view,
              ...(model === 'wgs84' ? {status: {converged: converged.view}} : {})
            }
          })
        );
        const compiled = graph.compile();
        try {
          submitGraph(device, compiled, undefined);
          const expected = entry.expected[model];
          const caseName = `${entry.name}/${model}`;
          if (model === 'wgs84')
            compareParity(
              'GPUGeodesicPairs',
              caseName,
              fixture.reference,
              'convergence and explicit sphere-fallback flags',
              await readUint32(converged.buffer, length),
              expected.converged
            );
          compareParity(
            'GPUGeodesicPairs',
            caseName,
            fixture.reference,
            'every distance in meters',
            await readFloat32(distances.buffer, length),
            expected.distances,
            0.001,
            2e-6
          );
          compareGeodesicAngles(
            'GPUGeodesicPairs',
            caseName,
            'initial bearing degrees',
            await readFloat32(initialBearings.buffer, length),
            expected.initialBearings,
            entry.bearingBounds,
            true,
            entry.singular
          );
          compareGeodesicAngles(
            'GPUGeodesicPairs',
            caseName,
            'forward arrival bearing degrees',
            await readFloat32(finalBearings.buffer, length),
            expected.finalBearings,
            entry.bearingBounds,
            true,
            entry.singular
          );
          compareGeodesicAngles(
            'GPUGeodesicPairs',
            caseName,
            'spherical midpoint continuous longitude/latitude',
            await readFloat32(midpoints.buffer, length * 2),
            expected.midpoints,
            entry.positionBounds.flatMap(bound => [bound, bound]),
            false,
            entry.singular.flatMap(value => [value, value])
          );
        } finally {
          compiled.destroy();
          fixtureGraph.destroy();
          device.destroy?.();
        }
      }, 120_000);
    }
