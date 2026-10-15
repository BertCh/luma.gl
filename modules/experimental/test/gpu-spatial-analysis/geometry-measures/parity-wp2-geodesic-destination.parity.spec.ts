// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {GPUGeodesicDestination} from '../../../src/gpu-spatial-analysis/geometry-measures/gpu-geodesic-destination';
import {requireTestDevice} from '../../utils/require-test-device';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {compareParity} from '../../oracles/parity-wp2-assert';
import {createParityGraph, compareGeodesicAngles} from '../../oracles/parity-wp2-geodesic-harness';
import fixture0 from '../../oracles/fixtures/parity-wp2-geodesic-destination-edges.json';
import fixture1 from '../../oracles/fixtures/parity-wp2-geodesic-destination-flights-00.json';
import fixture2 from '../../oracles/fixtures/parity-wp2-geodesic-destination-flights-01.json';
import fixture3 from '../../oracles/fixtures/parity-wp2-geodesic-destination-flights-02.json';
import fixture4 from '../../oracles/fixtures/parity-wp2-geodesic-destination-flights-03.json';
import fixture5 from '../../oracles/fixtures/parity-wp2-geodesic-destination-flights-04.json';
import fixture6 from '../../oracles/fixtures/parity-wp2-geodesic-destination-flights-05.json';
import fixture7 from '../../oracles/fixtures/parity-wp2-geodesic-destination-flights-06.json';
import fixture8 from '../../oracles/fixtures/parity-wp2-geodesic-destination-flights-07.json';
import fixture9 from '../../oracles/fixtures/parity-wp2-geodesic-destination-flights-08.json';
import fixture10 from '../../oracles/fixtures/parity-wp2-geodesic-destination-flights-09.json';
type DestinationCase = {
  name: string;
  model: string;
  origins: number[];
  bearings: number[];
  distances: number[];
  positionAbsolute: number;
  bearingBounds: number[];
  expected: {destinations: number[]; finalBearings: number[]; converged: number[]};
};
const fixtures: {reference: string; cases: DestinationCase[]}[] = [
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
  fixture10
];
for (const fixture of fixtures)
  for (const entry of fixture.cases) {
    it(`GPUGeodesicDestination PROJ parity: ${entry.name}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const fixtureGraph = createParityGraph(device);
      const {graph, input, output} = fixtureGraph;
      const length = entry.distances.length;
      const destinations = output('destinations', length, 'float32x2');
      const finalBearings = output('final', length, 'float32');
      const converged = output('converged', length, 'uint32');
      graph.add(
        new GPUGeodesicDestination({
          spatialContext: {
            coordinateSpace: 'longitude-latitude',
            metric: entry.model === 'sphere' ? 'great-circle' : 'ellipsoidal',
            units: 'meters'
          },
          origins: input('origins', entry.origins, 'float32x2').view,
          bearings: input('bearings', entry.bearings, 'float32').view,
          distances: input('distances', entry.distances, 'float32').view,
          output: {
            destinations: destinations.view,
            finalBearings: finalBearings.view,
            ...(entry.model === 'wgs84' ? {status: {converged: converged.view}} : {})
          }
        })
      );
      const compiled = graph.compile();
      try {
        submitGraph(device, compiled, undefined);
        if (entry.model === 'wgs84')
          compareParity(
            'GPUGeodesicDestination',
            entry.name,
            fixture.reference,
            'every convergence flag',
            await readUint32(converged.buffer, length),
            entry.expected.converged
          );
        compareParity(
          'GPUGeodesicDestination',
          entry.name,
          fixture.reference,
          'every continuous longitude/latitude',
          await readFloat32(destinations.buffer, length * 2),
          entry.expected.destinations,
          entry.positionAbsolute
        );
        compareGeodesicAngles(
          'GPUGeodesicDestination',
          entry.name,
          'every forward arrival bearing degrees',
          await readFloat32(finalBearings.buffer, length),
          entry.expected.finalBearings,
          entry.bearingBounds,
          true
        );
      } finally {
        compiled.destroy();
        fixtureGraph.destroy();
        device.destroy?.();
      }
    }, 120_000);
  }
