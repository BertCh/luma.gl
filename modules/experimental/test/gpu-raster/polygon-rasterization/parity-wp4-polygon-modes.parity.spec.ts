// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {
  GPUPolygonRasterization,
  GPURasterJoin,
  getGPUPolygonRasterizationParameterValues
} from '../../../src/gpu-raster/polygon-rasterization';
import {makeGPURasterMetadata} from '../../../src/gpu-raster';
import {
  GPULonLatPolygonPreparation,
  getGPULonLatPolygonPreparationParameterValues
} from '../../../src/gpu-spatial-analysis/lon-lat-polygons';
import fixtureJSON from '../../oracles/fixtures/parity-wp4-polygon-modes.json';
import {decodeOracleFixture} from '../../oracles/oracle-fixture';
import {ParityRasterGraph} from '../../oracles/parity-wp4-round3-graph';
import {compareRasterParity} from '../../oracles/parity-wp4-compare';
import {readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

type PolygonCase = Omit<
  (typeof fixtureJSON.cases)[number],
  'rowDirection' | 'origin' | 'cellSize'
> & {rowDirection: 'north' | 'south'; origin: [number, number]; cellSize: [number, number]};
const fixture = decodeOracleFixture<{reference: string; cases: PolygonCase[]}>(fixtureJSON);
for (const scene of fixture.cases)
  for (const withBoundary of [false, true]) {
    it(`Independent rasterio rasterize to join: ${scene.name}/boundary-${withBoundary}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const owner = new ParityRasterGraph(device, scene.name);
      const grid = makeGPURasterMetadata(scene);
      const parameters = owner.input(
        'parameters',
        'float32',
        getGPUPolygonRasterizationParameterValues(grid)
      );
      const zones = owner.output('zones', 'uint32', scene.zones.length);
      const boundary = owner.output('boundary', 'uint32', scene.zones.length);
      const overflow = owner.output('overflow', 'uint32', 1);
      const counts = owner.output('counts', 'uint32', 2);
      const pointZones = owner.output('point-zones', 'uint32', scene.pointZones.length);
      let geometry = {
        positions: owner.input('positions', 'float32x2', new Float32Array(scene.positions)),
        featureOffsets: owner.input('features', 'uint32', Uint32Array.of(0, 1, 2)),
        polygonOffsets: owner.input('polygons', 'uint32', Uint32Array.of(0, 1, 2)),
        ringOffsets: owner.input('rings', 'uint32', Uint32Array.of(0, 4, 8))
      };
      const preparationOverflow = owner.output('preparation-overflow', 'uint32', 1);
      const preparationInvalid = owner.output('preparation-invalid', 'uint32', 1);
      if (scene.geographic) {
        // Crossing rings must be split before rasterization, as in the showcase's
        // polygon-measures-global preparation graph. Keep the rasterio oracle unchanged.
        const prepared = {
          positions: owner.output('prepared-positions', 'float32x2', 64).view,
          featureOffsets: owner.output('prepared-features', 'uint32', 3).view,
          polygonOffsets: owner.output('prepared-polygons', 'uint32', 5).view,
          ringOffsets: owner.output('prepared-rings', 'uint32', 5).view
        };
        owner.graph.add(
          new GPULonLatPolygonPreparation({
            ...geometry,
            spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'none'},
            parameters: owner.input(
              'preparation-parameters',
              'float32',
              getGPULonLatPolygonPreparationParameterValues('linear')
            ),
            output: {
              ...prepared,
              status: {
                overflow: preparationOverflow.view,
                invalidCount: preparationInvalid.view
              }
            }
          })
        );
        geometry = prepared;
      }
      owner.graph.add(
        new GPUPolygonRasterization({
          grid,
          parameters,
          spatialContext: scene.geographic
            ? {coordinateSpace: 'longitude-latitude', metric: 'none'}
            : undefined,
          ...geometry,
          crossingCapacity: 128,
          output: {
            zones: zones.view,
            boundary: withBoundary ? boundary.view : undefined,
            status: {overflow: overflow.view}
          }
        })
      );
      owner.graph.add(
        new GPURasterJoin({
          grid,
          parameters,
          points: owner.input('points', 'float32x2', new Float32Array(scene.points)),
          zones: zones.view,
          boundary: withBoundary ? boundary.view : undefined,
          zoneCount: 2,
          output: {counts: counts.view, pointZones: pointZones.view}
        })
      );
      const compiled = owner.graph.compile();
      try {
        submitGraph(device, compiled, undefined);
        const compare = (
          className: string,
          metric: string,
          actual: ArrayLike<number>,
          expected: ArrayLike<number>
        ) =>
          compareRasterParity(
            className,
            `${scene.name}/boundary-${withBoundary}`,
            fixture.reference,
            metric,
            actual,
            expected,
            'E'
          );
        if (scene.geographic) {
          compare(
            'GPULonLatPolygonPreparation',
            'overflow',
            await readUint32(preparationOverflow.buffer, 1),
            [0]
          );
          compare(
            'GPULonLatPolygonPreparation',
            'invalid',
            await readUint32(preparationInvalid.buffer, 1),
            [0]
          );
        }
        compare(
          'GPUPolygonRasterization',
          'full-raster',
          await readUint32(zones.buffer, scene.zones.length),
          scene.zones
        );
        compare('GPUPolygonRasterization', 'overflow', await readUint32(overflow.buffer, 1), [0]);
        compare(
          'GPURasterJoin',
          'all-point-zones',
          await readUint32(pointZones.buffer, scene.pointZones.length),
          scene.pointZones
        );
        compare(
          'GPURasterJoin',
          'all-zone-counts',
          await readUint32(counts.buffer, 2),
          scene.counts
        );
      } finally {
        compiled.destroy();
        owner.destroy();
      }
    });
  }
