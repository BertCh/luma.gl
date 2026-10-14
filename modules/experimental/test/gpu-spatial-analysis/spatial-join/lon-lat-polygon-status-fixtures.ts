// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUCommandGraph,
  type GraphDataView,
  type CompiledGPUCommandGraph
} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import type {
  GPULonLatPolygonStatusPort,
  GPUPackedGeometryPort,
  GPUPolygonGeometryPort,
  GPUSpatialContext
} from '../../../src/analysis-core/contracts';
import {
  GPUNearestFeatureJoin,
  GPUPointInPolygonJoin,
  GPUSpatialPredicateJoin
} from '../../../src/gpu-spatial-analysis/spatial-join';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {
  ARCTIC_RING,
  CHICAGO_SQUARE,
  FIJI_SPLIT,
  FIJI_TRANSMERIDIAN,
  createLonLatPolygonFixtureViews
} from '../lon-lat-polygons/fixtures';
import {
  classifyRing,
  getLonLatFeatures,
  prepareLonLatPolygons,
  type LonLatPolygonData
} from '../lon-lat-polygons/lon-lat-polygon-oracle';

/** Operators covered by the shared consumer-status cases. */
export type PolygonJoinName =
  | 'GPUPointInPolygonJoin'
  | 'GPUSpatialPredicateJoin'
  | 'GPUNearestFeatureJoin';

/** Declares a minimal point/polygon or polygon/polygon join for node and GPU tests. */
export function createPolygonStatusJoin(
  graph: GPUCommandGraph,
  operation: PolygonJoinName,
  spatialContext: GPUSpatialContext | undefined,
  status: GPULonLatPolygonStatusPort,
  options: {
    polygons?: GPUPackedGeometryPort<GPUPolygonGeometryPort>;
    points?: GraphDataView<'float32x2'>;
    result?: GraphDataView<'uint32'>;
    distance?: number;
    polygonQueries?: boolean;
  } = {}
) {
  const scalar = (name: string) => createTransientView(graph, name, 'uint32', 1);
  const polygons = options.polygons ?? {
    kind: 'polygons' as const,
    positions: createTransientView(graph, 'positions', 'float32x2', 4),
    featureOffsets: createTransientView(graph, 'feature-offsets', 'uint32', 2),
    polygonOffsets: createTransientView(graph, 'polygon-offsets', 'uint32', 2),
    ringOffsets: createTransientView(graph, 'ring-offsets', 'uint32', 2)
  };
  const points = {
    kind: 'points' as const,
    positions: options.points ?? createTransientView(graph, 'points', 'float32x2', 1)
  };
  const outputStatus = {overflow: scalar('overflow'), ...status};
  const result = options.result ?? scalar('result');
  if (operation === 'GPUPointInPolygonJoin') {
    return new GPUPointInPolygonJoin({
      id: operation,
      spatialContext,
      points,
      polygons,
      candidateCapacity: 4,
      output: {pointFeatureIds: result, status: outputStatus}
    });
  }
  if (operation === 'GPUSpatialPredicateJoin') {
    return new GPUSpatialPredicateJoin({
      id: operation,
      spatialContext,
      left: options.polygonQueries ? polygons : points,
      right: polygons,
      predicate: options.distance === undefined ? 'intersects' : 'dwithin',
      distance: options.distance,
      candidateCapacity: 4,
      output: {
        pairs: {
          leftIds: scalar('left-ids'),
          rightIds: scalar('right-ids'),
          status: {count: result, overflow: scalar('pair-overflow')}
        },
        status: outputStatus
      }
    });
  }
  return new GPUNearestFeatureJoin({
    id: operation,
    spatialContext,
    queries: options.polygonQueries ? polygons : points,
    features: polygons,
    k: 1,
    neighborCapacity: 1,
    output: {neighborIds: result, neighborCounts: scalar('neighbor-counts'), status: outputStatus}
  });
}

/** Binary64 reference counts for every ring in a feature collection. */
export function getPolygonStatusCounts(fixture: LonLatPolygonData): number[] {
  const rings = getLonLatFeatures(fixture)
    .flat(2)
    .map(ring => classifyRing(ring));
  return [rings.filter(ring => ring.crossing).length, rings.filter(ring => ring.poleRing).length];
}

/** Shared GPU cases, registered by each operator's existing GPU spec. */
export function registerLonLatPolygonStatusTests(operation: PolygonJoinName): void {
  for (const [name, fixture, planar] of [
    ['Fiji transmeridian', FIJI_TRANSMERIDIAN, false],
    ['Arctic', ARCTIC_RING, false],
    ['split Fiji', FIJI_SPLIT, false],
    ['plain', CHICAGO_SQUARE, false],
    ['prepared with empty copy', prepareLonLatPolygons(CHICAGO_SQUARE), false],
    ['prepared Fiji', prepareLonLatPolygons(FIJI_TRANSMERIDIAN), false],
    ['prepared Arctic', prepareLonLatPolygons(ARCTIC_RING), false],
    ['planar counters untouched', FIJI_TRANSMERIDIAN, true]
  ] as const) {
    it(`${operation} lon/lat status: ${name}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const graph = new GPUCommandGraph(device);
      const polygons = createLonLatPolygonFixtureViews(device, graph, fixture);
      const buffers = [
        createOutputBuffer(device, 1),
        createOutputBuffer(device, 1),
        createInputBuffer(device, new Float32Array([-87.65, 41.85])),
        createOutputBuffer(device, 1)
      ];
      buffers[0].write(new Uint32Array([17]));
      buffers[1].write(new Uint32Array([19]));
      buffers[3].write(new Uint32Array([0xffffffff]));
      let compiled: CompiledGPUCommandGraph | undefined;
      try {
        const join = createPolygonStatusJoin(
          graph,
          operation,
          planar
            ? undefined
            : {
                coordinateSpace: 'longitude-latitude',
                metric: operation === 'GPUNearestFeatureJoin' ? 'great-circle' : 'none'
              },
          {
            antimeridianUncertainCount: importGraphBuffer(
              graph,
              'antimeridian',
              buffers[0],
              'uint32',
              1
            ),
            poleUncertainCount: importGraphBuffer(graph, 'pole', buffers[1], 'uint32', 1)
          },
          {
            polygons: {kind: 'polygons', ...polygons},
            points: importGraphBuffer(graph, 'points', buffers[2], 'float32x2', 1),
            result: importGraphBuffer(graph, 'result', buffers[3], 'uint32', 1)
          }
        );
        graph.add(join);
        compiled = graph.compile();
        // Repeat to prove consumer-owned counters are cleared on every encoding.
        for (let encoding = 0; encoding < 2; encoding++) {
          submitGraph(device, compiled, undefined);
          expect([
            ...(await readUint32(buffers[0], 1)),
            ...(await readUint32(buffers[1], 1))
          ]).toEqual(planar ? [17, 19] : getPolygonStatusCounts(fixture));
          if (name === 'prepared with empty copy') {
            expect(
              Array.from(fixture.ringOffsets).some(
                (offset, index, offsets) => index > 0 && offset === offsets[index - 1]
              )
            ).toBe(true);
            expect(await readUint32(buffers[3], 1)).toEqual([
              operation === 'GPUSpatialPredicateJoin' ? 1 : 0
            ]);
          }
        }
      } finally {
        compiled?.destroy();
        polygons.destroy();
        for (const buffer of buffers) buffer.destroy();
      }
    });
  }
}
