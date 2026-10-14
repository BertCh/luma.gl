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
import {WgslReflect} from 'wgsl_reflect';
import type {
  GPULonLatPolygonStatusPort,
  GPUPackedGeometryPort,
  GPUPolygonGeometryPort,
  GPUSpatialContext
} from '../../../src/analysis-core/contracts';
import {GPUZonalStatistics} from '../../../src/gpu-spatial-analysis/zonal-statistics/gpu-zonal-statistics';
import {GPURegionMask} from '../../../src/gpu-spatial-analysis/region-statistics/gpu-region-mask';
import {GPURegionStatistics} from '../../../src/gpu-spatial-analysis/region-statistics/gpu-region-statistics';
import {GPULineLengthPerPolygon} from '../../../src/gpu-spatial-analysis/line-density/gpu-line-length-per-polygon';
import {GPUPointSpatialQuery} from '../../../src/geospatial/gpu-point-spatial-query';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  readUint32,
  readFloat32,
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
  flattenLonLatFeatures,
  getLonLatFeatures,
  prepareLonLatPolygons
} from '../lon-lat-polygons/lon-lat-polygon-oracle';
import {computeLineLengthPerPolygon} from '../line-density/line-length-per-polygon-oracle';
import {getPolygonStatusCounts} from '../spatial-join/lon-lat-polygon-status-fixtures';

/** Consumer names sharing the polygon preparation diagnostics. */
export type PolygonConsumerName =
  | 'GPUZonalStatistics'
  | 'GPURegionMask'
  | 'GPURegionStatistics'
  | 'GPULineLengthPerPolygon'
  | 'GPUPointSpatialQuery';

/** Declares the smallest valid consumer, with optional imported GPU fixtures. */
export function createPolygonConsumer(
  graph: GPUCommandGraph,
  operation: PolygonConsumerName,
  spatialContext: GPUSpatialContext | undefined,
  status: GPULonLatPolygonStatusPort,
  options: {
    polygons?: GPUPackedGeometryPort<GPUPolygonGeometryPort>;
    positions?: GraphDataView<'float32x2'>;
    parameters?: GraphDataView<'float32'>;
    vertexCount?: GraphDataView<'uint32'>;
    lineOffsets?: GraphDataView<'uint32'>;
    result?: GraphDataView<'uint32'>;
    lengths?: GraphDataView<'float32'>;
    screenTransform?: GraphDataView<'float32'>;
  } = {}
) {
  const scalar = (name: string) => createTransientView(graph, name, 'uint32', 1);
  const polygons = options.polygons ?? {
    kind: 'polygons' as const,
    positions: createTransientView(graph, 'polygon-positions', 'float32x2', 4),
    featureOffsets: createTransientView(graph, 'feature-offsets', 'uint32', 2),
    polygonOffsets: createTransientView(graph, 'polygon-offsets', 'uint32', 2),
    ringOffsets: createTransientView(graph, 'ring-offsets', 'uint32', 2)
  };
  const positions = options.positions ?? createTransientView(graph, 'points', 'float32x2', 2);
  const outputStatus = {overflow: scalar('overflow'), ...status};
  const common = {id: operation, spatialContext, positions};
  if (operation === 'GPUZonalStatistics')
    return new GPUZonalStatistics({
      ...common,
      features: {...polygons, candidateCapacity: 8},
      output: {counts: options.result ?? scalar('counts'), status: outputStatus}
    });
  if (operation === 'GPULineLengthPerPolygon')
    return new GPULineLengthPerPolygon({
      ...common,
      polygons,
      pairCapacity: 8,
      lineOffsets: options.lineOffsets ?? createTransientView(graph, 'line-offsets', 'uint32', 2),
      output: {
        lengths: options.lengths ?? createTransientView(graph, 'lengths', 'float32', 1),
        status: outputStatus
      }
    });
  if (operation === 'GPUPointSpatialQuery')
    return new GPUPointSpatialQuery({
      ...common,
      kind: 'polygon',
      polygon: polygons,
      parameters: options.parameters ?? createTransientView(graph, 'parameters', 'float32', 4),
      output: {
        ids: createTransientView(graph, 'ids', 'uint32', 2),
        status: {...outputStatus, count: options.result ?? scalar('count')}
      }
    });
  const region = {
    kind: 'polygon' as const,
    vertices: polygons.positions,
    vertexCount: options.vertexCount ?? scalar('vertex-count'),
    screenTransform: options.screenTransform
  };
  if (operation === 'GPURegionMask')
    return new GPURegionMask({
      ...common,
      region,
      output: {
        mask: options.result ?? createTransientView(graph, 'mask', 'uint32', 2),
        status: outputStatus
      }
    });
  return new GPURegionStatistics({
    ...common,
    region,
    output: {summary: options.result ?? createTransientView(graph, 'summary', 'uint32', 8), status}
  });
}

/** Node coverage for context validation, scalar shape, graph ownership and opt-in construction. */
export function registerPolygonConsumerNodeTests(operation: PolygonConsumerName): void {
  const context: GPUSpatialContext = {
    coordinateSpace: 'longitude-latitude',
    metric: operation === 'GPULineLengthPerPolygon' ? 'great-circle' : 'none'
  };
  it(`${operation} diagnostic expectations use the shared binary64 oracle`, () => {
    expect(getPolygonStatusCounts(FIJI_TRANSMERIDIAN)).toEqual([1, 0]);
    expect(getPolygonStatusCounts(ARCTIC_RING)).toEqual([1, 1]);
    expect(getPolygonStatusCounts(prepareLonLatPolygons(FIJI_TRANSMERIDIAN))).toEqual([0, 0]);
    expect(getPolygonStatusCounts(prepareLonLatPolygons(ARCTIC_RING))).toEqual([0, 0]);
  });
  it(`${operation} accepts default/linear edges and rejects great-circle edges`, () => {
    const device = createNullWebGPUDevice();
    try {
      for (const edgeModel of [undefined, 'linear', 'great-circle'] as const) {
        const graph = new GPUCommandGraph(device);
        const create = () => createPolygonConsumer(graph, operation, {...context, edgeModel}, {});
        if (edgeModel === 'great-circle')
          expect(create).toThrow(
            `${operation} needs linear edges; prepare polygons with GPULonLatPolygonPreparation`
          );
        else expect(create).not.toThrow();
      }
    } finally {
      device.destroy();
    }
  });
  it(`${operation} only builds requested lon/lat checks before its main kernels`, () => {
    const device = createNullWebGPUDevice();
    try {
      for (const planar of [false, true])
        for (const fields of [
          [],
          ['antimeridianUncertainCount'],
          ['poleUncertainCount'],
          ['antimeridianUncertainCount', 'poleUncertainCount']
        ] as const) {
          const graph = new GPUCommandGraph(device);
          const status = Object.fromEntries(
            fields.map(name => [name, createTransientView(graph, name, 'uint32', 1)])
          );
          const consumer = createPolygonConsumer(
            graph,
            operation,
            planar ? undefined : context,
            status
          );
          const nodes = consumer.getCommandNodes(graph);
          if (!planar && fields.length === 2) {
            for (const {source} of getGPUCommandNodeKernelSources(nodes)) {
              expect(() => new WgslReflect(source)).not.toThrow();
            }
          }
          const checks = nodes.filter(node => node.id.includes('-lon-lat-polygon-check-'));
          expect(checks).toHaveLength(!planar && fields.length ? 1 : 0);
          expect(
            nodes.filter(
              node =>
                node.id.includes('-clear-antimeridianUncertainCount') ||
                node.id.includes('-clear-poleUncertainCount')
            )
          ).toHaveLength(planar ? 0 : fields.length);
          if (checks.length)
            expect(nodes.indexOf(checks[0])).toBe(
              fields.length + (operation.startsWith('GPURegion') ? 2 : 0)
            );
        }
      for (const name of ['antimeridianUncertainCount', 'poleUncertainCount']) {
        const graph = new GPUCommandGraph(device);
        const separator =
          operation === 'GPULineLengthPerPolygon' || operation === 'GPUPointSpatialQuery'
            ? '.'
            : ' ';
        expect(() =>
          createPolygonConsumer(graph, operation, context, {
            [name]: createTransientView(graph, name, 'uint32', 0)
          })
        ).toThrow(`${operation} output.status${separator}${name} must contain one uint32 row`);
        const aliasGraph = new GPUCommandGraph(device);
        const input = createTransientView(aliasGraph, 'aliased', 'uint32', 2);
        expect(() =>
          createPolygonConsumer(
            aliasGraph,
            operation,
            context,
            {[name]: input},
            {
              polygons: {
                kind: 'polygons',
                positions: createTransientView(aliasGraph, 'vertices', 'float32x2', 4),
                featureOffsets: input,
                polygonOffsets: input,
                ringOffsets: input
              },
              vertexCount: input
            }
          )
        ).toThrow(/overlap|share buffers/);
        const otherGraph = new GPUCommandGraph(device);
        const targetGraph = new GPUCommandGraph(device);
        const consumer = createPolygonConsumer(targetGraph, operation, context, {
          [name]: createTransientView(otherGraph, name, 'uint32', 1)
        });
        expect(() => consumer.getCommandNodes(targetGraph)).toThrow();
      }
    } finally {
      device.destroy();
    }
  });
  if (operation === 'GPURegionMask' || operation === 'GPURegionStatistics') {
    it(`${operation} leaves screen-space lasso diagnostics untouched`, () => {
      const device = createNullWebGPUDevice();
      try {
        const graph = new GPUCommandGraph(device);
        const consumer = createPolygonConsumer(
          graph,
          operation,
          context,
          {
            antimeridianUncertainCount: createTransientView(graph, 'antimeridian', 'uint32', 1)
          },
          {screenTransform: createTransientView(graph, 'transform', 'float32', 20)}
        );
        expect(
          consumer
            .getCommandNodes(graph)
            .some(
              node =>
                node.id.includes('lon-lat-polygon-check') || node.id.includes('clear-antimeridian')
            )
        ).toBe(false);
      } finally {
        device.destroy();
      }
    });
  }
}

/** GPU cases reuse the P-5a binary64 ring oracle and P-1 geometry fixtures. */
export function registerPolygonConsumerGPUTests(operation: PolygonConsumerName): void {
  const isRegion = operation === 'GPURegionMask' || operation === 'GPURegionStatistics';
  for (const [name, fixture, planar] of [
    ['Fiji transmeridian', FIJI_TRANSMERIDIAN, false],
    ['Arctic', ARCTIC_RING, false],
    ['split Fiji', FIJI_SPLIT, false],
    ['plain', CHICAGO_SQUARE, false],
    ['prepared with empty copy', prepareLonLatPolygons(CHICAGO_SQUARE), false],
    ['prepared Fiji', prepareLonLatPolygons(FIJI_TRANSMERIDIAN), false],
    ['prepared Arctic', prepareLonLatPolygons(ARCTIC_RING), false],
    ['planar untouched', FIJI_TRANSMERIDIAN, true]
  ] as const) {
    it(`${operation} polygon diagnostics: ${name}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      // A lasso has one ring. Check every prepared component, including the unused empty copy.
      const fixtures = isRegion
        ? getLonLatFeatures(fixture)
            .flat(2)
            .map(ring => flattenLonLatFeatures([[[ring]]]))
        : [fixture];
      for (const data of fixtures) {
        const graph = new GPUCommandGraph(device);
        const polygons = createLonLatPolygonFixtureViews(device, graph, data);
        const buffers = [
          createOutputBuffer(device, 1),
          createOutputBuffer(device, 1),
          createInputBuffer(device, new Float32Array([-87.65, 41.85, -87.64, 41.85])),
          createInputBuffer(device, new Float32Array([-180, -90, 180, 90])),
          createInputBuffer(device, new Uint32Array([0, 2])),
          createInputBuffer(device, new Uint32Array([data.positions.length / 2])),
          createOutputBuffer(
            device,
            operation === 'GPURegionStatistics' ? 8 : operation === 'GPURegionMask' ? 2 : 1
          )
        ];
        buffers[0].write(new Uint32Array([17]));
        buffers[1].write(new Uint32Array([19]));
        // Lassos retain a minimum storage capacity of three even when their active ring is empty.
        const lassoBuffer =
          isRegion && data.positions.length < 6
            ? createInputBuffer(device, new Float32Array(6))
            : undefined;
        let compiled: CompiledGPUCommandGraph | undefined;
        try {
          const consumer = createPolygonConsumer(
            graph,
            operation,
            planar
              ? undefined
              : {
                  coordinateSpace: 'longitude-latitude',
                  metric: operation === 'GPULineLengthPerPolygon' ? 'great-circle' : 'none'
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
              polygons: {
                kind: 'polygons',
                ...polygons,
                ...(lassoBuffer
                  ? {
                      positions: importGraphBuffer(
                        graph,
                        'empty-lasso',
                        lassoBuffer,
                        'float32x2',
                        3
                      )
                    }
                  : {})
              },
              positions: importGraphBuffer(graph, 'points', buffers[2], 'float32x2', 2),
              parameters: importGraphBuffer(graph, 'parameters', buffers[3], 'float32', 4),
              lineOffsets: importGraphBuffer(graph, 'lines', buffers[4], 'uint32', 2),
              vertexCount: importGraphBuffer(graph, 'vertex-count', buffers[5], 'uint32', 1),
              result:
                operation === 'GPULineLengthPerPolygon'
                  ? undefined
                  : importGraphBuffer(
                      graph,
                      'result',
                      buffers[6],
                      'uint32',
                      operation === 'GPURegionStatistics'
                        ? 8
                        : operation === 'GPURegionMask'
                          ? 2
                          : 1
                    ),
              lengths:
                operation === 'GPULineLengthPerPolygon'
                  ? importGraphBuffer(graph, 'lengths', buffers[6], 'float32', 1)
                  : undefined
            }
          );
          graph.add(consumer);
          compiled = graph.compile();
          for (let encoding = 0; encoding < 2; encoding++) {
            submitGraph(device, compiled, undefined);
            expect([
              ...(await readUint32(buffers[0], 1)),
              ...(await readUint32(buffers[1], 1))
            ]).toEqual(planar ? [17, 19] : getPolygonStatusCounts(data));
            if (name === 'prepared with empty copy' && operation === 'GPULineLengthPerPolygon') {
              const expected = computeLineLengthPerPolygon(
                [
                  [
                    [Math.fround(-87.65), Math.fround(41.85)],
                    [Math.fround(-87.64), Math.fround(41.85)]
                  ]
                ],
                undefined,
                getLonLatFeatures(data),
                true
              ).lengths[0];
              expect(Math.abs((await readFloat32(buffers[6], 1))[0] - expected)).toBeLessThan(
                expected * 0.002
              );
            }
            if (name === 'prepared with empty copy' && operation !== 'GPULineLengthPerPolygon') {
              const empty = data.positions.length === 0;
              expect((await readUint32(buffers[6], 1))[0]).toBe(
                empty ? 0 : operation === 'GPURegionMask' ? 1 : 2
              );
            }
          }
          if (isRegion && !planar) {
            buffers[5].write(new Uint32Array([0]));
            submitGraph(device, compiled, undefined);
            expect([
              ...(await readUint32(buffers[0], 1)),
              ...(await readUint32(buffers[1], 1))
            ]).toEqual([0, 0]);
            buffers[5].write(new Uint32Array([0xffffffff]));
            submitGraph(device, compiled, undefined);
            expect([
              ...(await readUint32(buffers[0], 1)),
              ...(await readUint32(buffers[1], 1))
            ]).toEqual(getPolygonStatusCounts(data));
          }
        } finally {
          compiled?.destroy();
          polygons.destroy();
          lassoBuffer?.destroy();
          for (const buffer of buffers) buffer.destroy();
        }
      }
    });
  }
}
