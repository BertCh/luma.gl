// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import type {GPUCommandGraph, GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  GPU_OFFSET_EXPANSION_NO_OWNER,
  GPUOffsetExpansion
} from '../../../src/gpu-spatial-analysis/offset-expansion/index';
import {createGeometryFixture} from '../outline-geometry/geometry-fixture';

// Pinned from shapely 2.1.2 / geopandas 1.2.0 (scratchpad build/D/gen_offsets.py) on
// [Polygon with hole, Polygon(), MultiPolygon(triangle, square with hole), LineString, triangle].
const COORD_OFFSETS = [0, 10, 10, 23, 25, 29];
const COORD_OWNERS = [
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 4, 4, 4, 4
]; // shapely.get_coordinates(return_index=True)[1]
const COORD_LOCAL = [
  0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 1, 0, 1, 2, 3
];
const PART_OFFSETS = [0, 1, 2, 4, 5, 6]; // cumsum(get_num_geometries): the empty polygon is one part
const PART_OWNERS = [0, 1, 2, 2, 3, 4]; // get_parts(return_index=True)[1], explode(index_parts=True) level 0
const PART_LOCAL = [0, 0, 0, 1, 0, 0]; // explode(index_parts=True) level 1
// shapely.to_ragged_array of the three non-empty polygonal features (feature ids 0, 2, 4).
const RING_OFFSETS = [0, 5, 10, 14, 19, 23, 27];
const POLYGON_OFFSETS = [0, 2, 3, 5, 6];
const GEOMETRY_OFFSETS = [0, 1, 3, 4];
const FEATURE_IDS = [0, 2, 4];
const VERTEX_FEATURE = [
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4
];

type TestDevice = NonNullable<Awaited<ReturnType<typeof getWebGPUTestDevice>>>;

function createExpansionFixture(
  device: TestDevice,
  offsets: number[],
  capacity: number,
  withLocal: boolean
) {
  return createGeometryFixture(device, {
    inputs: {offsets: {values: new Uint32Array(offsets), format: 'uint32'}},
    outputs: {
      owners: {format: 'uint32', length: capacity},
      ...(withLocal ? {localIndex: {format: 'uint32' as const, length: capacity}} : {}),
      count: {format: 'uint32', length: 1},
      overflow: {format: 'uint32', length: 1},
      requiredCount: {format: 'uint32', length: 1}
    },
    create: ({inputs, outputs}) =>
      new GPUOffsetExpansion({
        offsets: inputs['offsets'] as never,
        output: {
          owners: outputs['owners'] as never,
          localIndex: outputs['localIndex'] as never,
          count: outputs['count'] as never,
          overflow: outputs['overflow'] as never,
          requiredCount: outputs['requiredCount'] as never
        }
      })
  });
}

it('GPUOffsetExpansion matches shapely get_coordinates(return_index=True) with local indices', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const fixture = createExpansionFixture(device, COORD_OFFSETS, 40, true);
  const result = await fixture.run();
  expect(result['count'][0]).toBe(29);
  expect(result['requiredCount'][0]).toBe(29);
  expect(result['overflow'][0]).toBe(0);
  expect(result['owners'].slice(0, 29)).toEqual(COORD_OWNERS);
  expect(result['localIndex'].slice(0, 29)).toEqual(COORD_LOCAL);
  expect(new Set(result['owners'].slice(29))).toEqual(new Set([GPU_OFFSET_EXPANSION_NO_OWNER]));
  expect(fixture.getCompileCount()).toBe(0);
  fixture.destroy();
});

it('GPUOffsetExpansion matches shapely get_parts and geopandas explode index maps', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const fixture = createExpansionFixture(device, PART_OFFSETS, 6, true);
  const result = await fixture.run();
  expect(result['count'][0]).toBe(6);
  expect(result['owners']).toEqual(PART_OWNERS);
  expect(result['localIndex']).toEqual(PART_LOCAL);
  fixture.destroy();
});

it('GPUOffsetExpansion skips empty owners and reports overflow', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  // Empty owners at the start, middle and end; rows 0..4 belong to owners 1, 1, 3, 3, 3.
  const fixture = createExpansionFixture(device, [0, 0, 2, 2, 5, 5, 5], 3, false);
  const result = await fixture.run();
  expect(result['requiredCount'][0]).toBe(5);
  expect(result['count'][0]).toBe(3);
  expect(result['overflow'][0]).toBe(1);
  expect(result['owners']).toEqual([1, 1, 3]);
  fixture.destroy();
});

it('GPUOffsetExpansion chains vertex, ring, polygon and feature levels with ownerMap', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const outputs = {
    polygonOwners: {format: 'uint32' as const, length: 4},
    ringOwners: {format: 'uint32' as const, length: 6},
    vertexOwners: {format: 'uint32' as const, length: 27},
    ...Object.fromEntries(
      ['a', 'b', 'c'].flatMap(stage => [
        [`${stage}Count`, {format: 'uint32' as const, length: 1}],
        [`${stage}Overflow`, {format: 'uint32' as const, length: 1}]
      ])
    )
  };
  const fixture = createGeometryFixture(device, {
    inputs: {
      geometryOffsets: {values: new Uint32Array(GEOMETRY_OFFSETS), format: 'uint32'},
      polygonOffsets: {values: new Uint32Array(POLYGON_OFFSETS), format: 'uint32'},
      ringOffsets: {values: new Uint32Array(RING_OFFSETS), format: 'uint32'},
      featureIds: {values: new Uint32Array(FEATURE_IDS), format: 'uint32'}
    },
    outputs,
    create: ({inputs, outputs: views}) => {
      const stages = [
        new GPUOffsetExpansion({
          id: 'polygon-to-feature',
          offsets: inputs['geometryOffsets'] as never,
          ownerMap: inputs['featureIds'] as never,
          output: {
            owners: views['polygonOwners'] as never,
            count: views['aCount'] as never,
            overflow: views['aOverflow'] as never
          }
        }),
        new GPUOffsetExpansion({
          id: 'ring-to-feature',
          offsets: inputs['polygonOffsets'] as never,
          ownerMap: views['polygonOwners'] as never,
          output: {
            owners: views['ringOwners'] as never,
            count: views['bCount'] as never,
            overflow: views['bOverflow'] as never
          }
        }),
        new GPUOffsetExpansion({
          id: 'vertex-to-feature',
          offsets: inputs['ringOffsets'] as never,
          ownerMap: views['ringOwners'] as never,
          output: {
            owners: views['vertexOwners'] as never,
            count: views['cCount'] as never,
            overflow: views['cOverflow'] as never
          }
        })
      ];
      const producer: GPUCommandNodeProducer = {
        getCommandNodes: <Parameters>(graph: GPUCommandGraph<Parameters>) =>
          stages.flatMap(stage => stage.getCommandNodes(graph))
      };
      return producer;
    }
  });
  const result = await fixture.run();
  expect(result['polygonOwners'].slice(0, 4)).toEqual([0, 2, 2, 4]);
  expect(result['ringOwners'].slice(0, 6)).toEqual([0, 0, 2, 2, 2, 4]);
  expect(result['cCount'][0]).toBe(27);
  expect(result['vertexOwners'].slice(0, 27)).toEqual(VERTEX_FEATURE);
  fixture.destroy();
});
