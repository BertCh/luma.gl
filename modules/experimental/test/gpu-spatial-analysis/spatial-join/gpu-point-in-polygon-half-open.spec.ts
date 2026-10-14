// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  GPU_SPATIAL_JOIN_NO_FEATURE as NO_FEATURE,
  GPUPointInPolygonJoin,
  type GPUPointInPolygonBoundaryRule
} from '../../../src/gpu-spatial-analysis/spatial-join';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {HALF_OPEN_SCENES} from './half-open-fixtures';
import {buildPolygonFeatureArrays, type OraclePolygonFeature} from './spatial-join-oracle';
import {requireTestDevice} from '../../utils/require-test-device';

/** Stable IDs are `100 + original row`, so assignments can be compared after reordering rows. */
const ID_BASE = 100;

/** Joins `points` to `features` (rows in the given order, IDs `100 + originalRow[row]`). */
async function joinWithRule(
  device: Device,
  features: OraclePolygonFeature[],
  originalRows: number[],
  points: [number, number][],
  boundaryRule: GPUPointInPolygonBoundaryRule | undefined
): Promise<{ids: number[]; uncertain: number}> {
  const graph = new GPUCommandGraph(device, {id: 'pip-half-open'});
  const arrays = buildPolygonFeatureArrays(features);
  const buffers: Buffer[] = [];
  const input = (data: Float32Array | Uint32Array) => {
    const buffer = createInputBuffer(device, data);
    buffers.push(buffer);
    return buffer;
  };
  const output = (length: number) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return buffer;
  };
  const pointFeatureIds = output(points.length);
  const uncertainCount = output(1);
  const overflow = output(1);
  graph.add(
    new GPUPointInPolygonJoin({
      points: {
        kind: 'points',
        positions: importGraphBuffer(
          graph,
          'points',
          input(Float32Array.from(points.flat())),
          'float32x2',
          points.length
        )
      },
      polygons: {
        kind: 'polygons',
        positions: importGraphBuffer(
          graph,
          'polygon-positions',
          input(arrays.polygonPositions),
          'float32x2',
          arrays.polygonPositions.length / 2
        ),
        featureOffsets: importGraphBuffer(
          graph,
          'feature-offsets',
          input(arrays.featureOffsets),
          'uint32',
          arrays.featureOffsets.length
        ),
        polygonOffsets: importGraphBuffer(
          graph,
          'polygon-offsets',
          input(arrays.polygonOffsets),
          'uint32',
          arrays.polygonOffsets.length
        ),
        ringOffsets: importGraphBuffer(
          graph,
          'ring-offsets',
          input(arrays.ringOffsets),
          'uint32',
          arrays.ringOffsets.length
        ),
        sourceIds: importGraphBuffer(
          graph,
          'feature-ids',
          input(Uint32Array.from(originalRows.map(row => ID_BASE + row))),
          'uint32',
          features.length
        )
      },
      candidateCapacity: points.length * features.length,
      boundaryRule,
      output: {
        pointFeatureIds: importGraphBuffer(
          graph,
          'point-feature-ids',
          pointFeatureIds,
          'uint32',
          points.length
        ),
        status: {
          overflow: importGraphBuffer(graph, 'overflow', overflow, 'uint32', 1),
          uncertainCount: importGraphBuffer(graph, 'uncertain-count', uncertainCount, 'uint32', 1)
        }
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const ids = await readUint32(pointFeatureIds, points.length);
  const [uncertain] = await readUint32(uncertainCount, 1);
  expect((await readUint32(overflow, 1))[0]).toBe(0);
  compiled.destroy();
  for (const buffer of buffers) {
    buffer.destroy();
  }
  return {ids, uncertain};
}

/** Original feature row of a stable ID, or `null` for no feature. */
const toOwner = (id: number): number | null => (id === NO_FEATURE ? null : id - ID_BASE);

/** Every permutation of `[0, count)`. */
function getPermutations(count: number): number[][] {
  if (count <= 1) {
    return [Array.from({length: count}, (_, row) => row)];
  }
  return getPermutations(count - 1).flatMap(permutation =>
    Array.from({length: count}, (_, slot) => [
      ...permutation.slice(0, slot),
      count - 1,
      ...permutation.slice(slot)
    ])
  );
}

for (const [name, scene] of Object.entries(HALF_OPEN_SCENES)) {
  it(`GPUPointInPolygonJoin half-open rule assigns boundary points once, independent of feature order (${name})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    for (const permutation of getPermutations(scene.features.length)) {
      const features = permutation.map(row => scene.features[row]);
      // The default rule is the half-open one.
      for (const rule of [undefined, 'half-open'] as const) {
        const {ids, uncertain} = await joinWithRule(
          device,
          features,
          permutation,
          scene.points,
          rule
        );
        expect(ids.map(toOwner), `${name} order ${permutation} rule ${rule}`).toEqual(scene.owners);
        expect(uncertain).toBe(0);
      }
    }
  });
}

it('GPUPointInPolygonJoin closed rule keeps the smallest row, so a shared edge follows feature order', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const {features, points} = HALF_OPEN_SCENES.squares;
  // (2, 1) lies on the edge shared by squares 0 and 1.
  const sharedEdge = points.findIndex(([x, y]) => x === 2 && y === 1);
  const forward = await joinWithRule(device, features, [0, 1, 2, 3], points, 'closed');
  const reversed = await joinWithRule(
    device,
    [features[1], features[0], features[2], features[3]],
    [1, 0, 2, 3],
    points,
    'closed'
  );
  expect(toOwner(forward.ids[sharedEdge])).toBe(0);
  expect(toOwner(reversed.ids[sharedEdge])).toBe(1);
  // The closed rule keeps points on the outer boundary that the half-open rule leaves out.
  const outerRight = points.findIndex(([x, y]) => x === 4 && y === 1);
  expect(toOwner(forward.ids[outerRight])).toBe(1);
});

it('GPUPointInPolygonJoin open rule leaves every boundary point unassigned', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const {features, points, owners} = HALF_OPEN_SCENES.squares;
  const {ids} = await joinWithRule(device, features, [0, 1, 2, 3], points, 'open');
  // Strictly interior probe points (1, 1) and (3, 3) keep their feature; boundary points do not.
  const interior = new Set(
    points
      .map(([x, y], row) => [x > 0 && x < 4 && y > 0 && y < 4 && x !== 2 && y !== 2, row] as const)
      .filter(([isInterior]) => isInterior)
      .map(([, row]) => row)
  );
  expect(interior.size).toBe(2);
  for (const [row, id] of ids.entries()) {
    expect(toOwner(id), `point ${row}`).toBe(interior.has(row) ? owners[row] : null);
  }
});
