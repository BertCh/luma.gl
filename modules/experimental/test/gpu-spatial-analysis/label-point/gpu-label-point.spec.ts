// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPULabelPoint} from '../../../src/gpu-spatial-analysis/label-point';
import {
  createGeometryFixture,
  createRandom,
  flattenFeatures
} from '../outline-geometry/geometry-fixture';
import {findPoleOfInaccessibility, getSignedDistance, type Ring} from './label-point-oracle';

function createStar(
  random: () => number,
  cx: number,
  cy: number,
  radius: number,
  count: number,
  jitter: number,
  clockwise: boolean
): Ring {
  return Array.from({length: count}, (_, index) => {
    const angle = ((clockwise ? -1 : 1) * 2 * Math.PI * index) / count;
    const r = radius * (1 - jitter + jitter * random());
    return [cx + r * Math.cos(angle), cy + r * Math.sin(angle)];
  });
}

function createFeatures(): Ring[][] {
  const random = createRandom(5);
  const features: Ring[][] = [];
  for (let index = 0; index < 30; index++) {
    const far = index % 3 === 0 ? 4e5 : 0;
    const cx = far + random() * 500;
    const cy = far + random() * 500;
    const radius = 50 + random() * 300;
    const rings = [createStar(random, cx, cy, radius, 6 + (index % 17), 0.5, index % 4 === 1)];
    if (index % 5 === 2) {
      // Hole around the center: the pole must move off the centroid.
      rings.push(createStar(random, cx, cy, radius * 0.3, 8, 0.1, index % 4 !== 1));
    }
    features.push(rings);
  }
  // C shape, L shape, thin diagonal strip, two-part multipolygon, tiny triangle.
  features.push([
    [
      [0, 0],
      [100, 0],
      [100, 20],
      [20, 20],
      [20, 80],
      [100, 80],
      [100, 100],
      [0, 100]
    ]
  ]);
  features.push([
    [
      [0, 0],
      [60, 0],
      [60, 20],
      [20, 20],
      [20, 100],
      [0, 100]
    ]
  ]);
  features.push([
    [
      [0, 0],
      [3, 0],
      [103, 100],
      [100, 100]
    ]
  ]);
  features.push([
    [
      [0, 0],
      [10, 0],
      [10, 10],
      [0, 10]
    ],
    [
      [100, 0],
      [140, 0],
      [140, 40],
      [100, 40]
    ]
  ]);
  features.push([
    [
      [0, 0],
      [0.01, 0],
      [0.005, 0.01]
    ]
  ]);
  return features;
}

it('GPULabelPoint lands inside and near the polylabel pole of every polygon', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const features = createFeatures();
  const flat = flattenFeatures(features);
  const rounded: Ring[][] = features.map(rings =>
    rings.map(ring => ring.map(([x, y]) => [Math.fround(x), Math.fround(y)]))
  );
  const featureCount = features.length;
  const fixture = createGeometryFixture(device, {
    inputs: {
      positions: {values: flat.positions, format: 'float32x2'},
      ringOffsets: {values: flat.ringOffsets, format: 'uint32'},
      featureRingOffsets: {values: flat.featureRingOffsets, format: 'uint32'}
    },
    outputs: {
      points: {format: 'float32x2', length: featureCount},
      distances: {format: 'float32', length: featureCount},
      degenerate: {format: 'uint32', length: featureCount}
    },
    create: ({inputs, outputs}) =>
      new GPULabelPoint({
        positions: inputs['positions'] as never,
        ringOffsets: inputs['ringOffsets'] as never,
        featureRingOffsets: inputs['featureRingOffsets'] as never,
        output: {
          points: outputs['points'] as never,
          distances: outputs['distances'] as never,
          degenerate: outputs['degenerate'] as never
        }
      })
  });
  const result = await fixture.run();
  expect(Math.max(...result['distances'])).toBeGreaterThan(10);
  let worstRatio = 1;
  for (let feature = 0; feature < featureCount; feature++) {
    const rings = rounded[feature];
    const x = result['points'][2 * feature];
    const y = result['points'][2 * feature + 1];
    const xs = rings.flat().map(p => p[0]);
    const ys = rings.flat().map(p => p[1]);
    const size = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys));
    const truth = findPoleOfInaccessibility(rings, 1e-4 * size);
    const actualDistance = getSignedDistance(x, y, rings);
    // Strictly inside (and outside holes) for every non-degenerate polygon.
    expect(actualDistance, `feature ${feature} inside`).toBeGreaterThan(0);
    expect(result['degenerate'][feature]).toBe(0);
    // Reported distance matches the f64 distance at the reported point.
    expect(Math.abs(result['distances'][feature] - actualDistance)).toBeLessThanOrEqual(
      2e-3 * Math.max(truth.distance, 1e-9) + 1e-3 * Math.abs(x) * 1e-3
    );
    const ratio = actualDistance / truth.distance;
    worstRatio = Math.min(worstRatio, ratio);
    expect(ratio, `feature ${feature} ratio`).toBeGreaterThan(0.97);
  }
  // Shapely 2.1.2 polylabel (tolerance 0.001) radius for the C and L shapes: 11.7157 and 11.7154.
  expect(result['distances'][featureCount - 5]).toBeGreaterThan(11.7157 * 0.995);
  expect(result['distances'][featureCount - 5]).toBeLessThan(11.7157 * 1.001);
  expect(result['distances'][featureCount - 4]).toBeGreaterThan(11.7154 * 0.995);
  expect(result['distances'][featureCount - 4]).toBeLessThan(11.7154 * 1.001);
  console.log(`GPULabelPoint worst distance ratio to polylabel ${worstRatio.toFixed(4)}`);
  fixture.destroy();
});

it('GPULabelPoint flags degenerate polygons and reports empty features as NaN', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const features: Ring[][] = [
    [
      [
        [0, 0],
        [4, 0],
        [8, 0]
      ]
    ],
    [[]],
    [
      [
        [0, 0],
        [4, 0],
        [4, 4],
        [0, 4]
      ]
    ]
  ];
  const flat = flattenFeatures(features);
  const fixture = createGeometryFixture(device, {
    inputs: {
      positions: {values: flat.positions, format: 'float32x2'},
      ringOffsets: {values: flat.ringOffsets, format: 'uint32'}
    },
    outputs: {
      points: {format: 'float32x2', length: 3},
      distances: {format: 'float32', length: 3},
      degenerate: {format: 'uint32', length: 3}
    },
    create: ({inputs, outputs}) =>
      new GPULabelPoint({
        positions: inputs['positions'] as never,
        ringOffsets: inputs['ringOffsets'] as never,
        output: {
          points: outputs['points'] as never,
          distances: outputs['distances'] as never,
          degenerate: outputs['degenerate'] as never
        }
      })
  });
  const result = await fixture.run();
  expect(result['degenerate']).toEqual([1, 1, 0]);
  expect(result['distances'][1]).toBeNaN();
  expect(result['points'][2]).toBeNaN();
  // Unit square (4 x 4): the pole is the center with radius 2.
  expect(result['points'][4]).toBeCloseTo(2, 3);
  expect(result['points'][5]).toBeCloseTo(2, 3);
  expect(result['distances'][2]).toBeCloseTo(2, 3);
  fixture.destroy();
});

function createWavyStar(count: number): Ring {
  return Array.from({length: count}, (_, index) => {
    const angle = (2 * Math.PI * index) / count;
    const radius = 800 * (1 + 0.3 * Math.sin(7 * angle) + 0.1 * Math.sin(23 * angle));
    return [radius * Math.cos(angle), radius * Math.sin(angle)];
  });
}

async function runLabelPoint(
  device: NonNullable<Awaited<ReturnType<typeof getWebGPUTestDevice>>>,
  features: Ring[][],
  smallFeatureVertexLimit?: number
): Promise<Record<string, number[]>> {
  const flat = flattenFeatures(features);
  const featureCount = features.length;
  const fixture = createGeometryFixture(device, {
    inputs: {
      positions: {values: flat.positions, format: 'float32x2'},
      ringOffsets: {values: flat.ringOffsets, format: 'uint32'},
      featureRingOffsets: {values: flat.featureRingOffsets, format: 'uint32'}
    },
    outputs: {
      points: {format: 'float32x2', length: featureCount},
      distances: {format: 'float32', length: featureCount},
      degenerate: {format: 'uint32', length: featureCount}
    },
    create: ({inputs, outputs}) =>
      new GPULabelPoint({
        positions: inputs['positions'] as never,
        ringOffsets: inputs['ringOffsets'] as never,
        featureRingOffsets: inputs['featureRingOffsets'] as never,
        smallFeatureVertexLimit,
        output: {
          points: outputs['points'] as never,
          distances: outputs['distances'] as never,
          degenerate: outputs['degenerate'] as never
        }
      })
  });
  const result = await fixture.run();
  fixture.destroy();
  return result as unknown as Record<string, number[]>;
}

it('GPULabelPoint matches the polylabel oracle for a 20k-vertex polygon', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const star = createWavyStar(20000).map(([x, y]) => [Math.fround(x), Math.fround(y)]);
  // The 20k-vertex star takes the workgroup path by default; a small neighbor takes the thread path.
  const result = await runLabelPoint(device, [
    [star],
    [createStar(createRandom(9), 0, 0, 100, 12, 0.3, false)]
  ]);
  const x = result['points'][0];
  const y = result['points'][1];
  const truth = findPoleOfInaccessibility([star], 1e-4 * 2 * 1300);
  const actualDistance = getSignedDistance(x, y, [star]);
  expect(actualDistance).toBeGreaterThan(0);
  expect(result['degenerate'][0]).toBe(0);
  expect(Math.abs(result['distances'][0] - actualDistance)).toBeLessThanOrEqual(
    2e-3 * truth.distance
  );
  expect(actualDistance / truth.distance).toBeGreaterThan(0.97);
});

it('GPULabelPoint thread and workgroup paths give identical results', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const features = createFeatures();
  const threadResult = await runLabelPoint(device, features, 1 << 30);
  const workgroupResult = await runLabelPoint(device, features, 0);
  expect(Array.from(workgroupResult['points'])).toEqual(Array.from(threadResult['points']));
  expect(Array.from(workgroupResult['distances'])).toEqual(Array.from(threadResult['distances']));
  expect(Array.from(workgroupResult['degenerate'])).toEqual(Array.from(threadResult['degenerate']));
});
