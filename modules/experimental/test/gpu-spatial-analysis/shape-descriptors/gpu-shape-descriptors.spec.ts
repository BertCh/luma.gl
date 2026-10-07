// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {
  getGPUShapeDescriptorsParameterValues,
  GPUShapeDescriptors
} from '../../../src/gpu-spatial-analysis/shape-descriptors';
import {
  createGeometryFixture,
  createRandom,
  expectClose,
  flattenFeatures
} from '../outline-geometry/geometry-fixture';
import {describeFeature, type Ring} from './shape-descriptors-oracle';

function createStar(
  random: () => number,
  centerX: number,
  centerY: number,
  radius: number,
  vertexCount: number,
  jitter: number,
  clockwise: boolean
): Ring {
  const ring: Ring = [];
  for (let index = 0; index < vertexCount; index++) {
    const angle = ((clockwise ? -1 : 1) * 2 * Math.PI * index) / vertexCount;
    const r = radius * (1 - jitter + jitter * random());
    ring.push([centerX + r * Math.cos(angle), centerY + r * Math.sin(angle)]);
  }
  return ring;
}

function createFeatures(): Ring[][] {
  const random = createRandom(21);
  const features: Ring[][] = [];
  for (let index = 0; index < 40; index++) {
    // Mix of near-origin and projected-meter coordinates.
    const far = index % 2 === 0 ? 5e5 : 0;
    const centerX = far + random() * 1000;
    const centerY = far * 3 + random() * 1000;
    const radius = 20 + random() * 200;
    const rings: Ring[] = [
      createStar(random, centerX, centerY, radius, 5 + (index % 11), 0.6, index % 7 === 3)
    ];
    if (index % 5 === 1) {
      rings.push(createStar(random, centerX, centerY, radius * 0.15, 6, 0.2, index % 7 !== 3));
    }
    features.push(rings);
  }
  // One very large feature among small ones: the skew the monotone chain hull exists for.
  features.splice(7, 0, [createStar(random, 2000, 2000, 900, 2500, 0.35, false)]);
  // Analytic shapes: a 10 x 2 rectangle rotated by 30 degrees (elongation 0.8, orientation 30 deg),
  // a unit square, and a thin sliver.
  const rotate = (x: number, y: number, angle: number) => [
    x * Math.cos(angle) - y * Math.sin(angle),
    x * Math.sin(angle) + y * Math.cos(angle)
  ];
  const theta = Math.PI / 6;
  features.push([
    [
      [-5, -1],
      [5, -1],
      [5, 1],
      [-5, 1]
    ].map(([x, y]) => rotate(x, y, theta))
  ]);
  features.push([
    [
      [0, 0],
      [1, 0],
      [1, 1],
      [0, 1]
    ]
  ]);
  features.push([
    [
      [0, 0],
      [100, 0],
      [100, 0.5],
      [0, 0.5]
    ]
  ]);
  return features;
}

const CASES = [
  ['winding', 'gift-wrapping'],
  ['first-ring-exterior', 'monotone-chain'],
  ['winding', 'monotone-chain']
] as const;
for (const [holeRule, convexityMethod] of CASES) {
  it(`GPUShapeDescriptors matches the f64 oracle with holeRule ${holeRule} and ${convexityMethod} convexity`, async () => {
    const device = await getWebGPUTestDevice();
    if (!device) {
      return;
    }
    const features = createFeatures();
    // Under winding, a hole must wind opposite to its exterior; the oracle and kernel agree on it.
    const flat = flattenFeatures(features);
    // Compare against what the GPU sees: f32-rounded positions.
    const rounded: Ring[][] = features.map(rings =>
      rings.map(ring => ring.map(([x, y]) => [Math.fround(x), Math.fround(y)]))
    );
    const featureCount = features.length;
    const names = [
      'areas',
      'perimeters',
      'polsbyPopper',
      'schwartzberg',
      'elongation',
      'orientation',
      'convexity'
    ];
    const fixture = createGeometryFixture(device, {
      inputs: {
        positions: {values: flat.positions, format: 'float32x2'},
        ringOffsets: {values: flat.ringOffsets, format: 'uint32'},
        featureRingOffsets: {values: flat.featureRingOffsets, format: 'uint32'}
      },
      outputs: {
        ...Object.fromEntries(
          names.map(name => [name, {format: 'float32' as const, length: featureCount}])
        ),
        clockwise: {format: 'uint32', length: featureCount},
        sliver: {format: 'uint32', length: featureCount}
      },
      create: ({inputs, outputs, parameters}) =>
        new GPUShapeDescriptors({
          positions: inputs['positions'] as never,
          ringOffsets: inputs['ringOffsets'] as never,
          featureRingOffsets: inputs['featureRingOffsets'] as never,
          holeRule,
          convexityMethod,
          parameters,
          output: {
            areas: outputs['areas'] as never,
            perimeters: outputs['perimeters'] as never,
            polsbyPopper: outputs['polsbyPopper'] as never,
            schwartzberg: outputs['schwartzberg'] as never,
            elongation: outputs['elongation'] as never,
            orientation: outputs['orientation'] as never,
            convexity: outputs['convexity'] as never,
            clockwise: outputs['clockwise'] as never,
            sliver: outputs['sliver'] as never
          }
        })
    });
    let maximumRelativeError = 0;
    for (const sliverThreshold of [0.05, 0.4]) {
      const result = await fixture.run(getGPUShapeDescriptorsParameterValues({sliverThreshold}));
      expect(Math.max(...result['areas'])).toBeGreaterThan(1);
      let slivers = 0;
      for (let feature = 0; feature < featureCount; feature++) {
        const expected = describeFeature(rounded[feature], holeRule, sliverThreshold);
        const label = `feature ${feature}`;
        // First-ring-exterior hole rule means a CW exterior still counts positive.
        for (const [name, key, tolerance] of [
          ['areas', 'area', 3e-4],
          ['perimeters', 'perimeter', 1e-4],
          ['polsbyPopper', 'polsbyPopper', 5e-4],
          ['schwartzberg', 'schwartzberg', 5e-4],
          ['elongation', 'elongation', 1e-3],
          ['convexity', 'convexity', 1e-3]
        ] as const) {
          const actual = result[name][feature];
          expectClose(actual, expected[key], tolerance, 1e-5, `${label} ${name}`);
          maximumRelativeError = Math.max(
            maximumRelativeError,
            Math.abs(actual - expected[key]) / Math.max(Math.abs(expected[key]), 1e-9)
          );
        }
        // Orientation is periodic and meaningless for near-circles: compare when elongated.
        if (expected.elongation > 0.05) {
          const difference = Math.abs(result['orientation'][feature] - expected.orientation);
          expect(Math.min(difference, Math.PI - difference), `${label} orientation`).toBeLessThan(
            2e-3
          );
        }
        if (Math.abs(expected.polsbyPopper - sliverThreshold) > 2e-3) {
          expect(result['sliver'][feature], `${label} sliver`).toBe(expected.sliver);
        }
        slivers += result['sliver'][feature];
        expect(result['clockwise'][feature], `${label} clockwise`).toBe(expected.clockwise);
      }
      expect(slivers).toBeGreaterThan(0);
    }
    console.log(`GPUShapeDescriptors max relative error ${maximumRelativeError.toExponential(2)}`);
    // Analytic checks on the last three features.
    const result = await fixture.run(getGPUShapeDescriptorsParameterValues({}));
    const rectangle = featureCount - 3;
    expectClose(result['elongation'][rectangle], 0.8, 1e-3, 1e-3, 'rectangle elongation');
    expectClose(result['orientation'][rectangle], Math.PI / 6, 1e-3, 1e-3, 'rectangle orientation');
    expectClose(result['convexity'][rectangle], 1, 1e-4, 1e-4, 'rectangle convexity');
    expectClose(result['polsbyPopper'][rectangle + 1], Math.PI / 4, 1e-4, 1e-5, 'square PP');
    expectClose(result['elongation'][rectangle + 1], 0, 0, 1e-3, 'square elongation');
    expect(result['sliver'][rectangle + 2]).toBe(1);
    expect(result['sliver'][rectangle + 1]).toBe(0);
    expect(fixture.getCompileCount()).toBe(0);
    fixture.destroy();
  });
}
