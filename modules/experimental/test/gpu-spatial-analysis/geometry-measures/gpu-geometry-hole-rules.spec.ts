// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUGeometryMeasures} from '../../../src/gpu-spatial-analysis/geometry-measures';
import {
  getGPUShapeDescriptorsParameterValues,
  GPUShapeDescriptors
} from '../../../src/gpu-spatial-analysis/shape-descriptors';
import {createGeometryFixture, expectClose} from '../outline-geometry/geometry-fixture';
import holeRulesJson from './fixtures/hole-rules.json';
import {requireTestDevice} from '../../utils/require-test-device';

// Oracle: fixtures/hole-rules.json, written by test/oracles/geometry-hole-rules.py (shapely area,
// perimeter and centroid of multi-polygons and holed polygons built with deliberately mixed
// winding). Review D-13: ring roles come from the GeoArrow four-level layout by position.
type HoleRuleFixture = {
  positions: number[][];
  ringOffsets: number[];
  polygonOffsets: number[];
  featureOffsets: number[];
  expected: {
    name: string;
    area: number;
    perimeter: number;
    centroid: [number, number];
    polsbyPopper: number;
  }[];
  provenance: {tolerance: {relative: number}};
};

const FIXTURE = holeRulesJson as unknown as HoleRuleFixture;
const TOLERANCE = FIXTURE.provenance.tolerance.relative;

function getInputs() {
  return {
    positions: {values: new Float32Array(FIXTURE.positions.flat()), format: 'float32x2' as const},
    ringOffsets: {values: new Uint32Array(FIXTURE.ringOffsets), format: 'uint32' as const},
    polygonOffsets: {values: new Uint32Array(FIXTURE.polygonOffsets), format: 'uint32' as const},
    featureOffsets: {values: new Uint32Array(FIXTURE.featureOffsets), format: 'uint32' as const}
  };
}

it('GPUGeometryMeasures matches shapely for mixed winding and holes', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const featureCount = FIXTURE.expected.length;
  const fixture = createGeometryFixture(device, {
    inputs: getInputs(),
    outputs: {
      areas: {format: 'float32', length: featureCount},
      lengths: {format: 'float32', length: featureCount},
      centroids: {format: 'float32x2', length: featureCount}
    },
    create: ({inputs, outputs}) =>
      new GPUGeometryMeasures({
        spatialContext: {coordinateSpace: 'planar', metric: 'native', units: 'native'},
        positions: inputs['positions'] as never,
        geometryType: 'polygons',
        ringOffsets: inputs['ringOffsets'] as never,
        polygonOffsets: inputs['polygonOffsets'] as never,
        featureOffsets: inputs['featureOffsets'] as never,
        output: {
          areas: outputs['areas'] as never,
          lengths: outputs['lengths'] as never,
          centroids: outputs['centroids'] as never
        }
      })
  });
  const result = await fixture.run();
  FIXTURE.expected.forEach((expected, feature) => {
    expectClose(result['areas'][feature], expected.area, TOLERANCE, 1e-6, `${expected.name} area`);
    expectClose(
      result['lengths'][feature],
      expected.perimeter,
      TOLERANCE,
      1e-6,
      `${expected.name} perimeter`
    );
    expectClose(
      result['centroids'][2 * feature],
      expected.centroid[0],
      TOLERANCE,
      1e-5,
      `${expected.name} centroid x`
    );
    expectClose(
      result['centroids'][2 * feature + 1],
      expected.centroid[1],
      TOLERANCE,
      1e-5,
      `${expected.name} centroid y`
    );
  });
  fixture.destroy();
});

it('GPUShapeDescriptors matches shapely area and Polsby-Popper for mixed winding and holes', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const featureCount = FIXTURE.expected.length;
  const fixture = createGeometryFixture(device, {
    inputs: getInputs(),
    outputs: {
      areas: {format: 'float32', length: featureCount},
      polsbyPopper: {format: 'float32', length: featureCount},
      elongation: {format: 'float32', length: featureCount}
    },
    create: ({inputs, outputs, parameters}) =>
      new GPUShapeDescriptors({
        positions: inputs['positions'] as never,
        ringOffsets: inputs['ringOffsets'] as never,
        polygonOffsets: inputs['polygonOffsets'] as never,
        featureOffsets: inputs['featureOffsets'] as never,
        parameters,
        output: {
          areas: outputs['areas'] as never,
          polsbyPopper: outputs['polsbyPopper'] as never,
          elongation: outputs['elongation'] as never
        }
      })
  });
  const result = await fixture.run(getGPUShapeDescriptorsParameterValues({}));
  FIXTURE.expected.forEach((expected, feature) => {
    expectClose(result['areas'][feature], expected.area, TOLERANCE, 1e-6, `${expected.name} area`);
    expectClose(
      result['polsbyPopper'][feature],
      expected.polsbyPopper,
      TOLERANCE,
      1e-6,
      `${expected.name} Polsby-Popper`
    );
    // Moments use the same roles: finite and in [0, 1] means the covered region was integrated
    // (a wrongly signed hole drives the second moments negative or NaN).
    expect(result['elongation'][feature], `${expected.name} elongation`).toBeGreaterThanOrEqual(0);
    expect(result['elongation'][feature], `${expected.name} elongation`).toBeLessThanOrEqual(1);
  });
  fixture.destroy();
});
