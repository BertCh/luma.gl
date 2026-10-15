// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {polygonCases, decodePackedNumbers} from '../../oracles/parity-wp4-round3-fixtures';
import {compareRasterParity} from '../../oracles/parity-wp4-compare';
import centreAmbiguities from '../../oracles/fixtures/parity-wp4-polygon-centres.json';
import {submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {
  createRasterizationFixture,
  destroyRasterizationFixture,
  readRasterization
} from './polygon-rasterization-fixture';

for (const scene of polygonCases) {
  for (const insufficient of [false, true]) {
    it(`GPUPolygonRasterization rasterio parity: ${scene.name}, overflow=${insufficient}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const [zones, minimum, maximum] = await Promise.all([
        decodePackedNumbers(scene.zones),
        decodePackedNumbers(scene.boundaryMinimum),
        decodePackedNumbers(scene.boundaryMaximum)
      ]);
      const fixture = createRasterizationFixture(device, {
        width: scene.width,
        height: scene.height,
        extent: [0, 0, 1, 1],
        polygons: {
          positions: new Float32Array(scene.positions),
          featureOffsets: new Uint32Array(scene.featureOffsets),
          polygonOffsets: new Uint32Array(scene.polygonOffsets),
          ringOffsets: new Uint32Array(scene.ringOffsets)
        },
        crossingCapacity: insufficient ? 2 : scene.crossingCount
      });
      const compiled = fixture.graph.compile();
      const compare = (metric: string, actual: ArrayLike<number>, expected: ArrayLike<number>) =>
        compareRasterParity(
          'GPUPolygonRasterization',
          `${scene.name}/overflow=${insufficient}`,
          scene.reference,
          metric,
          actual,
          expected,
          'E'
        );
      try {
        submitGraph(device, compiled, undefined);
        const actual = await readRasterization(fixture);
        const ambiguous = new Map(
          scene.name === centreAmbiguities.case && !insufficient
            ? centreAmbiguities.cells.map(cell => [cell.index, cell.features] as const)
            : []
        );
        // Exact rational crossing at pixel 76206 is 291/466714624 cells right of
        // its centre: less than half an f32 ULP (1/65536 at x=430.5). Even a
        // correctly rounded crossing lands ON the centre. Exclude this enumerated
        // shared-edge cell from strict ownership; require one of its two tracts.
        // Every other pixel retains E equality, including all overflow outputs.
        compare(
          'all-zones-center-inclusion-and-overlap-precedence',
          actual.zones.filter((_, index) => !ambiguous.has(index)),
          (insufficient ? new Uint32Array(zones.length).fill(0xffffffff) : zones).filter(
            (_, index) => !ambiguous.has(index)
          )
        );
        compare(
          'enumerated-shared-edge-centres-belong-to-adjacent-tracts',
          [...ambiguous].map(([index, features]) => Number(features.includes(actual.zones[index]))),
          new Uint32Array(ambiguous.size).fill(1)
        );
        compare(
          'crossing-count-and-overflow',
          [actual.crossingCount, actual.overflow],
          [scene.crossingCount, Number(insufficient)]
        );
        // All non-ambiguous cells require equality. Only the fixture's declared padding band
        // allows either flag; every exact edge intersection must still be marked.
        const admissible = actual.boundary.map((flag, index) =>
          Number(flag >= minimum[index] && flag <= maximum[index])
        );
        compare(
          'every-conservative-boundary-flag-within-shapely-bounds',
          admissible,
          new Uint32Array(zones.length).fill(1)
        );
      } finally {
        compiled.destroy();
        destroyRasterizationFixture(fixture);
      }
    });
  }
}
