// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {expect, it} from 'vitest';
import {makeGPURasterMetadata} from '../../src/gpu-raster';
import {getGPUIsobandsParameterValues} from '../../src/gpu-raster/isolines';
import {
  roundFiveFixtures,
  lineCases,
  ringCases,
  statisticsCases,
  allStatistics
} from '../oracles/parity-wp4-round5-fixtures';
import {decodePackedNumbers} from '../oracles/parity-wp4-round3-fixtures';
import {compareRoundFour} from '../oracles/parity-wp4-round4-compare';
import {
  compareLineSegments,
  compareContourRings,
  ringSymmetricDifference
} from '../oracles/parity-wp4-contour-compare';
import {computeIsolinesOnCPU} from './isolines/isolines-oracle';
import {buildIsobandTrianglesOnCPU} from './isolines/isobands-oracle';
import {computeCellStatisticsOnCPU} from './raster-algebra/raster-algebra-oracle';
import {
  assembleRingsOnCPU,
  type OracleSegment
} from '../gpu-spatial-analysis/ring-assembly/ring-assembly-oracle';

it('WP4 round5 independent fixtures preserve versions, hashes, size limits and complete real tile', () => {
  for (const fixture of roundFiveFixtures) {
    expect(fixture.versions.python).toBeTruthy();
    expect(fixture.versions.numpy).toBeTruthy();
    expect(Buffer.byteLength(JSON.stringify(fixture))).toBeLessThan(1_500_000);
    for (const source of fixture.sources)
      expect(createHash('sha256').update(readFileSync(source.path)).digest('hex')).toBe(
        source.sha256
      );
  }
  expect(statisticsCases.slice(0, 8).reduce((sum, scene) => sum + scene.cellCount, 0)).toBe(
    512 * 512
  );
  const nested = ringCases.find(scene => scene.name.startsWith('nested-'))!;
  expect(nested.rings.filter(ring => ring.isHole).length).toBeGreaterThan(1);
  for (const scene of ringCases)
    for (const ring of scene.rings) {
      expect(scene.rings[ring.shell].isHole).toBe(0);
      expect(scene.rings[ring.shell].band).toBe(ring.band);
    }
});

it('symmetric-difference measurement detects equal-area translated shapes and concavities', () => {
  const square = [0, 0, 2, 0, 2, 2, 0, 2, 0, 0];
  const translated = [1, 0, 3, 0, 3, 2, 1, 2, 1, 0];
  const triangle = [0, 0, 2, 0, 0, 2, 0, 0];
  const concave = [0, 0, 2, 0, 2, 1, 1, 1, 1, 2, 0, 2, 0, 0];
  expect(ringSymmetricDifference(square, translated)).toBeCloseTo(4, 12);
  expect(ringSymmetricDifference(square, triangle)).toBeCloseTo(2, 12);
  expect(ringSymmetricDifference(square, concave)).toBeCloseTo(1, 12);
  expect(ringSymmetricDifference(triangle, [0, 0, 2, 0, 2, 2, 0, 0])).toBeCloseTo(2, 12);
  expect(ringSymmetricDifference(concave, concave)).toBe(0);
});

for (const scene of lineCases) {
  it(`Isolines CPU diagnostic versus skimage (not GPU evidence): ${scene.name}`, async () => {
    const [values, validity, expected] = await Promise.all([
      decodePackedNumbers(scene.values),
      decodePackedNumbers(scene.validity),
      decodePackedNumbers(scene.segments)
    ]);
    const segments = computeIsolinesOnCPU(
      {
        width: scene.width,
        height: scene.height,
        values: new Float32Array(values),
        validity: new Uint32Array(validity),
        noDataValue: -9999,
        levels: scene.levels,
        extent: [0, 0, scene.width, scene.height]
      },
      scene.levels.length
    );
    compareLineSegments(
      `CPU diagnostic/${scene.name}`,
      scene.reference,
      scene.width,
      scene.height,
      segments.flatMap(segment => [...segment.p0, ...segment.p1]),
      segments.map(segment => segment.level),
      segments.flatMap(segment => [segment.startEdge, segment.endEdge]),
      expected
    );
  });
}

for (const scene of ringCases) {
  it(`Ring CPU diagnostic versus Shapely union (not GPU evidence): ${scene.name}`, async () => {
    const [values, validity] = await Promise.all([
      decodePackedNumbers(scene.values),
      decodePackedNumbers(scene.validity)
    ]);
    const grid = makeGPURasterMetadata(scene);
    const triangles = buildIsobandTrianglesOnCPU(
      {
        width: scene.width,
        height: scene.height,
        values: new Float32Array(values),
        validity: new Uint32Array(validity),
        noDataValue: -9999,
        breaks: new Float32Array(scene.breaks)
      },
      getGPUIsobandsParameterValues(
        {
          breakCount: scene.breaks.length,
          extent: [0, 0, scene.width, scene.height],
          firstBand: scene.firstBand,
          lastBand: scene.lastBand
        },
        grid
      ),
      scene.breaks.length
    );
    // This local CPU diagnostic is never used to generate GPU expected output.
    // Cancel only exactly reversed same-band fan edges; assemble existing CPU rings.
    const boundary = new Map<string, {segment: OracleSegment; band: number}>();
    for (let triangle = 0; triangle < triangles.bands.length; triangle++) {
      const band = triangles.bands[triangle];
      for (let corner = 0; corner < 3; corner++) {
        const first = triangles.triangles.slice(
          triangle * 6 + corner * 2,
          triangle * 6 + corner * 2 + 2
        );
        const next = (corner + 1) % 3;
        const second = triangles.triangles.slice(
          triangle * 6 + next * 2,
          triangle * 6 + next * 2 + 2
        );
        if (first[0] === second[0] && first[1] === second[1]) continue;
        if (!boundary.delete(`${band}/${second}/${first}`))
          boundary.set(`${band}/${first}/${second}`, {
            segment: [first[0], first[1], second[0], second[1]],
            band
          });
      }
    }
    const boundaryEntries = [...boundary.values()];
    const assembled = assembleRingsOnCPU(
      boundaryEntries.map(entry => entry.segment),
      {
        tolerance: 1e-4,
        interiorSide: 'left',
        geographic: false,
        splitTouchingRings: true,
        groups: boundaryEntries.map(entry => entry.band)
      }
    );
    compareRoundFour(
      'GPUIsobandRings CPU diagnostic',
      scene.name,
      scene.reference,
      'complete-boundary-edges-and-open-count',
      [boundary.size, assembled.openSegments],
      [scene.rings.reduce((sum, ring) => sum + ring.positions.length / 2 - 1, 0), 0],
      'E'
    );
    compareContourRings(
      `CPU diagnostic/${scene.name}`,
      scene.reference,
      scene.width,
      scene.height,
      assembled.rings.map(ring => ({
        positions: ring.vertices.flat(),
        area: ring.area,
        band: ring.group,
        isHole: Number(ring.isHole),
        shell: ring.shell ?? -1
      })),
      scene.rings
    );
  });
}

for (const scene of statisticsCases) {
  for (const [frameIndex, frame] of scene.frames.entries()) {
    it(`Cell statistics CPU diagnostic versus NumPy (not GPU evidence): ${scene.name}/${frameIndex}`, async () => {
      const stack = new Float32Array(await decodePackedNumbers(scene.stack));
      const actual = computeCellStatisticsOnCPU(stack, scene.layerCount, scene.cellCount, {
        noDataValue: -9999,
        propagateNoData: frame.settings.noDataPolicy === 'propagate',
        minimumValidCount: frame.settings.minimumValidCount
      });
      for (const statistic of allStatistics) {
        compareRoundFour(
          'GPURasterCellStatistics CPU diagnostic',
          `${scene.name}/frame-${frameIndex}`,
          scene.reference,
          statistic,
          actual[statistic === 'counts' ? 'count' : statistic],
          await decodePackedNumbers(frame.expected[statistic]),
          ['minimum', 'maximum', 'majority', 'minority', 'counts', 'variety'].includes(statistic)
            ? 'E'
            : 'R'
        );
      }
    });
  }
}
