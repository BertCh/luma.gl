// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {expect, it} from 'vitest';
import {getGPUPolygonRasterizationParameterValues} from '../../src/gpu-raster/polygon-rasterization';
import {getGPURasterSamplingParameterValues} from '../../src/gpu-raster/raster-sampling';
import {
  roundSixFixtures,
  joinCases,
  joinMetrics,
  patchCases,
  patchMetrics,
  samplingCases
} from '../oracles/parity-wp4-round6-fixtures';
import {decodePackedNumbers} from '../oracles/parity-wp4-round3-fixtures';
import {compareRasterParity} from '../oracles/parity-wp4-compare';
import {compareRoundFour} from '../oracles/parity-wp4-round4-compare';
import {joinPointsOnCPU} from './polygon-rasterization/polygon-rasterization-oracle';
import {computePatchMetricsOracle, getAcceptedLabels} from './raster-patches/raster-patches-oracle';
import {sampleRasterPointsOnCPU} from './raster-sampling/raster-sampling-oracle';

it('WP4 round6 fixtures preserve sources, versions, full extents and exact-sum conditioning', async () => {
  for (const fixture of roundSixFixtures) {
    expect(Buffer.byteLength(JSON.stringify(fixture))).toBeLessThan(1_500_000);
    for (const version of Object.values(fixture.versions)) expect(version).toBeTruthy();
    for (const source of fixture.sources)
      expect(createHash('sha256').update(readFileSync(source.path)).digest('hex')).toBe(
        source.sha256
      );
  }
  expect(joinCases[0].pointCount).toBe(43557);
  for (const scene of joinCases) {
    const values = await decodePackedNumbers(scene.values);
    expect(values.filter(Number.isFinite).every(value => Number.isInteger(value * 4))).toBe(true);
    expect(
      values.filter(Number.isFinite).reduce((sum, value) => sum + Math.abs(value * 4), 0)
    ).toBeLessThan(2 ** 24);
    expect((await decodePackedNumbers(scene.zones)).length).toBe(
      scene.grid.width * scene.grid.height
    );
  }
  for (const scene of patchCases.slice(0, 4)) {
    expect((await decodePackedNumbers(scene.labels)).length).toBe(750 * 750);
    expect(scene.capacity).toBeGreaterThan(1900);
  }
  expect((await decodePackedNumbers(samplingCases[0].values)).length).toBe(512 * 512);
  expect(samplingCases[0].pointCount).toBe(5380);
});

for (const scene of joinCases) {
  it(`Join CPU diagnostic versus rasterio (not GPU evidence): ${scene.name}`, async () => {
    const [zones, boundary, points, values] = await Promise.all(
      [scene.zones, scene.boundary, scene.points, scene.values].map(decodePackedNumbers)
    );
    const reverseRows = (array: number[]) => {
      const result: number[] = [];
      for (let row = scene.grid.height - 1; row >= 0; row--)
        result.push(...array.slice(row * scene.grid.width, (row + 1) * scene.grid.width));
      return result;
    };
    for (const [frameIndex, frame] of scene.frames.entries()) {
      const placement = getGPUPolygonRasterizationParameterValues(frame.grid);
      const northUp = placement[3] < 0;
      const result = joinPointsOnCPU({
        raster: {
          width: scene.grid.width,
          height: scene.grid.height,
          extent: [placement[0], placement[1], placement[2], Math.abs(placement[3])]
        },
        zones: new Uint32Array(northUp ? reverseRows(zones) : zones),
        boundary: northUp ? reverseRows(boundary) : boundary,
        points: new Float32Array(points),
        values: new Float32Array(values),
        zoneCount: scene.zoneCount
      });
      for (const metric of joinMetrics) {
        const actual = result[metric];
        compareRasterParity(
          'GPURasterJoin',
          `CPU diagnostic/${scene.name}/frame-${frameIndex}`,
          scene.reference,
          metric,
          typeof actual === 'number' ? [actual] : actual,
          await decodePackedNumbers(frame.expected[metric]),
          'E'
        );
      }
    }
  });
}

for (const scene of patchCases) {
  it(`Patch CPU diagnostic versus regionprops/polygons (not GPU evidence): ${scene.name}`, async () => {
    const labels = new Uint32Array(await decodePackedNumbers(scene.labels));
    const validity = new Uint32Array(await decodePackedNumbers(scene.validity));
    const [stepX, shearX, , shearY, stepY] = scene.grid.affine;
    for (const [frameIndex, frame] of scene.frames.entries()) {
      // The legacy oracle has no failed-state override; those exact contract words
      // are checked in the GPU spec, not synthesized as a second CPU measurement.
      if (!frame.converged || frame.overflow) {
        for (const metric of patchMetrics) {
          const expected = await decodePackedNumbers(frame.expected[metric]);
          expect(
            expected.every(value =>
              metric === 'areas' || metric === 'perimeters'
                ? Number.isNaN(value)
                : value === 0xffffffff
            )
          ).toBe(true);
        }
        continue;
      }
      const accepted = getAcceptedLabels(labels, {
        validity,
        capacity: scene.capacity,
        componentCount: frame.componentCount
      });
      const measured = computePatchMetricsOracle(
        accepted,
        scene.grid.width,
        scene.grid.height,
        scene.capacity,
        scene.countRasterBorder
      );
      const result = {
        pixelCounts: measured.counts,
        minColumns: measured.minColumns,
        minRows: measured.minRows,
        maxColumns: measured.maxColumns,
        maxRows: measured.maxRows,
        perimeterFaceCounts: measured.rowFaces.map(
          (faces, index) => faces + measured.columnFaces[index]
        ),
        areas: measured.counts.map(count => count * Math.abs(stepX * stepY - shearX * shearY)),
        perimeters: measured.rowFaces.map(
          (faces, index) =>
            faces * Math.hypot(stepX, shearY) +
            measured.columnFaces[index] * Math.hypot(shearX, stepY)
        )
      };
      for (const metric of patchMetrics)
        compareRasterParity(
          'GPURasterPatchMetrics',
          `CPU diagnostic/${scene.name}/frame-${frameIndex}`,
          scene.reference,
          metric,
          result[metric],
          await decodePackedNumbers(frame.expected[metric]),
          metric === 'areas' || metric === 'perimeters' ? 'S' : 'E'
        );
    }
  });
}

for (const scene of samplingCases) {
  it(`Sampling CPU diagnostic versus SciPy (not GPU evidence): ${scene.name}`, async () => {
    const [values, validity, points] = await Promise.all(
      [scene.values, scene.validity, scene.points].map(decodePackedNumbers)
    );
    for (const [frameIndex, frame] of scene.frames.entries()) {
      const parameters = getGPURasterSamplingParameterValues(
        {...frame, localOrigin: scene.localOrigin},
        scene.grid
      );
      const actual = sampleRasterPointsOnCPU(
        {
          width: scene.grid.width,
          height: scene.grid.height,
          values: new Float32Array(values),
          validity: new Uint32Array(validity),
          noDataValue: -9999
        },
        parameters,
        new Float32Array(points),
        frame.activeCount
      );
      const caseName = `CPU diagnostic/${scene.name}/${frame.method}/${frame.noDataPolicy}/frame-${frameIndex}`;
      compareRoundFour(
        'GPURasterSampling',
        caseName,
        scene.reference,
        'values',
        actual,
        await decodePackedNumbers(frame.expected.values),
        frame.method === 'nearest' ? 'E' : 'R'
      );
      compareRoundFour(
        'GPURasterSampling',
        caseName,
        scene.reference,
        'outputValidity',
        Array.from(actual, value => (Number.isNaN(value) ? 0 : 1)),
        await decodePackedNumbers(frame.expected.outputValidity),
        'E'
      );
    }
  });
}
