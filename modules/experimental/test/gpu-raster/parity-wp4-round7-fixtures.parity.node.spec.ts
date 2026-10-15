// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {expect, it} from 'vitest';
import {getGPUWeightedOverlayParameterValues} from '../../src/gpu-raster/raster-algebra';
import {
  roundSevenFixtures,
  sieveCases,
  sieveMetrics,
  statisticsCases,
  statisticsMetrics,
  overlayCases,
  overlayMetrics
} from '../oracles/parity-wp4-round7-fixtures';
import {decodePackedNumbers} from '../oracles/parity-wp4-round3-fixtures';
import {compareRasterParity} from '../oracles/parity-wp4-compare';
import {compareRoundFour} from '../oracles/parity-wp4-round4-compare';
import {computeSieveOracle} from './raster-patches/raster-patches-oracle';
import {weightedOverlayOnCPU} from './raster-algebra/raster-algebra-oracle';

it('WP4 final fixtures preserve versions, source hashes and complete declared scales', async () => {
  for (const fixture of roundSevenFixtures) {
    expect(Buffer.byteLength(JSON.stringify(fixture))).toBeLessThan(1_500_000);
    for (const version of Object.values(fixture.versions)) expect(version).toBeTruthy();
    for (const source of fixture.sources)
      expect(createHash('sha256').update(readFileSync(source.path)).digest('hex')).toBe(
        source.sha256
      );
  }
  expect(sieveCases.slice(0, 2).every(scene => scene.width === 750 && scene.height === 750)).toBe(
    true
  );
  expect(statisticsCases[0].width * statisticsCases[0].height).toBe(512 * 512);
  expect(overlayCases.slice(0, 8).reduce((total, scene) => total + scene.cellCount, 0)).toBe(
    512 * 512
  );
  for (const scene of overlayCases)
    expect((await decodePackedNumbers(scene.stack)).length).toBe(
      scene.cellCount * scene.layerCount
    );
});

for (const scene of sieveCases) {
  it(`Sieve CPU diagnostic versus independent reference (not GPU evidence): ${scene.name}`, async () => {
    const labels = new Uint32Array(await decodePackedNumbers(scene.labels));
    expect(labels.length).toBe(scene.width * scene.height);
    for (const [frameIndex, frame] of scene.frames.entries()) {
      const result = computeSieveOracle(
        labels,
        scene.width,
        scene.height,
        scene.capacity,
        frame.minimumPixels,
        scene.mode,
        scene.connectivity
      );
      for (const metric of sieveMetrics)
        compareRasterParity(
          'GPURasterSieve',
          `CPU diagnostic/${scene.name}/frame-${frameIndex}`,
          scene.reference,
          metric,
          metric === 'sievedCount' ? [result.sievedCount] : result[metric],
          await decodePackedNumbers(frame.expected[metric]),
          'E'
        );
    }
  });
}

for (const scene of statisticsCases) {
  it(`Statistics fixture arithmetic diagnostic (not GPU evidence): ${scene.name}`, async () => {
    const values = await decodePackedNumbers(scene.values);
    const validity = await decodePackedNumbers(scene.validity);
    expect(values.length).toBe(scene.width * scene.height);
    for (const [frameIndex, frame] of scene.frames.entries()) {
      const selected = values
        .filter(
          (value, index) =>
            !frame.masked && validity[index] !== 0 && Number.isFinite(value) && value !== -9999
        )
        .map(value => value * scene.scale + scene.offset);
      let minimum = Infinity;
      let maximum = -Infinity;
      let sum = 0;
      for (const value of selected) {
        sum += value;
        minimum = Math.min(minimum, value);
        maximum = Math.max(maximum, value);
      }
      const actual = {
        count: [selected.length],
        sum: [sum],
        mean: [selected.length ? sum / selected.length : 0],
        extent: selected.length ? [minimum, maximum] : [0, 0]
      };
      for (const metric of statisticsMetrics)
        compareRasterParity(
          'GPURasterStatistics',
          `CPU diagnostic/${scene.name}/frame-${frameIndex}`,
          scene.reference,
          metric,
          actual[metric],
          frame.expected[metric],
          metric === 'count' ? 'E' : 'S'
        );
    }
  });
}

for (const scene of overlayCases) {
  it(`Overlay CPU diagnostic versus independent NumPy (not GPU evidence): ${scene.name}`, async () => {
    const [stack, remapBreaks, remapValues] = await Promise.all(
      [scene.stack, scene.remapBreaks, scene.remapValues].map(decodePackedNumbers)
    );
    for (const [frameIndex, frame] of scene.frames.entries()) {
      const result = weightedOverlayOnCPU(
        {
          stack: new Float32Array(stack),
          layerCount: scene.layerCount,
          cellCount: scene.cellCount,
          noDataValue: -9999,
          remapBreaks: new Float32Array(remapBreaks),
          remapValues: new Float32Array(remapValues),
          maximumBreakCount: 4
        },
        getGPUWeightedOverlayParameterValues(frame.settings)
      );
      for (const metric of overlayMetrics)
        compareRoundFour(
          'GPUWeightedOverlay',
          `CPU diagnostic/${scene.name}/frame-${frameIndex}`,
          scene.reference,
          metric,
          metric === 'outputValidity' ? result.validity : result[metric],
          await decodePackedNumbers(frame.expected[metric]),
          metric === 'outputValidity' ? 'E' : 'R'
        );
    }
  });
}
