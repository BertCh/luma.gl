// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {expect, it} from 'vitest';
import {convolveLineIntegralOnCPU} from '../../src/gpu-raster/flow-texture/line-integral-convolution-cpu';
import {getGPULineIntegralConvolutionParameterValues} from '../../src/gpu-raster/flow-texture';
import licJSON from '../oracles/fixtures/parity-wp4-lic.json';
import {
  licCases,
  componentCases,
  componentFixtures,
  decodeComponentRuns
} from '../oracles/parity-wp4-round2-fixtures';
import {compareRasterParity} from '../oracles/parity-wp4-compare';

it('WP4 round2 independent fixture provenance, versions and byte limits', () => {
  for (const fixture of [licJSON, ...componentFixtures]) {
    expect(fixture.versions.python).toBeTruthy();
    expect(fixture.versions.numpy).toBeTruthy();
    expect(fixture.versions.scipy).toBeTruthy();
    const sources = 'sources' in fixture ? fixture.sources : [fixture.source];
    for (const source of sources) {
      expect(
        createHash('sha256')
          .update(readFileSync(resolve(source.path)))
          .digest('hex')
      ).toBe(source.sha256);
    }
  }
  expect(licJSON.versions.randomgen).toBeTruthy();
  for (const name of ['lic', 'components-dixie-4', 'components-dixie-8', 'components-synthetic']) {
    expect(
      readFileSync(resolve(`modules/experimental/test/oracles/fixtures/parity-wp4-${name}.json`))
        .byteLength
    ).toBeLessThanOrEqual(1_500_000);
  }
});

for (const scene of licCases) {
  it(`LIC CPU-mirror diagnostic versus independent fixture (not GPU evidence): ${scene.name}`, () => {
    const result = convolveLineIntegralOnCPU(
      {
        width: scene.fieldWidth,
        height: scene.fieldHeight,
        velocities: new Float32Array(scene.field)
      },
      scene.width,
      scene.height,
      scene.stepCount,
      getGPULineIntegralConvolutionParameterValues(scene.settings),
      scene.settings.seed!
    );
    compareRasterParity(
      'GPULineIntegralConvolution',
      `${scene.name}/CPU-diagnostic`,
      scene.reference,
      'all-noise-versus-randomgen',
      result.noise,
      // The independent top-24-bit reference is stored and compared as exact f32.
      new Float32Array(scene.noise),
      'E'
    );
    compareRasterParity(
      'GPULineIntegralConvolution',
      `${scene.name}/CPU-diagnostic`,
      scene.reference,
      'all-lic-values',
      result.values,
      scene.values,
      'L'
    );
    compareRasterParity(
      'GPULineIntegralConvolution',
      `${scene.name}/CPU-diagnostic`,
      scene.reference,
      'all-centre-speeds',
      result.speeds,
      scene.speeds,
      'L'
    );
  });
}

it('component fixtures preserve full arrays, canonical roots, dense rank and global seams', () => {
  for (const scene of componentCases) {
    const length = scene.width * scene.height;
    const values = decodeComponentRuns(scene.values, length);
    const validity = decodeComponentRuns(scene.validity, length);
    const expectedValidity = decodeComponentRuns(scene.expectedValidity, length);
    const sparse = decodeComponentRuns(scene.sparse, length);
    const dense = decodeComponentRuns(scene.dense, length);
    const roots = [...new Set(sparse)]
      .filter(value => value !== 0)
      .sort((left, right) => left - right);
    expect(roots.length).toBe(scene.componentCount);
    const rank = new Map(roots.map((root, index) => [root, index + 1]));
    let foregroundCount = 0;
    let seamConnections = 0;
    for (let index = 0; index < length; index++) {
      const valid = Number(validity[index] !== 0 && values[index] !== scene.noDataValue);
      if (expectedValidity[index] !== valid) throw new Error(`${scene.name}: validity at ${index}`);
      if (valid && values[index] !== 0) {
        foregroundCount++;
        if (
          !sparse[index] ||
          sparse[index] > index + 1 ||
          sparse[sparse[index] - 1] !== sparse[index]
        ) {
          throw new Error(`${scene.name}: minimum representative at ${index}`);
        }
        if (dense[index] !== rank.get(sparse[index]))
          throw new Error(`${scene.name}: dense rank at ${index}`);
      } else if (sparse[index] !== 0 || dense[index] !== 0) {
        throw new Error(`${scene.name}: background at ${index}`);
      }
      // These global labels cross the 512 tile boundary in both directions.
      if (
        sparse[index] &&
        ((index % scene.width === 512 && sparse[index] === sparse[index - 1]) ||
          (Math.floor(index / scene.width) === 512 &&
            sparse[index] === sparse[index - scene.width]))
      ) {
        seamConnections++;
      }
    }
    expect(foregroundCount).toBe(scene.foregroundCount);
    if (scene.name.startsWith('dixie')) {
      expect(length).toBe(562_500);
      expect(seamConnections).toBeGreaterThan(100);
    }
  }
});

it('synthetic components exercise diagonal-only joins, unequal foreground and independent nodata', () => {
  for (const connectivity of [4, 8]) {
    const scene = componentCases.find(
      entry => entry.name === `diagonal-unequal-values-nodata-${connectivity}`
    )!;
    const labels = decodeComponentRuns(scene.sparse, scene.width * scene.height);
    expect(labels[4]).toBe(labels[5]);
    expect(labels[0] === labels[7]).toBe(connectivity === 8);
    expect(labels[14]).toBe(0);
    expect(labels[21]).toBe(0);
    expect(scene.componentCount).toBe(connectivity === 4 ? 3 : 2);
  }
  expect(() => decodeComponentRuns([2, 7], 3)).toThrow('Incomplete raster');
  expect(() => decodeComponentRuns([4, 7], 3)).toThrow('Invalid lossless run length');
});
