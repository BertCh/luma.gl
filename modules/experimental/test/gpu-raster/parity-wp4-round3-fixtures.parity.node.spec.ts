// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {expect, it} from 'vitest';
import {generateStreamlinesOnCPU} from '../../src/gpu-raster/flow-texture/streamlines-cpu';
import {
  getGPUStreamlinesParameterValues,
  getGPUStreamlinesUint32ParameterValues
} from '../../src/gpu-raster/flow-texture';
import {getGPUDistanceFieldParameterValues} from '../../src/gpu-raster/distance-field';
import {makeGPURasterMetadata} from '../../src/gpu-raster/raster-grid';
import {computeDistanceFieldOnCPU} from './distance-field/distance-field-oracle';
import {computeZonalStatistics} from './raster-zonal-statistics/raster-zonal-statistics-oracle';
import {
  roundThreeFixtures,
  streamlineCases,
  decodeStreamlineCase,
  distanceCases,
  polygonCases,
  zonalCases,
  decodePackedNumbers,
  decodePackedRuns
} from '../oracles/parity-wp4-round3-fixtures';
import {compareRasterParity} from '../oracles/parity-wp4-compare';

it('WP4 round3 fixture provenance, versions and byte limits', () => {
  for (const fixture of roundThreeFixtures) {
    expect(fixture.versions.python).toBeTruthy();
    expect(fixture.versions.numpy).toBeTruthy();
    expect(fixture.versions.scipy).toBeTruthy();
    for (const source of fixture.sources) {
      expect(
        createHash('sha256')
          .update(readFileSync(resolve(source.path)))
          .digest('hex')
      ).toBe(source.sha256);
    }
  }
  for (const name of [
    'streamlines',
    'distance-isotropic',
    'distance-anisotropic',
    'distance-synthetic',
    'polygons',
    'zonal'
  ]) {
    expect(
      readFileSync(resolve(`modules/experimental/test/oracles/fixtures/parity-wp4-${name}.json`))
        .byteLength
    ).toBeLessThanOrEqual(1_500_000);
  }
});

for (const encoded of streamlineCases) {
  it(`Streamlines CPU diagnostic versus independent ODE/spacing (not GPU evidence): ${encoded.name}`, async () => {
    const scene = await decodeStreamlineCase(encoded);
    const result = generateStreamlinesOnCPU(
      {
        width: scene.fieldWidth,
        height: scene.fieldHeight,
        velocities: new Float32Array(scene.field)
      },
      scene.config,
      getGPUStreamlinesParameterValues(scene.settings),
      getGPUStreamlinesUint32ParameterValues(scene.settings)
    );
    const compare = (
      metric: string,
      actual: ArrayLike<number>,
      expected: ArrayLike<number>,
      profile: 'E' | 'L'
    ) =>
      compareRasterParity(
        'GPUStreamlines',
        `${scene.name}/CPU-diagnostic`,
        scene.reference,
        metric,
        actual,
        expected,
        profile
      );
    compare('randomgen-priorities', result.keys, scene.keys, 'E');
    compare(
      'all-candidate-points-and-invalid-mask',
      result.candidates.points,
      scene.candidates,
      'L'
    );
    compare('all-candidate-spans', result.candidates.spans, scene.spans, 'E');
    compare('all-accepted-ids', result.ids, scene.ids, 'E');
    compare('all-offsets', result.pathOffsets, scene.offsets, 'E');
    compare('all-published-points', result.points, scene.points, 'L');
  });
}

it('full distance transport retains every cell and row/column 511/512 seams', async () => {
  for (const scene of distanceCases.filter(scene => scene.width === 750)) {
    const length = scene.width * scene.height;
    const mask = await decodePackedRuns(scene.mask, length);
    const squared = await decodePackedRuns(scene.squared, length);
    expect(mask.length).toBe(562500);
    expect(squared.length).toBe(562500);
    for (let index = 0; index < length; index++) {
      if (mask[index]) expect(squared[index]).toBe(0);
    }
    // Global oracle distances must agree with a full-mask brute-force nearest search
    // at selected seam cells; GPU specs compare every distance and every nearest cell.
    const seeds = Array.from(mask.keys()).filter(index => mask[index] !== 0);
    for (const row of [0, 31, 32, 511, 512, 749]) {
      for (const column of [0, 31, 32, 511, 512, 749]) {
        let minimum = Infinity;
        for (const seed of seeds) {
          minimum = Math.min(
            minimum,
            (((seed % 750) - column) * scene.cellSize[0]) ** 2 +
              ((Math.floor(seed / 750) - row) * scene.cellSize[1]) ** 2
          );
        }
        expect(squared[row * 750 + column]).toBe(minimum);
      }
    }
  }
});

it('small EDT fixtures agree with existing CPU diagnostic including labels and cutoff', async () => {
  for (const scene of distanceCases.filter(scene => scene.width < 750)) {
    const length = scene.width * scene.height;
    const mask = await decodePackedRuns(scene.mask, length);
    const squared = await decodePackedRuns(scene.squared, length);
    const expected = Float64Array.from(squared, value =>
      value === 0xffffffff || Math.sqrt(value) > scene.maximumDistance ? Infinity : Math.sqrt(value)
    );
    const result = computeDistanceFieldOnCPU({
      width: scene.width,
      height: scene.height,
      settings: getGPUDistanceFieldParameterValues(
        {maximumDistance: scene.maximumDistance},
        makeGPURasterMetadata({
          width: scene.width,
          height: scene.height,
          cellSize: [scene.cellSize[0], scene.cellSize[1]],
          rowDirection: 'north'
        })
      ),
      mask,
      positions: new Float32Array(scene.positions),
      ids: new Uint32Array(scene.ids),
      count: scene.count
    });
    compareRasterParity(
      'GPUDistanceField',
      `${scene.name}/CPU-diagnostic`,
      scene.reference,
      'all-distances',
      result.distances,
      expected,
      'E'
    );
    if (scene.allocation.length) {
      compareRasterParity(
        'GPUDistanceField',
        `${scene.name}/CPU-diagnostic`,
        scene.reference,
        'all-labels',
        result.allocation,
        scene.allocation.map((value, index) =>
          Number.isFinite(expected[index]) ? value : 0xffffffff
        ),
        'E'
      );
    }
  }
});

it('polygon transport preserves complete topology, cells and explicit conservative boundary ambiguity', async () => {
  for (const scene of polygonCases) {
    const length = scene.width * scene.height;
    const [zones, minimum, maximum] = await Promise.all([
      decodePackedNumbers(scene.zones),
      decodePackedNumbers(scene.boundaryMinimum),
      decodePackedNumbers(scene.boundaryMaximum)
    ]);
    expect(zones.length).toBe(length);
    expect(minimum.length).toBe(length);
    expect(maximum.length).toBe(length);
    expect(scene.featureOffsets.at(-1)).toBe(scene.polygonOffsets.length - 1);
    expect(scene.polygonOffsets.at(-1)).toBe(scene.ringOffsets.length - 1);
    expect(scene.ringOffsets.at(-1)).toBe(scene.positions.length / 2);
    expect(minimum.flatMap((value, index) => (value !== maximum[index] ? [index] : []))).toEqual(
      scene.boundaryAmbiguousCells
    );
    expect(minimum.every((value, index) => value <= maximum[index])).toBe(true);
    if (scene.width === 512) {
      expect(scene.featureOffsets.length - 1).toBe(791);
      expect(new Set(zones.filter(value => value !== 0xffffffff)).size).toBeGreaterThan(780);
    }
    if (scene.boundaryShift) {
      expect(zones.slice(0, 10)).toEqual([0, 0, 0, 0, 1, 1, 1, 1, 0xffffffff, 0xffffffff]);
      expect(zones.slice(40)).toEqual(new Array(20).fill(0xffffffff));
    }
  }
});

for (const scene of zonalCases) {
  it(`zonal fixture CPU diagnostic versus independent SciPy (not GPU evidence): ${scene.name}`, async () => {
    const [zones, values, validity] = await Promise.all([
      decodePackedNumbers(scene.zones),
      decodePackedNumbers(scene.values),
      decodePackedNumbers(scene.validity)
    ]);
    expect(zones.length).toBe(scene.width * scene.height);
    expect(values.length).toBe(zones.length);
    expect(validity.length).toBe(zones.length);
    const result = computeZonalStatistics(zones, values, {...scene, validity});
    for (const name of [
      'cellCounts',
      'valueCounts',
      'sums',
      'means',
      'minimums',
      'maximums',
      'weightSums',
      'variances',
      'standardDeviations'
    ] as const) {
      compareRasterParity(
        'GPURasterZonalStatistics',
        `${scene.name}/CPU-diagnostic`,
        scene.reference,
        name,
        result[name],
        scene.expected[name],
        name.endsWith('Counts') ? 'E' : 'S'
      );
    }
    expect(result.overflow).toBe(scene.overflow);
  });
}
