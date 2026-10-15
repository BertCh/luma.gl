// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {expect, it} from 'vitest';
import {makeGPURasterMetadata} from '../../src/gpu-raster';
import {getGPUIsobandsParameterValues} from '../../src/gpu-raster/isolines';
import {getGPURasterArithmeticParameterValues} from '../../src/gpu-raster/raster-algebra';
import {getGPURasterExtremaPyramidLayout} from '../../src/gpu-raster/raster-pyramid';
import {
  roundFourFixtures,
  isobandCases,
  arithmeticCases,
  pyramidCases
} from '../oracles/parity-wp4-round4-fixtures';
import {decodePackedNumbers} from '../oracles/parity-wp4-round3-fixtures';
import {compareRoundFour} from '../oracles/parity-wp4-round4-compare';
import {compareIsobandGeometry} from '../oracles/parity-wp4-isoband-geometry';
import centreAmbiguities from '../oracles/fixtures/parity-wp4-polygon-centres.json';
import {buildIsobandTrianglesOnCPU, computeBandClassesOnCPU} from './isolines/isobands-oracle';
import {arithmeticOnCPU} from './raster-algebra/raster-algebra-oracle';
import {computeRasterExtremaPyramid} from './raster-pyramid/raster-pyramid-oracle';

it('WP4 round4 independent references retain versions, raw-source hashes and byte limits', () => {
  for (const fixture of roundFourFixtures) {
    expect(fixture.versions.python).toBeTruthy();
    expect(fixture.versions.numpy).toBeTruthy();
    expect(Buffer.byteLength(JSON.stringify(fixture))).toBeLessThanOrEqual(1_500_000);
    for (const source of fixture.sources)
      expect(createHash('sha256').update(readFileSync(source.path)).digest('hex')).toBe(
        source.sha256
      );
  }
  expect(arithmeticCases.slice(0, 4).reduce((sum, scene) => sum + scene.length, 0)).toBe(512 * 512);
  expect(pyramidCases.slice(0, 2).map(scene => scene.width * scene.height)).toEqual([
    512 * 512,
    512 * 512
  ]);
});

it('Chicago has exactly one shared-edge crossing within half an f32 ULP of a pixel centre', () => {
  for (const source of centreAmbiguities.sources) {
    expect(createHash('sha256').update(readFileSync(source.path)).digest('hex')).toBe(
      source.sha256
    );
  }
  expect(centreAmbiguities.cells).toHaveLength(1);
  const cell = centreAmbiguities.cells[0];
  expect([cell.index, cell.row, cell.column, ...cell.features]).toEqual([
    76206, 148, 430, 709, 710
  ]);
  expect(cell.exactMargin).toBe('291/466714624');
  expect(cell.crossing - cell.centre[0]).toBeGreaterThan(0);
  expect(cell.crossing - cell.centre[0]).toBeLessThan(cell.halfUlp);
  expect(Math.fround(cell.crossing)).toBe(cell.centre[0]);
});

for (const scene of isobandCases) {
  it(`Isobands CPU diagnostic versus independent polygons (not GPU evidence): ${scene.name}`, async () => {
    const [values, validity, classes, triangles, edges] = await Promise.all([
      decodePackedNumbers(scene.values),
      decodePackedNumbers(scene.validity),
      decodePackedNumbers(scene.classes),
      decodePackedNumbers(scene.triangles),
      decodePackedNumbers(scene.edges)
    ]);
    const grid = makeGPURasterMetadata(scene);
    const input = {
      width: scene.width,
      height: scene.height,
      values: new Float32Array(values),
      validity: new Uint32Array(validity),
      noDataValue: -9999,
      breaks: new Float32Array(scene.breaks)
    };
    const parameters = getGPUIsobandsParameterValues(
      {
        breakCount: scene.breaks.length,
        extent: [0, 0, scene.width, scene.height],
        firstBand: scene.firstBand,
        lastBand: scene.lastBand
      },
      grid
    );
    const actual = buildIsobandTrianglesOnCPU(input, parameters, scene.breaks.length);
    compareRoundFour(
      'GPUIsobands CPU diagnostic',
      scene.name,
      scene.reference,
      'classes',
      computeBandClassesOnCPU(input, scene.breaks.length),
      classes,
      'E'
    );
    compareIsobandGeometry(
      `CPU diagnostic/${scene.name}`,
      scene.reference,
      scene.width,
      scene.height,
      actual.triangles,
      actual.bands,
      triangles,
      edges
    );
  });
}

for (const scene of arithmeticCases) {
  it(`Arithmetic CPU diagnostic versus NumPy (not GPU evidence): ${scene.name}`, async () => {
    const [first, second, expected] = await Promise.all([
      decodePackedNumbers(scene.first),
      scene.second ? decodePackedNumbers(scene.second) : Promise.resolve(undefined),
      decodePackedNumbers(scene.expected)
    ]);
    const actual = arithmeticOnCPU(
      new Float32Array(first),
      second ? new Float32Array(second) : undefined,
      getGPURasterArithmeticParameterValues(scene.settings),
      -9999
    );
    compareRoundFour(
      'GPURasterArithmetic CPU diagnostic',
      scene.name,
      scene.reference,
      'every-value-and-mask',
      actual,
      expected,
      'R'
    );
  });
}

for (const scene of pyramidCases) {
  it(`Pyramid CPU diagnostic versus direct NumPy footprints (not GPU evidence): ${scene.name}`, async () => {
    const [values, validity, maximum, minimum] = await Promise.all([
      decodePackedNumbers(scene.values),
      decodePackedNumbers(scene.validity),
      decodePackedNumbers(scene.maximum),
      decodePackedNumbers(scene.minimum)
    ]);
    const valid = Uint32Array.from(values, (value, index) =>
      Number(Number.isFinite(value) && value !== -9999 && validity[index] !== 0)
    );
    const calibrated = Float32Array.from(values, value => value * scene.scale + scene.offset);
    const layout = getGPURasterExtremaPyramidLayout(makeGPURasterMetadata(scene), scene);
    expect(layout.levels.map(({level, ...rest}) => rest)).toEqual(scene.levels);
    const actual = computeRasterExtremaPyramid(calibrated, valid, layout);
    compareRoundFour(
      'GPURasterExtremaPyramid CPU diagnostic',
      scene.name,
      scene.reference,
      'all-maxima',
      actual.maximum,
      maximum,
      'E'
    );
    compareRoundFour(
      'GPURasterExtremaPyramid CPU diagnostic',
      scene.name,
      scene.reference,
      'all-minima',
      actual.minimum,
      minimum,
      'E'
    );
  });
}
