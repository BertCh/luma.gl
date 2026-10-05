// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {submitGraph} from '../../utils/gpu-contributor-test-utils';
import {
  getGPUSolarIrradianceParameterValues,
  getGPUSolarIrradianceSunTable,
  GPU_SOLAR_IRRADIANCE_PARAMETER_LENGTH,
  GPUSolarIrradiance,
  type GPUSolarIrradianceSettings
} from '../../../src/gpu-terrain/terrain-illumination/gpu-solar-irradiance';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32
} from '../../utils/gpu-contributor-test-utils';
import {computeSolarIrradiance} from './solar-irradiance-oracle';
import {
  createRandom,
  decodeHorizonUnorm16Oracle,
  encodeHorizonUnorm16Oracle
} from './terrain-horizon-oracle';

const DAY_START = Date.UTC(2024, 5, 20);
const DAY_END = Date.UTC(2024, 5, 21);

/** Worst relative error with an absolute floor, and the NaN pattern must match exactly. */
function expectRelativelyClose(
  actual: number[],
  expected: number[],
  relativeTolerance: number,
  absoluteFloor: number
): void {
  expect(actual.length).toBe(expected.length);
  let worst = 0;
  for (const [index, value] of expected.entries()) {
    if (Number.isNaN(value)) {
      expect(Number.isNaN(actual[index]), `index ${index}`).toBe(true);
      continue;
    }
    expect(Number.isNaN(actual[index]), `index ${index} expected ${value}`).toBe(false);
    worst = Math.max(
      worst,
      Math.abs(actual[index] - value) / Math.max(Math.abs(value), absoluteFloor)
    );
  }
  expect(worst).toBeLessThan(relativeTolerance);
}

function packUnorm16(horizon: Float32Array): Uint32Array {
  const words = new Uint32Array(Math.ceil(horizon.length / 2));
  for (const [element, angle] of horizon.entries()) {
    words[element >> 1] |= encodeHorizonUnorm16Oracle(angle) << ((element & 1) * 16);
  }
  return words;
}

it('GPUSolarIrradiance matches the oracle for chunked, unchunked, and unorm16 horizons', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const width = 24;
  const height = 18;
  const pixelCount = width * height;
  const directionCount = 16;
  const random = createRandom(31);
  const horizon = Float32Array.from(
    {length: pixelCount * directionCount},
    () => -2 + random() * 30
  );
  for (let sector = 0; sector < directionCount; sector++) {
    horizon[3 * directionCount + sector] = NaN;
  }
  const slope = Float32Array.from({length: pixelCount}, () => random() * 45);
  const aspect = Float32Array.from({length: pixelCount}, () => random() * 360);
  aspect[5] = -1;
  aspect[6] = -1;
  const skyViewFactor = Float32Array.from({length: pixelCount}, () => 0.5 + random() * 0.5);
  skyViewFactor[40] = NaN;
  const {values: table, sampleCount} = getGPUSolarIrradianceSunTable({
    longitude: 8.2,
    latitude: 46.8,
    start: DAY_START,
    end: DAY_END
  });
  expect(sampleCount).toBe(288);
  const capacity = 300;
  const sunTable = new Float32Array(capacity * 4);
  sunTable.set(table);
  const settingsValues: GPUSolarIrradianceSettings = {
    sampleCount,
    diffuseIrradiance: 60
  };
  for (const variant of [
    {format: 'float32' as const, samplesPerNode: undefined},
    {format: 'float32' as const, samplesPerNode: 100},
    {format: 'unorm16' as const, samplesPerNode: 128}
  ]) {
    const horizonData = variant.format === 'unorm16' ? packUnorm16(horizon) : horizon;
    const oracleHorizon =
      variant.format === 'unorm16'
        ? Array.from(horizonData as Uint32Array, (word, wordIndex) =>
            [wordIndex * 2, wordIndex * 2 + 1].map(element =>
              decodeHorizonUnorm16Oracle((word >>> ((element & 1) * 16)) & 0xffff)
            )
          ).flat()
        : horizon;
    const inputs = [horizonData, sunTable, slope, aspect, skyViewFactor].map(values =>
      createInputBuffer(device, values)
    );
    const outputs = [
      createOutputBuffer(device, pixelCount),
      createOutputBuffer(device, pixelCount),
      createOutputBuffer(device, pixelCount)
    ];
    const settings = new GPUParameterBuffer(device, {
      id: 'irradiance-settings',
      format: 'float32',
      length: GPU_SOLAR_IRRADIANCE_PARAMETER_LENGTH,
      values: getGPUSolarIrradianceParameterValues(settingsValues)
    });
    const graph = new GPUCommandGraph(device, {id: 'solar-irradiance-test'});
    const float = (id: string, buffer: (typeof inputs)[number], length = pixelCount) =>
      importGraphBuffer(graph, id, buffer, 'float32', length);
    const contributor = new GPUSolarIrradiance({
      width,
      height,
      directionCount,
      horizonFormat: variant.format,
      horizon:
        variant.format === 'unorm16'
          ? importGraphBuffer(graph, 'horizon', inputs[0], 'uint32', Math.ceil(horizon.length / 2))
          : float('horizon', inputs[0], horizon.length),
      sunTable: float('sun-table', inputs[1], capacity * 4),
      sampleCapacity: capacity,
      samplesPerNode: variant.samplesPerNode,
      settings: settings.importToGraph(graph),
      slope: float('slope', inputs[2]),
      aspect: float('aspect', inputs[3]),
      skyViewFactor: float('svf', inputs[4]),
      sunHours: float('sun-hours', outputs[0]),
      insolation: float('insolation', outputs[1]),
      validity: importGraphBuffer(graph, 'validity', outputs[2], 'uint32', pixelCount)
    });
    graph.add(contributor);
    const compiled = graph.compile();
    submitGraph(device, compiled, undefined);
    const expected = computeSolarIrradiance({
      pixelCount,
      directionCount,
      horizon: oracleHorizon,
      sunTable,
      sampleCount,
      angularRadiusDegrees: 0.2666,
      diffuseIrradiance: 60,
      slope,
      aspect,
      skyViewFactor
    });
    const hours = await readFloat32(outputs[0], pixelCount);
    const insolation = await readFloat32(outputs[1], pixelCount);
    expectRelativelyClose(hours, expected.sunHours, 1e-4, 0.05);
    expectRelativelyClose(insolation, expected.insolation, 1e-4, 50);
    expect(await readUint32(outputs[2], pixelCount)).toEqual(expected.validity);
    // Pixels 3 (horizon) and 40 (svf) are invalid; everything else is nontrivial.
    expect(expected.validity.reduce((sum, value) => sum + value, 0)).toBe(pixelCount - 2);
    const finiteHours = hours.filter(value => !Number.isNaN(value));
    expect(Math.max(...finiteHours)).toBeGreaterThan(5);
    expect(Math.min(...finiteHours)).toBeGreaterThan(0);
    expect(Math.max(...insolation.filter(value => !Number.isNaN(value)))).toBeGreaterThan(1000);
    // Per-frame rewrite: fewer rows and a hard-edged sun reuse the compiled graph.
    settings.write(
      getGPUSolarIrradianceParameterValues({
        sampleCount: 150,
        angularRadiusDegrees: 0,
        diffuseIrradiance: 0
      })
    );
    submitGraph(device, compiled, undefined);
    const shorter = computeSolarIrradiance({
      pixelCount,
      directionCount,
      horizon: oracleHorizon,
      sunTable,
      sampleCount: 150,
      angularRadiusDegrees: 0,
      slope,
      aspect,
      skyViewFactor
    });
    const shorterHours = await readFloat32(outputs[0], pixelCount);
    // Hard-edged visibility is a step in time; a float32 horizon tie may flip one 5 minute row.
    for (const [pixel, value] of shorter.sunHours.entries()) {
      if (Number.isNaN(value)) {
        expect(Number.isNaN(shorterHours[pixel])).toBe(true);
      } else {
        expect(Math.abs(shorterHours[pixel] - value)).toBeLessThan(2 / 12 + 1e-4);
      }
    }
    expect(Math.max(...shorterHours.filter(value => !Number.isNaN(value)))).toBeLessThan(
      Math.max(...finiteHours)
    );
    expect(variant.format).toBe(contributor.horizonFormat);
    compiled.destroy();
    settings.destroy();
    for (const buffer of [...inputs, ...outputs]) buffer.destroy();
  }
});

it('GPUSolarIrradiance gives about 12 hours at the equator on an equinox and fewer in a pit', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const width = 4;
  const height = 2;
  const pixelCount = width * height;
  const directionCount = 8;
  const horizon = new Float32Array(pixelCount * directionCount);
  // Pixel 1 sits in a pit with a 50 degree horizon in every direction; pixel 2 is a 10 degree
  // horizon; the rest is open flat ground.
  horizon.fill(50, directionCount, 2 * directionCount);
  horizon.fill(10, 2 * directionCount, 3 * directionCount);
  const {values: table, sampleCount} = getGPUSolarIrradianceSunTable({
    longitude: 0,
    latitude: 0,
    start: Date.UTC(2024, 2, 20),
    end: Date.UTC(2024, 2, 21),
    refraction: false
  });
  const inputs = [horizon, table].map(values => createInputBuffer(device, values));
  const outputs = [createOutputBuffer(device, pixelCount), createOutputBuffer(device, pixelCount)];
  const settings = new GPUParameterBuffer(device, {
    id: 'equinox-settings',
    format: 'float32',
    length: GPU_SOLAR_IRRADIANCE_PARAMETER_LENGTH,
    values: getGPUSolarIrradianceParameterValues({sampleCount})
  });
  const graph = new GPUCommandGraph(device, {id: 'solar-irradiance-equinox'});
  graph.add(
    new GPUSolarIrradiance({
      width,
      height,
      directionCount,
      horizon: importGraphBuffer(graph, 'horizon', inputs[0], 'float32', horizon.length),
      sunTable: importGraphBuffer(graph, 'sun-table', inputs[1], 'float32', table.length),
      sampleCapacity: sampleCount,
      settings: settings.importToGraph(graph),
      sunHours: importGraphBuffer(graph, 'sun-hours', outputs[0], 'float32', pixelCount),
      insolation: importGraphBuffer(graph, 'insolation', outputs[1], 'float32', pixelCount)
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const hours = await readFloat32(outputs[0], pixelCount);
  const insolation = await readFloat32(outputs[1], pixelCount);
  // The disk-area-weighted visibility integrates to the time the center is up: 12 h +- one step.
  expect(Math.abs(hours[0] - 12)).toBeLessThan(5 / 60);
  expect(hours[7]).toBeCloseTo(hours[0], 6);
  expect(hours[2]).toBeLessThan(hours[0] - 1);
  expect(hours[1]).toBeLessThan(hours[2] - 1);
  // Sun above 50 degrees at the equator on an equinox: 12 * (1 - 2 * asin(...))... from the oracle.
  const expected = computeSolarIrradiance({
    pixelCount,
    directionCount,
    horizon,
    sunTable: table,
    sampleCount,
    angularRadiusDegrees: 0.2666
  });
  expect(Math.abs(hours[1] - expected.sunHours[1])).toBeLessThan(1e-3);
  // Analytic: hours with altitude > 50 degrees = 12 / 90 * (2 * 40) ... altitude 90 - 15|t|:
  // above 50 degrees for |hour angle| < 40 degrees, i.e. 2 * 40 / 15 = 5.33 hours.
  expect(Math.abs(hours[1] - 80 / 15)).toBeLessThan(0.2);
  // Horizontal clear-sky energy of a day at the equator is several kWh/m2.
  expect(insolation[0]).toBeGreaterThan(3000);
  expect(insolation[0]).toBeLessThan(9000);
  expect(insolation[1]).toBeLessThan(insolation[0]);
  expectRelativelyClose(
    insolation,
    computeSolarIrradiance({
      pixelCount,
      directionCount,
      horizon,
      sunTable: table,
      sampleCount,
      angularRadiusDegrees: 0.2666
    }).insolation,
    1e-4,
    50
  );
  compiled.destroy();
  settings.destroy();
  for (const buffer of [...inputs, ...outputs]) buffer.destroy();
});

it('GPUSolarIrradiance times a day at 5 minute steps on 1024 x 1024 with 16 sectors', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const width = 1024;
  const height = 1024;
  const pixelCount = width * height;
  const directionCount = 16;
  const random = createRandom(77);
  const horizon = Float32Array.from({length: pixelCount * directionCount}, () => random() * 25);
  const {values: table, sampleCount} = getGPUSolarIrradianceSunTable({
    longitude: 8.2,
    latitude: 46.8,
    start: DAY_START,
    end: DAY_END
  });
  const inputs = [horizon, table].map(values => createInputBuffer(device, values));
  const outputs = [createOutputBuffer(device, pixelCount), createOutputBuffer(device, pixelCount)];
  const settings = new GPUParameterBuffer(device, {
    id: 'timing-settings',
    format: 'float32',
    length: GPU_SOLAR_IRRADIANCE_PARAMETER_LENGTH,
    values: getGPUSolarIrradianceParameterValues({sampleCount, diffuseIrradiance: 50})
  });
  const graph = new GPUCommandGraph(device, {id: 'solar-irradiance-timing'});
  graph.add(
    new GPUSolarIrradiance({
      width,
      height,
      directionCount,
      horizon: importGraphBuffer(graph, 'horizon', inputs[0], 'float32', horizon.length),
      sunTable: importGraphBuffer(graph, 'sun-table', inputs[1], 'float32', table.length),
      sampleCapacity: sampleCount,
      samplesPerNode: 96,
      settings: settings.importToGraph(graph),
      sunHours: importGraphBuffer(graph, 'sun-hours', outputs[0], 'float32', pixelCount),
      insolation: importGraphBuffer(graph, 'insolation', outputs[1], 'float32', pixelCount)
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  await readFloat32(outputs[0], 4);
  const startTime = performance.now();
  submitGraph(device, compiled, undefined);
  const hours = await readFloat32(outputs[0], pixelCount);
  const milliseconds = performance.now() - startTime;
  // Midsummer in the Alps: a long day, visible disk hours between 4 and 16.
  expect(hours[0]).toBeGreaterThan(4);
  expect(hours[pixelCount - 1]).toBeLessThan(17);
  expect(Number.isNaN(hours[pixelCount >> 1])).toBe(false);
  expect(milliseconds).toBeLessThan(20000);
  compiled.destroy();
  settings.destroy();
  for (const buffer of [...inputs, ...outputs]) buffer.destroy();
});
