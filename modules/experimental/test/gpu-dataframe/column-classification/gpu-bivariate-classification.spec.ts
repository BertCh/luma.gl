// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer, submitGraph} from '../../../src/utils/gpu-contributor-utils';
import {GPUBivariateClassification} from '../../../src/gpu-dataframe/column-classification/gpu-bivariate-classification';
import {
  getGPUBivariateClassificationParameterValues,
  type GPUBivariateClassificationParameterOptions
} from '../../../src/gpu-dataframe/column-classification/bivariate-classification-parameters';
import {packGPUColor} from '../../../src/gpu-dataframe/column-classification/color-scale-parameters';
import {computeBivariateClassificationOnCPU} from './bivariate-classification-oracle';
import {createInputBuffer, createOutputBuffer, readUint32} from '../../utils/gpu-contributor-test-utils';

const NONE = 0xffffffff;
const NO_DATA = 0x0a0b0c0d;

/** Deterministic xorshift in [0, 1). */
function createRandom(seed: number): () => number {
  let state = seed >>> 0 || 1;
  return () => {
    state ^= state << 13;
    state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5;
    state >>>= 0;
    return state / 4294967296;
  };
}

type Scene = {
  valuesX: Float32Array;
  valuesY: Float32Array;
  mask?: Uint32Array;
  alphaValues?: Float32Array;
  maximumClassCount: number;
};

type Frame = {
  breaksX: number[];
  breaksY: number[];
  palette: number[];
  options: GPUBivariateClassificationParameterOptions;
};

type Result = {classIds: Uint32Array; colors: Uint32Array; classCounts: Uint32Array};

type Fixture = {
  compileCount: number;
  run(frame: Frame): Promise<Result>;
  destroy(): void;
};

function createFixture(device: Device, scene: Scene): Fixture {
  const rows = scene.valuesX.length;
  const maximum = scene.maximumClassCount;
  const graph = new GPUCommandGraph(device, {id: 'bivariate-graph'});
  const buffers: Buffer[] = [];
  const track = (buffer: Buffer) => {
    buffers.push(buffer);
    return buffer;
  };
  const parameterBuffer = new GPUParameterBuffer(device, {
    id: 'bivariate-parameters',
    format: 'float32',
    length: 8
  });
  const breaksXBuffer = track(createInputBuffer(device, new Float32Array(maximum + 1)));
  const breaksYBuffer = track(createInputBuffer(device, new Float32Array(maximum + 1)));
  const paletteBuffer = track(createInputBuffer(device, new Uint32Array(maximum * maximum)));
  const outputs = {
    classIds: track(createOutputBuffer(device, rows)),
    colors: track(createOutputBuffer(device, rows)),
    classCounts: track(createOutputBuffer(device, maximum * maximum))
  };
  graph.add(
    new GPUBivariateClassification({
      id: 'bivariate',
      valuesX: importGraphBuffer(
        graph,
        'x',
        track(createInputBuffer(device, scene.valuesX)),
        'float32',
        rows
      ),
      valuesY: importGraphBuffer(
        graph,
        'y',
        track(createInputBuffer(device, scene.valuesY)),
        'float32',
        rows
      ),
      mask: scene.mask
        ? importGraphBuffer(
            graph,
            'mask',
            track(createInputBuffer(device, scene.mask)),
            'uint32',
            rows
          )
        : undefined,
      breaksX: importGraphBuffer(graph, 'breaks-x', breaksXBuffer, 'float32', maximum + 1),
      breaksY: importGraphBuffer(graph, 'breaks-y', breaksYBuffer, 'float32', maximum + 1),
      palette: importGraphBuffer(graph, 'palette', paletteBuffer, 'uint32', maximum * maximum),
      alphaValues: scene.alphaValues
        ? importGraphBuffer(
            graph,
            'alpha',
            track(createInputBuffer(device, scene.alphaValues)),
            'float32',
            rows
          )
        : undefined,
      parameters: parameterBuffer.importToGraph(graph),
      maximumClassCount: maximum,
      output: {
        classIds: importGraphBuffer(graph, 'o-ids', outputs.classIds, 'uint32', rows),
        colors: importGraphBuffer(graph, 'o-colors', outputs.colors, 'uint32', rows),
        classCounts: importGraphBuffer(
          graph,
          'o-counts',
          outputs.classCounts,
          'uint32',
          maximum * maximum
        )
      }
    })
  );
  const compiled = graph.compile();
  const fixture: Fixture = {
    compileCount: 1,
    async run(frame) {
      const breaksX = new Float32Array(maximum + 1);
      breaksX.set(frame.breaksX);
      const breaksY = new Float32Array(maximum + 1);
      breaksY.set(frame.breaksY);
      const palette = new Uint32Array(maximum * maximum);
      palette.set(frame.palette);
      breaksXBuffer.write(breaksX);
      breaksYBuffer.write(breaksY);
      paletteBuffer.write(palette);
      parameterBuffer.write(getGPUBivariateClassificationParameterValues(frame.options));
      submitGraph(device, compiled, undefined);
      return {
        classIds: Uint32Array.from(await readUint32(outputs.classIds, rows)),
        colors: Uint32Array.from(await readUint32(outputs.colors, rows)),
        classCounts: Uint32Array.from(await readUint32(outputs.classCounts, maximum * maximum))
      };
    },
    destroy() {
      compiled.destroy();
      parameterBuffer.destroy();
      for (const buffer of buffers) {
        buffer.destroy();
      }
    }
  };
  return fixture;
}

/** Compares a frame with the brute-force oracle; alpha may differ by 1 when value-by-alpha is on. */
function expectParity(actual: Result, scene: Scene, frame: Frame, label: string) {
  const palette = new Uint32Array(scene.maximumClassCount ** 2);
  palette.set(frame.palette);
  const expected = computeBivariateClassificationOnCPU({
    valuesX: scene.valuesX,
    valuesY: scene.valuesY,
    mask: scene.mask,
    breaksX: frame.breaksX,
    breaksY: frame.breaksY,
    classCountX: frame.options.classCountX,
    classCountY: frame.options.classCountY,
    maximumClassCount: scene.maximumClassCount,
    palette,
    alphaValues: scene.alphaValues,
    noDataColor: frame.options.noDataColor,
    valueByAlpha: frame.options.valueByAlpha
  });
  expect(Array.from(actual.classIds), `${label} classIds`).toEqual(Array.from(expected.classIds));
  expect(Array.from(actual.classCounts), `${label} counts`).toEqual(
    Array.from(expected.classCounts)
  );
  const fadeActive = Boolean(frame.options.valueByAlpha && scene.alphaValues);
  for (let row = 0; row < expected.colors.length; row++) {
    if (fadeActive) {
      expect(actual.colors[row] & 0x00ffffff, `${label} rgb row ${row}`).toBe(
        expected.colors[row] & 0x00ffffff
      );
      expect(
        Math.abs((actual.colors[row] >>> 24) - (expected.colors[row] >>> 24)),
        `${label} alpha row ${row}`
      ).toBeLessThanOrEqual(1);
    } else {
      expect(actual.colors[row], `${label} color row ${row}`).toBe(expected.colors[row]);
    }
  }
  return expected;
}

function createGridPalette(size: number): number[] {
  return Array.from({length: size * size}, (_, index) =>
    packGPUColor(40 + index * 11, 200 - index * 9, (index * 53) & 255, 220)
  );
}

function createScene(seed: number, rows: number, maximumClassCount: number): Scene {
  const random = createRandom(seed);
  const valuesX = new Float32Array(rows);
  const valuesY = new Float32Array(rows);
  const mask = new Uint32Array(rows);
  const alphaValues = new Float32Array(rows);
  for (let row = 0; row < rows; row++) {
    // Integer-valued data on a coarse grid puts many values exactly on break edges.
    valuesX[row] = random() < 0.03 ? NaN : Math.floor(random() * 24) - 2;
    valuesY[row] = random() < 0.03 ? NaN : Math.floor(random() * 24) - 2;
    mask[row] = random() < 0.05 ? 0 : 1;
    alphaValues[row] = random() < 0.05 ? NaN : random() * 3 - 0.5;
  }
  valuesX[0] = Infinity;
  valuesY[0] = -Infinity;
  valuesX[1] = -0;
  valuesY[1] = 0;
  return {valuesX, valuesY, mask, alphaValues, maximumClassCount};
}

it('GPUBivariateClassification classifies 3x3 and 4x4 grids and changes frames without recompiling', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scene = createScene(3, 3000, 4);
  const fixture = createFixture(device, scene);
  const palette9 = createGridPalette(3);
  const palette16 = createGridPalette(4);
  const frames: Frame[] = [
    {
      breaksX: [0, 5, 12, 20],
      breaksY: [0, 8, 16, 22],
      palette: palette9,
      options: {classCountX: 3, classCountY: 3, noDataColor: NO_DATA}
    },
    {
      breaksX: [-Infinity, 0, 6, 12, Infinity],
      breaksY: [-5, 4, 8, 16, 30],
      palette: palette16,
      options: {classCountX: 4, classCountY: 4, noDataColor: NO_DATA}
    },
    // Unequal axes use classY * classCountX + classX.
    {
      breaksX: [0, 10, 22],
      breaksY: [0, 3, 9, 12, 22],
      palette: palette16,
      options: {classCountX: 2, classCountY: 4, noDataColor: 0xff112233}
    },
    // Duplicate inner edges leave an empty class.
    {
      breaksX: [0, 7, 7, 20],
      breaksY: [0, 7, 7, 20],
      palette: palette9,
      options: {classCountX: 3, classCountY: 3, noDataColor: NO_DATA}
    },
    // Zero classes on an axis makes every row no-data.
    {
      breaksX: [0, 7, 14],
      breaksY: [0, 7, 14],
      palette: palette9,
      options: {classCountX: 0, classCountY: 2, noDataColor: NO_DATA}
    },
    {
      breaksX: [0, 5, 12, 20],
      breaksY: [0, 8, 16, 22],
      palette: palette9,
      options: {classCountX: 3, classCountY: 3, noDataColor: NO_DATA}
    }
  ];
  for (const [index, frame] of frames.entries()) {
    const actual = await fixture.run(frame);
    const expected = expectParity(actual, scene, frame, `frame ${index}`);
    if (index === 0) {
      // Row 0 is (+Inf, -Inf): last X class, first Y class. Row 1 is (-0, 0) on the first edge.
      expect(actual.classIds[0]).toBe(0 * 3 + 2);
      expect(actual.classIds[1]).toBe(0);
      const counted = Array.from(expected.classCounts).reduce((sum, count) => sum + count, 0);
      expect(counted).toBeGreaterThan(2000);
    }
    if (index === 4) {
      expect(actual.classIds.every(classId => classId === NONE)).toBe(true);
      expect(actual.colors.every(color => color === NO_DATA)).toBe(true);
    }
  }
  expect(fixture.compileCount).toBe(1);
  fixture.destroy();
});

it('GPUBivariateClassification applies value-by-alpha per frame and treats NaN alpha as minimum', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scene = createScene(8, 2500, 3);
  // Hand-checkable rows: the first five use a class-0 palette entry with alpha 200.
  scene.valuesX.set([1, 1, 1, 1, 1]);
  scene.valuesY.set([1, 1, 1, 1, 1]);
  scene.mask!.set([1, 1, 1, 1, 1]);
  scene.alphaValues!.set([0, 1, 0.5, NaN, 2]);
  const fixture = createFixture(device, scene);
  const palette = createGridPalette(3).map(color => ((color & 0x00ffffff) | (200 << 24)) >>> 0);
  const base: Frame = {
    breaksX: [0, 5, 12, 20],
    breaksY: [0, 8, 16, 22],
    palette,
    options: {classCountX: 3, classCountY: 3, noDataColor: NO_DATA}
  };
  const off = await fixture.run(base);
  expectParity(off, scene, base, 'vba off');
  expect(Array.from(off.colors.slice(0, 5), color => color >>> 24)).toEqual([
    200, 200, 200, 200, 200
  ]);
  const on: Frame = {
    ...base,
    options: {...base.options, valueByAlpha: {domain: [0, 1], minimumAlpha: 0.25}}
  };
  const faded = await fixture.run(on);
  expectParity(faded, scene, on, 'vba on');
  // factor = 0.25 + 0.75 * clamp(a): alpha 0 -> 50, 1 -> 200, 0.5 -> 125, NaN -> 50, 2 -> 200.
  expect(Array.from(faded.colors.slice(0, 5), color => color >>> 24)).toEqual([
    50, 200, 125, 50, 200
  ]);
  // Class counts are unaffected by the fade.
  expect(Array.from(faded.classCounts)).toEqual(Array.from(off.classCounts));
  const other: Frame = {
    ...on,
    options: {...on.options, valueByAlpha: {domain: [-1, 2], minimumAlpha: 0.5}}
  };
  expectParity(await fixture.run(other), scene, other, 'vba other');
  // Back to off on the same graph.
  expectParity(await fixture.run(base), scene, base, 'vba off again');
  expect(fixture.compileCount).toBe(1);
  fixture.destroy();
});

it('GPUBivariateClassification ignores value-by-alpha without an alphaValues view', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scene = createScene(4, 500, 3);
  delete scene.alphaValues;
  const fixture = createFixture(device, scene);
  const frame: Frame = {
    breaksX: [0, 5, 12, 20],
    breaksY: [0, 8, 16, 22],
    palette: createGridPalette(3),
    options: {
      classCountX: 3,
      classCountY: 3,
      valueByAlpha: {domain: [0, 1], minimumAlpha: 0}
    }
  };
  expectParity(await fixture.run(frame), scene, frame, 'no alpha view');
  fixture.destroy();
});

it('GPUBivariateClassification is bitwise reproducible', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scene = createScene(21, 4000, 4);
  const fixture = createFixture(device, scene);
  const frame: Frame = {
    breaksX: [-Infinity, 0, 6, 12, Infinity],
    breaksY: [-5, 4, 8, 16, 30],
    palette: createGridPalette(4),
    options: {
      classCountX: 4,
      classCountY: 4,
      noDataColor: NO_DATA,
      valueByAlpha: {domain: [0, 2], minimumAlpha: 0.3}
    }
  };
  const first = await fixture.run(frame);
  const second = await fixture.run(frame);
  expect(Array.from(second.colors)).toEqual(Array.from(first.colors));
  expect(Array.from(second.classIds)).toEqual(Array.from(first.classIds));
  expect(Array.from(second.classCounts)).toEqual(Array.from(first.classCounts));
  fixture.destroy();
});
