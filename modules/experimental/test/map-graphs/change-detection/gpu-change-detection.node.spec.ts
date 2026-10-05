// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  getGPUChangeDetectionParameterValues,
  GPUChangeDetection,
  type GPUChangeDetectionProps
} from '../../../src/map-graphs/change-detection';
import {createNullWebGPUDevice} from '../map-graph-test-utils';
import {
  complementaryErrorOnCPU,
  detectChangeOnCPU,
  studentTTwoSidedPOnCPU
} from './change-detection-cpu';

let serial = 0;

function createProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUChangeDetectionProps> = {}
): GPUChangeDetectionProps {
  const view = <Format extends 'uint32' | 'float32' | 'sint32'>(format: Format, length: number) =>
    createTransientView(graph, `view-${serial++}`, format, length);
  return {
    slices: view('float32', 3 * 6),
    parameters: view('float32', 5),
    cellCount: 3,
    sliceCount: 6,
    output: {difference: view('float32', 3), tPValue: view('float32', 3)},
    ...overrides
  };
}

function expectThrows(
  overrides: (graph: GPUCommandGraph) => Partial<GPUChangeDetectionProps>,
  message: RegExp
): void {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  expect(() => new GPUChangeDetection(createProps(graph, overrides(graph)))).toThrow(message);
  device.destroy();
}

it('getGPUChangeDetectionParameterValues packs the layout and defaults', () => {
  expect(
    Array.from(
      getGPUChangeDetectionParameterValues({
        beforeSlice: 1,
        afterSlice: 4,
        splitSlice: 3
      })
    )
  ).toEqual([1, 4, Math.fround(1e-6), Math.fround(0.05), 3]);
  expect(() =>
    getGPUChangeDetectionParameterValues({beforeSlice: 0, afterSlice: 1}, new Float32Array(2))
  ).toThrow(/hold/);
});

it('GPUChangeDetection rejects invalid properties', () => {
  expectThrows(() => ({cellCount: 0}), /cellCount/);
  expectThrows(() => ({sliceCount: 1}), /sliceCount/);
  expectThrows(() => ({sliceCount: 257}), /sliceCount/);
  expectThrows(() => ({bandCount: 0}), /bandCount/);
  expectThrows(() => ({output: {}}), /at least one output/);
  expectThrows(graph => ({slices: createTransientView(graph, 'short', 'float32', 17)}), /slices/);
  expectThrows(
    graph => ({
      parameters: createTransientView(graph, 'short-p', 'float32', 4)
    }),
    /parameters must hold/
  );
  expectThrows(
    graph => ({mask: createTransientView(graph, 'short-m', 'uint32', 2)}),
    /mask must hold/
  );
  expectThrows(
    graph => ({
      output: {
        difference: createTransientView(graph, 'short-d', 'float32', 2)
      }
    }),
    /output.difference/
  );
  expectThrows(
    graph => ({
      bandCount: 2,
      slices: createTransientView(graph, 'multi', 'float32', 36),
      output: {tPValue: createTransientView(graph, 'p-multi', 'float32', 3)}
    }),
    /requires bandCount 1/
  );
  expectThrows(
    graph => ({
      output: {
        changeMagnitude: createTransientView(graph, 'mag', 'float32', 3)
      }
    }),
    /bandCount/
  );
  expectThrows(
    graph => ({
      sliceCount: 65,
      slices: createTransientView(graph, 'long', 'float32', 3 * 65),
      output: {senSlope: createTransientView(graph, 'sen', 'float32', 3)}
    }),
    /at most 64/
  );
  expectThrows(
    graph => ({
      bandCount: 3,
      slices: createTransientView(graph, 'tri', 'float32', 54),
      output: {
        changeDirection: createTransientView(graph, 'dir', 'float32', 3)
      }
    }),
    /changeDirection/
  );
  expectThrows(graph => {
    const slices = createTransientView(graph, 'alias', 'float32', 18);
    return {slices, output: {difference: slices}};
  }, /must not share/);
  expectThrows(
    graph => ({
      output: {
        mannKendallS: createTransientView(graph, 's-float', 'float32', 3) as never
      }
    }),
    /mannKendallS/
  );
});

it('GPUChangeDetection creates deterministic nodes for the requested modes only', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const f32 = (length: number) => createTransientView(graph, `o-${serial++}`, 'float32', length);
  const recipe = new GPUChangeDetection(
    createProps(graph, {
      output: {
        difference: f32(3),
        logRatio: f32(3),
        tStatistic: f32(3),
        senSlope: f32(3),
        mannKendallS: createTransientView(graph, 'mk-s', 'sint32', 3),
        significance: createTransientView(graph, 'sig', 'uint32', 3)
      }
    })
  );
  const ids = recipe.getCommandNodes(graph).map(node => node.id);
  expect(ids).toEqual([
    'change-detection-two-slice',
    'change-detection-t-test',
    'change-detection-sen-slope',
    'change-detection-mann-kendall'
  ]);
  const multiband = new GPUChangeDetection({
    ...createProps(graph),
    id: 'multi',
    bandCount: 2,
    slices: createTransientView(graph, 'multi-slices', 'float32', 36),
    output: {
      changeMagnitude: f32(3),
      changeDirection: f32(3)
    }
  });
  expect(multiband.getCommandNodes(graph).map(node => node.id)).toEqual(['multi-multiband']);
  device.destroy();
});

it('detectChangeOnCPU matches the hand-checked Welch t-test', () => {
  // before [1, 2, 3], after [4, 5, 6]: means 2 and 5, variances 1, t = 3 / sqrt(2 / 3), df = 4.
  const result = detectChangeOnCPU({
    slices: Float32Array.from([1, 2, 3, 4, 5, 6]),
    cellCount: 1,
    sliceCount: 6,
    beforeSlice: 0,
    afterSlice: 5,
    splitSlice: 3
  });
  expect(result.tStatistic[0]).toBeCloseTo(3 / Math.sqrt(2 / 3), 12);
  expect(result.tDegreesOfFreedom[0]).toBeCloseTo(4, 12);
  // Closed form for even df = 4: P(|T| < t) = s (1 + (1 - s^2) / 2) with s = t / sqrt(4 + t^2).
  const t2 = 13.5;
  const s = Math.sqrt(t2 / (4 + t2));
  expect(result.tPValue[0]).toBeCloseTo(1 - s * (1 + (1 - s * s) / 2), 6);
  expect(result.tPValue[0]).toBeCloseTo(0.0213, 3);
  expect(result.significance[0]).toBe(1);
  expect(result.difference[0]).toBe(5);
  // df = 2 closed form: p = 1 - t / sqrt(2 + t^2).
  for (const t of [0.3, 1.5, 4]) {
    expect(studentTTwoSidedPOnCPU(t, 2)).toBeCloseTo(1 - t / Math.sqrt(2 + t * t), 8);
  }
  // Large df approaches the normal: p(1.96, 1000) near 0.0503.
  expect(studentTTwoSidedPOnCPU(1.96, 1000)).toBeCloseTo(0.0503, 3);
  expect(complementaryErrorOnCPU(1.96 / Math.SQRT2)).toBeCloseTo(0.04999, 4);
});

it('detectChangeOnCPU matches hand-checked Sen slopes and Mann-Kendall with ties', () => {
  const sen = (values: number[]) =>
    detectChangeOnCPU({
      slices: Float32Array.from(values),
      cellCount: 1,
      sliceCount: values.length,
      beforeSlice: 0,
      afterSlice: 1
    });
  // Ten slopes: six 1s, 25, 33, 49, 97. The median is 1 despite the outlier.
  expect(sen([0, 1, 2, 3, 100]).senSlope[0]).toBe(1);
  // Valid slices 0, 2, 3 -> slopes 1, 1 / 3, -1.
  expect(sen([1, NaN, 3, 2]).senSlope[0]).toBeCloseTo(1 / 3, 12);
  // Even pair count: sorted slopes -1, 0.5, 2/3, 1, 1.5, 2 -> median (2/3 + 1) / 2.
  expect(sen([1, 2, 4, 3]).senSlope[0]).toBeCloseTo(5 / 6, 12);
  // Mann-Kendall of [1, 2, 2, 3, 3, 3]: S = 5 + 3 + 3 = 11, tie term 2*1*9 + 3*2*11 = 84,
  // var = (6*5*17 - 84) / 18, Z = (11 - 1) / sqrt(var).
  const mannKendall = sen([1, 2, 2, 3, 3, 3]);
  expect(mannKendall.mannKendallS[0]).toBe(11);
  expect(mannKendall.mannKendallZ[0]).toBeCloseTo(10 / Math.sqrt(426 / 18), 12);
  expect(mannKendall.mannKendallP[0]).toBeCloseTo(0.0398, 3);
  // A constant series has zero variance: Z 0, p 1, S 0.
  const constant = sen([2, 2, 2, 2]);
  expect([constant.mannKendallS[0], constant.mannKendallZ[0], constant.mannKendallP[0]]).toEqual([
    0, 0, 1
  ]);
});

it('detectChangeOnCPU handles two-slice edge cases and multiband vectors', () => {
  const result = detectChangeOnCPU({
    // Cells: (before, after) = (2, 5), (0, 3), (-2, 1), (NaN, 1).
    slices: Float32Array.from([2, 5, 0, 3, -2, 1, NaN, 1]),
    cellCount: 4,
    sliceCount: 2,
    beforeSlice: 0,
    afterSlice: 1,
    epsilon: 0.5
  });
  expect(Array.from(result.difference.slice(0, 3))).toEqual([3, 3, 3]);
  expect(result.difference[3]).toBeNaN();
  expect(result.logRatio[0]).toBeCloseTo(Math.log(5.5 / 2.5), 12);
  expect(result.logRatio[2]).toBeNaN();
  expect(result.percentChange[0]).toBe(150);
  expect(result.percentChange[1]).toBeNaN();
  expect(result.percentChange[2]).toBe(150);
  const multiband = detectChangeOnCPU({
    // One cell, two slices, two bands: before (1, 1), after (4, 5) -> difference (3, 4).
    slices: Float32Array.from([1, 1, 4, 5]),
    cellCount: 1,
    sliceCount: 2,
    bandCount: 2,
    beforeSlice: 0,
    afterSlice: 1
  });
  expect(multiband.changeMagnitude[0]).toBe(5);
  expect(multiband.changeDirection[0]).toBeCloseTo(Math.atan2(4, 3), 12);
  const threeBands = detectChangeOnCPU({
    slices: Float32Array.from([1, 1, 1, 2, 0, 1]),
    cellCount: 1,
    sliceCount: 2,
    bandCount: 3,
    beforeSlice: 0,
    afterSlice: 1
  });
  // Band 0 increased (bit 0), band 1 decreased (bit 17), band 2 unchanged.
  expect(threeBands.changeDirection[0]).toBe(1 | (1 << 17));
});
