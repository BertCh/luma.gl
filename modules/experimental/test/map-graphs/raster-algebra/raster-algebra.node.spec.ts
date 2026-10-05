// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  getGPURasterArithmeticParameterValues,
  getGPURasterCellStatisticsParameterValues,
  getGPURasterConditionalParameterValues,
  getGPURasterReclassifyParameterValues,
  getGPUWeightedOverlayParameterLength,
  getGPUWeightedOverlayParameterValues,
  GPURasterArithmetic,
  GPURasterCellStatistics,
  GPURasterConditional,
  GPURasterReclassify,
  GPUWeightedOverlay
} from '../../../src/map-graphs/raster-algebra';
import {createNullWebGPUDevice} from '../map-graph-test-utils';
import {
  arithmeticOnCPU,
  computeCellStatisticsOnCPU,
  conditionalOnCPU,
  countBreaksBelow,
  reclassifyOnCPU,
  weightedOverlayOnCPU
} from './raster-algebra-oracle';

let serial = 0;

function view<Format extends 'uint32' | 'float32'>(
  graph: GPUCommandGraph,
  format: Format,
  length: number
) {
  return createTransientView(graph, `view-${serial++}`, format, length);
}

function withGraph(callback: (graph: GPUCommandGraph) => void): void {
  const device = createNullWebGPUDevice();
  callback(new GPUCommandGraph(device));
  device.destroy();
}

it('raster-algebra parameter helpers pack layouts', () => {
  expect(
    Array.from(getGPURasterReclassifyParameterValues({breakCount: 3, closed: 'right'}))
  ).toEqual([3, 1, 0, 0]);
  expect(() => getGPURasterReclassifyParameterValues({breakCount: -1})).toThrow(/breakCount/);
  expect(getGPUWeightedOverlayParameterLength(3)).toBe(32);
  const overlay = getGPUWeightedOverlayParameterValues({
    layers: [
      {weight: 2, inputMin: 10, inputMax: 20, invert: true},
      {weight: -1, mode: 'table', breakCount: 2, closed: 'right'},
      {weight: 1, inputMin: 5, inputMax: 5}
    ],
    normalizeWeights: true,
    noDataPolicy: 'ignore'
  });
  expect(Array.from(overlay.subarray(0, 2))).toEqual([1, 1]);
  expect(Array.from(overlay.subarray(8, 16))).toEqual([2, 0, 10, Math.fround(0.1), 1, 0, 0, 0]);
  expect(Array.from(overlay.subarray(16, 24))).toEqual([-1, 1, 0, 1, 0, 2, 1, 0]);
  expect(Array.from(overlay.subarray(24, 32))).toEqual([1, 0, 5, 0, 0, 0, 0, 1]);
  expect(() => getGPUWeightedOverlayParameterValues({layers: [{weight: NaN}]})).toThrow(/weight/);
  expect(
    Array.from(getGPURasterCellStatisticsParameterValues({noDataPolicy: 'propagate'}))
  ).toEqual([1, 1, 0, 0]);
  expect(
    Array.from(
      getGPURasterConditionalParameterValues({
        comparison: 'between',
        threshold: 2,
        upperThreshold: 5,
        constantA: 7
      })
    )
  ).toEqual([6, 2, 5, 7, 0, 0, 0, 0]);
  expect(
    Array.from(getGPURasterArithmeticParameterValues({operation: 'normalizedDifference'}))
  ).toEqual([8, 1, 0, 1, 0, 0, -Infinity, Infinity]);
  expect(() => getGPURasterArithmeticParameterValues({operation: 'modulo' as 'add'})).toThrow(
    /not supported/
  );
});

it('raster-algebra recipes validate props', () => {
  withGraph(graph => {
    const values = view(graph, 'float32', 12);
    const breaks = view(graph, 'float32', 4);
    const parameters = view(graph, 'float32', 8);
    expect(() => new GPURasterReclassify({values, breaks, parameters, output: {}})).toThrow(
      /at least one output/
    );
    expect(
      () =>
        new GPURasterReclassify({
          values,
          breaks,
          parameters,
          output: {reclassified: view(graph, 'float32', 12)}
        })
    ).toThrow(/requires classValues/);
    expect(
      () =>
        new GPURasterReclassify({
          values,
          breaks,
          parameters,
          output: {classes: view(graph, 'uint32', 11)}
        })
    ).toThrow(/output.classes/);
    expect(
      () =>
        new GPURasterReclassify({
          values,
          breaks,
          parameters,
          output: {classes: values as never}
        })
    ).toThrow(/uint32/);
    expect(
      () =>
        new GPUWeightedOverlay({
          stack: view(graph, 'float32', 34),
          layerCount: 17,
          cellCount: 2,
          parameters: view(graph, 'float32', 200),
          output: {score: view(graph, 'float32', 2)}
        })
    ).toThrow(/layerCount/);
    expect(
      () =>
        new GPUWeightedOverlay({
          stack: view(graph, 'float32', 6),
          layerCount: 3,
          cellCount: 2,
          parameters: view(graph, 'float32', 32),
          remapBreaks: view(graph, 'float32', 6),
          output: {score: view(graph, 'float32', 2)}
        })
    ).toThrow(/together/);
    expect(
      () =>
        new GPUWeightedOverlay({
          stack: view(graph, 'float32', 6),
          layerCount: 3,
          cellCount: 2,
          parameters: view(graph, 'float32', 31),
          output: {score: view(graph, 'float32', 2)}
        })
    ).toThrow(/parameters/);
    expect(
      () =>
        new GPURasterCellStatistics({
          stack: view(graph, 'float32', 130),
          layerCount: 65,
          cellCount: 2,
          parameters: view(graph, 'float32', 4),
          output: {sum: view(graph, 'float32', 2)}
        })
    ).toThrow(/layerCount/);
    expect(
      () =>
        new GPURasterConditional({
          cellCount: 12,
          parameters,
          output: {values: view(graph, 'float32', 12)}
        })
    ).toThrow(/exactly one/);
    const a = view(graph, 'float32', 12);
    expect(
      () =>
        new GPURasterArithmetic({
          cellCount: 12,
          a,
          parameters,
          output: {values: a}
        })
    ).toThrow(/share buffers/);
    expect(
      () =>
        new GPURasterArithmetic({
          cellCount: 12,
          a,
          noDataValue: NaN,
          parameters,
          output: {values: view(graph, 'float32', 12)}
        })
    ).toThrow(/noDataValue/);
  });
});

it('raster-algebra recipes return deterministic node IDs', () => {
  withGraph(graph => {
    const reclassify = new GPURasterReclassify({
      id: 'r',
      values: view(graph, 'float32', 12),
      breaks: view(graph, 'float32', 4),
      parameters: view(graph, 'float32', 4),
      output: {classes: view(graph, 'uint32', 12), classCounts: view(graph, 'uint32', 5)}
    });
    expect(reclassify.getCommandNodes(graph).map(node => node.id)).toEqual([
      'r-clear-counts',
      'r-classify'
    ]);
    const overlay = new GPUWeightedOverlay({
      id: 'o',
      stack: view(graph, 'float32', 24),
      layerCount: 2,
      cellCount: 12,
      parameters: view(graph, 'float32', 24),
      output: {score: view(graph, 'float32', 12), scoreRange: view(graph, 'float32', 2)}
    });
    expect(overlay.getCommandNodes(graph).map(node => node.id)).toEqual([
      'o-clear-range',
      'o-score',
      'o-decode-range'
    ]);
    const statistics = new GPURasterCellStatistics({
      id: 's',
      stack: view(graph, 'float32', 24),
      layerCount: 2,
      cellCount: 12,
      parameters: view(graph, 'float32', 4),
      output: {majority: view(graph, 'float32', 12), minimum: view(graph, 'float32', 12)}
    });
    expect(statistics.getCommandNodes(graph).map(node => node.id)).toEqual([
      's-extremes',
      's-frequencies'
    ]);
  });
});

it('raster-algebra oracles agree with hand-computed cases', () => {
  expect(countBreaksBelow([1, 2, 2, 3], 0, 4, 2, false)).toBe(3);
  expect(countBreaksBelow([1, 2, 2, 3], 0, 4, 2, true)).toBe(1);
  const reclassified = reclassifyOnCPU(
    {
      values: Float32Array.from([0, 1, 2.5, NaN, 9]),
      breaks: Float32Array.from([1, 2, 3]),
      classValues: Float32Array.from([10, 20, 30, 40])
    },
    3,
    false
  );
  expect(Array.from(reclassified.classes)).toEqual([0, 1, 2, 0xffffffff, 3]);
  expect(Array.from(reclassified.classCounts)).toEqual([1, 1, 1, 1]);
  const overlay = weightedOverlayOnCPU(
    {stack: Float32Array.from([0, 5, 10, 10, 10, NaN]), layerCount: 2, cellCount: 3},
    getGPUWeightedOverlayParameterValues({
      layers: [
        {weight: 2, inputMin: 0, inputMax: 10},
        {weight: 1, inputMin: 0, inputMax: 10, invert: true}
      ],
      noDataPolicy: 'ignore',
      normalizeWeights: true
    })
  );
  expect(Array.from(overlay.score)).toEqual([0, Math.fround(1 / 3), 1]);
  const statistics = computeCellStatisticsOnCPU(Float32Array.from([1, 2, 2, 1, NaN, 2]), 3, 2, {});
  expect(Array.from(statistics.majority)).toEqual([1, 2]);
  expect(Array.from(statistics.minority)).toEqual([1, 1]);
  expect(Array.from(statistics.variety)).toEqual([2, 2]);
  const conditional = conditionalOnCPU(
    3,
    {conditionValues: Float32Array.from([1, 5, NaN])},
    getGPURasterConditionalParameterValues({comparison: '>', threshold: 2, constantA: 9})
  );
  expect(Array.from(conditional.values)).toEqual([0, 9, NaN]);
  const rounded = arithmeticOnCPU(
    Float32Array.from([0.5, 1.5, 2.5, -0.5, -1.5]),
    undefined,
    getGPURasterArithmeticParameterValues({operation: 'round'})
  );
  expect(Array.from(rounded)).toEqual([0, 2, 2, -0, -2]);
});
