// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  getGPURasterArithmeticParameterValues,
  getGPURasterCellStatisticsParameterValues,
  getGPURasterConditionalParameterValues,
  GPURasterArithmetic,
  GPURasterCellStatistics,
  GPURasterConditional,
  type GPURasterArithmeticSettings,
  type GPURasterCellStatisticsSettings,
  type GPURasterConditionalSettings
} from '../../../src/gpu-raster/raster-algebra';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {
  arithmeticOnCPU,
  computeCellStatisticsOnCPU,
  conditionalOnCPU,
  type CellStatisticsResult
} from './raster-algebra-oracle';
import {createRandom, expectFloatArraysClose} from './raster-algebra-test-utils';

const FLOAT_STATISTICS = [
  'minimum',
  'maximum',
  'range',
  'sum',
  'mean',
  'standardDeviation',
  'majority',
  'minority'
] as const;

it('GPURasterCellStatistics matches the CPU oracle for 64 layers under both nodata policies', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const random = createRandom(3);
  const layerCount = 64;
  const cellCount = 517;
  const stack = new Float32Array(layerCount * cellCount);
  for (let index = 0; index < stack.length; index++) {
    const roll = random();
    // Few distinct values so majority/minority ties are common; some cells are all nodata.
    const cell = index % cellCount;
    stack[index] =
      cell % 97 === 0 || roll < 0.05 ? NaN : roll < 0.08 ? -9999 : Math.floor(random() * 7) - 3;
  }
  // Cell 1: only -0 and +0, which compare equal.
  for (let layer = 0; layer < layerCount; layer++) {
    stack[layer * cellCount + 1] = layer % 2 === 0 ? -0 : 0;
  }
  // Cell 2: all layers valid, so the propagate policy keeps it.
  for (let layer = 0; layer < layerCount; layer++) {
    stack[layer * cellCount + 2] = 1000 + layer * 0.25;
  }
  const parameterBuffer = new GPUParameterBuffer(device, {
    id: 'cell-statistics-parameters',
    format: 'float32',
    length: 4
  });
  const stackBuffer = createInputBuffer(device, stack);
  const outputBuffers = Object.fromEntries(
    [...FLOAT_STATISTICS, 'variety', 'count'].map(name => [
      name,
      createOutputBuffer(device, cellCount)
    ])
  ) as Record<keyof CellStatisticsResult, Buffer>;
  const graph = new GPUCommandGraph(device, {id: 'cell-statistics-graph'});
  const float = (name: (typeof FLOAT_STATISTICS)[number]) =>
    importGraphBuffer(graph, `out-${name}`, outputBuffers[name], 'float32', cellCount);
  graph.add(
    new GPURasterCellStatistics({
      id: 'cell-statistics',
      stack: importGraphBuffer(graph, 'stack', stackBuffer, 'float32', stack.length),
      layerCount,
      cellCount,
      noDataValue: -9999,
      parameters: parameterBuffer.importToGraph(graph),
      output: {
        minimum: float('minimum'),
        maximum: float('maximum'),
        range: float('range'),
        sum: float('sum'),
        mean: float('mean'),
        standardDeviation: float('standardDeviation'),
        majority: float('majority'),
        minority: float('minority'),
        variety: importGraphBuffer(
          graph,
          'out-variety',
          outputBuffers.variety,
          'uint32',
          cellCount
        ),
        count: importGraphBuffer(graph, 'out-count', outputBuffers.count, 'uint32', cellCount)
      }
    })
  );
  const compiled = graph.compile();
  const frames: GPURasterCellStatisticsSettings[] = [
    {},
    {noDataPolicy: 'propagate'},
    {minimumValidCount: 60},
    {noDataPolicy: 'ignore', minimumValidCount: 0}
  ];
  for (const settings of frames) {
    parameterBuffer.write(getGPURasterCellStatisticsParameterValues(settings));
    submitGraph(device, compiled, undefined);
    const expected = computeCellStatisticsOnCPU(stack, layerCount, cellCount, {
      noDataValue: -9999,
      propagateNoData: settings.noDataPolicy === 'propagate',
      minimumValidCount: settings.minimumValidCount
    });
    expect(await readUint32(outputBuffers.count, cellCount)).toEqual(Array.from(expected.count));
    expect(await readUint32(outputBuffers.variety, cellCount)).toEqual(
      Array.from(expected.variety)
    );
    for (const name of FLOAT_STATISTICS) {
      // Mean divides once (WGSL within 2.5 ULP); the deviation also takes a square root.
      const tolerance = name === 'mean' ? 3 : name === 'standardDeviation' ? 8 : 0;
      expectFloatArraysClose(
        await readFloat32(outputBuffers[name], cellCount),
        expected[name],
        tolerance,
        name
      );
    }
  }
  const allValid = computeCellStatisticsOnCPU(stack, layerCount, cellCount, {
    noDataValue: -9999,
    propagateNoData: true
  });
  expect(allValid.count[2]).toBe(layerCount);
  expect(allValid.variety[1]).toBe(1);
  compiled.destroy();
  parameterBuffer.destroy();
  stackBuffer.destroy();
  for (const buffer of Object.values(outputBuffers)) {
    buffer.destroy();
  }
});

it('GPURasterConditional evaluates comparisons, masks, views, and constants per frame', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const random = createRandom(5);
  const cellCount = 301;
  const conditionValues = Float32Array.from({length: cellCount}, (_, cell) =>
    cell % 23 === 0 ? NaN : cell % 29 === 0 ? -9999 : Math.floor(random() * 10)
  );
  const a = Float32Array.from({length: cellCount}, (_, cell) => (cell % 31 === 0 ? NaN : cell));
  const mask = Uint32Array.from({length: cellCount}, () => (random() < 0.5 ? 0 : 7));
  const parameterBuffer = new GPUParameterBuffer(device, {
    id: 'conditional-parameters',
    format: 'float32',
    length: 8
  });
  const conditionBuffer = createInputBuffer(device, conditionValues);
  const aBuffer = createInputBuffer(device, a);
  const maskBuffer = createInputBuffer(device, mask);
  const valuesBuffer = createOutputBuffer(device, cellCount);
  const maskOutBuffer = createOutputBuffer(device, cellCount);
  const maskValuesBuffer = createOutputBuffer(device, cellCount);
  const graph = new GPUCommandGraph(device, {id: 'conditional-graph'});
  graph.add(
    new GPURasterConditional({
      id: 'conditional-values',
      cellCount,
      conditionValues: importGraphBuffer(graph, 'condition', conditionBuffer, 'float32', cellCount),
      a: importGraphBuffer(graph, 'a', aBuffer, 'float32', cellCount),
      noDataValue: -9999,
      parameters: parameterBuffer.importToGraph(graph),
      output: {
        values: importGraphBuffer(graph, 'values', valuesBuffer, 'float32', cellCount),
        mask: importGraphBuffer(graph, 'mask-out', maskOutBuffer, 'uint32', cellCount)
      }
    })
  );
  graph.add(
    new GPURasterConditional({
      id: 'conditional-mask',
      cellCount,
      mask: importGraphBuffer(graph, 'mask', maskBuffer, 'uint32', cellCount),
      a: importGraphBuffer(graph, 'a-again', aBuffer, 'float32', cellCount),
      parameters: parameterBuffer.importToGraph(graph, 'conditional-parameters-again'),
      output: {
        values: importGraphBuffer(graph, 'mask-values', maskValuesBuffer, 'float32', cellCount)
      }
    })
  );
  const compiled = graph.compile();
  const frames: GPURasterConditionalSettings[] = [
    {comparison: '<', threshold: 4, constantB: -1},
    {comparison: '<=', threshold: 4},
    {comparison: '>', threshold: 6.5, constantB: 2.5},
    {comparison: '>=', threshold: 6},
    {comparison: '==', threshold: 3, constantB: 100},
    {comparison: '!=', threshold: 3},
    {comparison: 'between', threshold: 2, upperThreshold: 5, constantB: -7}
  ];
  for (const settings of frames) {
    const parameters = getGPURasterConditionalParameterValues(settings);
    parameterBuffer.write(parameters);
    submitGraph(device, compiled, undefined);
    const expected = conditionalOnCPU(
      cellCount,
      {conditionValues, a, noDataValue: -9999},
      parameters
    );
    expectFloatArraysClose(
      await readFloat32(valuesBuffer, cellCount),
      expected.values,
      0,
      'values'
    );
    expect(await readUint32(maskOutBuffer, cellCount)).toEqual(Array.from(expected.mask));
    const expectedMask = conditionalOnCPU(cellCount, {mask, a}, parameters);
    expectFloatArraysClose(
      await readFloat32(maskValuesBuffer, cellCount),
      expectedMask.values,
      0,
      'mask values'
    );
  }
  compiled.destroy();
  parameterBuffer.destroy();
  for (const buffer of [
    conditionBuffer,
    aBuffer,
    maskBuffer,
    valuesBuffer,
    maskOutBuffer,
    maskValuesBuffer
  ]) {
    buffer.destroy();
  }
});

/**
 * Tolerances: add, subtract, multiply, minimum, maximum, absolute value, absolute difference,
 * floor, ceil, and round are exact when the operand scales are 1 and offsets 0 (no FMA can form).
 * Division and normalized difference allow 3 ULP (WGSL `/` is within 2.5 ULP), square root 3 ULP.
 * WGSL `pow` is `exp2(y * log2(x))` and `exp` has `3 + 2 |x|` ULP error, so both are compared with
 * a `1e-5` relative bound; logarithm near 1 has a large relative but tiny absolute error, so it
 * allows `4e-7` absolute.
 */
const OPERATION_TOLERANCES: Partial<Record<GPURasterArithmeticSettings['operation'], number>> = {
  divide: 3,
  normalizedDifference: 3,
  squareRoot: 3
};
const RELATIVE_TOLERANCE_OPERATIONS = new Set(['power', 'exponential']);

it('GPURasterArithmetic switches operations per frame without rebuilding', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const random = createRandom(9);
  const cellCount = 409;
  const a = Float32Array.from({length: cellCount}, (_, cell) =>
    cell % 37 === 0 ? NaN : cell % 41 === 0 ? -9999 : cell % 13 === 0 ? 0 : (random() - 0.3) * 20
  );
  const b = Float32Array.from({length: cellCount}, (_, cell) =>
    cell % 43 === 0 ? NaN : cell % 11 === 0 ? 0 : cell % 17 === 0 ? -a[cell] : (random() - 0.5) * 4
  );
  // Exact halves exercise round-half-to-even.
  for (let cell = 0; cell < 8; cell++) {
    a[100 + cell] = cell - 3.5;
  }
  const parameterBuffer = new GPUParameterBuffer(device, {
    id: 'arithmetic-parameters',
    format: 'float32',
    length: 8
  });
  const aBuffer = createInputBuffer(device, a);
  const bBuffer = createInputBuffer(device, b);
  const outBuffer = createOutputBuffer(device, cellCount);
  const constantOutBuffer = createOutputBuffer(device, cellCount);
  const graph = new GPUCommandGraph(device, {id: 'arithmetic-graph'});
  graph.add(
    new GPURasterArithmetic({
      id: 'arithmetic',
      cellCount,
      a: importGraphBuffer(graph, 'a', aBuffer, 'float32', cellCount),
      b: importGraphBuffer(graph, 'b', bBuffer, 'float32', cellCount),
      noDataValue: -9999,
      parameters: parameterBuffer.importToGraph(graph),
      output: {values: importGraphBuffer(graph, 'out', outBuffer, 'float32', cellCount)}
    })
  );
  graph.add(
    new GPURasterArithmetic({
      id: 'arithmetic-constant',
      cellCount,
      a: importGraphBuffer(graph, 'a-again', aBuffer, 'float32', cellCount),
      noDataValue: -9999,
      parameters: parameterBuffer.importToGraph(graph, 'arithmetic-parameters-again'),
      output: {
        values: importGraphBuffer(graph, 'constant-out', constantOutBuffer, 'float32', cellCount)
      }
    })
  );
  const compiled = graph.compile();
  const operations: GPURasterArithmeticSettings['operation'][] = [
    'add',
    'subtract',
    'multiply',
    'divide',
    'minimum',
    'maximum',
    'power',
    'absoluteDifference',
    'normalizedDifference',
    'absolute',
    'squareRoot',
    'logarithm',
    'exponential',
    'floor',
    'ceil',
    'round'
  ];
  for (const operation of operations) {
    for (const settings of [
      {operation, constantB: 1.5},
      {operation, constantB: 2, clampMin: -1, clampMax: 3}
    ] as GPURasterArithmeticSettings[]) {
      const parameters = getGPURasterArithmeticParameterValues(settings);
      parameterBuffer.write(parameters);
      submitGraph(device, compiled, undefined);
      const tolerance = OPERATION_TOLERANCES[operation] ?? 0;
      for (const [buffer, expected] of [
        [outBuffer, arithmeticOnCPU(a, b, parameters, -9999)],
        [constantOutBuffer, arithmeticOnCPU(a, undefined, parameters, -9999)]
      ] as const) {
        const actual = await readFloat32(buffer, cellCount);
        if (RELATIVE_TOLERANCE_OPERATIONS.has(operation)) {
          for (let cell = 0; cell < cellCount; cell++) {
            if (!Number.isNaN(expected[cell])) {
              expect(
                Math.abs(actual[cell] - expected[cell]),
                `${operation}[${cell}]`
              ).toBeLessThanOrEqual(1e-5 * Math.abs(expected[cell]));
            } else {
              expect(Number.isNaN(actual[cell])).toBe(true);
            }
          }
        } else if (operation === 'logarithm') {
          for (let cell = 0; cell < cellCount; cell++) {
            if (!Number.isNaN(expected[cell])) {
              expect(Math.abs(actual[cell] - expected[cell]), `log[${cell}]`).toBeLessThanOrEqual(
                4e-7 + 3 * 2 ** -23 * Math.abs(expected[cell])
              );
            } else {
              expect(Number.isNaN(actual[cell])).toBe(true);
            }
          }
        } else {
          expectFloatArraysClose(actual, expected, tolerance, operation);
        }
      }
    }
  }
  // Affine operand scales (may form an FMA, so compared with a small tolerance).
  const scaled = getGPURasterArithmeticParameterValues({
    operation: 'normalizedDifference',
    scaleA: 0.5,
    offsetA: 3,
    scaleB: 2,
    offsetB: -1
  });
  parameterBuffer.write(scaled);
  submitGraph(device, compiled, undefined);
  const scaledActual = await readFloat32(outBuffer, cellCount);
  const scaledExpected = arithmeticOnCPU(a, b, scaled, -9999);
  for (let cell = 0; cell < cellCount; cell++) {
    if (Number.isNaN(scaledExpected[cell])) {
      expect(Number.isNaN(scaledActual[cell])).toBe(true);
    } else {
      expect(Math.abs(scaledActual[cell] - scaledExpected[cell])).toBeLessThanOrEqual(1e-4);
    }
  }
  compiled.destroy();
  parameterBuffer.destroy();
  for (const buffer of [aBuffer, bBuffer, outBuffer, constantOutBuffer]) {
    buffer.destroy();
  }
});
