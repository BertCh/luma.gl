// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer, submitGraph} from '../../../src/utils/gpu-contributor-utils';
import {
  getGPUWeightedOverlayParameterLength,
  getGPUWeightedOverlayParameterValues,
  GPUWeightedOverlay,
  type GPUWeightedOverlaySettings
} from '../../../src/gpu-raster/raster-algebra';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32
} from '../../utils/gpu-contributor-test-utils';
import {weightedOverlayOnCPU, type WeightedOverlayScene} from './raster-algebra-oracle';
import {createRandom, expectFloatArraysClose} from './raster-algebra-test-utils';

/**
 * Scores accumulate `sum + weight * remapped` in fixed layer order, so one device always returns
 * the same bits, but backends may contract the multiply-add into an FMA (Metal does), which
 * changes the rounding against the unfused oracle. Each step can then differ by one rounding of
 * the running sum, so the bound is `(layerCount + 2) * 2^-24 * sum(|weight * remapped|)` (plus the
 * WGSL division, within 2.5 ULP, when normalized).
 */
function expectScoresClose(
  actual: number[],
  expected: {score: Float32Array; magnitude: Float64Array},
  layerCount: number
): void {
  for (let cell = 0; cell < expected.score.length; cell++) {
    const gpu = actual[cell];
    const cpu = expected.score[cell];
    if (Number.isNaN(cpu) || Number.isNaN(gpu)) {
      expect(Number.isNaN(gpu), `score[${cell}] nodata`).toBe(Number.isNaN(cpu));
      continue;
    }
    const tolerance =
      (layerCount + 2) * 2 ** -24 * expected.magnitude[cell] + 4 * 2 ** -24 * Math.abs(cpu);
    expect(Math.abs(gpu - cpu), `score[${cell}] gpu ${gpu} cpu ${cpu}`).toBeLessThanOrEqual(
      tolerance
    );
  }
}

it('GPUWeightedOverlay matches the CPU oracle as weights, remaps, and policies change without rebuilding', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const random = createRandom(11);
  const layerCount = 16;
  const cellCount = 997;
  const maximumBreakCount = 4;
  const stack = new Float32Array(layerCount * cellCount);
  for (let index = 0; index < stack.length; index++) {
    const roll = random();
    stack[index] = roll < 0.01 ? NaN : roll < 0.02 ? -1 : Math.round(random() * 1000) / 10;
  }
  const remapBreaks = new Float32Array(layerCount * maximumBreakCount);
  const remapValues = new Float32Array(layerCount * (maximumBreakCount + 1));
  for (let layer = 0; layer < layerCount; layer++) {
    remapBreaks.set([20, 40, 60, 80], layer * maximumBreakCount);
    remapValues.set(
      layer === 3 ? [NaN, 1, 2, 3, 4] : [1, 2, 3, 4, 5],
      layer * (maximumBreakCount + 1)
    );
  }
  const scene: WeightedOverlayScene = {
    stack,
    layerCount,
    cellCount,
    noDataValue: -1,
    remapBreaks,
    remapValues,
    maximumBreakCount
  };
  const parameterBuffer = new GPUParameterBuffer(device, {
    id: 'overlay-parameters',
    format: 'float32',
    length: getGPUWeightedOverlayParameterLength(layerCount)
  });
  const stackBuffer = createInputBuffer(device, stack);
  const breaksBuffer = createInputBuffer(device, remapBreaks);
  const valuesBuffer = createInputBuffer(device, remapValues);
  const scoreBuffer = createOutputBuffer(device, cellCount);
  const validityBuffer = createOutputBuffer(device, cellCount);
  const rangeBuffer = createOutputBuffer(device, 2);
  const graph = new GPUCommandGraph(device, {id: 'overlay-graph'});
  graph.add(
    new GPUWeightedOverlay({
      id: 'overlay',
      stack: importGraphBuffer(graph, 'stack', stackBuffer, 'float32', stack.length),
      layerCount,
      cellCount,
      noDataValue: -1,
      parameters: parameterBuffer.importToGraph(graph),
      remapBreaks: importGraphBuffer(graph, 'breaks', breaksBuffer, 'float32', remapBreaks.length),
      remapValues: importGraphBuffer(graph, 'values', valuesBuffer, 'float32', remapValues.length),
      maximumBreakCount,
      output: {
        score: importGraphBuffer(graph, 'score', scoreBuffer, 'float32', cellCount),
        validity: importGraphBuffer(graph, 'validity', validityBuffer, 'uint32', cellCount),
        scoreRange: importGraphBuffer(graph, 'range', rangeBuffer, 'float32', 2)
      }
    })
  );
  const compiled = graph.compile();
  const createLayers = (
    pick: (layer: number) => GPUWeightedOverlaySettings['layers'][number]
  ): GPUWeightedOverlaySettings['layers'] =>
    Array.from({length: layerCount}, (_, layer) => pick(layer));
  const frames: GPUWeightedOverlaySettings[] = [
    // Linear only, no restricted layer used in table mode, propagate nodata.
    {layers: createLayers(layer => ({weight: (layer % 5) - 1.5, inputMin: 10, inputMax: 90}))},
    // Mixed linear/inverted/table layers, layer 3 has a restricted lowest class.
    {
      layers: createLayers(layer =>
        layer % 3 === 0
          ? {weight: 0.25 * layer, mode: 'table', breakCount: layer === 6 ? 2 : 4}
          : {weight: 1 + layer, inputMin: 100, inputMax: 0, invert: layer % 2 === 0}
      ),
      noDataPolicy: 'ignore'
    },
    {
      layers: createLayers(layer => ({weight: layer + 1, inputMin: 30, inputMax: 30})),
      normalizeWeights: true,
      noDataPolicy: 'ignore'
    },
    {
      layers: createLayers(layer =>
        layer < 8
          ? {weight: 2, mode: 'table', breakCount: 4, closed: 'right'}
          : {weight: -0.5, inputMin: 0, inputMax: 100}
      ),
      normalizeWeights: true
    }
  ];
  for (const settings of frames) {
    const parameters = getGPUWeightedOverlayParameterValues(settings);
    parameterBuffer.write(parameters);
    submitGraph(device, compiled, undefined);
    const expected = weightedOverlayOnCPU(scene, parameters);
    const score = await readFloat32(scoreBuffer, cellCount);
    expectScoresClose(score, expected, layerCount);
    expect(await readUint32(validityBuffer, cellCount)).toEqual(Array.from(expected.validity));
    // The range is the exact min/max of the GPU's own scores.
    const definedScores = score.filter(value => !Number.isNaN(value));
    expectFloatArraysClose(
      await readFloat32(rangeBuffer, 2),
      [Math.min(...definedScores), Math.max(...definedScores)],
      0,
      'range'
    );
    expect(expected.validity.some(value => value === 1)).toBe(true);
  }
  // The restricted class (layer 3, values below 20) removes cells whatever the policy.
  const restricted = weightedOverlayOnCPU(scene, getGPUWeightedOverlayParameterValues(frames[1]));
  const restrictedCell = Array.from({length: cellCount}, (_, cell) => cell).find(
    cell => stack[3 * cellCount + cell] < 20 && stack[3 * cellCount + cell] >= 0
  )!;
  expect(Number.isNaN(restricted.score[restrictedCell])).toBe(true);
  compiled.destroy();
  parameterBuffer.destroy();
  for (const buffer of [
    stackBuffer,
    breaksBuffer,
    valuesBuffer,
    scoreBuffer,
    validityBuffer,
    rangeBuffer
  ]) {
    buffer.destroy();
  }
});

it('GPUWeightedOverlay reports an empty range as NaN', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const cellCount = 5;
  const stack = Float32Array.from([NaN, NaN, NaN, NaN, NaN]);
  const parameterBuffer = new GPUParameterBuffer(device, {
    id: 'overlay-empty-parameters',
    format: 'float32',
    length: getGPUWeightedOverlayParameterLength(1),
    values: getGPUWeightedOverlayParameterValues({layers: [{weight: 1}]})
  });
  const stackBuffer = createInputBuffer(device, stack);
  const scoreBuffer = createOutputBuffer(device, cellCount);
  const rangeBuffer = createOutputBuffer(device, 2);
  const graph = new GPUCommandGraph(device, {id: 'overlay-empty-graph'});
  graph.add(
    new GPUWeightedOverlay({
      stack: importGraphBuffer(graph, 'stack', stackBuffer, 'float32', cellCount),
      layerCount: 1,
      cellCount,
      parameters: parameterBuffer.importToGraph(graph),
      output: {
        score: importGraphBuffer(graph, 'score', scoreBuffer, 'float32', cellCount),
        scoreRange: importGraphBuffer(graph, 'range', rangeBuffer, 'float32', 2)
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  expect((await readFloat32(scoreBuffer, cellCount)).every(Number.isNaN)).toBe(true);
  expect((await readFloat32(rangeBuffer, 2)).every(Number.isNaN)).toBe(true);
  compiled.destroy();
  parameterBuffer.destroy();
  for (const buffer of [stackBuffer, scoreBuffer, rangeBuffer]) {
    buffer.destroy();
  }
});
