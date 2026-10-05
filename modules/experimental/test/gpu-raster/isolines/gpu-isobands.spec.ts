// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {GPUIsobands} from '../../../src/gpu-raster/isolines/gpu-isobands';
import {
  getGPUIsobandsParameterValues,
  type GPUIsobandsSettings
} from '../../../src/gpu-raster/isolines/isobands-parameters';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {createRandom} from '../raster-algebra/raster-algebra-test-utils';
import {
  buildIsobandTrianglesOnCPU,
  computeBandClassesOnCPU,
  type IsobandsScene
} from './isobands-oracle';

const WIDTH = 17;
const HEIGHT = 13;
const MAXIMUM_BREAK_COUNT = 6;
const CAPACITY = 4000;

function createScene(): IsobandsScene {
  const random = createRandom(21);
  const values = new Float32Array(WIDTH * HEIGHT);
  for (let sample = 0; sample < values.length; sample++) {
    const roll = random();
    // A mix of smooth noise, exact integers (equal to breaks), and nodata.
    values[sample] =
      roll < 0.04
        ? NaN
        : roll < 0.07
          ? -9999
          : roll < 0.4
            ? Math.floor(random() * 6)
            : Math.fround(random() * 6);
  }
  // Saddles of both kinds: (v0, v2) high / (v1, v3) low, with a high and a low centre.
  const placeSaddle = (column: number, row: number, high: number, low: number) => {
    values[row * WIDTH + column] = high;
    values[row * WIDTH + column + 1] = low;
    values[(row + 1) * WIDTH + column + 1] = high;
    values[(row + 1) * WIDTH + column] = low;
  };
  placeSaddle(2, 2, 4.5, 0.5);
  placeSaddle(5, 2, 3.2, 2.6);
  placeSaddle(8, 2, 5.5, 0.1);
  placeSaddle(2, 6, 0.4, 4.4);
  placeSaddle(10, 8, 2.2, 2.1);
  // Neighbouring samples shared with other saddles are overwritten; clear NaN/sentinel near them.
  return {
    width: WIDTH,
    height: HEIGHT,
    values,
    validity: Uint32Array.from({length: WIDTH * HEIGHT}, (_, i) => (i === 100 ? 0 : 1)),
    noDataValue: -9999,
    breaks: new Float32Array(MAXIMUM_BREAK_COUNT)
  };
}

it('GPUIsobands matches the CPU oracle as breaks, counts and windows change without rebuilding', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scene = createScene();
  const sampleCount = WIDTH * HEIGHT;
  const valuesBuffer = createInputBuffer(device, scene.values);
  const validityBuffer = createInputBuffer(device, scene.validity!);
  const breaksBuffer = createInputBuffer(device, scene.breaks);
  const parameterBuffer = new GPUParameterBuffer(device, {
    id: 'isobands-parameters',
    format: 'float32',
    length: 12
  });
  const classesBuffer = createOutputBuffer(device, sampleCount);
  const trianglesBuffer = createOutputBuffer(device, CAPACITY * 6);
  const bandsBuffer = createOutputBuffer(device, CAPACITY);
  const countBuffer = createOutputBuffer(device, 1);
  const overflowBuffer = createOutputBuffer(device, 1);
  const totalBuffer = createOutputBuffer(device, 1);
  const vertexCountBuffer = createOutputBuffer(device, 1);
  const graph = new GPUCommandGraph(device, {id: 'isobands-graph'});
  const contributor = new GPUIsobands({
    id: 'isobands',
    width: WIDTH,
    height: HEIGHT,
    values: importGraphBuffer(graph, 'values', valuesBuffer, 'float32', sampleCount),
    validity: importGraphBuffer(graph, 'validity', validityBuffer, 'uint32', sampleCount),
    noDataValue: -9999,
    breaks: importGraphBuffer(graph, 'breaks', breaksBuffer, 'float32', MAXIMUM_BREAK_COUNT),
    parameters: parameterBuffer.importToGraph(graph),
    output: {
      bandClasses: importGraphBuffer(graph, 'classes', classesBuffer, 'uint32', sampleCount),
      triangles: importGraphBuffer(graph, 'triangles', trianglesBuffer, 'float32x2', CAPACITY * 3),
      triangleBands: importGraphBuffer(graph, 'bands', bandsBuffer, 'uint32', CAPACITY),
      count: importGraphBuffer(graph, 'count', countBuffer, 'uint32', 1),
      overflow: importGraphBuffer(graph, 'overflow', overflowBuffer, 'uint32', 1),
      totalCount: importGraphBuffer(graph, 'total', totalBuffer, 'uint32', 1),
      vertexCount: importGraphBuffer(graph, 'vertex-count', vertexCountBuffer, 'uint32', 1)
    }
  });
  graph.add(contributor);
  let compileCount = 0;
  const compiled = graph.compile();
  compileCount++;
  const baseSettings = {width: WIDTH, height: HEIGHT, extent: [10, -5, 44, 21]} as const;
  const frames: {
    breaks: number[];
    settings: Omit<GPUIsobandsSettings, 'width' | 'height' | 'extent'>;
    extent?: GPUIsobandsSettings['extent'];
    capacity?: number;
  }[] = [
    {breaks: [1, 2, 3, 4, 5, 0], settings: {breakCount: 5}},
    {breaks: [0.5, 2.5, 4.5, 0, 0, 0], settings: {breakCount: 3}},
    {breaks: [2.5, 0, 0, 0, 0, 0], settings: {breakCount: 1}},
    {breaks: [1, 2, 3, 4, 5, 5.5], settings: {breakCount: 6, firstBand: 2, lastBand: 4}},
    {breaks: [1, 2, 3, 4, 5, 5.5], settings: {breakCount: 99}},
    // A degenerate equal pair makes band 2 empty; breakCount 0 is one whole-cell band.
    {breaks: [1, 3, 3, 4, 0, 0], settings: {breakCount: 4}},
    {breaks: [0, 0, 0, 0, 0, 0], settings: {breakCount: 0}},
    {breaks: [1, 2, 3, 4, 5, 0], settings: {breakCount: 5}, extent: [-3, 4, 11, 9]}
  ];
  let outputIndex = 0;
  for (const frame of frames) {
    scene.breaks.set(frame.breaks);
    breaksBuffer.write(scene.breaks);
    const settings: GPUIsobandsSettings = {
      ...baseSettings,
      extent: frame.extent ?? baseSettings.extent,
      ...frame.settings
    };
    const parameters = getGPUIsobandsParameterValues(settings);
    parameterBuffer.write(parameters);
    submitGraph(device, compiled, undefined);
    const label = `frame ${outputIndex++}`;
    const breakCount = Math.min(settings.breakCount, MAXIMUM_BREAK_COUNT);
    expect(await readUint32(classesBuffer, sampleCount), `${label} classes`).toEqual(
      Array.from(computeBandClassesOnCPU(scene, breakCount))
    );
    const expected = buildIsobandTrianglesOnCPU(scene, parameters, MAXIMUM_BREAK_COUNT, CAPACITY);
    const [count] = await readUint32(countBuffer, 1);
    expect(count, `${label} count`).toBe(expected.bands.length);
    expect(await readUint32(totalBuffer, 1), `${label} total`).toEqual([expected.totalCount]);
    expect(await readUint32(overflowBuffer, 1), `${label} overflow`).toEqual([0]);
    expect(await readUint32(vertexCountBuffer, 1), `${label} vertexCount`).toEqual([3 * count]);
    expect(count, `${label} nonempty`).toBeGreaterThan(0);
    expect(await readUint32(bandsBuffer, count), `${label} bands`).toEqual(expected.bands);
    const triangles = await readFloat32(trianglesBuffer, count * 6);
    let worst = 0;
    for (let i = 0; i < triangles.length; i++) {
      const scale = Math.max(1, Math.abs(expected.triangles[i]));
      worst = Math.max(worst, Math.abs(triangles[i] - expected.triangles[i]) / scale);
    }
    // Division is 2.5 ULP on GPUs and a*b+c may be contracted to FMA, so allow a few ULP of the
    // coordinate magnitude (2^-19 is 16 ULP; observed about 4).
    expect(worst, `${label} triangle error`).toBeLessThan(2 ** -19);
  }
  expect(compileCount - 1, 'rebuilds').toBe(0);

  // Capacity overflow: a smaller graph reuses the same oracle; totalCount stays unclamped.
  const smallCapacity = 40;
  const smallGraph = new GPUCommandGraph(device, {id: 'isobands-small'});
  smallGraph.add(
    new GPUIsobands({
      id: 'small',
      width: WIDTH,
      height: HEIGHT,
      values: importGraphBuffer(smallGraph, 'values', valuesBuffer, 'float32', sampleCount),
      noDataValue: -9999,
      breaks: importGraphBuffer(smallGraph, 'breaks', breaksBuffer, 'float32', MAXIMUM_BREAK_COUNT),
      parameters: parameterBuffer.importToGraph(smallGraph),
      output: {
        triangles: importGraphBuffer(
          smallGraph,
          'triangles',
          trianglesBuffer,
          'float32x2',
          smallCapacity * 3
        ),
        triangleBands: importGraphBuffer(smallGraph, 'bands', bandsBuffer, 'uint32', smallCapacity),
        count: importGraphBuffer(smallGraph, 'count', countBuffer, 'uint32', 1),
        overflow: importGraphBuffer(smallGraph, 'overflow', overflowBuffer, 'uint32', 1),
        totalCount: importGraphBuffer(smallGraph, 'total', totalBuffer, 'uint32', 1)
      }
    })
  );
  const smallCompiled = smallGraph.compile();
  scene.breaks.set([1, 2, 3, 4, 5, 0]);
  breaksBuffer.write(scene.breaks);
  const overflowParameters = getGPUIsobandsParameterValues({...baseSettings, breakCount: 5});
  parameterBuffer.write(overflowParameters);
  submitGraph(device, smallCompiled, undefined);
  // The small graph's scene has no validity buffer, so compare against a validity-free oracle.
  const noValidityScene = {...scene, validity: undefined};
  const expected = buildIsobandTrianglesOnCPU(
    noValidityScene,
    overflowParameters,
    MAXIMUM_BREAK_COUNT,
    smallCapacity
  );
  expect(expected.totalCount).toBeGreaterThan(smallCapacity);
  expect(await readUint32(countBuffer, 1)).toEqual([smallCapacity]);
  expect(await readUint32(overflowBuffer, 1)).toEqual([1]);
  expect(await readUint32(totalBuffer, 1)).toEqual([expected.totalCount]);
  expect(await readUint32(bandsBuffer, smallCapacity)).toEqual(expected.bands);

  smallCompiled.destroy();
  compiled.destroy();
  parameterBuffer.destroy();
  for (const buffer of [
    valuesBuffer,
    validityBuffer,
    breaksBuffer,
    classesBuffer,
    trianglesBuffer,
    bandsBuffer,
    countBuffer,
    overflowBuffer,
    totalBuffer,
    vertexCountBuffer
  ]) {
    buffer.destroy();
  }
});
