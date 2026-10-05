// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer, submitGraph} from '../../../src/utils/gpu-contributor-utils';
import {
  getGPURasterReclassifyParameterValues,
  GPURasterReclassify,
  type GPURasterReclassifySettings
} from '../../../src/gpu-raster/raster-algebra';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32
} from '../../utils/gpu-contributor-test-utils';
import {reclassifyOnCPU, type ReclassifyScene} from './raster-algebra-oracle';
import {createRandom, expectFloatArraysClose} from './raster-algebra-test-utils';

it('GPURasterReclassify matches the CPU oracle as breaks and closure change without rebuilding', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const random = createRandom(7);
  const rowCount = 1531;
  const maximumBreakCount = 6;
  const values = new Float32Array(rowCount);
  const validity = new Uint32Array(rowCount);
  for (let row = 0; row < rowCount; row++) {
    const roll = random();
    // Integer-valued data lands exactly on breaks, which exercises the interval closure.
    values[row] =
      roll < 0.03
        ? NaN
        : roll < 0.06
          ? -9999
          : Math.floor(random() * 12) - 1 + (roll < 0.5 ? 0 : 0.5);
    validity[row] = random() < 0.04 ? 0 : 1;
  }
  const scene: ReclassifyScene = {
    values,
    validity,
    noDataValue: -9999,
    breaks: new Float32Array(maximumBreakCount),
    classValues: new Float32Array(maximumBreakCount + 1)
  };
  const breaksBuffer = createInputBuffer(device, scene.breaks);
  const classValuesBuffer = createInputBuffer(device, scene.classValues!);
  const parameterBuffer = new GPUParameterBuffer(device, {
    id: 'reclassify-parameters',
    format: 'float32',
    length: 4
  });
  const valuesBuffer = createInputBuffer(device, values);
  const validityBuffer = createInputBuffer(device, validity);
  const classesBuffer = createOutputBuffer(device, rowCount);
  const reclassifiedBuffer = createOutputBuffer(device, rowCount);
  const countsBuffer = createOutputBuffer(device, maximumBreakCount + 1);
  const graph = new GPUCommandGraph(device, {id: 'reclassify-graph'});
  graph.add(
    new GPURasterReclassify({
      id: 'reclassify',
      values: importGraphBuffer(graph, 'values', valuesBuffer, 'float32', rowCount),
      validity: importGraphBuffer(graph, 'validity', validityBuffer, 'uint32', rowCount),
      noDataValue: -9999,
      breaks: importGraphBuffer(graph, 'breaks', breaksBuffer, 'float32', maximumBreakCount),
      classValues: importGraphBuffer(
        graph,
        'class-values',
        classValuesBuffer,
        'float32',
        maximumBreakCount + 1
      ),
      parameters: parameterBuffer.importToGraph(graph),
      output: {
        classes: importGraphBuffer(graph, 'classes', classesBuffer, 'uint32', rowCount),
        reclassified: importGraphBuffer(
          graph,
          'reclassified',
          reclassifiedBuffer,
          'float32',
          rowCount
        ),
        classCounts: importGraphBuffer(
          graph,
          'counts',
          countsBuffer,
          'uint32',
          maximumBreakCount + 1
        )
      }
    })
  );
  const compiled = graph.compile();
  const frames: {breaks: number[]; classValues: number[]; settings: GPURasterReclassifySettings}[] =
    [
      {
        breaks: [0, 3, 5.5, 8, 10, 11],
        classValues: [10, 20, 30, 40, 50, 60, 70],
        settings: {breakCount: 6}
      },
      {
        breaks: [0, 3, 5.5, 8, 10, 11],
        classValues: [10, 20, 30, 40, 50, 60, 70],
        settings: {breakCount: 6, closed: 'right'}
      },
      {breaks: [2, 4, 6, 0, 0, 0], classValues: [NaN, 1, 2, 3, 0, 0, 0], settings: {breakCount: 3}},
      {breaks: [0, 0, 0, 0, 0, 0], classValues: [5, 0, 0, 0, 0, 0, 0], settings: {breakCount: 0}},
      // Counts above the compile-time maximum clamp to it.
      {breaks: [-1, 1, 2, 3, 4, 5], classValues: [0, 1, 2, 3, 4, 5, 6], settings: {breakCount: 99}}
    ];
  for (const frame of frames) {
    scene.breaks.set(frame.breaks);
    scene.classValues!.set(frame.classValues);
    breaksBuffer.write(scene.breaks);
    classValuesBuffer.write(scene.classValues!);
    parameterBuffer.write(getGPURasterReclassifyParameterValues(frame.settings));
    submitGraph(device, compiled, undefined);
    const expected = reclassifyOnCPU(
      scene,
      frame.settings.breakCount,
      frame.settings.closed === 'right'
    );
    expect(await readUint32(classesBuffer, rowCount)).toEqual(Array.from(expected.classes));
    expect(await readUint32(countsBuffer, maximumBreakCount + 1)).toEqual(
      Array.from(expected.classCounts)
    );
    expectFloatArraysClose(
      await readFloat32(reclassifiedBuffer, rowCount),
      expected.reclassified,
      0,
      'reclassified'
    );
  }
  // Closure matters on exact break values: 3 is class 2 left-closed, class 1 right-closed.
  const closureScene = {...scene, values: Float32Array.from([3])};
  closureScene.breaks.set([0, 3, 5.5, 8, 10, 11]);
  expect(reclassifyOnCPU(closureScene, 6, false).classes[0]).toBe(2);
  expect(reclassifyOnCPU(closureScene, 6, true).classes[0]).toBe(1);
  compiled.destroy();
  parameterBuffer.destroy();
  for (const buffer of [
    breaksBuffer,
    classValuesBuffer,
    valuesBuffer,
    validityBuffer,
    classesBuffer,
    reclassifiedBuffer,
    countsBuffer
  ]) {
    buffer.destroy();
  }
});
