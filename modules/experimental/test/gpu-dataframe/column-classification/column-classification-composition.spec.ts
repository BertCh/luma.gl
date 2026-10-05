// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {getGPUBivariateClassificationParameterValues} from '../../../src/gpu-dataframe/column-classification/bivariate-classification-parameters';
import {
  getGPUClassBreaksParameterLength,
  getGPUClassBreaksParameterValues,
  type GPUClassBreaksParameters
} from '../../../src/gpu-dataframe/column-classification/class-breaks-parameters';
import {
  getGPUColorScaleParameterValues,
  GPU_COLOR_SCALE_PARAMETER_LENGTH,
  packGPUColor
} from '../../../src/gpu-dataframe/column-classification/color-scale-parameters';
import {getOrderedFloat32Key} from '../../../src/gpu-dataframe/column-classification/column-classification-shared';
import {
  getGPUColumnQuantilesParameterLength,
  getGPUColumnQuantilesParameterValues
} from '../../../src/gpu-dataframe/column-classification/column-quantiles-parameters';
import {GPUBivariateClassification} from '../../../src/gpu-dataframe/column-classification/gpu-bivariate-classification';
import {GPUClassBreaks} from '../../../src/gpu-dataframe/column-classification/gpu-class-breaks';
import {GPUColorScale} from '../../../src/gpu-dataframe/column-classification/gpu-color-scale';
import {GPUColumnQuantiles} from '../../../src/gpu-dataframe/column-classification/gpu-column-quantiles';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {computeColumnQuantilesOracle} from './column-quantiles-oracle';

const NO_CLASS = 0xffffffff;
const MAXIMUM_CLASS_COUNT = 9;
const ROWS = 5000;

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

/** Class by the stream rule: the number of inner edges whose key is at or below the value's key. */
function classify(value: number, breaks: number[], classCount: number): number {
  const key = getOrderedFloat32Key(value);
  let classIndex = 0;
  for (let edge = 1; edge < classCount; edge++) {
    if (getOrderedFloat32Key(breaks[edge]) <= key) {
      classIndex++;
    }
  }
  return classIndex;
}

it('quantile filter, class breaks, colour scale and bivariate classes compose in one graph without readback', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const random = createRandom(31);
  const valuesX = new Float32Array(ROWS).map(() =>
    random() < 0.01 ? NaN : 10 / (0.02 + random())
  );
  const valuesY = new Float32Array(ROWS).map(() => random() * 100 - 20);
  const palette = new Uint32Array(MAXIMUM_CLASS_COUNT).map((_, index) =>
    packGPUColor(index * 20, 255 - index * 20, index * 7, 255)
  );
  const bivariatePalette = new Uint32Array(9).map((_, index) =>
    packGPUColor(index, index * 2, 3, 200)
  );
  const buffers: Buffer[] = [];
  const keep = (buffer: Buffer) => {
    buffers.push(buffer);
    return buffer;
  };
  const output = (length: number) => keep(createOutputBuffer(device, length));
  const graph = new GPUCommandGraph(device, {id: 'composition'});
  const parameterBuffers = {
    quantiles: new GPUParameterBuffer(device, {
      id: 'quantile-parameters',
      format: 'float32',
      length: getGPUColumnQuantilesParameterLength(1)
    }),
    breaks: new GPUParameterBuffer(device, {
      id: 'breaks-parameters',
      format: 'float32',
      length: getGPUClassBreaksParameterLength(MAXIMUM_CLASS_COUNT)
    }),
    axisX: new GPUParameterBuffer(device, {
      id: 'axis-x-parameters',
      format: 'float32',
      length: getGPUClassBreaksParameterLength(MAXIMUM_CLASS_COUNT),
      values: getGPUClassBreaksParameterValues(
        {method: 'quantile', classCount: 3},
        MAXIMUM_CLASS_COUNT
      )
    }),
    axisY: new GPUParameterBuffer(device, {
      id: 'axis-y-parameters',
      format: 'float32',
      length: getGPUClassBreaksParameterLength(MAXIMUM_CLASS_COUNT),
      values: getGPUClassBreaksParameterValues(
        {method: 'equal-interval', classCount: 3},
        MAXIMUM_CLASS_COUNT
      )
    }),
    colorScale: new GPUParameterBuffer(device, {
      id: 'color-scale-parameters',
      format: 'float32',
      length: GPU_COLOR_SCALE_PARAMETER_LENGTH,
      values: getGPUColorScaleParameterValues({
        scale: 'threshold',
        domainCount: 2,
        paletteCount: MAXIMUM_CLASS_COUNT,
        noDataColor: packGPUColor(1, 2, 3, 4)
      })
    }),
    bivariate: new GPUParameterBuffer(device, {
      id: 'bivariate-parameters',
      format: 'float32',
      length: 8,
      values: getGPUBivariateClassificationParameterValues({classCountX: 3, classCountY: 3})
    })
  };
  const x = importGraphBuffer(
    graph,
    'x',
    keep(createInputBuffer(device, valuesX)),
    'float32',
    ROWS
  );
  const y = importGraphBuffer(
    graph,
    'y',
    keep(createInputBuffer(device, valuesY)),
    'float32',
    ROWS
  );
  // The percentile filter mask is a graph transient: it never leaves the GPU.
  const filterMask = createTransientView(graph, 'filter-mask', 'uint32', ROWS);
  const outputs = {
    filterBounds: output(2),
    breaks: output(MAXIMUM_CLASS_COUNT + 1),
    classCount: output(1),
    breakClassCounts: output(MAXIMUM_CLASS_COUNT),
    colors: output(ROWS),
    classIndices: output(ROWS),
    scaleClassCounts: output(MAXIMUM_CLASS_COUNT),
    breaksX: output(MAXIMUM_CLASS_COUNT + 1),
    breaksY: output(MAXIMUM_CLASS_COUNT + 1),
    bivariateIds: output(ROWS),
    bivariateColors: output(ROWS)
  };
  const floatView = (name: keyof typeof outputs, length: number) =>
    importGraphBuffer(graph, `o-${name}`, outputs[name], 'float32', length);
  const uintView = (name: keyof typeof outputs, length: number) =>
    importGraphBuffer(graph, `o-${name}`, outputs[name], 'uint32', length);
  const breaksView = floatView('breaks', MAXIMUM_CLASS_COUNT + 1);
  const classCountView = uintView('classCount', 1);
  const breaksXView = floatView('breaksX', MAXIMUM_CLASS_COUNT + 1);
  const breaksYView = floatView('breaksY', MAXIMUM_CLASS_COUNT + 1);
  graph.add(
    new GPUColumnQuantiles({
      id: 'percentile-filter',
      values: x,
      parameters: parameterBuffers.quantiles.importToGraph(graph),
      quantileCount: 1,
      output: {
        quantiles: createTransientView(graph, 'unused-quantile', 'float32', 1),
        validCount: createTransientView(graph, 'unused-valid', 'uint32', 1),
        filterMask,
        filterBounds: floatView('filterBounds', 2)
      }
    })
  );
  graph.add(
    new GPUClassBreaks({
      id: 'breaks',
      values: x,
      mask: filterMask,
      parameters: parameterBuffers.breaks.importToGraph(graph),
      maximumClassCount: MAXIMUM_CLASS_COUNT,
      methods: ['quantile', 'natural-breaks', 'head-tail', 'equal-interval', 'standard-deviation'],
      naturalBreaksBinCount: 512,
      output: {
        breaks: breaksView,
        classCount: classCountView,
        classCounts: uintView('breakClassCounts', MAXIMUM_CLASS_COUNT)
      }
    })
  );
  graph.add(
    new GPUColorScale({
      id: 'scale',
      values: x,
      mask: filterMask,
      domain: breaksView,
      domainCount: classCountView,
      palette: importGraphBuffer(
        graph,
        'palette',
        keep(createInputBuffer(device, palette)),
        'uint32'
      ),
      parameters: parameterBuffers.colorScale.importToGraph(graph),
      maximumDomainCount: MAXIMUM_CLASS_COUNT + 1,
      maximumPaletteCount: MAXIMUM_CLASS_COUNT,
      output: {
        colors: uintView('colors', ROWS),
        classIndices: uintView('classIndices', ROWS),
        classCounts: uintView('scaleClassCounts', MAXIMUM_CLASS_COUNT)
      }
    })
  );
  for (const [axis, values, breaks] of [
    ['x', x, breaksXView],
    ['y', y, breaksYView]
  ] as const) {
    graph.add(
      new GPUClassBreaks({
        id: `axis-${axis}`,
        values,
        parameters: (axis === 'x' ? parameterBuffers.axisX : parameterBuffers.axisY).importToGraph(
          graph
        ),
        maximumClassCount: MAXIMUM_CLASS_COUNT,
        methods: ['quantile', 'equal-interval'],
        output: {breaks, classCount: createTransientView(graph, `axis-${axis}-count`, 'uint32', 1)}
      })
    );
  }
  graph.add(
    new GPUBivariateClassification({
      id: 'bivariate',
      valuesX: x,
      valuesY: y,
      breaksX: breaksXView,
      breaksY: breaksYView,
      palette: importGraphBuffer(
        graph,
        'bivariate-palette',
        keep(createInputBuffer(device, bivariatePalette)),
        'uint32'
      ),
      parameters: parameterBuffers.bivariate.importToGraph(graph),
      maximumClassCount: 3,
      output: {
        classIds: uintView('bivariateIds', ROWS),
        colors: uintView('bivariateColors', ROWS)
      }
    })
  );
  const compiled = graph.compile();
  try {
    const frames: {filterRange: [number, number]; breaks: GPUClassBreaksParameters}[] = [
      {filterRange: [0, 1], breaks: {method: 'quantile', classCount: 5}},
      {filterRange: [0.05, 0.95], breaks: {method: 'natural-breaks', classCount: 6}},
      {filterRange: [0.1, 0.8], breaks: {method: 'head-tail', classCount: 9}},
      {filterRange: [0, 0.5], breaks: {method: 'equal-interval', classCount: 4}},
      {filterRange: [0.2, 1], breaks: {method: 'standard-deviation', classCount: 7}}
    ];
    for (const frame of frames) {
      const label = `${frame.breaks.method} ${frame.filterRange}`;
      parameterBuffers.quantiles.write(
        getGPUColumnQuantilesParameterValues({quantiles: [0.5], filterRange: frame.filterRange})
      );
      parameterBuffers.breaks.write(
        getGPUClassBreaksParameterValues(frame.breaks, MAXIMUM_CLASS_COUNT)
      );
      submitGraph(device, compiled, undefined);
      const oracle = computeColumnQuantilesOracle({
        values: valuesX,
        quantiles: [0.5],
        filterRange: frame.filterRange
      });
      const [lower, upper] = await readFloat32(outputs.filterBounds, 2);
      expect([lower, upper], label).toEqual(Array.from(oracle.filterBounds));
      const breaks = await readFloat32(outputs.breaks, MAXIMUM_CLASS_COUNT + 1);
      const [classCount] = await readUint32(outputs.classCount, 1);
      expect(classCount, label).toBeGreaterThan(1);
      const classIndices = await readUint32(outputs.classIndices, ROWS);
      const colors = await readUint32(outputs.colors, ROWS);
      const expectedCounts = new Array(MAXIMUM_CLASS_COUNT).fill(0);
      for (let row = 0; row < ROWS; row++) {
        if (oracle.filterMask[row] === 0) {
          expect(classIndices[row], `${label} row ${row}`).toBe(NO_CLASS);
          expect(colors[row]).toBe(packGPUColor(1, 2, 3, 4));
          continue;
        }
        const classIndex = classify(valuesX[row], breaks, classCount);
        expect(classIndices[row], `${label} row ${row}`).toBe(classIndex);
        expect(colors[row]).toBe(palette[classIndex]);
        expectedCounts[classIndex]++;
      }
      // The colour scale's legend counts equal the class-breaks counts for the same mask.
      expect(await readUint32(outputs.scaleClassCounts, MAXIMUM_CLASS_COUNT), label).toEqual(
        expectedCounts
      );
      expect(await readUint32(outputs.breakClassCounts, MAXIMUM_CLASS_COUNT), label).toEqual(
        expectedCounts
      );
      const breaksX = await readFloat32(outputs.breaksX, 4);
      const breaksY = await readFloat32(outputs.breaksY, 4);
      const bivariateIds = await readUint32(outputs.bivariateIds, ROWS);
      for (let row = 0; row < ROWS; row++) {
        const expected = Number.isNaN(valuesX[row])
          ? NO_CLASS
          : classify(valuesY[row], breaksY, 3) * 3 + classify(valuesX[row], breaksX, 3);
        expect(bivariateIds[row], `${label} bivariate row ${row}`).toBe(expected);
      }
    }
  } finally {
    compiled.destroy();
    for (const parameterBuffer of Object.values(parameterBuffers)) {
      parameterBuffer.destroy();
    }
    for (const buffer of buffers) {
      buffer.destroy();
    }
  }
}, 120000);
