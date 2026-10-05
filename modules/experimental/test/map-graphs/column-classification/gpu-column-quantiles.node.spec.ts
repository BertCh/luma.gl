// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {GPUColumnQuantiles} from '../../../src/map-graphs/column-classification/gpu-column-quantiles';
import type {GPUColumnQuantilesProps} from '../../../src/map-graphs/column-classification/gpu-column-quantiles';
import {
  getGPUColumnQuantilesParameterLength,
  getGPUColumnQuantilesParameterValues,
  GPU_COLUMN_QUANTILE_INTERPOLATION_CODES,
  GPU_COLUMN_QUANTILES_PARAMETER_HEADER_LENGTH
} from '../../../src/map-graphs/column-classification/column-quantiles-parameters';
import {createNullWebGPUDevice, createVectorView} from '../map-graph-test-utils';
import {computeColumnQuantilesOracle, roundHalfToEven} from './column-quantiles-oracle';

let serial = 0;

function createProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUColumnQuantilesProps> = {}
): GPUColumnQuantilesProps {
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `view-${serial++}`, format, length);
  return {
    values: view('float32', 10),
    parameters: view('float32', getGPUColumnQuantilesParameterLength(3)),
    quantileCount: 3,
    output: {
      quantiles: view('float32', 3),
      validCount: view('uint32', 1),
      filterMask: view('uint32', 10),
      filterBounds: view('float32', 2)
    },
    ...overrides
  };
}

function expectThrows(
  overrides: (graph: GPUCommandGraph) => Partial<GPUColumnQuantilesProps>,
  message: RegExp
): void {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  expect(() => new GPUColumnQuantiles(createProps(graph, overrides(graph)))).toThrow(message);
  device.destroy();
}

it('GPUColumnQuantiles validates its inputs', () => {
  const view = <Format extends 'uint32' | 'float32'>(
    graph: GPUCommandGraph,
    format: Format,
    length: number
  ) => createTransientView(graph, `bad-${serial++}`, format, length);
  expectThrows(() => ({quantileCount: 0}), /quantileCount/);
  expectThrows(() => ({quantileCount: 65}), /quantileCount/);
  expectThrows(() => ({quantileCount: 1.5}), /quantileCount/);
  expectThrows(
    graph => ({
      values: createVectorView('chunks', 'float32', [view(graph, 'float32', 5)]) as never
    }),
    /single packed view/
  );
  expectThrows(graph => ({values: view(graph, 'float32', 0)}), /at least one row/);
  expectThrows(graph => ({values: view(graph, 'float32', 2 ** 24 + 1)}), /at most 16777216 rows/);
  expectThrows(graph => ({values: view(graph, 'uint32', 10) as never}), /values/);
  expectThrows(graph => ({mask: view(graph, 'float32', 10) as never}), /mask/);
  expectThrows(graph => ({mask: view(graph, 'uint32', 9)}), /mask length/);
  expectThrows(graph => ({parameters: view(graph, 'float32', 6)}), /parameters must hold 7/);
  expectThrows(graph => ({parameters: view(graph, 'uint32', 7) as never}), /parameters/);
  expectThrows(
    graph => ({
      output: {
        quantiles: view(graph, 'float32', 2),
        validCount: view(graph, 'uint32', 1)
      }
    }),
    /output.quantiles/
  );
  expectThrows(
    graph => ({
      output: {
        quantiles: view(graph, 'uint32', 3) as never,
        validCount: view(graph, 'uint32', 1)
      }
    }),
    /output.quantiles/
  );
  expectThrows(
    graph => ({
      output: {
        quantiles: view(graph, 'float32', 3),
        validCount: view(graph, 'float32', 1) as never
      }
    }),
    /output.validCount/
  );
  expectThrows(
    graph => ({
      output: {
        quantiles: view(graph, 'float32', 3),
        validCount: view(graph, 'uint32', 1),
        filterMask: view(graph, 'uint32', 9)
      }
    }),
    /filterMask/
  );
  expectThrows(
    graph => ({
      output: {
        quantiles: view(graph, 'float32', 3),
        validCount: view(graph, 'uint32', 1),
        filterBounds: view(graph, 'float32', 1)
      }
    }),
    /filterBounds/
  );
});

it('GPUColumnQuantiles rejects outputs that alias inputs and foreign graphs', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const props = createProps(graph);
  expect(
    () =>
      new GPUColumnQuantiles({
        ...props,
        output: {...props.output, quantiles: props.values}
      })
  ).toThrow(/share buffers/);
  expect(
    () =>
      new GPUColumnQuantiles({
        ...props,
        output: {
          ...props.output,
          validCount: createTransientView(graph, 'mask-alias', 'uint32', 1)
        },
        mask: createTransientView(graph, 'mask-source', 'uint32', 10)
      })
  ).not.toThrow();
  const otherGraph = new GPUCommandGraph(device);
  expect(() => new GPUColumnQuantiles(props).getCommandNodes(otherGraph)).toThrow(/target graph/);
  device.destroy();
});

it('GPUColumnQuantiles emits deterministic node IDs', () => {
  const device = createNullWebGPUDevice();
  const ids = (withFilter: boolean) => {
    const graph = new GPUCommandGraph(device);
    const props = createProps(graph, {id: 'cq'});
    if (!withFilter) {
      props.output = {
        quantiles: props.output.quantiles,
        validCount: props.output.validCount
      };
    }
    return new GPUColumnQuantiles(props).getCommandNodes(graph).map(node => node.id);
  };
  const expected = [
    'cq-init',
    ...[0, 1, 2, 3].flatMap(pass => [`cq-histogram-${pass}`, `cq-select-${pass}`]),
    'cq-finish'
  ];
  expect(ids(true)).toEqual([...expected, 'cq-filter-mask']);
  expect(ids(false)).toEqual(expected);
  expect(ids(true)).toEqual(ids(true));
  expect(new GPUColumnQuantiles(createProps(new GPUCommandGraph(device))).id).toBe(
    'column-quantiles'
  );
  device.destroy();
});

it('column quantile parameter helper packs the documented layout', () => {
  expect(GPU_COLUMN_QUANTILES_PARAMETER_HEADER_LENGTH).toBe(4);
  expect(getGPUColumnQuantilesParameterLength(5)).toBe(9);
  expect(GPU_COLUMN_QUANTILE_INTERPOLATION_CODES).toEqual({
    lower: 0,
    higher: 1,
    nearest: 2,
    linear: 3,
    midpoint: 4
  });
  const values = getGPUColumnQuantilesParameterValues({
    quantiles: [0.25, 0.5],
    interpolation: 'nearest',
    filterRange: [0.1, 0.9]
  });
  expect(Array.from(values)).toEqual([2, Math.fround(0.1), Math.fround(0.9), 0, 0.25, 0.5]);
  const padded = getGPUColumnQuantilesParameterValues({
    quantiles: [0.5],
    quantileCount: 3
  });
  expect(Array.from(padded.slice(0, 5))).toEqual([3, 0, 1, 0, 0.5]);
  expect(Number.isNaN(padded[5]) && Number.isNaN(padded[6])).toBe(true);
  const target = new Float32Array(10).fill(7);
  expect(getGPUColumnQuantilesParameterValues({quantiles: [1]}, target)).toBe(target);
  expect(target[4]).toBe(1);
  expect(() => getGPUColumnQuantilesParameterValues({quantiles: []})).toThrow(/count/);
  expect(() =>
    getGPUColumnQuantilesParameterValues({
      quantiles: [0.1, 0.2],
      quantileCount: 1
    })
  ).toThrow(/exceed/);
  expect(() =>
    getGPUColumnQuantilesParameterValues({quantiles: [0.1]}, new Float32Array(4))
  ).toThrow(/hold/);
  expect(() =>
    getGPUColumnQuantilesParameterValues({
      quantiles: [0.5],
      interpolation: 'bogus' as never
    })
  ).toThrow(/interpolation/);
});

it('column quantile oracle matches hand-computed numpy and d3 values', () => {
  const run = (
    values: number[],
    probability: number,
    interpolation: 'lower' | 'higher' | 'nearest' | 'linear' | 'midpoint'
  ) =>
    computeColumnQuantilesOracle({
      values,
      quantiles: [probability],
      interpolation
    }).quantiles[0];
  const four = [4, 1, 3, 2];
  // h = 1.5: numpy lower 2, higher 3, nearest rounds half to even to 2 (value 3), linear and midpoint 2.5.
  expect(run(four, 0.5, 'lower')).toBe(2);
  expect(run(four, 0.5, 'higher')).toBe(3);
  expect(run(four, 0.5, 'nearest')).toBe(3);
  expect(run(four, 0.5, 'linear')).toBe(2.5);
  expect(run(four, 0.5, 'midpoint')).toBe(2.5);
  // h = 0.75: d3.quantile([1,2,3,4], 0.25) = 1.75.
  expect(run(four, 0.25, 'linear')).toBe(1.75);
  expect(run(four, 0.25, 'nearest')).toBe(2);
  expect(run(four, 0.25, 'midpoint')).toBe(1.5);
  const five = [50, 10, 40, 20, 30];
  // h = 0.5 rounds to index 0, h = 1.5 rounds to index 2.
  expect(run(five, 0.125, 'nearest')).toBe(10);
  expect(run(five, 0.375, 'nearest')).toBe(30);
  expect(run(five, 0, 'linear')).toBe(10);
  expect(run(five, 1, 'linear')).toBe(50);
  expect(run(five, 0.5, 'higher')).toBe(30);
  expect([
    roundHalfToEven(0.5),
    roundHalfToEven(1.5),
    roundHalfToEven(2.5),
    roundHalfToEven(2.4)
  ]).toEqual([0, 2, 2, 2]);
  // Invalid probabilities and an empty column give NaN.
  expect(run(five, -0.1, 'linear')).toBeNaN();
  expect(run(five, 1.1, 'linear')).toBeNaN();
  expect(run(five, NaN, 'lower')).toBeNaN();
  expect(run([NaN, NaN], 0.5, 'linear')).toBeNaN();
});

it('column quantile oracle skips NaN and masked rows and filters by percentile', () => {
  const values = [5, NaN, 1, 9, 3, 7, 100, -0, 0];
  const mask = [1, 1, 1, 1, 1, 1, 0, 1, 1];
  const result = computeColumnQuantilesOracle({
    values,
    mask,
    quantiles: [0.5],
    interpolation: 'lower',
    filterRange: [0.25, 0.75]
  });
  // Valid sorted: -0, 0, 1, 3, 5, 7, 9 (n = 7); -0 sorts below +0.
  expect(result.validCount).toBe(7);
  expect(Object.is(result.sorted[0], -0) && Object.is(result.sorted[1], 0)).toBe(true);
  expect(result.quantiles[0]).toBe(3);
  // lowerIndex = floor(7 * 0.25) = 1, upperIndex = ceil(7 * 0.75) - 1 = 5.
  expect(Array.from(result.filterBounds)).toEqual([0, 7]);
  expect(Array.from(result.filterMask)).toEqual([1, 0, 1, 0, 1, 1, 0, 0, 1]);
  const empty = computeColumnQuantilesOracle({
    values: [1, 2],
    mask: [0, 0],
    quantiles: [0.5]
  });
  expect(empty.validCount).toBe(0);
  expect(Array.from(empty.filterMask)).toEqual([0, 0]);
  expect(empty.filterBounds.every(Number.isNaN)).toBe(true);
});
