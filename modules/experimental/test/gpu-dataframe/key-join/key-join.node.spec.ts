// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {GPUKeyJoin, type GPUKeyJoinProps} from '../../../src/gpu-dataframe/key-join';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {decodeFixedPointSum, getScaledValue, joinOnCPU, packKeys} from './key-join-oracle';

let serial = 0;

function createProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUKeyJoinProps> = {},
  keyFormat: 'uint32' | 'uint32x2' = 'uint32'
): GPUKeyJoinProps {
  const view = <Format extends 'uint32' | 'float32' | 'uint32x2'>(format: Format, length: number) =>
    createTransientView(graph, `view-${serial++}`, format, length);
  return {
    leftKeys: view(keyFormat, 8),
    rightKeys: view(keyFormat, 6),
    output: {matched: view('uint32', 8)},
    ...overrides
  } as GPUKeyJoinProps;
}

it('joinOnCPU oracle joins duplicates, masks, reserved keys and aggregates', () => {
  const result = joinOnCPU({
    keyBits: 32,
    leftKeys: [1n, 2n, 3n, 0xffffffffn, 1n, 2n],
    rightKeys: [2n, 1n, 2n, 0xffffffffn, 1n],
    rightMask: [1, 1, 1, 1, 0],
    leftMask: [1, 1, 1, 1, 1, 0],
    kind: 'inner',
    capacity: 2,
    sumScale: 1,
    gatherColumns: [{words: Uint32Array.from([10, 11, 12, 13, 14]), isFloat: false}],
    aggregates: [
      {operation: 'sum', values: Float32Array.from([1.5, 2, 3, 4, 5])},
      {operation: 'count'}
    ]
  });
  expect(result.rightRows).toEqual([1, 0, 0xffffffff, 0xffffffff, 1, 0xffffffff]);
  expect(result.matchCounts).toEqual([1, 2, 0, 0, 1, 0]);
  expect(result.matched).toEqual([1, 1, 0, 0, 1, 0]);
  expect(result.rightMatched).toEqual([1, 1, 1, 0, 0]);
  expect(result.gathered[0]).toEqual([11, 10, 0xffffffff, 0xffffffff, 11, 0xffffffff]);
  expect(result.innerRows).toEqual([0, 1]);
  expect(result.innerTotal).toBe(3);
  expect(result.innerOverflow).toBe(1);
  expect(result.aggregates[1].values).toEqual([1, 2, 0, 0, 1, 0]);
  // sumScale 1 rounds half to even: round(1.5) = 2, plus 3 = 5.
  expect(result.aggregates[0].sums).toEqual([2n, 5n, 0n, 0n, 2n, 0n]);
  expect(result.aggregates[0].values.slice(0, 2)).toEqual([2, 5]);
  expect(Number.isNaN(result.aggregates[0].values[2])).toBe(true);
});

it('oracle helpers scale, decode and pack', () => {
  expect(getScaledValue(2.5, 1)).toBe(2n);
  expect(getScaledValue(-0.25, 4)).toBe(-1n);
  expect(decodeFixedPointSum(-3n * 65536n, 65536)).toBe(-3);
  expect(decodeFixedPointSum(2n ** 40n, 1)).toBe(2 ** 40);
  expect(Array.from(packKeys([0x100000002n], 64))).toEqual([2, 1]);
  expect(Array.from(packKeys([7n], 32))).toEqual([7]);
});

it('GPUKeyJoin validates its props', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {
    id: 'key-join-validation'
  });
  const view = <Format extends 'uint32' | 'float32' | 'uint32x2'>(format: Format, length: number) =>
    createTransientView(graph, `v-${serial++}`, format, length);
  expect(() => new GPUKeyJoin(createProps(graph)).getCommandNodes(graph)).not.toThrow();
  expect(new GPUKeyJoin(createProps(graph)).recipe).toBe('key-join');
  expect(() => new GPUKeyJoin(createProps(graph, {sumScale: -1}))).toThrow(/sumScale/);
  expect(
    () =>
      new GPUKeyJoin(
        createProps(graph, {
          rightKeys: view('uint32x2', 6)
        } as Partial<GPUKeyJoinProps>)
      )
  ).toThrow(/rightKeys/);
  expect(() => new GPUKeyJoin(createProps(graph, {leftMask: view('uint32', 3)}))).toThrow(
    /leftMask length/
  );
  expect(() => new GPUKeyJoin(createProps(graph, {output: {matched: view('uint32', 3)}}))).toThrow(
    /output.matched length/
  );
  expect(
    () => new GPUKeyJoin(createProps(graph, {output: {rightMatched: view('uint32', 8)}}))
  ).toThrow(/rightMatched length/);
  // Inner needs rows, left refuses them.
  expect(() => new GPUKeyJoin(createProps(graph, {kind: 'inner'}))).toThrow(/rows is required/);
  const rows = {
    ids: view('uint32', 4),
    count: view('uint32', 1),
    overflow: view('uint32', 1)
  };
  expect(() => new GPUKeyJoin(createProps(graph, {output: {rows}}))).toThrow(/only supported/);
  expect(() => new GPUKeyJoin(createProps(graph, {kind: 'inner', output: {rows}}))).not.toThrow();
  // Gather columns.
  expect(
    () =>
      new GPUKeyJoin(
        createProps(graph, {
          gather: [{column: view('float32', 6), output: view('uint32', 8)}]
        })
      )
  ).toThrow(/gather\[0\]\.output/);
  expect(
    () =>
      new GPUKeyJoin(
        createProps(graph, {
          gather: [{column: view('float32', 5), output: view('float32', 8)}]
        })
      )
  ).toThrow(/column length/);
  // Aggregates.
  expect(
    () =>
      new GPUKeyJoin(
        createProps(graph, {
          aggregates: [{operation: 'sum', output: view('float32', 8)}]
        })
      )
  ).toThrow(/needs a column/);
  expect(
    () =>
      new GPUKeyJoin(
        createProps(graph, {
          aggregates: [
            {
              operation: 'maximum',
              column: view('float32', 6),
              output: view('float32', 8),
              sums: view('uint32x2', 8)
            }
          ]
        })
      )
  ).toThrow(/sum or mean/);
  // Nodes are deterministic and prefixed.
  const props = createProps(graph, {
    gather: [{column: view('float32', 6), output: view('float32', 8)}],
    aggregates: [
      {
        operation: 'mean',
        column: view('float32', 6),
        output: view('float32', 8)
      }
    ]
  });
  const first = new GPUKeyJoin({...props, id: 'a'}).getCommandNodes(graph).map(node => node.id);
  expect(first.every(id => id.startsWith('a-'))).toBe(true);
  expect(first).toContain('a-probe');
});
