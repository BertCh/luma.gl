// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCellTable} from '../../src/gpu-spatial-analysis/cell-aggregation';
import {importGraphBuffer} from '../../src/utils/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32
} from '../utils/gpu-contributor-test-utils';
import {compareParity} from './parity-wp2-assert';

export type CellTableFixture = {
  cells: number[];
  counts: number[];
  sums: number[];
  sumValues: number[];
  minimums: number[];
  maximums: number[];
  count: number;
  requiredCount: number;
  overflow: number;
};
export type CellsExpected = {
  name: string;
  reference: string;
  capacity: number;
  frames: {name: string; levels: CellTableFixture[]}[];
  boundedFrames: {name: string; levels: CellTableFixture[]}[];
};
export type CellsInput = {keys: number[]; values: number[]; masks: number[][]};
export type CellsComparison = {
  name: string;
  reference: string;
  comparisons: {
    before: number;
    after: number;
    capacity: number;
    measure: 'count' | 'sum';
    expected: {
      cells: number[];
      presence: number[];
      before: number[];
      after: number[];
      delta: number[];
      ratio: number[];
      percentChange: number[];
      zScore: number[];
      count: number;
      requiredCount: number;
      overflow: number;
    };
  }[];
};

export function createCellRig(device: Device) {
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  function input<Format extends 'uint32' | 'uint32x2' | 'float32'>(
    name: string,
    values: number[],
    format: Format
  ) {
    const buffer = createInputBuffer(
      device,
      format === 'float32' ? Float32Array.from(values) : Uint32Array.from(values)
    );
    buffers.push(buffer);
    return {
      buffer,
      view: importGraphBuffer(
        graph,
        name,
        buffer,
        format,
        values.length / (format === 'uint32x2' ? 2 : 1)
      )
    };
  }
  function output<Format extends 'uint32' | 'uint32x2' | 'float32'>(
    name: string,
    length: number,
    format: Format
  ) {
    const buffer = createOutputBuffer(device, length * (format === 'uint32x2' ? 2 : 1));
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, name, buffer, format, length)};
  }
  function table(name: string, capacity: number) {
    const columns = {
      cells: output(`${name}-cells`, capacity, 'uint32x2'),
      counts: output(`${name}-counts`, capacity, 'uint32'),
      sums: output(`${name}-sums`, capacity, 'uint32x2'),
      sumValues: output(`${name}-values`, capacity, 'float32'),
      minimums: output(`${name}-minimums`, capacity, 'float32'),
      maximums: output(`${name}-maximums`, capacity, 'float32'),
      count: output(`${name}-count`, 1, 'uint32'),
      requiredCount: output(`${name}-required`, 1, 'uint32'),
      overflow: output(`${name}-overflow`, 1, 'uint32')
    };
    const view: GPUCellTable = {
      cells: columns.cells.view,
      counts: columns.counts.view,
      sums: columns.sums.view,
      sumValues: columns.sumValues.view,
      minimums: columns.minimums.view,
      maximums: columns.maximums.view,
      status: {
        count: columns.count.view,
        requiredCount: columns.requiredCount.view,
        overflow: columns.overflow.view
      }
    };
    return {columns, view, capacity};
  }
  return {
    graph,
    input,
    output,
    table,
    destroy() {
      for (const buffer of buffers) buffer.destroy();
    }
  };
}

export type CellTableBuffers = ReturnType<ReturnType<typeof createCellRig>['table']>;

export async function compareCellTable(
  className: string,
  label: string,
  reference: string,
  table: CellTableBuffers,
  expected: CellTableFixture
) {
  const compare = (
    metric: string,
    actual: number[],
    wanted: number[],
    absolute = 0,
    relative = 0
  ) => compareParity(className, label, reference, metric, actual, wanted, absolute, relative);
  for (const field of ['count', 'requiredCount', 'overflow'] as const)
    compare(field, await readUint32(table.columns[field].buffer, 1), [expected[field]]);
  for (const field of ['cells', 'counts', 'sums'] as const) {
    const components = field === 'counts' ? 1 : 2;
    const tail = Array((table.capacity - expected.count) * components).fill(
      field === 'cells' ? 0xffffffff : 0
    );
    compare(
      `every ${field} and tail`,
      await readUint32(table.columns[field].buffer, table.capacity * components),
      [...expected[field], ...tail]
    );
  }
  for (const field of ['sumValues', 'minimums', 'maximums'] as const) {
    // Fixed sums convert to float32; extrema are selected source values, hence exact.
    compare(
      `every ${field} and tail`,
      await readFloat32(table.columns[field].buffer, table.capacity),
      [
        ...expected[field],
        ...Array(table.capacity - expected.count).fill(field === 'sumValues' ? 0 : NaN)
      ],
      field === 'sumValues' ? 1e-6 : 0,
      field === 'sumValues' ? 1e-4 : 0
    );
  }
}

/** External tables as inputs; poison unused rows to test count-bounded reads after updates. */
export function writeCellTable(
  table: CellTableBuffers,
  expected: CellTableFixture,
  overflow = expected.overflow
) {
  for (const field of ['cells', 'counts', 'sums'] as const) {
    const values = new Uint32Array(table.capacity * (field === 'counts' ? 1 : 2)).fill(777);
    values.set(expected[field]);
    table.columns[field].buffer.write(values);
  }
  table.columns.count.buffer.write(new Uint32Array([expected.count]));
  table.columns.requiredCount.buffer.write(new Uint32Array([expected.requiredCount]));
  table.columns.overflow.buffer.write(new Uint32Array([overflow]));
}
