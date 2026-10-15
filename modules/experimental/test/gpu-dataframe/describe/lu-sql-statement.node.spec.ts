// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {GPUDataFrame} from '@luma.gl/experimental/gpu-dataframe';
import {LuSQLContext} from '@luma.gl/experimental/gpu-sql';
import {GPURecordBatch, GPUTable, type GPUTypeMap} from '@luma.gl/experimental/gpu-tables';
import {GPUData} from '@luma.gl/gpgpu/gpu-data';
import {NullDevice} from '@luma.gl/test-utils';
import {expect, test} from 'vitest';

function createTracts(): {device: NullDevice; tracts: GPUDataFrame} {
  const device = new NullDevice({id: 'lu-sql-statement-node'});
  const upload = <Format extends 'float32' | 'uint32'>(
    values: Float32Array | Uint32Array,
    format: Format
  ): GPUData<Format> =>
    new GPUData({
      buffer: device.createBuffer({
        data: values,
        usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
      }),
      format,
      length: values.length,
      ownsBuffer: true
    });
  const batch = new GPURecordBatch<GPUTypeMap>({
    gpuData: {
      population: upload(Float32Array.of(4284, 8210), 'float32'),
      households: upload(Float32Array.of(2322, 3028), 'float32'),
      noVehicle: upload(Float32Array.of(885, 928), 'float32'),
      category: upload(Uint32Array.of(0, 1), 'uint32')
    },
    fields: [
      {name: 'population', format: 'float32', nullable: false},
      {name: 'households', format: 'float32', nullable: false},
      {name: 'noVehicle', format: 'float32', nullable: false},
      {name: 'category', format: 'uint32', nullable: false}
    ]
  });
  return {
    device,
    tracts: new GPUDataFrame({
      table: new GPUTable<GPUTypeMap>({batches: [batch]}),
      dictionaries: {category: {values: ['a', 'b'], ordered: false}},
      ownership: 'owned'
    })
  };
}

const formatEdges = (edges: readonly {from: string; to: string; kind: string}[]) =>
  edges.map(edge => `${edge.from} -> ${edge.to} (${edge.kind})`);

test('describeStatement lists one node per clause and expression node of the example SQL', () => {
  const {tracts} = createTracts();
  const query = new LuSQLContext({tracts}).query(
    'SELECT population, noVehicle / households AS noVehicleRate FROM tracts WHERE population > :minimumPopulation',
    {parameters: {minimumPopulation: 1000}}
  );

  expect(query.statement.table).toBe('tracts');
  expect(query.statement.items.map(item => item.alias)).toEqual([undefined, 'noVehicleRate']);

  const graph = query.describeStatement();
  expect(graph.id).toBe('lu-sql-statement');
  expect(graph.form).toBe('intent');
  expect(graph.domain).toBe('gpu-sql');
  expect(graph.nodes.map(node => node.id)).toEqual([
    'from',
    'where',
    'where/expr',
    'where/expr/left',
    'where/expr/right',
    'select-0',
    'select-0/expr',
    'select-1',
    'select-1/expr',
    'select-1/expr/left',
    'select-1/expr/right'
  ]);
  expect(Object.fromEntries(graph.nodes.map(node => [node.id, node.label]))).toEqual({
    from: 'tracts',
    where: 'WHERE',
    'where/expr': '>',
    'where/expr/left': 'population',
    'where/expr/right': ':minimumPopulation',
    'select-0': 'population',
    'select-0/expr': 'population',
    'select-1': 'noVehicleRate',
    'select-1/expr': '/',
    'select-1/expr/left': 'noVehicle',
    'select-1/expr/right': 'households'
  });
  expect(formatEdges(graph.edges)).toEqual([
    'where/expr/left -> where/expr (dataflow)',
    'where/expr/right -> where/expr (dataflow)',
    'where/expr -> where (dataflow)',
    'from -> where (dataflow)',
    'select-0/expr -> select-0 (dataflow)',
    'where -> select-0 (dataflow)',
    'select-1/expr/left -> select-1/expr (dataflow)',
    'select-1/expr/right -> select-1/expr (dataflow)',
    'select-1/expr -> select-1 (dataflow)',
    'where -> select-1 (dataflow)'
  ]);
  const nodeIds = new Set(graph.nodes.map(node => node.id));
  expect(graph.edges.every(edge => nodeIds.has(edge.from) && nodeIds.has(edge.to))).toBe(true);
});

test('describeStatement covers aggregates, GROUP BY, ORDER BY and LIMIT', () => {
  const {tracts} = createTracts();
  const context = new LuSQLContext({tracts});

  const grouped = context
    .query(
      'SELECT category, COUNT(*) AS observations, SUM(population) AS residents FROM tracts GROUP BY category'
    )
    .describeStatement();
  expect(grouped.nodes.map(node => `${node.id}:${node.op}`)).toEqual([
    'from:from',
    'group-by:groupBy',
    'select-0:select',
    'select-0/expr:column',
    'select-1:aggregate',
    'select-2:aggregate'
  ]);
  expect(formatEdges(grouped.edges)).toEqual([
    'from -> group-by (dataflow)',
    'select-0/expr -> select-0 (dataflow)',
    'group-by -> select-0 (dataflow)',
    'group-by -> select-1 (dataflow)',
    'group-by -> select-2 (dataflow)'
  ]);

  const ordered = context
    .query('SELECT population FROM tracts ORDER BY population DESC LIMIT 10')
    .describeStatement();
  expect(ordered.nodes.map(node => node.id).slice(-2)).toEqual(['order-by', 'limit']);
  expect(formatEdges(ordered.edges).slice(-2)).toEqual([
    'select-0 -> order-by (dataflow)',
    'order-by -> limit (dataflow)'
  ]);
});
