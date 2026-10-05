// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {
  createTransientView,
  GPUCommandGraph,
  GPUCOOToCSR,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {GPUVector} from '@luma.gl/gpgpu/gpu-data';
import {
  GPUGraph,
  GPUGraphConnectedComponents,
  GPUGraphCoreNumber,
  GPUGraphDegree,
  GPUGraphLabelPropagation,
  GPUGraphPageRank,
  GPUGraphTopology,
  GPUGraphTopologyView,
  type GPUGraphAdjacency,
  type GPUGraphAdjacencyView
} from '@luma.gl/gpgpu/gpu-graph';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';

type Edge = readonly [number, number];

type Columns = {
  degree: number[];
  inDegree?: number[];
  pageRank: number[];
  coreNumber: number[];
  components: number[];
  communities: number[];
};

const VERTEX_COUNT = 9;
// Two components, a self-loop on 2, a duplicate 0->1, and isolated vertex 8.
const EDGES: Edge[] = [
  [0, 1],
  [0, 1],
  [1, 2],
  [2, 2],
  [2, 0],
  [3, 2],
  [4, 5],
  [5, 6],
  [6, 4],
  [6, 7]
];
const ITERATIONS = 32;

function createBuffer(device: Device, buffers: Buffer[], values: Uint32Array | number): Buffer {
  const length = typeof values === 'number' ? values : values.length;
  const buffer = device.createBuffer({
    byteLength: Math.max(length, 1) * 4,
    usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
  });
  if (typeof values !== 'number' && length > 0) {
    buffer.write(values);
  }
  buffers.push(buffer);
  return buffer;
}

async function readBuffer(buffer: Buffer, format: 'uint32' | 'float32', length: number) {
  const bytes = await buffer.readAsync();
  const values =
    format === 'float32'
      ? new Float32Array(bytes.buffer, bytes.byteOffset, length)
      : new Uint32Array(bytes.buffer, bytes.byteOffset, length);
  return Array.from(values);
}

/** Runs the five algorithms on a conventional `GPUGraphTopology` rebuilt from COO edges. */
async function runPhysical(device: Device, directed: boolean): Promise<Columns> {
  const buffers: Buffer[] = [];
  const vector = <Format extends 'uint32' | 'float32'>(
    name: string,
    format: Format,
    values: Uint32Array | number
  ) =>
    new GPUVector<Format>({
      type: 'buffer',
      name,
      buffer: createBuffer(device, buffers, values),
      format,
      length: typeof values === 'number' ? values : values.length
    });
  const graph = new GPUGraph({
    vertexCount: VERTEX_COUNT,
    sourceVertices: vector(
      'source',
      'uint32',
      Uint32Array.from(EDGES, edge => edge[0])
    ),
    targetVertices: vector(
      'target',
      'uint32',
      Uint32Array.from(EDGES, edge => edge[1])
    ),
    directed
  });
  const adjacency = (name: string, capacity: number): GPUGraphAdjacency => ({
    offsets: vector(`${name}-offsets`, 'uint32', VERTEX_COUNT + 1),
    neighbors: vector(`${name}-neighbors`, 'uint32', capacity),
    edgeIds: vector(`${name}-edge-ids`, 'uint32', capacity),
    count: vector(`${name}-count`, 'uint32', 1),
    overflow: vector(`${name}-overflow`, 'uint32', 1)
  });
  const topology = new GPUGraphTopology({
    graph,
    forward: adjacency('forward', 2 * EDGES.length),
    reverse: directed ? adjacency('reverse', EDGES.length) : undefined,
    invalidEdgeCount: vector('invalid', 'uint32', 1)
  });
  const outputs = {
    degree: vector('degree', 'uint32', VERTEX_COUNT),
    inDegree: directed ? vector('in-degree', 'uint32', VERTEX_COUNT) : undefined,
    pageRank: vector('page-rank', 'float32', VERTEX_COUNT),
    coreNumber: vector('core-number', 'uint32', VERTEX_COUNT),
    components: vector('components', 'uint32', VERTEX_COUNT),
    communities: vector('communities', 'uint32', VERTEX_COUNT)
  };
  const commandGraph = new GPUCommandGraph(device, {id: 'physical'});
  topology.addToGraph(commandGraph);
  addAlgorithms(commandGraph, topology, outputs);
  const compiled = commandGraph.compile();
  submit(device, compiled);
  const read = (column: GPUVector<'uint32'> | GPUVector<'float32'>) =>
    readBuffer(column.data[0].buffer as Buffer, column.format, VERTEX_COUNT);
  const columns: Columns = {
    degree: await read(outputs.degree),
    inDegree: outputs.inDegree ? await read(outputs.inDegree) : undefined,
    pageRank: await read(outputs.pageRank),
    coreNumber: await read(outputs.coreNumber),
    components: await read(outputs.components),
    communities: await read(outputs.communities)
  };
  compiled.destroy();
  for (const buffer of buffers) buffer.destroy();
  return columns;
}

/** Runs the five algorithms on transient CSR views built in-graph by `GPUCOOToCSR`. */
async function runViews(
  device: Device,
  directed: boolean,
  overflowValue?: number
): Promise<Columns> {
  const buffers: Buffer[] = [];
  const commandGraph = new GPUCommandGraph(device, {id: 'views'});
  const outputBuffers: Record<string, {buffer: Buffer; format: 'uint32' | 'float32'}> = {};
  const output = <Format extends 'uint32' | 'float32'>(name: string, format: Format) => {
    const buffer = createBuffer(device, buffers, VERTEX_COUNT);
    outputBuffers[name] = {buffer, format};
    return importView(commandGraph, name, buffer, format, VERTEX_COUNT);
  };
  const overflow =
    overflowValue === undefined
      ? undefined
      : importView(
          commandGraph,
          'overflow',
          createBuffer(device, buffers, Uint32Array.of(overflowValue)),
          'uint32',
          1
        );
  // Undirected adjacency lists both directions of every edge and each self-loop once.
  const forwardEdges = directed
    ? EDGES
    : EDGES.flatMap(([from, to]) =>
        from === to
          ? [[from, to] as Edge]
          : [
              [from, to],
              [to, from]
            ]
      );
  const forward = addCSR(device, commandGraph, buffers, 'forward', forwardEdges, overflow);
  const reverse = directed
    ? addCSR(
        device,
        commandGraph,
        buffers,
        'reverse',
        EDGES.map(([from, to]) => [to, from] as Edge),
        overflow
      )
    : undefined;
  const topology = new GPUGraphTopologyView({vertexCount: VERTEX_COUNT, forward, reverse});
  expect(topology.graph.directed).toBe(directed);
  addAlgorithms(commandGraph, topology, {
    degree: output('degree', 'uint32'),
    inDegree: directed ? output('inDegree', 'uint32') : undefined,
    pageRank: output('pageRank', 'float32'),
    coreNumber: output('coreNumber', 'uint32'),
    components: output('components', 'uint32'),
    communities: output('communities', 'uint32')
  });
  const compiled = commandGraph.compile();
  submit(device, compiled);
  const read = (name: string) =>
    readBuffer(outputBuffers[name].buffer, outputBuffers[name].format, VERTEX_COUNT);
  const columns: Columns = {
    degree: await read('degree'),
    inDegree: directed ? await read('inDegree') : undefined,
    pageRank: await read('pageRank'),
    coreNumber: await read('coreNumber'),
    components: await read('components'),
    communities: await read('communities')
  };
  compiled.destroy();
  for (const buffer of buffers) buffer.destroy();
  return columns;
}

function importView<Format extends 'uint32' | 'float32'>(
  commandGraph: GPUCommandGraph,
  id: string,
  buffer: Buffer,
  format: Format,
  length: number
): GraphDataView<Format> {
  const handle = commandGraph.importBuffer(
    {id, byteLength: buffer.byteLength, usage: buffer.usage},
    buffer
  );
  return commandGraph.createDataView(handle, {format, length});
}

function addCSR(
  device: Device,
  commandGraph: GPUCommandGraph,
  buffers: Buffer[],
  name: string,
  edges: readonly Edge[],
  overflow: GraphDataView<'uint32'> | undefined
): GPUGraphAdjacencyView {
  const sorted = [...edges].sort((left, right) => left[0] - right[0] || left[1] - right[1]);
  const edgeCount = sorted.length;
  const coo = (suffix: string, values: Uint32Array) =>
    importView(
      commandGraph,
      `${name}-${suffix}`,
      createBuffer(device, buffers, values),
      'uint32',
      edgeCount
    );
  const offsets = createTransientView(commandGraph, `${name}-offsets`, 'uint32', VERTEX_COUNT + 1);
  const neighbors = createTransientView(commandGraph, `${name}-neighbors`, 'uint32', edgeCount);
  const valuesBuffer = createBuffer(device, buffers, edgeCount);
  commandGraph.add(
    new GPUCOOToCSR({
      id: `${name}-coo-to-csr`,
      rows: VERTEX_COUNT,
      rowIndices: coo(
        'rows',
        Uint32Array.from(sorted, edge => edge[0])
      ),
      columnIndices: coo(
        'columns',
        Uint32Array.from(sorted, edge => edge[1])
      ),
      values: importView(commandGraph, `${name}-values`, valuesBuffer, 'float32', edgeCount),
      rowOffsets: offsets,
      outputColumnIndices: neighbors,
      outputValues: createTransientView(commandGraph, `${name}-csr-values`, 'float32', edgeCount)
    })
  );
  return {offsets, neighbors, ...(overflow ? {overflow} : {})};
}

type AlgorithmOutputs<Uint32Column, Float32Column> = {
  degree: Uint32Column;
  inDegree?: Uint32Column;
  pageRank: Float32Column;
  coreNumber: Uint32Column;
  components: Uint32Column;
  communities: Uint32Column;
};

function addAlgorithms(
  commandGraph: GPUCommandGraph,
  topology: GPUGraphTopology | GPUGraphTopologyView,
  outputs:
    | AlgorithmOutputs<GPUVector<'uint32'>, GPUVector<'float32'>>
    | AlgorithmOutputs<GraphDataView<'uint32'>, GraphDataView<'float32'>>
): void {
  const algorithms = [
    new GPUGraphDegree({id: 'degree', topology, output: outputs.degree}),
    ...(outputs.inDegree
      ? [
          new GPUGraphDegree({
            id: 'in-degree',
            topology,
            output: outputs.inDegree,
            direction: 'incoming'
          })
        ]
      : []),
    new GPUGraphPageRank({id: 'page-rank', topology, output: outputs.pageRank}),
    new GPUGraphCoreNumber({
      id: 'core-number',
      topology,
      output: outputs.coreNumber,
      iterations: ITERATIONS
    }),
    new GPUGraphConnectedComponents({
      id: 'components',
      topology,
      output: outputs.components,
      iterations: ITERATIONS
    }),
    new GPUGraphLabelPropagation({
      id: 'communities',
      topology,
      output: outputs.communities,
      iterations: ITERATIONS
    })
  ];
  for (const algorithm of algorithms) {
    algorithm.addToGraph(commandGraph);
  }
}

function submit(device: Device, compiled: ReturnType<GPUCommandGraph['compile']>): void {
  const commandEncoder = device.createCommandEncoder();
  compiled.encode(commandEncoder, {parameters: undefined});
  device.submit(commandEncoder.finish());
}

function expectClose(actual: number[], expected: number[]): void {
  expect(actual.length).toBe(expected.length);
  for (const [index, value] of actual.entries()) {
    expect(Math.abs(value - expected[index])).toBeLessThanOrEqual(1e-6);
  }
}

it('GPUGraphTopologyView runs gpu-graph algorithms on transient GPUCOOToCSR adjacency', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  for (const directed of [false, true]) {
    const physical = await runPhysical(device, directed);
    const views = await runViews(device, directed);
    expect(views.degree).toEqual(physical.degree);
    expect(views.inDegree).toEqual(physical.inDegree);
    expectClose(views.pageRank, physical.pageRank);
    expect(views.coreNumber).toEqual(physical.coreNumber);
    expect(views.components).toEqual(physical.components);
    expect(views.communities).toEqual(physical.communities);
    // Sanity: weak components of the fixture.
    expect(views.components).toEqual([0, 0, 0, 0, 4, 4, 4, 4, 8]);
  }
  // A caller-supplied zero overflow word behaves like an omitted one.
  const explicitZero = await runViews(device, false, 0);
  expect(explicitZero.components).toEqual([0, 0, 0, 0, 4, 4, 4, 4, 8]);
  // A nonzero overflow word makes the algorithms fail closed, as with GPUGraphTopology.
  const overflowed = await runViews(device, false, 1);
  expect(overflowed.pageRank).toEqual(new Array(VERTEX_COUNT).fill(0));
  expect(overflowed.coreNumber).toEqual(new Array(VERTEX_COUNT).fill(0xffffffff));
});
