// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUSnapRoundedNoding} from '../../../src/gpu-spatial-analysis/geometry-topology/gpu-snap-rounded-noding';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {expectNodingMatchesReference, type NodingRun} from './snap-rounding-assertions';
import {referenceSnapRound, referenceLatticeFrame} from './snap-rounding-reference';

// GPU run of GPUSnapRoundedNoding against the BigInt snap-rounding reference
// (snap-rounding-reference.ts): the same fixtures as snap-rounded-noding-executed.node.spec.ts,
// then larger random linework, the Morton-sorted crossing BVH path, and a near-capacity run.

type Line = (readonly [number, number])[];

async function runOnGPU(
  device: Device,
  lines: readonly Line[],
  options: {
    gridSize?: number;
    intersectionCapacity?: number;
    capacity?: number;
    spatialSort?: boolean;
  } = {}
): Promise<NodingRun & {overflow: number}> {
  const graph = new GPUCommandGraph(device, {id: 'snap-rounded-noding-gpu'});
  const buffers: Buffer[] = [];
  const input = <Format extends 'float32x2' | 'uint32'>(
    name: string,
    values: Float32Array | Uint32Array,
    format: Format
  ) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return importGraphBuffer(
      graph,
      name,
      buffer,
      format,
      values.length / (format === 'float32x2' ? 2 : 1)
    );
  };
  const output = <Format extends 'uint32' | 'float32' | 'float32x4'>(
    name: string,
    format: Format,
    length: number
  ) => {
    const buffer = createOutputBuffer(
      device,
      length * {uint32: 1, float32: 1, float32x4: 4}[format]
    );
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, name, buffer, format, length)};
  };
  const positions = Float32Array.from(lines.flat().flat());
  const offsets = Uint32Array.from(
    lines.reduce<number[]>((all, line) => [...all, all[all.length - 1] + line.length], [0])
  );
  const capacity = options.capacity ?? 4096;
  const gridSize = options.gridSize ?? 1;
  const endpoints = output('endpoints', 'float32x4', capacity);
  const latticeEndpoints = output('lattice-endpoints', 'float32x4', capacity);
  const lattice = output('lattice', 'float32', 4);
  const features = output('features', 'uint32', capacity);
  const rings = output('rings', 'uint32', capacity);
  const edges = output('edges', 'uint32', capacity);
  const starts = output('starts', 'float32', capacity);
  const ends = output('ends', 'float32', capacity);
  const count = output('count', 'uint32', 1);
  const required = output('required', 'uint32', 1);
  const overflow = output('overflow', 'uint32', 1);
  const candidateOverflow = output('candidate-overflow', 'uint32', 1);
  const uncertain = output('uncertain', 'uint32', 1);
  graph.add(
    new GPUSnapRoundedNoding({
      lines: {
        kind: 'lines',
        positions: input('positions', positions, 'float32x2'),
        lineOffsets: input('line-offsets', offsets, 'uint32')
      },
      gridSize,
      intersectionCapacity: options.intersectionCapacity ?? 4096,
      spatialSort: options.spatialSort,
      output: {
        endpoints: endpoints.view,
        latticeEndpoints: latticeEndpoints.view,
        lattice: lattice.view,
        sourceFeatureIds: features.view,
        sourceRingIds: rings.view,
        sourceEdgeIds: edges.view,
        sourceStartParameters: starts.view,
        sourceEndParameters: ends.view,
        status: {
          count: count.view,
          requiredCount: required.view,
          overflow: overflow.view,
          candidateOverflow: candidateOverflow.view,
          uncertainCount: uncertain.view
        }
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const written = (await readUint32(count.buffer, 1))[0];
  const lattices = await readFloat32(latticeEndpoints.buffer, 4 * capacity);
  const world = await readFloat32(endpoints.buffer, 4 * capacity);
  const edgeIds = await readUint32(edges.buffer, capacity);
  const result = {
    positions,
    lines,
    gridSize,
    lattice: await readFloat32(lattice.buffer, 4),
    count: written,
    required: (await readUint32(required.buffer, 1))[0],
    candidateOverflow: (await readUint32(candidateOverflow.buffer, 1))[0],
    uncertain: (await readUint32(uncertain.buffer, 1))[0],
    overflow: (await readUint32(overflow.buffer, 1))[0],
    pieces: Array.from({length: written}, (_, row) => ({
      segment: edgeIds[row],
      from: [lattices[4 * row], lattices[4 * row + 1]] as [number, number],
      to: [lattices[4 * row + 2], lattices[4 * row + 3]] as [number, number]
    })),
    worldRows: Float32Array.from(world.slice(0, 4 * written))
  };
  compiled.destroy();
  for (const buffer of buffers) {
    buffer.destroy();
  }
  return result;
}

function createRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const FIXTURES: {label: string; lines: Line[]; gridSize?: number}[] = [
  {
    label: 'cross',
    lines: [
      [
        [0, 0],
        [4, 3]
      ],
      [
        [0, 3],
        [4, 0]
      ]
    ]
  },
  {
    label: 'reroute',
    lines: [
      [
        [0, 0],
        [10, 3]
      ],
      [
        [5, 1],
        [5, 8]
      ]
    ]
  },
  {
    label: 'overlap',
    lines: [
      [
        [0, 0],
        [6, 0]
      ],
      [
        [2, 0],
        [8, 0]
      ]
    ]
  },
  {
    label: 't-junction',
    lines: [
      [
        [0, 0],
        [6, 0]
      ],
      [
        [3, 0],
        [3, 5]
      ]
    ]
  },
  {
    label: 'spike',
    lines: [
      [
        [0, 0],
        [4, 0],
        [1, 0]
      ]
    ]
  },
  {
    label: 'ring crosses ring',
    lines: [
      [
        [0, 0],
        [6, 0],
        [6, 6],
        [0, 6],
        [0, 0]
      ],
      [
        [3, 3],
        [9, 3],
        [9, 9],
        [3, 9],
        [3, 3]
      ]
    ]
  },
  {
    label: 'collapses',
    lines: [
      [
        [0, 0],
        [0.2, 0.3],
        [5, 0]
      ],
      [
        [0.1, 4],
        [3, -2]
      ]
    ]
  },
  {
    label: 'offset origin',
    gridSize: 0.3,
    lines: [
      [
        [1000.25, 2000.5],
        [1004.75, 2003.25],
        [1001, 2006]
      ],
      [
        [1000.5, 2003.75],
        [1005, 2001]
      ]
    ]
  },
  {
    label: 'fine cell',
    gridSize: 1e-9,
    lines: [
      [
        [0, 0],
        [100, 37]
      ],
      [
        [0, 37],
        [100, 0]
      ]
    ]
  }
];

it('GPUSnapRoundedNoding matches the BigInt snap-rounding reference on fixtures', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const {label, lines, gridSize} of FIXTURES) {
    const result = await runOnGPU(device, lines, {
      gridSize,
      intersectionCapacity: 256,
      capacity: 512
    });
    expect(result.overflow, label).toBe(0);
    expectNodingMatchesReference(result, label);
  }
  device.destroy?.();
});

it('GPUSnapRoundedNoding matches the reference on random linework, with and without spatial sort', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (let seed = 1; seed <= 8; seed++) {
    const random = createRandom(900 + seed);
    const lines: Line[] = Array.from({length: 12}, () =>
      Array.from(
        {length: 2 + Math.floor(random() * 6)},
        () => [Math.floor(random() * 160) / 4, Math.floor(random() * 160) / 4] as [number, number]
      )
    );
    for (const spatialSort of [false, true]) {
      const result = await runOnGPU(device, lines, {
        gridSize: seed % 2 ? 1 : 0.5,
        spatialSort,
        intersectionCapacity: 4096,
        capacity: 8192
      });
      expect(result.overflow, `seed ${seed}`).toBe(0);
      expectNodingMatchesReference(result, `seed ${seed} sort ${spatialSort}`);
    }
  }
  device.destroy?.();
});

it('GPUSnapRoundedNoding is deterministic and reports capacity overflow exactly', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const random = createRandom(77);
  const lines: Line[] = Array.from({length: 10}, () =>
    Array.from(
      {length: 4},
      () => [Math.floor(random() * 80) / 4, Math.floor(random() * 80) / 4] as [number, number]
    )
  );
  const frame = referenceLatticeFrame(Float32Array.from(lines.flat().flat()), 1);
  let row = 0;
  const reference = referenceSnapRound(
    lines.map(line => {
      const points = frame.points.slice(row, row + line.length);
      row += line.length;
      return points;
    })
  );
  const first = await runOnGPU(device, lines, {capacity: 4096});
  const second = await runOnGPU(device, lines, {capacity: 4096});
  expect(second.pieces).toEqual(first.pieces);
  const small = await runOnGPU(device, lines, {capacity: 8});
  expect(small.overflow).toBe(1);
  expect(small.candidateOverflow).toBe(0);
  expect(small.required).toBe(reference.pieces.length);
  expect(small.count).toBe(8);
  expect(small.pieces).toEqual(reference.pieces.slice(0, 8));
  device.destroy?.();
});
