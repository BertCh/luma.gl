// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  GPULineMerge,
  GPU_LINE_MERGE_NONE
} from '../../../src/gpu-spatial-analysis/line-merge/index';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  readFloat32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {LINE_MERGE_LINES, LINE_MERGE_SHAPELY_CHAINS} from './line-merge-fixture';

type Vertex = number[];

type MergeResult = {
  chains: Vertex[][];
  lineChains: number[];
  lineOrders: number[];
  lineReversed: number[];
};

async function runLineMerge(
  device: NonNullable<Awaited<ReturnType<typeof getWebGPUTestDevice>>>,
  lines: Vertex[][],
  directed = false
): Promise<MergeResult> {
  const flat = lines.flat();
  const offsets = [0];
  for (const line of lines) {
    offsets.push(offsets[offsets.length - 1] + line.length);
  }
  const lineCount = lines.length;
  const graph = new GPUCommandGraph(device, {id: 'line-merge'});
  const buffers: Buffer[] = [];
  const input = (
    id: string,
    values: Float32Array | Uint32Array,
    format: 'uint32' | 'float32x2'
  ) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return importGraphBuffer(
      graph,
      id,
      buffer,
      format,
      format === 'uint32' ? values.length : values.length / 2
    );
  };
  const output = (id: string, format: 'uint32' | 'float32x2', length: number) => {
    const buffer = createOutputBuffer(device, format === 'uint32' ? length : length * 2);
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, id, buffer, format, length)};
  };
  const chainOffsets = output('chain-offsets', 'uint32', lineCount + 1);
  const positions = output('out-positions', 'float32x2', flat.length);
  const count = output('count', 'uint32', 1);
  const lineChains = output('line-chains', 'uint32', lineCount);
  const lineOrders = output('line-orders', 'uint32', lineCount);
  const lineReversed = output('line-reversed', 'uint32', lineCount);
  graph.add(
    new GPULineMerge({
      id: 'merge',
      directed,
      positions: input('positions', Float32Array.from(flat.flat()), 'float32x2'),
      lineOffsets: input('line-offsets', Uint32Array.from(offsets), 'uint32'),
      output: {
        chainOffsets: chainOffsets.view,
        positions: positions.view,
        count: count.view,
        lineChains: lineChains.view,
        lineOrders: lineOrders.view,
        lineReversed: lineReversed.view
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const [chainCount] = await readUint32(count.buffer, 1);
  const offsetValues = await readUint32(chainOffsets.buffer, lineCount + 1);
  const coordinates = await readFloat32(positions.buffer, flat.length * 2);
  const chains: Vertex[][] = [];
  for (let chain = 0; chain < chainCount; chain++) {
    const chainVertices: Vertex[] = [];
    for (let vertex = offsetValues[chain]; vertex < offsetValues[chain + 1]; vertex++) {
      chainVertices.push([coordinates[vertex * 2], coordinates[vertex * 2 + 1]]);
    }
    chains.push(chainVertices);
  }
  // Rows past the chain count repeat the final offset.
  for (let row = chainCount; row <= lineCount; row++) {
    expect(offsetValues[row]).toBe(offsetValues[chainCount]);
  }
  const result = {
    chains,
    lineChains: await readUint32(lineChains.buffer, lineCount),
    lineOrders: await readUint32(lineOrders.buffer, lineCount),
    lineReversed: await readUint32(lineReversed.buffer, lineCount)
  };
  compiled.destroy();
  for (const buffer of buffers) {
    buffer.destroy();
  }
  return result;
}

/**
 * Orientation-free chain key: the lexicographically smallest of the chain and its reverse, and for
 * closed rings also of every rotation (shapely may start a ring anywhere).
 */
function canonicalize(chain: Vertex[]): Vertex[] {
  const text = (vertices: Vertex[]) => JSON.stringify(vertices);
  const variants: Vertex[][] = [chain, [...chain].reverse()];
  const isClosed = text(chain[0]) === text(chain[chain.length - 1]);
  if (isClosed) {
    const ring = chain.slice(0, -1);
    for (const direction of [ring, [...ring].reverse()]) {
      for (let shift = 0; shift < direction.length; shift++) {
        const rotated = [...direction.slice(shift), ...direction.slice(0, shift)];
        variants.push([...rotated, rotated[0]]);
      }
    }
  }
  return variants.sort((a, b) => text(a).localeCompare(text(b)))[0];
}

function sortChains(chains: Vertex[][]): Vertex[][] {
  return chains
    .map(canonicalize)
    .sort((a, b) => JSON.stringify(a).localeCompare(JSON.stringify(b)));
}

it('GPULineMerge matches shapely linemerge on a mixed fixture', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const result = await runLineMerge(device, LINE_MERGE_LINES);
  expect(sortChains(result.chains)).toEqual(sortChains(LINE_MERGE_SHAPELY_CHAINS));
  // Every line sits in exactly one chain, in order 0..n-1 per chain.
  const orders = new Map<number, number[]>();
  result.lineChains.forEach((chain, line) => {
    expect(chain).not.toBe(GPU_LINE_MERGE_NONE);
    orders.set(chain, [...(orders.get(chain) ?? []), result.lineOrders[line]]);
  });
  for (const chainOrders of orders.values()) {
    expect([...chainOrders].sort((a, b) => a - b)).toEqual(chainOrders.map((_, i) => i));
  }
  // Chains are ordered by leader (order-0) line index.
  const leaders = Array.from(orders.keys())
    .sort((a, b) => a - b)
    .map(chain =>
      result.lineChains.findIndex((c, line) => c === chain && result.lineOrders[line] === 0)
    );
  expect([...leaders].sort((a, b) => a - b)).toEqual(leaders);
});

it('GPULineMerge orients paths from the lower endpoint id and ignores degenerate lines', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  // Line 0 runs right, line 1 is stored reversed; the end of line 0 (id 1) joins the end of line 1
  // (id 3), the free ends are ids 0 and 2, so the chain starts at line 0's start and runs right.
  const lines: Vertex[][] = [
    [
      [0, 0],
      [1, 0]
    ],
    [
      [2, 0],
      [1, 0]
    ],
    [[5, 5]],
    [
      [7, 7],
      [8, 8]
    ]
  ];
  const result = await runLineMerge(device, lines);
  expect(result.chains).toEqual([
    [
      [0, 0],
      [1, 0],
      [2, 0]
    ],
    [
      [7, 7],
      [8, 8]
    ]
  ]);
  expect(result.lineChains).toEqual([0, 0, GPU_LINE_MERGE_NONE, 1]);
  expect(result.lineReversed).toEqual([0, 1, 0, 0]);
});

it('GPULineMerge ranks long shuffled paths and rings with list ranking', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  // A 700-line path and a 130-line ring, plus a short path, all shuffled with random directions.
  let seed = 12345;
  const random = () => {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    return seed / 4294967296;
  };
  const expectedChains: Vertex[][] = [];
  const lines: Vertex[][] = [];
  const addChain = (vertices: Vertex[], verticesPerLine: number) => {
    expectedChains.push(vertices);
    for (let first = 0; first + 1 < vertices.length; first += verticesPerLine - 1) {
      lines.push(vertices.slice(first, first + verticesPerLine));
    }
  };
  const path: Vertex[] = [];
  for (let i = 0; i <= 1400; i++) {
    path.push([i, (i * 7) % 5]);
  }
  addChain(path, 3);
  const ring: Vertex[] = [];
  for (let i = 0; i < 130; i++) {
    const angle = (i / 130) * 2 * Math.PI;
    ring.push([Math.fround(5000 + Math.cos(angle) * 50), Math.fround(Math.sin(angle) * 50)]);
  }
  ring.push(ring[0]);
  addChain(ring, 2);
  addChain(
    [
      [-5, -5],
      [-6, -5],
      [-7, -6]
    ],
    2
  );
  for (let i = lines.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [lines[i], lines[j]] = [lines[j], lines[i]];
  }
  const flipped = lines.map(line => (random() < 0.5 ? [...line].reverse() : line));
  const result = await runLineMerge(device, flipped);
  expect(sortChains(result.chains)).toEqual(sortChains(expectedChains));
  const orders = new Map<number, number[]>();
  result.lineChains.forEach((chain, line) => {
    expect(chain).not.toBe(GPU_LINE_MERGE_NONE);
    orders.set(chain, [...(orders.get(chain) ?? []), result.lineOrders[line]]);
  });
  for (const chainOrders of orders.values()) {
    expect([...chainOrders].sort((a, b) => a - b)).toEqual(chainOrders.map((_, i) => i));
  }
});

// Generated by shapely 2.1.2 line_merge(directed=True)
const DIRECTED_CASES: Record<string, {lines: Vertex[][]; chains: Vertex[][]}> = {
  chainAndOrder: {
    lines: [
      [
        [1, 0],
        [2, 0]
      ],
      [
        [0, 0],
        [1, 0]
      ],
      [
        [5, 5],
        [6, 6]
      ],
      [
        [2, 0],
        [3, 0]
      ]
    ],
    chains: [
      [
        [0, 0],
        [1, 0],
        [2, 0],
        [3, 0]
      ],
      [
        [5, 5],
        [6, 6]
      ]
    ]
  },
  reversedStaysSplit: {
    lines: [
      [
        [0, 0],
        [1, 0]
      ],
      [
        [2, 0],
        [1, 0]
      ]
    ],
    chains: [
      [
        [0, 0],
        [1, 0]
      ],
      [
        [2, 0],
        [1, 0]
      ]
    ]
  },
  sharedStartStaysSplit: {
    lines: [
      [
        [1, 0],
        [0, 0]
      ],
      [
        [1, 0],
        [2, 0]
      ]
    ],
    chains: [
      [
        [1, 0],
        [2, 0]
      ],
      [
        [1, 0],
        [0, 0]
      ]
    ]
  },
  junction: {
    lines: [
      [
        [0, 0],
        [1, 0]
      ],
      [
        [1, 0],
        [2, 0]
      ],
      [
        [1, 1],
        [1, 0]
      ]
    ],
    chains: [
      [
        [0, 0],
        [1, 0]
      ],
      [
        [1, 0],
        [2, 0]
      ],
      [
        [1, 1],
        [1, 0]
      ]
    ]
  },
  cycle: {
    lines: [
      [
        [0, 0],
        [1, 0]
      ],
      [
        [1, 0],
        [1, 1]
      ],
      [
        [1, 1],
        [0, 0]
      ]
    ],
    chains: [
      [
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 0]
      ]
    ]
  },
  closedWithTail: {
    lines: [
      [
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 0]
      ],
      [
        [0, 0],
        [-1, 0]
      ]
    ],
    chains: [
      [
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 0]
      ],
      [
        [0, 0],
        [-1, 0]
      ]
    ]
  },
  mixed: {
    lines: [
      [
        [0, 0],
        [1, 0]
      ],
      [
        [1, 0],
        [2, 0]
      ],
      [
        [4, 0],
        [3, 0]
      ],
      [
        [3, 0],
        [2, 0]
      ],
      [
        [2, 0],
        [2, 1]
      ]
    ],
    chains: [
      [
        [0, 0],
        [1, 0],
        [2, 0]
      ],
      [
        [2, 0],
        [2, 1]
      ],
      [
        [4, 0],
        [3, 0],
        [2, 0]
      ]
    ]
  }
};

/** Directed chains keep orientation; closed rings may start anywhere, so rotate to the smallest. */
function canonicalizeDirected(chain: Vertex[]): Vertex[] {
  const text = (vertices: Vertex[]) => JSON.stringify(vertices);
  if (text(chain[0]) !== text(chain[chain.length - 1])) {
    return chain;
  }
  const ring = chain.slice(0, -1);
  const rotations = ring.map((_, shift) => {
    const rotated = [...ring.slice(shift), ...ring.slice(0, shift)];
    return [...rotated, rotated[0]];
  });
  return rotations.sort((a, b) => text(a).localeCompare(text(b)))[0];
}

function sortDirectedChains(chains: Vertex[][]): Vertex[][] {
  return chains
    .map(canonicalizeDirected)
    .sort((a, b) => JSON.stringify(a).localeCompare(JSON.stringify(b)));
}

it('GPULineMerge directed matches shapely line_merge(directed=True)', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  for (const [name, {lines, chains}] of Object.entries(DIRECTED_CASES)) {
    const result = await runLineMerge(device, lines, true);
    expect(sortDirectedChains(result.chains), name).toEqual(sortDirectedChains(chains));
    expect(result.lineReversed, name).toEqual(lines.map(() => 0));
  }
});
