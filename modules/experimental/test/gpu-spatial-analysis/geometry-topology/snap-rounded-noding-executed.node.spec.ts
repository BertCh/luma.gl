// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Runs the real exact lattice WGSL (utils/exact-lattice-wgsl.ts) and the real GPUSnapRoundedNoding
// kernels on the CPU (wgsl_reflect interpreter, see segment-intersection/wgsl-graph-executor.ts)
// and compares them with the BigInt snap-rounding reference (snap-rounding-reference.ts).

import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';
import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {describe, expect, it} from 'vitest';
import {createWGSLKernelNode} from '../../../src/utils/wgsl-kernel-nodes';
import {EXACT_LATTICE_WGSL} from '../../../src/utils/exact-lattice-wgsl';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {GPUSnapRoundedNoding} from '../../../src/gpu-spatial-analysis/geometry-topology/gpu-snap-rounded-noding';
import {expectNodingMatchesReference as expectMatchesReference} from './snap-rounding-assertions';
import {
  isProperCrossing,
  referenceMeetsPixel,
  referenceRoundedCrossing,
  type LatticePoint,
  type ReferencePiece
} from './snap-rounding-reference';

useCPUExecutorTestHooks();

type Quad = [LatticePoint, LatticePoint, LatticePoint, LatticePoint];

/** Runs `body` (WGSL over `cases`, ten i32 inputs per case, `width` i32 outputs) on the CPU. */
function runLatticeKernel(cases: readonly number[][], width: number, body: string): number[][] {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'lattice-helper-cpu'});
  const inputs = createTransientView(graph, 'inputs', 'uint32', cases.length * 10);
  const outputs = createTransientView(graph, 'outputs', 'uint32', cases.length * width);
  const node = createWGSLKernelNode(graph, {
    id: 'lattice-helper',
    operation: 'test',
    bindings: [
      {name: 'inputs', view: inputs, type: 'i32', access: 'read'},
      {name: 'outputs', view: outputs, type: 'i32', access: 'read_write'}
    ],
    invocationCount: cases.length,
    declarations: `${EXACT_LATTICE_WGSL}
fn input(row: u32, slot: u32) -> i32 { return inputs[inputsOffset + row * 10u + slot]; }`,
    body
  });
  const memory = createCPUGraphMemory();
  memory.write(
    inputs,
    Int32Array.from(cases.flatMap(row => [...row, ...new Array(10 - row.length).fill(0)]))
  );
  executeCommandNodesOnCPU([node], memory);
  const words = new Int32Array(memory.readUint32(outputs).buffer);
  return cases.map((_, row) => Array.from(words.slice(row * width, row * width + width)));
}

const roundBody = `let a = vec2i(input(index, 0u), input(index, 1u));
  let b = vec2i(input(index, 2u), input(index, 3u));
  let c = vec2i(input(index, 4u), input(index, 5u));
  let d = vec2i(input(index, 6u), input(index, 7u));
  let terms = latticeCrossingTerms(a, b, c, d);
  let rounded = latticeRoundedCrossing(a, b, terms);
  outputs[outputsOffset + index * 2u] = rounded.x;
  outputs[outputsOffset + index * 2u + 1u] = rounded.y;`;

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

const MAXIMUM = 2 ** 22 + 1;

/** Random proper crossings; `pick` draws one lattice coordinate. */
function generateCrossings(
  count: number,
  seed: number,
  pick: (random: () => number) => number
): Quad[] {
  const random = createRandom(seed);
  const quads: Quad[] = [];
  let attempts = 0;
  while (quads.length < count && attempts++ < count * 200) {
    const quad = Array.from(
      {length: 4},
      () => [pick(random), pick(random)] as LatticePoint
    ) as Quad;
    if (isProperCrossing(...quad)) {
      quads.push(quad);
    }
  }
  return quads;
}

function generateNearParallel(count: number, seed: number): Quad[] {
  const random = createRandom(seed);
  const quads: Quad[] = [];
  let attempts = 0;
  while (quads.length < count && attempts++ < count * 100) {
    const height = () => Math.floor(random() * 6);
    const base = random() < 0.5 ? 0 : MAXIMUM - 5;
    const quad: Quad = [
      [0, base + height()],
      [MAXIMUM, base + height()],
      [0, base + height()],
      [MAXIMUM, base + height()]
    ];
    const swapped = random() < 0.5;
    const oriented = (swapped ? quad.map(([x, y]) => [y, x]) : quad) as Quad;
    if (isProperCrossing(...oriented)) {
      quads.push(oriented);
    }
  }
  return quads;
}

function isHalfTie(quad: Quad): boolean {
  const [a, b, c, d] = quad;
  const denominator =
    BigInt(b[0] - a[0]) * BigInt(d[1] - c[1]) - BigInt(b[1] - a[1]) * BigInt(d[0] - c[0]);
  const numerator =
    BigInt(c[0] - a[0]) * BigInt(d[1] - c[1]) - BigInt(c[1] - a[1]) * BigInt(d[0] - c[0]);
  // x - a = N (bx - ax) / D has a fractional part of exactly 1/2 on either axis.
  return [0, 1].some(axis => {
    const scaled = 2n * numerator * BigInt(b[axis] - a[axis]);
    const divisor = 2n * denominator;
    return ((scaled % divisor) + divisor) % divisor === divisor / 2n;
  });
}

describe('exact lattice WGSL executed on the CPU', () => {
  it('rounds proper crossings half up exactly (small lattice, ties, long and extreme segments)', () => {
    const groups: {name: string; quads: Quad[]}[] = [
      {name: 'small', quads: generateCrossings(120, 1, random => Math.floor(random() * 8))},
      {name: 'medium', quads: generateCrossings(60, 2, random => Math.floor(random() * 1000))},
      {
        name: 'extreme',
        quads: generateCrossings(80, 3, random =>
          random() < 0.5 ? Math.floor(random() * 4) : MAXIMUM - Math.floor(random() * 4)
        )
      },
      {
        name: 'wide',
        quads: generateCrossings(60, 4, random => Math.floor(random() * (MAXIMUM + 1)))
      },
      {
        // Two segments spanning the whole lattice with end heights within a few units: the
        // crossing parameter is a ratio of numbers below 2^3 over a difference of 2^22 products.
        name: 'near-parallel',
        quads: generateNearParallel(40, 5)
      }
    ];
    let ties = 0;
    for (const {name, quads} of groups) {
      expect(quads.length, name).toBeGreaterThanOrEqual(5);
      const results = runLatticeKernel(
        quads.map(quad => quad.flat()),
        2,
        roundBody
      );
      quads.forEach((quad, row) => {
        expect(isProperCrossing(...quad), `${name} ${row}`).toBe(true);
        expect(results[row], `${name} ${JSON.stringify(quad)}`).toEqual(
          referenceRoundedCrossing(...quad)
        );
        ties += isHalfTie(quad) ? 1 : 0;
      });
    }
    // The small lattice must exercise exact ties at half pixels.
    expect(ties).toBeGreaterThan(10);
  });

  it('classifies half-open pixel incidence exactly', () => {
    const random = createRandom(7);
    const cases: {a: LatticePoint; b: LatticePoint; center: LatticePoint}[] = [];
    // Axis, diagonal and corner-grazing segments around one pixel, then random small ones.
    const center: LatticePoint = [3, 3];
    for (let ax = 0; ax <= 6; ax++) {
      for (let ay = 0; ay <= 6; ay++) {
        for (const [bx, by] of [
          [6, 6],
          [0, 6],
          [6, 0],
          [0, 0],
          [3, 6],
          [6, 3],
          [1, 5],
          [5, 1]
        ]) {
          if ((ax !== bx || ay !== by) && cases.length < 400) {
            cases.push({a: [ax, ay], b: [bx, by], center});
          }
        }
      }
    }
    for (let index = 0; index < 500; index++) {
      const pick = () => Math.floor(random() * 7);
      cases.push({a: [pick(), pick()], b: [pick(), pick()], center: [pick(), pick()]});
    }
    for (let index = 0; index < 40; index++) {
      const pick = () =>
        random() < 0.5 ? Math.floor(random() * 3) : MAXIMUM - Math.floor(random() * 3);
      const a: LatticePoint = [pick(), pick()];
      const b: LatticePoint = [pick(), pick()];
      cases.push({a, b, center: [pick(), pick()]});
    }
    const eligible = cases.filter(
      ({a, b, center: pixel}) =>
        (a[0] !== b[0] || a[1] !== b[1]) && referenceMeetsPixel(a, b, pixel, true)
    );
    const results = runLatticeKernel(
      eligible.map(({a, b, center: pixel}) => [...a, ...b, ...pixel]),
      1,
      `let a = vec2i(input(index, 0u), input(index, 1u));
  let b = vec2i(input(index, 2u), input(index, 3u));
  let center = vec2i(input(index, 4u), input(index, 5u));
  outputs[outputsOffset + index] = select(0, 1, latticeSegmentMeetsPixel(a, b, center));`
    );
    let missed = 0;
    eligible.forEach(({a, b, center: pixel}, row) => {
      const expected = referenceMeetsPixel(a, b, pixel);
      missed += expected ? 0 : 1;
      expect(results[row][0] === 1, JSON.stringify({a, b, pixel})).toBe(expected);
    });
    expect(eligible.length).toBeGreaterThan(120);
    // Corner grazes (top-left, top-right, bottom-right) must occur among the closed hits.
    expect(missed).toBeGreaterThan(5);
  });
});

type NodingOptions = {
  gridSize?: number;
  intersectionCapacity?: number;
  outputCapacity?: number;
  sourceIds?: number[];
  spatialSort?: boolean;
};

/** Runs `GPUSnapRoundedNoding` on the CPU executor for lines of world vertices (NaN allowed). */
function runNoding(lines: readonly (readonly [number, number])[][], options: NodingOptions = {}) {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'snap-rounded-noding-cpu'});
  const memory = createCPUGraphMemory();
  const positions = Float32Array.from(lines.flat().flat());
  const lineOffsets = Uint32Array.from(
    lines.reduce<number[]>(
      (offsets, line) => [...offsets, offsets[offsets.length - 1] + line.length],
      [0]
    )
  );
  const positionsView = createTransientView(graph, 'positions', 'float32x2', positions.length / 2);
  const lineOffsetsView = createTransientView(graph, 'line-offsets', 'uint32', lineOffsets.length);
  memory.write(positionsView, positions);
  memory.write(lineOffsetsView, lineOffsets);
  let sourceIds;
  if (options.sourceIds) {
    sourceIds = createTransientView(graph, 'source-ids', 'uint32', options.sourceIds.length);
    memory.write(sourceIds, Uint32Array.from(options.sourceIds));
  }
  const capacity = options.outputCapacity ?? 128;
  const scalar = (name: string) => createTransientView(graph, name, 'uint32', 1);
  const output = {
    endpoints: createTransientView(graph, 'endpoints', 'float32x4', capacity),
    latticeEndpoints: createTransientView(graph, 'lattice-endpoints', 'float32x4', capacity),
    lattice: createTransientView(graph, 'lattice', 'float32', 4),
    sourceFeatureIds: createTransientView(graph, 'feature-ids', 'uint32', capacity),
    sourceRingIds: createTransientView(graph, 'ring-ids', 'uint32', capacity),
    sourceEdgeIds: createTransientView(graph, 'edge-ids', 'uint32', capacity),
    sourceStartParameters: createTransientView(graph, 'start-parameters', 'float32', capacity),
    sourceEndParameters: createTransientView(graph, 'end-parameters', 'float32', capacity),
    status: {
      count: scalar('count'),
      requiredCount: scalar('required'),
      overflow: scalar('overflow'),
      candidateOverflow: scalar('candidate-overflow'),
      uncertainCount: scalar('uncertain')
    }
  };
  const gridSize = options.gridSize ?? 1;
  const contributor = new GPUSnapRoundedNoding({
    lines: {kind: 'lines', positions: positionsView, lineOffsets: lineOffsetsView, sourceIds},
    gridSize,
    intersectionCapacity: options.intersectionCapacity ?? 64,
    spatialSort: options.spatialSort,
    output
  });
  executeCommandNodesOnCPU(contributor.getCommandNodes(graph), memory);
  const read = (view: typeof output.status.count) => memory.readUint32(view)[0];
  const count = read(output.status.count);
  const lattice = Array.from(memory.readFloat32(output.lattice));
  const latticeRows = memory.readFloat32(output.latticeEndpoints, 4 * count);
  const worldRows = memory.readFloat32(output.endpoints, 4 * count);
  const edges = memory.readUint32(output.sourceEdgeIds, count);
  const pieces: ReferencePiece[] = Array.from({length: count}, (_, row) => ({
    segment: edges[row],
    from: [latticeRows[4 * row], latticeRows[4 * row + 1]],
    to: [latticeRows[4 * row + 2], latticeRows[4 * row + 3]]
  }));
  return {
    positions,
    lattice,
    count,
    required: read(output.status.requiredCount),
    overflow: read(output.status.overflow),
    candidateOverflow: read(output.status.candidateOverflow),
    uncertain: read(output.status.uncertainCount),
    pieces,
    worldRows,
    features: Array.from(memory.readUint32(output.sourceFeatureIds, count)),
    rings: Array.from(memory.readUint32(output.sourceRingIds, count)),
    startParameters: Array.from(memory.readFloat32(output.sourceStartParameters, count)),
    endParameters: Array.from(memory.readFloat32(output.sourceEndParameters, count)),
    gridSize,
    lines
  };
}

const NAN = Number.NaN;

describe('GPUSnapRoundedNoding WGSL executed on the CPU', () => {
  it('nodes a crossing at a half pixel and reroutes through a nearby vertex', () => {
    const cross = runNoding([
      [
        [0, 0],
        [4, 3]
      ],
      [
        [0, 3],
        [4, 0]
      ]
    ]);
    const reference = expectMatchesReference(cross, 'cross');
    // The crossing (2, 1.5) rounds half up to (2, 2): four pieces meeting at one lattice point.
    expect(reference.crossings.map(({point}) => point)).toEqual([[2, 2]]);
    expect(cross.count).toBe(4);
    expect(cross.overflow).toBe(0);

    const reroute = runNoding([
      [
        [0, 0],
        [10, 3]
      ],
      [
        [5, 1],
        [5, 8]
      ]
    ]);
    const rerouted = expectMatchesReference(reroute, 'reroute');
    // The long segment passes through the pixel of the vertex (5, 1) and is cut there.
    expect(rerouted.hotPixels).toContainEqual([5, 1]);
    expect(reroute.pieces.filter(piece => piece.segment === 0).length).toBeGreaterThan(1);
  });

  it('keeps coincident, collinear-overlapping, touching and degenerate input bit-identical', () => {
    for (const [label, lines] of [
      [
        'overlap',
        [
          [
            [0, 0],
            [6, 0]
          ],
          [
            [2, 0],
            [8, 0]
          ]
        ]
      ],
      [
        't-junction',
        [
          [
            [0, 0],
            [6, 0]
          ],
          [
            [3, 0],
            [3, 5]
          ]
        ]
      ],
      [
        'shared endpoint',
        [
          [
            [0, 0],
            [3, 3]
          ],
          [
            [3, 3],
            [6, 0]
          ]
        ]
      ],
      [
        'spike',
        [
          [
            [0, 0],
            [4, 0],
            [1, 0]
          ]
        ]
      ],
      [
        'ring crosses ring',
        [
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
      ],
      [
        'collapsing segment',
        [
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
      ],
      [
        'collapsed line',
        [
          [
            [1.1, 1.1],
            [1.2, 0.9]
          ],
          [
            [0, 0],
            [3, 3]
          ]
        ]
      ],
      [
        'three lines through one pixel',
        [
          [
            [0, 0],
            [8, 8]
          ],
          [
            [0, 8],
            [8, 0]
          ],
          [
            [0, 3],
            [8, 5]
          ]
        ]
      ],
      [
        'repeated vertices',
        [
          [
            [1, 1],
            [1, 1],
            [5, 3],
            [5, 3],
            [2, 6]
          ]
        ]
      ]
    ] as const) {
      const result = runNoding(
        lines.map(line => line.map(point => [...point] as [number, number]))
      );
      expectMatchesReference(result, label);
      expect(result.overflow, label).toBe(0);
    }
  });

  it('reports non-finite vertices and capacities, and orders pieces along each segment', () => {
    const nonFinite = runNoding([
      [
        [0, 0],
        [NAN, 1],
        [4, 4]
      ],
      [
        [0, 4],
        [4, 0]
      ]
    ]);
    expect(nonFinite.overflow).toBe(1);
    const small = runNoding(
      [
        [
          [0, 0],
          [8, 2]
        ],
        [
          [0, 2],
          [8, 0]
        ],
        [
          [0, 1],
          [8, 1]
        ]
      ],
      {outputCapacity: 3}
    );
    expect(small.overflow).toBe(1);
    expect(small.count).toBe(3);
    expect(small.required).toBeGreaterThan(3);
    const bounded = runNoding(
      [
        [
          [0, 0],
          [8, 2]
        ],
        [
          [0, 2],
          [8, 0]
        ]
      ],
      {intersectionCapacity: 1}
    );
    expect(bounded.overflow).toBe(0);
    // Provenance: source ids, line indices and edge ids; parameters are monotone along a segment.
    const provenance = runNoding(
      [
        [
          [0, 0],
          [6, 6],
          [0, 6]
        ],
        [
          [0, 4],
          [6, 4]
        ]
      ],
      {sourceIds: [70, 80]}
    );
    expectMatchesReference(provenance, 'provenance');
    provenance.pieces.forEach((piece, index) => {
      expect(provenance.rings[index]).toBe(piece.segment < 3 ? 0 : 1);
      expect(provenance.features[index]).toBe(piece.segment < 3 ? 70 : 80);
      expect(provenance.endParameters[index]).toBeGreaterThanOrEqual(
        provenance.startParameters[index]
      );
    });
  });

  it('matches the reference on a non-power-of-two grid, an offset origin and an extent-limited cell', () => {
    // gridSize 0.3 rounds up to the cell 0.5; coordinates far from zero keep their exact offsets.
    const offset = runNoding(
      [
        [
          [1000.25, 2000.5],
          [1004.75, 2003.25],
          [1001, 2006]
        ],
        [
          [1000.5, 2003.75],
          [1005, 2001]
        ]
      ],
      {gridSize: 0.3}
    );
    expect(offset.lattice[2]).toBe(0.5);
    expectMatchesReference(offset, 'offset');
    // A tiny grid on a large extent: the cell is the extent's power of two times 2^-22.
    const fine = runNoding(
      [
        [
          [0, 0],
          [100, 37]
        ],
        [
          [0, 37],
          [100, 0]
        ]
      ],
      {gridSize: 1e-9}
    );
    expect(fine.lattice[2]).toBe(2 ** (7 - 22));
    expectMatchesReference(fine, 'fine');
    // Negative coordinates and a half-pixel tie in a vertex.
    const negative = runNoding(
      [
        [
          [-3.5, -2.5],
          [2.5, 3.5]
        ],
        [
          [-2.5, 1.5],
          [1.5, -3.5]
        ]
      ],
      {gridSize: 1}
    );
    expectMatchesReference(negative, 'negative');
  });

  it('matches the reference on random polylines', () => {
    let crossings = 0;
    let pieces = 0;
    let reroutes = 0;
    for (let seed = 1; seed <= 6; seed++) {
      const random = createRandom(100 + seed);
      const lines = Array.from({length: 4}, () =>
        Array.from(
          {length: 2 + Math.floor(random() * 3)},
          () => [Math.floor(random() * 56) / 4, Math.floor(random() * 56) / 4] as [number, number]
        )
      );
      const result = runNoding(lines, {gridSize: 1, intersectionCapacity: 64, outputCapacity: 200});
      const reference = expectMatchesReference(result, `random ${seed}`);
      expect(result.overflow, `random ${seed}`).toBe(0);
      crossings += reference.crossings.length;
      pieces += result.count;
      reroutes += reference.pieces.length - reference.usableSegments;
    }
    // The seeds must exercise crossings and reroutes, not just pass trivially.
    expect(crossings).toBeGreaterThan(8);
    expect(pieces).toBeGreaterThan(60);
    expect(reroutes).toBeGreaterThan(30);
  });

  it('matches the reference on dense random input with many ties and collapses', () => {
    let collapsed = 0;
    for (let seed = 1; seed <= 6; seed++) {
      const random = createRandom(500 + seed);
      // Half-integer world coordinates on a grid of 2: every vertex is a rounding tie.
      const lines = Array.from({length: 7}, () =>
        Array.from(
          {length: 3 + Math.floor(random() * 4)},
          () => [Math.floor(random() * 20) / 2, Math.floor(random() * 20) / 2] as [number, number]
        )
      );
      const result = runNoding(lines, {
        gridSize: seed % 2 === 0 ? 2 : 1,
        intersectionCapacity: 400,
        outputCapacity: 600,
        spatialSort: seed % 3 === 0
      });
      const reference = expectMatchesReference(result, `dense ${seed}`);
      expect(result.overflow, `dense ${seed}`).toBe(0);
      collapsed += reference.hotPixels.length < result.positions.length / 2 ? 1 : 0;
    }
    expect(collapsed).toBeGreaterThan(2);
  });

  it('handles a degenerate or unrepresentable coordinate range', () => {
    // Zero extent: every vertex lands on one lattice point and the lines collapse to nothing.
    const point = runNoding(
      [
        [
          [5, 5],
          [5, 5],
          [5, 5]
        ],
        [
          [5, 5],
          [5, 5]
        ]
      ],
      {gridSize: 0.25}
    );
    expect(point.count).toBe(0);
    expect(point.overflow).toBe(0);
    expect(point.lattice[2]).toBe(0.25);
    // An extent beyond the float32 range cannot be placed on the lattice: reported, not guessed.
    const huge = runNoding([
      [
        [-3e38, 0],
        [3e38, 1]
      ],
      [
        [0, 1],
        [1, 0]
      ]
    ]);
    expect(huge.overflow).toBe(1);
    // The frame falls back to the unit lattice at the origin: the two huge vertices are dropped
    // (NaN) and the small line survives.
    expect(huge.lattice).toEqual([0, 0, 1, 0]);
    expect(huge.pieces.map(piece => piece.segment)).toEqual([2]);
    // A large offset with a small extent: the lattice origin is a huge multiple of the cell.
    const far = runNoding(
      [
        [
          [10000000, 20000000],
          [10000008, 20000004]
        ],
        [
          [10000000, 20000004],
          [10000008, 20000000]
        ]
      ],
      {gridSize: 0.25}
    );
    expectMatchesReference(far, 'far offset');
    expect(far.count).toBe(4);
    // Huge coordinates with a tiny relative extent.
    const wide = runNoding(
      [
        [
          [1e30, 1e30],
          [1.0000001e30, 1.0000002e30]
        ],
        [
          [1e30, 1.0000002e30],
          [1.0000001e30, 1e30]
        ]
      ],
      {gridSize: 1}
    );
    expectMatchesReference(wide, 'wide');
  });
});

it('keeps the full incidence budget when only eight output pieces are requested', () => {
  const random = createRandom(77);
  const lines: [number, number][][] = Array.from({length: 10}, () =>
    Array.from({length: 4}, () => [Math.floor(random() * 80) / 4, Math.floor(random() * 80) / 4])
  );
  const full = runNoding(lines, {outputCapacity: 256, intersectionCapacity: 128});
  const reference = expectMatchesReference(full, 'full random linework');
  const small = runNoding(lines, {outputCapacity: 8, intersectionCapacity: 128});
  expect(small.candidateOverflow).toBe(0);
  expect(small.overflow).toBe(1);
  expect(small.required).toBe(reference.pieces.length);
  expect(small.count).toBe(8);
  expect(small.pieces).toEqual(reference.pieces.slice(0, 8));
}, 120_000);
