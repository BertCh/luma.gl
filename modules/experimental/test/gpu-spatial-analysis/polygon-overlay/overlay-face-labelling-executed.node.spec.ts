// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Runs the real face-labelling WGSL of GPUPolygonOverlay on the CPU (wgsl_reflect's interpreter,
// see segment-intersection/wgsl-graph-executor.ts) on small inputs. Oracle for the crossing order:
// exact BigInt rational arithmetic at the symbolic ray height.

import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';
import {createTransientView, GPUCommandGraph, type GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {describe, expect, it} from 'vitest';
import {GPUPolygonOverlay} from '../../../src/gpu-spatial-analysis/polygon-overlay/index';
import type {
  GPUPolygonOverlayFillRule,
  GPUPolygonOverlayOperation
} from '../../../src/gpu-spatial-analysis/polygon-overlay/index';
import {createWGSLKernelNode} from '../../../src/utils/wgsl-kernel-nodes';
import {
  createPointerJumpNodes,
  FACE_CROSSING_ORDER_WGSL,
  getPointerJumpRoundCount
} from '../../../src/gpu-spatial-analysis/polygon-overlay/overlay-face-labelling-wgsl';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {createOverlayABScene} from './ab/overlay-ab-scene';
import fixtureJson from './fixtures/overlay-snap-rounding.json';
import {packOperand, planOverlayCapacity, type OverlayOperand} from './overlay-gpu-harness';
import {referenceSnapRound} from '../geometry-topology/snap-rounding-reference';

// The sorts, scans and the BVH build are cooperative upstream kernels (workgroup barriers): CPU
// stand-ins, see segment-intersection/cpu-gpu-core-emulations.ts.
useCPUExecutorTestHooks();

type Edge = [number, number, number, number];
type CrossingCase = {first: Edge; second: Edge; doubledHeight: number; offsetSign: number};

const OFFSET_SCALE = 2n ** 100n;

/**
 * Sign of x1 - x2 where the edges meet the ray at height Y + offsetSign * tau, tau = 2^-100 (far
 * below the smallest nonzero gap between two such abscissae, about 2^-92), in exact rationals.
 */
function compareCrossingsExact(testCase: CrossingCase): number {
  const abscissa = (edge: Edge) => {
    const [lowX, lowY, highX, highY] = edge.map(BigInt);
    const deltaX = highX - lowX;
    const deltaY = highY - lowY;
    const height = BigInt(testCase.doubledHeight) * OFFSET_SCALE + 2n * BigInt(testCase.offsetSign);
    // x = lowX + (height / (2 S) - lowY) * deltaX / deltaY, as numerator / (2 S deltaY).
    return {
      numerator: 2n * OFFSET_SCALE * deltaY * lowX + (height - 2n * OFFSET_SCALE * lowY) * deltaX,
      denominator: 2n * OFFSET_SCALE * deltaY
    };
  };
  const first = abscissa(testCase.first);
  const second = abscissa(testCase.second);
  const difference = first.numerator * second.denominator - second.numerator * first.denominator;
  return difference > 0n ? 1 : difference < 0n ? -1 : 0;
}

function runCompareCrossingsOnCPU(cases: readonly CrossingCase[]): number[] {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'crossing-order'});
  const inputs = createTransientView(graph, 'cases', 'float32', cases.length * 10);
  const results = createTransientView(graph, 'results', 'uint32', cases.length);
  const node = createWGSLKernelNode(graph, {
    id: 'compare-crossings',
    operation: 'test',
    bindings: [
      {name: 'cases', view: inputs, type: 'f32', access: 'read'},
      {name: 'results', view: results, type: 'i32', access: 'read_write'}
    ],
    invocationCount: cases.length,
    workgroupSize: 32,
    declarations: FACE_CROSSING_ORDER_WGSL,
    body: `let base = casesOffset + index * 10u;
  let first = vec4f(cases[base], cases[base + 1u], cases[base + 2u], cases[base + 3u]);
  let second = vec4f(cases[base + 4u], cases[base + 5u], cases[base + 6u], cases[base + 7u]);
  results[resultsOffset + index] = u32(compareCrossings(first, second, i32(cases[base + 8u]), i32(cases[base + 9u])));`
  });
  const memory = createCPUGraphMemory();
  memory.write(
    inputs,
    Float32Array.from(
      cases.flatMap(testCase => [
        ...testCase.first,
        ...testCase.second,
        testCase.doubledHeight,
        testCase.offsetSign
      ])
    )
  );
  executeCommandNodesOnCPU([node], memory);
  return Array.from(new Int32Array(memory.readUint32(results).buffer));
}

function createRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

describe('face labelling crossing order executed on the CPU', () => {
  it('orders crossings exactly: random, near-equal, shared-endpoint and extreme cases', () => {
    const random = createRandom(20261014);
    const maximum = 2 ** 22 + 1;
    const integer = (limit: number) => Math.floor(random() * (limit + 1));
    const edge = (limit: number): Edge => {
      const lowX = integer(limit);
      const lowY = integer(limit - 1);
      return [lowX, lowY, integer(limit), lowY + 1 + integer(limit - lowY - 1)];
    };
    const cases: CrossingCase[] = [];
    const offsetSign = () => (random() < 0.5 ? -1 : 1);
    // Random edges at a random half-integer height inside the first edge.
    for (const limit of [8, 64, maximum]) {
      for (let k = 0; k < 60; k++) {
        const first = edge(limit);
        cases.push({
          first,
          second: edge(limit),
          doubledHeight: 2 * first[1] + integer(2 * (first[3] - first[1])),
          offsetSign: offsetSign()
        });
      }
    }
    // Shared lower endpoint at height Y: equal x(Y), decided by the slopes in both directions.
    for (let k = 0; k < 80; k++) {
      const first = edge(k < 40 ? 16 : maximum);
      const second: Edge = [
        first[0],
        first[1],
        integer(k < 40 ? 16 : maximum),
        first[1] + 1 + integer(Math.max(0, (k < 40 ? 16 : maximum) - first[1] - 1))
      ];
      cases.push({first, second, doubledHeight: 2 * first[1], offsetSign: offsetSign()});
    }
    // Shared upper endpoint at height Y.
    for (let k = 0; k < 40; k++) {
      const first = edge(maximum);
      const second: Edge = [integer(maximum), integer(first[3] - 1), first[2], first[3]];
      cases.push({first, second, doubledHeight: 2 * first[3], offsetSign: offsetSign()});
    }
    // Same line, different extents: equal slopes, order 0 at any height.
    cases.push({first: [0, 0, 6, 3], second: [2, 1, 4, 2], doubledHeight: 3, offsetSign: 1});
    // Nearly parallel long edges whose abscissae differ by 1 / (2 dy1 dy2), the closest possible.
    cases.push({
      first: [0, 0, maximum, maximum],
      second: [0, 0, maximum - 1, maximum],
      doubledHeight: 2 * maximum - 1,
      offsetSign: 1
    });
    cases.push({
      first: [maximum, 0, 0, maximum],
      second: [maximum - 1, 0, 0, maximum - 1],
      doubledHeight: maximum,
      offsetSign: -1
    });
    cases.push({
      first: [0, 0, maximum, maximum - 1],
      second: [1, 0, maximum, maximum],
      doubledHeight: 1,
      offsetSign: -1
    });
    const results = runCompareCrossingsOnCPU(cases);
    const expected = cases.map(compareCrossingsExact);
    const mismatches = cases
      .map((testCase, k) => ({testCase, result: results[k], expected: expected[k]}))
      .filter(entry => entry.result !== entry.expected);
    expect(mismatches.slice(0, 5)).toEqual([]);
    // The suite must exercise all three outcomes and the slope tie-break.
    expect(new Set(expected).size).toBe(3);
  });
});

type FixtureCase = {
  name: string;
  template: string;
  gridSize: number;
  lattice: {origin: [number, number]; cell: number};
  operation: GPUPolygonOverlayOperation;
  fillRule: GPUPolygonOverlayFillRule;
  method: 'shapely' | 'faces' | 'snap-rounded-faces';
  left: OverlayOperand;
  right: OverlayOperand | null;
  expected: {edges: [number, number, number, number][]; ringCount: number; area2: number};
  shapelyExpected?: {edges: [number, number, number, number][]; ringCount: number; area2: number};
  oracleDifference?: {
    reason: string;
    roundedInputValidity: string[];
    witness: {
      latticePoint: [number, number];
      insideOperands: boolean[];
      insideShapelyIntersection: boolean;
    };
  };
};

const FIXTURE = fixtureJson as unknown as {cases: FixtureCase[]};

/** Small capacities keep the interpreter fast; a case that overflows them reruns with the large ones. */
const SMALL_CAPACITY = {
  intersections: 48,
  nodedSegments: 96,
  boundary: 96,
  vertices: 96,
  rings: 12
};
const LARGE_CAPACITY = {
  intersections: 512,
  nodedSegments: 768,
  boundary: 768,
  vertices: 768,
  rings: 48
};
type Capacity = typeof SMALL_CAPACITY;

/** Directed lattice edges split into primitive steps: comparable across any noding of one boundary. */
function toPrimitiveSteps(edges: readonly (readonly number[])[]): string[] {
  const steps: string[] = [];
  const gcd = (a: number, b: number): number => (b === 0 ? a : gcd(b, a % b));
  for (const [x0, y0, x1, y1] of edges) {
    const dx = x1 - x0;
    const dy = y1 - y0;
    const divisor = gcd(Math.abs(dx), Math.abs(dy));
    for (let k = 0; k < divisor; k++) {
      steps.push(
        [
          x0 + (k * dx) / divisor,
          y0 + (k * dy) / divisor,
          x0 + ((k + 1) * dx) / divisor,
          y0 + ((k + 1) * dy) / divisor
        ].join(',')
      );
    }
  }
  return steps.sort();
}

function runOverlayOnCPU(
  fixture: FixtureCase,
  CAPACITY: Capacity = SMALL_CAPACITY,
  publishCapacity = false
) {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'overlay-cpu'});
  const memory = createCPUGraphMemory();
  let serial = 0;
  const upload = (operand: OverlayOperand) => {
    const packed = packOperand(operand);
    const make = <Format extends 'float32x2' | 'uint32'>(
      format: Format,
      data: Float32Array | Uint32Array
    ) => {
      const view = createTransientView(
        graph,
        `operand-${serial++}`,
        format,
        format === 'uint32' ? data.length : data.length / 2
      );
      memory.write(view, data);
      return view;
    };
    return {
      kind: 'polygons' as const,
      positions: make('float32x2', packed.positions),
      featureOffsets: make('uint32', packed.featureOffsets),
      polygonOffsets: make('uint32', packed.polygonOffsets),
      ringOffsets: make('uint32', packed.ringOffsets)
    };
  };
  const scalar = () => createTransientView(graph, `scalar-${serial++}`, 'uint32', 1);
  const status = () => ({
    count: scalar(),
    requiredCount: scalar(),
    overflow: scalar(),
    candidateOverflow: scalar()
  });
  const column = (length: number) =>
    createTransientView(graph, `column-${serial++}`, 'uint32', length);
  const output = {
    geometry: {
      kind: 'polygons' as const,
      positions: createTransientView(graph, 'out-positions', 'float32x2', CAPACITY.vertices),
      featureOffsets: column(CAPACITY.rings + 1),
      polygonOffsets: column(CAPACITY.rings + 1),
      ringOffsets: column(CAPACITY.rings + 1),
      sourceIds: column(CAPACITY.rings)
    },
    status: {...status(), nodedSegmentRequiredCount: scalar()},
    boundary: {
      endpoints: createTransientView(graph, 'out-boundary', 'float32x4', CAPACITY.boundary),
      operandIds: column(CAPACITY.boundary),
      sourceFeatureIds: column(CAPACITY.boundary),
      status: status()
    }
  };
  const overlay = new GPUPolygonOverlay({
    left: upload(fixture.left),
    right: fixture.right ? upload(fixture.right) : undefined,
    operation: fixture.operation,
    fillRule: fixture.fillRule,
    capacity: {intersections: CAPACITY.intersections, nodedSegments: CAPACITY.nodedSegments},
    vertexTolerance: fixture.gridSize,
    output
  });
  const nodes = overlay.getCommandNodes(graph);
  for (const node of nodes) {
    // Up to the boundary emission: the ring assembly (analysis-core) has its own specs.
    if (!publishCapacity && node.id.includes('-assemble-')) {
      break;
    }
    try {
      executeCommandNodesOnCPU([node], memory);
      if (publishCapacity && node.id.endsWith('-assemble-ring-count')) {
        // Ring/vertex counts are now complete. The cooperative ring-stats kernel cannot run in
        // the interpreter; it does not feed these counts. Execute the actual publication path.
        for (const suffix of ['-assemble-publish', '-source-ids', '-publish']) {
          const publish = nodes.find(candidate => candidate.id === `polygon-overlay${suffix}`)!;
          executeCommandNodesOnCPU([publish], memory);
        }
        break;
      }
    } catch (error) {
      throw new Error(`node ${node.id}: ${(error as Error).message}`, {cause: error});
    }
  }
  const views = new Map<string, GraphDataView>();
  for (const node of nodes as unknown as {resources?: {buffer?: GraphDataView}[]}[]) {
    for (const resource of node.resources ?? []) {
      if (resource.buffer?.buffer) {
        views.set((resource.buffer.buffer as {id: string}).id, resource.buffer);
      }
    }
  }
  const nodedCount = memory.readUint32(views.get('polygon-overlay-noded-count')!)[0];
  const nodedLattice = memory.readFloat32(
    views.get('polygon-overlay-noded-lattice-endpoints')!,
    4 * nodedCount
  );
  const nodedRings = memory.readUint32(views.get('polygon-overlay-noded-rings')!, nodedCount);
  const boundaryCount = memory.readUint32(output.boundary.status.count)[0];
  const world = memory.readFloat32(output.boundary.endpoints, 4 * boundaryCount);
  const {origin, cell} = fixture.lattice;
  const edges: [number, number, number, number][] = [];
  for (let row = 0; row < boundaryCount; row++) {
    edges.push([
      (world[4 * row] - origin[0]) / cell,
      (world[4 * row + 1] - origin[1]) / cell,
      (world[4 * row + 2] - origin[0]) / cell,
      (world[4 * row + 3] - origin[1]) / cell
    ]);
  }
  return {
    edges,
    nodedPieces: Array.from({length: nodedCount}, (_, row) => ({
      from: [nodedLattice[4 * row], nodedLattice[4 * row + 1]] as [number, number],
      to: [nodedLattice[4 * row + 2], nodedLattice[4 * row + 3]] as [number, number],
      ring: nodedRings[row]
    })),
    overflow: memory.readUint32(output.status.overflow)[0],
    candidateOverflow: memory.readUint32(output.status.candidateOverflow)[0],
    boundaryRequired: memory.readUint32(output.boundary.status.requiredCount)[0],
    ringRequired: memory.readUint32(output.status.requiredCount)[0],
    boundaryOverflow: memory.readUint32(output.boundary.status.overflow)[0],
    nodedRequired: memory.readUint32(output.status.nodedSegmentRequiredCount)[0],
    writtenRingCount: publishCapacity
      ? memory.readUint32(views.get('polygon-overlay-ring-count')!)[0]
      : 0
  };
}

type NodedPiece = {from: [number, number]; to: [number, number]; ring: number};

/** Operand (0 left, 1 right) and polygon index of every ring, in the overlay's combined ring order. */
function getRingTable(fixture: FixtureCase) {
  const rings: {operand: number; polygon: number}[] = [];
  let polygon = 0;
  [fixture.left, fixture.right ?? []].forEach((operand, operandIndex) => {
    for (const feature of operand) {
      for (const polygonRings of feature) {
        for (let ring = 0; ring < polygonRings.length; ring++) {
          rings.push({operand: operandIndex, polygon});
        }
        polygon++;
      }
    }
  });
  return rings;
}

const isInside = (rule: GPUPolygonOverlayFillRule, winding: number) =>
  rule === 'even-odd'
    ? winding % 2 !== 0
    : rule === 'positive'
      ? winding > 0
      : rule === 'negative'
        ? winding < 0
        : winding !== 0;

/**
 * Independent oracle of the labelling: the directed boundary edges (as primitive steps) of the
 * Boolean result, from the noded pieces. A probe a hair beside the midpoint of every piece group is
 * classified by a ray cast against the pieces of each polygon (double precision is exact enough:
 * lattice coordinates are below 100, probes sit 1e-3 from the pieces and pieces are at least
 * 1/100 apart), then the fill rule and the operation give the result on both sides.
 */
function getLabellingOracleSteps(fixture: FixtureCase, pieces: readonly NodedPiece[]): string[] {
  const rings = getRingTable(fixture);
  const polygonOperand = new Map<number, number>();
  for (const ring of rings) {
    polygonOperand.set(ring.polygon, ring.operand);
  }
  const windingOf = (polygon: number, x: number, y: number) => {
    let winding = 0;
    for (const piece of pieces) {
      if (rings[piece.ring].polygon !== polygon) {
        continue;
      }
      const [x0, y0] = piece.from;
      const [x1, y1] = piece.to;
      if (y0 > y !== y1 > y) {
        const crossingX = x0 + ((y - y0) * (x1 - x0)) / (y1 - y0);
        if (crossingX > x) {
          winding += y1 > y0 ? 1 : -1;
        }
      }
    }
    return winding;
  };
  const resultAt = (x: number, y: number) => {
    const inside = [false, false];
    for (const polygon of polygonOperand.keys()) {
      if (isInside(fixture.fillRule, windingOf(polygon, x, y))) {
        inside[polygonOperand.get(polygon)!] = true;
      }
    }
    const [a, b] = inside;
    switch (fixture.operation) {
      case 'intersection':
        return a && b;
      case 'union':
        return a || b;
      case 'difference':
        return a && !b;
      case 'symmetric-difference':
        return a !== b;
      default:
        return a;
    }
  };
  const groups = new Map<string, [number, number, number, number]>();
  for (const {from, to} of pieces) {
    const reversed = from[1] > to[1] || (from[1] === to[1] && from[0] > to[0]);
    const [low, high] = reversed ? [to, from] : [from, to];
    groups.set(`${low}|${high}`, [low[0], low[1], high[0], high[1]]);
  }
  const edges: [number, number, number, number][] = [];
  for (const [x0, y0, x1, y1] of groups.values()) {
    const length = Math.hypot(x1 - x0, y1 - y0);
    // Left normal of lo -> hi, and a fixed irrational shift so no probe ray hits a vertex.
    const offsetX = (-(y1 - y0) / length) * 1e-3;
    const offsetY = ((x1 - x0) / length) * 1e-3;
    const middleX = (x0 + x1) / 2 + 1.7e-6;
    const middleY = (y0 + y1) / 2 + 1.3e-6;
    const left = resultAt(middleX + offsetX, middleY + offsetY);
    const right = resultAt(middleX - offsetX, middleY - offsetY);
    if (left !== right) {
      edges.push(left ? [x0, y0, x1, y1] : [x1, y1, x0, y0]);
    }
  }
  return toPrimitiveSteps(edges);
}

/** Independent BigInt noding of the rounded source rings, including their directed provenance. */
function getReferenceNodedPieces(fixture: FixtureCase): NodedPiece[] {
  const {origin, cell} = fixture.lattice;
  const sourceRings = [fixture.left, fixture.right ?? []].flatMap(operand =>
    operand.flatMap(feature => feature.flat())
  );
  const lines = sourceRings.map(ring => {
    const points = ring.map(([x, y]): [number, number] => [
      Math.floor((x - origin[0]) / cell + 0.5),
      Math.floor((y - origin[1]) / cell + 0.5)
    ]);
    return [...points, points[0]];
  });
  const sourceRingIds = lines.flatMap((line, ring) => line.map(() => ring));
  return referenceSnapRound(lines).pieces.map(({from, to, segment}) => ({
    from: [from[0], from[1]],
    to: [to[0], to[1]],
    ring: sourceRingIds[segment]
  }));
}

// Every template on the unit grid by default (about a minute in the interpreter); the 0.25 and
// 1/64 grids rerun the same scenes at other scales (A5_ALL_FIXTURES=1).
const EXECUTED_CASES = process.env['A5_ALL_FIXTURES']
  ? FIXTURE.cases
  : FIXTURE.cases.filter(fixture => fixture.gridSize === 1 || fixture.oracleDifference);

describe('face labelling pointer jumping executed on the CPU', () => {
  it('sums the contribution of every ancestor along chains and forests', () => {
    const random = createRandom(7);
    for (const [count, componentCount] of [
      [70, 1],
      [100, 2],
      [33, 2]
    ] as const) {
      // A forest: node i has a parent below it (acyclic), a long chain among the first nodes.
      const parents = Array.from({length: count}, (_, node) =>
        node === count - 1 || (node > 5 && random() < 0.1)
          ? 0xffffffff
          : node < 40
            ? node + 1
            : node + 1 + Math.floor(random() * Math.min(3, count - node - 1))
      );
      const deltas = Array.from(
        {length: count * componentCount},
        () => Math.floor(random() * 5) - 2
      );
      const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: `jump-${count}`});
      const parentsView = createTransientView(graph, 'parents', 'uint32', count);
      const valuesView = createTransientView(graph, 'values', 'uint32', count * componentCount);
      const {nodes, values} = createPointerJumpNodes(graph, {
        id: 'jump',
        operation: 'GPUPolygonOverlay',
        parents: parentsView,
        values: valuesView,
        componentCount,
        rounds: getPointerJumpRoundCount(count)
      });
      const memory = createCPUGraphMemory();
      memory.write(parentsView, Uint32Array.from(parents));
      // Initial value: the contribution of the parent.
      memory.write(
        valuesView,
        Int32Array.from(
          parents.flatMap((parent, node) =>
            Array.from({length: componentCount}, (_, component) =>
              parent === 0xffffffff ? 0 : deltas[parent * componentCount + component]
            )
          )
        )
      );
      executeCommandNodesOnCPU(nodes, memory);
      const actual = new Int32Array(memory.readUint32(values).buffer);
      const expected = parents.flatMap((_, node) =>
        Array.from({length: componentCount}, (_, component) => {
          let sum = 0;
          for (
            let ancestor = parents[node];
            ancestor !== 0xffffffff;
            ancestor = parents[ancestor]
          ) {
            sum += deltas[ancestor * componentCount + component];
          }
          return sum;
        })
      );
      expect(Array.from(actual)).toEqual(expected);
    }
  });
});

describe('GPUPolygonOverlay face labelling executed on the CPU', () => {
  it('reproduces the A/B intersection ring-capacity overflow at 1000 vertices per operand', () => {
    const scene = createOverlayABScene(1000);
    const fixture: FixtureCase = {
      name: 'ab-1000',
      template: 'ab-1000',
      gridSize: 1 / 64,
      lattice: {origin: [0, 0], cell: 1 / 64},
      operation: 'intersection',
      fillRule: 'even-odd',
      method: 'shapely',
      left: scene.left,
      right: scene.right,
      expected: {edges: [], ringCount: 521, area2: 0}
    };
    const result = runOverlayOnCPU(
      fixture,
      {
        intersections: 20000,
        nodedSegments: 80000,
        vertices: 20000,
        boundary: 40000,
        rings: 512
      },
      true
    );
    expect(result).toMatchObject({
      overflow: 1,
      candidateOverflow: 0,
      boundaryOverflow: 0,
      ringRequired: 521,
      writtenRingCount: 512
    });
    expect(result.nodedRequired).toBeLessThan(80000);
  }, 1_200_000);

  it('publishes unclamped ring and noded requirements and sizes a complete repeat run', () => {
    const fixture: FixtureCase = {
      name: 'capacity-plan',
      template: 'capacity-plan',
      gridSize: 1,
      lattice: {origin: [-1, -1], cell: 1},
      operation: 'dissolve',
      fillRule: 'even-odd',
      method: 'shapely',
      right: null,
      left: [0, 4, 8].map(offset => [
        [
          [
            [offset, 0],
            [offset + 2, 0],
            [offset + 2, 2],
            [offset, 2]
          ]
        ]
      ]),
      expected: {edges: [], ringCount: 3, area2: 24}
    };
    const capacity = {...SMALL_CAPACITY, boundary: 12, rings: 1};
    const limited = runOverlayOnCPU(fixture, capacity, true);
    expect(limited).toMatchObject({
      overflow: 1,
      candidateOverflow: 0,
      boundaryOverflow: 0,
      ringRequired: 3,
      writtenRingCount: 1,
      boundaryRequired: 12,
      nodedRequired: 12
    });
    const planned = planOverlayCapacity(capacity, {
      rings: [],
      featureCount: 1,
      requiredRingCount: limited.ringRequired,
      overflow: limited.overflow,
      candidateOverflow: limited.candidateOverflow,
      boundaryCount: limited.edges.length,
      boundaryRequiredCount: limited.boundaryRequired,
      boundaryOverflow: limited.boundaryOverflow,
      nodedSegmentRequiredCount: limited.nodedRequired
    });
    expect(planned).toEqual({...capacity, rings: 3, vertices: 15, boundary: 12, nodedSegments: 12});
    const complete = runOverlayOnCPU(fixture, planned, true);
    expect(complete).toMatchObject({
      overflow: 0,
      ringRequired: 3,
      writtenRingCount: 3,
      nodedRequired: 12
    });
    const nodedLimited = runOverlayOnCPU(fixture, {...capacity, nodedSegments: 4});
    expect(nodedLimited.nodedRequired).toBe(12);
    expect(nodedLimited.boundaryOverflow).toBe(1);
    expect(() =>
      planOverlayCapacity(capacity, {
        rings: [],
        featureCount: 0,
        requiredRingCount: 0,
        overflow: 1,
        candidateOverflow: 0,
        boundaryCount: 4,
        boundaryRequiredCount: 4,
        boundaryOverflow: 1,
        nodedSegmentRequiredCount: 12
      })
    ).toThrow(/Incomplete overlay plan/);
  }, 600_000);

  for (const fixture of EXECUTED_CASES) {
    it(fixture.name, () => {
      let result = runOverlayOnCPU(fixture);
      if (result.overflow || result.boundaryOverflow) {
        result = runOverlayOnCPU(fixture, LARGE_CAPACITY);
      }
      expect(result.overflow, 'overflow').toBe(0);
      expect(result.candidateOverflow, 'candidateOverflow').toBe(0);
      expect(result.boundaryOverflow, 'boundary overflow').toBe(0);
      const actual = toPrimitiveSteps(result.edges);
      // The labelling equals an independent winding oracle on the same noded pieces ...
      expect(actual).toEqual(getLabellingOracleSteps(fixture, result.nodedPieces));
      // Every fixture has an independent expected boundary; no exception skips the oracle.
      expect(actual).toEqual(toPrimitiveSteps(fixture.expected.edges));
      if (fixture.oracleDifference) {
        const referencePieces = getReferenceNodedPieces(fixture);
        expect(result.nodedPieces).toEqual(referencePieces);
        expect(actual).toEqual(getLabellingOracleSteps(fixture, referencePieces));
        expect(fixture.method).toBe('snap-rounded-faces');
        expect(
          fixture.oracleDifference.roundedInputValidity.every(reason =>
            /intersection/i.test(reason)
          )
        ).toBe(true);
        expect(fixture.oracleDifference.witness).toEqual({
          latticePoint: [31.25, 29.5],
          insideOperands: [false, true],
          insideShapelyIntersection: true
        });
        const [pointX, pointY] = fixture.oracleDifference.witness.latticePoint;
        const windings = [0, 0];
        for (const {from, to, ring} of referencePieces) {
          const orientation =
            (to[0] - from[0]) * (pointY - from[1]) - (to[1] - from[1]) * (pointX - from[0]);
          if (from[1] <= pointY && pointY < to[1] && orientation > 0) {
            windings[ring]++;
          } else if (to[1] <= pointY && pointY < from[1] && orientation < 0) {
            windings[ring]--;
          }
        }
        expect(windings).toEqual([0, 1]);
        expect(fixture.expected.area2).toBe(5);
        expect(fixture.expected.ringCount).toBe(1);
        expect(fixture.shapelyExpected?.area2).toBe(107);
        expect(fixture.shapelyExpected?.ringCount).toBe(4);
        expect(actual).not.toEqual(toPrimitiveSteps(fixture.shapelyExpected!.edges));
      }
    }, 600_000);
  }
});
