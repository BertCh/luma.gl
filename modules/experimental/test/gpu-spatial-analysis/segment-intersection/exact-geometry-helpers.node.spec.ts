// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {WgslReflect} from 'wgsl_reflect';
import {describe, expect, it} from 'vitest';
import {GPUPolygonOverlay} from '../../../src/gpu-spatial-analysis/polygon-overlay';
import {GPUSegmentRingAssembly} from '../../../src/analysis-core/ring-assembly';
import {GPUCellCover} from '../../../src/gpu-spatial-analysis/cell-cover';
import {GPULineLengthPerPolygon} from '../../../src/gpu-spatial-analysis/line-density';
import {GPUMakeValid} from '../../../src/gpu-spatial-analysis/geometry-topology/gpu-make-valid';
import {GPUZoneEvents} from '../../../src/gpu-spatial-analysis/trajectory-zones';
import {GPULineNoding} from '../../../src/gpu-spatial-analysis/geometry-topology';
import {
  CORE_PRIVATE_MEMORY_LIMIT_BYTES,
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getPrivateMemoryBytes
} from '../../utils/gpu-contributor-test-utils';
import {getInlinedCallSites, getUndeclaredCalls} from './wgsl-call-sites';
import {
  createPointProbe,
  createSideProbe,
  doSegmentsCrossExact,
  getExactRingWinding,
  isInsideRingsExact,
  orientSignExact,
  type Point
} from './exact-geometry-reference';

// Kernels that decide topology through the shared exact helpers (utils/exact-geometry-wgsl.ts).
// Each must parse, declare every function once, and reach orientSign through a single inlined
// call chain: shader compilers inline functions, and a second copy of the 256-bit fallback made
// the spatial-join kernels 2.5x slower.

type Format = 'float32' | 'float32x2' | 'float32x4' | 'uint32' | 'uint32x2' | 'uint32x3';

function createViewFactory(graph: GPUCommandGraph) {
  let serial = 0;
  return <F extends Format>(format: F, length: number) =>
    createTransientView(graph, `exact-helpers-${serial++}`, format, length);
}

/** Function bodies by name, and the names of the compute entry points. */
function parseFunctions(source: string): {bodies: Map<string, string>; entries: string[]} {
  const bodies = new Map<string, string>();
  const entries: string[] = [];
  for (const match of source.matchAll(/(@compute[^\n]*\n?\s*)?fn\s+(\w+)\s*\(/g)) {
    const name = match[2];
    if (match[1]) {
      entries.push(name);
    }
    const open = source.indexOf('{', match.index);
    let depth = 0;
    let end = open;
    for (; end < source.length; end++) {
      if (source[end] === '{') {
        depth++;
      } else if (source[end] === '}') {
        depth--;
        if (depth === 0) {
          break;
        }
      }
    }
    bodies.set(name, source.slice(open, end));
  }
  return {bodies, entries};
}

function calls(body: string, name: string): number {
  return body.match(new RegExp(`\\b${name}\\s*\\(`, 'g'))?.length ?? 0;
}

/**
 * Number of inlined copies of every function on a call chain to `orientSign`, counted over the
 * functions reachable from the compute entry point (dead helpers are dropped by compilers).
 */
function getOrientCallSites(source: string): Map<string, number> {
  const {bodies, entries} = parseFunctions(source);
  const reachable = new Set(entries);
  const queue = [...entries];
  while (queue.length > 0) {
    const body = bodies.get(queue.pop()!) ?? '';
    for (const name of bodies.keys()) {
      if (!reachable.has(name) && calls(body, name) > 0) {
        reachable.add(name);
        queue.push(name);
      }
    }
  }
  const chain = new Set(['orientSign']);
  let grown = true;
  while (grown) {
    grown = false;
    for (const name of reachable) {
      const body = bodies.get(name) ?? '';
      if (!chain.has(name) && [...chain].some(callee => calls(body, callee) > 0)) {
        chain.add(name);
        grown = true;
      }
    }
  }
  const sites = new Map<string, number>();
  for (const callee of chain) {
    let count = 0;
    for (const name of reachable) {
      count += calls(bodies.get(name) ?? '', callee);
    }
    if (!entries.includes(callee)) {
      sites.set(callee, count);
    }
  }
  return sites;
}

function expectExactKernels(nodes: readonly unknown[], label: string): number {
  const kernels = getGPUCommandNodeKernelSources(nodes);
  expect(kernels.length, label).toBeGreaterThan(0);
  let exactKernels = 0;
  for (const kernel of kernels) {
    const {source} = kernel;
    expect(() => new WgslReflect(source), `${label} ${kernel.id}`).not.toThrow();
    const definitions = [...source.matchAll(/fn\s+(\w+)\s*\(/g)].map(match => match[1]);
    const duplicates = definitions.filter((name, index) => definitions.indexOf(name) !== index);
    expect(duplicates, `${label} ${kernel.id} duplicate functions`).toEqual([]);
    // Core-device limit (library rule 16): at most eight storage buffers per stage.
    const storageBindings = source.match(/var<storage/g)?.length ?? 0;
    expect(storageBindings, `${label} ${kernel.id} storage bindings`).toBeLessThanOrEqual(8);
    if (!source.includes('fn exactOrientSigns(')) {
      continue;
    }
    exactKernels++;
    for (const [name, count] of getOrientCallSites(source)) {
      expect(count, `${label} ${kernel.id} call sites of ${name}`).toBeLessThanOrEqual(1);
    }
  }
  return exactKernels;
}

describe('exact geometry kernels', () => {
  it('GPUPolygonOverlay face labelling is exact for every fill rule', () => {
    for (const fillRule of ['even-odd', 'positive', 'negative', 'nonzero'] as const) {
      const graph = new GPUCommandGraph(createNullWebGPUDevice());
      const view = createViewFactory(graph);
      const polygons = () => ({
        kind: 'polygons' as const,
        positions: view('float32x2', 8),
        featureOffsets: view('uint32', 2),
        polygonOffsets: view('uint32', 2),
        ringOffsets: view('uint32', 3)
      });
      const status = () => ({
        count: view('uint32', 1),
        requiredCount: view('uint32', 1),
        overflow: view('uint32', 1),
        candidateOverflow: view('uint32', 1)
      });
      const sourceIds = view('uint32', 8);
      const overlay = new GPUPolygonOverlay({
        left: polygons(),
        right: polygons(),
        operation: 'intersection',
        fillRule,
        capacity: {intersections: 64, nodedSegments: 128},
        vertexTolerance: 1e-5,
        output: {
          geometry: {
            kind: 'polygons',
            positions: view('float32x2', 64),
            featureOffsets: view('uint32', 9),
            polygonOffsets: view('uint32', 9),
            ringOffsets: view('uint32', 9),
            sourceIds
          },
          sourceIds,
          status: status(),
          boundary: {
            endpoints: view('float32x4', 64),
            operandIds: view('uint32', 64),
            sourceFeatureIds: view('uint32', 64),
            status: status()
          }
        }
      } as never);
      const nodes = overlay.getCommandNodes(graph);
      // Both ray shoots (polygon and coverage levels) and the ring-assembly hole assignment.
      expect(expectExactKernels(nodes, `overlay ${fillRule}`)).toBeGreaterThanOrEqual(3);
      const kernels = getGPUCommandNodeKernelSources(nodes);
      for (const level of [1, 2]) {
        const rayShoot = kernels.find(kernel => kernel.id.endsWith(`-ray-shoot-${level}`));
        expect(rayShoot, `ray shoot ${level}`).toBeDefined();
        const source = rayShoot!.source;
        // Exact crossing order in 96-bit integers, not an f32 interpolation of the crossing x.
        expect(source).toContain('compareCrossings');
        expect(source).not.toMatch(/\/ \(b\.y - a\.y\)/);
        expect(getUndeclaredCalls(source), `ray shoot ${level} undeclared calls`).toEqual([]);
        for (const helper of ['exactProbeCrossing', 'compareCrossings']) {
          for (const [name, count] of getInlinedCallSites(source, helper)) {
            expect(count, `ray shoot ${level} call sites of ${name}`).toBeLessThanOrEqual(1);
          }
        }
        expect(getPrivateMemoryBytes(source)).toBeLessThanOrEqual(CORE_PRIVATE_MEMORY_LIMIT_BYTES);
      }
      // The per-piece membership loop over every ring is gone.
      expect(kernels.some(kernel => /membership|insidePolygons/.test(kernel.source))).toBe(false);
    }
  });

  it('GPULineNoding reports an incomplete split and the pair-list overflow', () => {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    const view = createViewFactory(graph);
    const status = {
      count: view('uint32', 1),
      requiredCount: view('uint32', 1),
      overflow: view('uint32', 1),
      candidateOverflow: view('uint32', 1)
    };
    const noding = new GPULineNoding({
      lines: {kind: 'lines', positions: view('float32x2', 8), lineOffsets: view('uint32', 3)},
      intersectionCapacity: 16,
      output: {
        endpoints: view('float32x4', 32),
        sourceFeatureIds: view('uint32', 32),
        sourceRingIds: view('uint32', 32),
        sourceEdgeIds: view('uint32', 32),
        sourceStartParameters: view('float32', 32),
        sourceEndParameters: view('float32', 32),
        status
      }
    });
    const kernels = getGPUCommandNodeKernelSources(noding.getCommandNodes(graph));
    for (const kernel of kernels) {
      expect(() => new WgslReflect(kernel.source), kernel.id).not.toThrow();
    }
    const ids = kernels.map(kernel => kernel.id);
    expect(ids).toContain('line-noding-split-flag-non-finite');
    expect(ids).toContain('line-noding-split-incomplete');
    const scalars = kernels.find(kernel => kernel.id === 'line-noding-split-scalars');
    // The split's candidate overflow (pair list) reaches the noding's candidateOverflow.
    expect(scalars?.source).toMatch(/candidateOverflow\[candidateOverflowOffset\] = state/);
    expect(scalars?.source).toMatch(/incomplete\[incompleteOffset\] != 0u/);
  });

  it('GPUSegmentRingAssembly orients rings exactly and assigns holes with the exact side probe', () => {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    const view = createViewFactory(graph);
    const assembly = new GPUSegmentRingAssembly({
      endpoints: view('float32x4', 16),
      output: {
        ringOffsets: view('uint32', 5),
        positions: view('float32x2', 32),
        ringShells: view('uint32', 4),
        status: {
          count: view('uint32', 1),
          overflow: view('uint32', 1)
        }
      }
    });
    // Two exact kernels, one call site each: the lowest-vertex orientation and the shell search.
    expect(expectExactKernels(assembly.getCommandNodes(graph), 'ring assembly')).toBe(2);
  });

  it('GPUCellCover containment is exact with one call site', () => {
    for (const family of ['quadbin', 'h3'] as const) {
      for (const containment of ['center', 'full', 'intersects'] as const) {
        if (family === 'h3' && containment !== 'center') {
          continue;
        }
        for (const edgeSlabs of [false, true]) {
          const graph = new GPUCommandGraph(createNullWebGPUDevice());
          const view = createViewFactory(graph);
          const cover = new GPUCellCover({
            family,
            resolution: 4,
            containment,
            edgeSlabs,
            positions: view('float32x2', 64),
            featureOffsets: view('uint32', 2),
            polygonOffsets: view('uint32', 2),
            ringOffsets: view('uint32', 2),
            candidateCapacity: 256,
            output: {
              featureIds: view('uint32', 64),
              cells: view('uint32x2', 64),
              status: {
                count: view('uint32', 1),
                overflow: view('uint32', 1),
                candidateOverflow: view('uint32', 1)
              }
            }
          });
          const label = `cover ${family} ${containment} ${edgeSlabs ? 'slabs' : 'walk'}`;
          expect(expectExactKernels(cover.getCommandNodes(graph), label)).toBeGreaterThan(0);
        }
      }
    }
  });

  it('GPULineLengthPerPolygon, GPUMakeValid and GPUZoneEvents keep one exact call chain', () => {
    {
      const graph = new GPUCommandGraph(createNullWebGPUDevice());
      const view = createViewFactory(graph);
      const lengths = new GPULineLengthPerPolygon({
        positions: view('float32x2', 8),
        lineOffsets: view('uint32', 3),
        polygons: {
          kind: 'polygons',
          positions: view('float32x2', 8),
          featureOffsets: view('uint32', 2),
          polygonOffsets: view('uint32', 2),
          ringOffsets: view('uint32', 2)
        },
        output: {lengths: view('float32', 1), status: {overflow: view('uint32', 1)}}
      } as never);
      expect(expectExactKernels(lengths.getCommandNodes(graph), 'line length')).toBe(1);
    }
    {
      const graph = new GPUCommandGraph(createNullWebGPUDevice());
      const view = createViewFactory(graph);
      const zones = new GPUZoneEvents({
        tracks: {
          kind: 'trajectories',
          positions: view('float32x2', 8),
          timestamps: view('float32', 8),
          trackOffsets: view('uint32', 3)
        },
        zones: {
          kind: 'polygons',
          positions: view('float32x2', 8),
          featureOffsets: view('uint32', 3),
          polygonOffsets: view('uint32', 3),
          ringOffsets: view('uint32', 3)
        },
        candidateCapacity: 16,
        trackEventCapacity: 4,
        output: {
          events: {
            ids: view('uint32', 8),
            status: {count: view('uint32', 1), overflow: view('uint32', 1)},
            eventTimes: view('float32', 8),
            eventZones: view('uint32', 8),
            eventTypes: view('uint32', 8),
            eventRows: view('uint32', 8)
          },
          dwellTimes: view('float32', 4),
          visitCounts: view('uint32', 4),
          trackEventCounts: view('uint32', 2)
        }
      } as never);
      expect(expectExactKernels(zones.getCommandNodes(graph), 'zones')).toBe(1);
    }
    {
      const graph = new GPUCommandGraph(createNullWebGPUDevice());
      const view = createViewFactory(graph);
      const status = () => ({
        count: view('uint32', 1),
        requiredCount: view('uint32', 1),
        overflow: view('uint32', 1),
        candidateOverflow: view('uint32', 1)
      });
      const segments = 64;
      const rings = 8;
      const makeValid = new GPUMakeValid({
        polygons: {
          kind: 'polygons',
          positions: view('float32x2', 8),
          featureOffsets: view('uint32', 2),
          polygonOffsets: view('uint32', 2),
          ringOffsets: view('uint32', 3)
        },
        intersectionCapacity: 64,
        precision: {vertexTolerance: 1e-5, predicates: 'exact-or-uncertain', storage: 'float32'},
        output: {
          nodedSegments: {
            endpoints: view('float32x4', segments),
            sourceFeatureIds: view('uint32', segments),
            sourceRingIds: view('uint32', segments),
            sourceEdgeIds: view('uint32', segments),
            sourceStartParameters: view('float32', segments),
            sourceEndParameters: view('float32', segments),
            status: status()
          },
          directedEdges: {
            endpoints: view('float32x4', 2 * segments),
            sourceSegmentIds: view('uint32', 2 * segments),
            sourceFeatureIds: view('uint32', 2 * segments),
            twinIds: view('uint32', 2 * segments),
            ringIds: view('uint32', 2 * segments),
            flags: view('uint32', 2 * segments),
            status: status()
          },
          polygons: {
            kind: 'polygons',
            positions: view('float32x2', 2 * segments),
            featureOffsets: view('uint32', rings + 1),
            polygonOffsets: view('uint32', rings + 1),
            ringOffsets: view('uint32', rings + 1),
            sourceIds: view('uint32', rings)
          },
          diagnostics: {
            edgeClasses: view('uint32', segments),
            closedRingCount: view('uint32', 1),
            openChainCount: view('uint32', 1),
            cutEdgeCount: view('uint32', 1),
            dangleCount: view('uint32', 1),
            unusedEdgeCount: view('uint32', 1)
          },
          status: status()
        }
      } as never);
      expect(expectExactKernels(makeValid.getCommandNodes(graph), 'make-valid')).toBeGreaterThan(0);
    }
  });
});

// Deterministic generator (mulberry32) for the property checks of the CPU reference rule.
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

/** A random simple star-shaped ring with integer vertices (so vertices and edges are hit exactly). */
function createIntegerStar(random: () => number, count: number): Point[] {
  const ring: Point[] = [];
  for (let index = 0; index < count; index++) {
    const angle = (2 * Math.PI * index) / count;
    const radius = 4 + Math.floor(random() * 8);
    ring.push([
      Math.round(10 + radius * Math.cos(angle)),
      Math.round(10 + radius * Math.sin(angle))
    ]);
  }
  return ring.filter((point, index) => {
    const next = ring[(index + 1) % ring.length];
    return point[0] !== next[0] || point[1] !== next[1];
  });
}

/**
 * Independent oracle: exact on-boundary test, otherwise the winding number from the sum of signed
 * angles (f64 is ample for small integer data whose point is not on the boundary).
 */
function classifyIntegerPoint(
  ring: readonly Point[],
  point: Point
): 'inside' | 'outside' | 'boundary' {
  let angle = 0;
  for (let index = 0; index < ring.length; index++) {
    const a = ring[index];
    const b = ring[(index + 1) % ring.length];
    const inBox =
      point[0] >= Math.min(a[0], b[0]) &&
      point[0] <= Math.max(a[0], b[0]) &&
      point[1] >= Math.min(a[1], b[1]) &&
      point[1] <= Math.max(a[1], b[1]);
    if (inBox && orientSignExact(a, b, point) === 0) {
      return 'boundary';
    }
    const first = Math.atan2(a[1] - point[1], a[0] - point[0]);
    const second = Math.atan2(b[1] - point[1], b[0] - point[0]);
    let step = second - first;
    if (step > Math.PI) {
      step -= 2 * Math.PI;
    } else if (step < -Math.PI) {
      step += 2 * Math.PI;
    }
    angle += step;
  }
  return Math.abs(Math.round(angle / (2 * Math.PI))) % 2 === 1 ? 'inside' : 'outside';
}

describe('exact point-in-ring reference rule (Simulation of Simplicity)', () => {
  it('agrees with an exact containment oracle for points off the boundary', () => {
    const random = createRandom(7);
    let checked = 0;
    for (let trial = 0; trial < 40; trial++) {
      const ring = createIntegerStar(random, 5 + Math.floor(random() * 10));
      for (let x = 0; x <= 22; x++) {
        for (let y = 0; y <= 22; y++) {
          const point: Point = [x, y];
          const truth = classifyIntegerPoint(ring, point);
          const {boundary} = getExactRingWinding(createPointProbe(point), ring);
          expect(boundary, `${trial} ${x} ${y}`).toBe(truth === 'boundary');
          if (truth !== 'boundary') {
            // Off the boundary every consistent perturbation gives the true answer.
            expect(isInsideRingsExact([ring], point), `${trial} ${x} ${y}`).toBe(
              truth === 'inside'
            );
            checked++;
          }
        }
      }
    }
    expect(checked).toBeGreaterThan(1000);
  });

  it('segment crossing parity equals the change of the inside state, ties included', () => {
    const random = createRandom(11);
    for (let trial = 0; trial < 40; trial++) {
      const ring = createIntegerStar(random, 5 + Math.floor(random() * 10));
      for (let step = 0; step < 200; step++) {
        // Integer track samples hit vertices, run along edges and touch them.
        const p: Point = [Math.floor(random() * 23), Math.floor(random() * 23)];
        const q: Point = [Math.floor(random() * 23), Math.floor(random() * 23)];
        let crossings = 0;
        for (let index = 0; index < ring.length; index++) {
          if (doSegmentsCrossExact(p, q, ring[index], ring[(index + 1) % ring.length])) {
            crossings++;
          }
        }
        const change = isInsideRingsExact([ring], p) !== isInsideRingsExact([ring], q);
        expect((crossings & 1) === 1, `${trial} ${p} ${q}`).toBe(change);
      }
    }
  });

  it('side probes of every ring edge split inside from outside', () => {
    const random = createRandom(13);
    for (let trial = 0; trial < 40; trial++) {
      const ring = createIntegerStar(random, 5 + Math.floor(random() * 10));
      for (let index = 0; index < ring.length; index++) {
        const start = ring[index];
        const end = ring[(index + 1) % ring.length];
        const left = getExactRingWinding(createSideProbe(start, end, 1), ring).winding & 1;
        const right = getExactRingWinding(createSideProbe(start, end, -1), ring).winding & 1;
        // A simple ring's edge separates its interior from its exterior.
        expect(left !== right, `${trial} edge ${index}`).toBe(true);
      }
    }
  });

  it('side probes are exact where a rounded probe point is not (review finding D-05)', () => {
    // A slanted edge near x = 100: its rounded midpoint lies off the edge's line.
    const a: Point = [Math.fround(100.1), Math.fround(3.7)];
    const b: Point = [Math.fround(100.9), Math.fround(7.3)];
    const square: Point[] = [
      a,
      b,
      [Math.fround(90), Math.fround(7.3)],
      [Math.fround(90), Math.fround(3.7)]
    ];
    const left = getExactRingWinding(createSideProbe(a, b, 1), square).winding & 1;
    const right = getExactRingWinding(createSideProbe(a, b, -1), square).winding & 1;
    expect([left, right]).toEqual([1, 0]);
  });
});
