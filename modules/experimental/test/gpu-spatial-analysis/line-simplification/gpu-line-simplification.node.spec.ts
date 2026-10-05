// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  getGPULineSimplificationParameterValues,
  GPULineSimplification,
  type GPULineSimplificationProps
} from '../../../src/gpu-spatial-analysis/line-simplification';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  computeParallelImportance,
  divideFloor,
  getFloatBits,
  getKeptRowsFromImportance,
  simplifyDouglasPeucker,
  sqrtFloor,
  type LineSimplificationScene
} from './line-simplification-oracle';
import {
  createCornerCaseScene,
  createRandom,
  createRandomTracksScene,
  createSpiralScene
} from './line-simplification-scenes';

let serial = 0;

function createProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPULineSimplificationProps> = {}
): GPULineSimplificationProps {
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `view-${serial++}`, format, length);
  return {
    positions: view('float32x2', 10),
    trackOffsets: view('uint32', 3),
    importance: view('float32', 10),
    parameters: view('float32', 4),
    selection: {
      output: {
        ids: view('uint32', 10),
        count: view('uint32', 1),
        overflow: view('uint32', 1)
      },
      lineCounts: view('uint32', 2),
      lineStarts: view('uint32', 2)
    },
    ...overrides
  };
}

function expectThrows(
  overrides: (graph: GPUCommandGraph) => Partial<GPULineSimplificationProps>,
  message: RegExp
): void {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  expect(() => new GPULineSimplification(createProps(graph, overrides(graph)))).toThrow(message);
  device.destroy();
}

/** Every distinct finite importance value, its f32 neighbors, and a few round numbers. */
function getProbeTolerances(importanceBits: Uint32Array): number[] {
  const values = new Set<number>([0, 0.1, 0.5, 1, 2.5, 10, 1e6]);
  const view = new Float32Array(importanceBits.buffer.slice(0));
  for (const value of view) {
    if (Number.isFinite(value) && value >= 0) {
      values.add(value);
      values.add(Math.fround(value * (1 - 2 ** -23)));
      values.add(Math.fround(value * (1 + 2 ** -23)));
    }
  }
  return [...values].filter(value => value >= 0);
}

function expectImportanceMatchesDouglasPeucker(scene: LineSimplificationScene): void {
  const {importanceBits, converged} = computeParallelImportance(scene, 1024);
  expect(converged).toBe(true);
  for (const tolerance of getProbeTolerances(importanceBits)) {
    expect(
      getKeptRowsFromImportance(scene, importanceBits, tolerance),
      `tolerance ${tolerance}`
    ).toEqual(simplifyDouglasPeucker(scene, tolerance));
  }
}

it('GPULineSimplification parameter helper packs and validates the tolerance', () => {
  expect(Array.from(getGPULineSimplificationParameterValues({tolerance: 2.5}))).toEqual([
    2.5, 0, 0, 0
  ]);
  expect(() => getGPULineSimplificationParameterValues({tolerance: -1})).toThrow(/non-negative/);
  expect(() => getGPULineSimplificationParameterValues({tolerance: NaN})).toThrow(/finite/);
  expect(() =>
    getGPULineSimplificationParameterValues({tolerance: 1}, new Float32Array(2))
  ).toThrow(/4 elements/);
});

it('GPULineSimplification validates props', () => {
  expectThrows(() => ({maximumRounds: 0}), /maximumRounds/);
  expectThrows(() => ({maximumRounds: 1025}), /maximumRounds/);
  expectThrows(() => ({metric: 'line' as never}), /metric/);
  expectThrows(() => ({metric: 'time-ratio'}), /requires timestamps/);
  expectThrows(
    graph => ({
      metric: 'time-ratio',
      timestamps: createTransientView(graph, 't', 'float32', 4)
    }),
    /timestamps length/
  );
  expectThrows(
    graph => ({importance: createTransientView(graph, 'i', 'float32', 4)}),
    /importance length/
  );
  expectThrows(
    graph => ({trackOffsets: createTransientView(graph, 'o', 'uint32', 1)}),
    /at least two rows/
  );
  expectThrows(() => ({parameters: undefined}), /parameters are required/);
  expectThrows(
    graph => ({parameters: createTransientView(graph, 'p', 'float32', 2)}),
    /4 float32 values/
  );
  expectThrows(() => ({computeImportance: false, selection: undefined}), /computeImportance or/);
  expectThrows(
    graph => ({
      computeImportance: false,
      status: {converged: createTransientView(graph, 'c', 'uint32', 1)}
    }),
    /status requires/
  );
  expectThrows(
    graph => ({
      selection: {
        output: {
          ids: createTransientView(graph, 'ids', 'uint32', 4),
          count: createTransientView(graph, 'count', 'uint32', 1),
          overflow: createTransientView(graph, 'overflow', 'uint32', 1)
        },
        lineCounts: createTransientView(graph, 'lc', 'uint32', 5)
      }
    }),
    /lineCounts length/
  );
});

it('GPULineSimplification emits deterministic node IDs and five nodes per round', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const contributor = new GPULineSimplification(
    createProps(graph, {id: 'simplify', maximumRounds: 3})
  );
  const nodes = contributor.getCommandNodes(graph);
  const ids = nodes.map(node => node.id);
  expect(ids.slice(0, 2)).toEqual(['simplify-rounds-reset', 'simplify-init']);
  expect(ids.filter(id => id.startsWith('simplify-round-'))).toHaveLength(15);
  expect(ids).toContain('simplify-round-2-gate');
  expect(ids).toContain('simplify-unresolved');
  expect(ids).toContain('simplify-keep-mask');
  expect(ids).toContain('simplify-publish');
  expect(ids.at(-1)).toBe('simplify-line-ranges');

  const selectionGraph = new GPUCommandGraph(device);
  const selectionOnly = new GPULineSimplification(
    createProps(selectionGraph, {id: 'select', computeImportance: false})
  );
  const selectionIds = selectionOnly.getCommandNodes(selectionGraph).map(node => node.id);
  expect(selectionIds.some(id => id.includes('round'))).toBe(false);
  expect(selectionIds[0]).toBe('select-keep-mask');
  device.destroy();
});

it('divideFloor and sqrtFloor return the largest admissible f32', () => {
  const random = createRandom(3);
  for (let sample = 0; sample < 2000; sample++) {
    const numerator = Math.fround(random() * 1000);
    const denominator = Math.fround(random() * 50 + 1e-3);
    const quotient = divideFloor(numerator, denominator);
    expect(Math.fround(quotient * denominator)).toBeLessThanOrEqual(numerator);
    const next = new Float32Array(new Uint32Array([getFloatBits(quotient) + 1]).buffer)[0];
    expect(Math.fround(next * denominator)).toBeGreaterThan(numerator);
    const root = sqrtFloor(numerator);
    expect(Math.fround(root * root)).toBeLessThanOrEqual(numerator);
  }
  expect(sqrtFloor(16)).toBe(4);
  expect(divideFloor(6, 3)).toBe(2);
});

it('parallel importance equals recursive Douglas-Peucker on corner cases', () => {
  const scene = createCornerCaseScene();
  expectImportanceMatchesDouglasPeucker(scene);
  const {importanceBits} = computeParallelImportance(scene, 64);
  const importance = new Float32Array(importanceBits.buffer);
  // Collinear interior rows have importance 0 and are dropped at tolerance 0.
  expect(Array.from(importance.slice(0, 5))).toEqual([Infinity, 0, 0, 0, Infinity]);
  // Doubling back along a line: every row lies on the chord 0..10, so every row is 0.
  expect(Array.from(importance.slice(5, 11))).toEqual([Infinity, 0, 0, 0, 0, Infinity]);
  // Identical vertices.
  expect(Array.from(importance.slice(11, 15))).toEqual([Infinity, 0, 0, Infinity]);
  // Single and two-vertex lines are endpoints only.
  expect(Array.from(importance.slice(15, 18))).toEqual([Infinity, Infinity, Infinity]);
  // Tie: rows 1 and 3 of the last line are both 1 from the chord; the smaller row splits first,
  // then rows 2 and 3 tie against chord 1..4 and row 3 is clamped to row 2's importance.
  const tieStart = scene.trackOffsets[8];
  const tie = Array.from(importance.slice(tieStart, tieStart + 5));
  expect([tie[0], tie[1], tie[4]]).toEqual([Infinity, 1, Infinity]);
  expect(tie[2]).toBeCloseTo(2 / Math.sqrt(10), 6);
  expect(tie[3]).toBe(tie[2]);
});

it('parallel importance equals recursive Douglas-Peucker on many random tracks', () => {
  for (const seed of [1, 2, 3, 4]) {
    expectImportanceMatchesDouglasPeucker(createRandomTracksScene(seed, 40, 60));
    expectImportanceMatchesDouglasPeucker(createRandomTracksScene(seed + 10, 30, 50, 'time-ratio'));
  }
});

it('a round cap leaves a deep spiral unconverged with a superset of Douglas-Peucker', () => {
  const scene = createSpiralScene(300);
  const full = computeParallelImportance(scene, 1024);
  expect(full.converged).toBe(true);
  expect(full.roundCount).toBeGreaterThan(12);
  const capped = computeParallelImportance(scene, 6);
  expect(capped.converged).toBe(false);
  expect(capped.roundCount).toBe(6);
  for (const tolerance of [0.5, 2, 5, 20]) {
    const kept = new Set(getKeptRowsFromImportance(scene, capped.importanceBits, tolerance));
    for (const row of simplifyDouglasPeucker(scene, tolerance)) {
      expect(kept.has(row)).toBe(true);
    }
  }
  // A cap equal to the depth still converges.
  expect(computeParallelImportance(scene, full.roundCount).converged).toBe(true);
});
