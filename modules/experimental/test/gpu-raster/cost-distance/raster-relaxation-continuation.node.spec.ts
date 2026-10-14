// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  getGPURasterRelaxationContinuationStateLength,
  GPU_RASTER_RELAXATION_RESTART_WORD,
  GPUCostDistance,
  type GPUCostDistanceProps
} from '../../../src/gpu-raster/cost-distance';
import {makeGPURasterBand, makeGPURasterMetadata} from '../../../src/gpu-raster';
import {
  createNullWebGPUDevice,
  getGPUCommandNodePipelineKeys,
  getParsedKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {emulateTiledRelaxation, getNeighbor} from './raster-relaxation-emulator';

const WIDTH = 40;
const HEIGHT = 36;
const CELLS = WIDTH * HEIGHT;
const GRID = makeGPURasterMetadata({width: WIDTH, height: HEIGHT});
const STATE_LENGTH = getGPURasterRelaxationContinuationStateLength(GRID);

function buildNodes(overrides: Partial<GPUCostDistanceProps> = {}, continued = true) {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const nodes = new GPUCostDistance({
    grid: GRID,
    friction: makeGPURasterBand(createTransientView(graph, 'friction', 'float32', CELLS), GRID, {
      id: 'friction'
    }),
    parameters: createTransientView(graph, 'settings', 'float32', 8),
    sources: createTransientView(graph, 'sources', 'uint32', 3),
    sweepAfterIteration: 8,
    ...(continued
      ? {
          continuation: {
            state: createTransientView(graph, 'state', 'uint32', STATE_LENGTH),
            values: createTransientView(graph, 'values', 'float32', CELLS)
          }
        }
      : {}),
    output: {
      costs: createTransientView(graph, 'costs', 'float32', CELLS),
      status: {
        converged: createTransientView(graph, 'converged', 'uint32', 1),
        iterationCount: createTransientView(graph, 'iteration-count', 'uint32', 1)
      }
    },
    ...overrides
  }).getCommandNodes(graph);
  return {graph, nodes};
}

const getRoundIds = (ids: string[]) => ids.filter(id => /^cost-distance-relax-\d+$/.test(id));

it('continuation state holds 256 bytes of loop words, then one stamp per 16x16 tile', () => {
  // 40 x 36 cells are 3 x 3 tiles.
  expect(STATE_LENGTH).toBe(64 + 9);
  expect(GPU_RASTER_RELAXATION_RESTART_WORD).toBeLessThan(64);
});

it('without continuation the node list and defaults are unchanged', () => {
  const {nodes} = buildNodes({}, false);
  const ids = nodes.map(node => node.id);
  expect(getRoundIds(ids)).toHaveLength(512);
  expect(ids).toContain('cost-distance-relax-reset');
  expect(ids).not.toContain('cost-distance-restart-gate');
  expect(ids).not.toContain('cost-distance-publish');
  // Nothing but the relaxation rounds, gates and the unresolved pass is conditional.
  for (const id of ['relax-reset', 'tile-reset', 'initialize', 'seed']) {
    expect(nodes.find(node => node.id === `cost-distance-${id}`)?.condition, id).toBeUndefined();
  }
  const sources = getParsedKernelSources(nodes);
  expect(sources.get('cost-distance-relax-0')).toMatch(/tileStamps\[tileStampsOffset/);
  expect(sources.get('cost-distance-relax-0')).not.toMatch(/roundPhase/);
  // The gate still stops at the baked iteration limit.
  expect(sources.get('cost-distance-relax-gate-0')).toMatch(/iterationCount >= 512u/);
  // The same call with the explicit default builds the same list.
  expect(buildNodes({maximumIterations: 512}, false).nodes.map(node => node.id)).toEqual(ids);
});

it('with continuation: 64 rounds per execution and a GPU restart gate on the one-time nodes', () => {
  const {nodes} = buildNodes();
  const ids = nodes.map(node => node.id);
  expect(getRoundIds(ids)).toHaveLength(64);
  const order = [
    'cost-distance-restart-gate',
    'cost-distance-relax-reset',
    'cost-distance-tile-reset',
    'cost-distance-initialize',
    'cost-distance-seed',
    'cost-distance-relax-0'
  ].map(id => ids.indexOf(id));
  expect(order.every(index => index >= 0)).toBe(true);
  expect([...order].sort((a, b) => a - b)).toEqual(order);
  // The decision nodes are ungated (they write the dispatch buffers); the one-time nodes are
  // indirect on slots 0, 1, 2 of one restart-gate buffer.
  expect(nodes[order[0]].condition).toBeUndefined();
  expect(nodes[order[1]].condition).toBeUndefined();
  const gated = order.slice(2, 5).map(index => nodes[index].condition);
  for (const condition of gated) {
    expect(condition).toMatchObject({source: 'gpu', mode: 'indirect'});
  }
  const [tileReset, initialize, seed] = gated as {buffer: unknown; byteOffset: number}[];
  expect(initialize.buffer).toBe(tileReset.buffer);
  expect(seed.buffer).toBe(tileReset.buffer);
  expect([tileReset.byteOffset, initialize.byteOffset, seed.byteOffset]).toEqual([
    tileReset.byteOffset,
    tileReset.byteOffset + 12,
    tileReset.byteOffset + 24
  ]);
  // The publish copy precedes the unresolved pass; the explicit budget wins over the default.
  expect(ids.indexOf('cost-distance-publish')).toBeLessThan(
    ids.indexOf('cost-distance-unresolved-arm')
  );
  expect(getRoundIds(buildNodes({maximumIterations: 24}).nodes.map(node => node.id))).toHaveLength(
    24
  );
});

it('with continuation the gates use a per-execution limit word and the restart word is consumed', () => {
  const sources = getParsedKernelSources(buildNodes().nodes);
  const gate = sources.get('cost-distance-relax-gate-0')!;
  expect(gate).toMatch(/iterationCount >= status\[statusOffset \+ 4u\]/);
  expect(gate).not.toMatch(/iterationCount >= 64u/);
  const arm = sources.get('cost-distance-relax-reset')!;
  expect(arm).toMatch(
    new RegExp(`state\\[stateOffset \\+ ${GPU_RASTER_RELAXATION_RESTART_WORD}u\\] = 0u`)
  );
  expect(arm).toMatch(/state\[stateOffset \+ 4u\] = state\[stateOffset \+ 2u\] \+ 64u/);
  const decision = sources.get('cost-distance-restart-gate')!;
  expect(decision).toMatch(/canResume/);
  // One kernel for every pass: the pass comes from the cumulative count, never from the round.
  const round = sources.get('cost-distance-relax-0')!;
  expect(round).toMatch(/roundPhase/);
  expect(round).toMatch(/status\[statusOffset \+ 64u \+ neighborTile\]/);
  expect(round).not.toMatch(/tileStampsOffset/);
  // The schedule constant is the compile-time sweep switch, not a loop index.
  expect(round).toMatch(/iteration >= 8u/);
});

it('rule 12: continued rounds share one pipeline and the pipeline count ignores the round count', () => {
  const counts = [16, 20, 21].map(
    iterations =>
      getGPUCommandNodePipelineKeys(buildNodes({maximumIterations: iterations}).nodes).size
  );
  expect(new Set(counts).size).toBe(1);
  const roundKeys = [16, 21].map(
    iterations =>
      getGPUCommandNodePipelineKeys(buildNodes({maximumIterations: iterations}).nodes, {
        idPattern: /^cost-distance-relax-\d+$/
      }).size
  );
  expect(roundKeys).toEqual([1, 1]);
  // The unrolled-schedule engine needs one pipeline per pass variant once sweeps are reached.
  const unrolled = getGPUCommandNodePipelineKeys(buildNodes({maximumIterations: 21}, false).nodes, {
    idPattern: /^cost-distance-relax-\d+$/
  }).size;
  expect(unrolled).toBeGreaterThan(1);
});

it('every continued kernel parses as WGSL and stays within the core limits', () => {
  const sources = getParsedKernelSources(buildNodes().nodes);
  expect(sources.size).toBeGreaterThan(8);
  for (const source of sources.values()) {
    expect(source).toMatch(/@compute/);
  }
});

it('rejects unsupported or malformed continuations', () => {
  const make = (overrides: Partial<GPUCostDistanceProps>, stateLength = STATE_LENGTH) => {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    return new GPUCostDistance({
      grid: GRID,
      friction: makeGPURasterBand(createTransientView(graph, 'friction', 'float32', CELLS), GRID, {
        id: 'friction'
      }),
      parameters: createTransientView(graph, 'settings', 'float32', 8),
      sources: createTransientView(graph, 'sources', 'uint32', 3),
      continuation: {
        state: createTransientView(graph, 'state', 'uint32', stateLength),
        values: createTransientView(graph, 'values', 'float32', CELLS)
      },
      output: {costs: createTransientView(graph, 'costs', 'float32', CELLS)},
      ...overrides
    });
  };
  expect(() => make({})).not.toThrow();
  expect(() => make({}, STATE_LENGTH - 1)).toThrow(/continuation.state has the wrong length/);
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  expect(() =>
    make({
      output: {
        costs: createTransientView(graph, 'costs2', 'float32', CELLS),
        backLinks: createTransientView(graph, 'links', 'uint32', CELLS)
      }
    })
  ).toThrow(/does not support output.backLinks/);
});

// ---- CPU model of resume semantics ------------------------------------------------------------

/** Spiral corridors: nested square rings 4 cells wide, each wall ring with one gap on alternating sides. */
function createSpiral(size: number): Float32Array {
  const friction = new Float32Array(size * size).fill(1);
  for (let row = 0; row < size; row++) {
    for (let column = 0; column < size; column++) {
      const ring = Math.min(row, column, size - 1 - row, size - 1 - column);
      if ((ring & 7) < 4) continue;
      const gapOnLeft = (ring >> 3) % 2 === 0;
      const inGap =
        Math.abs(row - (size >> 1)) < 2 && (gapOnLeft ? column < size >> 1 : column >= size >> 1);
      if (!inGap) friction[row * size + column] = NaN;
    }
  }
  return friction;
}

const DIAGONAL = Math.SQRT2;

function createModelProblem(size: number, friction: Float32Array) {
  const passable = (cell: number) => !Number.isNaN(friction[cell]);
  const getEdgeCost = (cell: number, direction: number) => {
    const neighbor = getNeighbor(size, size, cell, direction);
    return (direction % 2 === 0 ? 1 : DIAGONAL) * 0.5 * (friction[cell] + friction[neighbor]);
  };
  return {passable, getEdgeCost};
}

/** Dijkstra over the same float64 edge costs; the fold of a path is monotone, so it equals the fixpoint. */
function computeDijkstra(size: number, friction: Float32Array, source: number): Float64Array {
  const {passable, getEdgeCost} = createModelProblem(size, friction);
  const distance = new Float64Array(size * size).fill(Infinity);
  const done = new Uint8Array(size * size);
  distance[source] = 0;
  for (;;) {
    let best = -1;
    for (let cell = 0; cell < distance.length; cell++) {
      if (
        !done[cell] &&
        distance[cell] < Infinity &&
        (best < 0 || distance[cell] < distance[best])
      ) {
        best = cell;
      }
    }
    if (best < 0) return distance;
    done[best] = 1;
    for (let direction = 0; direction < 8; direction++) {
      const neighbor = getNeighbor(size, size, best, direction);
      if (neighbor < 0 || !passable(neighbor)) continue;
      // The edge cost is symmetric, so the cost into `neighbor` equals the cost into `best`.
      distance[neighbor] = Math.min(
        distance[neighbor],
        distance[best] + getEdgeCost(best, direction)
      );
    }
  }
}

/** Continuation state as the GPU keeps it: stamps, cumulative count, converged flag, restart word. */
type ModelState = {
  stamps: Uint32Array;
  iterationCount: number;
  converged: boolean;
  restart: boolean;
};

/**
 * One execution of the continued relaxation, mirroring the WGSL of `createContinuedRelaxation`:
 * resume only when the previous execution stopped unconverged and no restart was requested,
 * otherwise reset the stamps, initialize the values and seed; then run a budget of rounds.
 */
function runModelExecution(
  size: number,
  friction: Float32Array,
  source: number,
  values: Float64Array,
  state: ModelState,
  budget: number,
  sweepAfterIteration: number
) {
  const {passable, getEdgeCost} = createModelProblem(size, friction);
  const canResume = !state.restart && !state.converged && state.iterationCount > 0;
  state.restart = false;
  if (!canResume) {
    state.stamps.fill(0);
    state.iterationCount = 0;
    values.fill(Infinity);
    values[source] = 0;
    const tilesX = Math.ceil(size / 16);
    state.stamps[Math.floor(source / size / 16) * tilesX + Math.floor((source % size) / 16)] = 1;
  }
  const status = emulateTiledRelaxation({
    width: size,
    height: size,
    values,
    participates: passable,
    getEdgeCost,
    maxIterations: budget,
    sweepAfterIteration,
    activateAllTiles: false,
    state
  });
  state.converged = status.converged;
  return status;
}

function createModelState(size: number): ModelState {
  const tiles = Math.ceil(size / 16) ** 2;
  return {stamps: new Uint32Array(tiles), iterationCount: 0, converged: false, restart: false};
}

function runToConvergence(
  size: number,
  friction: Float32Array,
  budget: number,
  sweepAfterIteration: number
) {
  const values = new Float64Array(size * size);
  const state = createModelState(size);
  let executions = 0;
  while (!state.converged) {
    runModelExecution(size, friction, 0, values, state, budget, sweepAfterIteration);
    executions++;
    expect(executions).toBeLessThan(10000);
  }
  return {values, iterationCount: state.iterationCount, executions};
}

const sameBits = (left: Float64Array, right: Float64Array) =>
  left.length === right.length && left.every((value, index) => Object.is(value, right[index]));

it('model: executions of k rounds equal one long execution, bit for bit, on a spiral', () => {
  const size = 96;
  const friction = createSpiral(size);
  const sweepAfterIteration = 4;
  const oneShot = runToConvergence(size, friction, 4096, sweepAfterIteration);
  expect(oneShot.executions).toBe(1);
  // The spiral needs many rounds, so the small budgets really resume several times.
  expect(oneShot.iterationCount).toBeGreaterThan(10);
  expect(sameBits(oneShot.values, computeDijkstra(size, friction, 0))).toBe(true);
  for (const budget of [1, 2, 3, 4, 5, 8]) {
    const continued = runToConvergence(size, friction, budget, sweepAfterIteration);
    expect(continued.executions, `k=${budget}`).toBeGreaterThan(1);
    expect(sameBits(continued.values, oneShot.values), `k=${budget}`).toBe(true);
    // The schedule follows the cumulative count, so the trajectory and its length are identical.
    expect(continued.iterationCount, `k=${budget}`).toBe(oneShot.iterationCount);
  }
});

it('model: a plain tiled schedule (no sweeps) also resumes to the same fixpoint', () => {
  const size = 48;
  const friction = createSpiral(size);
  const oneShot = runToConvergence(size, friction, 4096, 4096);
  const continued = runToConvergence(size, friction, 5, 4096);
  expect(sameBits(continued.values, oneShot.values)).toBe(true);
  expect(continued.iterationCount).toBe(oneShot.iterationCount);
});

it('model: a converged run and a restart request start from scratch, and inputs may change', () => {
  const size = 64;
  const first = createSpiral(size);
  const values = new Float64Array(size * size);
  const state = createModelState(size);
  while (!state.converged) runModelExecution(size, first, 0, values, state, 8, 6);
  expect(sameBits(values, computeDijkstra(size, first, 0))).toBe(true);

  // After convergence the next execution restarts (the inputs may have changed meanwhile).
  const changed = Float32Array.from(first, (value, cell) => (cell % 7 === 0 ? 3 : value));
  while (true) {
    runModelExecution(size, changed, 0, values, state, 8, 6);
    if (state.converged) break;
  }
  expect(sameBits(values, computeDijkstra(size, changed, 0))).toBe(true);

  // Mid-run change: without the restart word the run would mix two surfaces; with it, it restarts.
  runModelExecution(size, changed, 0, values, state, 8, 6);
  expect(state.converged).toBe(false);
  expect(state.iterationCount).toBe(8);
  state.restart = true;
  const reset = runModelExecution(size, first, 0, values, state, 8, 6);
  expect(reset.iterationCount).toBe(8);
  while (!state.converged) runModelExecution(size, first, 0, values, state, 8, 6);
  expect(sameBits(values, computeDijkstra(size, first, 0))).toBe(true);
});
