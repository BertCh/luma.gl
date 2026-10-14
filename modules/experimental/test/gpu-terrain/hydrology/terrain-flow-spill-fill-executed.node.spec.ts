// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Runs the real spill fill WGSL (everything but the barrier-based tile kernel) on the CPU with
// wgsl_reflect's interpreter (see ../../gpu-spatial-analysis/segment-intersection/wgsl-graph-executor.ts)
// on small grids. The tile kernel's outputs (local fill and labels) come from the CPU model, which
// the GPU spec checks against the kernel; every later kernel (implicit edges, Boruvka rounds with
// pointer jumping, Euler tour, ranking, rooting, path maximum, final values, status) is executed.

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {WgslExec, WgslParser} from 'wgsl_reflect';
import {createTerrainFlowSpillFillNodes} from '../../../src/gpu-terrain/hydrology/terrain-flow-spill-fill';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../../gpu-spatial-analysis/segment-intersection/wgsl-graph-executor';
import {modelSpillFill, modelTileFill} from './terrain-flow-spill-fill-model';
import {
  fillDepressionsPriorityFlood,
  makeSpiralCase,
  type SpillFillCase
} from './terrain-flow-spill-fill-oracle';

function runSpillFillOnCPU(testCase: SpillFillCase, epsilon = 0, maximumIterations = 512) {
  const {width, height, elevation} = testCase;
  const cellCount = width * height;
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'spill-cpu'});
  const elevationView = createTransientView(graph, 'elevation', 'float32', cellCount);
  const filled = createTransientView(graph, 'filled', 'float32', cellCount);
  const settings = createTransientView(graph, 'settings', 'float32', 8);
  const converged = createTransientView(graph, 'converged', 'uint32', 1);
  const rounds = createTransientView(graph, 'rounds', 'uint32', 1);
  const {nodes, status} = createTerrainFlowSpillFillNodes(graph, {
    id: 'flow',
    width,
    height,
    cellSizeMode: 'uniform',
    maxIterations: maximumIterations,
    elevation: elevationView,
    filled,
    settings,
    converged,
    iterationCount: rounds
  });
  const memory = createCPUGraphMemory();
  memory.write(elevationView, elevation);
  memory.write(settings, Float32Array.from([0, 0, 0, 0, epsilon, 0, 0, 0]));
  // The tile kernel is a workgroup-barrier kernel: seed its outputs from the model.
  const tileNode = nodes.find(node => /-spill-tile-fill$/.test(node.id)) as unknown as {
    resources: {buffer: Parameters<typeof memory.write>[0]}[];
  };
  const model = modelSpillFill(elevation, width, height);
  memory.write(tileNode.resources[1].buffer, model.tile.localFill);
  memory.write(tileNode.resources[2].buffer, model.tile.labels);
  // The relaxation rounds are barrier kernels: only the nodes that set them up run here.
  executeCommandNodesOnCPU(
    nodes.filter(node => !/-spill-tile-fill$|-fill-relax-\d+$/.test(node.id)),
    memory
  );
  return {
    filled: memory.readFloat32(filled),
    converged: memory.readUint32(converged)[0],
    rounds: memory.readUint32(rounds)[0],
    status: memory.readUint32(status),
    model,
    workgroupCounts: memory.workgroupCounts
  };
}

function makeCase(name: string, width: number, height: number, values: (cell: number) => number) {
  return {
    name,
    width,
    height,
    elevation: Float32Array.from({length: width * height}, (_, cell) => values(cell))
  } satisfies SpillFillCase;
}

const small = [
  makeCase(
    'noise-6x5',
    6,
    5,
    cell =>
      [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3, 2, 3, 8, 4, 6, 2, 6, 4, 3, 3, 8, 3, 2, 7][
        cell
      ]
  ),
  makeCase('pit-in-ridge-9x9', 9, 9, cell => {
    const column = cell % 9;
    const row = Math.floor(cell / 9);
    const ring = Math.max(Math.abs(column - 4), Math.abs(row - 4));
    return ring === 0 ? 1 : ring === 1 ? 6 : ring === 2 ? 9 : ring === 3 ? 4 : 7;
  }),
  makeCase('strip-1x12', 1, 12, cell => (cell * 7) % 5),
  makeCase('strip-12x1', 12, 1, cell => (cell * 3) % 4),
  makeCase('nodata-ring-10x9', 10, 9, cell => {
    const column = cell % 10;
    const row = Math.floor(cell / 10);
    if (column === 5 && row === 4) return 2;
    if (Math.max(Math.abs(column - 5), Math.abs(row - 4)) === 2) return Number.NaN;
    return 3 + ((column + row) % 3);
  }),
  makeCase('single-cell', 1, 1, () => 4),
  makeCase('all-nodata-4x3', 4, 3, () => Number.NaN),
  makeCase('flat-5x5', 5, 5, () => 2),
  makeSpiralCase(11, 1)
];

it('the executed WGSL equals the CPU model, Priority-Flood, and reports convergence', () => {
  for (const testCase of small) {
    const run = runSpillFillOnCPU(testCase);
    const expected = fillDepressionsPriorityFlood(
      testCase.elevation,
      testCase.width,
      testCase.height
    );
    expect(Array.from(run.filled).map(String), testCase.name).toEqual(
      Array.from(expected).map(String)
    );
    expect(Array.from(run.model.filled).map(String), testCase.name).toEqual(
      Array.from(expected).map(String)
    );
    expect(run.converged, testCase.name).toBe(1);
    expect(run.rounds, testCase.name).toBe(run.model.boruvkaRounds);
    expect(run.status[3], testCase.name).toBe(1);
    expect(run.status[2], testCase.name).toBe(run.model.boruvkaRounds);
  }
});

it('the executed spiral needed pointer jumping and several Boruvka rounds', () => {
  const run = runSpillFillOnCPU(makeSpiralCase(11, 1));
  expect(run.rounds).toBeGreaterThanOrEqual(2);
  expect(run.model.maximumJumpRounds).toBeGreaterThanOrEqual(2);
  // The relaxation path launched nothing for epsilon 0.
  for (const [id, count] of run.workgroupCounts) {
    if (/flow-fill-(init|relax-\d+)$/.test(id)) expect(count, id).toBe(0);
  }
});

it('zero-epsilon spill fill converges even with a one-round relaxation budget', () => {
  const testCase = makeSpiralCase(11, 1);
  const run = runSpillFillOnCPU(testCase, 0, 1);
  expect(run.converged).toBe(1);
  expect(run.rounds).toBeGreaterThan(1);
  expect(Array.from(run.filled)).toEqual(
    Array.from(fillDepressionsPriorityFlood(testCase.elevation, testCase.width, testCase.height))
  );
  expect(run.workgroupCounts.get('flow-fill-init')).toBe(0);
});

// The tile kernel synchronizes a workgroup with barriers, which the interpreter cannot do. It runs
// here with ONE invocation per workgroup that owns every cell (stride 1, barriers removed): the
// fill and label fixpoints are monotone, so the sequential sweep reaches the same unique fixpoint,
// and this checks the kernel's categories, neighbor arithmetic, keys, labels and output writes.
// The barrier placement itself is covered by the GPU spec.
function runTileKernelSequentially(testCase: SpillFillCase, tileSize: number) {
  const {width, height, elevation} = testCase;
  const cellCount = width * height;
  const device = createNullWebGPUDevice();
  Object.defineProperty(device, 'limits', {
    value: {...device.limits, maxComputeWorkgroupStorageSize: tileSize === 16 ? 8192 : 16384}
  });
  const graph = new GPUCommandGraph(device, {id: 'spill-tile'});
  const {nodes} = createTerrainFlowSpillFillNodes(graph, {
    id: 'flow',
    width,
    height,
    cellSizeMode: 'uniform',
    maxIterations: 512,
    elevation: createTransientView(graph, 'elevation', 'float32', cellCount),
    filled: createTransientView(graph, 'filled', 'float32', cellCount),
    settings: createTransientView(graph, 'settings', 'float32', 8)
  });
  const kernel = getGPUCommandNodeKernelSources(
    nodes.filter(node => /-spill-tile-fill$/.test(node.id))
  )[0];
  const tileCells = tileSize * tileSize;
  const source = kernel.source
    // The interpreter has no atomic workgroup variables; one invocation needs none.
    .replace(/array<atomic<u32>,/g, 'array<u32,')
    .replace(/var<workgroup> tileChanged: atomic<u32>;/, 'var<workgroup> tileChanged: u32;')
    .replace(/atomicLoad\(&(\w+)\[(\w+)\]\)/g, '$1[$2]')
    .replace(/atomicStore\(&(\w+)\[(\w+)\], ([^;]+)\);/g, '$1[$2] = $3;')
    .replace(/atomicMin\(&(\w+)\[(\w+)\], (\w+)\);/g, '$1[$2] = min($1[$2], $3);')
    .replace(/atomicLoad\(&tileChanged\)/g, 'tileChanged')
    .replace(/atomicStore\(&tileChanged, (\w+)\);/g, 'tileChanged = $1;')
    .replace(/workgroupBarrier\(\);/g, '')
    .replace(/workgroupUniformLoad\(&tileContinue\)/g, 'tileContinue')
    .replace(/slot \* 256u/g, 'slot * 1u')
    .replace(/const CELLS_PER_THREAD: u32 = \d+u;/, `const CELLS_PER_THREAD: u32 = ${tileCells}u;`)
    .replace(/@workgroup_size\(\s*256\s*\)/, '@workgroup_size(1, 1, 1)')
    .replace(/^\s*_\s*=[^;]*;/gm, '');
  const localFill = new Float32Array(cellCount);
  const labels = new Uint32Array(cellCount);
  const tileCount = Math.ceil(width / tileSize) * Math.ceil(height / tileSize);
  const exec = new WgslExec(new WgslParser().parse(source));
  exec.dispatchWorkgroups(
    'main',
    [tileCount, 1, 1],
    {0: {0: Float32Array.from(elevation), 1: localFill, 2: labels}},
    {constants: kernel.constants}
  );
  return {localFill, labels};
}

it('the tile kernel (sequential stand-in) writes the model local fill and labels', () => {
  const tileCases: [SpillFillCase, number][] = [
    [makeSpiralCase(11, 1), 16],
    [small[1], 16],
    [small[4], 16],
    [makeCase('noise-21x19', 21, 19, cell => (cell * 37 + (cell >> 3) * 11) % 9), 16],
    [makeCase('plateau-20x18', 20, 18, cell => (cell % 7 === 3 ? 2 : 1)), 16]
  ];
  for (const [testCase, tileSize] of tileCases) {
    const model = modelTileFill(testCase.elevation, testCase.width, testCase.height, tileSize);
    const run = runTileKernelSequentially(testCase, tileSize);
    expect(Array.from(run.labels), testCase.name).toEqual(Array.from(model.labels));
    expect(Array.from(run.localFill).map(String), testCase.name).toEqual(
      Array.from(model.localFill).map(String)
    );
  }
});

it('fillEpsilon above 0 selects the relaxation on the GPU; everything else selects the spill graph', () => {
  const testCase = small[0];
  const spillIds = /-spill-(init|boruvka-0-(clear|hook)|tree-(lists|link|root)|final)$/;
  for (const epsilon of [0, -1, Number.NaN, Number.POSITIVE_INFINITY]) {
    const run = runSpillFillOnCPU(testCase, epsilon);
    expect(run.workgroupCounts.get('flow-fill-init') ?? 0, `epsilon ${epsilon}`).toBe(0);
    for (const [id, count] of run.workgroupCounts) {
      if (spillIds.test(id)) expect(count, `${id} epsilon ${epsilon}`).toBeGreaterThan(0);
    }
    expect(run.converged, `epsilon ${epsilon}`).toBe(1);
  }
  for (const epsilon of [0.001, 1e-30, 5]) {
    const run = runSpillFillOnCPU(testCase, epsilon);
    expect(run.workgroupCounts.get('flow-fill-init'), `epsilon ${epsilon}`).toBeGreaterThan(0);
    for (const [id, count] of run.workgroupCounts) {
      if (spillIds.test(id)) expect(count, `${id} epsilon ${epsilon}`).toBe(0);
    }
    // The relaxation state (not executed here) reports: enabled, not yet converged.
    expect(run.converged, `epsilon ${epsilon}`).toBe(0);
  }
});
