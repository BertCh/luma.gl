// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {GPUGeometryCleanup} from '../../../src/gpu-spatial-analysis/geometry-edit/index';
import {createSeededRandom} from '../neighbor-search/neighbor-search-oracle';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys,
  getParsedKernelSources,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';
import {
  createCleanupModelInput,
  runParallelCleanupModel,
  runSequentialCleanupModel,
  type CleanupModelInput
} from './geometry-cleanup-model';

type Ring = number[][];

function expectModelsAgree(input: CleanupModelInput, label: string) {
  const sequential = runSequentialCleanupModel({...input, tolerance: 0});
  const parallel = runParallelCleanupModel(input);
  expect(parallel.keepFlags, label).toEqual(sequential.keepFlags);
  expect(parallel.collapsedCount, label).toBe(sequential.collapsedCount);
  return sequential;
}

const NAN = Number.NaN;
const ADVERSARIAL_RINGS: [string, Ring][] = [
  [
    'all identical',
    [
      [1, 1],
      [1, 1],
      [1, 1],
      [1, 1],
      [1, 1]
    ]
  ],
  [
    'repeats across the closing vertex',
    [
      [0, 0],
      [4, 0],
      [4, 4],
      [4, 4],
      [0, 0]
    ]
  ],
  [
    'run of repeats before closing vertex',
    [
      [0, 0],
      [4, 0],
      [4, 4],
      [4, 4],
      [4, 4],
      [4, 4]
    ]
  ],
  [
    'collapses polygon below three vertices',
    [
      [0, 0],
      [0, 0],
      [0, 0],
      [3, 3]
    ]
  ],
  [
    'collapses after the closing replacement',
    [
      [0, 0],
      [3, 3],
      [3, 3]
    ]
  ],
  [
    'repeats around a distinct vertex',
    [
      [0, 0],
      [0, 0],
      [3, 3],
      [3, 3],
      [0, 0]
    ]
  ],
  [
    'run that reaches the first vertex',
    [
      [2, 2],
      [2, 2],
      [2, 2],
      [2, 2]
    ]
  ],
  [
    'negative zero equals positive zero',
    [
      [0, 0],
      [-0, -0],
      [0, -0],
      [5, 5],
      [-0, 0]
    ]
  ],
  [
    'NaN vertices are never repeats',
    [
      [0, 0],
      [NAN, 1],
      [NAN, 1],
      [2, NAN],
      [2, NAN],
      [3, 3]
    ]
  ],
  [
    'infinite vertices are never repeats',
    [
      [0, 0],
      [Infinity, 1],
      [Infinity, 1],
      [-Infinity, 1],
      [-Infinity, 1],
      [3, 3]
    ]
  ],
  ['single vertex', [[7, 7]]],
  [
    'two vertices equal',
    [
      [7, 7],
      [7, 7]
    ]
  ],
  [
    'two vertices different',
    [
      [7, 7],
      [8, 8]
    ]
  ],
  [
    'three vertices, middle repeats first',
    [
      [1, 1],
      [1, 1],
      [2, 2]
    ]
  ],
  [
    'three vertices, middle repeats last',
    [
      [1, 1],
      [2, 2],
      [2, 2]
    ]
  ],
  ['empty', []],
  [
    'repeats only after snapping',
    [
      [0.01, 0.01],
      [0.02, -0.02],
      [0.4, 0.4],
      [0.45, 0.35],
      [0.01, 0.01]
    ]
  ]
];

it('the tolerance-0 parallel passes equal the sequential pass on adversarial rings', () => {
  for (const isPolygon of [false, true]) {
    for (const gridSize of [0, 0.5, 1]) {
      for (const removeRepeatedPoints of [true, false]) {
        for (const [name, ring] of ADVERSARIAL_RINGS) {
          // Each ring alone, then all rings together with vertices outside every ring.
          expectModelsAgree(
            createCleanupModelInput([ring], {isPolygon, gridSize, removeRepeatedPoints}),
            `${name} polygon=${isPolygon} grid=${gridSize} remove=${removeRepeatedPoints}`
          );
        }
        expectModelsAgree(
          createCleanupModelInput(
            ADVERSARIAL_RINGS.map(([, ring]) => ring),
            {isPolygon, gridSize, removeRepeatedPoints},
            [
              [9, 9],
              [9, 9]
            ]
          ),
          `all rings polygon=${isPolygon} grid=${gridSize}`
        );
      }
    }
  }
});

it('the sequential model keeps the documented rules (guards the model itself)', () => {
  const line = (ring: Ring) => runSequentialCleanupModel(createCleanupModelInput([ring])).keepFlags;
  expect(
    line([
      [1, 1],
      [1, 1],
      [1, 1],
      [1, 1],
      [1, 1]
    ])
  ).toEqual([1, 0, 0, 0, 1]);
  // The kept interior vertex that the closing vertex repeats is replaced by the closing vertex.
  expect(
    line([
      [0, 0],
      [4, 0],
      [4, 4],
      [4, 4]
    ])
  ).toEqual([1, 1, 0, 1]);
  // A repeat of the first vertex is not replaced (lastKept is the first vertex).
  expect(
    line([
      [2, 2],
      [2, 2],
      [2, 2]
    ])
  ).toEqual([1, 0, 1]);
  expect(line([[7, 7]])).toEqual([1]);
  const nan = line([
    [NAN, 1],
    [NAN, 1],
    [NAN, 1]
  ]);
  expect(nan).toEqual([1, 1, 1]);
  const polygon = runSequentialCleanupModel(
    createCleanupModelInput(
      [
        [
          [0, 0],
          [0, 0],
          [0, 0],
          [3, 3]
        ]
      ],
      {isPolygon: true}
    )
  );
  expect(polygon).toEqual({keepFlags: [0, 0, 0, 0], collapsedCount: 1});
});

it('the tolerance-0 parallel passes equal the sequential pass on random rings', () => {
  const random = createSeededRandom(2026);
  const values = [0, -0, 1, 1.5, 2, NAN, Infinity, 0.25];
  for (let trial = 0; trial < 400; trial++) {
    const ringCount = 1 + Math.floor(random() * 5);
    const rings: Ring[] = [];
    for (let ring = 0; ring < ringCount; ring++) {
      const length = Math.floor(random() * (trial % 5 === 0 ? 150 : 10));
      // A tiny alphabet makes repeats, runs and collapses common.
      const alphabet = trial % 3 === 0 ? values : values.slice(0, 1 + Math.floor(random() * 4));
      rings.push(
        Array.from({length}, () => [
          alphabet[Math.floor(random() * alphabet.length)],
          alphabet[Math.floor(random() * alphabet.length)]
        ])
      );
    }
    expectModelsAgree(
      createCleanupModelInput(rings, {
        isPolygon: trial % 2 === 0,
        gridSize: [0, 0.5, 1][trial % 3],
        removeRepeatedPoints: trial % 7 !== 0
      }),
      `trial ${trial}`
    );
  }
});

function createCleanupFixture(options: {forceSequential?: boolean} = {}) {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'cleanup-repeat'});
  let serial = 0;
  const view = <Format extends 'float32x2' | 'uint32' | 'float32'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `cleanup-${serial++}`, format, length);
  const cleanup = new GPUGeometryCleanup({
    geometryType: 'polygons',
    positions: view('float32x2', 40),
    ringOffsets: view('uint32', 5),
    polygonOffsets: view('uint32', 3),
    featureOffsets: view('uint32', 3),
    parameters: view('float32', 4),
    forceSequentialRepeatRemoval: options.forceSequential,
    output: {
      positions: view('float32x2', 40),
      ringOffsets: view('uint32', 5),
      status: {count: view('uint32', 1), overflow: view('uint32', 1)}
    }
  });
  return {graph, nodes: cleanup.getCommandNodes(graph)};
}

it('GPUGeometryCleanup compiles both flag paths behind GPU indirect gates', () => {
  const {nodes} = createCleanupFixture();
  const ids = nodes.map(node => node.id);
  expect(ids).toEqual(
    expect.arrayContaining([
      'geometry-cleanup-repeat-gate',
      'geometry-cleanup-flags',
      'geometry-cleanup-repeat-flags',
      'geometry-cleanup-repeat-resolve',
      'geometry-cleanup-repeat-apply'
    ])
  );
  // The gate arms the passes before any of them, and the scan follows them all.
  const index = (id: string) => ids.indexOf(id);
  expect(index('geometry-cleanup-repeat-gate')).toBeLessThan(index('geometry-cleanup-flags'));
  expect(index('geometry-cleanup-repeat-apply')).toBeLessThan(
    index('geometry-cleanup-scan-0') >= 0 ? index('geometry-cleanup-scan-0') : ids.length
  );
  const gatedIds = [
    'geometry-cleanup-flags',
    'geometry-cleanup-repeat-flags',
    'geometry-cleanup-repeat-resolve',
    'geometry-cleanup-repeat-apply'
  ];
  const gated = nodes.filter(node => gatedIds.includes(node.id));
  expect(gated).toHaveLength(4);
  for (const node of gated) {
    const condition = (node as unknown as {condition?: {source: string; mode: string}}).condition;
    expect(condition, node.id).toMatchObject({source: 'gpu', mode: 'indirect'});
  }
  const gateNode = nodes.find(node => node.id === 'geometry-cleanup-repeat-gate');
  expect((gateNode as unknown as {condition?: unknown}).condition).toBeUndefined();
});

it('GPUGeometryCleanup forced sequential removal compiles no parallel passes', () => {
  const {nodes} = createCleanupFixture({forceSequential: true});
  const ids = nodes.map(node => node.id);
  expect(ids).toContain('geometry-cleanup-flags');
  expect(ids.some(id => /repeat/.test(id))).toBe(false);
  expect(
    (nodes.find(node => node.id === 'geometry-cleanup-flags') as unknown as {condition?: unknown})
      .condition
  ).toBeUndefined();
});

it('the cleanup repeat kernels parse and stay within the core binding limit', () => {
  const {nodes} = createCleanupFixture();
  const sources = getParsedKernelSources(nodes);
  for (const id of [
    'geometry-cleanup-repeat-gate',
    'geometry-cleanup-repeat-flags',
    'geometry-cleanup-repeat-resolve',
    'geometry-cleanup-repeat-apply'
  ]) {
    const source = sources.get(id);
    expect(source, id).toBeDefined();
    expect(getStorageBindingCount(source!), id).toBeLessThanOrEqual(8);
  }
  // The flag pass walks a constant-size chunk and issues atomics per ring segment, not per vertex.
  expect(sources.get('geometry-cleanup-repeat-flags')).toMatch(/const CHUNK: u32 = 32u;/);
  expect(sources.get('geometry-cleanup-repeat-flags')?.match(/atomicAdd|atomicMax/g)).toHaveLength(
    2
  );
  expect(sources.get('geometry-cleanup-repeat-resolve')).not.toMatch(/for \(/);
  expect(sources.get('geometry-cleanup-repeat-apply')).not.toMatch(/for \(var row/);
  // The tolerance choice is made on the GPU, from the parameters word.
  expect(sources.get('geometry-cleanup-repeat-gate')).toMatch(
    /parameters\[parametersOffset \+ 1u\]/
  );
});

it('GPUGeometryCleanup keeps one pipeline set regardless of tolerance (no recompile per frame)', () => {
  // Tolerance is a per-frame word: building twice yields identical pipeline keys.
  const first = getGPUCommandNodePipelineKeys(createCleanupFixture().nodes);
  const second = getGPUCommandNodePipelineKeys(createCleanupFixture().nodes);
  expect([...second]).toEqual([...first]);
  const forced = getGPUCommandNodeKernelSources(
    createCleanupFixture({forceSequential: true}).nodes
  ).length;
  // Gate writer, flag, resolve and apply passes (plus the ring-state clear).
  expect(
    getGPUCommandNodeKernelSources(createCleanupFixture().nodes).length
  ).toBeGreaterThanOrEqual(forced + 4);
});

it('guards both cleanup operands before zero-tolerance subtraction can be folded', () => {
  const sources = getParsedKernelSources(createCleanupFixture().nodes);
  for (const id of [
    'geometry-cleanup-flags',
    'geometry-cleanup-repeat-flags',
    'geometry-cleanup-repeat-resolve'
  ]) {
    const source = sources.get(id)!;
    expect(source).toContain('!isFiniteBits(first.x) || !isFiniteBits(first.y)');
    expect(source).toContain('!isFiniteBits(second.x) || !isFiniteBits(second.y)');
    expect(source.indexOf('!isFiniteBits(first.x)')).toBeLessThan(
      source.indexOf('let delta = first - second')
    );
  }
});
