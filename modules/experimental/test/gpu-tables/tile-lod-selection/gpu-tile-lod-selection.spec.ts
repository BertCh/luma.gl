// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {
  DispatchCommandBuffer,
  DrawCommandBuffer,
  GPUCommandGraph,
  GPUVirtualGeometrySelection
} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  GPU_TILE_LOD_VIEW_LENGTH,
  GPU_TILE_LOD_VIEW_OFFSETS,
  GPUTileLODSelection,
  makeGPUTileLODQuadtree,
  type GPUTileLODSelectionProps
} from '../../../src/gpu-tables/tile-lod-selection';
import {
  createInputBuffer,
  createOutputBuffer,
  readCompactIds,
  readFloat32,
  readUint32,
  sortNumbers,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {
  FIXTURE_A,
  INVALID,
  makeBoxFrustum,
  makeFixtureView,
  selectTileLODOnCPU,
  UNLIMITED,
  type TileLODFixture
} from './tile-lod-oracle';

type TileOptions = {
  fixture?: TileLODFixture;
  capacity?: number;
  budget?: boolean;
  refinement?: 'replace' | 'add';
  identityIds?: boolean;
  requestCapacity?: number;
  indirectDispatchWorkgroupSize?: number;
};

type TileRun = {
  ids: number[];
  count: number;
  overflow: number;
  requiredCount: number;
  statistics: number[];
  instanceCount: number;
  requestIds: number[];
  requestCount: number;
  requestOverflow: number;
  requestTotal: number;
  requestPriorities: number[];
  drawnAncestors: number[];
  dispatch: number[];
};

function createTileFixture(device: Device, options: TileOptions = {}) {
  const fixture = options.fixture ?? FIXTURE_A;
  const nodeCount = fixture.levelOffsets[fixture.levelOffsets.length - 1];
  const capacity = options.capacity ?? nodeCount;
  const requestCapacity = options.requestCapacity ?? nodeCount;
  const graph = new GPUCommandGraph(device, {id: 'tile-lod-test'});
  const owned: Buffer[] = [];
  const input = (values: Float32Array | Uint32Array) => {
    const buffer = createInputBuffer(device, values);
    owned.push(buffer);
    return buffer;
  };
  const output = (length: number) => {
    const buffer = createOutputBuffer(device, length);
    owned.push(buffer);
    return buffer;
  };
  const parameters = {
    view: new GPUParameterBuffer(device, {
      id: 'view',
      format: 'float32',
      length: GPU_TILE_LOD_VIEW_LENGTH
    }),
    budget: new GPUParameterBuffer(device, {
      id: 'budget',
      format: 'uint32',
      length: 2,
      values: Uint32Array.from([UNLIMITED, UNLIMITED])
    }),
    residency: new GPUParameterBuffer(device, {
      id: 'residency',
      format: 'uint32',
      length: nodeCount,
      values: new Uint32Array(nodeCount).fill(1)
    }),
    enabledNodes: new GPUParameterBuffer(device, {
      id: 'enabled',
      format: 'uint32',
      length: nodeCount,
      values: new Uint32Array(nodeCount).fill(1)
    })
  };
  const buffers = {
    ids: output(capacity),
    count: output(1),
    overflow: output(1),
    requiredCount: output(1),
    statistics: output(8),
    requestIds: output(requestCapacity),
    requestCount: output(1),
    requestOverflow: output(1),
    requestTotal: output(1),
    requestPriorities: output(requestCapacity),
    drawnAncestors: output(nodeCount)
  };
  const drawCommands = new DrawCommandBuffer(device, {
    id: 'tile-draw',
    type: 'draw',
    commands: [{vertexCount: 6, instanceCount: 99}]
  });
  const dispatchCommands = options.indirectDispatchWorkgroupSize
    ? new DispatchCommandBuffer(device, {id: 'tile-dispatch', capacity: 1})
    : undefined;
  const hierarchy: GPUTileLODSelectionProps['hierarchy'] = {
    sphereBounds: importGraphBuffer(
      graph,
      'bounds',
      input(fixture.sphereBounds),
      'float32x4',
      nodeCount
    ),
    geometricErrors: importGraphBuffer(
      graph,
      'errors',
      input(fixture.geometricErrors),
      'float32',
      nodeCount
    ),
    children: importGraphBuffer(graph, 'children', input(fixture.children), 'uint32x2', nodeCount),
    levelOffsets: fixture.levelOffsets,
    tileIds:
      fixture.tileIds && !options.identityIds
        ? importGraphBuffer(graph, 'tile-ids', input(fixture.tileIds), 'uint32', nodeCount)
        : undefined,
    parents: fixture.parents
      ? importGraphBuffer(graph, 'parents', input(fixture.parents), 'uint32', nodeCount)
      : undefined,
    nodeCosts: fixture.nodeCosts
      ? importGraphBuffer(graph, 'costs', input(fixture.nodeCosts), 'uint32', nodeCount)
      : undefined
  };
  graph.add(
    new GPUTileLODSelection({
      hierarchy,
      view: parameters.view.importToGraph(graph),
      refinement: options.refinement,
      residency: parameters.residency.importToGraph(graph),
      enabledNodes: parameters.enabledNodes.importToGraph(graph),
      budget: options.budget ? parameters.budget.importToGraph(graph) : undefined,
      output: {
        ids: importGraphBuffer(graph, 'ids', buffers.ids, 'uint32', capacity),
        count: importGraphBuffer(graph, 'count', buffers.count, 'uint32', 1),
        overflow: importGraphBuffer(graph, 'overflow', buffers.overflow, 'uint32', 1),
        requiredCount: importGraphBuffer(graph, 'total', buffers.requiredCount, 'uint32', 1)
      },
      requests: {
        ids: importGraphBuffer(graph, 'request-ids', buffers.requestIds, 'uint32', requestCapacity),
        count: importGraphBuffer(graph, 'request-count', buffers.requestCount, 'uint32', 1),
        overflow: importGraphBuffer(
          graph,
          'request-overflow',
          buffers.requestOverflow,
          'uint32',
          1
        ),
        requiredCount: importGraphBuffer(graph, 'request-total', buffers.requestTotal, 'uint32', 1),
        priorities: importGraphBuffer(
          graph,
          'request-priorities',
          buffers.requestPriorities,
          'float32',
          requestCapacity
        )
      },
      drawnAncestors: hierarchy.parents
        ? importGraphBuffer(graph, 'ancestors', buffers.drawnAncestors, 'uint32', nodeCount)
        : undefined,
      statistics: importGraphBuffer(graph, 'statistics', buffers.statistics, 'uint32', 8),
      indirectDraw: {commands: drawCommands.importToGraph(graph)},
      indirectDispatch: dispatchCommands
        ? {
            command: importGraphBuffer(graph, 'dispatch', dispatchCommands.buffer, 'uint32', 3),
            workgroupSize: options.indirectDispatchWorkgroupSize ?? 1
          }
        : undefined
    })
  );
  const compiled = graph.compile();

  const run = async (
    view: Float32Array,
    state: {budget?: [number, number]; residency?: number[]; enabledNodes?: number[]} = {}
  ): Promise<TileRun> => {
    parameters.view.write(view);
    parameters.budget.write(Uint32Array.from(state.budget ?? [UNLIMITED, UNLIMITED]));
    parameters.residency.write(Uint32Array.from(state.residency ?? new Array(nodeCount).fill(1)));
    parameters.enabledNodes.write(
      Uint32Array.from(state.enabledNodes ?? new Array(nodeCount).fill(1))
    );
    submitGraph(device, compiled, undefined);
    const [requestCount] = await readUint32(buffers.requestCount, 1);
    const instanceBytes = await drawCommands.buffer.readAsync(
      drawCommands.getInstanceCountByteOffset(0),
      4
    );
    return {
      ids: await readCompactIds(buffers.ids, buffers.count),
      count: (await readUint32(buffers.count, 1))[0],
      overflow: (await readUint32(buffers.overflow, 1))[0],
      requiredCount: (await readUint32(buffers.requiredCount, 1))[0],
      statistics: await readUint32(buffers.statistics, 8),
      instanceCount: new Uint32Array(instanceBytes.buffer, instanceBytes.byteOffset, 1)[0],
      requestIds: await readUint32(buffers.requestIds, requestCount),
      requestCount,
      requestOverflow: (await readUint32(buffers.requestOverflow, 1))[0],
      requestTotal: (await readUint32(buffers.requestTotal, 1))[0],
      requestPriorities: await readFloat32(buffers.requestPriorities, requestCount),
      drawnAncestors: await readUint32(buffers.drawnAncestors, nodeCount),
      dispatch: dispatchCommands ? await readUint32(dispatchCommands.buffer, 3) : []
    };
  };

  const expectOracle = async (
    view: Float32Array,
    state: {budget?: [number, number]; residency?: number[]; enabledNodes?: number[]} = {}
  ) => {
    const result = await run(view, state);
    const oracle = selectTileLODOnCPU(
      {
        ...fixture,
        tileIds: options.identityIds ? undefined : fixture.tileIds,
        residency: state.residency ? Uint32Array.from(state.residency) : undefined,
        enabledNodes: state.enabledNodes ? Uint32Array.from(state.enabledNodes) : undefined
      },
      view,
      {
        refinement: options.refinement,
        budget: options.budget ? (state.budget ?? [UNLIMITED, UNLIMITED]) : undefined
      }
    );
    expect(result.requiredCount).toBe(oracle.drawnIds.length);
    expect(result.ids).toEqual(oracle.drawnIds.slice(0, capacity));
    expect(result.statistics).toEqual(oracle.statistics);
    expect(result.requestTotal).toBe(oracle.requestedIds.length);
    expect(result.requestIds).toEqual(oracle.requestedIds.slice(0, requestCapacity));
    return {result, oracle};
  };

  const destroy = () => {
    compiled.destroy();
    for (const parameter of Object.values(parameters)) parameter.destroy();
    drawCommands.destroy();
    dispatchCommands?.destroy();
    for (const buffer of owned) buffer.destroy();
  };
  return {run, expectOracle, destroy};
}

const at = (threshold: number, overrides = {}) =>
  makeFixtureView({maximumScreenSpaceError: threshold, ...overrides});

it('GPUTileLODSelection sweeps thresholds per frame with statistics, indirect draw, and overflow', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const tiles = createTileFixture(device);
  for (const [threshold, expected] of [
    [100, [1000]],
    [50, [1100, 1101]],
    [10, [1101, 1200, 1201]],
    [5, [1200, 1201, 1202, 1203]]
  ] as const) {
    const {result} = await tiles.expectOracle(at(threshold));
    expect(result.ids).toEqual(expected);
    expect(result.instanceCount).toBe(expected.length);
    expect(result.overflow).toBe(0);
  }
  expect((await tiles.run(at(5))).statistics).toEqual([4, 160, 4, 160, 0, 0, 7, 0]);
  tiles.destroy();

  const bounded = createTileFixture(device, {capacity: 2});
  let result = await bounded.run(at(5));
  expect(result).toMatchObject({
    ids: [1200, 1201],
    count: 2,
    overflow: 1,
    requiredCount: 4,
    instanceCount: 2
  });
  result = await bounded.run(at(100));
  expect(result).toMatchObject({ids: [1000], count: 1, overflow: 0});
  bounded.destroy();

  const identity = createTileFixture(device, {identityIds: true});
  expect((await identity.run(at(10))).ids).toEqual([2, 3, 4]);
  identity.destroy();
});

it('GPUTileLODSelection culls by frustum and matches GPUVirtualGeometrySelection', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const culled = at(5, {frustumPlanes: Array.from(makeBoxFrustum(-1000, -1, 1000))});
  const tiles = createTileFixture(device);
  expect((await tiles.expectOracle(culled)).result.ids).toEqual([1200, 1201]);

  const nodeCount = 7;
  const owned: Buffer[] = [];
  const input = (values: Float32Array | Uint32Array) => {
    const buffer = createInputBuffer(device, values);
    owned.push(buffer);
    return buffer;
  };
  const planesBuffer = createInputBuffer(device, new Float32Array(24));
  const cameraBuffer = createInputBuffer(device, new Float32Array(3));
  const scaleBuffer = createInputBuffer(device, Float32Array.of(100));
  const thresholdBuffer = createInputBuffer(device, Float32Array.of(0));
  const idsBuffer = createOutputBuffer(device, nodeCount);
  const countBuffer = createOutputBuffer(device, 1);
  const overflowBuffer = createOutputBuffer(device, 1);
  owned.push(
    planesBuffer,
    cameraBuffer,
    scaleBuffer,
    thresholdBuffer,
    idsBuffer,
    countBuffer,
    overflowBuffer
  );
  const graph = new GPUCommandGraph(device, {id: 'tile-lod-parity'});
  const selection = new GPUVirtualGeometrySelection({
    hierarchy: {
      sphereBounds: importGraphBuffer(
        graph,
        'bounds',
        input(FIXTURE_A.sphereBounds),
        'float32x4',
        nodeCount
      ),
      geometricErrors: importGraphBuffer(
        graph,
        'errors',
        input(FIXTURE_A.geometricErrors),
        'float32',
        nodeCount
      ),
      children: importGraphBuffer(
        graph,
        'children',
        input(FIXTURE_A.children),
        'uint32x2',
        nodeCount
      ),
      clusterIds: importGraphBuffer(
        graph,
        'ids-in',
        input(FIXTURE_A.tileIds as Uint32Array),
        'uint32',
        nodeCount
      ),
      levelOffsets: FIXTURE_A.levelOffsets
    },
    view: {
      frustumPlanes: importGraphBuffer(graph, 'planes', planesBuffer, 'float32x4', 6),
      cameraPosition: importGraphBuffer(graph, 'camera', cameraBuffer, 'float32x3', 1),
      pixelProjectionScale: importGraphBuffer(graph, 'scale', scaleBuffer, 'float32', 1),
      maximumScreenSpaceError: importGraphBuffer(graph, 'threshold', thresholdBuffer, 'float32', 1)
    },
    output: importGraphBuffer(graph, 'ids', idsBuffer, 'uint32', nodeCount),
    count: importGraphBuffer(graph, 'count', countBuffer, 'uint32', 1),
    overflow: importGraphBuffer(graph, 'overflow', overflowBuffer, 'uint32', 1)
  });
  graph.add(selection);
  const compiled = graph.compile();
  for (const view of [at(100), at(50), at(10), at(5), at(0), culled]) {
    planesBuffer.write(view.slice(0, 24));
    thresholdBuffer.write(
      view.slice(
        GPU_TILE_LOD_VIEW_OFFSETS.maximumScreenSpaceError,
        GPU_TILE_LOD_VIEW_OFFSETS.maximumScreenSpaceError + 1
      )
    );
    submitGraph(device, compiled, undefined);
    const reference = (await readCompactIds(idsBuffer, countBuffer)).sort(sortNumbers);
    expect((await tiles.run(view)).ids.sort(sortNumbers)).toEqual(reference);
  }
  compiled.destroy();
  selection.destroy();
  for (const buffer of owned) buffer.destroy();
  tiles.destroy();
});

it('GPUTileLODSelection spends cost and count budgets deterministically', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const tiles = createTileFixture(device, {budget: true});
  for (const [budget, expected, exhausted] of [
    [[200, UNLIMITED], [1200, 1201, 1202, 1203], 0],
    [[150, UNLIMITED], [1101, 1200, 1201], 1],
    [[50, UNLIMITED], [1000], 1],
    [[UNLIMITED, UNLIMITED], [1200, 1201, 1202, 1203], 0],
    [[UNLIMITED, 3], [1101, 1200, 1201], 1],
    [[UNLIMITED, 1], [1000], 1]
  ] as const) {
    const {result} = await tiles.expectOracle(at(5), {budget: [...budget]});
    expect(result.ids).toEqual(expected);
    expect(result.statistics[5]).toBe(exhausted);
  }
  expect((await tiles.run(at(5), {budget: [150, UNLIMITED]})).statistics[1]).toBe(140);
  tiles.destroy();

  const additive = createTileFixture(device, {budget: true, refinement: 'add'});
  expect((await additive.expectOracle(at(5))).result.ids).toEqual([
    1000, 1100, 1101, 1200, 1201, 1202, 1203
  ]);
  const capped = await additive.expectOracle(at(5), {budget: [300, UNLIMITED]});
  expect(capped.result.ids).toEqual([1000, 1100, 1101, 1200, 1201]);
  expect(capped.result.statistics[5]).toBe(1);
  additive.destroy();
});

it('GPUTileLODSelection draws resident stand-ins and requests missing tiles', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const tiles = createTileFixture(device, {requestCapacity: 3});
  let run = (await tiles.expectOracle(at(5))).result;
  expect(run.ids).toEqual([1200, 1201, 1202, 1203]);
  expect([run.requestCount, run.requestOverflow]).toEqual([0, 0]);

  run = (await tiles.expectOracle(at(5), {residency: [1, 1, 1, 1, 0, 1, 1]})).result;
  expect(run.ids).toEqual([1100, 1202, 1203]);
  expect(run.requestIds).toEqual([1201]);
  expect(run.requestPriorities[0]).toBeCloseTo(2.832, 2);
  expect(run.drawnAncestors).toEqual([INVALID, 1, INVALID, 1, 1, 5, 6]);

  run = (await tiles.expectOracle(at(5), {residency: [0, 1, 1, 0, 0, 0, 0]})).result;
  expect(run.ids).toEqual([1100, 1101]);
  expect(run).toMatchObject({
    requestIds: [1000, 1200, 1201],
    requestCount: 3,
    requestOverflow: 1,
    requestTotal: 5
  });

  run = (await tiles.expectOracle(at(5), {residency: [1, 0, 0, 1, 1, 1, 1]})).result;
  expect(run.ids).toEqual([1000]);
  expect(run.requestIds).toEqual([1100, 1101]);

  run = (await tiles.expectOracle(at(5), {enabledNodes: [1, 0, 1, 1, 1, 1, 1]})).result;
  expect(run.ids).toEqual([1202, 1203]);
  tiles.destroy();
});

it('GPUTileLODSelection relaxes error with foveation and distance falloff', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const tiles = createTileFixture(device);
  const matrix = [0.2, 0, 0, 0, 0, 0.2, 0, 0, 0, 0, 0.01, 0, 0, 0, 0, 1];
  const foveated = (strength: number) =>
    at(10, {viewProjectionMatrix: matrix, foveation: {center: [0.5, 0.5], radius: 0.1, strength}});
  expect((await tiles.expectOracle(foveated(0))).result.ids).toEqual([1101, 1200, 1201]);
  expect((await tiles.expectOracle(foveated(4))).result.ids).toEqual([1100, 1101]);
  expect(
    (await tiles.expectOracle(at(10, {focusDistance: 10, distanceFalloff: 1}))).result.ids
  ).toEqual([1100, 1101]);
  tiles.destroy();
});

it('GPUTileLODSelection writes indirect dispatch records and matches a quadtree oracle', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const dispatch = createTileFixture(device, {indirectDispatchWorkgroupSize: 2});
  expect((await dispatch.run(at(10))).dispatch).toEqual([2, 1, 1]);
  const allCulled = await dispatch.run(
    at(10, {frustumPlanes: Array.from(makeBoxFrustum(-1000, -100, 1000))})
  );
  expect(allCulled.dispatch).toEqual([0, 1, 1]);
  expect(allCulled.count).toBe(0);
  dispatch.destroy();

  const quadtree = makeGPUTileLODQuadtree({bounds: [0, 0, 256, 256], maximumLevel: 3, maximumZ: 0});
  const fixture: TileLODFixture = {
    sphereBounds: quadtree.sphereBounds,
    geometricErrors: quadtree.geometricErrors,
    children: quadtree.children,
    levelOffsets: quadtree.levelOffsets,
    parents: quadtree.parents
  };
  for (const budget of [false, true]) {
    const tiles = createTileFixture(device, {fixture, budget});
    for (const threshold of [2, 8, 32]) {
      for (const frustumPlanes of [undefined, Array.from(makeBoxFrustum(-1000, 128, 1000))]) {
        const view = makeFixtureView({
          cameraPosition: [64, 64, 40],
          maximumScreenSpaceError: threshold,
          ...(frustumPlanes ? {frustumPlanes} : {})
        });
        await tiles.expectOracle(view, budget ? {budget: [UNLIMITED, 20]} : {});
      }
    }
    tiles.destroy();
  }
});
