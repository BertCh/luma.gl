// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  GPUSplatGraphRenderer,
  makeGPUSplatData,
  SplatResidencyManager,
  type GPUSplatData
} from '@luma.gl/splats';
import {NullDevice} from '@luma.gl/test-utils';
import {
  SplatHierarchyManager,
  getSplatHierarchyFoveatedPriority,
  getSplatHierarchyRefinementError,
  getSplatHierarchyScreenSpaceError,
  isSplatHierarchyNodeVisible,
  type SplatHierarchyLoadContext,
  type SplatHierarchyNode,
  type SplatHierarchyView
} from '../src/splat-hierarchy';

const IDENTITY_MATRIX = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] as const;

it('Splat hierarchy projects geometric error and conservatively culls bounding spheres', () => {
  const centeredNode = makeSplatHierarchyNode('center', [0, 0, 0], 1, 0.1);
  const peripheralNode = makeSplatHierarchyNode('peripheral', [0.8, 0, 0], 1, 0.1);
  const outsideNode = makeSplatHierarchyNode('outside', [1.3, 0, 0], 1, 0.1);
  const intersectingNode = makeSplatHierarchyNode('intersecting', [1.1, 0, 0], 1, 0.2);
  const nearView = makeSplatHierarchyView([0, 0, 2]);
  const farView = makeSplatHierarchyView([0, 0, 20]);
  const nearError = getSplatHierarchyScreenSpaceError(centeredNode, nearView);
  const farError = getSplatHierarchyScreenSpaceError(centeredNode, farView);

  expect(
    Boolean(nearError > farError),
    'increases projected geometric error for nearby source pages'
  ).toBe(true);
  expect(Boolean(nearError > 0), 'reports finite physical-pixel approximation error').toBe(true);
  expect(
    Boolean(isSplatHierarchyNodeVisible(centeredNode, IDENTITY_MATRIX)),
    'retains visible pages'
  ).toBe(true);
  expect(
    Boolean(isSplatHierarchyNodeVisible(outsideNode, IDENTITY_MATRIX)),
    'culls a page whose complete bounding sphere lies outside a clip plane'
  ).toBe(false);
  expect(
    Boolean(isSplatHierarchyNodeVisible(intersectingNode, IDENTITY_MATRIX)),
    'retains a bounding sphere intersecting the conservative clip volume'
  ).toBe(true);

  const foveation = {center: [0.5, 0.5] as const, radius: 0.05, strength: 8};
  const centeredPriority = getSplatHierarchyFoveatedPriority(
    centeredNode,
    nearView,
    foveation,
    100
  );
  const peripheralPriority = getSplatHierarchyFoveatedPriority(
    peripheralNode,
    nearView,
    foveation,
    100
  );
  expect(centeredPriority, 'preserves full detail around the gaze position').toBe(100);
  expect(
    Boolean(peripheralPriority < centeredPriority),
    'relaxes refinement away from the gaze position'
  ).toBe(true);
  expect(
    getSplatHierarchyFoveatedPriority(peripheralNode, nearView, {strength: 0}, 100),
    'preserves geometric priority when foveation is disabled'
  ).toBe(100);
  void 0;
});

it('SplatHierarchyManager retains a parent until every replacing child is resident', async () => {
  const device = new NullDevice({});
  const parentBatch = makeSplatHierarchyBatch(device, 0, 10);
  const firstChildBatch = makeSplatHierarchyBatch(device, 1, 24);
  const secondChildBatch = makeSplatHierarchyBatch(device, 2, 48);
  const residencyManager = new SplatResidencyManager({maxResidentChunks: 3});
  const pendingPages = new Map<string, (batch: GPUSplatData) => void>();
  const frontierEvents: number[][] = [];
  const root: SplatHierarchyNode = {
    ...makeSplatHierarchyNode('root', [0, 0, 0], 1, 0.2),
    data: parentBatch,
    children: [
      {...makeSplatHierarchyNode('first', [-0.25, 0, 0], 0), parentId: 'root'},
      {...makeSplatHierarchyNode('second', [0.25, 0, 0], 0), parentId: 'root'}
    ]
  };
  const manager = new SplatHierarchyManager({
    roots: [root],
    residencyManager,
    maximumScreenSpaceError: 2,
    maxConcurrentLoads: 2,
    loadPage: node =>
      new Promise(resolve => {
        pendingPages.set(node.id, resolve);
      }),
    onFrontierChange: batches => {
      frontierEvents.push(batches.map(batch => batch.sourceBatchIndex));
    }
  });

  const initialFrontier = manager.update(makeSplatHierarchyView());
  expect(
    initialFrontier.map(entry => entry.node.id),
    'immediately renders the root'
  ).toEqual(['root']);
  expect(
    Boolean(initialFrontier[0].isFallback),
    'marks a resident parent as a temporary refinement fallback'
  ).toBe(true);
  expect(
    Boolean(residencyManager.getChunk('root')?.pinned),
    'protects the visible fallback against eviction'
  ).toBe(true);
  expect(manager.stats.pendingLoadCount, 'starts bounded independent child page requests').toBe(2);

  manager.update(makeSplatHierarchyView());
  expect(
    manager.stats.pendingLoadCount,
    'coalesces repeated camera updates into one load per page'
  ).toBe(2);

  await flushSplatHierarchyMicrotasks();
  pendingPages.get('first')?.(firstChildBatch);
  await flushSplatHierarchyMicrotasks();

  expect(
    Boolean(residencyManager.has('first')),
    'retains the completed first source page independently'
  ).toBe(true);
  expect(
    manager.frontier.map(entry => entry.node.id),
    'does not mix a replacing child with an overlapping resident fallback'
  ).toEqual(['root']);
  expect(
    Boolean(residencyManager.getChunk('first')?.pinned),
    'protects a completed replacement sibling while the parent still covers its missing peer'
  ).toBe(true);
  expect(manager.stats.pendingLoadCount, 'keeps the missing sibling request in flight').toBe(1);

  pendingPages.get('second')?.(secondChildBatch);
  await manager.waitForIdle();

  expect(
    manager.frontier.map(entry => entry.node.id),
    'atomically replaces the parent once both child source pages are resident'
  ).toEqual(['first', 'second']);
  expect(
    manager.frontierBatches.map(batch => batch.sourceInfo),
    'preserves exact source-batch and global-row identities for rendering and picking'
  ).toEqual([firstChildBatch.sourceInfo, secondChildBatch.sourceInfo]);
  expect(frontierEvents, 'notifies renderers only when visible coverage changes').toEqual([
    [0],
    [1, 2]
  ]);
  expect(
    Boolean(residencyManager.getChunk('root')?.pinned),
    'releases obsolete hierarchy-owned parent pins'
  ).toBe(false);
  expect(manager.stats.completedLoadCount, 'records each independently completed source page').toBe(
    2
  );
  expect(manager.getNode('second')?.parentId, 'preserves original source parent metadata').toBe(
    'root'
  );

  manager.destroy();
  expect(
    Boolean(residencyManager.destroyed),
    'never destroys a caller-owned shared residency window'
  ).toBe(false);
  residencyManager.destroy();
  parentBatch.destroy();
  firstChildBatch.destroy();
  secondChildBatch.destroy();
  void 0;
});

it('SplatHierarchyManager synchronizes replacing and empty frontiers with a GPU graph', async () => {
  const device = new NullDevice({});
  Object.defineProperties(device, {
    type: {value: 'webgpu'},
    info: {value: {...device.info, type: 'webgpu', shadingLanguage: 'wgsl'}}
  });
  const rootBatch = makeSplatHierarchyBatch(device, 26, 1300);
  const firstChildBatch = makeSplatHierarchyBatch(device, 27, 1600);
  const secondChildBatch = makeSplatHierarchyBatch(device, 28, 2400);
  const graphRenderer = new GPUSplatGraphRenderer(device, {
    expectedSplatCount: 2,
    expectedBatchCount: 2,
    viewportSize: [256, 256]
  });
  const graphFrontiers: Array<readonly GPUSplatData[]> = [];
  const manager = new SplatHierarchyManager({
    maximumScreenSpaceError: 1,
    roots: [
      {
        ...makeSplatHierarchyNode('root', [0, 0, 0], 1, 0.3),
        data: rootBatch,
        children: [
          makeSplatHierarchyNode('first', [-0.2, 0, 0], 0),
          makeSplatHierarchyNode('second', [0.2, 0, 0], 0)
        ]
      }
    ],
    loadPage: node => (node.id === 'first' ? firstChildBatch : secondChildBatch),
    onFrontierChange: batches => {
      graphFrontiers.push([...batches]);
      graphRenderer.setProps({data: batches});
    }
  });

  manager.update(makeSplatHierarchyView());
  expect(graphRenderer.batches, 'publishes the intact resident graph fallback').toEqual([
    rootBatch
  ]);
  expect(graphRenderer.compiledGraph, 'does not force graph compilation from traversal').toBe(
    undefined
  );

  await manager.waitForIdle();
  expect(
    graphRenderer.batches,
    'atomically replaces borrowed graph source slots once every child page is resident'
  ).toEqual([firstChildBatch, secondChildBatch]);
  expect(
    graphRenderer.batches.map(batch => batch.sourceInfo),
    'preserves graph picking source-batch and noncontiguous global-row identities'
  ).toEqual([firstChildBatch.sourceInfo, secondChildBatch.sourceInfo]);
  expect(
    graphRenderer.batches[0].positions.data[0].buffer,
    'shares the original source allocation with the graph instead of repacking it'
  ).toBe(firstChildBatch.positions.data[0].buffer);

  manager.update({
    ...makeSplatHierarchyView(),
    modelViewProjectionMatrix: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 3, 0, 0, 1]
  });
  expect(graphRenderer.batches, 'detaches every graph source when the frontier is empty').toEqual(
    []
  );
  expect(manager.stats.frontierNodeCount, 'reports the fully culled hierarchy frontier').toBe(0);
  expect(
    Boolean(firstChildBatch.destroyed),
    'preserves a borrowed page outside the graph-visible frontier'
  ).toBe(false);
  expect(
    Boolean(secondChildBatch.destroyed),
    'preserves every independently owned inactive source page'
  ).toBe(false);

  manager.update(makeSplatHierarchyView());
  expect(
    graphRenderer.batches,
    'restores the original resident graph frontier when the camera returns'
  ).toEqual([firstChildBatch, secondChildBatch]);
  expect(
    graphFrontiers.map(batches => batches.map(batch => batch.sourceBatchIndex)),
    'publishes only complete parent, child, empty, and restored graph frontiers'
  ).toEqual([[26], [27, 28], [], [27, 28]]);

  graphRenderer.destroy();
  manager.destroy();
  rootBatch.destroy();
  firstChildBatch.destroy();
  secondChildBatch.destroy();
  void 0;
});

it('SplatHierarchyManager supports additive source-page refinement without repacking', () => {
  const device = new NullDevice({});
  const parentBatch = makeSplatHierarchyBatch(device, 3, 100);
  const firstChildBatch = makeSplatHierarchyBatch(device, 4, 101);
  const secondChildBatch = makeSplatHierarchyBatch(device, 5, 102);
  const manager = new SplatHierarchyManager({
    maximumScreenSpaceError: 1,
    roots: [
      {
        ...makeSplatHierarchyNode('root', [0, 0, 0], 1),
        data: parentBatch,
        refinement: 'add',
        children: [
          {...makeSplatHierarchyNode('first', [-0.2, 0, 0], 0), data: firstChildBatch},
          {...makeSplatHierarchyNode('second', [0.2, 0, 0], 0), data: secondChildBatch}
        ]
      }
    ]
  });

  manager.update(makeSplatHierarchyView());
  expect(
    manager.frontierBatches,
    'retains intact parent and independent additive child source batches'
  ).toEqual([parentBatch, firstChildBatch, secondChildBatch]);
  expect(
    manager.frontier.map(entry => entry.levelOfDetail),
    'retains independently traversed source hierarchy levels'
  ).toEqual([0, 1, 1]);
  expect(
    Boolean(manager.frontier[0].isFallback),
    'marks complete additive refinement as resident'
  ).toBe(false);
  expect(manager.residencyManager.stats.residentChunkCount, 'never merges source allocations').toBe(
    3
  );

  manager.destroy();
  expect(
    Boolean(manager.residencyManager.destroyed),
    'destroys only the hierarchy-created residency window'
  ).toBe(true);
  expect(Boolean(parentBatch.destroyed), 'preserves borrowed parent source buffers').toBe(false);
  expect(Boolean(firstChildBatch.destroyed), 'preserves borrowed child source buffers').toBe(false);
  parentBatch.destroy();
  firstChildBatch.destroy();
  secondChildBatch.destroy();
  void 0;
});

it('SplatHierarchyManager prioritizes foveated source pages and limits decoder workers', async () => {
  const device = new NullDevice({});
  const rootBatch = makeSplatHierarchyBatch(device, 6, 200);
  const focusedBatch = makeSplatHierarchyBatch(device, 7, 201);
  const peripheralBatch = makeSplatHierarchyBatch(device, 8, 202);
  const pendingPages = new Map<string, (batch: GPUSplatData) => void>();
  const loadOrder: string[] = [];
  const loadContexts: SplatHierarchyLoadContext[] = [];
  const manager = new SplatHierarchyManager({
    maxConcurrentLoads: 1,
    maximumScreenSpaceError: 1,
    foveation: {center: [0.5, 0.5], radius: 0.05, strength: 10},
    roots: [
      {
        ...makeSplatHierarchyNode('root', [0, 0, 0], 2, 0.9),
        data: rootBatch,
        children: [
          {
            ...makeSplatHierarchyNode('peripheral', [0.8, 0, 0], 1),
            contentUri: 'tiles/peripheral.spz',
            metadata: {compression: 'spz-v2'}
          },
          {
            ...makeSplatHierarchyNode('focused', [0, 0, 0], 1),
            contentUri: 'tiles/focused.spz'
          }
        ]
      }
    ],
    loadPage: (node, context) => {
      loadOrder.push(node.id);
      loadContexts.push(context);
      return new Promise(resolve => pendingPages.set(node.id, resolve));
    }
  });

  manager.update(makeSplatHierarchyView());
  await flushSplatHierarchyMicrotasks();
  expect(loadOrder, 'schedules the gaze-centered page before peripheral content').toEqual([
    'focused'
  ]);
  expect(manager.stats.pendingLoadCount, 'limits simultaneously running decoder workers').toBe(1);
  expect(manager.stats.queuedLoadCount, 'applies explicit source-page request backpressure').toBe(
    1
  );
  expect(loadContexts[0].levelOfDetail, 'provides hierarchy depth to worker-style loaders').toBe(1);
  expect(
    Boolean(loadContexts[0].signal.aborted),
    'provides a live worker cancellation signal'
  ).toBe(false);
  expect(
    manager.getNode('peripheral')?.contentUri,
    'preserves format-independent source content locations'
  ).toBe('tiles/peripheral.spz');
  expect(
    manager.getNode('peripheral')?.metadata,
    'passes compression or feature metadata through without a source-format dependency'
  ).toEqual({compression: 'spz-v2'});

  pendingPages.get('focused')?.(focusedBatch);
  await flushSplatHierarchyMicrotasks();
  expect(loadOrder, 'starts the next source page only after a slot opens').toEqual([
    'focused',
    'peripheral'
  ]);
  expect(
    manager.stats.pendingLoadCount,
    'maintains the configured bounded decoder concurrency'
  ).toBe(1);
  expect(
    Boolean(loadContexts[0].priority > loadContexts[1].priority),
    'preserves foveated scheduling priorities'
  ).toBe(true);

  pendingPages.get('peripheral')?.(peripheralBatch);
  await manager.waitForIdle();
  expect(
    manager.frontierBatches,
    'keeps original child traversal order independent of worker completion order'
  ).toEqual([peripheralBatch, focusedBatch]);
  expect(manager.stats.pendingLoadCount, 'releases every completed decoder worker slot').toBe(0);
  expect(manager.stats.queuedLoadCount, 'drains every queued source page request').toBe(0);

  manager.destroy();
  rootBatch.destroy();
  focusedBatch.destroy();
  peripheralBatch.destroy();
  void 0;
});

it('SplatHierarchyManager cancels source workers after conservative view culling', async () => {
  const device = new NullDevice({});
  const rootBatch = makeSplatHierarchyBatch(device, 9, 300);
  let observedSignal: AbortSignal | undefined;
  let loadErrors = 0;
  const manager = new SplatHierarchyManager({
    maximumScreenSpaceError: 1,
    roots: [
      {
        ...makeSplatHierarchyNode('root', [0, 0, 0], 1, 2),
        data: rootBatch,
        children: [makeSplatHierarchyNode('cancelled', [0.8, 0, 0], 0, 0.05)]
      }
    ],
    loadPage: (_node, context) => {
      observedSignal = context.signal;
      return new Promise((_resolve, reject) => {
        context.signal.addEventListener('abort', () => reject(new Error('worker aborted')), {
          once: true
        });
      });
    },
    onLoadError: () => {
      loadErrors++;
    }
  });

  manager.update(makeSplatHierarchyView());
  await flushSplatHierarchyMicrotasks();
  expect(manager.stats.pendingLoadCount, 'starts one source decoder worker').toBe(1);

  manager.update({
    ...makeSplatHierarchyView(),
    modelViewProjectionMatrix: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -2.9, 0, 0, 1]
  });
  await manager.waitForIdle();

  expect(
    Boolean(observedSignal?.aborted),
    'aborts a decoder worker whose source page leaves the view'
  ).toBe(true);
  expect(
    manager.stats.abortedLoadCount,
    'records view-driven worker cancellation exactly once'
  ).toBe(1);
  expect(manager.stats.culledNodeCount, 'excludes the invisible child source branch').toBe(1);
  expect(loadErrors, 'does not report expected cancellation as a source loader failure').toBe(0);
  expect(manager.frontierBatches, 'preserves intersecting resident parent coverage').toEqual([
    rootBatch
  ]);

  manager.destroy();
  rootBatch.destroy();
  void 0;
});

it('SplatHierarchyManager never starts worker decoding after immediate cancellation', async () => {
  const device = new NullDevice({});
  const rootBatch = makeSplatHierarchyBatch(device, 25, 1200);
  let startedWorkers = 0;
  let reportedErrors = 0;
  const manager = new SplatHierarchyManager({
    maximumScreenSpaceError: 1,
    roots: [
      {
        ...makeSplatHierarchyNode('root', [0, 0, 0], 1, 2),
        data: rootBatch,
        children: [makeSplatHierarchyNode('cancelled', [0.8, 0, 0], 0, 0.05)]
      }
    ],
    loadPage: () => {
      startedWorkers++;
      throw new Error('cancelled worker started');
    },
    onLoadError: () => {
      reportedErrors++;
    }
  });

  manager.update(makeSplatHierarchyView());
  manager.update({
    ...makeSplatHierarchyView(),
    modelViewProjectionMatrix: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -2.9, 0, 0, 1]
  });
  await manager.waitForIdle();

  expect(startedWorkers, 'checks cancellation before dispatching a queued decoder microtask').toBe(
    0
  );
  expect(reportedErrors, 'does not report expected pre-dispatch cancellation as a failure').toBe(0);
  expect(manager.stats.abortedLoadCount, 'records immediate cancellation once').toBe(1);
  expect(manager.stats.pendingLoadCount, 'releases the cancelled decoder slot').toBe(0);
  expect(manager.frontierBatches, 'preserves the visible resident source fallback').toEqual([
    rootBatch
  ]);

  manager.destroy();
  rootBatch.destroy();
  void 0;
});

it('SplatHierarchyManager reserves estimated source capacity before starting decoder work', async () => {
  const device = new NullDevice({});
  const rootBatch = makeSplatHierarchyBatch(device, 10, 400);
  const childBatch = makeSplatHierarchyBatch(device, 11, 401);
  const residencyManager = new SplatResidencyManager({maxResidentChunks: 1});
  let requestedLoads = 0;
  const manager = new SplatHierarchyManager({
    residencyManager,
    maximumScreenSpaceError: 1,
    roots: [
      {
        ...makeSplatHierarchyNode('root', [0, 0, 0], 1),
        data: rootBatch,
        children: [
          {
            ...makeSplatHierarchyNode('child', [0, 0, 0], 0),
            estimatedGpuBytes: childBatch.byteLength,
            estimatedSplatCount: childBatch.length
          }
        ]
      }
    ],
    loadPage: () => {
      requestedLoads++;
      return childBatch;
    }
  });

  manager.update(makeSplatHierarchyView());
  await manager.waitForIdle();

  expect(requestedLoads, 'rejects over-budget worker requests before invoking the decoder').toBe(0);
  expect(manager.stats.rejectedLoadCount, 'records transactional source-page rejection').toBe(1);
  expect(manager.frontierBatches, 'retains the protected fallback under pressure').toEqual([
    rootBatch
  ]);
  expect(
    Boolean(residencyManager.getChunk('root')?.pinned),
    'never evicts the visible parent to start a page'
  ).toBe(true);

  residencyManager.setBudget({maxResidentChunks: 2});
  manager.update(makeSplatHierarchyView());
  await manager.waitForIdle();

  expect(
    requestedLoads,
    'retries rejected source pages after a later camera or budget update'
  ).toBe(1);
  expect(manager.frontierBatches, 'replaces fallback after bounded page admission').toEqual([
    childBatch
  ]);
  expect(manager.stats.completedLoadCount, 'records the successfully prepared source page').toBe(1);

  manager.destroy();
  residencyManager.destroy();
  rootBatch.destroy();
  childBatch.destroy();
  void 0;
});

it('SplatHierarchyManager reserves unknown page slots and avoids reloading prepared sources', async () => {
  const device = new NullDevice({});
  const rootBatch = makeSplatHierarchyBatch(device, 19, 1000);
  const preparedChildBatch = makeSplatHierarchyBatch(device, 20, 1001);
  const unloadedChildBatch = makeSplatHierarchyBatch(device, 21, 1002);
  const residencyManager = new SplatResidencyManager({maxResidentChunks: 1});
  let decoderRequests = 0;
  const manager = new SplatHierarchyManager({
    residencyManager,
    maximumScreenSpaceError: 1,
    roots: [
      {
        ...makeSplatHierarchyNode('root', [0, 0, 0], 1),
        data: rootBatch,
        children: [
          {...makeSplatHierarchyNode('prepared', [-0.2, 0, 0], 0), data: preparedChildBatch},
          makeSplatHierarchyNode('unloaded', [0.2, 0, 0], 0)
        ]
      }
    ],
    loadPage: () => {
      decoderRequests++;
      return unloadedChildBatch;
    }
  });

  manager.update(makeSplatHierarchyView());
  await manager.waitForIdle();

  expect(decoderRequests, 'reserves an intact page slot even when source sizes are unknown').toBe(
    0
  );
  expect(manager.stats.rejectedLoadCount, 'rejects only the genuinely unloaded source page').toBe(
    1
  );
  expect(
    Boolean(residencyManager.has('prepared')),
    'does not upload a prepared source page beyond the active residency budget'
  ).toBe(false);
  expect(manager.frontierBatches, 'retains protected resident fallback coverage').toEqual([
    rootBatch
  ]);

  residencyManager.setBudget({maxResidentChunks: 3});
  manager.update(makeSplatHierarchyView());
  await manager.waitForIdle();

  expect(decoderRequests, 'never invokes the source loader for an already prepared page').toBe(1);
  expect(
    manager.frontierBatches,
    'preserves caller-prepared and asynchronously loaded source batches independently'
  ).toEqual([preparedChildBatch, unloadedChildBatch]);

  manager.destroy();
  residencyManager.destroy();
  rootBatch.destroy();
  preparedChildBatch.destroy();
  unloadedChildBatch.destroy();
  void 0;
});

it('SplatHierarchyManager never thrashes replacement siblings that cannot all fit', async () => {
  const device = new NullDevice({});
  const rootBatch = makeSplatHierarchyBatch(device, 22, 1100);
  const firstChildBatch = makeSplatHierarchyBatch(device, 23, 1101);
  const secondChildBatch = makeSplatHierarchyBatch(device, 24, 1102);
  const residencyManager = new SplatResidencyManager({maxResidentChunks: 2});
  const decoderRequests: string[] = [];
  const manager = new SplatHierarchyManager({
    residencyManager,
    maximumScreenSpaceError: 1,
    maxConcurrentLoads: 1,
    roots: [
      {
        ...makeSplatHierarchyNode('root', [0, 0, 0], 1),
        data: rootBatch,
        children: [
          makeSplatHierarchyNode('first', [-0.2, 0, 0], 0),
          makeSplatHierarchyNode('second', [0.2, 0, 0], 0)
        ]
      }
    ],
    loadPage: node => {
      decoderRequests.push(node.id);
      return node.id === 'first' ? firstChildBatch : secondChildBatch;
    }
  });

  manager.update(makeSplatHierarchyView());
  await manager.waitForIdle();

  expect(decoderRequests, 'never decodes a sibling beyond atomic replacement capacity').toEqual([
    'first'
  ]);
  expect(manager.stats.completedLoadCount, 'prepares the first replacement sibling once').toBe(1);
  expect(manager.stats.rejectedLoadCount, 'rejects the blocked second sibling once').toBe(1);
  expect(manager.stats.pendingLoadCount, 'settles bounded workers without a retry loop').toBe(0);
  expect(
    manager.stats.queuedLoadCount,
    'does not leave an unfulfillable source request queued'
  ).toBe(0);
  expect(manager.frontierBatches, 'retains complete resident fallback coverage').toEqual([
    rootBatch
  ]);
  expect(
    Boolean(residencyManager.getChunk('root')?.pinned),
    'protects the rendered source parent'
  ).toBe(true);
  expect(
    Boolean(residencyManager.getChunk('first')?.pinned),
    'protects the already decoded replacement sibling'
  ).toBe(true);
  expect(
    residencyManager.stats.evictedChunkCount,
    'never evicts and reloads replacing siblings'
  ).toBe(0);

  manager.update(makeSplatHierarchyView());
  await manager.waitForIdle();
  expect(
    decoderRequests,
    'remains stable across repeated views while replacement capacity is unavailable'
  ).toEqual(['first']);

  residencyManager.setBudget({maxResidentChunks: 3});
  manager.update(makeSplatHierarchyView());
  await manager.waitForIdle();

  expect(decoderRequests, 'loads only the missing sibling after capacity grows').toEqual([
    'first',
    'second'
  ]);
  expect(
    manager.frontierBatches,
    'atomically activates intact children after all replacing pages are resident'
  ).toEqual([firstChildBatch, secondChildBatch]);
  expect(
    Boolean(residencyManager.getChunk('root')?.pinned),
    'releases the replaced fallback source page'
  ).toBe(false);

  manager.destroy();
  residencyManager.destroy();
  rootBatch.destroy();
  firstChildBatch.destroy();
  secondChildBatch.destroy();
  void 0;
});

it('SplatHierarchyManager retries failed source workers only on a new explicit update', async () => {
  const device = new NullDevice({});
  const rootBatch = makeSplatHierarchyBatch(device, 12, 500);
  const childBatch = makeSplatHierarchyBatch(device, 13, 501);
  let attemptedLoads = 0;
  const failures: string[] = [];
  const manager = new SplatHierarchyManager({
    maximumScreenSpaceError: 1,
    roots: [
      {
        ...makeSplatHierarchyNode('root', [0, 0, 0], 1),
        data: rootBatch,
        children: [makeSplatHierarchyNode('child', [0, 0, 0], 0)]
      }
    ],
    loadPage: async () => {
      attemptedLoads++;
      if (attemptedLoads === 1) {
        throw new Error('source decoder unavailable');
      }
      return childBatch;
    },
    onLoadError: (error, node) => {
      failures.push(`${node.id}:${error instanceof Error ? error.message : String(error)}`);
    }
  });

  manager.update(makeSplatHierarchyView());
  await manager.waitForIdle();
  expect(attemptedLoads, 'does not automatically loop on a failed source worker request').toBe(1);
  expect(failures, 'reports source errors alongside their original hierarchy metadata').toEqual([
    'child:source decoder unavailable'
  ]);
  expect(manager.frontierBatches, 'retains resident coverage after decoder failure').toEqual([
    rootBatch
  ]);

  manager.update(makeSplatHierarchyView());
  await manager.waitForIdle();
  expect(attemptedLoads, 'allows a later explicit update to retry the source decoder').toBe(2);
  expect(manager.frontierBatches, 'publishes the independently recovered source').toEqual([
    childBatch
  ]);

  manager.destroy();
  rootBatch.destroy();
  childBatch.destroy();
  void 0;
});

it('SplatHierarchyManager retains caller-owned pins and discards stale worker results', async () => {
  const device = new NullDevice({});
  const rootBatch = makeSplatHierarchyBatch(device, 14, 600);
  const staleBatch = makeSplatHierarchyBatch(device, 15, 601);
  const residencyManager = new SplatResidencyManager();
  residencyManager.add(rootBatch, {id: 'root', pinned: true, ownsData: false});
  let finishStalePage: ((batch: GPUSplatData) => void) | undefined;
  const manager = new SplatHierarchyManager({
    residencyManager,
    maximumScreenSpaceError: 1,
    roots: [
      {
        ...makeSplatHierarchyNode('root', [0, 0, 0], 1),
        data: rootBatch,
        children: [
          {...makeSplatHierarchyNode('stale', [0, 0, 0], 0), ownsData: true, parentId: 'root'}
        ]
      }
    ],
    loadPage: () =>
      new Promise(resolve => {
        finishStalePage = resolve;
      })
  });

  manager.update(makeSplatHierarchyView());
  await flushSplatHierarchyMicrotasks();
  manager.destroy();

  expect(Boolean(manager.destroyed), 'marks hierarchy traversal as destroyed').toBe(true);
  expect(
    Boolean(residencyManager.getChunk('root')?.pinned),
    'never releases an existing caller-owned source pin'
  ).toBe(true);
  expect(manager.stats.abortedLoadCount, 'cancels an outstanding worker on destruction').toBe(1);

  finishStalePage?.(staleBatch);
  await manager.waitForIdle();

  expect(
    Boolean(residencyManager.has('stale')),
    'discards a page completed after hierarchy destruction'
  ).toBe(false);
  expect(
    Boolean(staleBatch.destroyed),
    'destroys only stale source buffers with explicitly transferred ownership'
  ).toBe(true);
  expect(Boolean(rootBatch.destroyed), 'preserves borrowed externally pinned source buffers').toBe(
    false
  );
  expect(
    Boolean(residencyManager.destroyed),
    'preserves the caller-owned shared residency manager'
  ).toBe(false);

  residencyManager.destroy();
  rootBatch.destroy();
  void 0;
});

it('SplatHierarchyManager prunes large source hierarchies before loading invisible branches', () => {
  const device = new NullDevice({});
  const visibleBatch = makeSplatHierarchyBatch(device, 16, 700);
  const hierarchyDepth = 11;

  function makeLargeHierarchy(
    depth: number,
    branchIdentity: string,
    visible: boolean
  ): SplatHierarchyNode {
    const node: SplatHierarchyNode = {
      ...makeSplatHierarchyNode(
        branchIdentity,
        visible ? [0, 0, 0] : [4, 0, 0],
        depth > 0 ? 1 : 0,
        depth === hierarchyDepth ? 4 : 0.05
      ),
      ...(depth === 0 && visible ? {data: visibleBatch} : {})
    };
    if (depth === 0) {
      return node;
    }
    node.children = [
      makeLargeHierarchy(depth - 1, `${branchIdentity}/visible`, visible),
      makeLargeHierarchy(depth - 1, `${branchIdentity}/hidden`, false)
    ];
    return node;
  }

  const root = makeLargeHierarchy(hierarchyDepth, 'root', true);
  const manager = new SplatHierarchyManager({
    roots: [root],
    maximumScreenSpaceError: 1
  });

  manager.update(makeSplatHierarchyView());

  expect(
    manager.stats.nodeCount,
    'indexes every caller-owned source node without preparing GPU data'
  ).toBe(2 ** (hierarchyDepth + 1) - 1);
  expect(
    manager.stats.visibleNodeCount,
    'visits only the root-to-leaf source branch intersecting the camera view'
  ).toBe(hierarchyDepth + 1);
  expect(
    manager.stats.culledNodeCount,
    'rejects invisible sibling subtrees before touching their descendants'
  ).toBe(hierarchyDepth);
  expect(manager.frontierBatches, 'returns the single intact visible source page').toEqual([
    visibleBatch
  ]);
  expect(
    manager.stats.pendingLoadCount,
    'does not schedule decoder work for invisible branches'
  ).toBe(0);
  expect(
    manager.residencyManager.stats.residentChunkCount,
    'uploads only the visible source page'
  ).toBe(1);

  manager.destroy();
  visibleBatch.destroy();
  void 0;
});

it('SplatHierarchyManager replaces same-identity roots without losing externally owned pins', () => {
  const device = new NullDevice({});
  const originalBatch = makeSplatHierarchyBatch(device, 17, 800);
  const replacementBatch = makeSplatHierarchyBatch(device, 18, 900);
  const originalNode = {
    ...makeSplatHierarchyNode('shared-root', [0, 0, 0], 0),
    data: originalBatch,
    ownsData: false
  };
  const replacementNode = {
    ...makeSplatHierarchyNode('shared-root', [0.4, 0, 0], 0, 0.2),
    data: replacementBatch,
    ownsData: true
  };
  const evictionEvents: Array<{data: GPUSplatData; reason: string}> = [];
  const frontierEvents: number[][] = [];
  const residencyManager = new SplatResidencyManager({
    maxResidentChunks: 1,
    onEvict: (chunk, reason) => evictionEvents.push({data: chunk.data, reason})
  });
  residencyManager.add(originalBatch, {
    id: originalNode.id,
    pinned: true,
    ownsData: false,
    bounds: originalNode.bounds
  });
  const manager = new SplatHierarchyManager({
    roots: [originalNode],
    residencyManager,
    onFrontierChange: batches => {
      frontierEvents.push(batches.map(batch => batch.sourceBatchIndex));
    }
  });

  manager.update(makeSplatHierarchyView());
  manager.setRoots([replacementNode]);

  expect(manager.frontierBatches, 'renders the replacement source batch').toEqual([
    replacementBatch
  ]);
  expect(manager.frontier[0].node, 'publishes replacement source metadata').toBe(replacementNode);
  expect(
    manager.frontier[0].chunk.data.sourceInfo,
    'preserves the replacement source-batch and global-row identity'
  ).toBe(replacementBatch.sourceInfo);
  expect(
    residencyManager.getChunk(originalNode.id)?.bounds,
    'updates the retained spatial metadata for the replacement source node'
  ).toEqual(replacementNode.bounds);
  expect(
    residencyManager.getChunk(originalNode.id)?.ownsData,
    'honors explicitly transferred ownership for the replacement source batch'
  ).toBe(true);
  expect(
    Boolean(residencyManager.getChunk(originalNode.id)?.pinned),
    'preserves the caller-owned source pin'
  ).toBe(true);
  expect(residencyManager.stats.residentChunkCount, 'reuses the existing bounded chunk slot').toBe(
    1
  );
  expect(
    evictionEvents,
    'reports exact replacement ownership before updating the rendered frontier'
  ).toEqual([{data: originalBatch, reason: 'replace'}]);
  expect(frontierEvents, 'notifies renderers when stable IDs gain new data').toEqual([[17], [18]]);
  expect(
    Boolean(originalBatch.destroyed),
    'never destroys the original caller-owned source batch'
  ).toBe(false);

  manager.destroy();
  expect(
    Boolean(residencyManager.getChunk(originalNode.id)?.pinned),
    'never releases externally owned pins after a same-identity replacement'
  ).toBe(true);
  residencyManager.destroy();
  expect(
    Boolean(replacementBatch.destroyed),
    'destroys the explicitly manager-owned replacement batch'
  ).toBe(true);
  originalBatch.destroy();
  void 0;
});

it('SplatHierarchyManager transfers hierarchy-owned pins and honors prior source ownership', () => {
  const device = new NullDevice({});
  const originalBatch = makeSplatHierarchyBatch(device, 19, 950);
  const replacementBatch = makeSplatHierarchyBatch(device, 20, 1_000);
  const residencyManager = new SplatResidencyManager({maxResidentChunks: 1});
  const manager = new SplatHierarchyManager({
    roots: [
      {
        ...makeSplatHierarchyNode('owned-root', [0, 0, 0], 0),
        data: originalBatch,
        ownsData: true
      }
    ],
    residencyManager
  });

  manager.update(makeSplatHierarchyView());
  expect(
    Boolean(residencyManager.getChunk('owned-root')?.pinned),
    'protects the original visible root'
  ).toBe(true);
  manager.setRoots([
    {
      ...makeSplatHierarchyNode('owned-root', [0.2, 0, 0], 0),
      data: replacementBatch,
      ownsData: false
    }
  ]);

  expect(
    Boolean(originalBatch.destroyed),
    'destroys the replaced source batch when its ownership was held'
  ).toBe(true);
  expect(manager.frontierBatches, 'publishes the intact borrowed source').toEqual([
    replacementBatch
  ]);
  expect(
    Boolean(residencyManager.getChunk('owned-root')?.pinned),
    'transfers hierarchy-owned pin protection'
  ).toBe(true);
  expect(
    residencyManager.stats.pinnedChunkCount,
    'keeps pin accounting balanced across transactional replacement'
  ).toBe(1);

  manager.destroy();
  expect(
    Boolean(residencyManager.getChunk('owned-root')?.pinned),
    'releases transferred pins owned only by this hierarchy'
  ).toBe(false);
  residencyManager.destroy();
  expect(
    Boolean(replacementBatch.destroyed),
    'preserves the explicitly borrowed replacement source'
  ).toBe(false);
  replacementBatch.destroy();
  void 0;
});

it('SplatHierarchyManager replaces source roots and rejects duplicate source identities', () => {
  const device = new NullDevice({});
  const firstBatch = makeSplatHierarchyBatch(device, 17, 800);
  const secondBatch = makeSplatHierarchyBatch(device, 18, 900);
  const firstRoot = {...makeSplatHierarchyNode('first', [0, 0, 0], 0), data: firstBatch};
  const secondRoot = {...makeSplatHierarchyNode('second', [0, 0, 0], 0), data: secondBatch};
  const manager = new SplatHierarchyManager({roots: [firstRoot]});

  manager.update(makeSplatHierarchyView());
  expect(manager.frontierBatches, 'publishes the initial intact source root').toEqual([firstBatch]);
  const initialChunk = manager.residencyManager.getChunk('first');
  manager.setRoots([firstRoot, secondRoot]);
  expect(
    manager.frontierBatches,
    'appends streamed independent roots without dropping existing visible source pages'
  ).toEqual([firstBatch, secondBatch]);
  expect(
    manager.residencyManager.getChunk('first'),
    'retains existing source allocations and residency metadata while appending streamed roots'
  ).toBe(initialChunk);
  expect(
    manager.frontierBatches.map(batch => batch.sourceInfo),
    'preserves original streamed source-batch and global-row identities'
  ).toEqual([firstBatch.sourceInfo, secondBatch.sourceInfo]);
  manager.setRoots([secondRoot]);
  expect(manager.frontierBatches, 'updates the active frontier for replacement roots').toEqual([
    secondBatch
  ]);
  expect(manager.getNode('first'), 'removes superseded caller-owned hierarchy metadata').toBe(
    undefined
  );
  expect(manager.getNode('second'), 'indexes the replacement source root directly').toBe(
    secondRoot
  );
  expect(
    () => new SplatHierarchyManager({roots: [firstRoot, {...firstRoot}]}),
    'rejects ambiguous source-page identities before traversing the hierarchy'
  ).toThrow(/unique/);

  manager.destroy();
  expect(
    () => manager.update(makeSplatHierarchyView()),
    'does not traverse a destroyed hierarchy'
  ).toThrow(/destroyed/);
  firstBatch.destroy();
  secondBatch.destroy();
  void 0;
});

it('SplatHierarchyManager does not re-traverse for loads the residency budget rejects', async () => {
  const device = new NullDevice({});
  const rootBatch = makeSplatHierarchyBatch(device, 30, 1200);
  const residencyManager = new SplatResidencyManager({maxResidentChunks: 1});
  const manager = new SplatHierarchyManager({
    residencyManager,
    maximumScreenSpaceError: 1,
    maxConcurrentLoads: 1,
    roots: [
      {
        ...makeSplatHierarchyNode('root', [0, 0, 0], 1),
        data: rootBatch,
        children: [
          makeSplatHierarchyNode('a', [-0.2, -0.2, 0], 0),
          makeSplatHierarchyNode('b', [0.2, -0.2, 0], 0),
          makeSplatHierarchyNode('c', [-0.2, 0.2, 0], 0),
          makeSplatHierarchyNode('d', [0.2, 0.2, 0], 0)
        ]
      }
    ],
    loadPage: () => {
      throw new Error('an over-budget page must never reach the decoder');
    }
  });
  // Private, and counted here because a traversal per rejection is the regression: under a full
  // budget every visible child is rejected, and each one used to re-walk the whole tree.
  const internals = manager as unknown as {refresh: () => void};
  const refresh = internals.refresh.bind(manager);
  let traversalCount = 0;
  internals.refresh = () => {
    traversalCount++;
    refresh();
  };

  manager.update(makeSplatHierarchyView());
  await manager.waitForIdle();

  expect(manager.stats.rejectedLoadCount, 'rejects every child that cannot fit').toBe(4);
  expect(traversalCount, 'traverses once for the view, not once per rejection').toBe(1);
  expect(manager.stats.queuedLoadCount, 'drains the queue through the freed load slots').toBe(0);
  expect(manager.frontierBatches, 'keeps drawing the resident parent').toEqual([rootBatch]);

  manager.destroy();
  residencyManager.destroy();
  rootBatch.destroy();
});

it('SplatHierarchyManager evicts pages it no longer wants for pages it does', async () => {
  const device = new NullDevice({});
  const nearBatch = makeSplatHierarchyBatch(device, 40, 1600);
  const farBatch = makeSplatHierarchyBatch(device, 41, 1700);
  const residencyManager = new SplatResidencyManager({maxResidentChunks: 1, ownsData: false});
  const manager = new SplatHierarchyManager({
    residencyManager,
    maximumScreenSpaceError: 1,
    roots: [
      makeSplatHierarchyNode('near', [0, 0, 0], 1),
      makeSplatHierarchyNode('far', [1.3, 0, 0], 1)
    ],
    loadPage: node => (node.id === 'near' ? nearBatch : farBatch)
  });

  // A close camera loads `near` at a large projected error; `far` is outside the frustum.
  manager.update(makeSplatHierarchyView([0, 0, 0.2]));
  await manager.waitForIdle();
  manager.update(makeSplatHierarchyView([0, 0, 0.2]));
  expect(manager.frontierBatches, 'loads the page in view').toEqual([nearBatch]);

  // Turn to `far` from much further away, so it is wanted at a far smaller error than `near` had.
  const turnedView = {
    ...makeSplatHierarchyView([1.3, 0, 40]),
    modelViewProjectionMatrix: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -1.3, 0, 0, 1]
  };
  manager.update(turnedView);
  await manager.waitForIdle();
  manager.update(turnedView);

  expect(
    manager.frontierBatches,
    'admits the page in view by evicting the one that left it, whatever its old priority'
  ).toEqual([farBatch]);
  expect(manager.stats.rejectedLoadCount, 'without refusing it first').toBe(0);

  manager.destroy();
  residencyManager.destroy();
  nearBatch.destroy();
  farBatch.destroy();
});

it('Splat hierarchy relaxes refinement error by focus distance', () => {
  const nearNode = makeSplatHierarchyNode('near', [0, 0, 0], 1);
  const farNode = makeSplatHierarchyNode('far', [0, 0, -36], 1);
  const view = {...makeSplatHierarchyView(), focusDistance: 4};
  const nearError = getSplatHierarchyScreenSpaceError(nearNode, view);
  const farError = getSplatHierarchyScreenSpaceError(farNode, view);

  expect(
    getSplatHierarchyRefinementError(nearNode, view, undefined, 1),
    'leaves a page nearer than the focus distance at its projected error'
  ).toBeCloseTo(nearError);
  expect(
    getSplatHierarchyRefinementError(farNode, view, undefined, 1),
    'divides the error of a page ten focus distances away by ten'
  ).toBeCloseTo(farError / ((40 - 0.05) / 4));
  expect(
    getSplatHierarchyRefinementError(farNode, view, undefined, 0),
    'applies no distance relaxation by default'
  ).toBeCloseTo(farError);
});

it('Splat hierarchy foveates by the nearest edge of a page, not its center', () => {
  const view = makeSplatHierarchyView([0, 0, 4]);
  const foveation = {radius: 0.05, strength: 8};
  // Centered at the right edge of the view, but wide enough to reach well into the gaze.
  const wideNode = makeSplatHierarchyNode('wide', [0.9, 0, 0], 1, 3.5);
  const smallNode = makeSplatHierarchyNode('small', [0.9, 0, 0], 1, 0.01);
  const aroundCamera = makeSplatHierarchyNode('around', [0.9, 0, 3.5], 1, 1.5);

  expect(
    getSplatHierarchyFoveatedPriority(wideNode, view, foveation, 100),
    'leaves a page reaching into the gaze unrelaxed'
  ).toBe(100);
  expect(
    getSplatHierarchyFoveatedPriority(smallNode, view, foveation, 100) < 100,
    'still relaxes a page wholly outside it'
  ).toBe(true);
  expect(
    getSplatHierarchyFoveatedPriority(aroundCamera, view, foveation, 100),
    'never relaxes a page the camera is inside'
  ).toBe(100);
});

it('SplatHierarchyManager spends a splat budget where the view is looking', async () => {
  const device = new NullDevice({});
  const batches: GPUSplatData[] = [];
  const makeRoot = (id: string, center: readonly [number, number, number]) => {
    const batch = makeSplatHierarchyBatch(device, batches.length, batches.length * 10);
    batches.push(batch);
    return {
      ...makeSplatHierarchyNode(id, center, 1, 0.1),
      data: batch,
      estimatedSplatCount: 1,
      children: [0, 1, 2, 3].map(index => ({
        ...makeSplatHierarchyNode(`${id}-${index}`, center, 0),
        estimatedSplatCount: 10
      }))
    };
  };
  // Named so an unfoveated tie goes to the edge: only foveation can hand the budget to the center.
  const roots = [makeRoot('a-edge', [0.8, 0, 0]), makeRoot('b-center', [0, 0, 0])];
  // Both roots, plus one refinement of 40 - 1 splats. Not two.
  const splatBudget = 2 + 39;

  const getRequestedIds = async (
    props: Partial<ConstructorParameters<typeof SplatHierarchyManager>[0]>,
    view: SplatHierarchyView = makeSplatHierarchyView()
  ): Promise<string[]> => {
    const requested: string[] = [];
    const residencyManager = new SplatResidencyManager();
    const manager = new SplatHierarchyManager({
      roots,
      residencyManager,
      maximumScreenSpaceError: 0.01,
      maxConcurrentLoads: 16,
      splatBudget,
      loadPage: node => {
        requested.push(node.id);
        return new Promise<GPUSplatData>(() => {});
      },
      ...props
    });
    manager.update(view);
    await flushSplatHierarchyMicrotasks();
    manager.destroy();
    residencyManager.destroy();
    return requested.map(id => id.split('-').slice(0, 2).join('-'));
  };

  expect(
    new Set(await getRequestedIds({})),
    'refines the tie-break winner without foveation'
  ).toEqual(new Set(['a-edge']));
  expect(
    new Set(await getRequestedIds({foveation: {radius: 0.05, strength: 8}})),
    'plans the budget against the foveated error'
  ).toEqual(new Set(['b-center']));
  expect(
    await getRequestedIds({}, {...makeSplatHierarchyView(), requestErrorScale: 1e6}),
    'requests no finer pages while the view is coarsened for motion'
  ).toEqual([]);

  for (const batch of batches) {
    batch.destroy();
  }
});

it('SplatHierarchyManager keeps resident children up while a coarsened parent reloads', async () => {
  const device = new NullDevice({});
  const rootBatch = makeSplatHierarchyBatch(device, 0, 0);
  const leafBatches = [
    makeSplatHierarchyBatch(device, 1, 10),
    makeSplatHierarchyBatch(device, 2, 20)
  ];
  const residencyManager = new SplatResidencyManager();
  const leaves = [
    makeSplatHierarchyNode('leaf-0', [-0.2, 0, 0], 0),
    makeSplatHierarchyNode('leaf-1', [0.2, 0, 0], 0)
  ];
  leaves.forEach((leaf, index) =>
    residencyManager.add(leafBatches[index], {id: leaf.id, bounds: leaf.bounds, ownsData: false})
  );
  const requested: string[] = [];
  const manager = new SplatHierarchyManager({
    residencyManager,
    maximumScreenSpaceError: 2,
    roots: [
      {
        ...makeSplatHierarchyNode('root', [0, 0, 0], 100, 0.5),
        data: rootBatch,
        children: [{...makeSplatHierarchyNode('middle', [0, 0, 0], 1, 0.4), children: leaves}]
      }
    ],
    loadPage: node => {
      requested.push(node.id);
      return new Promise<GPUSplatData>(() => {});
    }
  });

  expect(
    manager.update(makeSplatHierarchyView()).map(entry => entry.node.id),
    'draws the resident leaves at full detail'
  ).toEqual(['leaf-0', 'leaf-1']);

  // Pulled back until the middle page is good enough and the leaves are no longer wanted.
  const coarsened = manager.update(makeSplatHierarchyView([0, 0, 400]));
  expect(
    coarsened.map(entry => entry.node.id),
    'keeps the finer resident pages rather than falling back past them'
  ).toEqual(['leaf-0', 'leaf-1']);
  expect(
    coarsened.every(entry => entry.isFallback),
    'marks them as standing in'
  ).toBe(true);
  await flushSplatHierarchyMicrotasks();
  expect(requested, 'still asks for the page the coarser view selects').toContain('middle');

  manager.destroy();
  residencyManager.destroy();
  rootBatch.destroy();
  leafBatches.forEach(batch => batch.destroy());
});

it('SplatHierarchyManager keeps loaded detail while moving and only defers what is missing', async () => {
  const device = new NullDevice({});
  const rootBatch = makeSplatHierarchyBatch(device, 0, 0);
  const loadedBatch = makeSplatHierarchyBatch(device, 1, 10);
  const residencyManager = new SplatResidencyManager();
  const loaded = makeSplatHierarchyNode('loaded', [-0.2, 0, 0], 0);
  const missing = makeSplatHierarchyNode('missing', [0.2, 0, 0], 0);
  const otherRoot = {
    ...makeSplatHierarchyNode('other', [0, 0.5, 0], 1, 0.1),
    data: makeSplatHierarchyBatch(device, 2, 20),
    children: [
      makeSplatHierarchyNode('other-0', [0, 0.4, 0], 0),
      makeSplatHierarchyNode('other-1', [0, 0.6, 0], 0)
    ]
  };
  const leftRoot = {
    ...makeSplatHierarchyNode('left', [0, 0, 0], 1, 0.3),
    data: rootBatch,
    children: [
      {...makeSplatHierarchyNode('left-a', [-0.1, 0, 0], 0.5, 0.15), children: [loaded]},
      {...makeSplatHierarchyNode('left-b', [0.1, 0, 0], 0.5, 0.15), children: [missing]}
    ]
  };
  // Only the left branch's leaves matter here; its middle level is resident up front.
  const middleBatches = [
    makeSplatHierarchyBatch(device, 3, 30),
    makeSplatHierarchyBatch(device, 4, 40)
  ];
  residencyManager.add(middleBatches[0], {id: 'left-a', ownsData: false});
  residencyManager.add(middleBatches[1], {id: 'left-b', ownsData: false});
  residencyManager.add(loadedBatch, {id: 'loaded', ownsData: false});
  const requested: string[] = [];
  const manager = new SplatHierarchyManager({
    residencyManager,
    maximumScreenSpaceError: 1,
    maxConcurrentLoads: 8,
    roots: [leftRoot, otherRoot],
    loadPage: node => {
      requested.push(node.id);
      return new Promise<GPUSplatData>(() => {});
    }
  });
  const moving = {...makeSplatHierarchyView(), requestErrorScale: 1e6};

  const frontier = manager.update(moving).map(entry => entry.node.id);
  await flushSplatHierarchyMicrotasks();
  expect(
    frontier,
    'draws the loaded fine page and stands the resident parent in for the missing one'
  ).toEqual(['loaded', 'left-b', 'other']);
  expect(requested, 'requests nothing finer while moving').toEqual([]);

  manager.update(makeSplatHierarchyView());
  await flushSplatHierarchyMicrotasks();
  expect(new Set(requested), 'requests the missing detail once the camera settles').toEqual(
    new Set(['missing', 'other-0', 'other-1'])
  );

  manager.update(moving);
  await flushSplatHierarchyMicrotasks();
  expect(manager.stats.abortedLoadCount, 'does not abort detail already in flight').toBe(0);
  expect(manager.stats.pendingLoadCount, 'and keeps it loading').toBe(3);

  manager.destroy();
  residencyManager.destroy();
  for (const batch of [rootBatch, loadedBatch, otherRoot.data, ...middleBatches]) {
    batch.destroy();
  }
});

it('SplatHierarchyManager keeps a cancelled page the view asks for again before it settles', async () => {
  const device = new NullDevice({});
  const pageBatch = makeSplatHierarchyBatch(device, 50, 5000);
  const residencyManager = new SplatResidencyManager({ownsData: false});
  const resolvers: ((batch: GPUSplatData) => void)[] = [];
  const manager = new SplatHierarchyManager({
    residencyManager,
    roots: [makeSplatHierarchyNode('page', [0, 0, 0], 1)],
    // Ignores cancellation, as a fetch already past its network phase or a busy worker may.
    loadPage: () => new Promise<GPUSplatData>(resolve => resolvers.push(resolve))
  });
  const lookingAway = {
    ...makeSplatHierarchyView(),
    modelViewProjectionMatrix: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -1.3, 0, 0, 1]
  };

  manager.update(makeSplatHierarchyView());
  await flushSplatHierarchyMicrotasks();
  manager.update(lookingAway);
  expect(manager.stats.abortedLoadCount, 'turning away cancels the page').toBe(1);
  manager.update(makeSplatHierarchyView());

  resolvers[0](pageBatch);
  await flushSplatHierarchyMicrotasks();
  expect(
    manager.frontierBatches,
    'draws the page as soon as it settles, without another update()'
  ).toEqual([pageBatch]);
  expect(resolvers.length, 'and does not fetch it a second time').toBe(1);
  expect(manager.stats.pendingLoadCount, 'leaving nothing in flight').toBe(0);

  // The same cancellation with the view still turned away discards what the loader produced.
  manager.update(lookingAway);
  expect(manager.frontierBatches, 'nothing is drawn while looking away').toEqual([]);

  manager.destroy();
  residencyManager.destroy();
  pageBatch.destroy();
});

it('SplatHierarchyManager re-requests a cancelled page the view wants back once it settles', async () => {
  const device = new NullDevice({});
  const pageBatch = makeSplatHierarchyBatch(device, 51, 5100);
  const residencyManager = new SplatResidencyManager({ownsData: false});
  const resolvers: ((batch: GPUSplatData) => void)[] = [];
  const loadErrors: unknown[] = [];
  const manager = new SplatHierarchyManager({
    residencyManager,
    roots: [makeSplatHierarchyNode('page', [0, 0, 0], 1)],
    // Honours cancellation by rejecting, so nothing is left to keep when it settles.
    loadPage: (_node, {signal}) =>
      new Promise<GPUSplatData>((resolve, reject) => {
        resolvers.push(resolve);
        signal.addEventListener('abort', () => reject(signal.reason));
      }),
    onLoadError: error => loadErrors.push(error)
  });
  const lookingAway = {
    ...makeSplatHierarchyView(),
    modelViewProjectionMatrix: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -1.3, 0, 0, 1]
  };

  manager.update(makeSplatHierarchyView());
  await flushSplatHierarchyMicrotasks();
  // Away and back within one frame: the cancelled load has not settled when the view returns.
  manager.update(lookingAway);
  manager.update(makeSplatHierarchyView());
  await flushSplatHierarchyMicrotasks();

  expect(resolvers.length, 'requests the page again once the cancelled load settles').toBe(2);
  resolvers[1](pageBatch);
  await manager.waitForIdle();
  expect(manager.frontierBatches, 'and draws it without another update()').toEqual([pageBatch]);
  expect(loadErrors, 'a cancellation is not reported as a load error').toEqual([]);

  manager.destroy();
  residencyManager.destroy();
  pageBatch.destroy();
});

it('SplatHierarchyManager fades only children of additive parents in over lodFadeBand', () => {
  const device = new NullDevice({});
  const batches: GPUSplatData[] = [];
  const makeTree = (refinement: 'add' | 'replace'): SplatHierarchyNode => {
    const makeBatch = () => {
      const batch = makeSplatHierarchyBatch(device, batches.length, batches.length * 10);
      batches.push(batch);
      return batch;
    };
    return {
      // 128 px focal length at 3.95 units: about 3 px of error, halfway through a band of 2 to 4.
      ...makeSplatHierarchyNode('root', [0, 0, 0], (3 * 3.95) / 128),
      refinement,
      data: makeBatch(),
      children: [
        {
          ...makeSplatHierarchyNode('sharp', [-0.02, 0, 0], 0),
          data: makeBatch(),
          filterVariance: 0.3
        },
        {
          ...makeSplatHierarchyNode('clamped', [0.02, 0, 0], 0),
          data: makeBatch(),
          filterVariance: -1
        }
      ]
    };
  };
  const getFrontier = (refinement: 'add' | 'replace') => {
    const residencyManager = new SplatResidencyManager({ownsData: false});
    const manager = new SplatHierarchyManager({
      residencyManager,
      roots: [makeTree(refinement)],
      maximumScreenSpaceError: 2,
      lodFadeBand: 1
    });
    const frontier = manager.update(makeSplatHierarchyView()).map(entry => ({
      id: entry.node.id,
      fadeOpacity: entry.fadeOpacity,
      filterVariance: entry.filterVariance
    }));
    manager.destroy();
    residencyManager.destroy();
    return frontier;
  };

  const replaced = getFrontier('replace');
  expect(
    replaced.map(entry => entry.id),
    'replacing children stand in for their parent'
  ).toEqual(['sharp', 'clamped']);
  expect(
    replaced.map(entry => entry.fadeOpacity),
    'and are drawn opaque, since nothing is drawn beneath them'
  ).toEqual([1, 1]);
  expect(
    replaced.map(entry => entry.filterVariance),
    'each carries its own filter variance, never negative'
  ).toEqual([0.3, 0]);

  const added = getFrontier('add');
  expect(
    added.map(entry => entry.id),
    'additive children draw over their parent'
  ).toEqual(['root', 'sharp', 'clamped']);
  expect(added[0].fadeOpacity, 'the root has no parent to fade against').toBe(1);
  expect(added[1].fadeOpacity, 'additive children fade in across the band').toBeCloseTo(0.5, 2);
  expect(added[2].fadeOpacity, 'all of them').toBeCloseTo(0.5, 2);

  for (const batch of batches) {
    batch.destroy();
  }
});

it('SplatHierarchyManager evicts pages from roots replaced by setRoots', async () => {
  const device = new NullDevice({});
  const nearBatch = makeSplatHierarchyBatch(device, 60, 6000);
  const farBatch = makeSplatHierarchyBatch(device, 61, 6100);
  const residencyManager = new SplatResidencyManager({maxResidentChunks: 1, ownsData: false});
  const manager = new SplatHierarchyManager({
    residencyManager,
    maximumScreenSpaceError: 1,
    roots: [makeSplatHierarchyNode('old-near', [0, 0, 0], 1)],
    loadPage: node => (node.id === 'old-near' ? nearBatch : farBatch)
  });

  // Loaded close up, so the old page carries a very large priority.
  manager.update(makeSplatHierarchyView([0, 0, 0.2]));
  await manager.waitForIdle();
  expect(manager.frontierBatches, 'loads the original page').toEqual([nearBatch]);

  // New roots that no longer name the old page, wanted at a far smaller error.
  manager.setRoots([makeSplatHierarchyNode('new-far', [1.3, 0, 0], 1)]);
  const turnedView = {
    ...makeSplatHierarchyView([1.3, 0, 40]),
    modelViewProjectionMatrix: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -1.3, 0, 0, 1]
  };
  manager.update(turnedView);
  await manager.waitForIdle();

  expect(
    manager.frontierBatches,
    'evicts the page the old roots loaded for the one the new roots want'
  ).toEqual([farBatch]);
  expect(manager.stats.rejectedLoadCount, 'without refusing it first').toBe(0);

  manager.destroy();
  residencyManager.destroy();
  nearBatch.destroy();
  farBatch.destroy();
});

it('SplatHierarchyManager counts budgetUpdateInterval in update() calls, not traversals', async () => {
  const device = new NullDevice({});
  const rootBatch = makeSplatHierarchyBatch(device, 70, 7000);
  const childBatches = [0, 1].map(index => makeSplatHierarchyBatch(device, 71 + index, 7100));
  const residencyManager = new SplatResidencyManager({ownsData: false});
  const manager = new SplatHierarchyManager({
    residencyManager,
    maximumScreenSpaceError: 1,
    splatBudget: 100,
    budgetUpdateInterval: 3,
    roots: [
      {
        ...makeSplatHierarchyNode('root', [0, 0, 0], 1, 0.1),
        data: rootBatch,
        estimatedSplatCount: 1,
        children: [0, 1].map(index => ({
          ...makeSplatHierarchyNode(`child-${index}`, [index ? 0.05 : -0.05, 0, 0], 0),
          estimatedSplatCount: 1
        }))
      }
    ],
    loadPage: async node => childBatches[Number(node.id.split('-')[1])]
  });

  manager.update(makeSplatHierarchyView());
  expect(manager.stats.framesSinceBudgetPlan, 'plans on the first update').toBe(0);
  await manager.waitForIdle();
  expect(manager.stats.completedLoadCount, 'both children loaded and re-traversed').toBe(2);
  expect(manager.stats.framesSinceBudgetPlan, 'traversals for settled loads are not frames').toBe(
    0
  );
  expect(manager.frontierBatches, 'and they reuse the plan to select the children').toEqual(
    childBatches
  );

  const framesSincePlan: number[] = [];
  for (let frame = 0; frame < 4; frame++) {
    manager.update(makeSplatHierarchyView());
    framesSincePlan.push(manager.stats.framesSinceBudgetPlan);
  }
  expect(framesSincePlan, 'reuses each plan for two further frames').toEqual([1, 2, 0, 1]);
  expect(manager.stats.budgetedSplatCount, 'reports what the plan spends').toBe(2);

  manager.destroy();
  residencyManager.destroy();
  rootBatch.destroy();
  childBatches.forEach(batch => batch.destroy());
});

it('SplatHierarchyManager treats a requestErrorScale below one or NaN as one', async () => {
  const device = new NullDevice({});
  const rootBatch = makeSplatHierarchyBatch(device, 80, 8000);
  const getRequestedIds = async (requestErrorScale?: number): Promise<string[]> => {
    const requested: string[] = [];
    const residencyManager = new SplatResidencyManager({ownsData: false});
    const manager = new SplatHierarchyManager({
      residencyManager,
      maximumScreenSpaceError: 1,
      roots: [
        {
          ...makeSplatHierarchyNode('root', [0, 0, 0], 1, 0.1),
          data: rootBatch,
          children: [
            makeSplatHierarchyNode('a', [-0.05, 0, 0], 0),
            makeSplatHierarchyNode('b', [0.05, 0, 0], 0)
          ]
        }
      ],
      loadPage: node => {
        requested.push(node.id);
        return new Promise<GPUSplatData>(() => {});
      }
    });
    manager.update({...makeSplatHierarchyView(), requestErrorScale});
    await flushSplatHierarchyMicrotasks();
    manager.destroy();
    residencyManager.destroy();
    return requested.sort();
  };

  expect(await getRequestedIds(), 'requests the children by default').toEqual(['a', 'b']);
  for (const scale of [Number.NaN, 0.25, 0, -3]) {
    expect(await getRequestedIds(scale), `requestErrorScale ${scale} acts as 1`).toEqual([
      'a',
      'b'
    ]);
  }
  expect(
    await getRequestedIds(Number.POSITIVE_INFINITY),
    'an infinite scale requests nothing below the roots'
  ).toEqual([]);

  rootBatch.destroy();
});

it('Splat hierarchy foveation clamps the field of view and respects the viewport aspect', () => {
  const foveation = {center: [0.5, 0.5] as const, radius: 0.05, strength: 8};
  const node = makeSplatHierarchyNode('side', [0.5, 0, 0], 1, 0.2);
  const clampedView = {...makeSplatHierarchyView(), verticalFieldOfView: Math.PI - 1e-6};
  const clamped = getSplatHierarchyFoveatedPriority(node, clampedView, foveation, 100);
  for (const verticalFieldOfView of [2 * Math.PI, 4, Math.PI]) {
    expect(
      getSplatHierarchyFoveatedPriority(
        node,
        {...makeSplatHierarchyView(), verticalFieldOfView},
        foveation,
        100
      ),
      `a field of view of ${verticalFieldOfView} is clamped like screen-space error clamps it`
    ).toBeCloseTo(clamped, 9);
  }
  expect(
    Number.isFinite(
      getSplatHierarchyFoveatedPriority(
        node,
        {...makeSplatHierarchyView(), verticalFieldOfView: 0},
        foveation,
        100
      )
    ),
    'a zero field of view stays finite'
  ).toBe(true);

  // A sphere 0.31 viewport heights across, a quarter of the width to the side of the gaze. On a
  // square viewport it reaches the foveal radius; on one twice as wide it covers half as much of
  // the normalized width, and no longer does.
  const wideSphere = makeSplatHierarchyNode('wide', [0.5, 0, 0], 1, 2.5);
  const tallSphere = makeSplatHierarchyNode('tall', [0, 0.5, 0], 1, 2.5);
  const squareView = makeSplatHierarchyView();
  const wideView = {...makeSplatHierarchyView(), viewportSize: [512, 256] as const};
  expect(
    getSplatHierarchyFoveatedPriority(wideSphere, squareView, foveation, 100),
    'reaches the gaze horizontally on a square viewport'
  ).toBe(100);
  expect(
    getSplatHierarchyFoveatedPriority(wideSphere, wideView, foveation, 100) < 100,
    'but not on a wide one, where its normalized horizontal extent halves'
  ).toBe(true);
  expect(
    getSplatHierarchyFoveatedPriority(tallSphere, wideView, foveation, 100),
    'its vertical extent is measured in viewport heights either way'
  ).toBe(100);
});

function makeSplatHierarchyNode(
  id: string,
  center: readonly [number, number, number],
  geometricError: number,
  radius = 0.05
): SplatHierarchyNode {
  return {id, bounds: {center, radius}, geometricError};
}

function makeSplatHierarchyView(
  cameraPosition: readonly [number, number, number] = [0, 0, 4]
): SplatHierarchyView {
  return {
    cameraPosition,
    viewportSize: [256, 256],
    verticalFieldOfView: Math.PI / 2,
    modelViewProjectionMatrix: IDENTITY_MATRIX
  };
}

function makeSplatHierarchyBatch(
  device: NullDevice,
  sourceBatchIndex: number,
  rowIndexBase: number
): GPUSplatData {
  return makeGPUSplatData(device, {
    positions: new Float32Array([0, 0, 0]),
    scales: new Float32Array([0.1, 0.1, 0.1]),
    rotations: new Float32Array([1, 0, 0, 0]),
    colors: new Uint8Array([255, 128, 64, 255]),
    opacities: new Float32Array([1]),
    sourceBatchIndex,
    rowIndexBase
  });
}

async function flushSplatHierarchyMicrotasks(): Promise<void> {
  for (let flushIndex = 0; flushIndex < 12; flushIndex++) {
    await Promise.resolve();
  }
}
