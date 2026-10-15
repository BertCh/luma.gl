// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {inflateSync} from 'node:zlib';
import {
  createGPUComputationRecord,
  createGPUComputationCaptures,
  createGPUComputationMetrics,
  type GPUComputationCapture
} from '@luma.gl/gpgpu';
import {toRecordedLensFile} from '../../../src/computation-lenses/registry';
import {createViewContext} from '../../../src/computation-views/registry/view-registry';
import {buildLensTwin, validateLensTwin} from '../../../src/computation-views/twin/static-twin';
import {NETWORK_WAVEFRONT_VIEW} from '../../../src/gpu-network/computation-pack/index';
import {
  buildNetworkWavefrontModel,
  renderNetworkWavefront
} from '../../../src/gpu-network/computation-pack/views/network-wavefront-view';
import {findViewNodes} from '../../../src/computation-views/view-node';

function makeRecord() {
  const record = createGPUComputationRecord({
    id: 'gpu-network.fixture',
    adapter: {backend: 'webgpu', description: 'Fixture GPU'},
    domains: ['gpu-network'],
    createdAt: '2026-10-14T00:00:00Z',
    graphs: [
      {
        id: 'network',
        domain: 'gpu-network',
        form: 'compiled',
        edges: [],
        nodes: [
          {
            id: 'relax',
            label: 'relax',
            namespace: '',
            kind: 'compute',
            reads: [],
            writes: [],
            attrs: {chaotic: true},
            loop: {id: 'relax', iteration: 0, maximumIterations: 4}
          }
        ]
      }
    ],
    axes: {
      keySpaces: [{id: 'node', label: 'node', size: 3}],
      axes: [{id: 'iteration', label: 'round', kind: 'execution'}]
    }
  });
  const snapshots: GPUComputationCapture[] = [
    {
      id: 'layout',
      name: 'nodePositions',
      values: [-87.64, 41.88, -87.63, 41.88, -87.63, 41.89],
      elementType: 'f32',
      components: 2,
      provenance: 'gpu-readback'
    },
    {
      id: 'offsets',
      name: 'offsets',
      values: [0, 1, 2, 2],
      elementType: 'u32',
      provenance: 'gpu-readback'
    },
    {
      id: 'targets',
      name: 'neighbors',
      values: [1, 2],
      elementType: 'u32',
      provenance: 'gpu-readback'
    },
    {
      id: 'roads',
      name: 'roads',
      values: [],
      elementType: 'f32',
      provenance: 'gpu-readback',
      role: 'context',
      geometry: {
        kind: 'graph',
        offsetsCapture: 'offsets',
        targetsCapture: 'targets',
        layoutCapture: 'layout',
        coordinateModel: 'geographic'
      }
    },
    ...[
      [0, Infinity, Infinity],
      [0, 2, Infinity],
      [0, 2, 4]
    ].map(
      (values, iteration): GPUComputationCapture => ({
        id: `costs-${iteration}`,
        name: 'costs',
        keySpace: 'node',
        iteration,
        values,
        elementType: 'f32',
        provenance: 'gpu-readback'
      })
    )
  ];
  return {
    ...record,
    captures: createGPUComputationCaptures({
      runId: record.runId,
      snapshots,
      replays: [
        {
          id: 'replay',
          name: 'costs',
          keySpace: 'node',
          iteration: 1,
          values: [0, 2, Infinity],
          elementType: 'f32',
          provenance: 'gpu-replay',
          meaning: 'replay: activeIterations = 1'
        }
      ]
    }),
    metrics: [
      createGPUComputationMetrics({
        runId: record.runId,
        graph: 'network',
        timing: {nodes: []},
        encoding: {coalesced: false, timestamped: false, taps: 0, lensTwin: true},
        series: [{name: 'frontier-size', axis: 'iteration', values: [1, 201, 0], unit: 'nodes'}]
      })
    ]
  };
}

test('network iterations twin has recorded frames, final default, play, strip and replay hatching', () => {
  const record = makeRecord();
  const file = toRecordedLensFile(record, {
    scenarioId: 'gpu-network.fixture',
    recordedAt: record.createdAt,
    adapterLabel: 'Fixture GPU'
  });
  const directive = {scenario: file.scenarioId, rung: 'iterations'};
  const twin = buildLensTwin(file, directive);
  expect(twin.viewId).toBe(NETWORK_WAVEFRONT_VIEW.id);
  expect(validateLensTwin(twin)).toEqual([]);
  expect(twin.figure.html).toBe(buildLensTwin(file, directive).figure.html);
  expect(twin.figure.html.match(/class="cv-frame"/g)).toHaveLength(3);
  expect(twin.figure.html.match(/ hidden/g)).toHaveLength(2);
  expect(twin.figure.html).toContain('data-cv-play');
  expect(twin.figure.html).toContain('cv-key-frames');
  expect(twin.figure.html).toContain('replay · round 1');
  expect(twin.figure.html).toContain('cv-replay-hatch');
  expect(twin.figure.html).toContain('data-scale="log1p"');
  expect(twin.figure.html).toContain('Each frame is a valid execution prefix on Fixture GPU');
  expect(buildNetworkWavefrontModel(createViewContext(record)).arrivals).toEqual([0, 1, 2]);
});

test('wavefront names missing layout or snapshots instead of inventing evidence', () => {
  const record = makeRecord();
  record.captures.snapshots = record.captures.snapshots.filter(capture => capture.id !== 'layout');
  expect(NETWORK_WAVEFRONT_VIEW.satisfies(createViewContext(record))).toEqual({
    ok: false,
    reason: 'needs a context graph with a captured node layout'
  });
  const noSnapshots = makeRecord();
  noSnapshots.captures.snapshots = noSnapshots.captures.snapshots.filter(
    capture => capture.iteration === undefined
  );
  expect(NETWORK_WAVEFRONT_VIEW.satisfies(createViewContext(noSnapshots))).toEqual({
    ok: false,
    reason: 'needs at least two iteration snapshots of one capture on key space node'
  });
});

test('even sampling limits large traces and always includes the final capture', () => {
  const record = makeRecord();
  record.captures.snapshots = [
    ...record.captures.snapshots.filter(capture => capture.iteration === undefined),
    ...Array.from(
      {length: 30},
      (_, iteration): GPUComputationCapture => ({
        id: `round-${iteration}`,
        name: 'costs',
        keySpace: 'node',
        iteration,
        values: [0, 1, iteration > 0 ? 2 : Infinity],
        elementType: 'f32',
        provenance: 'gpu-readback'
      })
    )
  ];
  const model = buildNetworkWavefrontModel(createViewContext(record));
  expect(model.frames).toHaveLength(8);
  expect(model.frames[0].iteration).toBe(0);
  expect(model.frames.at(-1)?.iteration).toBe(29);
});

test('end-only chained evidence falls back to RealizedView without fabricated frames', () => {
  const record = makeRecord();
  record.captures.snapshots = record.captures.snapshots.filter(
    capture => capture.iteration === undefined
  );
  record.captures.replays = [];
  record.captures.stamps = [
    {
      id: 'lastQueuedRound',
      name: 'lastQueuedRound',
      keySpace: 'node',
      values: [0, 1, NaN],
      elementType: 'f32',
      provenance: 'gpu-readback',
      meaning: 'last global-queue round (chained hops inside a round are not stamped)'
    }
  ];
  const file = toRecordedLensFile(record, {
    scenarioId: 'gpu-network.fixture',
    recordedAt: record.createdAt,
    adapterLabel: 'Fixture GPU'
  });
  const twin = buildLensTwin(file, {scenario: file.scenarioId, rung: 'iterations'});
  expect(twin.viewId).toBe('realized-view');
  expect(validateLensTwin(twin)).toEqual([]);
  expect(twin.figure.html).not.toContain('class="cv-frame"');
});

function makeLargeRecord() {
  const record = makeRecord();
  const nodeCount = 2048;
  const captures = record.captures.snapshots;
  captures.find(capture => capture.id === 'layout')!.values = Array.from(
    {length: nodeCount},
    (_, node) => [node % 64, Math.floor(node / 64)]
  ).flat();
  captures.find(capture => capture.id === 'offsets')!.values = Array.from(
    {length: nodeCount + 1},
    (_, node) => node
  );
  captures.find(capture => capture.id === 'targets')!.values = Array.from(
    {length: nodeCount},
    (_, node) => (node + 1) % nodeCount
  );
  const roadGeometry = captures.find(capture => capture.id === 'roads')!.geometry!;
  roadGeometry.coordinateModel = 'projected';
  record.captures.snapshots = [
    ...captures.filter(capture => capture.iteration === undefined),
    ...Array.from(
      {length: 30},
      (_, iteration): GPUComputationCapture => ({
        id: `round-${iteration}`,
        name: 'costs',
        keySpace: 'node',
        iteration,
        values: Array.from({length: nodeCount}, (_, node) =>
          node % 30 <= iteration ? node : Infinity
        ),
        elementType: 'f32',
        provenance: 'gpu-readback'
      })
    )
  ];
  record.captures.replays = Array.from({length: 30}, (_, iteration) => ({
    ...record.captures.replays[0],
    id: `replay-${iteration}`,
    iteration,
    values: Array.from({length: nodeCount}, (_, node) => (node % 30 <= iteration ? node : Infinity))
  }));
  record.metrics[0].series![0].values = Array.from({length: 10000}, (_, round) => round % 100);
  return record;
}

test('dense recorded wavefronts stay within full and thumbnail budgets without changing evidence', () => {
  const record = makeLargeRecord();
  const evidence = JSON.stringify(record);
  const file = toRecordedLensFile(record, {
    scenarioId: 'gpu-network.fixture',
    recordedAt: record.createdAt,
    adapterLabel: 'Fixture GPU'
  });
  const directive = {scenario: file.scenarioId, rung: 'iterations', datum: '420'};
  const twin = buildLensTwin(file, directive);
  const thumbnail = buildLensTwin(file, directive, {size: 'thumbnail'});
  expect(Buffer.byteLength(JSON.stringify(twin, null, 2))).toBeLessThanOrEqual(256 * 1024);
  expect(Buffer.byteLength(JSON.stringify(thumbnail, null, 2))).toBeLessThanOrEqual(48 * 1024);
  expect(validateLensTwin(twin)).toEqual([]);
  expect(validateLensTwin(thumbnail)).toEqual([]);
  expect(twin.figure.html.match(/class="cv-frame"/g)).toHaveLength(8);
  expect(twin.figure.html.match(/data:image\/png;base64,/g)).toHaveLength(14);
  expect(twin.figure.html).toContain('data-key="420"');
  expect(twin.figure.html).toContain('8 of 30 recorded frames');
  expect(twin.figure.html).toContain('All 2048 nodes and 2048 edges rasterized');
  expect(twin.figure.html).toContain('Frontier sparkline decimated to 256');
  expect(twin.figure.html).toContain('data-iteration="29"');
  expect(twin.keyNumber.value).toBe('2,048');
  expect(thumbnail.figure.html.match(/data:image\/png;base64,/g)).toHaveLength(1);
  expect(thumbnail.figure.html).toContain('data-iteration="29"');
  expect(thumbnail.figure.html).toContain('Final recorded frame only');
  expect(thumbnail.figure.html).not.toContain('data-cv-play');
  expect(thumbnail.figure.html).not.toContain('cv-key-frames');
  expect(JSON.stringify(record)).toBe(evidence);
});

test('raster frames retain recorded colours, native layout and replay hatching on both grounds', () => {
  const record = makeLargeRecord();
  for (const ground of ['light', 'dark'] as const) {
    const context = createViewContext(record, {ground});
    const model = buildNetworkWavefrontModel(context);
    const tree = renderNetworkWavefront(model, context);
    const images = findViewNodes(tree, node => node.tag === 'image');
    for (const image of images) {
      const bytes = Buffer.from(String(image.attributes['href']).split(',')[1], 'base64');
      expect([...bytes.subarray(0, 8)]).toEqual([137, 80, 78, 71, 13, 10, 26, 10]);
      const width = bytes.readUInt32BE(16);
      const height = bytes.readUInt32BE(20);
      expect(width).toBe(image.attributes['data-raster-width']);
      expect(height).toBe(image.attributes['data-raster-height']);
      let offset = 8;
      const payload: Buffer[] = [];
      while (offset < bytes.length) {
        const length = bytes.readUInt32BE(offset);
        if (bytes.toString('ascii', offset + 4, offset + 8) === 'IDAT') {
          payload.push(bytes.subarray(offset + 8, offset + 8 + length));
        }
        offset += 12 + length;
      }
      const scanlines = inflateSync(Buffer.concat(payload));
      expect(scanlines.length).toBe((width + 1) * height);
      expect(scanlines.includes(3)).toBe(true); // Earliest reached nodes retain their time class.
      expect(scanlines.includes(8)).toBe(image.attributes['data-provenance'] === 'replay');
      const [horizontal, vertical] = model.projection.project(...model.points[0]);
      const column = Math.round((horizontal * width) / 640);
      const row = Math.round((vertical * height) / 360);
      expect([3, 8]).toContain(scanlines[row * (width + 1) + column + 1]);
    }
  }
});
