// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {execFileSync} from 'node:child_process';
import {expect, test} from 'vitest';
import {decodeChicagoTracts} from '../../../src/computation-lenses/scenarios/gpu-spatial-analysis/spatial-lens-data';

const path = 'modules/experimental/src/computation-lenses/data/s1-chicago-tracts.json';
const source = readFileSync(path);
const json = JSON.parse(source.toString());

test('Chicago extraction is compact and byte-identical on regeneration', () => {
  expect(source.byteLength).toBeLessThanOrEqual(150_000);
  execFileSync(process.execPath, ['scripts/lens-data/s1-chicago-tracts.mjs']);
  expect(readFileSync(path).equals(source)).toBe(true);
});

test('decodes all 791 tracts with source topology and f32 vertex precision', () => {
  const tracts = decodeChicagoTracts(json);
  expect(tracts.count).toBe(791);
  expect(tracts.dataset.id).toBe('s1-chicago-tracts/1');
  for (const offsets of [tracts.ringOffsets, tracts.polygonRingOffsets]) {
    expect(offsets[0]).toBe(0);
    for (let index = 1; index < offsets.length; index++)
      expect(offsets[index]).toBeGreaterThanOrEqual(offsets[index - 1]);
  }
  expect(tracts.ringOffsets.at(-1)).toBe(tracts.vertices.length / 2);
  expect(tracts.polygonRingOffsets.at(-1)).toBe(tracts.ringOffsets.length - 1);
  // ResultView draws a single ring per feature: verify this dataset needs no multipart flattening.
  expect(Array.from(tracts.partFeature)).toEqual(Array.from({length: 791}, (_, index) => index));
  expect(Array.from(tracts.polygonRingOffsets)).toEqual(
    Array.from({length: 792}, (_, index) => index)
  );
  const original = readFileSync(
    'examples/deck/spatial-analysis-showcase/public/data/chicago-tracts/vertices.bin'
  );
  expect(tracts.vertices.length * 4).toBe(original.byteLength);
  const sharedVertices = new Map<string, string>();
  for (let index = 0; index < tracts.vertices.length; index++) {
    expect(Math.abs(tracts.vertices[index] - original.readFloatLE(index * 4))).toBeLessThanOrEqual(
      1e-5
    );
    if (index % 2) continue;
    const sourceKey = `${original.readUInt32LE(index * 4)},${original.readUInt32LE((index + 1) * 4)}`;
    const decodedKey = `${tracts.vertices[index]},${tracts.vertices[index + 1]}`;
    if (sharedVertices.has(sourceKey)) expect(decodedKey).toBe(sharedVertices.get(sourceKey));
    sharedVertices.set(sourceKey, decodedKey);
  }
  for (const [name, column] of Object.entries(tracts.columns)) {
    expect(column.length, name).toBe(791);
    json.columns[name].forEach((value: number | null, index: number) => {
      expect(column[index]).toBe(value === null ? NaN : Math.fround(value));
    });
  }
});

test('geometry tags produce two 791-polygon panels with unchanged scalar readbacks', async () => {
  const {tagSpatialCaptures} = await import(
    '../../../src/computation-lenses/scenarios/gpu-spatial-analysis/spatial-scenario-utils'
  );
  const {
    createGPUComputationRecord,
    createGPUComputationCaptures,
    validateGPUComputationRecord,
    validateGPUComputationSidecars
  } = await import('@luma.gl/gpgpu');
  const {buildResultViewModel} = await import('../../../src/computation-views/views/result-view');
  const {createViewContext} = await import('../../../src/computation-views/registry/view-registry');
  const tracts = decodeChicagoTracts(json);
  // Synthetic readback fixtures test the metadata contract only; they are never published as a result.
  const inputs = [
    {
      id: 'vertices',
      name: 'vertices',
      elementType: 'f32' as const,
      components: 2,
      values: tracts.vertices,
      provenance: 'gpu-readback' as const
    },
    {
      id: 'ringOffsets',
      name: 'ringOffsets',
      elementType: 'u32' as const,
      values: tracts.ringOffsets,
      provenance: 'gpu-readback' as const
    },
    {
      id: 'values',
      name: 'values',
      elementType: 'f32' as const,
      values: tracts.columns['poverty150Pct'],
      provenance: 'gpu-readback' as const
    },
    {
      id: 'localI',
      name: 'localI',
      elementType: 'f32' as const,
      values: new Float32Array(791),
      provenance: 'gpu-readback' as const
    }
  ];
  const geometry = {verticesCapture: 'vertices', offsetsCapture: 'ringOffsets'};
  const snapshots = tagSpatialCaptures(inputs, {
    vertices: {role: 'intermediate', keySpace: 'vertex'},
    values: {role: 'input', keySpace: 'tract', polygons: geometry},
    localI: {role: 'output', keySpace: 'tract', polygons: geometry}
  });
  const record = createGPUComputationRecord({
    id: 'test-geometry',
    domains: ['gpu-spatial-analysis'],
    adapter: {backend: 'test'},
    graphs: [],
    axes: {
      axes: [{id: 'entity', label: 'tract', kind: 'data', keySpace: 'tract'}],
      keySpaces: [
        {id: 'tract', label: 'tract', size: 791},
        {id: 'vertex', label: 'vertex'}
      ]
    }
  });
  const captures = createGPUComputationCaptures({runId: record.runId, snapshots});
  expect(validateGPUComputationRecord(record)).toMatchObject({valid: true});
  expect(validateGPUComputationSidecars(record, {captures})).toMatchObject({valid: true});
  expect(snapshots.find(capture => capture.name === 'localI')?.values).toBe(inputs[3].values);
  expect(snapshots.find(capture => capture.name === 'localI.geometry')?.values).toBe(
    tracts.vertices
  );
  const model = buildResultViewModel(createViewContext({...record, captures}));
  expect(model.layout).toBe('side-by-side');
  expect(model.panels.map(panel => [panel.role, panel.count])).toEqual([
    ['input', 791],
    ['output', 791]
  ]);
  expect(model.panels.map(panel => panel.values?.name)).toEqual(['values', 'localI']);
  expect(snapshots.find(capture => capture.name === 'localI.geometry')?.keySpace).toBe('vertex');
  expect(model.output.points.length).toBe(tracts.vertices.length / 2);
  expect(model.projection.model).toBe('geographic');
});
