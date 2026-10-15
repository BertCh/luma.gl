// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createHash} from 'node:crypto';
import {readFileSync, statSync} from 'node:fs';
import {resolve} from 'node:path';
import {expect, it, vi} from 'vitest';
import {WgslReflect} from 'wgsl_reflect';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import type {GPUDataFrameQueryParameters} from '@luma.gl/experimental/gpu-dataframe';
import {createNullWebGPUDevice} from '../utils/gpu-contributor-test-utils';
import {
  dataframeFixture,
  getDataFrameBatches,
  decodeDataFramePayload,
  createParityDataFrame,
  createParityDataFrameQuery
} from '../oracles/parity-wp6-dataframe-harness';
import momentsJson from '../oracles/fixtures/parity-wp6-moment-diagnostics.json';
import priorMomentsJson from '../oracles/fixtures/parity-wp6-offset-moments.json';
import {decodeOracleFixture} from '../oracles/oracle-fixture';

const root = resolve(import.meta.dirname, '../../../..');
const fixtureDirectory = resolve(root, 'modules/experimental/test/oracles/fixtures');

it('WP6 round 6 fixtures preserve full OSM input, hashes, versions and all output arrays', () => {
  const names = [
    'parity-wp6-dataframe-queries.json',
    'parity-wp6-moment-diagnostics.json',
    ...dataframeFixture.cases[0].batchFiles!
  ];
  for (const name of names) {
    const path = resolve(fixtureDirectory, name);
    expect(statSync(path).size).toBeLessThan(1500000);
    const fixture = JSON.parse(readFileSync(path, 'utf8'));
    expect(fixture.generator).toBe('parity-wp6-round6.py');
    expect(fixture.versions.pandas).toMatch(/^\d+\./);
    expect(fixture.versions.numpy).toMatch(/^\d+\./);
    for (const [source, hash] of Object.entries(fixture.sources)) {
      expect(
        createHash('sha256')
          .update(readFileSync(resolve(root, source)))
          .digest('hex')
      ).toBe(hash);
    }
  }
  const realBatches = getDataFrameBatches(dataframeFixture.cases[0]);
  expect(realBatches.reduce((total, batch) => total + batch.length, 0)).toBe(400000);
  expect(realBatches.some(batch => batch.length === 0)).toBe(true);
  const originalCategories = readFileSync(
    resolve(root, 'examples/deck/spatial-analysis-showcase/public/data/poopdeck-osm-nyc/kind.bin')
  );
  const originalContributors = readFileSync(
    resolve(
      root,
      'examples/deck/spatial-analysis-showcase/public/data/poopdeck-osm-nyc/contributor.bin'
    )
  );
  const categories = realBatches.flatMap(batch => decodeDataFramePayload(batch.category));
  const values = realBatches.flatMap(batch => decodeDataFramePayload(batch.values));
  expect(categories).toEqual(Array.from(originalCategories));
  expect(values).toEqual(
    Array.from({length: 400000}, (_, index) => originalContributors.readUInt16LE(index * 2) / 16)
  );
  for (const scene of dataframeFixture.cases) {
    const batches = getDataFrameBatches(scene);
    for (const batch of batches) {
      for (const payload of [
        batch.category,
        batch.values,
        batch.categoryValidity,
        batch.valueValidity,
        batch.score
      ]) {
        expect(decodeDataFramePayload(payload)).toHaveLength(batch.length);
      }
      for (const frame of batch.frames) {
        const mask = decodeDataFramePayload(frame.mask);
        const ids = decodeDataFramePayload(frame.ids);
        expect(mask).toHaveLength(batch.length);
        expect(ids).toEqual(
          mask.flatMap((selected, index) => (selected ? [index + batch.sourceOffset] : []))
        );
      }
    }
    for (const [frameIndex, frame] of scene.frames.entries()) {
      const selectedCount = batches.reduce(
        (total, batch) => total + batch.frames[frameIndex].ids.length,
        0
      );
      expect(frame.global['count']).toEqual([selectedCount]);
      expect(frame.groups['count'].reduce((sum, value) => sum + value, 0)).toBe(selectedCount);
      expect(frame.groups['category']).toHaveLength(scene.groupCount);
      expect(frame.histogram).toHaveLength(scene.edges.length - 1);
      expect(frame.equalHistogram).toHaveLength(scene.binCount);
    }
  }
});

it('WP6 moment references use rounded inputs, preserve prior targets, and expose tile M2', () => {
  type Scene = {
    name: string;
    values: number[];
    mask: number[];
    expected: number[];
    tiles: number[][];
  };
  const fixture = decodeOracleFixture<{cases: Scene[]}>(momentsJson);
  const prior = decodeOracleFixture<{cases: Scene[]}>(priorMomentsJson);
  for (const scene of fixture.cases) {
    expect(scene.values.filter(Number.isFinite).every(value => Math.fround(value) === value)).toBe(
      true
    );
    expect(scene.tiles).toHaveLength(Math.ceil(scene.values.length / 2048));
    expect(scene.tiles.reduce((total, tile) => total + tile[0], 0)).toBe(scene.expected[0]);
    const original = prior.cases.find(entry => entry.name === scene.name);
    if (original) {
      expect(scene.expected).toEqual(original.expected);
      expect(scene.values).toEqual(original.values);
    }
    if (scene.name.endsWith('-translated')) {
      const unshifted = fixture.cases.find(
        entry => entry.name === scene.name.replace('-translated', '')
      )!;
      expect(scene.expected[2]).toBeCloseTo(unshifted.expected[2], 7);
    }
  }
  const rounded = fixture.cases.at(-1)!;
  expect(new Set(rounded.values).size).toBe(4);
  // f32 ULP at 2^24 is 2. A quarter-unit source increment is intentionally lost BEFORE
  // reference calculation; the mathematical pre-rounding variance is not a valid GPU target.
  expect(rounded.values.every(value => value % 2 === 0)).toBe(true);
});

it('WP6 query, grouped/global aggregate and histogram graphs compile and reflect WGSL', () => {
  for (const scene of dataframeFixture.cases) {
    const device = createNullWebGPUDevice();
    const frame = createParityDataFrame(device, scene);
    Object.defineProperty(device, 'props', {
      value: {...device.props, _cacheShaders: false, _cachePipelines: false}
    });
    const shaderSpy = vi.spyOn(device, 'createShader');
    // As in getGPUCommandNodeKernelSources, record/parse shaders on the null device.
    // Pipeline stubs are node-only plumbing; no encoding or numerical GPU result is claimed.
    Object.defineProperty(device, 'createComputePipeline', {
      value: (props: {id?: string}) => ({id: props.id ?? '', hash: '', destroy: () => {}})
    });
    try {
      for (const mode of ['query', 'group', 'global', 'histogram', 'equalHistogram']) {
        const graph = new GPUCommandGraph<GPUDataFrameQueryParameters>(device, {
          id: `parity-wp6-${mode}`
        });
        const query = createParityDataFrameQuery(frame);
        const definitions = {
          count: 'count',
          sum: {sum: 'score'},
          minimum: {min: 'score'},
          maximum: {max: 'score'},
          mean: {mean: 'score'}
        } as const;
        const plan =
          mode === 'query'
            ? query.select(['category', 'score'])
            : mode === 'group'
              ? query.groupBy('category').aggregate(definitions)
              : mode === 'global'
                ? query.aggregate(definitions)
                : query.histogram(
                    'score',
                    mode === 'histogram'
                      ? {edges: scene.edges}
                      : {bins: scene.binCount, domain: scene.domain}
                  );
        const compiled = plan.compile(graph);
        try {
          expect(compiled.selectionMask.data.map(chunk => chunk.length)).toEqual(
            getDataFrameBatches(scene).map(batch => batch.length)
          );
        } finally {
          compiled.destroy();
        }
      }
      expect(shaderSpy.mock.calls.length).toBeGreaterThan(0);
      for (const [props] of shaderSpy.mock.calls) {
        expect(() => new WgslReflect(props.source), props.id).not.toThrow();
      }
    } finally {
      shaderSpy.mockRestore();
      frame.destroy();
      device.destroy();
    }
  }
});
