// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync, statSync} from 'node:fs';
import {resolve} from 'node:path';
import {NullDevice} from '@luma.gl/test-utils';
import {validateGPUComputationRecord} from '@luma.gl/gpgpu';
import {expect, test, vi} from 'vitest';
import {getLensScenario} from '../../../src/computation-lenses/registry';
import {
  extractLensExample,
  findNonPublicImports,
  getLensExamplePath
} from '../../../src/computation-lenses/lens-example';
import {getCoreStationInputs} from '../../../src/computation-lenses/scenarios/gpu-core/k-core-data';
import data from '../../../src/computation-lenses/data/k-core.json';
import {scanLensMarkers} from '../../computation-docs/lens-directives';
import '../../../src/computation-lenses/scenarios/gpu-core/index';

const cases = [
  ['gpu-core.scan', 'GPUScan', true],
  ['gpu-core.scan.segmented', 'GPUScan', true],
  ['gpu-core.scan.anatomy', 'GPUScan', true],
  ['gpu-core.flag-offsets', 'GPUFlagOffsets', false],
  ['gpu-core.compaction', 'GPUCompaction', false],
  ['gpu-core.mask', 'GPUMask', false]
] as const;

test.each(
  cases
)('%s registers its element, rungs, and public executable example', (id, element, iterations) => {
  const scenario = getLensScenario(id)!;
  expect(scenario).toMatchObject({
    id,
    element,
    entry: 'gpu-core',
    audience: 'engineer',
    defaultRung: 'result'
  });
  expect(scenario.rungs).toEqual([
    'result',
    'kernel',
    ...(iterations ? ['iterations'] : []),
    'graph',
    'datum',
    'formula',
    'table'
  ]);
  const path = scenario.example?.path ?? getLensExamplePath(id);
  const source = readFileSync(resolve('modules/experimental/src/computation-lenses', path), 'utf8');
  expect(extractLensExample(source).lineCount).toBeLessThanOrEqual(25);
  expect(findNonPublicImports(source)).toEqual([]);
  expect(source).toContain('export async function runExample');
});

test('real-data fixture has aligned finite stations, sparse keys, CSR state starts, and a small DEM', () => {
  const {stations, dem} = data;
  for (const name of [
    'row',
    'lon',
    'lat',
    'prcpMm',
    'prcpPrevDayMm',
    'elevationM',
    'state'
  ] as const) {
    expect(stations[name]).toHaveLength(1024);
    expect(stations[name].every(Number.isFinite)).toBe(true);
  }
  expect(new Set(stations.row).size).toBe(1024);
  expect(Math.max(...stations.row)).toBeGreaterThan(1024);
  expect(stations.states).toEqual([...stations.states].sort());
  expect(stations.stateOffsets).toHaveLength(stations.states.length + 1);
  expect(stations.stateOffsets[0]).toBe(0);
  expect(stations.stateOffsets.at(-1)).toBe(1024);
  stations.stateOffsets.slice(0, -1).forEach((first, state) => {
    expect(
      stations.state.slice(first, stations.stateOffsets[state + 1]).every(value => value === state)
    ).toBe(true);
    const rows = stations.row.slice(first, stations.stateOffsets[state + 1]);
    expect(rows).toEqual([...rows].sort((left, right) => left - right));
  });
  expect(dem.width).toBe(64);
  expect(dem.height).toBe(64);
  expect(dem.elevationM).toHaveLength(4096);
  expect(dem.elevationM.every(value => Number.isFinite(value) && Number.isInteger(value * 4))).toBe(
    true
  );
  expect(dem.cellSizeM).toBe(212.4);
  expect(dem.bounds).toHaveLength(4);
  expect(
    statSync('modules/experimental/src/computation-lenses/data/k-core.json').size
  ).toBeLessThanOrEqual(150000);
  const input = getCoreStationInputs(data);
  expect(input.first).toBeGreaterThan(0);
  expect(
    input.segmentFlags.slice(input.first, input.first + 16).reduce((sum, flag) => sum + flag, 0)
  ).toBeGreaterThanOrEqual(2);
});

test.each(
  cases
)('%s compiles and validates its real graph without execution', async (id, element) => {
  const device = new NullDevice({_cacheShaders: false, _cachePipelines: false});
  Object.defineProperty(device, 'type', {value: 'webgpu'});
  Object.defineProperty(device, 'limits', {
    value: {...device.limits, maxComputeWorkgroupsPerDimension: 65535}
  });
  // NullDevice cannot create compute pipelines: keep real graph compilation, stub only the backend.
  const pipeline = vi
    .spyOn(device, 'createComputePipeline')
    .mockImplementation(props => ({id: props?.id, hash: '', destroy() {}}) as never);
  const submit = vi.spyOn(device, 'submit');
  try {
    const scenario = await getLensScenario(id)!.load();
    const record = await scenario.describeStructure!(device);
    expect(validateGPUComputationRecord(record)).toMatchObject({valid: true});
    expect(record.graphs[0].nodes.some(node => node.op === element)).toBe(true);
    expect(record.graphs[0].nodes.every(node => node.datum?.keySpace === 'stations')).toBe(true);
    expect(record.graphs[0].nodes.every(node => node.kind === 'compute')).toBe(true);
    expect(record.refs.sources.length).toBeGreaterThan(0);
    expect(pipeline).toHaveBeenCalled();
    expect(submit).not.toHaveBeenCalled();
    expect(record).not.toHaveProperty('captures');
    expect(JSON.stringify(record).length).toBeLessThan(60 * 1024);
    if (id === 'gpu-core.scan.anatomy') {
      const resources = record.graphs[0].resources!;
      expect(
        resources.some(
          resource => resource.id === 'heavyRainHighStations' && resource.ownership === 'transient'
        )
      ).toBe(true);
      expect(resources.every(resource => !resource.retained)).toBe(true);
      expect(
        record.graphs[0].nodes.some(
          node => node.op === 'GPUScan' && node.namespace?.includes('gpu-compaction')
        )
      ).toBe(true);
      expect(new Set(record.graphs[0].nodes.map(node => node.op))).toEqual(
        new Set(['GPUMask', 'GPUScan', 'GPUCompaction', 'GPUReduction'])
      );
    }
  } finally {
    device.destroy();
  }
});

test('K-1 documentation cards use registered rungs and bounded captions', () => {
  for (const [page, element] of [
    ['scan', 'GPUScan'],
    ['segmented-scan', undefined],
    ['flag-offsets', 'GPUFlagOffsets'],
    ['compaction', 'GPUCompaction'],
    ['mask', 'GPUMask']
  ]) {
    const path = `docs/api-reference/experimental/gpu-core/gpu-${page}.md`;
    const source = readFileSync(path, 'utf8');
    const markers = scanLensMarkers(path, source);
    const directives = [
      ...source.matchAll(/::lens\{scenario=([^ ]+) rung=([^ ]+) caption="([^"]+)"\}/g)
    ];
    expect(directives[0][2]).toBe('result');
    for (const [, id, rung, caption] of directives) {
      expect(getLensScenario(id)?.rungs).toContain(rung);
      expect(caption.split(/\s+/).length).toBeLessThanOrEqual(18);
    }
    if (element) {
      expect(markers.map(marker => marker.name)).toEqual([element]);
      expect(source).toContain('#### How it computes');
    } else {
      expect(directives.map(directive => directive[2])).toEqual(['result', 'kernel']);
      expect(markers).toEqual([]);
    }
  }
});

const sharedRungs = ['result', 'kernel', 'graph', 'datum', 'formula', 'table'];
const sortRungs = ['result', 'kernel', 'graph', 'datum', 'table'];
const k3Cases = [
  ['gpu-core.reduction', 'GPUReduction', 'reduction', sharedRungs],
  ['gpu-core.segmented-reduction', 'GPUSegmentedReduction', 'segmented-reduction', sharedRungs],
  ['gpu-core.sort', 'GPUSort', 'sort', sortRungs],
  ['gpu-core.segmented-sort', 'GPUSegmentedSort', 'segmented-sort', sortRungs],
  ['gpu-core.gather', 'GPUGather', 'gather', sharedRungs],
  ['gpu-core.histogram', 'GPUHistogram', 'histogram', sharedRungs],
  ['gpu-core.elementwise', 'GPUElementwise', 'elementwise', sharedRungs],
  ['gpu-core.elementwise.id-probe', 'GPUElementwise', undefined, ['result', 'kernel', 'graph']]
] as const;

test.each(
  k3Cases
)('%s registers its element, rungs, and public executable example', (id, element, _page, rungs) => {
  const scenario = getLensScenario(id)!;
  expect(scenario).toMatchObject({
    id,
    element,
    entry: 'gpu-core',
    audience: 'engineer',
    defaultRung: 'result'
  });
  expect(scenario.rungs).toEqual(rungs);
  const path = scenario.example?.path ?? getLensExamplePath(id.replace('.id-probe', ''));
  const source = readFileSync(resolve('modules/experimental/src/computation-lenses', path), 'utf8');
  expect(extractLensExample(source).lineCount).toBeLessThanOrEqual(25);
  expect(findNonPublicImports(source)).toEqual([]);
  expect(source).toContain('export async function runExample');
});

test.each(
  k3Cases
)('%s compiles and validates its real graph without execution', async (id, element) => {
  const device = new NullDevice({_cacheShaders: false, _cachePipelines: false});
  Object.defineProperty(device, 'type', {value: 'webgpu'});
  Object.defineProperty(device, 'limits', {
    value: {
      ...device.limits,
      maxComputeWorkgroupsPerDimension: 65535,
      maxStorageBufferBindingSize: 2 ** 30
    }
  });
  vi.spyOn(device, 'createComputePipeline').mockImplementation(
    props => ({id: props?.id, hash: '', destroy() {}}) as never
  );
  const submit = vi.spyOn(device, 'submit');
  try {
    const scenario = await getLensScenario(id)!.load();
    const record = await scenario.describeStructure!(device);
    expect(validateGPUComputationRecord(record)).toMatchObject({valid: true});
    expect(record.graphs[0].nodes.some(node => node.op === element)).toBe(true);
    expect(record.graphs[0].nodes.every(node => node.kind === 'compute')).toBe(true);
    expect(submit).not.toHaveBeenCalled();
    expect(record).not.toHaveProperty('captures');
  } finally {
    device.destroy();
  }
});

test('K-3 documentation cards use registered rungs and bounded captions', () => {
  for (const [, element, page] of k3Cases) {
    if (!page) continue;
    const path = `docs/api-reference/experimental/gpu-core/gpu-${page}.md`;
    const source = readFileSync(path, 'utf8');
    const markers = scanLensMarkers(path, source);
    const directives = [
      ...source.matchAll(/::lens\{scenario=([^ ]+) rung=([^ ]+) caption="([^"]+)"\}/g)
    ];
    expect(directives[0][2]).toBe('result');
    for (const [, id, rung, caption] of directives) {
      expect(getLensScenario(id)?.rungs).toContain(rung);
      expect(caption.split(/\s+/).length).toBeLessThanOrEqual(18);
    }
    expect(markers.map(marker => marker.name)).toEqual([element]);
    expect(source).toContain('#### How it computes');
  }
});

const k4Cases = [
  ['gpu-core.grid-index', 'GPUGridIndex', 'grid-index', sortRungs],
  ['gpu-core.grid-binning', 'GPUGridBinning', 'grid-binning', sharedRungs],
  ['gpu-core.grid-aggregation', 'GPUGridAggregation', 'grid-aggregation', sharedRungs],
  ['gpu-core.group-aggregation', 'GPUGroupAggregation', 'group-aggregation', sharedRungs],
  ['gpu-core.hash-index', 'GPUHashIndex', 'hash-index', sortRungs],
  ['gpu-core.hash-join', 'GPUHashJoin', 'hash-join', sortRungs]
] as const;

test.each(
  k4Cases
)('%s registers its element, rungs, and public executable example', (id, element, _page, rungs) => {
  const scenario = getLensScenario(id)!;
  expect(scenario).toMatchObject({
    id,
    element,
    entry: 'gpu-core',
    audience: 'engineer',
    defaultRung: 'result'
  });
  expect(scenario.rungs).toEqual(rungs);
  const source = readFileSync(
    resolve('modules/experimental/src/computation-lenses', getLensExamplePath(id)),
    'utf8'
  );
  expect(extractLensExample(source).lineCount).toBeLessThanOrEqual(25);
  expect(findNonPublicImports(source)).toEqual([]);
  expect(source).toContain('export async function runExample');
});

test.each(
  k4Cases
)('%s compiles and validates its real graph without execution', async (id, element, _page, rungs) => {
  const device = new NullDevice({_cacheShaders: false, _cachePipelines: false});
  Object.defineProperty(device, 'type', {value: 'webgpu'});
  Object.defineProperty(device, 'limits', {
    value: {
      ...device.limits,
      maxComputeWorkgroupsPerDimension: 65535,
      maxStorageBufferBindingSize: 2 ** 30
    }
  });
  vi.spyOn(device, 'createComputePipeline').mockImplementation(
    props => ({id: props?.id, hash: '', destroy() {}}) as never
  );
  const submit = vi.spyOn(device, 'submit');
  try {
    const scenario = await getLensScenario(id)!.load();
    const record = await scenario.describeStructure!(device);
    expect(validateGPUComputationRecord(record)).toMatchObject({valid: true});
    expect(record.graphs[0].nodes.some(node => node.op === element)).toBe(true);
    expect(record.graphs[0].nodes.every(node => node.kind === 'compute')).toBe(true);
    expect(record.graphs[0].nodes.every(node => node.datum?.keySpace === 'stations')).toBe(true);
    expect(submit).not.toHaveBeenCalled();
    expect(record).not.toHaveProperty('captures');
    // Scenarios with a curated formula reference it from their instrumented nodes.
    const hasFormula = (rungs as readonly string[]).includes('formula');
    expect(record.graphs[0].nodes.some(node => node.formula === id)).toBe(hasFormula);
  } finally {
    device.destroy();
  }
});

test('K-4 documentation cards use registered rungs and bounded captions', () => {
  for (const [, element, page] of k4Cases) {
    const path = `docs/api-reference/experimental/gpu-core/gpu-${page}.md`;
    const source = readFileSync(path, 'utf8');
    const markers = scanLensMarkers(path, source);
    const directives = [
      ...source.matchAll(/::lens\{scenario=([^ ]+) rung=([^ ]+) caption="([^"]+)"\}/g)
    ];
    expect(directives[0][2]).toBe('result');
    for (const [, id, rung, caption] of directives) {
      expect(getLensScenario(id)?.rungs).toContain(rung);
      expect(caption.split(/\s+/).length).toBeLessThanOrEqual(18);
    }
    expect(markers.map(marker => marker.name)).toEqual([element]);
    expect(source).toContain('#### How it computes');
    // Formula-free elements explain themselves with the graph rung.
    if (element === 'GPUGridIndex' || element === 'GPUHashIndex' || element === 'GPUHashJoin')
      expect(source).toContain('rung=graph');
  }
});

const k5Rungs = ['result', 'kernel', 'graph', 'datum', 'formula', 'table'];
const k5Cases = [
  ['gpu-core.mat-vec', 'GPUMatVec', 'matvec', k5Rungs],
  ['gpu-core.mat-mul', 'GPUMatMul', 'matmul', k5Rungs],
  ['gpu-core.convolution', 'GPUConvolution', 'convolution', k5Rungs],
  ['gpu-core.fft1d', 'GPUFFT1D', 'fft1d', k5Rungs],
  ['gpu-core.fft2d', 'GPUFFT2D', 'fft2d', k5Rungs],
  [
    'gpu-core.conjugate-gradient-program',
    'createGPUConjugateGradientProgram',
    'preconditioned-conjugate-gradient',
    ['result', 'kernel', 'graph', 'datum', 'iterations', 'formula', 'table']
  ]
] as const;

test.each(
  k5Cases
)('%s registers its element, rungs, and public executable example', (id, element, _page, rungs) => {
  const scenario = getLensScenario(id)!;
  expect(scenario).toMatchObject({
    id,
    element,
    entry: 'gpu-core',
    audience: 'engineer',
    defaultRung: 'result'
  });
  expect(scenario.rungs).toEqual(rungs);
  const source = readFileSync(
    resolve('modules/experimental/src/computation-lenses', getLensExamplePath(id)),
    'utf8'
  );
  expect(extractLensExample(source).lineCount).toBeLessThanOrEqual(25);
  expect(findNonPublicImports(source)).toEqual([]);
  expect(source).toContain('export async function runExample');
});

test.each(
  k5Cases
)('%s compiles and validates its real graph without execution', async (id, element) => {
  const device = new NullDevice({_cacheShaders: false, _cachePipelines: false});
  Object.defineProperty(device, 'type', {value: 'webgpu'});
  Object.defineProperty(device, 'limits', {
    value: {
      ...device.limits,
      maxComputeWorkgroupsPerDimension: 65535,
      maxComputeInvocationsPerWorkgroup: 256,
      maxComputeWorkgroupSizeX: 256,
      maxComputeWorkgroupSizeY: 256,
      maxComputeWorkgroupStorageSize: 32768,
      maxStorageBuffersPerShaderStage: 8,
      maxStorageBufferBindingSize: 2 ** 30
    }
  });
  vi.spyOn(device, 'createComputePipeline').mockImplementation(
    props => ({id: props?.id, hash: '', destroy() {}}) as never
  );
  const submit = vi.spyOn(device, 'submit');
  try {
    const scenario = await getLensScenario(id)!.load();
    const record = await scenario.describeStructure!(device);
    expect(validateGPUComputationRecord(record)).toMatchObject({valid: true});
    expect(submit).not.toHaveBeenCalled();
    expect(record).not.toHaveProperty('captures');
    // Every scenario references its curated formula from the record.
    // The program scenario is named after its factory; its formula keeps the shorter id.
    const formulaId =
      id === 'gpu-core.conjugate-gradient-program' ? 'gpu-core.conjugate-gradient' : id;
    expect(record.refs.formulas.some(formula => formula.id === formulaId)).toBe(true);
    if (id === 'gpu-core.conjugate-gradient-program') {
      // Both graph forms are recorded; compiled nodes link back to the intent graph by derivedFrom.
      expect(record.graphs.map(graph => graph.form)).toEqual(['intent', 'compiled']);
      expect(
        record.graphs[1].nodes.some(node =>
          node.derivedFrom?.some(reference => reference.startsWith(`${record.graphs[0].id}:`))
        )
      ).toBe(true);
      expect(record.graphs[0].nodes.some(node => node.formula === formulaId)).toBe(true);
      expect(record.graphs[1].nodes.some(node => node.loop)).toBe(true);
    } else {
      expect(record.graphs[0].nodes.some(node => node.op === element)).toBe(true);
      expect(record.graphs[0].nodes.every(node => node.kind === 'compute')).toBe(true);
      expect(record.graphs[0].nodes.some(node => node.formula === id)).toBe(true);
    }
    if (id === 'gpu-core.mat-vec' || id === 'gpu-core.mat-mul') {
      expect(record.graphs[0].nodes.every(node => node.datum !== undefined)).toBe(true);
    }
    expect(JSON.stringify(record).length).toBeLessThan(512 * 1024);
  } finally {
    device.destroy();
  }
});

test('K-5 documentation cards use registered rungs and bounded captions', () => {
  for (const [, element, page] of k5Cases) {
    const path = `docs/api-reference/experimental/gpu-core/gpu-${page}.md`;
    const source = readFileSync(path, 'utf8');
    const markers = scanLensMarkers(path, source);
    const directives = [
      ...source.matchAll(/::lens\{scenario=([^ ]+) rung=([^ ]+) caption="([^"]+)"\}/g)
    ];
    expect(directives[0][2]).toBe('result');
    for (const [, id, rung, caption] of directives) {
      expect(getLensScenario(id)?.rungs).toContain(rung);
      expect(caption.split(/\s+/).length).toBeLessThanOrEqual(18);
    }
    expect(markers.map(marker => marker.name)).toEqual([element]);
    expect(source).toContain('#### How it computes');
  }
});
