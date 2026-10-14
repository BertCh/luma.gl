// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  GPUCrossfilter,
  type GPUCrossfilterProps,
  type GPUCrossfilterView
} from '@luma.gl/experimental/gpu-crossfilter';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../src/utils/gpu-contributor-utils';
import {readUint32, submitGraph} from '../utils/gpu-contributor-test-utils';
import {
  buildCubeOracle,
  computeExactHistogram,
  readCubeOracle,
  type OracleBinning
} from './histogram-cube-oracle';

/**
 * `skipUnchangedDimensions` and `histogramCubes` against the always-exact path, and an A/B harness
 * at 1e7 rows. Integer-valued data on power-of-two or width-10 bins keeps float32 binning exact.
 */

type Step = {name: string; apply: (filter: GPUCrossfilter) => void; rewrite?: boolean};

type Dashboard = {
  filter: GPUCrossfilter;
  compiled: ReturnType<GPUCommandGraph['compile']>;
  outputs: Record<string, {buffer: Buffer; length: number}>;
  destroy: () => void;
};

const SKIP_ROW_COUNT = 4096;

it('skipUnchangedDimensions matches the exact path after a brush sequence', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;

  const columns = {
    a: createColumn(device, 'a', SKIP_ROW_COUNT, 100, 1, 'float32'),
    b: createColumn(device, 'b', SKIP_ROW_COUNT, 100, 2, 'float32'),
    c: createColumn(device, 'c', SKIP_ROW_COUNT, 100, 3, 'uint32')
  };
  const exact = createDashboard(device, columns, SKIP_ROW_COUNT, {});
  const skipping = createDashboard(device, columns, SKIP_ROW_COUNT, {
    skipUnchangedDimensions: true
  });

  const steps: Step[] = [
    {name: 'initial', apply: () => {}},
    {name: 'brush a', apply: filter => filter.setRange('a', [10, 60])},
    {name: 'brush b', apply: filter => filter.setRange('b', [20, 80])},
    {name: 'move a', apply: filter => filter.setRange('a', [30, 40])},
    {name: 'clear a', apply: filter => filter.clear('a')},
    {name: 'brush c', apply: filter => filter.setRange('c', [5, 25])},
    {
      name: 'rewrite a',
      apply: filter => filter.invalidate('a'),
      rewrite: true
    },
    {name: 'brush a again', apply: filter => filter.setRange('a', [0, 49])},
    {name: 'clear all', apply: filter => filter.clearAll()}
  ];
  for (const step of steps) {
    if (step.rewrite) {
      // New source values; only the skipping controller needs to be told.
      columns.a.buffer.write(createValues(SKIP_ROW_COUNT, 100, 7, 'float32'));
    }
    step.apply(exact.filter);
    step.apply(skipping.filter);
    submitGraph(device, exact.compiled, undefined);
    submitGraph(device, skipping.compiled, undefined);
    expect(await readOutputs(skipping), step.name).toEqual(await readOutputs(exact));
  }

  // A second encoding with nothing changed must leave every output as it was.
  const before = await readOutputs(skipping);
  submitGraph(device, skipping.compiled, undefined);
  expect(await readOutputs(skipping), 'unchanged re-encode').toEqual(before);

  exact.destroy();
  skipping.destroy();
  for (const column of Object.values(columns)) column.buffer.destroy();
});

it('histogram cubes equal the exact path for edge-aligned brushes', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;

  const rowCount = 3000;
  // a: 8 bins of width 8 on [0, 64); b: 10 bins of width 10 on [0, 100); c: irregular edges.
  const aValues = createValues(rowCount, 64, 11, 'float32') as Float32Array;
  const bValues = createValues(rowCount, 100, 12, 'uint32') as Uint32Array;
  const cValues = Int32Array.from(
    createValues(rowCount, 100, 13, 'uint32') as Uint32Array,
    v => v - 50
  );
  const columns = {
    a: {buffer: createBuffer(device, 'a', aValues), format: 'float32' as const},
    b: {buffer: createBuffer(device, 'b', bValues), format: 'uint32' as const},
    c: {buffer: createBuffer(device, 'c', cValues), format: 'sint32' as const}
  };
  const bins = {
    a: {binCount: 8, domain: [0, 64]},
    b: {binCount: 10, domain: [0, 100]},
    c: {binCount: 5, edges: [-50, -30, -10, 0, 20, 50]}
  } satisfies Record<string, OracleBinning>;
  const options: Partial<GPUCrossfilterProps> = {};
  const exact = createCubeDashboard(device, columns, rowCount, bins, options);
  const cubes = createCubeDashboard(device, columns, rowCount, bins, {histogramCubes: true});

  const run = async (name: string, activeView: string) => {
    submitGraph(device, exact.compiled, undefined);
    submitGraph(device, cubes.compiled, undefined);
    const expected = await readOutputs(exact);
    const actual = await readOutputs(cubes);
    for (const viewId of ['ha', 'hb', 'hc']) {
      // The active view excludes its own brush, so it is exact on both paths.
      expect(actual[viewId], `${name}: ${viewId}`).toEqual(expected[viewId]);
    }
    void activeView;
  };

  cubes.filter.activateView('ha');
  await run('activated', 'ha');

  // Aligned to a's bins: [8 * first, 8 * (last + 1) - 1] on integers.
  for (const [first, last] of [
    [0, 7],
    [0, 0],
    [2, 5],
    [7, 7],
    [3, 4]
  ]) {
    const brush: [number, number] = [8 * first, 8 * (last + 1) - 1];
    exact.filter.setRange('a', brush);
    cubes.filter.setRange('a', brush);
    await run(`a ${brush}`, 'ha');
  }

  // A non-active brush rebuilds only the cubes that read it; results stay exact.
  exact.filter.setRange('b', [10, 69]);
  cubes.filter.setRange('b', [10, 69]);
  await run('b brush, a brushed', 'ha');
  exact.filter.setRange('a', [16, 39]);
  cubes.filter.setRange('a', [16, 39]);
  await run('b brush, a moved', 'ha');

  // Switching the active view rebuilds every cube; aligned to b's bins and c's edges.
  cubes.filter.activateView('hb');
  exact.filter.setRange('b', [20, 59]);
  cubes.filter.setRange('b', [20, 59]);
  exact.filter.clear('a');
  cubes.filter.clear('a');
  exact.filter.setRange('c', [-30, -1]);
  cubes.filter.setRange('c', [-30, -1]);
  await run('hb active', 'hb');

  // Leaving the cubes returns every view to the exact path.
  cubes.filter.deactivateView();
  await run('deactivated', 'hb');

  // The CPU oracle agrees with the exact GPU histograms it replaces.
  const accepted = Array.from({length: rowCount}, () => true);
  const oracleCube = buildCubeOracle(
    Array.from(aValues),
    Array.from(bValues),
    accepted,
    bins.a,
    bins.b
  );
  expect(readCubeOracle(oracleCube, [0, 63], bins.a)).toEqual(
    computeExactHistogram(Array.from(bValues), accepted, bins.b)
  );

  exact.destroy();
  cubes.destroy();
  for (const column of Object.values(columns)) column.buffer.destroy();
});

it('A/B: 1e7 rows, 4 histogram views, brush sweep, exact vs histogram cube', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;

  const rowCount = 10_000_000;
  const binCount = 32;
  if (device.limits.maxStorageBufferBindingSize < rowCount * 4) return;

  const names = ['w', 'x', 'y', 'z'];
  const columns: Record<string, {buffer: Buffer; format: 'float32'}> = {};
  names.forEach((name, index) => {
    columns[name] = {
      buffer: createBuffer(device, name, createValues(rowCount, 1024, 100 + index, 'float32')),
      format: 'float32'
    };
  });
  const bins: Record<string, OracleBinning> = Object.fromEntries(
    names.map(name => [name, {binCount, domain: [0, 1024]} as OracleBinning])
  );

  const timings = new Map<string, number[]>();
  const results = new Map<string, number[][]>();
  for (const mode of ['exact', 'cube'] as const) {
    const dashboard = createCubeDashboard(
      device,
      columns,
      rowCount,
      bins,
      mode === 'cube' ? {histogramCubes: true} : {},
      names
    );
    if (mode === 'cube') dashboard.filter.activateView('hw');
    // Link the other dimensions so the cube also covers a composed mask.
    dashboard.filter.setRange('x', [128, 895]);
    submitGraph(device, dashboard.compiled, undefined);
    await readOutputs(dashboard);

    const modeTimings: number[] = [];
    const modeResults: number[][] = [];
    timings.set(mode, modeTimings);
    results.set(mode, modeResults);
    for (let step = 0; step < binCount; step++) {
      // Slide a 4-bin window across the active dimension: width 32 bins, aligned.
      const first = (step * 32) % 896;
      dashboard.filter.setRange('w', [first, first + 127]);
      const start = performance.now();
      submitGraph(device, dashboard.compiled, undefined);
      const outputs = await readOutputs(dashboard);
      modeTimings.push(performance.now() - start);
      const {hx, hy, hz} = outputs;
      modeResults.push(hx, hy, hz);
    }
    dashboard.destroy();
  }
  const exactTimings = timings.get('exact')!;
  const cubeTimings = timings.get('cube')!;
  expect(results.get('cube')).toEqual(results.get('exact'));

  const median = (values: number[]) => [...values].sort((a, b) => a - b)[values.length >> 1];
  // eslint-disable-next-line no-console
  console.log(
    `[A/B gpu-crossfilter cube] rows=${rowCount} bins=${binCount} views=4 ` +
      `exact median ${median(exactTimings).toFixed(2)} ms, cube median ${median(cubeTimings).toFixed(2)} ms ` +
      `(exact ${exactTimings.map(ms => ms.toFixed(1)).join(',')}; cube ${cubeTimings
        .map(ms => ms.toFixed(1))
        .join(',')})`
  );
  for (const column of Object.values(columns)) column.buffer.destroy();
});

type Column = {buffer: Buffer; format: 'float32' | 'uint32' | 'sint32'};

function createDashboard(
  device: Device,
  columns: Record<'a' | 'b' | 'c', Column>,
  rowCount: number,
  options: Partial<GPUCrossfilterProps>
): Dashboard {
  const graph = new GPUCommandGraph(device, {id: `skip-${options.skipUnchangedDimensions}`});
  const inputs = {
    a: importGraphBuffer(graph, 'a', columns.a.buffer, 'float32', rowCount),
    b: importGraphBuffer(graph, 'b', columns.b.buffer, 'float32', rowCount),
    c: importGraphBuffer(graph, 'c', columns.c.buffer, 'uint32', rowCount)
  };
  const outputs: Dashboard['outputs'] = {};
  const output = (id: string, length: number) => {
    const buffer = createBuffer(device, `${id}-out`, new Uint32Array(length));
    outputs[id] = {buffer, length};
    return importGraphBuffer(graph, `${id}-out`, buffer, 'uint32', length);
  };
  const views: GPUCrossfilterView[] = [
    {
      id: 'ha',
      kind: 'histogram',
      dimension: 'a',
      input: inputs.a,
      domain: [0, 100],
      output: output('ha', 10)
    },
    {
      id: 'hb',
      kind: 'histogram',
      dimension: 'b',
      input: inputs.b,
      domain: 'auto',
      output: output('hb', 10)
    },
    {
      id: 'hc',
      kind: 'histogram',
      dimension: 'c',
      input: inputs.c,
      edges: [0, 10, 30, 60, 90, 100],
      output: output('hc', 5)
    },
    {id: 'groups', kind: 'group', keys: inputs.c, output: output('groups', 100)},
    {id: 'count', kind: 'count', output: output('count', 1)},
    {
      id: 'visible',
      kind: 'visibility',
      output: output('visible-ids', rowCount),
      count: output('visible-count', 1)
    }
  ];
  const filter = new GPUCrossfilter(graph, {
    id: `skip-${options.skipUnchangedDimensions}`,
    dimensions: [
      {id: 'a', kind: 'range', input: inputs.a},
      {id: 'b', kind: 'range', input: inputs.b},
      {id: 'c', kind: 'range', input: inputs.c}
    ],
    views,
    ...options
  });
  filter.addToGraph(graph);
  const compiled = graph.compile();
  return {
    filter,
    compiled,
    outputs,
    destroy: () => {
      compiled.destroy();
      filter.destroy();
      for (const entry of Object.values(outputs)) entry.buffer.destroy();
    }
  };
}

function createCubeDashboard(
  device: Device,
  columns: Record<string, {buffer: Buffer; format: 'float32' | 'uint32' | 'sint32'}>,
  rowCount: number,
  bins: Record<string, OracleBinning>,
  options: Partial<GPUCrossfilterProps>,
  names: string[] = ['a', 'b', 'c']
): Dashboard {
  const tag = options.histogramCubes ? 'cube' : 'exact';
  const graph = new GPUCommandGraph(device, {id: `${tag}-${names.join('')}`});
  const inputs = Object.fromEntries(
    names.map(name => [
      name,
      importGraphBuffer(graph, name, columns[name].buffer, columns[name].format, rowCount)
    ])
  );
  const outputs: Dashboard['outputs'] = {};
  const views = names.map((name): GPUCrossfilterView => {
    const binning = bins[name];
    const id = `h${name}`;
    const buffer = createBuffer(device, `${id}-out`, new Uint32Array(binning.binCount));
    outputs[id] = {buffer, length: binning.binCount};
    const common = {
      id,
      kind: 'histogram' as const,
      dimension: name,
      input: inputs[name],
      output: importGraphBuffer(graph, `${id}-out`, buffer, 'uint32', binning.binCount)
    };
    return (
      binning.edges
        ? {...common, edges: [...binning.edges]}
        : {...common, domain: [...binning.domain!] as [number, number]}
    ) as GPUCrossfilterView;
  });
  const filter = new GPUCrossfilter(graph, {
    id: `${tag}-${names.join('')}`,
    dimensions: names.map(name => ({id: name, kind: 'range' as const, input: inputs[name]})),
    views,
    ...options
  });
  filter.addToGraph(graph);
  const compiled = graph.compile();
  return {
    filter,
    compiled,
    outputs,
    destroy: () => {
      compiled.destroy();
      filter.destroy();
      for (const entry of Object.values(outputs)) entry.buffer.destroy();
    }
  };
}

async function readOutputs(dashboard: Dashboard): Promise<Record<string, number[]>> {
  const result: Record<string, number[]> = {};
  for (const [id, {buffer, length}] of Object.entries(dashboard.outputs)) {
    result[id] = await readUint32(buffer, length);
  }
  return result;
}

function createColumn(
  device: Device,
  id: string,
  rowCount: number,
  range: number,
  seed: number,
  format: 'float32' | 'uint32'
): Column {
  return {buffer: createBuffer(device, id, createValues(rowCount, range, seed, format)), format};
}

/** Deterministic integers in `[0, range)` stored as float32 or uint32. */
function createValues(
  rowCount: number,
  range: number,
  seed: number,
  format: 'float32' | 'uint32'
): Float32Array | Uint32Array {
  const values = format === 'float32' ? new Float32Array(rowCount) : new Uint32Array(rowCount);
  let state = (seed * 2654435761) >>> 0;
  for (let row = 0; row < rowCount; row++) {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    values[row] = (state >>> 8) % range;
  }
  return values;
}

function createBuffer(
  device: Device,
  id: string,
  values: Float32Array | Uint32Array | Int32Array
): Buffer {
  return device.createBuffer({
    id,
    data: values,
    usage: Buffer.STORAGE | Buffer.COPY_DST | Buffer.COPY_SRC
  });
}
