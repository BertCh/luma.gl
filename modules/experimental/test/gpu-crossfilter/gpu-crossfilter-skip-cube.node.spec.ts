// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {GPUCommandGraph, GraphVectorView, type GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {
  GPUCrossfilter,
  type GPUCrossfilterMask,
  type GPUCrossfilterProps,
  type GPUCrossfilterScalarFormat,
  type GPUCrossfilterScalarInput,
  type GPUCrossfilterView
} from '@luma.gl/experimental/gpu-crossfilter';
import {NullDevice} from '@luma.gl/test-utils';
import {WgslReflect} from 'wgsl_reflect';
import {describe, expect, test, vi} from 'vitest';
import {auditWGSLModule} from '../analysis-core/kernel-audit/kernel-audit-wgsl';
import {
  countDistinctPipelines,
  getGPUCommandNodeKernelSources
} from '../utils/gpu-contributor-test-utils';
import {
  buildCubeOracle,
  computeExactHistogram,
  readCubeOracle,
  snapBrush,
  type OracleBinning
} from './histogram-cube-oracle';

type Pass = {
  id: string;
  condition?: {id: string; source: string; evaluate: (parameters: unknown) => boolean};
  workload?: {operation?: string; variant?: string};
  compile?: (context: {device: unknown}) => unknown;
};

type Fixture = {
  device: NullDevice;
  graph: GPUCommandGraph;
  buffers: Buffer[];
  passes: Pass[];
};

const ROW_COUNT = 8;

describe('GPUCrossfilter skipUnchangedDimensions node contracts', () => {
  test('adds no conditions unless the option is on', () => {
    const {passes} = createDashboard({});
    expect(passes.length).toBeGreaterThan(0);
    expect(passes.every(pass => pass.condition === undefined)).toBe(true);
  });

  test('conditions every node and runs everything once, then nothing', () => {
    const dashboard = createDashboard({skipUnchangedDimensions: true});
    expect(dashboard.passes.every(pass => pass.condition?.source === 'cpu')).toBe(true);
    expect(dashboard.encode()).toEqual(new Set(dashboard.passes.map(pass => pass.id)));
    expect(dashboard.encode().size).toBe(0);
  });

  test.each([
    'a',
    'b',
    'c'
  ])('setRange(%s) re-evaluates only the nodes that depend on it', brushed => {
    const dashboard = createDashboard({skipUnchangedDimensions: true});
    dashboard.encode();
    dashboard.controller.setRange(brushed, [0, 4]);
    const ran = dashboard.encode();
    for (const dimension of ['a', 'b', 'c']) {
      const isBrushed = dimension === brushed;
      // The dimension's predicate runs only for its own brush.
      expect(dashboard.ranExactly(ran, `xf-${dimension}`)).toBe(isBrushed);
      // The leave-one-out mask and the self-excluding histogram do not depend on that brush.
      expect(dashboard.ranExactly(ran, `xf/controller/compose-without/${dimension}`)).toBe(
        !isBrushed
      );
      expect(dashboard.ran(ran, `xf/view/2:h${dimension}`)).toBe(!isBrushed);
    }
    expect(dashboard.ranExactly(ran, 'xf/controller/compose')).toBe(true);
    expect(dashboard.ran(ran, 'xf/view/5:count')).toBe(true);
    expect(dashboard.encode().size).toBe(0);
  });

  test('clear and clearAll count as changes', () => {
    const dashboard = createDashboard({skipUnchangedDimensions: true});
    dashboard.encode();
    dashboard.controller.clear('b');
    expect(dashboard.ranExactly(dashboard.encode(), 'xf-b')).toBe(true);
    dashboard.controller.clearAll();
    const ran = dashboard.encode();
    for (const dimension of ['a', 'b', 'c'])
      expect(dashboard.ranExactly(ran, `xf-${dimension}`)).toBe(true);
  });

  test('invalidate(dimension) refreshes its predicate and its own histogram', () => {
    const dashboard = createDashboard({skipUnchangedDimensions: true});
    dashboard.encode();
    dashboard.controller.invalidate('a');
    const ran = dashboard.encode();
    expect(dashboard.ranExactly(ran, 'xf-a')).toBe(true);
    expect(dashboard.ranExactly(ran, 'xf-b')).toBe(false);
    // Its histogram excludes its own brush but reads its source values.
    expect(dashboard.ran(ran, 'xf/view/2:ha')).toBe(true);
    expect(dashboard.ran(ran, 'xf/view/2:hb')).toBe(true);
    // The mask that leaves out `a` reads neither its brush nor its values.
    expect(dashboard.ranExactly(ran, 'xf/controller/compose-without/a')).toBe(false);
    expect(dashboard.ranExactly(ran, 'xf/controller/compose-without/b')).toBe(true);
    expect(() => dashboard.controller.invalidate('missing')).toThrow(/does not contain/);
  });

  test('invalidate() refreshes every node', () => {
    const dashboard = createDashboard({skipUnchangedDimensions: true});
    dashboard.encode();
    dashboard.controller.invalidate();
    expect(dashboard.encode()).toEqual(new Set(dashboard.passes.map(pass => pass.id)));
  });

  test('invalidate does nothing observable without conditions', () => {
    const dashboard = createDashboard({});
    expect(() => dashboard.controller.invalidate().invalidate('a')).not.toThrow();
    expect(dashboard.passes.every(pass => pass.condition === undefined)).toBe(true);
  });

  test('keeps masks in persistent buffers only when skipping', () => {
    const skipping = createDashboard({skipUnchangedDimensions: true});
    const exact = createDashboard({});
    expect(isTransient(skipping.controller.getDimensionMask('a'))).toBe(false);
    expect(isTransient(skipping.controller.mask)).toBe(false);
    expect(isTransient(skipping.controller.getViewMask('hb')!)).toBe(false);
    expect(isTransient(exact.controller.getDimensionMask('a'))).toBe(true);
    expect(isTransient(exact.controller.mask)).toBe(true);

    const owned = skipping.createBufferSpy.mock.results
      .map(result => result.value as Buffer)
      .filter(buffer => buffer.id.includes('mask'));
    expect(owned.length).toBeGreaterThan(0);
    skipping.controller.destroy();
    expect(owned.every(buffer => buffer.destroyed)).toBe(true);
  });
});

describe('GPUCrossfilter histogramCubes node contracts', () => {
  test('requires the option and a histogram of a dimension input', () => {
    const plain = createDashboard({});
    expect(() => plain.controller.activateView('ha')).toThrow(/histogramCubes/);

    const dashboard = createDashboard({histogramCubes: true});
    expect(() => dashboard.controller.activateView('count')).toThrow(/cannot be the active/);
    expect(() => dashboard.controller.activateView('missing')).toThrow(/cannot be the active/);
    expect(() => dashboard.controller.activateView('ha').deactivateView()).not.toThrow();
  });

  test('rejects exclusive-maximum dimensions and foreign histogram inputs as active views', () => {
    const fixture = createFixture('cube-candidates');
    const input = createView(fixture, 'values', new Float32Array(ROW_COUNT), 'float32');
    const other = createView(fixture, 'other', new Float32Array(ROW_COUNT), 'float32');
    const controller = new GPUCrossfilter(fixture.graph, {
      id: 'candidates',
      histogramCubes: true,
      dimensions: [
        {id: 'x', kind: 'range', input, exclusiveMaximum: true},
        {id: 'y', kind: 'range', input: other}
      ],
      views: [
        histogramView(fixture, 'hx', 'x', input),
        histogramView(fixture, 'hy', 'y', input) // a different view object than y's input
      ]
    });
    expect(() => controller.activateView('hx')).toThrow(/cannot be the active/);
    expect(() => controller.activateView('hy')).toThrow(/cannot be the active/);
  });

  test('builds one cube per (active, linked) pair with a readout each', () => {
    const dashboard = createDashboard({histogramCubes: true});
    const cubeIds = dashboard.passes
      .filter(pass => pass.workload?.operation === 'GPUCrossfilterHistogramCube')
      .map(pass => pass.id);
    // Three histogram views: 3 active x 2 linked = 6 pairs, each clear + accumulate + prefix + readout.
    expect(cubeIds.filter(id => id.endsWith('/clear'))).toHaveLength(6);
    expect(cubeIds.filter(id => id.endsWith('/prefix'))).toHaveLength(6);
    expect(cubeIds.filter(id => id.endsWith('/readout'))).toHaveLength(6);
    expect(cubeIds.filter(id => id.includes('/accumulate-'))).toHaveLength(6);
  });

  test('routes linked histograms to cubes only while a brushed view is active', () => {
    const dashboard = createDashboard({histogramCubes: true});
    const initial = dashboard.encode();
    expect(dashboard.cubeIds(initial)).toEqual([]);
    expect(dashboard.ran(initial, 'xf/view/2:hb')).toBe(true);

    // Active but unbrushed: cubes build, the exact path still answers.
    dashboard.controller.activateView('ha');
    const built = dashboard.encode();
    expect(dashboard.cubeIds(built).filter(id => id.endsWith('/clear'))).toHaveLength(2);
    expect(dashboard.cubeIds(built).filter(id => id.endsWith('/readout'))).toHaveLength(0);
    expect(dashboard.ran(built, 'xf/view/2:hb')).toBe(true);

    // Brushing the active dimension serves hb and hc from their cubes.
    dashboard.controller.setRange('a', [0, 3]);
    const brushed = dashboard.encode();
    expect(dashboard.cubeIds(brushed).filter(id => id.endsWith('/readout'))).toHaveLength(2);
    expect(dashboard.cubeIds(brushed).filter(id => id.endsWith('/clear'))).toHaveLength(0);
    expect(dashboard.ran(brushed, 'xf/view/2:hb')).toBe(false);
    expect(dashboard.ran(brushed, 'xf/view/2:hc')).toBe(false);
    // The active view excludes its own brush, so it keeps its exact path.
    expect(dashboard.ran(brushed, 'xf/view/2:ha')).toBe(true);
    // Every consumer is a cube-served histogram, so the brushed dimension's mask passes idle.
    expect(dashboard.ranExactly(brushed, 'xf-a')).toBe(false);
    expect(dashboard.ranExactly(brushed, 'xf/controller/compose')).toBe(false);
    expect(dashboard.ranExactly(brushed, 'xf-b')).toBe(true);

    dashboard.controller.setRange('a', [1, 5]);
    const moved = dashboard.encode();
    expect(dashboard.cubeIds(moved).filter(id => id.endsWith('/clear'))).toHaveLength(0);
    expect(dashboard.cubeIds(moved).filter(id => id.endsWith('/readout'))).toHaveLength(2);
  });

  test('rebuilds only the cubes that read a changed non-active dimension', () => {
    const dashboard = createDashboard({histogramCubes: true});
    dashboard.controller.activateView('ha').setRange('a', [0, 3]);
    dashboard.encode();
    dashboard.controller.setRange('b', [0, 4]);
    const clears = dashboard.cubeIds(dashboard.encode()).filter(id => id.endsWith('/clear'));
    // hb excludes its own brush, so only the (ha, hc) cube reads dimension b.
    expect(clears).toHaveLength(1);
    expect(clears[0]).toContain('2:hc');
  });

  test('returns to the exact path when the active view or its brush goes away', () => {
    const dashboard = createDashboard({histogramCubes: true});
    dashboard.controller.activateView('ha').setRange('a', [0, 3]);
    dashboard.encode();
    dashboard.controller.clear('a');
    const cleared = dashboard.encode();
    expect(dashboard.cubeIds(cleared).filter(id => id.endsWith('/readout'))).toHaveLength(0);
    expect(dashboard.ran(cleared, 'xf/view/2:hb')).toBe(true);
    expect(dashboard.ranExactly(cleared, 'xf-a')).toBe(true);

    dashboard.controller.setRange('a', [0, 3]);
    dashboard.encode();
    dashboard.controller.deactivateView();
    const deactivated = dashboard.encode();
    expect(dashboard.cubeIds(deactivated)).toEqual([]);
    expect(dashboard.ran(deactivated, 'xf/view/2:hb')).toBe(true);
    expect(dashboard.ranExactly(deactivated, 'xf-a')).toBe(true);
  });

  test('switching the active view rebuilds every cube', () => {
    const dashboard = createDashboard({histogramCubes: true});
    dashboard.controller.activateView('ha');
    dashboard.encode();
    dashboard.controller.activateView('hb');
    const clears = dashboard.cubeIds(dashboard.encode()).filter(id => id.endsWith('/clear'));
    expect(clears).toHaveLength(2);
    expect(clears.every(id => id.includes('/2:hb/'))).toBe(true);
  });

  test('invalidate rebuilds the active cubes', () => {
    const dashboard = createDashboard({histogramCubes: true});
    dashboard.controller.activateView('ha').setRange('a', [0, 3]);
    dashboard.encode();
    expect(dashboard.cubeIds(dashboard.encode()).filter(id => id.endsWith('/clear'))).toHaveLength(
      0
    );
    dashboard.controller.invalidate();
    expect(dashboard.cubeIds(dashboard.encode()).filter(id => id.endsWith('/clear'))).toHaveLength(
      2
    );
  });

  test('keeps mask passes running when a non-histogram view consumes the brush', () => {
    const dashboard = createDashboard({histogramCubes: true, withCountView: true});
    dashboard.controller.activateView('ha').setRange('a', [0, 3]);
    dashboard.encode();
    const brushed = dashboard.encode();
    expect(dashboard.ranExactly(brushed, 'xf-a')).toBe(true);
    expect(dashboard.ran(brushed, 'xf/view/5:count')).toBe(true);
    // The histograms are still served by their cubes.
    expect(dashboard.ran(brushed, 'xf/view/2:hb')).toBe(false);
  });

  test('combined with skipping, a view rewrites itself when it leaves its cube', () => {
    const dashboard = createDashboard({histogramCubes: true, skipUnchangedDimensions: true});
    dashboard.controller.activateView('ha').setRange('a', [0, 3]);
    dashboard.encode();
    expect(dashboard.ran(dashboard.encode(), 'xf/view/2:hb')).toBe(false);
    dashboard.controller.deactivateView();
    expect(dashboard.ran(dashboard.encode(), 'xf/view/2:hb')).toBe(true);
    expect(dashboard.ran(dashboard.encode(), 'xf/view/2:hb')).toBe(false);
  });

  test('rejects cubes larger than the storage binding limit', () => {
    const fixture = createFixture('cube-memory');
    const input = createView(fixture, 'values', new Float32Array(ROW_COUNT), 'float32');
    const second = createView(fixture, 'second', new Float32Array(ROW_COUNT), 'float32');
    expect(
      () =>
        new GPUCrossfilter(fixture.graph, {
          id: 'big',
          histogramCubes: true,
          dimensions: [
            {id: 'x', kind: 'range', input},
            {id: 'y', kind: 'range', input: second}
          ],
          views: [
            histogramView(fixture, 'hx', 'x', input, 1 << 15),
            histogramView(fixture, 'hy', 'y', second, 1 << 15)
          ]
        })
    ).toThrow(/device limit/);
  });

  test('generated kernels parse and satisfy the core shader limits', () => {
    // 4 x 4 bins use workgroup-local cells; 64 x 64 exceed the local limit and use global atomics.
    for (const binCount of [4, 64]) {
      for (const format of ['float32', 'uint32', 'sint32'] as const) {
        const dashboard = createDashboard({histogramCubes: true, format, binCount});
        const kernels = getGPUCommandNodeKernelSources(
          dashboard.passes.filter(
            pass => pass.workload?.operation === 'GPUCrossfilterHistogramCube'
          )
        );
        expect(kernels.length).toBeGreaterThan(0);
        for (const kernel of kernels) {
          expect(() => new WgslReflect(kernel.source), kernel.id).not.toThrow();
          expect(auditWGSLModule(kernel.source, kernel.constants), kernel.id).toEqual([]);
        }
      }
    }
  });

  test('chunk count does not reach the WGSL text', () => {
    const twoChunks = createDashboard({histogramCubes: true, chunks: 2});
    const threeChunks = createDashboard({histogramCubes: true, chunks: 3});
    const cubePasses = (dashboard: ReturnType<typeof createDashboard>) =>
      dashboard.passes.filter(pass => pass.workload?.operation === 'GPUCrossfilterHistogramCube');
    expect(countDistinctPipelines(cubePasses(threeChunks))).toBe(
      countDistinctPipelines(cubePasses(twoChunks))
    );
  });
});

describe('Falcon cube oracle', () => {
  const rowCount = 600;
  const activeValues = Array.from({length: rowCount}, (_, row) => (row * 37) % 100);
  const linkedValues = Array.from({length: rowCount}, (_, row) => (row * 53 + 11) % 100);
  const otherValues = Array.from({length: rowCount}, (_, row) => (row * 29 + 7) % 100);
  const activeBinning: OracleBinning = {binCount: 10, domain: [0, 100]};
  const linkedBinning: OracleBinning = {binCount: 5, edges: [0, 10, 30, 60, 90, 100]};
  const accepted = otherValues.map(value => value >= 20 && value <= 80);
  const cube = buildCubeOracle(activeValues, linkedValues, accepted, activeBinning, linkedBinning);

  test('equals the exact histogram for edge-aligned brushes', () => {
    // Integer values with width-10 bins: [10 * first, 10 * (last + 1) - 1] covers whole bins.
    for (const [first, last] of [
      [0, 9],
      [0, 0],
      [3, 3],
      [2, 6],
      [9, 9]
    ]) {
      const brush = [first * 10, (last + 1) * 10 - 1] as const;
      const exactMask = accepted.map(
        (pass, row) => pass && activeValues[row] >= brush[0] && activeValues[row] <= brush[1]
      );
      expect(readCubeOracle(cube, brush, activeBinning)).toEqual(
        computeExactHistogram(linkedValues, exactMask, linkedBinning)
      );
    }
  });

  test('snaps unaligned brushes outward to whole bins', () => {
    expect(snapBrush([23, 47], activeBinning)).toEqual([2, 4]);
    expect(snapBrush([100, 100], activeBinning)).toEqual([9, 9]);
    expect(snapBrush([-20, -1], activeBinning)).toBeUndefined();
    expect(snapBrush([101, 200], activeBinning)).toBeUndefined();
    const snapped = readCubeOracle(cube, [23, 47], activeBinning);
    const exactMask = accepted.map(
      (pass, row) => pass && activeValues[row] >= 23 && activeValues[row] <= 47
    );
    const exact = computeExactHistogram(linkedValues, exactMask, linkedBinning);
    snapped.forEach((count, bin) => expect(count).toBeGreaterThanOrEqual(exact[bin]));
  });

  test('counts each accepted row once, so the last prefix row is the unbrushed histogram', () => {
    expect(cube.at(-1)).toEqual(computeExactHistogram(linkedValues, accepted, linkedBinning));
  });
});

type DashboardOptions = Partial<
  Pick<GPUCrossfilterProps, 'skipUnchangedDimensions' | 'histogramCubes'>
> & {
  withCountView?: boolean;
  format?: GPUCrossfilterScalarFormat;
  binCount?: number;
  chunks?: number;
};

/** Three dimensions with a histogram each (and optionally a count), recorded as graph passes. */
function createDashboard(options: DashboardOptions) {
  const fixture = createFixture('dash');
  const format = options.format ?? 'float32';
  const binCount = options.binCount ?? 4;
  const inputs = {
    a: createInput(fixture, 'a', format, options.chunks ?? 1),
    b: createInput(fixture, 'b', format, options.chunks ?? 1),
    c: createInput(fixture, 'c', format, options.chunks ?? 1)
  };
  const views: GPUCrossfilterView[] = [
    histogramView(fixture, 'ha', 'a', inputs.a, binCount, 'domain'),
    histogramView(fixture, 'hb', 'b', inputs.b, binCount, 'auto'),
    histogramView(fixture, 'hc', 'c', inputs.c, binCount, 'edges')
  ];
  if (options.withCountView || !options.histogramCubes) {
    views.push({
      id: 'count',
      kind: 'count',
      output: createView(fixture, 'count-output', new Uint32Array(1), 'uint32')
    });
  }
  const controller = new GPUCrossfilter(fixture.graph, {
    id: 'xf',
    skipUnchangedDimensions: options.skipUnchangedDimensions,
    histogramCubes: options.histogramCubes,
    dimensions: [
      {id: 'a', kind: 'range', input: inputs.a},
      {id: 'b', kind: 'range', input: inputs.b},
      {id: 'c', kind: 'range', input: inputs.c}
    ],
    views
  });
  controller.addToGraph();
  return {
    controller,
    passes: fixture.passes,
    createBufferSpy: fixture.createBufferSpy,
    /** Evaluates every node's condition once, as one graph encoding does. */
    encode(): Set<string> {
      return new Set(
        fixture.passes
          .filter(pass => (pass.condition ? pass.condition.evaluate(undefined) : true))
          .map(pass => pass.id)
      );
    },
    /** Whether a node of the histogram or count view `prefix` (`xf/view/<n>:<id>`) ran. */
    ran(ran: ReadonlySet<string>, prefix: string): boolean {
      return [...ran].some(
        id => id.startsWith(`${prefix}-`) || id.startsWith(`${prefix}/`) || id === prefix
      );
    },
    /** Whether the node with exactly this id ran. */
    ranExactly(ran: ReadonlySet<string>, id: string): boolean {
      return ran.has(id);
    },
    cubeIds(ran: ReadonlySet<string>): string[] {
      return [...ran].filter(id => id.includes('/cube/'));
    }
  };
}

function createFixture(id: string) {
  const device = new NullDevice({id: `${id}-device`});
  Object.defineProperty(device, 'type', {value: 'webgpu'});
  for (const [name, value] of Object.entries({
    maxComputeWorkgroupsPerDimension: 65535,
    maxStorageBufferBindingSize: 134217728,
    maxStorageBuffersPerShaderStage: 8
  })) {
    Object.defineProperty(device.limits, name, {value});
  }
  const graph = new GPUCommandGraph(device, {id});
  const passes: Pass[] = [];
  const addComputePass = graph.addComputePass.bind(graph);
  vi.spyOn(graph, 'addComputePass').mockImplementation(node => {
    passes.push(node as unknown as Pass);
    addComputePass(node);
  });
  const createBufferSpy = vi.spyOn(device, 'createBuffer');
  return {device, graph, buffers: [] as Buffer[], passes, createBufferSpy};
}

function createView<Format extends GPUCrossfilterScalarFormat>(
  fixture: Fixture,
  id: string,
  values: Float32Array | Int32Array | Uint32Array,
  format: Format
): GraphDataView<Format> {
  const buffer = fixture.device.createBuffer({
    id,
    data: values,
    usage: Buffer.STORAGE | Buffer.COPY_DST
  });
  fixture.buffers.push(buffer);
  const handle = fixture.graph.importBuffer(
    {id, byteLength: buffer.byteLength, usage: buffer.usage},
    buffer
  );
  return fixture.graph.createDataView(handle, {format, length: values.length});
}

/** One scalar column, optionally split into equal chunks of a graph vector. */
function createInput(
  fixture: Fixture,
  name: string,
  format: GPUCrossfilterScalarFormat,
  chunkCount: number
): GPUCrossfilterScalarInput {
  const values =
    format === 'float32'
      ? new Float32Array(ROW_COUNT)
      : format === 'sint32'
        ? new Int32Array(ROW_COUNT)
        : new Uint32Array(ROW_COUNT);
  if (chunkCount === 1) return createView(fixture, name, values, format);
  const data = Array.from({length: chunkCount}, (_, chunk) =>
    createView(fixture, `${name}-chunk-${chunk}`, values, format)
  );
  return new GraphVectorView({
    id: name,
    name,
    format,
    length: ROW_COUNT * chunkCount,
    valueLength: ROW_COUNT * chunkCount,
    stride: 1,
    byteStride: 4,
    rowByteLength: 4,
    data
  }) as GPUCrossfilterScalarInput;
}

function histogramView(
  fixture: Fixture,
  id: string,
  dimension: string,
  input: GPUCrossfilterScalarInput,
  binCount = 4,
  binning: 'domain' | 'auto' | 'edges' = 'domain'
): GPUCrossfilterView {
  const output = createView(fixture, `${id}-output`, new Uint32Array(binCount), 'uint32');
  const common = {id, kind: 'histogram' as const, dimension, input, output};
  if (binning === 'edges') {
    return {
      ...common,
      edges: Array.from({length: binCount + 1}, (_, edge) => edge * 2)
    } as GPUCrossfilterView;
  }
  return {
    ...common,
    domain: binning === 'auto' ? 'auto' : [0, 8]
  } as GPUCrossfilterView;
}

/** Whether a mask lives in graph-transient storage rather than a persistent buffer. */
function isTransient(mask: GPUCrossfilterMask): boolean {
  return (mask instanceof GraphVectorView ? mask.data[0] : mask).buffer.transient;
}
