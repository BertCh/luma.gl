// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {WgslReflect} from 'wgsl_reflect';
import {
  GPURipley,
  GPUVariogram,
  GPUSpatialCorrelogram,
  getGPURipleyParameterValues,
  getGPUVariogramParameterValues,
  getGPUSpatialCorrelogramParameterValues
} from '../../../src/gpu-dataframe/pair-statistics';
import type {GPUStudyWindow} from '../../../src/analysis-core/contracts/study-window';
import {GPUStudyWindowPreparation} from '../../../src/gpu-spatial-analysis/study-window/gpu-study-window-preparation';
import {getPairStatisticsInputNodes} from '../../../src/gpu-dataframe/pair-statistics/pair-statistics-grid';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';
import {ORACLE_RECTANGLE, getOracleWindowBounds} from '../../analysis-core/study-window-oracle';
import {computeRipleyOnCPU} from './ripley-oracle';

function createFixture() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `window-test-${serial++}`, format, length);
  const window = new GPUStudyWindowPreparation({
    positions: view('float32x2', 10),
    featureOffsets: view('uint32', 2),
    polygonOffsets: view('uint32', 2),
    ringOffsets: view('uint32', 3),
    samplingGrid: [8, 8]
  });
  const create = (kind: string, studyWindow?: GPUStudyWindow) => {
    const common = {
      id: 'consumer',
      positions: view('float32x2', 20),
      parameters: view('float32', 10),
      gridSize: [4, 4] as const,
      mask: view('uint32', 20),
      studyWindow
    };
    if (kind === 'ripley')
      return new GPURipley({
        ...common,
        radiusCount: 4,
        output: {
          k: view('float32', 4),
          l: view('float32', 4),
          lMinusR: view('float32', 4),
          pairCorrelation: view('float32', 4),
          radii: view('float32', 4),
          pairCounts: view('uint32', 4)
        }
      });
    if (kind === 'variogram')
      return new GPUVariogram({
        ...common,
        values: view('float32', 20),
        lagCount: 4,
        output: {
          semivariances: view('float32', 4),
          robustSemivariances: view('float32', 4),
          meanDistances: view('float32', 4),
          pairCounts: view('uint32', 4),
          statistics: view('float32', 5),
          status: {overflow: view('uint32', 1)}
        }
      });
    return new GPUSpatialCorrelogram({
      ...common,
      values: view('float32', 20),
      bandCount: 4,
      output: {
        moransI: view('float32', 4),
        zScores: view('float32', 4),
        pValues: view('float32', 4),
        expectedI: view('float32', 4),
        varianceI: view('float32', 4),
        pairCounts: view('uint32', 4),
        peakBands: view('uint32', 2),
        statistics: view('float32', 5),
        status: {overflow: view('uint32', 1)}
      }
    });
  };
  return {device, graph, view, window, create};
}

for (const [name, pack] of [
  ['GPURipley', getGPURipleyParameterValues],
  ['GPUVariogram', getGPUVariogramParameterValues],
  ['GPUSpatialCorrelogram', getGPUSpatialCorrelogramParameterValues]
] as const) {
  it(`${name} packs study-window bounds as NaN and validates distance`, () => {
    const parameters = pack({bounds: 'study-window', maximumDistance: 12});
    expect(Array.from(parameters.slice(0, 4)).every(Number.isNaN)).toBe(true);
    expect(parameters[4]).toBe(12);
    expect(() => pack({bounds: [] as never, maximumDistance: 1})).toThrow(
      `${name} bounds must be [minX, minY, maxX, maxY] or 'study-window'`
    );
    expect(() => pack({bounds: 'study-window', maximumDistance: NaN})).toThrow(/finite/);
    expect(() => pack({bounds: 'study-window', maximumDistance: 0})).toThrow(/positive/);
  });
}

for (const kind of ['ripley', 'variogram', 'correlogram']) {
  it(`${kind} resolves study windows and rejects foreign window views including packed`, () => {
    const {device, graph, window, create} = createFixture();
    try {
      expect(create(kind).studyWindow).toBe('rectangle');
      expect(() => create(kind, {} as never)).toThrow(
        "consumer studyWindow must be 'rectangle' or a GPUStudyWindowPreparation"
      );
      expect(() =>
        create(kind, {
          kind: 'polygon',
          id: 'foreign-context',
          getViews: () => window.getViews(graph),
          spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'}
        })
      ).toThrow(/consumer studyWindow incompatible spatial contexts:/);
      expect(() => create(kind, window).getCommandNodes(graph)).toThrow(
        /study window needs graph.add/
      );
      graph.add(window);
      const consumer = create(kind, window);
      expect(consumer.studyWindow).toBe(window);
      const other = new GPUCommandGraph(device);
      expect(() => window.getViews(other)).toThrow(/study window needs graph.add/);
      const views = window.getViews(graph);
      const foreignPacked = createTransientView(
        other,
        'foreign-packed',
        'uint32',
        views.packed.length
      );
      expect(() =>
        create(kind, {
          kind: 'polygon',
          id: 'foreign',
          spatialContext: window.spatialContext,
          getViews: () => ({...views, packed: foreignPacked})
        }).getCommandNodes(graph)
      ).toThrow(/target graph/);
      // Also reject a handle returning all of another graph's views.
      expect(() => consumer.getCommandNodes(other)).toThrow(/target graph/);
    } finally {
      device.destroy();
    }
  });

  it(`${kind} adds effective parameters, one inclusion call site, and stays within eight bindings`, () => {
    const {device, graph, window, create} = createFixture();
    try {
      graph.add(window);
      const nodes = create(kind, window).getCommandNodes(graph);
      expect(nodes[0].id).toBe('consumer-window-parameters');
      const sources = getGPUCommandNodeKernelSources(nodes);
      for (const {id, source} of sources) {
        expect(() => new WgslReflect(source), id).not.toThrow();
        expect(getStorageBindingCount(source), id).toBeLessThanOrEqual(8);
        if (source.includes('fn isInsideStudyWindow')) {
          expect(source.match(/fn isFiniteBits\(/g)).toHaveLength(1);
          expect(source.match(/fn exactProbeCrossing\(/g)).toHaveLength(1);
          const references = source.match(/isInsideStudyWindow\(/g) ?? [];
          expect(references.length, id).toBe(id === 'consumer-cell-keys' ? 2 : 1);
        }
      }
      const parameters = sources.find(source => source.id === 'consumer-window-parameters')!.source;
      expect(parameters).toContain('isStudyWindowUsable()');
      expect(parameters).toContain('studyWindowSummaryValue(index)');
      expect(parameters).toContain('sourceParameters[sourceParametersOffset + index]');
      expect(parameters).toContain('10u');
      if (kind === 'ripley') {
        const pairs = sources.find(source => source.id === 'consumer-histogram-pairs')!.source;
        expect(pairs).toContain('if (boundaryDistance < pairDistance)');
        expect(pairs).toContain('getStudyWindowCircleFraction(vec2f(x, y), pairDistance)');
        expect(sources.find(source => source.id === 'consumer-finish')!.source).toContain(
          'let area = studyWindowSummaryValue(4u)'
        );
      }
    } finally {
      device.destroy();
    }
  });

  it(`${kind} keeps omitted and explicit rectangle nodes and WGSL identical`, () => {
    const build = (studyWindow?: GPUStudyWindow) => {
      const {device, graph, create} = createFixture();
      try {
        const nodes = create(kind, studyWindow).getCommandNodes(graph);
        return {ids: nodes.map(node => node.id), sources: getGPUCommandNodeKernelSources(nodes)};
      } finally {
        device.destroy();
      }
    };
    expect(build('rectangle')).toEqual(build());
  });
}

it('shared input nodes keep omitted and undefined studyWindow WGSL identical', () => {
  const build = (withOption: boolean) => {
    const {device, graph, view} = createFixture();
    try {
      const inputs = getPairStatisticsInputNodes(graph, {
        id: 'input',
        operation: 'test',
        positions: view('float32x2', 20),
        parameters: view('float32', 8),
        gridSize: [4, 4],
        ...(withOption ? {studyWindow: undefined} : {})
      });
      return {
        ids: inputs.nodes.map(node => node.id),
        sources: getGPUCommandNodeKernelSources(inputs.nodes)
      };
    } finally {
      device.destroy();
    }
  };
  expect(build(true)).toEqual(build(false));
});

it('Ripley polygon oracle agrees with rectangle oracle to 1e-9 in all corrections', () => {
  const scene = {
    positions: new Float32Array([
      123, 144, 321, 222, 550, 450, 880, 650, 1100, 810, 1700, 311, 2200, 500
    ])
  };
  for (const edgeCorrection of ['none', 'border', 'isotropic'] as const) {
    const frame = {
      bounds: getOracleWindowBounds(ORACLE_RECTANGLE),
      maximumDistance: 700,
      edgeCorrection
    };
    const rectangle = computeRipleyOnCPU(scene, frame, 7);
    const polygon = computeRipleyOnCPU(scene, {...frame, studyWindow: ORACLE_RECTANGLE}, 7);
    expect(polygon.n).toBe(rectangle.n);
    for (const key of ['k', 'l', 'pairCorrelation'] as const) {
      polygon[key].forEach((value, index) => {
        if (Number.isNaN(rectangle[key][index])) expect(value).toBeNaN();
        else expect(Math.abs(value - rectangle[key][index])).toBeLessThanOrEqual(1e-9);
      });
    }
  }
});
