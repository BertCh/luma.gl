// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  getGPUClassBreaksParameterLength,
  getGPUClassBreaksParameterValues,
  GPU_CLASS_BREAKS_METHOD_CODES,
  type GPUClassBreaksMethod
} from '../../../src/map-graphs/column-classification/class-breaks-parameters';
import {
  GPUClassBreaks,
  type GPUClassBreaksProps
} from '../../../src/map-graphs/column-classification/gpu-class-breaks';
import {createNullWebGPUDevice, createVectorView} from '../map-graph-test-utils';
import {computeClassBreaksOracle, countClassesOracle} from './class-breaks-oracle';

let graphCount = 0;

/** Returns a fresh graph, so transient IDs never collide between cases. */
function createGraph(): GPUCommandGraph {
  return new GPUCommandGraph(createNullWebGPUDevice(), {id: `graph-${graphCount++}`});
}

function createProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUClassBreaksProps> = {},
  rows = 100,
  maximumClassCount = 8
): GPUClassBreaksProps {
  return {
    values: createTransientView(graph, 'values', 'float32', rows),
    mask: createTransientView(graph, 'mask', 'uint32', rows),
    parameters: createTransientView(
      graph,
      'parameters',
      'float32',
      getGPUClassBreaksParameterLength(maximumClassCount)
    ),
    maximumClassCount,
    output: {
      breaks: createTransientView(graph, 'breaks', 'float32', maximumClassCount + 1),
      classCount: createTransientView(graph, 'class-count', 'uint32', 1),
      classCounts: createTransientView(graph, 'class-counts', 'uint32', maximumClassCount)
    },
    ...overrides
  };
}

it('getGPUClassBreaksParameterValues packs the header and method data', () => {
  const maximumClassCount = 8;
  expect(getGPUClassBreaksParameterLength(maximumClassCount)).toBe(17);
  const quantile = getGPUClassBreaksParameterValues({method: 'quantile', classCount: 4}, 8);
  expect(Array.from(quantile.slice(0, 5))).toEqual([1, 4, 1, Math.fround(0.4), 1.5]);
  expect(Array.from(quantile.slice(8, 11))).toEqual([0.25, 0.5, 0.75]);
  expect(Number.isNaN(quantile[11])).toBe(true);
  const boxPlot = getGPUClassBreaksParameterValues({method: 'box-plot', boxPlotHinge: 3}, 8);
  expect(boxPlot[1]).toBe(6);
  expect(boxPlot[4]).toBe(3);
  expect(Array.from(boxPlot.slice(8, 11))).toEqual([0.25, 0.5, 0.75]);
  const custom = getGPUClassBreaksParameterValues({method: 'custom', customEdges: [0, 1, 5]}, 8);
  expect(custom[0]).toBe(GPU_CLASS_BREAKS_METHOD_CODES.custom);
  expect(custom[1]).toBe(2);
  expect(Array.from(custom.slice(8, 11))).toEqual([0, 1, 5]);
  const third = getGPUClassBreaksParameterValues({method: 'quantile', classCount: 3}, 8);
  expect(third[8]).toBe(Math.fround(1 / 3));
  expect(() => getGPUClassBreaksParameterValues({method: 'quantile', classCount: 9}, 8)).toThrow(
    /Class count/
  );
  expect(() => getGPUClassBreaksParameterValues({method: 'custom', customEdges: [1]}, 8)).toThrow(
    /two edges/
  );
  expect(() =>
    getGPUClassBreaksParameterValues({method: 'nope' as GPUClassBreaksMethod}, 8)
  ).toThrow(/Unknown/);
  expect(() =>
    getGPUClassBreaksParameterValues({method: 'quantile'}, 8, new Float32Array(4))
  ).toThrow(/must hold/);
});

it('GPUClassBreaks validates its views and options', () => {
  /** Builds the recipe on a fresh graph with overrides derived from that graph. */
  const build = (
    getOverrides: (graph: GPUCommandGraph) => Partial<GPUClassBreaksProps> = () => ({}),
    rows = 100,
    maximumClassCount = 8
  ) => {
    const graph = createGraph();
    return () =>
      new GPUClassBreaks(createProps(graph, getOverrides(graph), rows, maximumClassCount));
  };
  expect(build(() => ({maximumClassCount: 0}))).toThrow(/maximumClassCount/);
  expect(build(() => ({maximumClassCount: 65}))).toThrow(/maximumClassCount/);
  expect(build(() => ({methods: []}))).toThrow(/at least one/);
  expect(build(() => ({methods: ['bogus' as GPUClassBreaksMethod]}))).toThrow(/unknown method/);
  expect(build(undefined, 100, 5)).toThrow(/box-plot/);
  expect(build(() => ({methods: ['quantile']}), 100, 5)).not.toThrow();
  expect(build(() => ({naturalBreaksBinCount: 1}))).toThrow(/naturalBreaksBinCount/);
  expect(build(() => ({naturalBreaksBinCount: 4096}))).toThrow(/naturalBreaksBinCount/);
  expect(build(graph => ({mask: createTransientView(graph, 'short-mask', 'uint32', 99)}))).toThrow(
    /mask length/
  );
  expect(
    build(
      graph =>
        ({
          values: createVectorView('vector', 'float32', [
            createTransientView(graph, 'chunk', 'float32', 100)
          ])
        }) as unknown as Partial<GPUClassBreaksProps>
    )
  ).toThrow(/chunked vector/);
  expect(build(graph => ({parameters: createTransientView(graph, 'short', 'float32', 4)}))).toThrow(
    /parameters must hold/
  );
  const getOutput = (graph: GPUCommandGraph) => ({
    breaks: createTransientView(graph, 'other-breaks', 'float32', 9),
    classCount: createTransientView(graph, 'other-class-count', 'uint32', 1),
    classCounts: createTransientView(graph, 'other-class-counts', 'uint32', 8)
  });
  expect(
    build(graph => ({
      output: {...getOutput(graph), breaks: createTransientView(graph, 'few', 'float32', 4)}
    }))
  ).toThrow(/output.breaks/);
  expect(
    build(graph => ({
      output: {
        ...getOutput(graph),
        classCounts: createTransientView(graph, 'few-counts', 'uint32', 2)
      }
    }))
  ).toThrow(/output.classCounts/);
  const graph = createGraph();
  const props = createProps(graph);
  expect(
    () => new GPUClassBreaks({...props, output: {...props.output, classCount: props.mask!}})
  ).toThrow(/share buffers/);
});

it('GPUClassBreaks returns deterministic node IDs and rejects foreign views', () => {
  const getNodeIds = () => {
    const graph = createGraph();
    return new GPUClassBreaks({...createProps(graph), id: 'cb'})
      .getCommandNodes(graph)
      .map(node => node.id);
  };
  const first = getNodeIds();
  const second = getNodeIds();
  expect(second).toEqual(first);
  expect(new Set(first).size).toBe(first.length);
  expect(first[0]).toBe('cb-prepare');
  expect(first[1]).toBe('cb-extremes');
  expect(first).toContain('cb-head-tail-0-split');
  expect(first).toContain('cb-natural-breaks-backtrack');
  expect(first).toContain('cb-maximum-breaks-finish');
  expect(first.slice(-4)).toEqual([
    'cb-finish',
    'cb-publish',
    'cb-class-counts-clear',
    'cb-class-counts'
  ]);
  expect(first.every(id => id.startsWith('cb-'))).toBe(true);
  // Leaving out the sort-based and iterative methods drops their nodes.
  const leanGraph = createGraph();
  const lean = new GPUClassBreaks({
    ...createProps(leanGraph),
    id: 'lean',
    methods: ['equal-interval']
  });
  expect(lean.getCommandNodes(leanGraph).map(node => node.id)).toEqual([
    'lean-prepare',
    'lean-extremes',
    'lean-finish',
    'lean-publish',
    'lean-class-counts-clear',
    'lean-class-counts'
  ]);
  const recipe = new GPUClassBreaks(createProps(createGraph()));
  expect(() => recipe.getCommandNodes(createGraph())).toThrow(/target graph/);
});

it('class breaks oracle reproduces textbook breaks', () => {
  const values = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, NaN, Infinity]);
  const base = {values, maximumClassCount: 8, naturalBreaksBinCount: 10};
  const equal = computeClassBreaksOracle({
    ...base,
    parameters: {method: 'equal-interval', classCount: 3}
  });
  expect(equal.classCount).toBe(3);
  expect(Array.from(equal.breaks.slice(0, 4))).toEqual([1, 4, 7, 10]);
  const quantile = computeClassBreaksOracle({
    ...base,
    parameters: {method: 'quantile', classCount: 2}
  });
  // Infinity takes part in quantiles (the median of 1..10 and Infinity is 6); the end edges are finite.
  expect(Array.from(quantile.breaks.slice(0, 3))).toEqual([1, 6, 10]);
  const gaps = new Float32Array([1, 2, 3, 10, 11, 30]);
  const maximum = computeClassBreaksOracle({
    ...base,
    values: gaps,
    parameters: {method: 'maximum-breaks', classCount: 3}
  });
  expect(Array.from(maximum.breaks.slice(0, 4))).toEqual([1, 6.5, 20.5, 30]);
  const natural = computeClassBreaksOracle({
    ...base,
    values: new Float32Array([0, 0.5, 1, 9, 9.5, 10]),
    parameters: {method: 'natural-breaks', classCount: 2}
  });
  // Every start inside the empty gap ties; the lowest start wins.
  expect(natural.naturalStartBins).toEqual([2]);
  const headTail = computeClassBreaksOracle({
    ...base,
    values: new Float32Array([1, 1, 1, 1, 1, 1, 1, 2, 4, 100]),
    parameters: {method: 'head-tail', classCount: 8, headTailRatio: 0.4}
  });
  // Mean 11.3 splits off [100]; a single-row head stops.
  expect(headTail.classCount).toBe(2);
  expect(headTail.breaks[1]).toBeCloseTo(11.3, 5);
  expect(Array.from(countClassesOracle(values, undefined, equal.breaks, 3, 3))).toEqual([3, 3, 5]);
});

it('GPUClassBreaks fits every kernel in 8 storage buffers with all methods compiled', () => {
  // The default WebGPU limit; the null device reports it, so an extra binding throws here.
  const graph = createGraph();
  const recipe = new GPUClassBreaks(createProps(graph));
  expect(() => recipe.getCommandNodes(graph)).not.toThrow();
});
