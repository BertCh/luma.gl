// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  getGPULineLocateParameterValues,
  GPULinearReferencing,
  GPULineLocate,
  type GPULinearReferencingProps,
  type GPULineLocateProps
} from '../../../src/map-graphs/linear-referencing';
import {createNullWebGPUDevice} from '../map-graph-test-utils';
import {
  findNearestSegment,
  getVertexMeasures,
  locateAlong,
  type ReferencePaths
} from './linear-referencing-oracle';

let viewSerial = 0;

function view<Format extends 'uint32' | 'float32' | 'float32x2' | 'sint32'>(
  graph: GPUCommandGraph,
  format: Format,
  length: number
) {
  return createTransientView(graph, `view-${viewSerial++}`, format, length);
}

function createReferencingProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPULinearReferencingProps> = {}
): GPULinearReferencingProps {
  return {
    points: view(graph, 'float32x2', 8),
    positions: view(graph, 'float32x2', 12),
    pathOffsets: view(graph, 'uint32', 4),
    radius: view(graph, 'float32', 1),
    candidateCapacity: 64,
    output: {measures: view(graph, 'float32', 8), sides: view(graph, 'sint32', 8)},
    overflow: view(graph, 'uint32', 1),
    ...overrides
  };
}

function createLocateProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPULineLocateProps> = {}
): GPULineLocateProps {
  return {
    positions: view(graph, 'float32x2', 12),
    pathOffsets: view(graph, 'uint32', 4),
    eventPaths: view(graph, 'uint32', 5),
    eventMeasures: view(graph, 'float32', 5),
    output: {positions: view(graph, 'float32x2', 5)},
    ...overrides
  };
}

it('GPULinearReferencing composes the nearest-feature join and validates outputs', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'referencing'});
  const ids = new GPULinearReferencing(createReferencingProps(graph))
    .getCommandNodes(graph)
    .map(node => node.id);
  expect(ids.slice(0, 2)).toEqual([
    'linear-referencing-segments',
    'linear-referencing-vertex-measures'
  ]);
  expect(ids.some(id => id.startsWith('linear-referencing-join-'))).toBe(true);
  expect(ids).toContain('linear-referencing-project');
  expect(ids.at(-1)).toBe('linear-referencing-output');

  expect(() => new GPULinearReferencing(createReferencingProps(graph, {output: {}}))).toThrow(
    /at least one output/
  );
  expect(
    () =>
      new GPULinearReferencing(
        createReferencingProps(graph, {output: {measures: view(graph, 'float32', 3)}})
      )
  ).toThrow(/must hold 8 rows/);
  expect(
    () =>
      new GPULinearReferencing(
        createReferencingProps(graph, {output: {vertexMeasures: view(graph, 'float32', 8)}})
      )
  ).toThrow(/must hold 12 rows/);
  expect(
    () =>
      new GPULinearReferencing(
        createReferencingProps(graph, {positions: view(graph, 'float32x2', 1)})
      )
  ).toThrow(/at least two positions/);
});

it('GPULineLocate validates events and parameters', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'locate'});
  const ids = new GPULineLocate(
    createLocateProps(graph, {
      output: {positions: view(graph, 'float32x2', 5), angles: view(graph, 'float32', 5)}
    })
  )
    .getCommandNodes(graph)
    .map(node => node.id);
  expect(ids).toEqual(['line-locate-vertex-measures', 'line-locate-locate', 'line-locate-output']);
  expect(
    () => new GPULineLocate(createLocateProps(graph, {eventMeasures: view(graph, 'float32', 4)}))
  ).toThrow(/eventMeasures length/);
  expect(
    () => new GPULineLocate(createLocateProps(graph, {measureMode: 'percent' as never}))
  ).toThrow(/measureMode/);
  expect(
    () => new GPULineLocate(createLocateProps(graph, {parameters: view(graph, 'float32', 2)}))
  ).toThrow(/parameters/);
  expect(Array.from(getGPULineLocateParameterValues({measureOffset: 3}))).toEqual([1, 3, 0, 0]);
  expect(() => getGPULineLocateParameterValues({measureScale: Number.NaN})).toThrow(/finite/);
});

it('linear referencing oracles agree with each other', () => {
  const paths: ReferencePaths = {
    positions: new Float32Array([0, 0, 10, 0, 10, 10, 20, 20]),
    pathOffsets: new Uint32Array([0, 3, 4])
  };
  const measures = getVertexMeasures(paths);
  expect(measures).toEqual([0, 10, 20, 0]);
  const projection = findNearestSegment(paths, measures, [12, 4], 100);
  expect(projection?.segmentRow).toBe(1);
  expect(projection?.measure).toBe(14);
  expect(projection?.side).toBe(-1);
  const located = locateAlong(paths, measures, 0, projection?.measure ?? 0);
  expect(located.position).toEqual([10, 4]);
  expect(locateAlong(paths, measures, 0, 25).status).toBe(1);
  expect(locateAlong(paths, measures, 5, 1).status).toBe(2);
  expect(locateAlong(paths, measures, 1, 0).tangent).toEqual([0, 0]);
});
