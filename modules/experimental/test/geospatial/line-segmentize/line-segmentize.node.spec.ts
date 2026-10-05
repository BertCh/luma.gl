// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  getGPUGreatCircleArcsParameterValues,
  getGPULineChunkParameterValues,
  getGPULineSegmentizeParameterValues,
  getGPULineSmoothParameterValues,
  GPUGreatCircleArcs,
  GPULineChunk,
  GPULineSegmentize,
  GPULineSmooth,
  type GPULinePathOutput,
  type GPULineSegmentizeProps
} from '../../../src/geospatial/line-segmentize';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

let viewSerial = 0;

/** Returns a graph-unique resource ID. */
function getViewId(name: string): string {
  return `${name}-${viewSerial++}`;
}
import {
  createFlatPaths,
  segmentizePaths,
  tessellateGreatCircleArcs
} from './line-segmentize-oracle';

function createOutput(graph: GPUCommandGraph, pathCount: number, capacity = 64): GPULinePathOutput {
  return {
    positions: createTransientView(graph, getViewId('o-positions'), 'float32x2', capacity),
    pathOffsets: createTransientView(graph, getViewId('o-offsets'), 'uint32', pathCount + 1),
    count: createTransientView(graph, getViewId('o-count'), 'uint32', 1),
    overflow: createTransientView(graph, getViewId('o-overflow'), 'uint32', 1),
    measures: createTransientView(graph, getViewId('o-measures'), 'float32', capacity)
  };
}

function createSegmentizeProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPULineSegmentizeProps> = {}
): GPULineSegmentizeProps {
  return {
    positions: createTransientView(graph, getViewId('positions'), 'float32x2', 10),
    pathOffsets: createTransientView(graph, getViewId('offsets'), 'uint32', 3),
    parameters: createTransientView(graph, getViewId('parameters'), 'float32', 4),
    output: createOutput(graph, 2),
    ...overrides
  };
}

it('GPULineSegmentize validates inputs and emits deterministic node IDs', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'segmentize'});
  const recipe = new GPULineSegmentize({
    ...createSegmentizeProps(graph),
    coordinateSystem: 'spherical'
  });
  const ids = recipe.getCommandNodes(graph).map(node => node.id);
  expect(ids[0]).toBe('line-segmentize-prefix');
  expect(ids).toContain('line-segmentize-count');
  expect(ids).toContain('line-segmentize-emit');
  expect(ids.at(-1)).toBe('line-segmentize-publish');

  expect(
    () =>
      new GPULineSegmentize(createSegmentizeProps(graph, {coordinateSystem: 'mercator' as never}))
  ).toThrow(/coordinateSystem/);
  expect(() => new GPULineSegmentize(createSegmentizeProps(graph, {radius: -1}))).toThrow(/radius/);
  expect(
    () =>
      new GPULineSegmentize(
        createSegmentizeProps(graph, {
          parameters: createTransientView(graph, getViewId('short-parameters'), 'float32', 2)
        })
      )
  ).toThrow(/parameters/);
  expect(
    () => new GPULineSegmentize(createSegmentizeProps(graph, {output: createOutput(graph, 5)}))
  ).toThrow(/pathOffsets/);
  expect(
    () => new GPULineSegmentize(createSegmentizeProps(graph, {maximumPiecesPerSegment: 2 ** 30}))
  ).toThrow(/maximumPiecesPerSegment/);
  const props = createSegmentizeProps(graph);
  expect(
    () =>
      new GPULineSegmentize({
        ...props,
        output: {...props.output, measures: undefined, positions: props.positions}
      })
  ).toThrow(/share buffers/);
});

it('GPUGreatCircleArcs validates pair inputs', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'arcs'});
  const sources = createTransientView(graph, getViewId('sources'), 'float32x2', 3);
  const parameters = createTransientView(graph, getViewId('parameters'), 'float32', 4);
  const recipe = new GPUGreatCircleArcs({
    sources,
    targets: createTransientView(graph, getViewId('targets'), 'float32x2', 3),
    parameters,
    output: createOutput(graph, 3)
  });
  expect(recipe.getCommandNodes(graph).map(node => node.id)).toContain('great-circle-arcs-emit');
  expect(
    () =>
      new GPUGreatCircleArcs({
        sources,
        targets: createTransientView(graph, getViewId('short-targets'), 'float32x2', 2),
        parameters,
        output: createOutput(graph, 3)
      })
  ).toThrow(/targets length/);
});

it('line segmentize parameter packing validates values', () => {
  expect(Array.from(getGPULineSegmentizeParameterValues({maximumSegmentLength: 2.5}))).toEqual([
    2.5, 0, 0, 0
  ]);
  expect(Array.from(getGPULineSegmentizeParameterValues({maximumSegmentLength: Infinity}))).toEqual(
    [0, 0, 0, 0]
  );
  expect(() => getGPULineSegmentizeParameterValues({maximumSegmentLength: -1})).toThrow();
  expect(
    Array.from(
      getGPUGreatCircleArcsParameterValues({maximumSegmentLength: 1000, minimumSegments: 8})
    )
  ).toEqual([1000, 8, 0, 0]);
  expect(() =>
    getGPUGreatCircleArcsParameterValues({maximumSegmentLength: 0, minimumSegments: 0.5})
  ).toThrow(/minimumSegments/);
  expect(() =>
    getGPULineSegmentizeParameterValues({maximumSegmentLength: 1}, new Float32Array(2))
  ).toThrow(/target/);
});

it('line segmentize oracles keep vertices, measures and endpoints', () => {
  const paths = createFlatPaths([
    [
      [0, 0],
      [3, 4],
      [3, 4],
      [3, 5]
    ],
    [],
    [[1, 1]]
  ]);
  const result = segmentizePaths(paths, {maximumSegmentLength: 2, maximumPieces: 100});
  expect(result.pathOffsets).toEqual([0, 6, 6, 7]);
  expect(result.positions[1]).toEqual([1, 4 / 3]);
  expect(result.measures[3]).toBe(5);
  expect(result.measures[5]).toBe(6);
  const arcs = tessellateGreatCircleArcs([[170, 0]], [[-170, 0]], {
    maximumSegmentLength: 0,
    minimumSegments: 4,
    maximumSegments: 10,
    radius: 1
  });
  expect(arcs.positions.at(-1)).toEqual([190, 0]);
  expect(arcs.positions[2][0]).toBeCloseTo(180, 10);
});

it('GPULineSmooth and GPULineChunk validate their layouts', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'smooth-chunk'});
  const positions = createTransientView(graph, getViewId('positions'), 'float32x2', 10);
  const pathOffsets = createTransientView(graph, getViewId('offsets'), 'uint32', 3);
  const parameters = createTransientView(graph, getViewId('parameters'), 'float32', 4);
  const smooth = new GPULineSmooth({
    positions,
    pathOffsets,
    iterations: 3,
    parameters,
    output: {...createOutput(graph, 2), measures: undefined}
  });
  const smoothIds = smooth.getCommandNodes(graph).map(node => node.id);
  expect(smoothIds.filter(id => /^line-smooth-level-\d$/.test(id))).toEqual([
    'line-smooth-level-0',
    'line-smooth-level-1',
    'line-smooth-level-2'
  ]);
  expect(
    () =>
      new GPULineSmooth({
        positions,
        pathOffsets,
        iterations: 3,
        parameters,
        output: createOutput(graph, 2)
      })
  ).toThrow(/not supported/);
  expect(
    () =>
      new GPULineSmooth({
        positions,
        pathOffsets,
        iterations: 11,
        parameters,
        output: {...createOutput(graph, 2), measures: undefined}
      })
  ).toThrow(/iterations/);
  expect(Array.from(getGPULineSmoothParameterValues())).toEqual([0.25, 0, 0, 0]);
  expect(() => getGPULineSmoothParameterValues({ratio: 0.75})).toThrow(/ratio/);

  // Chunk mode: path capacity comes from the output; substring mode needs one path per input.
  const chunk = new GPULineChunk({
    positions,
    pathOffsets,
    mode: 'chunk',
    parameters,
    output: createOutput(graph, 7)
  });
  expect(chunk.pathCapacity).toBe(7);
  expect(chunk.getCommandNodes(graph).map(node => node.id)).toContain('line-chunk-piece-range');
  expect(
    () =>
      new GPULineChunk({
        positions,
        pathOffsets,
        mode: 'substring',
        parameters,
        output: createOutput(graph, 7)
      })
  ).toThrow(/pathOffsets/);
  expect(Array.from(getGPULineChunkParameterValues({chunkLength: 2}))).toEqual([
    2,
    0,
    Math.fround(3.4e38),
    0
  ]);
  expect(() => getGPULineChunkParameterValues({chunkLength: -1})).toThrow();
});
