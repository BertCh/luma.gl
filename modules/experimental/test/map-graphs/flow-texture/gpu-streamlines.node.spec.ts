// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  generateStreamlinesOnCPU,
  getGPUStreamlinesParameterValues,
  getGPUStreamlinesWordParameterValues,
  GPUStreamlines,
  type GPUStreamlinesProps,
  type StreamlinesCPUConfig
} from '../../../src/map-graphs/flow-texture';
import {
  getStreamlineGridCell,
  getStreamlineKey
} from '../../../src/map-graphs/flow-texture/streamlines-cpu';
import {createNullWebGPUDevice} from '../map-graph-test-utils';
import {createUniformField, createVortexField} from './flow-texture-scenes';

let serial = 0;

function createProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUStreamlinesProps> = {}
): GPUStreamlinesProps {
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `view-${serial++}`, format, length);
  return {
    velocities: view('float32x2', 16),
    fieldWidth: 4,
    fieldHeight: 4,
    gridWidth: 8,
    gridHeight: 8,
    seedColumns: 3,
    seedRows: 2,
    stepsPerDirection: 5,
    roundCount: 4,
    parameters: view('float32', 12),
    wordParameters: view('uint32', 4),
    output: {
      lines: {
        ids: view('uint32', 6),
        count: view('uint32', 1),
        overflow: view('uint32', 1)
      },
      pathOffsets: view('uint32', 7),
      points: view('float32x2', 66),
      pointCount: view('uint32', 1),
      unconverged: view('uint32', 1)
    },
    ...overrides
  };
}

it('GPUStreamlines validates props and builds deterministic nodes', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {
    id: 'streamline-validation'
  });
  const recipe = new GPUStreamlines(createProps(graph, {id: 'lines'}));
  expect(recipe.recipe).toBe('streamlines');
  expect(recipe.seedCount).toBe(6);
  const ids = recipe.getCommandNodes(graph).map(node => node.id);
  expect(ids[0]).toBe('lines-trace');
  expect(ids).toContain('lines-round-3-decide');
  expect(ids).not.toContain('lines-round-4-claim');
  expect(ids[ids.length - 1]).toBe('lines-finalize');
  expect(() => new GPUStreamlines(createProps(graph, {seedColumns: 300, seedRows: 300}))).toThrow(
    /at most 65535/
  );
  expect(() => new GPUStreamlines(createProps(graph, {stepsPerDirection: 0}))).toThrow(
    /stepsPerDirection/
  );
  const props = createProps(graph);
  expect(
    () =>
      new GPUStreamlines({
        ...props,
        output: {
          ...props.output,
          pathOffsets: createTransientView(graph, 'offsets', 'uint32', 6)
        }
      })
  ).toThrow(/pathOffsets/);
  expect(
    () =>
      new GPUStreamlines({
        ...props,
        output: {
          ...props.output,
          candidates: {
            points: createTransientView(graph, 'candidates', 'float32x2', 10),
            spans: createTransientView(graph, 'spans', 'uint32', 24)
          }
        }
      })
  ).toThrow(/candidates/);
});

it('streamline parameter helpers pack the documented layout', () => {
  expect(
    Array.from(
      getGPUStreamlinesParameterValues({
        fieldExtent: [1, 2, 3, 4],
        gridExtent: [5, 6, 0.5, 0.25],
        stepLength: 0.1,
        minimumSpeed: 0.2
      })
    )
  ).toEqual([1, 2, 3, 4, 5, 6, 0.5, 0.25, 2, 4, Math.fround(0.1), Math.fround(0.2)]);
  expect(Array.from(getGPUStreamlinesWordParameterValues({seed: 3}))).toEqual([3, 2, 0, 0]);
  expect(() =>
    getGPUStreamlinesParameterValues({
      fieldExtent: [0, 0, 1, 1],
      gridExtent: [0, 0, 0, 1],
      stepLength: 1
    })
  ).toThrow(/positive size/);
});

it('generateStreamlinesOnCPU spaces lines by the grid and orders priorities uniquely', () => {
  const config: StreamlinesCPUConfig = {
    seedColumns: 12,
    seedRows: 12,
    stepsPerDirection: 30,
    gridWidth: 24,
    gridHeight: 24,
    lineCapacity: 144,
    pointCapacity: 144 * 61
  };
  const settings = {
    fieldExtent: [0, 0, 1, 1] as const,
    gridExtent: [0, 0, 0.5, 0.5] as const,
    stepLength: 0.25,
    seed: 4,
    minimumPoints: 3
  };
  const parameters = getGPUStreamlinesParameterValues(settings);
  const words = getGPUStreamlinesWordParameterValues(settings);
  for (const field of [createUniformField(12, 12, 0.6, 0.8), createVortexField(12, 0.2)]) {
    const result = generateStreamlinesOnCPU(field, config, parameters, words);
    expect(result.ids.length).toBeGreaterThan(5);
    expect(new Set(result.keys).size).toBe(result.keys.length);
    // No grid cell is shared by two published lines, and every line keeps minimumPoints.
    const owner = new Map<number, number>();
    for (let line = 0; line < result.ids.length; line++) {
      expect(result.pathOffsets[line + 1] - result.pathOffsets[line]).toBeGreaterThanOrEqual(3);
      for (let point = result.pathOffsets[line]; point < result.pathOffsets[line + 1]; point++) {
        const cell = getStreamlineGridCell(
          result.points[2 * point],
          result.points[2 * point + 1],
          parameters,
          config
        );
        expect(cell).toBeGreaterThanOrEqual(0);
        expect(owner.get(cell) ?? line).toBe(line);
        owner.set(cell, line);
      }
    }
    // Published IDs ascend.
    expect([...result.ids].sort((left, right) => left - right)).toEqual(result.ids);
  }
  // Keys break ties toward the lower seed index.
  expect(getStreamlineKey(0, 1) & 0xffff).toBe(0xffff);
  expect(getStreamlineKey(5, 1) & 0xffff).toBe(0xffff - 5);
});
