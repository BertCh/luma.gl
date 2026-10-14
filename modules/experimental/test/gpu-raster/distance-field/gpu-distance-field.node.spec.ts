// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  getGPUDistanceFieldParameterValues,
  GPUDistanceField,
  GPU_DISTANCE_FIELD_MAXIMUM_DIMENSION,
  GPU_DISTANCE_FIELD_NONE,
  GPU_DISTANCE_FIELD_PARAMETER_LENGTH,
  type GPUDistanceFieldProps
} from '../../../src/gpu-raster/distance-field';
import {getDistanceFieldJumpFloodSteps} from '../../../src/gpu-raster/distance-field/distance-field-kernels';
import {makeGPURasterMetadata} from '../../../src/gpu-raster/raster-grid';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {getRasterPipelines} from '../raster-pipeline-count';
import {computeDistanceFieldOnCPU, getCellSeeds, getUlpDistance} from './distance-field-oracle';

const WIDTH = 20;
const HEIGHT = 10;
const CELLS = WIDTH * HEIGHT;
const NONE = GPU_DISTANCE_FIELD_NONE;

function createProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUDistanceFieldProps> = {}
): GPUDistanceFieldProps {
  return {
    grid: makeGPURasterMetadata({width: WIDTH, height: HEIGHT}),
    parameters: createTransientView(graph, 'parameters', 'float32', 8),
    seedPositions: createTransientView(graph, 'positions', 'float32x2', 4),
    output: {
      distances: createTransientView(graph, 'distances', 'float32', CELLS)
    },
    ...overrides
  };
}

it('packs distance-field settings', () => {
  expect(GPU_DISTANCE_FIELD_PARAMETER_LENGTH).toBe(8);
  expect(GPU_DISTANCE_FIELD_NONE).toBe(0xffffffff);
  expect(GPU_DISTANCE_FIELD_MAXIMUM_DIMENSION).toBe(32768);
  // North-up grid: the row step is negative and the origin is the top-left corner.
  expect(
    Array.from(
      getGPUDistanceFieldParameterValues(
        {},
        makeGPURasterMetadata({width: 4, height: 3, cellSize: [2, 0.5], origin: [3, -4]})
      )
    )
  ).toEqual([3, -4, 2, 0.5, 0.5, -2, Infinity, 0]);
  // North-down grid: row 0 is the minimum-y edge.
  expect(
    Array.from(
      getGPUDistanceFieldParameterValues(
        {maximumDistance: 9, localOrigin: [1, 1]},
        makeGPURasterMetadata({
          width: 50,
          height: 4,
          cellSize: [2, 5],
          origin: [0, 10],
          rowDirection: 'north'
        })
      )
    )
  ).toEqual([-1, 9, 2, 5, 0.5, 0.20000000298023224, 9, 0]);
  const grid = makeGPURasterMetadata({width: 2, height: 2});
  expect(() =>
    getGPUDistanceFieldParameterValues({}, {...grid, affine: [0, 0, 0, 0, -1, 0]})
  ).toThrow(/positive/);
  expect(() =>
    getGPUDistanceFieldParameterValues({}, {...grid, affine: [1, 0.5, 0, 0, -1, 0]})
  ).toThrow(/axis-aligned/);
  expect(() => getGPUDistanceFieldParameterValues({}, grid, new Float32Array(4))).toThrow(
    /8 values/
  );
});

it('GPUDistanceField builds deterministic exact and jump-flood node lists', () => {
  const device = createNullWebGPUDevice();
  const getNodeIds = (overrides: Partial<GPUDistanceFieldProps>) => {
    const graph = new GPUCommandGraph(device);
    const contributor = new GPUDistanceField({
      ...createProps(graph),
      id: 'df',
      ...overrides
    });
    const first = contributor.getCommandNodes(graph).map(node => node.id);
    return first;
  };
  expect(getNodeIds({})).toEqual([
    'df-seed-clear',
    'df-seed-scatter',
    'df-column-bands',
    'df-column-carry',
    'df-column-fill',
    'df-row-bands',
    'df-row-marker-clear',
    'df-row-start-markers',
    'df-row-marker-bands',
    'df-row-marker-carry',
    'df-row-color',
    'df-finalize'
  ]);
  // 20x10 rounds up to 32: steps 16, 8, 4, 2, 1 plus one refinement pass.
  expect(getNodeIds({mode: 'jump-flood'})).toEqual([
    'df-seed-clear',
    'df-seed-scatter',
    'df-jump-flood-initialize',
    'df-jump-flood-steps',
    ...[0, 1, 2, 3, 4, 5].map(step => `df-jump-flood-${step}`),
    'df-finalize'
  ]);
  const maskOnly = (graph: GPUCommandGraph) => ({
    seedPositions: undefined,
    seedMask: createTransientView(graph, 'mask', 'uint32', CELLS)
  });
  const graph = new GPUCommandGraph(device);
  const contributor = new GPUDistanceField({
    ...createProps(graph),
    ...maskOnly(graph)
  });
  expect(contributor.id).toBe('distance-field');
  expect(contributor.getCommandNodes(graph).map(node => node.id)).toEqual([
    'distance-field-seed-clear',
    'distance-field-column-bands',
    'distance-field-column-carry',
    'distance-field-column-fill',
    'distance-field-row-bands',
    'distance-field-row-marker-clear',
    'distance-field-row-start-markers',
    'distance-field-row-marker-bands',
    'distance-field-row-marker-carry',
    'distance-field-row-color',
    'distance-field-finalize'
  ]);
});

it('getDistanceFieldJumpFloodSteps halves from the padded size and appends refinement', () => {
  expect(getDistanceFieldJumpFloodSteps({width: 1, height: 1}, 1)).toEqual([1]);
  expect(getDistanceFieldJumpFloodSteps({width: 1, height: 1}, 0)).toEqual([]);
  expect(getDistanceFieldJumpFloodSteps({width: 256, height: 3}, 0)).toEqual([
    128, 64, 32, 16, 8, 4, 2, 1
  ]);
  expect(getDistanceFieldJumpFloodSteps({width: 5, height: 9}, 2)).toEqual([8, 4, 2, 1, 2, 1]);
});

it('GPUDistanceField validates its props', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const baseProps = createProps(graph);
  const create = (overrides: Partial<GPUDistanceFieldProps>) =>
    new GPUDistanceField({...baseProps, ...overrides});
  const otherGraph = new GPUCommandGraph(device);
  expect(() => create({grid: makeGPURasterMetadata({width: 0, height: HEIGHT})})).toThrow(
    /grid dimensions/
  );
  expect(() =>
    create({
      grid: makeGPURasterMetadata({width: 1, height: GPU_DISTANCE_FIELD_MAXIMUM_DIMENSION + 1})
    })
  ).toThrow(/height/);
  expect(() => create({mode: 'nearest' as never})).toThrow(/mode/);
  expect(() => create({jumpFloodRefinementIterationCount: 3 as never})).toThrow(
    /refinement|0, 1, or 2/
  );
  expect(() => create({parameters: createTransientView(graph, 's4', 'float32', 4)})).toThrow(
    /at least 8 values/
  );
  expect(() => create({seedPositions: undefined})).toThrow(/seedPositions or seedMask/);
  expect(() =>
    create({
      seedPositions: undefined,
      seedMask: createTransientView(graph, 'm', 'uint32', CELLS),
      seedCount: createTransientView(graph, 'c', 'uint32', 1)
    })
  ).toThrow(/require seedPositions/);
  expect(() => create({seedIds: createTransientView(graph, 'ids3', 'uint32', 3)})).toThrow(
    /seedIds length/
  );
  expect(() => create({seedCount: createTransientView(graph, 'c2', 'uint32', 2)})).toThrow(
    /exactly one row/
  );
  expect(() => create({seedMask: createTransientView(graph, 'm2', 'uint32', 5)})).toThrow(
    /seedMask must contain one row per cell/
  );
  expect(() =>
    create({
      output: {distances: createTransientView(graph, 'd2', 'float32', 5)}
    })
  ).toThrow(/output.distances/);
  const distances = createTransientView(graph, 'd3', 'float32', CELLS);
  expect(() =>
    create({
      output: {
        distances,
        allocation: createTransientView(graph, 'a3', 'uint32', CELLS),
        withinDistance: graph.createDataView(distances.buffer, {
          format: 'uint32',
          length: CELLS
        })
      }
    })
  ).toThrow(/separate buffers/);
  const parameters = createTransientView(graph, 's8', 'float32', 8);
  expect(() =>
    create({
      parameters,
      output: {
        distances: graph.createDataView(parameters.buffer, {
          format: 'float32',
          length: CELLS
        })
      }
    })
  ).toThrow();
  const foreign = new GPUDistanceField(createProps(otherGraph));
  expect(() => foreign.getCommandNodes(graph)).toThrow(/target graph/);
});

it('the CPU oracle snaps seeds like the GPU and breaks ties on the smallest ID', () => {
  const settings = getGPUDistanceFieldParameterValues(
    {},
    makeGPURasterMetadata({width: 5, height: 2, rowDirection: 'north'})
  );
  const positions = Float32Array.of(0.5, 0.5, 4.5, 0.5, 0.99, 0.2, -1, 0, Number.NaN, 1);
  const cellSeeds = getCellSeeds({
    width: 5,
    height: 2,
    settings,
    positions,
    ids: Uint32Array.of(4, 3, 1, 0, 0)
  });
  expect(Array.from(cellSeeds)).toEqual([1, NONE, NONE, NONE, 3, NONE, NONE, NONE, NONE, NONE]);
  const result = computeDistanceFieldOnCPU({
    width: 5,
    height: 1,
    settings,
    positions: Float32Array.of(0.5, 0.5, 4.5, 0.5),
    ids: Uint32Array.of(7, 2)
  });
  expect(Array.from(result.allocation)).toEqual([7, 7, 2, 2, 2]);
  expect(Array.from(result.distances)).toEqual([0, 1, 2, 1, 0]);
  expect(getUlpDistance(1, 1 + 2 ** -23)).toBe(1);
  expect(getUlpDistance(Infinity, Infinity)).toBe(0);
});

it('rule 12: every jump-flood step reads its offset from a step word', () => {
  const device = createNullWebGPUDevice();
  for (const [width, height] of [
    [20, 10],
    [70, 9],
    [300, 260]
  ]) {
    const graph = new GPUCommandGraph(device);
    const nodes = new GPUDistanceField({
      ...createProps(graph),
      grid: makeGPURasterMetadata({width, height}),
      output: {distances: createTransientView(graph, 'distances-sized', 'float32', width * height)},
      mode: 'jump-flood'
    }).getCommandNodes(graph);
    const steps = getRasterPipelines(nodes).filter(pipeline =>
      /-jump-flood-\d+$/.test(pipeline.id)
    );
    expect(steps.length).toBeGreaterThan(4);
    expect(new Set(steps.map(pipeline => pipeline.key)).size).toBe(1);
    expect(steps[0].source).not.toMatch(/const STEP:/);
  }
  device.destroy();
});
