// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUStudyWindowPreparation} from '../../../src/gpu-spatial-analysis/study-window/gpu-study-window-preparation';
import {
  addOracleDegenerateSlots,
  ORACLE_RECTANGLE,
  ORACLE_L_SHAPE,
  getOracleWindowArea
} from '../../analysis-core/study-window-oracle';
import {
  getPolygonKernelDensityOracle,
  getPolygonKernelDensityWeights,
  type PolygonKernelDensityInput
} from './point-density-oracle';
import {createWGSLKernelNode} from '../../../src/utils/wgsl-kernel-nodes';
import {getStudyWindowBinding} from '../../../src/utils/study-window-wgsl';
import {getKernelDensityStudyWindowWGSL} from '../../../src/gpu-spatial-analysis/point-density/kernel-density-wgsl';
import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  getGPUKernelDensityParameterValues,
  GPUKernelDensity,
  type GPUKernelDensityEdgeCorrection,
  type GPUKernelDensityKernel,
  type GPUKernelDensityNormalization
} from '../../../src/gpu-spatial-analysis/point-density';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {makeExtentGrid} from '../spatial-interpolation/extent-grid';
import {KERNEL_DENSITY_ORACLE} from './kernel-density-oracle-values';
import {requireTestDevice} from '../../utils/require-test-device';

/**
 * GPU versus float64 oracle (sklearn / numpy / scipy dblquad, see kernel-density-oracle-values.ts):
 * `|gpu - oracle| <= 2e-4 * max|oracle|`. Covers f32 accumulation, the f32 erfc fit (1.2e-7), and
 * the f32 quadrant mass cancellation of the edge correction.
 */
const RELATIVE_TOLERANCE = 2e-4;

const {field} = KERNEL_DENSITY_ORACLE;
const WINDOW = field.window as unknown as [number, number, number, number];

type Run = {
  values: Float32Array;
  weightSum?: number;
  invalidCount?: number;
};

async function runKernelDensity(
  device: Device,
  options: {
    kernel: GPUKernelDensityKernel;
    edgeCorrection: GPUKernelDensityEdgeCorrection;
    normalization?: GPUKernelDensityNormalization;
    weighted: boolean;
    extraRows?: {positions: number[]; weights: number[]};
    readStatus?: boolean;
  }
): Promise<Run> {
  const positions = [...field.points, ...(options.extraRows?.positions ?? [])];
  const weights = [...field.weights, ...(options.extraRows?.weights ?? [])];
  const pointCount = positions.length / 2;
  const cellCount = field.columns * field.rows;
  // Rows advance north from the window minimum, as the old extent props did.
  const grid = makeExtentGrid(field.columns, field.rows, WINDOW);
  const buffers: Buffer[] = [];
  const track = (buffer: Buffer) => {
    buffers.push(buffer);
    return buffer;
  };
  const parameterBuffer = new GPUParameterBuffer(device, {
    id: 'kernel-density-parameters',
    format: 'float32',
    length: 9
  });
  const valuesOut = track(createOutputBuffer(device, cellCount));
  const weightSumOut = track(createOutputBuffer(device, 1));
  const invalidCountOut = track(createOutputBuffer(device, 1));
  const graph = new GPUCommandGraph(device, {id: 'kernel-density-graph'});
  graph.add(
    new GPUKernelDensity({
      id: 'kde',
      positions: importGraphBuffer(
        graph,
        'positions',
        track(createInputBuffer(device, Float32Array.from(positions))),
        'float32x2',
        pointCount
      ),
      weights: options.weighted
        ? importGraphBuffer(
            graph,
            'weights',
            track(createInputBuffer(device, Float32Array.from(weights))),
            'float32',
            pointCount
          )
        : undefined,
      parameters: parameterBuffer.importToGraph(graph),
      grid,
      indexGridSize: [4, 3],
      indexBounds: WINDOW,
      kernel: options.kernel,
      edgeCorrection: options.edgeCorrection,
      normalization: options.normalization,
      output: {
        values: importGraphBuffer(graph, 'values-out', valuesOut, 'float32', cellCount),
        weightSum: options.readStatus
          ? importGraphBuffer(graph, 'weight-sum-out', weightSumOut, 'float32', 1)
          : undefined,
        status: options.readStatus
          ? {
              invalidCount: importGraphBuffer(
                graph,
                'invalid-count-out',
                invalidCountOut,
                'uint32',
                1
              )
            }
          : undefined
      }
    })
  );
  const compiled = graph.compile();
  try {
    parameterBuffer.write(getGPUKernelDensityParameterValues({bandwidth: field.bandwidth}, grid));
    submitGraph(device, compiled, undefined);
    return {
      values: Float32Array.from(await readFloat32(valuesOut, cellCount)),
      weightSum: options.readStatus ? (await readFloat32(weightSumOut, 1))[0] : undefined,
      invalidCount: options.readStatus ? (await readUint32(invalidCountOut, 1))[0] : undefined
    };
  } finally {
    compiled.destroy();
    parameterBuffer.destroy();
    for (const buffer of buffers) {
      buffer.destroy();
    }
  }
}

function expectField(actual: Float32Array, expected: readonly number[], label: string): void {
  const scale = Math.max(...expected.map(Math.abs));
  let worst = 0;
  for (const [cell, value] of expected.entries()) {
    worst = Math.max(worst, Math.abs(actual[cell] - value));
  }
  expect(worst, label).toBeLessThanOrEqual(RELATIVE_TOLERANCE * scale);
}

it('GPUKernelDensity matches sklearn, numpy and dblquad for every kernel and edge correction', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const testCase of field.cases) {
    const run = await runKernelDensity(device, {
      kernel: testCase.kernel as GPUKernelDensityKernel,
      edgeCorrection: testCase.edgeCorrection as GPUKernelDensityEdgeCorrection,
      weighted: testCase.weighted
    });
    expectField(
      run.values,
      testCase.values,
      `${testCase.kernel} ${testCase.edgeCorrection} weighted=${testCase.weighted}`
    );
  }
});

it('GPUKernelDensity probability output divides by the included weight', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const testCase = field.cases.find(
    candidate =>
      candidate.kernel === 'gaussian' && candidate.edgeCorrection === 'none' && candidate.weighted
  )!;
  const run = await runKernelDensity(device, {
    kernel: 'gaussian',
    edgeCorrection: 'none',
    normalization: 'probability',
    weighted: true,
    readStatus: true
  });
  expectField(
    run.values,
    testCase.values.map(value => value / testCase.weightSum),
    'probability'
  );
  expect(run.weightSum!).toBeCloseTo(testCase.weightSum, 4);
  expect(run.invalidCount).toBe(0);
});

it('GPUKernelDensity excludes non-finite, negative and out-of-domain rows and counts them', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const testCase = field.cases.find(
    candidate =>
      candidate.kernel === 'gaussian' && candidate.edgeCorrection === 'none' && candidate.weighted
  )!;
  // NaN position, negative weight, and a position outside indexBounds: all excluded.
  const run = await runKernelDensity(device, {
    kernel: 'gaussian',
    edgeCorrection: 'none',
    weighted: true,
    readStatus: true,
    extraRows: {positions: [Number.NaN, 1, 2, 2, 50, 50], weights: [1, -1, 1]}
  });
  expectField(run.values, testCase.values, 'excluded rows leave the field unchanged');
  expect(run.invalidCount).toBe(3);
  expect(run.weightSum!).toBeCloseTo(testCase.weightSum, 4);
});

/** Executes one polygon-window fixture, or its rectangle counterpart, without CPU analysis. */
async function runPolygonDensity(
  device: Device,
  input: PolygonKernelDensityInput,
  options: {
    method: 'exact' | 'binned' | 'auto';
    rectangle?: readonly [number, number, number, number];
    massOnly?: boolean;
  }
): Promise<{values: Float32Array; weightSum: number; invalidCount: number}> {
  const buffers: Buffer[] = [];
  const graph = new GPUCommandGraph(device);
  const upload = (values: Float32Array | Uint32Array) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return buffer;
  };
  const allocate = (length: number) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return buffer;
  };
  const valuesBuffer = allocate(input.width * input.height);
  const weightSumBuffer = allocate(1);
  const invalidBuffer = allocate(1);
  const grid = makeExtentGrid(input.width, input.height, [
    input.start[0],
    input.start[1],
    input.start[0] + input.width * input.cellSize[0],
    input.start[1] + input.height * input.cellSize[1]
  ]);
  const positions = importGraphBuffer(
    graph,
    'points',
    upload(Float32Array.from(input.points)),
    'float32x2',
    input.points.length / 2
  );
  const parameters = importGraphBuffer(
    graph,
    'parameters',
    upload(
      getGPUKernelDensityParameterValues(
        {
          bandwidth: input.bandwidth,
          window: options.rectangle ?? 'study-window'
        },
        grid
      )
    ),
    'float32',
    9
  );
  const values = importGraphBuffer(
    graph,
    'values',
    valuesBuffer,
    'float32',
    input.width * input.height
  );
  const studyWindow = new GPUStudyWindowPreparation({
    positions: importGraphBuffer(
      graph,
      'vertices',
      upload(Float32Array.from(input.window.positions)),
      'float32x2',
      input.window.positions.length / 2
    ),
    featureOffsets: importGraphBuffer(
      graph,
      'features',
      upload(
        input.window.featureOffsets ?? new Uint32Array([0, input.window.polygonOffsets.length - 1])
      ),
      'uint32',
      2
    ),
    polygonOffsets: importGraphBuffer(
      graph,
      'polygons',
      upload(input.window.polygonOffsets),
      'uint32',
      input.window.polygonOffsets.length
    ),
    ringOffsets: importGraphBuffer(
      graph,
      'rings',
      upload(input.window.ringOffsets),
      'uint32',
      input.window.ringOffsets.length
    ),
    samplingGrid: [8, 8]
  });
  if (!options.rectangle) {
    graph.add(studyWindow);
  }
  if (options.massOnly) {
    const views = studyWindow.getViews(graph);
    graph.add(
      createWGSLKernelNode(graph, {
        id: 'kernel-mass',
        operation: 'GPUKernelDensity',
        variant: 'mass-integral',
        bindings: [
          getStudyWindowBinding(views),
          {name: 'values', view: values, type: 'f32', access: 'read_write'}
        ],
        invocationCount: input.width * input.height,
        declarations: getKernelDensityStudyWindowWGSL(views, input.kernel),
        body: `let center = vec2f(${input.start[0]}.0, ${input.start[1]}.0) +
        (vec2f(f32(index % ${input.width}u), f32(index / ${input.width}u)) + vec2f(0.5)) * vec2f(${input.cellSize[0]}, ${input.cellSize[1]});
        values[valuesOffset + index] = getStudyWindowKernelMass(center, ${input.bandwidth}.0);`
      })
    );
  } else {
    graph.add(
      new GPUKernelDensity({
        positions,
        parameters,
        studyWindow: options.rectangle ? 'rectangle' : studyWindow,
        weights: input.weights
          ? importGraphBuffer(
              graph,
              'weights',
              upload(Float32Array.from(input.weights)),
              'float32',
              input.weights.length
            )
          : undefined,
        grid,
        indexGridSize: [8, 8],
        indexBounds: [-4000, -4000, 4000, 4000],
        kernel: input.kernel,
        edgeCorrection: input.edgeCorrection,
        method: options.method,
        normalization: input.normalization,
        binnedPadding: input.padding,
        output: {
          values,
          weightSum: importGraphBuffer(graph, 'weight-sum', weightSumBuffer, 'float32', 1),
          status: {invalidCount: importGraphBuffer(graph, 'invalid', invalidBuffer, 'uint32', 1)}
        }
      })
    );
  }
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    return {
      values: Float32Array.from(await readFloat32(valuesBuffer, input.width * input.height)),
      weightSum: (await readFloat32(weightSumBuffer, 1))[0],
      invalidCount: (await readUint32(invalidBuffer, 1))[0]
    };
  } finally {
    compiled.destroy();
    studyWindow.destroy();
    for (const buffer of buffers) {
      buffer.destroy();
    }
  }
}

/** Compare every finite cell relatively, including exact agreement of the outside NaN mask. */
function expectPolygonField(
  actual: Float32Array,
  expected: ArrayLike<number>,
  tolerance: number
): void {
  for (let index = 0; index < actual.length; index++) {
    expect(Number.isNaN(actual[index]), `NaN mask at ${index}`).toBe(Number.isNaN(expected[index]));
    if (Number.isFinite(expected[index])) {
      expect(Math.abs(actual[index] - expected[index]), `cell ${index}`).toBeLessThanOrEqual(
        tolerance * Math.abs(expected[index])
      );
    }
  }
}

it('polygon rectangle matches rectangle KDE for all kernels and corrections', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const kernel of ['gaussian', 'quartic', 'epanechnikov'] as const) {
    for (const edgeCorrection of ['uniform', 'diggle'] as const) {
      const input: PolygonKernelDensityInput = {
        window: ORACLE_RECTANGLE,
        kernel,
        edgeCorrection,
        points: [100, 100, 500, 400, 1600, 750, -10, 500, 2100, 500],
        weights: [2, 1, 3, 100, 100],
        width: 24,
        height: 14,
        start: [-200, -200],
        cellSize: [100, 100],
        bandwidth: 400
      };
      const polygon = await runPolygonDensity(device, input, {method: 'exact'});
      const rectangle = await runPolygonDensity(device, input, {
        method: 'exact',
        rectangle: [0, 0, 2000, 1000]
      });
      expectPolygonField(polygon.values, rectangle.values, 2e-5);
      expect(polygon.weightSum).toBe(6);
      expect(polygon.invalidCount).toBe(2);
    }
  }
});

it('polygon exact KDE matches fan oracle on an L-shape and excludes outside points', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const kernel of ['gaussian', 'quartic', 'epanechnikov'] as const) {
    for (const edgeCorrection of ['uniform', 'diggle'] as const) {
      const input: PolygonKernelDensityInput = {
        window: ORACLE_L_SHAPE,
        kernel,
        edgeCorrection,
        normalization: 'probability',
        points: [100, 100, 600, 650, 1600, 200, 350, 1600, 1000, 1000, -100, 500],
        weights: [2, 1, 3, 4, 100, 100],
        width: 24,
        height: 24,
        start: [-200, -200],
        cellSize: [100, 100],
        bandwidth: 600
      };
      const actual = await runPolygonDensity(device, input, {method: 'exact'});
      expectPolygonField(actual.values, getPolygonKernelDensityOracle(input), 2e-5);
      const included = getPolygonKernelDensityWeights(input);
      expect(actual.weightSum).toBe(included.weightSum);
      expect(actual.invalidCount).toBe(included.invalidCount);
    }
  }
});

it('polygon kernel mass integrates to window area over a grid containing its support', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const kernel of ['gaussian', 'quartic', 'epanechnikov'] as const) {
    const input: PolygonKernelDensityInput = {
      window: ORACLE_L_SHAPE,
      kernel,
      edgeCorrection: 'uniform',
      points: [100, 100],
      width: 80,
      height: 80,
      start: [-1000, -1000],
      cellSize: [50, 50],
      bandwidth: 150
    };
    const actual = await runPolygonDensity(device, input, {method: 'exact', massOnly: true});
    const area = actual.values.reduce((total, mass) => total + mass * 2500, 0);
    expect(Math.abs(area / getOracleWindowArea(input.window) - 1)).toBeLessThan(0.01);
  }
});

it('polygon KDE mass and corrections ignore empty and degenerate clip slots', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const window = {
    positions: new Float64Array([
      ...ORACLE_L_SHAPE.positions,
      150,
      150,
      350,
      150,
      350,
      350,
      150,
      350
    ]),
    featureOffsets: new Uint32Array([0, 1]),
    polygonOffsets: new Uint32Array([0, 2]),
    ringOffsets: new Uint32Array([0, 6, 10])
  };
  for (const kernel of ['gaussian', 'quartic', 'epanechnikov'] as const) {
    const input: PolygonKernelDensityInput = {
      window,
      kernel,
      edgeCorrection: 'uniform',
      points: [80, 90, 430, 200, 550, 480, 900, 310],
      width: 10,
      height: 10,
      start: [0, 0],
      cellSize: [200, 200],
      bandwidth: 600
    };
    for (const massOnly of [true, false]) {
      const clean = await runPolygonDensity(device, input, {method: 'exact', massOnly});
      const clipped = await runPolygonDensity(
        device,
        {...input, window: addOracleDegenerateSlots(window)},
        {method: 'exact', massOnly}
      );
      expectPolygonField(clipped.values, clean.values, 1e-6);
    }
  }
});
