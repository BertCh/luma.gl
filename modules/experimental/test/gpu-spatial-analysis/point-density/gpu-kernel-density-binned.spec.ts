// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUStudyWindowPreparation} from '../../../src/gpu-spatial-analysis/study-window/gpu-study-window-preparation';
import {ORACLE_RECTANGLE, ORACLE_L_SHAPE} from '../../analysis-core/study-window-oracle';
import {
  getPolygonKernelDensityOracle,
  getPolygonKernelDensityWeights,
  type PolygonKernelDensityInput
} from './point-density-oracle';
import {getPolygonBinnedDensityOracle} from './kernel-density-binned-oracle';
import {
  getOracleKernelMassFan,
  isOracleInsideWindow
} from '../../analysis-core/study-window-oracle';
import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  getGPUKernelDensityParameterValues,
  GPUKernelDensity,
  type GPUKernelDensityEdgeCorrection,
  type GPUKernelDensityNormalization
} from '../../../src/gpu-spatial-analysis/point-density';
import type {GPUKernelDensityMethod} from '../../../src/gpu-spatial-analysis/point-density/kernel-density-math';
import {getGPUKernelDensityBinnedErrorBound} from '../../../src/gpu-spatial-analysis/point-density/kernel-density-math';
import fixtureJson from '../../oracles/fixtures/geometry-kernel-density-binned.json';
import {decodeOracleFixture} from '../../oracles/oracle-fixture';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {makeExtentGrid} from '../spatial-interpolation/extent-grid';

/**
 * GPU `method: 'binned'` and `'auto'` versus the float64 gaussian kernel sum (numpy, kernel
 * normalisation cross-checked with scipy.stats.multivariate_normal, window mass with
 * scipy.stats.norm; see test/oracles/geometry-kernel-density-binned.py).
 *
 * Tolerances:
 * - exact path (bandwidth of at most 6 cells): `2e-4 * max|expected|`, the existing exact tolerance
 *   (f32 accumulation and the f32 erfc fit);
 * - binned path (40 to 200 cells): `getGPUKernelDensityBinnedErrorBound(cell / h) * W / (2 pi h^2)`,
 *   derived in kernel-density-math.ts (linear binning r^2/4, filter shape 0.016, truncation, f32),
 *   with a 'uniform' edge correction multiplying the bound by the largest `1 / e(u)` of the
 *   compared cells.
 */
const fixture = decodeOracleFixture<{
  width: number;
  height: number;
  cellSize: number;
  window: [number, number, number, number];
  points: number[];
  weights: number[];
  weightSum: number;
  excludedPositions: number[];
  excludedWeights: number[];
  cells: [number, number][];
  cases: {
    bandwidthCells: number;
    bandwidth: number;
    edgeCorrection: GPUKernelDensityEdgeCorrection;
    expected: number[];
  }[];
}>(fixtureJson);

const {width, height, cellSize, window: WINDOW} = fixture;
const CELL_COUNT = width * height;
const EXACT_RELATIVE_TOLERANCE = 2e-4;
/** The largest bandwidth at which `'auto'` still gathers: `5 h` of at most 32 cells. */
const AUTO_EXACT_CELLS = 6;

type Run = {
  values: Float32Array;
  weightSum: number;
  invalidCount: number;
  insufficientPadding?: number;
};

type Session = {
  run: (bandwidthCells: number) => Promise<Run>;
  destroy: () => void;
};

/** Builds one compiled graph; `run` rewrites the per-frame parameters and re-submits. */
function createSession(
  device: Device,
  options: {
    method: GPUKernelDensityMethod;
    edgeCorrection: GPUKernelDensityEdgeCorrection;
    normalization?: GPUKernelDensityNormalization;
    binnedPadding?: number;
    readPadding?: boolean;
  }
): Session {
  // The rows every kernel excludes follow the oracle points.
  const positions = [...fixture.points, ...fixture.excludedPositions];
  const weights = [...fixture.weights, ...fixture.excludedWeights];
  const pointCount = positions.length / 2;
  const grid = makeExtentGrid(width, height, WINDOW);
  const buffers: Buffer[] = [];
  const track = (buffer: Buffer) => {
    buffers.push(buffer);
    return buffer;
  };
  const parameterBuffer = new GPUParameterBuffer(device, {
    id: 'kernel-density-binned-parameters',
    format: 'float32',
    length: 9
  });
  const valuesOut = track(createOutputBuffer(device, CELL_COUNT));
  const weightSumOut = track(createOutputBuffer(device, 1));
  const invalidCountOut = track(createOutputBuffer(device, 1));
  const paddingOut = track(createOutputBuffer(device, 1));
  const graph = new GPUCommandGraph(device, {id: 'kernel-density-binned-graph'});
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
      weights: importGraphBuffer(
        graph,
        'weights',
        track(createInputBuffer(device, Float32Array.from(weights))),
        'float32',
        pointCount
      ),
      parameters: parameterBuffer.importToGraph(graph),
      grid,
      indexGridSize: [16, 12],
      indexBounds: WINDOW,
      method: options.method,
      binnedPadding: options.binnedPadding,
      edgeCorrection: options.edgeCorrection,
      normalization: options.normalization,
      output: {
        values: importGraphBuffer(graph, 'values-out', valuesOut, 'float32', CELL_COUNT),
        weightSum: importGraphBuffer(graph, 'weight-sum-out', weightSumOut, 'float32', 1),
        status: {
          invalidCount: importGraphBuffer(graph, 'invalid-count-out', invalidCountOut, 'uint32', 1),
          insufficientPadding:
            options.readPadding && options.method !== 'exact'
              ? importGraphBuffer(graph, 'padding-out', paddingOut, 'uint32', 1)
              : undefined
        }
      }
    })
  );
  const compiled = graph.compile();
  return {
    run: async bandwidthCells => {
      parameterBuffer.write(
        getGPUKernelDensityParameterValues({bandwidth: bandwidthCells * cellSize}, grid)
      );
      submitGraph(device, compiled, undefined);
      return {
        values: Float32Array.from(await readFloat32(valuesOut, CELL_COUNT)),
        weightSum: (await readFloat32(weightSumOut, 1))[0],
        invalidCount: (await readUint32(invalidCountOut, 1))[0],
        insufficientPadding:
          options.readPadding && options.method !== 'exact'
            ? (await readUint32(paddingOut, 1))[0]
            : undefined
      };
    },
    destroy: () => {
      compiled.destroy();
      parameterBuffer.destroy();
      for (const buffer of buffers) {
        buffer.destroy();
      }
    }
  };
}

/** Margin of 2.5 bandwidths (plus the guard cell) for the largest bandwidth of a session. */
function getPadding(bandwidthCells: number): number {
  return Math.ceil(2.5 * bandwidthCells);
}

function getCase(bandwidthCells: number, edgeCorrection: GPUKernelDensityEdgeCorrection) {
  const found = fixture.cases.find(
    candidate =>
      candidate.bandwidthCells === bandwidthCells && candidate.edgeCorrection === edgeCorrection
  );
  expect(found, `fixture case ${bandwidthCells} ${edgeCorrection}`).toBeDefined();
  return found!;
}

/** Window mass of a cell center, for the uniform-correction bound (gaussian CDF differences). */
function getWindowMassFactor(bandwidth: number, cell: [number, number]): number {
  const cdf = (z: number) => 0.5 * (1 + erf(z / Math.SQRT2));
  const centerX = (cell[0] + 0.5) * cellSize;
  const centerY = (cell[1] + 0.5) * cellSize;
  return (
    (cdf((WINDOW[2] - centerX) / bandwidth) - cdf((WINDOW[0] - centerX) / bandwidth)) *
    (cdf((WINDOW[3] - centerY) / bandwidth) - cdf((WINDOW[1] - centerY) / bandwidth))
  );
}

/** Abramowitz and Stegun 7.1.26, absolute error 1.5e-7: enough for a tolerance factor. */
function erf(x: number): number {
  const t = 1 / (1 + 0.3275911 * Math.abs(x));
  const polynomial =
    ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
  return Math.sign(x) * (1 - polynomial * Math.exp(-x * x));
}

function expectBinned(
  run: Run,
  bandwidthCells: number,
  edgeCorrection: GPUKernelDensityEdgeCorrection
) {
  const testCase = getCase(bandwidthCells, edgeCorrection);
  const bandwidth = testCase.bandwidth;
  const scale = fixture.weightSum / (2 * Math.PI * bandwidth * bandwidth);
  const baseBound = getGPUKernelDensityBinnedErrorBound(cellSize / bandwidth) * scale;
  for (const [index, cell] of fixture.cells.entries()) {
    const bound =
      edgeCorrection === 'uniform' ? baseBound / getWindowMassFactor(bandwidth, cell) : baseBound;
    const actual = run.values[cell[1] * width + cell[0]];
    expect(
      Math.abs(actual - testCase.expected[index]),
      `h = ${bandwidthCells} cells ${edgeCorrection} cell ${cell}`
    ).toBeLessThanOrEqual(bound);
  }
}

function expectExact(
  run: Run,
  bandwidthCells: number,
  edgeCorrection: GPUKernelDensityEdgeCorrection
) {
  const testCase = getCase(bandwidthCells, edgeCorrection);
  const scale = Math.max(...testCase.expected.map(Math.abs));
  for (const [index, cell] of fixture.cells.entries()) {
    const actual = run.values[cell[1] * width + cell[0]];
    expect(
      Math.abs(actual - testCase.expected[index]),
      `exact h = ${bandwidthCells} cells ${edgeCorrection} cell ${cell}`
    ).toBeLessThanOrEqual(EXACT_RELATIVE_TOLERANCE * scale);
  }
}

it('binned matches the float64 kernel sum within the derived bound at 40 to 200 cells', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const edgeCorrection of ['none', 'uniform'] as const) {
    for (const bandwidthCells of [40, 80, 200]) {
      const session = createSession(device, {
        method: 'binned',
        edgeCorrection,
        binnedPadding: getPadding(bandwidthCells),
        readPadding: true
      });
      try {
        const run = await session.run(bandwidthCells);
        expectBinned(run, bandwidthCells, edgeCorrection);
        // Rows with a NaN position, a NaN weight and a negative weight are excluded and counted.
        expect(run.invalidCount).toBe(3);
        expect(run.weightSum).toBeCloseTo(fixture.weightSum, 2);
        expect(run.insufficientPadding).toBe(0);
      } finally {
        session.destroy();
      }
    }
  }
});

it('binned reports a margin shortfall, and auto stays exact when the margin is short', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const binned = createSession(device, {
    method: 'binned',
    edgeCorrection: 'none',
    binnedPadding: 0,
    readPadding: true
  });
  try {
    expect((await binned.run(80)).insufficientPadding).toBe(1);
  } finally {
    binned.destroy();
  }
  // Auto with no margin never takes the binned branch, so 80 cells is still within the exact
  // tolerance.
  const auto = createSession(device, {
    method: 'auto',
    edgeCorrection: 'none',
    binnedPadding: 0,
    readPadding: true
  });
  try {
    const run = await auto.run(80);
    expectExact(run, 80, 'none');
    expect(run.insufficientPadding).toBe(0);
  } finally {
    auto.destroy();
  }
});

it('auto gathers up to 6 cells and blurs above, from one compiled graph', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const edgeCorrection of ['none', 'uniform'] as const) {
    const session = createSession(device, {
      method: 'auto',
      edgeCorrection,
      binnedPadding: getPadding(200)
    });
    try {
      // Alternate paths on one compiled graph: the gate is re-armed every encoding.
      for (const bandwidthCells of [2, 80, 6, 200, 40, 2]) {
        const run = await session.run(bandwidthCells);
        if (bandwidthCells <= AUTO_EXACT_CELLS) {
          expectExact(run, bandwidthCells, edgeCorrection);
        } else {
          expectBinned(run, bandwidthCells, edgeCorrection);
        }
      }
    } finally {
      session.destroy();
    }
  }
});

it('exact is unchanged by the method option', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const edgeCorrection of ['none', 'uniform'] as const) {
    const session = createSession(device, {method: 'exact', edgeCorrection});
    try {
      for (const bandwidthCells of [2, 6]) {
        expectExact(await session.run(bandwidthCells), bandwidthCells, edgeCorrection);
      }
    } finally {
      session.destroy();
    }
  }
});

it('binned probability divides by the included weight', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const bandwidthCells = 80;
  const session = createSession(device, {
    method: 'binned',
    edgeCorrection: 'none',
    normalization: 'probability',
    binnedPadding: getPadding(bandwidthCells)
  });
  try {
    const run = await session.run(bandwidthCells);
    const testCase = getCase(bandwidthCells, 'none');
    const bandwidth = testCase.bandwidth;
    const bound =
      getGPUKernelDensityBinnedErrorBound(cellSize / bandwidth) / (2 * Math.PI * bandwidth ** 2);
    for (const [index, cell] of fixture.cells.entries()) {
      expect(
        Math.abs(
          run.values[cell[1] * width + cell[0]] - testCase.expected[index] / fixture.weightSum
        )
      ).toBeLessThanOrEqual(bound);
    }
  } finally {
    session.destroy();
  }
});

it('binned gives NaN for a bandwidth the packer cannot reject (GPU-computed NaN)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // bandwidthRule is undefined for fewer than two included rows: every cell is NaN, as for 'exact'.
  const positions = Float32Array.from([10, 10]);
  const grid = makeExtentGrid(8, 8, [0, 0, 8, 8]);
  const parameterBuffer = new GPUParameterBuffer(device, {
    id: 'kde-nan-parameters',
    format: 'float32',
    length: 9
  });
  const valuesOut = createOutputBuffer(device, 64);
  const graph = new GPUCommandGraph(device, {id: 'kde-nan-graph'});
  const positionsBuffer = createInputBuffer(device, positions);
  graph.add(
    new GPUKernelDensity({
      id: 'kde-nan',
      positions: importGraphBuffer(graph, 'positions', positionsBuffer, 'float32x2', 1),
      parameters: parameterBuffer.importToGraph(graph),
      grid,
      indexGridSize: [2, 2],
      indexBounds: [0, 0, 16, 16],
      method: 'binned',
      binnedPadding: 8,
      bandwidthRule: 'scott',
      edgeCorrection: 'none',
      output: {values: importGraphBuffer(graph, 'values', valuesOut, 'float32', 64)}
    })
  );
  const compiled = graph.compile();
  try {
    parameterBuffer.write(getGPUKernelDensityParameterValues({bandwidth: 1}, grid));
    submitGraph(device, compiled, undefined);
    const values = await readFloat32(valuesOut, 64);
    expect(values.every(Number.isNaN)).toBe(true);
  } finally {
    compiled.destroy();
    parameterBuffer.destroy();
    positionsBuffer.destroy();
    valuesOut.destroy();
  }
});

/** Executes one polygon-window fixture, or its rectangle counterpart, without CPU analysis. */
async function runPolygonDensity(
  device: Device,
  input: PolygonKernelDensityInput,
  options: {
    method: 'exact' | 'binned' | 'auto';
    rectangle?: readonly [number, number, number, number];
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

it('polygon rectangle matches rectangular binned and auto KDE with uniform and Diggle correction', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const method of ['binned', 'auto'] as const) {
    for (const edgeCorrection of ['uniform', 'diggle'] as const) {
      for (const bandwidth of method === 'auto' ? [100, 800] : [800]) {
        const input: PolygonKernelDensityInput = {
          window: ORACLE_RECTANGLE,
          kernel: 'gaussian',
          edgeCorrection,
          points: [100, 100, 500, 400, 1600, 750, -10, 500, 2100, 500],
          weights: [2, 1, 3, 100, 100],
          width: 24,
          height: 14,
          start: [-200, -200],
          cellSize: [100, 100],
          bandwidth,
          padding: 24
        };
        const polygon = await runPolygonDensity(device, input, {method});
        const rectangle = await runPolygonDensity(device, input, {
          method,
          rectangle: [0, 0, 2000, 1000]
        });
        expectPolygonField(polygon.values, rectangle.values, 2e-5);
        expect(polygon.weightSum).toBe(6);
        expect(polygon.invalidCount).toBe(2);
      }
    }
  }
});

it('polygon binned KDE matches its float64 model and stays within the exact error bound on an L-shape', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const edgeCorrection of ['uniform', 'diggle'] as const) {
    const input: PolygonKernelDensityInput = {
      window: ORACLE_L_SHAPE,
      kernel: 'gaussian',
      edgeCorrection,
      points: [100, 100, 600, 650, 1600, 200, 350, 1600, 1000, 1000, -100, 500],
      weights: [2, 1, 3, 4, 100, 100],
      width: 24,
      height: 24,
      start: [-200, -200],
      cellSize: [100, 100],
      bandwidth: 800,
      padding: 24
    };
    const actual = await runPolygonDensity(device, input, {method: 'binned'});
    expectPolygonField(actual.values, getPolygonBinnedDensityOracle(input), 2e-5);
    const included = getPolygonKernelDensityWeights(input);
    expect(actual.weightSum).toBe(included.weightSum);
    expect(actual.invalidCount).toBe(included.invalidCount);
    const exact = getPolygonKernelDensityOracle(input);
    const correctedWeight = included.weights.reduce((total, weight) => total + weight, 0);
    const bound =
      (getGPUKernelDensityBinnedErrorBound(100 / input.bandwidth) * correctedWeight) /
      (2 * Math.PI * input.bandwidth ** 2);
    for (let index = 0; index < actual.values.length; index++) {
      const center = [
        input.start[0] + ((index % input.width) + 0.5) * 100,
        input.start[1] + (Math.floor(index / input.width) + 0.5) * 100
      ] as const;
      if (!isOracleInsideWindow(input.window, center)) {
        continue;
      }
      const mass =
        edgeCorrection === 'uniform'
          ? getOracleKernelMassFan(input.window, center, input.bandwidth, 'gaussian')
          : 1;
      expect(Math.abs(actual.values[index] - exact[index])).toBeLessThanOrEqual(bound / mass);
    }
  }
});
