// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUMapGraphParameterBuffer, importGraphBuffer, submitGraph} from '../../../src/map-graphs';
import {GPUCompositeScore} from '../../../src/map-graphs/composite-indicators/gpu-composite-score';
import {
  getGPUCompositeScoreParameterValues,
  GPU_COMPOSITE_SCORE_PARAMETER_LENGTH,
  type GPUCompositeScoreSettings
} from '../../../src/map-graphs/composite-indicators/composite-score-parameters';
import {createInputBuffer, createOutputBuffer, readFloat32} from '../map-graph-test-utils';
import {
  computeCompositeScoreOnCPU,
  type CompositeScoreOracleResult
} from './composite-score-oracle';

type Scene = {indicators: Float32Array; indicatorCount: number; mask?: Uint32Array};

type GPUResult = {
  score: number[];
  scaled: number[];
  columnStatistics: number[];
  loadings: number[];
  principalComponentSummary: number[];
};

type Fixture = {
  run(settings: GPUCompositeScoreSettings): Promise<GPUResult>;
  readonly rebuildCount: number;
  destroy(): void;
};

function createRandom(seed: number): () => number {
  let state = seed >>> 0 || 1;
  return () => {
    state ^= state << 13;
    state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5;
    state >>>= 0;
    return state / 4294967296;
  };
}

/** Correlated indicators with integer-rounded ties, a few NaN rows and a masked block. */
function createScene(rows: number, indicatorCount: number, seed: number): Scene {
  const random = createRandom(seed);
  const indicators = new Float32Array(rows * indicatorCount);
  const mask = new Uint32Array(rows).fill(1);
  for (let row = 0; row < rows; row++) {
    const latent = random() * 10;
    for (let column = 0; column < indicatorCount; column++) {
      const noise = random() * (column + 1);
      const value = column % 2 === 0 ? latent + noise : 20 - latent + noise;
      // Even columns are rounded so the rank scaler sees many ties.
      indicators[row * indicatorCount + column] = column % 2 === 0 ? Math.round(value) : value;
    }
    if (row % 97 === 5) {
      indicators[row * indicatorCount + (row % indicatorCount)] = NaN;
    }
    if (row % 50 === 7) {
      mask[row] = 0;
    }
  }
  return {indicators, indicatorCount, mask};
}

function createFixture(device: Device, scene: Scene): Fixture {
  const {indicatorCount} = scene;
  const rows = scene.indicators.length / indicatorCount;
  const buffers: Buffer[] = [];
  const track = (buffer: Buffer) => {
    buffers.push(buffer);
    return buffer;
  };
  const parameterBuffer = new GPUMapGraphParameterBuffer(device, {
    id: 'composite-parameters',
    format: 'float32',
    length: GPU_COMPOSITE_SCORE_PARAMETER_LENGTH
  });
  const outputs = {
    score: track(createOutputBuffer(device, rows)),
    scaled: track(createOutputBuffer(device, rows * indicatorCount)),
    columnStatistics: track(createOutputBuffer(device, indicatorCount * 4)),
    loadings: track(createOutputBuffer(device, indicatorCount)),
    principalComponentSummary: track(createOutputBuffer(device, 3))
  };
  const graph = new GPUCommandGraph(device, {id: 'composite-graph'});
  const float = (name: keyof typeof outputs, length: number) =>
    importGraphBuffer(graph, `out-${name}`, outputs[name], 'float32', length);
  graph.add(
    new GPUCompositeScore({
      id: 'composite',
      indicators: importGraphBuffer(
        graph,
        'indicators',
        track(createInputBuffer(device, scene.indicators)),
        'float32',
        rows * indicatorCount
      ),
      indicatorCount,
      mask: scene.mask
        ? importGraphBuffer(
            graph,
            'mask',
            track(createInputBuffer(device, scene.mask)),
            'uint32',
            rows
          )
        : undefined,
      parameters: parameterBuffer.importToGraph(graph),
      enableRank: true,
      enablePrincipalComponent: true,
      output: {
        score: float('score', rows),
        scaled: float('scaled', rows * indicatorCount),
        columnStatistics: float('columnStatistics', indicatorCount * 4),
        loadings: float('loadings', indicatorCount),
        principalComponentSummary: float('principalComponentSummary', 3)
      }
    })
  );
  let compileCount = 0;
  const compiled = graph.compile();
  compileCount++;
  return {
    async run(settings) {
      parameterBuffer.write(getGPUCompositeScoreParameterValues(settings));
      submitGraph(device, compiled, undefined);
      return {
        score: await readFloat32(outputs.score, rows),
        scaled: await readFloat32(outputs.scaled, rows * indicatorCount),
        columnStatistics: await readFloat32(outputs.columnStatistics, indicatorCount * 4),
        loadings: await readFloat32(outputs.loadings, indicatorCount),
        principalComponentSummary: await readFloat32(outputs.principalComponentSummary, 3)
      };
    },
    get rebuildCount() {
      return compileCount - 1;
    },
    destroy() {
      compiled.destroy();
      parameterBuffer.destroy();
      for (const buffer of buffers) {
        buffer.destroy();
      }
    }
  };
}

function expectClose(
  actual: ArrayLike<number>,
  expected: ArrayLike<number>,
  tolerance: number,
  label: string
): void {
  expect(actual.length, label).toBe(expected.length);
  for (let index = 0; index < expected.length; index++) {
    const want = expected[index];
    const got = actual[index];
    if (Number.isNaN(want)) {
      expect(Number.isNaN(got), `${label}[${index}] should be NaN, got ${got}`).toBe(true);
    } else {
      const scale = Math.max(1, Math.abs(want));
      expect(Math.abs(got - want), `${label}[${index}] ${got} vs ${want}`).toBeLessThanOrEqual(
        tolerance * scale
      );
    }
  }
}

function compare(result: GPUResult, oracle: CompositeScoreOracleResult, label: string): void {
  expectClose(result.columnStatistics, oracle.columnStatistics, 1e-4, `${label} statistics`);
  expectClose(result.scaled, oracle.scaled, 1e-4, `${label} scaled`);
  expectClose(result.score, oracle.score, 1e-4, `${label} score`);
}

const SETTINGS: GPUCompositeScoreSettings[] = [
  {scaler: 'min-max', aggregation: 'weighted-sum', weights: [1, 2, 0.5, 3, 1]},
  {scaler: 'z-score', aggregation: 'weighted-sum', weights: [1, -1, 2, 0, 1], directions: [1, -1]},
  {scaler: 'rank', aggregation: 'weighted-sum', weights: [1, 1, 1, 1, 1], directions: [-1, 1, -1]},
  {
    scaler: 'min-max',
    aggregation: 'weighted-geometric-mean',
    weights: [1, 2, 0, 3, 1],
    epsilon: 0.01
  },
  {
    scaler: 'rank',
    aggregation: 'weighted-geometric-mean',
    weights: [2, 1, 1, 1, 0.5],
    epsilon: 0.05
  },
  {aggregation: 'principal-component', weights: [], directions: [1, -1, 1, -1, 1]}
];

it('GPUCompositeScore matches the CPU oracle across scalers and aggregations without rebuilding', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scene = createScene(3000, 5, 11);
  const fixture = createFixture(device, scene);
  try {
    for (const settings of SETTINGS) {
      const label = `${settings.scaler ?? 'pca'}/${settings.aggregation}`;
      const result = await fixture.run(settings);
      const oracle = computeCompositeScoreOnCPU({...scene, settings});
      compare(result, oracle, label);
      if (settings.aggregation === 'principal-component') {
        expectClose(result.loadings, oracle.loadings, 1e-4, 'loadings');
        expectClose(
          result.principalComponentSummary.slice(0, 2),
          oracle.principalComponent,
          1e-4,
          'principal component'
        );
        expect(result.principalComponentSummary[2]).toBeLessThan(1e-3);
      }
    }
    expect(fixture.rebuildCount).toBe(0);
  } finally {
    fixture.destroy();
  }
});

it('GPUCompositeScore averages tied ranks and the output is bitwise reproducible', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scene = createScene(1500, 3, 3);
  const fixture = createFixture(device, scene);
  try {
    const settings: GPUCompositeScoreSettings = {scaler: 'rank', weights: [1, 1, 1]};
    const first = await fixture.run(settings);
    const oracle = computeCompositeScoreOnCPU({...scene, settings});
    // Ranks are exact integers; only the final f32 division (2.5 ULP in WGSL) may round.
    expectClose(first.scaled, oracle.scaled, 1e-6, 'ranks');
    const zScore = await fixture.run({scaler: 'z-score', weights: [1, 2, 3]});
    const again = await fixture.run({scaler: 'z-score', weights: [1, 2, 3]});
    expect(again.score).toEqual(zScore.score);
    expect(again.columnStatistics).toEqual(zScore.columnStatistics);
  } finally {
    fixture.destroy();
  }
});

it('GPUCompositeScore writes NaN for every row when nothing is included', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scene: Scene = {
    indicators: Float32Array.from([1, 2, 3, 4]),
    indicatorCount: 2,
    mask: Uint32Array.from([0, 0])
  };
  const fixture = createFixture(device, scene);
  try {
    const result = await fixture.run({weights: [1, 1]});
    expect(result.score.every(Number.isNaN)).toBe(true);
    expect(result.columnStatistics.every(Number.isNaN)).toBe(true);
  } finally {
    fixture.destroy();
  }
});
