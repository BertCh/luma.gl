// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Chicago polygon-isotropic cost, with persistent window preparation and CSR envelopes.
 * Run with:
 * LUMA_TEST_BROWSER_BENCHMARKS=true npx vitest run --project headless --no-file-parallelism --silent=false modules/experimental/test/gpu-dataframe/pair-statistics/ripley-isotropic-envelope-benchmark.spec.ts
 *
 * Times are wall-clock encode + submit + queue completion, excluding compilation and readback.
 * "window build" is the first FULL encode (including the consumer and first-use GPU overhead),
 * not an isolated preparation dispatch. Later encodes must reuse that build. Envelopes include
 * the observed K pass. Each variant compiles once, warms up three times, then takes seven samples.
 * The optional 199-simulation pair runs only if extrapolation from 39 fits a 60-second budget
 * and the suite has spent less than five minutes. Its omission is explicit in the single table.
 */

import type {Buffer} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPURipley} from '../../../src/gpu-dataframe/pair-statistics/gpu-ripley';
import {GPURipleyEnvelope} from '../../../src/gpu-dataframe/pair-statistics/gpu-ripley-envelope';
import {
  getGPURipleyParameterValues,
  GPU_RIPLEY_PARAMETER_LENGTH,
  GPU_RIPLEY_ENVELOPE_STATISTICS_LENGTH
} from '../../../src/gpu-dataframe/pair-statistics/ripley-parameters';
import {GPUStudyWindowPreparation} from '../../../src/gpu-spatial-analysis/study-window/gpu-study-window-preparation';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {
  CHICAGO_WINDOW,
  CHICAGO_WINDOW_AREA,
  CHICAGO_WINDOW_BOUNDS,
  createChicagoPattern
} from './chicago-window-fixture';

const RADIUS_COUNT = 30;
const SEED = 20261014;
const WARMUP_COUNT = 3;
const SAMPLE_COUNT = 7;
const ENCODE_COUNT = 1 + WARMUP_COUNT + SAMPLE_COUNT;

type Measurement = {
  milliseconds: number;
  firstEncodeMilliseconds: number;
  elapsedMilliseconds: number;
};
type Variant = {
  polygon: boolean;
  correction: 'isotropic' | 'border';
  simulationCount?: number;
};

async function measureVariant(
  device: NonNullable<Awaited<ReturnType<typeof getWebGPUTestDevice>>>,
  positions: Float32Array,
  maximumDistance: number,
  variant: Variant
): Promise<Measurement> {
  const started = performance.now();
  const graph = new GPUCommandGraph(device, {id: 'ripley-isotropic-envelope-benchmark'});
  const buffers: Buffer[] = [];
  const importInput = <Format extends 'float32x2' | 'uint32'>(
    name: string,
    format: Format,
    values: Float32Array | Uint32Array,
    length: number
  ) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return importGraphBuffer(graph, name, buffer, format, length);
  };
  const createOutput = (name: string, length: number) => {
    const buffer = createOutputBuffer(device, length);
    // Detect passes that silently leave caller-owned outputs unwritten.
    buffer.write(new Float32Array(length).fill(NaN));
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, name, buffer, 'float32', length)};
  };
  const parameters = new GPUParameterBuffer(device, {
    id: 'parameters',
    format: 'float32',
    length: GPU_RIPLEY_PARAMETER_LENGTH,
    values: getGPURipleyParameterValues({
      bounds: variant.polygon ? 'study-window' : CHICAGO_WINDOW_BOUNDS,
      maximumDistance,
      edgeCorrection: variant.correction,
      seed: SEED
    })
  });
  let compiled: ReturnType<GPUCommandGraph['compile']> | undefined;
  try {
    const windowSummary = variant.polygon ? createOutput('window-summary', 8) : undefined;
    const studyWindow = variant.polygon
      ? new GPUStudyWindowPreparation({
          positions: importInput(
            'window-positions',
            'float32x2',
            Float32Array.from(CHICAGO_WINDOW.positions),
            CHICAGO_WINDOW.positions.length / 2
          ),
          ringOffsets: importInput(
            'window-rings',
            'uint32',
            Uint32Array.from(CHICAGO_WINDOW.ringOffsets),
            CHICAGO_WINDOW.ringOffsets.length
          ),
          polygonOffsets: importInput(
            'window-polygons',
            'uint32',
            Uint32Array.from(CHICAGO_WINDOW.polygonOffsets),
            CHICAGO_WINDOW.polygonOffsets.length
          ),
          featureOffsets: importInput(
            'window-features',
            'uint32',
            Uint32Array.from(CHICAGO_WINDOW.featureOffsets),
            CHICAGO_WINDOW.featureOffsets.length
          ),
          samplingGrid: [64, 64],
          output: {summary: windowSummary!.view}
        })
      : undefined;
    if (studyWindow) graph.add(studyWindow);
    const common = {
      studyWindow,
      positions: importInput('positions', 'float32x2', positions, positions.length / 2),
      parameters: parameters.importToGraph(graph),
      gridSize: [
        Math.max(
          1,
          Math.ceil((CHICAGO_WINDOW_BOUNDS[2] - CHICAGO_WINDOW_BOUNDS[0]) / maximumDistance)
        ),
        Math.max(
          1,
          Math.ceil((CHICAGO_WINDOW_BOUNDS[3] - CHICAGO_WINDOW_BOUNDS[1]) / maximumDistance)
        )
      ] as const,
      radiusCount: RADIUS_COUNT
    };
    const observed = createOutput('observed-k', RADIUS_COUNT);
    graph.add(new GPURipley({...common, output: {k: observed.view}}));
    const finiteOutputs = [observed];
    let statistics: ReturnType<typeof createOutput> | undefined;
    let unplaced: Buffer | undefined;
    if (variant.simulationCount) {
      const lower = createOutput('lower', RADIUS_COUNT);
      const upper = createOutput('upper', RADIUS_COUNT);
      statistics = createOutput('statistics', GPU_RIPLEY_ENVELOPE_STATISTICS_LENGTH);
      unplaced = createOutputBuffer(device, 1);
      unplaced.write(new Uint32Array([0xffffffff]));
      buffers.push(unplaced);
      finiteOutputs.push(lower, upper, statistics);
      graph.add(
        new GPURipleyEnvelope({
          ...common,
          simulationCount: variant.simulationCount,
          observedK: observed.view,
          statistic: 'k',
          output: {
            lower: lower.view,
            upper: upper.view,
            statistics: statistics.view,
            unplacedCount: importGraphBuffer(graph, 'unplaced', unplaced, 'uint32', 1)
          }
        })
      );
    }
    compiled = graph.compile();
    // Drain unrelated queued work before starting the first-encode clock.
    await device.handle.queue.onSubmittedWorkDone();
    const samples: number[] = [];
    let firstEncodeMilliseconds = 0;
    for (let run = 0; run < ENCODE_COUNT; run++) {
      const start = performance.now();
      submitGraph(device, compiled, undefined);
      await device.handle.queue.onSubmittedWorkDone();
      const elapsed = performance.now() - start;
      if (run === 0) firstEncodeMilliseconds = elapsed;
      else if (run > WARMUP_COUNT) samples.push(elapsed);
    }
    for (const {buffer, view} of finiteOutputs) {
      const values = await readFloat32(buffer, view.length);
      expect(values.every(Number.isFinite)).toBe(true);
    }
    expect((await readFloat32(observed.buffer, RADIUS_COUNT)).every(value => value > 0)).toBe(true);
    if (windowSummary && studyWindow) {
      const summary = await readFloat32(windowSummary.buffer, 8);
      expect(summary.every(Number.isFinite)).toBe(true);
      expect(Math.abs(summary[4] / CHICAGO_WINDOW_AREA - 1)).toBeLessThan(1e-4);
      expect(summary[7]).toBe(0);
      expect(studyWindow.encodedBuildCount).toBe(1);
    }
    if (statistics && unplaced) {
      const summary = await readFloat32(statistics.buffer, GPU_RIPLEY_ENVELOPE_STATISTICS_LENGTH);
      expect(summary[0]).toBe(positions.length / 2);
      expect(summary[1]).toBe(variant.simulationCount);
      expect(await readUint32(unplaced, 1)).toEqual([0]);
    }
    samples.sort((first, second) => first - second);
    return {
      milliseconds: samples[Math.floor(samples.length / 2)],
      firstEncodeMilliseconds,
      elapsedMilliseconds: performance.now() - started
    };
  } finally {
    compiled?.destroy();
    parameters.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
}

it('Chicago isotropic Ripley and envelope cost benchmark', async context => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    context.skip('no WebGPU test device for the opt-in benchmark');
    return;
  }
  const started = performance.now();
  const rows = [
    'n | maxD (m) | variant | simulations | median ms/encode | window build (first full encode) ms | c/b | d/e'
  ];
  const append = (
    count: number,
    maximumDistance: number,
    name: string,
    simulations: number,
    result: Measurement,
    ratio = ''
  ) => {
    rows.push(
      `${count} | ${maximumDistance} | ${name} | ${simulations || '-'} | ${result.milliseconds.toFixed(3)} | ${name === 'a bbox isotropic' ? '-' : result.firstEncodeMilliseconds.toFixed(3)} | ${name.startsWith('c ') ? ratio : ''} | ${name.startsWith('d ') ? ratio : ''}`
    );
  };
  for (const count of [2000, 8000]) {
    const positions = createChicagoPattern(count, SEED);
    for (const maximumDistance of [2000, 5000]) {
      const rectangle = await measureVariant(device, positions, maximumDistance, {
        polygon: false,
        correction: 'isotropic'
      });
      append(count, maximumDistance, 'a bbox isotropic', 0, rectangle);
      const border = await measureVariant(device, positions, maximumDistance, {
        polygon: true,
        correction: 'border'
      });
      append(count, maximumDistance, 'b polygon border', 0, border);
      const isotropic = await measureVariant(device, positions, maximumDistance, {
        polygon: true,
        correction: 'isotropic'
      });
      append(
        count,
        maximumDistance,
        'c polygon isotropic',
        0,
        isotropic,
        (isotropic.milliseconds / border.milliseconds).toFixed(3)
      );
      const measureEnvelopes = async (simulationCount: number) => {
        const envelopeIsotropic = await measureVariant(device, positions, maximumDistance, {
          polygon: true,
          correction: 'isotropic',
          simulationCount
        });
        const envelopeBorder = await measureVariant(device, positions, maximumDistance, {
          polygon: true,
          correction: 'border',
          simulationCount
        });
        append(
          count,
          maximumDistance,
          'd envelope isotropic',
          simulationCount,
          envelopeIsotropic,
          (envelopeIsotropic.milliseconds / envelopeBorder.milliseconds).toFixed(3)
        );
        append(count, maximumDistance, 'e envelope border', simulationCount, envelopeBorder);
        return envelopeIsotropic.elapsedMilliseconds + envelopeBorder.elapsedMilliseconds;
      };
      const envelopeElapsed = await measureEnvelopes(39);
      if (envelopeElapsed * (200 / 40) < 60_000 && performance.now() - started < 300_000) {
        await measureEnvelopes(199);
      } else {
        rows.push(
          `${count} | ${maximumDistance} | d/e envelopes | 199 | skipped: projected pair cost exceeds budget | - | |`
        );
      }
    }
  }
  console.log(
    `Chicago Ripley (encode + submit + queue completion; envelopes include observed K)\n${rows.join('\n')}`
  );
}, 1_800_000);
