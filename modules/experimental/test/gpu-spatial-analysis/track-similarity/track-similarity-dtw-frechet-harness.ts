// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {GPUTrackSimilarity} from '../../../src/gpu-spatial-analysis/track-similarity/index';
import type {GPUSpatialContext} from '../../../src/analysis-core/contracts/spatial-context';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {decodeOracleFixture} from '../../oracles/oracle-fixture';
import fixtureJson from '../../oracles/fixtures/track-similarity-dtw-frechet.json';

/** One case of `test/oracles/track-similarity-dtw-frechet.py`. */
export type DynamicProgramCase = {
  name: string;
  a: [number, number][];
  b: [number, number][];
  frechet: number;
  dynamicTimeWarping: number;
  banded: {radius: number; dynamicTimeWarping: number}[];
  decisions: {epsilon: number; within: number}[];
};

/** The decoded oracle fixture. */
export const DYNAMIC_PROGRAM_FIXTURE = decodeOracleFixture<{
  tolerances: {
    frechetRelative: number;
    frechetAbsolutePlanar: number;
    frechetAbsoluteMeters: number;
    dynamicTimeWarpingRelativePerStep: number;
    dynamicTimeWarpingAbsolutePlanar: number;
    dynamicTimeWarpingAbsoluteMetersPerStep: number;
  };
  planar: DynamicProgramCase[];
  greatCircle: DynamicProgramCase[];
}>(fixtureJson);

function packSet(polylines: readonly (readonly (readonly number[])[])[]) {
  const positions: number[] = [];
  const offsets = [0];
  for (const polyline of polylines) {
    for (const [x, y] of polyline) {
      positions.push(x, y);
    }
    offsets.push(positions.length / 2);
  }
  return {positions: Float32Array.from(positions), offsets: Uint32Array.from(offsets)};
}

/**
 * Compiles one `GPUTrackSimilarity` with the Frechet distance, the Frechet decision and dynamic
 * time warping over pairs `(a[i], b[i])`; `run(epsilon)` rewrites the per-frame epsilon only.
 * `outputMode` isolates decision or distance work for benchmarks; only that result is meaningful.
 */
export function createDynamicProgramHarness(
  device: Device,
  setA: readonly (readonly (readonly number[])[])[],
  setB: readonly (readonly (readonly number[])[])[],
  options: {
    spatialContext?: GPUSpatialContext;
    bandRadius?: number;
    maximumFrechetVertexCount?: number;
    outputMode?: 'all' | 'decision' | 'distance';
  } = {}
) {
  const graph = new GPUCommandGraph(device, {id: 'similarity-dynamic-programs'});
  const buffers: Buffer[] = [];
  const input = (values: Float32Array | Uint32Array) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return buffer;
  };
  const output = (length: number) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return buffer;
  };
  const packedA = packSet(setA);
  const packedB = packSet(setB);
  const pairCount = setA.length;
  const epsilon = new GPUParameterBuffer(device, {id: 'epsilon', format: 'float32', length: 1});
  const out = {
    frechet: output(pairCount),
    within: output(pairCount),
    dynamicTimeWarping: output(pairCount),
    status: output(pairCount)
  };
  const pairs = Uint32Array.from({length: pairCount}, (_, index) => index);
  graph.add(
    new GPUTrackSimilarity({
      spatialContext: options.spatialContext,
      linesA: {
        kind: 'lines',
        positions: importGraphBuffer(graph, 'positions-a', input(packedA.positions), 'float32x2'),
        lineOffsets: importGraphBuffer(graph, 'offsets-a', input(packedA.offsets), 'uint32')
      },
      linesB: {
        kind: 'lines',
        positions: importGraphBuffer(graph, 'positions-b', input(packedB.positions), 'float32x2'),
        lineOffsets: importGraphBuffer(graph, 'offsets-b', input(packedB.offsets), 'uint32')
      },
      pairA: importGraphBuffer(graph, 'pair-a', input(pairs), 'uint32'),
      pairB: importGraphBuffer(graph, 'pair-b', input(pairs), 'uint32'),
      frechetEpsilon: options.outputMode === 'distance' ? undefined : epsilon.importToGraph(graph),
      dynamicTimeWarpingBandRadius: options.bandRadius,
      maximumFrechetVertexCount: options.maximumFrechetVertexCount,
      output: {
        frechet:
          options.outputMode === 'decision'
            ? undefined
            : importGraphBuffer(graph, 'o-frechet', out.frechet, 'float32', pairCount),
        frechetWithin:
          options.outputMode === 'distance'
            ? undefined
            : importGraphBuffer(graph, 'o-within', out.within, 'uint32', pairCount),
        dynamicTimeWarping:
          options.outputMode && options.outputMode !== 'all'
            ? undefined
            : importGraphBuffer(
                graph,
                'o-dynamic-time-warping',
                out.dynamicTimeWarping,
                'float32',
                pairCount
              ),
        pairStatus: importGraphBuffer(graph, 'o-status', out.status, 'uint32', pairCount)
      }
    })
  );
  const compiled = graph.compile();
  return {
    async run(epsilonValue: number) {
      epsilon.write(new Float32Array([epsilonValue]));
      submitGraph(device, compiled, undefined);
      return {
        frechet: await readFloat32(out.frechet, pairCount),
        within: await readUint32(out.within, pairCount),
        dynamicTimeWarping: await readFloat32(out.dynamicTimeWarping, pairCount),
        status: await readUint32(out.status, pairCount)
      };
    },
    /** Submits once more and waits for a one-word readback; for A/B timing. */
    async time(epsilonValue: number): Promise<number> {
      epsilon.write(new Float32Array([epsilonValue]));
      const start = performance.now();
      submitGraph(device, compiled, undefined);
      if (options.outputMode === 'distance') {
        await readFloat32(out.frechet, 1);
      } else {
        await readUint32(out.within, 1);
      }
      return performance.now() - start;
    },
    destroy() {
      compiled.destroy();
      epsilon.destroy();
      for (const buffer of buffers) {
        buffer.destroy();
      }
    }
  };
}
