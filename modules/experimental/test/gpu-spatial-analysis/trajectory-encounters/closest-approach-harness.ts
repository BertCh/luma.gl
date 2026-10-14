// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {getInt64TimeWords} from '../../../src/gpu-dataframe/time-window-filter/time-words';
import {GPUTrajectoryClosestApproach} from '../../../src/gpu-spatial-analysis/trajectory-encounters/index';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import type {ClosestApproachTrack} from './closest-approach-oracle';

/** GPU readback of one `GPUTrajectoryClosestApproach` run. */
export type ClosestApproachGPUResult = {
  count: number;
  overflow: number;
  requiredCount: number;
  ids: number[];
  partners: number[];
  minimumDistances: number[];
  closestTimes: number[];
  firstTimes: number[];
  durations: number[];
  entryOverflow: number;
  hitOverflow: number;
  medianEncodingMilliseconds?: number;
};

/**
 * Runs one closest-approach graph. With `wordBase`, timestamps are uploaded as Int64 words
 * `wordBase + time` (times must then be integers).
 */
export async function runClosestApproach(
  device: Device,
  tracks: readonly ClosestApproachTrack[],
  options: {
    distance: number;
    cellSize: number;
    timeCellSize: number;
    entryCapacity: number;
    hitCapacity: number;
    pairCapacity: number;
    wordBase?: bigint;
    timedEncodings?: number;
  }
): Promise<ClosestApproachGPUResult> {
  const graph = new GPUCommandGraph(device, {id: 'closest-approach-test'});
  const buffers: Buffer[] = [];
  const input = (values: Float32Array | Uint32Array) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return buffer;
  };
  const output = (length: number) => {
    const buffer = createOutputBuffer(device, length);
    buffer.write(new Uint32Array(Math.max(length, 1)).fill(0x7f7f7f7f));
    buffers.push(buffer);
    return buffer;
  };
  const positions = Float32Array.from(tracks.flatMap(track => track.positions.flat()));
  const times = tracks.flatMap(track => track.times);
  const trackOffsets = [0];
  for (const track of tracks) {
    trackOffsets.push(trackOffsets[trackOffsets.length - 1] + track.positions.length);
  }
  const rowCount = times.length;
  const {pairCapacity} = options;
  const out = {
    ids: output(pairCapacity),
    count: output(1),
    overflow: output(1),
    total: output(1),
    partners: output(pairCapacity),
    minimumDistances: output(pairCapacity),
    closestTimes: output(pairCapacity),
    firstTimes: output(pairCapacity),
    durations: output(pairCapacity),
    entryOverflow: output(1),
    hitOverflow: output(1)
  };
  const view = <Format extends 'uint32' | 'float32'>(
    name: keyof typeof out,
    format: Format,
    length: number
  ) => importGraphBuffer(graph, `o-${name}`, out[name], format, length);
  const timestampBuffer = input(
    options.wordBase === undefined
      ? Float32Array.from(times)
      : getInt64TimeWords(
          BigInt64Array.from(times, time => (options.wordBase as bigint) + BigInt(time))
        )
  );
  graph.add(
    new GPUTrajectoryClosestApproach({
      positions: importGraphBuffer(graph, 'positions', input(positions), 'float32x2', rowCount),
      timestamps:
        options.wordBase === undefined
          ? importGraphBuffer(graph, 'timestamps', timestampBuffer, 'float32', rowCount)
          : importGraphBuffer(graph, 'timestamps', timestampBuffer, 'uint32x2', rowCount),
      trackOffsets: importGraphBuffer(
        graph,
        'track-offsets',
        input(Uint32Array.from(trackOffsets)),
        'uint32'
      ),
      distance: importGraphBuffer(
        graph,
        'distance',
        input(Float32Array.of(options.distance)),
        'float32',
        1
      ),
      cellSize: options.cellSize,
      timeCellSize: options.timeCellSize,
      entryCapacity: options.entryCapacity,
      hitCapacity: options.hitCapacity,
      output: {
        pairs: {
          ids: view('ids', 'uint32', pairCapacity),
          status: {
            count: view('count', 'uint32', 1),
            overflow: view('overflow', 'uint32', 1),
            requiredCount: view('total', 'uint32', 1)
          },
          partners: view('partners', 'uint32', pairCapacity),
          minimumDistances: view('minimumDistances', 'float32', pairCapacity),
          closestTimes: view('closestTimes', 'float32', pairCapacity),
          firstTimes: view('firstTimes', 'float32', pairCapacity),
          durations: view('durations', 'float32', pairCapacity)
        },
        status: {
          entryOverflow: view('entryOverflow', 'uint32', 1),
          hitOverflow: view('hitOverflow', 'uint32', 1)
        }
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  let medianEncodingMilliseconds: number | undefined;
  if (options.timedEncodings) {
    await readUint32(out.count, 1);
    const samples: number[] = [];
    for (let encoding = 0; encoding < options.timedEncodings; encoding++) {
      const start = performance.now();
      submitGraph(device, compiled, undefined);
      await readUint32(out.count, 1);
      samples.push(performance.now() - start);
    }
    samples.sort((left, right) => left - right);
    medianEncodingMilliseconds = samples[Math.floor(samples.length / 2)];
  }
  const [count] = await readUint32(out.count, 1);
  const result: ClosestApproachGPUResult = {
    count,
    overflow: (await readUint32(out.overflow, 1))[0],
    requiredCount: (await readUint32(out.total, 1))[0],
    ids: await readUint32(out.ids, pairCapacity),
    partners: await readUint32(out.partners, pairCapacity),
    minimumDistances: await readFloat32(out.minimumDistances, pairCapacity),
    closestTimes: await readFloat32(out.closestTimes, pairCapacity),
    firstTimes: await readFloat32(out.firstTimes, pairCapacity),
    durations: await readFloat32(out.durations, pairCapacity),
    entryOverflow: (await readUint32(out.entryOverflow, 1))[0],
    hitOverflow: (await readUint32(out.hitOverflow, 1))[0],
    medianEncodingMilliseconds
  };
  compiled.destroy();
  for (const buffer of buffers) {
    buffer.destroy();
  }
  return result;
}
