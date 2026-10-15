// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {it} from 'vitest';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {GPUTrajectoryEncounters} from '../../../src/gpu-spatial-analysis/trajectory-encounters/index';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {decodeOracleFixture} from '../../oracles/oracle-fixture';
import {compareParity} from '../../oracles/parity-wp2-assert';
import fixtureJson from '../../oracles/fixtures/parity-wp2-trajectory-encounters.json';

type EncounterCase = {
  name: string;
  samples: number[];
  bucketTimes: number[];
  trackCount: number;
  bucketCount: number;
  trackValid: number[];
  bounds: [number, number, number, number];
  distance: number;
  cellSize: number;
  hitCount: number;
  distanceAbsolute: number;
  expected: Record<'ids' | 'partners' | 'first' | 'distances' | 'counts' | 'times', number[]>;
};

const fixture = decodeOracleFixture<{reference: string; cases: EncounterCase[]}>(fixtureJson);
type RunOptions = {
  bounds: [number, number, number, number];
  bucketTimes: number[];
  distance: number;
  cellSize: number;
  hitCapacity: number;
  pairCapacity: number;
  trackValid?: Uint32Array;
};

async function runEncounters(
  device: Device,
  samples: Float32Array,
  trackCount: number,
  bucketCount: number,
  options: RunOptions
) {
  const {pairCapacity} = options;
  const graph = new GPUCommandGraph(device, {id: 'encounters-test'});
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
  const bucketTimes = Float32Array.from(options.bucketTimes);
  const outputs = {
    ids: output(pairCapacity),
    partners: output(pairCapacity),
    first: output(pairCapacity),
    distances: output(pairCapacity),
    counts: output(pairCapacity),
    times: output(pairCapacity),
    count: output(1),
    overflow: output(1),
    candidateOverflow: output(1),
    total: output(1)
  };
  graph.add(
    new GPUTrajectoryEncounters({
      id: 'encounters',
      samples: importGraphBuffer(graph, 'samples', input(samples), 'float32x2', samples.length / 2),
      trackCount,
      bucketCount,
      trackValid: options.trackValid
        ? importGraphBuffer(graph, 'valid', input(options.trackValid), 'uint32', trackCount)
        : undefined,
      distance: importGraphBuffer(
        graph,
        'distance',
        input(Float32Array.of(options.distance)),
        'float32',
        1
      ),
      cellSize: options.cellSize,
      bounds: options.bounds,
      hitCapacity: options.hitCapacity,
      bucketTimes: importGraphBuffer(
        graph,
        'bucket-times',
        input(bucketTimes),
        'float32',
        bucketCount
      ),
      output: {
        pairs: {
          ids: importGraphBuffer(graph, 'o-ids', outputs.ids, 'uint32', pairCapacity),
          status: {
            count: importGraphBuffer(graph, 'o-count', outputs.count, 'uint32', 1),
            overflow: importGraphBuffer(graph, 'o-overflow', outputs.overflow, 'uint32', 1),
            requiredCount: importGraphBuffer(graph, 'o-total', outputs.total, 'uint32', 1)
          },
          partners: importGraphBuffer(
            graph,
            'o-partners',
            outputs.partners,
            'uint32',
            pairCapacity
          ),
          firstBuckets: importGraphBuffer(graph, 'o-first', outputs.first, 'uint32', pairCapacity),
          minimumDistances: importGraphBuffer(
            graph,
            'o-distances',
            outputs.distances,
            'float32',
            pairCapacity
          ),
          bucketCounts: importGraphBuffer(
            graph,
            'o-counts',
            outputs.counts,
            'uint32',
            pairCapacity
          ),
          firstTimes: importGraphBuffer(graph, 'o-times', outputs.times, 'float32', pairCapacity)
        },
        status: {
          candidateOverflow: importGraphBuffer(
            graph,
            'o-candidate-overflow',
            outputs.candidateOverflow,
            'uint32',
            1
          )
        }
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const [count] = await readUint32(outputs.count, 1);
  const [overflow] = await readUint32(outputs.overflow, 1);
  const [candidateOverflow] = await readUint32(outputs.candidateOverflow, 1);
  const [requiredCount] = await readUint32(outputs.total, 1);
  const result = {
    count,
    overflow,
    candidateOverflow,
    requiredCount,
    ids: await readUint32(outputs.ids, pairCapacity),
    partners: await readUint32(outputs.partners, pairCapacity),
    first: await readUint32(outputs.first, pairCapacity),
    distances: await readFloat32(outputs.distances, pairCapacity),
    counts: await readUint32(outputs.counts, pairCapacity),
    times: await readFloat32(outputs.times, pairCapacity),
    bucketTimes
  };
  compiled.destroy();
  for (const buffer of buffers) {
    buffer.destroy();
  }
  return result;
}

for (const entry of fixture.cases) {
  it(`GPUTrajectoryEncounters independent parity: ${entry.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const compare = (
      metric: string,
      actual: ArrayLike<number>,
      expected: ArrayLike<number>,
      absolute = 0,
      relative = 0
    ) =>
      compareParity(
        'GPUTrajectoryEncounters',
        entry.name,
        fixture.reference,
        metric,
        actual,
        expected,
        absolute,
        relative
      );
    try {
      const pairCount = entry.expected.ids.length;
      for (const pairCapacity of [pairCount + 1, 1]) {
        const actual = await runEncounters(
          device,
          Float32Array.from(entry.samples),
          entry.trackCount,
          entry.bucketCount,
          {
            ...entry,
            trackValid: Uint32Array.from(entry.trackValid),
            hitCapacity: Math.max(16, 2 ** Math.ceil(Math.log2(entry.hitCount + 1))),
            pairCapacity
          }
        );
        const count = Math.min(pairCount, pairCapacity);
        compare(
          `status capacity=${pairCapacity}`,
          [actual.count, actual.requiredCount, actual.overflow, actual.candidateOverflow],
          [count, pairCount, Number(pairCount > pairCapacity), 0]
        );
        for (const column of ['ids', 'partners', 'first', 'counts', 'times'] as const) {
          compare(
            `${column} capacity=${pairCapacity}`,
            actual[column].slice(0, count),
            entry.expected[column].slice(0, count)
          );
        }
        compare(
          `distances capacity=${pairCapacity}`,
          actual.distances.slice(0, count),
          entry.expected.distances.slice(0, count),
          entry.distanceAbsolute,
          2e-6
        );
        if (count < pairCapacity) {
          compare(
            'tail sentinels',
            [
              actual.ids[count],
              actual.partners[count],
              actual.first[count],
              actual.counts[count],
              actual.times[count],
              actual.distances[count]
            ],
            [0xffffffff, 0xffffffff, 0xffffffff, 0, 0, 0]
          );
        }
      }
      const starved = await runEncounters(
        device,
        Float32Array.from(entry.samples),
        entry.trackCount,
        entry.bucketCount,
        {
          ...entry,
          trackValid: Uint32Array.from(entry.trackValid),
          hitCapacity: 1,
          pairCapacity: pairCount + 1
        }
      );
      compare('candidate capacity overflow', [starved.candidateOverflow, starved.overflow], [1, 0]);
    } finally {
      device.destroy?.();
    }
  }, 120_000);
}
