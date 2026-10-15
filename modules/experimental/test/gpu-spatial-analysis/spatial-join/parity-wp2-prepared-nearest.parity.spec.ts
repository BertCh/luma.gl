// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {
  GPUSpatialJoinPrepared,
  GPUSpatialJoinCandidates,
  GPUNearestFeatureJoin
} from '../../../src/gpu-spatial-analysis/spatial-join';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {compareParity} from '../../oracles/parity-wp2-assert';
import {decodeOracleFixture} from '../../oracles/oracle-fixture';
import fixtureJson from '../../oracles/fixtures/parity-wp2-prepared-nearest.json';

type Neighbor = {row: number; id: number; distance: number; foot: number[]};
type Frame = {
  name: string;
  positions: number[];
  queries: number[];
  candidates: number[][];
  dwithin: number[][];
  neighbors: Neighbor[][];
};
type JoinCase = {
  name: string;
  radius: number;
  lineOffsets: number[];
  sourceIds: number[];
  absolute: number;
  frames: Frame[];
};
const fixture = decodeOracleFixture<{reference: string; cases: JoinCase[]}>(fixtureJson);

function createRun(
  device: Device,
  entry: JoinCase,
  pairCapacity: number,
  neighborCapacity: number,
  ties: 'all' | 'lowest-id'
) {
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const input = <Format extends 'float32x2' | 'float32' | 'uint32'>(
    name: string,
    values: Float32Array | Uint32Array,
    format: Format
  ) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return {
      buffer,
      view: importGraphBuffer(
        graph,
        name,
        buffer,
        format,
        values.length / (format === 'float32x2' ? 2 : 1)
      )
    };
  };
  const output = <Format extends 'float32x2' | 'float32' | 'uint32'>(
    name: string,
    length: number,
    format: Format
  ) => {
    const buffer = createOutputBuffer(device, length * (format === 'float32x2' ? 2 : 1));
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, name, buffer, format, length)};
  };
  const firstFrame = entry.frames[0];
  const queryCount = firstFrame.queries.length / 2;
  const positions = input(
    'feature-positions',
    Float32Array.from(firstFrame.positions),
    'float32x2'
  );
  const queries = input('query-positions', Float32Array.from(firstFrame.queries), 'float32x2');
  const lineOffsets = input('line-offsets', Uint32Array.from(entry.lineOffsets), 'uint32');
  const sourceIds = input('source-ids', Uint32Array.from(entry.sourceIds), 'uint32');
  const radius = input('radius', Float32Array.of(entry.radius), 'float32');
  const right = {kind: 'lines' as const, positions: positions.view, lineOffsets: lineOffsets.view};
  const left = {kind: 'points' as const, positions: queries.view};
  const prepared = new GPUSpatialJoinPrepared({geometry: right, spatialSort: false});
  graph.add(prepared);
  const pairLeft = output('pair-left', pairCapacity, 'uint32');
  const pairRight = output('pair-right', pairCapacity, 'uint32');
  const pairCount = output('pair-count', 1, 'uint32');
  const pairRequired = output('pair-required', 1, 'uint32');
  const pairOverflow = output('pair-overflow', 1, 'uint32');
  graph.add(
    new GPUSpatialJoinCandidates({
      left,
      right,
      prepared,
      distance: entry.radius,
      output: {
        leftIds: pairLeft.view,
        rightIds: pairRight.view,
        status: {
          count: pairCount.view,
          requiredCount: pairRequired.view,
          overflow: pairOverflow.view
        }
      }
    })
  );
  const slotCount = queryCount * neighborCapacity;
  const neighborIds = output('neighbor-ids', slotCount, 'uint32');
  const neighborDistances = output('neighbor-distances', slotCount, 'float32');
  const neighborFootPoints = output('neighbor-foot', slotCount, 'float32x2');
  const neighborCounts = output('neighbor-counts', queryCount, 'uint32');
  const neighborOverflow = output('neighbor-overflow', 1, 'uint32');
  graph.add(
    new GPUNearestFeatureJoin({
      queries: left,
      features: {...right, sourceIds: sourceIds.view},
      prepared,
      k: 1,
      neighborCapacity,
      ties,
      radius: radius.view,
      output: {
        neighborIds: neighborIds.view,
        neighborDistances: neighborDistances.view,
        neighborFootPoints: neighborFootPoints.view,
        neighborCounts: neighborCounts.view,
        status: {overflow: neighborOverflow.view}
      }
    })
  );
  const compiled = graph.compile();
  return {
    prepared,
    async run(frame: Frame, invalidate: boolean) {
      positions.buffer.write(Float32Array.from(frame.positions));
      queries.buffer.write(Float32Array.from(frame.queries));
      if (invalidate) prepared.invalidate();
      submitGraph(device, compiled, undefined);
      return {
        pairCount: (await readUint32(pairCount.buffer, 1))[0],
        pairRequired: (await readUint32(pairRequired.buffer, 1))[0],
        pairOverflow: (await readUint32(pairOverflow.buffer, 1))[0],
        pairLeft: await readUint32(pairLeft.buffer, pairCapacity),
        pairRight: await readUint32(pairRight.buffer, pairCapacity),
        neighborIds: await readUint32(neighborIds.buffer, slotCount),
        neighborDistances: await readFloat32(neighborDistances.buffer, slotCount),
        neighborFootPoints: await readFloat32(neighborFootPoints.buffer, slotCount * 2),
        neighborCounts: await readUint32(neighborCounts.buffer, queryCount),
        neighborOverflow: (await readUint32(neighborOverflow.buffer, 1))[0]
      };
    },
    destroy() {
      compiled.destroy();
      prepared.destroy();
      for (const buffer of buffers) buffer.destroy();
    }
  };
}

for (const entry of fixture.cases) {
  it(`prepared candidates and nearest Shapely parity: ${entry.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fullPairCapacity = Math.max(...entry.frames.map(frame => frame.candidates.length)) + 1;
    const fullNeighborCapacity =
      Math.max(
        ...entry.frames.flatMap(frame => frame.neighbors.map(neighbors => neighbors.length))
      ) + 1;
    const configurations = [
      {
        pairCapacity: fullPairCapacity,
        neighborCapacity: fullNeighborCapacity,
        ties: 'all' as const
      },
      {pairCapacity: 1, neighborCapacity: 1, ties: 'all' as const},
      {pairCapacity: fullPairCapacity, neighborCapacity: 1, ties: 'lowest-id' as const}
    ];
    try {
      for (const configuration of configurations) {
        const {pairCapacity, neighborCapacity, ties} = configuration;
        const run = createRun(device, entry, pairCapacity, neighborCapacity, ties);
        try {
          // Repeated first frame proves reuse; moved queries must not reuse query results.
          for (const frameIndex of [0, 0, 1, 2]) {
            const frame = entry.frames[frameIndex];
            const actual = await run.run(frame, frameIndex === 2);
            const caseName = `${entry.name}/${frame.name}/pairs=${pairCapacity}/neighbors=${neighborCapacity}/${ties}`;
            const compare = (
              className: string,
              metric: string,
              measured: ArrayLike<number>,
              expected: ArrayLike<number>,
              absolute = 0
            ) =>
              compareParity(
                className,
                caseName,
                fixture.reference,
                metric,
                measured,
                expected,
                absolute
              );
            compare(
              'GPUSpatialJoinPrepared',
              'build count after reuse/invalidation',
              [run.prepared.encodedBuildCount],
              [frameIndex === 2 ? 2 : 1]
            );
            const expectedPairs = frame.candidates.slice(0, pairCapacity);
            compare(
              'GPUSpatialJoinCandidates',
              'candidate count/required/overflow',
              [actual.pairCount, actual.pairRequired, actual.pairOverflow],
              [
                expectedPairs.length,
                frame.candidates.length,
                Number(frame.candidates.length > pairCapacity)
              ]
            );
            compare(
              'GPUSpatialJoinCandidates',
              'every bounding-box candidate pair',
              actual.pairLeft
                .slice(0, actual.pairCount)
                .flatMap((left, index) => [left, actual.pairRight[index]]),
              expectedPairs.flat()
            );
            if (pairCapacity >= frame.candidates.length) {
              const pairSet = new Set(
                actual.pairLeft
                  .slice(0, actual.pairCount)
                  .map((left, index) => `${left},${actual.pairRight[index]}`)
              );
              compare(
                'GPUSpatialJoinPrepared',
                'every exact dwithin pair retained by index',
                frame.dwithin.map(pair => Number(pairSet.has(pair.join(',')))),
                frame.dwithin.map(() => 1)
              );
            }
            const expectedCounts: number[] = [];
            const expectedIds: number[] = [];
            const expectedDistances: number[] = [];
            const expectedFeet: number[] = [];
            for (const neighbors of frame.neighbors) {
              const selected = neighbors.slice(0, ties === 'lowest-id' ? 1 : neighborCapacity);
              expectedCounts.push(selected.length);
              for (let slot = 0; slot < neighborCapacity; slot++) {
                const neighbor = selected[slot];
                expectedIds.push(neighbor?.id ?? 0xffffffff);
                expectedDistances.push(neighbor?.distance ?? -1);
                expectedFeet.push(...(neighbor?.foot ?? [NaN, NaN]));
              }
            }
            compare(
              'GPUNearestFeatureJoin',
              'tie overflow flag',
              [actual.neighborOverflow],
              [
                Number(
                  ties === 'all' &&
                    frame.neighbors.some(neighbors => neighbors.length > neighborCapacity)
                )
              ]
            );
            compare(
              'GPUNearestFeatureJoin',
              'every neighbor count',
              actual.neighborCounts,
              expectedCounts
            );
            compare(
              'GPUNearestFeatureJoin',
              'every nearest ID including ties and sentinels',
              actual.neighborIds,
              expectedIds
            );
            compare(
              'GPUNearestFeatureJoin',
              'every nearest distance',
              actual.neighborDistances,
              expectedDistances,
              entry.absolute
            );
            compare(
              'GPUNearestFeatureJoin',
              'every nearest foot point and validity mask',
              actual.neighborFootPoints,
              expectedFeet,
              entry.absolute
            );
          }
        } finally {
          run.destroy();
        }
      }
    } finally {
      device.destroy?.();
    }
  }, 120_000);
}
