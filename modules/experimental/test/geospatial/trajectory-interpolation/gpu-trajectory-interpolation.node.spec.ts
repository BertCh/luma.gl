// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  getGPUTrajectoryPlayheadParameterValues,
  getGPUTrajectoryPlayheadWordParameterValues,
  GPUTrajectoryPlayhead,
  GPUTrajectoryResample,
  GPU_TRAJECTORY_PLAYHEAD_PARAMETER_LENGTH,
  GPU_TRAJECTORY_PLAYHEAD_STATUS as STATUS,
  type GPUTrajectoryPlayheadProps,
  type GPUTrajectoryResampleProps
} from '../../../src/geospatial/trajectory-interpolation';
import {joinTimeWords} from '../../../src/gpu-dataframe/time-window-filter/time-words';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {EDGE_CASE_TRACKS, packTracks} from './trajectory-interpolation-fixtures';
import {computePlayheadOracle, computeResampleOracle} from './trajectory-interpolation-oracle';

const NO_ROW = 0xffffffff;
let uniqueIndex = 0;
const unique = (id: string) => `${id}-${uniqueIndex++}`;

function createPlayheadProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUTrajectoryPlayheadProps> = {}
): GPUTrajectoryPlayheadProps {
  return {
    positions: createTransientView(graph, unique('positions'), 'float32x2', 8),
    timestamps: createTransientView(graph, unique('timestamps'), 'float32', 8),
    trackOffsets: createTransientView(graph, unique('offsets'), 'uint32', 3),
    parameters: createTransientView(graph, unique('parameters'), 'float32', 4),
    currentPositions: createTransientView(graph, unique('current'), 'float32x2', 2),
    ...overrides
  };
}

function createResampleProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUTrajectoryResampleProps> = {}
): GPUTrajectoryResampleProps {
  return {
    positions: createTransientView(graph, unique('positions'), 'float32x2', 8),
    timestamps: createTransientView(graph, unique('timestamps'), 'float32', 8),
    trackOffsets: createTransientView(graph, unique('offsets'), 'uint32', 3),
    sampleCount: 4,
    samples: createTransientView(graph, unique('samples'), 'float32x2', 8),
    ...overrides
  };
}

it('getGPUTrajectoryPlayheadParameterValues packs float32 and word playheads', () => {
  expect(GPU_TRAJECTORY_PLAYHEAD_PARAMETER_LENGTH).toBe(4);
  expect(Array.from(getGPUTrajectoryPlayheadParameterValues({playhead: 12.5, maxGap: 3}))).toEqual([
    12.5, 3, 0, 0
  ]);
  expect(Array.from(getGPUTrajectoryPlayheadParameterValues({playhead: -1}))).toEqual([
    -1, 0, 0, 0
  ]);
  const base = 1_700_000_000_000n;
  const words = getGPUTrajectoryPlayheadWordParameterValues({
    playhead: base,
    maxGap: 60_000
  });
  expect(joinTimeWords(words[0], words[1])).toBe(base);
  const floats = new Float32Array(words.buffer);
  expect([floats[2], floats[3]]).toEqual([0, 60_000]);
  const fractional = getGPUTrajectoryPlayheadWordParameterValues({
    playhead: -2.25
  });
  expect(joinTimeWords(fractional[0], fractional[1])).toBe(-3n);
  expect(new Float32Array(fractional.buffer)[2]).toBe(0.75);
  const target = new Uint32Array(6).fill(9);
  getGPUTrajectoryPlayheadWordParameterValues({playhead: 5n}, target);
  expect(Array.from(target)).toEqual([5, 0, 0, 0, 9, 9]);

  expect(() => getGPUTrajectoryPlayheadParameterValues({playhead: Number.NaN})).toThrow(/finite/);
  expect(() => getGPUTrajectoryPlayheadParameterValues({playhead: 1n})).toThrow(/numbers/);
  expect(() => getGPUTrajectoryPlayheadParameterValues({playhead: 1, maxGap: -1})).toThrow(
    /maxGap/
  );
  expect(() =>
    getGPUTrajectoryPlayheadWordParameterValues({
      playhead: 1,
      maxGap: Infinity
    })
  ).toThrow(/maxGap/);
  expect(() => getGPUTrajectoryPlayheadParameterValues({playhead: 1}, new Float32Array(3))).toThrow(
    /4 elements/
  );
});

it('GPUTrajectoryPlayhead validates inputs and outputs', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  expect(() => new GPUTrajectoryPlayhead(createPlayheadProps(graph))).not.toThrow();
  expect(
    () =>
      new GPUTrajectoryPlayhead(
        createPlayheadProps(graph, {
          timestamps: createTransientView(graph, unique('t'), 'float32', 7)
        })
      )
  ).toThrow(/timestamps length/);
  expect(
    () =>
      new GPUTrajectoryPlayhead(
        createPlayheadProps(graph, {
          timestamps: createTransientView(graph, unique('t'), 'uint32x2', 8)
        })
      )
  ).toThrow(/uint32 parameters/);
  expect(
    () =>
      new GPUTrajectoryPlayhead(
        createPlayheadProps(graph, {
          parameters: createTransientView(graph, unique('p'), 'float32', 3)
        })
      )
  ).toThrow(/4 float32/);
  expect(
    () =>
      new GPUTrajectoryPlayhead(
        createPlayheadProps(graph, {
          trackOffsets: createTransientView(graph, unique('o'), 'uint32', 1)
        })
      )
  ).toThrow(/two rows/);
  expect(
    () =>
      new GPUTrajectoryPlayhead(
        createPlayheadProps(graph, {
          speeds: createTransientView(graph, unique('s'), 'float32', 3)
        })
      )
  ).toThrow(/track count/);
  expect(
    () =>
      new GPUTrajectoryPlayhead(
        createPlayheadProps(graph, {
          currentElevations: createTransientView(graph, unique('z'), 'float32', 2)
        })
      )
  ).toThrow(/requires elevations/);
  expect(
    () =>
      new GPUTrajectoryPlayhead(
        createPlayheadProps(graph, {
          drawInstanceCount: createTransientView(graph, unique('d'), 'uint32', 1)
        })
      )
  ).toThrow(/requires activeTracks/);
  expect(
    () => new GPUTrajectoryPlayhead(createPlayheadProps(graph, {currentPositions: undefined}))
  ).toThrow(/at least one output/);
  const positions = createTransientView(graph, unique('positions'), 'float32x2', 2);
  expect(
    () =>
      new GPUTrajectoryPlayhead(
        createPlayheadProps(graph, {
          positions,
          timestamps: createTransientView(graph, unique('t'), 'float32', 2),
          currentPositions: positions
        })
      )
  ).toThrow(/share buffers/);
});

it('GPUTrajectoryPlayhead emits deterministic nodes and skips unused kernels', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const minimal = new GPUTrajectoryPlayhead(createPlayheadProps(graph, {id: 'minimal'}));
  expect(minimal.getCommandNodes(graph).map(node => node.id)).toEqual([
    'minimal-bracket',
    'minimal-geometry'
  ]);
  const full = new GPUTrajectoryPlayhead(
    createPlayheadProps(graph, {
      id: 'full',
      timestamps: createTransientView(graph, unique('t'), 'uint32x2', 8),
      parameters: createTransientView(graph, unique('p'), 'uint32', 4),
      currentPositions: undefined,
      status: createTransientView(graph, unique('status'), 'uint32', 2),
      activeTracks: {
        ids: createTransientView(graph, unique('ids'), 'uint32', 2),
        count: createTransientView(graph, unique('count'), 'uint32', 1),
        overflow: createTransientView(graph, unique('overflow'), 'uint32', 1)
      },
      drawInstanceCount: createTransientView(graph, unique('draw'), 'uint32', 1)
    })
  );
  const ids = full.getCommandNodes(graph).map(node => node.id);
  expect(ids[0]).toBe('full-bracket');
  expect(ids).not.toContain('full-geometry');
  expect(ids).toContain('full-columns');
  expect(ids.some(id => id.startsWith('full-active'))).toBe(true);
  expect(ids[ids.length - 1]).toBe('full-publish');
});

it('GPUTrajectoryResample validates and builds arc-length nodes', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  expect(
    new GPUTrajectoryResample(createResampleProps(graph, {id: 'time'}))
      .getCommandNodes(graph)
      .map(node => node.id)
  ).toEqual(['time-samples']);
  expect(
    new GPUTrajectoryResample(
      createResampleProps(graph, {
        id: 'arc',
        spacing: 'arc-length',
        timestamps: undefined
      })
    )
      .getCommandNodes(graph)
      .map(node => node.id)
  ).toEqual(['arc-arc-length', 'arc-samples']);
  expect(
    () => new GPUTrajectoryResample(createResampleProps(graph, {timestamps: undefined}))
  ).toThrow(/timestamps are required/);
  expect(() => new GPUTrajectoryResample(createResampleProps(graph, {sampleCount: 0}))).toThrow(
    /positive integer/
  );
  expect(() => new GPUTrajectoryResample(createResampleProps(graph, {sampleCount: 3}))).toThrow(
    /trackCount \* sampleCount/
  );
  expect(
    () =>
      new GPUTrajectoryResample(
        createResampleProps(graph, {
          sampleElevations: createTransientView(graph, unique('z'), 'float32', 8)
        })
      )
  ).toThrow(/requires elevations/);
});

it('playhead oracle: exact sample times, duplicates, single points, empty tracks, outside range', () => {
  const tracks = packTracks(EDGE_CASE_TRACKS);
  const at = (playhead: number, maxGap = 0) =>
    computePlayheadOracle(tracks, {kind: 'float32', value: playhead}, maxGap);

  // Exact sample time: fraction 0 on the outgoing segment, exact position.
  const exact = at(10);
  expect(exact.status[0]).toBe(STATUS.active);
  expect(exact.segmentRows[0]).toBe(1);
  expect(exact.segmentFractions[0]).toBe(0);
  expect(exact.positions.slice(0, 2)).toEqual([10, 0]);
  expect(exact.headings[0]).toBeCloseTo(Math.PI / 2, 6);
  expect(exact.speeds[0]).toBe(2);

  // Duplicate timestamps: the last of the rows at t=5 (absolute rows 5-7) wins and the interval
  // to t=15 is used. Segment rows are absolute; track 1 starts at row 4.
  const duplicate = at(5);
  expect(duplicate.segmentRows[1]).toBe(7);
  expect(duplicate.positions.slice(2, 4)).toEqual([9, 9]);
  expect(duplicate.speeds[1]).toBeCloseTo(1.1, 6);
  // Final duplicated time: the last row, a zero-duration last segment, speed 0.
  const finalDuplicate = at(20);
  expect(finalDuplicate.status[1]).toBe(STATUS.active);
  expect(finalDuplicate.segmentRows[1]).toBe(9);
  expect(finalDuplicate.positions.slice(2, 4)).toEqual([22, 9]);
  expect(finalDuplicate.speeds[1]).toBe(0);

  // Single sample: active only at exactly its time.
  expect([at(11).status[2], at(12).status[2], at(13).status[2]]).toEqual([
    STATUS.beforeStart,
    STATUS.active,
    STATUS.afterEnd
  ]);
  expect(at(12).positions.slice(4, 6)).toEqual([3, 4]);
  expect(at(12).speeds[2]).toBe(0);

  // Empty tracks.
  for (const track of [3, 6]) {
    expect(at(12).status[track]).toBe(STATUS.empty);
    expect(at(12).segmentRows[track]).toBe(NO_ROW);
  }

  // Outside the range: clamped positions, end headings, zero speed.
  const late = at(50);
  expect(late.status[0]).toBe(STATUS.afterEnd);
  expect(late.positions.slice(0, 2)).toEqual([0, 20]);
  expect(late.headings[0]).toBeCloseTo(Math.PI, 6);
  expect(late.speeds[0]).toBe(0);
  expect(late.status[5]).toBe(STATUS.beforeStart);
  expect(late.positions.slice(10, 12)).toEqual([50, 50]);
  expect(at(-1).status.slice(0, 3)).toEqual([
    STATUS.beforeStart,
    STATUS.beforeStart,
    STATUS.beforeStart
  ]);
  expect(late.activeTracks).toEqual([4]);
});

it('playhead oracle: gap rule holds at the last fix and never flags an exact fix', () => {
  const tracks = packTracks(EDGE_CASE_TRACKS);
  const at = (playhead: number, maxGap: number) =>
    computePlayheadOracle(tracks, {kind: 'float32', value: playhead}, maxGap);
  expect(at(50, 10).status[4]).toBe(STATUS.gap);
  expect(at(50, 10).positions.slice(8, 10)).toEqual([1, 0]);
  expect(at(50, 10).speeds[4]).toBe(0);
  expect(at(50, 0).status[4]).toBe(STATUS.active);
  expect(at(50, 0).positions.slice(8, 10)).toEqual([50, 0]);
  expect(at(50, 100).status[4]).toBe(STATUS.active);
  expect(at(50, 99.5).status[4]).toBe(STATUS.gap);
  expect(at(1, 10).status[4]).toBe(STATUS.active);
  expect(at(101, 10).status[4]).toBe(STATUS.active);
  expect(at(101, 10).positions.slice(8, 10)).toEqual([101, 0]);
});

it('playhead oracle: word playheads with fractions match the float32 model', () => {
  const base = 397n * 2n ** 32n - 20n;
  const floatTracks = packTracks(EDGE_CASE_TRACKS);
  const wordTracks = packTracks(EDGE_CASE_TRACKS, base);
  for (const playhead of [-3.5, 0, 4.25, 5, 7.5, 12, 20, 50.75, 101, 200]) {
    const integer = Math.floor(playhead);
    const expected = computePlayheadOracle(floatTracks, {kind: 'float32', value: playhead}, 10);
    const actual = computePlayheadOracle(
      wordTracks,
      {
        kind: 'words',
        integer: base + BigInt(integer),
        fraction: playhead - integer
      },
      10
    );
    expect(actual.status).toEqual(expected.status);
    expect(actual.segmentRows).toEqual(expected.segmentRows);
    expect(actual.positions).toEqual(expected.positions);
  }
});

it('resample oracle: time and arc-length spacing, single points, and empty tracks', () => {
  const tracks = packTracks(EDGE_CASE_TRACKS);
  const time = computeResampleOracle(tracks, 5, 'time');
  expect(time.samples.slice(0, 10)).toEqual([0, 0, 10, 0, 10, 20, 5, 20, 0, 20]);
  expect(time.sampleTimes.slice(0, 5)).toEqual([0, 10, 20, 30, 40]);
  const arc = computeResampleOracle(tracks, 5, 'arc-length');
  expect(arc.samples.slice(0, 10)).toEqual([0, 0, 10, 0, 10, 10, 10, 20, 0, 20]);
  expect(arc.sampleTimes.slice(0, 5)).toEqual([0, 10, 15, 20, 40]);
  // Duplicates: sample 0 starts at the first row, the last sample is the last duplicate.
  expect(time.samples.slice(10, 12)).toEqual([0, 0]);
  expect(time.samples.slice(18, 20)).toEqual([22, 9]);
  // Single point repeats; empty tracks are zero.
  expect(time.samples.slice(20, 30)).toEqual([3, 4, 3, 4, 3, 4, 3, 4, 3, 4]);
  expect(time.samples.slice(30, 40)).toEqual(new Array(10).fill(0));
  expect(computeResampleOracle(tracks, 1, 'time').samples.slice(0, 2)).toEqual([0, 0]);
});
