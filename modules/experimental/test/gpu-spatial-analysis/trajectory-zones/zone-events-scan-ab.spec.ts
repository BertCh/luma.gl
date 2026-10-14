// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// A/B timing for the parallel zone-event formulation (algorithms roadmap P2: the per-track walk
// as a cell sort plus segmented scan). Prints the median wall time of 10 encodings (one-word
// readback) for few tracks with very many events and many tracks with few events. To compare with
// the per-track walk, run it on the current tree and once with
// `src/gpu-spatial-analysis/trajectory-zones/{gpu-zone-events,zone-events-kernels}.ts` replaced by
// the baseline copies the A5-MOVE stream saved (see its handoff):
//   npx vitest run --project headless modules/experimental/test/gpu-spatial-analysis/trajectory-zones/zone-events-scan-ab.spec.ts

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {
  createRandom,
  packTracks,
  type FixtureRow
} from '../trajectory-interpolation/trajectory-interpolation-fixtures';
import {runZoneEvents} from './zone-events-harness';
import {createRingEdges} from './zone-events-oracle';
import {requireTestDevice} from '../../utils/require-test-device';

const EDGES = createRingEdges([
  [0, [100.5, 100.5, 300.5, 100.5, 300.5, 300.5, 100.5, 300.5]],
  [1, [300.5, 100.5, 500.5, 100.5, 500.5, 300.5, 300.5, 300.5]]
]);

function createZigzagTracks(trackCount: number, rowsPerTrack: number): FixtureRow[][] {
  const random = createRandom(trackCount);
  return Array.from({length: trackCount}, () => {
    const y = 110 + random() * 180;
    return Array.from(
      {length: rowsPerTrack},
      (_, row): FixtureRow => [row % 2 === 0 ? 50.25 : 450.75, y + random() * 0.5, row * 2]
    );
  });
}

it('GPUZoneEvents A/B: few tracks with many events and many tracks with few events', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const lines: string[] = [];
  for (const [trackCount, rowsPerTrack] of [
    [4, 8000],
    [8000, 4]
  ] as const) {
    const tracks = packTracks(createZigzagTracks(trackCount, rowsPerTrack));
    const events = trackCount * (rowsPerTrack - 1) * 3;
    const result = await runZoneEvents(device, tracks, EDGES, {
      zoneCount: 2,
      candidateCapacity: 2 * events + 1024,
      trackEventCapacity: rowsPerTrack * 4,
      eventCapacity: events + 1024,
      timedEncodings: 10
    });
    expect(result.candidateOverflow).toBe(0);
    lines.push(
      `zone-events-scan-ab tracks=${trackCount} rows=${rowsPerTrack} events=${result.requiredCount} median=${result.medianEncodingMilliseconds?.toFixed(2)} ms`
    );
  }
  // eslint-disable-next-line no-console
  console.log(lines.join('\n'));
}, 300000);
