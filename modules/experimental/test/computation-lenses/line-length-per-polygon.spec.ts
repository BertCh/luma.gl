// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, test} from 'vitest';
import {getLensScenario} from '../../src/computation-lenses/registry';
import '../../src/computation-lenses/scenarios/gpu-spatial-analysis/line-length-per-polygon.lens';
import {requireTestDevice} from '../utils/require-test-device';

test('line length lens measures streets in all three community areas without overflow', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const scenario = getLensScenario('gpu-spatial-analysis.line-length-per-polygon');
  expect(scenario).toBeDefined();
  if (!scenario) return;
  const result = await (await scenario.load()).run(device);
  const snapshots = result.captures?.snapshots ?? [];
  const lengths = snapshots.find(capture => capture.name === 'lengths')?.values;
  expect(lengths).toHaveLength(3);
  if (!lengths) return;
  // Float64 oracle expectations are checked independently in the node data spec.
  for (const [index, expected] of [7845.95, 27341.81, 22321.82].entries()) {
    expect(Math.abs(Number(lengths[index]) - expected)).toBeLessThan(1);
  }
  expect(
    Array.from(snapshots.find(capture => capture.name === 'segmentCounts')?.values ?? [])
  ).toEqual([77, 329, 270]);
  expect(Array.from(snapshots.find(capture => capture.name === 'overflow')?.values ?? [])).toEqual([
    0
  ]);
});
