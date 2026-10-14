// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {
  canonicalizeLabels,
  computeHierarchicalDensityModel,
  getAdjustedRandIndex,
  HIERARCHICAL_DENSITY_FIXTURE
} from './hierarchical-density-clustering-oracle';
import {runHierarchicalDensityClustering} from './hierarchical-density-clustering-harness';
import {requireTestDevice} from '../../utils/require-test-device';

// Oracle: test/oracles/spatial-clustering-hdbscan.py (scikit-learn HDBSCAN labels, core distances
// and spanning-tree weights) and the float64 model of the GPU formulation in
// hierarchical-density-clustering-oracle.ts, which uses the same (weight, low, high) tie order.
const {tolerances} = HIERARCHICAL_DENSITY_FIXTURE;

it('GPUHierarchicalDensityClustering matches scikit-learn HDBSCAN and the model', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const scene of HIERARCHICAL_DENSITY_FIXTURE.scenes) {
    const positions = Float32Array.from(scene.positions.flat());
    const actual = await runHierarchicalDensityClustering(device, positions, {
      minimumClusterSize: scene.minimumClusterSize,
      minimumSamples: scene.minimumSamples
    });
    actual.coreDistances.forEach((core, row) => {
      const expected = scene.coreDistances[row];
      if (Number.isFinite(expected)) {
        expect(Math.abs(core - expected), `${scene.name} core ${row}`).toBeLessThanOrEqual(
          tolerances.coreAbsolute + tolerances.coreRelative * expected
        );
      }
    });
    const labels = canonicalizeLabels(actual.labels);
    const model = computeHierarchicalDensityModel(
      scene.positions,
      scene.minimumClusterSize,
      scene.minimumSamples
    );
    // Same tie order as the model; float32 weights can only reorder near-ties.
    expect(
      getAdjustedRandIndex(labels, model.labels),
      `${scene.name} model`
    ).toBeGreaterThanOrEqual(0.99);
    expect(
      getAdjustedRandIndex(labels, scene.labels),
      `${scene.name} scikit-learn`
    ).toBeGreaterThanOrEqual(0.97);
    expect(actual.clusterCount, `${scene.name} cluster count`).toBe(Math.max(...scene.labels) + 1);
  }
  device.destroy?.();
}, 300000);

it('GPUHierarchicalDensityClustering treats non-finite points as noise', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const scene = HIERARCHICAL_DENSITY_FIXTURE.scenes.find(entry => entry.name === 'moons');
  if (!scene) {
    throw new Error('moons scene missing');
  }
  const positions = Float32Array.from([...scene.positions.flat(), NaN, 0, Infinity, 1, 0, NaN]);
  const actual = await runHierarchicalDensityClustering(device, positions, {
    minimumClusterSize: scene.minimumClusterSize,
    minimumSamples: scene.minimumSamples
  });
  const rows = scene.positions.length;
  expect(actual.labels.slice(rows)).toEqual([0xffffffff, 0xffffffff, 0xffffffff]);
  expect(getAdjustedRandIndex(canonicalizeLabels(actual.labels.slice(0, rows)), scene.labels)).toBe(
    1
  );
  device.destroy?.();
}, 120000);
