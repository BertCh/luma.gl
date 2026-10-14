// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// A/B (algorithms roadmap P3 #7): GPU HDBSCAN on 20k and 100k points in four blobs (median of 3
// encodings, one-word readback). There is no previous GPU path; the speed bar is scikit-learn's
// time on the same 100k scene shape, recorded by test/oracles/spatial-clustering-hdbscan.py
// (`cpuSeconds.blobs100k` in the fixture).
//   npx vitest run --project headless modules/experimental/test/gpu-spatial-analysis/spatial-clustering/hierarchical-density-clustering-ab.spec.ts

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {createClusteredPoints} from './spatial-clustering-oracle';
import {HIERARCHICAL_DENSITY_FIXTURE} from './hierarchical-density-clustering-oracle';
import {runHierarchicalDensityClustering} from './hierarchical-density-clustering-harness';
import {requireTestDevice} from '../../utils/require-test-device';

it('GPUHierarchicalDensityClustering A/B: 20k and 100k points against scikit-learn', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const lines: string[] = [];
  for (const rows of [20_000, 100_000]) {
    const positions = createClusteredPoints(3, {
      pointCount: rows,
      blobCount: 4,
      extent: 60,
      blobSigma: 4,
      noiseFraction: 0.05
    });
    const result = await runHierarchicalDensityClustering(device, positions, {
      minimumClusterSize: 50,
      timedEncodings: 3
    });
    lines.push(
      `hdbscan-ab rows=${rows} clusters=${result.clusterCount} median=${result.medianEncodingMilliseconds?.toFixed(2)} ms`
    );
  }
  lines.push(
    `hdbscan-ab scikit-learn 100k=${HIERARCHICAL_DENSITY_FIXTURE.cpuSeconds.blobs100k.toFixed(2)} s`
  );
  // eslint-disable-next-line no-console
  console.log(lines.join('\n'));
  expect(lines.length).toBe(3);
}, 900000);
