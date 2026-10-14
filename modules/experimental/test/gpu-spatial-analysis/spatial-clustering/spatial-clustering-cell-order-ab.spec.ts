// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// A/B timing for the DBSCAN cell-ordered position and core-flag copies (algorithms roadmap P1).
// Not an assertion spec: it prints the median GPU time per encoding. To compare with the previous
// row-order scans, run it once on the current tree and once with
// `src/gpu-spatial-analysis/spatial-clustering/{gpu-spatial-clustering,spatial-clustering-kernels}.ts`
// temporarily replaced by the baseline copies the A5-MOVE stream saved (see its handoff):
//   npx vitest run --project headless modules/experimental/test/gpu-spatial-analysis/spatial-clustering/spatial-clustering-cell-order-ab.spec.ts

import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  getGPUSpatialClusteringParameterValues,
  GPUSpatialClustering,
  GPU_SPATIAL_CLUSTERING_PARAMETER_LENGTH
} from '../../../src/gpu-spatial-analysis/spatial-clustering';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {createClusteredPoints, createSeededRandom} from './spatial-clustering-oracle';
import {requireTestDevice} from '../../utils/require-test-device';

const ROWS = 200_000;
const ENCODINGS = 10;

it('DBSCAN cell-ordered scans: median time on 200k shuffled points in 4 blobs', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const points = createClusteredPoints(5, {
    pointCount: ROWS,
    blobCount: 4,
    extent: 1000,
    blobSigma: 40,
    noiseFraction: 0.1
  });
  // Shuffle so that the input order carries no locality (the case the cell-ordered copy targets).
  const random = createSeededRandom(9);
  for (let row = ROWS - 1; row > 0; row--) {
    const other = Math.floor(random() * (row + 1));
    for (const axis of [0, 1]) {
      const value = points[row * 2 + axis];
      points[row * 2 + axis] = points[other * 2 + axis];
      points[other * 2 + axis] = value;
    }
  }
  const positionsBuffer = createInputBuffer(device, points);
  const labelsBuffer = createOutputBuffer(device, ROWS);
  const parameterBuffer = new GPUParameterBuffer(device, {
    id: 'ab-parameters',
    format: 'float32',
    length: GPU_SPATIAL_CLUSTERING_PARAMETER_LENGTH,
    values: getGPUSpatialClusteringParameterValues({
      bounds: [-200, -200, 1200, 1200],
      epsilon: 4,
      minimumPoints: 8
    })
  });
  const graph = new GPUCommandGraph(device, {id: 'dbscan-ab'});
  graph.add(
    new GPUSpatialClustering({
      positions: importGraphBuffer(graph, 'positions', positionsBuffer, 'float32x2', ROWS),
      parameters: parameterBuffer.importToGraph(graph),
      gridSize: [512, 512],
      output: {labels: importGraphBuffer(graph, 'labels', labelsBuffer, 'uint32', ROWS)}
    })
  );
  const compiled = await graph.compileAsync();
  const times: number[] = [];
  try {
    for (let encoding = 0; encoding <= ENCODINGS; encoding++) {
      const start = performance.now();
      submitGraph(device, compiled, undefined);
      await readUint32(labelsBuffer, 1);
      if (encoding > 0) {
        times.push(performance.now() - start);
      }
    }
  } finally {
    compiled.destroy();
    positionsBuffer.destroy();
    labelsBuffer.destroy();
    parameterBuffer.destroy();
  }
  times.sort((left, right) => left - right);
  const median = times[Math.floor(times.length / 2)];
  // eslint-disable-next-line no-console
  console.log(`dbscan-cell-order-ab rows=${ROWS} median=${median.toFixed(2)} ms`);
  expect(median).toBeGreaterThan(0);
}, 300000);
