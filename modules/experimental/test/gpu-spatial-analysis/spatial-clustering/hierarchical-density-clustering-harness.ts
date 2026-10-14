// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {GPUHierarchicalDensityClustering} from '../../../src/gpu-spatial-analysis/spatial-clustering/index';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

/** Runs one `GPUHierarchicalDensityClustering` graph and reads labels, core distances and the cluster count. */
export async function runHierarchicalDensityClustering(
  device: Device,
  positions: Float32Array,
  options: {minimumClusterSize: number; minimumSamples?: number; timedEncodings?: number}
): Promise<{
  labels: number[];
  coreDistances: number[];
  clusterCount: number;
  medianEncodingMilliseconds?: number;
}> {
  const rows = positions.length / 2;
  const graph = new GPUCommandGraph(device, {id: 'hierarchical-density-clustering-test'});
  const positionsBuffer = createInputBuffer(device, positions);
  const labelsBuffer = createOutputBuffer(device, rows);
  const coreBuffer = createOutputBuffer(device, rows);
  const countBuffer = createOutputBuffer(device, 1);
  graph.add(
    new GPUHierarchicalDensityClustering({
      positions: importGraphBuffer(graph, 'positions', positionsBuffer, 'float32x2', rows),
      minimumClusterSize: options.minimumClusterSize,
      minimumSamples: options.minimumSamples,
      output: {
        labels: importGraphBuffer(graph, 'labels', labelsBuffer, 'uint32', rows),
        coreDistances: importGraphBuffer(graph, 'core-distances', coreBuffer, 'float32', rows),
        status: {count: importGraphBuffer(graph, 'count', countBuffer, 'uint32', 1)}
      }
    })
  );
  const compiled = await graph.compileAsync();
  submitGraph(device, compiled, undefined);
  let medianEncodingMilliseconds: number | undefined;
  if (options.timedEncodings) {
    await readUint32(countBuffer, 1);
    const times: number[] = [];
    for (let encoding = 0; encoding < options.timedEncodings; encoding++) {
      const start = performance.now();
      submitGraph(device, compiled, undefined);
      await readUint32(countBuffer, 1);
      times.push(performance.now() - start);
    }
    times.sort((left, right) => left - right);
    medianEncodingMilliseconds = times[Math.floor(times.length / 2)];
  }
  const result = {
    labels: await readUint32(labelsBuffer, rows),
    coreDistances: await readFloat32(coreBuffer, rows),
    clusterCount: (await readUint32(countBuffer, 1))[0],
    medianEncodingMilliseconds
  };
  compiled.destroy();
  for (const buffer of [positionsBuffer, labelsBuffer, coreBuffer, countBuffer]) {
    buffer.destroy();
  }
  return result;
}
