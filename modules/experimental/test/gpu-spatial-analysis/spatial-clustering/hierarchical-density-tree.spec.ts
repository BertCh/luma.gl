// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../utils/require-test-device';
import {
  computeHierarchicalDensityModel,
  HIERARCHICAL_DENSITY_FIXTURE
} from './hierarchical-density-clustering-oracle';
import type {DensityTreeEdge} from './hierarchical-density-tree-model';
import {runDensityTreeStages} from './hierarchical-density-tree-harness';

it('parallel tree stages exactly preserve the serial labels on every sklearn fixture', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  try {
    for (const scene of HIERARCHICAL_DENSITY_FIXTURE.scenes) {
      const model = computeHierarchicalDensityModel(
        scene.positions,
        scene.minimumClusterSize,
        scene.minimumSamples
      );
      const rows = scene.positions.length;
      const serial = await runDensityTreeStages(
        device,
        rows,
        model.sortedEdges,
        scene.minimumClusterSize,
        'serial'
      );
      const parallel = await runDensityTreeStages(
        device,
        rows,
        model.sortedEdges,
        scene.minimumClusterSize,
        'parallel'
      );
      expect(parallel.labels, scene.name).toEqual(serial.labels);
      expect(parallel.count, scene.name).toBe(serial.count);
      expect(parallel.firstLabels, `${scene.name} re-encode`).toEqual(parallel.labels);
      for (let node = 0; node < rows * 2; node++) {
        const base = node * 5;
        expect(parallel.dendrogram.slice(base, base + 3), `${scene.name} node ${node}`).toEqual(
          serial.dendrogram.slice(base, base + 3)
        );
        expect(
          parallel.dendrogram.slice(base + 3, base + 5).sort(),
          `${scene.name} children ${node}`
        ).toEqual(serial.dendrogram.slice(base + 3, base + 5).sort());
      }
    }
  } finally {
    device.destroy();
  }
}, 300000);

it('parallel tree stages preserve deep tied chains, stars, isolated rows and empty hierarchies', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  try {
    for (const rows of [1, 2, 257, 4097]) {
      for (const kind of ['chain', 'star', 'comb', 'forest', 'empty']) {
        const edgeCount =
          kind === 'empty' ? 0 : kind === 'forest' ? Math.floor((rows - 1) / 2) : rows - 1;
        const edges: DensityTreeEdge[] = Array.from({length: edgeCount}, (_, rank) => [
          kind === 'star' ? 0 : rank,
          rank + 1,
          Math.floor(rank / 7)
        ]);
        if (kind === 'comb') {
          // Merge pairs first so every later split survives min_cluster_size=2.
          // Unlike a point-at-a-time chain, this is also a deep condensed EOM tree.
          edges.length = 0;
          for (let row = 0; row + 1 < rows; row += 2) edges.push([row, row + 1, 1]);
          for (let row = 2; row + 1 < rows; row += 2) edges.push([0, row, row]);
        }
        const serial = await runDensityTreeStages(device, rows, edges, 2, 'serial');
        const parallel = await runDensityTreeStages(device, rows, edges, 2, 'parallel');
        expect(parallel.labels, `${rows}/${kind}`).toEqual(serial.labels);
        expect(parallel.count, `${rows}/${kind}`).toBe(serial.count);
        expect(parallel.firstLabels).toEqual(parallel.labels);
        for (let node = 0; node < rows * 2; node++) {
          expect(parallel.dendrogram.slice(node * 5, node * 5 + 3)).toEqual(
            serial.dendrogram.slice(node * 5, node * 5 + 3)
          );
        }
      }
    }
  } finally {
    device.destroy();
  }
}, 300000);
