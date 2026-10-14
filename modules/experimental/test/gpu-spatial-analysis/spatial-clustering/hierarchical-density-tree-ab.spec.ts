// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../utils/require-test-device';
import type {DensityTreeEdge} from './hierarchical-density-tree-model';
import {runDensityTreeStages} from './hierarchical-density-tree-harness';

// Same-device, same-run old/new tree-stage A/B; does not include the unchanged kNN / Boruvka cost.
// Run separately after parity specs; implementation sessions must not run GPU timing specs.
it('HDBSCAN tree-stage A/B: balanced, chain and comb MSTs at 20k / 100k', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  try {
    for (const rows of [20_000, 100_000]) {
      for (const kind of ['balanced', 'chain', 'comb']) {
        const edges: DensityTreeEdge[] = [];
        if (kind === 'chain') {
          for (let row = 1; row < rows; row++) edges.push([row - 1, row, row]);
        } else if (kind === 'comb') {
          for (let row = 0; row + 1 < rows; row += 2) edges.push([row, row + 1, 1]);
          for (let row = 2; row + 1 < rows; row += 2) edges.push([0, row, row]);
        } else {
          for (let stride = 1; stride < rows; stride *= 2) {
            for (let first = 0; first + stride < rows; first += stride * 2)
              edges.push([first, first + stride, stride]);
          }
        }
        const serial = await runDensityTreeStages(device, rows, edges, 2, 'serial', 3);
        const parallel = await runDensityTreeStages(device, rows, edges, 2, 'parallel', 3);
        expect(parallel.labels).toEqual(serial.labels);
        expect(parallel.count).toBe(serial.count);
        // eslint-disable-next-line no-console
        console.log(
          `hdbscan-tree-ab rows=${rows} shape=${kind} serial=${serial.medianMilliseconds.toFixed(2)} ms parallel=${parallel.medianMilliseconds.toFixed(2)} ms`
        );
      }
    }
  } finally {
    device.destroy();
  }
}, 900000);
