// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {
  GPUSkaterRegions,
  getGPUSkaterRegionsParameterValues
} from '../../../src/gpu-spatial-analysis/spatial-regionalization';
import fixture from '../../oracles/fixtures/parity-wp1-skater-lag.json';
import {compareParity} from '../../oracles/parity-wp1-compare';
import {requireTestDevice} from '../../utils/require-test-device';
import {AnalysisRig} from '../catchment-accessibility/rig';

for (const testCase of fixture.skater) {
  for (const run of testCase.runs) {
    it(`GPUSkaterRegions spopt parity: ${testCase.name}/${run.target}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const rig = new AnalysisRig(device);
      const comparison = {...testCase, name: `${testCase.name}/${run.target}`};
      try {
        const labels = rig.output('uint32', testCase.n);
        const regionCount = rig.output('uint32', 1);
        const cutEdges = rig.output('uint32', run.target - 1);
        const cutGains = rig.output('float32', run.target - 1);
        rig.run(
          new GPUSkaterRegions({
            weights: rig.weights(testCase, 0),
            treeEdgeFlags: rig.input(Uint32Array.from(testCase.treeFlags), 'uint32'),
            componentLabels: rig.input(Uint32Array.from(testCase.components), 'uint32'),
            values: rig.input(Float32Array.from(testCase.values), 'float32'),
            columnCount: testCase.columns,
            maximumRegionCount: run.target,
            parameters: rig.input(
              getGPUSkaterRegionsParameterValues({
                regionCount: run.target,
                minimumRegionSize: testCase.floor
              }),
              'uint32'
            ),
            output: {labels, regionCount, cutEdges, cutGains}
          })
        );
        const actualLabels = await rig.readUint(labels);
        const minimumRows = new Map<number, number>();
        actualLabels.forEach((label, row) => {
          if (!minimumRows.has(label)) minimumRows.set(label, row);
        });
        const canonical = actualLabels.map(label => minimumRows.get(label)!);
        compareParity(
          'GPUSkaterRegions',
          comparison,
          'canonical-partition',
          canonical,
          run.labels,
          'E'
        );
        compareParity(
          'GPUSkaterRegions',
          comparison,
          'region-count',
          await rig.readUint(regionCount),
          [run.regionIds.length],
          'E'
        );
        const regionIds = [...new Set(canonical)].sort((first, second) => first - second);
        const sizes: number[] = [];
        const objectives: number[] = [];
        const disconnected: number[] = [];
        for (const region of regionIds) {
          const members = canonical.flatMap((label, row) => (label === region ? [row] : []));
          sizes.push(members.length);
          let objective = 0;
          for (let column = 0; column < testCase.columns; column++) {
            const mean =
              members.reduce(
                (sum, row) => sum + testCase.values[row * testCase.columns + column],
                0
              ) / members.length;
            objective += members.reduce(
              (sum, row) => sum + (testCase.values[row * testCase.columns + column] - mean) ** 2,
              0
            );
          }
          objectives.push(objective);
          const visited = new Set<number>([members[0]]);
          const pending = [members[0]];
          for (let cursor = 0; cursor < pending.length; cursor++) {
            const row = pending[cursor];
            for (let entry = testCase.offsets[row]; entry < testCase.offsets[row + 1]; entry++) {
              const neighbor = testCase.neighbors[entry];
              if (canonical[neighbor] === region && !visited.has(neighbor)) {
                visited.add(neighbor);
                pending.push(neighbor);
              }
            }
          }
          disconnected.push(members.length - visited.size);
        }
        compareParity('GPUSkaterRegions', comparison, 'region-sizes', sizes, run.sizes, 'E');
        compareParity(
          'GPUSkaterRegions',
          comparison,
          'disconnected-rows',
          disconnected,
          regionIds.map(() => 0),
          'E'
        );
        compareParity(
          'GPUSkaterRegions',
          comparison,
          'per-region-objective',
          objectives,
          run.scores
        );
        const objective = objectives.reduce((sum, value) => sum + value, 0);
        compareParity(
          'GPUSkaterRegions',
          comparison,
          'objective-quality-ratio',
          [objective / run.objective],
          [1],
          'H'
        );
        const cuts = await rig.readUint(cutEdges);
        const gains = await rig.readFloat(cutGains);
        compareParity('GPUSkaterRegions', comparison, 'per-step-cut-gains', gains, run.cutGains);
        const applied = cuts.filter(edge => edge !== 0xffffffff);
        compareParity(
          'GPUSkaterRegions',
          comparison,
          'applied-cuts',
          [applied.length],
          [regionIds.length - testCase.componentCount],
          'E'
        );
        compareParity(
          'GPUSkaterRegions',
          comparison,
          'cut-gain-sum',
          [gains.reduce((sum, value) => sum + value, 0)],
          [testCase.initialObjective - objective]
        );
        // Every reported cut must be an original tree edge separating final regions.
        compareParity(
          'GPUSkaterRegions',
          comparison,
          'cut-edge-validity',
          applied.map(entry => {
            const row = testCase.offsets.findIndex(
              (offset, index) =>
                index < testCase.n && offset <= entry && entry < testCase.offsets[index + 1]
            );
            return row >= 0 &&
              testCase.treeFlags[entry] === 1 &&
              canonical[row] !== canonical[testCase.neighbors[entry]]
              ? 1
              : 0;
          }),
          applied.map(() => 1),
          'E'
        );
      } finally {
        rig.destroy();
      }
    });
  }
}
