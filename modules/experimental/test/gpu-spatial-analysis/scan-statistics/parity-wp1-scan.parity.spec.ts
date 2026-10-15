// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUSpatialScanStatistic,
  getGPUSpatialScanStatisticParameterValues
} from '../../../src/gpu-spatial-analysis/scan-statistics';
import type {
  GPUSpatialScanModel,
  GPUSpatialScanRateDirection,
  GPUSpatialScanWindowShape
} from '../../../src/gpu-spatial-analysis/scan-statistics/scan-statistic-parameters';
import {createWGSLKernelNode} from '../../../src/utils/wgsl-kernel-nodes';
import fixtureJson from '../../oracles/fixtures/parity-wp1-scan.json';
import showcaseJson from '../../oracles/fixtures/parity-wp1-scan-showcase.json';
import {decodeRound4Fixture} from '../../oracles/parity-wp1-round4-fixture';
import {decodeOracleFixture} from '../../oracles/oracle-fixture';
import {compareReference} from '../../oracles/parity-wp1-round8-compare';
import {AnalysisRig} from '../catchment-accessibility/rig';
import {requireTestDevice} from '../../utils/require-test-device';

type ScanCase = {
  name: string;
  reference: string;
  model: GPUSpatialScanModel;
  timeBuckets: number;
  maximumTimeBuckets: number;
  positions: number[];
  cases: number[];
  baseline: number[];
  controls: number[];
  replicates: number[][];
  shape: GPUSpatialScanWindowShape;
  direction: GPUSpatialScanRateDirection;
  fraction: number;
  expected: {
    optimalWindows: {
      indices: number[];
      members: number[];
      observed: number;
      expected: number;
      radius: number;
    }[];
    maxima: number[];
    zoneStatistics: number[];
    indices: number[];
    members: number[];
    observed: number;
    expected: number;
    radius: number;
    pValue: number;
  };
};
const fixture = decodeOracleFixture<{cases: ScanCase[]}>(fixtureJson);
const showcase = await decodeRound4Fixture<{cases: ScanCase[]}>(showcaseJson);

for (const scene of [...fixture.cases, ...showcase.cases]) {
  it(`GPUSpatialScanStatistic SpatialEpi supplied tables ${scene.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = new AnalysisRig(device);
    const clusterIndices = rig.output('uint32', 4);
    const clusterStatistics = rig.output('float32', 8);
    const statistics = rig.output('float32', scene.replicates.length + 1);
    const summary = rig.output('uint32', 4);
    const zoneStatistics = rig.output('float32', scene.positions.length / 2);
    try {
      const producer = new GPUSpatialScanStatistic({
        id: 'reference-scan',
        positions: rig.input(
          new Float32Array(scene.positions),
          'float32x2',
          scene.positions.length / 2
        ),
        cases: rig.input(new Uint32Array(scene.cases), 'uint32'),
        baseline: rig.input(new Float32Array(scene.baseline), 'float32'),
        controls: rig.input(new Uint32Array(scene.controls), 'uint32'),
        model: scene.model,
        timeBuckets: scene.timeBuckets,
        maximumPermutations: scene.replicates.length,
        batchReplicates: scene.replicates.length,
        maximumClusters: 1,
        parameters: rig.input(
          getGPUSpatialScanStatisticParameterValues({
            seed: 8201,
            permutationCount: scene.replicates.length,
            maximumPopulationFraction: scene.fraction,
            maximumTimeBuckets: scene.maximumTimeBuckets,
            windowShape: scene.shape,
            rateDirection: scene.direction
          }),
          'uint32'
        ),
        output: {clusterIndices, clusterStatistics, statistics, summary, zoneStatistics}
      });
      const supplied = rig.input(new Uint32Array(scene.replicates.flat()), 'uint32');
      let replacedDraws = 0;
      for (const node of producer.getCommandNodes(rig.graph)) {
        // Test-only supplied-randomness seam. All production likelihood, search,
        // reduction, cluster selection and p-value kernels remain unmodified.
        // This deliberately makes no claim about the contributor's RNG sampler.
        if (/^reference-scan-(scatter|draw-labels|draw-times)-0$/.test(node.id)) {
          const resource = node.resources?.find(
            use =>
              'buffer' in use &&
              use.buffer instanceof GraphDataView &&
              use.buffer.buffer.id === 'reference-scan-counts'
          );
          if (!resource || !('buffer' in resource) || !(resource.buffer instanceof GraphDataView))
            throw new Error('Scan counts resource missing');
          rig.graph.add(
            createWGSLKernelNode(rig.graph, {
              id: node.id,
              operation: 'ParitySuppliedScanDraws',
              invocationCount: supplied.length,
              bindings: [
                {name: 'supplied', view: supplied, type: 'u32', access: 'read'},
                {name: 'counts', view: resource.buffer, type: 'u32', access: 'read_write'}
              ],
              body: `counts[countsOffset + ${scene.cases.length}u + index] = supplied[suppliedOffset + index];`
            })
          );
          replacedDraws++;
        } else rig.graph.add(node);
      }
      expect(replacedDraws).toBe(1);
      rig.run();
      const compare = (metric: string, actual: number[], expected: number[], exact = false) =>
        compareReference('GPUSpatialScanStatistic', scene, metric, actual, expected, exact);
      const cluster = await rig.readFloat(clusterStatistics);
      compare(
        'all-observed-and-null-maxima',
        await rig.readFloat(statistics),
        scene.expected.maxima
      );
      compare(
        'all-center-maxima',
        await rig.readFloat(zoneStatistics),
        scene.expected.zoneStatistics
      );
      const indices = await rig.readUint(clusterIndices);
      // Different centers can represent the same maximizing cylinder. Only
      // exact R likelihood ties are admitted, never near-optima or a wider bound.
      const selected = scene.expected.optimalWindows.find(window =>
        window.indices.every((value, index) => indices[index] === value)
      );
      compare('independently-maximizing-window', [Number(Boolean(selected))], [1], true);
      if (!selected) throw new Error('Scan window is not a reference maximizer');
      compare(
        'summary',
        await rig.readUint(summary),
        [1, scene.cases.reduce((sum, value) => sum + value, 0), scene.replicates.length, 0],
        true
      );
      compare('cluster-llr-observed-expected', cluster.slice(0, 3), [
        scene.expected.maxima[0],
        selected.observed,
        selected.expected
      ]);
      // Supplied tables require the exact integer exceedance count. WGSL f32
      // division permits 2.5 ULP error; multiplying p by B+1 and rounding recovers
      // that count (the bound here is far below half one count).
      // https://www.w3.org/TR/WGSL/#accuracy-of-concrete-expressions
      const denominator = scene.replicates.length + 1;
      compare(
        'monte-carlo-exceedance-count',
        [Math.round(cluster[3] * denominator) - 1],
        [Math.round(scene.expected.pValue * denominator) - 1],
        true
      );
      const expectedProbability = scene.expected.pValue;
      const probabilityUlp = 2 ** (Math.floor(Math.log2(expectedProbability)) - 23);
      const probabilityBound = 2.5 * probabilityUlp;
      const probabilityError = Math.abs(cluster[3] - expectedProbability);
      console.log(
        'PARITY',
        JSON.stringify({
          class: 'GPUSpatialScanStatistic',
          case: scene.name,
          reference: scene.reference,
          n: 1,
          metric: 'monte-carlo-p',
          maxAbs: probabilityError,
          maxRel: probabilityError / expectedProbability,
          mismatches: Number(probabilityError > probabilityBound),
          tolerance: {
            absolute: probabilityBound,
            relative: 0,
            derivation: 'WGSL f32 division: 2.5 ULP'
          }
        })
      );
      expect(probabilityError).toBeLessThanOrEqual(probabilityBound);
      compare('radius-ratio-cap', cluster.slice(4, 7), [
        selected.radius,
        selected.observed / selected.expected,
        0
      ]);
    } finally {
      rig.destroy();
    }
  });
}
