// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {
  getGPUSpatialClusteringParameterValues,
  GPUSpatialClustering,
  GPU_SPATIAL_CLUSTERING_NOISE
} from '../../../src/gpu-spatial-analysis/spatial-clustering';
import fixtureJson from '../../oracles/fixtures/parity-wp1-spatial-clustering.json';
import {decodeOracleFixture} from '../../oracles/oracle-fixture';
import {requireTestDevice} from '../../utils/require-test-device';
import {GraphRig} from '../areal-interpolation/areal-interpolation-harness';

type ClusteringCase = {
  name: string;
  reference: string;
  n: number;
  positions: number[];
  bounds: [number, number, number, number];
  epsilon: number;
  minimumPoints: number;
  labels: number[];
  coreMask: number[];
  clusterCount: number;
  allowedBorderLabels: Record<string, number[]>;
};
const fixture = decodeOracleFixture<{cases: ClusteringCase[]}>(fixtureJson);

/** Canonicalize using core representatives, so an allowed border tie cannot renumber clusters. */
function canonicalizeLabels(labels: number[], coreMask: number[]): number[] {
  const roots = new Map<number, number>();
  labels.forEach((label, row) => {
    if (label !== GPU_SPATIAL_CLUSTERING_NOISE && coreMask[row] && !roots.has(label)) {
      roots.set(label, row);
    }
  });
  const mapping = new Map(
    [...roots.entries()]
      .sort((left, right) => left[1] - right[1])
      .map(([label], index) => [label, index])
  );
  return labels.map(label =>
    label === GPU_SPATIAL_CLUSTERING_NOISE ? -1 : (mapping.get(label) ?? -2)
  );
}

function compareExact(
  testCase: ClusteringCase,
  metric: string,
  actual: number[],
  expected: number[],
  allowed: Record<string, number[]> = {}
): void {
  let mismatches = Math.abs(actual.length - expected.length);
  let maxAbs = 0;
  let maxRel = 0;
  expected.forEach((value, row) => {
    // Only explicitly enumerated NON-CORE border points may choose another adjacent component.
    const candidates = allowed[String(row)] ?? [value];
    const error = Math.min(...candidates.map(candidate => Math.abs(actual[row] - candidate)));
    maxAbs = Math.max(maxAbs, error);
    maxRel = Math.max(maxRel, error / Math.max(1, Math.abs(value)));
    if (!candidates.includes(actual[row])) mismatches++;
  });
  console.log(
    'PARITY',
    JSON.stringify({
      class: 'GPUSpatialClustering',
      case: testCase.name,
      reference: testCase.reference,
      n: testCase.n,
      metric,
      maxAbs,
      maxRel,
      mismatches,
      tolerance: {
        profile: 'E',
        absolute: 0,
        relative: 0,
        allowedBorderTies: Object.keys(allowed).length
      }
    })
  );
  expect(mismatches, `${testCase.name}: ${metric}`).toBe(0);
}

for (const testCase of fixture.cases) {
  it(`GPUSpatialClustering sklearn parity: ${testCase.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    // The full data case exercises both the ordinary indexed path and dense-box optimization.
    for (const denseBoxShortcut of testCase.n > 100 ? [false, true] : [false]) {
      const rig = new GraphRig(device, `parity-wp1-clustering-${denseBoxShortcut}`);
      try {
        const labels = rig.output('uint32', testCase.n);
        const coreMask = rig.output('uint32', testCase.n);
        const clusterCount = rig.output('uint32', 1);
        rig.run(
          new GPUSpatialClustering({
            positions: rig.input(Float32Array.from(testCase.positions), 'float32x2', testCase.n),
            parameters: rig.input(getGPUSpatialClusteringParameterValues(testCase), 'float32'),
            gridSize: [256, 256],
            denseBoxShortcut,
            output: {labels: labels.view, coreFlags: coreMask.view, clusterCount: clusterCount.view}
          })
        );
        const actualLabels = await labels.readUint32();
        const actualCore = await coreMask.readUint32();
        const comparisonCase = {...testCase, name: `${testCase.name}/denseBox=${denseBoxShortcut}`};
        compareExact(comparisonCase, 'core-mask', actualCore, testCase.coreMask);
        compareExact(
          comparisonCase,
          'noise-mask',
          actualLabels.map(label => Number(label === GPU_SPATIAL_CLUSTERING_NOISE)),
          testCase.labels.map(label => Number(label === -1))
        );
        compareExact(
          comparisonCase,
          'canonical-partition',
          canonicalizeLabels(actualLabels, actualCore),
          testCase.labels,
          testCase.allowedBorderLabels
        );
        compareExact(comparisonCase, 'cluster-count', await clusterCount.readUint32(), [
          testCase.clusterCount
        ]);
      } finally {
        rig.destroy();
      }
    }
  }, 240000);
}
