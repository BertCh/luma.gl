// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Edge-policy tables (quality roadmap rule 22) on the GPU. Oracle:
// test/oracles/statistics-edge-policy.py (esda on the analysis rows; NaN, +-Infinity are missing,
// -0 equals +0, empty input and undefined denominators give NaN). The CPU oracles are checked
// against the same rows in statistics-edge-policy.node.spec.ts.

import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  GPU_GLOBAL_JOIN_COUNT_FIELD,
  GPU_GLOBAL_SPATIAL_STATISTIC_FIELD,
  GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT,
  GPU_GLOBAL_SPATIAL_STATISTICS_SUMMARY,
  GPUGlobalSpatialStatistics
} from '../../../src/gpu-spatial-analysis/global-spatial-statistics';
import {
  getGPUSpatialAutocorrelationParameterValues,
  GPU_LOCAL_MORAN_QUADRANT
} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {createSpatialAutocorrelationHarness} from '../spatial-autocorrelation/spatial-autocorrelation-harness';
import {
  getEdgePolicyMask,
  getEdgePolicyWeights,
  isWithinEdgePolicyTolerance,
  STATISTICS_EDGE_POLICY
} from './statistics-edge-policy-fixture';

const {tolerances} = STATISTICS_EDGE_POLICY;
const LAYOUT = GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT;
const ROW_COUNT = STATISTICS_EDGE_POLICY.rows.length;

it('GPUGlobalSpatialStatistics follows the edge-policy table', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const weights = getEdgePolicyWeights();
  const mismatches: string[] = [];
  for (const testCase of STATISTICS_EDGE_POLICY.global) {
    const mask = getEdgePolicyMask(testCase.mask);
    const buffers = {
      offsets: createInputBuffer(device, weights.offsets),
      neighbors: createInputBuffer(device, weights.neighbors),
      weights: createInputBuffer(device, weights.weights),
      values: createInputBuffer(device, Float32Array.from(testCase.values)),
      mask: mask && createInputBuffer(device, mask),
      results: createOutputBuffer(device, LAYOUT.length),
      joinCounts: createOutputBuffer(device, 3)
    };
    const capacity = weights.neighbors.length;
    const graph = new GPUCommandGraph(device, {id: 'statistics-edge-policy-global'});
    graph.add(
      new GPUGlobalSpatialStatistics({
        weights: {
          offsets: importGraphBuffer(graph, 'offsets', buffers.offsets, 'uint32', ROW_COUNT + 1),
          neighbors: importGraphBuffer(graph, 'neighbors', buffers.neighbors, 'uint32', capacity),
          weights: importGraphBuffer(graph, 'weights', buffers.weights, 'float32', capacity)
        },
        islands: 'keep',
        values: importGraphBuffer(graph, 'values', buffers.values, 'float32', ROW_COUNT),
        mask: buffers.mask && importGraphBuffer(graph, 'mask', buffers.mask, 'uint32', ROW_COUNT),
        statistics: ['moran', 'geary', 'getisOrdG', 'joinCount'],
        output: {
          summary: importGraphBuffer(graph, 'results', buffers.results, 'float32', LAYOUT.length),
          joinCounts: importGraphBuffer(graph, 'join-counts', buffers.joinCounts, 'uint32', 3)
        }
      })
    );
    const compiled = graph.compile();
    submitGraph(device, compiled, undefined);
    const results = await readFloat32(buffers.results, LAYOUT.length);
    const joinCounts = await readUint32(buffers.joinCounts, 3);
    compiled.destroy();
    for (const buffer of Object.values(buffers)) {
      buffer?.destroy();
    }
    const {expected} = testCase;
    const label = testCase.name;
    const summary = GPU_GLOBAL_SPATIAL_STATISTICS_SUMMARY;
    expect(results[summary.count], `${label} n`).toBe(expected.count);
    for (const statistic of ['moran', 'geary', 'getisOrdG'] as const) {
      for (const [field, value] of Object.entries(expected[statistic])) {
        const tolerance = field.startsWith('p')
          ? tolerances.pValue
          : field.startsWith('z')
            ? tolerances.zScore
            : tolerances.statistic;
        const index =
          LAYOUT[statistic] +
          GPU_GLOBAL_SPATIAL_STATISTIC_FIELD[
            field as keyof typeof GPU_GLOBAL_SPATIAL_STATISTIC_FIELD
          ];
        if (!isWithinEdgePolicyTolerance(results[index], value, tolerance)) {
          mismatches.push(
            `${label} ${statistic}.${field}: GPU ${results[index]} != table ${value}`
          );
        }
      }
    }
    expect(results[summary.blackCount], `${label} blacks`).toBe(expected.blackCount);
    const {blackBlack, blackWhite, whiteWhite, joins} = expected.joinCount;
    expect(joinCounts, `${label} ordered join counts`).toEqual([
      2 * blackBlack,
      2 * blackWhite,
      2 * whiteWhite
    ]);
    expect(results[LAYOUT.joinCount + GPU_GLOBAL_JOIN_COUNT_FIELD.joins], label).toBe(joins);
  }
  expect(mismatches).toEqual([]);
});

it('GPUHotSpotAnalysis and GPULocalMoran follow the edge-policy table', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const weights = getEdgePolicyWeights();
  const mismatches: string[] = [];
  for (const testCase of STATISTICS_EDGE_POLICY.local) {
    const scene = {
      values: Float32Array.from(testCase.values),
      mask: getEdgePolicyMask(testCase.mask)
    };
    for (const contributor of ['hot-spot', 'local-moran'] as const) {
      const harness = createSpatialAutocorrelationHarness(device, {
        contributor,
        scene,
        weights: {kind: 'csr', ...weights},
        islands: 'keep',
        selfWeight: contributor === 'hot-spot' ? 1 : undefined
      });
      try {
        const result = await harness.run();
        expect(result.globalStatistics[0], `${testCase.name} ${contributor} n`).toBe(
          testCase.expected.count
        );
        for (let row = 0; row < ROW_COUNT; row++) {
          const [actual, expected, tolerance] =
            contributor === 'hot-spot'
              ? [result.zScores[row], testCase.expected.giStar[row], tolerances.zScore]
              : [result.localI[row], testCase.expected.localI[row], tolerances.statistic];
          if (!isWithinEdgePolicyTolerance(actual, expected, tolerance)) {
            mismatches.push(
              `${testCase.name} ${contributor} row ${row}: GPU ${actual} != ${expected}`
            );
          }
        }
      } finally {
        harness.destroy();
      }
    }
  }
  expect(mismatches).toEqual([]);
});

it('GPULocalMoran reports no quadrant when the raw significance level is NaN', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // The packer refuses a NaN level (node spec); written raw past it, `p <= NaN` is false.
  const baseline = STATISTICS_EDGE_POLICY.local[0];
  const harness = createSpatialAutocorrelationHarness(device, {
    contributor: 'local-moran',
    scene: {values: Float32Array.from(baseline.values)},
    weights: {kind: 'csr', ...getEdgePolicyWeights()},
    islands: 'keep',
    quadrantGating: 'analytic'
  });
  try {
    const permissive = await harness.run({significanceLevel: 0.999});
    expect(
      permissive.quadrants.some(quadrant => quadrant !== GPU_LOCAL_MORAN_QUADRANT.NOT_SIGNIFICANT)
    ).toBe(true);
    const parameters = getGPUSpatialAutocorrelationParameterValues({significanceLevel: 0.5});
    parameters[0] = NaN;
    const result = await harness.run(parameters);
    expect(result.quadrants).toEqual(
      new Array(ROW_COUNT).fill(GPU_LOCAL_MORAN_QUADRANT.NOT_SIGNIFICANT)
    );
  } finally {
    harness.destroy();
  }
});
