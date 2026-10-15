// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/// <reference types="vite/client" />
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {GPUNetworkCostMatrix} from '../../../src/gpu-network/network-accessibility';
import {GPUParameterBuffer} from '../../../src/utils/gpu-contributor-utils';
import {submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {decodeOracleFixture} from '../../oracles/oracle-fixture';
import raw from '../../oracles/fixtures/parity-wp3-cost-matrix-live.json?raw';
import {compareParity, EXACT, type Tolerance} from '../parity-wp3-comparison';
import {createParityResources} from '../parity-wp3-round2-resources';

const fixture = decodeOracleFixture<{
  case: string;
  reference: string;
  nodeCount: number;
  offsets: number[];
  neighbors: number[];
  startNodes: number[];
  cases: {case: string; weights: number[]; costs: number[]; tolerance: Tolerance}[];
}>(JSON.parse(raw));
for (const laneCount of [8, 14, 16, 32, 64]) {
  // Transit uses one lane per hub. Job accessibility caps lanes at its row count;
  // use its smallest scene row count so every requested job lane count is valid.
  // Repeating independently solved rows exercises batching, not new origins.
  const rowCount = laneCount === 14 ? fixture.startNodes.length : 128;
  const referenceRows = Array.from(
    {length: rowCount},
    (_, rowIndex) => rowIndex % fixture.startNodes.length
  );
  const startNodes = referenceRows.map(rowIndex => fixture.startNodes[rowIndex]);
  it(`NetworkX live raw-cost matrix and closures, rows ${rowCount}, lanes ${laneCount}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const resources = createParityResources(device);
    const weights = new GPUParameterBuffer(device, {
      id: 'live-weights',
      format: 'float32',
      length: fixture.neighbors.length
    });
    resources.graph.add(
      new GPUNetworkCostMatrix({
        topology: {
          offsets: resources.input('offsets', 'uint32', fixture.offsets),
          neighbors: resources.input('neighbors', 'uint32', fixture.neighbors),
          weights: weights.importToGraph(resources.graph)
        },
        startNodes: resources.input('hubs', 'uint32', startNodes),
        laneCount,
        maximumIterations: laneCount === 14 ? 16 : 48,
        localIterations: 16,
        output: {
          costs: resources.output('costs', 'float32', rowCount * fixture.nodeCount),
          status: {converged: resources.output('converged', 'uint32', 1)}
        }
      })
    );
    const compiled = resources.graph.compile();
    try {
      for (const state of [...fixture.cases, fixture.cases[0]]) {
        weights.write(Float32Array.from(state.weights));
        submitGraph(device, compiled, undefined);
        const comparison = {...fixture, case: `${state.case}/rows-${rowCount}/lanes-${laneCount}`};
        const expectedCosts = referenceRows.flatMap(rowIndex =>
          state.costs.slice(rowIndex * fixture.nodeCount, (rowIndex + 1) * fixture.nodeCount)
        );
        compareParity(
          'GPUNetworkCostMatrix',
          comparison,
          'every raw-seconds cost and reachability mask',
          await resources.readFloat('costs'),
          expectedCosts,
          state.tolerance
        );
        compareParity(
          'GPUNetworkCostMatrix',
          comparison,
          'all rows converged after live rewrite',
          await resources.readUint('converged'),
          [1],
          EXACT
        );
      }
    } finally {
      compiled.destroy();
      weights.destroy();
      resources.destroy();
    }
  });
}
