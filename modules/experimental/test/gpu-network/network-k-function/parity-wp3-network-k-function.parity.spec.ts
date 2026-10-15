// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {
  GPUNetworkKFunction,
  getGPUNetworkKFunctionParameterValues
} from '../../../src/gpu-network/network-k-function';
import {submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {compareParity, EXACT} from '../parity-wp3-comparison';
import {createParityResources} from '../parity-wp3-round2-resources';
import {getDistributionDistance, kFixtures} from './parity-wp3-network-k-function-fixture';

for (const fixture of kFixtures) {
  for (const seed of fixture.gpuSeeds) {
    it(`GPUNetworkKFunction independent Spaghetti ${fixture.case}, seed ${seed}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const resources = createParityResources(device);
      const bandCount = fixture.radii.length;
      const patternCount = fixture.gpuSimulationCount + 1;
      const className = 'GPUNetworkKFunction';
      const caseName = `${fixture.case}/seed-${seed}`;
      const comparison = {...fixture, case: caseName};
      const {graph} = resources;
      graph.add(
        new GPUNetworkKFunction({
          positions: resources.input('points', 'float32x2', fixture.points.flat()),
          topology: {
            offsets: resources.input('offsets', 'uint32', fixture.offsets),
            neighbors: resources.input('neighbors', 'uint32', fixture.neighbors),
            weights: resources.input('weights', 'float32', fixture.weights),
            nodePositions: resources.input('positions', 'float32x2', fixture.positions.flat())
          },
          networkLength: resources.input('length', 'float32', [fixture.networkLength]),
          maximumDistance: resources.input('radius', 'float32', [fixture.maximumDistance]),
          parameters: resources.input(
            'parameters',
            'uint32',
            Array.from(
              getGPUNetworkKFunctionParameterValues({
                seed,
                activeSimulations: fixture.gpuSimulationCount
              })
            )
          ),
          bandCount,
          simulationCount: fixture.gpuSimulationCount,
          rowsPerBlock: 32,
          maximumIterations: 64,
          output: {
            kValues: resources.output('k', 'float32', patternCount * bandCount),
            pairCounts: resources.output('pairs', 'uint32', patternCount * bandCount),
            envelope: fixture.gpuSimulationCount
              ? resources.output('envelope', 'float32', 3 * bandCount)
              : undefined,
            snappedEventCount: resources.output('snapped', 'uint32', 1),
            status: {converged: resources.output('converged', 'uint32', 1)}
          }
        })
      );
      const compiled = graph.compile();
      try {
        submitGraph(device, compiled, undefined);
        const values = await resources.readFloat('k');
        const counts = await resources.readUint('pairs');
        compareParity(
          className,
          comparison,
          'snapped events',
          await resources.readUint('snapped'),
          [fixture.points.length],
          EXACT
        );
        compareParity(
          className,
          comparison,
          'convergence',
          await resources.readUint('converged'),
          [1],
          EXACT
        );
        // Real-data pairs inside the analytically derived distance-error interval are
        // enumerated in the fixture, not silently forced to one side of a threshold.
        const observedCounts = counts.slice(0, bandCount);
        const mismatches = observedCounts.filter(
          (value, index) => value < fixture.lowerCounts[index] || value > fixture.upperCounts[index]
        ).length;
        console.log(
          'PARITY',
          JSON.stringify({
            class: className,
            case: caseName,
            reference: fixture.reference,
            n: fixture.points.length,
            metric: 'observed pair counts at every radius',
            maxAbs: Math.max(
              ...observedCounts.map((value, index) => Math.abs(value - fixture.counts[index]))
            ),
            maxRel: Math.max(
              ...observedCounts.map(
                (value, index) =>
                  Math.abs(value - fixture.counts[index]) / Math.max(1, fixture.counts[index])
              )
            ),
            mismatches,
            tolerance: {
              profile: 'P/E',
              lower: fixture.lowerCounts,
              upper: fixture.upperCounts,
              distanceErrorBound: fixture.distanceErrorBound
            }
          })
        );
        expect.soft(mismatches).toBe(0);
        // Compare K to the closest independently allowed strict-pair-count value;
        // only explicitly enumerated numerical boundary pairs contribute uncertainty.
        const expected = fixture.observed.map((value, index) => {
          const scale = (2 * fixture.exactNetworkLength) / fixture.points.length ** 2;
          return Math.max(
            fixture.lowerCounts[index] * scale,
            Math.min(
              fixture.upperCounts[index] * scale,
              value + (observedCounts[index] - fixture.counts[index]) * scale
            )
          );
        });
        compareParity(
          className,
          comparison,
          'observed K (enumerated boundary intervals)',
          values.slice(0, bandCount),
          expected,
          fixture.tolerance
        );
        if (fixture.gpuSimulationCount) {
          const envelope = await resources.readFloat('envelope');
          const expectedEnvelope = new Array<number>(3 * bandCount).fill(0);
          const distributionDistances: number[] = [];
          for (let band = 0; band < bandCount; band++) {
            const simulations = Array.from(
              {length: fixture.gpuSimulationCount},
              (_, index) => values[(index + 1) * bandCount + band]
            );
            const simulationCounts = Array.from(
              {length: fixture.gpuSimulationCount},
              (_, index) => counts[(index + 1) * bandCount + band]
            );
            const referenceCounts = fixture.simulations.map(row =>
              Math.round(
                (row[band] * fixture.points.length ** 2) / (2 * fixture.exactNetworkLength)
              )
            );
            distributionDistances.push(getDistributionDistance(simulationCounts, referenceCounts));
            expectedEnvelope[band] = Math.min(...simulations);
            expectedEnvelope[bandCount + band] =
              simulations.reduce((sum, value) => sum + value, 0) / simulations.length;
            expectedEnvelope[2 * bandCount + band] = Math.max(...simulations);
          }
          // Union bound over all seeds/bands and both empirical distributions:
          // each CDF is within sqrt(log(4*m/alpha)/(2*n)); triangle inequality.
          const comparisons = fixture.gpuSeeds.length * bandCount;
          const logarithm = Math.log((4 * comparisons) / fixture.distributionAlpha);
          const bound =
            Math.sqrt(logarithm / (2 * fixture.gpuSimulationCount)) +
            Math.sqrt(logarithm / (2 * fixture.simulations.length));
          compareParity(
            className,
            comparison,
            'uniform-network simulation CDF distances',
            distributionDistances,
            new Array(bandCount).fill(0),
            {absolute: bound, relative: 0}
          );
          compareParity(
            className,
            {...comparison, reference: 'GPU simulation rows (envelope aggregation integrity only)'},
            'every envelope min/mean/max',
            envelope,
            expectedEnvelope,
            fixture.tolerance
          );
          // Integrating the CDF bound over the theoretical K support gives a
          // distribution-free bound for the independently simulated envelope mean.
          const maximumK =
            (fixture.exactNetworkLength * (fixture.points.length - 1)) / fixture.points.length;
          const referenceMeans = fixture.radii.map(
            (_, band) =>
              fixture.simulations.reduce((sum, row) => sum + row[band], 0) /
              fixture.simulations.length
          );
          compareParity(
            className,
            comparison,
            'every simulation-envelope mean (DKW integrated support bound)',
            envelope.slice(bandCount, 2 * bandCount),
            referenceMeans,
            {
              absolute: maximumK * bound + fixture.tolerance.absolute,
              relative: fixture.tolerance.relative
            }
          );
          const maximumPairs = (fixture.points.length * (fixture.points.length - 1)) / 2;
          compareParity(
            className,
            {...comparison, reference: 'unordered strict-pair-count definition (integrity only)'},
            'all simulation count ranges and monotone radii',
            counts.map((value, index) =>
              Number(
                value > maximumPairs ||
                  (index % bandCount === 0 ? value !== 0 : value < counts[index - 1])
              )
            ),
            new Array(counts.length).fill(0),
            EXACT
          );
          const normalized = counts
            .slice(bandCount)
            .map(value => (2 * value * fixture.exactNetworkLength) / fixture.points.length ** 2);
          compareParity(
            className,
            {
              ...comparison,
              reference:
                'Spaghetti normalization applied to every GPU simulation count (integrity only)'
            },
            'all simulated K values',
            values.slice(bandCount),
            normalized,
            fixture.tolerance
          );
        }
      } finally {
        compiled.destroy();
        resources.destroy();
      }
    }, 120000);
  }
}
