// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUGraphPageRank} from '@luma.gl/gpgpu/gpu-graph';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {GPUNetworkAnalyticsColumns} from '../../../experimental/src/gpu-network/network-analysis/gpu-network-analytics-columns';
import {submitGraph} from '../../../experimental/test/utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../../experimental/test/utils/require-test-device';
import {compareParity, EXACT} from '../../../experimental/test/gpu-network/parity-wp3-comparison';
import {createParityResources} from '../../../experimental/test/gpu-network/parity-wp3-round2-resources';
import {createMetricsTopology} from './parity-wp3-graph-metrics-fixture';
import {evaluatePartition} from './parity-wp3-graph-communities-fixture';
import {showcaseGraphs} from './parity-wp3-graph-showcase-fixture';

for (const fixture of showcaseGraphs) {
  for (const run of fixture.runs) {
    it(`finite PageRank/analytics ${fixture.case}/${run.damping}/${run.iterations}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const resources = createParityResources(device);
      const topology = createMetricsTopology(resources, fixture);
      new GPUGraphPageRank({
        topology,
        damping: run.damping,
        iterations: run.iterations,
        output: resources.vector('ranks', 'float32', new Array(fixture.nodeCount).fill(0)),
        residual: resources.vector('residual', 'float32', [0])
      }).addToGraph(resources.graph);
      resources.graph.add(
        new GPUNetworkAnalyticsColumns({
          topology: {
            offsets: resources.input('csr-offsets', 'uint32', fixture.offsets),
            neighbors: resources.input('csr-neighbors', 'uint32', fixture.neighbors)
          },
          reverseTopology: fixture.directed
            ? {
                offsets: resources.input('reverse-csr-offsets', 'uint32', fixture.reverseOffsets),
                neighbors: resources.input(
                  'reverse-csr-neighbors',
                  'uint32',
                  fixture.reverseNeighbors
                )
              }
            : undefined,
          pageRank: {
            output: resources.output('column', 'float32', fixture.nodeCount),
            normalized: resources.output('normalized', 'float32', fixture.nodeCount),
            extent: resources.output('extent', 'float32', 2),
            damping: run.damping,
            iterationCount: run.iterations,
            residual: resources.output('column-residual', 'float32', 1)
          }
        })
      );
      const compiled = resources.graph.compile();
      try {
        submitGraph(device, compiled, undefined);
        const comparison = {
          ...fixture,
          case: `${fixture.case}/alpha-${run.damping}/iterations-${run.iterations}`
        };
        const tolerance = {absolute: 1e-6 * Math.max(...run.ranks), relative: 1e-4};
        for (const [className, output, residual] of [
          ['GPUGraphPageRank', 'ranks', 'residual'],
          ['GPUNetworkAnalyticsColumns', 'column', 'column-residual']
        ]) {
          compareParity(
            className,
            comparison,
            'every finite-iteration rank versus Google matrix power',
            await resources.readFloat(output),
            run.ranks,
            tolerance
          );
          // L1 change is Lipschitz in both iterates: sum their unchanged S budgets.
          const residualBound = [...run.ranks, ...run.previous].reduce(
            (sum, value) => sum + tolerance.absolute + tolerance.relative * value,
            0
          );
          compareParity(
            className,
            comparison,
            'finite-iteration L1 change',
            await resources.readFloat(residual),
            [run.residual],
            {absolute: residualBound, relative: 0}
          );
        }
        const minimum = Math.min(...run.ranks);
        const maximum = Math.max(...run.ranks);
        compareParity(
          'GPUNetworkAnalyticsColumns',
          comparison,
          'rank extent',
          await resources.readFloat('extent'),
          [minimum, maximum],
          tolerance
        );
        compareParity(
          'GPUNetworkAnalyticsColumns',
          comparison,
          'normalized finite rank',
          await resources.readFloat('normalized'),
          run.ranks.map(value => (maximum > minimum ? (value - minimum) / (maximum - minimum) : 0)),
          {absolute: 1e-6, relative: 1e-4}
        );
      } finally {
        compiled.destroy();
        resources.destroy();
      }
    });
  }
  for (const communityIterations of [2, 5, 10, 32, 100]) {
    for (const componentIterations of [4, 16, 32, 128]) {
      it(`analytics communities ${fixture.case}/${communityIterations}/${componentIterations}`, async () => {
        const device = await getWebGPUTestDevice();
        requireTestDevice(device);
        const resources = createParityResources(device);
        resources.graph.add(
          new GPUNetworkAnalyticsColumns({
            topology: {
              offsets: resources.input('offsets', 'uint32', fixture.offsets),
              neighbors: resources.input('neighbors', 'uint32', fixture.neighbors)
            },
            reverseTopology: fixture.directed
              ? {
                  offsets: resources.input('reverse-offsets', 'uint32', fixture.reverseOffsets),
                  neighbors: resources.input(
                    'reverse-neighbors',
                    'uint32',
                    fixture.reverseNeighbors
                  )
                }
              : undefined,
            communities: {
              output: resources.output('communities', 'uint32', fixture.nodeCount),
              maximumIterations: communityIterations,
              status: {converged: resources.output('community-converged', 'uint32', 1)}
            },
            components: {
              output: resources.output('components', 'uint32', fixture.nodeCount),
              maximumIterations: componentIterations,
              status: {converged: resources.output('component-converged', 'uint32', 1)}
            }
          })
        );
        const compiled = resources.graph.compile();
        try {
          submitGraph(device, compiled, undefined);
          const comparison = {
            ...fixture,
            case: `${fixture.case}/communities-${communityIterations}/components-${componentIterations}`,
            reference: 'networkx.connected_components; asyn_lpa_communities; modularity'
          };
          compareParity(
            'GPUNetworkAnalyticsColumns',
            comparison,
            'every weak component',
            await resources.readUint('components'),
            fixture.components,
            EXACT
          );
          compareParity(
            'GPUNetworkAnalyticsColumns',
            comparison,
            'component convergence',
            await resources.readUint('component-converged'),
            [1],
            EXACT
          );
          const labels = await resources.readUint('communities');
          compareParity(
            'GPUNetworkAnalyticsColumns',
            comparison,
            'every community label is valid and inside its weak component',
            labels.map((label, node) =>
              Number(
                label >= fixture.nodeCount || fixture.components[label] !== fixture.components[node]
              )
            ),
            new Array(fixture.nodeCount).fill(0),
            EXACT
          );
          if (labels.some(label => label >= fixture.nodeCount)) return;
          const score = evaluatePartition(
            {...fixture, directed: false, edges: fixture.communityEdges},
            labels,
            1
          ).reduce((sum, value) => sum + value, 0);
          console.log(
            'PARITY',
            JSON.stringify({
              class: 'GPUNetworkAnalyticsColumns',
              case: comparison.case,
              reference: comparison.reference,
              n: fixture.nodeCount,
              metric: 'community modularity quality',
              maxAbs: Math.abs(score - fixture.communityScore),
              maxRel: Math.abs(score / fixture.communityScore - 1),
              mismatches: 0,
              tolerance: {
                profile: 'H',
                comparison: 'descriptive ratio, synchronous and asynchronous algorithms differ',
                gpuScore: score,
                referenceScore: fixture.communityScore,
                ratio: score / fixture.communityScore
              }
            })
          );
        } finally {
          compiled.destroy();
          resources.destroy();
        }
      });
    }
  }
}
