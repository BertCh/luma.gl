// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUGraph, GPUGraphTopology, GPUGraphLabelPropagation} from '@luma.gl/gpgpu/gpu-graph';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {submitGraph} from '../../../experimental/test/utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../../experimental/test/utils/require-test-device';
import {compareParity, EXACT} from '../../../experimental/test/gpu-network/parity-wp3-comparison';
import {createParityResources} from '../../../experimental/test/gpu-network/parity-wp3-round2-resources';
import {
  canonicalizePartition,
  communityFixtures,
  evaluatePartition,
  getAdjustedRandIndex,
  getUnstableVertices
} from './parity-wp3-graph-communities-fixture';

const propagationTrials = communityFixtures
  .filter(value => value.runs.length)
  .flatMap(fixture =>
    (fixture.exactPartition ? Array.from({length: 100}, (_, index) => index + 1) : [4, 32, 64]).map(
      iterations => ({fixture, iterations})
    )
  );
for (const {fixture, iterations} of propagationTrials) {
  it(`GPUGraphLabelPropagation NetworkX quality and stability: ${fixture.case}/iterations-${iterations}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const partitions: number[][] = [];
    for (const run of fixture.runs) {
      const resources = createParityResources(device);
      const vertexCount = fixture.nodeCount;
      const capacity = 2 * fixture.edges.length;
      const graph = new GPUGraph({
        vertexCount,
        directed: false,
        sourceVertices: resources.vector(
          'sources',
          'uint32',
          fixture.edges.map(edge => run.permutation[edge[0]])
        ),
        targetVertices: resources.vector(
          'targets',
          'uint32',
          fixture.edges.map(edge => run.permutation[edge[1]])
        )
      });
      const topology = new GPUGraphTopology({
        graph,
        forward: {
          offsets: resources.vector('offsets', 'uint32', new Array(vertexCount + 1).fill(0)),
          neighbors: resources.vector('neighbors', 'uint32', new Array(capacity).fill(0)),
          edgeIds: resources.vector('edges', 'uint32', new Array(capacity).fill(0)),
          count: resources.vector('count', 'uint32', [0]),
          overflow: resources.vector('overflow', 'uint32', [0])
        },
        invalidEdgeCount: resources.vector('invalid', 'uint32', [0])
      });
      topology.addToGraph(resources.graph);
      new GPUGraphLabelPropagation({
        topology,
        iterations,
        output: resources.vector('labels', 'uint32', new Array(vertexCount).fill(0)),
        converged: resources.vector('converged', 'uint32', [0])
      }).addToGraph(resources.graph);
      const compiled = resources.graph.compile();
      try {
        submitGraph(device, compiled, undefined);
        const labels = await resources.readUint('labels');
        const className = 'GPUGraphLabelPropagation';
        const comparison = {
          ...fixture,
          case: `${fixture.case}/iterations-${iterations}/permutation-${run.seed}`
        };
        compareParity(
          className,
          comparison,
          'complete valid partition',
          labels.map(label => Number(label >= vertexCount)),
          new Array(vertexCount).fill(0),
          EXACT
        );
        compareParity(
          className,
          comparison,
          'topology overflow and invalid edges',
          [...(await resources.readUint('overflow')), ...(await resources.readUint('invalid'))],
          [0, 0],
          EXACT
        );
        // Avoid downstream indexing invalid labels, after the parity report has recorded them.
        if (labels.some(label => label >= vertexCount)) continue;
        const inverse = new Array<number>(vertexCount);
        run.permutation.forEach((permuted, original) => {
          inverse[permuted] = original;
        });
        const originalLabels = run.permutation.map(permuted => inverse[labels[permuted]]);
        const canonical = canonicalizePartition(originalLabels);
        partitions.push(canonical);
        compareParity(
          className,
          comparison,
          'all labels remain inside NetworkX connected components',
          originalLabels.map((label, node) =>
            Number(fixture.components[label] !== fixture.components[node])
          ),
          new Array(vertexCount).fill(0),
          EXACT
        );
        const unstable = getUnstableVertices(fixture, originalLabels, run.permutation);
        const converged = (await resources.readUint('converged'))[0];
        compareParity(
          className,
          {
            ...comparison,
            reference: 'documented synchronous vote fixed-point contract (integrity only)'
          },
          'reported convergence implies fixed point',
          unstable.map(value => (converged ? value : 0)),
          new Array(vertexCount).fill(0),
          EXACT
        );
        const score = evaluatePartition(fixture, originalLabels, 1, false).reduce(
          (sum, value) => sum + value,
          0
        );
        const qualityRatio = run.score === 0 ? null : score / run.score;
        console.log(
          'PARITY',
          JSON.stringify({
            class: className,
            case: comparison.case,
            reference: 'networkx.community.asyn_lpa_communities; networkx.community.modularity',
            n: vertexCount,
            metric: 'heuristic modularity quality (different algorithms)',
            maxAbs: Math.abs(score - run.score),
            maxRel: qualityRatio === null ? null : Math.abs(qualityRatio - 1),
            mismatches: 0,
            tolerance: {
              profile: 'H',
              comparison: 'descriptive quality ratio; no equality assertion',
              gpuScore: score,
              referenceScore: run.score,
              qualityRatio,
              converged,
              unstableVertices: unstable.reduce((sum, value) => sum + value, 0),
              communities: new Set(canonical).size
            }
          })
        );
        if (fixture.exactPartition) {
          compareParity(
            className,
            comparison,
            'unambiguous clique and island partition',
            canonical,
            run.labels,
            EXACT
          );
          compareParity(
            className,
            comparison,
            'unambiguous partition objective',
            [score],
            [run.score],
            fixture.tolerance
          );
          if (iterations > 1)
            compareParity(className, comparison, 'cliques converge', [converged], [1], EXACT);
        }
      } finally {
        compiled.destroy();
        resources.destroy();
      }
    }
    for (let first = 0; first < partitions.length; first++) {
      for (let second = first + 1; second < partitions.length; second++) {
        const actual = getAdjustedRandIndex(partitions[first], partitions[second]);
        const reference = fixture.stability[first][second];
        console.log(
          'PARITY',
          JSON.stringify({
            class: 'GPUGraphLabelPropagation',
            case: `${fixture.case}/stability-${first}-${second}`,
            reference: 'sklearn.metrics.adjusted_rand_score on NetworkX LPA runs',
            n: fixture.nodeCount,
            metric: 'partition stability across vertex permutations versus reference seeds',
            maxAbs: Math.abs(actual - reference),
            maxRel: Math.abs(actual - reference) / Math.max(Math.abs(reference), Number.EPSILON),
            mismatches: 0,
            tolerance: {
              profile: 'H',
              comparison: 'descriptive only; algorithms and seeds differ',
              gpuAdjustedRand: actual,
              referenceAdjustedRand: reference
            }
          })
        );
      }
    }
  }, 180000);
}
