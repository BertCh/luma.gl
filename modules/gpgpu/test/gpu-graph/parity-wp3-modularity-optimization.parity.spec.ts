// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUGraph,
  GPUGraphTopology,
  GPUGraphModularityOptimization,
  type GPUGraphAdjacency
} from '@luma.gl/gpgpu/gpu-graph';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {submitGraph} from '../../../experimental/test/utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../../experimental/test/utils/require-test-device';
import {compareParity, EXACT} from '../../../experimental/test/gpu-network/parity-wp3-comparison';
import {createParityResources} from '../../../experimental/test/gpu-network/parity-wp3-round2-resources';
import {
  maximumMoveGain,
  optimizationFixtures,
  scorePartition
} from './parity-wp3-modularity-optimization-fixture';

for (const fixture of optimizationFixtures) {
  const runs = [
    ...fixture.runs.map(run => ({...run, minimumGain: 0})),
    ...(fixture.nodeCount < 20
      ? [0, 128, 256, 512, 1024].flatMap(iterations =>
          [0, 0.0001, 0.001].map(minimumGain => ({...fixture.runs[0], iterations, minimumGain}))
        )
      : [])
  ];
  for (const run of runs) {
    const caseName = `${fixture.case}/${run.start}/resolution-${run.resolution}/seed-${run.seed}/iterations-${run.iterations}/gain-${run.minimumGain}`;
    it(`GPUGraphModularityOptimization independent objective and quality: ${caseName}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const resources = createParityResources(device);
      const vertexCount = fixture.nodeCount;
      const capacity = 2 * fixture.edges.length;
      const empty = () => new Array<number>(vertexCount).fill(0);
      const adjacency = (prefix: string): GPUGraphAdjacency => ({
        offsets: resources.vector(
          `${prefix}-offsets`,
          'uint32',
          new Array(vertexCount + 1).fill(0)
        ),
        neighbors: resources.vector(`${prefix}-neighbors`, 'uint32', new Array(capacity).fill(0)),
        edgeIds: resources.vector(`${prefix}-edges`, 'uint32', new Array(capacity).fill(0)),
        edgeWeights: resources.vector(`${prefix}-weights`, 'float32', new Array(capacity).fill(0)),
        count: resources.vector(`${prefix}-count`, 'uint32', [0]),
        overflow: resources.vector(`${prefix}-overflow`, 'uint32', [0])
      });
      const graph = new GPUGraph({
        vertexCount,
        directed: fixture.directed,
        sourceVertices: resources.vector(
          'sources',
          'uint32',
          fixture.edges.map(edge => run.permutation[edge[0]])
        ),
        targetVertices: resources.vector(
          'targets',
          'uint32',
          fixture.edges.map(edge => run.permutation[edge[1]])
        ),
        edgeWeights: resources.vector(
          'weights',
          'float32',
          fixture.edges.map(edge => edge[2])
        )
      });
      const topology = new GPUGraphTopology({
        graph,
        forward: adjacency('forward'),
        reverse: fixture.directed ? adjacency('reverse') : undefined,
        invalidEdgeCount: resources.vector('invalid', 'uint32', [0])
      });
      const initial = empty();
      run.permutation.forEach((permuted, original) => {
        initial[permuted] = run.permutation[run.initial[original]];
      });
      topology.addToGraph(resources.graph);
      new GPUGraphModularityOptimization({
        topology,
        initialCommunities: resources.vector('initial', 'uint32', initial),
        output: resources.vector('labels', 'uint32', empty()),
        modularity: resources.vector('score', 'float32', [0]),
        converged: resources.vector('converged', 'uint32', [0]),
        valid: resources.vector('valid', 'uint32', [0]),
        resolution: run.resolution,
        iterations: run.iterations,
        minimumGain: run.minimumGain
      }).addToGraph(resources.graph);
      const compiled = resources.graph.compile();
      try {
        submitGraph(device, compiled, undefined);
        const comparison = {...fixture, case: caseName};
        const className = 'GPUGraphModularityOptimization';
        const labels = await resources.readUint('labels');
        compareParity(
          className,
          comparison,
          'every label is valid',
          labels.map(label => Number(label >= vertexCount)),
          empty(),
          EXACT
        );
        compareParity(
          className,
          comparison,
          'validity',
          await resources.readUint('valid'),
          [1],
          EXACT
        );
        for (const output of [
          'invalid',
          'forward-overflow',
          ...(fixture.directed ? ['reverse-overflow'] : [])
        ]) {
          compareParity(
            className,
            comparison,
            output,
            await resources.readUint(output),
            [0],
            EXACT
          );
        }
        if (labels.some(label => label >= vertexCount)) return;
        const inverse = empty();
        run.permutation.forEach((permuted, original) => {
          inverse[permuted] = original;
        });
        const originalLabels = run.permutation.map(permuted => inverse[labels[permuted]]);
        // Community IDs need not be member vertex IDs after a singleton split.
        const communityComponents = new Map<number, number>();
        const violations = originalLabels.map((label, vertex) => {
          const component = fixture.components[vertex];
          const prior = communityComponents.get(label);
          communityComponents.set(label, component);
          return Number(prior !== undefined && prior !== component);
        });
        compareParity(
          className,
          comparison,
          'no community spans disconnected components',
          violations,
          empty(),
          EXACT
        );
        const independentScore = scorePartition(fixture, originalLabels, run.resolution);
        compareParity(
          className,
          comparison,
          'reported objective versus independent Newman definition',
          await resources.readFloat('score'),
          [independentScore],
          fixture.tolerance
        );
        compareParity(
          className,
          comparison,
          'objective does not decrease from initial partition',
          [Math.min(0, independentScore - run.initialScore)],
          [0],
          fixture.tolerance
        );
        compareParity(
          className,
          comparison,
          'changed vertices fit accepted-move budget',
          [
            Number(
              originalLabels.filter((label, vertex) => label !== run.initial[vertex]).length >
                run.iterations
            )
          ],
          [0],
          EXACT
        );
        const converged = (await resources.readUint('converged'))[0];
        compareParity(
          className,
          comparison,
          'convergence flag is Boolean',
          [Number(converged > 1)],
          [0],
          EXACT
        );
        if (vertexCount < 20 && converged) {
          const gain = maximumMoveGain(fixture, originalLabels, run.resolution);
          // A gain subtracts two S-profile objective evaluations; sum their error budgets.
          const gainTolerance =
            2 * fixture.tolerance.absolute +
            fixture.tolerance.relative *
              (Math.abs(independentScore) + Math.abs(independentScore + gain));
          compareParity(
            className,
            comparison,
            'converged small partition has no improving single-vertex move',
            [Math.max(0, gain - run.minimumGain)],
            [0],
            {absolute: gainTolerance, relative: 0}
          );
        }
        console.log(
          'PARITY',
          JSON.stringify({
            class: className,
            case: caseName,
            reference: fixture.reference,
            n: vertexCount,
            metric: 'heuristic quality across reference seeds (not equal-answer parity)',
            maxAbs: Math.abs(independentScore - run.score),
            maxRel: run.score ? Math.abs(independentScore / run.score - 1) : null,
            mismatches: 0,
            tolerance: {
              profile: 'H',
              comparison: 'descriptive',
              gpuScore: independentScore,
              initialScore: run.initialScore,
              louvainScore: run.score,
              greedyScore: run.greedyScore,
              louvainQualityRatio: run.score ? independentScore / run.score : null,
              greedyQualityRatio: run.greedyScore ? independentScore / run.greedyScore : null,
              referenceDistribution: fixture.runs
                .filter(value => value.resolution === run.resolution && value.start === run.start)
                .map(value => value.score),
              communities: new Set(originalLabels).size,
              converged,
              acceptedMoveBudget: run.iterations
            }
          })
        );
      } finally {
        compiled.destroy();
        resources.destroy();
      }
    }, 180000);
  }
}
