// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {
  GPUSpatialWeightsAlgebra,
  type GPUSpatialWeightsAlgebraProps
} from '../../../src/gpu-spatial-analysis/spatial-weights';
import fixture from '../../oracles/fixtures/parity-wp1-round8-algebra.json';
import {decodeRound4Fixture} from '../../oracles/parity-wp1-round4-fixture';
import {sumSparseRows, type AlgebraFixture} from '../../oracles/parity-wp1-round6-fixture';
import {compareParity} from '../../oracles/parity-wp1-compare';
import {readUint32} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {WeightsRig} from './spatial-weights-harness';

const decoded = decodeRound4Fixture<AlgebraFixture>(fixture);
for (let caseIndex = 0; caseIndex < 57; caseIndex++) {
  it(`GPUSpatialWeightsAlgebra showcase controls SciPy output case ${caseIndex}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const {datasets, cases} = await decoded;
    const testCase = cases[caseIndex];
    const dataset = datasets[testCase.datasetIndex];
    const fullCount = testCase.expected.neighbors.length;
    const capacities = [fullCount + 5];
    if (testCase.operation === 'union' && testCase.weightRule === 'sum') capacities.push(3);
    for (const capacity of capacities) {
      const rig = new WeightsRig(device);
      try {
        const left = rig.uploadWeights(dataset.left, 3);
        const right = rig.uploadWeights(dataset.right, 2);
        const destination = rig.weightsOutput(dataset.n, capacity);
        const overflow = rig.output('uint32', 1);
        const requiredCount = rig.output('uint32', 1);
        const output = {
          ...destination.spatialWeights,
          status: {overflow: overflow.view, requiredCount: requiredCount.view}
        };
        let props: GPUSpatialWeightsAlgebraProps;
        switch (testCase.operation) {
          case 'block':
            props = {
              operation: 'block',
              groupIds: rig.input(Uint32Array.from(dataset.groups), 'uint32', dataset.n),
              groupCount: dataset.groupCount,
              output
            };
            break;
          case 'subgraph':
            props = {
              operation: 'subgraph',
              weights: left,
              mask: rig.input(Uint32Array.from(dataset.mask), 'uint32', dataset.n),
              output
            };
            break;
          case 'selfWeight':
            props = {
              operation: 'selfWeight',
              weights: left,
              selfWeight: rig.input(
                new Float32Array(dataset.n).fill(testCase.selfWeight!),
                'float32',
                dataset.n
              ),
              output
            };
            break;
          case 'higherOrder':
            // Enough room for all intermediate frontiers; only the output bound
            // is tested here. This is a capacity bound, not a numerical tolerance.
            props = {
              operation: 'higherOrder',
              weights: left,
              order: testCase.order!,
              cumulative: testCase.cumulative,
              workCapacity: testCase.workCapacity!,
              output
            };
            break;
          default:
            props = {
              operation: testCase.operation,
              left,
              right,
              weightRule: testCase.weightRule,
              output
            };
        }
        rig.run(new GPUSpatialWeightsAlgebra(props));
        const actual = await destination.read();
        const expected = {
          offsets: testCase.expected.offsets.map(value => Math.min(value, capacity)),
          neighbors: testCase.expected.neighbors.slice(0, capacity),
          weights: testCase.expected.weights.slice(0, capacity),
          distances: []
        };
        const comparison = {...testCase, name: `${testCase.name}/capacity-${capacity}`};
        const compare = (metric: string, measured: number[], reference: number[], exact = false) =>
          compareParity(
            'GPUSpatialWeightsAlgebra',
            comparison,
            metric,
            measured,
            reference,
            exact ? 'E' : 'S'
          );
        compare('offsets', actual.offsets, expected.offsets, true);
        compare('neighbors', actual.neighbors, expected.neighbors, true);
        compare('weights', actual.weights, expected.weights);
        compare(
          'rowSums',
          sumSparseRows(actual),
          capacity >= fullCount ? testCase.rowSums : sumSparseRows(expected)
        );
        compare(
          'overflow',
          await readUint32(overflow.buffer, 1),
          [Number(fullCount > capacity)],
          true
        );
        compare('requiredCount', await readUint32(requiredCount.buffer, 1), [fullCount], true);
      } finally {
        rig.destroy();
      }
    }
  });
}
