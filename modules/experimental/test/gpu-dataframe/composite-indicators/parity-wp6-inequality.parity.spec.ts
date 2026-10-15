// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {GPUInequality} from '../../../src/gpu-dataframe/composite-indicators/gpu-inequality';
import {
  getGPUInequalityParameterValues,
  type GPUInequalitySettings
} from '../../../src/gpu-dataframe/composite-indicators/inequality-parameters';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {decodeOracleFixture} from '../../oracles/oracle-fixture';
import {compareParity} from '../../oracles/parity-wp6-compare';
import fixtureJSON from '../../oracles/fixtures/parity-wp6-inequality.json';

type FloatMetric =
  | 'gini'
  | 'theilT'
  | 'theilL'
  | 'atkinson'
  | 'hoover'
  | 'palma'
  | 'mean'
  | 'lorenzKnots';
type InequalityCase = {
  name: string;
  values: number[];
  zoneIds: number[];
  zoneCount: number;
  weights?: number[];
  mask: number[];
  runs: {
    settings: GPUInequalitySettings;
    expected: Record<FloatMetric | 'counts' | 'globalSummary', number[]>;
  }[];
};
const fixture = decodeOracleFixture<{lorenzKnotCount: number; cases: InequalityCase[]}>(
  fixtureJSON
);
const references: Record<FloatMetric, string> = {
  gini: 'inequality.gini.Gini',
  theilT: 'inequality.theil.Theil + scipy.stats.entropy',
  theilL: 'scipy.stats.entropy(population, income); documented zero-income NaN',
  atkinson: 'inequality.atkinson.Atkinson; zero-income composite Atkinson(1970)/scipy.stats.pmean',
  hoover: 'inequality.schutz.Schutz.distance',
  palma: 'inequality.schutz.Schutz Lorenz table + numpy.interp',
  mean: 'numpy.average',
  lorenzKnots: 'inequality.schutz.Schutz Lorenz table + numpy.interp'
};
const summaryNames = [
  'totalTheilT',
  'betweenTheilT',
  'withinTheilT',
  'pooledGini',
  'count',
  'mean',
  'totalWeight',
  'totalIncome'
];
const summaryReferences = [
  'inequality.theil.TheilD.T',
  'inequality.theil.TheilD.bg',
  'inequality.theil.TheilD.wg',
  'inequality.gini.Gini',
  'numpy.count_nonzero',
  'numpy.average',
  'numpy.sum',
  'numpy.dot'
];

for (const scene of fixture.cases) {
  it(`GPUInequality external parity: ${scene.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const buffers: Buffer[] = [];
    const graph = new GPUCommandGraph(device);
    const track = (buffer: Buffer) => {
      buffers.push(buffer);
      return buffer;
    };
    const inputFloat = (name: string, values: number[]) =>
      importGraphBuffer(
        graph,
        name,
        track(createInputBuffer(device, Float32Array.from(values))),
        'float32',
        values.length
      );
    const inputUint = (name: string, values: number[]) =>
      importGraphBuffer(
        graph,
        name,
        track(createInputBuffer(device, Uint32Array.from(values))),
        'uint32',
        values.length
      );
    const lengths: Record<FloatMetric, number> = {
      gini: scene.zoneCount,
      theilT: scene.zoneCount,
      theilL: scene.zoneCount,
      atkinson: scene.zoneCount,
      hoover: scene.zoneCount,
      palma: scene.zoneCount,
      mean: scene.zoneCount,
      lorenzKnots: scene.zoneCount * fixture.lorenzKnotCount
    };
    const outputBuffers = new Map<FloatMetric, Buffer>();
    const outputFloat = (name: FloatMetric) => {
      const buffer = track(createOutputBuffer(device, lengths[name]));
      outputBuffers.set(name, buffer);
      return importGraphBuffer(graph, name, buffer, 'float32', lengths[name]);
    };
    const counts = track(createOutputBuffer(device, scene.zoneCount));
    const globalSummary = track(createOutputBuffer(device, 8));
    const parameters = new GPUParameterBuffer(device, {
      id: 'parity-wp6-parameters',
      format: 'float32',
      length: 4
    });
    graph.add(
      new GPUInequality({
        values: inputFloat('values', scene.values),
        zoneIds: inputUint('zones', scene.zoneIds),
        weights: scene.weights ? inputFloat('weights', scene.weights) : undefined,
        mask: inputUint('mask', scene.mask),
        zoneCount: scene.zoneCount,
        lorenzKnotCount: fixture.lorenzKnotCount,
        parameters: parameters.importToGraph(graph),
        output: {
          gini: outputFloat('gini'),
          theilT: outputFloat('theilT'),
          theilL: outputFloat('theilL'),
          atkinson: outputFloat('atkinson'),
          hoover: outputFloat('hoover'),
          palma: outputFloat('palma'),
          mean: outputFloat('mean'),
          lorenzKnots: outputFloat('lorenzKnots'),
          counts: importGraphBuffer(graph, 'counts', counts, 'uint32', scene.zoneCount),
          globalSummary: importGraphBuffer(graph, 'global-summary', globalSummary, 'float32', 8)
        }
      })
    );
    const compiled = graph.compile();
    try {
      for (const run of scene.runs) {
        parameters.write(getGPUInequalityParameterValues(run.settings));
        submitGraph(device, compiled, undefined);
        const caseName = `${scene.name}/epsilon=${run.settings.epsilon}`;
        for (const [metric, buffer] of outputBuffers) {
          compareParity(
            'GPUInequality',
            caseName,
            references[metric],
            scene.values.length,
            metric,
            await readFloat32(buffer, lengths[metric]),
            run.expected[metric]
          );
        }
        compareParity(
          'GPUInequality',
          caseName,
          'numpy.count_nonzero',
          scene.values.length,
          'counts',
          await readUint32(counts, scene.zoneCount),
          run.expected.counts,
          true
        );
        const summary = await readFloat32(globalSummary, 8);
        for (let index = 0; index < summaryNames.length; index++) {
          compareParity(
            'GPUInequality',
            caseName,
            summaryReferences[index],
            scene.values.length,
            `globalSummary.${summaryNames[index]}`,
            [summary[index]],
            [run.expected.globalSummary[index]],
            index === 4
          );
        }
      }
    } finally {
      compiled.destroy();
      parameters.destroy();
      for (const buffer of buffers) buffer.destroy();
    }
  });
}
