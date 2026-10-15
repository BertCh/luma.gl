// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUHistogram, type GraphVectorView} from '@luma.gl/gpgpu/gpu-core';
import {GPUData, GPUVector} from '@luma.gl/gpgpu/gpu-data';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {
  createInputBuffer,
  submitGraph
} from '../../../experimental/test/utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../../experimental/test/utils/require-test-device';
import {decodeOracleFixture} from '../../../experimental/test/oracles/oracle-fixture';
import {compareParity} from '../../../experimental/test/oracles/parity-wp6-compare';
import {
  ParityReductionHarness,
  getParityEarthquakes
} from '../../../experimental/test/oracles/parity-wp6-reduction-harness';
import fixtureJson from '../../../experimental/test/oracles/fixtures/parity-wp6-histogram.json';

import round8Json from '../../../experimental/test/oracles/fixtures/parity-wp6-round8-histograms.json';

type ScalarFormat = 'float32' | 'uint32' | 'sint32';
type Scene = {
  name: string;
  layouts?: string[];
  source: boolean;
  format: ScalarFormat;
  values: number[];
  mask: number[];
  bins: number;
  edges?: number[];
  domain?: [number, number] | 'auto';
  resolvedDomain?: [number, number];
  expected: number[];
};
const fixture = decodeOracleFixture<{reference: string; cases: Scene[]}>(fixtureJson);

for (const scene of [
  ...fixture.cases,
  ...decodeOracleFixture<{cases: Scene[]}>(round8Json).cases
]) {
  for (const layout of [
    'literal',
    'interleaved',
    'unmasked',
    'gpu-parameters',
    'independent-chunks',
    'auto-domain'
  ] as const) {
    if (scene.layouts && !scene.layouts.includes(layout)) continue;
    if (layout === 'unmasked' && !scene.name.startsWith('showcase')) continue;
    if (layout === 'auto-domain' && scene.domain !== 'auto') continue;
    it(`GPUHistogram NumPy parity: ${scene.name}/${layout}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const harness = new ParityReductionHarness(device);
      const vectors: GPUVector[] = [];
      const values = scene.source ? getParityEarthquakes().values : scene.values;
      const masks = scene.source ? Array(values.length).fill(1) : scene.mask;
      const ScalarArray =
        scene.format === 'uint32'
          ? Uint32Array
          : scene.format === 'sint32'
            ? Int32Array
            : Float32Array;
      const makeVector = <Format extends ScalarFormat>(
        name: string,
        entries: number[],
        format: Format,
        firstBoundary: number,
        secondBoundary: number
      ): GraphVectorView<Format> => {
        const ArrayType =
          format === 'uint32' ? Uint32Array : format === 'sint32' ? Int32Array : Float32Array;
        // Explicit empty chunks and different mask/value seams at 255/256/257 exercise alignment.
        const boundaries = [
          0,
          0,
          Math.min(firstBoundary, entries.length),
          Math.min(secondBoundary, entries.length),
          entries.length
        ];
        const data = boundaries.slice(1).map((end, index) => {
          const chunk = ArrayType.from(entries.slice(boundaries[index], end));
          const buffer = createInputBuffer(device, chunk.length ? chunk : new Uint32Array(1));
          harness.buffers.push(buffer);
          return new GPUData({buffer, format, length: chunk.length, ownsBuffer: false});
        });
        const vector = new GPUVector({type: 'data', name, format, data, ownsData: true});
        vectors.push(vector);
        return harness.graph.importGPUVector(name, vector);
      };
      const interleavedBuffer =
        layout === 'interleaved'
          ? createInputBuffer(
              device,
              // Even an empty view must have storage through its four-byte offset.
              ScalarArray.from(values.length ? values.flatMap(value => [123, value, 456]) : [123])
            )
          : undefined;
      if (interleavedBuffer) harness.buffers.push(interleavedBuffer);
      const input = interleavedBuffer
        ? harness.graph.createDataView(
            harness.graph.importBuffer(
              {
                id: 'interleaved',
                byteLength: interleavedBuffer.byteLength,
                usage: interleavedBuffer.usage
              },
              interleavedBuffer
            ),
            {format: scene.format, length: values.length, byteOffset: 4, byteStride: 12}
          )
        : layout === 'independent-chunks'
          ? makeVector('values', values, scene.format, 1, 256)
          : harness.input('values', ScalarArray.from(values), scene.format);
      const mask =
        layout === 'unmasked'
          ? undefined
          : layout === 'independent-chunks'
            ? makeVector('mask', masks, 'uint32', 2, 257)
            : harness.input('mask', Uint32Array.from(masks), 'uint32');
      const output = harness.output('counts', scene.bins, 'uint32');
      if (scene.edges) {
        const edges =
          layout === 'gpu-parameters'
            ? harness.input('edges', ScalarArray.from(scene.edges), scene.format)
            : scene.edges;
        harness.graph.add(new GPUHistogram({input, mask, output, edges}));
      } else {
        const domain =
          scene.domain === 'auto' &&
          (layout === 'literal' || layout === 'gpu-parameters' || layout === 'interleaved')
            ? scene.resolvedDomain!
            : scene.domain!;
        harness.graph.add(
          new GPUHistogram({
            input,
            mask,
            output,
            domain:
              layout === 'gpu-parameters' && domain !== 'auto'
                ? harness.input('domain', ScalarArray.from(domain), scene.format)
                : domain
          })
        );
      }
      let compiled: ReturnType<typeof harness.graph.compile> | undefined;
      try {
        compiled = harness.graph.compile();
        // A second encoding must clear all bins; parity is checked for every bin both times.
        for (let encoding = 0; encoding < 2; encoding++) {
          submitGraph(device, compiled, undefined);
          const actual = await harness.read();
          compareParity(
            'GPUHistogram',
            `${scene.name}/${layout}/encoding=${encoding}`,
            fixture.reference,
            values.length,
            'counts',
            actual['counts'],
            scene.expected,
            true
          );
        }
      } finally {
        compiled?.destroy();
        for (const vector of vectors) vector.destroy();
        harness.destroy();
      }
    }, 120000);
  }
}
