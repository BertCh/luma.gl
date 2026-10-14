// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  createFixedPointClearNode,
  createFixedPointDecodeNode,
  createFixedPointScaleNodes,
  FIXED_POINT_WGSL,
  getFixedPointAtomicAddWGSL,
  getFixedPointHeadroomBits
} from '../../src/utils/fixed-point-accumulator';
import {createWGSLKernelNode} from '../../src/utils/wgsl-kernel-nodes';
import {
  createNullWebGPUDevice,
  createVectorView,
  getGPUCommandNodePipelineKeys,
  getParsedKernelSources,
  getStorageBindingCount
} from './gpu-contributor-test-utils';
import {
  decodeOracle,
  getOracleUnitExponent,
  quantizeOracle,
  sumFixedPointOracle
} from './fixed-point-oracle';

it('getFixedPointHeadroomBits is the bit length of the term count', () => {
  expect(getFixedPointHeadroomBits(0)).toBe(1);
  expect(getFixedPointHeadroomBits(1)).toBe(1);
  expect(getFixedPointHeadroomBits(255)).toBe(8);
  expect(getFixedPointHeadroomBits(256)).toBe(9);
  expect(getFixedPointHeadroomBits(0xffffffff)).toBe(32);
  expect(() => getFixedPointHeadroomBits(2 ** 32)).toThrow(/maximumTermCount/);
  expect(() => getFixedPointHeadroomBits(1.5)).toThrow(/maximumTermCount/);
});

it('fixed-point nodes parse, share modules across sizes and bind few buffers', () => {
  const build = (rowCount: number, cellCount: number) => {
    const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: `fixed-point-${rowCount}`});
    const values = createTransientView(graph, 'values', 'float32', rowCount);
    const keys = createTransientView(graph, 'keys', 'uint32', rowCount);
    const scale = createTransientView(graph, 'scale', 'uint32', 1);
    const words = createTransientView(graph, 'words', 'uint32', cellCount * 3);
    const output = createTransientView(graph, 'output', 'float32', cellCount);
    return [
      ...createFixedPointScaleNodes(graph, {
        id: 'scale',
        operation: 'FixedPointSpec',
        values,
        scale
      }),
      createFixedPointClearNode(graph, {id: 'clear', operation: 'FixedPointSpec', words}),
      createWGSLKernelNode(graph, {
        id: 'accumulate',
        operation: 'FixedPointSpec',
        bindings: [
          {name: 'values', view: values, type: 'f32', access: 'read'},
          {name: 'keys', view: keys, type: 'u32', access: 'read'},
          {name: 'scale', view: scale, type: 'u32', access: 'read'},
          {name: 'words', view: words, type: 'atomic<u32>', access: 'read_write'}
        ],
        invocationCount: rowCount,
        workgroupSize: 64,
        declarations: `override HEADROOM_BITS: u32;
${FIXED_POINT_WGSL}
${getFixedPointAtomicAddWGSL('words')}`,
        constants: {HEADROOM_BITS: getFixedPointHeadroomBits(rowCount)},
        body: `let unitExponent = getFixedPointUnitExponent(scale[scaleOffset], HEADROOM_BITS);
  addFixedPointToWords(keys[keysOffset + index], quantizeFixedPoint(values[valuesOffset + index], unitExponent));`
      }),
      createFixedPointDecodeNode(graph, {
        id: 'decode',
        operation: 'FixedPointSpec',
        words,
        scale,
        maximumTermCount: rowCount,
        output
      })
    ];
  };
  const nodes = build(10_000, 64);
  const sources = getParsedKernelSources(nodes);
  expect([...sources.keys()]).toEqual(['scale-clear', 'scale-0', 'clear', 'accumulate', 'decode']);
  for (const [id, source] of sources) {
    expect(getStorageBindingCount(source), id).toBeLessThanOrEqual(4);
  }
  // Rule 12: sizes and headroom are overrides; a different size changes no module text.
  const sourceSet = (rowCount: number, cellCount: number) =>
    new Set([...getParsedKernelSources(build(rowCount, cellCount)).values()]);
  expect(sourceSet(70_000, 1000)).toEqual(sourceSet(10_000, 64));
  expect(getGPUCommandNodePipelineKeys(nodes).size).toBe(5);
});

it('createFixedPointScaleNodes adds one max node per non-empty chunk', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'fixed-point-chunks'});
  const chunks = [
    createTransientView(graph, 'chunk-0', 'float32', 300),
    createTransientView(graph, 'chunk-1', 'float32', 0),
    createTransientView(graph, 'chunk-2', 'float32', 10)
  ];
  const nodes = createFixedPointScaleNodes(graph, {
    id: 'scale',
    operation: 'FixedPointSpec',
    values: createVectorView('chunks', 'float32', chunks),
    scale: createTransientView(graph, 'scale', 'uint32', 1)
  });
  expect(nodes.map(node => node.id)).toEqual(['scale-clear', 'scale-0', 'scale-2']);
});

it('the fixed-point oracle keeps typical terms beside a 1e8 outlier and is order-independent', () => {
  // Library rule 7 / review B-01: one outlier must not zero the typical terms.
  const values = [1, 0.1, -0.25, 3e-3, 1e8, 1.5, -1e8];
  const keys = [0, 0, 0, 0, 1, 0, 1];
  const {sums, unitExponent} = sumFixedPointOracle(values, keys, 2);
  expect(sums[0]).toBe(Math.fround(1 + Math.fround(0.1) - 0.25 + Math.fround(3e-3) + 1.5));
  expect(sums[1]).toBe(0);
  // A term of magnitude 2^-20 next to 1e8 (2^26.6) with 7 terms still keeps all 24 bits.
  expect(quantizeOracle(2 ** -20, unitExponent) * 2n ** 23n).toBeGreaterThan(0n);
  const reversed = sumFixedPointOracle([...values].reverse(), [...keys].reverse(), 2);
  expect(reversed.sums).toEqual(sums);
  // Decoding rounds ties to even.
  expect(decodeOracle((1n << 24n) + 1n, 0)).toBe(2 ** 24);
  expect(decodeOracle((1n << 24n) + 3n, 0)).toBe(2 ** 24 + 4);
  expect(getOracleUnitExponent(0, 1)).toBe(-219);
});
