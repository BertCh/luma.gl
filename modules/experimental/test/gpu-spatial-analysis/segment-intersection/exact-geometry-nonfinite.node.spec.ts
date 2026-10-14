// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {EXACT_GEOMETRY_WGSL} from '../../../src/utils/exact-geometry-wgsl';
import {EXACT_ORIENTATION_WGSL} from '../../../src/utils/exact-orientation-wgsl';
import {createWGSLKernelNode} from '../../../src/utils/wgsl-kernel-nodes';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {createPointProbe, getExactProbeCrossing} from './exact-geometry-reference';
import {buildArrays, type OraclePoint} from './segment-intersection-oracle';
import {createCPUGraphMemory, executeCommandNodesOnCPU} from './wgsl-graph-executor';

it('non-finite edges contribute no crossing and preserve the WGSL uncertainty flag', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const values = [NaN, Infinity, -Infinity];
  const inputs = createTransientView(graph, 'inputs', 'float32', values.length);
  const outputs = createTransientView(graph, 'outputs', 'uint32x2', values.length);
  const node = createWGSLKernelNode(graph, {
    id: 'nonfinite-crossings',
    operation: 'Test',
    invocationCount: values.length,
    bindings: [
      {name: 'inputs', view: inputs, type: 'f32', access: 'read'},
      {name: 'outputs', view: outputs, type: 'i32', access: 'read_write'}
    ],
    declarations: `${EXACT_ORIENTATION_WGSL}\n${EXACT_GEOMETRY_WGSL}`,
    body: `let point = vec2f(0.0, 1.0);
      let result = exactProbeCrossing(ExactProbe(point, point, point, EXACT_PROBE_POINT),
        vec2f(2.0, inputs[inputsOffset + index]), vec2f(2.0, 2.0));
      outputs[outputsOffset + 2u * index] = result.x;
      outputs[outputsOffset + 2u * index + 1u] = result.y;`
  });
  const memory = createCPUGraphMemory();
  memory.write(inputs, Float32Array.from(values));
  executeCommandNodesOnCPU([node], memory);
  expect(Array.from(memory.readUint32(outputs))).toEqual([0, 2, 0, 2, 0, 2]);
  for (const value of values) {
    expect(getExactProbeCrossing(createPointProbe([0, 1]), [2, value], [2, 2])).toEqual({
      crossing: 0,
      boundary: false
    });
  }
});

it('flattens 200000-vertex lines and polygons without a function argument limit', () => {
  const vertices: OraclePoint[] = Array.from({length: 200_000}, (_, row) => [row, -row]);
  const lines = buildArrays([{kind: 'lines', vertices}]);
  const polygons = buildArrays([{kind: 'polygons', polygons: [[vertices]]}]);
  expect(lines.positions.length).toBe(400_000);
  expect(polygons.positions).toEqual(lines.positions);
  expect(Array.from(lines.lineOffsets)).toEqual([0, 200_000]);
  expect(Array.from(polygons.ringOffsets)).toEqual([0, 200_000]);
  expect(Array.from(polygons.positions.slice(-2))).toEqual([199_999, -199_999]);
});
