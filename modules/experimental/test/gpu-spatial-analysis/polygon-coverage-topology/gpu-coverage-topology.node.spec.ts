// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUCoverageDissolve,
  GPUCoverageValidity,
  getGPUCoverageValidityParameterValues,
  GPU_COVERAGE_VALIDITY_PARAMETER_LENGTH
} from '../../../src/gpu-spatial-analysis/polygon-coverage-topology/index';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

it('getGPUCoverageValidityParameterValues packs and validates the gap width', () => {
  expect([...getGPUCoverageValidityParameterValues({gapWidth: 0.5})]).toEqual([0.5, 0, 0, 0]);
  expect(() => getGPUCoverageValidityParameterValues({gapWidth: -1})).toThrow(/gapWidth/);
  expect(() => getGPUCoverageValidityParameterValues({gapWidth: Number.NaN})).toThrow(/gapWidth/);
  expect(() => getGPUCoverageValidityParameterValues({gapWidth: 1}, new Float32Array(2))).toThrow(
    /target/
  );
  expect(GPU_COVERAGE_VALIDITY_PARAMETER_LENGTH).toBe(4);
});

it('GPUCoverageValidity and GPUCoverageDissolve validate props and declare nodes', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'coverage-topology-nodes'});
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `view-${serial++}`, format, length);
  const geometry = () => ({
    positions: view('float32x2', 12),
    ringOffsets: view('uint32', 4),
    polygonOffsets: view('uint32', 4)
  });
  const validity = {
    ...geometry(),
    parameters: view('float32', 4),
    output: {segmentFlags: view('uint32', 12)}
  };
  expect(() => new GPUCoverageValidity({...validity, parameters: view('float32', 2)})).toThrow(
    /parameters/
  );
  expect(
    () => new GPUCoverageValidity({...validity, output: {segmentFlags: view('uint32', 11)}})
  ).toThrow(/segmentFlags length/);
  expect(() => new GPUCoverageValidity({...validity, leafCapacity: 8})).toThrow(/leafCapacity/);
  expect(
    () =>
      new GPUCoverageValidity({
        ...validity,
        output: {...validity.output, polygonInvalidCounts: view('uint32', 5)}
      })
  ).toThrow(/polygonInvalidCounts/);
  const ids = new GPUCoverageValidity({
    ...validity,
    output: {
      ...validity.output,
      polygonInvalidCounts: view('uint32', 3),
      invalidSegmentCount: view('uint32', 1),
      isValid: view('uint32', 1)
    }
  })
    .getCommandNodes(graph)
    .map(node => node.id);
  expect(ids[0]).toBe('coverage-validity-ring-orientation');
  expect(ids).toContain('coverage-validity-probe');
  expect(ids.at(-1)).toBe('coverage-validity-finalize');

  const dissolve = {
    ...geometry(),
    labels: view('uint32', 3),
    output: {
      ringOffsets: view('uint32', 5),
      positions: view('float32x2', 16),
      count: view('uint32', 1),
      overflow: view('uint32', 1)
    }
  };
  expect(() => new GPUCoverageDissolve({...dissolve, labels: view('uint32', 4)})).toThrow(
    /labels length/
  );
  expect(() => new GPUCoverageDissolve({...dissolve, snapTolerance: -1})).toThrow(/snapTolerance/);
  expect(() => new GPUCoverageDissolve({...dissolve, vertexTolerance: 0})).toThrow(
    /vertexTolerance/
  );
  expect(
    () =>
      new GPUCoverageDissolve({
        ...dissolve,
        output: {...dissolve.output, boundarySegmentCount: view('uint32', 0)}
      })
  ).toThrow(/boundarySegmentCount/);
  const dissolveIds = new GPUCoverageDissolve(dissolve).getCommandNodes(graph).map(node => node.id);
  expect(dissolveIds[0]).toBe('coverage-dissolve-ring-orientation');
  expect(dissolveIds).toContain('coverage-dissolve-edge-keep');
  expect(dissolveIds).toContain('coverage-dissolve-compact');
  expect(dissolveIds.some(id => id.startsWith('coverage-dissolve-assembly'))).toBe(true);
});
