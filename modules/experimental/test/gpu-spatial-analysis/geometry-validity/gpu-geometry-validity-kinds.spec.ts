// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  GPU_GEOMETRY_VALIDITY_BIT,
  GPU_GEOMETRY_VALIDITY_STRUCTURAL_MASK,
  GPUGeometryValidity
} from '../../../src/gpu-spatial-analysis/geometry-validity/index';
import {createOutputBuffer, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {
  createPredicateGeometry,
  type PredicateGeometrySpec
} from '../geometry-predicates/gpu-geometry-predicates-harness';
import {SHAPELY_PREDICATES_FIXTURE as FIXTURE} from '../geometry-predicates/shapely-predicates-fixture';

async function runKindValidity(
  device: Device,
  spec: PredicateGeometrySpec & ({kind: 'lines'} | {kind: 'points'}),
  featureCount: number
): Promise<number[]> {
  const graph = new GPUCommandGraph(device, {id: 'geometry-validity-kinds'});
  const buffers: Buffer[] = [];
  const geometry = createPredicateGeometry(device, graph, 'geometry', spec, buffers);
  const maskBuffer = createOutputBuffer(device, featureCount);
  buffers.push(maskBuffer);
  const mask = importGraphBuffer(graph, 'mask', maskBuffer, 'uint32', featureCount);
  graph.add(
    new GPUGeometryValidity(
      geometry.kind === 'lines' ? {lines: geometry, mask} : {points: geometry as never, mask}
    )
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const result = await readUint32(maskBuffer, featureCount);
  compiled.destroy();
  for (const buffer of buffers) {
    buffer.destroy();
  }
  return result;
}

it('GPUGeometryValidity lines match Shapely is_valid and is_valid_reason', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const {lines} = FIXTURE;
  // Two hand-pinned rows Shapely cannot build from coordinates: an empty line (valid) and a
  // one-vertex line (too few points).
  const vertices = [...lines.vertices, [], [[1, 1]]];
  const mask = await runKindValidity(device, {kind: 'lines', vertices}, vertices.length);
  const BIT = GPU_GEOMETRY_VALIDITY_BIT;
  lines.names.forEach((name, row) => {
    const structural = mask[row] & GPU_GEOMETRY_VALIDITY_STRUCTURAL_MASK;
    expect(structural === 0, `${name} valid`).toBe(lines.expected.isValid[row] === 1);
  });
  const named = (name: string) => mask[lines.names.indexOf(name)];
  expect(named('two_same')).toBe(BIT.tooFewPoints);
  expect(named('three_same')).toBe(BIT.tooFewPoints);
  expect(named('closed_bowtie')).toBe(0);
  expect(named('cross')).toBe(0);
  expect(mask[lines.names.length]).toBe(0);
  expect(mask[lines.names.length + 1]).toBe(BIT.tooFewPoints);
  expect(lines.expected.isValid.filter(valid => valid === 0).length).toBeGreaterThan(1);
});

it('GPUGeometryValidity points match Shapely is_valid', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const {points} = FIXTURE;
  const mask = await runKindValidity(
    device,
    {kind: 'points', positions: points.positions},
    points.names.length
  );
  points.names.forEach((name, row) => {
    expect(mask[row] === 0, `${name} valid`).toBe(points.isValid[row] === 1);
    if (!points.isValid[row]) {
      expect(mask[row]).toBe(GPU_GEOMETRY_VALIDITY_BIT.nonFinite);
    }
  });
  expect(points.isValid.filter(valid => valid === 0)).toHaveLength(2);
});
