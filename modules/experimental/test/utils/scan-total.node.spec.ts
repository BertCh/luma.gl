// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {WgslReflect} from 'wgsl_reflect';
import {createSaturatedScanTotalNodes} from '../../src/utils/scan-total';
import {createNullWebGPUDevice} from './gpu-contributor-test-utils';

/** WGSL source and constants of every pipeline that `nodes` compile. */
function getPipelineKeys(nodes: readonly unknown[]): string[] {
  const device = createNullWebGPUDevice();
  Object.defineProperty(device, 'props', {
    value: {...device.props, _cacheShaders: false, _cachePipelines: false}
  });
  const keys: string[] = [];
  Object.defineProperty(device, 'createComputePipeline', {
    value: (props: {shader: {source: string}; constants?: object}) => {
      new WgslReflect(props.shader.source);
      keys.push(`${props.shader.source}\n${JSON.stringify(props.constants ?? null)}`);
      return {id: '', hash: '', destroy: () => {}};
    }
  });
  for (const node of nodes as {compile?: (context: {device: unknown}) => unknown}[]) {
    node.compile?.({device});
  }
  device.destroy();
  return keys;
}

function buildTotalNodes(rowCount: number) {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: `scan-total-${rowCount}`});
  return createSaturatedScanTotalNodes(graph, {
    id: 'pairs',
    operation: 'ScanTotalTest',
    counts: createTransientView(graph, 'counts', 'uint32', rowCount),
    offsets: createTransientView(graph, 'offsets', 'uint32', rowCount),
    rowCount,
    total: createTransientView(graph, 'state', 'uint32', 4),
    totalRow: 2
  });
}

it('saturated scan total: a total node then a carry node over every row', () => {
  const nodes = buildTotalNodes(1000);
  expect(nodes.map(node => node.id)).toEqual(['pairs-total', 'pairs-carry']);
  const sources = getPipelineKeys(nodes);
  expect(sources).toHaveLength(2);
  expect(sources[1]).toMatch(/var<storage, read_write> total: array<atomic<u32>>/);
  expect(sources[1]).toMatch(/offset \+ counts\[countsOffset \+ index\] < offset/);
});

it('saturated scan total shares its WGSL modules across row counts', () => {
  const sources = (rowCount: number) =>
    getPipelineKeys(buildTotalNodes(rowCount)).map(key => key.slice(0, key.lastIndexOf('\n')));
  expect(sources(1000)).toEqual(sources(1001));
});

it('GPUSegmentIntersection saturates its pair total before the write and finalize nodes', async () => {
  const {GEOMETRY_CASES} = await import('../analysis-core/kernel-audit/cases-geometry');
  const {createKernelAuditContext, getDeclaredNodes} = await import(
    '../analysis-core/kernel-audit/kernel-audit-context'
  );
  const auditCase = GEOMETRY_CASES.find(
    candidate =>
      candidate.exportName === 'GPUSegmentIntersection' &&
      candidate.name === 'lines x self, spatialSort false'
  );
  expect(auditCase).toBeDefined();
  const context = createKernelAuditContext('scan-total');
  auditCase?.build(context);
  const ids = (getDeclaredNodes(context) as {id: string}[]).map(node => node.id);
  context.device.destroy();
  const total = ids.indexOf('segment-intersection-total');
  const carry = ids.indexOf('segment-intersection-carry');
  expect(total).toBeGreaterThanOrEqual(0);
  expect(carry).toBe(total + 1);
  expect(ids.indexOf('segment-intersection-write')).toBeGreaterThan(carry);
});
