// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandNode} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {getPolygonEdgeSlabNodes} from '../../src/gpu-spatial-analysis/spatial-join/polygon-edge-slabs';
import {CELLS_AND_JOINS_CASES} from '../analysis-core/kernel-audit/cases-cells-and-joins';
import {GEOMETRY_CASES} from '../analysis-core/kernel-audit/cases-geometry';
import {NETWORK_CASES} from '../analysis-core/kernel-audit/cases-network';
import {RASTER_CASES} from '../analysis-core/kernel-audit/cases-raster';
import {STATISTICS_CASES} from '../analysis-core/kernel-audit/cases-statistics';
import {
  createKernelAuditContext,
  getDeclaredNodes
} from '../analysis-core/kernel-audit/kernel-audit-context';
import {getParsedKernelSources, getStorageBindingCount} from './gpu-contributor-test-utils';

const CASES = [
  ...CELLS_AND_JOINS_CASES,
  ...GEOMETRY_CASES,
  ...NETWORK_CASES,
  ...RASTER_CASES,
  ...STATISTICS_CASES
];

const SCAN_SITES = [
  {
    exportName: 'GPUNetworkCostWeights',
    cases: ['rows, every column, every output', 'one row, outputs without diagnostics'],
    prefix: 'network-cost-weights',
    consumer: 'network-cost-weights-finalize',
    totalRead: 'let total = scanTotal[scanTotalOffset];'
  },
  {
    exportName: 'GPUNetworkLineGraph',
    cases: ['planar, banned turns, every output', 'no bans, no arc count'],
    prefix: 'network-line-graph',
    consumer: 'network-line-graph-publish',
    totalRead: 'let total = scanTotal[scanTotalOffset];'
  },
  {
    exportName: 'GPUPolygonRasterization',
    cases: ['every output, single sort', 'every output, two sorts'],
    prefix: 'polygon-rasterization',
    consumer: 'polygon-rasterization-publish',
    totalRead: 'let crossings = total[totalOffset];'
  },
  {
    exportName: 'GPUCellCompaction',
    cases: ['quadbin uncompact depth 15 every option', 'h3 uncompact depth 11 every option'],
    prefix: 'cell-compaction',
    consumer: 'cell-compaction-expand',
    totalRead: 'let totalCells = scanTotal[scanTotalOffset];'
  },
  {
    exportName: 'GPULineDensity',
    cases: ['planar, every output', 'spherical, every output'],
    prefix: 'line-density',
    consumer: 'line-density-publish',
    totalRead: 'let total = scanTotal[scanTotalOffset];'
  },
  {
    exportName: 'GPUContiguityWeights',
    cases: ['queen every output large pair capacity', 'rook every output large pair capacity'],
    prefix: 'contiguity-weights-pair',
    consumer: 'contiguity-weights-totals',
    totalRead: 'let pairTotal = scanTotal[scanTotalOffset];'
  },
  {
    exportName: 'GPULatticeWeights',
    cases: ['queen radius 1 mask and cell size', 'rook radius 1 mask and cell size'],
    prefix: 'lattice-weights',
    consumer: 'lattice-weights-offsets',
    totalRead: 'let total = scanTotal[scanTotalOffset];'
  },
  {
    exportName: 'GPUSpatialWeightsAlgebra',
    cases: ['block with many groups', 'union left rule and distances'],
    prefix: 'spatial-weights-algebra',
    consumer: 'spatial-weights-algebra-offsets',
    totalRead: 'let total = scanTotal[scanTotalOffset];'
  },
  {
    exportName: 'GPUSharedPaths',
    cases: ['every option'],
    prefix: 'shared-paths-vertex',
    consumer: 'shared-paths-finalize',
    totalRead: 'let totalVertices = vertexTotal[vertexTotalOffset];'
  }
];

for (const site of SCAN_SITES) {
  for (const name of site.cases) {
    it(`${site.exportName} saturates its scan total before publishing (${name})`, () => {
      const auditCase = CASES.find(
        candidate => candidate.exportName === site.exportName && candidate.name === name
      );
      expect(auditCase, name).toBeDefined();
      const context = createKernelAuditContext('scan-total-contributors');
      try {
        auditCase!.build(context);
        const nodes = getDeclaredNodes(context) as readonly GPUCommandNode[];
        const ids = nodes.map(node => node.id);
        const totalIndex = ids.indexOf(`${site.prefix}-total`);
        const carryIndex = ids.indexOf(`${site.prefix}-carry`);
        expect(totalIndex).toBeGreaterThanOrEqual(0);
        expect(carryIndex).toBe(totalIndex + 1);
        expect(ids.indexOf(site.consumer)).toBeGreaterThan(carryIndex);
        // The carry must update the word initialized by the total pass and read by the status pass.
        const carryOutput = nodes[carryIndex].resources?.find(
          resource => resource.usage === 'storage-read-write'
        );
        expect(carryOutput).toBeDefined();
        expect(nodes[totalIndex].resources).toContainEqual(carryOutput);
        expect(nodes[ids.indexOf(site.consumer)].resources).toContainEqual({
          ...carryOutput,
          usage: 'storage-read'
        });
        const sources = getParsedKernelSources(nodes);
        expect(sources.get(`${site.prefix}-carry`)).toContain(
          'atomicStore(&total[totalOffset + TOTAL_ROW], 4294967295u);'
        );
        expect(sources.get(site.consumer)).toContain(site.totalRead);
        for (const source of sources.values()) {
          expect(getStorageBindingCount(source)).toBeLessThanOrEqual(8);
        }
      } finally {
        context.device.destroy();
      }
    });
  }
}

it('higher-order weights protect every intermediate CSR stage before offsets and overflow', () => {
  const context = createKernelAuditContext('scan-total-higher-order');
  try {
    const auditCase = CASES.find(
      candidate =>
        candidate.exportName === 'GPUSpatialWeightsAlgebra' &&
        candidate.name === 'higherOrder order 2 cumulative'
    );
    expect(auditCase).toBeDefined();
    auditCase!.build(context);
    const nodes = getDeclaredNodes(context) as readonly GPUCommandNode[];
    const ids = nodes.map(node => node.id);
    const sources = getParsedKernelSources(nodes);
    const offsetsIds = ids.filter(id =>
      /^spatial-weights-algebra-(exact|visited)-[0-9]+-offsets$/.test(id)
    );
    expect(offsetsIds).toHaveLength(3);
    for (const offsetsId of offsetsIds) {
      const prefix = offsetsId.slice(0, -'-offsets'.length);
      const carryIndex = ids.indexOf(`${prefix}-carry`);
      expect(carryIndex).toBe(ids.indexOf(`${prefix}-total`) + 1);
      expect(ids.indexOf(offsetsId)).toBe(carryIndex + 1);
      expect(sources.get(offsetsId)).toContain('let total = scanTotal[scanTotalOffset];');
    }
  } finally {
    context.device.destroy();
  }
});

it('polygon edge slabs detect carry before clearing counts and disable wrapped indexes', () => {
  const context = createKernelAuditContext('scan-total-slabs');
  try {
    const {nodes} = getPolygonEdgeSlabNodes(context.graph, {
      id: 'slabs',
      operation: 'ScanTotalTest',
      owner: 'polygon',
      positions: context.view('float32x2', 64),
      polygonOffsets: context.view('uint32', 2),
      ringOffsets: context.view('uint32', 2)
    });
    const ids = nodes.map(node => node.id);
    const totalIndex = ids.indexOf('slabs-slab-entry-total');
    const carryIndex = ids.indexOf('slabs-slab-entry-carry');
    expect(totalIndex).toBeGreaterThanOrEqual(0);
    expect(carryIndex).toBe(totalIndex + 1);
    expect(ids[carryIndex + 1]).toBe('slabs-slab-entry-fallback');
    expect(ids.indexOf('slabs-slab-edges-fill')).toBeGreaterThan(carryIndex + 1);
    const sources = getParsedKernelSources(nodes);
    expect(sources.get('slabs-slab-entry-fallback')).toContain(
      'if (scanTotal[scanTotalOffset] == 0xffffffffu)'
    );
    expect(sources.get('slabs-slab-entry-fallback')).toContain(
      'edgeSlabParams[edgeSlabParamsOffset + index * 6u] = 0u;'
    );
    for (const source of sources.values()) {
      expect(getStorageBindingCount(source)).toBeLessThanOrEqual(8);
    }
  } finally {
    context.device.destroy();
  }
});
