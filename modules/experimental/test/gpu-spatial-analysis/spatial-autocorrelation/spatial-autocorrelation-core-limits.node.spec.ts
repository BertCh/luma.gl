// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {describe, expect, it} from 'vitest';
import {
  GPUAssuncaoReisStandardization,
  GPUGammaStatistic,
  GPUHotSpotAnalysis,
  GPULocalGeary,
  GPULocalMoran,
  GPUSpatialPearson,
  GPU_RATE_STANDARDIZATION_STATISTICS_LENGTH,
  GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH,
  GPU_SPATIAL_AUTOCORRELATION_STATISTICS_LENGTH,
  type GPUGammaOperation
} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  CORE_PRIVATE_MEMORY_LIMIT_BYTES,
  CORE_WORKGROUP_STORAGE_LIMIT_BYTES,
  getPrivateMemoryBytes,
  getStorageBindingCount,
  getWorkgroupMemoryBytes
} from '../../utils/gpu-contributor-test-utils';

const SLOTS = 8;
const METHODS = [
  'none',
  'bonferroni',
  'holm',
  'benjamini-hochberg',
  'benjamini-yekutieli'
] as const;
let serial = 0;

type Resource = {buffer: {byteLength?: number; length?: number; rowByteLength?: number}};

/** Bytes of every buffer or view bound by the nodes. */
function getBoundBytes(nodes: readonly unknown[]): {id: string; bytes: number}[] {
  const result: {id: string; bytes: number}[] = [];
  for (const node of nodes as {
    id: string;
    resources?: Resource[];
    workload?: {resources?: Resource[]};
  }[]) {
    for (const resource of node.resources ?? node.workload?.resources ?? []) {
      const buffer = resource.buffer;
      const bytes =
        buffer.byteLength ??
        (buffer.length ?? 0) * (buffer.rowByteLength ?? Uint32Array.BYTES_PER_ELEMENT);
      result.push({id: node.id, bytes});
    }
  }
  return result;
}

function expectCoreLimits(nodes: readonly unknown[], device = createNullWebGPUDevice()): void {
  const sources = getGPUCommandNodeKernelSources(nodes);
  expect(sources.length).toBeGreaterThan(0);
  for (const {id, source} of sources) {
    expect(getStorageBindingCount(source), `${id} storage bindings`).toBeLessThanOrEqual(8);
    expect(getPrivateMemoryBytes(source), `${id} private bytes`).toBeLessThanOrEqual(
      CORE_PRIVATE_MEMORY_LIMIT_BYTES
    );
    expect(getWorkgroupMemoryBytes(source), `${id} workgroup bytes`).toBeLessThanOrEqual(
      CORE_WORKGROUP_STORAGE_LIMIT_BYTES
    );
  }
  const bound = getBoundBytes(nodes);
  expect(bound.length).toBeGreaterThan(0);
  for (const {id, bytes} of bound) {
    expect(bytes, `${id} bound bytes`).toBeLessThanOrEqual(
      device.limits.maxStorageBufferBindingSize
    );
  }
}

function createContext(rows: number) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device, {id: `autocorrelation-limits-${serial++}`});
  const view = <Format extends 'uint32' | 'float32' | 'sint32'>(format: Format, length: number) =>
    createTransientView(graph, `autocorrelation-limits-${serial++}`, format, length);
  const weights = {
    offsets: view('uint32', rows + 1),
    neighbors: view('uint32', rows * SLOTS),
    weights: view('float32', rows * SLOTS)
  };
  return {device, graph, view, weights};
}

const ROW_COUNTS = [1000, 2_000_000];

describe('spatial autocorrelation at core limits', () => {
  for (const rows of ROW_COUNTS) {
    for (const multipleTesting of METHODS) {
      // With every output wired and multipleTesting 'none' the fused neighbors kernel used to
      // bind 9 storage buffers; the classify step now runs as its own pass in that case.
      it(`GPUHotSpotAnalysis with ${multipleTesting}, ${rows} rows and every output`, () => {
        const {device, graph, view, weights} = createContext(rows);
        const nodes = new GPUHotSpotAnalysis({
          weights,
          selfWeight: 1,
          values: view('float32', rows),
          parameters: view('float32', GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH),
          mask: view('uint32', rows),
          islands: 'keep',
          minimumNeighbors: 8,
          multipleTesting,
          output: {
            zScores: view('float32', rows),
            bins: view('sint32', rows),
            pValues: view('float32', rows),
            adjustedPValues: view('float32', rows),
            neighborCounts: view('uint32', rows),
            globalStatistics: view('float32', GPU_SPATIAL_AUTOCORRELATION_STATISTICS_LENGTH)
          }
        }).getCommandNodes(graph);
        expectCoreLimits(nodes, device);
      });

      // quadrantGating 'none' cannot be combined with multipleTesting.
      for (const quadrantGating of multipleTesting === 'none'
        ? (['analytic', 'none'] as const)
        : (['analytic'] as const)) {
        it(`GPULocalMoran with ${multipleTesting}, ${quadrantGating} gating, ${rows} rows`, () => {
          const {device, graph, view, weights} = createContext(rows);
          const nodes = new GPULocalMoran({
            weights,
            values: view('float32', rows),
            parameters: view('float32', GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH),
            mask: view('uint32', rows),
            islands: 'keep',
            minimumNeighbors: 8,
            quadrantGating,
            multipleTesting,
            output: {
              zScores: view('float32', rows),
              localI: view('float32', rows),
              spatialLag: view('float32', rows),
              quadrants: view('uint32', rows),
              pValues: view('float32', rows),
              adjustedPValues: view('float32', rows),
              neighborCounts: view('uint32', rows),
              globalStatistics: view('float32', GPU_SPATIAL_AUTOCORRELATION_STATISTICS_LENGTH)
            }
          }).getCommandNodes(graph);
          expectCoreLimits(nodes, device);
        });
      }
    }

    it(`GPULocalGeary with every output, ${rows} rows`, () => {
      const {device, graph, view, weights} = createContext(rows);
      const nodes = new GPULocalGeary({
        weights,
        values: view('float32', rows),
        parameters: view('float32', GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH),
        mask: view('uint32', rows),
        islands: 'keep',
        output: {
          statistic: view('float32', rows),
          globalStatistics: view('float32', GPU_SPATIAL_AUTOCORRELATION_STATISTICS_LENGTH)
        }
      }).getCommandNodes(graph);
      expectCoreLimits(nodes, device);
    });

    it(`GPUSpatialPearson with every output, ${rows} rows`, () => {
      const {device, graph, view, weights} = createContext(rows);
      const nodes = new GPUSpatialPearson({
        weights,
        values: view('float32', rows),
        secondValues: view('float32', rows),
        mask: view('uint32', rows),
        output: {statistic: view('float32', 4)}
      }).getCommandNodes(graph);
      expectCoreLimits(nodes, device);
    });

    for (const operation of [
      'cross-product',
      'squared-difference',
      'absolute-difference'
    ] as GPUGammaOperation[]) {
      it(`GPUGammaStatistic ${operation} with every output, ${rows} rows`, () => {
        const {device, graph, view, weights} = createContext(rows);
        const nodes = new GPUGammaStatistic({
          weights,
          operation,
          values: view('float32', rows),
          secondValues: view('float32', rows),
          mask: view('uint32', rows),
          output: {statistic: view('float32', 4)}
        }).getCommandNodes(graph);
        expectCoreLimits(nodes, device);
      });
    }

    it(`GPUAssuncaoReisStandardization with every output, ${rows} rows`, () => {
      const {device, graph, view} = createContext(rows);
      const nodes = new GPUAssuncaoReisStandardization({
        events: view('float32', rows),
        populations: view('float32', rows),
        mask: view('uint32', rows),
        output: {
          standardizedRates: view('float32', rows),
          statistics: view('float32', GPU_RATE_STANDARDIZATION_STATISTICS_LENGTH),
          status: {invalidCount: view('uint32', 1)}
        }
      }).getCommandNodes(graph);
      expectCoreLimits(nodes, device);
    });
  }
});
