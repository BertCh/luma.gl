// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {GPUNetworkNoding} from '../../../src/gpu-network/network-noding';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {compareParity, EXACT} from '../parity-wp3-comparison';
import {createParityResources} from '../parity-wp3-round2-resources';
import {canonicalizeNoding, nodingFixtures} from './parity-wp3-noding-fixture';

for (const fixture of nodingFixtures) {
  for (const spatialSort of [false, true]) {
    it(`GPUNetworkNoding Shapely ${fixture.case}/sort=${spatialSort}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const resources = createParityResources(device);
      const allocations: Record<string, {buffer: Buffer; length: number}> = {};
      const output = <Format extends 'uint32' | 'float32' | 'float32x2'>(
        name: string,
        format: Format,
        length: number
      ) => {
        const scalarLength = length * (format === 'float32x2' ? 2 : 1);
        const buffer = createOutputBuffer(device, scalarLength);
        allocations[name] = {buffer, length: scalarLength};
        return importGraphBuffer(resources.graph, name, buffer, format, length);
      };
      const readUint = (name: string) =>
        readUint32(allocations[name].buffer, allocations[name].length);
      const readFloat = (name: string) =>
        readFloat32(allocations[name].buffer, allocations[name].length);
      const pieceCount = fixture.sourceIds.length;
      const nodeCount = fixture.nodePositions.length / 2;
      const expectedTopology = canonicalizeNoding(fixture);
      const nodeCapacity = nodeCount + 3;
      const pieceCapacity = pieceCount + 3;
      const vertexCount = fixture.piecePositions.length / 2;
      resources.graph.add(
        new GPUNetworkNoding({
          lines: {
            kind: 'lines',
            positions: resources.input('positions', 'float32x2', fixture.positions),
            lineOffsets: resources.input('line-offsets', 'uint32', fixture.lineOffsets)
          },
          tolerance: resources.input('tolerance', 'float32', [fixture.tolerance]),
          intersectionCapacity: fixture.intersectionCapacity,
          spatialSort,
          output: {
            pieces: {
              geometry: {
                kind: 'lines',
                positions: output('piece-positions', 'float32x2', vertexCount + 8),
                lineOffsets: output('piece-offsets', 'uint32', pieceCapacity + 1)
              },
              sourceIds: output('source-ids', 'uint32', pieceCapacity),
              status: {
                count: output('piece-count', 'uint32', 1),
                requiredCount: output('piece-required', 'uint32', 1),
                overflow: output('piece-overflow', 'uint32', 1),
                vertexCount: output('vertex-count', 'uint32', 1),
                requiredVertexCount: output('vertex-required', 'uint32', 1)
              }
            },
            nodes: {
              positions: output('node-positions', 'float32x2', nodeCapacity),
              status: {
                count: output('node-count', 'uint32', 1),
                requiredCount: output('node-required', 'uint32', 1)
              }
            },
            edges: {
              fromNodes: output('from', 'uint32', pieceCapacity),
              toNodes: output('to', 'uint32', pieceCapacity),
              lengths: output('lengths', 'float32', pieceCapacity)
            },
            topology: {
              offsets: output('offsets', 'uint32', nodeCapacity + 1),
              neighbors: output('neighbors', 'uint32', 2 * pieceCapacity),
              weights: output('weights', 'float32', 2 * pieceCapacity),
              edgeIds: output('edge-ids', 'uint32', 2 * pieceCapacity)
            },
            status: {
              overflow: output('overflow', 'uint32', 1),
              uncertainCount: output('uncertain', 'uint32', 1)
            }
          }
        })
      );
      const compiled = resources.graph.compile();
      try {
        for (let encoding = 0; encoding < 2; encoding++) {
          submitGraph(device, compiled, undefined);
          const comparison = {
            ...fixture,
            case: `${fixture.case}/sort=${spatialSort}/encoding=${encoding}`
          };
          const compare = (metric: string, actual: number[], expected: number[], absolute = 0) =>
            compareParity('GPUNetworkNoding', comparison, metric, actual, expected, {
              absolute,
              relative: 0
            });
          compare(
            'all counts, overflow and uncertainty',
            (
              await Promise.all(
                [
                  'piece-count',
                  'piece-required',
                  'vertex-count',
                  'vertex-required',
                  'node-count',
                  'node-required',
                  'piece-overflow',
                  'overflow',
                  'uncertain'
                ].map(readUint)
              )
            ).flat(),
            [pieceCount, pieceCount, vertexCount, vertexCount, nodeCount, nodeCount, 0, 0, 0]
          );
          for (const [name, expected] of [
            ['source-ids', fixture.sourceIds],
            ['piece-offsets', fixture.pieceOffsets]
          ] as const)
            compare(name, (await readUint(name)).slice(0, expected.length), expected);
          const offsets = await readUint('offsets');
          compare(
            'CSR start, end and unused row offsets',
            [offsets[0], ...offsets.slice(nodeCount)],
            [0, ...new Array(4).fill(2 * pieceCount)]
          );
          // Canonicalize GPU and reference independently. This changes only node names
          // and CSR row placement; every edge, neighbor and duplicate stays observable.
          const actualTopology = canonicalizeNoding({
            nodePositions: (await readFloat('node-positions')).slice(0, 2 * nodeCount),
            fromNodes: (await readUint('from')).slice(0, pieceCount),
            toNodes: (await readUint('to')).slice(0, pieceCount),
            offsets: offsets.slice(0, nodeCount + 1),
            neighbors: (await readUint('neighbors')).slice(0, 2 * pieceCount),
            edgeIds: (await readUint('edge-ids')).slice(0, 2 * pieceCount),
            weights: (await readFloat('weights')).slice(0, 2 * pieceCount)
          });
          for (const field of [
            'fromNodes',
            'toNodes',
            'offsets',
            'neighbors',
            'edgeIds'
          ] as const) {
            compare(`canonical ${field}`, actualTopology[field], expectedTopology[field]);
          }
          compare(
            'every piece vertex',
            (await readFloat('piece-positions')).slice(0, fixture.piecePositions.length),
            fixture.piecePositions,
            fixture.coordinateBound
          );
          compare(
            'every canonical node position',
            actualTopology.nodePositions,
            expectedTopology.nodePositions,
            fixture.coordinateBound
          );
          const lengths = (await readFloat('lengths')).slice(0, pieceCount);
          // Reverse triangle inequality: each segment changes by <=2*sqrt(2)*G.
          // The fixture adds gamma_(3*k)*length for f32 norm/reduction rounding.
          // Per-piece budgets are checked separately, rather than widened globally.
          compare('all piece lengths', lengths, fixture.lengths, Math.max(...fixture.lengthBounds));
          compare(
            'every piece length / derived error budget',
            lengths.map(
              (value, index) =>
                Math.abs(value - fixture.lengths[index]) / fixture.lengthBounds[index]
            ),
            new Array(pieceCount).fill(0),
            1
          );
          const weights = actualTopology.weights;
          compare(
            'every CSR weight',
            weights,
            expectedTopology.weights,
            Math.max(...fixture.lengthBounds)
          );
          compare(
            'every CSR weight / derived error budget',
            weights.map(
              (value, index) =>
                Math.abs(value - expectedTopology.weights[index]) /
                fixture.lengthBounds[expectedTopology.edgeIds[index]]
            ),
            new Array(weights.length).fill(0),
            1
          );
          const sourceTotals = fixture.sourceLengths.map(() => 0);
          const sourceBounds = fixture.sourceLengths.map(() => 0);
          for (const [index, sourceId] of fixture.sourceIds.entries()) {
            sourceTotals[sourceId] += lengths[index];
            sourceBounds[sourceId] += fixture.lengthBounds[index];
          }
          compare(
            'every source length conservation / summed budget',
            sourceTotals.map(
              (value, index) => Math.abs(value - fixture.sourceLengths[index]) / sourceBounds[index]
            ),
            new Array(sourceTotals.length).fill(0),
            1
          );
          compareParity(
            'GPUNetworkNoding',
            comparison,
            'unused edge provenance sentinels',
            (await readUint('source-ids')).slice(pieceCount),
            new Array(3).fill(0xffffffff),
            EXACT
          );
        }
      } finally {
        compiled.destroy();
        resources.destroy();
        for (const {buffer} of Object.values(allocations)) buffer.destroy();
      }
    }, 180000);
  }
}
