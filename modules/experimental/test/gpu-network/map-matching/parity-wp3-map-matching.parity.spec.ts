// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  GPUMapMatching,
  GPU_MAP_MATCHING_NONE,
  getGPUMapMatchingParameterValues
} from '../../../src/gpu-network/map-matching';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {
  MATCH_FIXTURES,
  EXACT,
  compareValues,
  type MatchFixture,
  type MatchAlternative
} from './parity-wp3-map-matching-fixture';

async function runMatching(device: Device, fixture: MatchFixture) {
  const graph = new GPUCommandGraph(device, {id: `parity-wp3-${fixture.case}`});
  const buffers: Buffer[] = [];
  const outputs: Record<string, Buffer> = {};
  let compiled: ReturnType<GPUCommandGraph['compile']> | undefined;
  const pointCount = fixture.points.length / 2;
  const trackCount = fixture.trackOffsets.length - 1;
  const input = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    name: string,
    values: Float32Array | Uint32Array,
    format: Format
  ) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return importGraphBuffer(
      graph,
      name,
      buffer,
      format,
      values.length / (format === 'float32x2' ? 2 : 1)
    );
  };
  const output = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    name: string,
    length: number,
    format: Format
  ) => {
    const buffer = createOutputBuffer(device, length * (format === 'float32x2' ? 2 : 1));
    buffers.push(buffer);
    outputs[name] = buffer;
    return importGraphBuffer(graph, name, buffer, format, length);
  };
  try {
    graph.add(
      new GPUMapMatching({
        positions: input('points', Float32Array.from(fixture.points), 'float32x2'),
        trackOffsets: input('tracks', Uint32Array.from(fixture.trackOffsets), 'uint32'),
        topology: {
          nodePositions: input('nodes', Float32Array.from(fixture.positions), 'float32x2'),
          offsets: input('adjacency', Uint32Array.from(fixture.offsets), 'uint32'),
          neighbors: input('neighbors', Uint32Array.from(fixture.neighbors), 'uint32')
        },
        parameters: input(
          'parameters',
          getGPUMapMatchingParameterValues(fixture.settings),
          'float32'
        ),
        candidateCount: 8,
        routeNodeBudget: 128,
        cellSize: fixture.cellSize,
        bounds: fixture.bounds,
        entryCapacity: fixture.entryCapacity,
        output: {
          matchedEdges: output('edges', pointCount, 'uint32'),
          matchedFractions: output('fractions', pointCount, 'float32'),
          matchedOffsets: output('offsets', pointCount, 'float32'),
          snapDistances: output('distances', pointCount, 'float32'),
          snappedPositions: output('positions', pointCount, 'float32x2'),
          breaks: output('breaks', pointCount, 'uint32'),
          trackLogLikelihoods: output('likelihoods', trackCount, 'float32'),
          matchedCount: output('matchedCount', 1, 'uint32'),
          breakCount: output('breakCount', 1, 'uint32'),
          status: {
            overflow: output('overflow', 1, 'uint32'),
            routeOverflow: output('routeOverflow', 1, 'uint32')
          }
        }
      })
    );
    compiled = graph.compile();
    submitGraph(device, compiled, undefined);
    return {
      edges: await readUint32(outputs['edges'], pointCount),
      fractions: await readFloat32(outputs['fractions'], pointCount),
      offsets: await readFloat32(outputs['offsets'], pointCount),
      distances: await readFloat32(outputs['distances'], pointCount),
      positions: await readFloat32(outputs['positions'], pointCount * 2),
      breaks: await readUint32(outputs['breaks'], pointCount),
      likelihoods: await readFloat32(outputs['likelihoods'], trackCount),
      status: [
        ...(await readUint32(outputs['matchedCount'], 1)),
        ...(await readUint32(outputs['breakCount'], 1)),
        ...(await readUint32(outputs['overflow'], 1)),
        ...(await readUint32(outputs['routeOverflow'], 1))
      ]
    };
  } finally {
    compiled?.destroy();
    for (const buffer of buffers) {
      buffer.destroy();
    }
  }
}

for (const fixture of MATCH_FIXTURES) {
  it(`GPUMapMatching independent HMM parity: ${fixture.case}`, async context => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device, context);
    const result = await runMatching(device, fixture);
    const selectedAlternatives: MatchAlternative[] = [];
    const membership: number[] = [];
    for (let track = 0; track < fixture.expected.length; track++) {
      const first = fixture.trackOffsets[track];
      const last = fixture.trackOffsets[track + 1];
      const edges = result.edges.slice(first, last);
      // Whole-path membership prevents combining incompatible per-observation ties.
      const alternative = fixture.expected[track].alternatives.find(reference =>
        reference.edges.every((edge, index) => edge === edges[index])
      );
      membership.push(Number(Boolean(alternative)));
      selectedAlternatives.push(alternative ?? fixture.expected[track].alternatives[0]);
    }
    compareValues(
      fixture,
      'whole matched edge path in exact optimum tie set',
      membership,
      membership.map(() => 1),
      EXACT
    );
    compareValues(
      fixture,
      'matchedEdges',
      result.edges,
      selectedAlternatives.flatMap(reference => reference.edges),
      EXACT
    );
    compareValues(
      fixture,
      'matchedFractions',
      result.fractions,
      selectedAlternatives.flatMap(reference => reference.fractions),
      fixture.tolerances.fraction
    );
    compareValues(
      fixture,
      'matchedOffsets',
      result.offsets,
      selectedAlternatives.flatMap(reference => reference.offsets),
      fixture.tolerances.distance
    );
    compareValues(
      fixture,
      'snapDistances',
      result.distances,
      selectedAlternatives.flatMap(reference => reference.distances),
      fixture.tolerances.distance
    );
    compareValues(
      fixture,
      'snappedPositions',
      result.positions,
      selectedAlternatives.flatMap(reference => reference.positions),
      fixture.tolerances.distance
    );
    compareValues(
      fixture,
      'trackLogLikelihoods',
      result.likelihoods,
      fixture.expected.map(reference => reference.logLikelihood),
      fixture.tolerances.likelihood
    );
    compareValues(
      fixture,
      'breaks',
      result.breaks,
      result.breaks.map(() => 0),
      EXACT
    );
    const matchedCount = selectedAlternatives
      .flatMap(reference => reference.edges)
      .filter(edge => edge !== GPU_MAP_MATCHING_NONE).length;
    compareValues(
      fixture,
      'matchedCount/breakCount/overflow/routeOverflow',
      result.status,
      [matchedCount, 0, 0, 0],
      EXACT
    );
  });
}
