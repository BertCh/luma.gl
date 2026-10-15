// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {
  GPUNetworkSnapping,
  type GPUNetworkSnappingStartDirection
} from '../../../src/gpu-network/network-accessibility';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {createOutputBuffer, readFloat32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {compareParity, EXACT} from '../parity-wp3-comparison';
import {createParityResources} from '../parity-wp3-round2-resources';
import {snappingFixtures, type SnappingFixture} from './parity-wp3-snapping-fixture';

const NONE = 0xffffffff;
const directions: GPUNetworkSnappingStartDirection[] = ['both', 'forward', 'reverse'];

/** Emit raw errors while applying the fixture's derived per-element G bounds. */
function compareGeometry(
  fixture: SnappingFixture,
  metric: string,
  actual: number[],
  expected: number[],
  bounds: number[],
  sourceIndices?: number[]
): void {
  const failures: {index: number; actual: number; expected: number; bound: number}[] = [];
  let maxAbs = 0;
  let maxRel = 0;
  let mismatches = Math.abs(actual.length - expected['length']);
  expected['forEach']((value, index) => {
    const absolute = Math.abs(actual[index] - value);
    const relative = absolute / Math.max(Math.abs(value), Number.MIN_VALUE);
    maxAbs = Math.max(maxAbs, absolute);
    maxRel = Math.max(maxRel, relative);
    const failed = !Number.isFinite(actual[index]) || absolute > bounds[index];
    mismatches += Number(failed);
    if (failed)
      failures.push({
        index: sourceIndices?.[index] ?? index,
        actual: actual[index],
        expected: value,
        bound: bounds[index]
      });
  });
  console.log(
    'PARITY',
    JSON.stringify({
      class: 'GPUNetworkSnapping',
      case: fixture.case,
      reference: fixture.reference,
      n: expected['length'],
      metric,
      maxAbs: Number.isFinite(maxAbs) ? maxAbs : 'Infinity',
      maxRel: Number.isFinite(maxRel) ? maxRel : 'Infinity',
      mismatches,
      tolerance: {
        profile: 'G',
        relative: 0,
        absolute: 'per-element derived bound recorded by generator',
        maximumAbsolute: Math.max(...bounds),
        coordinateBound: fixture.coordinateBound,
        failures
      }
    })
  );
  expect.soft(mismatches, `${fixture.case}: ${metric}`).toBe(0);
}

for (const fixture of snappingFixtures) {
  for (const direction of directions) {
    for (const weighted of fixture.case === 'chicago' ? [true] : [false, true]) {
      for (const indexed of fixture.case === 'chicago' ? [true] : [false, true]) {
        const caseName = `${fixture.case}/${direction}/${weighted ? 'travel-cost' : 'length-cost'}/${indexed ? 'bvh' : 'scan'}`;
        it(`GPUNetworkSnapping Shapely ${caseName}`, async () => {
          const device = await getWebGPUTestDevice();
          requireTestDevice(device);
          const resources = createParityResources(device);
          const pointCount = fixture.queries.length;
          const positionBuffer = createOutputBuffer(device, pointCount * 2);
          const pointValues = fixture.queries.flatMap(query => query.point);
          const pointBuffer = device.createBuffer({
            data: Float32Array.from(pointValues),
            usage: Buffer.STORAGE | Buffer.COPY_SRC
          });
          resources.graph.add(
            new GPUNetworkSnapping({
              positions: importGraphBuffer(
                resources.graph,
                'points',
                pointBuffer,
                'float32x2',
                pointCount
              ),
              topology: {
                offsets: resources.input('offsets', 'uint32', fixture.offsets),
                neighbors: resources.input('neighbors', 'uint32', fixture.neighbors),
                weights: weighted
                  ? resources.input('weights', 'float32', fixture.weights)
                  : undefined,
                nodePositions: resources.input('nodes', 'float32x2', fixture.positions)
              },
              maximumSnapDistance: resources.input('radius', 'float32', [fixture.radius]),
              candidateCapacity: indexed ? fixture.candidateCapacity : undefined,
              spatialSort: indexed,
              startDirection: direction,
              output: {
                snappedEdges: resources.output('edges', 'uint32', pointCount),
                snapFractions: resources.output('fractions', 'float32', pointCount),
                snapDistances: resources.output('distances', 'float32', pointCount),
                sourceCosts: resources.output('sourceCosts', 'float32', pointCount),
                targetCosts: resources.output('targetCosts', 'float32', pointCount),
                snappedPositions: importGraphBuffer(
                  resources.graph,
                  'snappedPositions',
                  positionBuffer,
                  'float32x2',
                  pointCount
                ),
                startNodes: resources.output('startNodes', 'uint32', 2 * pointCount),
                startCosts: resources.output('startCosts', 'float32', 2 * pointCount),
                status: indexed ? {overflow: resources.output('overflow', 'uint32', 1)} : undefined
              }
            })
          );
          const compiled = resources.graph.compile();
          try {
            submitGraph(device, compiled, undefined);
            const comparison = {...fixture, case: caseName};
            const edges = await resources.readUint('edges');
            compareParity(
              'GPUNetworkSnapping',
              comparison,
              fixture.exactTieBreak
                ? 'every smallest-ID exact tie winner and unsnapped mask'
                : 'every winner belongs to enumerated G uncertainty interval; exact unmatched mask',
              edges.map((edge, index) =>
                Number(
                  fixture.queries[index].candidates.length
                    ? !fixture.queries[index].candidates.some(candidate => candidate.edge === edge)
                    : edge !== NONE
                )
              ),
              new Array(pointCount).fill(0),
              EXACT
            );
            if (indexed)
              compareParity(
                'GPUNetworkSnapping',
                comparison,
                'no candidate overflow',
                await resources.readUint('overflow'),
                [0],
                EXACT
              );
            const expected: Record<string, number[]> = {
              fractions: [],
              distances: [],
              sourceCosts: [],
              targetCosts: [],
              positions: [],
              startNodes: [],
              startCosts: []
            };
            const bounds: Record<string, number[]> = {
              fractions: [],
              distances: [],
              sourceCosts: [],
              targetCosts: [],
              positions: [],
              startCosts: []
            };
            fixture.queries.forEach((query, index) => {
              // An invalid edge has already failed the membership assertion; use the first reference
              // only to keep reporting numerical errors without indexing untrusted GPU IDs.
              const candidate =
                query.candidates.find(value => value.edge === edges[index]) ?? query.candidates[0];
              const cost = candidate ? (weighted ? candidate.weight : candidate.length) : 0;
              const costUlp = cost > 0 ? 2 ** (Math.floor(Math.log2(cost)) - 23) : 0;
              const costBound = candidate
                ? cost * candidate.fractionBound +
                  (weighted ? 8 * costUlp : fixture.coordinateBound)
                : 0;
              const sourceCost = candidate ? candidate.fraction * cost : -1;
              const targetCost = candidate ? (1 - candidate.fraction) * cost : -1;
              expected['fractions'].push(candidate?.fraction ?? -1);
              expected['distances'].push(candidate?.distance ?? -1);
              expected['sourceCosts'].push(sourceCost);
              expected['targetCosts'].push(targetCost);
              expected['positions'].push(...(candidate?.position ?? query.point));
              bounds['fractions'].push(candidate?.fractionBound ?? 0);
              bounds['distances'].push(candidate ? fixture.coordinateBound : 0);
              bounds['sourceCosts'].push(costBound);
              bounds['targetCosts'].push(costBound);
              bounds['positions'].push(
                candidate ? fixture.coordinateBound : 0,
                candidate ? fixture.coordinateBound : 0
              );
              const useSource = Boolean(candidate && direction !== 'forward');
              const useTarget = Boolean(candidate && direction !== 'reverse');
              expected['startNodes'].push(
                useSource ? candidate.source : NONE,
                useTarget ? candidate.target : NONE
              );
              expected['startCosts'].push(useSource ? sourceCost : -1, useTarget ? targetCost : -1);
              bounds['startCosts'].push(useSource ? costBound : 0, useTarget ? costBound : 0);
            });
            compareParity(
              'GPUNetworkSnapping',
              comparison,
              'every directional start node',
              await resources.readUint('startNodes'),
              expected['startNodes'],
              EXACT
            );
            for (const metric of [
              'fractions',
              'distances',
              'sourceCosts',
              'targetCosts',
              'startCosts'
            ]) {
              compareGeometry(
                comparison,
                metric,
                await resources.readFloat(metric),
                expected[metric],
                bounds[metric]
              );
            }
            compareParity(
              'GPUNetworkSnapping',
              comparison,
              'every uploaded input coordinate remains exact',
              await readFloat32(pointBuffer, pointCount * 2),
              pointValues,
              EXACT
            );
            const positions = await readFloat32(positionBuffer, pointCount * 2);
            // Chicago inputs are f32 AEQD metres, |coordinate| <= 32768: ulp <= 2^-9 m.
            // The matched-position budget is 16 ulps = 0.03125 m; the logged 0.000961 m
            // maximum is below even one ulp. The four failures came from JSON-module
            // transformation perturbing unmatched f64 references, fixed by raw JSON parsing.
            // Unmatched rows only copy already-f32 inputs: their bound MUST remain zero.
            // Separate these populations so a large permitted matched error cannot hide the
            // location or size of an exact-copy failure. Edge-specific projections above
            // already handle reciprocal/near ties without comparing unrelated edge positions.
            for (const matched of [true, false]) {
              const indices = positions.flatMap((_, index) =>
                Boolean(fixture.queries[Math.floor(index / 2)].candidates.length) === matched
                  ? [index]
                  : []
              );
              compareGeometry(
                comparison,
                matched ? 'matched positions (metres)' : 'unmatched input copies (exact metres)',
                indices.map(index => positions[index]),
                indices.map(index => expected['positions'][index]),
                indices.map(index => bounds['positions'][index]),
                indices
              );
            }
          } finally {
            compiled.destroy();
            positionBuffer.destroy();
            pointBuffer.destroy();
            resources.destroy();
          }
        }, 180000);
      }
    }
  }
}
