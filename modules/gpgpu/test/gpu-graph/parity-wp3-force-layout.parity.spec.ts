// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {GPUVector} from '@luma.gl/gpgpu/gpu-data';
import {GPUGraphForceLayout} from '@luma.gl/gpgpu/gpu-graph';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {
  readFloat32,
  submitGraph
} from '../../../experimental/test/utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../../experimental/test/utils/require-test-device';
import {compareParity, EXACT} from '../../../experimental/test/gpu-network/parity-wp3-comparison';
import {createParityResources} from '../../../experimental/test/gpu-network/parity-wp3-round2-resources';
import {createMetricsTopology} from './parity-wp3-graph-metrics-fixture';
import {
  evaluateLayout,
  layoutFixtures,
  type LayoutEvaluation,
  type LayoutFixture
} from './parity-wp3-force-layout-fixture';

function reportQuality(
  fixture: LayoutFixture,
  caseName: string,
  metric: string,
  actual: readonly number[],
  expected: readonly number[],
  quality: Record<string, unknown>
): void {
  console.log(
    'PARITY',
    JSON.stringify({
      class: 'GPUGraphForceLayout',
      case: caseName,
      reference: fixture.reference,
      n: expected.length,
      metric,
      maxAbs: Math.max(0, ...actual.map((value, index) => Math.abs(value - expected[index]))),
      maxRel: Math.max(
        0,
        ...actual.map(
          (value, index) =>
            Math.abs(value - expected[index]) / Math.max(Math.abs(expected[index]), Number.EPSILON)
        )
      ),
      mismatches: 0,
      tolerance: {
        profile: 'H',
        comparison: 'descriptive quality; different optimizers, no position equality',
        ...quality
      }
    })
  );
}

for (const fixture of layoutFixtures) {
  for (const run of fixture.runs) {
    it(`external layout quality ${fixture.case}/seed-${run.seed}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const resources = createParityResources(device);
      const topology = createMetricsTopology(resources, fixture);
      const positionBuffer = device.createBuffer({
        data: Float32Array.from(run.initial.positions),
        usage: Buffer.STORAGE | Buffer.VERTEX | Buffer.COPY_SRC | Buffer.COPY_DST
      });
      const velocityBuffer = device.createBuffer({
        data: new Float32Array(fixture.nodeCount * 2),
        usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
      });
      const positions = new GPUVector({
        name: 'positions',
        type: 'buffer',
        format: 'float32x2',
        buffer: positionBuffer,
        length: fixture.nodeCount,
        ownsBuffer: false
      });
      const velocities = new GPUVector({
        name: 'velocities',
        type: 'buffer',
        format: 'float32x2',
        buffer: velocityBuffer,
        length: fixture.nodeCount,
        ownsBuffer: false
      });
      new GPUGraphForceLayout({
        topology,
        positions,
        velocities,
        ...fixture.parameters,
        iterationsPerFrame: fixture.iterationsPerFrame
      }).addToGraph(resources.graph);
      const compiled = resources.graph.compile();
      try {
        for (let frame = 0; frame < fixture.frames; frame++)
          submitGraph(device, compiled, undefined);
        const actualPositions = await readFloat32(positionBuffer, fixture.nodeCount * 2);
        const actualVelocities = await readFloat32(velocityBuffer, fixture.nodeCount * 2);
        const comparison = {...fixture, case: `${fixture.case}/seed-${run.seed}`};
        compareParity(
          'GPUGraphForceLayout',
          comparison,
          'every position and velocity finite',
          [...actualPositions, ...actualVelocities].map(value => Number(!Number.isFinite(value))),
          new Array(fixture.nodeCount * 4).fill(0),
          EXACT
        );
        // Per-step clipping implies displacement <= steps*dt*maxVelocity by triangle
        // inequality. F32 accumulation adds gamma_(2*steps) times the coordinate scale.
        const steps = fixture.frames * fixture.iterationsPerFrame;
        const displacementBound =
          steps * fixture.parameters.timeStep * fixture.parameters.maxVelocity;
        const unitRoundoff = 2 ** -24;
        const accumulationBound =
          ((2 * steps * unitRoundoff) / (1 - 2 * steps * unitRoundoff)) *
          (Math.max(...run.initial.positions.map(Math.abs)) + displacementBound);
        const violations = Array.from({length: fixture.nodeCount}, (_, node) =>
          Number(
            Math.hypot(
              actualPositions[node * 2] - run.initial.positions[node * 2],
              actualPositions[node * 2 + 1] - run.initial.positions[node * 2 + 1]
            ) >
              displacementBound + 2 * accumulationBound ||
              Math.hypot(actualVelocities[node * 2], actualVelocities[node * 2 + 1]) >
                fixture.parameters.maxVelocity * (1 + 8 * unitRoundoff)
          )
        );
        compareParity(
          'GPUGraphForceLayout',
          comparison,
          'every displacement and speed obeys derived integration bounds',
          violations,
          violations.map(() => 0),
          EXACT
        );
        const evaluated = evaluateLayout(fixture, actualPositions);
        for (const method of ['optimized', 'networkx'] as const) {
          const reference: LayoutEvaluation = run[method];
          const referenceGain = run.initial.energy - reference.energy;
          const quality = {
            method,
            initialEnergy: run.initial.energy,
            gpuEnergy: evaluated.energy,
            referenceEnergy: reference.energy,
            improvementRatio:
              referenceGain === 0 ? null : (run.initial.energy - evaluated.energy) / referenceGain,
            gpuGradientRms: evaluated.rootMeanSquareGradient,
            referenceGradientRms: reference.rootMeanSquareGradient,
            optimizer:
              method === 'optimized'
                ? run.optimizer
                : 'NetworkX FR; centered and energy-scaled; different force law'
          };
          reportQuality(
            fixture,
            comparison.case,
            `${method}: all edge lengths`,
            evaluated.edgeLengths,
            reference.edgeLengths,
            quality
          );
          reportQuality(
            fixture,
            comparison.case,
            `${method}: common energy components`,
            evaluated.components,
            reference.components,
            quality
          );
          reportQuality(
            fixture,
            comparison.case,
            `${method}: every vertex equilibrium residual magnitude`,
            Array.from({length: fixture.nodeCount}, (_, node) =>
              Math.hypot(evaluated.gradient[2 * node], evaluated.gradient[2 * node + 1])
            ),
            Array.from({length: fixture.nodeCount}, (_, node) =>
              Math.hypot(reference.gradient[2 * node], reference.gradient[2 * node + 1])
            ),
            quality
          );
        }
        if (fixture.exactEquilibrium) {
          compareParity(
            'GPUGraphForceLayout',
            comparison,
            'analytic stationary pair positions',
            actualPositions,
            run.optimized.positions,
            {absolute: 1e-6, relative: 1e-4}
          );
          compareParity(
            'GPUGraphForceLayout',
            comparison,
            'analytic stationary pair velocities',
            actualVelocities,
            actualVelocities.map(() => 0),
            {absolute: 1e-6, relative: 0}
          );
        }
        const status = [
          ...(await resources.readUint('invalid')),
          ...(await resources.readUint('forward-overflow'))
        ];
        compareParity(
          'GPUGraphForceLayout',
          comparison,
          'complete topology: invalid/overflow',
          status,
          [0, 0],
          EXACT
        );
      } finally {
        compiled.destroy();
        positions.destroy();
        velocities.destroy();
        positionBuffer.destroy();
        velocityBuffer.destroy();
        resources.destroy();
      }
    }, 180000);
  }
}
