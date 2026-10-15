// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {GPUVector} from '@luma.gl/gpgpu/gpu-data';
import {GPUGraphForceLayout, GPUGraphSpatialForceLayout} from '@luma.gl/gpgpu/gpu-graph';
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
import {evaluateLayout} from './parity-wp3-force-layout-fixture';
import {canonicalizeGridIds, spatialFixtures} from './parity-wp3-spatial-force-layout-fixture';

for (const fixture of spatialFixtures) {
  for (const run of fixture.runs) {
    for (const mode of ['gradient', 'exact', 'approximate'] as const) {
      it(`external spatial force ${fixture.case}/${run.seed}/${mode}`, async () => {
        const device = await getWebGPUTestDevice();
        requireTestDevice(device);
        const resources = createParityResources(device);
        const topology = createMetricsTopology(resources, fixture);
        const buffers: Buffer[] = [];
        const vectors: GPUVector<'float32x2'>[] = [];
        const vectorPair = (name: string, values: number[], renderable = false) => {
          const buffer = device.createBuffer({
            data: Float32Array.from(values),
            usage:
              Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST | (renderable ? Buffer.VERTEX : 0)
          });
          const vector = new GPUVector({
            name,
            type: 'buffer',
            format: 'float32x2',
            buffer,
            length: values.length / 2,
            ownsBuffer: false
          });
          buffers.push(buffer);
          vectors.push(vector);
          return {buffer, vector};
        };
        const positions = vectorPair('positions', run.initial.positions, true);
        const velocities = vectorPair('velocities', new Array(fixture.nodeCount * 2).fill(0));
        const cellCount = fixture.gridSize[0] * fixture.gridSize[1];
        const centers = vectorPair('cell-centers', new Array(cellCount * 2).fill(0));
        const parameters =
          mode === 'gradient'
            ? {...fixture.parameters, timeStep: Math.fround(0.001), damping: 1, maxVelocity: 1e6}
            : fixture.parameters;
        const layout = new GPUGraphForceLayout({
          topology,
          positions: positions.vector,
          velocities: velocities.vector,
          pinned: resources.vector('pinned', 'uint32', fixture.pinned),
          ...parameters,
          iterationsPerFrame: mode === 'gradient' ? 1 : fixture.iterationsPerFrame
        });
        new GPUGraphSpatialForceLayout({
          layout,
          gridSize: fixture.gridSize,
          bounds: fixture.bounds,
          theta: mode === 'approximate' ? 0.6 : 0,
          nearCellRadius: 1,
          cellOffsets: resources.vector('cell-offsets', 'uint32', new Array(cellCount + 1).fill(0)),
          vertexIds: resources.vector('vertex-ids', 'uint32', new Array(fixture.nodeCount).fill(0)),
          cellCenters: centers.vector,
          count: resources.vector('count', 'uint32', [0]),
          overflow: resources.vector('overflow', 'uint32', [0])
        }).addToGraph(resources.graph);
        const compiled = resources.graph.compile();
        const comparison = {...fixture, case: `${fixture.case}/${run.seed}/${mode}`};
        const className = 'GPUGraphSpatialForceLayout';
        try {
          const frames = mode === 'gradient' ? 1 : fixture.frames;
          for (let frame = 0; frame < frames; frame++) submitGraph(device, compiled, undefined);
          const actualPositions = await readFloat32(positions.buffer, fixture.nodeCount * 2);
          const actualVelocities = await readFloat32(velocities.buffer, fixture.nodeCount * 2);
          compareParity(
            className,
            comparison,
            'every position and velocity finite',
            [...actualPositions, ...actualVelocities].map(value => Number(!Number.isFinite(value))),
            new Array(fixture.nodeCount * 4).fill(0),
            EXACT
          );
          const anchorIndices = run.initial.positions
            .map((_, index) => index)
            .filter(index => fixture.pinned[Math.floor(index / 2)]);
          compareParity(
            className,
            comparison,
            'all fixed anchor coordinates',
            anchorIndices.map(index => actualPositions[index]),
            anchorIndices.map(index => run.initial.positions[index]),
            EXACT
          );
          compareParity(
            className,
            comparison,
            'all fixed anchor velocities',
            anchorIndices.map(index => actualVelocities[index]),
            anchorIndices.map(() => 0),
            EXACT
          );
          compareParity(
            className,
            comparison,
            'index count/overflow and topology status',
            [
              ...(await resources.readUint('count')),
              ...(await resources.readUint('overflow')),
              ...(await resources.readUint('invalid')),
              ...(await resources.readUint('forward-overflow'))
            ],
            [fixture.nodeCount, 0, 0, 0],
            EXACT
          );
          if (mode === 'gradient') {
            // One unconstrained step from rest with damping=1 exposes -grad(E)*dt.
            // Compare every free force coordinate to SciPy's independently differentiated
            // scalar energy. dt=1e-3 and maxVelocity=1e6 prevent clipping (node-checked).
            const expectedGradient = run.initial.gradient.map((value, index) =>
              fixture.pinned[Math.floor(index / 2)] ? 0 : value
            );
            compareParity(
              className,
              comparison,
              'exact theta-zero force versus SciPy energy derivative',
              actualVelocities.map(value => -value / parameters.timeStep),
              expectedGradient,
              {absolute: 1e-6 * Math.max(1, ...expectedGradient.map(Math.abs)), relative: 1e-4}
            );
            const gridOffsets = await resources.readUint('cell-offsets');
            compareParity(
              className,
              comparison,
              'every initial grid offset versus SciPy bins',
              gridOffsets,
              run.gridReference.offsets,
              EXACT
            );
            compareParity(
              className,
              comparison,
              'every initial grid vertex ID, canonical within each cell',
              canonicalizeGridIds(await resources.readUint('vertex-ids'), gridOffsets),
              canonicalizeGridIds(run.gridReference.vertexIds, run.gridReference.offsets),
              EXACT
            );
            compareParity(
              className,
              comparison,
              'every initial grid center versus SciPy bin means',
              await readFloat32(centers.buffer, 2 * cellCount),
              run.gridReference.centers,
              {absolute: 1e-6 * Math.max(1, ...run.initial.positions.map(Math.abs)), relative: 1e-4}
            );
          } else {
            // Per-step clipping bounds each displacement by steps*dt*maxVelocity.
            // Sequential f32 updates add gamma_(2*steps)*(initial scale + bound).
            const steps = frames * fixture.iterationsPerFrame;
            const bound = steps * parameters.timeStep * parameters.maxVelocity;
            const unitRoundoff = 2 ** -24;
            const rounding =
              ((2 * steps * unitRoundoff) / (1 - 2 * steps * unitRoundoff)) *
              (Math.max(...run.initial.positions.map(Math.abs)) + bound);
            const violations = Array.from({length: fixture.nodeCount}, (_, node) =>
              Number(
                Math.hypot(
                  actualPositions[2 * node] - run.initial.positions[2 * node],
                  actualPositions[2 * node + 1] - run.initial.positions[2 * node + 1]
                ) >
                  bound + 2 * rounding ||
                  Math.hypot(actualVelocities[2 * node], actualVelocities[2 * node + 1]) >
                    parameters.maxVelocity * (1 + 8 * unitRoundoff)
              )
            );
            compareParity(
              className,
              comparison,
              'all displacement and speed bounds',
              violations,
              violations.map(() => 0),
              EXACT
            );
          }
          const evaluated = evaluateLayout(fixture, actualPositions);
          for (const method of ['optimized', 'networkx'] as const) {
            const reference = run[method];
            const freeResiduals = (gradient: number[]) =>
              Array.from({length: fixture.nodeCount}, (_, node) =>
                fixture.pinned[node] ? 0 : Math.hypot(gradient[2 * node], gradient[2 * node + 1])
              );
            for (const [metric, actual, expected] of [
              ['all edge lengths', evaluated.edgeLengths, reference.edgeLengths],
              ['energy components', evaluated.components, reference.components],
              [
                'every free-vertex equilibrium residual',
                freeResiduals(evaluated.gradient),
                freeResiduals(reference.gradient)
              ]
            ] as const) {
              // H profile: different optimizers and a monopole approximation have no
              // equal-coordinate contract. Report full-array errors and energy quality.
              console.log(
                'PARITY',
                JSON.stringify({
                  class: className,
                  case: comparison.case,
                  reference: fixture.reference,
                  n: expected.length,
                  metric: `${method}: ${metric}`,
                  maxAbs: Math.max(
                    0,
                    ...actual.map((value, index) => Math.abs(value - expected[index]))
                  ),
                  maxRel: Math.max(
                    0,
                    ...actual.map(
                      (value, index) =>
                        Math.abs(value - expected[index]) /
                        Math.max(Math.abs(expected[index]), Number.EPSILON)
                    )
                  ),
                  mismatches: 0,
                  tolerance: {
                    profile: 'H',
                    comparison: 'descriptive quality; no optimizer equality',
                    initialEnergy: run.initial.energy,
                    gpuEnergy: evaluated.energy,
                    referenceEnergy: reference.energy,
                    improvementRatio:
                      run.initial.energy === reference.energy
                        ? null
                        : (run.initial.energy - evaluated.energy) /
                          (run.initial.energy - reference.energy),
                    optimizer:
                      method === 'optimized'
                        ? run.optimizer
                        : 'NetworkX FR with fixed anchors; different attraction law'
                  }
                })
              );
            }
          }
          if (fixture.exactEquilibrium) {
            compareParity(
              className,
              comparison,
              'analytic stationary pair',
              actualPositions,
              run.initial.positions,
              {absolute: 1e-6, relative: 1e-4}
            );
          }
        } finally {
          compiled.destroy();
          for (const vector of vectors) vector.destroy();
          for (const buffer of buffers) buffer.destroy();
          resources.destroy();
        }
      }, 180000);
    }
  }
}
