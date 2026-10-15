// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {GEODESIC_WGSL} from '../../../src/gpu-spatial-analysis/geometry-measures/geodesic-wgsl';
import {createWGSLKernelNode} from '../../../src/utils/wgsl-kernel-nodes';
import fixture from '../../oracles/fixtures/parity-wp2-great-circle-arcs-seam-regression.json';
import {createParityGraph} from '../../oracles/parity-wp2-geodesic-harness';
import {compareParity} from '../../oracles/parity-wp2-assert';
import {readFloat32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

it('shared geodesic seam delta and distance retain f32 endpoint precision', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = createParityGraph(device);
  try {
    const entry = fixture.cases[0];
    const count = entry.sources.length / 2;
    const sources = rig.input('sources', entry.sources, 'float32x2');
    const destinations = rig.input('destinations', entry.targets, 'float32x2');
    const deltas = rig.output('deltas', count, 'float32');
    const distances = rig.output('distances', count, 'float32');
    rig.graph.add(
      createWGSLKernelNode(rig.graph, {
        id: 'seam-regression',
        operation: 'GPUGreatCircleArcs',
        variant: 'shared-geodesic-isolation',
        invocationCount: count,
        bindings: [
          {name: 'sources', view: sources.view, type: 'f32', access: 'read'},
          {name: 'destinations', view: destinations.view, type: 'f32', access: 'read'},
          {name: 'deltas', view: deltas.view, type: 'f32', access: 'read_write'},
          {name: 'distances', view: distances.view, type: 'f32', access: 'read_write'}
        ],
        declarations: GEODESIC_WGSL,
        body: `let source = vec2f(sources[sourcesOffset + 2u * index], sources[sourcesOffset + 2u * index + 1u]);
let destination = vec2f(destinations[destinationsOffset + 2u * index], destinations[destinationsOffset + 2u * index + 1u]);
deltas[deltasOffset + index] = geodesicLongitudeDelta(source.x, destination.x);
distances[distancesOffset + index] = geodesicCentralAngle(source, destination) * 6371008.8;`
      })
    );
    const compiled = rig.graph.compile();
    try {
      for (let iteration = 0; iteration < 2; iteration++) {
        submitGraph(device, compiled, undefined);
        const expectedDeltas = Array.from({length: count}, (_, index) => {
          const difference = entry.targets[2 * index] - entry.sources[2 * index];
          return difference - 360 * Math.round(difference / 360);
        });
        compareParity(
          'GPUGreatCircleArcs',
          `isolated-seam/${iteration}`,
          'exact f64 wrapped difference of f32 fixture endpoints',
          'longitude separation',
          await readFloat32(deltas.buffer, count),
          expectedDeltas
        );
        const frame = entry.frames[0];
        compareParity(
          'GPUGreatCircleArcs',
          `isolated-seam/${iteration}`,
          fixture.reference,
          'PROJ full length',
          await readFloat32(distances.buffer, count),
          frame.lineOffsets.slice(1).map(end => frame.measures[end - 1]),
          fixture.tolerance.measureAbsolute,
          fixture.tolerance.measureRelative
        );
      }
    } finally {
      compiled.destroy();
    }
  } finally {
    rig.destroy();
    device.destroy?.();
  }
});
