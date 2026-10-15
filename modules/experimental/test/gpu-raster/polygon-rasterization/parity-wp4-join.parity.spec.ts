// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {
  GPURasterJoin,
  getGPUPolygonRasterizationParameterValues
} from '../../../src/gpu-raster/polygon-rasterization';
import {GPUParameterBuffer} from '../../../src/utils/gpu-contributor-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {ParityRasterGraph} from '../../oracles/parity-wp4-round3-graph';
import {decodePackedNumbers} from '../../oracles/parity-wp4-round3-fixtures';
import {compareRasterParity} from '../../oracles/parity-wp4-compare';
import {joinCases, joinMetrics} from '../../oracles/parity-wp4-round6-fixtures';

for (const scene of joinCases) {
  it(`GPURasterJoin independent rasterio parity: ${scene.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const [zones, boundary, points, values] = await Promise.all(
      [scene.zones, scene.boundary, scene.points, scene.values].map(decodePackedNumbers)
    );
    const rig = new ParityRasterGraph(device, scene.name);
    const parameters = new GPUParameterBuffer(device, {
      id: 'join-parameters',
      format: 'float32',
      length: 4
    });
    const outputs = {
      pointZones: rig.output('point-zones', 'uint32', scene.pointCount),
      pointBoundaryMask: rig.output('point-boundary', 'uint32', scene.pointCount),
      counts: rig.output('counts', 'uint32', scene.zoneCount),
      sums: rig.output('sums', 'float32', scene.zoneCount),
      boundaryCounts: rig.output('boundary-counts', 'uint32', scene.zoneCount),
      unassignedBoundaryCount: rig.output('unassigned', 'uint32', 1),
      outsideCount: rig.output('outside', 'uint32', 1)
    };
    rig.graph.add(
      new GPURasterJoin({
        grid: scene.grid,
        parameters: parameters.importToGraph(rig.graph),
        points: rig.input('points', 'float32x2', new Float32Array(points)),
        values: rig.input('values', 'float32', new Float32Array(values)),
        zones: rig.input('zones', 'uint32', new Uint32Array(zones)),
        boundary: rig.input('boundary', 'uint32', new Uint32Array(boundary)),
        zoneCount: scene.zoneCount,
        output: {
          pointZones: outputs.pointZones.view,
          pointBoundaryMask: outputs.pointBoundaryMask.view,
          counts: outputs.counts.view,
          sums: outputs.sums.view,
          boundaryCounts: outputs.boundaryCounts.view,
          unassignedBoundaryCount: outputs.unassignedBoundaryCount.view,
          outsideCount: outputs.outsideCount.view
        }
      })
    );
    const compiled = rig.graph.compile();
    try {
      for (const [frameIndex, frame] of scene.frames.entries()) {
        parameters.write(getGPUPolygonRasterizationParameterValues(frame.grid));
        submitGraph(device, compiled, undefined);
        for (const metric of joinMetrics) {
          const output = outputs[metric];
          const actual = await (metric === 'sums' ? readFloat32 : readUint32)(
            output.buffer,
            output.view.length
          );
          // E also applies to sums: all finite contributions are quarter integers and
          // even their sum of absolute values is < 2^24 / 4. Every reduction is exact.
          compareRasterParity(
            'GPURasterJoin',
            `${scene.name}/frame-${frameIndex}`,
            scene.reference,
            metric,
            actual,
            await decodePackedNumbers(frame.expected[metric]),
            'E'
          );
        }
      }
    } finally {
      compiled.destroy();
      parameters.destroy();
      rig.destroy();
    }
  });
}
