// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {GPUIsolines, getGPUIsolinesParameterValues} from '../../../src/gpu-raster/isolines';
import {makeGPURasterMetadata} from '../../../src/gpu-raster';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {lineCases} from '../../oracles/parity-wp4-round5-fixtures';
import {decodePackedNumbers} from '../../oracles/parity-wp4-round3-fixtures';
import {ParityRasterGraph} from '../../oracles/parity-wp4-round3-graph';
import {compareRoundFour} from '../../oracles/parity-wp4-round4-compare';
import {compareLineSegments} from '../../oracles/parity-wp4-contour-compare';
import {
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

for (const scene of lineCases) {
  it(`GPUIsolines scikit-image parity: ${scene.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const [values, validity, expected] = await Promise.all([
      decodePackedNumbers(scene.values),
      decodePackedNumbers(scene.validity),
      decodePackedNumbers(scene.segments)
    ]);
    const owner = new ParityRasterGraph(device, scene.name);
    const grid = makeGPURasterMetadata(scene);
    const capacity = 2 * (scene.width - 1) * (scene.height - 1) * scene.levels.length;
    const coordinates = createOutputBuffer(device, capacity * 4);
    const edgeIds = createOutputBuffer(device, capacity * 2);
    owner.buffers.push(coordinates, edgeIds);
    const levels = owner.output('segment-levels', 'uint32', capacity);
    const count = owner.output('count', 'uint32', 1);
    const requiredCount = owner.output('required-count', 'uint32', 1);
    const overflow = owner.output('overflow', 'uint32', 1);
    owner.graph.add(
      new GPUIsolines({
        grid,
        values: owner.input('values', 'float32', new Float32Array(values)),
        validity: owner.input('validity', 'uint32', new Uint32Array(validity)),
        noDataValue: -9999,
        levels: owner.input('levels', 'float32', new Float32Array(scene.levels)),
        parameters: owner.input(
          'parameters',
          'float32',
          getGPUIsolinesParameterValues(
            {
              levelCount: scene.levels.length,
              extent: [0, 0, scene.width, scene.height]
            },
            grid
          )
        ),
        output: {
          segments: importGraphBuffer(owner.graph, 'segments', coordinates, 'float32x4', capacity),
          segmentEdges: importGraphBuffer(owner.graph, 'edge-ids', edgeIds, 'uint32x2', capacity),
          segmentLevels: levels.view,
          status: {count: count.view, requiredCount: requiredCount.view, overflow: overflow.view}
        }
      })
    );
    const compiled = owner.graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      const [segmentCount] = await readUint32(count.buffer, 1);
      compareRoundFour(
        'GPUIsolines',
        scene.name,
        scene.reference,
        'complete-output-status',
        [
          segmentCount,
          ...(await readUint32(requiredCount.buffer, 1)),
          ...(await readUint32(overflow.buffer, 1))
        ],
        [scene.segmentCount, scene.segmentCount, 0],
        'E'
      );
      compareLineSegments(
        scene.name,
        scene.reference,
        scene.width,
        scene.height,
        await readFloat32(coordinates, segmentCount * 4),
        await readUint32(levels.buffer, segmentCount),
        await readUint32(edgeIds, segmentCount * 2),
        expected
      );
    } finally {
      compiled.destroy();
      owner.destroy();
    }
  });
}
