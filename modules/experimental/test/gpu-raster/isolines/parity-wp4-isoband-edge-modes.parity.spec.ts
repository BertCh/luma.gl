// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {GPUIsobands, getGPUIsobandsParameterValues} from '../../../src/gpu-raster/isolines';
import {makeGPURasterMetadata} from '../../../src/gpu-raster';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {ringCases} from '../../oracles/parity-wp4-round5-fixtures';
import {decodePackedNumbers} from '../../oracles/parity-wp4-round3-fixtures';
import {ParityRasterGraph} from '../../oracles/parity-wp4-round3-graph';
import {compareRoundFour} from '../../oracles/parity-wp4-round4-compare';
import {contourBound} from '../../oracles/parity-wp4-contour-compare';
import {
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

for (const scene of ringCases) {
  it(`GPUIsobands direct boundaries against independent Shapely rings: ${scene.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const owner = new ParityRasterGraph(device, scene.name);
    const grid = makeGPURasterMetadata(scene);
    const reference = scene.rings.flatMap(ring =>
      Array.from({length: ring.positions.length / 2 - 1}, (_, index) => ({
        band: ring.band,
        points: ring.positions.slice(2 * index, 2 * index + 4)
      }))
    );
    const capacity = Math.max(32, reference.length * 2);
    const buffer = createOutputBuffer(device, capacity * 4);
    owner.buffers.push(buffer);
    const bands = owner.output('bands', 'uint32', capacity);
    const count = owner.output('count', 'uint32', 1);
    const overflow = owner.output('overflow', 'uint32', 1);
    owner.graph.add(
      new GPUIsobands({
        grid,
        values: owner.input(
          'values',
          'float32',
          new Float32Array(await decodePackedNumbers(scene.values))
        ),
        validity: owner.input(
          'validity',
          'uint32',
          new Uint32Array(await decodePackedNumbers(scene.validity))
        ),
        noDataValue: -9999,
        breaks: owner.input('breaks', 'float32', new Float32Array(scene.breaks)),
        parameters: owner.input(
          'parameters',
          'float32',
          getGPUIsobandsParameterValues(
            {
              breakCount: scene.breaks.length,
              extent: [0, 0, scene.width, scene.height],
              firstBand: scene.firstBand,
              lastBand: scene.lastBand
            },
            grid
          )
        ),
        output: {
          edges: importGraphBuffer(owner.graph, 'edges', buffer, 'float32x4', capacity),
          edgeBands: bands.view,
          status: {edgeCount: count.view, edgeOverflow: overflow.view}
        }
      })
    );
    const compiled = owner.graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      const [edgeCount] = await readUint32(count.buffer, 1);
      compareRoundFour(
        'GPUIsobands',
        scene.name,
        scene.reference,
        'boundary-count-and-overflow',
        [edgeCount, ...(await readUint32(overflow.buffer, 1))],
        [reference.length, 0],
        'E'
      );
      const coordinates = await readFloat32(buffer, edgeCount * 4);
      const labels = await readUint32(bands.buffer, edgeCount);
      const remaining = [...reference];
      const actual: number[] = [];
      const expected: number[] = [];
      let unmatched = 0;
      const bound = contourBound(scene.width, scene.height);
      for (let index = 0; index < edgeCount; index++) {
        const points = coordinates.slice(index * 4, index * 4 + 4);
        const selected = remaining.findIndex(
          edge =>
            edge.band === labels[index] &&
            edge.points.every((value, coordinate) => Math.abs(value - points[coordinate]) <= bound)
        );
        if (selected < 0) unmatched++;
        else {
          actual.push(...points);
          expected.push(...remaining[selected].points);
          remaining.splice(selected, 1);
        }
      }
      compareRoundFour(
        'GPUIsobands',
        scene.name,
        scene.reference,
        'complete-oriented-band-boundary-bijection',
        [unmatched, remaining.length],
        [0, 0],
        'E'
      );
      compareRoundFour(
        'GPUIsobands',
        scene.name,
        scene.reference,
        'all-boundary-coordinates',
        actual,
        expected,
        'G',
        bound
      );
    } finally {
      compiled.destroy();
      owner.destroy();
    }
  });
}
