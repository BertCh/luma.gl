// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {GPUIsobands, getGPUIsobandsParameterValues} from '../../../src/gpu-raster/isolines';
import {makeGPURasterMetadata} from '../../../src/gpu-raster';
import {isobandCases} from '../../oracles/parity-wp4-round4-fixtures';
import {decodePackedNumbers} from '../../oracles/parity-wp4-round3-fixtures';
import {ParityRasterGraph} from '../../oracles/parity-wp4-round3-graph';
import {compareRoundFour} from '../../oracles/parity-wp4-round4-compare';
import {compareIsobandGeometry} from '../../oracles/parity-wp4-isoband-geometry';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

for (const scene of isobandCases) {
  it(`GPUIsobands scikit-image/Shapely parity: ${scene.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const [values, validity, classes, referenceTriangles, referenceEdges] = await Promise.all([
      decodePackedNumbers(scene.values),
      decodePackedNumbers(scene.validity),
      decodePackedNumbers(scene.classes),
      decodePackedNumbers(scene.triangles),
      decodePackedNumbers(scene.edges)
    ]);
    const owner = new ParityRasterGraph(device, scene.name);
    const grid = makeGPURasterMetadata(scene);
    // Two pieces, at most eight vertices each: twelve triangles per band/cell.
    const capacity = 12 * (scene.width - 1) * (scene.height - 1) * (scene.breaks.length + 1);
    const outputClasses = owner.output('classes', 'uint32', values.length);
    const triangles = owner.output('triangles', 'float32x2', capacity * 3);
    const bands = owner.output('bands', 'uint32', capacity);
    const count = owner.output('count', 'uint32', 1);
    const requiredCount = owner.output('required-count', 'uint32', 1);
    const overflow = owner.output('overflow', 'uint32', 1);
    const vertexCount = owner.output('vertex-count', 'uint32', 1);
    owner.graph.add(
      new GPUIsobands({
        grid,
        values: owner.input('values', 'float32', new Float32Array(values)),
        validity: owner.input('validity', 'uint32', new Uint32Array(validity)),
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
          bandClasses: outputClasses.view,
          triangles: triangles.view,
          triangleBands: bands.view,
          status: {
            count: count.view,
            requiredCount: requiredCount.view,
            overflow: overflow.view,
            vertexCount: vertexCount.view
          }
        }
      })
    );
    const compiled = owner.graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      const [triangleCount] = await readUint32(count.buffer, 1);
      compareRoundFour(
        'GPUIsobands',
        scene.name,
        scene.reference,
        'every-sample-band-class',
        await readUint32(outputClasses.buffer, values.length),
        classes,
        'E'
      );
      compareRoundFour(
        'GPUIsobands',
        scene.name,
        scene.reference,
        'complete-output-status',
        [
          ...(await readUint32(requiredCount.buffer, 1)),
          ...(await readUint32(vertexCount.buffer, 1)),
          ...(await readUint32(overflow.buffer, 1))
        ],
        [triangleCount, triangleCount * 3, 0],
        'E'
      );
      compareIsobandGeometry(
        scene.name,
        scene.reference,
        scene.width,
        scene.height,
        await readFloat32(triangles.buffer, triangleCount * 6),
        await readUint32(bands.buffer, triangleCount),
        referenceTriangles,
        referenceEdges
      );
    } finally {
      compiled.destroy();
      owner.destroy();
    }
  });
}
