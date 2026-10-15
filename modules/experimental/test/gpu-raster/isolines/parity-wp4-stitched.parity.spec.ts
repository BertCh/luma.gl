// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {GPUIsolines, getGPUIsolinesParameterValues} from '../../../src/gpu-raster/isolines';
import {makeGPURasterMetadata} from '../../../src/gpu-raster';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import stitched from '../../oracles/fixtures/parity-wp4-stitched.json';
import {contourBound} from '../../oracles/parity-wp4-contour-compare';
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
    const polylineVertices = owner.output('polyline-vertices', 'float32x2', capacity * 2);
    const polylineOffsets = owner.output('polyline-offsets', 'uint32', capacity + 1);
    const polylineLevels = owner.output('polyline-levels', 'uint32', capacity);
    const polylineClosed = owner.output('polyline-closed', 'uint32', capacity);
    const polylineCount = owner.output('polyline-count', 'uint32', 1);
    const polylineVertexCount = owner.output('polyline-vertex-count', 'uint32', 1);
    const polylineOverflow = owner.output('polyline-overflow', 'uint32', 1);
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
          polylines: {
            vertices: polylineVertices.view,
            polylineOffsets: polylineOffsets.view,
            polylineLevels: polylineLevels.view,
            polylineClosed: polylineClosed.view,
            polylineCount: polylineCount.view,
            vertexCount: polylineVertexCount.view
          },
          status: {
            count: count.view,
            requiredCount: requiredCount.view,
            overflow: overflow.view,
            polylineOverflow: polylineOverflow.view
          }
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

      const reference = stitched.cases.find(entry => entry.name === scene.name)!;
      const [chainCount] = await readUint32(polylineCount.buffer, 1);
      const offsets = await readUint32(polylineOffsets.buffer, chainCount + 1);
      const chainCoordinates = await readFloat32(polylineVertices.buffer, offsets[chainCount] * 2);
      const chainLevels = await readUint32(polylineLevels.buffer, chainCount);
      const closed = await readUint32(polylineClosed.buffer, chainCount);
      compareRoundFour(
        'GPUIsolines',
        scene.name,
        stitched.reference,
        'independent-chain-count-and-status',
        [
          chainCount,
          ...(await readUint32(polylineOverflow.buffer, 1)),
          ...(await readUint32(polylineVertexCount.buffer, 1)),
          offsets[0]
        ],
        [
          reference.chains.length,
          0,
          reference.chains.reduce((sum, chain) => sum + chain.positions.length / 2, 0),
          0
        ],
        'E'
      );
      const remaining = [...reference.chains];
      const received: number[] = [];
      const target: number[] = [];
      let unmatched = 0;
      const bound = contourBound(scene.width, scene.height);
      for (let index = 0; index < chainCount; index++) {
        const points = chainCoordinates.slice(offsets[index] * 2, offsets[index + 1] * 2);
        let selected = -1;
        let rotated: number[] = [];
        for (let candidate = 0; candidate < remaining.length; candidate++) {
          const chain = remaining[candidate];
          if (
            chain.level !== chainLevels[index] ||
            chain.closed !== closed[index] ||
            chain.positions.length !== points.length
          )
            continue;
          const distinct = chain.closed
            ? chain.positions.length / 2 - 1
            : chain.positions.length / 2;
          for (let start = 0; start < (chain.closed ? Math.max(1, distinct) : 1); start++) {
            const expected = Array.from(
              {length: points.length},
              (_, coordinate) =>
                chain.positions[
                  2 * ((Math.floor(coordinate / 2) + start) % Math.max(1, distinct)) +
                    (coordinate % 2)
                ]
            );
            if (!chain.closed) expected.splice(0, expected.length, ...chain.positions);
            if (
              points.every((value, coordinate) => Math.abs(value - expected[coordinate]) <= bound)
            ) {
              selected = candidate;
              rotated = expected;
              break;
            }
          }
          if (selected >= 0) break;
        }
        if (selected < 0) unmatched++;
        else {
          remaining.splice(selected, 1);
          received.push(...points);
          target.push(...rotated);
        }
      }
      compareRoundFour(
        'GPUIsolines',
        scene.name,
        stitched.reference,
        'complete-chain-level-closure-and-vertex-bijection',
        [unmatched, remaining.length],
        [0, 0],
        'E'
      );
      compareRoundFour(
        'GPUIsolines',
        scene.name,
        stitched.reference,
        'every-stitched-vertex',
        received,
        target,
        'G',
        bound
      );
    } finally {
      compiled.destroy();
      owner.destroy();
    }
  });
}
