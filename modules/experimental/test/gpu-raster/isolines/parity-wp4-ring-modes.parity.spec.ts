// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {GPUIsobandRings, getGPUIsobandsParameterValues} from '../../../src/gpu-raster/isolines';
import {makeGPURasterMetadata} from '../../../src/gpu-raster';
import {ringCases} from '../../oracles/parity-wp4-round5-fixtures';
import {decodePackedNumbers} from '../../oracles/parity-wp4-round3-fixtures';
import {ParityRasterGraph} from '../../oracles/parity-wp4-round3-graph';
import {compareRoundFour} from '../../oracles/parity-wp4-round4-compare';
import {compareContourRings} from '../../oracles/parity-wp4-contour-compare';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

for (const original of ringCases.filter(
  scene => scene.name === 'nested-annulus-island-and-domain-notch'
)) {
  for (const vertexTolerance of [1e-4, 15, 31]) {
    for (const splitTouchingRings of [false, true]) {
      const scene = {
        ...original,
        name: `${original.name}/tolerance-${vertexTolerance}/split-${splitTouchingRings}`
      };
      const coordinateScale = 128;
      // Minimum independent boundary edge is 0.5 cells: 64 world units exceeds 31.
      // Maximum coordinate is 2176, below i32 hash saturation even at tolerance 1e-4.
      it(`GPUIsobandRings scikit-image/Shapely parity: ${scene.name}`, async () => {
        const device = await getWebGPUTestDevice();
        requireTestDevice(device);
        const [values, validity] = await Promise.all([
          decodePackedNumbers(scene.values),
          decodePackedNumbers(scene.validity)
        ]);
        const owner = new ParityRasterGraph(device, scene.name);
        const grid = makeGPURasterMetadata(scene);
        // Independent reference supplies the complete boundary size. A factor two
        // leaves spare capacity for degeneracies without constraining GPU ordering.
        const expectedEdges = scene.rings.reduce(
          (sum, ring) => sum + ring.positions.length / 2 - 1,
          0
        );
        const edgeCapacity = Math.max(32, expectedEdges * 2);
        const ringCapacity = Math.max(32, scene.rings.length * 2);
        const offsets = owner.output('ring-offsets', 'uint32', ringCapacity + 1);
        const positions = owner.output('positions', 'float32x2', edgeCapacity + ringCapacity);
        const areas = owner.output('areas', 'float32', ringCapacity);
        const holes = owner.output('holes', 'uint32', ringCapacity);
        const shells = owner.output('shells', 'uint32', ringCapacity);
        const groups = owner.output('groups', 'uint32', ringCapacity);
        const count = owner.output('count', 'uint32', 1);
        const overflow = owner.output('overflow', 'uint32', 1);
        const openCount = owner.output('open-count', 'uint32', 1);
        const edgeCount = owner.output('edge-count', 'uint32', 1);
        const edgeOverflow = owner.output('edge-overflow', 'uint32', 1);
        owner.graph.add(
          new GPUIsobandRings({
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
                  extent: [0, 0, scene.width * coordinateScale, scene.height * coordinateScale],
                  firstBand: scene.firstBand,
                  lastBand: scene.lastBand
                },
                grid
              )
            ),
            edgeCapacity,
            vertexTolerance,
            splitTouchingRings,
            output: {
              ringOffsets: offsets.view,
              positions: positions.view,
              ringAreas: areas.view,
              ringIsHole: holes.view,
              ringShells: shells.view,
              ringGroups: groups.view,
              status: {
                count: count.view,
                overflow: overflow.view,
                openSegmentCount: openCount.view,
                edgeCount: edgeCount.view,
                edgeOverflow: edgeOverflow.view
              }
            }
          })
        );
        const compiled = owner.graph.compile();
        try {
          submitGraph(device, compiled, undefined);
          const [ringCount] = await readUint32(count.buffer, 1);
          compareRoundFour(
            'GPUIsobandRings',
            scene.name,
            scene.reference,
            'complete-output-status',
            [
              ringCount,
              ...(await readUint32(overflow.buffer, 1)),
              ...(await readUint32(openCount.buffer, 1)),
              ...(await readUint32(edgeOverflow.buffer, 1)),
              ...(await readUint32(edgeCount.buffer, 1))
            ],
            [scene.rings.length, 0, 0, 0, expectedEdges],
            'E'
          );
          const ringOffsets = await readUint32(offsets.buffer, ringCount + 1);
          const coordinates = await readFloat32(positions.buffer, ringOffsets[ringCount] * 2);
          const ringAreas = await readFloat32(areas.buffer, ringCount);
          const ringHoles = await readUint32(holes.buffer, ringCount);
          const ringShells = await readUint32(shells.buffer, ringCount);
          const ringGroups = await readUint32(groups.buffer, ringCount);
          compareRoundFour(
            'GPUIsobandRings',
            scene.name,
            scene.reference,
            'valid-packed-offsets',
            [
              ringOffsets[0],
              Number(
                ringOffsets.every((value, index) => index === 0 || value >= ringOffsets[index - 1])
              )
            ],
            [0, 1],
            'E'
          );
          compareContourRings(
            scene.name,
            scene.reference,
            scene.width,
            scene.height,
            Array.from({length: ringCount}, (_, index) => ({
              positions: coordinates
                .slice(ringOffsets[index] * 2, ringOffsets[index + 1] * 2)
                .map(value => value / coordinateScale),
              area: ringAreas[index] / coordinateScale ** 2,
              isHole: ringHoles[index],
              shell: ringShells[index],
              band: ringGroups[index]
            })),
            scene.rings
          );
        } finally {
          compiled.destroy();
          owner.destroy();
        }
      });
    }
  }
}
