// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {
  createLonLatPolygonCheckNode,
  LON_LAT_RING_WGSL
} from '../../../src/analysis-core/lon-lat-ring-wgsl';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {createWGSLKernelNode} from '../../../src/utils/wgsl-kernel-nodes';
import {createOutputBuffer, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {createLonLatPolygonFixtureViews, LON_LAT_POLYGON_FIXTURES} from './fixtures';
import {
  classifyRing,
  flattenLonLatFeatures,
  getLonLatFeatures,
  isAmbiguousGreatCircleEdge,
  type LonLatPosition
} from './lon-lat-polygon-oracle';

const fixtures = {
  ...LON_LAT_POLYGON_FIXTURES,
  EMPTY_RING: flattenLonLatFeatures([[[[]]]]),
  TWO_DISTINCT: flattenLonLatFeatures([
    [
      [
        [
          [179, 0],
          [-179, 0],
          [179, 0],
          [-179, 0]
        ]
      ]
    ]
  ]),
  REPEATED_CLOSURE: flattenLonLatFeatures([
    [
      [
        [
          [177, -19],
          [-178, -19],
          [-178, -16],
          [177, -16],
          [177, -19]
        ]
      ]
    ]
  ])
};

for (const [name, fixture] of Object.entries(fixtures)) {
  it(`lon/lat consumer check counts ${name} rings like the binary64 oracle`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const graph = new GPUCommandGraph(device, {id: `lon-lat-check-${name}`});
    const fields = createLonLatPolygonFixtureViews(device, graph, fixture);
    const crossingBuffer = createOutputBuffer(device, 1);
    const poleBuffer = createOutputBuffer(device, 1);
    graph.add(
      createLonLatPolygonCheckNode(graph, {
        id: 'check',
        operation: 'LonLatCheckTest',
        positions: fields.positions,
        ringOffsets: fields.ringOffsets,
        status: {
          antimeridianUncertainCount: importGraphBuffer(
            graph,
            'crossing',
            crossingBuffer,
            'uint32',
            1
          ),
          poleUncertainCount: importGraphBuffer(graph, 'pole', poleBuffer, 'uint32', 1)
        }
      })
    );
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      const rings = getLonLatFeatures(fixture)
        .flat(2)
        .map(ring => classifyRing(ring));
      expect(await readUint32(crossingBuffer, 1)).toEqual([
        rings.filter(ring => ring.crossing).length
      ]);
      expect(await readUint32(poleBuffer, 1)).toEqual([rings.filter(ring => ring.poleRing).length]);
    } finally {
      compiled.destroy();
      fields.destroy();
      crossingBuffer.destroy();
      poleBuffer.destroy();
    }
  });
}

it('great-circle ambiguity helper evaluates antipodes and pole arcs on the GPU', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const edges: [LonLatPosition, LonLatPosition][] = [
    [
      [0, 0],
      [180, 0]
    ],
    [
      [0, 70],
      [180, 70]
    ],
    [
      [0, -70],
      [180, -70]
    ],
    [
      [0, 60],
      [0, 70]
    ],
    [
      [0, 70],
      [179, 70]
    ],
    [
      [0, 90],
      [0, 70]
    ],
    [
      [180, -90],
      [-180, -90]
    ],
    [
      [0, 0],
      [179.99999, 0]
    ],
    [
      [0, 0],
      [179.999, 0]
    ]
  ];
  const graph = new GPUCommandGraph(device);
  const output = createOutputBuffer(device, edges.length);
  const fields = createLonLatPolygonFixtureViews(device, graph, flattenLonLatFeatures([[edges]]));
  graph.add(
    createWGSLKernelNode(graph, {
      id: 'ambiguity',
      operation: 'LonLatAmbiguityTest',
      invocationCount: edges.length,
      bindings: [
        {name: 'positions', view: fields.positions, type: 'f32', access: 'read'},
        {
          name: 'output',
          view: importGraphBuffer(graph, 'output', output, 'uint32', edges.length),
          type: 'u32',
          access: 'read_write'
        }
      ],
      declarations: LON_LAT_RING_WGSL,
      body: `let offset = positionsOffset + index * 4u;
      let first = vec2<f32>(positions[offset], positions[offset + 1u]);
      let second = vec2<f32>(positions[offset + 2u], positions[offset + 3u]);
      output[outputOffset + index] = select(0u, 1u, lonLatIsAmbiguousGreatCircleEdge(first, second));`
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    expect(await readUint32(output, edges.length)).toEqual(
      edges.map(([first, second]) => Number(isAmbiguousGreatCircleEdge(first, second)))
    );
  } finally {
    compiled.destroy();
    fields.destroy();
    output.destroy();
  }
});
