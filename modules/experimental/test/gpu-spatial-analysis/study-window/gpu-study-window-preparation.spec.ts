// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {createTransientView, GPUCommandGraph, type GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUStudyWindowPreparation} from '../../../src/gpu-spatial-analysis/study-window';
import {GPU_STUDY_WINDOW_SUMMARY} from '../../../src/analysis-core/contracts';
import {
  GPUPolygonRasterization,
  GPU_POLYGON_RASTERIZATION_NO_ZONE
} from '../../../src/gpu-raster/polygon-rasterization';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  getStudyWindowPackedLayout,
  STUDY_WINDOW_BOUNDARY_CELL_BIT
} from '../../../src/utils/study-window-wgsl';
import {createWGSLKernelNode} from '../../../src/utils/wgsl-kernel-nodes';
import {createPolygons} from '../../gpu-raster/polygon-rasterization/polygon-rasterization-oracle';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {
  addOracleDegenerateSlots,
  ORACLE_L_SHAPE,
  getOracleWindowBounds,
  getOracleWindowArea,
  getOracleWindowPerimeter,
  getOracleOrientedEdges,
  type OracleStudyWindow
} from '../../analysis-core/study-window-oracle';
import {
  GPURectangleClip,
  getGPURectangleClipParameterValues
} from '../../../src/gpu-spatial-analysis/rectangle-clip';
import {GPURipley} from '../../../src/gpu-dataframe/pair-statistics/gpu-ripley';
import {
  getGPURipleyParameterValues,
  GPU_RIPLEY_PARAMETER_LENGTH
} from '../../../src/gpu-dataframe/pair-statistics/ripley-parameters';
import {createPairStatisticsWindowHarness} from '../../gpu-dataframe/pair-statistics/pair-statistics-harness';

type Point = [number, number];
const square: Point[] = [
  [0, 0],
  [1, 0],
  [1, 1],
  [0, 1]
];
const shell: Point[] = [
  [0, 0],
  [0, 4],
  [4, 4],
  [4, 0]
];
const hole: Point[] = [
  [1, 1],
  [3, 1],
  [3, 3],
  [1, 3]
];
const island: Point[] = [
  [2, 0],
  [3, 0],
  [3, 1],
  [2, 1]
];

for (const scene of [
  {
    name: 'vertices outside rings',
    polygons: {
      positions: new Float32Array([99, 99, ...square.flat(), -99, -99]),
      featureOffsets: new Uint32Array([0, 1]),
      polygonOffsets: new Uint32Array([0, 1]),
      ringOffsets: new Uint32Array([1, 5])
    },
    bounds: [0, 0, 1, 1],
    area: 1,
    perimeter: 4,
    ringSigns: [1]
  },
  {
    name: 'zero-area ring',
    polygons: createPolygons([
      [
        [
          [
            [0, 0],
            [1, 0],
            [2, 0]
          ]
        ]
      ]
    ]),
    bounds: [NaN, NaN, NaN, NaN],
    area: 0,
    perimeter: 0,
    ringSigns: [],
    flags: 2
  },
  {
    name: 'unit square',
    polygons: createPolygons([[[square]]]),
    bounds: [0, 0, 1, 1],
    area: 1,
    perimeter: 4,
    ringSigns: [1]
  },
  {
    name: 'CW shell and CCW hole',
    polygons: createPolygons([[[shell, hole]]]),
    bounds: [0, 0, 4, 4],
    area: 12,
    perimeter: 24,
    ringSigns: [1, -1]
  },
  {
    name: 'crossing overflow recovery',
    polygons: createPolygons([[[shell, hole]]]),
    bounds: [0, 0, 4, 4],
    area: 12,
    perimeter: 24,
    ringSigns: [1, -1]
  },
  {
    name: 'two islands',
    polygons: createPolygons([[[square], [island]]]),
    bounds: [0, 0, 3, 1],
    area: 2,
    perimeter: 8,
    ringSigns: [1, 1]
  },
  {
    name: 'translated square',
    polygons: createPolygons([[[square.map(([x, y]): Point => [x + 1000000, y + 1000000])]]]),
    bounds: [1000000, 1000000, 1000001, 1000001],
    area: 1,
    perimeter: 4,
    ringSigns: [1]
  }
]) {
  it(`GPUStudyWindowPreparation measures, orients, samples and packs ${scene.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const graph = new GPUCommandGraph(device);
    const owned: Buffer[] = [];
    const input = <Format extends 'uint32' | 'float32' | 'float32x2'>(
      name: string,
      values: Float32Array | Uint32Array,
      format: Format,
      length = values.length
    ) => {
      const buffer = createInputBuffer(device, values);
      owned.push(buffer);
      return importGraphBuffer(graph, name, buffer, format, length);
    };
    const output = (name: string, length: number) => {
      const buffer = createOutputBuffer(device, length);
      owned.push(buffer);
      return {buffer, view: importGraphBuffer(graph, name, buffer, 'uint32', length)};
    };
    const geometry = {
      positions: input(
        'positions',
        scene.polygons.positions,
        'float32x2',
        scene.polygons.positions.length / 2
      ),
      featureOffsets: input('features', scene.polygons.featureOffsets, 'uint32'),
      polygonOffsets: input('polygons', scene.polygons.polygonOffsets, 'uint32'),
      ringOffsets: input('rings', scene.polygons.ringOffsets, 'uint32')
    };
    const summaryBuffer = createOutputBuffer(device, 8);
    owned.push(summaryBuffer);
    const producer = new GPUStudyWindowPreparation({
      ...geometry,
      samplingGrid: [16, 16],
      crossingCapacity: scene.name === 'crossing overflow recovery' ? 2 : undefined,
      output: {
        summary: importGraphBuffer(graph, 'summary-copy', summaryBuffer, 'float32', 8)
      }
    });
    graph.add(producer);
    const views = producer.getViews(graph);
    const snapshot = (name: string, source: GraphDataView, words: number) => {
      const destination = output(`read-${name}`, words);
      graph.add(
        createWGSLKernelNode(graph, {
          id: `snapshot-${name}`,
          operation: 'TestStudyWindow',
          invocationCount: words,
          bindings: [
            {name: 'source', view: source, type: 'u32', access: 'read'},
            {name: 'destination', view: destination.view, type: 'u32', access: 'read_write'}
          ],
          body: 'destination[destinationOffset + index] = source[sourceOffset + index];'
        })
      );
      return destination.buffer;
    };
    const sources = {
      summary: snapshot('summary', views.summary, 8),
      edges: snapshot('edges', views.edges, views.edges.length),
      slabParams: snapshot('slabParams', views.slabs.params, views.slabs.params.length),
      slabOffsets: snapshot('slabOffsets', views.slabs.slabOffsets, views.slabs.slabOffsets.length),
      slabEntries: snapshot(
        'slabEntries',
        views.slabs.slabEntries,
        2 * views.slabs.slabEntries.length
      ),
      sampleCellCount: snapshot('sampleCellCount', views.sampleCellCount, 1),
      sampleCells: snapshot('sampleCells', views.sampleCells, 256)
    };
    const packedBuffer = snapshot('packed', views.packed, views.packed.length);
    const zones = output('reference-zones', 256);
    const boundary = output('reference-boundary', 256);
    const overflow = output('reference-overflow', 1);
    const [minimumX, minimumY, maximumX, maximumY] = scene.bounds;
    graph.add(
      new GPUPolygonRasterization({
        ...geometry,
        id: 'reference-raster',
        crossingCapacity: 256,
        grid: {width: 16, height: 16, affine: [1, 0, 0, 0, 1, 0], pixelInterpretation: 'area'},
        parameters: input(
          'reference-parameters',
          new Float32Array([
            minimumX,
            minimumY,
            (maximumX - minimumX) / 16,
            (maximumY - minimumY) / 16
          ]),
          'float32'
        ),
        output: {zones: zones.view, boundary: boundary.view, status: {overflow: overflow.view}}
      })
    );
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      const summary = await readFloat32(sources.summary, 8);
      expect(summary.slice(0, 4)).toEqual(scene.bounds);
      expect(
        Math.abs(summary[GPU_STUDY_WINDOW_SUMMARY.area] - scene.area) / Math.max(1, scene.area)
      ).toBeLessThanOrEqual(1e-6);
      expect(
        Math.abs(summary[GPU_STUDY_WINDOW_SUMMARY.perimeter] - scene.perimeter) /
          Math.max(1, scene.perimeter)
      ).toBeLessThanOrEqual(1e-6);
      expect(summary[GPU_STUDY_WINDOW_SUMMARY.edgeCount]).toBe(
        scene.name === 'zero-area ring'
          ? 0
          : scene.polygons.ringOffsets.at(-1)! - scene.polygons.ringOffsets[0]
      );
      expect(summary[GPU_STUDY_WINDOW_SUMMARY.flags]).toBe('flags' in scene ? scene.flags : 0);
      expect(await readFloat32(summaryBuffer, 8)).toEqual(summary);
      const edges = await readFloat32(sources.edges, views.edges.length);
      if (scene.name === 'zero-area ring') {
        expect(edges.every(Number.isNaN)).toBe(true);
      }
      if (scene.name === 'vertices outside rings') {
        expect(edges.slice(0, 4).every(Number.isNaN)).toBe(true);
        expect(edges.slice(-4).every(Number.isNaN)).toBe(true);
      }
      let totalSignedArea = 0;
      for (let ring = 0; ring < scene.ringSigns.length; ring++) {
        let signedArea = 0;
        for (
          let vertex = scene.polygons.ringOffsets[ring];
          vertex < scene.polygons.ringOffsets[ring + 1];
          vertex++
        ) {
          const [startX, startY, endX, endY] = edges.slice(4 * vertex, 4 * vertex + 4);
          signedArea += (startX * endY - startY * endX) / 2;
          const next =
            vertex + 1 === scene.polygons.ringOffsets[ring + 1]
              ? scene.polygons.ringOffsets[ring]
              : vertex + 1;
          const original = Array.from(scene.polygons.positions.slice(2 * vertex, 2 * vertex + 2));
          const following = Array.from(scene.polygons.positions.slice(2 * next, 2 * next + 2));
          expect([
            [...original, ...following],
            [...following, ...original]
          ]).toContainEqual([startX, startY, endX, endY]);
        }
        expect(Math.sign(signedArea)).toBe(scene.ringSigns[ring]);
        totalSignedArea += signedArea;
      }
      expect(totalSignedArea).toBe(scene.area);
      const [count] = await readUint32(sources.sampleCellCount, 1);
      const cells = await readUint32(sources.sampleCells, 256);
      const referenceZones = await readUint32(zones.buffer, 256);
      const referenceBoundary = await readUint32(boundary.buffer, 256);
      expect(await readUint32(overflow.buffer, 1)).toEqual([0]);
      const expectedCells = referenceZones.flatMap((zone, cell) =>
        zone !== GPU_POLYGON_RASTERIZATION_NO_ZONE || referenceBoundary[cell]
          ? [(cell | (referenceBoundary[cell] ? STUDY_WINDOW_BOUNDARY_CELL_BIT : 0)) >>> 0]
          : []
      );
      expect(count).toBe(expectedCells.length);
      expect(cells.slice(0, count)).toEqual(expectedCells);
      expect(cells.slice(count)).toEqual(new Array(256 - count).fill(0));
      const packed = await readUint32(packedBuffer, views.packed.length);
      const layout = getStudyWindowPackedLayout({
        vertexCount: geometry.positions.length,
        slabOffsetCount: views.slabs.slabOffsets.length,
        slabEntryCount: views.slabs.slabEntries.length,
        sampleCellCapacity: 256
      });
      const lengths = {
        summary: 8,
        edges: views.edges.length,
        slabParams: views.slabs.params.length,
        slabOffsets: views.slabs.slabOffsets.length,
        slabEntries: 2 * views.slabs.slabEntries.length,
        sampleCellCount: 1,
        sampleCells: 256
      };
      for (const name of Object.keys(sources) as (keyof typeof sources)[]) {
        expect(packed.slice(layout[name], layout[name] + lengths[name])).toEqual(
          await readUint32(sources[name], lengths[name])
        );
      }
      // Re-encoding reuses imported storage without rebuilding the preparation.
      submitGraph(device, compiled, undefined);
      expect(producer.encodedBuildCount).toBe(1);
      expect(await readFloat32(summaryBuffer, 8)).toEqual(summary);
      expect(await readUint32(sources.sampleCells, 256)).toEqual(cells);
    } finally {
      compiled.destroy();
      producer.destroy();
      owned.forEach(buffer => buffer.destroy());
    }
  });
}

const lShapeWithHole: OracleStudyWindow = {
  positions: new Float64Array([
    ...ORACLE_L_SHAPE.positions,
    150,
    150,
    350,
    150,
    350,
    350,
    150,
    350
  ]),
  featureOffsets: new Uint32Array([0, 1]),
  polygonOffsets: new Uint32Array([0, 2]),
  ringOffsets: new Uint32Array([0, 6, 10])
};

/** Imports geometry and tracks caller buffers independently of the preparation's owned storage. */
function createWindowFixture(device: Device, window: OracleStudyWindow) {
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const input = <Format extends 'float32x2' | 'float32' | 'uint32'>(
    name: string,
    data: Float32Array | Uint32Array,
    format: Format,
    length = data.length
  ) => {
    // Readable: the summary copy and clip status are read back, and positions are rewritten.
    const buffer = device.createBuffer({
      data,
      usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
    });
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, name, buffer, format, length)};
  };
  const positions = input(
    'vertices',
    Float32Array.from(window.positions),
    'float32x2',
    window.positions.length / 2
  );
  const summary = input('summary-copy', new Float32Array(8), 'float32');
  const geometry = {
    positions: positions.view,
    featureOffsets: input(
      'features',
      window.featureOffsets ?? new Uint32Array([0, window.polygonOffsets.length - 1]),
      'uint32'
    ).view,
    polygonOffsets: input('polygons', window.polygonOffsets, 'uint32').view,
    ringOffsets: input('rings', window.ringOffsets, 'uint32').view
  };
  return {graph, buffers, input, positions, summary, geometry};
}

for (const mode of ['invalidate', 'rebuildWhen'] as const) {
  it(`GPUStudyWindowPreparation persists the previous build until ${mode}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = createWindowFixture(device, lShapeWithHole);
    let changed = false;
    const preparation = new GPUStudyWindowPreparation({
      ...fixture.geometry,
      samplingGrid: [8, 8],
      output: {summary: fixture.summary.view},
      rebuildWhen: mode === 'rebuildWhen' ? () => changed : undefined
    });
    fixture.graph.add(preparation);
    const compiled = fixture.graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      const initial = await readFloat32(fixture.summary.buffer, 8);
      expect(preparation.encodedBuildCount).toBe(1);
      fixture.positions.buffer.write(
        Float32Array.from(lShapeWithHole.positions, value => value * 2)
      );
      submitGraph(device, compiled, undefined);
      expect(preparation.encodedBuildCount).toBe(1);
      expect(await readFloat32(fixture.summary.buffer, 8)).toEqual(initial);
      if (mode === 'invalidate') {
        preparation.invalidate();
      } else {
        changed = true;
      }
      submitGraph(device, compiled, undefined);
      expect(preparation.encodedBuildCount).toBe(2);
      const rebuilt = await readFloat32(fixture.summary.buffer, 8);
      expect(rebuilt).toEqual([
        ...initial.slice(0, 4).map(value => value * 2),
        initial[4] * 4,
        initial[5] * 2,
        initial[6],
        initial[7]
      ]);
      changed = false;
      submitGraph(device, compiled, undefined);
      expect(preparation.encodedBuildCount).toBe(2);
      expect(await readFloat32(fixture.summary.buffer, 8)).toEqual(rebuilt);
    } finally {
      compiled.destroy();
      preparation.destroy();
      fixture.buffers.forEach(buffer => buffer.destroy());
    }
  });
}

it('GPUStudyWindowPreparation ignores clip-shaped degenerate slots in summaries and isotropic K', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const summaries: number[][] = [];
  const results: number[][] = [];
  for (const window of [lShapeWithHole, addOracleDegenerateSlots(lShapeWithHole)]) {
    const fixture = createWindowFixture(device, window);
    const preparation = new GPUStudyWindowPreparation({
      ...fixture.geometry,
      samplingGrid: [8, 8],
      output: {summary: fixture.summary.view}
    });
    fixture.graph.add(preparation);
    const compiled = fixture.graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      summaries.push(await readFloat32(fixture.summary.buffer, 8));
    } finally {
      compiled.destroy();
      preparation.destroy();
      fixture.buffers.forEach(buffer => buffer.destroy());
    }
    const harness = createPairStatisticsWindowHarness(device, {
      window,
      positions: new Float32Array([
        80, 90, 430, 200, 550, 480, 900, 310, 1450, 260, 310, 900, 400, 1400, 250, 250, 1100, 1100
      ]),
      parameterLength: GPU_RIPLEY_PARAMETER_LENGTH,
      outputs: {k: {format: 'float32', length: 6}},
      createContributor: (views, studyWindow) =>
        new GPURipley({
          positions: views.positions,
          parameters: views.parameters,
          studyWindow,
          radiusCount: 6,
          gridSize: [8, 8],
          output: {k: views.outputs.k as GraphDataView<'float32'>}
        })
    });
    try {
      results.push(
        (
          await harness.run(
            getGPURipleyParameterValues({
              bounds: 'study-window',
              maximumDistance: 613,
              edgeCorrection: 'isotropic'
            })
          )
        ).k
      );
    } finally {
      harness.destroy();
    }
  }
  expect(summaries[1]).toEqual(summaries[0]);
  for (let index = 0; index < results[0].length; index++) {
    expect(Math.abs(results[1][index] - results[0][index])).toBeLessThanOrEqual(
      1e-6 * Math.abs(results[0][index])
    );
  }
});

it('GPUStudyWindowPreparation consumes real rectangle-clip output with vanished rings', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createWindowFixture(device, lShapeWithHole);
  const {graph} = fixture;
  const clippedPositions = createTransientView(graph, 'clipped-positions', 'float32x2', 64);
  const clippedRings = createTransientView(graph, 'clipped-rings', 'uint32', 3);
  const overflow = fixture.input('overflow', new Uint32Array(1), 'uint32');
  graph.add(
    new GPURectangleClip({
      ...fixture.geometry,
      geometryType: 'polygons',
      parameters: fixture.input(
        'clip-parameters',
        getGPURectangleClipParameterValues({minX: 500, minY: 0, maxX: 1800, maxY: 1500}),
        'float32'
      ).view,
      output: {
        positions: clippedPositions,
        lineOffsets: clippedRings,
        status: {
          count: createTransientView(graph, 'clip-count', 'uint32', 1),
          overflow: overflow.view
        }
      }
    })
  );
  const preparation = new GPUStudyWindowPreparation({
    ...fixture.geometry,
    positions: clippedPositions,
    ringOffsets: clippedRings,
    samplingGrid: [8, 8],
    output: {summary: fixture.summary.view}
  });
  graph.add(preparation);
  const compiled = graph.compile();
  const expected: OracleStudyWindow = {
    positions: new Float64Array([500, 0, 1800, 0, 1800, 700, 700, 700, 700, 1500, 500, 1500]),
    polygonOffsets: new Uint32Array([0, 1]),
    ringOffsets: new Uint32Array([0, 6])
  };
  try {
    submitGraph(device, compiled, undefined);
    expect(await readUint32(overflow.buffer, 1)).toEqual([0]);
    expect(await readFloat32(fixture.summary.buffer, 8)).toEqual([
      ...getOracleWindowBounds(expected),
      getOracleWindowArea(expected),
      getOracleWindowPerimeter(expected),
      getOracleOrientedEdges(expected).length,
      0
    ]);
  } finally {
    compiled.destroy();
    preparation.destroy();
    fixture.buffers.forEach(buffer => buffer.destroy());
  }
});
