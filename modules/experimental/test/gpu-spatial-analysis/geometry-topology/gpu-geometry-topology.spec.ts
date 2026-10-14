// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {
  GPUMakeValid,
  GPUPolygonize
} from '../../../src/gpu-spatial-analysis/geometry-topology/index';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {flattenLines, type Point} from '../line-split/line-split-harness';
import {requireTestDevice} from '../../utils/require-test-device';

type RunResult = {
  nodedCount: number;
  nodedRequiredCount: number;
  nodedOverflow: number;
  nodedCandidateOverflow: number;
  nodedEndpoints: number[];
  sourceEdgeIds: number[];
  startParameters: number[];
  endParameters: number[];
  polygonCount: number;
  polygonOffsets: number[];
  ringOffsets: number[];
  diagnostics: number[];
  overflow: number;
  requiredCount: number;
  candidateOverflow: number;
  polygonPositions: number[];
};

async function runPolygonize(
  device: Device,
  lines: readonly (readonly Point[])[],
  operation: 'polygonize' | 'make-valid' = 'polygonize',
  segmentCapacity = 64,
  intersectionCapacity = 256
): Promise<RunResult> {
  const graph = new GPUCommandGraph(device, {id: 'polygonize-test'});
  const buffers: Buffer[] = [];
  const arrays = flattenLines(lines);
  const input = (
    name: string,
    data: Float32Array | Uint32Array,
    format: 'float32x2' | 'uint32'
  ) => {
    const buffer = createInputBuffer(device, data);
    buffers.push(buffer);
    return importGraphBuffer(
      graph,
      name,
      buffer,
      format,
      format === 'uint32' ? data.length : data.length / 2
    ) as never;
  };
  const componentCount = {uint32: 1, float32: 1, float32x2: 2, float32x4: 4} as const;
  const output = <Format extends keyof typeof componentCount>(
    name: string,
    length: number,
    format: Format
  ) => {
    const buffer = createOutputBuffer(device, length * componentCount[format]);
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, name, buffer, format, length) as never};
  };
  const scalar = (name: string) => output(name, 1, 'uint32');
  const directedCapacity = 2 * segmentCapacity;
  const ringCapacity = 64;
  const vertexCapacity = 256;
  const nodedEndpoints = output('noded-endpoints', segmentCapacity, 'float32x4');
  const nodedSourceFeatures = output('noded-source-features', segmentCapacity, 'uint32');
  const nodedSourceRings = output('noded-source-rings', segmentCapacity, 'uint32');
  const nodedSourceEdges = output('noded-source-edges', segmentCapacity, 'uint32');
  const nodedStartParameters = output('noded-start-parameters', segmentCapacity, 'float32');
  const nodedEndParameters = output('noded-end-parameters', segmentCapacity, 'float32');
  const nodedCount = scalar('noded-count');
  const nodedRequired = scalar('noded-required');
  const nodedOverflow = scalar('noded-overflow');
  const nodedCandidateOverflow = scalar('noded-candidate-overflow');
  const directedEndpoints = output('directed-endpoints', directedCapacity, 'float32x4');
  const directedSourceSegments = output('directed-source-segments', directedCapacity, 'uint32');
  const directedSourceFeatures = output('directed-source-features', directedCapacity, 'uint32');
  const directedTwins = output('directed-twins', directedCapacity, 'uint32');
  const directedRings = output('directed-rings', directedCapacity, 'uint32');
  const directedFlags = output('directed-flags', directedCapacity, 'uint32');
  const directedCount = scalar('directed-count');
  const directedRequired = scalar('directed-required');
  const directedOverflow = scalar('directed-overflow');
  const directedCandidateOverflow = scalar('directed-candidate-overflow');
  const polygonPositions = output('polygon-positions', vertexCapacity, 'float32x2');
  const featureOffsets = output('feature-offsets', ringCapacity + 1, 'uint32');
  const polygonOffsets = output('polygon-offsets', ringCapacity + 1, 'uint32');
  const ringOffsets = output('ring-offsets', ringCapacity + 1, 'uint32');
  const polygonSourceIds = output('polygon-source-ids', ringCapacity, 'uint32');
  const edgeClasses = output('edge-classes', segmentCapacity, 'uint32');
  const diagnosticScalars = [
    scalar('closed-rings'),
    scalar('open-chains'),
    scalar('cut-edges'),
    scalar('dangles'),
    scalar('unused-edges')
  ];
  const count = scalar('count');
  const required = scalar('required');
  const overflow = scalar('overflow');
  const candidateOverflow = scalar('candidate-overflow');
  const inputPositions = input('positions', arrays.positions, 'float32x2');
  const inputLineOffsets = input('line-offsets', arrays.lineOffsets, 'uint32');
  const topologyOutput = {
    nodedSegments: {
      endpoints: nodedEndpoints.view,
      sourceFeatureIds: nodedSourceFeatures.view,
      sourceRingIds: nodedSourceRings.view,
      sourceEdgeIds: nodedSourceEdges.view,
      sourceStartParameters: nodedStartParameters.view,
      sourceEndParameters: nodedEndParameters.view,
      status: {
        count: nodedCount.view,
        requiredCount: nodedRequired.view,
        overflow: nodedOverflow.view,
        candidateOverflow: nodedCandidateOverflow.view
      }
    },
    directedEdges: {
      endpoints: directedEndpoints.view,
      sourceSegmentIds: directedSourceSegments.view,
      sourceFeatureIds: directedSourceFeatures.view,
      twinIds: directedTwins.view,
      ringIds: directedRings.view,
      flags: directedFlags.view,
      status: {
        count: directedCount.view,
        requiredCount: directedRequired.view,
        overflow: directedOverflow.view,
        candidateOverflow: directedCandidateOverflow.view
      }
    },
    polygons: {
      kind: 'polygons' as const,
      positions: polygonPositions.view,
      featureOffsets: featureOffsets.view,
      polygonOffsets: polygonOffsets.view,
      ringOffsets: ringOffsets.view,
      sourceIds: polygonSourceIds.view
    },
    diagnostics: {
      edgeClasses: edgeClasses.view,
      closedRingCount: diagnosticScalars[0].view,
      openChainCount: diagnosticScalars[1].view,
      cutEdgeCount: diagnosticScalars[2].view,
      dangleCount: diagnosticScalars[3].view,
      unusedEdgeCount: diagnosticScalars[4].view
    },
    status: {
      count: count.view,
      requiredCount: required.view,
      overflow: overflow.view,
      candidateOverflow: candidateOverflow.view
    }
  };
  const common = {
    intersectionCapacity,
    precision: {
      vertexTolerance: 1e-5,
      predicates: 'exact-or-uncertain' as const,
      storage: 'float32' as const
    },
    output: topologyOutput
  };
  graph.add(
    operation === 'make-valid'
      ? new GPUMakeValid({
          ...common,
          polygons: {
            kind: 'polygons',
            positions: inputPositions,
            ringOffsets: inputLineOffsets,
            polygonOffsets: input(
              'input-polygon-offsets',
              Uint32Array.of(0, lines.length),
              'uint32'
            ),
            featureOffsets: input('input-feature-offsets', Uint32Array.of(0, 1), 'uint32')
          }
        })
      : new GPUPolygonize({
          ...common,
          lines: {
            kind: 'lines',
            positions: inputPositions,
            lineOffsets: inputLineOffsets
          }
        })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const [writtenSegments] = await readUint32(nodedCount.buffer, 1);
  const [requiredSegments] = await readUint32(nodedRequired.buffer, 1);
  const [writtenPolygons] = await readUint32(count.buffer, 1);
  const polygonOffsetValues = await readUint32(polygonOffsets.buffer, ringCapacity + 1);
  const result: RunResult = {
    nodedCount: writtenSegments,
    nodedRequiredCount: requiredSegments,
    nodedOverflow: (await readUint32(nodedOverflow.buffer, 1))[0],
    nodedCandidateOverflow: (await readUint32(nodedCandidateOverflow.buffer, 1))[0],
    nodedEndpoints: (await readFloat32(nodedEndpoints.buffer, segmentCapacity * 4)).slice(
      0,
      writtenSegments * 4
    ),
    sourceEdgeIds: (await readUint32(nodedSourceEdges.buffer, segmentCapacity)).slice(
      0,
      writtenSegments
    ),
    startParameters: (await readFloat32(nodedStartParameters.buffer, segmentCapacity)).slice(
      0,
      writtenSegments
    ),
    endParameters: (await readFloat32(nodedEndParameters.buffer, segmentCapacity)).slice(
      0,
      writtenSegments
    ),
    polygonCount: writtenPolygons,
    polygonOffsets: polygonOffsetValues.slice(0, writtenPolygons + 1),
    ringOffsets: (await readUint32(ringOffsets.buffer, ringCapacity + 1)).slice(
      0,
      polygonOffsetValues[writtenPolygons] + 1
    ),
    diagnostics: await Promise.all(
      diagnosticScalars.map(async diagnostic => (await readUint32(diagnostic.buffer, 1))[0])
    ),
    overflow: (await readUint32(overflow.buffer, 1))[0],
    requiredCount: (await readUint32(required.buffer, 1))[0],
    candidateOverflow: (await readUint32(candidateOverflow.buffer, 1))[0],
    polygonPositions: await readFloat32(polygonPositions.buffer, vertexCapacity * 2)
  };
  compiled.destroy();
  for (const buffer of buffers) buffer.destroy();
  return result;
}

it('GPUPolygonize nodes crossings with edge and parameter provenance', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const result = await runPolygonize(device, [
    [
      [0, 0],
      [2, 2]
    ],
    [
      [0, 2],
      [2, 0]
    ]
  ]);
  expect(result.nodedCount).toBe(4);
  expect(result.sourceEdgeIds).toEqual([0, 0, 2, 2]);
  expect(result.startParameters).toEqual([0, 0.5, 0, 0.5]);
  expect(result.endParameters).toEqual([0.5, 1, 0.5, 1]);
  expect(result.polygonCount).toBe(0);
  expect(result.overflow).toBe(0);
});

it('GPUPolygonize handles endpoint touches, overlaps, and repeated vertices', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);

  const touch = await runPolygonize(device, [
    [
      [0, 0],
      [1, 0]
    ],
    [
      [1, 0],
      [1, 1]
    ]
  ]);
  expect(touch.nodedCount).toBe(2);
  expect(touch.polygonCount).toBe(0);
  expect(touch.overflow).toBe(0);

  const overlap = await runPolygonize(device, [
    [
      [0, 0],
      [2, 0]
    ],
    [
      [1, 0],
      [3, 0]
    ]
  ]);
  expect(overlap.nodedCount).toBe(4);
  expect(overlap.polygonCount).toBe(0);
  expect(overlap.overflow).toBe(0);

  const repeatedVertex = await runPolygonize(device, [
    [
      [0, 0],
      [1, 0],
      [1, 0],
      [1, 1],
      [0, 1],
      [0, 0]
    ]
  ]);
  expect(repeatedVertex.polygonCount).toBe(1);
  expect(repeatedVertex.overflow).toBe(0);
});

it('GPUPolygonize exposes exact required cardinality when noded output capacity is too small', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const result = await runPolygonize(
    device,
    [
      [
        [0, 0],
        [2, 2]
      ],
      [
        [0, 2],
        [2, 0]
      ]
    ],
    'polygonize',
    2,
    16
  );
  expect(result.nodedCount).toBe(2);
  expect(result.nodedRequiredCount).toBe(4);
  expect(result.nodedOverflow).toBe(1);
  expect(result.nodedCandidateOverflow).toBe(0);
  expect(result.overflow).toBe(1);
  // candidateOverflow is the noding candidate flag only; capacity overflow does not set it.
  expect(result.candidateOverflow).toBe(0);
});

it('GPUPolygonize publishes the ring-assembly requirement, not the written polygon count', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // A square with a hole polygonizes into the annulus ring, its hole ring, the island face and
  // the clockwise unbounded walk: more rings are required than polygons are written.
  const result = await runPolygonize(device, SQUARE_WITH_HOLE);
  expect(result.polygonCount).toBe(2);
  expect(result.requiredCount).toBeGreaterThan(result.polygonCount);
  expect(result.candidateOverflow).toBe(0);
});

it('GPUPolygonize extracts a bounded face and reports complete use', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const result = await runPolygonize(device, [
    [
      [0, 0],
      [1, 0]
    ],
    [
      [1, 0],
      [1, 1]
    ],
    [
      [1, 1],
      [0, 1]
    ],
    [
      [0, 1],
      [0, 0]
    ]
  ]);
  expect(result.nodedCount).toBe(4);
  expect(result.polygonCount).toBe(1);
  expect(result.polygonOffsets).toEqual([0, 1]);
  expect(result.ringOffsets[0]).toBe(0);
  expect(result.ringOffsets[1]).toBe(4);
  expect(result.diagnostics[0]).toBe(4);
  expect(result.diagnostics[4]).toBe(0);
  expect(result.overflow).toBe(0);
});

it('GPUPolygonize repairs a self-crossing bow tie into bounded faces', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const result = await runPolygonize(device, [
    [
      [0, 0],
      [2, 2],
      [0, 2],
      [2, 0],
      [0, 0]
    ]
  ]);
  expect(result.nodedCount).toBe(6);
  expect(result.polygonCount).toBe(2);
  expect(result.overflow).toBe(0);
});

it('GPUMakeValid rebuilds a self-crossing polygon boundary', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const result = await runPolygonize(
    device,
    [
      [
        [0, 0],
        [2, 2],
        [0, 2],
        [2, 0],
        [0, 0]
      ]
    ],
    'make-valid'
  );
  expect(result.polygonCount).toBe(2);
  expect(result.overflow).toBe(0);
});

type PolygonSummary = {area: number; ringCount: number};

/** Area (shell minus holes) and ring count of every written polygon, sorted by area then rings. */
function summarizePolygons(result: RunResult): PolygonSummary[] {
  const ringArea = (ring: number) => {
    const start = result.ringOffsets[ring];
    const end = result.ringOffsets[ring + 1];
    let twiceArea = 0;
    for (let i = start; i < end; i++) {
      const next = i + 1 < end ? i + 1 : start;
      twiceArea +=
        result.polygonPositions[2 * i] * result.polygonPositions[2 * next + 1] -
        result.polygonPositions[2 * next] * result.polygonPositions[2 * i + 1];
    }
    return Math.abs(twiceArea) / 2;
  };
  const summaries: PolygonSummary[] = [];
  for (let polygon = 0; polygon < result.polygonCount; polygon++) {
    const first = result.polygonOffsets[polygon];
    const end = result.polygonOffsets[polygon + 1];
    let area = ringArea(first);
    for (let ring = first + 1; ring < end; ring++) area -= ringArea(ring);
    summaries.push({area, ringCount: end - first});
  }
  return summaries.sort((a, b) => a.area - b.area || a.ringCount - b.ringCount);
}

function expectMakeValidMatches(result: RunResult, expected: readonly [number, number][]) {
  const summaries = summarizePolygons(result);
  expect(summaries.length).toBe(expected.length);
  for (let i = 0; i < expected.length; i++) {
    expect(summaries[i].area).toBeCloseTo(expected[i][0], 5);
    expect(summaries[i].ringCount).toBe(expected[i][1]);
  }
  expect(summaries.reduce((total, summary) => total + summary.area, 0)).toBeCloseTo(
    expected.reduce((total, [area]) => total + area, 0),
    5
  );
  expect(result.polygonOffsets[result.polygonCount]).toBe(
    expected.reduce((total, [, rings]) => total + rings, 0)
  );
  expect(result.overflow).toBe(0);
  expect(result.candidateOverflow).toBe(0);
}

// Expected (area, ring count) per output polygon, sorted by area: shapely 2.2.0
// `shapely.validation.make_valid` on the same rings, see
// scratchpad/fix/work/W2A/makevalid/oracle.py. The fill rule is source parity (even-odd).
const SQUARE_WITH_HOLE: Point[][] = [
  [
    [0, 0],
    [4, 0],
    [4, 4],
    [0, 4],
    [0, 0]
  ],
  [
    [1, 1],
    [1, 3],
    [3, 3],
    [3, 1],
    [1, 1]
  ]
];

it('GPUMakeValid keeps only the annulus of a valid polygon with a hole', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // make_valid: Polygon, area 12, shell + 1 hole. GPUPolygonize would add the island face.
  expectMakeValidMatches(await runPolygonize(device, SQUARE_WITH_HOLE, 'make-valid'), [[12, 2]]);
  const polygonized = await runPolygonize(device, SQUARE_WITH_HOLE);
  expect(polygonized.polygonCount).toBe(2);
});

it('GPUMakeValid keeps a polygon with two holes whole', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // make_valid: Polygon, area 24, shell + 2 holes.
  const result = await runPolygonize(
    device,
    [
      [
        [0, 0],
        [8, 0],
        [8, 4],
        [0, 4],
        [0, 0]
      ],
      [
        [1, 1],
        [1, 3],
        [3, 3],
        [3, 1],
        [1, 1]
      ],
      [
        [5, 1],
        [5, 3],
        [7, 3],
        [7, 1],
        [5, 1]
      ]
    ],
    'make-valid'
  );
  expectMakeValidMatches(result, [[24, 3]]);
});

it('GPUMakeValid splits a bow tie into two triangles', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // make_valid: MultiPolygon, two parts of area 1.
  const result = await runPolygonize(
    device,
    [
      [
        [0, 0],
        [2, 2],
        [0, 2],
        [2, 0],
        [0, 0]
      ]
    ],
    'make-valid'
  );
  expectMakeValidMatches(result, [
    [1, 1],
    [1, 1]
  ]);
});

it('GPUMakeValid drops the doubly covered lens of a hole ring crossing the shell', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // make_valid: MultiPolygon, parts of area 2 (the lobe outside the shell) and 14 (the shell
  // minus the lens); the lens covered by shell and hole ring has even parity and is dropped.
  const result = await runPolygonize(
    device,
    [
      [
        [0, 0],
        [4, 0],
        [4, 4],
        [0, 4],
        [0, 0]
      ],
      [
        [3, 1],
        [5, 1],
        [5, 3],
        [3, 3],
        [3, 1]
      ]
    ],
    'make-valid'
  );
  expectMakeValidMatches(result, [
    [2, 1],
    [14, 1]
  ]);
});

it('GPUMakeValid keeps an island inside a hole as its own polygon by parity', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // make_valid of MultiPolygon(lake with hole, island): parts of area 4 (island) and 64
  // (lake, shell + 1 hole), total 68. The island face is covered three times (odd).
  const result = await runPolygonize(
    device,
    [
      [
        [0, 0],
        [10, 0],
        [10, 10],
        [0, 10],
        [0, 0]
      ],
      [
        [2, 2],
        [2, 8],
        [8, 8],
        [8, 2],
        [2, 2]
      ],
      [
        [4, 4],
        [6, 4],
        [6, 6],
        [4, 6],
        [4, 4]
      ]
    ],
    'make-valid'
  );
  expectMakeValidMatches(result, [
    [4, 1],
    [64, 2]
  ]);
});
