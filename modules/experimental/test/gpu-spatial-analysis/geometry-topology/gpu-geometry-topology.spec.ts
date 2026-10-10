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
    polygonOffsets: (await readUint32(polygonOffsets.buffer, ringCapacity + 1)).slice(
      0,
      writtenPolygons + 1
    ),
    ringOffsets: (await readUint32(ringOffsets.buffer, ringCapacity + 1)).slice(
      0,
      writtenPolygons + 1
    ),
    diagnostics: await Promise.all(
      diagnosticScalars.map(async diagnostic => (await readUint32(diagnostic.buffer, 1))[0])
    ),
    overflow: (await readUint32(overflow.buffer, 1))[0]
  };
  compiled.destroy();
  for (const buffer of buffers) buffer.destroy();
  return result;
}

it('GPUPolygonize nodes crossings with edge and parameter provenance', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
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
  if (!device) return;

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
  if (!device) return;
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
});

it('GPUPolygonize extracts a bounded face and reports complete use', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
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
  if (!device) return;
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
  if (!device) return;
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
