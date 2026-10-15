// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  describeNetworkElement,
  recordNetworkElement,
  type NetworkElementScenario
} from './network-element-lens';
import {runExample} from './network-noding.example';
import {
  CONSTRUCTION_DATASETS,
  LOOP_POLYLINES,
  createLoopFrame,
  getKeySpaces,
  sliceCapture
} from './construction-lens-support';

const scenarioId = 'gpu-network.network-noding';
/** Snap tolerance in metres: piece end points in the same 0.5 m cell become one node. */
const TOLERANCE_METERS = 0.5;

const scenario: NetworkElementScenario = {
  id: scenarioId,
  weights: 'lengthMeters',
  withoutRoads: true,
  coordinateModel: 'projected',
  seed: 0,
  datasetVersion: `${CONSTRUCTION_DATASETS.polylines.id}-v1`,
  retain: [
    'pieceVertices',
    'pieceOffsets',
    'pieceSourceIds',
    'pieceCount',
    'pieceOverflow',
    'nodingPositions',
    'nodeCount',
    'edgeFrom',
    'edgeTo',
    'edgeLengths',
    'nodingOverflow'
  ],
  formulas: [],
  keySpaces: getKeySpaces([
    {id: 'line', label: 'raw centreline', size: LOOP_POLYLINES.lineCount},
    {id: 'piece', label: 'noded edge piece', size: 512},
    {id: 'node', label: 'noded node', size: 512}
  ]),
  entityKeySpace: 'piece',
  choices: [
    {
      stage: 'representation',
      name: 'toleranceMeters',
      value: TOLERANCE_METERS,
      alternatives: [0, 2],
      sweepable: true
    },
    {
      stage: 'representation',
      name: 'frame',
      value: 'local azimuthal-equidistant metres at the Loop centre',
      alternatives: [],
      sweepable: false
    }
  ],
  async build(context) {
    const frame = createLoopFrame();
    const linePositions = context.upload(
      'linePositions',
      frame.projectArray(LOOP_POLYLINES.positions),
      'float32x2'
    );
    const lineOffsets = context.upload(
      'lineOffsets',
      Uint32Array.from(LOOP_POLYLINES.lineOffsets),
      'uint32'
    );
    const tolerance = context.upload('tolerance', Float32Array.of(TOLERANCE_METERS), 'float32');
    const example = await runExample(context.device, {
      graph: context.graph,
      linePositions,
      lineOffsets,
      tolerance
    });
    const {pieces, nodes, edges} = example.output;
    return {
      views: {
        linePositions,
        lineOffsets,
        pieceVertices: pieces.geometry.positions,
        pieceOffsets: pieces.geometry.lineOffsets,
        pieceSourceIds: pieces.sourceIds,
        pieceCount: pieces.status.count,
        pieceOverflow: pieces.status.overflow!,
        nodingPositions: nodes.positions,
        nodeCount: nodes.status.count,
        edgeFrom: edges.fromNodes,
        edgeTo: edges.toNodes,
        edgeLengths: edges.lengths,
        nodingOverflow: example.output.status.overflow
      }
    };
  },
  capture(final) {
    if (final('pieceOverflow').values[0] !== 0 || final('nodingOverflow').values[0] !== 0)
      throw new Error(
        'Network noding overflowed a capacity (pieces, vertices, intersections or nodes)'
      );
    const pieceCount = final('pieceCount').values[0];
    const nodeCount = final('nodeCount').values[0];
    const pieceOffsets = sliceCapture(final('pieceOffsets'), 0, pieceCount + 1, {
      id: 'pieceOffsets',
      meaning:
        'first vertex of each noded piece, plus the end (GPU offsets up to the GPU piece count)'
    });
    const vertexCount = pieceOffsets.values[pieceCount];
    return [
      {
        ...final('linePositions'),
        keySpace: undefined,
        role: 'input',
        meaning:
          'input: raw road centrelines projected to local metres (16 streets, junction vertices removed on straight stretches); grade separations are not marked',
        geometry: {
          kind: 'lines',
          coordinateModel: 'projected',
          unit: 'm',
          offsetsCapture: 'lineOffsets'
        }
      },
      final('lineOffsets'),
      pieceOffsets,
      {
        ...sliceCapture(final('pieceVertices'), 0, vertexCount, {id: 'pieceVertices'}),
        role: 'output',
        meaning:
          'output: noded edge pieces, split at every intersection found on the GPU (local metres)',
        geometry: {
          kind: 'lines',
          coordinateModel: 'projected',
          unit: 'm',
          offsetsCapture: 'pieceOffsets'
        }
      },
      {
        ...sliceCapture(final('nodingPositions'), 0, nodeCount, {id: 'nodingPositions'}),
        keySpace: 'node',
        role: 'output',
        meaning: 'output: noded nodes, one per tolerance cell of piece end points (local metres)',
        geometry: {kind: 'points', coordinateModel: 'projected', unit: 'm'}
      },
      sliceCapture(final('pieceSourceIds'), 0, pieceCount, {
        id: 'pieceSourceIds',
        keySpace: 'piece',
        meaning: 'source centreline of each noded piece'
      }),
      sliceCapture(final('edgeFrom'), 0, pieceCount, {
        id: 'edgeFrom',
        keySpace: 'piece',
        meaning: 'node at the start of each piece'
      }),
      sliceCapture(final('edgeTo'), 0, pieceCount, {
        id: 'edgeTo',
        keySpace: 'piece',
        meaning: 'node at the end of each piece'
      }),
      sliceCapture(final('edgeLengths'), 0, pieceCount, {
        id: 'edgeLengths',
        keySpace: 'piece',
        meaning: 'length of each piece along its vertices, in metres'
      }),
      final('pieceCount'),
      final('nodeCount')
    ];
  }
};

/** Executes the line noding on raw Chicago Loop centrelines and reads the pieces back. */
export async function run(device: Device) {
  return recordNetworkElement(device, scenario);
}

/** Builds the same graph on a null device, without execution. */
export async function describeStructure(device: Device) {
  return describeNetworkElement(device, scenario);
}
