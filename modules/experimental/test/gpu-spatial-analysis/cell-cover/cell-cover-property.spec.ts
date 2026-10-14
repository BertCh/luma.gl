// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {GPUCellCover} from '../../../src/gpu-spatial-analysis/cell-cover';
import {joinCellKey} from '../cell-aggregation/cell-aggregation-oracle';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import fixtureJson from './fixtures/cell-cover-h3.json';
import {requireTestDevice} from '../../utils/require-test-device';

// Oracle: fixtures/cell-cover-h3.json, written by test/oracles/geometry-properties.py with
// h3.polygon_to_cells (h3-py, center containment). Eight float32 star polygons at resolutions 3-5,
// three of them between latitude 60 and 80 degrees. The generator keeps only polygons whose nearby
// cell centers are at least 1e-4 degrees from the boundary (see the fixture's derivation), so the
// comparison is an exact set equality per polygon.

type CoverCase = {
  resolution: number;
  center: [number, number];
  ring: [number, number][];
  cells: string[];
  minimumCenterDistance: number;
};

const fixture: {tolerances: {centerMargin: number}; cases: CoverCase[]} =
  fixtureJson as unknown as {tolerances: {centerMargin: number}; cases: CoverCase[]};

async function runCover(device: Device, testCase: CoverCase) {
  const buffers: Buffer[] = [];
  const input = (values: Float32Array | Uint32Array) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return buffer;
  };
  const output = (length: number) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return buffer;
  };
  const outputCapacity = testCase.cells.length + 256;
  const out = {
    featureIds: output(outputCapacity),
    cells: output(2 * outputCapacity),
    core: output(outputCapacity),
    count: output(1),
    overflow: output(1),
    candidateOverflow: output(1),
    total: output(1)
  };
  const graph = new GPUCommandGraph(device, {id: 'cell-cover-property'});
  const positions = Float32Array.from(testCase.ring.flat());
  graph.add(
    new GPUCellCover({
      family: 'h3',
      resolution: testCase.resolution,
      containment: 'center',
      positions: importGraphBuffer(
        graph,
        'positions',
        input(positions),
        'float32x2',
        positions.length / 2
      ),
      featureOffsets: importGraphBuffer(
        graph,
        'feature-offsets',
        input(Uint32Array.of(0, 1)),
        'uint32',
        2
      ),
      polygonOffsets: importGraphBuffer(
        graph,
        'polygon-offsets',
        input(Uint32Array.of(0, 1)),
        'uint32',
        2
      ),
      ringOffsets: importGraphBuffer(
        graph,
        'ring-offsets',
        input(Uint32Array.of(0, positions.length / 2)),
        'uint32',
        2
      ),
      candidateCapacity: 1_000_000,
      output: {
        featureIds: importGraphBuffer(graph, 'out-ids', out.featureIds, 'uint32', outputCapacity),
        cells: importGraphBuffer(graph, 'out-cells', out.cells, 'uint32x2', outputCapacity),
        core: importGraphBuffer(graph, 'out-core', out.core, 'uint32', outputCapacity),
        status: {
          count: importGraphBuffer(graph, 'out-count', out.count, 'uint32', 1),
          requiredCount: importGraphBuffer(graph, 'out-total', out.total, 'uint32', 1),
          overflow: importGraphBuffer(graph, 'out-overflow', out.overflow, 'uint32', 1),
          candidateOverflow: importGraphBuffer(
            graph,
            'out-candidate-overflow',
            out.candidateOverflow,
            'uint32',
            1
          )
        }
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const count = (await readUint32(out.count, 1))[0];
  const words = await readUint32(out.cells, 2 * outputCapacity);
  const result = {
    count,
    overflow: (await readUint32(out.overflow, 1))[0],
    candidateOverflow: (await readUint32(out.candidateOverflow, 1))[0],
    required: (await readUint32(out.total, 1))[0],
    cells: Array.from({length: count}, (_, row) =>
      joinCellKey(words[2 * row], words[2 * row + 1]).toString(16)
    )
  };
  compiled.destroy();
  for (const buffer of buffers) {
    buffer.destroy();
  }
  return result;
}

it('GPUCellCover h3 center equals h3.polygon_to_cells on random polygons, including 60-80 degrees', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  expect(fixture.cases.filter(testCase => testCase.center[1] >= 60).length).toBeGreaterThanOrEqual(
    3
  );
  for (const [index, testCase] of fixture.cases.entries()) {
    const label = `case ${index} res ${testCase.resolution} lat ${testCase.center[1].toFixed(1)}`;
    expect(testCase.minimumCenterDistance, `${label} margin`).toBeGreaterThanOrEqual(
      fixture.tolerances.centerMargin
    );
    const actual = await runCover(device, testCase);
    expect(actual.overflow, `${label} overflow`).toBe(0);
    expect(actual.candidateOverflow, `${label} candidate overflow`).toBe(0);
    const unique = new Set(actual.cells);
    expect(unique.size, `${label} unique cells`).toBe(actual.cells.length);
    const expected = new Set(testCase.cells);
    expect(
      {
        missing: testCase.cells.filter(cell => !unique.has(cell)),
        extra: actual.cells.filter(cell => !expected.has(cell))
      },
      `${label} set difference`
    ).toEqual({missing: [], extra: []});
    expect(actual.required, `${label} required`).toBe(testCase.cells.length);
  }
  device.destroy?.();
});
