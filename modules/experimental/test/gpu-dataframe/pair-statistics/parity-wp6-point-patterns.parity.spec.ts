// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {GPUPointPatternIndices} from '../../../src/gpu-dataframe/pair-statistics/gpu-point-pattern-indices';
import {getGPUPointPatternIndicesParameterValues} from '../../../src/gpu-dataframe/pair-statistics/point-pattern-indices-parameters';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {decodeOracleFixture} from '../../oracles/oracle-fixture';
import {compareParity} from '../../oracles/parity-wp6-compare';
import positionsJSON from '../../oracles/fixtures/parity-wp6-point-patterns-positions.json';
import distancesJSON from '../../oracles/fixtures/parity-wp6-point-patterns-distances.json';
import idsJSON from '../../oracles/fixtures/parity-wp6-point-patterns-ids.json';
import statisticsJSON from '../../oracles/fixtures/parity-wp6-point-patterns-statistics.json';

type PointCase = {
  name: string;
  positions: ArrayLike<number>;
  mask?: ArrayLike<number>;
  bounds: [number, number, number, number];
  quadratGrid: [number, number];
  nearestNeighborDistances: ArrayLike<number>;
  nearestNeighborIds: ArrayLike<number>;
  ambiguousFloat32Neighbors: {row: number; candidate: number; distanceGap: number}[];
  quadratCounts: number[];
  quadratAreas: number[];
  runs: {
    alternative: 'two-sided' | 'greater' | 'lesser';
    clarkEvans: number[];
    quadratStatistics: number[];
  }[];
};

/** Lossless LE payloads keep each full-data fixture below 1.5 MB; no output is summarized. */
function unpackArray(payload: {dtype: string; length: number; base64: string}): Float64Array {
  const bytes = Uint8Array.from(atob(payload.base64), character => character.charCodeAt(0));
  const view = new DataView(bytes.buffer);
  const stride = payload.dtype === '<f8' ? 8 : 4;
  return Float64Array.from({length: payload.length}, (_, index) =>
    payload.dtype === '<u4'
      ? view.getUint32(index * stride, true)
      : payload.dtype === '<f4'
        ? view.getFloat32(index * stride, true)
        : view.getFloat64(index * stride, true)
  );
}

const fixture = decodeOracleFixture<{
  real: Omit<PointCase, 'positions' | 'nearestNeighborDistances' | 'nearestNeighborIds'>;
  synthetic: PointCase[];
}>(statisticsJSON);
const scenes: PointCase[] = [
  {
    ...fixture.real,
    positions: unpackArray(positionsJSON.positions),
    nearestNeighborDistances: unpackArray(distancesJSON.nearestNeighborDistances),
    nearestNeighborIds: unpackArray(idsJSON.nearestNeighborIds)
  },
  ...fixture.synthetic
];
const clarkNames = [
  'count',
  'observedMean',
  'expectedMean',
  'ratio',
  'standardError',
  'zScore',
  'pValue'
];
const quadratNames = [
  'count',
  'mean',
  'variance',
  'varianceToMeanRatio',
  'chiSquare',
  'degreesOfFreedom',
  'pValue'
];

for (const scene of scenes) {
  it(`GPUPointPatternIndices external parity (rectangle/none): ${scene.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rowCount = scene.positions.length / 2;
    const quadratCount = scene.quadratGrid[0] * scene.quadratGrid[1];
    const graph = new GPUCommandGraph(device);
    const buffers: Buffer[] = [];
    const track = (buffer: Buffer) => {
      buffers.push(buffer);
      return buffer;
    };
    const parameters = new GPUParameterBuffer(device, {
      id: 'parity-wp6-point-parameters',
      format: 'float32',
      length: 8
    });
    const distances = track(createOutputBuffer(device, rowCount));
    const neighborIds = track(createOutputBuffer(device, rowCount));
    const clarkEvans = track(createOutputBuffer(device, 7));
    const counts = track(createOutputBuffer(device, quadratCount));
    const areas = track(createOutputBuffer(device, quadratCount));
    const quadratStatistics = track(createOutputBuffer(device, 7));
    graph.add(
      new GPUPointPatternIndices({
        positions: importGraphBuffer(
          graph,
          'positions',
          track(createInputBuffer(device, Float32Array.from(scene.positions))),
          'float32x2',
          rowCount
        ),
        mask: scene.mask
          ? importGraphBuffer(
              graph,
              'mask',
              track(createInputBuffer(device, Uint32Array.from(scene.mask))),
              'uint32',
              rowCount
            )
          : undefined,
        parameters: parameters.importToGraph(graph),
        gridSize: [256, 256],
        quadratGrid: scene.quadratGrid,
        output: {
          nearestNeighborDistances: importGraphBuffer(
            graph,
            'distances',
            distances,
            'float32',
            rowCount
          ),
          nearestNeighborIds: importGraphBuffer(graph, 'ids', neighborIds, 'uint32', rowCount),
          clarkEvans: importGraphBuffer(graph, 'clark', clarkEvans, 'float32', 7),
          quadratCounts: importGraphBuffer(graph, 'counts', counts, 'uint32', quadratCount),
          quadratAreas: importGraphBuffer(graph, 'areas', areas, 'float32', quadratCount),
          quadratStatistics: importGraphBuffer(graph, 'statistics', quadratStatistics, 'float32', 7)
        }
      })
    );
    const compiled = graph.compile();
    try {
      for (const run of scene.runs) {
        const caseName = `${scene.name}/none/${run.alternative}`;
        parameters.write(
          getGPUPointPatternIndicesParameterValues({
            bounds: scene.bounds,
            maximumDistance:
              Math.max(scene.bounds[2] - scene.bounds[0], scene.bounds[3] - scene.bounds[1]) / 256,
            clarkEvansCorrection: 'none',
            alternative: run.alternative
          })
        );
        submitGraph(device, compiled, undefined);
        compareParity(
          'GPUPointPatternIndices',
          caseName,
          'pointpats.PointPattern.nnd',
          rowCount,
          'nearestNeighborDistances',
          await readFloat32(distances, rowCount),
          scene.nearestNeighborDistances
        );
        const observedIds = await readUint32(neighborIds, rowCount);
        const alternatives = new Map(
          scene.ambiguousFloat32Neighbors.map(entry => [entry.row, entry.candidate])
        );
        // Every ID is checked: exact off enumerated f32 boundary ties, candidate membership on ties.
        const validCandidates = observedIds.map((neighborId, row) =>
          Number(
            neighborId === scene.nearestNeighborIds[row] || neighborId === alternatives.get(row)
          )
        );
        compareParity(
          'GPUPointPatternIndices',
          caseName,
          'scipy.spatial.cKDTree; lowest ID exact ties; separately enumerated f32 ambiguous candidates',
          rowCount,
          'nearestNeighborIds.referenceCandidateMembership',
          validCandidates,
          new Uint32Array(rowCount).fill(1),
          true
        );
        compareParity(
          'GPUPointPatternIndices',
          caseName,
          'numpy.histogram2d',
          rowCount,
          'quadratCounts',
          await readUint32(counts, quadratCount),
          scene.quadratCounts,
          true
        );
        compareParity(
          'GPUPointPatternIndices',
          caseName,
          'rectangle area / quadrat count',
          rowCount,
          'quadratAreas',
          await readFloat32(areas, quadratCount),
          scene.quadratAreas
        );
        const clarkValues = await readFloat32(clarkEvans, 7);
        const quadratValues = await readFloat32(quadratStatistics, 7);
        for (let index = 0; index < 7; index++) {
          compareParity(
            'GPUPointPatternIndices',
            caseName,
            'pointpats.PointPattern.mean_nnd; scipy.stats.rayleigh/ norm (Clark-Evans CSR)',
            rowCount,
            `clarkEvans.${clarkNames[index]}`,
            [clarkValues[index]],
            [run.clarkEvans[index]],
            index === 0
          );
          compareParity(
            'GPUPointPatternIndices',
            caseName,
            'numpy.mean/var(ddof=1); scipy.stats.chisquare/chi2',
            rowCount,
            `quadratStatistics.${quadratNames[index]}`,
            [quadratValues[index]],
            [run.quadratStatistics[index]],
            index === 0 || index === 5
          );
        }
      }
    } finally {
      compiled.destroy();
      parameters.destroy();
      for (const buffer of buffers) buffer.destroy();
    }
  }, 120000);
}
