// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import type {GPUSpatialContext} from '../../src/analysis-core/contracts';
import {
  GPUPointSpatialQuery,
  GPUPairwisePointInPolygon,
  GPUPairwisePointLinestringNearest
} from '../../src/geospatial';
import {createNullWebGPUDevice} from '../utils/gpu-contributor-test-utils';

function createOperator(
  side: string,
  spatialContext: GPUSpatialContext,
  secondary?: GPUSpatialContext
) {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  let serial = 0;
  const view = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `view-${serial++}`, format, length);
  const common = {id: 'query', spatialContext};
  if (side === 'polygon') {
    return new GPUPointSpatialQuery({
      ...common,
      polygonSpatialContext: secondary,
      kind: 'polygon',
      positions: view('float32x2', 1),
      parameters: view('float32', 4),
      polygon: {positions: view('float32x2', 4), ringOffsets: view('uint32', 2)},
      output: {
        ids: view('uint32', 1),
        status: {count: view('uint32', 1), overflow: view('uint32', 1)}
      }
    });
  }
  const rows = {
    queryPositions: view('float32x2', 1),
    positions: view('float32x2', 4),
    featureOffsets: view('uint32', 2)
  };
  if (side === 'polygons') {
    return new GPUPairwisePointInPolygon({
      ...common,
      ...rows,
      polygonsSpatialContext: secondary,
      polygonOffsets: view('uint32', 2),
      ringOffsets: view('uint32', 2),
      output: view('uint32', 1)
    });
  }
  return new GPUPairwisePointLinestringNearest({
    ...common,
    ...rows,
    linestringsSpatialContext: secondary,
    lineOffsets: view('uint32', 2),
    output: {distances: view('float32', 1)}
  });
}

for (const side of ['polygon', 'polygons', 'linestrings']) {
  const primary: GPUSpatialContext = {
    coordinateSpace: side === 'linestrings' ? 'planar' : 'longitude-latitude',
    metric: side === 'linestrings' ? 'native' : 'none',
    datum: 'EPSG:4326'
  };
  it(`${side} accepts absent and matching secondary contexts`, () => {
    expect(createOperator(side, primary).spatialContext).toBe(primary);
    expect(createOperator(side, primary, {...primary}).spatialContext).toBe(primary);
  });
  it(`${side} rejects a secondary datum mismatch`, () => {
    expect(() => createOperator(side, primary, {...primary, datum: 'EPSG:4267'})).toThrowError(
      new Error(
        `query ${side}SpatialContext incompatible spatial contexts: datums differ (EPSG:4267 and EPSG:4326)`
      )
    );
  });
  if (side !== 'linestrings') {
    it(`${side} requires linear secondary edges`, () => {
      expect(() =>
        createOperator(side, primary, {...primary, edgeModel: 'great-circle'})
      ).toThrowError(
        new Error(
          `query ${side}SpatialContext needs linear edges; prepare polygons with GPULonLatPolygonPreparation`
        )
      );
    });
  }
}
