// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import type {GPUSpatialContext} from '../../src/analysis-core/contracts';
import {GPULineClipByPolygon} from '../../src/gpu-spatial-analysis/line-clip/gpu-line-clip-by-polygon';
import {GPUPolygonOverlay} from '../../src/gpu-spatial-analysis/polygon-overlay/gpu-polygon-overlay';
import {GPUGeometryPredicates} from '../../src/gpu-spatial-analysis/geometry-predicates/gpu-geometry-predicates';
import {GPUZoneEvents} from '../../src/gpu-spatial-analysis/trajectory-zones/gpu-zone-events';
import {createNullWebGPUDevice} from '../utils/gpu-contributor-test-utils';

const PRIMARY: GPUSpatialContext = {
  coordinateSpace: 'planar',
  metric: 'native',
  datum: 'EPSG:4326',
  frame: {kind: 'local-azimuthal-equidistant', id: 'a'}
};
function createOperator(side: string, secondary?: GPUSpatialContext, omitSide = false) {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  let serial = 0;
  const view = <Format extends 'float32' | 'float32x2' | 'float32x4' | 'uint32'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `view-${serial++}`, format, length);
  const polygons = () => ({
    kind: 'polygons' as const,
    positions: view('float32x2', 4),
    featureOffsets: view('uint32', 2),
    polygonOffsets: view('uint32', 2),
    ringOffsets: view('uint32', 2)
  });
  const status = () => ({
    count: view('uint32', 1),
    requiredCount: view('uint32', 1),
    overflow: view('uint32', 1)
  });
  const common = {id: 'planar', spatialContext: PRIMARY};
  if (side === 'right') {
    return new GPUPolygonOverlay({
      ...common,
      rightSpatialContext: secondary,
      left: polygons(),
      right: omitSide ? undefined : polygons(),
      operation: omitSide ? 'dissolve' : 'intersection',
      capacity: {intersections: 16, nodedSegments: 32},
      vertexTolerance: 1e-5,
      output: {
        geometry: {
          kind: 'polygons',
          positions: view('float32x2', 32),
          featureOffsets: view('uint32', 9),
          polygonOffsets: view('uint32', 9),
          ringOffsets: view('uint32', 9),
          sourceIds: view('uint32', 8)
        },
        status: {...status(), candidateOverflow: view('uint32', 1)},
        boundary: {
          endpoints: view('float32x4', 32),
          operandIds: view('uint32', 32),
          sourceFeatureIds: view('uint32', 32),
          status: {...status(), candidateOverflow: view('uint32', 1)}
        }
      }
    });
  }
  if (side === 'other') {
    return new GPUGeometryPredicates({
      ...common,
      otherSpatialContext: secondary,
      geometry: polygons(),
      other: omitSide ? undefined : polygons(),
      output: {numCoordinates: view('uint32', 1)}
    });
  }
  if (side === 'polygons') {
    return new GPULineClipByPolygon({
      ...common,
      polygonsSpatialContext: secondary,
      polygons: polygons(),
      lines: {
        kind: 'lines',
        positions: view('float32x2', 2),
        lineOffsets: view('uint32', 2)
      },
      intersectionCapacity: 16,
      candidateCapacity: 16,
      output: {
        pieces: {
          geometry: {
            kind: 'lines',
            positions: view('float32x2', 32),
            lineOffsets: view('uint32', 9)
          },
          sourceIds: view('uint32', 8),
          status: status()
        }
      }
    });
  }
  return new GPUZoneEvents({
    ...common,
    zonesSpatialContext: secondary,
    zones: polygons(),
    tracks: {
      kind: 'trajectories',
      positions: view('float32x2', 2),
      timestamps: view('float32', 2),
      trackOffsets: view('uint32', 2)
    },
    candidateCapacity: 16,
    trackEventCapacity: 4,
    output: {
      events: {
        ids: view('uint32', 4),
        status: status(),
        eventTypes: view('uint32', 4),
        eventTimes: view('float32', 4)
      },
      dwellTimes: view('float32', 1),
      visitCounts: view('uint32', 1),
      trackEventCounts: view('uint32', 1)
    }
  });
}
for (const side of ['polygons', 'right', 'other', 'zones']) {
  it(`${side} accepts absent and matching secondary contexts`, () => {
    expect(createOperator(side).spatialContext).toBe(PRIMARY);
    expect(createOperator(side, {...PRIMARY}).spatialContext).toBe(PRIMARY);
  });
  it(`${side} rejects a secondary datum mismatch`, () => {
    expect(() => createOperator(side, {...PRIMARY, datum: 'EPSG:4267'})).toThrowError(
      new Error(
        `planar ${side}SpatialContext incompatible spatial contexts: datums differ (EPSG:4267 and EPSG:4326)`
      )
    );
  });
  it(`${side} rejects a secondary planar frame mismatch`, () => {
    expect(() =>
      createOperator(side, {...PRIMARY, frame: {kind: 'local-azimuthal-equidistant', id: 'b'}})
    ).toThrowError(
      new Error(
        `planar ${side}SpatialContext incompatible spatial contexts: planar frames differ (b and a)`
      )
    );
  });
}
for (const side of ['right', 'other']) {
  it(`${side} context requires its optional input`, () => {
    expect(createOperator(side, undefined, true).spatialContext).toBe(PRIMARY);
    expect(() => createOperator(side, PRIMARY, true)).toThrowError(
      new Error(`planar ${side}SpatialContext needs ${side}`)
    );
  });
}
