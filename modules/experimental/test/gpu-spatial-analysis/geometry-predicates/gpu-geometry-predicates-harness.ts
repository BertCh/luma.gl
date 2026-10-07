// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import type {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import type {GPUSpatialJoinGeometry} from '../../../src/gpu-spatial-analysis/spatial-join/index';
import {createInputBuffer} from '../../utils/gpu-contributor-test-utils';

/** A fixture coordinate: `null` encodes NaN and `'inf'` encodes Infinity. */
export type FixtureCoordinate = readonly (number | null | 'inf')[];

/** Test geometry description: one entry per feature. */
export type PredicateGeometrySpec =
  | {kind: 'points'; positions: readonly FixtureCoordinate[]}
  | {kind: 'lines'; vertices: readonly (readonly FixtureCoordinate[])[]}
  | {
      kind: 'polygons';
      /** Feature, polygon, ring, vertex. */
      polygons: readonly (readonly (readonly (readonly FixtureCoordinate[])[])[])[];
    };

function toNumber(value: number | null | 'inf' | undefined): number {
  return value === null || value === undefined ? Number.NaN : value === 'inf' ? Infinity : value;
}

/** Uploads a geometry description as GeoArrow-layout graph views. */
export function createPredicateGeometry(
  device: Device,
  graph: GPUCommandGraph,
  name: string,
  spec: PredicateGeometrySpec,
  buffers: Buffer[]
): GPUSpatialJoinGeometry {
  const positions: number[] = [];
  const lineOffsets = [0];
  const featureOffsets = [0];
  const polygonOffsets = [0];
  const ringOffsets = [0];
  const pushVertices = (vertices: readonly FixtureCoordinate[]) => {
    for (const vertex of vertices) {
      positions.push(toNumber(vertex[0]), toNumber(vertex[1]));
    }
  };
  if (spec.kind === 'points') {
    pushVertices(spec.positions);
  } else if (spec.kind === 'lines') {
    for (const vertices of spec.vertices) {
      pushVertices(vertices);
      lineOffsets.push(positions.length / 2);
    }
  } else {
    for (const feature of spec.polygons) {
      for (const polygon of feature) {
        for (const ring of polygon) {
          pushVertices(ring);
          ringOffsets.push(positions.length / 2);
        }
        polygonOffsets.push(ringOffsets.length - 1);
      }
      featureOffsets.push(polygonOffsets.length - 1);
    }
  }
  const upload = (data: Float32Array | Uint32Array, id: string, format: 'float32x2' | 'uint32') => {
    const buffer = createInputBuffer(device, data);
    buffers.push(buffer);
    return importGraphBuffer(
      graph,
      `${name}-${id}`,
      buffer,
      format,
      format === 'uint32' ? data.length : data.length / 2
    );
  };
  const positionView = upload(Float32Array.from(positions), 'positions', 'float32x2');
  const offsets = (id: string, data: number[]) => upload(Uint32Array.from(data), id, 'uint32');
  if (spec.kind === 'points') {
    return {kind: 'points', positions: positionView};
  }
  if (spec.kind === 'lines') {
    return {
      kind: 'lines',
      positions: positionView,
      lineOffsets: offsets('line-offsets', lineOffsets)
    };
  }
  return {
    kind: 'polygons',
    positions: positionView,
    featureOffsets: offsets('feature-offsets', featureOffsets),
    polygonOffsets: offsets('polygon-offsets', polygonOffsets),
    ringOffsets: offsets('ring-offsets', ringOffsets)
  };
}
