// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import type {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {createInputBuffer} from '../../utils/gpu-contributor-test-utils';
import {
  flattenLonLatFeatures,
  type LonLatPolygonData,
  type LonLatPosition
} from './lon-lat-polygon-oracle';

function createFixture(...polygons: LonLatPosition[][][]): LonLatPolygonData {
  return flattenLonLatFeatures([polygons]);
}

/** Fiji represented by two polygons already cut at the antimeridian. */
export const FIJI_SPLIT = createFixture(
  [
    [
      [177, -19],
      [180, -19],
      [180, -16],
      [177, -16]
    ]
  ],
  [
    [
      [-180, -19],
      [-178, -19],
      [-178, -16],
      [-180, -16]
    ]
  ]
);
/** Fiji represented by one short-way transmeridian shell. */
export const FIJI_TRANSMERIDIAN = createFixture([
  [
    [177, -19],
    [-178, -19],
    [-178, -16],
    [177, -16]
  ]
]);
/** A seam-crossing shell and seam-crossing hole, with opposite starting longitude copies. */
export const FIJI_WITH_SEAM_HOLE = createFixture([
  [
    [177, -19],
    [-178, -19],
    [-178, -16],
    [177, -16]
  ],
  [
    [-179, -18],
    [179, -18],
    [179, -17],
    [-179, -17]
  ]
]);
/** Natural-Earth-style Antarctica, explicitly closed through the south-pole degree line. */
export const ANTARCTICA_NATURAL_EARTH = createFixture([
  [
    [-180, -75],
    [-120, -72],
    [-60, -76],
    [0, -70],
    [60, -74],
    [120, -71],
    [180, -75],
    [180, -90],
    [-180, -90]
  ]
]);
/** North-pole enclosure with positive 360-degree winding and no explicit pole edge. */
export const ARCTIC_RING = createFixture([
  [
    [-180, 70],
    [-90, 72],
    [0, 70],
    [90, 72]
  ]
]);
/** South-pole enclosure with negative 360-degree winding and no explicit pole edge. */
export const SOUTHERN_POLE_RING = createFixture([
  [
    [180, -70],
    [90, -72],
    [0, -70],
    [-90, -72]
  ]
]);
/** An exactly half-turn linear edge with no unique short-way interpretation. */
export const AMBIGUOUS_180_EDGE = createFixture([
  [
    [0, 10],
    [180, 10],
    [90, 20]
  ]
]);
/** Two full longitude turns before implicit closure; preparation must reject it. */
export const SPIRAL_WIDER_THAN_360 = createFixture([
  [
    [0, 10],
    [120, 20],
    [-120, 30],
    [0, 40],
    [120, 50],
    [-120, 60]
  ]
]);
/** A small ordinary square with no preparation flags. */
export const CHICAGO_SQUARE = createFixture([
  [
    [-87.7, 41.8],
    [-87.6, 41.8],
    [-87.6, 41.9],
    [-87.7, 41.9]
  ]
]);
/** First edge spans sixty degrees at 45N; its minor great circle bows north. */
export const GREAT_CIRCLE_60_DEGREE_EDGE = createFixture([
  [
    [-30, 45],
    [30, 45],
    [0, 30]
  ]
]);

/** Every named fixture, for shared classification and consumer-check suites. */
export const LON_LAT_POLYGON_FIXTURES = {
  FIJI_SPLIT,
  FIJI_TRANSMERIDIAN,
  FIJI_WITH_SEAM_HOLE,
  ANTARCTICA_NATURAL_EARTH,
  ARCTIC_RING,
  SOUTHERN_POLE_RING,
  AMBIGUOUS_180_EDGE,
  SPIRAL_WIDER_THAN_360,
  CHICAGO_SQUARE,
  GREAT_CIRCLE_60_DEGREE_EDGE
};

/** Uploads a binary64 fixture as float32 positions and uint32 offsets; caller owns the buffers. */
export function createLonLatPolygonFixtureViews<Parameters>(
  device: Device,
  graph: GPUCommandGraph<Parameters>,
  fixture: LonLatPolygonData,
  id = 'lon-lat-fixture'
) {
  const buffers: Buffer[] = [];
  const upload = <Format extends 'float32x2' | 'uint32'>(
    name: string,
    values: Float32Array | Uint32Array,
    format: Format,
    length: number
  ) => {
    const buffer = createInputBuffer(device, values.length ? values : new Uint32Array(2));
    buffers.push(buffer);
    return importGraphBuffer(graph, `${id}-${name}`, buffer, format, length);
  };
  return {
    positions: upload(
      'positions',
      new Float32Array(fixture.positions),
      'float32x2',
      fixture.positions.length / 2
    ),
    featureOffsets: upload(
      'feature-offsets',
      fixture.featureOffsets,
      'uint32',
      fixture.featureOffsets.length
    ),
    polygonOffsets: upload(
      'polygon-offsets',
      fixture.polygonOffsets,
      'uint32',
      fixture.polygonOffsets.length
    ),
    ringOffsets: upload('ring-offsets', fixture.ringOffsets, 'uint32', fixture.ringOffsets.length),
    buffers,
    destroy: () => {
      for (const buffer of buffers) {
        buffer.destroy();
      }
    }
  };
}
