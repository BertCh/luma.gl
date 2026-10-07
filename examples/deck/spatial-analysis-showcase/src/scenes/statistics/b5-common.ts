// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {COORDINATE_SYSTEM, type Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import type {LoadedDataset} from '../../data/catalog';
import {SpatialAnalysisSegmentLayer} from '../../engine/layers';
import type {SpatialAnalysisResources} from '../../engine/resources';
import {ChoroplethFillLayer, type ChoroplethFillLayerProps} from './b5-choropleth-layer';
import {
  buildPolygonMesh,
  createFeatureLocator,
  getPolygonLayout,
  type FeatureLocator,
  type PolygonLayout,
  type PolygonMesh
} from './b5-geometry';

/** GPU geometry of one polygon dataset: static triangle and outline buffers plus a CPU locator. */
export type ChoroplethGeometry = {
  layout: PolygonLayout;
  mesh: PolygonMesh;
  locator: FeatureLocator;
  positions: Buffer;
  featureRows: Buffer;
  outline: Buffer;
  featureCount: number;
  vertexCount: number;
  segmentCount: number;
  /** Fill layer over a per-feature `values` buffer. */
  createFillLayer: (
    id: string,
    props: Omit<ChoroplethFillLayerProps, 'positions' | 'featureRows' | 'vertexCount' | 'id'>
  ) => Layer;
  /** Outline layer, `widthPixels` wide. */
  createOutlineLayer: (
    id: string,
    color: readonly [number, number, number, number],
    widthPixels: number
  ) => Layer;
};

/** Triangulates a polygon dataset once and uploads the static buffers. */
export function createChoroplethGeometry(
  resources: SpatialAnalysisResources,
  dataset: LoadedDataset
): ChoroplethGeometry {
  const layout = getPolygonLayout(dataset);
  const mesh = buildPolygonMesh(layout);
  const locator = createFeatureLocator(layout);
  const positions = resources.createBuffer('fill-positions', mesh.positions);
  const featureRows = resources.createBuffer('fill-features', mesh.featureRows);
  const outline = resources.createBuffer('outline', mesh.outline);
  const vertexCount = mesh.positions.length / 2;
  const segmentCount = mesh.outline.length / 4;
  return {
    layout,
    mesh,
    locator,
    positions,
    featureRows,
    outline,
    featureCount: layout.featureCount,
    vertexCount,
    segmentCount,
    createFillLayer: (id, props) =>
      new ChoroplethFillLayer({id, positions, featureRows, vertexCount, ...props}),
    createOutlineLayer: (id, color, widthPixels) =>
      new SpatialAnalysisSegmentLayer({
        id,
        coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
        segments: outline,
        instanceCount: segmentCount,
        widthPixels,
        color
      })
  };
}

/** Outline color that reads on the current basemap. */
export function getOutlineColor(
  theme: 'light' | 'dark',
  alpha = 130
): [number, number, number, number] {
  return theme === 'dark' ? [235, 240, 250, alpha] : [40, 50, 70, alpha];
}

/** Returns the sorted finite values of a column (a copy). */
export function getSortedFinite(values: ArrayLike<number>): Float64Array {
  const finite: number[] = [];
  for (let index = 0; index < values.length; index++) {
    if (Number.isFinite(values[index])) finite.push(values[index]);
  }
  return Float64Array.from(finite).sort();
}

/** Linear-interpolated quantile `q` in `[0, 1]` of a sorted array. */
export function getQuantile(sorted: ArrayLike<number>, q: number): number {
  if (sorted.length === 0) return NaN;
  const position = Math.min(Math.max(q, 0), 1) * (sorted.length - 1);
  const lower = Math.floor(position);
  const upper = Math.min(lower + 1, sorted.length - 1);
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
}

/** Reads a typed-array copy from a readback ArrayBuffer, advancing a byte cursor. */
export function createByteReader(bytes: ArrayBuffer) {
  let cursor = 0;
  return {
    floats(length: number): Float32Array {
      const values = new Float32Array(bytes, cursor, length);
      cursor += length * 4;
      return values;
    },
    words(length: number): Uint32Array {
      const values = new Uint32Array(bytes, cursor, length);
      cursor += length * 4;
      return values;
    }
  };
}

/** Formats a decimal with thousands separators. */
export function formatNumber(value: number, digits = 0): string {
  if (!Number.isFinite(value)) return 'n/a';
  return value.toLocaleString('en-US', {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits
  });
}
