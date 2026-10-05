// luma.gl
// SPDX-License-Identifier: MIT AND Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
// SPDX-FileCopyrightText: Copyright 2016-2024 Uber Technologies, Inc.

import {dggs} from '@luma.gl/shadertools';
import {cellToChildren, cellToLatLng, getRes0Cells, latLngToCell} from 'h3-js';
import {describe, expect, it} from 'vitest';
import {H3_INDEX_WGSL} from '../../../src/geospatial/cell-indexing/h3-index-wgsl';
import {
  CELL_INDEX_H3_BASE_CELL_PACKED_TABLE,
  CELL_INDEX_H3_CW_OFFSET_PENTAGON_FACES
} from '../../../src/geospatial/cell-indexing/h3-index-tables';
import {
  H3_FACE_BASES,
  createCellEdgePoints,
  createPentagonNeighborhoodPoints,
  createSpecialH3Points,
  createUniformSpherePoints,
  latLngToCellOracle
} from './h3-index-oracle';

function expectOracleMatchesH3(points: Float32Array, resolution: number, label: string): void {
  let mismatches = 0;
  let firstMismatch = '';
  for (let index = 0; index < points.length; index += 2) {
    const lng = points[index];
    const lat = points[index + 1];
    const expected = BigInt(`0x${latLngToCell(lat, lng, resolution)}`);
    const actual = latLngToCellOracle(lng, lat, resolution);
    if (actual !== expected) {
      mismatches++;
      firstMismatch ||= `(${lng}, ${lat}) expected ${expected.toString(16)} got ${actual.toString(16)}`;
    }
  }
  expect(mismatches, `${label} res ${resolution}: ${firstMismatch}`).toBe(0);
}

describe('H3_INDEX_WGSL tables and float64 port', () => {
  it('dggs face basis table agrees with the azimuth-derived float64 basis', () => {
    const start = dggs.source.indexOf('const DGGS_H3_FACE_UNIT_VECTOR_BASES');
    const end = dggs.source.indexOf(');', start);
    const rows = [...dggs.source.slice(start, end).matchAll(/vec4f\(([^)]*)\)/g)].map(match =>
      match[1].split(',').map(Number)
    );
    expect(rows.length).toBe(60);
    for (let face = 0; face < 20; face++) {
      for (let axis = 0; axis < 3; axis++) {
        for (let component = 0; component < 3; component++) {
          expect(
            Math.abs(rows[face * 3 + axis][component] - H3_FACE_BASES[face][axis][component])
          ).toBeLessThan(1e-9);
        }
      }
    }
  });

  it('table shape and pentagon data are well formed', () => {
    expect(CELL_INDEX_H3_BASE_CELL_PACKED_TABLE.length).toBe(540);
    expect(CELL_INDEX_H3_BASE_CELL_PACKED_TABLE.filter(value => value >= 0).length).toBe(320);
    expect(
      CELL_INDEX_H3_BASE_CELL_PACKED_TABLE.every(value => value < 0 || (value & 0xff) < 122)
    ).toBe(true);
    expect(CELL_INDEX_H3_CW_OFFSET_PENTAGON_FACES.length).toBe(12);
  });

  it('maps every resolution 0, 1 and 2 cell center back to itself', () => {
    let cellCount = 0;
    for (const base of getRes0Cells()) {
      const cells = [base];
      for (let resolution = 0; resolution <= 2; resolution++) {
        const next: string[] = [];
        for (const cell of cells) {
          const [lat, lng] = cellToLatLng(cell);
          expect(latLngToCellOracle(lng, lat, resolution)).toBe(BigInt(`0x${cell}`));
          cellCount++;
        }
        if (resolution < 2) {
          for (const cell of cells) {
            next.push(...cellToChildren(cell, resolution + 1));
          }
          cells.length = 0;
          cells.push(...next);
        }
      }
    }
    expect(cellCount).toBe(122 + 842 + 5882);
  });

  it('matches h3-js on uniform random points for resolutions 0..15', () => {
    for (let resolution = 0; resolution <= 15; resolution++) {
      expectOracleMatchesH3(
        createUniformSpherePoints(4000, 100 + resolution),
        resolution,
        'uniform'
      );
    }
  });

  it('matches h3-js near pentagons for resolutions 0..15', () => {
    for (let resolution = 0; resolution <= 15; resolution++) {
      expectOracleMatchesH3(
        createPentagonNeighborhoodPoints(resolution, 150, 200 + resolution),
        resolution,
        'pentagon'
      );
    }
  });

  it('matches h3-js near cell edges for resolutions 0..15', () => {
    for (let resolution = 0; resolution <= 15; resolution++) {
      expectOracleMatchesH3(
        createCellEdgePoints(resolution, 2500, 300 + resolution),
        resolution,
        'edge'
      );
    }
  });

  it('matches h3-js on poles, antimeridian and equator probes', () => {
    for (let resolution = 0; resolution <= 15; resolution++) {
      expectOracleMatchesH3(createSpecialH3Points(), resolution, 'special');
    }
  });

  it('returns the null cell for non-finite input and resolution above 15', () => {
    expect(latLngToCellOracle(NaN, 0, 5)).toBe(0n);
    expect(latLngToCellOracle(0, Infinity, 5)).toBe(0n);
    expect(latLngToCellOracle(0, 0, 16)).toBe(0n);
  });
});

describe('H3_INDEX_WGSL source', () => {
  it('defines the contract function and only prefixed helper identifiers', () => {
    expect(H3_INDEX_WGSL).toContain(
      'fn cellIndexH3FromLngLat(lngLatDegrees: vec2f, resolution: u32) -> vec2u'
    );
    const names = [...H3_INDEX_WGSL.matchAll(/^(?:fn|const|var|struct)\s+(\w+)/gm)].map(
      match => match[1]
    );
    expect(names.length).toBeGreaterThan(10);
    for (const name of names) {
      expect(name.startsWith('cellIndexH3') || name.startsWith('CELL_INDEX_H3_'), name).toBe(true);
      expect(dggs.source.includes(` ${name}(`) || dggs.source.includes(` ${name}:`), name).toBe(
        false
      );
    }
  });

  it('embeds the 540-entry base cell table', () => {
    const match = H3_INDEX_WGSL.match(/CELL_INDEX_H3_BASE_CELL_TABLE = array<u32, 540>\(([^)]*)\)/);
    expect(match?.[1].split(',').length).toBe(540);
  });
});
