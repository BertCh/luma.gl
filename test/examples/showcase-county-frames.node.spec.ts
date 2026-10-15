// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {afterEach, describe, expect, test, vi} from 'vitest';
import {
  LoadedDataset,
  type DatasetManifest,
  type LoadedColumn
} from '../../examples/deck/spatial-analysis-showcase/src/data/catalog';
import countyInfo from '../../examples/deck/spatial-analysis-showcase/src/data/datasets/us-counties.dataset';
import {haversineDistance} from '../../examples/deck/spatial-analysis-showcase/src/cartography/reference-geometry';
import {loadGeography} from '../../examples/deck/spatial-analysis-showcase/src/scenes/weights/b4-geography';

function loadCounties(): LoadedDataset {
  const directory = new URL(
    '../../examples/deck/spatial-analysis-showcase/public/data/us-counties/',
    import.meta.url
  );
  const manifest: DatasetManifest = JSON.parse(
    readFileSync(new URL('manifest.json', directory), 'utf8')
  );
  const columns: Record<string, LoadedColumn> = {};
  for (const name of [
    'vertices',
    'ringOffsets',
    'polygonRingOffsets',
    'countyPolygonOffsets',
    'centroid',
    'stateFips',
    'population'
  ]) {
    const column = manifest.columns![name];
    const bytes = Uint8Array.from(readFileSync(new URL(column.file, directory)));
    const Constructor =
      column.dtype === 'float32'
        ? Float32Array
        : column.dtype === 'uint32'
          ? Uint32Array
          : Uint8Array;
    columns[name] = {data: new Constructor(bytes.buffer), components: column.components ?? 1};
  }
  return new LoadedDataset(countyInfo, {manifest, columns, geojson: null, raster: null}, 'bundled');
}

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe('showcase county frames', () => {
  test('display preparation preserves the subsequent unsafe metric-frame warning', () => {
    const warning = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const counties = loadCounties();
    const display = counties.getDisplayProjection();
    expect(display.project(-96, 38).every(Number.isFinite)).toBe(true);
    expect(warning).not.toHaveBeenCalled();
    counties.projectColumn('vertices');
    expect(warning).toHaveBeenCalledExactlyOnceWith(expect.stringContaining('half-diagonal'));
    counties.getProjection();
    expect(warning).toHaveBeenCalledTimes(1);
  });

  test('county distance inputs retain the declared Albers frame independently of display', async () => {
    const warning = vi.spyOn(console, 'warn').mockImplementation(() => {});
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('Offline name lookup')));
    const counties = loadCounties();
    const geography = await loadGeography(
      'us-counties',
      {get: () => counties},
      new AbortController().signal
    );
    expect(warning).not.toHaveBeenCalled();
    expect(geography.analysisFrame.layerFrame.kind).toBe('albers-equal-area');
    expect(geography.analysisFrame.spatialContext.units).toBe('meters');
    expect(geography.analysisFrame.spatialContext.frame?.id).not.toBe(
      counties.getDisplayProjection().spatialContext.frame?.id
    );
    expect(geography.analysisContiguityVertices).not.toEqual(geography.contiguityVertices);
    const source = counties.column<Float32Array>('centroid');
    // Check actual county points across the coverage, including their local distance scale.
    for (let row = 0; row < geography.count; row += 37) {
      const longitude = source[row * 2];
      const latitude = source[row * 2 + 1];
      const recovered = geography.getCentroidLngLat(row);
      expect(recovered[0]).toBeCloseTo(longitude, 4);
      expect(recovered[1]).toBeCloseTo(latitude, 4);
      const start = geography.analysisFrame.project(longitude, latitude);
      const end = geography.analysisFrame.project(longitude + 0.01, latitude);
      const projectedDistance = Math.hypot(end[0] - start[0], end[1] - start[1]);
      const groundDistance = haversineDistance([longitude, latitude], [longitude + 0.01, latitude]);
      expect(Math.abs(projectedDistance / groundDistance - 1)).toBeLessThan(
        geography.analysisFrame.maxScaleError
      );
    }
    expect(geography.medianSpacing).toBeGreaterThan(1000);
    expect(geography.medianSpacing).toBeLessThan(100000);
  });
});
