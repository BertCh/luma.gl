// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import type coreData from '../../data/k-core.json';

/** Shape of the deterministic real-data fixture. */
export type CoreData = typeof coreData;

/** Shared provenance for the GPU Core fixture. */
export const CORE_DATASET = {
  id: 'k-core',
  licence: 'NOAA public domain; swissALTI3D © swisstopo OGD / Copernicus GLO-30 with attribution',
  source: 'Showcase ghcn-stations (NOAA NCEI GHCN-Daily via AWS Open Data), 2024-09-27; alps-dem'
};

/** Builds canonical binary flags and integer rainfall from the real station rows. */
export function getCoreStationInputs(data: CoreData) {
  const stations = data.stations;
  const segmentFlags = new Uint32Array(stations.row.length);
  for (const offset of stations.stateOffsets.slice(0, -1)) segmentFlags[offset] = 1;
  const first = stations.stateOffsets.find(
    (offset, index, offsets) =>
      offset > 0 && index < offsets.length - 2 && offsets[index + 1] - offset < 16
  );
  // The selected dataset has at least two state starts in a 16-row window.
  assert(first !== undefined);
  return {
    rainfall: Uint32Array.from(stations.prcpMm, Math.round),
    rainfallMm: Float32Array.from(stations.prcpMm),
    stationRows: Uint32Array.from(stations.row),
    wet: Uint32Array.from(stations.prcpMm, value => Number(value >= 100)),
    high: Uint32Array.from(stations.elevationM, value => Number(value >= 500)),
    segmentFlags,
    first
  };
}

/** Captures at most sixteen rows without changing the buffer's allocation or contents. */
export function getCoreWindow(graph: GPUCommandGraph<undefined>, view: GraphDataView, first = 0) {
  return graph.createDataView(view.buffer, {
    format: view.format,
    length: Math.min(16, view.length - first),
    byteOffset: view.byteOffset + first * view.rowByteLength
  });
}
