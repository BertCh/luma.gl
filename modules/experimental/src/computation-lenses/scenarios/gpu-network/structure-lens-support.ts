// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUComputationCapture, GPUComputationKeySpace} from '@luma.gl/gpgpu';
import type {NetworkRoads} from './network-element-lens';

/** Key spaces shared by the structure family: nodes and directed edges, plus the scenario's own. */
export function getStructureKeySpaces(
  data: NetworkRoads,
  extra: readonly GPUComputationKeySpace[] = []
): GPUComputationKeySpace[] {
  return [
    {id: 'node', label: 'network node', indexSpace: '1d', size: data.nodeCount},
    {id: 'edge', label: 'directed edge', indexSpace: 'csr', size: data.edgeCount},
    ...extra
  ];
}

/** The adjacency-matrix scenarios group intersections in a 4 x 2 grid over the data bounding box. */
export const MATRIX_ZONE_COLUMNS = 4;
export const MATRIX_ZONE_ROWS = 2;

/**
 * Deterministic input partition: the zone of every intersection in a `columns` x `rows` grid over
 * the data bounding box, as `row * columns + column`. This is data preparation computed in the
 * script and uploaded, never a GPU result.
 */
export function getGridZoneLabels(data: NetworkRoads, columns: number, rows: number): Uint32Array {
  const [west, south, east, north] = data.bbox;
  const labels = new Uint32Array(data.nodeCount);
  for (let node = 0; node < data.nodeCount; node++) {
    const column = Math.min(
      columns - 1,
      Math.max(0, Math.floor(((data.nodes[node * 2] - west) / (east - west)) * columns))
    );
    const row = Math.min(
      rows - 1,
      Math.max(0, Math.floor(((data.nodes[node * 2 + 1] - south) / (north - south)) * rows))
    );
    labels[node] = row * columns + column;
  }
  return labels;
}

/** Index of the first maximum of a value list, ignoring NaN (the GPU-selected highlight key). */
export function getMaximumKey(values: ArrayLike<number>): number {
  let best = 0;
  for (let key = 1; key < values.length; key++) if (values[key] > values[best]) best = key;
  return best;
}

/** The first `count` rows of a readback, for outputs whose live length is a GPU count. */
export function takeFirstRows(
  capture: GPUComputationCapture,
  count: number,
  props: {keySpace?: string; meaning: string}
): GPUComputationCapture {
  return {
    ...capture,
    keySpace: props.keySpace,
    values: Array.from(capture.values).slice(0, count),
    meaning: props.meaning
  };
}

/**
 * A capture assembled from values gathered out of readbacks (segment endpoints, midpoints, index
 * bookkeeping, slices of a summary). It names a source resource only when `source` is given, a
 * readback it is a slice of; its `meaning` states what it was gathered from.
 */
export function createGatheredCapture(props: {
  id: string;
  /** The readback this capture is a slice of; its resource links the capture to the kernel. */
  source?: GPUComputationCapture;
  elementType: 'f32' | 'u32';
  components?: number;
  values: ArrayLike<number>;
  keySpace?: string;
  role?: GPUComputationCapture['role'];
  meaning: string;
  geometry?: GPUComputationCapture['geometry'];
  highlight?: GPUComputationCapture['highlight'];
}): GPUComputationCapture {
  return {
    id: props.id,
    name: props.id,
    elementType: props.elementType,
    ...(props.components ? {components: props.components} : {}),
    values: props.values,
    provenance: 'gpu-readback',
    ...(props.source
      ? {
          resource: props.source.resource,
          graph: props.source.graph,
          afterNode: props.source.afterNode
        }
      : {}),
    ...(props.keySpace ? {keySpace: props.keySpace} : {}),
    ...(props.role ? {role: props.role} : {}),
    meaning: props.meaning,
    ...(props.geometry ? {geometry: props.geometry} : {}),
    ...(props.highlight ? {highlight: props.highlight} : {})
  };
}

/** Source node of every CSR slot, expanded from the offsets readback. */
export function getSlotSources(offsets: ArrayLike<number>): number[] {
  const sources: number[] = [];
  for (let node = 0; node + 1 < offsets.length; node++)
    for (let slot = offsets[node]; slot < offsets[node + 1]; slot++) sources.push(node);
  return sources;
}

/** Rounds a coordinate to six decimals so gathered geometry stays compact in the record. */
export function roundCoordinate(value: number): number {
  return Math.round(value * 1e6) / 1e6;
}
