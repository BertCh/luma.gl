// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUComputationCapture, GPUComputationKeySpace} from '@luma.gl/gpgpu';
import type {NetworkRoads} from './network-element-lens';

/** Facility rows of the roads file used by the search family: a hospital, two libraries and a fire station. */
export const SEARCH_FACILITY_ROWS = [0, 5, 3, 18] as const;

/** The selected facilities (at most four, so nominal assignments stay readable). */
export function getSearchFacilities(data: NetworkRoads) {
  return SEARCH_FACILITY_ROWS.map(row => data.facilities[row]);
}

/** Key spaces shared by the search family: nodes, edges, facilities and snapped points. */
export function getSearchKeySpaces(
  data: NetworkRoads,
  pointCount: number
): GPUComputationKeySpace[] {
  return [
    {id: 'node', label: 'network node', indexSpace: '1d', size: data.nodeCount},
    {id: 'edge', label: 'directed edge', indexSpace: 'csr', size: data.edgeCount},
    {id: 'facility', label: 'facility', indexSpace: '1d', size: SEARCH_FACILITY_ROWS.length},
    {id: 'point', label: 'point to snap', indexSpace: '1d', size: pointCount},
    {id: 'frontier-slot', label: 'frontier slot', indexSpace: '1d', size: 2 * data.nodeCount}
  ];
}

/** Gathers the positions of selected nodes from the node-position readback, as an input marker layer. */
export function gatherNodePoints(
  positions: GPUComputationCapture,
  nodes: readonly number[],
  props: {id: string; meaning: string; coordinateModel?: 'geographic' | 'projected'; unit?: string}
): GPUComputationCapture {
  return {
    ...positions,
    id: props.id,
    name: props.id,
    keySpace: undefined,
    values: Float32Array.from(
      nodes.flatMap(node => [positions.values[node * 2], positions.values[node * 2 + 1]])
    ),
    role: 'input',
    meaning: props.meaning,
    geometry: {
      kind: 'points',
      coordinateModel: props.coordinateModel ?? 'geographic',
      ...(props.unit ? {unit: props.unit} : {})
    }
  };
}

/** Reads a captured uint32 view as floats, with the `0xffffffff` "none" sentinel shown as no data. */
export function maskNoneSentinel(
  capture: GPUComputationCapture,
  meaning: string
): GPUComputationCapture {
  return {
    ...capture,
    elementType: 'f32',
    values: Float32Array.from(capture.values, value => (value === 0xffffffff ? NaN : value)),
    meaning
  };
}

/** Node-keyed output points: the node-position readback coloured by a captured per-node value. */
export function createNodeValuePoints(
  positions: GPUComputationCapture,
  props: {
    id: string;
    valuesCapture: string;
    meaning: string;
    coordinateModel?: 'geographic' | 'projected';
    unit?: string;
    highlight?: {key: number; label: string};
  }
): GPUComputationCapture {
  return {
    ...positions,
    id: props.id,
    name: props.id,
    keySpace: 'node',
    role: 'output',
    meaning: props.meaning,
    ...(props.highlight ? {highlight: props.highlight} : {}),
    geometry: {
      kind: 'points',
      coordinateModel: props.coordinateModel ?? 'geographic',
      valuesCapture: props.valuesCapture,
      ...(props.unit ? {unit: props.unit} : {})
    }
  };
}

/**
 * Reverses every directed edge of the road extract (deterministic counting sort by target), so a
 * search from a facility measures travel time from every intersection to that facility. This is
 * data preparation: the edge weights are carried over unchanged.
 */
export function reverseRoads(data: NetworkRoads): NetworkRoads {
  const offsets = new Array<number>(data.nodeCount + 1).fill(0);
  for (const target of data.neighbors) offsets[target + 1]++;
  for (let node = 0; node < data.nodeCount; node++) offsets[node + 1] += offsets[node];
  const cursor = offsets.slice(0, data.nodeCount);
  const neighbors = new Array<number>(data.edgeCount);
  const travelTimeSeconds = new Array<number>(data.edgeCount);
  const lengthMeters = new Array<number>(data.edgeCount);
  const roadClass = new Array<number>(data.edgeCount);
  for (let source = 0; source < data.nodeCount; source++) {
    for (let edge = data.offsets[source]; edge < data.offsets[source + 1]; edge++) {
      const slot = cursor[data.neighbors[edge]]++;
      neighbors[slot] = source;
      travelTimeSeconds[slot] = data.travelTimeSeconds[edge];
      lengthMeters[slot] = data.lengthMeters[edge];
      roadClass[slot] = data.roadClass[edge];
    }
  }
  return {...data, offsets, neighbors, travelTimeSeconds, lengthMeters, roadClass};
}

/** Packs the real longitude/latitude of facility rows as interleaved float32 pairs (for upload). */
export function getFacilityPositions(
  data: NetworkRoads,
  rows: readonly number[] = SEARCH_FACILITY_ROWS
): Float32Array {
  return Float32Array.from(
    rows.flatMap(row => [data.facilities[row].lon, data.facilities[row].lat])
  );
}
