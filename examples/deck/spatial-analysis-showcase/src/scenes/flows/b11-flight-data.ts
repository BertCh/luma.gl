// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {LoadedDataset} from '../../data/catalog';
import {getContinentIndex, type AirportRecord} from './b11-geography';

/** An undirected airport network with per-edge attributes. */
export type FlightNetwork = {
  id: 'world' | 'us';
  nodeCount: number;
  /** `[longitude, latitude]` per airport. */
  lonLat: Float32Array;
  airports: readonly AirportRecord[];
  /** Index into `CONTINENT_NAMES` per airport (the OpenFlights country table). */
  continent: Uint8Array;
  /** Distinct neighbours per airport. */
  degree: Uint32Array;
  edgeCount: number;
  source: Uint32Array;
  target: Uint32Array;
  distanceKm: Float32Array;
  /** Airline-route records (world) or scheduled flights in July 2023 (US) on the pair. */
  traffic: Float32Array;
  /** Distinct airlines serving the pair (world only). */
  airlines: Float32Array | null;
  /** Mean departure delay in minutes, flight-weighted over both directions (US only). */
  delay: Float32Array | null;
  /** Share of scheduled flights cancelled (US only). */
  cancelRate: Float32Array | null;
};

/** Reads the OpenFlights route network (one row per undirected airport pair). */
export function readWorldNetwork(dataset: LoadedDataset, airports: AirportRecord[]): FlightNetwork {
  const lonLat = dataset.column<Float32Array>('locations');
  const nodeCount = lonLat.length / 2;
  const source = dataset.column<Uint32Array>('origin');
  const target = dataset.column<Uint32Array>('destination');
  const degree = Uint32Array.from(dataset.column<Uint16Array>('degree'));
  return {
    id: 'world',
    nodeCount,
    lonLat,
    airports,
    continent: Uint8Array.from(airports.map(airport => getContinentIndex(airport.country))),
    degree,
    edgeCount: source.length,
    source,
    target,
    distanceKm: dataset.column<Float32Array>('distanceKm'),
    traffic: Float32Array.from(dataset.column<Uint16Array>('count')),
    airlines: Float32Array.from(dataset.column<Uint16Array>('airlineCount')),
    delay: null,
    cancelRate: null
  };
}

/** Reads the July 2023 US flights, merging the two directions of each airport pair. */
export function readUsNetwork(dataset: LoadedDataset, airports: AirportRecord[]): FlightNetwork {
  const lonLat = dataset.column<Float32Array>('locations');
  const nodeCount = lonLat.length / 2;
  const origin = dataset.column<Uint32Array>('origin');
  const destination = dataset.column<Uint32Array>('destination');
  const count = dataset.column<Uint32Array>('count');
  const cancelled = dataset.column<Uint16Array>('cancelled');
  const delay = dataset.column<Float32Array>('meanDepDelay');
  const distance = dataset.column<Float32Array>('distanceKm');
  const merged = new Map<
    number,
    {a: number; b: number; flights: number; cancelled: number; delaySum: number; distance: number}
  >();
  for (let row = 0; row < origin.length; row++) {
    const a = Math.min(origin[row], destination[row]);
    const b = Math.max(origin[row], destination[row]);
    const key = a * nodeCount + b;
    let entry = merged.get(key);
    if (!entry) {
      entry = {a, b, flights: 0, cancelled: 0, delaySum: 0, distance: distance[row]};
      merged.set(key, entry);
    }
    entry.flights += count[row];
    entry.cancelled += cancelled[row];
    if (Number.isFinite(delay[row])) entry.delaySum += delay[row] * count[row];
  }
  const entries = [...merged.values()];
  const edgeCount = entries.length;
  const source = new Uint32Array(edgeCount);
  const target = new Uint32Array(edgeCount);
  const distanceKm = new Float32Array(edgeCount);
  const traffic = new Float32Array(edgeCount);
  const meanDelay = new Float32Array(edgeCount);
  const cancelRate = new Float32Array(edgeCount);
  const degree = new Uint32Array(nodeCount);
  entries.forEach((entry, edge) => {
    source[edge] = entry.a;
    target[edge] = entry.b;
    distanceKm[edge] = entry.distance;
    traffic[edge] = entry.flights;
    meanDelay[edge] = entry.flights > 0 ? entry.delaySum / entry.flights : 0;
    cancelRate[edge] = entry.flights > 0 ? entry.cancelled / entry.flights : 0;
    degree[entry.a]++;
    degree[entry.b]++;
  });
  return {
    id: 'us',
    nodeCount,
    lonLat,
    airports,
    continent: new Uint8Array(nodeCount).fill(2),
    degree,
    edgeCount,
    source,
    target,
    distanceKm,
    traffic,
    airlines: null,
    delay: meanDelay,
    cancelRate
  };
}

/** Edges expanded for the antimeridian: some originals appear twice with shifted endpoints. */
export type ExpandedEdges = {
  /** Airport positions followed by shifted virtual copies (longitude plus or minus 360). */
  positions: Float32Array;
  source: Uint32Array;
  target: Uint32Array;
  /** Original edge of every expanded edge. */
  original: Uint32Array;
};

/**
 * Straight lines in longitude and latitude would cross the whole map for a pair that spans the
 * antimeridian (Sydney to Los Angeles). Such an edge is replaced by two: one to a copy of the
 * target shifted by 360 degrees, and one from a copy of the source shifted the other way, so each
 * half leaves the visible world on its own side.
 */
export function expandAntimeridianEdges(network: FlightNetwork): ExpandedEdges {
  const positions: number[] = Array.from(network.lonLat);
  const source: number[] = [];
  const target: number[] = [];
  const original: number[] = [];
  const copy = (airport: number, shift: number): number => {
    positions.push(network.lonLat[airport * 2] + shift, network.lonLat[airport * 2 + 1]);
    return positions.length / 2 - 1;
  };
  for (let edge = 0; edge < network.edgeCount; edge++) {
    const a = network.source[edge];
    const b = network.target[edge];
    const difference = network.lonLat[b * 2] - network.lonLat[a * 2];
    if (Math.abs(difference) <= 180) {
      source.push(a);
      target.push(b);
      original.push(edge);
    } else {
      const shift = difference > 0 ? -360 : 360;
      source.push(a);
      target.push(copy(b, shift));
      original.push(edge);
      source.push(copy(a, -shift));
      target.push(b);
      original.push(edge);
    }
  }
  return {
    positions: Float32Array.from(positions),
    source: Uint32Array.from(source),
    target: Uint32Array.from(target),
    original: Uint32Array.from(original)
  };
}

/** Great-circle-free equirectangular length in kilometres between two lon/lat points. */
export function getApproximateKilometres(
  longitudeA: number,
  latitudeA: number,
  longitudeB: number,
  latitudeB: number
): number {
  const meanLatitude = ((latitudeA + latitudeB) / 2) * (Math.PI / 180);
  const x = (longitudeB - longitudeA) * Math.cos(meanLatitude);
  const y = latitudeB - latitudeA;
  return Math.hypot(x, y) * 111.19;
}
