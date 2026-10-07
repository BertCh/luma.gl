// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Pure helpers of the flight-bundling story: looking routes up by IATA code, the CPU mirror of
 * the work box `GPUEdgeBundling` computes on the GPU, and the filters that decide which pairs are
 * live. No GPU imports.
 */

import type {ExpandedEdges, FlightNetwork} from './b11-flight-data';
import {CONTINENT_NAMES} from './b11-geography';

/** Kilometres per degree of latitude, the unit of the bundling work box. */
export const KILOMETRES_PER_DEGREE = 111.19;

/** Index of the airport with this IATA code, or `-1`. */
export function findAirportByIata(network: FlightNetwork, iata: string): number {
  return network.airports.findIndex(airport => airport.iata === iata);
}

/** A route resolved in a network: the airports, the original pair and its expanded edges. */
export type ResolvedRoute = {
  /** Airport rows of the two ends. */
  airports: readonly [number, number];
  /** IATA codes of the two ends as found (a fallback may have replaced the requested one). */
  codes: readonly [string, string];
  /** Row of the pair in the original edge list. */
  edge: number;
  /** Rows of the expanded edges drawn for the pair (two when it crosses the antimeridian). */
  expandedEdges: readonly number[];
};

/**
 * Resolves the route `from` to the first of `to` that exists as a pair in the network, or `null`
 * when none does (OpenFlights 2014 has no New York to Haneda pair, for example).
 */
export function resolveRoute(
  network: FlightNetwork,
  expanded: ExpandedEdges,
  from: string,
  to: readonly string[]
): ResolvedRoute | null {
  const a = findAirportByIata(network, from);
  if (a < 0) return null;
  for (const code of to) {
    const b = findAirportByIata(network, code);
    if (b < 0) continue;
    for (let edge = 0; edge < network.edgeCount; edge++) {
      const forward = network.source[edge] === a && network.target[edge] === b;
      const backward = network.source[edge] === b && network.target[edge] === a;
      if (!forward && !backward) continue;
      const expandedEdges: number[] = [];
      for (let row = 0; row < expanded.original.length; row++) {
        if (expanded.original[row] === edge) expandedEdges.push(row);
      }
      return {airports: [a, b], codes: [from, code], edge, expandedEdges};
    }
  }
  return null;
}

/**
 * Which original pairs pass the region and distance filters (1) and which do not (0). Both ends of
 * a pair must be inside the region; `'lower48'` is the contiguous United States by bounds.
 */
export function getLivePairs(
  network: FlightNetwork,
  region: string,
  distanceRange: readonly [number, number]
): Uint8Array {
  const passes = new Uint8Array(network.edgeCount);
  for (let edge = 0; edge < network.edgeCount; edge++) {
    let inside = true;
    if (region !== 'all') {
      for (const airport of [network.source[edge], network.target[edge]]) {
        if (region === 'lower48') {
          const longitude = network.lonLat[airport * 2];
          const latitude = network.lonLat[airport * 2 + 1];
          inside &&= longitude > -125 && longitude < -66 && latitude > 24 && latitude < 50;
        } else {
          inside &&= CONTINENT_NAMES[network.continent[airport]] === region;
        }
      }
    }
    const distance = network.distanceKm[edge];
    if (distance < distanceRange[0] || distance > distanceRange[1]) inside = false;
    passes[edge] = inside ? 1 : 0;
  }
  return passes;
}

/**
 * The side, in km, of the square work box `GPUEdgeBundling` builds around the live edges: the
 * larger of the longitude span (scaled by the cosine of the mid-latitude, at least 0.01) and the
 * latitude span, padded on both sides. The kernel radius is a fraction of this side, so
 * `kernelRadius * side` is the radius on the ground (the mirror of `getBox()` in the shader).
 *
 * @param padding `GPU_EDGE_BUNDLING_WORK_BOX_PADDING`.
 */
export function getWorkBoxSideKilometres(
  expanded: ExpandedEdges,
  livePairs: Uint8Array,
  padding: number
): number {
  let minX = Number.POSITIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;
  const {positions} = expanded;
  for (let row = 0; row < expanded.source.length; row++) {
    if (!livePairs[expanded.original[row]]) continue;
    for (const vertex of [expanded.source[row], expanded.target[row]]) {
      const x = positions[vertex * 2];
      const y = positions[vertex * 2 + 1];
      minX = Math.min(minX, x);
      maxX = Math.max(maxX, x);
      minY = Math.min(minY, y);
      maxY = Math.max(maxY, y);
    }
  }
  if (!Number.isFinite(minX)) return 0;
  const longitudeScale = Math.max(Math.cos(0.5 * (minY + maxY) * (Math.PI / 180)), 0.01);
  const extent = Math.max((maxX - minX) * longitudeScale, maxY - minY);
  const side = extent > 0 ? extent * (1 + 2 * padding) : 1;
  return side * KILOMETRES_PER_DEGREE;
}

/** Kernel radius at iteration `iteration`: `r0 * decay^iteration` (the annealing schedule). */
export function getRadiusAtIteration(
  startRadius: number,
  decay: number,
  iteration: number
): number {
  return startRadius * decay ** iteration;
}

/** Histogram counts of `values` over `bins` equal bins of `[low, high]`; values past `high` land in the last bin. */
export function countInBins(
  values: ArrayLike<number>,
  low: number,
  high: number,
  bins: number
): number[] {
  const counts = new Array<number>(bins).fill(0);
  for (let index = 0; index < values.length; index++) {
    const value = values[index];
    if (!Number.isFinite(value) || value < low) continue;
    counts[Math.min(bins - 1, Math.floor(((value - low) / (high - low)) * bins))]++;
  }
  return counts;
}
