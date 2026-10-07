// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {MapAnnotation} from '../../cartography/types';
import type {VesselTrackSet} from './b12-tracks';

/** A visible, data-derived explanation of the 3 x 3 neighbourhood searched for one encounter. */
export function getVesselEncounterSearchAnnotations(
  vessels: VesselTrackSet,
  first: readonly [number, number],
  second: readonly [number, number],
  cellSize: number,
  distance: number,
  leash: {from: readonly [number, number]; to: readonly [number, number]; meters: number} | null
): MapAnnotation[] {
  const annotations: MapAnnotation[] = [];
  const cellX = Math.floor(first[0] / cellSize);
  const cellY = Math.floor(first[1] / cellSize);
  const minX = (cellX - 1) * cellSize;
  const minY = (cellY - 1) * cellSize;
  const maxX = (cellX + 2) * cellSize;
  const maxY = (cellY + 2) * cellSize;
  for (let column = 0; column <= 3; column++) {
    const x = minX + column * cellSize;
    annotations.push({
      kind: 'line',
      id: `encounter-lattice-v-${column}`,
      coordinates: [vessels.unproject(x, minY), vessels.unproject(x, maxY)],
      dashed: true,
      widthPixels: 1,
      tone: 'muted',
      priority: 1
    });
  }
  for (let row = 0; row <= 3; row++) {
    const y = minY + row * cellSize;
    annotations.push({
      kind: 'line',
      id: `encounter-lattice-h-${row}`,
      coordinates: [vessels.unproject(minX, y), vessels.unproject(maxX, y)],
      dashed: true,
      widthPixels: 1,
      tone: 'muted',
      priority: 1
    });
  }
  const midpoint: [number, number] = [(first[0] + second[0]) / 2, (first[1] + second[1]) / 2];
  annotations.push(
    {
      kind: 'ring',
      id: 'encounter-search-radius',
      coordinate: vessels.unproject(midpoint[0], midpoint[1]),
      radiusMeters: distance,
      text: `${distance} m rule`,
      dashed: true,
      tone: 'signal',
      priority: 8
    },
    {
      kind: 'line',
      id: 'encounter-now-connector',
      coordinates: [vessels.unproject(first[0], first[1]), vessels.unproject(second[0], second[1])],
      text: `${Math.hypot(first[0] - second[0], first[1] - second[1]).toFixed(0)} m now`,
      widthPixels: 2.5,
      tone: 'signal',
      priority: 9
    },
    {
      kind: 'note',
      id: 'encounter-search-note',
      coordinate: vessels.unproject(midpoint[0], midpoint[1]),
      title: `3 × 3 cells of ${cellSize} m`,
      text: 'Only samples in this bucket and these neighbouring cells are tested.',
      tone: 'ink',
      priority: 8
    }
  );
  if (leash) {
    annotations.push({
      kind: 'line',
      id: 'encounter-route-leash',
      coordinates: [
        vessels.unproject(leash.from[0], leash.from[1]),
        vessels.unproject(leash.to[0], leash.to[1])
      ],
      text: `route sample ${leash.meters.toFixed(0)} m apart`,
      dashed: true,
      widthPixels: 2,
      tone: 'accent',
      priority: 7
    });
  }
  return annotations;
}

/** A representative separation between equally progressed route samples for the leash teaching aid. */
export function getVesselRouteLeashWitness(
  vessels: VesselTrackSet,
  firstTrack: number,
  secondTrack: number,
  sampleCount = 64
): {from: [number, number]; to: [number, number]; meters: number} | null {
  const firstLength = vessels.offsets[firstTrack + 1] - vessels.offsets[firstTrack];
  const secondLength = vessels.offsets[secondTrack + 1] - vessels.offsets[secondTrack];
  if (firstLength < 2 || secondLength < 2) return null;
  let witness: {from: [number, number]; to: [number, number]; meters: number} | null = null;
  for (let sample = 0; sample < sampleCount; sample++) {
    const fraction = sample / (sampleCount - 1);
    const from = getTrackFraction(vessels, firstTrack, fraction);
    const to = getTrackFraction(vessels, secondTrack, fraction);
    const meters = Math.hypot(from[0] - to[0], from[1] - to[1]);
    if (!witness || meters > witness.meters) witness = {from, to, meters};
  }
  return witness;
}

function getTrackFraction(
  vessels: VesselTrackSet,
  track: number,
  fraction: number
): [number, number] {
  const first = vessels.offsets[track];
  const last = vessels.offsets[track + 1] - 1;
  const position = first + fraction * (last - first);
  const lower = Math.floor(position);
  const upper = Math.min(last, lower + 1);
  const mix = position - lower;
  return [
    vessels.positions[lower * 2] +
      (vessels.positions[upper * 2] - vessels.positions[lower * 2]) * mix,
    vessels.positions[lower * 2 + 1] +
      (vessels.positions[upper * 2 + 1] - vessels.positions[lower * 2 + 1]) * mix
  ];
}
