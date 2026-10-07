// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {MapAnnotation} from '../../cartography/types';
import type {MigrationSeason, MigrationTrackSet} from './migration-shared';
import {MIGRATION_SEASONS, MIGRATION_SPECIES_LABELS, SECONDS_PER_DAY} from './migration-shared';

export type FlywayEvidence = {
  animalYears: number[];
  birds: number[];
  fixes: number[];
  annotations: MapAnnotation[];
};

/** Counts the convenience sample behind the density and puts its recorded origins on the map. */
export function getMigrationFlywayEvidence(
  tracks: MigrationTrackSet,
  season: MigrationSeason
): FlywayEvidence {
  const animalYears = [0, 0, 0];
  const fixes = [0, 0, 0];
  const birds = [new Set<number>(), new Set<number>(), new Set<number>()];
  const originSums = [
    {longitude: 0, latitude: 0, count: 0},
    {longitude: 0, latitude: 0, count: 0},
    {longitude: 0, latitude: 0, count: 0}
  ];
  const [startDay, endDay] = MIGRATION_SEASONS[season].days;
  const start = startDay * SECONDS_PER_DAY;
  const end = endDay * SECONDS_PER_DAY;
  for (let track = 0; track < tracks.trackCount; track++) {
    const species = tracks.species[track];
    let trackFixes = 0;
    for (let vertex = tracks.offsets[track]; vertex < tracks.offsets[track + 1]; vertex++) {
      if (tracks.timestamps[vertex] >= start && tracks.timestamps[vertex] <= end) trackFixes++;
    }
    if (trackFixes < 2) continue;
    animalYears[species]++;
    fixes[species] += trackFixes;
    birds[species].add(tracks.individual[track]);
    const first = tracks.offsets[track];
    originSums[species].longitude += tracks.lngLat[first * 2];
    originSums[species].latitude += tracks.lngLat[first * 2 + 1];
    originSums[species].count++;
  }
  const annotations: MapAnnotation[] = originSums.flatMap((origin, species) => {
    if (!origin.count) return [];
    return [
      {
        kind: 'point' as const,
        id: `flyway-evidence-origin-${species}`,
        coordinate: [origin.longitude / origin.count, origin.latitude / origin.count] as const,
        text: MIGRATION_SPECIES_LABELS[species],
        detail: `${animalYears[species]} tagged years / ${birds[species].size} birds`,
        marker: 'ring' as const,
        tone: 'muted' as const,
        priority: 3,
        minZoom: 2.7
      }
    ];
  });
  return {animalYears, birds: birds.map(group => group.size), fixes, annotations};
}
