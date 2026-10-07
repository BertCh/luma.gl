// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {MigrationTrackSet} from './migration-shared';
import {DAYS_IN_YEAR, SECONDS_PER_DAY} from './migration-shared';

export type MigrationSeasonComparison = {
  days: number[];
  shares: Float64Array[];
  panels: readonly {label: string; start: number; end: number}[];
};

/** Computes the folded-calendar comparison once from the recorded fixes, not from story claims. */
export function getMigrationSeasonComparison(
  tracks: MigrationTrackSet,
  southLatitude: number
): MigrationSeasonComparison {
  const days = Array.from({length: DAYS_IN_YEAR}, (_, day) => day + 0.5);
  const shares = [0, 1, 2].map(species => {
    const share = new Float64Array(DAYS_IN_YEAR);
    for (let day = 0; day < DAYS_IN_YEAR; day++) {
      let active = 0;
      let south = 0;
      for (let track = 0; track < tracks.trackCount; track++) {
        if (tracks.species[track] !== species) continue;
        const position = interpolateMigrationTrack(tracks, track, (day + 0.5) * SECONDS_PER_DAY);
        if (!position) continue;
        active++;
        if (position[1] < southLatitude) south++;
      }
      share[day] = active ? (100 * south) / active : Number.NaN;
    }
    return share;
  });
  return {
    days,
    shares,
    panels: [
      {label: 'Winter start', start: 0, end: 60},
      {label: 'Spring passage', start: 60, end: 167},
      {label: 'Breeding', start: 167, end: 213},
      {label: 'Autumn passage', start: 213, end: 335},
      {label: 'Winter return', start: 335, end: DAYS_IN_YEAR}
    ]
  };
}

function interpolateMigrationTrack(
  tracks: MigrationTrackSet,
  track: number,
  seconds: number
): [number, number] | null {
  let low = tracks.offsets[track];
  let high = tracks.offsets[track + 1] - 1;
  if (high < low || seconds < tracks.timestamps[low] || seconds > tracks.timestamps[high])
    return null;
  while (low < high) {
    const middle = (low + high + 1) >> 1;
    if (tracks.timestamps[middle] <= seconds) low = middle;
    else high = middle - 1;
  }
  const next = Math.min(low + 1, tracks.offsets[track + 1] - 1);
  const span = tracks.timestamps[next] - tracks.timestamps[low];
  const fraction = span > 0 ? (seconds - tracks.timestamps[low]) / span : 0;
  return [
    tracks.lngLat[low * 2] + (tracks.lngLat[next * 2] - tracks.lngLat[low * 2]) * fraction,
    tracks.lngLat[low * 2 + 1] +
      (tracks.lngLat[next * 2 + 1] - tracks.lngLat[low * 2 + 1]) * fraction
  ];
}
