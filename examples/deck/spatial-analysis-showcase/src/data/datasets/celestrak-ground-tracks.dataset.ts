import type {DatasetInfo} from '../dataset-types';

export default {
  id: 'celestrak-ground-tracks',
  title: 'Satellite ground tracks, 3 hours from 7 October 2026 00:00 UTC',
  description:
    '911 satellites (stations, an 800-satellite Starlink sample, GPS, weather and Earth observation) propagated with SGP4 from current CelesTrak TLEs at 30 s steps, with geodetic altitude per vertex. Tracks are cut at the antimeridian. Simulated positions, not observed ones.',
  license: 'CelesTrak element sets: free to use with attribution; SGP4 propagation by this project',
  attribution:
    'CelesTrak (celestrak.org, Dr T.S. Kelso), GP element sets; propagation with satellite.js (SGP4)',
  sourceUrl: 'https://celestrak.org/NORAD/elements/',
  approxBytes: 5_660_000,
  bbox: [-180, -83, 180, 83]
} satisfies DatasetInfo;
