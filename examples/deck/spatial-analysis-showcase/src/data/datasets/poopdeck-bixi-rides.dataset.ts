import type {DatasetInfo} from '../dataset-types';

export default {
  id: 'poopdeck-bixi-rides',
  title: 'BIXI rides on the street network, 15 August 2024, 07:30-10:00',
  description:
    '8,365 BIXI rides of the morning peak, routed on the bicycle network with OSRM (derived routes, not GPS), simplified to 2 m, with per-vertex times. From the poopdeck.gl bixi-points archive.',
  license:
    'ODbL 1.0 (OSRM routes on OpenStreetMap) and Creative Commons Attribution (BIXI Montreal trip history)',
  attribution:
    'BIXI Montréal open data; © OpenStreetMap contributors; routes via OSRM; poopdeck.gl',
  sourceUrl: 'https://tiles.poopdeck.gl/data/bixi-points/manifest.json',
  approxBytes: 2_031_552,
  bbox: [-73.755, 45.415, -73.449, 45.702]
} satisfies DatasetInfo;
