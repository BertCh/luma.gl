import type {DatasetInfo} from '../dataset-types';

export default {
  id: 'poopdeck-gtfs-nl',
  title: 'Randstad scheduled trips, Friday 3 July 2026, 07:00-09:00',
  description:
    '8,448 scheduled public-transport trips (tram, bus, metro, rail, ferry) in the Randstad during the morning peak, interpolated between stops along the route shape and simplified to 20 m. Scheduled positions, not real-time vehicle positions.',
  license: 'CC0 1.0 Universal (public domain)',
  attribution: 'OVapi / NDOV national GTFS feed, via poopdeck.gl (gtfs-nl)',
  sourceUrl: 'https://gtfs.ovapi.nl/nl/gtfs-nl.zip',
  approxBytes: 5_100_000,
  bbox: [4.2, 51.85, 5.2, 52.45]
} satisfies DatasetInfo;
