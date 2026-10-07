import type {DatasetInfo} from '../dataset-types';

export default {
  id: 'gtfs-nl-rail-graph',
  title: 'Dutch rail graph with scheduled travel times',
  description:
    '485 stations and 1,411 directed station-to-station edges (intercity, express and stopping trains) with the median scheduled in-vehicle time and trains per hour, derived from 6,251 rail trips of Friday 9 October 2026.',
  license: 'CC0 1.0 Universal (public domain)',
  attribution: 'OVapi / NDOV national GTFS feed (derived graph)',
  sourceUrl: 'https://gtfs.ovapi.nl/nl/gtfs-nl.zip',
  approxBytes: 100_000,
  bbox: [3.07, 50.62, 7.57, 53.46]
} satisfies DatasetInfo;
