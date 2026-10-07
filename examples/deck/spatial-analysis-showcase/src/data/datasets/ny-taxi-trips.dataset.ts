import type {DatasetInfo} from '../dataset-types';

export default {
  id: 'ny-taxi-trips',
  title: 'New York taxi trips (sample)',
  description:
    'A few thousand taxi trips across Manhattan with per-vertex timestamps, from the deck.gl trips example. Fetched from deck.gl-data; a synthetic street-grid walk replaces it offline.',
  license: 'Sample data from the deck.gl-data repository (MIT)',
  attribution: 'deck.gl-data trips-v7 (NYC Taxi and Limousine Commission trips, sampled)',
  sourceUrl: 'https://github.com/visgl/deck.gl-data/tree/master/examples/trips',
  approxBytes: 1_600_000,
  bbox: [-74.02, 40.7, -73.93, 40.8]
} satisfies DatasetInfo;
