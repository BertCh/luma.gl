import type {DatasetInfo} from '../dataset-types';

export default {
  id: 'sf-bike-parking',
  title: 'San Francisco bike parking',
  description:
    'About 2,700 bike-parking locations with their number of spaces, from the deck.gl website. Fetched from deck.gl-data; clustered synthetic points replace it offline.',
  license: 'DataSF Open Data (PDDL), via the deck.gl-data repository',
  attribution: 'DataSF, bicycle parking (via deck.gl-data)',
  sourceUrl: 'https://data.sfgov.org/Transportation/Bicycle-Parking-Public-/hn4x-zwk7',
  approxBytes: 330_000,
  bbox: [-122.52, 37.7, -122.36, 37.83]
} satisfies DatasetInfo;
