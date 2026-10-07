import type {DatasetInfo} from '../dataset-types';

export default {
  id: 'chicago-places',
  title: 'Chicago places (Overture Maps)',
  description:
    '106k open points of interest in Chicago in 14 broad categories (restaurants, grocery, health, schools, parks, transit, retail and more).',
  license: 'CDLA-Permissive-2.0 (Overture Places; some sources Apache-2.0 or CC0)',
  attribution: 'Overture Maps Foundation (release 2026-09-23.1)',
  sourceUrl: 'https://docs.overturemaps.org/attribution/places',
  approxBytes: 1271760,
  bbox: [-87.94, 41.645, -87.525, 42.023]
} satisfies DatasetInfo;
