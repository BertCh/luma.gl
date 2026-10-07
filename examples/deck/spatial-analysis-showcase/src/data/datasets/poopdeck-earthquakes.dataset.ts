import type {DatasetInfo} from '../dataset-types';

export default {
  id: 'poopdeck-earthquakes',
  title: 'Global earthquakes M4+, 2020 to 2024',
  description:
    'Every magnitude 4+ event of the USGS ComCat catalog from 2020 to the end of 2024 (77,231 rows) with depth, magnitude, magnitude band and event type, served by the poopdeck.gl earthquakes archive and sorted by time.',
  license: 'Public domain (US Government work)',
  attribution: 'USGS Earthquake Catalog (ComCat), via poopdeck.gl',
  sourceUrl: 'https://earthquake.usgs.gov/fdsnws/event/1/',
  approxBytes: 1_700_000,
  bbox: [-180, -83, 180, 87]
} satisfies DatasetInfo;
