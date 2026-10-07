import type {DatasetInfo} from '../dataset-types';

export default {
  id: 'bixi-flows',
  title: 'BIXI Montreal station flows, August 2024',
  description:
    '1.93 million BIXI rides between 905 stations in August 2024: 139,000 station pairs, 610,000 weekday/weekend x hour slices for the frequent pairs, residual rows that keep every station total exact, and the borough of each station.',
  license:
    'Creative Commons Attribution (BIXI Montreal trip history, as listed on the Montreal open data portal)',
  attribution: 'BIXI Montréal open data, trip history 2024',
  sourceUrl: 'https://bixi.com/en/open-data/',
  approxBytes: 5_723_716,
  bbox: [-73.8, 45.4, -73.45, 45.71]
} satisfies DatasetInfo;
