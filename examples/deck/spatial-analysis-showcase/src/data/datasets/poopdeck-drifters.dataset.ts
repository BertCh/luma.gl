import type {DatasetInfo} from '../dataset-types';

export default {
  id: 'poopdeck-drifters',
  title: 'Global Drifter Program tracks, 2017 (first 60 days of each record)',
  description:
    '2,811 satellite-tracked surface drifters seen in 2017, each clipped to the 60 days after its first 2017 fix (or to 31 December) and thinned to about 12-hourly positions with sea-surface temperature, plus the real position at each of the first 30 lead days (the reference for model comparisons).',
  license: 'Public domain (NOAA waives copyright on GDP data)',
  attribution:
    'NOAA Global Drifter Program (AOML / PMEL), 6-hourly interpolated drifter data, via the poopdeck.gl archive',
  sourceUrl: 'https://www.aoml.noaa.gov/phod/gdp/',
  approxBytes: 4_525_509,
  bbox: [-180, -77.1, 180, 80.7]
} satisfies DatasetInfo;
