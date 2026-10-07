import type {DatasetInfo} from '../dataset-types';

export default {
  id: 'poopdeck-mrms-storm3d-reports',
  title: 'SPC storm reports, 21-22 May 2024',
  description:
    '860 preliminary local storm reports (hail, wind, wind damage, tornado, flood, other) filed with the NOAA Storm Prediction Center from 17:30 UTC on 21 May to 03:00 UTC on 22 May 2024, with report time, location and reported magnitude.',
  license: 'Public domain (US Government work)',
  attribution: 'NOAA Storm Prediction Center preliminary local storm reports, via poopdeck.gl',
  sourceUrl: 'https://www.spc.noaa.gov/climo/reports/',
  approxBytes: 25_000,
  bbox: [-109.73, 25.64, -71.05, 47.94]
} satisfies DatasetInfo;
