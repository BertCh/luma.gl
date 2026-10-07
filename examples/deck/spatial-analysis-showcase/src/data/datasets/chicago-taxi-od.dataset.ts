import type {DatasetInfo} from '../dataset-types';

export default {
  id: 'chicago-taxi-od',
  title: 'Chicago taxi flows between community areas, 2023',
  description:
    'Taxi Trips 2023 aggregated to community-area origin-destination flows (full year, and by weekday/weekend x hour) plus a 365 x 77 daily pickup matrix.',
  license: 'City of Chicago Data Portal Terms of Use',
  attribution: 'City of Chicago (taxi trip data reported by licensed taxi companies)',
  sourceUrl: 'https://data.cityofchicago.org/resource/wrvz-psew',
  approxBytes: 2_998_122,
  bbox: [-87.94011, 41.64454, -87.52414, 42.02304]
} satisfies DatasetInfo;
