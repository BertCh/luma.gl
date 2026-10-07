import type {DatasetInfo} from '../dataset-types';

export default {
  id: 'ais-vessels',
  title: 'NY/NJ Harbor vessel tracks, 12 June 2024',
  description:
    '897 AIS trajectories (471 vessels) in New York / New Jersey Harbor over one UTC day at 1-minute sampling, with speed, heading, vessel category and length.',
  license: 'CC0 1.0 Universal (public domain)',
  attribution:
    'NOAA Office for Coastal Management / U.S. Coast Guard Navigation Center, Nationwide AIS 2024',
  sourceUrl: 'https://github.com/ocm-marinecadastre/ais-vessel-traffic',
  approxBytes: 2266042,
  bbox: [-74.3, 40.45, -73.75, 40.8]
} satisfies DatasetInfo;
