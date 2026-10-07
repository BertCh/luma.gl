import type {DatasetInfo} from '../dataset-types';

export default {
  id: 'us-airline-flows',
  title: 'US domestic flight flows, July 2023',
  description:
    '5,792 directed US airport-pair flows (601k flights in July 2023) with cancellations and departure delay, weighted by flight count.',
  license: 'US government public domain (BTS); airport coordinates ODbL 1.0 (OpenFlights)',
  attribution:
    'U.S. Bureau of Transportation Statistics On-Time Reporting Carrier data; airport positions OpenFlights.org (ODbL)',
  sourceUrl: 'https://www.transtats.bts.gov/',
  approxBytes: 167503,
  bbox: [-176.646, -14.331, 145.729, 71.285]
} satisfies DatasetInfo;
