import type {DatasetInfo} from '../dataset-types';

export default {
  id: 'us-county-births',
  title: 'US county births, 2021-2023',
  description:
    "Live births per county (by mother's residence) and women aged 15-44 for 2021-2023 from the Census population estimates: a general fertility rate whose small-county values are unstable, built for empirical Bayes rate smoothing.",
  license: 'US Government work (US Census Bureau): public domain',
  attribution: 'US Census Bureau, Population Estimates Program, Vintage 2023',
  sourceUrl: 'https://www.census.gov/programs-surveys/popest.html',
  approxBytes: 130000,
  bbox: [-124.7258, 24.4981, -66.9499, 49.3844]
} satisfies DatasetInfo;
