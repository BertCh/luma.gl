import type {DatasetInfo} from '../dataset-types';

export default {
  id: 'us-county-mortality',
  title: 'US county traffic deaths, 2017-2023',
  description:
    'Motor-vehicle crash deaths per county from NHTSA FARS with population at risk: small-number counts with unstable raw rates, built for empirical Bayes and rate smoothing.',
  license: 'US Government work (NHTSA FARS): public domain',
  attribution:
    'NHTSA Fatality Analysis Reporting System 2017-2023; US Census Bureau 2020 population',
  sourceUrl: 'https://www.nhtsa.gov/research-data/fatality-analysis-reporting-system-fars',
  approxBytes: 180000,
  bbox: [-124.7258, 24.4981, -66.9499, 49.3844]
} satisfies DatasetInfo;
