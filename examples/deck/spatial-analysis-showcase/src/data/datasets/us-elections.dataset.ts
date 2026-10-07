import type {DatasetInfo} from '../dataset-types';

export default {
  id: 'us-elections',
  title: 'US presidential election returns by county, 2000-2024',
  description:
    'Democratic and Republican vote shares, total votes, turnout proxy and swing for seven presidential elections per county, as float32 columns aligned to us-counties. 2024 covers about 2,860 counties.',
  license: 'CC0 1.0 (MIT Election Data and Science Lab)',
  attribution:
    'MIT Election Data and Science Lab, County Presidential Election Returns 2000-2020 and 2024 precinct returns, Harvard Dataverse (CC0)',
  sourceUrl: 'https://doi.org/10.7910/DVN/VOQCHQ',
  approxBytes: 520000,
  bbox: [-124.7258, 24.4981, -66.9499, 49.3844]
} satisfies DatasetInfo;
