import type {DatasetInfo} from '../dataset-types';

export default {
  id: 'chicago-311-rats',
  title: 'Chicago rodent complaints, 2023',
  description:
    '48.6k 311 rodent baiting / rat complaints in 2023: a second point process to compare with wildlife sightings.',
  license: 'City of Chicago Data Portal Terms of Use',
  attribution: 'City of Chicago 311',
  sourceUrl: 'https://data.cityofchicago.org/Service-Requests/311-Service-Requests/v6vf-nfxy',
  approxBytes: 682216,
  bbox: [-87.846, 41.648, -87.526, 42.023]
} satisfies DatasetInfo;
